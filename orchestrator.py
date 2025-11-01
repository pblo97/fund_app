# orchestrator.py
#
# Flujo:
# 1. build_universe() ->
#       trae NYSE / NASDAQ / AMEX desde el screener FMP
#       normaliza a DataFrame y filtra large caps
#
# 2. build_full_snapshot(kept_symbols) ->
#       enriquece SOLO los tickers que tú marcaste en "kept"
#       (watchlist enriquecida arriba en la app)
#
# 3. build_market_snapshot() ->
#       arma la shortlist global del mercado (Tab1/Tab2):
#       - baja métricas core por ticker (fundamentales)
#       - mete placeholders de calidad/crecimiento
#       - aplica filtro básico de apalancamiento
#
# 4. enrich_company_snapshot(snapshot_dict_de_un_ticker) ->
#       agrega info cualitativa + series históricas
#       para Tab2 (insiders, news, transcript, gráficos)

from typing import List, Dict, Any
import math
import json
import numpy as np
import pandas as pd

from config import MAX_NET_DEBT_TO_EBITDA

from fmp_api import (
    run_screener_for_exchange,
    # snapshots / estados financieros
    get_profile,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_ratios,
    # históricos crudos por año
    get_cashflow_history,
    get_balance_history,
    get_income_history,
    get_shares_history,
    # enriquecimiento cualitativo
    get_insider_trading,
    get_news,
    get_earnings_call_transcript,
)

from metrics import compute_core_financial_metrics


# -------------------------------------------------
# Utils internos
# -------------------------------------------------

EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]

def _safe_df(x) -> pd.DataFrame:
    """
    Convierte lo que entregue run_screener_for_exchange() en DataFrame.
    Acepta:
    - list[dict]
    - DataFrame
    """
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, list):
        return pd.DataFrame(x)
    # último recurso: intenta DataFrame directo
    return pd.DataFrame(x)


def _row_large_cap(row: pd.Series, min_mktcap: float = 10_000_000_000) -> bool:
    """
    True si la fila es large cap >=10B USD.
    """
    mc = (
        row.get("marketCap")
        or row.get("mktCap")
        or row.get("marketCapIntraday")
    )
    try:
        return float(mc) >= float(min_mktcap)
    except Exception:
        return False


def _linear_trend(values: pd.Series) -> float | None:
    """
    Slope lineal simple (np.polyfit) para ver la tendencia.
    Devuelve None si no hay suficientes puntos.
    """
    s = pd.Series(values).dropna()
    if len(s) < 2:
        return None
    x = np.arange(len(s), dtype=float)
    y = s.astype(float).values
    slope, _intercept = np.polyfit(x, y, 1)
    return slope


def _merge_scores_and_growth(base_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Placeholder seguro de calidad/crecimiento:
    Agregamos llaves esperadas por la UI aunque todavía no
    calculemos Altman Z, Piotroski, CAGR, etc.
    """
    enriched = []
    for row in base_rows:
        r = dict(row)

        # Placeholders numéricos / métricas de screening
        r.setdefault("altmanZScore", None)
        r.setdefault("piotroskiScore", None)

        r.setdefault("revenueGrowth", None)
        r.setdefault("operatingCashFlowGrowth", None)
        r.setdefault("freeCashFlowGrowth", None)
        r.setdefault("debtGrowth", None)

        r.setdefault("rev_CAGR_5y", None)
        r.setdefault("rev_CAGR_3y", None)
        r.setdefault("ocf_CAGR_5y", None)
        r.setdefault("ocf_CAGR_3y", None)

        # Campos sectoriales / moat (heurístico, placeholder)
        r.setdefault("moat_flag", "—")

        enriched.append(r)

    return enriched


def _quality_filter_final(d: Dict[str, Any]) -> bool:
    """
    Filtro mínimo para quedarse con candidatas 'sanas'.
    Por ahora:
    - Net Debt / EBITDA <= MAX_NET_DEBT_TO_EBITDA (si se conoce)
    - (Large cap ya lo filtramos antes en build_universe)
    """
    nde = d.get("netDebt_to_EBITDA")
    if nde is not None:
        try:
            if float(nde) > float(MAX_NET_DEBT_TO_EBITDA):
                return False
        except Exception:
            pass
    return True


def _historicals_for_detail(symbol: str) -> Dict[str, Any]:
    """
    Baja históricos anuales y construye series para los gráficos de Tab2:
    - fcf por acción
    - shares en circulación
    - deuda neta
    """
    cf   = get_cashflow_history(symbol)
    bal  = get_balance_history(symbol)
    inc  = get_income_history(symbol)
    shr  = get_shares_history(symbol)

    hist = (
        _safe_df(cf)
        .merge(_safe_df(shr), on="fiscalDate", how="left")
        .merge(_safe_df(bal), on="fiscalDate", how="left")
        .merge(_safe_df(inc), on="fiscalDate", how="left")
        .sort_values("fiscalDate")
        .reset_index(drop=True)
    )

    # calcula métricas derivadas por año
    hist["fcf"] = hist.get("operatingCashFlow", 0) - hist.get("capitalExpenditure", 0)
    hist["fcf_per_share"] = hist["fcf"] / hist.get("sharesDiluted", np.nan)
    hist["net_debt"] = hist.get("totalDebt", 0) - hist.get("cashAndShortTermInvestments", 0)

    years = []
    fcfps_hist = []
    shares_hist = []
    net_debt_hist = []

    for _, r in hist.iterrows():
        # tratar de deducir "año fiscal" desde fiscalDate
        fd = r.get("fiscalDate")
        try:
            year_str = str(fd)[:4]
        except Exception:
            year_str = "—"

        years.append(year_str)
        fcfps_hist.append(r.get("fcf_per_share", None))
        shares_hist.append(r.get("sharesDiluted", None))
        net_debt_hist.append(r.get("net_debt", None))

    return {
        "years": years,
        "fcf_per_share_hist": fcfps_hist,
        "shares_hist": shares_hist,
        "net_debt_hist": net_debt_hist,
    }


# -------------------------------------------------
# Paso 1. Universo inicial
# -------------------------------------------------

def build_universe() -> pd.DataFrame:
    """
    Trae screener para cada exchange, normaliza a DF,
    concatena, deduplica por symbol y deja solo large caps.
    """
    frames: List[pd.DataFrame] = []

    for exch in EXCHANGES:
        chunk = run_screener_for_exchange(exch)
        df_chunk = _safe_df(chunk)

        # nos aseguramos de tener columnas clave aunque vengan ausentes
        for col in ["symbol", "companyName", "sector", "industry", "marketCap"]:
            if col not in df_chunk.columns:
                df_chunk[col] = None

        frames.append(df_chunk)

    universe_raw = pd.concat(frames, ignore_index=True)

    # limpiamos duplicados por ticker
    universe_raw = (
        universe_raw
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )

    # quedarnos solo con large caps
    mask_large = universe_raw.apply(_row_large_cap, axis=1)
    universe_large = universe_raw[mask_large].reset_index(drop=True)

    return universe_large


# -------------------------------------------------
# Paso 2. Bloque fundamentals rápido (slope FCF/acción, recompras, deuda)
# -------------------------------------------------

def fetch_fundamentals_for_symbol(symbol: str) -> dict:
    """
    Calcula:
    - slope de FCF por acción a ~5y
    - recompras (% de reducción en sharesDiluted)
    - cambio deuda neta
    - net_debt_to_ebitda_last
    """
    cf   = get_cashflow_history(symbol)
    bal  = get_balance_history(symbol)
    inc  = get_income_history(symbol)
    shr  = get_shares_history(symbol)

    hist = (
        _safe_df(cf)
        .merge(_safe_df(shr), on="fiscalDate", how="left")
        .merge(_safe_df(bal), on="fiscalDate", how="left")
        .merge(_safe_df(inc), on="fiscalDate", how="left")
        .sort_values("fiscalDate")
        .reset_index(drop=True)
    )

    # derivadas anuales
    hist["fcf"] = hist.get("operatingCashFlow", 0) - hist.get("capitalExpenditure", 0)
    hist["fcf_per_share"] = hist["fcf"] / hist.get("sharesDiluted", np.nan)
    hist["net_debt"] = hist.get("totalDebt", 0) - hist.get("cashAndShortTermInvestments", 0)

    # slope de FCF/acción
    fcfps_slope = _linear_trend(hist["fcf_per_share"])

    # recompras (% reducción acciones diluidas)
    if len(hist) >= 2:
        shares_start = hist["sharesDiluted"].iloc[0]
        shares_end   = hist["sharesDiluted"].iloc[-1]
    else:
        shares_start = None
        shares_end   = None

    if shares_start and shares_end:
        buyback_pct_5y = (shares_start - shares_end) / shares_start
    else:
        buyback_pct_5y = None

    # cambio deuda neta en el período
    if "net_debt" in hist.columns and len(hist) >= 2:
        net_debt_change_5y = hist["net_debt"].iloc[-1] - hist["net_debt"].iloc[0]
    else:
        net_debt_change_5y = None

    # net debt / EBITDA del último año
    if len(hist) and ("ebitda" in hist.columns):
        last = hist.iloc[-1]
        if (
            last.get("ebitda") not in [None, 0]
            and last.get("net_debt") is not None
        ):
            net_debt_to_ebitda_last = last["net_debt"] / last["ebitda"]
        else:
            net_debt_to_ebitda_last = None
    else:
        net_debt_to_ebitda_last = None

    return {
        "symbol": symbol,
        "fcf_per_share_slope_5y": fcfps_slope,
        "buyback_pct_5y": buyback_pct_5y,
        "net_debt_change_5y": net_debt_change_5y,
        "net_debt_to_ebitda_last": net_debt_to_ebitda_last,
    }


def build_fundamentals_block(symbols: List[str]) -> pd.DataFrame:
    """
    Itera cada símbolo y arma las métricas de FCF, recompras, deuda.
    Nunca revienta: si falla un ticker, rellena None.
    """
    rows = []
    for sym in symbols:
        try:
            row = fetch_fundamentals_for_symbol(sym)
        except Exception:
            row = {
                "symbol": sym,
                "fcf_per_share_slope_5y": None,
                "buyback_pct_5y": None,
                "net_debt_change_5y": None,
                "net_debt_to_ebitda_last": None,
            }
        rows.append(row)

    return pd.DataFrame(rows)


def enrich_universe_with_fundamentals(
    universe_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Hace merge 1:1 por symbol y genera banderas de calidad.
    """
    full = universe_df.merge(
        fundamentals_df,
        on="symbol",
        how="left",
        validate="1:1"
    )

    def _flag_positive(x):
        return (x is not None) and (pd.notna(x)) and (x > 0)

    # fcf en alza
    full["flag_fcf_up"] = full["fcf_per_share_slope_5y"].apply(_flag_positive)

    # recompras agresivas (>5%)
    full["flag_buybacks"] = full["buyback_pct_5y"].apply(
        lambda x: (x is not None) and pd.notna(x) and (x > 0.05)
    )

    # deuda controlada (<2x EBITDA aprox)
    full["flag_net_debt_ok"] = full["net_debt_to_ebitda_last"].apply(
        lambda x: (x is not None) and pd.notna(x) and (x < 2)
    )

    full["is_quality_compounder"] = (
        full["flag_fcf_up"]
        & full["flag_buybacks"]
        & full["flag_net_debt_ok"]
    )

    return full


# -------------------------------------------------
# Paso 3. build_full_snapshot() -> Watchlist enriquecida
# -------------------------------------------------

def build_full_snapshot(kept_symbols: List[str]) -> pd.DataFrame:
    """
    Para la watchlist personalizada que tienes en st.session_state["kept"].
    - Filtra el universo a SOLO esos tickers.
    - Calcula fundamentals_block solo para esos tickers.
    - Devuelve DF mergeado con flags como is_quality_compounder.
    """
    if not kept_symbols:
        return pd.DataFrame()

    universe_all = build_universe()
    uni_subset = universe_all[universe_all["symbol"].isin(kept_symbols)].copy()

    fundamentals_block = build_fundamentals_block(kept_symbols)

    final_df = enrich_universe_with_fundamentals(
        uni_subset,
        fundamentals_block
    )

    return final_df


# -------------------------------------------------
# Paso 4. build_market_snapshot() -> Shortlist global Tab1/Tab2
# -------------------------------------------------

def build_company_core_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Métricas fundamentales core de UN TICKER usando estados financieros,
    ratios, etc. Llama a compute_core_financial_metrics().
    """
    profile = get_profile(ticker)
    income_hist = get_income_statement(ticker)
    balance_hist = get_balance_sheet(ticker)
    cash_hist = get_cash_flow(ticker)
    ratios_hist = get_ratios(ticker)

    # validación mínima para no romper
    if (
        not isinstance(income_hist, list) or len(income_hist) < 2 or
        not isinstance(balance_hist, list) or len(balance_hist) < 2 or
        not isinstance(cash_hist, list) or len(cash_hist) < 2
    ):
        raise ValueError("historial financiero insuficiente")

    base_metrics = compute_core_financial_metrics(
        ticker=ticker,
        profile=profile,
        ratios_hist=ratios_hist,
        income_hist=income_hist,
        balance_hist=balance_hist,
        cash_hist=cash_hist,
    )

    # compute_core_financial_metrics deberá darnos como mínimo:
    # {
    #   "ticker": ...,
    #   "name": ...,
    #   "sector": ...,
    #   "industry": ...,
    #   "marketCap": ...,
    #   "netDebt_to_EBITDA": ...,
    #   ...
    # }
    return base_metrics


def build_market_snapshot(limit: int = 40) -> List[Dict[str, Any]]:
    """
    Construye la shortlist global (lo que Tab1 y Tab2 usan).
    Ahora además mezcla:
    - snapshot core (P&L, balance, ratios)
    - bloque fundamentals (FCF/acción slope, recompras, deuda, etc.)
    y luego mete los placeholders de growth/Altman/etc.
    """

    uni = build_universe()

    tickers = (
        uni["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    tickers = tickers[:limit]  # puedes subir/eliminar este límite

    rows_raw: List[Dict[str, Any]] = []

    for tkr in tickers:
        try:
            # métrica core (income/balance/cash/ratios)
            core = build_company_core_snapshot(tkr)
        except Exception:
            # si ni siquiera podemos sacar los básicos, saltamos este ticker
            continue

        # tratamos también de sacar fundamentals "anualizados"
        # (fcf_per_share_slope_5y, recompras, deuda neta, etc.)
        try:
            fblock = fetch_fundamentals_for_symbol(tkr)
        except Exception:
            fblock = {
                "fcf_per_share_slope_5y": None,
                "buyback_pct_5y": None,
                "net_debt_change_5y": None,
                "net_debt_to_ebitda_last": None,
            }

        # mergeamos ambos dicts
        merged = dict(core)
        merged.update(fblock)

        # aseguramos campos que tu UI usa con nombres consistentes
        # ticker vs symbol
        if "ticker" not in merged and "symbol" in merged:
            merged["ticker"] = merged["symbol"]
        if "symbol" not in merged and "ticker" in merged:
            merged["symbol"] = merged["ticker"]

        # agregamos a rows_raw
        rows_raw.append(merged)

    # ahora metemos placeholders tipo altmanZScore, growth, CAGR, etc.
    rows_enriched = _merge_scores_and_growth(rows_raw)

    # filtramos apalancamiento
    final_rows = [r for r in rows_enriched if _quality_filter_final(r)]

    return final_rows


# -------------------------------------------------
# Paso 5. enrich_company_snapshot() -> Detalle 1 ticker (Tab2)
# -------------------------------------------------

def enrich_company_snapshot(base_core_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parte de lo que ya tenemos en Tab1 para un ticker y
    le agrega:
    - insiders (señal muy burda)
    - sentimiento noticias
    - resumen de última transcript
    - series históricas para los charts
    """
    out = dict(base_core_snapshot)
    ticker = out.get("ticker") or out.get("symbol")

    if ticker is None:
        return out  # sin ticker no podemos enriquecer más

    # ---------- insiders ----------
    try:
        insider_raw = get_insider_trading(ticker)
    except Exception:
        insider_raw = []

    insider_signal = "neutral"
    if isinstance(insider_raw, list) and len(insider_raw) > 0:
        # Heurística MUY básica:
        # si hay más compras que ventas recientes -> "bullish"
        buys = 0.0
        sells = 0.0
        for tr in insider_raw:
            # intentamos leer tipo y cantidad
            typ = str(tr.get("transactionType", "")).lower()
            val = tr.get("amount", 0) or tr.get("shares", 0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            if "buy" in typ:
                buys += val
            if "sell" in typ:
                sells += val
        if buys > sells * 1.2:
            insider_signal = "bullish"
        elif sells > buys * 1.2:
            insider_signal = "bearish"

    out["insider_signal"] = insider_signal

    # ---------- news / sentimiento ----------
    try:
        news_raw = get_news(ticker)
    except Exception:
        news_raw = []

    sentiment_flag = "neutral"
    sentiment_reason = "tono mixto/sectorial"
    if isinstance(news_raw, list) and len(news_raw) > 0:
        # micro heurística: si en títulos sale 'record', 'beat', etc => bullish
        heads = " ".join([str(n.get("title", "")).lower() for n in news_raw[:5]])
        if any(w in heads for w in ["record", "beat", "surge", "strong"]):
            sentiment_flag = "bullish"
            sentiment_reason = "titulares positivos recientes"
        elif any(w in heads for w in ["probe", "fraud", "miss", "lawsuit", "sec "]):
            sentiment_flag = "bearish"
            sentiment_reason = "titulares de riesgo/regulatorio"

    out["sentiment_flag"] = sentiment_flag
    out["sentiment_reason"] = sentiment_reason

    # ---------- earnings call transcript ----------
    try:
        transcript_raw = get_earnings_call_transcript(ticker)
    except Exception:
        transcript_raw = None

    transcript_summary = "Sin señales fuertes en la última call."
    if transcript_raw:
        # tratamos de construir un mini resumen muy defensivo
        if isinstance(transcript_raw, dict):
            body = transcript_raw.get("content") or transcript_raw.get("text") or ""
        elif isinstance(transcript_raw, list) and len(transcript_raw) > 0:
            body = transcript_raw[0].get("content", "") or transcript_raw[0].get("text","")
        else:
            body = ""

        if body:
            # tomamos las ~300 primeras chars, pero limpiando saltos
            snippet = " ".join(str(body).split())[:300]
            transcript_summary = (
                "Extracto call: "
                + snippet
                + ("..." if len(snippet) == 300 else "")
            )

    out["transcript_summary"] = transcript_summary

    # ---------- series históricas para gráficos ----------
    hist_block = _historicals_for_detail(ticker)
    out.update(hist_block)

    # normalizamos nombres para que Tab2 no reviente
    if "symbol" not in out and "ticker" in out:
        out["symbol"] = out["ticker"]
    if "ticker" not in out and "symbol" in out:
        out["ticker"] = out["symbol"]

    # aseguramos que haya sector/industry aunque el DF base no las traiga
    out.setdefault("sector", base_core_snapshot.get("sector", "—"))
    out.setdefault("industry", base_core_snapshot.get("industry", "—"))
    out.setdefault("moat_flag", base_core_snapshot.get("moat_flag", "—"))

    # business summary / why it matters / risk note:
    # si compute_core_financial_metrics no lo trae, dejamos textos placeholder
    out.setdefault("business_summary", base_core_snapshot.get("business_summary", "—"))
    out.setdefault("why_it_matters", base_core_snapshot.get("why_it_matters", "—"))
    out.setdefault(
        "core_risk_note",
        base_core_snapshot.get("core_risk_note", "riesgo principal no crítico visible"),
    )

    return out
