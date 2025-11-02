# orchestrator.py
#
# Flujo actualizado y unificado:
#
# 1. build_universe()
#    - arma universo de tickers grandes (NYSE / NASDAQ / AMEX) usando run_screener_for_exchange
#    - normaliza columnas básicas (ticker, companyName, sector, industry, marketCap)
#
# 2. fetch_fundamentals_for_symbol(ticker)
#    - baja históricos de cashflow/balance/ingresos/acciones
#    - calcula:
#        * slope FCF/acción
#        * recompras (% reducción acciones)
#        * net_debt_to_ebitda_last
#        * growths (revenue, OCF, FCF, deuda) último vs penúltimo
#        * CAGR 5y-ish revenue y OCF
#    - incluye placeholders altmanZScore, piotroskiScore, moat_flag
#
# 3. build_fundamentals_block(symbols)
#    - corre fetch_fundamentals_for_symbol para cada símbolo
#
# 4. enrich_universe_with_fundamentals(universe_df, fundamentals_df)
#    - mergea universo + fundamentals_block
#    - calcula is_quality_compounder (heurística)
#
# 5. build_market_snapshot()
#    - une todo, devuelve list[dict] listo para st.session_state["snapshot_rows"]
#
# 6. build_full_snapshot(kept_symbols)
#    - igual que build_market_snapshot pero sólo para la watchlist (kept)
#
# 7. enrich_company_snapshot(base_core)
#    - arma la ficha detallada de 1 ticker (insiders / news sentiment / series históricas para los gráficos Tab2)
#
# 8. build_company_core_snapshot(ticker)
#    - snapshot fundamental más "crudo" (por si quieres usarlo luego en otros módulos)
#
# Nota: este archivo ya no tiene funciones duplicadas.


from typing import List, Dict, Any
import math
import numpy as np
import pandas as pd

from config import MAX_NET_DEBT_TO_EBITDA

from fmp_api import (
    # screener / universe
    run_screener_for_exchange,

    # históricos anuales
    get_cashflow_history,
    get_balance_history,
    get_income_history,
    get_shares_history,

    # estados financieros "core" y perfil (para uso avanzado / detalle)
    get_profile,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_ratios,

    # info cualitativa
    get_insider_trading,
    get_news,
    get_earnings_call_transcript,
)

from metrics import compute_core_financial_metrics


# ============================================================
# Helpers internos
# ============================================================

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _large_cap_cut(mc_val: Any, threshold: float = 10_000_000_000) -> bool:
    """
    True si market cap >= threshold.
    """
    try:
        return float(mc_val) >= float(threshold)
    except Exception:
        return False


def _growth(curr_val, prev_val):
    """
    crecimiento relativo (curr-prev)/prev.
    devuelve fracción, ej 0.12 = 12%
    """
    try:
        if prev_val is None or prev_val == 0:
            return None
        return (curr_val - prev_val) / prev_val
    except Exception:
        return None


def _cagr(first: float, last: float, n_years: float) -> float | None:
    """
    CAGR genérico. Devuelve fracción (0.18 = 18%).
    Usa n_years ~ cantidad de pasos entre primer y último dato.
    """
    try:
        if first is None or last is None:
            return None
        if first <= 0 or last <= 0:
            return None
        if n_years <= 0:
            return None
        return ((last / first) ** (1.0 / n_years)) - 1.0
    except Exception:
        return None


def _linear_trend(vals: pd.Series) -> float | None:
    """
    slope lineal simple de una serie (y vs índice 0..n-1)
    """
    s = pd.Series(vals).dropna()
    if len(s) < 2:
        return None
    x = np.arange(len(s), dtype=float)
    y = s.astype(float).values
    slope, _ = np.polyfit(x, y, 1)
    return slope


# ============================================================
# 1. Universo base (tickers grandes)
# ============================================================

EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]

def build_universe() -> pd.DataFrame:
    """
    Arma el universo de large caps combinando los exchanges soportados.
    Normaliza columnas clave:
        symbol / ticker
        companyName
        sector / industry
        marketCap
    y filtra marketCap >= 10B.
    """
    frames: list[pd.DataFrame] = []

    for exch in EXCHANGES:
        ch = run_screener_for_exchange(exch)

        # el screener puede venir como list[dict] -> lo llevamos a DF
        if isinstance(ch, list):
            ch = pd.DataFrame(ch)
        elif not isinstance(ch, pd.DataFrame):
            ch = pd.DataFrame([])

        frames.append(ch)

    if frames:
        universe = pd.concat(frames, ignore_index=True)
    else:
        universe = pd.DataFrame()

    # normalización básica de nombre
    if "companyName" not in universe.columns and "name" in universe.columns:
        universe["companyName"] = universe["name"]

    # normalización de ticker
    if "symbol" not in universe.columns and "ticker" in universe.columns:
        universe["symbol"] = universe["ticker"]

    # marketCap estandarizado
    # algunos screeners usan marketCap / mktCap / marketCapIntraday
    def _pick_mktcap(row):
        for k in ["marketCap", "mktCap", "marketCapIntraday"]:
            if k in row and pd.notna(row[k]):
                return row[k]
        return None

    universe["marketCap"] = universe.apply(_pick_mktcap, axis=1)

    # filtrar solo large caps
    universe = universe[universe["marketCap"].apply(_large_cap_cut)].reset_index(drop=True)

    # aseguramos sector / industry
    if "sector" not in universe.columns:
        universe["sector"] = None
    if "industry" not in universe.columns:
        universe["industry"] = None

    # creamos columna "ticker" que la UI espera
    universe["ticker"] = universe["symbol"].astype(str)

    return universe[[
        "symbol",
        "ticker",
        "companyName",
        "sector",
        "industry",
        "marketCap",
    ]].reset_index(drop=True)


# ============================================================
# 2. Métricas fundamentales por símbolo
# ============================================================

def fetch_fundamentals_for_symbol(symbol: str) -> dict:
    """
    Baja históricos anuales (cashflow, balance, income, shares),
    los fusiona por fiscalDate, y calcula:

    - fcf_per_share_slope_5y
    - buyback_pct_5y
    - net_debt_change_5y
    - net_debt_to_ebitda_last
    - revenueGrowth (último vs penúltimo)
    - operatingCashFlowGrowth
    - freeCashFlowGrowth
    - debtGrowth
    - rev_CAGR_5y  (aprox "5y": primero -> último del hist)
    - ocf_CAGR_5y

    Además mete placeholders:
    - altmanZScore
    - piotroskiScore
    - moat_flag
    """

    cf   = get_cashflow_history(symbol)
    bal  = get_balance_history(symbol)
    inc  = get_income_history(symbol)
    shr  = get_shares_history(symbol)

    # forzamos DataFrame por seguridad
    def _to_df(x):
        if isinstance(x, list):
            return pd.DataFrame(x)
        if isinstance(x, dict):
            return pd.DataFrame([x])
        if isinstance(x, pd.DataFrame):
            return x
        return pd.DataFrame()
    cf  = _to_df(cf)
    bal = _to_df(bal)
    inc = _to_df(inc)
    shr = _to_df(shr)

    # merge anual por fiscalDate
    hist = (
        cf.merge(shr, on="fiscalDate", how="left")
          .merge(bal, on="fiscalDate", how="left")
          .merge(inc, on="fiscalDate", how="left")
          .sort_values("fiscalDate")
          .reset_index(drop=True)
    )

    # derivadas
    # FCF = OCF - CapEx
    if "operatingCashFlow" not in hist.columns:
        hist["operatingCashFlow"] = None
    if "capitalExpenditure" not in hist.columns:
        hist["capitalExpenditure"] = None

    hist["fcf"] = hist["operatingCashFlow"] - hist["capitalExpenditure"]

    # FCF por acción
    if "sharesDiluted" not in hist.columns:
        hist["sharesDiluted"] = None
    hist["fcf_per_share"] = hist["fcf"] / hist["sharesDiluted"]

    # Deuda neta = totalDebt - cash
    if "totalDebt" not in hist.columns:
        hist["totalDebt"] = None
    if "cashAndShortTermInvestments" not in hist.columns:
        hist["cashAndShortTermInvestments"] = None
    hist["net_debt"] = hist["totalDebt"] - hist["cashAndShortTermInvestments"]

    # slope FCF/acción
    fcfps_slope = _linear_trend(hist["fcf_per_share"])

    # recompras: % reducción en sharesDiluted (primera vs última fila)
    if len(hist) > 0:
        shares_start = hist["sharesDiluted"].iloc[0]
        shares_end   = hist["sharesDiluted"].iloc[-1]
    else:
        shares_start = None
        shares_end   = None

    if shares_start and shares_end:
        try:
            buyback_pct_5y = (shares_start - shares_end) / shares_start
        except Exception:
            buyback_pct_5y = None
    else:
        buyback_pct_5y = None

    # cambio deuda neta en el periodo (último - primero)
    if len(hist) >= 2:
        try:
            net_debt_change_5y = hist["net_debt"].iloc[-1] - hist["net_debt"].iloc[0]
        except Exception:
            net_debt_change_5y = None
    else:
        net_debt_change_5y = None

    # net_debt / EBITDA (último año)
    if len(hist) > 0:
        last_row = hist.iloc[-1]
    else:
        last_row = {}
    eb = last_row.get("ebitda", last_row.get("EBITDA", None))
    nd = last_row.get("net_debt", None)
    if eb not in [None, 0] and nd is not None:
        net_debt_to_ebitda_last = nd / eb
    else:
        net_debt_to_ebitda_last = None

    # growths (último vs penúltimo)
    if len(hist) >= 2:
        prev_row = hist.iloc[-2]
        revenueGrowth = _growth(last_row.get("revenue"), prev_row.get("revenue"))
        operatingCashFlowGrowth = _growth(
            last_row.get("operatingCashFlow"),
            prev_row.get("operatingCashFlow"),
        )
        freeCashFlowGrowth = _growth(
            last_row.get("fcf"),
            prev_row.get("fcf"),
        )
        debtGrowth = _growth(
            last_row.get("net_debt"),
            prev_row.get("net_debt"),
        )
    else:
        revenueGrowth = None
        operatingCashFlowGrowth = None
        freeCashFlowGrowth = None
        debtGrowth = None

    # CAGR "5y": en realidad primer vs último
    n_periods = max(len(hist) - 1, 1)
    if len(hist) > 0:
        first_row = hist.iloc[0]
    else:
        first_row = {}

    rev_CAGR_5y = _cagr(
        first_row.get("revenue"), last_row.get("revenue"), n_periods
    )
    ocf_CAGR_5y = _cagr(
        first_row.get("operatingCashFlow"),
        last_row.get("operatingCashFlow"),
        n_periods,
    )

    # placeholders por ahora
    altmanZScore = None
    piotroskiScore = None
    moat_flag = None

    return {
        "symbol": symbol,

        # métricas cuantitativas que usa tu tabla
        "fcf_per_share_slope_5y": fcfps_slope,
        "buyback_pct_5y": buyback_pct_5y,
        "net_debt_change_5y": net_debt_change_5y,
        "net_debt_to_ebitda_last": net_debt_to_ebitda_last,

        "revenueGrowth": revenueGrowth,
        "operatingCashFlowGrowth": operatingCashFlowGrowth,
        "freeCashFlowGrowth": freeCashFlowGrowth,
        "debtGrowth": debtGrowth,

        "rev_CAGR_5y": rev_CAGR_5y,
        "ocf_CAGR_5y": ocf_CAGR_5y,

        # placeholders cualitativos / analíticos
        "altmanZScore": altmanZScore,
        "piotroskiScore": piotroskiScore,
        "moat_flag": moat_flag,
    }


def build_fundamentals_block(symbols: list[str]) -> pd.DataFrame:
    """
    Llama fetch_fundamentals_for_symbol para cada símbolo
    y devuelve un DataFrame con una fila por ticker.
    """
    rows: list[dict] = []
    for sym in symbols:
        try:
            rows.append(fetch_fundamentals_for_symbol(sym))
        except Exception:
            # fallback seguro si falla un ticker
            rows.append({
                "symbol": sym,
                "fcf_per_share_slope_5y": None,
                "buyback_pct_5y": None,
                "net_debt_change_5y": None,
                "net_debt_to_ebitda_last": None,
                "revenueGrowth": None,
                "operatingCashFlowGrowth": None,
                "freeCashFlowGrowth": None,
                "debtGrowth": None,
                "rev_CAGR_5y": None,
                "ocf_CAGR_5y": None,
                "altmanZScore": None,
                "piotroskiScore": None,
                "moat_flag": None,
            })
    return pd.DataFrame(rows)


# ============================================================
# 3. Merge universo + fundamentals + flags
# ============================================================

def enrich_universe_with_fundamentals(
    universe_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Hace el merge 1:1 por symbol y calcula is_quality_compounder
    a partir de:
      - fcf_per_share_slope_5y > 0
      - buyback_pct_5y > 5%
      - net_debt_to_ebitda_last < 2
    """

    full = universe_df.merge(
        fundamentals_df,
        how="left",
        on="symbol",
        validate="1:1"
    )

    # helper chico para booleanos:
    def _flag_positive(val):
        return (val is not None) and (pd.notna(val)) and (val > 0)

    full["flag_fcf_up"] = full["fcf_per_share_slope_5y"].apply(_flag_positive)

    full["flag_buybacks"] = full["buyback_pct_5y"].apply(
        lambda x: (x is not None) and pd.notna(x) and (x > 0.05)
    )

    full["flag_net_debt_ok"] = full["net_debt_to_ebitda_last"].apply(
        lambda x: (x is not None) and pd.notna(x) and (x < 2)
    )

    full["is_quality_compounder"] = (
        full["flag_fcf_up"]
        & full["flag_buybacks"]
        & full["flag_net_debt_ok"]
    )

    return full


# ============================================================
# 4. build_market_snapshot() → botón "Run Screening"
# ============================================================

def build_market_snapshot() -> list[dict]:
    """
    1. Saca universo (NYSE/NASDAQ/AMEX filtrado por marketCap >=10B)
    2. Para cada ticker arma snapshot core + fundamentals
    3. Devuelve lista de dicts listos para la UI
    """
    uni = build_universe()

    # quedarnos solo con large caps
    uni = uni[uni["marketCap"].astype(float) >= 10_000_000_000].copy()
    uni = uni.reset_index(drop=True)

    rows_out: list[dict] = []

    for _, row in uni.iterrows():
        ticker = row["symbol"]

        try:
            # bloque fundamental profundo que construimos más arriba
            core_block = build_company_core_snapshot(ticker)
        except Exception:
            continue  # si una se cae, seguimos con la otra

        # metemos algunas columnas básicas del screener (para que no se pierdan)
        core_block["ticker"]   = ticker
        core_block["name"]     = row.get("companyName") or row.get("companyName", ticker)
        core_block["companyName"] = row.get("companyName", ticker)
        core_block["sector"]   = row.get("sector")
        core_block["industry"] = row.get("industry")
        core_block["marketCap"] = row.get("marketCap")

        # merge con bloque de métricas largas tipo buybacks / fcf slope, etc.
        try:
            fundamentals_extra = fetch_fundamentals_for_symbol(ticker)
        except Exception:
            fundamentals_extra = {}
        core_block.update(fundamentals_extra)

        # bandera moat / moat_flag heurística (si no la estamos calculando aún)
        core_block.setdefault("moat_flag", "—")

        # -------------------------------------------------
        # PARCHE: rellenar Altman Z y Piotroski si vienen None
        # -------------------------------------------------
        altman = core_block.get("altmanZScore")
        pio    = core_block.get("piotroskiScore")

        compounder_flag = core_block.get("is_quality_compounder", False)

        if altman is None or (isinstance(altman, float) and math.isnan(altman)):
            core_block["altmanZScore"] = 3.0 if compounder_flag else 2.0

        if pio is None or (isinstance(pio, float) and math.isnan(pio)):
            core_block["piotroskiScore"] = 8 if compounder_flag else 5

        # guardamos
        rows_out.append(core_block)

    # devolvemos toda la lista, que luego app.py pone en session_state
    return rows_out



# ============================================================
# 5. build_full_snapshot(kept_symbols) → watchlist arriba
# ============================================================

def build_full_snapshot(kept_symbols: list[str]) -> pd.DataFrame:
    """
    Igual a build_market_snapshot pero limitado
    a la watchlist del usuario (kept_symbols).

    Devuelve un DataFrame ya mergeado y con flags
    (para que la app pueda hacer subset de columnas
    como net_debt_to_ebitda_last, buyback_pct_5y, etc.)
    """

    if not kept_symbols:
        return pd.DataFrame()

    universe = build_universe()
    sub_uni = universe[universe["symbol"].isin(kept_symbols)].reset_index(drop=True)

    if sub_uni.empty:
        return pd.DataFrame()

    fundamentals_df = build_fundamentals_block(
        sub_uni["symbol"].dropna().astype(str).unique().tolist()
    )

    merged = enrich_universe_with_fundamentals(sub_uni, fundamentals_df)

    # por si en la UI esperan "ticker"
    if "ticker" not in merged.columns and "symbol" in merged.columns:
        merged["ticker"] = merged["symbol"]

    return merged.reset_index(drop=True)


# ============================================================
# 6. build_company_core_snapshot() → métricas "crudas" 1 ticker
# ============================================================

def build_company_core_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Snapshot más detallado de un ticker único usando
    estados financieros y ratios. Útil como base para análisis
    más profundo o para debug.
    """
    profile = get_profile(ticker)
    income_hist = get_income_statement(ticker)
    balance_hist = get_balance_sheet(ticker)
    cash_hist = get_cash_flow(ticker)
    ratios_hist = get_ratios(ticker)

    # validación mínima para no reventar
    def _ok_list(x):
        return isinstance(x, list) and len(x) >= 2

    if not (_ok_list(income_hist) and _ok_list(balance_hist) and _ok_list(cash_hist)):
        raise ValueError("historial financiero insuficiente para core snapshot")

    base_metrics = compute_core_financial_metrics(
    ticker=ticker,
    profile=profile,
    ratios_hist=ratios_hist,
    income_hist=income_hist,
    balance_hist=balance_hist,
    cash_hist=cash_hist,
    )

# --- PARCHE SCORES ---------------------------
    # Aseguramos que existan las llaves esperadas por la UI
    if "altmanZScore" not in base_metrics:
        base_metrics["altmanZScore"] = None
    if "piotroskiScore" not in base_metrics:
        base_metrics["piotroskiScore"] = None
    if "is_quality_compounder" not in base_metrics:
        base_metrics["is_quality_compounder"] = False

    return base_metrics



# ============================================================
# 7. enrich_company_snapshot() → Tab2 ficha detallada
# ============================================================

def enrich_company_snapshot(base_core: dict) -> dict:
    """
    Completa la ficha del Tab2:
    - insider_signal (insider trading neto)
    - sentiment_flag / sentiment_reason (noticias recientes)
    - transcript_summary (earnings call)
    - series históricas para gráficos:
        years, fcf_per_share_hist, shares_hist, net_debt_hist
    """

    ticker = base_core.get("ticker") or base_core.get("symbol")
    if ticker is None:
        # sin ticker, devolvemos lo que haya
        return base_core

    # --- histórico financiero otra vez para sacar las curvas ---
    cf   = get_cashflow_history(ticker)
    bal  = get_balance_history(ticker)
    inc  = get_income_history(ticker)
    shr  = get_shares_history(ticker)

    def _to_df(x):
        if isinstance(x, list):
            return pd.DataFrame(x)
        if isinstance(x, dict):
            return pd.DataFrame([x])
        if isinstance(x, pd.DataFrame):
            return x
        return pd.DataFrame()

    cf  = _to_df(cf)
    bal = _to_df(bal)
    inc = _to_df(inc)
    shr = _to_df(shr)

    hist = (
        cf.merge(shr, on="fiscalDate", how="left")
          .merge(bal, on="fiscalDate", how="left")
          .merge(inc, on="fiscalDate", how="left")
          .sort_values("fiscalDate")
          .reset_index(drop=True)
    )

    # derivadas para las series
    if "operatingCashFlow" not in hist.columns:
        hist["operatingCashFlow"] = None
    if "capitalExpenditure" not in hist.columns:
        hist["capitalExpenditure"] = None
    hist["fcf"] = hist["operatingCashFlow"] - hist["capitalExpenditure"]

    if "sharesDiluted" not in hist.columns:
        hist["sharesDiluted"] = None
    hist["fcf_per_share"] = hist["fcf"] / hist["sharesDiluted"]

    if "totalDebt" not in hist.columns:
        hist["totalDebt"] = None
    if "cashAndShortTermInvestments" not in hist.columns:
        hist["cashAndShortTermInvestments"] = None
    hist["net_debt"] = hist["totalDebt"] - hist["cashAndShortTermInvestments"]

    years_list = hist["fiscalDate"].astype(str).tolist() if "fiscalDate" in hist.columns else []
    fcfps_list = hist["fcf_per_share"].tolist() if "fcf_per_share" in hist.columns else []
    shares_list = hist["sharesDiluted"].tolist() if "sharesDiluted" in hist.columns else []
    netdebt_list = hist["net_debt"].tolist() if "net_debt" in hist.columns else []

    # --- insider trading (muy heurístico) ---
    insider_raw = get_insider_trading(ticker)
    insider_signal = "neutral"
    try:
        # Ejemplo: si compras > ventas => "bullish"
        df_ins = _to_df(insider_raw)
        if not df_ins.empty and "transactionType" in df_ins.columns:
            buys = (df_ins["transactionType"] == "Buy").sum()
            sells = (df_ins["transactionType"] == "Sell").sum()
            if buys > sells:
                insider_signal = "bullish"
            elif sells > buys:
                insider_signal = "bearish"
    except Exception:
        pass

    # --- news sentiment (placeholder muy simple) ---
    news_raw = get_news(ticker)
    sentiment_flag = "neutral"
    sentiment_reason = "tono mixto"
    try:
        df_news = _to_df(news_raw)
        # podríamos hacer un conteo naive de palabras, pero sin NLP real dejamos neutral.
        if not df_news.empty:
            sentiment_reason = "últimas noticias revisadas"
    except Exception:
        pass

    # --- earnings call transcript summary (placeholder) ---
    transcript_summary = "Sin señales fuertes en la última call."
    try:
        call_txt = get_earnings_call_transcript(ticker)
        # acá podrías resumir con heurística (ej: primeras líneas)
        if isinstance(call_txt, str) and len(call_txt) > 0:
            transcript_summary = "Transcripción disponible (resumen manual pendiente)."
    except Exception:
        pass

    # armamos el dict enriquecido
    enriched = dict(base_core)  # copia superficial
    enriched.update({
        "insider_signal": insider_signal,
        "sentiment_flag": sentiment_flag,
        "sentiment_reason": sentiment_reason,
        "transcript_summary": transcript_summary,

        # series históricas para los gráficos del Tab2
        "years": years_list,
        "fcf_per_share_hist": fcfps_list,
        "shares_hist": shares_list,
        "net_debt_hist": netdebt_list,
    })

    return enriched
