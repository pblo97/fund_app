# orchestrator.py
#
# Flujo que consume la app:
#
# 1. build_universe()
#    -> DataFrame con tickers large cap (NYSE/NASDAQ/AMEX).
#
# 2. build_company_core_snapshot(symbol)
#    -> dict con todas las llaves que app.py espera en cada fila:
#       identidad, deuda, buybacks, slope FCF/acción, etc.
#       OJO: compute_core_financial_metrics() (nuevo) ya NO trae
#       Altman Z, Piotroski, growth ni CAGR. Esos campos los
#       rellenamos con None.
#
# 3. build_market_snapshot()
#    -> recorre el universo y devuelve lista[dict] lista para Tab1.
#
# 4. enrich_company_snapshot(base_row)
#    -> agrega insiders / news / transcript para Tab2.
#

from __future__ import annotations

from typing import List, Dict, Any
import math
import numpy as np
import pandas as pd

from config import MAX_NET_DEBT_TO_EBITDA

# ---- API FMP ----
from fmp_api import (
    run_screener_for_exchange,
    get_profile,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_ratios,
    get_cashflow_history,
    get_balance_history,
    get_income_history,
    get_shares_history,
    get_insider_trading,
    get_news,
    get_earnings_call_transcript,
)

# ---- NLP / heurísticas cualitativas ----
# Hacemos el import "best-effort". Si falla, definimos dummies.
try:
    from text_analysis import (
        summarize_business,
        flag_moat,
        summarize_transcript,
        summarize_news_sentiment,
        summarize_insiders,
    )
except Exception:
    def summarize_business(_profile_like):
        return ""

    def flag_moat(_profile_like):
        # si tenemos moat heurístico desde metrics igual,
        # esto solo se usaría como fallback
        return "—"

    def summarize_transcript(_txt):
        return ""

    def summarize_news_sentiment(_news_list):
        # devolvemos bandera neutral y mini-explicación
        return ("neutral", "tono mixto/sectorial")

    def summarize_insiders(_insider_raw):
        # (signal, note)
        return ("neutral", "")

# ---- métricas cuantitativas core ----
from metrics import compute_core_financial_metrics


# =========================================================
# Helpers internos
# =========================================================

EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]


def _is_large_cap(row: Dict[str, Any], min_mktcap: float = 10_000_000_000) -> bool:
    """
    Filtro large cap >= 10B USD.
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


def _linear_trend(values: pd.Series | list[float]) -> float | None:
    """
    Pendiente lineal (slope) sobre una serie temporal simple.
    Devuelve None si hay <2 puntos válidos.
    """
    s = pd.Series(values).dropna()
    if len(s) < 2:
        return None
    x = np.arange(len(s), dtype=float)
    y = s.astype(float).values
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _safe_div(a, b):
    try:
        if b is None or float(b) == 0:
            return None
        return float(a) / float(b)
    except Exception:
        return None


def _fetch_histories(symbol: str) -> pd.DataFrame:
    """
    Descarga historia anual de cashflow/balance/income/shares,
    la alinea por fiscalDate ascendente (viejo -> nuevo),
    y deriva columnas auxiliares:
      - fcf (OCF - CapEx)
      - fcf_per_share
      - net_debt
      - ebitda
      - sharesDiluted
    """
    cf = get_cashflow_history(symbol)
    bal = get_balance_history(symbol)
    inc = get_income_history(symbol)
    shr = get_shares_history(symbol)

    hist = (
        pd.merge(cf, shr, on="fiscalDate", how="outer")
          .merge(bal, on="fiscalDate", how="outer")
          .merge(inc, on="fiscalDate", how="outer")
    )

    hist = hist.sort_values("fiscalDate").reset_index(drop=True)

    # FCF y FCF/acción
    hist["fcf"] = hist["operatingCashFlow"] - hist["capitalExpenditure"]
    hist["fcf_per_share"] = _safe_div(hist["fcf"], hist["sharesDiluted"])

    # Net Debt = totalDebt - cashAndShortTermInvestments
    hist["net_debt"] = hist["totalDebt"] - hist["cashAndShortTermInvestments"]

    return hist


def _compute_buyback_pct_5y(hist: pd.DataFrame) -> float | None:
    """
    Aproximación de recompras netas:
    (shares_start - shares_end)/shares_start.
    """
    if hist is None or hist.empty or "sharesDiluted" not in hist.columns:
        return None
    shares = hist["sharesDiluted"].dropna()
    if len(shares) < 2:
        return None
    start = float(shares.iloc[0])
    end   = float(shares.iloc[-1])
    if start == 0:
        return None
    return (start - end) / start


def _compute_net_debt_change_5y(hist: pd.DataFrame) -> float | None:
    """
    Cambio absoluto en deuda neta entre primer y último punto.
    """
    if hist is None or hist.empty or "net_debt" not in hist.columns:
        return None
    nd = hist["net_debt"].dropna()
    if len(nd) < 2:
        return None
    return float(nd.iloc[-1]) - float(nd.iloc[0])


def _compute_last_net_debt_to_ebitda(hist: pd.DataFrame) -> float | None:
    """
    Usa el último punto disponible con net_debt y ebitda.
    """
    if hist is None or hist.empty:
        return None
    last_valid = hist.dropna(subset=["net_debt", "ebitda"]).iloc[-1:]
    if last_valid.empty:
        return None
    nd = last_valid["net_debt"].values[0]
    eb = last_valid["ebitda"].values[0]
    if eb is None or eb == 0:
        return None
    try:
        return float(nd) / float(eb)
    except Exception:
        return None


def _flag_positive(x):
    """
    True si x es numérico > 0.
    """
    if x is None:
        return False
    try:
        return float(x) > 0
    except Exception:
        return False


# =========================================================
# Paso 1: universo inicial
# =========================================================

def build_universe() -> pd.DataFrame:
    """
    Descarga screener por cada exchange, concatena, dedup,
    y filtra solo large caps.
    """
    frames: List[pd.DataFrame] = []
    for exch in EXCHANGES:
        try:
            chunk = run_screener_for_exchange(exch)
            if chunk is None:
                continue
            frames.append(chunk)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["symbol", "companyName", "sector", "industry", "marketCap"])

    uni_raw = pd.concat(frames, ignore_index=True)

    # normalizar nombre
    if "name" in uni_raw.columns and "companyName" not in uni_raw.columns:
        uni_raw = uni_raw.rename(columns={"name": "companyName"})

    # dedupe
    uni_raw = (
        uni_raw
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )

    # filtrar large cap
    mask_large = uni_raw.apply(_is_large_cap, axis=1)
    uni_large = uni_raw[mask_large].reset_index(drop=True)

    return uni_large


# =========================================================
# Paso 2: snapshot core por empresa
# =========================================================

def build_company_core_snapshot(symbol: str) -> Dict[str, Any] | None:
    """
    Para 1 ticker:
    - baja info cruda de FMP
    - llama compute_core_financial_metrics() (nuevo), que SOLO da:
        * ticker, name, sector, industry, marketCap, beta
        * business_summary
        * netDebt_to_EBITDA
        * moat_flag
        * years / fcf_per_share_hist / shares_hist / net_debt_hist
    - calcula buybacks, slope FCF/acción, net_debt_to_ebitda_last, etc.
    - rellena AltmanZScore, Piotroski, growth, CAGR con None.
    """

    # 1. bajar data cruda para compute_core_financial_metrics
    try:
        profile_data      = get_profile(symbol)
        income_hist_raw   = get_income_statement(symbol)
        balance_hist_raw  = get_balance_sheet(symbol)
        cash_hist_raw     = get_cash_flow(symbol)
        ratios_hist_raw   = get_ratios(symbol)
    except Exception:
        return None

    # sanity mínima
    if (
        not isinstance(income_hist_raw, list) or len(income_hist_raw) < 1 or
        not isinstance(balance_hist_raw, list) or len(balance_hist_raw) < 1 or
        not isinstance(cash_hist_raw, list) or len(cash_hist_raw) < 1
    ):
        return None

    # 2. métricas core básicas
    core = compute_core_financial_metrics(
        ticker=symbol,
        profile=profile_data,
        ratios_hist=ratios_hist_raw,
        income_hist=income_hist_raw,
        balance_hist=balance_hist_raw,
        cash_hist=cash_hist_raw,
    )

    # 3. históricos detallados
    hist_df = _fetch_histories(symbol)

    fcf_slope_5y = None
    buyback_pct_5y = None
    net_debt_change_5y = None
    net_debt_to_ebitda_last = None

    # series que enviaremos al Tab2
    years_series = core.get("years", [])
    fcfps_series = core.get("fcf_per_share_hist", [])
    shares_series = core.get("shares_hist", [])
    net_debt_series = core.get("net_debt_hist", [])

    if hist_df is not None and not hist_df.empty:
        # slope FCF/acción
        if "fcf_per_share" in hist_df.columns:
            fcf_slope_5y = _linear_trend(hist_df["fcf_per_share"])

        # recompras (% reducción acciones)
        buyback_pct_5y = _compute_buyback_pct_5y(hist_df)

        # cambio deuda neta
        net_debt_change_5y = _compute_net_debt_change_5y(hist_df)

        # último ND/EBITDA del hist real
        net_debt_to_ebitda_last = _compute_last_net_debt_to_ebitda(hist_df)

        # sobrescribimos series con hist_df cronológico limpio
        years_series = hist_df["fiscalDate"].astype(str).tolist()
        fcfps_series = (
            hist_df["fcf_per_share"]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .tolist()
        )
        shares_series = (
            hist_df["sharesDiluted"]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .tolist()
        )
        net_debt_series = (
            hist_df["net_debt"]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .tolist()
        )

    # quality flag
    flag_fcf_up = _flag_positive(fcf_slope_5y)
    flag_buybacks = (
        buyback_pct_5y is not None
        and not (isinstance(buyback_pct_5y, float) and math.isnan(buyback_pct_5y))
        and buyback_pct_5y > 0.05
    )
    flag_net_debt_ok = (
        net_debt_to_ebitda_last is not None
        and not (isinstance(net_debt_to_ebitda_last, float) and math.isnan(net_debt_to_ebitda_last))
        and net_debt_to_ebitda_last < 2
    )
    is_quality_compounder = bool(flag_fcf_up and flag_buybacks and flag_net_debt_ok)

    # 4. fila final con TODAS las llaves que app.py espera
    row: Dict[str, Any] = {
        # identidad
        "ticker":            core.get("ticker", symbol),
        "name":              core.get("name", ""),
        "companyName":       core.get("name", ""),
        "sector":            core.get("sector", "—"),
        "industry":          core.get("industry", "—"),
        "marketCap":         core.get("marketCap"),
        "beta":              core.get("beta"),
        "business_summary":  core.get("business_summary", ""),

        # scores (placeholder porque metrics nuevo no calcula esto)
        "altmanZScore":      None,
        "piotroskiScore":    None,

        # growth / CAGR (placeholder)
        "revenueGrowth":            None,
        "operatingCashFlowGrowth":  None,
        "freeCashFlowGrowth":       None,
        "debtGrowth":               None,
        "rev_CAGR_5y":              None,
        "rev_CAGR_3y":              None,
        "ocf_CAGR_5y":              None,
        "ocf_CAGR_3y":              None,

        # deuda / leverage
        "netDebt_to_EBITDA":       core.get("netDebt_to_EBITDA"),
        "net_debt_to_ebitda_last": net_debt_to_ebitda_last,
        "net_debt_change_5y":      net_debt_change_5y,

        # recompras / FCF/acción
        "fcf_per_share_slope_5y":  fcf_slope_5y,
        "buyback_pct_5y":          buyback_pct_5y,

        # moat heurístico
        "moat_flag":               core.get("moat_flag", "—"),

        # flag compuesto tipo "✅ COMPOUNDER"
        "is_quality_compounder":   is_quality_compounder,

        # notas cualitativas básicas (se llenan mejor en enrich_company_snapshot)
        "why_it_matters":          "",
        "core_risk_note":          "",

        # series históricas para Tab2
        "years":                years_series,
        "fcf_per_share_hist":   fcfps_series,
        "shares_hist":          shares_series,
        "net_debt_hist":        net_debt_series,
    }

    return row


# =========================================================
# Paso 3: shortlist de mercado (Tab1)
# =========================================================

def build_market_snapshot() -> List[Dict[str, Any]]:
    """
    1. arma universo large caps
    2. corre build_company_core_snapshot() para cada symbol
    3. devuelve lista[dict] lista para la UI
    """
    uni = build_universe()
    rows: List[Dict[str, Any]] = []

    for _idx, r in uni.iterrows():
        sym = str(r.get("symbol"))
        if not sym:
            continue

        snap = build_company_core_snapshot(sym)
        if snap is None:
            continue

        # podríamos filtrar apalancadas acá, pero dejamos que la UI lo haga
        rows.append(snap)

    return rows


# =========================================================
# Paso 4: enriquecimiento cualitativo (Tab2)
# =========================================================

def enrich_company_snapshot(base_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recibe la fila base y agrega:
      - insiders
      - sentimiento prensa
      - resumen earnings call
    Si text_analysis falló al importar, usamos dummies neutras.
    """
    symbol = base_row.get("ticker") or base_row.get("symbol")
    if not symbol:
        return base_row

    # insiders
    try:
        insider_raw = get_insider_trading(symbol)
    except Exception:
        insider_raw = []
    insider_signal, insider_note = summarize_insiders(insider_raw)

    # news / sentimiento
    try:
        news_raw = get_news(symbol)
    except Exception:
        news_raw = []
    sentiment_flag, sentiment_reason = summarize_news_sentiment(news_raw)

    # earnings call transcript
    try:
        transcript_txt = get_earnings_call_transcript(symbol)
    except Exception:
        transcript_txt = ""
    transcript_summary = summarize_transcript(transcript_txt)

    enriched = dict(base_row)
    enriched["insider_signal"] = insider_signal or "neutral"
    enriched["insider_note"] = insider_note or ""
    enriched["sentiment_flag"] = sentiment_flag or "neutral"
    enriched["sentiment_reason"] = sentiment_reason or "tono mixto/sectorial"
    enriched["transcript_summary"] = (
        transcript_summary or "Sin señales fuertes en la última call."
    )

    return enriched
