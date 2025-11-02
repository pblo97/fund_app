from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd

from config import MAX_NET_DEBT_TO_EBITDA

# --- llamadas a FMP que ya tienes en fmp_api.py ---
from fmp_api import (
    run_screener_for_exchange,
    get_profile,
    get_income_statement,
    get_cash_flow,
    get_balance_sheet,
    get_ratios,
    get_shares_history,
    get_insider_trading,
    get_news,
    get_earnings_call_transcript,
)

# métrica cuantitativa consolidada (tu metrics.py nuevo expone esto)
from metrics import compute_core_metrics


# ===============================
# Helpers internos
# ===============================

EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]


def _is_large_cap_row(row: Dict[str, Any], min_mktcap: float = 10_000_000_000) -> bool:
    mc = (
        row.get("marketCap")
        or row.get("mktCap")
        or row.get("marketCapIntraday")
    )
    try:
        return float(mc) >= float(min_mktcap)
    except Exception:
        return False


def _normalize_chunk_to_df(chunk) -> pd.DataFrame:
    """
    Asegura que run_screener_for_exchange(...) termine siendo DataFrame,
    aunque FMP a veces responda lista o dict raro.
    """
    if chunk is None:
        return pd.DataFrame()

    if isinstance(chunk, pd.DataFrame):
        return chunk.copy()

    if isinstance(chunk, list):
        return pd.DataFrame(chunk)

    if isinstance(chunk, dict):
        # caso {"data":[...]}
        if "data" in chunk and isinstance(chunk["data"], list):
            return pd.DataFrame(chunk["data"])
        # caso dict simple (1 fila)
        return pd.DataFrame([chunk])

    # fallback
    return pd.DataFrame()


# ===============================
# 1. Universo large cap
# ===============================

def build_universe() -> pd.DataFrame:
    """
    Une screeners de cada exchange, limpia columnas,
    filtra solo large caps.
    Devuelve columnas mínimas:
    symbol, companyName, sector, industry, marketCap
    """
    frames: List[pd.DataFrame] = []

    for exch in EXCHANGES:
        try:
            raw = run_screener_for_exchange(exch)
        except Exception:
            raw = None

        df_chunk = _normalize_chunk_to_df(raw)
        if df_chunk.empty:
            continue
        if "symbol" not in df_chunk.columns:
            continue

        frames.append(df_chunk)

    if not frames:
        return pd.DataFrame(columns=["symbol", "companyName", "sector", "industry", "marketCap"])

    uni = pd.concat(frames, ignore_index=True)

    # renombrar 'name' -> 'companyName' si hace falta
    if "companyName" not in uni.columns and "name" in uni.columns:
        uni = uni.rename(columns={"name": "companyName"})

    # asegurar marketCap
    if "marketCap" not in uni.columns:
        for fb in ["mktCap", "marketCapIntraday"]:
            if fb in uni.columns:
                uni["marketCap"] = uni[fb]
                break
        if "marketCap" not in uni.columns:
            uni["marketCap"] = None

    # dedupe
    uni = (
        uni.drop_duplicates(subset=["symbol"])
           .reset_index(drop=True)
    )

    # filtrar large cap
    mask_large = uni.apply(_is_large_cap_row, axis=1)
    uni = uni[mask_large].reset_index(drop=True)

    # columnas mínimas garantizadas
    for col in ["symbol", "companyName", "sector", "industry", "marketCap"]:
        if col not in uni.columns:
            uni[col] = None

    return uni[["symbol", "companyName", "sector", "industry", "marketCap"]].copy()


# ===============================
# 2. Snapshot de UNA empresa
# ===============================

def build_company_row(symbol: str) -> Dict[str, Any] | None:
    """
    Descarga estados financieros crudos para un ticker
    y arma el diccionario consolidado que la app espera.
    """
    try:
        profile      = get_profile(symbol) or {}
        income_hist  = get_income_statement(symbol) or []
        cash_hist    = get_cash_flow(symbol) or []
        balance_hist = get_balance_sheet(symbol) or []
        ratios_hist  = get_ratios(symbol) or []
        shares_hist  = get_shares_history(symbol) or []
    except Exception:
        return None

    # si no hay data histórica mínima, salimos
    if len(income_hist) < 2 or len(cash_hist) < 2 or len(balance_hist) < 2:
        return None

    core = compute_core_metrics(
        ticker=symbol,
        profile=profile,
        income_hist=income_hist,
        cash_hist=cash_hist,
        balance_hist=balance_hist,
        ratios_hist=ratios_hist,
        shares_hist=shares_hist,
    )

    # devolvemos con las llaves que usa app.py/dataframe_from_rows()
    return {
        # identidad
        "ticker": core["ticker"],
        "name": core["name"] or core["ticker"],
        "companyName": core["name"] or core["ticker"],
        "sector": core["sector"],
        "industry": core["industry"],
        "marketCap": core["marketCap"],

        # scores de calidad/solvencia
        "altmanZScore": core["altmanZScore"],
        "piotroskiScore": core["piotroskiScore"],

        # crecimiento último año
        "revenueGrowth": core["revenueGrowth"],
        "operatingCashFlowGrowth": core["operatingCashFlowGrowth"],
        "freeCashFlowGrowth": core["freeCashFlowGrowth"],
        "debtGrowth": core["debtGrowth"],

        # CAGR multianual
        "rev_CAGR_5y": core["rev_CAGR_5y"],
        "ocf_CAGR_5y": core["ocf_CAGR_5y"],

        # apalancamiento
        "netDebt_to_EBITDA": core["netDebt_to_EBITDA"],

        # capital allocation / compounding
        "buyback_pct_5y": core["buyback_pct_5y"],
        "fcf_per_share_slope_5y": core["fcf_per_share_slope_5y"],
        "is_quality_compounder": core["is_quality_compounder"],

        # moat heurístico
        "moat_flag": core["moat_flag"],

        # descripción corta del negocio
        "business_summary": core["business_summary"],

        # series históricas para Tab2
        "years": core["years"],
        "fcf_per_share_hist": core["fcf_per_share_hist"],
        "shares_hist": core["shares_hist"],
        "net_debt_hist": core["net_debt_hist"],
    }


# ===============================
# 3. Shortlist de TODO el mercado (Tab1)
# ===============================

def build_market_snapshot() -> List[Dict[str, Any]]:
    """
    - arma universo large cap
    - para cada symbol construye una fila
    - devuelve lista[dict] (la app después la convierte a DataFrame bonito)
    """
    uni = build_universe()
    out: List[Dict[str, Any]] = []

    for _, r in uni.iterrows():
        sym = str(r["symbol"])
        if not sym:
            continue

        snap = build_company_row(sym)
        if snap is None:
            continue

        # opcional: si quisieras filtrar empresas ultra apalancadas acá,
        # podrías mirar snap["netDebt_to_EBITDA"] contra MAX_NET_DEBT_TO_EBITDA.
        out.append(snap)

    return out


# ===============================
# 4. Snapshot de tu watchlist (arriba en la app)
# ===============================

def build_full_snapshot(symbols: List[str]) -> pd.DataFrame:
    """
    Recibe ['AAPL','MSFT',...] y retorna un DataFrame
    con las mismas columnas que usa la grilla de la app.
    Esto es EXACTAMENTE lo que app.py espera para la watchlist.
    """
    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        snap = build_company_row(sym)
        if snap is not None:
            rows.append(snap)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ===============================
# 5. Enriquecer 1 empresa (Tab2 cualitativo)
# ===============================

def enrich_company_snapshot(base_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agrega señales cualitativas para el panel detalle:
    insiders, sentimiento de prensa y resumen de la earnings call.

    IMPORTANTE:
    Hacemos import local de text_analysis para no romper el import global.
    Si text_analysis falla, metemos defaults neutros.
    """
    symbol = base_row.get("ticker") or base_row.get("symbol")
    if not symbol:
        return base_row

    # import perezoso para evitar que la app explote al levantar
    try:
        from text_analysis import (
            summarize_business,
            summarize_news_sentiment,
            summarize_insiders,
            summarize_transcript,
        )
    except Exception:
        summarize_business = lambda prof_or_desc: base_row.get("business_summary", "")
        summarize_news_sentiment = (
            lambda news: ("neutral", "tono mixto/sectorial")
        )
        summarize_insiders = (
            lambda insider: ("neutral", "")
        )
        summarize_transcript = (
            lambda txt: "Sin señales fuertes en la última call."
        )

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

    # si business_summary venía vacío, intentamos fabricarlo
    if not enriched.get("business_summary"):
        enriched["business_summary"] = summarize_business(
            {"description": base_row.get("business_summary", "")}
        )

    return enriched
