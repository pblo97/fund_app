from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd

from config import MAX_NET_DEBT_TO_EBITDA

# FMP accessors que ya tienes en fmp_api.py
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

from metrics import compute_core_metrics

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
    Asegura que run_screener_for_exchange(...) termine siendo DataFrame.
    Soporta:
      - list[dict]
      - dict con key "data"
      - DataFrame
    """
    if chunk is None:
        return pd.DataFrame()

    if isinstance(chunk, pd.DataFrame):
        return chunk.copy()

    if isinstance(chunk, list):
        return pd.DataFrame(chunk)

    if isinstance(chunk, dict):
        # caso tipo {"data":[...]}
        if "data" in chunk and isinstance(chunk["data"], list):
            return pd.DataFrame(chunk["data"])
        # caso dict plano con info de 1 ticker
        return pd.DataFrame([chunk])

    # fallback
    return pd.DataFrame()


def build_universe() -> pd.DataFrame:
    """
    Une los screeners de cada exchange, normaliza columnas,
    filtra solo large caps y devuelve symbol, companyName, sector, industry, marketCap.
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

    # renombrar name -> companyName si es necesario
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

    # limpiar duplicados
    uni = (
        uni.drop_duplicates(subset=["symbol"])
           .reset_index(drop=True)
    )

    # filtrar large cap
    mask_large = uni.apply(_is_large_cap_row, axis=1)
    uni = uni[mask_large].reset_index(drop=True)

    # columnas mínimas
    for col in ["symbol", "companyName", "sector", "industry", "marketCap"]:
        if col not in uni.columns:
            uni[col] = None

    return uni[["symbol", "companyName", "sector", "industry", "marketCap"]].copy()


def build_company_row(symbol: str) -> Dict[str, Any] | None:
    """
    Para un ticker:
    - Descarga los estados financieros crudos (histórico anual).
    - Llama compute_core_metrics(...) para consolidar todo.
    - Devuelve un dict con todas las llaves que app.py y dataframe_from_rows() esperan.
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

    # sanity mínimo: necesitamos al menos 2 años para calcular growth
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

    # fila final coherente con app.py/dataframe_from_rows()
    return {
        "ticker": core["ticker"],
        "name": core["name"] or core["ticker"],
        "companyName": core["name"] or core["ticker"],
        "sector": core["sector"],
        "industry": core["industry"],
        "marketCap": core["marketCap"],

        "altmanZScore": core["altmanZScore"],
        "piotroskiScore": core["piotroskiScore"],

        "revenueGrowth": core["revenueGrowth"],
        "operatingCashFlowGrowth": core["operatingCashFlowGrowth"],
        "freeCashFlowGrowth": core["freeCashFlowGrowth"],
        "debtGrowth": core["debtGrowth"],

        "rev_CAGR_5y": core["rev_CAGR_5y"],
        "ocf_CAGR_5y": core["ocf_CAGR_5y"],

        "netDebt_to_EBITDA": core["netDebt_to_EBITDA"],

        "buyback_pct_5y": core["buyback_pct_5y"],
        "fcf_per_share_slope_5y": core["fcf_per_share_slope_5y"],
        "is_quality_compounder": core["is_quality_compounder"],

        "moat_flag": core["moat_flag"],
        "business_summary": core["business_summary"],

        # series históricas para Tab2
        "years": core["years"],
        "fcf_per_share_hist": core["fcf_per_share_hist"],
        "shares_hist": core["shares_hist"],
        "net_debt_hist": core["net_debt_hist"],
    }


def build_market_snapshot() -> List[Dict[str, Any]]:
    """
    1. arma universo large cap
    2. recorre cada symbol
    3. junta las filas válidas
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

        # (opcional) podríamos filtrar por apalancamiento aquí con MAX_NET_DEBT_TO_EBITDA,
        # pero no es obligatorio porque la UI igual aplica el slider leverage_ok.
        out.append(snap)

    return out


def enrich_company_snapshot(base_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriquecimiento cualitativo para Tab2.
    Este paso usa insiders, noticias y earnings call,
    y hace resúmenes de texto.

    Hacemos import local de text_analysis para no romper el import global.
    Si text_analysis falla, devolvemos defaults y seguimos.
    """
    symbol = base_row.get("ticker") or base_row.get("symbol")
    if not symbol:
        return base_row

    # intentamos importar análisis de texto acá adentro
    try:
        from text_analysis import (
            summarize_business,
            summarize_news_sentiment,
            summarize_insiders,
            summarize_transcript,
        )
    except Exception:
        summarize_business = lambda prof_or_desc: base_row.get("business_summary", "")
        summarize_news_sentiment = lambda news: ("neutral", "tono mixto/sectorial")
        summarize_insiders = lambda insider: ("neutral", "")
        summarize_transcript = lambda txt: "Sin señales fuertes en la última call."

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

    # Si no teníamos resumen de negocio legible, lo intentamos generar
    if not enriched.get("business_summary"):
        enriched["business_summary"] = summarize_business(
            {"description": base_row.get("business_summary", "")}
        )

    return enriched
