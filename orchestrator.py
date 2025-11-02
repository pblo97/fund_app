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
from text_analysis import (
    summarize_business,
    summarize_news_sentiment,
    summarize_insiders,
    summarize_transcript,
)

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


def build_universe() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for exch in EXCHANGES:
        try:
            chunk = run_screener_for_exchange(exch)
        except Exception:
            continue

        if chunk is None:
            continue

        # normalizar a DataFrame
        if isinstance(chunk, list):
            df_chunk = pd.DataFrame(chunk)
        elif isinstance(chunk, dict):
            if "data" in chunk and isinstance(chunk["data"], list):
                df_chunk = pd.DataFrame(chunk["data"])
            else:
                df_chunk = pd.DataFrame([chunk])
        elif isinstance(chunk, pd.DataFrame):
            df_chunk = chunk
        else:
            continue

        if "symbol" not in df_chunk.columns:
            continue

        frames.append(df_chunk)

    if not frames:
        return pd.DataFrame(columns=["symbol","companyName","sector","industry","marketCap"])

    uni = pd.concat(frames, ignore_index=True)

    # renombrar name -> companyName si es necesario
    if "companyName" not in uni.columns and "name" in uni.columns:
        uni = uni.rename(columns={"name":"companyName"})

    # asegurar marketCap
    if "marketCap" not in uni.columns:
        for fallback in ["mktCap", "marketCapIntraday"]:
            if fallback in uni.columns:
                uni["marketCap"] = uni[fallback]
                break
        if "marketCap" not in uni.columns:
            uni["marketCap"] = None

    # limpiar duplicados
    uni = (
        uni
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )

    # filtrar large cap
    mask_large = uni.apply(_is_large_cap_row, axis=1)
    uni = uni[mask_large].reset_index(drop=True)

    # columnas mínimas
    for col in ["symbol","companyName","sector","industry","marketCap"]:
        if col not in uni.columns:
            uni[col] = None

    return uni[["symbol","companyName","sector","industry","marketCap"]].copy()


def build_company_row(symbol: str) -> Dict[str, Any] | None:
    """
    Baja toda la data cruda de FMP para un ticker,
    llama compute_core_metrics,
    y devuelve una fila lista para la tabla principal (Tab1),
    más info base que Tab2 usará.
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

    # sanity mínimo para no reventar
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

    # fila final coherente con app.py/dataframe_from_rows
    row: Dict[str, Any] = {
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

        # series para el Tab2
        "years": core["years"],
        "fcf_per_share_hist": core["fcf_per_share_hist"],
        "shares_hist": core["shares_hist"],
        "net_debt_hist": core["net_debt_hist"],
    }

    return row


def build_market_snapshot() -> List[Dict[str, Any]]:
    """
    Arma la shortlist para Tab1.
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

        # filtro suave de leverage (no las boto si están pasadas,
        # porque tú igual las filtras con el slider en la UI, pero
        # si quisieras cortarlas acá puedes hacerlo)
        nde = snap.get("netDebt_to_EBITDA")
        try:
            if nde is not None and float(nde) > float(MAX_NET_DEBT_TO_EBITDA):
                pass  # actualmente dejamos pasar igual
        except Exception:
            pass

        out.append(snap)

    return out


def enrich_company_snapshot(base_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tab2: agrega insiders, news sentiment y earnings call.
    """
    symbol = base_row.get("ticker") or base_row.get("symbol")
    if not symbol:
        return base_row

    # insiders
    try:
        insider_raw = get_insider_trading(symbol)
    except Exception:
        insider_raw = []
    ins_sig, ins_note = summarize_insiders(insider_raw)

    # news sentiment
    try:
        news_raw = get_news(symbol)
    except Exception:
        news_raw = []
    sent_flag, sent_reason = summarize_news_sentiment(news_raw)

    # transcript
    try:
        transcript_txt = get_earnings_call_transcript(symbol)
    except Exception:
        transcript_txt = ""
    call_sum = summarize_transcript(transcript_txt)

    enriched = dict(base_row)
    enriched["insider_signal"] = ins_sig or "neutral"
    enriched["insider_note"] = ins_note or ""
    enriched["sentiment_flag"] = sent_flag or "neutral"
    enriched["sentiment_reason"] = sent_reason or "tono mixto/sectorial"
    enriched["transcript_summary"] = (
        call_sum or "Sin señales fuertes en la última call."
    )

    # por si en algún caso base_row no traía un resumen legible
    if not enriched.get("business_summary"):
        enriched["business_summary"] = summarize_business({"description": base_row.get("business_summary", "")})

    return enriched
