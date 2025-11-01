# orchestrator.py
#
# - Arma el universo con screener NASDAQ/NYSE
# - Filtra ROE mínimo
# - Para cada ticker que pasa:
#   descarga perfil, estados financieros, insiders, noticias, transcripts
#   calcula métricas cuantitativas y cualitativas
# - Devuelve lista de snapshots (dicts)


from typing import List, Dict, Any
import traceback

from config import MIN_ROE_TTM
from fmp_api import (
    run_screener_for_exchange,
    get_ratios,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_profile,
    get_insider_trading,
    get_news,
    get_earnings_call_transcript,
)
from metrics import compute_core_financial_metrics, extract_roe_ttm
from text_analysis import (
    summarize_insiders,
    summarize_news_sentiment,
    summarize_transcript,
    infer_core_risk,
    infer_why_it_matters,
)


def build_universe() -> list[str]:
    base_list = []

    for exch in ["NASDAQ", "NYSE"]:
        chunk = run_screener_for_exchange(exch)
        base_list.extend(chunk)

    seen = {}
    for row in base_list:
        sym = row.get("symbol")
        if sym and sym not in seen:
            seen[sym] = row

    tickers = list(seen.keys())

    filtered_tickers = []

    for t in tickers:
        try:
            ratios_hist = get_ratios(t)
            roe_ttm = extract_roe_ttm(ratios_hist)
            if roe_ttm is None or roe_ttm < MIN_ROE_TTM:
                continue
            filtered_tickers.append((t, roe_ttm))
        except Exception:
            continue

    # 🔥 NUEVO: priorizar las mejores empresas primero
    # ordenamos por ROE descendente y nos quedamos con las 30 mejores
    filtered_tickers.sort(key=lambda x: (x[1] if x[1] is not None else 0), reverse=True)
    top_symbols = [t for (t, _) in filtered_tickers[:30]]

    return top_symbols

# en orchestrator.py

def build_company_core_snapshot(ticker: str) -> dict:
    profile = get_profile(ticker)
    income_hist = get_income_statement(ticker)
    balance_hist = get_balance_sheet(ticker)
    cash_hist = get_cash_flow(ticker)
    ratios_hist = get_ratios(ticker)

    if len(income_hist) < 3 or len(balance_hist) < 3 or len(cash_hist) < 3:
        raise ValueError("historial financiero insuficiente")

    base_metrics = compute_core_financial_metrics(
        ticker=ticker,
        profile=profile,
        ratios_hist=ratios_hist,
        income_hist=income_hist,
        balance_hist=balance_hist,
        cash_hist=cash_hist
    )

    # todavía sin insiders/news/transcript
    return base_metrics


def enrich_company_snapshot(snapshot: dict) -> dict:
    ticker = snapshot["ticker"]

    insider_trades = get_insider_trading(ticker)
    news_list = get_news(ticker)
    transcripts = get_earnings_call_transcript(ticker)

    from text_analysis import (
        summarize_insiders,
        summarize_news_sentiment,
        summarize_transcript,
        infer_core_risk,
        infer_why_it_matters,
    )

    insider_signal = summarize_insiders(insider_trades)
    sentiment_flag, sentiment_reason = summarize_news_sentiment(news_list)
    transcript_summary = summarize_transcript(transcripts)
    why_matters = infer_why_it_matters(
        sector=snapshot["sector"],
        industry=snapshot["industry"],
        moat_flag=snapshot["moat_flag"],
        beta=snapshot["beta"],
    )
    core_risk = infer_core_risk(
        net_debt_to_ebitda=snapshot["netDebt_to_EBITDA"],
        sentiment_flag=sentiment_flag,
        sentiment_reason=sentiment_reason,
    )

    snapshot["insider_signal"] = insider_signal
    snapshot["sentiment_flag"] = sentiment_flag
    snapshot["sentiment_reason"] = sentiment_reason
    snapshot["why_it_matters"] = why_matters
    snapshot["core_risk_note"] = core_risk
    snapshot["transcript_summary"] = transcript_summary

    return snapshot



def build_full_snapshot() -> list[dict]:
    final_rows = []
    universe = build_universe()  # ahora ya viene reducido a top ~30

    for tkr in universe:
        try:
            snap_core = build_company_core_snapshot(tkr)
            # sin enrich todavía
            final_rows.append(snap_core)
        except Exception:
            continue

    return final_rows


if __name__ == "__main__":
    # Ejecución directa rápida
    rows = build_full_snapshot()
    print(f"Tickers procesados: {len(rows)}")
    if rows:
        from pprint import pprint
        pprint(rows[0])
