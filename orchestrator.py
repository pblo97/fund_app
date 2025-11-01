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


def build_universe() -> List[str]:
    """
    1. Correr screener en NASDAQ y NYSE con filtros duros.
    2. Unir tickers únicos.
    3. Filtrar por ROE mínimo usando get_ratios.
    """
    base_list = []

    for exch in ["NASDAQ", "NYSE"]:
        chunk = run_screener_for_exchange(exch)
        base_list.extend(chunk)

    # remover duplicados por símbolo
    seen = {}
    for row in base_list:
        sym = row.get("symbol")
        if sym and sym not in seen:
            seen[sym] = row

    tickers = list(seen.keys())

    # ahora filtramos por ROE
    filtered_tickers = []
    for t in tickers:
        try:
            ratios_hist = get_ratios(t)
            roe_ttm = extract_roe_ttm(ratios_hist)
            if roe_ttm is None or roe_ttm < MIN_ROE_TTM:
                continue
            filtered_tickers.append(t)
        except Exception:
            # si falla ratios, lo saltamos
            continue

    return filtered_tickers


def build_company_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Para un ticker:
    - descargar profile, income, balance, cash, ratios, insiders, news, transcript
    - calcular métricas cuantitativas
    - calcular señales cualitativas
    - devolver dict final listo para mostrar en Streamlit
    """
    profile = get_profile(ticker)
    income_hist = get_income_statement(ticker)
    balance_hist = get_balance_sheet(ticker)
    cash_hist = get_cash_flow(ticker)
    ratios_hist = get_ratios(ticker)
    insider_trades = get_insider_trading(ticker)
    news_list = get_news(ticker)
    transcripts = get_earnings_call_transcript(ticker)

    # sanity básico: necesitamos historial suficiente
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

    # insiders
    insider_signal = summarize_insiders(insider_trades)

    # news sentiment
    sentiment_flag, sentiment_reason = summarize_news_sentiment(news_list)

    # transcript summary
    transcript_summary = summarize_transcript(transcripts)

    # why it matters / riesgo
    why_matters = infer_why_it_matters(
        sector=base_metrics["sector"],
        industry=base_metrics["industry"],
        moat_flag=base_metrics["moat_flag"],
        beta=base_metrics["beta"]
    )

    core_risk = infer_core_risk(
        net_debt_to_ebitda=base_metrics["netDebt_to_EBITDA"],
        sentiment_flag=sentiment_flag,
        sentiment_reason=sentiment_reason
    )

    # completar snapshot
    base_metrics["insider_signal"] = insider_signal
    base_metrics["sentiment_flag"] = sentiment_flag
    base_metrics["sentiment_reason"] = sentiment_reason
    base_metrics["why_it_matters"] = why_matters
    base_metrics["core_risk_note"] = core_risk
    base_metrics["transcript_summary"] = transcript_summary

    return base_metrics


def build_full_snapshot() -> List[Dict[str, Any]]:
    """
    Orquesta todo:
    - arma universo limpio
    - para cada ticker baja la foto completa
    - guarda cada snapshot en una lista (aún sin persistir)
    """
    final_rows: List[Dict[str, Any]] = []
    universe = build_universe()

    for tkr in universe:
        try:
            snap = build_company_snapshot(tkr)
            final_rows.append(snap)
        except Exception:
            traceback.print_exc()
            continue

    return final_rows


if __name__ == "__main__":
    # Ejecución directa rápida
    rows = build_full_snapshot()
    print(f"Tickers procesados: {len(rows)}")
    if rows:
        from pprint import pprint
        pprint(rows[0])
