# orchestrator.py
#
# NUEVA VERSIÓN:
# - Arma universo solo con large caps (>=10B) desde NASDAQ/NYSE/AMEX.
# - Filtra por solvencia/calidad (Altman Z alto, Piotroski alto).
# - Filtra por crecimiento sano (ventas, EBIT, FCF, OCF creciendo; deuda no subiendo)
#   y CAGR >= 15% anual compuesto (3y/5y) en revenue/OCF por acción.
#
# - SOLO para esa shortlist final:
#   descarga perfil, estados financieros, ratios
#   y construye métricas core (aún sin insiders/news/transcripts).
#
# - Devuelve snapshots listos para mostrar en la app.
#
# NOTA: Puedes después, bajo demanda (por ticker clickeado), llamar enrich_company_snapshot()
# para insiders/news/transcript y análisis cualitativo.

from typing import List, Dict, Any
import traceback
import pandas as pd

from fmp_api import (
    run_screener_for_exchange,
    get_financial_scores_batch,
    get_growth_batch,
    get_ratios,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_profile,
    get_insider_trading,
    get_news,
    get_earnings_call_transcript,
)
from metrics import compute_core_financial_metrics
from text_analysis import (
    summarize_insiders,
    summarize_news_sentiment,
    summarize_transcript,
    infer_core_risk,
    infer_why_it_matters,
)


# -------------------------------------------------
# 1. Construir universo filtrado (large cap + calidad + crecimiento)
# -------------------------------------------------

def build_universe_pipeline() -> pd.DataFrame:
    """
    Paso 1: screener por exchange (ya filtra large caps >=10B en fmp_api.run_screener_for_exchange)
    Paso 2: Financial Scores (Altman Z >=3, Piotroski >=7)
    Paso 3: Growth sano, deuda controlada, CAGR >=15% (get_growth_batch)

    Devuelve DataFrame final_candidates con columnas combinadas:
        symbol, companyName, sector, industry, marketCap, ...
        altmanZScore, piotroskiScore, ...
        revenueGrowth, operatingCashFlowGrowth, freeCashFlowGrowth,
        debtGrowth, rev_CAGR_5y, ocf_CAGR_5y, high_growth_flag, etc.
    """

    # --- Paso 1: screener NASDAQ/NYSE/AMEX ---
    universes = []
    for exch in ("NASDAQ", "NYSE", "AMEX"):
        try:
            data = run_screener_for_exchange(exch)
        except Exception:
            data = []
        if data:
            df_part = pd.DataFrame(data)
            universes.append(df_part)

    if not universes:
        return pd.DataFrame()

    base_universe = (
        pd.concat(universes, ignore_index=True)
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )

    # (opcional) si quieres sacar bancos/seguros porque distorsionan Piotroski/Altman:
    # if "sector" in base_universe.columns:
    #     base_universe = base_universe[
    #         ~base_universe["sector"].str.contains("Financial", case=False, na=False)
    #     ].reset_index(drop=True)

    if base_universe.empty:
        return pd.DataFrame()

    # --- Paso 2: Altman Z / Piotroski ---
    symbols_all = base_universe["symbol"].dropna().astype(str).unique().tolist()
    scores_df = get_financial_scores_batch(symbols_all)

    if scores_df.empty:
        return pd.DataFrame()

    merged_step2 = (
        base_universe.merge(scores_df, on="symbol", how="inner")
        .reset_index(drop=True)
    )
    if merged_step2.empty:
        return pd.DataFrame()

    # --- Paso 3: Growth sano + high_growth_flag (CAGR >=15%) ---
    symbols_step2 = merged_step2["symbol"].dropna().astype(str).unique().tolist()
    growth_df = get_growth_batch(symbols_step2)

    if growth_df.empty:
        return pd.DataFrame()

    final_candidates = (
        merged_step2.merge(growth_df, on="symbol", how="inner")
        .reset_index(drop=True)
    )

    # final_candidates ahora es tu shortlist "de verdad":
    # - Large cap
    # - Solvente / contablemente sana
    # - Creciendo ventas/EBIT/FCF/OCF
    # - Sin expansión de deuda
    # - CAGR >=15% en revenue/OCF por acción
    return final_candidates


# -------------------------------------------------
# 2. Snapshot financiero core por empresa
# -------------------------------------------------

def build_company_core_snapshot(ticker: str) -> dict:
    """
    Descarga perfil y estados financieros de 1 compañía,
    calcula métricas core cuantitativas (apalancamiento, márgenes, etc.).
    NOTA: no mete todavía insiders/news/transcripts.
    """
    profile = get_profile(ticker)
    income_hist = get_income_statement(ticker)
    balance_hist = get_balance_sheet(ticker)
    cash_hist = get_cash_flow(ticker)
    ratios_hist = get_ratios(ticker)

    # chequeo básico de profundidad de historial;
    # si no hay al menos 3 períodos anuales es difícil sacar estabilidad
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

    # base_metrics típicamente ya incluye:
    # - ticker, name, sector, industry
    # - márgenes, FCF margin, ROIC, leverage, netDebt/EBITDA
    # - moat_flag o similar si lo calculas ahí
    return base_metrics


# -------------------------------------------------
# 3. Enriquecer snapshot con señal cualitativa (bajo demanda)
# -------------------------------------------------

def enrich_company_snapshot(snapshot: dict) -> dict:
    """
    Añade info cualitativa y de flujo reciente:
    insiders, sentimiento de noticias, resumen de transcript de earnings,
    'por qué importa', y el riesgo central.
    Esto es caro en llamadas, así que úsalo solo cuando el usuario abra el detalle.
    """
    ticker = snapshot["ticker"]

    insider_trades = get_insider_trading(ticker)
    news_list = get_news(ticker)
    transcripts = get_earnings_call_transcript(ticker)

    insider_signal = summarize_insiders(insider_trades)
    sentiment_flag, sentiment_reason = summarize_news_sentiment(news_list)
    transcript_summary = summarize_transcript(transcripts)

    why_matters = infer_why_it_matters(
        sector=snapshot.get("sector"),
        industry=snapshot.get("industry"),
        moat_flag=snapshot.get("moat_flag"),
        beta=snapshot.get("beta"),
    )

    core_risk = infer_core_risk(
        net_debt_to_ebitda=snapshot.get("netDebt_to_EBITDA"),
        sentiment_flag=sentiment_flag,
        sentiment_reason=sentiment_reason,
    )

    snapshot["insider_signal"] = insider_signal
    snapshot["sentiment_flag"] = sentiment_flag
    snapshot["sentiment_reason"] = sentiment_reason
    snapshot["transcript_summary"] = transcript_summary
    snapshot["why_it_matters"] = why_matters
    snapshot["core_risk_note"] = core_risk

    return snapshot


# -------------------------------------------------
# 4. Construir snapshot completo para la app (solo core)
# -------------------------------------------------

def build_full_snapshot() -> List[dict]:
    """
    - Saca la shortlist final (large cap + calidad + crecimiento estructural).
    - Para cada ticker en esa shortlist, baja fundamentals y arma snapshot core.
    - NO enriquece con insiders/news/transcript para no quemar la API en masa.
    """
    final_rows: List[dict] = []

    universe_df = build_universe_pipeline()
    if universe_df.empty:
        return final_rows

    # la columna estándar que necesitamos es 'symbol'
    tickers_final = (
        universe_df["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for tkr in tickers_final:
        try:
            snap_core = build_company_core_snapshot(tkr)
            # adjuntamos algunas columnas clave del screening (altman z, piotroski, growth)
            row_screen = universe_df[universe_df["symbol"] == tkr].iloc[0].to_dict()

            snap_core["altmanZScore"] = row_screen.get("altmanZScore")
            snap_core["piotroskiScore"] = row_screen.get("piotroskiScore")
            snap_core["revenueGrowth"] = row_screen.get("revenueGrowth")
            snap_core["operatingCashFlowGrowth"] = row_screen.get("operatingCashFlowGrowth")
            snap_core["freeCashFlowGrowth"] = row_screen.get("freeCashFlowGrowth")
            snap_core["debtGrowth"] = row_screen.get("debtGrowth")
            snap_core["rev_CAGR_5y"] = row_screen.get("rev_CAGR_5y")
            snap_core["rev_CAGR_3y"] = row_screen.get("rev_CAGR_3y")
            snap_core["ocf_CAGR_5y"] = row_screen.get("ocf_CAGR_5y")
            snap_core["ocf_CAGR_3y"] = row_screen.get("ocf_CAGR_3y")
            snap_core["high_growth_flag"] = row_screen.get("high_growth_flag")

            final_rows.append(snap_core)

        except Exception:
            # si una empresa falla al bajar estados financieros,
            # seguimos con las demás en vez de romper todo
            traceback.print_exc()
            continue

    return final_rows


# -------------------------------------------------
# Ejecución directa para testing local
# -------------------------------------------------
if __name__ == "__main__":
    rows = build_full_snapshot()
    print(f"Tickers procesados: {len(rows)}")
    if rows:
        from pprint import pprint
        pprint(rows[0])
