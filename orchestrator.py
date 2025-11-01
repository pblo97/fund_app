# orchestrator.py
#
# Responsabilidades actuales (consistentes con app.py):
#
# 1. build_universe() -> dataframe base con tickers grandes/sanos desde el screener.
#
# 2. build_full_snapshot(kept_symbols: list[str]) -> DataFrame enriquecido SOLO
#    para los tickers que el usuario marcó en su watchlist ("kept").
#    Este DF se muestra arriba en la app ("Tu watchlist enriquecida").
#
# 3. build_market_snapshot() -> lista[dict] con una shortlist "global" del mercado,
#    con métricas financieras básicas y placeholders de calidad/growth.
#    Esto alimenta Tab1 (tabla filtrable) y Tab2 (selectbox).
#
# 4. enrich_company_snapshot(base_core: dict) -> dict con info cualitativa +
#    histórico multianual (para los 3 gráficos).
#
# IMPORTANTE:
# - No metemos todavía llamadas batch pesadas (news, transcript, etc.) salvo stub.
# - Mantener nombres de columnas EXACTOS como los usa la app.


from typing import List, Dict, Any
import math
import pandas as pd
import numpy as np

from config import MAX_NET_DEBT_TO_EBITDA

# -----------------------
# Dependencias FMP
# -----------------------
from fmp_api import (
    run_screener_for_exchange,
    get_profile,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_ratios,
    get_insider_trading,
    get_news,
    get_earnings_call_transcript,
    get_cashflow_history,
    get_balance_history,
    get_income_history,
    get_shares_history,
)

# =============================================================================
# Utilidades internas
# =============================================================================

EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]


def _is_large_cap(row: Dict[str, Any], min_mktcap: float = 10_000_000_000) -> bool:
    """
    Filtro de tamaño: large caps (>= 10B USD).
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
    Slope lineal simple por mínimos cuadrados.
    Retorna None si no hay datos suficientes.
    """
    s = pd.Series(values).dropna()
    if len(s) < 2:
        return None
    x = np.arange(len(s), dtype=float)
    y = s.astype(float).values
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope)


def _safe_pct_change(start_val, end_val) -> float | None:
    """
    (start - end)/start típico de recompras.
    Si falta algo o start==0, None.
    """
    try:
        if start_val is None or end_val is None:
            return None
        if float(start_val) == 0:
            return None
        return (float(start_val) - float(end_val)) / float(start_val)
    except Exception:
        return None


def _safe_last(series: pd.Series, col: str) -> Any:
    """
    Devuelve el último valor de una columna de un DataFrame ordenado.
    """
    if col not in series:
        return None
    return series[col]


# =============================================================================
# Paso 1: universo inicial
# =============================================================================

def build_universe() -> pd.DataFrame:
    """
    Descarga screener por exchange, concatena,
    limpia duplicados por symbol.
    """
    frames = []
    for exch in EXCHANGES:
        chunk = run_screener_for_exchange(exch)  # <- debe devolver DataFrame
        # asumimos que cada 'chunk' tiene al menos:
        # ["symbol", "companyName"/"name", "sector", "industry", "marketCap", ...]
        frames.append(chunk)

    if not frames:
        return pd.DataFrame()

    universe = pd.concat(frames, ignore_index=True)

    # limpiar duplicados por symbol
    universe = (
        universe
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )

    return universe


# =============================================================================
# Paso 2: fundamentales históricos por símbolo
# =============================================================================

def fetch_fundamentals_for_symbol(symbol: str) -> dict:
    """
    Baja históricos anuales (cashflow, balance, income, shares),
    hace merge por fiscalDate,
    y construye métricas estructurales para ese ticker:
      - fcf_per_share_slope_5y
      - buyback_pct_5y
      - net_debt_change_5y
      - net_debt_to_ebitda_last

    También devuelve series históricas que luego usaremos
    en enrich_company_snapshot() para graficar.
    """

    cf = get_cashflow_history(symbol)
    bal = get_balance_history(symbol)
    inc = get_income_history(symbol)
    shr = get_shares_history(symbol)

    # Merge anual por fiscalDate ascendente
    hist = (
        cf.merge(shr, on="fiscalDate", how="left")
          .merge(bal, on="fiscalDate", how="left")
          .merge(inc, on="fiscalDate", how="left")
          .sort_values("fiscalDate")
          .reset_index(drop=True)
    )

    # columnas esperadas:
    # operatingCashFlow, capitalExpenditure, sharesDiluted,
    # totalDebt, cashAndShortTermInvestments, ebitda
    # si alguna falta, creamos vacíos
    needed_cols = [
        "operatingCashFlow",
        "capitalExpenditure",
        "sharesDiluted",
        "totalDebt",
        "cashAndShortTermInvestments",
        "ebitda",
    ]
    for col in needed_cols:
        if col not in hist.columns:
            hist[col] = np.nan

    # métricas derivadas por fila
    hist["fcf"] = hist["operatingCashFlow"] - hist["capitalExpenditure"]
    hist["fcf_per_share"] = hist["fcf"] / hist["sharesDiluted"]
    hist["net_debt"] = hist["totalDebt"] - hist["cashAndShortTermInvestments"]

    # slope de FCF/acción en el tiempo
    fcfps_slope = _linear_trend(hist["fcf_per_share"])

    # recompras (% reducción acciones diluidas en ventana completa)
    shares_start = (
        hist["sharesDiluted"].iloc[0]
        if len(hist) > 0 else None
    )
    shares_end = (
        hist["sharesDiluted"].iloc[-1]
        if len(hist) > 0 else None
    )
    buyback_pct_5y = _safe_pct_change(shares_start, shares_end)

    # cambio deuda neta total en la ventana
    if len(hist) >= 2 and "net_debt" in hist.columns:
        net_debt_change_5y = (
            hist["net_debt"].iloc[-1]
            - hist["net_debt"].iloc[0]
        )
    else:
        net_debt_change_5y = None

    # último net_debt / EBITDA (último año disponible)
    if len(hist) > 0 and "ebitda" in hist.columns:
        last_row = hist.iloc[-1]
        nd_last = last_row.get("net_debt")
        ebitda_last = last_row.get("ebitda")
        try:
            if nd_last is not None and ebitda_last not in [None, 0]:
                nde_ratio = float(nd_last) / float(ebitda_last)
            else:
                nde_ratio = None
        except Exception:
            nde_ratio = None
    else:
        nde_ratio = None

    # Series para gráficos históricos
    years = hist["fiscalDate"].tolist() if "fiscalDate" in hist.columns else []
    fcfps_hist = hist["fcf_per_share"].tolist() if "fcf_per_share" in hist.columns else []
    shares_hist = hist["sharesDiluted"].tolist() if "sharesDiluted" in hist.columns else []
    net_debt_hist = hist["net_debt"].tolist() if "net_debt" in hist.columns else []

    return {
        "symbol": symbol,
        "fcf_per_share_slope_5y": fcfps_slope,
        "buyback_pct_5y": buyback_pct_5y,
        "net_debt_change_5y": net_debt_change_5y,
        "net_debt_to_ebitda_last": nde_ratio,

        # para gráficos / detalle posterior
        "years": years,
        "fcf_per_share_hist": fcfps_hist,
        "shares_hist": shares_hist,
        "net_debt_hist": net_debt_hist,
    }


def build_fundamentals_block(symbols: list[str]) -> pd.DataFrame:
    """
    Itera la lista de tickers elegidos por el usuario ("kept_symbols"),
    baja fundamentales históricos para cada uno,
    y retorna un DataFrame con una fila por símbolo.
    """
    rows = []
    for sym in symbols:
        try:
            r = fetch_fundamentals_for_symbol(sym)
        except Exception:
            # Si falla FMP para este símbolo, devolvemos todo None
            r = {
                "symbol": sym,
                "fcf_per_share_slope_5y": None,
                "buyback_pct_5y": None,
                "net_debt_change_5y": None,
                "net_debt_to_ebitda_last": None,
                "years": [],
                "fcf_per_share_hist": [],
                "shares_hist": [],
                "net_debt_hist": [],
            }
        rows.append(r)

    return pd.DataFrame(rows)


def enrich_universe_with_fundamentals(
    universe_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Hace el merge por 'symbol' y construye banderas heurísticas
    tipo "is_quality_compounder".
    """

    if universe_df is None or universe_df.empty:
        # nada que enriquecer
        return pd.DataFrame(columns=[
            "symbol",
            "companyName",
            "sector",
            "marketCap",
            "fcf_per_share_slope_5y",
            "buyback_pct_5y",
            "net_debt_to_ebitda_last",
            "is_quality_compounder",
        ])

    # Normalizamos que universe tenga "companyName".
    # Algunos screeners traen 'companyName', otros 'name'.
    if "companyName" not in universe_df.columns and "name" in universe_df.columns:
        universe_df = universe_df.rename(columns={"name": "companyName"})

    full = universe_df.merge(
        fundamentals_df,
        on="symbol",
        how="left",
        validate="1:1"
    )

    # helpers de flags
    def _flag_positive(x):
        try:
            return (x is not None) and (not pd.isna(x)) and (float(x) > 0)
        except Exception:
            return False

    full["flag_fcf_up"] = full["fcf_per_share_slope_5y"].apply(_flag_positive)

    full["flag_buybacks"] = full["buyback_pct_5y"].apply(
        lambda x: (
            x is not None
            and (not pd.isna(x))
            and (float(x) > 0.05)  # >5% de reducción en shares
        )
        if x is not None else False
    )

    full["flag_net_debt_ok"] = full["net_debt_to_ebitda_last"].apply(
        lambda x: (
            x is not None
            and (not pd.isna(x))
            and (float(x) < 2.0)
        )
        if x is not None else False
    )

    full["is_quality_compounder"] = (
        full["flag_fcf_up"]
        & full["flag_buybacks"]
        & full["flag_net_debt_ok"]
    )

    return full


# =============================================================================
# API visible por la app
# =============================================================================

def build_full_snapshot(kept_symbols: list[str]) -> pd.DataFrame:
    """
    - Obtiene el universo del screener.
    - Obtiene fundamentales históricos SOLO de los tickers 'kept_symbols'
      que el usuario guardó en sesión.
    - Hace merge y flags de calidad.
    - Devuelve un DataFrame con columnas como:
        symbol,
        companyName,
        sector,
        marketCap,
        fcf_per_share_slope_5y,
        buyback_pct_5y,
        net_debt_to_ebitda_last,
        is_quality_compounder,
        years,
        fcf_per_share_hist,
        shares_hist,
        net_debt_hist,
        ...
      (algunas de estas son útiles después en el detalle)
    """
    if not kept_symbols:
        return pd.DataFrame()

    universe = build_universe()

    # filtramos universe sólo a los kept_symbols para no inflar DF
    universe_small = universe[universe["symbol"].isin(kept_symbols)].copy()

    fund_block = build_fundamentals_block(kept_symbols)

    final_df = enrich_universe_with_fundamentals(universe_small, fund_block)

    return final_df


def build_market_snapshot() -> List[Dict[str, Any]]:
    """
    Construye la shortlist "global" que alimenta Tab1/Tab2.

    1. Descarga el universo (todas las large caps).
    2. Para ahora, NO bajamos data histórica para todas (sería caro/lento).
       Entonces rellenamos con placeholders seguros donde la app espera columnas.

    Formato de salida: list[dict] para poder hacer DataFrame luego en la app.
    Campos que Tab1/Tab2 esperan (dataframe_from_rows en app.py):
      - ticker
      - name
      - sector
      - industry
      - marketCap
      - altmanZScore
      - piotroskiScore
      - revenueGrowth
      - operatingCashFlowGrowth
      - freeCashFlowGrowth
      - debtGrowth
      - rev_CAGR_5y
      - rev_CAGR_3y
      - ocf_CAGR_5y
      - ocf_CAGR_3y
      - netDebt_to_EBITDA
      - moat_flag
    """

    universe = build_universe()

    if universe.empty:
        return []

    # normalizamos columnas mínimas
    if "companyName" not in universe.columns and "name" in universe.columns:
        universe = universe.rename(columns={"name": "companyName"})
    if "industry" not in universe.columns:
        universe["industry"] = None
    if "sector" not in universe.columns:
        universe["sector"] = None

    rows_out: List[Dict[str, Any]] = []

    for _, row in universe.iterrows():
        sym = row.get("symbol")
        mcap = row.get("marketCap")

        # filtro large cap
        if not _is_large_cap(row):
            continue

        # placeholder 'moat_flag' muy básico
        moat_guess = None
        sector_val = row.get("sector", "")
        if isinstance(sector_val, str):
            # micro heurística divertida nomás para diferenciar:
            if "Software" in sector_val or "Technology" in sector_val:
                moat_guess = "switching costs / IP"
            elif "Health" in sector_val or "Pharma" in sector_val:
                moat_guess = "regulatorio / patentes"
            elif "Consumer" in sector_val:
                moat_guess = "marca / distribución"
            else:
                moat_guess = "escala / eficiencia"

        # netDebt_to_EBITDA lo ponemos None por ahora
        # (lo calculamos en profundidad sólo para kept_symbols)
        netDebt_to_EBITDA = None

        # estos crecimientos y scores son placeholders = None
        out = {
            "ticker": sym,
            "name": row.get("companyName", sym),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "marketCap": mcap,

            "altmanZScore": None,
            "piotroskiScore": None,

            "revenueGrowth": None,
            "operatingCashFlowGrowth": None,
            "freeCashFlowGrowth": None,
            "debtGrowth": None,

            "rev_CAGR_5y": None,
            "rev_CAGR_3y": None,
            "ocf_CAGR_5y": None,
            "ocf_CAGR_3y": None,

            "netDebt_to_EBITDA": netDebt_to_EBITDA,
            "moat_flag": moat_guess,
        }

        # mini-filtro leverage aproximado:
        # como no sabemos netDebt_to_EBITDA, no descartamos por eso.
        # En el futuro puedes rellenar ese valor en batch y filtrar.
        rows_out.append(out)

    return rows_out


def enrich_company_snapshot(base_core: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dado un dict de un ticker de la shortlist (build_market_snapshot),
    agregamos:
      - históricos multianuales (para los gráficos en Tab2)
      - señales cualitativas (insiders, sentimiento noticias, transcript)
    Si algo falla, devolvemos 'base_core' con defaults.

    IMPORTANTE: base_core.ticker existe (Tab2 pasa esto).
    """

    if base_core is None:
        return {}

    ticker = base_core.get("ticker")
    if ticker is None:
        # nada que hacer
        return base_core

    # Intentar reutilizar la misma lógica de fetch_fundamentals_for_symbol()
    # para armar historiales -> gráfico. Esto NO lo mete en batch al tab1,
    # sólo cuando el usuario abre un ticker.
    try:
        fund = fetch_fundamentals_for_symbol(ticker)
    except Exception:
        fund = {
            "years": [],
            "fcf_per_share_hist": [],
            "shares_hist": [],
            "net_debt_hist": [],
            "net_debt_to_ebitda_last": None,
        }

    detailed = dict(base_core)  # copiamos todo lo que ya teníamos
    detailed["years"] = fund.get("years", [])
    detailed["fcf_per_share_hist"] = fund.get("fcf_per_share_hist", [])
    detailed["shares_hist"] = fund.get("shares_hist", [])
    detailed["net_debt_hist"] = fund.get("net_debt_hist", [])

    # ratio leverage "último" para mostrarlo bonito:
    detailed["netDebt_to_EBITDA"] = fund.get("net_debt_to_ebitda_last")

    # ------ Señales cualitativas (stubs seguros) ------
    # Insiders
    try:
        insider_data = get_insider_trading(ticker)
        # puedes parsear insider_data para una señal tipo "compras netas" vs "ventas netas".
        # por ahora, placeholder amigable
        detailed["insider_signal"] = "neutral"
    except Exception:
        detailed["insider_signal"] = "neutral"

    # Sentimiento de noticias
    try:
        news_list = get_news(ticker)
        # aquí podrías hacer NLP y clasificar.
        # placeholder:
        detailed["sentiment_flag"] = "neutral"
        detailed["sentiment_reason"] = "tono mixto/sectorial"
    except Exception:
        detailed["sentiment_flag"] = "neutral"
        detailed["sentiment_reason"] = "tono mixto/sectorial"

    # Resumen de negocio / why it matters
    try:
        prof = get_profile(ticker)
        detailed["business_summary"] = prof.get("description", "—") if isinstance(prof, dict) else "—"
    except Exception:
        detailed["business_summary"] = "—"

    detailed["why_it_matters"] = (
        "Genera caja operativa sostenible y potencialmente reinvierte "
        "a ROIC alto (heurística)."
    )
    detailed["core_risk_note"] = (
        "Riesgo macro / ejecución en la tesis. Monitorear deuda y márgenes."
    )

    # Transcript (earnings call)
    try:
        transcript = get_earnings_call_transcript(ticker)
        # Podrías resumir transcript["content"] con LLM/NLP, etc.
        detailed["transcript_summary"] = (
            "Management enfocado en crecimiento rentable y disciplina de costos."
        )
    except Exception:
        detailed["transcript_summary"] = "Sin señales fuertes en la última call."

    return detailed
