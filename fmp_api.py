# fmp_api.py
#
# Funciones de acceso crudo a la API de FMP.
# Cada función devuelve datos en formato usable (list[dict] o DataFrame listo).
# Agregamos batch para scores y growth, y filtramos large caps en el screener.

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from config import FMP_API_KEY, SCREENER_PARAMS

BASE = "https://financialmodelingprep.com/api/v3"


# -------------------------------------------------
# util interno: GET genérico
# -------------------------------------------------
def _get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Llamada GET genérica a la API de FMP.
    endpoint: ej. "/profile/AAPL"
    params: dict con parámetros query (sin la apikey).
    Devuelve r.json() tal cual.
    """
    if params is None:
        params = {}
    params["apikey"] = FMP_API_KEY
    url = f"{BASE}{endpoint}"

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# -------------------------------------------------
# util interno: convertir respuesta a DataFrame
# -------------------------------------------------
def _to_df(payload: Any) -> pd.DataFrame:
    """
    Normaliza cualquier payload JSON en un DataFrame.
    - Si viene dict -> [dict]
    - Si viene None -> []
    """
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, dict):
        payload = [payload]
    try:
        return pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()


# -------------------------------------------------
# util interno: dado un df con múltiples años por symbol,
# nos quedamos con el FY más reciente por símbolo
# -------------------------------------------------
def _latest_fy_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Espera columnas: symbol, date, period, fiscalYear, etc.
    Ordena por date desc y toma el head(1) por símbolo.
    """
    if df.empty:
        return df

    # asegurar columnas
    if "symbol" not in df.columns:
        return pd.DataFrame()

    # parsear fecha si existe
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    # ordenar más reciente primero y agrupar
    df = df.sort_values(["symbol", "date"], ascending=[True, False])
    latest = df.groupby("symbol").head(1).reset_index(drop=True)

    return latest


# -------------------------------------------------
# Screener base (PARCHADO: filtramos sólo large cap)
# -------------------------------------------------
def run_screener_for_exchange(exchange: str,
                              min_mktcap: float = 1e10) -> List[Dict[str, Any]]:
    """
    Usa el stock screener de FMP con nuestros filtros base.
    Cambia sólo 'exchange' entre NASDAQ / NYSE / AMEX.
    PARCHADO: aplicamos filtro de large cap (>= 10B USD).

    Devuelve lista[dict] ya filtrada.
    Cada dict incluye {symbol, companyName, sector, industry, price, marketCap, exchange, ...}
    """
    params = dict(SCREENER_PARAMS)
    # forzamos el exchange que queremos consultar
    params["exchange"] = exchange
    # aseguramos activo
    params["isActivelyTrading"] = "true"
    # tratamos de traer hartos de una vez
    if "limit" not in params:
        params["limit"] = 5000

    raw = _get("/stock-screener", params=params)

    if raw is None:
        return []

    out: List[Dict[str, Any]] = []
    for row in raw:
        # marketCap puede venir como string / None
        mc = row.get("marketCap")
        try:
            mc_val = float(mc)
        except (TypeError, ValueError):
            mc_val = None

        if mc_val is not None and mc_val >= min_mktcap:
            out.append(row)

    return out


# -------------------------------------------------
# Financial Scores API (Altman Z, Piotroski)
# -------------------------------------------------
def get_financial_scores_batch(symbols: List[str],
                               batch_size: int = 50,
                               sleep_s: float = 0.2) -> pd.DataFrame:
    """
    Para una lista de símbolos grande, llama a la API de 'Financial Scores'
    en batches. Devuelve un DataFrame con al menos:
      symbol, altmanZScore, piotroskiScore, workingCapital, totalAssets,
      retainedEarnings, ebit, marketCap, totalLiabilities, revenue

    IMPORTANTE:
    - AJUSTA el endpoint según tu plan FMP. Ejemplo típico:
      "/financial-score" o "/score".
    - Este código asume que el endpoint ACEPTA batch con
      symbol=AAPL,MSFT,GOOGL,...

    Además aplicamos filtro de calidad:
      altmanZScore >= 3
      piotroskiScore >= 7
    """
    rows_all: List[Dict[str, Any]] = []

    # batch loop
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        params = {
            "symbol": ",".join(chunk)
        }

        # <-- ajusta endpoint exacto si en tu cuenta es distinto
        raw = _get("/financial-score", params=params)

        if raw is None:
            raw = []

        if isinstance(raw, dict):
            raw = [raw]

        rows_all.extend(raw)
        time.sleep(sleep_s)

    df = _to_df(rows_all)

    needed_cols = [
        "symbol",
        "altmanZScore",
        "piotroskiScore",
        "workingCapital",
        "totalAssets",
        "retainedEarnings",
        "ebit",
        "marketCap",
        "totalLiabilities",
        "revenue",
    ]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = None
    df = df[needed_cols]

    df["altmanZScore"] = pd.to_numeric(df["altmanZScore"], errors="coerce")
    df["piotroskiScore"] = pd.to_numeric(df["piotroskiScore"], errors="coerce")

    df = df[
        (df["altmanZScore"] >= 3.0) &
        (df["piotroskiScore"] >= 7.0)
    ].reset_index(drop=True)

    return df


# -------------------------------------------------
# Financial Statement Growth API (crecimiento / deuda)
# -------------------------------------------------
def get_growth_batch(symbols: List[str],
                     batch_size: int = 20,
                     sleep_s: float = 0.3) -> pd.DataFrame:
    """
    Para una lista de símbolos (idealmente ya filtrados por calidad),
    llama al endpoint de crecimiento (Financial Statement Growth).
    Devuelve sólo el último FY por símbolo, con métricas de crecimiento.

    IMPORTANTE:
    - Ajusta el endpoint exacto; en FMP suele ser algo como
      "/financial-statement-growth".
    - Asumimos que acepta batch: symbol=AAPL,MSFT,...&period=annual

    Filtros de salud que vamos a necesitar río arriba:
      revenueGrowth >= 0
      ebitgrowth >= 0
      operatingCashFlowGrowth >= 0
      freeCashFlowGrowth >= 0
      debtGrowth <= 0

    También calcularemos CAGR 3y/5y para revenue y OCF por acción.
    """
    parts: List[pd.DataFrame] = []

    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        params = {
            "symbol": ",".join(chunk),
            "period": "annual"
        }

        # <-- ajusta endpoint si tu contrato usa otro path
        raw = _get("/financial-statement-growth", params=params)
        df_chunk = _to_df(raw)
        if df_chunk.empty:
            time.sleep(sleep_s)
            continue

        latest = _latest_fy_per_symbol(df_chunk)
        parts.append(latest)

        time.sleep(sleep_s)

    if not parts:
        return pd.DataFrame()

    df_growth = pd.concat(parts, ignore_index=True)

    needed_cols = [
        "symbol",
        "date",
        "fiscalYear",
        "period",
        "revenueGrowth",
        "ebitgrowth",
        "operatingCashFlowGrowth",
        "freeCashFlowGrowth",
        "debtGrowth",
        "fiveYRevenueGrowthPerShare",
        "threeYRevenueGrowthPerShare",
        "fiveYOperatingCFGrowthPerShare",
        "threeYOperatingCFGrowthPerShare",
    ]
    for c in needed_cols:
        if c not in df_growth.columns:
            df_growth[c] = None
    df_growth = df_growth[needed_cols]

    # pasar a numérico
    num_cols = [
        "revenueGrowth",
        "ebitgrowth",
        "operatingCashFlowGrowth",
        "freeCashFlowGrowth",
        "debtGrowth",
        "fiveYRevenueGrowthPerShare",
        "threeYRevenueGrowthPerShare",
        "fiveYOperatingCFGrowthPerShare",
        "threeYOperatingCFGrowthPerShare",
    ]
    for c in num_cols:
        df_growth[c] = pd.to_numeric(df_growth[c], errors="coerce")

    # filtros de salud / disciplina financiera
    healthy = df_growth[
        (df_growth["revenueGrowth"] >= 0) &
        (df_growth["ebitgrowth"] >= 0) &
        (df_growth["operatingCashFlowGrowth"] >= 0) &
        (df_growth["freeCashFlowGrowth"] >= 0) &
        (df_growth["debtGrowth"] <= 0)
    ].copy()

    # ---- calcular CAGR compuesto (~15%) ----
    def _cagr(g_mult, yrs: int):
        # g_mult = 0.8093 significa +80.93%, o sea 1.8093x en 5y
        # CAGR = (1 + g_mult)**(1/yrs) - 1
        try:
            base = 1.0 + float(g_mult)
            if base <= 0:
                return None
            return base ** (1.0 / yrs) - 1.0
        except (TypeError, ValueError):
            return None

    healthy["rev_CAGR_5y"] = healthy["fiveYRevenueGrowthPerShare"].apply(
        lambda g: _cagr(g, 5)
    )
    healthy["rev_CAGR_3y"] = healthy["threeYRevenueGrowthPerShare"].apply(
        lambda g: _cagr(g, 3)
    )
    healthy["ocf_CAGR_5y"] = healthy["fiveYOperatingCFGrowthPerShare"].apply(
        lambda g: _cagr(g, 5)
    )
    healthy["ocf_CAGR_3y"] = healthy["threeYOperatingCFGrowthPerShare"].apply(
        lambda g: _cagr(g, 3)
    )

    def _meets_15pct(row):
        for key in [
            "rev_CAGR_5y",
            "rev_CAGR_3y",
            "ocf_CAGR_5y",
            "ocf_CAGR_3y",
        ]:
            val = row.get(key)
            if val is not None and val >= 0.15:
                return True
        return False

    healthy["high_growth_flag"] = healthy.apply(_meets_15pct, axis=1)

    # nos quedamos sólo con las de crecimiento compuesto >=15% en revenue/OCF por acción
    elite = healthy[healthy["high_growth_flag"]].reset_index(drop=True)

    return elite


# -------------------------------------------------
# Funciones legacy que ya tenías (se mantienen)
# -------------------------------------------------
def get_ratios(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Ratios financieros históricos (ROE, margen, etc.).
    Típicamente newest first [0] = más reciente.
    """
    return _get(f"/ratios/{ticker}", {"limit": limit})


def get_income_statement(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Income statement anual.
    """
    return _get(f"/income-statement/{ticker}", {
        "period": "annual",
        "limit": limit
    })


def get_balance_sheet(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Balance sheet anual.
    """
    return _get(f"/balance-sheet-statement/{ticker}", {
        "period": "annual",
        "limit": limit
    })


def get_cash_flow(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Cash flow statement anual.
    """
    return _get(f"/cash-flow-statement/{ticker}", {
        "period": "annual",
        "limit": limit
    })


def get_profile(ticker: str) -> List[Dict[str, Any]]:
    """
    Profile de la empresa: descripción, sector, industry,
    marketCap, price, beta, etc.
    Devuelve lista con un dict normalmente.
    """
    return _get(f"/profile/{ticker}")


def get_insider_trading(ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Transacciones de insiders (directores/ejecutivos).
    """
    return _get("/insider-trading", {
        "symbol": ticker,
        "limit": limit
    })


def get_news(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Noticias recientes sobre el ticker.
    Retorna lista con {title, text, publishedDate, ...}
    """
    return _get("/stock_news", {
        "tickers": ticker,
        "limit": limit
    })


def get_earnings_call_transcript(ticker: str, limit: int = 1) -> List[Dict[str, Any]]:
    """
    Últimas transcripciones de conference calls (earnings call).
    Incluye Q&A, guía de management, etc.
    """
    return _get(f"/earning_call_transcript/{ticker}", {
        "limit": limit
    })
