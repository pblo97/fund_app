# fmp_api.py
#
# Capa de acceso a FMP.
#
# - Todos los requests tienen timeout y no revientan la app.
# - No asumimos batch en endpoints que no lo soportan.
# - Entregamos helpers "bulk" que internamente iteran símbolo por símbolo.
#
# Lo importante que exporta este módulo:
#
#   run_screener_for_exchange(exchange)
#   get_scores_bulk(symbols)
#   get_growth_bulk(symbols)
#
#   get_profile(ticker)
#   get_income_statement(ticker)
#   get_balance_sheet(ticker)
#   get_cash_flow(ticker)
#   get_ratios(ticker)
#
#   get_cashflow_history(ticker)
#   get_balance_history(ticker)
#   get_income_history(ticker)
#   get_shares_history(ticker)
#
#   get_insider_trading(ticker)
#   get_news(ticker)
#   get_earnings_call_transcript(ticker)
#
# Estas funciones son las únicas que debería usar orchestrator.
#
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from config import FMP_API_KEY, SCREENER_PARAMS

BASE = "https://financialmodelingprep.com/api/v3"
_REQ_TIMEOUT = 10.0
_SLEEP_SCORE = 0.2
_SLEEP_GROWTH = 0.3


# -------------------------------------------------
# util interno HTTP
# -------------------------------------------------
def _get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Hace GET a FMP y devuelve r.json() ya parseado.
    - Siempre agrega apikey.
    - Captura errores de conexión / timeout.
    - Si algo falla o la respuesta no es JSON válido -> []
      (o {} en algunos casos downstream lo normalizamos igual).
    """
    if params is None:
        params = {}
    params["apikey"] = FMP_API_KEY

    url = f"{BASE}{endpoint}"

    try:
        r = requests.get(url, params=params, timeout=_REQ_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        return []
    except requests.exceptions.RequestException:
        return []

    try:
        return r.json()
    except Exception:
        return []


def _to_df(payload: Any) -> pd.DataFrame:
    """
    Convierte payload (list[dict] | dict | None) en DataFrame.
    Si no se puede -> df vacío.
    """
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, dict):
        payload = [payload]
    try:
        return pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()


def _safe_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Devuelve sólo las columnas pedidas que existan.
    Si ninguna existe, devuelve DF vacío con esas columnas.
    """
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.DataFrame(columns=cols)
    return df[existing].copy()


# -------------------------------------------------
# 1) Screener de mercado (large caps)
# -------------------------------------------------
def run_screener_for_exchange(
    exchange: str,
    min_mktcap: float = 10_000_000_000.0,
) -> List[Dict[str, Any]]:
    """
    Usa el screener de FMP para un exchange (NYSE / NASDAQ / AMEX / etc.).
    SCREENER_PARAMS viene de config (tus filtros base).
    Además:
      - isActivelyTrading="true"
      - limit grande
      - filtramos por marketCap >= min_mktcap
    Devuelve lista[dict] con:
        symbol, companyName, sector, industry, marketCap, ...
    """
    params = dict(SCREENER_PARAMS)
    params["exchange"] = exchange
    params["isActivelyTrading"] = "true"
    params.setdefault("limit", 5000)

    raw = _get("/stock-screener", params=params)
    if raw is None:
        raw = []
    if isinstance(raw, dict):
        raw = [raw]

    out: List[Dict[str, Any]] = []
    for row in raw:
        mc = row.get("marketCap")
        try:
            mc_val = float(mc)
        except (TypeError, ValueError):
            mc_val = None

        if mc_val is not None and mc_val >= min_mktcap:
            out.append(row)

    return out


# -------------------------------------------------
# 2) Scores de salud financiera (Altman Z, Piotroski)
#    - versión por ticker
#    - versión bulk
# -------------------------------------------------
def _get_financial_score_one(symbol: str) -> pd.DataFrame:
    """
    Pide /financial-score/{symbol}
    Devuelve DF UNA FILA con columnas:
      symbol, altmanZScore, piotroskiScore, workingCapital, totalAssets,
      retainedEarnings, ebit, marketCap, totalLiabilities, revenue
    Si no hay datos -> DF vacío con esas columnas.
    """
    raw = _get(f"/financial-score/{symbol}")
    df = _to_df(raw)

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

    df = df[needed_cols].copy()

    # numéricos
    df["altmanZScore"] = pd.to_numeric(df["altmanZScore"], errors="coerce")
    df["piotroskiScore"] = pd.to_numeric(df["piotroskiScore"], errors="coerce")

    # Si viene más de una fila anual/histórica, nos quedamos con la primera
    # (FMP suele ya entregar 1, pero por seguridad)
    if not df.empty:
        return df.iloc[[0]].reset_index(drop=True)
    else:
        # DF vacío consistente
        return pd.DataFrame(columns=needed_cols)


def get_scores_bulk(symbols: List[str]) -> pd.DataFrame:
    """
    Itera símbolo por símbolo llamando _get_financial_score_one.
    Apila resultados en un DF grande.
    Al final aplica tu filtro de "empresa sana":
      altmanZScore >= 3
      piotroskiScore >= 7
    """
    frames = []
    for sym in symbols:
        df_one = _get_financial_score_one(sym)
        if not df_one.empty:
            frames.append(df_one)
        time.sleep(_SLEEP_SCORE)

    if not frames:
        return pd.DataFrame(
            columns=[
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
        )

    out = pd.concat(frames, ignore_index=True)

    mask_ok = (
        (out["altmanZScore"] >= 3.0)
        & (out["piotroskiScore"] >= 7.0)
    )
    out = out[mask_ok].reset_index(drop=True)

    return out


# -------------------------------------------------
# 3) Crecimiento y disciplina financiera
#    - versión por ticker
#    - versión bulk
# -------------------------------------------------
def _get_growth_one(symbol: str) -> pd.DataFrame:
    """
    Pide /financial-statement-growth/{symbol}?period=annual&limit=40
    Devuelve SOLO la fila más reciente para ese ticker, con métricas como:
      revenueGrowth, ebitgrowth, operatingCashFlowGrowth, freeCashFlowGrowth,
      debtGrowth,
      fiveYRevenueGrowthPerShare, threeYRevenueGrowthPerShare,
      fiveYOperatingCFGrowthPerShare, threeYOperatingCFGrowthPerShare

    Retorna DataFrame con UNA fila y columna 'symbol'.
    """
    raw = _get(f"/financial-statement-growth/{symbol}", {
        "period": "annual",
        "limit": 40,
    })
    df_all = _to_df(raw)

    # si no hay nada devolvemos DF vacío con columnas necesarias
    base_cols = [
        "symbol",
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
    if df_all.empty:
        return pd.DataFrame(columns=base_cols)

    # ordenar más reciente: preferimos 'date', si no 'fiscalYear'
    if "date" in df_all.columns:
        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
        df_all = df_all.sort_values("date", ascending=False)
    elif "fiscalYear" in df_all.columns:
        df_all["fiscalYear_num"] = pd.to_numeric(
            df_all["fiscalYear"], errors="coerce"
        )
        df_all = df_all.sort_values("fiscalYear_num", ascending=False)

    latest = df_all.iloc[[0]].copy()
    latest["symbol"] = symbol

    for c in base_cols:
        if c not in latest.columns:
            latest[c] = None

    latest = latest[base_cols].copy()

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
        latest[c] = pd.to_numeric(latest[c], errors="coerce")

    return latest.reset_index(drop=True)


def get_growth_bulk(symbols: List[str]) -> pd.DataFrame:
    """
    Itera _get_growth_one() para todos los tickers.
    Calcula CAGR 3y/5y y 'high_growth_flag'.
    Aplica filtro de disciplina financiera:
      - revenueGrowth >= 0
      - ebitgrowth >= 0
      - operatingCashFlowGrowth >= 0
      - freeCashFlowGrowth >= 0
      - debtGrowth <= 0
    Luego marca high_growth_flag si alguna CAGR >= 15%.
    """
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df_one = _get_growth_one(sym)
        if not df_one.empty:
            frames.append(df_one)
        time.sleep(_SLEEP_GROWTH)

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "revenueGrowth",
                "ebitgrowth",
                "operatingCashFlowGrowth",
                "freeCashFlowGrowth",
                "debtGrowth",
                "fiveYRevenueGrowthPerShare",
                "threeYRevenueGrowthPerShare",
                "fiveYOperatingCFGrowthPerShare",
                "threeYOperatingCFGrowthPerShare",
                "rev_CAGR_5y",
                "rev_CAGR_3y",
                "ocf_CAGR_5y",
                "ocf_CAGR_3y",
                "high_growth_flag",
            ]
        )

    df = pd.concat(frames, ignore_index=True)

    # helper CAGR
    def _cagr(g_mult, yrs: int):
        """
        g_mult ~ crecimiento acumulado (ej 0.80 => +80% total => factor 1.80).
        CAGR = (1 + g_mult)**(1/yrs) - 1
        """
        try:
            base = 1.0 + float(g_mult)
            if base <= 0:
                return None
            return base ** (1.0 / yrs) - 1.0
        except (TypeError, ValueError):
            return None

    df["rev_CAGR_5y"] = df["fiveYRevenueGrowthPerShare"].apply(
        lambda g: _cagr(g, 5)
    )
    df["rev_CAGR_3y"] = df["threeYRevenueGrowthPerShare"].apply(
        lambda g: _cagr(g, 3)
    )
    df["ocf_CAGR_5y"] = df["fiveYOperatingCFGrowthPerShare"].apply(
        lambda g: _cagr(g, 5)
    )
    df["ocf_CAGR_3y"] = df["threeYOperatingCFGrowthPerShare"].apply(
        lambda g: _cagr(g, 3)
    )

    # filtro de disciplina en el último FY
    df = df[
        (df["revenueGrowth"] >= 0)
        & (df["ebitgrowth"] >= 0)
        & (df["operatingCashFlowGrowth"] >= 0)
        & (df["freeCashFlowGrowth"] >= 0)
        & (df["debtGrowth"] <= 0)
    ].copy()

    # flag high_growth_flag si CAGR >=15% en algo
    def _meets_15pct(row: pd.Series) -> bool:
        for k in [
            "rev_CAGR_5y",
            "rev_CAGR_3y",
            "ocf_CAGR_5y",
            "ocf_CAGR_3y",
        ]:
            v = row.get(k)
            if v is not None and v >= 0.15:
                return True
        return False

    df["high_growth_flag"] = df.apply(_meets_15pct, axis=1)

    return df.reset_index(drop=True)


# -------------------------------------------------
# 4) Estados financieros básicos (anual)
# -------------------------------------------------
def get_income_statement(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    /income-statement/{ticker}?period=annual&limit=5
    Devuelve lista de dicts (más reciente primero).
    """
    out = _get(f"/income-statement/{ticker}", {
        "period": "annual",
        "limit": limit,
    })
    if isinstance(out, dict):
        out = [out]
    return out


def get_balance_sheet(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    /balance-sheet-statement/{ticker}?period=annual&limit=5
    """
    out = _get(f"/balance-sheet-statement/{ticker}", {
        "period": "annual",
        "limit": limit,
    })
    if isinstance(out, dict):
        out = [out]
    return out


def get_cash_flow(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    /cash-flow-statement/{ticker}?period=annual&limit=5
    """
    out = _get(f"/cash-flow-statement/{ticker}", {
        "period": "annual",
        "limit": limit,
    })
    if isinstance(out, dict):
        out = [out]
    return out


def get_ratios(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    /ratios/{ticker}?period=annual&limit=5
    Ratios históricos (ROE, márgenes, etc.).
    """
    out = _get(f"/ratios/{ticker}", {
        "period": "annual",
        "limit": limit,
    })
    if isinstance(out, dict):
        out = [out]
    return out


def get_profile(ticker: str) -> List[Dict[str, Any]]:
    """
    /profile/{ticker}
    Info estática de la empresa (sector, industry, marketCap, beta, descripción...).
    """
    out = _get(f"/profile/{ticker}")
    if isinstance(out, dict):
        out = [out]
    if out is None:
        out = []
    return out


# -------------------------------------------------
# 5) Históricos alineados por año (para slope FCF/acción, recompras, net debt)
# -------------------------------------------------
def get_cashflow_history(symbol: str) -> pd.DataFrame:
    """
    Devuelve DF anual con:
      fiscalDate, operatingCashFlow, capitalExpenditure
    """
    raw = _get(f"/cash-flow-statement/{symbol}", {
        "period": "annual",
        "limit": 5,
    })
    df = _to_df(raw).copy()

    # normalizamos fiscalDate
    if "date" in df.columns and "fiscalDate" not in df.columns:
        df["fiscalDate"] = df["date"]

    cols = [
        "fiscalDate",
        "operatingCashFlow",
        "capitalExpenditure",
    ]
    return _safe_cols(df, cols)


def get_balance_history(symbol: str) -> pd.DataFrame:
    """
    Devuelve DF anual con:
      fiscalDate, totalDebt, cashAndShortTermInvestments
    """
    raw = _get(f"/balance-sheet-statement/{symbol}", {
        "period": "annual",
        "limit": 5,
    })
    df = _to_df(raw).copy()

    if "date" in df.columns and "fiscalDate" not in df.columns:
        df["fiscalDate"] = df["date"]

    cols = [
        "fiscalDate",
        "totalDebt",
        "cashAndShortTermInvestments",
    ]
    return _safe_cols(df, cols)


def get_income_history(symbol: str) -> pd.DataFrame:
    """
    Devuelve DF anual con:
      fiscalDate, ebitda, revenue
    """
    raw = _get(f"/income-statement/{symbol}", {
        "period": "annual",
        "limit": 5,
    })
    df = _to_df(raw).copy()

    if "date" in df.columns and "fiscalDate" not in df.columns:
        df["fiscalDate"] = df["date"]

    cols = [
        "fiscalDate",
        "ebitda",
        "revenue",
    ]
    return _safe_cols(df, cols)


def get_shares_history(symbol: str) -> pd.DataFrame:
    """
    De /income-statement/ obtenemos weightedAverageShsOutDil,
    lo exponemos como sharesDiluted.
    """
    raw = _get(f"/income-statement/{symbol}", {
        "period": "annual",
        "limit": 5,
    })
    df = _to_df(raw).copy()

    if "date" in df.columns and "fiscalDate" not in df.columns:
        df["fiscalDate"] = df["date"]

    if "weightedAverageShsOutDil" in df.columns and "sharesDiluted" not in df.columns:
        df["sharesDiluted"] = df["weightedAverageShsOutDil"]

    cols = [
        "fiscalDate",
        "sharesDiluted",
    ]
    return _safe_cols(df, cols)


# -------------------------------------------------
# 6) Señales cualitativas (insiders, noticias, transcript)
# -------------------------------------------------
def get_insider_trading(ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    /insider-trading?symbol={ticker}&limit={limit}
    """
    out = _get("/insider-trading", {
        "symbol": ticker,
        "limit": limit,
    })
    if isinstance(out, dict):
        out = [out]
    if out is None:
        out = []
    return out


def get_news(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    /stock_news?tickers={ticker}&limit={limit}
    """
    out = _get("/stock_news", {
        "tickers": ticker,
        "limit": limit,
    })
    if isinstance(out, dict):
        out = [out]
    if out is None:
        out = []
    return out


def get_earnings_call_transcript(ticker: str, limit: int = 1) -> List[Dict[str, Any]]:
    """
    /earning_call_transcript/{ticker}?limit={limit}
    """
    out = _get(f"/earning_call_transcript/{ticker}", {
        "limit": limit,
    })
    if isinstance(out, dict):
        out = [out]
    if out is None:
        out = []
    return out
