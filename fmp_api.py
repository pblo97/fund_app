from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from config import FMP_API_KEY, SCREENER_PARAMS

BASE = "https://financialmodelingprep.com/api/v3"


def _get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if params is None:
        params = {}
    params["apikey"] = FMP_API_KEY
    url = f"{BASE}{endpoint}"

    try:
        r = requests.get(url, params=params, timeout=10)
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
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, dict):
        payload = [payload]
    try:
        return pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()


def _latest_fy_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()

    if "symbol" not in df.columns:
        return pd.DataFrame()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "fiscalDate" in df.columns:
        df["date"] = pd.to_datetime(df["fiscalDate"], errors="coerce")
    else:
        df["date"] = pd.NaT

    df = df.sort_values(["symbol", "date"], ascending=[True, False])
    latest = df.groupby("symbol").head(1).reset_index(drop=True)
    return latest


def run_screener_for_exchange(
    exchange: str,
    min_mktcap: float = 1e10
) -> List[Dict[str, Any]]:
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


def get_financial_scores_batch(
    symbols: List[str],
    batch_size: int = 50,
    sleep_s: float = 0.2
) -> pd.DataFrame:
    rows_all: List[Dict[str, Any]] = []

    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        if not chunk:
            continue

        params = {"symbol": ",".join(chunk)}
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


def get_growth_batch(
    symbols: List[str],
    batch_size: int = 20,
    sleep_s: float = 0.3
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        if not chunk:
            continue

        params = {
            "symbol": ",".join(chunk),
            "period": "annual",
        }

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

    healthy = df_growth[
        (df_growth["revenueGrowth"] >= 0) &
        (df_growth["ebitgrowth"] >= 0) &
        (df_growth["operatingCashFlowGrowth"] >= 0) &
        (df_growth["freeCashFlowGrowth"] >= 0) &
        (df_growth["debtGrowth"] <= 0)
    ].copy()

    def _cagr(g_mult, yrs: int):
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

    def _meets_15pct(row: pd.Series) -> bool:
        for key in ["rev_CAGR_5y","rev_CAGR_3y","ocf_CAGR_5y","ocf_CAGR_3y"]:
            val = row.get(key)
            if val is not None and val >= 0.15:
                return True
        return False

    healthy["high_growth_flag"] = healthy.apply(_meets_15pct, axis=1)

    elite = healthy[healthy["high_growth_flag"]].reset_index(drop=True)

    return elite


def get_ratios(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    out = _get(f"/ratios/{ticker}", {"limit": limit, "period": "annual"})
    if isinstance(out, dict):
        out = [out]
    return out


def get_income_statement(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    out = _get(f"/income-statement/{ticker}", {
        "period": "annual",
        "limit": limit
    })
    if isinstance(out, dict):
        out = [out]
    return out


def get_balance_sheet(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    out = _get(f"/balance-sheet-statement/{ticker}", {
        "period": "annual",
        "limit": limit
    })
    if isinstance(out, dict):
        out = [out]
    return out


def get_cash_flow(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    out = _get(f"/cash-flow-statement/{ticker}", {
        "period": "annual",
        "limit": limit
    })
    if isinstance(out, dict):
        out = [out]
    return out


def get_profile(ticker: str) -> List[Dict[str, Any]]:
    out = _get(f"/profile/{ticker}")
    if isinstance(out, dict):
        out = [out]
    if out is None:
        out = []
    return out


def get_insider_trading(ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
    out = _get("/insider-trading", {
        "symbol": ticker,
        "limit": limit
    })
    if isinstance(out, dict):
        out = [out]
    if out is None:
        out = []
    return out


def get_news(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    out = _get("/stock_news", {
        "tickers": ticker,
        "limit": limit
    })
    if isinstance(out, dict):
        out = [out]
    if out is None:
        out = []
    return out


def get_earnings_call_transcript(ticker: str, limit: int = 1) -> List[Dict[str, Any]]:
    out = _get(f"/earning_call_transcript/{ticker}", {
        "limit": limit
    })
    if isinstance(out, dict):
        out = [out]
    if out is None:
        out = []
    return out


# ---- históricos anuales por ticker ----

def _safe_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.DataFrame(columns=cols)
    return df[existing].copy()


def get_cashflow_history(symbol: str) -> pd.DataFrame:
    raw = _get(f"/cash-flow-statement/{symbol}", {
        "period": "annual",
        "limit": 5
    })
    df = _to_df(raw).copy()
    if "date" in df.columns and "fiscalDate" not in df.columns:
        df["fiscalDate"] = df["date"]
    cols = ["fiscalDate", "operatingCashFlow", "capitalExpenditure"]
    return _safe_cols(df, cols)


def get_balance_history(symbol: str) -> pd.DataFrame:
    raw = _get(f"/balance-sheet-statement/{symbol}", {
        "period": "annual",
        "limit": 5
    })
    df = _to_df(raw).copy()
    if "date" in df.columns and "fiscalDate" not in df.columns:
        df["fiscalDate"] = df["date"]
    cols = ["fiscalDate", "totalDebt", "cashAndShortTermInvestments"]
    return _safe_cols(df, cols)


def get_income_history(symbol: str) -> pd.DataFrame:
    raw = _get(f"/income-statement/{symbol}", {
        "period": "annual",
        "limit": 5
    })
    df = _to_df(raw).copy()
    if "date" in df.columns and "fiscalDate" not in df.columns:
        df["fiscalDate"] = df["date"]
    cols = ["fiscalDate", "ebitda", "revenue"]
    return _safe_cols(df, cols)


def get_shares_history(symbol: str) -> pd.DataFrame:
    raw = _get(f"/income-statement/{symbol}", {
        "period": "annual",
        "limit": 5
    })
    df = _to_df(raw).copy()
    if "date" in df.columns and "fiscalDate" not in df.columns:
        df["fiscalDate"] = df["date"]
    if "weightedAverageShsOutDil" in df.columns and "sharesDiluted" not in df.columns:
        df["sharesDiluted"] = df["weightedAverageShsOutDil"]
    cols = ["fiscalDate", "sharesDiluted"]
    return _safe_cols(df, cols)
