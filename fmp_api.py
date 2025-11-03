import requests
from typing import List, Dict, Any
from config import FMP_API_KEY

FMP_BASE = "https://financialmodelingprep.com/api/v3"

def _get_json(path: str, params: Dict[str, Any] | None = None) -> Any:
    if params is None:
        params = {}
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE}/{path}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# -------------- Universe / Screener --------------

def get_companies_in_exchange(exchange: str) -> list[dict]:
    # similar a /stock-screener
    return _get_json(
        "stock-screener",
        {
            "exchange": exchange,
            "isActivelyTrading": "true",
            "limit": 2000,
        },
    )

# -------------- Scores (Altman / Piotroski) --------------

def get_financial_scores_bulk(symbols: List[str]) -> dict[str, dict]:
    """
    Llama /financial-score/{symbol} para cada ticker.
    Devuelve dict[ticker] = {...scores...}
    """
    out = {}
    for sym in symbols:
        try:
            data = _get_json(f"financial-score/{sym}")
            # API suele devolver lista con un objeto
            if isinstance(data, list) and data:
                out[sym] = data[0]
        except Exception:
            continue
    return out

# -------------- Growth / Ratios de crecimiento --------------

def get_key_metrics(sym: str) -> dict:
    # Placeholder: ajusta al endpoint real que usas para growth, margins, debt, etc.
    # Por ejemplo /key-metrics-ttm/{symbol} o similar
    try:
        data = _get_json(f"key-metrics-ttm/{sym}")
        if isinstance(data, list) and data:
            return data[0]
    except Exception:
        pass
    return {}

def get_growth_statement(sym: str) -> dict:
    # Placeholder para growth multiyear tipo /income-statement-growth/{sym}
    try:
        data = _get_json(f"income-statement-growth/{sym}")
        if isinstance(data, list) and data:
            return data[-1]
    except Exception:
        pass
    return {}

# -------------- Fundamentos históricos --------------

def get_income_statement_annual(sym: str) -> list[dict]:
    return _get_json(f"income-statement/{sym}", {"period": "annual", "limit": 10})

def get_balance_sheet_annual(sym: str) -> list[dict]:
    return _get_json(f"balance-sheet-statement/{sym}", {"period": "annual", "limit": 10})

def get_cashflow_statement_annual(sym: str) -> list[dict]:
    return _get_json(f"cash-flow-statement/{sym}", {"period": "annual", "limit": 10})

def get_profile(sym: str) -> dict:
    data = _get_json(f"profile/{sym}")
    if isinstance(data, list) and data:
        return data[0]
    return {}

def get_ratios_ttm(sym: str) -> dict:
    # por ejemplo /ratios-ttm/{symbol}
    try:
        data = _get_json(f"ratios-ttm/{sym}")
        if isinstance(data, list) and data:
            return data[0]
    except Exception:
        pass
    return {}

# -------------- Texto / Sentimiento --------------

def get_insider_trading(sym: str) -> list[dict]:
    # /insider-trading?symbol=...
    return _get_json("insider-trading", {"symbol": sym, "limit": 60})

def get_company_news(sym: str) -> list[dict]:
    # /stock_news?tickers=...  (endpoint puede variar)
    return _get_json("stock_news", {"tickers": sym, "limit": 20})

def get_earnings_call_transcript(sym: str) -> list[dict]:
    # /earning_call_transcript/{symbol}?limit=1
    # (endpoint real puede diferir, ajústalo)
    try:
        data = _get_json(f"earning_call_transcript/{sym}", {"limit": 1})
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []
