# fmp_api.py
#
# Funciones de acceso crudo a la API de FMP.
# Cada función devuelve el JSON tal cual (listas/dicts).


import requests
from typing import Any, Dict, List, Optional
from config import FMP_API_KEY, SCREENER_PARAMS

BASE = "https://financialmodelingprep.com/api/v3"


def _get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Llamada GET genérica a la API de FMP.
    endpoint: ej. "/profile/AAPL"
    params: dict con parámetros query.
    """
    if params is None:
        params = {}
    params["apikey"] = FMP_API_KEY
    url = f"{BASE}{endpoint}"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def run_screener_for_exchange(exchange: str) -> List[Dict[str, Any]]:
    """
    Usa el stock screener de FMP con nuestros filtros base.
    Cambia sólo 'exchange' entre NASDAQ y NYSE.
    """
    params = dict(SCREENER_PARAMS)
    params["exchange"] = exchange
    data = _get("/stock-screener", params=params)
    return data  # list[dict] con {symbol, companyName, marketCap, ...}


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
