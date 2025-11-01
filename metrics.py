# metrics.py
#
# Cálculos financieros cuantitativos:
# - FCF, FCF/acción, CAGR FCF/acción
# - Owner's Yield (dividend + buyback)
# - Expected Return (Owner's Yield + crecimiento FCF/acción)
# - Apalancamiento, moat_flag
# - Estructuras históricas para graficar


from typing import Dict, Any, List, Optional
import math

from config import (
    MIN_ROE_TTM,
    MAX_NET_DEBT_TO_EBITDA,
    EXPECTED_RETURN_HURDLE
)


def _safe_get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def build_history_series(
    income_hist: List[Dict[str, Any]],
    balance_hist: List[Dict[str, Any]],
    cash_hist: List[Dict[str, Any]]
) -> Dict[str, List[Optional[float]]]:
    """
    Normaliza y alinea la info por año.
    Suponemos que income_hist, balance_hist, cash_hist están en orden newest first.
    Invertimos a old -> new para las series.
    """
    inc_rev = list(reversed(income_hist))
    bal_rev = list(reversed(balance_hist))
    cf_rev  = list(reversed(cash_hist))

    years = []
    fcf_list = []
    shares_list = []
    net_debt_list = []
    ebitda_list = []

    for i in range(min(len(inc_rev), len(bal_rev), len(cf_rev))):
        inc = inc_rev[i]
        bal = bal_rev[i]
        cf  = cf_rev[i]

        # Año de referencia
        yr = (
            inc.get("calendarYear")
            or inc.get("date")
            or inc.get("fillingDate")
        )
        years.append(str(yr))

        # EBITDA
        ebitda_val = _safe_get(inc, "ebitda")
        ebitda_list.append(ebitda_val)

        # CFO y CapEx
        cfo = _safe_get(cf, "netCashProvidedByOperatingActivities")
        capex = _safe_get(cf, "capitalExpenditure")
        if cfo is not None and capex is not None:
            fcf_val = cfo - capex
        else:
            fcf_val = None
        fcf_list.append(fcf_val)

        # Shares outstanding
        shares_val = _safe_get(
            inc,
            "weightedAverageShsOutDil",
            "weightedAverageShsOut",
            default=_safe_get(bal, "commonStockSharesOutstanding")
        )
        shares_list.append(shares_val)

        # Net Debt
        total_debt = _safe_get(bal, "totalDebt")
        if total_debt is None:
            short_debt = bal.get("shortTermDebt") or 0
            long_debt = bal.get("longTermDebt") or 0
            total_debt = short_debt + long_debt
        cash_val = bal.get("cashAndCashEquivalents") or 0
        if total_debt is None:
            net_debt_val = None
        else:
            net_debt_val = total_debt - cash_val
        net_debt_list.append(net_debt_val)

    return {
        "years": years,
        "fcf_list": fcf_list,
        "shares_list": shares_list,
        "net_debt_list": net_debt_list,
        "ebitda_list": ebitda_list,
    }


def compute_fcf_per_share_series(
    fcf_list: List[Optional[float]],
    shares_list: List[Optional[float]]
) -> List[Optional[float]]:
    fcfps_hist = []
    for fcf, sh in zip(fcf_list, shares_list):
        if fcf is None or sh is None or sh == 0:
            fcfps_hist.append(None)
        else:
            fcfps_hist.append(fcf / sh)
    return fcfps_hist


def compute_cagr(series: List[Optional[float]]) -> Optional[float]:
    """
    CAGR aproximado entre el primer valor y el último valor válido.
    Devuelve decimal (0.12 => 12% anual aprox).
    """
    vals = [v for v in series if v is not None and v > 0]
    if len(vals) < 2:
        return None
    first = vals[0]
    last = vals[-1]
    n_years = max(1, len(vals) - 1)
    if first <= 0:
        return None
    ratio = last / first
    return ratio ** (1 / n_years) - 1


def compute_buyback_yield(shares_list: List[Optional[float]]) -> Optional[float]:
    """
    Buyback yield aprox = -%cambio en acciones último año.
    Positivo => reducción de acciones (bueno para el dueño).
    """
    if len(shares_list) < 2:
        return None
    prev = shares_list[-2]
    curr = shares_list[-1]
    if prev is None or curr is None or prev == 0:
        return None
    dilution_pct = (curr - prev) / prev
    return -dilution_pct  # si reducen acciones, yield positivo


def compute_fcf_yield_now(
    fcf_list: List[Optional[float]],
    market_cap: Optional[float]
) -> Optional[float]:
    if not fcf_list or market_cap is None or market_cap == 0:
        return None
    last_fcf = fcf_list[-1]
    if last_fcf is None:
        return None
    return last_fcf / market_cap  # decimal, ej 0.08 = 8%


def compute_net_debt_to_ebitda(
    net_debt_list: List[Optional[float]],
    ebitda_list: List[Optional[float]]
) -> Optional[float]:
    if not net_debt_list or not ebitda_list:
        return None
    nd = net_debt_list[-1]
    eb = ebitda_list[-1]
    if nd is None or eb is None or eb == 0:
        return None
    return nd / eb


def extract_roe_ttm(ratios_hist: List[Dict[str, Any]]) -> Optional[float]:
    """
    Busca returnOnEquityTTM o returnOnEquity en el registro más reciente.
    """
    if not ratios_hist:
        return None
    latest = ratios_hist[0]
    return latest.get("returnOnEquityTTM") or latest.get("returnOnEquity")


def extract_dividend_yield(
    profile: List[Dict[str, Any]],
    ratios_hist: List[Dict[str, Any]]
) -> Optional[float]:
    """
    Dividend yield como decimal.
    Usaremos ratios_hist[0].dividendYield si existe.
    Si no, como fallback heurístico:
    lastDiv / price (asumiendo lastDiv ~ DPS anual).
    """
    dy = None
    if ratios_hist and ratios_hist[0].get("dividendYield") is not None:
        dy = ratios_hist[0]["dividendYield"]

    if dy is None and profile:
        last_div = profile[0].get("lastDiv")
        price = profile[0].get("price")
        if last_div and price and price != 0:
            dy = last_div / price  # heurístico
    return dy


def moat_flag_from_numbers(
    roe_ttm: Optional[float],
    cagr_fcfps_5y: Optional[float],
    net_debt_to_ebitda: Optional[float]
) -> str:
    """
    Heurística básica para etiquetar moat.
    """
    moat = "débil"

    if (roe_ttm and roe_ttm >= 0.15) or (cagr_fcfps_5y and cagr_fcfps_5y > 0.10):
        moat = "media"

    if (
        roe_ttm and roe_ttm >= 0.20 and
        cagr_fcfps_5y and cagr_fcfps_5y > 0.10 and
        (net_debt_to_ebitda is not None and net_debt_to_ebitda <= MAX_NET_DEBT_TO_EBITDA)
    ):
        moat = "fuerte"

    return moat


def compute_core_financial_metrics(
    ticker: str,
    profile: List[dict],
    ratios_hist: List[dict],
    income_hist: List[dict],
    balance_hist: List[dict],
    cash_hist: List[dict]
) -> Dict[str, Any]:
    """
    Produce las métricas cuantitativas principales para el ticker.
    No hace análisis de insiders / noticias todavía.
    """

    # Datos base empresa
    company_name = profile[0].get("companyName") if profile else None
    sector = profile[0].get("sector") if profile else None
    industry = profile[0].get("industry") if profile else None
    market_cap = profile[0].get("mktCap") or profile[0].get("marketCap")
    price = profile[0].get("price")
    beta = profile[0].get("beta")

    # Históricos estructurados
    hist = build_history_series(income_hist, balance_hist, cash_hist)
    years = hist["years"]
    fcf_list = hist["fcf_list"]
    shares_list = hist["shares_list"]
    net_debt_list = hist["net_debt_list"]
    ebitda_list = hist["ebitda_list"]

    fcfps_hist = compute_fcf_per_share_series(fcf_list, shares_list)
    cagr_fcfps_5y = compute_cagr(fcfps_hist)

    fcf_yield_now = compute_fcf_yield_now(fcf_list, market_cap)
    buyback_yield = compute_buyback_yield(shares_list)
    dividend_yield = extract_dividend_yield(profile, ratios_hist)

    if dividend_yield is None and buyback_yield is None:
        owners_yield = None
    else:
        owners_yield = (dividend_yield or 0.0) + (buyback_yield or 0.0)

    roe_ttm = extract_roe_ttm(ratios_hist)
    net_debt_to_ebitda = compute_net_debt_to_ebitda(net_debt_list, ebitda_list)

    expected_return = None
    if owners_yield is not None and cagr_fcfps_5y is not None:
        expected_return = owners_yield + cagr_fcfps_5y

    moat_flag = moat_flag_from_numbers(
        roe_ttm=roe_ttm,
        cagr_fcfps_5y=cagr_fcfps_5y,
        net_debt_to_ebitda=net_debt_to_ebitda
    )

    out = {
        "ticker": ticker,
        "companyName": company_name,
        "sector": sector,
        "industry": industry,
        "marketCap": market_cap,
        "price": price,
        "beta": beta,

        "roe_ttm": roe_ttm,
        "netDebt_to_EBITDA": net_debt_to_ebitda,
        "fcf_yield_now": fcf_yield_now,
        "owners_yield": owners_yield,
        "cagr_fcfps_5y": cagr_fcfps_5y,
        "expected_return": expected_return,

        "moat_flag": moat_flag,

        "years": years,
        "fcf_per_share_hist": fcfps_hist,
        "shares_hist": shares_list,
        "net_debt_hist": net_debt_list,

        # todavía se llenan en text_analysis / orchestrator
        "insider_signal": None,
        "sentiment_flag": None,
        "sentiment_reason": None,
        "business_summary": profile[0].get("description") if profile else None,
        "why_it_matters": None,
        "core_risk_note": None,
        "transcript_summary": None,
    }

    return out
