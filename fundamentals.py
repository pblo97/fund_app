from typing import Dict, Any, List
import pandas as pd
from utils import safe_float, recent_growth, linear_slope_y_per_year, pct_change, series_tail
from config import MAX_NET_DEBT_TO_EBITDA
from fmp_api import (
    get_profile,
    get_income_statement_annual,
    get_balance_sheet_annual,
    get_cashflow_statement_annual,
    get_ratios_ttm,
)

def _extract_series(fin_list: List[dict], key: str) -> List[float]:
    """
    fin_list: lista de estados año a año (ya ordenada del más reciente al más antiguo)
    key: campo numérico (ej "revenue")
    Retorna lista en orden cronológico ascendente.
    """
    vals = []
    # invertimos para ir de más viejo a más nuevo
    for rec in reversed(fin_list):
        vals.append(safe_float(rec.get(key)))
    return [v for v in vals if v is not None]

def _extract_years(fin_list: List[dict], key="calendarYear") -> List[int]:
    yrs = []
    for rec in reversed(fin_list):
        try:
            yrs.append(int(rec.get(key)))
        except Exception:
            yrs.append(None)
    return yrs

def build_fundamental_snapshot(ticker: str,
                               quality_row: pd.Series | None,
                               growth_row: pd.Series | None) -> Dict[str, Any]:
    """
    Devuelve un dict con TODO lo cuantitativo que definimos en el Paso 2.
    """

    profile = get_profile(ticker)
    inc = get_income_statement_annual(ticker)  # revenue, ebit, netIncome...
    bs  = get_balance_sheet_annual(ticker)    # totalDebt, cashAndShortTermInvestments...
    cf  = get_cashflow_statement_annual(ticker)  # operatingCashFlow, capitalExpenditure...
    ratios = get_ratios_ttm(ticker)  # margins, ROIC, etc.

    # Series históricas
    years = _extract_years(inc)
    revenue_hist = _extract_series(inc, "revenue")
    ebit_hist    = _extract_series(inc, "ebit")
    # flujo operativo y capex vienen del cash flow
    ocf_hist     = _extract_series(cf, "netCashProvidedByOperatingActivities")
    capex_hist   = _extract_series(cf, "capitalExpenditure")
    # FCF ~ OCF - CapEx
    free_cf_hist = []
    for ocf, capex in zip(ocf_hist, capex_hist):
        if ocf is None or capex is None:
            free_cf_hist.append(None)
        else:
            free_cf_hist.append(ocf - abs(capex))

    # acciones diluidas
    shares_hist = _extract_series(inc, "weightedAverageShsOutDil")
    # deuda neta = debt total - cash
    total_debt_hist = _extract_series(bs, "totalDebt")
    cash_hist       = _extract_series(bs, "cashAndShortTermInvestments")
    net_debt_hist = []
    for debt, cash in zip(total_debt_hist, cash_hist):
        if debt is None or cash is None:
            net_debt_hist.append(None)
        else:
            net_debt_hist.append(debt - cash)

    # FCF por acción
    fcf_per_share_hist = []
    for fcf, sh in zip(free_cf_hist, shares_hist):
        if fcf is None or sh in (None, 0):
            fcf_per_share_hist.append(None)
        else:
            fcf_per_share_hist.append(fcf / sh)

    # Crecimientos YoY reales desde las series
    revenueGrowth_calc = recent_growth(revenue_hist)
    ebitgrowth_calc    = recent_growth(ebit_hist)
    ocfGrowth_calc     = recent_growth(ocf_hist)
    fcfGrowth_calc     = recent_growth(free_cf_hist)
    debtGrowth_calc    = recent_growth(total_debt_hist)

    # márgenes y ROIC actuales (ttm)
    gross_margin_last      = safe_float(ratios.get("grossProfitMargin"))
    operating_margin_last  = safe_float(ratios.get("operatingProfitMargin"))
    fcf_margin_last        = None  # si la API expone FCF margin ttm, úsalo aquí
    roic_last              = safe_float(ratios.get("roic"))

    # apalancamiento actual
    net_debt_to_ebitda_last = safe_float(ratios.get("netDebtToEBITDA"))
    leverage_ok = (
        (net_debt_to_ebitda_last is not None)
        and (net_debt_to_ebitda_last <= MAX_NET_DEBT_TO_EBITDA)
    )

    # buyback_pct_5y
    buyback_pct_5y = None
    if len(shares_hist) >= 2 and shares_hist[0] and shares_hist[-1]:
        buyback_pct_5y = pct_change(shares_hist[0], shares_hist[-1])
        # pct_change da (new-old)/old, aquí queremos % reducción,
        # si salió negativo (porque subieron acciones), queda negativo.

    # slope de FCF por acción (tendencia compounder)
    slope_fcf_ps = None
    if years and fcf_per_share_hist:
        # usar las últimas 5 observaciones para limpiar ruido
        yrs_tail = series_tail(years, n=5)
        fcf_tail = series_tail(fcf_per_share_hist, n=5)
        slope_fcf_ps = linear_slope_y_per_year(yrs_tail, fcf_tail)

    # alta calidad compounder (heurística inicial)
    is_quality_compounder = False
    if (
        slope_fcf_ps is not None and slope_fcf_ps > 0 and
        buyback_pct_5y is not None and buyback_pct_5y < 0 and  # menor float => recompras netas
        leverage_ok
    ):
        is_quality_compounder = True

    # armar snapshot base
    snap: Dict[str, Any] = {
        # Identidad
        "ticker": ticker,
        "companyName": profile.get("companyName"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "beta": safe_float(profile.get("beta")),
        "marketCap": safe_float(profile.get("mktCap")),
        "country": profile.get("country"),

        # Series históricas
        "years": years,
        "revenue_hist": revenue_hist,
        "ebit_hist": ebit_hist,
        "operating_cf_hist": ocf_hist,
        "capex_hist": capex_hist,
        "free_cf_hist": free_cf_hist,
        "shares_hist": shares_hist,
        "net_debt_hist": net_debt_hist,
        "fcf_per_share_hist": fcf_per_share_hist,

        # Crecimiento YoY calculado localmente
        "revenueGrowth_calc": revenueGrowth_calc,
        "ebitgrowth_calc": ebitgrowth_calc,
        "operatingCashFlowGrowth_calc": ocfGrowth_calc,
        "freeCashFlowGrowth_calc": fcfGrowth_calc,
        "debtGrowth_calc": debtGrowth_calc,

        # Márgenes / eficiencia
        "gross_margin_last": gross_margin_last,
        "operating_margin_last": operating_margin_last,
        "fcf_margin_last": fcf_margin_last,
        "roic_last": roic_last,

        # Apalancamiento
        "net_debt_to_ebitda_last": net_debt_to_ebitda_last,
        "leverage_ok": leverage_ok,

        # Disciplina de capital
        "buyback_pct_5y": buyback_pct_5y,
        "fcf_per_share_slope_5y": slope_fcf_ps,
        "is_quality_compounder": is_quality_compounder,
    }

    # Mezclar señales de calidad y crecimiento traídas en pasos anteriores
    if quality_row is not None:
        snap["altmanZScore"] = quality_row.get("altmanZScore")
        snap["piotroskiScore"] = quality_row.get("piotroskiScore")

    if growth_row is not None:
        snap["revenueGrowth"] = growth_row.get("revenueGrowth")
        snap["ebitgrowth"] = growth_row.get("ebitgrowth")
        snap["operatingCashFlowGrowth"] = growth_row.get("operatingCashFlowGrowth")
        snap["freeCashFlowGrowth"] = growth_row.get("freeCashFlowGrowth")
        snap["debtGrowth"] = growth_row.get("debtGrowth")
        snap["rev_CAGR_3y"] = growth_row.get("rev_CAGR_3y")
        snap["rev_CAGR_5y"] = growth_row.get("rev_CAGR_5y")
        snap["ocf_CAGR_3y"] = growth_row.get("ocf_CAGR_3y")
        snap["ocf_CAGR_5y"] = growth_row.get("ocf_CAGR_5y")
        snap["high_growth_flag"] = growth_row.get("high_growth_flag")

    # placeholders para las fases siguientes
    snap["insider_signal"] = None
    snap["insider_note"] = None
    snap["news_sentiment"] = None
    snap["news_note"] = None
    snap["transcript_summary"] = None
    snap["core_risk_note"] = None
    snap["why_it_matters"] = None
    snap["expected_return_cagr"] = None
    snap["valuation_note"] = None

    return snap
