from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import numpy as np
import pandas as pd


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _pct_change(curr, prev):
    """
    (curr - prev)/|prev|
    """
    c = _safe_float(curr)
    p = _safe_float(prev)
    if c is None or p is None or p == 0:
        return None
    return (c - p) / abs(p)


def _cagr(first, last, years):
    """
    CAGR = (last/first)^(1/years) - 1
    years ~ cantidad_de_periodos-1
    """
    a = _safe_float(first)
    b = _safe_float(last)
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    if years <= 0:
        return None
    try:
        return (b / a) ** (1.0 / years) - 1.0
    except Exception:
        return None


def _linear_trend(vals: List[Optional[float]]) -> Optional[float]:
    """
    slope simple de una serie (x = 0..n-1)
    """
    s = pd.Series(vals, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return None
    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def compute_core_metrics(
    ticker: str,
    profile: Dict[str, Any],
    income_hist: List[Dict[str, Any]],     # más reciente primero
    cash_hist: List[Dict[str, Any]],       # más reciente primero
    balance_hist: List[Dict[str, Any]],    # más reciente primero
    ratios_hist: List[Dict[str, Any]],     # más reciente primero
    shares_hist: List[Dict[str, Any]],     # más reciente primero (diluited shares)
) -> Dict[str, Any]:
    """
    Devuelve TODO lo que app.py / orchestrator necesitan en una sola pasada.
    """

    # -------------------------
    # 0. Identidad básica
    # -------------------------
    name = (
        profile.get("companyName")
        or profile.get("companyNameLong")
        or profile.get("companyNameShort")
        or profile.get("companyName")
    )

    sector = profile.get("sector")
    industry = profile.get("industry")

    market_cap = (
        profile.get("mktCap")
        or profile.get("marketCap")
        or profile.get("marketCapIntraday")
    )

    business_summary = profile.get("description", "")

    # -------------------------
    # 1. Scores financieros base
    # -------------------------
    r0 = ratios_hist[0] if isinstance(ratios_hist, list) and ratios_hist else {}

    altman_z = r0.get("altmanZScore") or r0.get("altmanZScoreTTM")
    try:
        if altman_z is not None:
            altman_z = float(altman_z)
    except Exception:
        altman_z = None

    piotroski = r0.get("piotroskiScore") or r0.get("piotroskiFScore")
    try:
        if piotroski is not None:
            piotroski = float(piotroski)
    except Exception:
        piotroski = None

    net_debt_to_ebitda = (
        r0.get("netDebtToEBITDA")
        or r0.get("netDebtToEBITDARatio")
        or r0.get("netDebtToEBITDATTM")
    )
    try:
        if net_debt_to_ebitda is not None:
            net_debt_to_ebitda = float(net_debt_to_ebitda)
    except Exception:
        net_debt_to_ebitda = None

    # -------------------------
    # 2. Construir dataframe histórico anual
    #    Necesitamos revenue, OCF, CapEx, FCF, Deuda neta, acciones
    # -------------------------
    # income_hist[i] ~ { date, revenue, ebitda, ... } más reciente primero
    # cash_hist[i]   ~ { date, operatingCashFlow, capitalExpenditure, freeCashFlow, ... }
    # balance_hist[i]~ { date, totalDebt, cashAndShortTermInvestments, ... }
    # shares_hist[i] ~ { date, sharesDiluted, ... }

    def _row_at(lst, idx):
        return lst[idx] if (isinstance(lst, list) and len(lst) > idx) else {}

    # Armamos una tabla (más viejo -> más nuevo) para series limpias
    rows_joined = []
    # tomamos el máximo largo que tengamos de cualquiera
    max_len = max(len(income_hist), len(cash_hist), len(balance_hist), len(shares_hist))

    for i in range(max_len):
        inc = _row_at(income_hist, i)
        csh = _row_at(cash_hist, i)
        bal = _row_at(balance_hist, i)
        shs = _row_at(shares_hist, i)

        # fiscal date más reciente primero, pero guardamos por año/fiscalDate
        # luego invertimos
        rows_joined.append({
            "fiscalDate": (
                inc.get("calendarYear") or
                inc.get("date") or
                inc.get("fillingDate") or
                csh.get("calendarYear") or
                csh.get("date") or
                csh.get("fillingDate")
            ),
            "revenue": inc.get("revenue"),
            "operatingCashFlow": csh.get("operatingCashFlow"),
            "capitalExpenditure": csh.get("capitalExpenditure"),
            "freeCashFlow": csh.get("freeCashFlow"),
            "totalDebt": bal.get("totalDebt"),
            "cashAndShortTermInvestments": (
                bal.get("cashAndShortTermInvestments") or
                bal.get("cashAndCashEquivalents")
            ),
            "sharesDiluted": shs.get("sharesDiluted") or shs.get("weightedAverageShsOutDil"),
        })

    hist_df = pd.DataFrame(rows_joined)

    # quitamos filas totalmente vacías
    hist_df = hist_df.dropna(how="all").copy()

    # invertimos a cronológico ASC (viejo -> nuevo)
    hist_df = hist_df.iloc[::-1].reset_index(drop=True)

    # métricas derivadas
    hist_df["fcf_calc"] = hist_df["operatingCashFlow"] - hist_df["capitalExpenditure"]
    # prefer freeCashFlow de la API si viene, si no usamos calc
    hist_df["fcf_final"] = hist_df["freeCashFlow"].fillna(hist_df["fcf_calc"])

    # deuda neta
    hist_df["net_debt"] = (
        hist_df["totalDebt"].astype(float).fillna(0.0)
        - hist_df["cashAndShortTermInvestments"].astype(float).fillna(0.0)
    )

    # FCF por acción
    hist_df["fcf_per_share"] = hist_df.apply(
        lambda r: (
            float(r["fcf_final"]) / float(r["sharesDiluted"])
            if (
                r.get("fcf_final") is not None
                and r.get("sharesDiluted") not in [None, 0, "0"]
            )
            else np.nan
        ),
        axis=1,
    )

    # -------------------------
    # 3. Crecimientos YoY (último vs penúltimo)
    # -------------------------
    # usamos income_hist[0] más reciente, income_hist[1] año previo, etc
    rev_growth = _pct_change(
        _row_at(income_hist, 0).get("revenue"),
        _row_at(income_hist, 1).get("revenue"),
    )

    ocf_growth = _pct_change(
        _row_at(cash_hist, 0).get("operatingCashFlow"),
        _row_at(cash_hist, 1).get("operatingCashFlow"),
    )

    fcf_growth = _pct_change(
        _row_at(cash_hist, 0).get("freeCashFlow"),
        _row_at(cash_hist, 1).get("freeCashFlow"),
    )

    debt_growth = _pct_change(
        _row_at(balance_hist, 0).get("totalDebt"),
        _row_at(balance_hist, 1).get("totalDebt"),
    )

    # -------------------------
    # 4. CAGR multi-año (5y aprox)
    #    tomamos primer valor no-nulo y último
    # -------------------------
    def _first_last(series: pd.Series):
        s = series.dropna()
        if len(s) < 2:
            return (None, None, 0)
        return (s.iloc[0], s.iloc[-1], len(s) - 1)

    rev_first, rev_last, rev_years = _first_last(hist_df["revenue"].astype(float))
    ocf_first, ocf_last, ocf_years = _first_last(hist_df["operatingCashFlow"].astype(float))

    rev_cagr_5y = _cagr(rev_first, rev_last, rev_years)
    ocf_cagr_5y = _cagr(ocf_first, ocf_last, ocf_years)

    # -------------------------
    # 5. Buybacks y slope de FCF/acción
    # -------------------------
    buyback_pct_5y = None
    shares_valid = hist_df["sharesDiluted"].dropna()
    if len(shares_valid) >= 2:
        start_sh = _safe_float(shares_valid.iloc[0])
        end_sh   = _safe_float(shares_valid.iloc[-1])
        if start_sh and start_sh != 0:
            buyback_pct_5y = (start_sh - end_sh) / start_sh

    fcf_per_share_slope_5y = _linear_trend(hist_df["fcf_per_share"].tolist())

    # -------------------------
    # 6. Bandera "is_quality_compounder"
    #    criterio simple:
    #      slope fcf/acc > 0
    #      recompras >5%
    #      netDebt_to_EBITDA < 2
    # -------------------------
    def _pos(x): 
        try:
            return float(x) > 0
        except Exception:
            return False

    compounder = (
        _pos(fcf_per_share_slope_5y)
        and (buyback_pct_5y is not None and buyback_pct_5y > 0.05)
        and (net_debt_to_ebitda is not None and net_debt_to_ebitda < 2)
    )

    # -------------------------
    # 7. Heurística moat muy simple (puedes tunear luego)
    #    usamos ROIC / margen FCF si vienen en ratios_hist[0]
    # -------------------------
    roic = r0.get("roic") or r0.get("returnOnInvestedCapital")
    try:
        roic = float(roic) if roic is not None else None
    except Exception:
        roic = None

    fcf_margin = r0.get("freeCashFlowMargin") or r0.get("fcfMargin")
    try:
        fcf_margin = float(fcf_margin) if fcf_margin is not None else None
    except Exception:
        fcf_margin = None

    moat_flag = "—"
    if roic is not None and fcf_margin is not None:
        if roic >= 0.15 and fcf_margin >= 0.10:
            moat_flag = "fuerte"
        elif roic >= 0.10:
            moat_flag = "media"

    # -------------------------
    # 8. net_debt_hist para gráfico
    # -------------------------
    net_debt_hist = hist_df["net_debt"].replace([np.inf, -np.inf], np.nan).tolist()
    fcf_ps_hist   = hist_df["fcf_per_share"].replace([np.inf, -np.inf], np.nan).tolist()
    shares_hist_v = hist_df["sharesDiluted"].replace([np.inf, -np.inf], np.nan).tolist()
    years_vec     = hist_df["fiscalDate"].astype(str).tolist()

    # -------------------------
    # 9. Armar dict final
    # -------------------------
    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "industry": industry,
        "marketCap": market_cap,
        "business_summary": business_summary,

        "altmanZScore": altman_z,
        "piotroskiScore": piotroski,
        "netDebt_to_EBITDA": net_debt_to_ebitda,
        "moat_flag": moat_flag,

        "revenueGrowth": rev_growth,
        "operatingCashFlowGrowth": ocf_growth,
        "freeCashFlowGrowth": fcf_growth,
        "debtGrowth": debt_growth,

        "rev_CAGR_5y": rev_cagr_5y,
        "ocf_CAGR_5y": ocf_cagr_5y,

        "buyback_pct_5y": buyback_pct_5y,
        "fcf_per_share_slope_5y": fcf_per_share_slope_5y,
        "is_quality_compounder": compounder,

        "years": years_vec,
        "fcf_per_share_hist": fcf_ps_hist,
        "shares_hist": shares_hist_v,
        "net_debt_hist": net_debt_hist,
    }
