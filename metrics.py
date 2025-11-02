# metrics.py
#
# CALCULO DE MÉTRICAS CORE PARA UN TICKER
#
# Entradas:
#   profile: list[dict] (normalmente [0] = info empresa)
#   ratios_hist: list[dict] (más reciente primero)
#   income_hist: list[dict] (anual, más reciente primero)
#   balance_hist: list[dict] (anual, más reciente primero)
#   cash_hist: list[dict] (anual, más reciente primero)
#
# Salida:
#   dict con TODAS las claves que usa la app/orchestrator:
#     - ticker, name, sector, industry, marketCap, beta, business_summary
#     - altmanZScore, piotroskiScore
#     - revenueGrowth, operatingCashFlowGrowth, freeCashFlowGrowth, debtGrowth
#     - rev_CAGR_5y, rev_CAGR_3y, ocf_CAGR_5y, ocf_CAGR_3y
#     - netDebt_to_EBITDA, moat_flag
#     - years, fcf_per_share_hist, shares_hist, net_debt_hist
#
# Cualquier cosa que no podamos calcular -> None
# para que luego la UI muestre "—".
#

from typing import Any, Dict, List, Tuple
import math


# -------------------------
# Helpers genéricos
# -------------------------

def _safe_get(d: dict, *keys, default=None):
    """
    Devuelve la PRIMERA key presente y no None en d.
    _safe_get(row, "weightedAverageShsOut", "weightedAverageShsOutDil")
    """
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _pct_growth(latest: float, prev: float) -> float | None:
    """
    Crecimiento simple year-over-year: (latest - prev) / |prev|.
    Devuelve None si no se puede.
    """
    try:
        if prev is None or latest is None:
            return None
        prev_f = float(prev)
        latest_f = float(latest)
        if prev_f == 0:
            return None
        return (latest_f - prev_f) / abs(prev_f)
    except Exception:
        return None


def _cagr_from_series(vals: List[float], years: int) -> float | None:
    """
    Calcula CAGR usando el primer y último valor de la serie.
    - vals debe estar en orden cronológico viejo->reciente.
    - years = número de años entre esos extremos.
    Si years < 2 o valores inválidos, devuelve None.
    """
    try:
        if not vals or len(vals) < 2 or years < 2:
            return None
        start = vals[0]
        end = vals[-1]
        if start is None or end is None:
            return None
        start_f = float(start)
        end_f = float(end)
        if start_f <= 0 or end_f <= 0:
            # evitamos dividir por 0 / log negativos (no definimos CAGR si valores <=0)
            return None
        ratio = end_f / start_f
        # si ratio <=0, no sirve
        if ratio <= 0:
            return None
        # CAGR clásica
        return (ratio ** (1.0 / years)) - 1.0
    except Exception:
        return None


def _extract_profile_info(profile: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    profile suele venir como lista [ { ... } ].
    Sacamos nombre, sector, industry, marketCap, beta, summary.
    """
    p0 = profile[0] if isinstance(profile, list) and profile else {}

    name = (
        p0.get("companyName")
        or p0.get("companyNameLong")
        or p0.get("companyNameShort")
        or p0.get("companyName")  # fallback redundante pero inofensivo
    )
    sector = p0.get("sector")
    industry = p0.get("industry")
    market_cap = (
        p0.get("mktCap")
        or p0.get("marketCap")
        or p0.get("marketCapIntraday")
    )
    beta = p0.get("beta")
    business_summary = p0.get("description", "")

    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "marketCap": market_cap,
        "beta": beta,
        "business_summary": business_summary,
    }


def _extract_balance_leverage(
    ratios_hist: List[Dict[str, Any]],
    balance_hist: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    - netDebt_to_EBITDA: tratamos de leerlo del ratio más reciente.
    - net_debt_hist y years_balance_crono: trayectoria de deuda neta.
    """

    # ----- netDebt_to_EBITDA -----
    nde = None
    if ratios_hist and isinstance(ratios_hist, list):
        r0 = ratios_hist[0]  # más reciente
        nde = (
            r0.get("netDebtToEBITDA")
            or r0.get("netDebtToEBITDARatio")
            or r0.get("netDebt_to_EBITDA")
        )
        try:
            nde = float(nde) if nde is not None else None
        except Exception:
            nde = None

    # ----- historial de net debt -----
    # balance_hist viene más reciente primero, lo invertimos a viejo->nuevo
    bal_rev = list(reversed(balance_hist or []))

    years_balance = []
    net_debt_hist = []

    for row in bal_rev:
        # año / etiqueta temporal
        year_tag = (
            row.get("calendarYear")
            or row.get("date")
            or row.get("fillingDate")
            or row.get("acceptedDate")
        )

        # si ya viene netDebt directamente, bacán
        net_debt_val = row.get("netDebt")

        # si no, calculamos totalDebt - cash
        if net_debt_val is None:
            total_debt = row.get("totalDebt")
            cash_equiv = (
                row.get("cashAndShortTermInvestments")
                or row.get("cashAndCashEquivalents")
            )
            try:
                if total_debt is not None and cash_equiv is not None:
                    net_debt_val = float(total_debt) - float(cash_equiv)
            except Exception:
                net_debt_val = None

        years_balance.append(year_tag)
        net_debt_hist.append(net_debt_val)

    return {
        "netDebt_to_EBITDA": nde,
        "net_debt_hist": net_debt_hist,
        "years_balance_crono": years_balance,
    }


def _extract_cashflow_per_share(
    income_hist: List[Dict[str, Any]],
    cash_hist: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Construímos series cronológicas (viejo->nuevo) para:
    - FCF por acción (fcf_per_share_hist)
    - acciones diluidas (shares_hist)
    - eje temporal (years_cf_crono)

    income_hist[i] y cash_hist[i] son anual más reciente primero,
    así que invertimos para viejo->nuevo y emparejamos por índice.
    """

    inc_rev = list(reversed(income_hist or []))
    cf_rev = list(reversed(cash_hist or []))

    years_cf = []
    fcf_per_share_hist = []
    shares_hist = []

    n = min(len(inc_rev), len(cf_rev))
    for i in range(n):
        inc_row = inc_rev[i]
        cf_row = cf_rev[i]

        # etiqueta temporal
        year_tag = (
            inc_row.get("calendarYear")
            or inc_row.get("date")
            or inc_row.get("fillingDate")
            or inc_row.get("acceptedDate")
        )

        # acciones (diluidas si hay)
        shares_val = _safe_get(
            inc_row,
            "weightedAverageShsOutDil",
            "weightedAverageShsOut",
            "weightedAverageShsOutDiluted",
            default=None
        )

        # FCF absoluto
        fcf_val = cf_row.get("freeCashFlow")

        # FCF / acción
        fcf_ps = None
        try:
            if shares_val and fcf_val is not None:
                fcf_ps = float(fcf_val) / float(shares_val)
        except Exception:
            fcf_ps = None

        years_cf.append(year_tag)
        fcf_per_share_hist.append(fcf_ps)
        shares_hist.append(shares_val)

    return {
        "years_cf_crono": years_cf,
        "fcf_per_share_hist": fcf_per_share_hist,
        "shares_hist": shares_hist,
    }


def _infer_moat_flag(ratios_hist: List[Dict[str, Any]]) -> str:
    """
    Heurística muy simple:
    - si ROIC alto y margen FCF decente → "fuerte"
    - si ROIC OK → "media"
    - sino → "—"
    """
    if not ratios_hist:
        return "—"

    r0 = ratios_hist[0]
    roic = r0.get("roic") or r0.get("returnOnInvestedCapital")
    fcf_margin = r0.get("freeCashFlowMargin") or r0.get("fcfMargin")

    try:
        if roic is not None:
            roic = float(roic)
    except Exception:
        roic = None

    try:
        if fcf_margin is not None:
            fcf_margin = float(fcf_margin)
    except Exception:
        fcf_margin = None

    if (roic is not None and roic >= 0.15) and (fcf_margin is not None and fcf_margin >= 0.10):
        return "fuerte"
    if (roic is not None and roic >= 0.10):
        return "media"
    return "—"


def _extract_quality_scores_and_growth(
    ratios_hist: List[Dict[str, Any]],
    income_hist: List[Dict[str, Any]],
    cash_hist: List[Dict[str, Any]],
    balance_hist: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    1. Sacamos Altman Z y Piotroski del ratio más reciente.
    2. Growth último año (revenue, OCF, FCF, deuda).
    3. CAGR 5y y 3y para revenue y OCF.
       - revenue lo sacamos de income_hist
       - operatingCashFlow lo sacamos de cash_hist
    """

    # ---------- 1. Altman Z / Piotroski ----------
    altman_z = None
    piotroski = None
    if ratios_hist and isinstance(ratios_hist, list):
        r0 = ratios_hist[0]
        altman_z = (
            r0.get("altmanZScore")
            or r0.get("altmanScore")
            or r0.get("altmanZ")
        )
        piotroski = (
            r0.get("piotroskiScore")
            or r0.get("piotroskiFScore")
            or r0.get("piotroski")
        )
        try:
            altman_z = float(altman_z) if altman_z is not None else None
        except Exception:
            altman_z = None
        try:
            # piotroski suele ser entero 0-9
            if piotroski is not None:
                piotroski = float(piotroski)
        except Exception:
            piotroski = None

    # ---------- 2. Crecimiento último año ----------
    # Tomamos income_hist[0] y income_hist[1] para revenue
    rev_growth = None
    if len(income_hist) >= 2:
        rev_latest = income_hist[0].get("revenue")
        rev_prev = income_hist[1].get("revenue")
        rev_growth = _pct_growth(rev_latest, rev_prev)

    # OCF growth (operatingCashFlow)
    ocf_growth = None
    if len(cash_hist) >= 2:
        ocf_latest = cash_hist[0].get("operatingCashFlow")
        ocf_prev = cash_hist[1].get("operatingCashFlow")
        ocf_growth = _pct_growth(ocf_latest, ocf_prev)

    # FCF growth
    fcf_growth = None
    if len(cash_hist) >= 2:
        fcf_latest = cash_hist[0].get("freeCashFlow")
        fcf_prev = cash_hist[1].get("freeCashFlow")
        fcf_growth = _pct_growth(fcf_latest, fcf_prev)

    # Deuda growth (usamos netDebt del balance si existe)
    debt_growth = None
    if len(balance_hist) >= 2:
        # balance_hist[0] es más reciente
        nd_latest = balance_hist[0].get("netDebt")
        nd_prev = balance_hist[1].get("netDebt")
        if nd_latest is None or nd_prev is None:
            # si netDebt no está, aproximamos totalDebt - cash
            def _net_debt_fallback(row):
                td = row.get("totalDebt")
                cash_ = (
                    row.get("cashAndShortTermInvestments")
                    or row.get("cashAndCashEquivalents")
                )
                try:
                    if td is not None and cash_ is not None:
                        return float(td) - float(cash_)
                except Exception:
                    return None
                return None

            if nd_latest is None:
                nd_latest = _net_debt_fallback(balance_hist[0])
            if nd_prev is None:
                nd_prev = _net_debt_fallback(balance_hist[1])

        debt_growth = _pct_growth(nd_latest, nd_prev)

    # ---------- 3. CAGR multi-año ----------
    # Para revenue CAGR usamos income_hist invertido a cronológico viejo->nuevo
    inc_rev = list(reversed(income_hist or []))
    rev_series = [row.get("revenue") for row in inc_rev]
    # Para OCF CAGR usamos cash_hist invertido
    cash_rev = list(reversed(cash_hist or []))
    ocf_series = [row.get("operatingCashFlow") for row in cash_rev]

    # cuántos años reales hay?
    # Si tengo N puntos anuales cronológicos, años = N-1
    n_rev_years = len(rev_series) - 1
    n_ocf_years = len(ocf_series) - 1

    # CAGR 5y = usar hasta 5 años atrás si existe
    def _cagr_window(series: List[Any], max_years: int) -> float | None:
        """
        Toma los últimos max_years+1 puntos (para tener max_years de diferencia).
        Ej: max_years=5 -> intenta usar 6 puntos si hay.
        """
        if not series:
            return None
        if len(series) < 2:
            return None

        # Queremos ventana de ~5 años => 6 puntos
        # pero si no tenemos tantos datos, usamos lo que haya.
        window = series[-(max_years+1):]  # últimos 6 para 5y
        yrs = len(window) - 1
        # convertimos posibles None a None -> _cagr_from_series lo validará
        return _cagr_from_series(window, yrs)

    rev_CAGR_5y = _cagr_window(rev_series, 5)
    rev_CAGR_3y = _cagr_window(rev_series, 3)
    ocf_CAGR_5y = _cagr_window(ocf_series, 5)
    ocf_CAGR_3y = _cagr_window(ocf_series, 3)

    return {
        "altmanZScore": altman_z,
        "piotroskiScore": piotroski,
        "revenueGrowth": rev_growth,
        "operatingCashFlowGrowth": ocf_growth,
        "freeCashFlowGrowth": fcf_growth,
        "debtGrowth": debt_growth,
        "rev_CAGR_5y": rev_CAGR_5y,
        "rev_CAGR_3y": rev_CAGR_3y,
        "ocf_CAGR_5y": ocf_CAGR_5y,
        "ocf_CAGR_3y": ocf_CAGR_3y,
    }


# -------------------------
# FUNCIÓN PRINCIPAL
# -------------------------

def compute_core_financial_metrics(
    ticker: str,
    profile: List[Dict[str, Any]],
    ratios_hist: List[Dict[str, Any]],
    income_hist: List[Dict[str, Any]],
    balance_hist: List[Dict[str, Any]],
    cash_hist: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produce un snapshot homogéneo para la app.
    Todas las claves que la UI/orchestrator esperan salen de acá.
    """

    prof_info   = _extract_profile_info(profile)
    lev_info    = _extract_balance_leverage(ratios_hist, balance_hist)
    cfps_info   = _extract_cashflow_per_share(income_hist, cash_hist)
    qual_growth = _extract_quality_scores_and_growth(
        ratios_hist,
        income_hist,
        cash_hist,
        balance_hist,
    )
    moat_flag   = _infer_moat_flag(ratios_hist)

    # Elegimos un eje temporal principal para los gráficos del Tab2.
    # Preferimos el eje de cashflow_per_share (years_cf_crono).
    # Si está vacío, usamos years_balance_crono.
    years = (
        cfps_info.get("years_cf_crono")
        or lev_info.get("years_balance_crono")
        or []
    )

    snapshot: Dict[str, Any] = {
        # --------- Identidad / descripción ---------
        "ticker":            ticker,
        "name":              prof_info.get("name"),
        "sector":            prof_info.get("sector"),
        "industry":          prof_info.get("industry"),
        "marketCap":         prof_info.get("marketCap"),
        "beta":              prof_info.get("beta"),
        "business_summary":  prof_info.get("business_summary", ""),

        # --------- Calidad financiera / riesgo ---------
        "altmanZScore":      qual_growth.get("altmanZScore"),
        "piotroskiScore":    qual_growth.get("piotroskiScore"),
        "netDebt_to_EBITDA": lev_info.get("netDebt_to_EBITDA"),
        "moat_flag":         moat_flag,

        # --------- Crecimiento reciente (últ. FY) ---------
        "revenueGrowth":              qual_growth.get("revenueGrowth"),
        "operatingCashFlowGrowth":    qual_growth.get("operatingCashFlowGrowth"),
        "freeCashFlowGrowth":         qual_growth.get("freeCashFlowGrowth"),
        "debtGrowth":                 qual_growth.get("debtGrowth"),

        # --------- CAGR multi-año ---------
        "rev_CAGR_5y":   qual_growth.get("rev_CAGR_5y"),
        "rev_CAGR_3y":   qual_growth.get("rev_CAGR_3y"),
        "ocf_CAGR_5y":   qual_growth.get("ocf_CAGR_5y"),
        "ocf_CAGR_3y":   qual_growth.get("ocf_CAGR_3y"),

        # --------- Series históricas para la ficha (Tab2) ---------
        "years":                years,
        "fcf_per_share_hist":   cfps_info.get("fcf_per_share_hist", []),
        "shares_hist":          cfps_info.get("shares_hist", []),
        "net_debt_hist":        lev_info.get("net_debt_hist", []),
    }

    return snapshot
