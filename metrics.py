# metrics.py
#
# Este módulo calcula métricas financieras "core" para un ticker,
# usando los dumps crudos que vienen de fmp_api:
#
#   profile: list[dict] (normalmente [0] = info de la empresa)
#   ratios_hist: list[dict] (más reciente primero)
#   income_hist: list[dict] (anual, más reciente primero)
#   balance_hist: list[dict] (anual, más reciente primero)
#   cash_hist: list[dict] (anual, más reciente primero)
#
# Devuelve un dict listo para que lo consuma la app y que luego
# el orchestrator pueda enriquecer con insiders/news/transcript.
#
# IMPORTANTE:
# - NO importamos config aquí.
# - NO usamos EXPECTED_RETURN_HURDLE ni MIN_ROE_TTM ni nada legacy.
# - Todo debe ser cálculos locales y heurísticas suaves.


from typing import Any, Dict, List, Tuple
import math


def _safe_get(d: dict, *keys, default=None):
    """
    Navega en cascada varias keys posibles y te da la primera que exista.
    Ej:
        _safe_get(row, "weightedAverageShsOut", "weightedAverageShsOutDil")
    """
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _extract_profile_info(profile: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Profile suele venir como lista [ { ... } ].
    Extraemos nombre, sector, industria, marketCap, beta.
    """
    p0 = profile[0] if isinstance(profile, list) and profile else {}

    name = p0.get("companyName") or p0.get("companyNameLong") or p0.get("companyNameShort")
    sector = p0.get("sector")
    industry = p0.get("industry")
    market_cap = (
        p0.get("mktCap")
        or p0.get("marketCap")
        or p0.get("marketCapIntraday")
    )
    beta = p0.get("beta")

    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "marketCap": market_cap,
        "beta": beta,
        # texto descriptivo corto opcional
        "business_summary": p0.get("description", ""),
    }


def _extract_leverage_metrics(ratios_hist: List[Dict[str, Any]], balance_hist: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Queremos netDebt_to_EBITDA y la serie histórica de deuda neta.
    ratios_hist[0] = más reciente.
    balance_hist = anual más reciente primero.
    """
    net_debt_to_ebitda = None
    if isinstance(ratios_hist, list) and ratios_hist:
        rh0 = ratios_hist[0]
        # FMP suele exponer netDebtToEBITDA o netDebtToEBITDARatio
        net_debt_to_ebitda = (
            rh0.get("netDebtToEBITDA")
            or rh0.get("netDebtToEBITDARatio")
        )

        # sanity cast
        try:
            if net_debt_to_ebitda is not None:
                net_debt_to_ebitda = float(net_debt_to_ebitda)
        except Exception:
            net_debt_to_ebitda = None

    # construir hist de deuda neta para gráficos
    years_hist = []
    net_debt_hist = []

    # balance_hist[i] más reciente primero → invertimos para verlo cronológico
    for row in reversed(balance_hist or []):
        year = (
            row.get("calendarYear")
            or row.get("date")
            or row.get("fillingDate")
            or row.get("acceptedDate")
        )
        # netDebt a veces viene directo
        net_debt_val = row.get("netDebt")

        # si no viene "netDebt", lo calculamos como totalDebt - cash
        if net_debt_val is None:
            total_debt = row.get("totalDebt")
            cash_equiv = (
                row.get("cashAndCashEquivalents")
                or row.get("cashAndShortTermInvestments")
            )
            if total_debt is not None and cash_equiv is not None:
                try:
                    net_debt_val = float(total_debt) - float(cash_equiv)
                except Exception:
                    net_debt_val = None

        years_hist.append(year)
        net_debt_hist.append(net_debt_val)

    return {
        "netDebt_to_EBITDA": net_debt_to_ebitda,
        "net_debt_hist": net_debt_hist,
        "years_for_net_debt": years_hist,  # lo usamos luego para mergear
    }


def _extract_cashflow_per_share(
    income_hist: List[Dict[str, Any]],
    cash_hist: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Construimos historiales (cronológicos) de:
    - FCF por acción
    - acciones en circulación
    también devolvemos vector years para alinear luego en la vista final.

    Suponemos:
    cash_hist[i]["freeCashFlow"]
    income_hist[i]["weightedAverageShsOut"] / ["weightedAverageShsOutDil"]
    """

    # Convertimos a cronológico viejo→reciente:
    inc_rev = list(reversed(income_hist or []))
    cfs_rev = list(reversed(cash_hist or []))

    years = []
    fcf_per_share_hist = []
    shares_hist = []

    # Emparejamos por índice (mismo orden temporal ya que ambos son anuales):
    n = min(len(inc_rev), len(cfs_rev))

    for i in range(n):
        inc_row = inc_rev[i]
        cf_row = cfs_rev[i]

        year = (
            inc_row.get("calendarYear")
            or inc_row.get("date")
            or inc_row.get("fillingDate")
            or inc_row.get("acceptedDate")
        )

        shares = _safe_get(
            inc_row,
            "weightedAverageShsOut",
            "weightedAverageShsOutDil",
            default=None
        )

        fcf = cf_row.get("freeCashFlow")

        # calculamos FCF/acción
        fcf_ps = None
        try:
            if shares and fcf is not None:
                # shares a veces viene como float grande tipo 15_000_000_000
                fcf_ps = float(fcf) / float(shares)
        except Exception:
            fcf_ps = None

        years.append(year)
        fcf_per_share_hist.append(fcf_ps)
        shares_hist.append(shares)

    return {
        "years_fcfps": years,
        "fcf_per_share_hist": fcf_per_share_hist,
        "shares_hist": shares_hist,
    }


def _infer_moat_flag(ratios_hist: List[Dict[str, Any]]) -> str:
    """
    Heurística MUY simple.
    Puedes tunear esto más adelante. La idea es tener algo legible/estable.
    Ejemplo:
    - si ROIC alto y margen FCF decente → "fuerte"
    - si algo aceptable → "media"
    - si nada claro → "—"
    """
    if not ratios_hist:
        return "—"

    r0 = ratios_hist[0]  # más reciente
    roic = r0.get("roic") or r0.get("returnOnInvestedCapital")
    fcf_margin = (
        r0.get("freeCashFlowMargin")
        or r0.get("fcfMargin")
    )

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

    # reglas MUY básicas, ajustables
    if (roic is not None and roic >= 0.15) and (fcf_margin is not None and fcf_margin >= 0.10):
        return "fuerte"
    if (roic is not None and roic >= 0.10):
        return "media"
    return "—"


def compute_core_financial_metrics(
    ticker: str,
    profile: List[Dict[str, Any]],
    ratios_hist: List[Dict[str, Any]],
    income_hist: List[Dict[str, Any]],
    balance_hist: List[Dict[str, Any]],
    cash_hist: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Punto de entrada principal.

    Devuelve un dict con:
    - Identidad básica (ticker, name, sector, industry, marketCap, beta)
    - Métricas clave (netDebt_to_EBITDA, moat_flag)
    - Series históricas para gráficos (years, fcf_per_share_hist, shares_hist, net_debt_hist)
    - business_summary corto
    """

    prof_info = _extract_profile_info(profile)
    lev_info = _extract_leverage_metrics(ratios_hist, balance_hist)
    fcfps_info = _extract_cashflow_per_share(income_hist, cash_hist)

    moat_flag = _infer_moat_flag(ratios_hist)

    # Ahora consolidamos las series históricas en una única escala de tiempo:
    # preferimos years_fcfps para los gráficos, y si falta usamos years_for_net_debt.
    years = fcfps_info.get("years_fcfps") or lev_info.get("years_for_net_debt") or []

    snapshot: Dict[str, Any] = {
        "ticker": ticker,
        "name": prof_info.get("name"),
        "sector": prof_info.get("sector"),
        "industry": prof_info.get("industry"),
        "marketCap": prof_info.get("marketCap"),
        "beta": prof_info.get("beta"),
        "business_summary": prof_info.get("business_summary", ""),

        "netDebt_to_EBITDA": lev_info.get("netDebt_to_EBITDA"),
        "moat_flag": moat_flag,

        # histórico para panel detalle
        "years": years,
        "fcf_per_share_hist": fcfps_info.get("fcf_per_share_hist", []),
        "shares_hist": fcfps_info.get("shares_hist", []),
        "net_debt_hist": lev_info.get("net_debt_hist", []),
    }

    # valores adicionales opcionales que podrías querer más adelante:
    # Podrías extraer ROE TTM, márgenes, etc.,
    # pero la app nueva ya no los exige en la tabla principal.
    # Puedes añadirlos acá si luego quieres mostrarlos:
    #
    # ej:
    # if ratios_hist:
    #     r0 = ratios_hist[0]
    #     snapshot["roe_ttm"] = r0.get("returnOnEquityTTM") or r0.get("roeTTM")

    return snapshot
