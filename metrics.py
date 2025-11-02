# metrics.py
#
# Define las métricas base para un ticker.
# Esta función NO hace llamados a la API ni calcula nada pesado.
# Solo:
#   - limpia y normaliza lo que viene de profile
#   - construye un esqueleto completo con TODAS las llaves que
#     el resto de la app (orchestrator / app.py) espera.
#
# orchestrator después:
#   - rellena Altman, Piotroski con get_scores_bulk()
#   - rellena growth, CAGRs, high_growth_flag con get_growth_bulk()
#   - arma históricos (fcf_per_share_hist, shares_hist, etc.)
#   - calcula buyback_pct_5y, slope FCF/acción, netDebt/EBITDA, etc.
#
# ventaja: siempre devolvemos el mismo shape y evitamos columnas vacías
# que luego rompan la UI o muestren "—" en todas partes sin sentido.


from __future__ import annotations
from typing import List, Dict, Any


def compute_core_financial_metrics(
    symbol: str,
    profile: List[Dict[str, Any]] | None,
    ratios_hist: List[Dict[str, Any]] | None,
    income_hist: List[Dict[str, Any]] | None,
    balance_hist: List[Dict[str, Any]] | None,
    cash_hist: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    """
    Crea el esqueleto base de métricas fundamentales para un ticker.

    NO calcula todavía:
      - Altman / Piotroski
      - Growth reciente (revenueGrowth, OCFGrowth, etc.)
      - CAGR 3y/5y
      - Flags de compounder

    Eso lo inyecta luego orchestrator al combinar:
      * get_scores_bulk()      -> Altman, Piotroski
      * get_growth_bulk()      -> revenueGrowth, debtGrowth, CAGRs, high_growth_flag
      * históricos anuales     -> slope FCF/acción, recompras, net_debt_to_ebitda_last

    Params
    ------
    symbol : str
        Ticker (ej. "AAPL")
    profile : list[dict] | None
        Resultado crudo de get_profile(symbol). Suele ser [ {companyName, sector, ...} ]
    ratios_hist, income_hist, balance_hist, cash_hist :
        Se dejan acá para compatibilidad de firma (pueden servir después si decides
        derivar más métricas contables directamente dentro de metrics.py).
        Hoy no las usamos dentro de esta función.

    Returns
    -------
    dict
        Un diccionario con TODAS las llaves que espera la app.
        Muchas vienen inicializadas en None o [] y luego se completan.
    """

    # "profile" típicamente viene como lista de un solo dict.
    p0 = profile[0] if isinstance(profile, list) and profile else {}

    # name fallback:
    # - si hay companyName úsalo
    # - si no, que sea el símbolo
    name_val = p0.get("companyName") or p0.get("companyName") or symbol

    # market cap puede venir como "mktCap" o "marketCap"
    mc_val = p0.get("mktCap") or p0.get("marketCap")

    return {
        # ---------------------------
        # Identidad / descripción
        # ---------------------------
        "ticker": symbol,
        "name": name_val,
        "sector": p0.get("sector"),
        "industry": p0.get("industry"),
        "marketCap": mc_val,
        "beta": p0.get("beta"),
        "business_summary": p0.get("description", ""),

        # ---------------------------
        # Salud financiera / riesgo
        # ---------------------------
        # netDebt_to_EBITDA:
        #   métrica "estática"/general si algún día decides sacarla directo de estados
        "netDebt_to_EBITDA": None,

        # net_debt_to_ebitda_last:
        #   la versión realmente importante para ti:
        #   última (más reciente) relación Deuda Neta / EBITDA calculada desde históricos
        "net_debt_to_ebitda_last": None,

        # moats / ventaja estructural (heurística que marcamos en orchestrator)
        "moat_flag": "—",

        # ---------------------------
        # Calidad contable / solvencia
        # ---------------------------
        # Altman Z y Piotroski salen de get_scores_bulk()
        "altmanZScore": None,
        "piotroskiScore": None,

        # ---------------------------
        # Crecimiento reciente (último FY)
        # ---------------------------
        # Estas vienen directo de get_growth_bulk():
        #   revenueGrowth, operatingCashFlowGrowth, freeCashFlowGrowth
        # y disciplina de deuda (debtGrowth <= 0)
        "revenueGrowth": None,
        "operatingCashFlowGrowth": None,
        "freeCashFlowGrowth": None,
        "debtGrowth": None,

        # ---------------------------
        # Crecimiento compuesto multianual
        # ---------------------------
        # rev_CAGR_*  -> CAGR de revenue por acción (3y / 5y)
        # ocf_CAGR_*  -> CAGR de OCF por acción (3y / 5y)
        "rev_CAGR_5y": None,
        "rev_CAGR_3y": None,
        "ocf_CAGR_5y": None,
        "ocf_CAGR_3y": None,

        # high_growth_flag:
        #   True si alguna de esas CAGRs >= ~15% anual compuesto
        "high_growth_flag": None,

        # ---------------------------
        # Compounder score (propietario)
        # ---------------------------
        # Estos se calculan con históricos en orchestrator:
        #   - fcf_per_share_slope_5y: pendiente lineal de FCF/acción
        #   - buyback_pct_5y: % de reducción en sharesDiluted (recompras)
        #   - is_quality_compounder: True si slope>0, recompras>5%, ND/EBITDA<2
        "fcf_per_share_slope_5y": None,
        "buyback_pct_5y": None,
        "is_quality_compounder": None,

        # ---------------------------
        # Series históricas para la vista detallada (tab 2)
        # ---------------------------
        # Estas listas se llenan con datos multi-year para graficar:
        #   years                -> eje X (fiscalDate ordenado)
        #   fcf_per_share_hist   -> FCF/acción año a año
        #   shares_hist          -> acciones diluidas (para ver buybacks/dilution)
        #   net_debt_hist        -> deuda neta año a año
        "years": [],
        "fcf_per_share_hist": [],
        "shares_hist": [],
        "net_debt_hist": [],

        # ---------------------------
        # Campos cualitativos que luego completa orchestrator.enrich()
        # ---------------------------
        # why_it_matters: argumento de inversión corto
        # core_risk_note: riesgo visible
        # insider / news / transcript summary también se agregan más tarde
        "why_it_matters": "",
        "core_risk_note": "",
    }
