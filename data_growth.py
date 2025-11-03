from typing import List, Dict
import pandas as pd
from fmp_api import get_key_metrics
from utils import safe_float, cagr
from config import HIGH_GROWTH_CAGR_THRESHOLD

def _build_growth_metrics_for_symbol(sym: str) -> Dict:
    """
    Aquí deberías mapear exactamente los campos que entrega tu endpoint real.
    Ejemplo de campos esperados:
      revenueGrowth, ebitgrowth, operatingCashFlowGrowth,
      freeCashFlowGrowth, debtGrowth,
      fiveYRevenueGrowthPerShare,
      fiveYOperatingCFGrowthPerShare,
      threeYRevenueGrowthPerShare, ...
    """
    km = get_key_metrics(sym)

    rev_yoy = safe_float(km.get("revenueGrowth"))
    ebit_yoy = safe_float(km.get("ebitgrowth"))
    ocf_yoy = safe_float(km.get("operatingCashFlowGrowth"))
    fcf_yoy = safe_float(km.get("freeCashFlowGrowth"))
    debt_yoy = safe_float(km.get("debtGrowth"))

    # Supongamos que FMP da crecimiento acumulado 3y y 5y por acción:
    rev_ps_3y_total = safe_float(km.get("threeYRevenueGrowthPerShare"))
    rev_ps_5y_total = safe_float(km.get("fiveYRevenueGrowthPerShare"))
    ocf_ps_3y_total = safe_float(km.get("threeYOperatingCFGrowthPerShare"))
    ocf_ps_5y_total = safe_float(km.get("fiveYOperatingCFGrowthPerShare"))

    rev_CAGR_3y = cagr(rev_ps_3y_total, 3) if rev_ps_3y_total is not None else None
    rev_CAGR_5y = cagr(rev_ps_5y_total, 5) if rev_ps_5y_total is not None else None
    ocf_CAGR_3y = cagr(ocf_ps_3y_total, 3) if ocf_ps_3y_total is not None else None
    ocf_CAGR_5y = cagr(ocf_ps_5y_total, 5) if ocf_ps_5y_total is not None else None

    # high_growth_flag si alguna CAGR >= threshold
    high_growth_flag = False
    for g in [rev_CAGR_3y, rev_CAGR_5y, ocf_CAGR_3y, ocf_CAGR_5y]:
        if g is not None and g >= HIGH_GROWTH_CAGR_THRESHOLD:
            high_growth_flag = True
            break

    return {
        "ticker": sym,
        "revenueGrowth": rev_yoy,
        "ebitgrowth": ebit_yoy,
        "operatingCashFlowGrowth": ocf_yoy,
        "freeCashFlowGrowth": fcf_yoy,
        "debtGrowth": debt_yoy,
        "rev_CAGR_3y": rev_CAGR_3y,
        "rev_CAGR_5y": rev_CAGR_5y,
        "ocf_CAGR_3y": ocf_CAGR_3y,
        "ocf_CAGR_5y": ocf_CAGR_5y,
        "high_growth_flag": high_growth_flag,
    }

def get_growth_signals_bulk(symbols: List[str]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            rows.append(_build_growth_metrics_for_symbol(sym))
        except Exception:
            continue
    return pd.DataFrame(rows)
