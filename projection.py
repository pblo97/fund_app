from typing import Dict, Any
from config import TARGET_YEARS_FORWARD, DEFAULT_FCF_MULTIPLE
from utils import safe_float

def estimate_forward_return(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Paso 4:
    - Estimar FCF/acción futuro simple
    - Asignar múltiplo terminal
    - Derivar CAGR esperado
    """
    fcf_hist = snapshot.get("fcf_per_share_hist", [])
    if not fcf_hist or len(fcf_hist) < 2:
        return {
            "expected_return_cagr": None,
            "valuation_note": "No hay suficiente historia de FCF/acción."
        }

    # Asumimos que el último valor de FCF/acción es base:
    last_fcf_ps = safe_float(fcf_hist[-1])
    if last_fcf_ps is None or last_fcf_ps <= 0:
        return {
            "expected_return_cagr": None,
            "valuation_note": "FCF/acción actual no usable."
        }

    # Crecimiento esperado: usamos slope anual (fcf_per_share_slope_5y)
    slope = snapshot.get("fcf_per_share_slope_5y")
    if slope is None:
        slope = 0.0

    # Proyectar FCF/acción a N años = último + slope*N
    projected_fcf_ps = last_fcf_ps + slope * TARGET_YEARS_FORWARD
    if projected_fcf_ps <= 0:
        projected_fcf_ps = last_fcf_ps  # fallback defensivo

    # Precio objetivo = múltiplo * FCF/acción proyectado
    target_price = projected_fcf_ps * DEFAULT_FCF_MULTIPLE

    # Necesitamos precio actual
    # profile.mktCap / shares_outstanding_last ~ aproximación
    # snapshot no guardó shares_outstanding_last explícito separado,
    # pero podemos tomar el último de shares_hist:
    shares_hist = snapshot.get("shares_hist", [])
    market_cap = snapshot.get("marketCap")
    if market_cap and shares_hist:
        last_shares = shares_hist[-1] or None
    else:
        last_shares = None

    if not (market_cap and last_shares and last_shares > 0):
        return {
            "expected_return_cagr": None,
            "valuation_note": "No pude estimar el precio actual."
        }

    current_price = market_cap / last_shares

    # CAGR esperado
    # (target/current)^(1/N) - 1
    try:
        expected_cagr = (target_price / current_price) ** (1.0 / TARGET_YEARS_FORWARD) - 1.0
    except Exception:
        expected_cagr = None

    note = (
        f"Asume múltiplo {DEFAULT_FCF_MULTIPLE}x FCF/acc en "
        f"{TARGET_YEARS_FORWARD} años con slope FCF/acc ~ {slope:.3f}."
    )

    return {
        "expected_return_cagr": expected_cagr,
        "valuation_note": note,
    }
