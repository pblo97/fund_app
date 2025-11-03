import math
import numpy as np
import pandas as pd
from typing import List, Sequence

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def cagr(total_growth_fraction: float, years: float) -> float | None:
    """
    total_growth_fraction = (final / initial) - 1
    years = número de años entre ambos puntos
    Retorna CAGR anualizado, ej 0.15 = 15% anual.
    """
    try:
        base = 1.0 + float(total_growth_fraction)
        if base <= 0 or years <= 0:
            return None
        return (base ** (1.0 / years)) - 1.0
    except Exception:
        return None

def linear_slope_y_per_year(years: Sequence[float], values: Sequence[float]) -> float | None:
    """
    Calcula la pendiente aproximada (regresión lineal simple)
    de 'values' en función de 'years'.
    Retorna unidades de "valor por año".
    """
    try:
        if len(years) < 2 or len(values) < 2:
            return None
        x = np.array(years, dtype=float)
        y = np.array(values, dtype=float)
        # y = a*x + b -> slope = a
        A = np.vstack([x, np.ones_like(x)]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(slope)
    except Exception:
        return None

def pct_change(old, new):
    try:
        old = float(old)
        new = float(new)
        if old == 0:
            return None
        return (new - old) / abs(old)
    except Exception:
        return None

def recent_growth(series: List[float]) -> float | None:
    """
    Crecimiento año contra año usando los 2 últimos puntos de la serie.
    """
    if not series or len(series) < 2:
        return None
    return pct_change(series[-2], series[-1])

def series_tail(series: List[float], n=5) -> List[float]:
    return list(series[-n:])
