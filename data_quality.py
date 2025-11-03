from typing import List
import pandas as pd
from config import ALTMAN_MIN, PIOTROSKI_MIN
from fmp_api import get_financial_scores_bulk
from utils import safe_float

def get_quality_signals_bulk(symbols: List[str]) -> pd.DataFrame:
    raw = get_financial_scores_bulk(symbols)

    rows = []
    for sym, dat in raw.items():
        altman = safe_float(dat.get("altmanZScore"))
        piot = safe_float(dat.get("piotroskiScore"))
        rows.append({
            "ticker": sym,
            "altmanZScore": altman,
            "piotroskiScore": piot,
        })

    # si no llegó nada, devolvemos DF vacío pero con las columnas correctas
    if not rows:
        return pd.DataFrame(
            columns=["ticker", "altmanZScore", "piotroskiScore"]
        )

    df = pd.DataFrame(rows)

    # si por alguna razón faltan columnas, créalas
    for col in ["altmanZScore", "piotroskiScore"]:
        if col not in df.columns:
            df[col] = None

    # ahora sí: filtrar NaN en estas dos métricas
    df = df.dropna(subset=["altmanZScore", "piotroskiScore"])
    return df.reset_index(drop=True)

def passes_quality_hard(row: pd.Series) -> bool:
    # protegemos accesos con get
    altman = row.get("altmanZScore")
    piot = row.get("piotroskiScore")

    if altman is None or piot is None:
        return False
    if altman < ALTMAN_MIN:
        return False
    if piot < PIOTROSKI_MIN:
        return False
    return True
