from typing import List
import pandas as pd
from config import ALTMAN_MIN, PIOTROSKI_MIN
from fmp_api import get_financial_scores_bulk
from utils import safe_float

def get_quality_signals_bulk(symbols: List[str]) -> pd.DataFrame:
    raw = get_financial_scores_bulk(symbols)
    rows = []
    for sym, dat in raw.items():
        rows.append({
            "ticker": sym,
            "altmanZScore": safe_float(dat.get("altmanZScore")),
            "piotroskiScore": safe_float(dat.get("piotroskiScore")),
        })
    df = pd.DataFrame(rows).dropna(subset=["altmanZScore", "piotroskiScore"])
    return df

def passes_quality_hard(row: pd.Series) -> bool:
    if row["altmanZScore"] is None or row["piotroskiScore"] is None:
        return False
    if row["altmanZScore"] < ALTMAN_MIN:
        return False
    if row["piotroskiScore"] < PIOTROSKI_MIN:
        return False
    return True
