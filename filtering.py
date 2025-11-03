import pandas as pd
from data_quality import passes_quality_hard

def premerge_candidates(universe_df: pd.DataFrame,
                        scores_df: pd.DataFrame,
                        growth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une universo + quality + growth.
    NO aplica todavía el filtro de leverage_ok (eso llega tras fundamentals).
    Limita a MAX_TICKERS_FOR_TEST más adelante.
    """
    df = (
        universe_df
        .merge(scores_df, on="ticker", how="left")
        .merge(growth_df, on="ticker", how="left")
    )

    # filtro duro inicial de Altman/Piotro/operating cash-flow growth etc.
    # acá aplicamos la parte "passes_quality_hard" usando Altman/Piotro
    mask_quality = df.apply(passes_quality_hard, axis=1)

    # ebitgrowth >= 0, operatingCashFlowGrowth >= 0
    mask_growth_sanity = (
        (df["ebitgrowth"] >= 0) &
        (df["operatingCashFlowGrowth"] >= 0)
    )

    df_filt = df[mask_quality & mask_growth_sanity].copy()
    return df_filt.reset_index(drop=True)

def final_filter_after_fundamentals(snapshots: list[dict]) -> list[dict]:
    """
    Aplica el filtro de deuda estructural (leverage_ok).
    """
    out = []
    for snap in snapshots:
        if snap.get("leverage_ok", False):
            out.append(snap)
    return out
