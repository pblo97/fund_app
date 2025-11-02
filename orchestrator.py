from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import math

from fmp_api import (
    run_screener_for_exchange,
    get_financial_scores_batch,
    get_growth_batch,
)

EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]


def build_universe_largecaps() -> pd.DataFrame:
    """
    Junta el screener de NYSE/NASDAQ/AMEX ya filtrado a >=10B mktcap.
    Devuelve DataFrame con columnas al menos:
      symbol, companyName, sector, industry, marketCap
    y sin duplicados.
    """
    frames: List[pd.DataFrame] = []

    for exch in EXCHANGES:
        block = run_screener_for_exchange(exch)
        # run_screener_for_exchange ya filtra marketCap >= 10B
        # y devuelve list[dict] con symbol, companyName, sector, industry, marketCap, etc
        if block:
            frames.append(pd.DataFrame(block))

    if not frames:
        return pd.DataFrame(columns=[
            "symbol", "companyName", "sector", "industry", "marketCap"
        ])

    uni = pd.concat(frames, ignore_index=True)

    # normalizar companyName
    if "companyName" not in uni.columns and "name" in uni.columns:
        uni = uni.rename(columns={"name": "companyName"})

    # quitar duplicados por ticker
    if "symbol" in uni.columns:
        uni = uni.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    return uni


def _fmt_float(x: Optional[float], ndigits: int = 2) -> Optional[float]:
    """
    Intento sano de castear a float sin reventar con None / NaN.
    Devuelvo float redondeado o None.
    """
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, ndigits)
    except Exception:
        return None


def build_core_table() -> pd.DataFrame:
    """
    Pipeline cuantitativo principal:
      1. Saca universo large caps.
      2. Hace batch de AltmanZ/Piotroski (get_financial_scores_batch).
      3. Hace batch de Growth/CAGR (get_growth_batch).
      4. Junta todo.

    Devuelve DataFrame con UNA FILA POR TICKER y columnas:
      ticker
      name
      sector
      industry
      marketCap

      altmanZScore
      piotroskiScore

      revenueGrowth
      operatingCashFlowGrowth
      freeCashFlowGrowth
      debtGrowth

      rev_CAGR_5y
      ocf_CAGR_5y
      rev_CAGR_3y
      ocf_CAGR_3y
    """

    # ---------- 1. universo ----------
    uni = build_universe_largecaps().copy()

    # sanity cols
    for col in ["symbol", "companyName", "sector", "industry", "marketCap"]:
        if col not in uni.columns:
            uni[col] = None

    tickers = (
        uni["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not tickers:
        # nada que hacer
        return pd.DataFrame(columns=[
            "ticker","name","sector","industry","marketCap",
            "altmanZScore","piotroskiScore",
            "revenueGrowth","operatingCashFlowGrowth","freeCashFlowGrowth","debtGrowth",
            "rev_CAGR_5y","ocf_CAGR_5y","rev_CAGR_3y","ocf_CAGR_3y",
        ])

    # ---------- 2. scores batch (Altman / Piotroski) ----------
    scores_df = get_financial_scores_batch(tickers)
    # normalizamos nombres
    if "symbol" in scores_df.columns:
        scores_df = scores_df.rename(columns={"symbol": "ticker"})
    # Nos quedamos con columnas relevantes
    needed_scores = [
        "ticker",
        "altmanZScore",
        "piotroskiScore",
        "marketCap",   # viene también acá muchas veces
    ]
    for c in needed_scores:
        if c not in scores_df.columns:
            scores_df[c] = None
    scores_df = scores_df[needed_scores].copy()

    # ---------- 3. growth batch (crecimiento + CAGRs) ----------
    growth_df = get_growth_batch(tickers)
    if "symbol" in growth_df.columns:
        growth_df = growth_df.rename(columns={"symbol": "ticker"})

    needed_growth = [
        "ticker",
        "revenueGrowth",
        "operatingCashFlowGrowth",
        "freeCashFlowGrowth",
        "debtGrowth",
        "rev_CAGR_5y",
        "rev_CAGR_3y",
        "ocf_CAGR_5y",
        "ocf_CAGR_3y",
    ]
    for c in needed_growth:
        if c not in growth_df.columns:
            growth_df[c] = None
    growth_df = growth_df[needed_growth].copy()

    # ---------- 4. merge universo + scores + growth ----------
    # Empezamos del universo, porque ahí tenemos el nombre y sector
    core = uni.rename(columns={
        "symbol": "ticker",
        "companyName": "name",
    })[["ticker", "name", "sector", "industry", "marketCap"]].copy()

    # merge LEFT para ir agregando info
    core = core.merge(scores_df, on="ticker", how="left", suffixes=("", "_scores"))
    # si marketCap del screener era None, usamos marketCap del score batch
    core["marketCap"] = core["marketCap"].fillna(core["marketCap_scores"])
    if "marketCap_scores" in core.columns:
        core = core.drop(columns=["marketCap_scores"])

    core = core.merge(growth_df, on="ticker", how="left")

    # ---------- 5. sanity cast / limpiar números raros ----------
    num_cols_round2 = [
        "marketCap",
        "altmanZScore",
        "piotroskiScore",
        "revenueGrowth",
        "operatingCashFlowGrowth",
        "freeCashFlowGrowth",
        "debtGrowth",
        "rev_CAGR_5y",
        "rev_CAGR_3y",
        "ocf_CAGR_5y",
        "ocf_CAGR_3y",
    ]
    for c in num_cols_round2:
        core[c] = core[c].apply(lambda v: _fmt_float(v, 4))

    # ---------- 6. orden final de columnas ----------
    core = core[[
        "ticker",
        "name",
        "sector",
        "industry",
        "marketCap",
        "altmanZScore",
        "piotroskiScore",
        "revenueGrowth",
        "operatingCashFlowGrowth",
        "freeCashFlowGrowth",
        "debtGrowth",
        "rev_CAGR_5y",
        "ocf_CAGR_5y",
        "rev_CAGR_3y",
        "ocf_CAGR_3y",
    ]]

    # listo: core es TU dataframe limpio con métricas cuantitativas
    return core.reset_index(drop=True)


def get_company_row(df_core: pd.DataFrame, ticker: str) -> Dict[str, Any] | None:
    """
    Devuelve la fila (dict) de un ticker específico desde el dataframe core.
    Útil para detalle en Tab 2 sin recalcular nada.
    """
    if df_core is None or df_core.empty:
        return None
    row = df_core[df_core["ticker"] == ticker]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
