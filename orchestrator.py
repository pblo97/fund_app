# orchestrator.py
#
# Flujo principal:
#   - Corre screener (large caps)
#   - Descarga métricas en lote (scores y growth)
#   - Baja perfil por ticker
#   - Construye una tabla final mergeada lista para Streamlit
#
# Nota: Esta versión NO mete insiders, transcript ni históricos largos.
#       Eso lo hacemos después para Tab2.
#
# Dependencias:
#   config.py
#   fmp_api.py
#   metrics.py

from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
import math

from config import SCREENER_PARAMS
from fmp_api import (
    run_screener_for_exchange,
    get_financial_scores_batch,
    get_growth_batch,
    get_profile,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_ratios,
)
from metrics import compute_core_financial_metrics


# ============================================================
# 1. Universo base (large caps activas en NYSE, NASDAQ, AMEX)
# ============================================================

EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]


def build_universe() -> pd.DataFrame:
    """
    Llama el screener por cada exchange y retorna un DataFrame con columnas:
      symbol, companyName, sector, industry, marketCap
    Sin duplicados y sólo large caps (>=10B USD) porque nuestro fmp_api ya filtra.
    """
    frames = []
    for exch in EXCHANGES:
        data = run_screener_for_exchange(exch, min_mktcap=1e10)
        if not data:
            continue
        frames.append(pd.DataFrame(data))

    if not frames:
        # vació duro, devolvemos DF con columnas mínimas esperadas
        return pd.DataFrame(columns=["symbol", "companyName", "sector", "industry", "marketCap"])

    uni = pd.concat(frames, ignore_index=True)

    # Normalizamos nombre de la empresa
    if "companyName" not in uni.columns and "name" in uni.columns:
        uni = uni.rename(columns={"name": "companyName"})

    # Borramos duplicados por symbol
    uni = uni.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    # Nos quedamos con columnas que vamos a usar en la UI
    keep_cols = ["symbol", "companyName", "sector", "industry", "marketCap"]
    for c in keep_cols:
        if c not in uni.columns:
            uni[c] = None
    uni = uni[keep_cols]

    return uni


# ============================================================
# 2. Descarga en lote (Altman, Piotroski, Growth/CAGR)
# ============================================================

def fetch_bulk_scores_and_growth(universe_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A partir del universo, arma:
      scores_df: Altman Z, Piotroski, etc. indexado por symbol
      growth_df: revenueGrowth, debtGrowth, CAGRs, high_growth_flag... indexado por symbol
    """
    symbols: List[str] = (
        universe_df["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Llamadas batch a la API
    scores_df = get_financial_scores_batch(symbols)
    growth_df = get_growth_batch(symbols)

    # Normalizamos índices para lookup rápido
    if not scores_df.empty:
        scores_df = scores_df.copy()
        scores_df["symbol"] = scores_df["symbol"].astype(str)
        scores_df = scores_df.set_index("symbol")
    else:
        scores_df = pd.DataFrame(columns=[
            "altmanZScore", "piotroskiScore"
        ]).set_index(pd.Index([], name="symbol"))

    if not growth_df.empty:
        growth_df = growth_df.copy()
        growth_df["symbol"] = growth_df["symbol"].astype(str)
        growth_df = growth_df.set_index("symbol")
    else:
        growth_df = pd.DataFrame(columns=[
            "revenueGrowth",
            "operatingCashFlowGrowth",
            "freeCashFlowGrowth",
            "debtGrowth",
            "rev_CAGR_5y",
            "rev_CAGR_3y",
            "ocf_CAGR_5y",
            "ocf_CAGR_3y",
            "high_growth_flag",
        ]).set_index(pd.Index([], name="symbol"))

    return scores_df, growth_df


# ============================================================
# 3. Construcción fila por ticker
# ============================================================

def _extract_float(df: pd.DataFrame, sym: str, col: str):
    """
    intenta leer df.loc[sym, col] como float de modo seguro.
    si falla o no existe, devuelve None.
    """
    try:
        v = df.loc[sym, col]
    except Exception:
        return None

    # si df.loc[...] devolvió una serie (duplicados), quédate con el primero
    if isinstance(v, pd.Series):
        v = v.iloc[0]

    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def build_company_row(sym: str,
                      scores_df: pd.DataFrame,
                      growth_df: pd.DataFrame) -> Dict[str, Any] | None:
    """
    Baja el perfil/base statements de un ticker
    y construye UNA FILA lista para el DataFrame final,
    inyectando scores (Altman, Piotroski) y growth (CAGRs, etc.).
    """

    # --- bajar data cruda necesaria para compute_core_financial_metrics ---
    profile = get_profile(sym)
    income_hist = get_income_statement(sym, limit=5)
    balance_hist = get_balance_sheet(sym, limit=5)
    cash_hist = get_cash_flow(sym, limit=5)
    ratios_hist = get_ratios(sym, limit=5)

    # sanity: si no hay statements básicos, saltamos
    if (
        not isinstance(income_hist, list) or len(income_hist) == 0 or
        not isinstance(balance_hist, list) or len(balance_hist) == 0 or
        not isinstance(cash_hist, list) == 0
    ):
        # ojo: la línea de arriba tiene un bug común si no somos cuidadosos.
        # corrijamos:
        pass

    if (
        not isinstance(income_hist, list) or len(income_hist) == 0 or
        not isinstance(balance_hist, list) or len(balance_hist) == 0 or
        not isinstance(cash_hist, list) or len(cash_hist) == 0
    ):
        return None

    base_row = compute_core_financial_metrics(
        sym,
        profile,
        ratios_hist,
        income_hist,
        balance_hist,
        cash_hist,
    )

    # --- inyectar Altman / Piotroski ---
    base_row["altmanZScore"] = _extract_float(scores_df, sym, "altmanZScore")
    base_row["piotroskiScore"] = _extract_float(scores_df, sym, "piotroskiScore")

    # --- inyectar growth / disciplina / CAGRs ---
    base_row["revenueGrowth"] = _extract_float(growth_df, sym, "revenueGrowth")
    base_row["operatingCashFlowGrowth"] = _extract_float(growth_df, sym, "operatingCashFlowGrowth")
    base_row["freeCashFlowGrowth"] = _extract_float(growth_df, sym, "freeCashFlowGrowth")
    base_row["debtGrowth"] = _extract_float(growth_df, sym, "debtGrowth")

    base_row["rev_CAGR_5y"] = _extract_float(growth_df, sym, "rev_CAGR_5y")
    base_row["rev_CAGR_3y"] = _extract_float(growth_df, sym, "rev_CAGR_3y")
    base_row["ocf_CAGR_5y"] = _extract_float(growth_df, sym, "ocf_CAGR_5y")
    base_row["ocf_CAGR_3y"] = _extract_float(growth_df, sym, "ocf_CAGR_3y")

    # El flag high_growth_flag viene de growth_df["high_growth_flag"]:
    try:
        hg = growth_df.loc[sym, "high_growth_flag"]
        if isinstance(hg, pd.Series):
            hg = hg.iloc[0]
        base_row["high_growth_flag"] = bool(hg)
    except Exception:
        base_row["high_growth_flag"] = None

    # --- moat_flag heurístico inicial ---
    # si cumple high_growth_flag Y Piotroski >=7:
    if base_row["high_growth_flag"] and base_row["piotroskiScore"] is not None:
        if base_row["piotroskiScore"] >= 7:
            base_row["moat_flag"] = "moat_candidate"
        else:
            base_row["moat_flag"] = "—"
    else:
        base_row["moat_flag"] = "—"

    return base_row


# ============================================================
# 4. Build snapshot final
# ============================================================

def build_market_snapshot_df() -> pd.DataFrame:
    """
    Función de más alto nivel:
      - arma universo
      - baja métricas en lote
      - construye las filas completas
      - devuelve un DataFrame listo para mostrar en Streamlit
    """
    uni = build_universe()
    if uni.empty:
        return pd.DataFrame()

    scores_df, growth_df = fetch_bulk_scores_and_growth(uni)

    rows: List[Dict[str, Any]] = []

    for _, r in uni.iterrows():
        sym = str(r["symbol"]).strip()
        if not sym:
            continue

        row_dict = build_company_row(sym, scores_df, growth_df)
        if row_dict is None:
            continue

        # metemos también info directa del screener por si faltó en profile
        if row_dict.get("name") in (None, "", sym):
            # intenta rellenar con companyName del screener
            row_dict["name"] = r.get("companyName") or row_dict.get("name")

        if row_dict.get("sector") is None:
            row_dict["sector"] = r.get("sector")
        if row_dict.get("industry") is None:
            row_dict["industry"] = r.get("industry")
        if row_dict.get("marketCap") is None:
            row_dict["marketCap"] = r.get("marketCap")

        rows.append(row_dict)

    if not rows:
        return pd.DataFrame()

    df_final = pd.DataFrame(rows)

    # aseguramos columnas clave que la UI espera
    expected_cols = [
        "ticker", "name", "sector", "industry", "marketCap",
        "altmanZScore", "piotroskiScore",
        "revenueGrowth", "operatingCashFlowGrowth", "freeCashFlowGrowth", "debtGrowth",
        "rev_CAGR_5y", "rev_CAGR_3y", "ocf_CAGR_5y", "ocf_CAGR_3y",
        "high_growth_flag",
        "moat_flag",
    ]
    for c in expected_cols:
        if c not in df_final.columns:
            df_final[c] = None

    # orden amigable
    df_final = df_final[expected_cols]

    return df_final
