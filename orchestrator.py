from typing import List, Dict, Any
import pandas as pd

from config import EXCHANGES, MIN_MARKET_CAP, MAX_TICKERS_FOR_TEST
from fmp_api import get_companies_in_exchange
from data_quality import get_quality_signals_bulk
from data_growth import get_growth_signals_bulk
from fundamentals import build_fundamental_snapshot
from text_layer import fetch_text_signals_for_snapshot
from projection import estimate_forward_return
from filtering import premerge_candidates, final_filter_after_fundamentals

def build_universe() -> pd.DataFrame:
    rows = []
    for exch in EXCHANGES:
        try:
            data = get_companies_in_exchange(exch)
        except Exception:
            data = []
        for row in data:
            ticker = row.get("symbol")
            if not ticker:
                continue
            # filtro de market cap mínimo
            mcap = row.get("marketCap")
            if mcap is None:
                continue
            if mcap < MIN_MARKET_CAP:
                continue
            rows.append({
                "ticker": ticker,
                "companyName": row.get("companyName"),
                "sector": row.get("sector"),
                "industry": row.get("industry"),
                "marketCap": mcap,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
    # limitar cantidad total para debug
    df = df.sort_values("marketCap", ascending=False).head(MAX_TICKERS_FOR_TEST).reset_index(drop=True)
    return df

def pipeline_run() -> Dict[str, Any]:
    """
    Ejecuta TODO excepto la UI.
    Retorna:
      - df_list_view (para tabla screener)
      - snapshots_final (lista de dicts detallados)
    """

    # 1. Universo base
    universe_df = build_universe()
    tickers = universe_df["ticker"].tolist()

    # 2. Señales de calidad y crecimiento
    scores_df = get_quality_signals_bulk(tickers)
    growth_df = get_growth_signals_bulk(tickers)

    # 3. Merge + filtro inicial duro (Altman, Piotroski, EBIT↑, OCF↑)
    candidates_df = premerge_candidates(universe_df, scores_df, growth_df)

    # 4. Para cada candidato: snapshot fundamental detallado
    snapshots_raw: List[dict] = []
    for _, row in candidates_df.iterrows():
        tkr = row["ticker"]
        q_row = scores_df[scores_df["ticker"] == tkr].iloc[0] if not scores_df[scores_df["ticker"] == tkr].empty else None
        g_row = growth_df[growth_df["ticker"] == tkr].iloc[0] if not growth_df[growth_df["ticker"] == tkr].empty else None
        snap = build_fundamental_snapshot(tkr, q_row, g_row)
        snapshots_raw.append(snap)

    # 5. Filtro final por leverage_ok
    snapshots_after_debt = final_filter_after_fundamentals(snapshots_raw)

    # 6. Capa narrativa / texto
    snapshots_text: List[dict] = []
    for snap in snapshots_after_debt:
        text_bits = fetch_text_signals_for_snapshot(snap)
        snap.update(text_bits)
        snapshots_text.append(snap)

    # 7. Proyección / retorno esperado
    snapshots_final: List[dict] = []
    for snap in snapshots_text:
        proj = estimate_forward_return(snap)
        snap.update(proj)
        snapshots_final.append(snap)

    # 8. Crear df_list_view para la UI (screener tab)
    df_list_view = build_list_view_df(snapshots_final)

    return {
        "df_list_view": df_list_view,
        "snapshots_final": snapshots_final,
    }

def build_list_view_df(snapshots: List[dict]) -> pd.DataFrame:
    rows = []
    for s in snapshots:
        rows.append({
            "ticker": s.get("ticker"),
            "companyName": s.get("companyName"),
            "sector": s.get("sector"),
            "industry": s.get("industry"),
            "marketCap": s.get("marketCap"),
            "altmanZScore": s.get("altmanZScore"),
            "piotroskiScore": s.get("piotroskiScore"),
            "high_growth_flag": s.get("high_growth_flag"),
            "is_quality_compounder": s.get("is_quality_compounder"),
            "net_debt_to_ebitda_last": s.get("net_debt_to_ebitda_last"),
            "expected_return_cagr": s.get("expected_return_cagr"),
            "core_risk_note": s.get("core_risk_note"),
        })
    df = pd.DataFrame(rows)
    # Ordenar por expected_return_cagr descendente (mejores primero)
    if "expected_return_cagr" in df.columns:
        df = df.sort_values("expected_return_cagr", ascending=False)
    return df.reset_index(drop=True)

