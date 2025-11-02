import streamlit as st
import pandas as pd
import math
import json
import matplotlib.pyplot as plt

from config import MAX_NET_DEBT_TO_EBITDA
from orchestrator import (
    build_full_snapshot,
    build_market_snapshot_lite,  # <- usamos la versión lite
    enrich_company_snapshot,
)

# -------------------------------------------------
# Helpers de formato
# -------------------------------------------------

def _fmt_pct(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "—"
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "—"

def _fmt_num(x):
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—"
    try:
        val = float(x)
    except Exception:
        return "—"
    if abs(val) >= 1_000_000_000:
        return f"{val/1_000_000_000:.1f}B"
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    return f"{val:.0f}"

def _ensure_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    if isinstance(v, list):
        return v
    return []

def dataframe_from_rows(rows: list[dict], max_leverage: float):
    df = pd.DataFrame(rows).copy()

    needed = [
        "ticker","name","companyName","sector","industry","marketCap",
        "altmanZScore","piotroskiScore",
        "revenueGrowth","operatingCashFlowGrowth","freeCashFlowGrowth","debtGrowth",
        "rev_CAGR_5y","ocf_CAGR_5y",
        "rev_CAGR_3y","ocf_CAGR_3y",
        "netDebt_to_EBITDA","net_debt_to_ebitda_last",
        "fcf_per_share_slope_5y","buyback_pct_5y","is_quality_compounder","moat_flag",
        "years","fcf_per_share_hist","shares_hist","net_debt_hist",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = None

    if "name" in df.columns and "companyName" in df.columns:
        df["name"] = df["name"].fillna(df["companyName"])

    def _pick_nde(row):
        cand1 = row.get("net_debt_to_ebitda_last")
        cand2 = row.get("netDebt_to_EBITDA")
        if cand1 is not None and not (isinstance(cand1, float) and math.isnan(cand1)):
            return cand1
        return cand2
    df["netDebt_to_EBITDA_effective"] = df.apply(_pick_nde, axis=1)

    def _lev_ok(x):
        if x is None:
            return True
        if isinstance(x, float) and math.isnan(x):
            return True
        try:
            return float(x) <= max_leverage
        except Exception:
            return True
    df["leverage_ok"] = df["netDebt_to_EBITDA_effective"].apply(_lev_ok)

    df["netDebt_to_EBITDA_fmt"] = df["netDebt_to_EBITDA_effective"].apply(
        lambda x: ("—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.2f}")
    )
    df["marketCap_fmt"] = df["marketCap"].apply(_fmt_num)

    df["altmanZScore_fmt"] = df["altmanZScore"].apply(
        lambda x: ("—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.2f}")
    )
    df["piotroskiScore_fmt"] = df["piotroskiScore"].apply(
        lambda x: ("—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.0f}/9")
    )

    df["revenueGrowth_pct"] = df["revenueGrowth"].apply(_fmt_pct)
    df["ocfGrowth_pct"] = df["operatingCashFlowGrowth"].apply(_fmt_pct)
    df["fcfGrowth_pct"] = df["freeCashFlowGrowth"].apply(_fmt_pct)
    df["debtGrowth_pct"] = df["debtGrowth"].apply(_fmt_pct)

    df["rev_CAGR_5y_pct"] = df["rev_CAGR_5y"].apply(_fmt_pct)
    df["ocf_CAGR_5y_pct"] = df["ocf_CAGR_5y"].apply(_fmt_pct)

    if "rev_CAGR_3y" in df.columns:
        df["rev_CAGR_3y_pct"] = df["rev_CAGR_3y"].apply(_fmt_pct)
    else:
        df["rev_CAGR_3y_pct"] = "—"
    if "ocf_CAGR_3y" in df.columns:
        df["ocf_CAGR_3y_pct"] = df["ocf_CAGR_3y"].apply(_fmt_pct)
    else:
        df["ocf_CAGR_3y_pct"] = "—"

    df["buyback_pct_5y_fmt"] = df["buyback_pct_5y"].apply(_fmt_pct)

    def _fmt_slope(val):
        if val is None:
            return "—"
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return "—"
        try:
            return f"{float(val):.2f} /yr"
        except Exception:
            return "—"
    df["fcf_per_share_slope_5y_fmt"] = df["fcf_per_share_slope_5y"].apply(_fmt_slope)

    def _fmt_compounder(flag):
        return "✅ COMPOUNDER" if flag is True else "—"
    df["is_quality_compounder_fmt"] = df["is_quality_compounder"].apply(_fmt_compounder)

    return df


def render_detail_panel(row: pd.Series):
    ticker = row.get("ticker", "—")
    display_name = row.get("name", row.get("companyName", ticker))

    st.markdown(f"## {display_name} ({ticker})")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Altman Z", f"{row.get('altmanZScore', '—')}")
        st.metric("Piotroski", f"{row.get('piotroskiScore', '—')}")
    with col2:
        st.metric("Crec. OCF (últ. FY)", _fmt_pct(row.get("operatingCashFlowGrowth")))
        st.metric("Crec. FCF (últ. FY)", _fmt_pct(row.get("freeCashFlowGrowth")))
    with col3:
        nde = (
            row.get("netDebt_to_EBITDA")
            or row.get("netDebt_to_EBITDA_effective")
        )
        nde_fmt = (
            "—"
            if nde is None or (isinstance(nde, float) and math.isnan(nde))
            else f"{float(nde):.2f}"
        )
        st.metric("Net Debt / EBITDA", nde_fmt)
        st.metric("Deuda vs año prev.", _fmt_pct(row.get("debtGrowth")))

    st.subheader("Snapshot estratégico")
    st.write(f"**Sector / Industria:** {row.get('sector','—')} / {row.get('industry','—')}")
    st.write(f"**Moat (heurístico):** {row.get('moat_flag','—')}")
    st.write(f"**Insiders:** {row.get('insider_signal','(sin dato)')}")
    st.write(
        f"**Sentimiento noticias:** {row.get('sentiment_flag','neutral')} — "
        f"{row.get('sentiment_reason','tono mixto')}"
    )
    st.write(f"**Resumen negocio:** {row.get('business_summary','—')}")
    st.write(f"**Por qué importa:** {row.get('why_it_matters','—')}")
    st.write(f"**Riesgo visible:** {row.get('core_risk_note','—')}")
    st.write(f"**Última earnings call:** {row.get('transcript_summary','—')}")

    st.subheader("Históricos clave (tendencias 3-5y)")

    years = _ensure_list(row.get("years", []))
    fcfps_hist = _ensure_list(row.get("fcf_per_share_hist", []))
    shares_hist = _ensure_list(row.get("shares_hist", []))
    net_debt_hist = _ensure_list(row.get("net_debt_hist", []))

    colA, colB, colC = st.columns(3)

    with colA:
        st.caption("FCF por acción")
        fig, ax = plt.subplots()
        ax.plot(years, fcfps_hist, marker="o")
        ax.set_xlabel("Año")
        ax.set_ylabel("FCF/acción")
        st.pyplot(fig)

    with colB:
        st.caption("Acciones en circulación (dilución vs recompras)")
        fig2, ax2 = plt.subplots()
        ax2.plot(years, shares_hist, marker="o")
        ax2.set_xlabel("Año")
        ax2.set_ylabel("Acciones (unidades)")
        st.pyplot(fig2)

    with colC:
        st.caption("Deuda Neta")
        fig3, ax3 = plt.subplots()
        ax3.plot(years, net_debt_hist, marker="o")
        ax3.set_xlabel("Año")
        ax3.set_ylabel("Deuda Neta (USD)")
        st.pyplot(fig3)


# -------------------------------------------------
# CONFIG STREAMLIT
# -------------------------------------------------

st.set_page_config(
    page_title="FUND Screener",
    page_icon="💸",
    layout="wide"
)

st.title("💸 FUND Screener")
st.write(
    "1) Filtramos el mercado a solo large caps sanas y creciendo.\n"
    "2) Vemos la lista final.\n"
    "3) Abrimos una empresa y miramos insiders / news / transcript."
)

st.sidebar.header("Parámetros")

max_leverage = st.sidebar.slider(
    "Máximo Net Debt / EBITDA permitido",
    min_value=0.0,
    max_value=6.0,
    value=MAX_NET_DEBT_TO_EBITDA,
    step=0.5
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Screening / Refresh Data (lite)")


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "snapshot_rows" not in st.session_state:
    st.session_state["snapshot_rows"] = []

if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

if "kept" not in st.session_state:
    st.session_state["kept"] = pd.DataFrame(columns=["symbol"])


# -------------------------------------------------
# WATCHLIST ENRIQUECIDA (arriba de todo)
# -------------------------------------------------
# Para evitar colgarnos apenas carga la app, NO llamamos
# build_full_snapshot automáticamente. Solo mostramos si ya está en cache.

kept_syms = (
    st.session_state["kept"]["symbol"]
      .dropna()
      .astype(str)
      .unique()
      .tolist()
    if ("kept" in st.session_state and not st.session_state["kept"].empty)
    else []
)

if kept_syms and "watchlist_cache" not in st.session_state:
    try:
        st.session_state["watchlist_cache"] = build_full_snapshot(kept_syms)
    except Exception as e:
        st.session_state["watchlist_cache"] = pd.DataFrame()
        st.warning(f"No pude enriquecer tu watchlist: {e}")

final_df_watchlist = st.session_state.get("watchlist_cache", pd.DataFrame())

if not final_df_watchlist.empty:
    st.subheader("Watchlist enriquecida (tus tickers marcados)")
    cols_watch = [
        "ticker",
        "name",
        "sector",
        "industry",
        "marketCap",
        "fcf_per_share_slope_5y",
        "buyback_pct_5y",
        "netDebt_to_EBITDA",
        "is_quality_compounder"
    ]
    cols_watch = [c for c in cols_watch if c in final_df_watchlist.columns]

    st.dataframe(
        final_df_watchlist[cols_watch].reset_index(drop=True),
        use_container_width=True,
        height=300
    )
else:
    st.info("Aún no hay watchlist enriquecida en caché.")


# -------------------------------------------------
# BOTÓN: construir shortlist global de mercado (LITE)
# -------------------------------------------------

if run_btn:
    try:
        rows = build_market_snapshot_lite(limit=15)
        st.session_state["snapshot_rows"] = rows
        st.session_state["last_error"] = None
    except Exception as e:
        st.session_state["last_error"] = str(e)
        st.session_state["snapshot_rows"] = []

rows_data = st.session_state["snapshot_rows"]

if st.session_state["last_error"]:
    st.sidebar.error(f"Error al armar shortlist: {st.session_state['last_error']}")
else:
    st.sidebar.success(
        f"{len(rows_data)} tickers cargados" if rows_data else "Sin datos aún"
    )


# -------------------------------------------------
# TABS
# -------------------------------------------------

tab1, tab2 = st.tabs([
    "1. Shortlist final",
    "2. Detalle Ticker"
])

with tab1:
    st.subheader("1. Shortlist: large caps sólidas + crecimiento compuesto ≥15%")

    if not rows_data:
        st.info("Presiona 'Run Screening / Refresh Data (lite)' en el sidebar.")
    else:
        df_all = dataframe_from_rows(rows_data, max_leverage)

        df_screen = df_all[df_all["leverage_ok"]].copy()

        cols_basic = [
            "ticker",
            "name",
            "sector",
            "industry",
            "marketCap_fmt",
            "netDebt_to_EBITDA_fmt",
            "buyback_pct_5y_fmt",
            "fcf_per_share_slope_5y_fmt",
            "is_quality_compounder_fmt",
            "altmanZScore_fmt",
            "piotroskiScore_fmt",
            "revenueGrowth_pct",
            "ocfGrowth_pct",
            "fcfGrowth_pct",
            "debtGrowth_pct",
            "rev_CAGR_5y_pct",
            "ocf_CAGR_5y_pct",
            "moat_flag",
        ]
        cols_basic = [c for c in cols_basic if c in df_screen.columns]

        if df_screen.empty:
            st.warning("Con tu límite de apalancamiento no queda ninguna candidata.")
        else:
            st.dataframe(
                df_screen[cols_basic].reset_index(drop=True),
                use_container_width=True,
                height=500
            )

with tab2:
    st.subheader("2. Ficha detallada de la empresa")

    if not rows_data:
        st.info("Primero genera datos en '1. Shortlist final'.")
    else:
        df_all = dataframe_from_rows(rows_data, max_leverage)
        df_valid = df_all[df_all["leverage_ok"]].copy()
        tickers_available = df_valid["ticker"].dropna().tolist()

        if not tickers_available:
            st.warning("No quedan tickers válidos tras tu límite de apalancamiento.")
        else:
            picked = st.selectbox(
                "Elige un ticker para ver su ficha cualitativa y de tendencias:",
                tickers_available
            )

            base_core = next((r for r in rows_data if r.get("ticker") == picked), None)
            if base_core is None:
                st.error("No encontré datos base de ese ticker.")
            else:
                try:
                    detailed = enrich_company_snapshot(base_core.copy())
                except Exception as e:
                    st.error(f"No pude completar el detalle cualitativo: {e}")
                    detailed = base_core.copy()

                render_detail_panel(pd.Series(detailed))