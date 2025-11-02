import streamlit as st
import pandas as pd
import math

from orchestrator import (
    build_core_table,
    get_company_row,
)


# ---------- helpers de formato visual ----------
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

def _fmt_num_short(x):
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    # formateo market cap
    if abs(v) >= 1_000_000_000:
        return f"{v/1_000_000_000:.1f}B"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    return f"{v:.0f}"


def decorate_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia con columnas bonitas para mostrar en Tab1."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    # columnas bonitas
    df["marketCap_fmt"] = df["marketCap"].apply(_fmt_num_short)
    df["altmanZScore_fmt"] = df["altmanZScore"].apply(
        lambda v: "—" if v is None else f"{v:.2f}"
    )
    df["piotroskiScore_fmt"] = df["piotroskiScore"].apply(
        lambda v: "—" if v is None else f"{v:.0f}/9"
    )

    df["revenueGrowth_pct"] = df["revenueGrowth"].apply(_fmt_pct)
    df["ocfGrowth_pct"] = df["operatingCashFlowGrowth"].apply(_fmt_pct)
    df["fcfGrowth_pct"] = df["freeCashFlowGrowth"].apply(_fmt_pct)
    df["debtGrowth_pct"] = df["debtGrowth"].apply(_fmt_pct)

    df["rev_CAGR_5y_pct"] = df["rev_CAGR_5y"].apply(_fmt_pct)
    df["ocf_CAGR_5y_pct"] = df["ocf_CAGR_5y"].apply(_fmt_pct)

    df["rev_CAGR_3y_pct"] = df["rev_CAGR_3y"].apply(_fmt_pct)
    df["ocf_CAGR_3y_pct"] = df["ocf_CAGR_3y"].apply(_fmt_pct)

    show_cols = [
        "ticker",
        "name",
        "sector",
        "industry",
        "marketCap_fmt",
        "altmanZScore_fmt",
        "piotroskiScore_fmt",
        "revenueGrowth_pct",
        "ocfGrowth_pct",
        "fcfGrowth_pct",
        "debtGrowth_pct",
        "rev_CAGR_5y_pct",
        "ocf_CAGR_5y_pct",
        "rev_CAGR_3y_pct",
        "ocf_CAGR_3y_pct",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    return df[show_cols].reset_index(drop=True)


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="FUND Screener Core", page_icon="💸", layout="wide")
st.title("💸 FUND Screener (core cuantitativo)")

st.write(
    "Este panel arma un universo de large caps sólidas usando:\n"
    "- Altman Z / Piotroski (calidad financiera / solvencia)\n"
    "- Crecimiento reciente en Revenue / OCF / FCF\n"
    "- Disciplina de deuda (debtGrowth <= 0)\n"
    "- CAGR 3y/5y en ingresos y OCF por acción\n\n"
    "Luego puedes elegir un ticker y ver sus números crudos."
)

if "core_df" not in st.session_state:
    st.session_state["core_df"] = pd.DataFrame()

if "last_err" not in st.session_state:
    st.session_state["last_err"] = None

run_btn = st.sidebar.button("🚀 Run / Refresh datos del mercado")

if run_btn:
    try:
        st.session_state["core_df"] = build_core_table()
        st.session_state["last_err"] = None
    except Exception as e:
        st.session_state["last_err"] = str(e)
        st.session_state["core_df"] = pd.DataFrame()

core_df = st.session_state["core_df"]

if st.session_state["last_err"]:
    st.sidebar.error(st.session_state["last_err"])
else:
    st.sidebar.success(
        f"{len(core_df)} empresas cargadas" if not core_df.empty else "Sin datos aún"
    )


tab1, tab2 = st.tabs(["1. Shortlist cuantitativa", "2. Detalle Ticker"])

with tab1:
    st.subheader("Shortlist cuantitativa filtrable mentalmente")
    if core_df.empty:
        st.info("Aprieta el botón 🚀 en el sidebar para traer datos.")
    else:
        table = decorate_df(core_df)
        st.dataframe(
            table,
            use_container_width=True,
            height=600
        )
        st.caption(
            "- Altman Z alto y Piotroski alto = balance sano, baja prob. quiebra, contabilidad limpia.\n"
            "- Crecimiento positivo y deuda no creciendo = escala sin quemar el balance.\n"
            "- CAGR≥15% en rev/OCF por acción = potencial compuesto serio."
        )

with tab2:
    st.subheader("Detalle de la empresa (números crudos)")
    if core_df.empty:
        st.info("Primero genera la tabla en el tab 1.")
    else:
        tickers = core_df["ticker"].dropna().tolist()
        picked = st.selectbox("Elige ticker", tickers)
        row = get_company_row(core_df, picked)

        if row is None:
            st.error("No encontré ese ticker en la tabla base.")
        else:
            colA, colB, colC = st.columns(3)

            with colA:
                st.metric("Altman Z", f"{row.get('altmanZScore','—')}")
                st.metric("Piotroski", f"{row.get('piotroskiScore','—')}")

            with colB:
                st.metric("Revenue Growth (últ FY)",
                          _fmt_pct(row.get("revenueGrowth")))
                st.metric("OCF Growth (últ FY)",
                          _fmt_pct(row.get("operatingCashFlowGrowth")))

            with colC:
                st.metric("FCF Growth (últ FY)",
                          _fmt_pct(row.get("freeCashFlowGrowth")))
                st.metric("Deuda vs año prev.",
                          _fmt_pct(row.get("debtGrowth")))

            st.markdown("---")
            st.write("**CAGR por acción (compuesto)**")
            st.write(f"Revenue CAGR 5y: {_fmt_pct(row.get('rev_CAGR_5y'))}")
            st.write(f"Revenue CAGR 3y: {_fmt_pct(row.get('rev_CAGR_3y'))}")
            st.write(f"OCF CAGR 5y: {_fmt_pct(row.get('ocf_CAGR_5y'))}")
            st.write(f"OCF CAGR 3y: {_fmt_pct(row.get('ocf_CAGR_3y'))}")

            st.markdown("---")
            st.write("**Info básica**")
            st.write(f"Ticker: {row.get('ticker','—')}")
            st.write(f"Nombre: {row.get('name','—')}")
            st.write(f"Sector / Industria: {row.get('sector','—')} / {row.get('industry','—')}")
            st.write(f"Market Cap: {_fmt_num_short(row.get('marketCap'))}")
