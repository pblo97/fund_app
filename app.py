# app.py
#
# Streamlit UI para tu screener fundamental.
# Usa orchestrator.build_market_snapshot_df() para generar un DataFrame
# con métricas clave (Altman, Piotroski, crecimiento, CAGR).
#
# Flujo:
# - Sidebar: botón RUN → descarga todo y lo guarda en session_state
# - Tab 1: tabla shortlist
# - Tab 2: panel detalle del ticker seleccionado

import math
import pandas as pd
import streamlit as st

from config import MAX_NET_DEBT_TO_EBITDA
from orchestrator import build_market_snapshot_df

# -------------------------------------------------
# Helpers de formato visual
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


# -------------------------------------------------
# Render del panel detalle
# (por ahora solo texto numérico, sin gráficos históricos)
# -------------------------------------------------

def render_detail_panel(row: pd.Series):
    ticker = row.get("ticker", "—")
    name = row.get("name", ticker)

    st.markdown(f"## {name} ({ticker})")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Altman Z",
            "—" if pd.isna(row.get("altmanZScore")) else f"{row.get('altmanZScore'):.2f}"
        )
        st.metric(
            "Piotroski",
            "—" if pd.isna(row.get("piotroskiScore")) else f"{row.get('piotroskiScore'):.0f}/9"
        )

    with col2:
        st.metric("Revenue YoY", _fmt_pct(row.get("revenueGrowth")))
        st.metric("OCF YoY", _fmt_pct(row.get("operatingCashFlowGrowth")))

    with col3:
        st.metric("FCF YoY", _fmt_pct(row.get("freeCashFlowGrowth")))
        st.metric("Deuda YoY", _fmt_pct(row.get("debtGrowth")))

    st.write("---")

    st.subheader("Crecimiento compuesto (CAGR por acción)")
    colA, colB = st.columns(2)
    with colA:
        st.write(f"**Revenue CAGR 5y:** {_fmt_pct(row.get('rev_CAGR_5y'))}")
        st.write(f"**Revenue CAGR 3y:** {_fmt_pct(row.get('rev_CAGR_3y'))}")
    with colB:
        st.write(f"**OCF CAGR 5y:** {_fmt_pct(row.get('ocf_CAGR_5y'))}")
        st.write(f"**OCF CAGR 3y:** {_fmt_pct(row.get('ocf_CAGR_3y'))}")

    st.write("---")

    moat_flag = row.get("moat_flag", "—")
    high_growth = row.get("high_growth_flag", None)

    st.subheader("Lectura rápida")
    st.write(f"- **Sector / Industria:** {row.get('sector','—')} / {row.get('industry','—')}")
    st.write(f"- **MarketCap:** {_fmt_num(row.get('marketCap'))} USD aprox.")
    st.write(f"- **Moat (heurístico):** {moat_flag}")
    st.write(f"- **High growth flag:** {high_growth}")

    st.caption(
        "Interpretación inicial:\n"
        "- Altman Z y Piotroski altos ⇒ balance sano + eficiencia operativa.\n"
        "- Revenue/OCF creciendo y deuda controlada ⇒ caja que escala sin apalancarse.\n"
        "- CAGR ≥15% en Revenue/OCF ⇒ potencia compuesta.\n"
        "- 'moat_candidate' ⇒ señal de que combina disciplina financiera + crecimiento fuerte."
    )


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
    "Pipeline:\n"
    "1. Screener large caps (NYSE / NASDAQ / AMEX ≥10B).\n"
    "2. Filtrado rápido por salud financiera y crecimiento.\n"
    "3. Panel de detalle con métricas clave."
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("Parámetros")

# Esto queda acá aunque todavía no calculamos netDebt/EBITDA real,
# porque más adelante vamos a incorporar esa métrica en orchestrator.
max_leverage = st.sidebar.slider(
    "Máximo Net Debt / EBITDA permitido (placeholder)",
    min_value=0.0,
    max_value=6.0,
    value=MAX_NET_DEBT_TO_EBITDA,
    step=0.5
)

run_btn = st.sidebar.button("🚀 Run Screening / Refresh Data")

# -------------------------------------------------
# SESSION STATE init
# -------------------------------------------------

if "snapshot_df" not in st.session_state:
    st.session_state["snapshot_df"] = pd.DataFrame()

if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

# -------------------------------------------------
# BOTÓN: ejecutar screening y bajar métricas
# -------------------------------------------------

if run_btn:
    try:
        df_snapshot = build_market_snapshot_df()
        # guardamos en session_state
        st.session_state["snapshot_df"] = df_snapshot
        st.session_state["last_error"] = None
    except Exception as e:
        st.session_state["snapshot_df"] = pd.DataFrame()
        st.session_state["last_error"] = str(e)

df_snapshot = st.session_state["snapshot_df"]

# feedback en sidebar
if st.session_state["last_error"]:
    st.sidebar.error(f"Error al armar shortlist: {st.session_state['last_error']}")
else:
    st.sidebar.success(
        f"{len(df_snapshot)} tickers cargados" if not df_snapshot.empty else "Sin datos aún"
    )

# -------------------------------------------------
# TABS
# -------------------------------------------------

tab1, tab2 = st.tabs([
    "1. Shortlist final",
    "2. Detalle Ticker"
])

# ---- TAB 1 ----
with tab1:
    st.subheader("1. Shortlist fundamental")

    if df_snapshot.empty:
        st.info("Presiona 'Run Screening / Refresh Data' en el sidebar.")
    else:
        # Creamos columnas formateadas amigables para ver
        df_view = df_snapshot.copy()

        # market cap bonito
        df_view["marketCap_fmt"] = df_view["marketCap"].apply(_fmt_num)

        # formateos porcentuales
        df_view["revenueGrowth_pct"] = df_view["revenueGrowth"].apply(_fmt_pct)
        df_view["ocfGrowth_pct"] = df_view["operatingCashFlowGrowth"].apply(_fmt_pct)
        df_view["fcfGrowth_pct"] = df_view["freeCashFlowGrowth"].apply(_fmt_pct)
        df_view["debtGrowth_pct"] = df_view["debtGrowth"].apply(_fmt_pct)

        df_view["rev_CAGR_5y_pct"] = df_view["rev_CAGR_5y"].apply(_fmt_pct)
        df_view["ocf_CAGR_5y_pct"] = df_view["ocf_CAGR_5y"].apply(_fmt_pct)

        df_view["altmanZScore_fmt"] = df_view["altmanZScore"].apply(
            lambda x: (
                "—"
                if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
                else f"{float(x):.2f}"
            )
        )
        df_view["piotroskiScore_fmt"] = df_view["piotroskiScore"].apply(
            lambda x: (
                "—"
                if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
                else f"{float(x):.0f}/9"
            )
        )

        # columnas que vamos a mostrar en la tabla
        cols_basic = [
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
            "high_growth_flag",
            "moat_flag",
        ]

        # filtra por leverage si ya lo tuviéramos en el DF más adelante:
        # si no existe, simplemente usamos df_view tal cual
        if "netDebt_to_EBITDA" in df_view.columns:
            def _lev_ok(x):
                if x is None:
                    return True
                if isinstance(x, float) and math.isnan(x):
                    return True
                try:
                    return float(x) <= max_leverage
                except Exception:
                    return True
            df_view = df_view[df_view["netDebt_to_EBITDA"].apply(_lev_ok)].copy()

        cols_basic = [c for c in cols_basic if c in df_view.columns]

        if df_view.empty:
            st.warning("No quedaron candidatas después del filtro de leverage.")
        else:
            st.dataframe(
                df_view[cols_basic].reset_index(drop=True),
                use_container_width=True,
                height=500,
            )

        st.caption(
            "Lectura rápida:\n"
            "- Altman Z ↑ y Piotroski ↑ = balance sólido + disciplina operativa.\n"
            "- Crecimiento positivo en revenue/OCF/FCF y deuda que NO sube.\n"
            "- CAGR ≥15% en revenue/OCF sugiere capacidad de componer valor.\n"
            "- 'high_growth_flag' True = pasó el umbral de crecimiento compuesto.\n"
            "- 'moat_candidate' = crecimiento fuerte + buena disciplina financiera."
        )

# ---- TAB 2 ----
with tab2:
    st.subheader("2. Ficha detallada del ticker")

    if df_snapshot.empty:
        st.info("Primero genera datos en '1. Shortlist final'.")
    else:
        tickers_available = (
            df_snapshot["ticker"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if "ticker" in df_snapshot.columns
            else []
        )

        if not tickers_available:
            st.warning("No hay tickers disponibles.")
        else:
            picked = st.selectbox(
                "Elige un ticker para ver su ficha:",
                tickers_available
            )

            # buscamos la fila de ese ticker
            row_detail = df_snapshot[df_snapshot["ticker"] == picked]
            if row_detail.empty:
                st.error("No encontré datos para ese ticker.")
            else:
                render_detail_panel(row_detail.iloc[0])
