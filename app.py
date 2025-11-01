import math
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- imports de tu capa de datos / lógica ---
# build_full_snapshot: recibe kept_symbols -> DataFrame enriquecido con métricas históricas
# build_market_snapshot: no recibe args -> shortlist tipo lista[dict] con Altman, Piotroski, growth, etc.
# enrich_company_snapshot: dado un dict base de un ticker, lo enriquece con insiders / sentimiento / transcript (puede ser stub)
from orchestrator import (
    build_full_snapshot,
    build_market_snapshot,
    enrich_company_snapshot,
)

# --- config del umbral de leverage ---
try:
    from config import MAX_NET_DEBT_TO_EBITDA
except Exception:
    # fallback si no tienes config.py aún
    MAX_NET_DEBT_TO_EBITDA = 2.0


# ======================================================================================
# Helpers de formato numérico / porcentaje / parsing
# ======================================================================================

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


# ======================================================================================
# Transformar la shortlist (list[dict]) → DataFrame listo para UI
# ======================================================================================

def dataframe_from_rows(rows: list[dict], max_leverage: float):
    """
    rows viene de build_market_snapshot() y es una lista de dicts tipo:
    {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 2500000000000,
        "altmanZScore": 8.5,
        "piotroskiScore": 7,
        "revenueGrowth": 0.06,
        "operatingCashFlowGrowth": 0.08,
        "freeCashFlowGrowth": 0.07,
        "debtGrowth": -0.02,
        "rev_CAGR_5y": 0.10,
        "rev_CAGR_3y": 0.12,
        "ocf_CAGR_5y": 0.11,
        "ocf_CAGR_3y": 0.13,
        "netDebt_to_EBITDA": 1.2,
        "moat_flag": "ecosistema cerrado / marca",
        ...
    }
    """
    df = pd.DataFrame(rows).copy()

    needed = [
        "netDebt_to_EBITDA",
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
        "sector",
        "industry",
        "moat_flag",
        "ticker",
        "name",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = None

    def _lev_ok(x):
        if x is None:
            return True
        if isinstance(x, float) and math.isnan(x):
            return True
        try:
            return float(x) <= max_leverage
        except Exception:
            return True

    # flag: ¿pasa el umbral de apalancamiento?
    df["leverage_ok"] = df["netDebt_to_EBITDA"].apply(_lev_ok)

    # columnas formateadas para mostrar
    df["netDebt_to_EBITDA_fmt"] = df["netDebt_to_EBITDA"].apply(
        lambda x: "—"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{x:.2f}"
    )
    df["marketCap_fmt"] = df["marketCap"].apply(_fmt_num)

    df["revenueGrowth_pct"] = df["revenueGrowth"].apply(_fmt_pct)
    df["ocfGrowth_pct"] = df["operatingCashFlowGrowth"].apply(_fmt_pct)
    df["fcfGrowth_pct"] = df["freeCashFlowGrowth"].apply(_fmt_pct)
    df["debtGrowth_pct"] = df["debtGrowth"].apply(_fmt_pct)

    df["rev_CAGR_5y_pct"] = df["rev_CAGR_5y"].apply(_fmt_pct)
    df["rev_CAGR_3y_pct"] = df["rev_CAGR_3y"].apply(_fmt_pct)
    df["ocf_CAGR_5y_pct"] = df["ocf_CAGR_5y"].apply(_fmt_pct)
    df["ocf_CAGR_3y_pct"] = df["ocf_CAGR_3y"].apply(_fmt_pct)

    df["altmanZScore_fmt"] = df["altmanZScore"].apply(
        lambda x: "—"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{float(x):.2f}"
    )
    df["piotroskiScore_fmt"] = df["piotroskiScore"].apply(
        lambda x: "—"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{float(x):.0f}/9"
    )

    return df


# ======================================================================================
# Panel de detalle por ticker
# ======================================================================================

def _plot_safe(ax, xs, ys, xlabel, ylabel):
    """Evita que Streamlit reviente si las listas vienen vacías o de distinto largo."""
    if isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys) and len(xs) > 0:
        ax.plot(xs, ys, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


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
        nde = row.get("netDebt_to_EBITDA")
        nde_fmt = (
            "—"
            if nde is None or (isinstance(nde, float) and math.isnan(nde))
            else f"{nde:.2f}"
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
        _plot_safe(ax, years, fcfps_hist, "Año", "FCF/acción")
        st.pyplot(fig)

    with colB:
        st.caption("Acciones en circulación (dilución vs recompras)")
        fig2, ax2 = plt.subplots()
        _plot_safe(ax2, years, shares_hist, "Año", "Acciones (unidades)")
        st.pyplot(fig2)

    with colC:
        st.caption("Deuda Neta")
        fig3, ax3 = plt.subplots()
        _plot_safe(ax3, years, net_debt_hist, "Año", "Deuda Neta (USD)")
        st.pyplot(fig3)

    st.caption(
        "- FCF/acción subiendo = más caja real para el dueño.\n"
        "- Acciones bajando = recompras (alineación management/accionista).\n"
        "- Deuda neta estable o bajando = menos riesgo de liquidez futura."
    )


# ======================================================================================
# CONFIG DE PÁGINA
# ======================================================================================

st.set_page_config(
    page_title="FUND Screener",
    page_icon="💸",
    layout="wide"
)

st.title("💸 FUND Screener")
st.write(
    "1) Filtramos el mercado a solo large caps sanas y creciendo.\n"
    "2) Vemos la lista final (calidad financiera + growth sostenible).\n"
    "3) Abrimos una empresa y miramos insiders / news / transcript / tendencias históricas."
)

# ======================================================================================
# SIDEBAR
# ======================================================================================

st.sidebar.header("Parámetros")

max_leverage = st.sidebar.slider(
    "Máximo Net Debt / EBITDA permitido",
    min_value=0.0,
    max_value=6.0,
    value=MAX_NET_DEBT_TO_EBITDA,
    step=0.5
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Screening / Refresh Data")


# ======================================================================================
# SESIÓN
# ======================================================================================

if "snapshot_rows" not in st.session_state:
    st.session_state["snapshot_rows"] = []

if "last_error" not in st.session_state:
    st.session_state["last_error"] = None


# ======================================================================================
# BLOQUE KEPT_SYMS (lista elegida por el usuario en otra parte de la app)
# y panel chico arriba con métricas 'compounder'
# ======================================================================================

kept_syms = (
    st.session_state["kept"]["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    if ("kept" in st.session_state and not st.session_state["kept"].empty)
    else []
)

# build_full_snapshot necesita esa lista -> devuelve DF con columnas tipo:
#   symbol, companyName, sector, marketCap,
#   fcf_per_share_slope_5y, buyback_pct_5y,
#   net_debt_to_ebitda_last, is_quality_compounder, ...
try:
    final_df = build_full_snapshot(kept_syms) if kept_syms else pd.DataFrame()
except Exception as e:
    st.warning(f"No pude generar la vista enriquecida de tus tickers guardados: {e}")
    final_df = pd.DataFrame()

if not final_df.empty:
    st.subheader("Tu watchlist enriquecida (compounders / buybacks / deuda)")
    cols_show = [
        "symbol",
        "companyName",
        "sector",
        "marketCap",
        "fcf_per_share_slope_5y",
        "buyback_pct_5y",
        "net_debt_to_ebitda_last",
        "is_quality_compounder",
    ]
    cols_show = [c for c in cols_show if c in final_df.columns]

    st.dataframe(
        final_df[cols_show].reset_index(drop=True),
        use_container_width=True,
        height=300
    )

    st.caption(
        "- fcf_per_share_slope_5y ↑: FCF por acción acelera → valor real al dueño.\n"
        "- buyback_pct_5y ↑: recompras consistentes → menos dilución.\n"
        "- net_debt_to_ebitda_last bajo: menos riesgo en shock de liquidez.\n"
        "- is_quality_compounder: heurístico de 'flywheel' de caja reinvertible."
    )


# ======================================================================================
# RUN BUTTON: poblar shortlist completa del mercado
# ======================================================================================

if run_btn:
    with st.spinner("Generando shortlist..."):
        try:
            rows = build_market_snapshot()
            st.session_state["snapshot_rows"] = rows
            st.session_state["last_error"] = None
        except Exception as e:
            st.session_state["last_error"] = str(e)
            st.session_state["snapshot_rows"] = []


rows_data = st.session_state["snapshot_rows"]

# panel de debug en sidebar
if st.session_state["last_error"]:
    st.sidebar.error(f"Error al armar shortlist: {st.session_state['last_error']}")
else:
    st.sidebar.success(
        f"{len(rows_data)} tickers cargados" if rows_data else "Sin datos aún"
    )


# ======================================================================================
# TABS PRINCIPALES
# ======================================================================================

tab1, tab2 = st.tabs([
    "1. Shortlist final",
    "2. Detalle Ticker"
])


# ------------------------------------------------------------------
# TAB 1: tabla con filtros financieros / crecimiento
# ------------------------------------------------------------------
with tab1:
    st.subheader("1. Shortlist: large caps sólidas + crecimiento compuesto ≥15%")

    if not rows_data:
        st.info("Presiona 'Run Screening / Refresh Data' en el sidebar.")
    else:
        df_all = dataframe_from_rows(rows_data, max_leverage)

        st.write(
            "Esta lista pasó (heurística inicial):\n"
            "- Cap grande (≥10B USD)\n"
            "- Altman Z sano (bajo riesgo de quiebra)\n"
            "- Piotroski alto (calidad contable / eficiencia operativa)\n"
            "- Crecimiento positivo en ventas, OCF y FCF\n"
            "- Deuda que NO se dispara\n"
            "- CAGR ≥15% en revenue/OCF por acción (3-5y)\n"
            "- Apalancamiento ≤ tu umbral de Net Debt / EBITDA"
        )

        df_screen = df_all[df_all["leverage_ok"]].copy()

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
            "netDebt_to_EBITDA_fmt",
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

        st.caption(
            "Lectura rápida:\n"
            "- Altman Z ↑ y Piotroski ↑ = balance sólido + disciplina.\n"
            "- OCF/FCF creciendo y deuda controlada = caja real que escala sin apalancarse.\n"
            "- CAGR ≥15% = potencial de componer valor a largo plazo.\n"
            "- Net Debt/EBITDA bajo = resiliencia en estrés de liquidez."
        )


# ------------------------------------------------------------------
# TAB 2: ficha cualitativa de un ticker de la shortlist
# ------------------------------------------------------------------
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

            # base_core = dict original crudo para ese ticker
            base_core = next((r for r in rows_data if r.get("ticker") == picked), None)
            if base_core is None:
                st.error("No encontré datos base de ese ticker.")
            else:
                try:
                    detailed = enrich_company_snapshot(base_core.copy())
                except Exception as e:
                    st.error(f"No pude completar el detalle cualitativo: {e}")
                    detailed = base_core.copy()

                # asegurar campos mínimos para el panel
                nde = detailed.get("netDebt_to_EBITDA")
                if nde is None or (isinstance(nde, float) and math.isnan(nde)):
                    detailed["netDebt_to_EBITDA_fmt"] = "—"
                else:
                    detailed["netDebt_to_EBITDA_fmt"] = f"{nde:.2f}"

                # historiales para los gráficos
                detailed["years"] = detailed.get("years", [])
                detailed["fcf_per_share_hist"] = detailed.get("fcf_per_share_hist", [])
                detailed["shares_hist"] = detailed.get("shares_hist", [])
                detailed["net_debt_hist"] = detailed.get("net_debt_hist", [])

                # defaults cualitativos
                if "transcript_summary" not in detailed:
                    detailed["transcript_summary"] = "Sin señales fuertes en la última call."
                if "sector" not in detailed:
                    detailed["sector"] = base_core.get("sector", "—")
                if "industry" not in detailed:
                    detailed["industry"] = base_core.get("industry", "—")
                if "moat_flag" not in detailed:
                    detailed["moat_flag"] = base_core.get("moat_flag", "—")
                if "insider_signal" not in detailed:
                    detailed["insider_signal"] = base_core.get("insider_signal", "neutral")
                if "sentiment_flag" not in detailed:
                    detailed["sentiment_flag"] = base_core.get("sentiment_flag", "neutral")
                if "sentiment_reason" not in detailed:
                    detailed["sentiment_reason"] = detailed.get(
                        "sentiment_reason",
                        "tono mixto/sectorial"
                    )
                if "business_summary" not in detailed:
                    detailed["business_summary"] = base_core.get("business_summary", "—")
                if "why_it_matters" not in detailed:
                    detailed["why_it_matters"] = base_core.get("why_it_matters", "—")
                if "core_risk_note" not in detailed:
                    detailed["core_risk_note"] = base_core.get(
                        "core_risk_note",
                        "riesgo principal no crítico visible"
                    )

                render_detail_panel(pd.Series(detailed))
