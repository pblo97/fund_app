# app.py
#
# Interfaz Streamlit con 2 tabs:
# 1. Shortlist final (large caps sanas y creciendo)
# 2. Detalle Ticker (enriquecido on-demand)
#
# Flujo:
# - Tab 1: corres screening y ves la shortlist cuantitativa final
#          (ya pasó solvencia, crecimiento, deuda controlada, CAGR >=15%).
# - Tab 2: eliges 1 ticker y ves ficha más cualitativa (insiders, news, transcript).
#
# Nota:
# Ya NO usamos expected_return / owners_yield / hurdle.
# Mostramos métricas nuevas: Altman Z, Piotroski, crecimiento FCF/OCF,
# leverage, etc.

import streamlit as st
import pandas as pd
import math
import json
import matplotlib.pyplot as plt

from orchestrator import build_full_snapshot, enrich_company_snapshot
from config import MAX_NET_DEBT_TO_EBITDA

# ---------- Helpers de formato ----------

def _fmt_pct(x):
    """
    Formato porcentaje para números tipo 0.123 -> '12.3%'.
    Si es None o NaN -> '—'.
    """
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
    """
    Formato numérico grande en B/M.
    """
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
    """
    Convierto strings tipo '["a", "b"]' a list, o devuelvo [] si falla.
    """
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    if isinstance(v, list):
        return v
    return []


def dataframe_from_rows(rows: list[dict], max_leverage: float):
    """
    Convierte la lista de snapshots core (dicts) en DataFrame para mostrar en la tabla.
    Calcula:
    - leverage_ok (netDebt/EBITDA <= max_leverage)
    - formatos bonitos de market cap, leverage, crecimientos, etc.
    """

    df = pd.DataFrame(rows).copy()

    # asegurar columnas que podríamos no tener en algunas empresas
    for col in [
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
    ]:
        if col not in df.columns:
            df[col] = None

    # flag de apalancamiento sano
    def _lev_ok(x):
        if x is None:
            return True
        if isinstance(x, float) and math.isnan(x):
            return True
        try:
            return float(x) <= max_leverage
        except Exception:
            return True

    df["leverage_ok"] = df["netDebt_to_EBITDA"].apply(_lev_ok)

    # formatos para vista
    df["netDebt_to_EBITDA_fmt"] = df["netDebt_to_EBITDA"].apply(
        lambda x: "—"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{x:.2f}"
    )
    df["marketCap_fmt"] = df["marketCap"].apply(_fmt_num)

    # crecimiento puntual (último FY vs FY previo)
    df["revenueGrowth_pct"] = df["revenueGrowth"].apply(_fmt_pct)
    df["ocfGrowth_pct"] = df["operatingCashFlowGrowth"].apply(_fmt_pct)
    df["fcfGrowth_pct"] = df["freeCashFlowGrowth"].apply(_fmt_pct)
    df["debtGrowth_pct"] = df["debtGrowth"].apply(_fmt_pct)

    # CAGR compuesto 3y/5y
    df["rev_CAGR_5y_pct"] = df["rev_CAGR_5y"].apply(_fmt_pct)
    df["rev_CAGR_3y_pct"] = df["rev_CAGR_3y"].apply(_fmt_pct)
    df["ocf_CAGR_5y_pct"] = df["ocf_CAGR_5y"].apply(_fmt_pct)
    df["ocf_CAGR_3y_pct"] = df["ocf_CAGR_3y"].apply(_fmt_pct)

    # Altman / Piotroski directo
    # no los pasamos a % porque son scores
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


def render_detail_panel(row: pd.Series):
    """
    Panel de detalle fundamental + cualitativo para un ticker.
    row debe tener llaves ya enriquecidas por enrich_company_snapshot().
    """

    # aseguramos campos que podría no tener
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
        nde_fmt = "—" if nde is None or (isinstance(nde, float) and math.isnan(nde)) else f"{nde:.2f}"
        st.metric("Net Debt / EBITDA", nde_fmt)
        st.metric("Deuda vs año prev.", _fmt_pct(row.get("debtGrowth")))

    st.subheader("Snapshot estratégico")
    st.write(f"**Sector / Industria:** {row.get('sector','—')} / {row.get('industry','—')}")
    st.write(f"**Moat (heurístico):** {row.get('moat_flag','—')}")
    st.write(f"**Insiders:** {row.get('insider_signal','(sin dato)')}")
    st.write(f"**Sentimiento noticias:** {row.get('sentiment_flag','neutral')} — {row.get('sentiment_reason','tono mixto')}")
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

    st.caption(
        "- FCF/acción subiendo = más caja real para el dueño.\n"
        "- Acciones bajando = recompras (alineación management/accionista).\n"
        "- Deuda neta estable o bajando = menos riesgo de liquidez futura."
    )


# ---------- Configuración de página ----------

st.set_page_config(
    page_title="FUND Screener",
    page_icon="💸",
    layout="wide"
)

st.title("💸 FUND Screener")
st.write(
    "1) Filtramos el mercado a solo large caps sanas y creciendo.\n"
    "2) Vemos la lista final.\n"
    "3) Abrimos una empresa y miramos el cualitativo (insiders, news, transcript)."
)

# Sidebar con parámetros globales
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

# Estado global del snapshot
if "snapshot_rows" not in st.session_state:
    st.session_state["snapshot_rows"] = []

if run_btn:
    rows = build_full_snapshot()
    st.session_state["snapshot_rows"] = rows

rows_data = st.session_state["snapshot_rows"]

tab1, tab2 = st.tabs([
    "1. Shortlist final",
    "2. Detalle Ticker"
])


# =======================
# TAB 1 · SHORTLIST FINAL
# =======================
with tab1:
    st.subheader("1. Shortlist: large caps sólidas + crecimiento compuesto ≥15%")

    if not rows_data:
        st.info("Presiona 'Run Screening / Refresh Data' en el sidebar para generar el universo inicial.")
    else:
        df_all = dataframe_from_rows(rows_data, max_leverage)

        st.write(
            "Esta lista ya pasó:\n"
            "- Cap grande (≥10B USD)\n"
            "- Altman Z sano (riesgo de quiebra bajo)\n"
            "- Piotroski alto (calidad contable)\n"
            "- Crecimiento positivo en ventas, EBIT, OCF y FCF\n"
            "- Deuda no creciendo\n"
            "- CAGR ≥15% en revenue/OCF por acción (3-5y)\n"
            "- Apalancamiento dentro de tu umbral de Net Debt / EBITDA"
        )

        # filtramos por leverage_ok según tu slider
        df_screen = df_all[df_all["leverage_ok"]].copy()

        # columnas a mostrar
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

        # solo columnas que existan realmente
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
            "- AltmanZScore ↑ y Piotroski ↑ = balance fuerte + disciplina operativa.\n"
            "- Crec. OCF/FCF > 0 y DeudaGrowth ≤ 0 = la caja sube sin endeudarse más.\n"
            "- Rev/OCF CAGR ≥15% = motor de composición real a varios años.\n"
            "- Net Debt/EBITDA bajo = menos riesgo si el mercado se aprieta."
        )


# =======================
# TAB 2 · DETALLE TICKER
# =======================
with tab2:
    st.subheader("2. Ficha detallada de la empresa")

    if not rows_data:
        st.info("Primero genera datos en la pestaña '1. Shortlist final'.")
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

            # buscamos el snapshot base (antes de enriquecer)
            base_core = next((r for r in rows_data if r.get("ticker") == picked), None)
            if base_core is None:
                st.error("No pude encontrar datos base de ese ticker.")
                st.stop()

            # enriquecemos SOLO este ticker (insiders, news, transcript, riesgo cualitativo)
            try:
                detailed = enrich_company_snapshot(base_core.copy())
            except Exception as e:
                st.error(f"No pude completar el detalle cualitativo: {e}")
                detailed = base_core.copy()

            # netDebt_to_EBITDA_fmt para mostrar bonito
            nde = detailed.get("netDebt_to_EBITDA")
            if nde is None or (isinstance(nde, float) and math.isnan(nde)):
                detailed["netDebt_to_EBITDA_fmt"] = "—"
            else:
                detailed["netDebt_to_EBITDA_fmt"] = f"{nde:.2f}"

            # asegurar listas históricas, por si la métrica no se pudo calcular
            detailed["years"] = detailed.get("years", [])
            detailed["fcf_per_share_hist"] = detailed.get("fcf_per_share_hist", [])
            detailed["shares_hist"] = detailed.get("shares_hist", [])
            detailed["net_debt_hist"] = detailed.get("net_debt_hist", [])

            # asegurar campos cualitativos
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
                detailed["sentiment_reason"] = base_core.get("sentiment_reason", "tono mixto/sectorial")
            if "business_summary" not in detailed:
                detailed["business_summary"] = base_core.get("business_summary", "—")
            if "why_it_matters" not in detailed:
                detailed["why_it_matters"] = base_core.get("why_it_matters", "—")
            if "core_risk_note" not in detailed:
                detailed["core_risk_note"] = base_core.get("core_risk_note", "riesgo principal no crítico visible")

            render_detail_panel(pd.Series(detailed))
