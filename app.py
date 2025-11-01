# app.py
#
# Interfaz Streamlit con 3 tabs:
# 1. Screener base
# 2. Oportunidades vs Vigilancia
# 3. Detalle Ticker
#
# Flujo:
# - Tab 1: corres screening y ves universo limpio.
# - Tab 2: ves qué cumple tu hurdle (≥15%) y qué es watchlist (<15%).
# - Tab 3: eliges 1 ticker y ves la mini-tesis, riesgos y series históricas.


import streamlit as st
import pandas as pd
import math
import json
import matplotlib.pyplot as plt

from orchestrator import build_full_snapshot
from config import EXPECTED_RETURN_HURDLE, MAX_NET_DEBT_TO_EBITDA
from orchestrator import enrich_company_snapshot

# ---------- Helpers de formato ----------

def _fmt_pct(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:.1f}%"

def _fmt_num(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f}B"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    return f"{x:.0f}"

def _row_quality_tag(moat_flag, insider_signal):
    if moat_flag == "fuerte" and insider_signal == "buy":
        return "🏰🟢 moat+insider buy"
    if moat_flag == "fuerte":
        return "🏰 moat fuerte"
    if insider_signal == "buy":
        return "🟢 insider buy"
    if moat_flag == "media":
        return "🟡 moat medio"
    return "⚪ sin moat claro"

def _risk_badge(expected_return, hurdle, sentiment_flag):
    if expected_return is not None and expected_return >= hurdle:
        return "🟢"
    if sentiment_flag == "neg":
        return "🔴"
    return "🟡"

def _ensure_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    if isinstance(v, list):
        return v
    return []


def dataframe_from_rows(rows: list[dict], hurdle: float, max_leverage: float):
    """
    Convierte lista de snapshots (dicts) en DataFrame:
    - calcula columnas formateadas
    - marca leverage_ok
    - marca meets_hurdle
    """
    df = pd.DataFrame(rows).copy()

    df["leverage_ok"] = df["netDebt_to_EBITDA"].apply(
        lambda x: (x is None)
        or (not (isinstance(x, float) and math.isnan(x)))
        and (x <= max_leverage)
    )

    df["expected_return_pct"] = df["expected_return"].apply(_fmt_pct)
    df["owners_yield_pct"] = df["owners_yield"].apply(_fmt_pct)
    df["cagr_fcfps_5y_pct"] = df["cagr_fcfps_5y"].apply(_fmt_pct)
    df["fcf_yield_now_pct"] = df["fcf_yield_now"].apply(_fmt_pct)
    df["roe_ttm_pct"] = df["roe_ttm"].apply(_fmt_pct)

    df["netDebt_to_EBITDA_fmt"] = df["netDebt_to_EBITDA"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"
    )

    df["marketCap_fmt"] = df["marketCap"].apply(_fmt_num)

    df["quality_tag"] = df.apply(
        lambda r: _row_quality_tag(r.get("moat_flag"), r.get("insider_signal")), axis=1
    )
    df["risk_signal"] = df.apply(
        lambda r: _risk_badge(r.get("expected_return"), hurdle, r.get("sentiment_flag")),
        axis=1
    )

    df["meets_hurdle"] = df["expected_return"].apply(
        lambda x: (x is not None and x >= hurdle)
    )

    return df


def render_detail_panel(row: pd.Series):
    st.markdown(f"## {row['companyName']} ({row['ticker']})")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Return (≈ anual)", _fmt_pct(row["expected_return"]))
        st.metric("Owner's Yield", _fmt_pct(row["owners_yield"]))
    with col2:
        st.metric("CAGR FCF/acción (~5y)", _fmt_pct(row["cagr_fcfps_5y"]))
        st.metric("FCF Yield ahora", _fmt_pct(row["fcf_yield_now"]))
    with col3:
        st.metric("Net Debt / EBITDA", row["netDebt_to_EBITDA_fmt"])
        st.metric("ROE TTM", _fmt_pct(row["roe_ttm"]))

    st.subheader("Snapshot estratégico")
    st.write(f"**Sector / Industria:** {row['sector']} / {row['industry']}")
    st.write(f"**Moat (heurístico):** {row['moat_flag']}")
    st.write(f"**Insiders:** {row['insider_signal']}")
    st.write(f"**Sentimiento noticias:** {row['sentiment_flag']} — {row['sentiment_reason']}")
    st.write(f"**Resumen negocio:** {row['business_summary']}")
    st.write(f"**Por qué importa:** {row['why_it_matters']}")
    st.write(f"**Riesgo visible:** {row['core_risk_note']}")
    st.write(f"**Última earnings call:** {row['transcript_summary']}")

    st.subheader("Históricos clave (~5 años)")

    years = _ensure_list(row["years"])
    fcfps_hist = _ensure_list(row["fcf_per_share_hist"])
    shares_hist = _ensure_list(row["shares_hist"])
    net_debt_hist = _ensure_list(row["net_debt_hist"])

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
        "- FCF/acción subiendo = el dueño recibe más caja con el tiempo.\n"
        "- Acciones bajando = recompras reales (alineación management / accionista).\n"
        "- Deuda neta estable o bajando = menor riesgo de liquidez futura."
    )


# ---------- Configuración de página ----------

st.set_page_config(
    page_title="FUND Screener",
    page_icon="💸",
    layout="wide"
)

st.title("💸 FUND Screener")
st.write(
    "Flujo:\n"
    "1) limpiamos el universo → "
    "2) vemos oportunidades vs vigilancia → "
    "3) abrimos la ficha cualitativa."
)

# Sidebar con parámetros globales
st.sidebar.header("Parámetros")

hurdle_input = st.sidebar.slider(
    "Hurdle de retorno esperado (anual, %)",
    min_value=5,
    max_value=30,
    value=int(EXPECTED_RETURN_HURDLE * 100),
    step=1
) / 100.0

max_leverage = st.sidebar.slider(
    "Máximo Net Debt / EBITDA permitido",
    min_value=0.0,
    max_value=6.0,
    value=MAX_NET_DEBT_TO_EBITDA,
    step=0.5
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Screening / Refresh Data")

# Estado global de snapshot
if "snapshot_rows" not in st.session_state:
    st.session_state["snapshot_rows"] = []

if run_btn:
    rows = build_full_snapshot()
    st.session_state["snapshot_rows"] = rows

rows_data = st.session_state["snapshot_rows"]

tab1, tab2, tab3 = st.tabs([
    "1. Screener base",
    "2. Oportunidades / Vigilancia",
    "3. Detalle Ticker"
])


# =======================
# TAB 1 · SCREENER BASE
# =======================
with tab1:
    st.subheader("1. Screener base (calidad mínima)")
    if not rows_data:
        st.info("Presiona 'Run Screening / Refresh Data' en el sidebar para generar el universo inicial.")
    else:
        df_all = dataframe_from_rows(rows_data, hurdle_input, max_leverage)

        st.write(
            "Este es el universo depurado: liquidez mínima, market cap decente, "
            "ROE aceptable y deuda dentro de lo tolerado. "
            "O sea: empresas que valen la pena mirar."
        )

        cols_basic = [
            "ticker",
            "companyName",
            "sector",
            "industry",
            "marketCap_fmt",
            "roe_ttm_pct",
            "netDebt_to_EBITDA_fmt",
            "moat_flag",
            "insider_signal",
        ]

        df_screen_view = df_all[df_all["leverage_ok"]][cols_basic].sort_values("companyName")
        st.dataframe(
            df_screen_view.reset_index(drop=True),
            use_container_width=True,
            height=500
        )

        st.caption(
            "Claves:\n"
            "- ROE alto = gestión eficiente del capital.\n"
            "- Net Debt / EBITDA bajo = menor riesgo de ahogo financiero.\n"
            "- Moat / insiders = ventaja estructural y alineación con el accionista."
        )


# ================================
# TAB 2 · OPORTUNIDADES / WATCH
# ================================
with tab2:
    st.subheader("2. Oportunidades vs Vigilancia")

    if not rows_data:
        st.info("Aún no hay datos. Ve a '1. Screener base' y ejecuta el screening.")
    else:
        df_all = dataframe_from_rows(rows_data, hurdle_input, max_leverage)
        df_f = df_all[df_all["leverage_ok"]].copy()

        df_opps = df_f[df_f["meets_hurdle"]].copy()
        df_watch = df_f[~df_f["meets_hurdle"]].copy()

        df_opps = df_opps.sort_values("expected_return", ascending=False)
        df_watch = df_watch.sort_values("expected_return", ascending=False)

        col_top1, col_top2, col_top3 = st.columns(3)
        with col_top1:
            st.metric("Acciones Oportunidad (≥ hurdle)", f"{len(df_opps)}")
        with col_top2:
            st.metric("Acciones en Vigilancia (< hurdle)", f"{len(df_watch)}")
        with col_top3:
            best_er = df_opps["expected_return"].max() if not df_opps.empty else None
            st.metric(
                "Mejor Expected Return",
                _fmt_pct(best_er) if best_er is not None else "—"
            )

        st.markdown("### 🟢 Oportunidades (cumplen tu retorno objetivo)")
        st.caption("≥ hurdle de retorno esperado + deuda bajo tu umbral.")
        cols_show_opps = [
            "risk_signal",
            "ticker",
            "companyName",
            "expected_return_pct",
            "owners_yield_pct",
            "cagr_fcfps_5y_pct",
            "fcf_yield_now_pct",
            "netDebt_to_EBITDA_fmt",
            "roe_ttm_pct",
            "moat_flag",
            "insider_signal",
            "sentiment_flag",
            "quality_tag",
            "marketCap_fmt",
        ]
        if df_opps.empty:
            st.warning("No hay nada hoy que te pague tu hurdle con la calidad que exiges.")
        else:
            st.dataframe(
                df_opps[cols_show_opps].reset_index(drop=True),
                use_container_width=True,
                height=350
            )

        st.markdown("### 🟡 Vigilancia (buena calidad, pero aún cara)")
        st.caption(
            "Negocios sólidos (moat, disciplina de capital), "
            "pero expected_return < hurdle. Lista para cazarlas si bajan."
        )
        cols_show_watch = [
            "risk_signal",
            "ticker",
            "companyName",
            "expected_return_pct",
            "owners_yield_pct",
            "cagr_fcfps_5y_pct",
            "fcf_yield_now_pct",
            "netDebt_to_EBITDA_fmt",
            "roe_ttm_pct",
            "moat_flag",
            "insider_signal",
            "sentiment_flag",
            "quality_tag",
            "marketCap_fmt",
        ]
        if df_watch.empty:
            st.info("No hay candidatas de vigilancia bajo tus filtros actuales.")
        else:
            st.dataframe(
                df_watch[cols_show_watch].reset_index(drop=True),
                use_container_width=True,
                height=350
            )

        st.caption(
            "- 'risk_signal' 🟢 = supera hurdle.\n"
            "- 'risk_signal' 🔴 = prensa negativa / alerta.\n"
            "- 'quality_tag' resume moat e insiders en una línea."
        )


# =======================
# TAB 3 · DETALLE TICKER
# =======================
with tab3:
    st.subheader("3. Detalle de una empresa")

    if not rows_data:
        st.info("Primero genera datos en la pestaña '1. Screener base'.")
    else:
        # Construimos el universo filtrado con los mismos parámetros de hurdle y leverage
        df_all = dataframe_from_rows(rows_data, hurdle_input, max_leverage)
        df_f = df_all[df_all["leverage_ok"]].copy()

        tickers_available = df_f["ticker"].tolist()

        if not tickers_available:
            st.warning("No quedan tickers válidos tras tus filtros de hurdle y deuda.")
        else:
            picked = st.selectbox(
                "Elige un ticker para ver su ficha fundamental completa:",
                tickers_available
            )

            # buscamos en rows_data el snapshot 'core' ya calculado
            base_core = next((r for r in rows_data if r["ticker"] == picked), None)
            if base_core is None:
                st.error("No pude encontrar datos base de ese ticker.")
                st.stop()

            # enriquecemos en vivo con insiders, noticias, transcript, riesgo cualitativo, etc.
            try:
                from orchestrator import enrich_company_snapshot
                detailed = enrich_company_snapshot(base_core.copy())
            except Exception as e:
                st.error(f"No pude completar el detalle cualitativo: {e}")
                # en caso de fallo, mostramos al menos lo cuantitativo
                detailed = base_core.copy()

            # Aseguramos que estén los campos que usa render_detail_panel

            # formateo de Net Debt / EBITDA para la métrica
            if "netDebt_to_EBITDA" in detailed:
                ndeb = detailed["netDebt_to_EBITDA"]
                if ndeb is None or (isinstance(ndeb, float) and math.isnan(ndeb)):
                    detailed["netDebt_to_EBITDA_fmt"] = "—"
                else:
                    detailed["netDebt_to_EBITDA_fmt"] = f"{ndeb:.2f}"
            else:
                detailed["netDebt_to_EBITDA_fmt"] = "—"

            # nos aseguramos que estas listas existan aunque la API falle
            detailed["years"] = detailed.get("years", [])
            detailed["fcf_per_share_hist"] = detailed.get("fcf_per_share_hist", [])
            detailed["shares_hist"] = detailed.get("shares_hist", [])
            detailed["net_debt_hist"] = detailed.get("net_debt_hist", [])

            # también nos aseguramos de que transcript_summary exista
            if "transcript_summary" not in detailed:
                detailed["transcript_summary"] = "sin señales clave detectables en la última call"

            # igualamos algunos nombres que usa el panel (por seguridad)
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
                detailed["sentiment_reason"] = detailed.get("sentiment_reason", "tono mixto/sectorial")
            if "business_summary" not in detailed:
                detailed["business_summary"] = base_core.get("business_summary", "—")
            if "why_it_matters" not in detailed:
                detailed["why_it_matters"] = base_core.get("why_it_matters", "—")
            if "core_risk_note" not in detailed:
                detailed["core_risk_note"] = base_core.get("core_risk_note", "riesgo principal no crítico visible")

            # ROE TTM ya viene como número decimal en detailed["roe_ttm"]
            # render_detail_panel usa _fmt_pct internamente, así que estamos bien.

            render_detail_panel(detailed)
