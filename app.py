import streamlit as st
import pandas as pd

from orchestrator import pipeline_run
from views import build_detail_view

st.set_page_config(page_title="QVM Screener", layout="wide")

st.title("QVM / Compounder Screener")

# Ejecutar pipeline una vez por sesión
if "pipeline_cache" not in st.session_state:
    st.session_state["pipeline_cache"] = pipeline_run()

data_pack = st.session_state["pipeline_cache"]
df_list: pd.DataFrame = data_pack["df_list_view"]
snaps: list[dict] = data_pack["snapshots_final"]

tab1, tab2 = st.tabs(["1. Screener", "2. Detalle empresa"])

with tab1:
    st.subheader("Ranking de candidatas (limitado a 40 tickers)")
    st.write("Ordenado por retorno esperado (CAGR forward estimado).")
    st.dataframe(df_list, use_container_width=True)

with tab2:
    st.subheader("Ficha fundamental completa")

    # Validación defensiva
    if df_list is None or df_list.empty:
        st.info(
            "No hay empresas que pasen todos los filtros (calidad financiera, "
            "crecimiento operativo, deuda sana). "
            "Prueba relajar umbrales o verifica que la API esté devolviendo datos."
        )
    else:
        # ahora sí sabemos que df_list tiene columnas
        tickers_available = df_list["ticker"].dropna().unique().tolist()

        if not tickers_available:
            st.info(
                "No hay tickers disponibles en este momento, "
                "aunque el screener tiene filas sin ticker válido."
            )
        else:
            picked = st.selectbox("Elige un ticker:", tickers_available)

            snap = next((s for s in snaps if s.get("ticker") == picked), None)

            if snap is None:
                st.warning(
                    "No encontré el snapshot detallado de ese ticker. "
                    "Puede que fallara la descarga de fundamentales o texto."
                )
            else:
                detail = build_detail_view(snap)

                for section, block in detail.items():
                    st.markdown(f"### {section}")
                    for k, v in block.items():
                        st.write(f"**{k}:** {v}")
                    st.markdown("---")
