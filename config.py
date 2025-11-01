
import os

# Intento de leer la API key:
# 1. streamlit.secrets["FMP_API_KEY"]
# 2. variable de entorno FMP_API_KEY
# 3. fallback "" (vacío → error controlado más adelante)

def _load_fmp_key() -> str:
    # Intentar desde streamlit.secrets si estamos dentro de Streamlit
    try:
        import streamlit as st
        if "FMP_API_KEY" in st.secrets:
            return st.secrets["FMP_API_KEY"]
    except Exception:
        # No estamos en streamlit, o no existe secrets todavía
        pass

    # Intentar desde variable de entorno
    env_key = os.getenv("FMP_API_KEY")
    if env_key:
        return env_key

    # Fallback vacío → lo manejamos en runtime con error amigable
    return ""

FMP_API_KEY = _load_fmp_key()


# =========================
# Parámetros del screener
# =========================

SCREENER_PARAMS = {
    "isEtf": "false",
    "isFund": "false",
    "isActivelyTrading": "true",

    # evitamos basura/pennystocks
    "priceMoreThan": 1,

    # empresas con cierto tamaño
    "marketCapMoreThan": 1_000_000_000,  # >= 1B USD

    # solo mercados grandes/estables (podemos extender luego)
    "country": "US",

    # liquidez decente: evita papel muerto
    "volumeMoreThan": 100000,

    # beta en rango razonable para no tener cosas muertas ni cohetes suicidas
    "betaMoreThan": 0.5,
    "betaLowerThan": 1.5,

    # vamos a correrlo para NASDAQ y NYSE
    "includeAllShareClasses": "false",
    "limit": 1000,
}

# Corte mínimo de calidad para aceptar al universo inicial (calidad básica)
MIN_ROE_TTM = 0.12  # 12%

# Hurdle por defecto para "quiero al menos X% anual esperado"
EXPECTED_RETURN_HURDLE = 0.15  # 15%

# Apalancamiento máximo aceptable por defecto
MAX_NET_DEBT_TO_EBITDA = 3.0

# Palabras que marcan señales negativas/peligro en prensa
NEG_WORDS = [
    "fraud", "probe", "sec investigation", "lawsuit",
    "guidance cut", "misses", "missed estimates",
    "recall", "downgrade", "layoffs", "restructuring",
    "accounting irregularities", "restatement",
    "whistleblower", "governance issue", "antitrust"
]

# Palabras que suelen aparecer cuando el tono es positivo / bullish
POS_WORDS = [
    "beats expectations", "beats estimates", "raises guidance",
    "buyback", "repurchase authorization",
    "dividend increase", "contract win",
    "price target raised", "accretive acquisition",
    "strong demand", "record revenue", "margin expansion"
]

# Riesgos duros que queremos resaltar
CRITICAL_RISK_TERMS = [
    "sec investigation", "fraud", "accounting irregularities",
    "liquidity crisis", "going concern", "default risk",
    "covenant breach"
]