import os

# =========================
# API KEY HANDLING
# =========================

def _load_fmp_key() -> str:
    """
    Intenta obtener la API key de FMP en este orden:
    1. streamlit.secrets["FMP_API_KEY"]  (cuando estamos dentro de Streamlit)
    2. variable de entorno FMP_API_KEY   (cuando corres scripts sueltos)
    3. fallback "" (string vacío)        (lo tratamos como error luego al hacer llamadas reales)
    """
    try:
        import streamlit as st  # esto falla si no estás en Streamlit
        if "FMP_API_KEY" in st.secrets:
            return st.secrets["FMP_API_KEY"]
    except Exception:
        pass

    env_key = os.getenv("FMP_API_KEY")
    if env_key:
        return env_key

    return ""  # sin key → las llamadas a la API deberían fallar de forma controlada más adelante


FMP_API_KEY = _load_fmp_key()


# =========================
# Parámetros del screener
# =========================
# Estos son filtros base que se le pasan al /stock-screener de FMP.
# OJO: luego en run_screener_for_exchange() nosotros:
#  - inyectamos el exchange (NASDAQ / NYSE / AMEX)
#  - volvemos a filtrar sólo large caps >= 10B market cap
# así que esto es la "primera pasada", no el corte final.

SCREENER_PARAMS = {
    "isEtf": "false",
    "isFund": "false",
    "isActivelyTrading": "true",

    # evita penny garbage
    "priceMoreThan": 1,

    # tamaño mínimo razonable (>=1B). El corte real grande lo hacemos a >=10B luego.
    "marketCapMoreThan": 1_000_000_000,

    # liquidez mínima
    "volumeMoreThan": 100000,

    # rango beta razonable (evita zombies ultra-defensivos y cohetes memestock)
    "betaMoreThan": 0.5,
    "betaLowerThan": 1.5,

    # Sólo para EE.UU. por ahora, puedes quitar esto si quieres ADR / Europa grande
    "country": "US",

    # Vamos a ir exchange=NASDAQ / NYSE / AMEX manualmente
    "includeAllShareClasses": "false",

    # límite de resultados por llamada
    "limit": 1000,
}


# =========================
# Parámetros de riesgo y límites en la app
# =========================

# Apalancamiento máximo aceptable por defecto (usado como valor inicial del slider)
MAX_NET_DEBT_TO_EBITDA = 3.0


# =========================
# Señales semánticas para text_analysis
# =========================
# Estas listas las puede importar text_analysis.py para juzgar tono en news/transcripts.

NEG_WORDS = [
    "fraud", "probe", "sec investigation", "lawsuit",
    "guidance cut", "misses", "missed estimates",
    "recall", "downgrade", "layoffs", "restructuring",
    "accounting irregularities", "restatement",
    "whistleblower", "governance issue", "antitrust",
    "going concern",
    "liquidity crisis",
    "default risk",
    "covenant breach",
]

POS_WORDS = [
    "beats expectations", "beats estimates", "raises guidance",
    "buyback", "repurchase authorization",
    "dividend increase", "contract win",
    "price target raised", "accretive acquisition",
    "strong demand", "record revenue", "margin expansion",
]

CRITICAL_RISK_TERMS = [
    "sec investigation", "fraud", "accounting irregularities",
    "liquidity crisis", "going concern", "default risk",
    "covenant breach"
]
