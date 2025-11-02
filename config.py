# config.py
#
# Configuración global de tu app cuant/fundamental.

# --- tu API key de FMP ----
FMP_API_KEY = "TU_API_KEY_FMP_AQUI"

# --- filtro base del screener ---
# Estos parámetros se mandan directo a /stock-screener de FMP.
# Puedes ajustarlos a tu gusto (sector, price > 5, etc.).
#
# OJO: además del screener de FMP, nosotros vamos a filtrar
#     marketCap >= 10B USD en código.
SCREENER_PARAMS = {
    # ejemplos típicos admitidos por /stock-screener:
    # "marketCapMoreThan": "10000000000",  # podrías forzarlo acá también
    "isEtf": "false",
    "isActivelyTrading": "true",
    # "betaMoreThan": "0",   # opcional
    # "volumeMoreThan": "100000",  # opcional
    "limit": 5000,
}

# Si quieres usar este umbral también desde Streamlit:
MAX_NET_DEBT_TO_EBITDA = 3.0  # placeholder hasta que calculemos el real
