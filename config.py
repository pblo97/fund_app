import os

# ============================
# Credenciales / API
# ============================
FMP_API_KEY = os.getenv("FMP_API_KEY", "YOUR_FMP_KEY_HERE")

# ============================
# Parámetros de screening
# ============================
EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]

MIN_MARKET_CAP = 5_000_000_000  # 5B para evitar microcaps
MAX_TICKERS_FOR_TEST = 40       # <- límite duro solicitado

ALTMAN_MIN = 3.0
PIOTROSKI_MIN = 7

MAX_NET_DEBT_TO_EBITDA = 2.0    # tolerancia de apalancamiento "aceptable"

HIGH_GROWTH_CAGR_THRESHOLD = 0.15  # 15% anual para marcar high_growth_flag

# ============================
# Parámetros de proyección
# ============================
TARGET_YEARS_FORWARD = 3
DEFAULT_FCF_MULTIPLE = 15  # múltiplo terminal FCF/acción conservador

# ============================
# Palabras para análisis de texto
# (puedes tunearlas después)
# ============================
POS_WORDS = ["strong demand", "record", "outperform", "robust", "margin expansion"]
NEG_WORDS = ["headwind", "slowdown", "regulatory", "investigation", "guidance cut"]

CRITICAL_RISK_TERMS = [
    "fraud",
    "sec investigation",
    "material weakness",
    "liquidity crisis",
    "going concern",
]
