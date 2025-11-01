# config.py

# === CONFIGURACIÓN GENERAL ===

# Pon tu API key de FinancialModelingPrep acá o léela de un env var
FMP_API_KEY = "TU_API_KEY_AQUI"

# Parámetros base del screener para definir el universo inicial "invertible"
SCREENER_PARAMS = {
    "isEtf": "false",
    "isFund": "false",
    "isActivelyTrading": "true",
    "priceMoreThan": 1,
    "marketCapMoreThan": 1_000_000_000,  # >= 1B USD
    "country": "US",
    # Vamos a correr esto para NASDAQ y NYSE
    "betaMoreThan": 0.5,
    "betaLowerThan": 1.5,
    "volumeMoreThan": 100000,
    "includeAllShareClasses": "false",
    "limit": 1000,
}

# Corte mínimo de calidad para ROE
MIN_ROE_TTM = 0.12  # 12%

# Hurdle default de retorno esperado anual
EXPECTED_RETURN_HURDLE = 0.15  # 15%

# Máximo apalancamiento aceptable por defecto (Net Debt / EBITDA)
MAX_NET_DEBT_TO_EBITDA = 3.0

# Palabras clave para análisis de sentimiento de noticias
NEG_WORDS = [
    "fraud", "probe", "sec investigation", "lawsuit",
    "guidance cut", "misses", "missed estimates",
    "recall", "downgrade", "layoffs", "restructuring",
    "accounting irregularities", "restatement",
    "whistleblower", "governance issue", "antitrust"
]

POS_WORDS = [
    "beats expectations", "beats estimates", "raises guidance",
    "buyback", "repurchase authorization",
    "dividend increase", "contract win",
    "price target raised", "accretive acquisition",
    "strong demand", "record revenue", "margin expansion"
]

# Riesgos críticos que queremos marcar fuerte
CRITICAL_RISK_TERMS = [
    "sec investigation", "fraud", "accounting irregularities",
    "liquidity crisis", "going concern", "default risk",
    "covenant breach"
]
