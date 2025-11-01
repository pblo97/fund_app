# orchestrator.py
#
# Flujo:
# 1. build_universe() -> shortlist de tickers grandes y sanos
#    (usa screener + filtros básicos en memoria, sin NLP ni news)
#
# 2. build_company_core_snapshot(ticker) -> métricas fundamentales core
#
# 3. build_full_snapshot() -> corre todo eso sobre la shortlist y devuelve
#    un array de dicts listo para usar en la tabla del Tab1.
#
# 4. enrich_company_snapshot(snapshot) -> agrega insiders, news, transcript
#    SOLO para un ticker puntual (Tab2).
#

from typing import List, Dict, Any
import math

from fmp_api import (
    run_screener_for_exchange,
    get_profile,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_ratios,
    get_insider_trading,
    get_news,
    get_earnings_call_transcript,
)

from metrics import compute_core_financial_metrics
from config import MAX_NET_DEBT_TO_EBITDA

# -----------------------
# Helpers internos
# -----------------------

def _is_large_cap(row: Dict[str, Any], min_mktcap: float = 10_000_000_000) -> bool:
    """
    Filtro de tamaño: nos quedamos con large caps (>=10B).
    """
    mc = row.get("marketCap") or row.get("mktCap") or row.get("marketCapIntraday")
    try:
        return float(mc) >= float(min_mktcap)
    except Exception:
        return False


def _merge_scores_and_growth(
    base_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    ACA ES DONDE EN LA VERSIÓN AVANZADA METERÍAMOS:
    - Altman Z
    - Piotroski
    - Growth
    usando endpoints batch.

    Para no quedarnos pegados aún (y para ver la app viva),
    vamos a poner placeholders seguros en vez de llamar APIs extra.
    """

    enriched = []
    for row in base_rows:
        r = dict(row)  # copia superficial

        # placeholders razonables hasta que afinemos llamadas reales:
        # los ponemos en None en vez de inventar números, para que la UI muestre "—"
        r.setdefault("altmanZScore", None)
        r.setdefault("piotroskiScore", None)

        # crecimientos "puntuales":
        r.setdefault("revenueGrowth", None)
        r.setdefault("operatingCashFlowGrowth", None)
        r.setdefault("freeCashFlowGrowth", None)
        r.setdefault("debtGrowth", None)

        # crecimientos compuestos (CAGR 3y/5y). Igual, None placeholder.
        r.setdefault("rev_CAGR_5y", None)
        r.setdefault("rev_CAGR_3y", None)
        r.setdefault("ocf_CAGR_5y", None)
        r.setdefault("ocf_CAGR_3y", None)

        enriched.append(r)

    return enriched


def _quality_filter_final(d: Dict[str, Any]) -> bool:
    """
    Esta función decide si el ticker pasa al shortlist final.
    Con placeholders, hacemos un filtro mínimo:
    - Debe tener netDebt_to_EBITDA <= MAX_NET_DEBT_TO_EBITDA (si está disponible)
    - Debe ser large cap (ya filtrado antes)
    Más adelante, cuando tengamos Altman Z, Piotroski, growth etc.,
    afinamos acá.
    """

    nde = d.get("netDebt_to_EBITDA")
    if nde is not None:
        try:
            if float(nde) > float(MAX_NET_DEBT_TO_EBITDA):
                return False
        except Exception:
            pass

    return True


# -----------------------
# Paso 1: universo inicial
# -----------------------

def build_universe() -> List[str]:
    """
    Descarga screener de NASDAQ y NYSE,
    deduplica tickers,
    filtra large caps >=10B,
    y devuelve la lista de tickers candidata.
    """

    base_list: List[Dict[str, Any]] = []

    for exch in ["NASDAQ", "NYSE"]:
        try:
            chunk = run_screener_for_exchange(exch)
            if isinstance(chunk, list):
                base_list.extend(chunk)
        except Exception:
            # si falla un exchange igual seguimos con el otro
            continue

    # dedupe
    seen: Dict[str, Dict[str, Any]] = {}
    for row in base_list:
        sym = row.get("symbol")
        if sym and sym not in seen:
            seen[sym] = row

    # large caps only
    large_caps = [
        sym for (sym, data) in seen.items()
        if _is_large_cap(data, min_mktcap=10_000_000_000)
    ]

    return large_caps


# -----------------------
# Paso 2: métricas core por empresa
# -----------------------

def build_company_core_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Para 1 ticker:
    - baja profile + estados financieros + ratios
    - arma snapshot base con compute_core_financial_metrics()
    """

    profile = get_profile(ticker)
    income_hist = get_income_statement(ticker)
    balance_hist = get_balance_sheet(ticker)
    cash_hist = get_cash_flow(ticker)
    ratios_hist = get_ratios(ticker)

    # validación mínima para no romper más adelante
    if (
        not isinstance(income_hist, list) or len(income_hist) < 2 or
        not isinstance(balance_hist, list) or len(balance_hist) < 2 or
        not isinstance(cash_hist, list) or len(cash_hist) < 2
    ):
        raise ValueError("historial financiero insuficiente")

    base_metrics = compute_core_financial_metrics(
        ticker=ticker,
        profile=profile,
        ratios_hist=ratios_hist,
        income_hist=income_hist,
        balance_hist=balance_hist,
        cash_hist=cash_hist,
    )

    return base_metrics


# -----------------------
# Paso 3: shortlist completo
# -----------------------

def build_full_snapshot() -> List[Dict[str, Any]]:
    """
    1. arma la lista de tickers grandes
    2. para cada ticker, crea snapshot core
    3. agrega placeholders de growth/scores (por ahora)
    4. aplica quality_filter_final
    5. retorna la lista final
    """

    final_rows: List[Dict[str, Any]] = []

    tickers = build_universe()

    for tkr in tickers:
        try:
            snap_core = build_company_core_snapshot(tkr)
            final_rows.append(snap_core)
        except Exception:
            # si una empresa no tiene historia limpia, saltamos
            continue

    # Enriquecemos con placeholders de scores/growth
    final_rows = _merge_scores_and_growth(final_rows)

    # Filtro final de calidad (apalancamiento, etc.)
    filtered_rows = [row for row in final_rows if _quality_filter_final(row)]

    return filtered_rows


# -----------------------
# Paso 4: enrich para detalle
# -----------------------

def enrich_company_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    SOLO cuando el usuario abre el Tab2 para ver un ticker.
    Agregamos insiders, news, transcript y hacemos inferencias cualitativas.
    """

    ticker = snapshot.get("ticker")

    # llamadas adicionales (estas sí gastan créditos por ticker)
    try:
        insider_trades = get_insider_trading(ticker)
    except Exception:
        insider_trades = []

    try:
        news_list = get_news(ticker)
    except Exception:
        news_list = []

    try:
        transcripts = get_earnings_call_transcript(ticker)
    except Exception:
        transcripts = []

    # ---- análisis cualitativo simple sin LLM extra ----
    # insider_signal básico:
    # si vemos compras ("Buy") de directores en los últimos registros -> "buy"
    insider_signal = "neutral"
    for it in insider_trades[:10]:
        act = (it.get("transactionType") or "").lower()
        if "buy" in act:
            insider_signal = "buy"
            break
        if "sell" in act and insider_signal != "buy":
            insider_signal = "sell"

    # sentimiento noticias muy naive:
    sentiment_flag = "neutral"
    sentiment_reason = "tono mixto"
    bad_hits = 0
    good_hits = 0
    from config import NEG_WORDS, POS_WORDS  # esto es seguro aquí (no circular)

    for nw in news_list[:20]:
        headline = (nw.get("title") or "").lower() + " " + (nw.get("text") or "").lower()
        if any(w in headline for w in NEG_WORDS):
            bad_hits += 1
        if any(w in headline for w in POS_WORDS):
            good_hits += 1

    if bad_hits > good_hits and bad_hits > 0:
        sentiment_flag = "neg"
        sentiment_reason = "noticias con banderas rojas recientes"
    elif good_hits > bad_hits and good_hits > 0:
        sentiment_flag = "pos"
        sentiment_reason = "tono mayormente constructivo en prensa"

    # transcript_summary súper básico:
    transcript_summary = "sin transcript reciente"
    if transcripts:
        first = transcripts[0]
        # muchos endpoints de FMP devuelven 'content' o 'qa' o 'transcript'
        transcript_summary = first.get("content") or first.get("transcript") or "call sin resumen parseable"

    # riesgo cualitativo básico:
    core_risk_note = "apalancamiento controlado"
    nde = snapshot.get("netDebt_to_EBITDA")
    try:
        if nde is not None and float(nde) > float(MAX_NET_DEBT_TO_EBITDA):
            core_risk_note = "apalancamiento elevado / riesgo de liquidez si el mercado se aprieta"
    except Exception:
        pass

    # why_it_matters placeholder:
    why_it_matters = (
        f"Negocio en {snapshot.get('sector','?')} / {snapshot.get('industry','?')} "
        f"con perfil de caja y recompras que podrían sostener composición de valor."
    )

    # merge final
    snapshot["insider_signal"] = insider_signal
    snapshot["sentiment_flag"] = sentiment_flag
    snapshot["sentiment_reason"] = sentiment_reason
    snapshot["transcript_summary"] = transcript_summary
    snapshot["core_risk_note"] = core_risk_note
    snapshot["why_it_matters"] = why_it_matters

    return snapshot
