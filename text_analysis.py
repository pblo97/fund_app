# text_analysis.py
#
# Análisis cualitativo:
# - insiders (buy / sell / neutral)
# - sentimiento de noticias (pos / neutral / neg)
# - riesgo visible
# - por qué importa (moat, beta, sector)
# - resumen corto de la última earnings call


from typing import List, Dict, Any, Optional
from config import NEG_WORDS, POS_WORDS, CRITICAL_RISK_TERMS, MAX_NET_DEBT_TO_EBITDA


def summarize_insiders(insider_trades: List[Dict[str, Any]]) -> str:
    """
    insider_trades: lista de transacciones de insiders.
    Heurística:
    - contar buys vs sells recientes
    """
    if not insider_trades:
        return "neutral"

    buys = 0
    sells = 0
    for t in insider_trades:
        ttype = (t.get("transactionType") or "").lower()
        if "buy" in ttype:
            buys += 1
        elif "sell" in ttype:
            sells += 1

    if buys > sells:
        return "buy"
    if sells > buys:
        return "sell"
    return "neutral"


def summarize_news_sentiment(news_list: List[Dict[str, Any]]) -> (str, str):
    """
    Retorna (flag, reason)
    flag = "pos" / "neutral" / "neg"
    reason = string corta con la principal señal.
    """
    if not news_list:
        return ("neutral", "sin noticias recientes")

    score = 0
    example_note = "tono mixto/sectorial"

    for item in news_list[:10]:
        title = item.get("title") or ""
        text = item.get("text") or ""
        blob = (title + " " + text).lower()

        hit_pos = any(w in blob for w in POS_WORDS)
        hit_neg = any(w in blob for w in NEG_WORDS)

        if hit_pos and not hit_neg:
            score += 1
            example_note = f"positivo: {title[:100]}"
        elif hit_neg and not hit_pos:
            score -= 1
            example_note = f"negativo: {title[:100]}"

    if score > 0:
        return ("pos", example_note)
    if score < 0:
        return ("neg", example_note)
    return ("neutral", example_note)


def summarize_transcript(transcripts: List[Dict[str, Any]]) -> Optional[str]:
    """
    transcripts: respuesta cruda de get_earnings_call_transcript.
    Suele traer: symbol, quarter, year, content (texto largo).
    Heurística rápida con palabras clave.
    """
    if not transcripts:
        return None
    latest = transcripts[0]
    content = (latest.get("content") or "").lower()

    points = []
    if "guidance" in content:
        points.append("hablaron de guidance")
    if "margin" in content or "margins" in content:
        points.append("comentaron márgenes")
    if "demand" in content:
        points.append("mencionaron demanda")
    if "headwind" in content or "headwinds" in content:
        points.append("advirtieron vientos en contra")
    if "ai" in content:
        points.append("mencionaron IA")

    if not points:
        return "sin señales clave detectables en la última call"
    return ", ".join(points)


def infer_core_risk(
    net_debt_to_ebitda: Optional[float],
    sentiment_flag: Optional[str],
    sentiment_reason: Optional[str]
) -> str:
    risks = []

    if net_debt_to_ebitda is not None and net_debt_to_ebitda > MAX_NET_DEBT_TO_EBITDA:
        risks.append("apalancamiento elevado")

    if sentiment_flag == "neg":
        risks.append("sentimiento de prensa negativo")

    if sentiment_reason:
        low = sentiment_reason.lower()
        if any(term in low for term in CRITICAL_RISK_TERMS):
            risks.append("riesgo regulatorio/contable serio")

    if not risks:
        return "riesgo principal no crítico visible"
    return "; ".join(risks)


def infer_why_it_matters(
    sector: Optional[str],
    industry: Optional[str],
    moat_flag: Optional[str],
    beta: Optional[float]
) -> str:
    """
    Texto corto que explica por qué podría importar este negocio:
    moat, posicionamiento, ciclicidad/defensividad.
    """
    bullets = []

    if moat_flag == "fuerte":
        bullets.append("ventaja competitiva defendible")
    elif moat_flag == "media":
        bullets.append("posicionamiento competitivo decente")
    else:
        bullets.append("moat limitado")

    if beta is not None:
        if beta < 0.8:
            bullets.append("perfil defensivo")
        elif beta > 1.2:
            bullets.append("exposición cíclica/de crecimiento")

    if sector:
        bullets.append(f"sector {sector}")
    if industry:
        bullets.append(f"industria {industry}")

    return ", ".join(bullets)
