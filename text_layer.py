from typing import Dict, Any
from config import POS_WORDS, NEG_WORDS, CRITICAL_RISK_TERMS, MAX_NET_DEBT_TO_EBITDA
from fmp_api import get_insider_trading, get_company_news, get_earnings_call_transcript

def summarize_insiders(insider_rows: list[dict]) -> tuple[str, str]:
    """
    Retorna (signal, note)
    signal ∈ {"buy","sell","neutral"}
    note = breve explicación
    Heurística simple: más compras $ que ventas $ en últimos filings => "buy"
    """
    total_buy = 0.0
    total_sell = 0.0
    for row in insider_rows:
        # estos campos exactos dependen de FMP, ajustar acá
        ttype = str(row.get("transactionType","")).lower()
        val = float(row.get("value", 0) or 0)
        if "buy" in ttype:
            total_buy += val
        elif "sell" in ttype:
            total_sell += val

    if total_buy > total_sell * 1.2:
        return "buy", f"Compras insider relevantes (${total_buy:,.0f} vs ${total_sell:,.0f} en ventas)."
    if total_sell > total_buy * 1.2:
        return "sell", f"Ventas insider relevantes (${total_sell:,.0f} vs ${total_buy:,.0f} en compras)."
    return "neutral", "Flujo insider balanceado / poco significativo."

def summarize_news_sentiment(news_rows: list[dict]) -> tuple[str, str]:
    """
    Analiza titulares y cuerpo buscando palabras positivas / negativas.
    Retorna (sentiment, note)
    sentiment ∈ {"pos","neutral","neg"}
    """
    score = 0
    hits_pos = []
    hits_neg = []
    for art in news_rows:
        text_blob = " ".join([
            str(art.get("title","")),
            str(art.get("text","")),
            str(art.get("content","")),
        ]).lower()

        for w in POS_WORDS:
            if w.lower() in text_blob:
                score += 1
                hits_pos.append(w)
        for w in NEG_WORDS:
            if w.lower() in text_blob:
                score -= 1
                hits_neg.append(w)

    if score > 0:
        return "pos", f"Prensa con sesgo positivo ({', '.join(set(hits_pos[:3]))})."
    if score < 0:
        return "neg", f"Prensa con alertas ({', '.join(set(hits_neg[:3]))})."
    return "neutral", "Sentimiento de prensa neutro / mixto."

def summarize_transcript(transcript_rows: list[dict]) -> str:
    """
    Retorna una frase corta con temas clave.
    Buscamos mentions de guidance, margin, headwind, AI...
    """
    if not transcript_rows:
        return "No transcript reciente disponible."
    blob = " ".join([str(r.get("content","")) for r in transcript_rows]).lower()

    tags = []
    if "guidance" in blob:
        tags.append("habló guidance")
    if "margin" in blob or "margins" in blob:
        tags.append("márgenes")
    if "headwind" in blob or "slowdown" in blob:
        tags.append("vientos en contra")
    if "ai" in blob or "artificial intelligence" in blob:
        tags.append("IA")

    if not tags:
        return "Transcript sin señales fuertes."
    return "Transcript menciona: " + ", ".join(tags)

def infer_core_risk(snap: Dict[str, Any],
                    insider_signal: str,
                    news_sentiment: str,
                    transcript_summary: str) -> str:
    """
    Combina:
    - Apalancamiento
    - Sentimiento negativo
    - Palabras críticas
    Devuelve nota corta de riesgo principal.
    """
    # 1. deuda
    risk_bits = []
    lev = snap.get("net_debt_to_ebitda_last")
    if lev is not None and lev > MAX_NET_DEBT_TO_EBITDA:
        risk_bits.append("apalancamiento elevado")

    # 2. prensa negativa
    if news_sentiment == "neg":
        risk_bits.append("prensa negativa reciente")

    # 3. transcript con headwinds
    if "vientos en contra" in transcript_summary:
        risk_bits.append("management advierte headwinds")

    # 4. términos críticos explícitos
    blob_full = " ".join([
        str(transcript_summary).lower(),
    ])
    for term in CRITICAL_RISK_TERMS:
        if term.lower() in blob_full:
            risk_bits.append("riesgo regulatorio/contable serio")
            break

    if not risk_bits:
        return "Sin riesgo crítico evidente a primera vista."
    return "; ".join(risk_bits) + "."

def infer_why_it_matters(snap: Dict[str, Any],
                         insider_signal: str,
                         news_sentiment: str) -> str:
    """
    Mini pitch de por qué la empresa importa.
    Usa sector, beta, compounder flag, growth flag.
    """
    sector = snap.get("sector")
    industry = snap.get("industry")
    beta = snap.get("beta")
    hg = snap.get("high_growth_flag")
    comp = snap.get("is_quality_compounder")

    bits = []
    if comp:
        bits.append("compounder disciplinada (FCF/acción ↑ y recompras)")
    if hg:
        bits.append("crecimiento secular alto (~15%+ CAGR)")
    if beta is not None:
        if beta < 1:
            bits.append("beta defensiva (<1)")
        else:
            bits.append("beta agresiva (>1)")
    if sector:
        bits.append(f"exposición sector {sector}")
    if industry:
        bits.append(f"nicho {industry}")

    if not bits:
        return "Empresa estable sin rasgos diferenciales fuertes aún."
    return "; ".join(bits)
    

def fetch_text_signals_for_snapshot(snap: Dict[str, Any]) -> Dict[str, Any]:
    ticker = snap["ticker"]

    insiders = get_insider_trading(ticker)
    news = get_company_news(ticker)
    transcript = get_earnings_call_transcript(ticker)

    insider_signal, insider_note = summarize_insiders(insiders)
    news_sentiment, news_note = summarize_news_sentiment(news)
    transcript_summary = summarize_transcript(transcript)

    core_risk = infer_core_risk(
        snap,
        insider_signal=insider_signal,
        news_sentiment=news_sentiment,
        transcript_summary=transcript_summary,
    )
    why_matters = infer_why_it_matters(
        snap,
        insider_signal=insider_signal,
        news_sentiment=news_sentiment,
    )

    return {
        "insider_signal": insider_signal,
        "insider_note": insider_note,
        "news_sentiment": news_sentiment,
        "news_note": news_note,
        "transcript_summary": transcript_summary,
        "core_risk_note": core_risk,
        "why_it_matters": why_matters,
    }
