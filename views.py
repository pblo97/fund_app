from typing import Dict, Any

def build_detail_view(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Devuelve un dict organizado por secciones para mostrar en Streamlit.
    """
    return {
        "Resumen clave": {
            "Ticker": snapshot.get("ticker"),
            "Nombre": snapshot.get("companyName"),
            "Sector / Industria": f"{snapshot.get('sector')} / {snapshot.get('industry')}",
            "Beta": snapshot.get("beta"),
            "MarketCap": snapshot.get("marketCap"),
        },
        "Calidad financiera": {
            "Altman Z": snapshot.get("altmanZScore"),
            "Piotroski": snapshot.get("piotroskiScore"),
            "Net Debt / EBITDA": snapshot.get("net_debt_to_ebitda_last"),
            "Leverage OK": snapshot.get("leverage_ok"),
            "ROIC": snapshot.get("roic_last"),
            "Margen Op": snapshot.get("operating_margin_last"),
            "Margen FCF": snapshot.get("fcf_margin_last"),
        },
        "Crecimiento / Compounder": {
            "CAGR Revenue 3y": snapshot.get("rev_CAGR_3y"),
            "CAGR OCF 3y": snapshot.get("ocf_CAGR_3y"),
            "High Growth Flag": snapshot.get("high_growth_flag"),
            "Slope FCF/acc 5y": snapshot.get("fcf_per_share_slope_5y"),
            "Buyback %5y": snapshot.get("buyback_pct_5y"),
            "Compounder?": snapshot.get("is_quality_compounder"),
        },
        "Narrativa / Riesgo": {
            "Why it matters": snapshot.get("why_it_matters"),
            "Core Risk": snapshot.get("core_risk_note"),
            "Insiders": f"{snapshot.get('insider_signal')} - {snapshot.get('insider_note')}",
            "News": f"{snapshot.get('news_sentiment')} - {snapshot.get('news_note')}",
            "Transcript": snapshot.get("transcript_summary"),
        },
        "Valuación forward": {
            "Expected CAGR": snapshot.get("expected_return_cagr"),
            "Nota Valuación": snapshot.get("valuation_note"),
        },
    }
