def compute_core_financial_metrics(
    symbol,
    profile,
    ratios_hist,
    income_hist,
    balance_hist,
    cash_hist,
):
    # profile normalmente viene como lista con un dict.
    p0 = profile[0] if isinstance(profile, list) and profile else {}

    return {
        "ticker": symbol,
        "name": p0.get("companyName") or p0.get("companyName") or p0.get("companyName") or symbol,
        "sector": p0.get("sector"),
        "industry": p0.get("industry"),
        "marketCap": p0.get("mktCap") or p0.get("marketCap"),
        "beta": p0.get("beta"),
        "business_summary": p0.get("description", ""),

        "netDebt_to_EBITDA": None,
        "moat_flag": "—",

        "altmanZScore": None,
        "piotroskiScore": None,
        "revenueGrowth": None,
        "operatingCashFlowGrowth": None,
        "freeCashFlowGrowth": None,
        "debtGrowth": None,
        "rev_CAGR_5y": None,
        "rev_CAGR_3y": None,
        "ocf_CAGR_5y": None,
        "ocf_CAGR_3y": None,

        "years": [],
        "fcf_per_share_hist": [],
        "shares_hist": [],
        "net_debt_hist": [],
    }
