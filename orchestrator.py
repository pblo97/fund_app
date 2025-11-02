from __future__ import annotations

from typing import List, Dict, Any
import math
import numpy as np
import pandas as pd

from fmp_api import (
    run_screener_for_exchange,
    get_profile,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_ratios,
    get_cashflow_history,
    get_balance_history,
    get_income_history,
    get_shares_history,
    get_insider_trading,
    get_news,
    get_earnings_call_transcript,
)

from metrics import compute_core_financial_metrics

EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]


def _linear_trend(values) -> float | None:
    s = pd.Series(values).dropna()
    if len(s) < 2:
        return None
    x = np.arange(len(s), dtype=float)
    y = s.astype(float).values
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _safe_div(a, b):
    try:
        if b is None or float(b) == 0:
            return None
        return float(a) / float(b)
    except Exception:
        return None


def _is_large_cap(row: Dict[str, Any], min_mktcap: float = 10_000_000_000) -> bool:
    mc = (
        row.get("marketCap")
        or row.get("mktCap")
        or row.get("marketCapIntraday")
    )
    try:
        return float(mc) >= float(min_mktcap)
    except Exception:
        return False


def _fetch_histories(symbol: str) -> pd.DataFrame:
    cf  = get_cashflow_history(symbol)
    bal = get_balance_history(symbol)
    inc = get_income_history(symbol)
    shr = get_shares_history(symbol)

    hist = (
        pd.merge(cf, shr, on="fiscalDate", how="outer")
          .merge(bal, on="fiscalDate", how="outer")
          .merge(inc, on="fiscalDate", how="outer")
    )

    if "fiscalDate" in hist.columns:
        hist = hist.sort_values("fiscalDate").reset_index(drop=True)
    else:
        hist = hist.reset_index(drop=True)

    hist["fcf"] = hist.get("operatingCashFlow") - hist.get("capitalExpenditure")
    hist["fcf_per_share"] = _safe_div(hist.get("fcf"), hist.get("sharesDiluted"))
    hist["net_debt"] = (
        hist.get("totalDebt") - hist.get("cashAndShortTermInvestments")
    )

    return hist


def _compute_buyback_pct_5y(hist: pd.DataFrame) -> float | None:
    if hist is None or hist.empty or "sharesDiluted" not in hist.columns:
        return None
    shares = hist["sharesDiluted"].dropna()
    if len(shares) < 2:
        return None
    start = float(shares.iloc[0])
    end   = float(shares.iloc[-1])
    if start == 0:
        return None
    return (start - end) / start


def _compute_last_net_debt_to_ebitda(hist: pd.DataFrame) -> float | None:
    if hist is None or hist.empty:
        return None
    usable = hist.dropna(subset=["net_debt", "ebitda"])
    if usable.empty:
        return None
    last = usable.iloc[-1]
    nd = last.get("net_debt")
    eb = last.get("ebitda")
    if eb is None or eb == 0:
        return None
    try:
        return float(nd) / float(eb)
    except Exception:
        return None


def build_universe() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for exch in EXCHANGES:
        block = run_screener_for_exchange(exch)
        if not block:
            continue
        frames.append(pd.DataFrame(block))

    if not frames:
        return pd.DataFrame(
            columns=["symbol", "companyName", "sector", "industry", "marketCap"]
        )

    uni_raw = pd.concat(frames, ignore_index=True)

    if "companyName" not in uni_raw.columns and "name" in uni_raw.columns:
        uni_raw = uni_raw.rename(columns={"name": "companyName"})

    uni_raw = uni_raw.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    mask_large = uni_raw.apply(_is_large_cap, axis=1)
    uni_lc = uni_raw[mask_large].reset_index(drop=True)

    return uni_lc


def build_company_core_snapshot(symbol: str) -> Dict[str, Any] | None:
    profile       = get_profile(symbol)
    income_hist   = get_income_statement(symbol)
    balance_hist  = get_balance_sheet(symbol)
    cash_hist     = get_cash_flow(symbol)
    ratios_hist   = get_ratios(symbol)

    if (
        not isinstance(income_hist, list) or len(income_hist) == 0 or
        not isinstance(balance_hist, list) or len(balance_hist) == 0 or
        not isinstance(cash_hist, list) or len(cash_hist) == 0
    ):
        return None

    core = compute_core_financial_metrics(
        symbol,
        profile,
        ratios_hist,
        income_hist,
        balance_hist,
        cash_hist,
    )
    # core debe (como mínimo) darnos:
    # ticker, name, sector, industry, marketCap, beta, business_summary,
    # netDebt_to_EBITDA, moat_flag,
    # altmanZScore, piotroskiScore,
    # revenueGrowth, operatingCashFlowGrowth, freeCashFlowGrowth, debtGrowth,
    # rev_CAGR_5y, rev_CAGR_3y, ocf_CAGR_5y, ocf_CAGR_3y,
    # years, fcf_per_share_hist, shares_hist, net_debt_hist

    hist = _fetch_histories(symbol)

    fcf_slope_5y            = None
    buyback_pct_5y          = None
    net_debt_to_ebitda_last = None
    hist_years              = core.get("years", [])
    shares_hist             = core.get("shares_hist", [])
    fcf_hist                = core.get("fcf_per_share_hist", [])
    net_debt_hist           = core.get("net_debt_hist", [])

    if hist is not None and not hist.empty:
        if "fcf_per_share" in hist.columns:
            fcf_slope_5y = _linear_trend(hist["fcf_per_share"])

        buyback_pct_5y = _compute_buyback_pct_5y(hist)
        net_debt_to_ebitda_last = _compute_last_net_debt_to_ebitda(hist)

        if "fiscalDate" in hist.columns:
            hist_years = hist["fiscalDate"].astype(str).tolist()
        if "fcf_per_share" in hist.columns:
            fcf_hist = (
                hist["fcf_per_share"]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .tolist()
            )
        if "sharesDiluted" in hist.columns:
            shares_hist = (
                hist["sharesDiluted"]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .tolist()
            )
        if "net_debt" in hist.columns:
            net_debt_hist = (
                hist["net_debt"]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .tolist()
            )

    def _pos(x):
        try:
            return float(x) > 0
        except Exception:
            return False

    flag_fcf_up       = _pos(fcf_slope_5y)
    flag_buybacks     = (buyback_pct_5y is not None and not math.isnan(buyback_pct_5y) and buyback_pct_5y > 0.05)
    flag_net_debt_ok  = (net_debt_to_ebitda_last is not None and (net_debt_to_ebitda_last < 2))

    is_quality_compounder = bool(flag_fcf_up and flag_buybacks and flag_net_debt_ok)

    row: Dict[str, Any] = {
        "ticker":               core.get("ticker", symbol),
        "name":                 core.get("name"),
        "companyName":          core.get("name"),
        "sector":               core.get("sector"),
        "industry":             core.get("industry"),
        "marketCap":            core.get("marketCap"),
        "beta":                 core.get("beta"),
        "business_summary":     core.get("business_summary", ""),
        "netDebt_to_EBITDA":    core.get("netDebt_to_EBITDA"),
        "moat_flag":            core.get("moat_flag", "—"),

        "altmanZScore":             core.get("altmanZScore"),
        "piotroskiScore":           core.get("piotroskiScore"),
        "revenueGrowth":            core.get("revenueGrowth"),
        "operatingCashFlowGrowth":  core.get("operatingCashFlowGrowth"),
        "freeCashFlowGrowth":       core.get("freeCashFlowGrowth"),
        "debtGrowth":               core.get("debtGrowth"),
        "rev_CAGR_5y":              core.get("rev_CAGR_5y"),
        "rev_CAGR_3y":              core.get("rev_CAGR_3y"),
        "ocf_CAGR_5y":              core.get("ocf_CAGR_5y"),
        "ocf_CAGR_3y":              core.get("ocf_CAGR_3y"),

        "fcf_per_share_slope_5y":   fcf_slope_5y,
        "buyback_pct_5y":           buyback_pct_5y,
        "net_debt_to_ebitda_last":  net_debt_to_ebitda_last,
        "is_quality_compounder":    is_quality_compounder,

        "years":                hist_years,
        "fcf_per_share_hist":   fcf_hist,
        "shares_hist":          shares_hist,
        "net_debt_hist":        net_debt_hist,

        "why_it_matters":   "",
        "core_risk_note":   "",
    }

    return row


def build_market_snapshot() -> List[Dict[str, Any]]:
    uni = build_universe()
    rows: List[Dict[str, Any]] = []

    for _, r in uni.iterrows():
        sym = str(r.get("symbol", "")).strip()
        if not sym:
            continue

        snap = build_company_core_snapshot(sym)
        if snap is None:
            continue

        nde_last = snap.get("net_debt_to_ebitda_last")
        try:
            if nde_last is not None and float(nde_last) > 20:
                continue
        except Exception:
            pass

        rows.append(snap)

    return rows


def enrich_company_snapshot(base_row: Dict[str, Any]) -> Dict[str, Any]:
    symbol = base_row.get("ticker") or base_row.get("symbol")
    if not symbol:
        return base_row.copy()

    insider_raw = get_insider_trading(symbol)
    insider_signal = "neutral"
    insider_note = ""
    if isinstance(insider_raw, list) and insider_raw:
        buys = 0.0
        sells = 0.0
        for tx in insider_raw:
            side = (tx.get("transactionType") or "").lower()
            val  = tx.get("value") or 0
            try:
                val = float(val)
            except Exception:
                val = 0.0
            if "buy" in side:
                buys += val
            elif "sell" in side:
                sells += val
        if buys > sells * 1.2:
            insider_signal = "bullish"
        elif sells > buys * 1.2:
            insider_signal = "bearish"
        insider_note = f"Compras insider ≈ {buys:.0f}, ventas insider ≈ {sells:.0f} USD"

    news_raw = get_news(symbol)
    sentiment_flag = "neutral"
    sentiment_reason = "tono mixto/sectorial"
    if isinstance(news_raw, list) and news_raw:
        text_blob = " ".join([
            str(n.get("text") or "") + " " + str(n.get("title") or "")
            for n in news_raw[:10]
        ]).lower()
        if "guidance raised" in text_blob or "beat expectations" in text_blob:
            sentiment_flag = "bullish"
        elif "lawsuit" in text_blob or "investigation" in text_blob:
            sentiment_flag = "bearish"
            sentiment_reason = "riesgo legal/regulatorio"

    transcript_raw = get_earnings_call_transcript(symbol)
    transcript_summary = "Sin señales fuertes en la última call."
    if isinstance(transcript_raw, list) and transcript_raw:
        body = (
            transcript_raw[0].get("content")
            or transcript_raw[0].get("text")
            or ""
        )
        short = str(body).strip().replace("\n", " ")
        if len(short) > 400:
            short = short[:400] + "..."
        if short:
            transcript_summary = short

    enriched = dict(base_row)
    enriched["insider_signal"]     = insider_signal
    enriched["insider_note"]       = insider_note
    enriched["sentiment_flag"]     = sentiment_flag
    enriched["sentiment_reason"]   = sentiment_reason
    enriched["transcript_summary"] = transcript_summary

    return enriched
