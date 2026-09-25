# -*- coding: utf-8 -*-
"""
smallcap_opening_advance_decline.py
===================================
Daily OPENING advance-decline breadth for Nifty Smallcap 100 & Nifty Smallcap 250,
computed from constituents, for every trading day in the dataset's range.

Breadth is measured at the OPEN: opening_change = (09:30 candle OPEN - prev_close)/prev_close*100
  advances = opening_change > 0 ; declines < 0 ; unchanged == 0 (kept in denominator only).

MEMBERSHIP (accuracy caveat): official CURRENT NSE constituent lists
(data/nifty_smallcap{100,250}_constituents_current.csv, fetched from NSE index archives).
This is CURRENT membership applied across ALL dates -> SURVIVORSHIP-BIASED / not point-in-time
(NSE's public endpoint serves only the current list; dated factsheets would be needed for true
point-in-time). Output is labelled accordingly.

PREV-CLOSE convention: VWAP of the previous trading day's last two candles (15:00 & 15:15),
tp=(H+L+C)/3 volume-weighted — matches prepare_data.py's locked-baseline definition.

OPEN candle: 09:30 candle open (HM 570), i.e. one candle AFTER the 09:15 opening-auction candle.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
DATA = rb.REPO / "data" if hasattr(rb, "REPO") else Path(__file__).resolve().parent.parent / "data"
OUTDIR = rb.RESULTS / "smallcap_breadth"
HM_930, HM_1500, HM_1515 = 570, 900, 915
SESSION_HMS = list(range(555, 916, 15))
INDICES = {
    "100": {"csv": DATA / "nifty_smallcap100_constituents_current.csv",
            "ohlc": DATA / "cnxsmallcap100_15min_ohlc.csv", "expected": 100},
    "250": {"csv": DATA / "nifty_smallcap250_constituents_current.csv",
            "ohlc": DATA / "cnxsmallcap250_15min_ohlc.csv", "expected": 250},
}


def opening_change_series(sym):
    """Per-symbol Series (index=date) of opening_change % = (09:30 open - prev VWAP close)/prev*100."""
    pq = rb.MASTER_DIR / f"{sym}.parquet"
    if not pq.exists():
        return None
    raw = pd.read_parquet(pq)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
    raw["date"] = raw["timestamp"].dt.date
    raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
    raw = raw[raw["hm"].isin(SESSION_HMS)]
    if raw.empty:
        return None
    # 09:30 candle open
    o930 = (raw[raw["hm"] == HM_930].drop_duplicates("date", keep="last")
            .set_index("date")["open"])
    # prev-day VWAP close (last two candles, volume-weighted typical price) — prepare_data convention
    l2 = raw[raw["hm"].isin([HM_1500, HM_1515])].copy()
    l2["tp"] = (l2["high"] + l2["low"] + l2["close"]) / 3.0
    l2["tpv"] = l2["tp"] * l2["volume"]
    g = l2.groupby("date").agg(tpv=("tpv", "sum"), vol=("volume", "sum"))
    vwap = (g["tpv"] / g["vol"]).where(g["vol"] > 0, np.nan).sort_index()
    prev = vwap.shift(1)                                  # previous trading day's VWAP close
    oc = (o930 - prev) / prev * 100.0
    return oc.dropna()


def index_daily_close(ohlc_csv):
    df = pd.read_csv(ohlc_csv)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    df["date"] = ts.dt.date; df["hm"] = ts.dt.hour * 60 + ts.dt.minute
    df = df[df["hm"].isin(SESSION_HMS)]
    # daily close = close of last session candle each day (prefer 15:15)
    last = df.sort_values("hm").drop_duplicates("date", keep="last").set_index("date")["close"]
    return last.sort_index()


def build_breadth(members, all_oc):
    """members: list of symbols; all_oc: dict sym->Series. Returns daily breadth DataFrame."""
    cols = {s: all_oc[s] for s in members if s in all_oc and all_oc[s] is not None and len(all_oc[s])}
    M = pd.DataFrame(cols)                                # rows=date, cols=symbol (opening_change)
    M = M.sort_index()
    adv = (M > 0).sum(axis=1)
    dec = (M < 0).sum(axis=1)
    unch = (M == 0).sum(axis=1)
    n = M.notna().sum(axis=1)                             # advances+declines+unchanged
    with np.errstate(divide="ignore", invalid="ignore"):
        ad_ratio = adv / dec.replace(0, np.nan)
    out = pd.DataFrame({
        "date": M.index, "advances": adv.values, "declines": dec.values, "unchanged": unch.values,
        "n_with_data": n.values,
        "advance_pct": (adv / n * 100).round(2).values,
        "decline_pct": (dec / n * 100).round(2).values,
        "ad_ratio": ad_ratio.round(3).values,
        "ad_difference": (adv - dec).values,
    })
    out["zero_declines_flag"] = np.where(dec.values == 0, "no_declines", "")
    return out.reset_index(drop=True)


def summarise(name, bdf, expected):
    full_cov = int((bdf["n_with_data"] >= expected).sum())
    below = int((bdf["n_with_data"] < expected).sum())
    return {
        "index": f"Nifty Smallcap {name}",
        "trading_days": len(bdf),
        "expected_constituents": expected,
        "avg_with_usable_data": round(float(bdf["n_with_data"].mean()), 1),
        "min_with_data": int(bdf["n_with_data"].min()),
        "avg_advance_pct": round(float(bdf["advance_pct"].mean()), 2),
        "avg_decline_pct": round(float(bdf["decline_pct"].mean()), 2),
        "strong_open_days_adv_gt70": int((bdf["advance_pct"] > 70).sum()),
        "weak_open_days_dec_gt70": int((bdf["decline_pct"] > 70).sum()),
        "days_full_coverage": full_cov,
        "days_below_full_coverage": below,
        "pct_days_below_full": round(below / len(bdf) * 100, 1),
    }


def chart(name, bdf, idx_close, path):
    d = bdf.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["cum_ad"] = d["ad_difference"].cumsum()
    ic = idx_close.copy(); ic.index = pd.to_datetime(ic.index)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(d["date"], d["cum_ad"], color="#1f77b4", lw=1.4, label="cumulative opening A/D (Σ adv−dec)")
    ax1.set_ylabel("cumulative opening advance−decline", color="#1f77b4")
    ax1.axhline(0, color="#888", lw=.7, ls=":")
    ax2 = ax1.twinx()
    ax2.plot(ic.index, ic.values, color="#d62728", lw=1.2, alpha=.75, label="index level (close)")
    ax2.set_ylabel(f"Nifty Smallcap {name} level", color="#d62728")
    ax1.set_title(f"Nifty Smallcap {name}: cumulative OPENING breadth vs index level "
                  f"(CURRENT membership — survivorship-biased)")
    ax1.grid(alpha=.25)
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # membership (current, official NSE)
    members = {}
    all_syms = set()
    for k, cfg in INDICES.items():
        m = pd.read_csv(cfg["csv"])["Symbol"].dropna().astype(str).str.strip().tolist()
        members[k] = m; all_syms.update(m)
    print(f"Members: Smallcap100={len(members['100'])}, Smallcap250={len(members['250'])}, "
          f"distinct={len(all_syms)}")

    # per-symbol opening_change (once per symbol; Smallcap100 ⊂ Smallcap250)
    print("Computing per-constituent opening_change …")
    all_oc = {}
    have = 0
    for i, s in enumerate(sorted(all_syms), 1):
        oc = opening_change_series(s)
        all_oc[s] = oc
        if oc is not None and len(oc):
            have += 1
        if i % 50 == 0:
            print(f"  …{i}/{len(all_syms)} ({have} with data)")
    print(f"  constituents with usable candle data: {have}/{len(all_syms)}")

    results, summaries = {}, []
    for k, cfg in INDICES.items():
        bdf = build_breadth(members[k], all_oc)
        results[k] = bdf
        summaries.append(summarise(k, bdf, cfg["expected"]))
        idx_close = index_daily_close(cfg["ohlc"])
        chart(k, bdf, idx_close, OUTDIR / f"smallcap{k}_cum_opening_ad.png")
        bdf.to_csv(OUTDIR / f"smallcap{k}_opening_ad_daily.csv", index=False)

    summ = pd.DataFrame(summaries)
    dmin = min(pd.to_datetime(results["250"]["date"]).min(), pd.to_datetime(results["100"]["date"]).min())
    dmax = max(pd.to_datetime(results["250"]["date"]).max(), pd.to_datetime(results["100"]["date"]).max())

    with pd.ExcelWriter(OUTDIR / "smallcap_opening_advance_decline.xlsx", engine="openpyxl") as w:
        results["100"].to_excel(w, sheet_name="smallcap100_daily", index=False)
        results["250"].to_excel(w, sheet_name="smallcap250_daily", index=False)
        summ.to_excel(w, sheet_name="summary", index=False)
        pd.DataFrame({"note": [
            "MEMBERSHIP: official CURRENT NSE constituents applied to ALL dates -> SURVIVORSHIP-BIASED,"
            " NOT point-in-time. NSE public endpoint serves only current membership.",
            "OPENING breadth: opening_change = (09:30 candle OPEN - prev_close)/prev_close*100.",
            "PREV_CLOSE: VWAP of previous day's 15:00 & 15:15 candles (prepare_data.py convention).",
            "'unchanged' (exactly 0.00%) kept in n_with_data denominator, excluded from adv/dec counts.",
            f"Date range: {dmin.date()} -> {dmax.date()}.",
        ]}).to_excel(w, sheet_name="notes", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 80)
    print("DAILY OPENING ADVANCE-DECLINE — Nifty Smallcap 100 & 250")
    print("  MEMBERSHIP: CURRENT official NSE lists (survivorship-biased, not point-in-time)")
    print(f"  Date range: {dmin.date()} -> {dmax.date()}")
    print("=" * 80)
    print("\n--- SUMMARY ---")
    print(summ.to_string(index=False))
    for k in ("100", "250"):
        print(f"\n--- Smallcap {k}: last 5 rows ---")
        print(results[k].tail(5).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
