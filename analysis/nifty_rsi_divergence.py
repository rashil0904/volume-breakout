# -*- coding: utf-8 -*-
"""
nifty_rsi_divergence.py — STEP 2: detect BULLISH / BEARISH RSI divergences on the hourly
Nifty RSI-14 series built in step 1 (results/nifty_hourly_rsi/nifty_hourly_rsi.csv).
DETECTION ONLY — flags divergences; no trades yet.

DEFINITIONS (two pivots, 1st then 2nd):
  BULLISH : price 2nd LOW  < price 1st LOW  (lower low)  AND rsi 2nd low  > rsi 1st low  (higher low)
            AND rsi_2nd_low  > 30
  BEARISH : price 2nd HIGH > price 1st HIGH (higher high) AND rsi 2nd high < rsi 1st high (lower high)
            AND rsi_2nd_high < 70

SWING POINTS (flag a — THE key choice; default = fractal window):
  method 'pivot' (default): a pivot low = an hourly candle whose LOW is strictly the lowest within
    ±k candles (k = pivot strength, default 2); pivot high symmetric on HIGH. Consecutive pivot lows
    (bullish) / highs (bearish) are compared, provided they're within L=30 candles (flag b: k, L tunable).
  RSI at a price pivot = the time-aligned rsi_14 on the SAME candle (flag c). Filter on the 2nd RSI
  pivot only (flag d). Each consecutive qualifying pivot-pair is flagged (flag e: not deduped).
  Confirming candle = the pivot's fractal confirmation, k candles AFTER the 2nd pivot.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "nifty_hourly_rsi" / "nifty_hourly_rsi.csv"
OUTDIR = rb.RESULTS / "nifty_rsi_divergence"
K = 2                 # pivot strength (±k candles)
L = 30                # max separation between the two pivots (hourly candles)
BULL_FLOOR, BEAR_CEIL = 30.0, 70.0
METHOD = "pivot"      # 'pivot' (fractal) — default


def fractals(vals, k, kind):
    """Strict fractal pivots: True where vals[i] is strictly the extreme within [i-k, i+k]."""
    n = len(vals); piv = np.zeros(n, bool)
    for i in range(k, n - k):
        seg = vals[i - k:i + k + 1]
        c = vals[i]
        others = np.concatenate([seg[:k], seg[k + 1:]])
        if kind == "low" and np.all(c < others):
            piv[i] = True
        elif kind == "high" and np.all(c > others):
            piv[i] = True
    return piv


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    H = pd.read_csv(SRC, parse_dates=["date"])
    H["date"] = H["date"].dt.date
    n = len(H)
    low = H["low"].values.astype(float); high = H["high"].values.astype(float)
    close = H["close"].values.astype(float); rsi = H["rsi_14"].values.astype(float)
    dt = H["date"].values; bk = H["hour_bucket"].values

    piv_low = fractals(low, K, "low"); piv_high = fractals(high, K, "high")
    low_i = [i for i in range(n) if piv_low[i] and not np.isnan(rsi[i])]
    high_i = [i for i in range(n) if piv_high[i] and not np.isnan(rsi[i])]

    def rec(kind, a, b, pcol):
        conf = min(b + K, n - 1)
        return {"type": kind,
                "p1_date": dt[a], "p1_bucket": bk[a], "p1_price": round(float(pcol[a]), 2), "p1_rsi": round(float(rsi[a]), 2),
                "p2_date": dt[b], "p2_bucket": bk[b], "p2_price": round(float(pcol[b]), 2), "p2_rsi": round(float(rsi[b]), 2),
                "confirm_date": dt[conf], "confirm_bucket": bk[conf],
                "n_candles_between_pivots": int(b - a),
                "p1_seq": a, "p2_seq": b, "confirm_seq": int(conf)}

    divs = []
    for a, b in zip(low_i, low_i[1:]):                       # bullish: consecutive pivot LOWS
        if b - a > L:
            continue
        if low[b] < low[a] and rsi[b] > rsi[a] and rsi[b] > BULL_FLOOR:
            divs.append(rec("bullish", a, b, low))
    for a, b in zip(high_i, high_i[1:]):                     # bearish: consecutive pivot HIGHS
        if b - a > L:
            continue
        if high[b] > high[a] and rsi[b] < rsi[a] and rsi[b] < BEAR_CEIL:
            divs.append(rec("bearish", a, b, high))
    D = pd.DataFrame(divs).sort_values("p2_seq").reset_index(drop=True)

    # overlay series (for plotting price+RSI vs flags)
    H2 = H.copy()
    H2["seq"] = range(n)
    H2["bull_div_at_p2"] = False; H2["bear_div_at_p2"] = False
    for _, r in D.iterrows():
        (H2.at[r["p2_seq"], "bull_div_at_p2"] if r["type"] == "bullish" else None)
        col = "bull_div_at_p2" if r["type"] == "bullish" else "bear_div_at_p2"
        H2.at[r["p2_seq"], col] = True
    H2[["date", "candle", "hour_bucket", "seq", "close", "rsi_14",
        "bull_div_at_p2", "bear_div_at_p2"]].to_csv(OUTDIR / "overlay_price_rsi_flags.csv", index=False)

    D.to_csv(OUTDIR / "divergences.csv", index=False)
    # counts by year / quarter
    Dd = D.copy(); ts = pd.to_datetime(Dd["p2_date"])
    Dd["year"] = ts.dt.year; Dd["quarter"] = ts.dt.year.astype(str) + "Q" + ts.dt.quarter.astype(str)
    by_year = Dd.groupby(["year", "type"]).size().unstack(fill_value=0)
    by_q = Dd.groupby(["quarter", "type"]).size().unstack(fill_value=0)
    with pd.ExcelWriter(OUTDIR / "nifty_rsi_divergence.xlsx", engine="openpyxl") as w:
        D.to_excel(w, sheet_name="divergences", index=False)
        by_year.to_excel(w, sheet_name="counts_by_year")
        by_q.to_excel(w, sheet_name="counts_by_quarter")

    # overview plot: close (top) + RSI (bottom) with markers at the 2nd pivots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, height_ratios=[2, 1])
    ax1.plot(H2["seq"], close, lw=0.5, color="#333")
    bl = D[D["type"] == "bullish"]; be = D[D["type"] == "bearish"]
    ax1.scatter(bl["p2_seq"], [close[i] for i in bl["p2_seq"]], c="green", s=18, marker="^", label=f"bullish ({len(bl)})")
    ax1.scatter(be["p2_seq"], [close[i] for i in be["p2_seq"]], c="red", s=18, marker="v", label=f"bearish ({len(be)})")
    ax1.set_ylabel("Nifty close (hourly)"); ax1.legend(); ax1.set_title("Nifty hourly close + RSI-14 divergences", fontweight="bold")
    ax2.plot(H2["seq"], rsi, lw=0.5, color="#2E74B5"); ax2.axhline(70, color="grey", ls=":"); ax2.axhline(30, color="grey", ls=":")
    ax2.scatter(bl["p2_seq"], [rsi[i] for i in bl["p2_seq"]], c="green", s=14, marker="^")
    ax2.scatter(be["p2_seq"], [rsi[i] for i in be["p2_seq"]], c="red", s=14, marker="v")
    ax2.set_ylabel("RSI-14"); ax2.set_xlabel("hourly candle sequence")
    fig.tight_layout(); fig.savefig(OUTDIR / "divergence_overview.png", dpi=120); plt.close(fig)

    pd.set_option("display.width", 220)
    print("=" * 90 + f"\nNIFTY HOURLY RSI-14 DIVERGENCES  (method={METHOD}, k={K}, L={L}, bull>30 / bear<70)\n" + "=" * 90)
    print(f"hourly candles: {n:,} | pivot lows: {len(low_i):,} | pivot highs: {len(high_i):,}")
    print(f"\nDETECTED: bullish {len(bl):,} | bearish {len(be):,} | total {len(D):,}")
    print("\n--- counts by year ---"); print(by_year.to_string())
    print("\n--- examples (5 bullish + 5 bearish; verify the lower-low/higher-RSI + filter by hand) ---")
    cols = ["type", "p1_date", "p1_bucket", "p1_price", "p1_rsi", "p2_date", "p2_bucket", "p2_price", "p2_rsi",
            "n_candles_between_pivots", "confirm_date", "confirm_bucket"]
    ex = pd.concat([D[D["type"] == "bullish"].head(5), D[D["type"] == "bearish"].head(5)])
    print(ex[cols].to_string(index=False))
    print("\n  bullish check: p2_price < p1_price (lower low) & p2_rsi > p1_rsi (higher RSI) & p2_rsi > 30")
    print("  bearish check: p2_price > p1_price (higher high) & p2_rsi < p1_rsi (lower RSI) & p2_rsi < 70")
    print(f"\nSaved -> {OUTDIR} (divergences.csv/.xlsx, overlay_price_rsi_flags.csv, divergence_overview.png)")


if __name__ == "__main__":
    main()
