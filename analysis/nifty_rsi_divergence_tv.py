# -*- coding: utf-8 -*-
"""
nifty_rsi_divergence_tv.py — EXACT TradingView "Divergence Indicator" (Pine v6) replication on the
hourly Nifty RSI-14 series, PLUS a 30/70 RSI filter on the 2nd (current) pivot.

Pine logic replicated exactly:
  osc = ta.rsi(close,14)                         # Wilder RMA (SMA-seeded), reused from step-1 rsi_14
  plFound = not na(ta.pivotlow(osc, 5, 5))       # pivots on the RSI, 5 left / 5 right, confirmed 5 bars late
  phFound = not na(ta.pivothigh(osc, 5, 5))
  _inRange(cond): rangeLower(5) <= bars_between_pivots <= rangeUpper(60)
  Regular Bullish : priceLL = low[5]  < valuewhen(plFound, low[5], 1)
                    oscHL   = osc[5]  > valuewhen(plFound, osc[5], 1)  and _inRange
                    bullCond = priceLL and oscHL and plFound
  Regular Bearish : priceHH = high[5] > valuewhen(phFound, high[5], 1)
                    oscLH   = osc[5]  < valuewhen(phFound, osc[5], 1)  and _inRange
                    bearCond = priceHH and oscLH and phFound
  osc[5] = rsiLBR = the RSI at the pivot bar (5 bars back from the confirmation bar).

ADDED 30/70 filter (this step) — on rsiLBR (2nd/current pivot RSI ONLY, strict):
  bullCond &= (rsiLBR > 30)     bearCond &= (rsiLBR < 70)
Filtered divergences are a strict SUBSET of the unfiltered TradingView ones; we report both counts.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "nifty_hourly_rsi" / "nifty_hourly_rsi.csv"
OUTDIR = rb.RESULTS / "nifty_rsi_divergence_tv"
LBL, LBR = 5, 5
RANGE_LOWER, RANGE_UPPER = 5, 60
BULL_FLOOR, BEAR_CEIL = 30.0, 70.0


def pivots_confirmed(v, lbl, lbr, kind):
    """Replicate ta.pivotlow/high(v, lbl, lbr): at bar t, the candidate at c=t-lbr is a pivot if it is
    strictly the extreme vs lbl bars left and lbr bars right. Confirmed at bar t. Returns bool array
    'found' (True at confirmation bar t) and 'candbar' (the pivot bar c=t-lbr)."""
    n = len(v); found = np.zeros(n, bool); candbar = np.full(n, -1)
    for t in range(lbl + lbr, n):
        c = t - lbr
        cand = v[c]
        if not (cand == cand):
            continue
        left = v[c - lbl:c]; right = v[c + 1:c + lbr + 1]
        if np.isnan(left).any() or np.isnan(right).any():
            continue
        if kind == "low" and np.all(cand < left) and np.all(cand < right):
            found[t] = True; candbar[t] = c
        elif kind == "high" and np.all(cand > left) and np.all(cand > right):
            found[t] = True; candbar[t] = c
    return found, candbar


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    H = pd.read_csv(SRC, parse_dates=["date"])
    H["date"] = H["date"].dt.date
    n = len(H)
    rsi = H["rsi_14"].values.astype(float)
    low = H["low"].values.astype(float); high = H["high"].values.astype(float)
    dt = H["date"].values; bk = H["hour_bucket"].values

    plF, plC = pivots_confirmed(rsi, LBL, LBR, "low")
    phF, phC = pivots_confirmed(rsi, LBL, LBR, "high")
    plbars = [t for t in range(n) if plF[t]]
    phbars = [t for t in range(n) if phF[t]]

    bullCond = np.zeros(n, bool); bearCond = np.zeros(n, bool)              # FILTERED
    bullU = np.zeros(n, bool); bearU = np.zeros(n, bool)                    # UNFILTERED (pure TV)
    bull_rows, bear_rows = [], []

    def rec(kind, t, prev_t, pcol):
        c, cp = t - LBR, prev_t - LBR
        return {"type": kind, "confirmation_bar": t, "pivot_bar_T-5": c,
                "date": dt[c], "hour_bucket": bk[c], "confirm_date": dt[t], "confirm_bucket": bk[t],
                "rsi_at_pivot": round(float(rsi[c]), 4), "rsi_at_prev_pivot": round(float(rsi[cp]), 4),
                "price_at_pivot": round(float(pcol[c]), 2), "price_at_prev_pivot": round(float(pcol[cp]), 2),
                "prev_pivot_date": dt[cp], "prev_pivot_bucket": bk[cp],
                "bars_between_pivots": int(t - prev_t)}

    # Regular Bullish — consecutive RSI pivot lows (valuewhen ,1 == previous plFound bar)
    for prev_t, t in zip(plbars, plbars[1:]):
        c, cp = t - LBR, prev_t - LBR
        bars_between = t - prev_t
        in_range = RANGE_LOWER <= bars_between <= RANGE_UPPER
        priceLL = low[c] < low[cp]
        oscHL = rsi[c] > rsi[cp] and in_range
        if priceLL and oscHL:                                              # unfiltered TV bullCond
            bullU[t] = True
            if rsi[c] > BULL_FLOOR:                                        # + 30/70 filter (rsiLBR>30)
                bullCond[t] = True; bull_rows.append(rec("bullish", t, prev_t, low))

    # Regular Bearish — consecutive RSI pivot highs
    for prev_t, t in zip(phbars, phbars[1:]):
        c, cp = t - LBR, prev_t - LBR
        bars_between = t - prev_t
        in_range = RANGE_LOWER <= bars_between <= RANGE_UPPER
        priceHH = high[c] > high[cp]
        oscLH = rsi[c] < rsi[cp] and in_range
        if priceHH and oscLH:
            bearU[t] = True
            if rsi[c] < BEAR_CEIL:
                bearCond[t] = True; bear_rows.append(rec("bearish", t, prev_t, high))

    D = pd.DataFrame(bull_rows + bear_rows).sort_values("confirmation_bar").reset_index(drop=True)
    n_bull_u, n_bear_u = int(bullU.sum()), int(bearU.sum())
    n_bull_f, n_bear_f = int(bullCond.sum()), int(bearCond.sum())

    # per-bar verification series
    H2 = H.copy(); H2["seq"] = range(n)
    H2["plFound"] = plF; H2["phFound"] = phF
    H2["bullCond"] = bullCond; H2["bearCond"] = bearCond
    H2[["date", "hour_bucket", "seq", "close", "rsi_14", "plFound", "phFound", "bullCond", "bearCond"]] \
        .to_csv(OUTDIR / "per_bar_series.csv", index=False)

    with pd.ExcelWriter(OUTDIR / "nifty_rsi_divergence_tv_filtered.xlsx", engine="openpyxl") as w:
        D.to_excel(w, sheet_name="divergences_filtered", index=False)
        pd.DataFrame([{"type": "bullish", "unfiltered_TV": n_bull_u, "filtered_30_70": n_bull_f, "removed": n_bull_u - n_bull_f},
                      {"type": "bearish", "unfiltered_TV": n_bear_u, "filtered_30_70": n_bear_f, "removed": n_bear_u - n_bear_f}]
                     ).to_excel(w, sheet_name="filter_effect", index=False)

    D.to_csv(OUTDIR / "divergences_filtered.csv", index=False)
    pd.set_option("display.width", 240)
    print("=" * 96 + "\nEXACT TradingView RSI-Divergence (RSI-pivots 5/5, valuewhen, _inRange 5-60) + 30/70 filter\n" + "=" * 96)
    print(f"hourly bars: {n:,} | RSI pivot lows (plFound): {len(plbars):,} | pivot highs (phFound): {len(phbars):,}")
    print("\n--- FILTER EFFECT (pure TradingView -> after adding rsiLBR>30 / rsiLBR<70) ---")
    print(f"  BULLISH : unfiltered TV {n_bull_u:>3}  ->  filtered {n_bull_f:>3}   (removed {n_bull_u - n_bull_f}: 2nd-RSI-low <= 30)")
    print(f"  BEARISH : unfiltered TV {n_bear_u:>3}  ->  filtered {n_bear_f:>3}   (removed {n_bear_u - n_bear_f}: 2nd-RSI-high >= 70)")
    print(f"  TOTAL   : {n_bull_u + n_bear_u} -> {n_bull_f + n_bear_f}  (removed {n_bull_u + n_bear_u - n_bull_f - n_bear_f})")
    print("\n--- FILTERED DIVERGENCES (first 6 bull + 6 bear; rsi_at_pivot guaranteed >30 / <70) ---")
    cols = ["type", "date", "hour_bucket", "pivot_bar_T-5", "confirmation_bar", "rsi_at_pivot", "rsi_at_prev_pivot",
            "price_at_pivot", "price_at_prev_pivot", "bars_between_pivots"]
    ex = pd.concat([D[D.type == "bullish"].head(6), D[D.type == "bearish"].head(6)])
    print(ex[cols].to_string(index=False))
    print(f"\nSaved -> {OUTDIR} (divergences_filtered.csv/.xlsx, per_bar_series.csv, filter_effect)")


if __name__ == "__main__":
    main()
