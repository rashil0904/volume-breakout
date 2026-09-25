# -*- coding: utf-8 -*-
"""
52w_high_breakout_forward_highlow.py
====================================
Filters the main strategy's trades to 52-WEEK-HIGH BREAKOUTS (entry day made a new
52-week high) and tracks each forward day's daily HIGH/LOW for T+1..T+5. Diagnostic only —
no exit-logic change. Reuses canonical trades (fpr.build_trades); not recomputed.

Filter: trailing 52w high = max daily HIGH over the trailing 252 trading days ending the
day BEFORE entry (excludes entry day). Keep trades where entry-day high >= that level.
Forward: each forward day's OWN daily high/low (not cumulative), % from entry_price
(15:15 open). Trades without a full 252-day prior history are dropped and counted.
"""
import sys, bisect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "breakout_52w_forward"
WIN_52 = 252
FWD = [1, 2, 3, 4, 5]


def scan(trades):
    """Per-trade: trailing 52w high (excl. entry day), entry-day high, forward day
    high/low (n,5 each). Returns arrays + n_no_history."""
    n = len(trades)
    tw52 = np.full(n, np.nan)
    entry_high = np.full(n, np.nan)
    fh = np.full((n, len(FWD)), np.nan)
    fl = np.full((n, len(FWD)), np.nan)
    n_no_hist = 0
    for sym, grp in trades.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        dh = raw.groupby("date")["high"].max()
        dl = raw.groupby("date")["low"].min()
        dates_sorted = sorted(dh.index)
        dh = dh.reindex(dates_sorted); dl = dl.reindex(dates_sorted)
        # 52w high ending the day BEFORE entry: rolling max then shift(1)
        r52 = dh.rolling(WIN_52, min_periods=WIN_52).max().shift(1)
        r52_map, dh_map, dl_map = r52.to_dict(), dh.to_dict(), dl.to_dict()
        for ridx, ed in zip(grp.index, grp["entry_date"]):
            dd = pd.Timestamp(ed).date()
            hi = r52_map.get(dd, np.nan)
            if hi is None or hi != hi:                 # < 252-day prior history
                n_no_hist += 1
                continue
            tw52[ridx] = hi
            entry_high[ridx] = dh_map.get(dd, np.nan)
            idx0 = bisect.bisect_right(dates_sorted, dd)   # T+1 position
            for k, d in enumerate(FWD):
                j = idx0 + d - 1
                if j >= len(dates_sorted):
                    break
                fdate = dates_sorted[j]
                fh[ridx, k] = dh_map.get(fdate, np.nan)
                fl[ridx, k] = dl_map.get(fdate, np.nan)
    return tw52, entry_high, fh, fl, n_no_hist


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = fpr.build_trades().reset_index(drop=True)
    e = T["entry_price"].values.astype(float)

    print("Scanning parquets for 52w high + forward high/low …")
    tw52, entry_high, fh, fl, n_no_hist = scan(T)

    has_hist = ~np.isnan(tw52)
    breakout = has_hist & (entry_high >= tw52)          # entry day made a new 52w high
    print(f"  total trades: {len(T):,}")
    print(f"  dropped (no full 252-day history): {n_no_hist:,}")
    print(f"  with history: {int(has_hist.sum()):,} | 52w-high BREAKOUTS: {int(breakout.sum()):,} "
          f"({int(breakout.sum())/len(T)*100:.1f}% of total)")

    ki = np.where(breakout)[0]
    hp = (fh - e[:, None]) / e[:, None] * 100           # high_pct_from_entry (n,5)
    lp = (fl - e[:, None]) / e[:, None] * 100           # low_pct_from_entry

    # ── 1. trade-level detail ──
    det = {"entry_date": T["entry_date"].values[ki], "symbol": T["symbol"].values[ki],
           "entry_price": np.round(e[ki], 2), "trailing_52w_high": np.round(tw52[ki], 2),
           "entry_day_high": np.round(entry_high[ki], 2)}
    for k, d in enumerate(FWD):
        det[f"T{d}_high_price"] = np.round(fh[ki, k], 2)
        det[f"T{d}_high_pct"] = np.round(hp[ki, k], 4)
        det[f"T{d}_low_price"] = np.round(fl[ki, k], 2)
        det[f"T{d}_low_pct"] = np.round(lp[ki, k], 4)
    det["n_forward_days_missing"] = np.isnan(fh[ki]).sum(axis=1)
    detail = pd.DataFrame(det)

    # ── 2. aggregate summary per forward day ──
    rows = []
    for k, d in enumerate(FWD):
        h = hp[ki, k]; l = lp[ki, k]
        vh = ~np.isnan(h); vl = ~np.isnan(l)
        nv = int(vh.sum())
        rows.append({
            "forward_day": f"T+{d}", "n_trades": nv,
            "avg_high_pct": round(np.nanmean(h), 4) if nv else np.nan,
            "median_high_pct": round(np.nanmedian(h), 4) if nv else np.nan,
            "avg_low_pct": round(np.nanmean(l), 4) if nv else np.nan,
            "median_low_pct": round(np.nanmedian(l), 4) if nv else np.nan,
            "max_high_pct": round(np.nanmax(h), 4) if nv else np.nan,
            "min_low_pct": round(np.nanmin(l), 4) if nv else np.nan,
            "pct_trades_high_above_entry": round(np.mean(h[vh] > 0) * 100, 2) if nv else np.nan,
            "pct_trades_low_below_entry": round(np.mean(l[vl] < 0) * 100, 2) if vl.sum() else np.nan,
        })
    summary = pd.DataFrame(rows)

    with pd.ExcelWriter(OUTDIR / "52w_high_breakout_forward_highlow.xlsx", engine="openpyxl") as w:
        detail.to_excel(w, sheet_name="breakout_trades_detail", index=False)
        summary.to_excel(w, sheet_name="forward_highlow_summary", index=False)

    # ── 3. avg high/low envelope chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [f"T+{d}" for d in FWD]
    ax.plot(x, summary["avg_high_pct"], "o-", color="#2ca25f", lw=2, label="avg high % from entry")
    ax.plot(x, summary["avg_low_pct"], "o-", color="#b2182b", lw=2, label="avg low % from entry")
    ax.fill_between(range(len(x)), summary["avg_low_pct"], summary["avg_high_pct"],
                    color="#1f4e79", alpha=0.08)
    for i, (hi, lo) in enumerate(zip(summary["avg_high_pct"], summary["avg_low_pct"])):
        ax.annotate(f"{hi:+.2f}%", (i, hi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.annotate(f"{lo:+.2f}%", (i, lo), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("forward day"); ax.set_ylabel("% from entry price")
    ax.set_title("52-week-high breakout — avg forward high/low envelope (T+1..T+5)\n"
                 f"({int(breakout.sum())} breakout trades)", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "forward_highlow_envelope.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 110)
    print("FORWARD HIGH/LOW SUMMARY — 52-week-high breakout trades")
    print("=" * 110)
    print(summary.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
