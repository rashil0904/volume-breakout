# -*- coding: utf-8 -*-
"""
exit_day_high_low_5k7p5k.py
===========================
Exit-day intraday high/low behaviour for the ₹5,000–7,500 Cr strategy variant
(lookback 36, volume 6x, 3:15pm entry, +5% daily return, 09:45/12:00 conditional
split exit + 14% target, pool-split ₹1L/trade). Same 4-chart structure as the baseline
exit_day_high_low_distribution.py, applied to THIS variant's existing trades.

Trades are read from the variant's saved trade list (results/mcap_band_comparison/
trades_mcap5k7p5k.csv) — the strategy is NOT recomputed. For each trade's exit day
(first trading day after entry), the stock's 09:15–15:15 15-min candles are scanned
chronologically:
  day_open              = 09:15 open
  "new high" event      = candle_high > running_max BEFORE updating this candle
  "new low"  event      = candle_low  < running_min BEFORE updating this candle
  pct_high_from_open    = (candle_high − day_open)/day_open × 100
  pct_low_from_open     = (candle_low  − day_open)/day_open × 100
The 09:15 bar (running max/min at -/+inf) always counts as both a new high and new low
— the literal "before updating" reading, identical to the baseline version.
"""
import sys
import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
MASTER_DIR = rb.MASTER_DIR
TRADES_CSV = rb.RESULTS / "mcap_band_comparison" / "trades_mcap5k7p5k.csv"
OUTDIR = rb.RESULTS / "exit_day_analysis_5k7p5k"
VNAME = "₹5,000–7,500 Cr"

HM_START, HM_END, STEP = 555, 915, 15                 # 09:15 .. 15:15
HMS = list(range(HM_START, HM_END + 1, STEP))         # 25 intervals
LABELS = [f"{hm // 60:02d}:{hm % 60:02d}" for hm in HMS]


def _empty_agg():
    return {hm: {"new_high": 0, "new_low": 0,
                 "sum_high_pct": 0.0, "sum_low_pct": 0.0, "n": 0} for hm in HMS}


def process():
    trades = pd.read_csv(TRADES_CSV, parse_dates=["entry_date"])
    agg = _empty_agg()
    n_instances = 0
    n_skipped = 0

    for sym, tr in trades.groupby("symbol"):
        pq = MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            n_skipped += len(tr)
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[(raw["hm"] >= HM_START) & (raw["hm"] <= HM_END)]

        by_day = {d: g.sort_values("hm") for d, g in raw.groupby("date")}
        dates_sorted = sorted(by_day.keys())

        for _, t in tr.iterrows():
            ed = t["entry_date"].date()
            idx = bisect.bisect_right(dates_sorted, ed)       # first trading day AFTER entry
            if idx >= len(dates_sorted):
                n_skipped += 1
                continue
            g = by_day[dates_sorted[idx]]
            hms = g["hm"].values
            highs = g["high"].values
            lows = g["low"].values
            opens = g["open"].values

            open_915 = g.loc[g["hm"] == HM_START, "open"]
            day_open = float(open_915.iloc[0]) if len(open_915) else float(opens[0])
            if day_open == 0:
                continue

            run_max, run_min = float("-inf"), float("inf")
            for i in range(len(hms)):
                hm = int(hms[i])
                if hm not in agg:
                    continue
                h, l = float(highs[i]), float(lows[i])
                is_new_high = h > run_max
                is_new_low = l < run_min
                if h > run_max:
                    run_max = h
                if l < run_min:
                    run_min = l
                a = agg[hm]
                a["new_high"] += int(is_new_high)
                a["new_low"] += int(is_new_low)
                a["sum_high_pct"] += (h - day_open) / day_open * 100
                a["sum_low_pct"] += (l - day_open) / day_open * 100
                a["n"] += 1
            n_instances += 1

    rows = []
    for hm, lab in zip(HMS, LABELS):
        a = agg[hm]
        n = a["n"]
        rows.append({
            "interval": lab,
            "new_high_count": a["new_high"],
            "new_low_count": a["new_low"],
            "avg_pct_high_from_open": round(a["sum_high_pct"] / n, 6) if n else np.nan,
            "avg_pct_low_from_open": round(a["sum_low_pct"] / n, 6) if n else np.nan,
            "n": n,
        })
    return pd.DataFrame(rows), n_instances, n_skipped


def bar_chart(labels, values, title, ylabel, out_png, color):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=color, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_xlabel("15-min interval (exit day)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not TRADES_CSV.exists():
        raise SystemExit(f"Missing {TRADES_CSV} — run analysis/mcap_band_comparison.py first.")

    summary, n_inst, n_skip = process()
    summary.to_csv(OUTDIR / "exit_day_high_low_summary_5k7p5k.csv", index=False)

    print(f"Variant: {VNAME}")
    print(f"  exit-day instances (n): {n_inst:,}   (skipped, no exit day/data: {n_skip})")
    print(f"  saved summary -> exit_day_high_low_summary_5k7p5k.csv")

    L = summary["interval"].tolist()
    bar_chart(L, summary["new_high_count"],
              f"New-high events by interval — exit day ({VNAME})",
              "count of new-high events",
              OUTDIR / "new_high_count_by_interval_5k7p5k.png", "#1f4e79")
    bar_chart(L, summary["avg_pct_high_from_open"],
              f"Avg candle-high % from 09:15 open by interval — exit day ({VNAME})",
              "avg pct_high_from_open (%)",
              OUTDIR / "avg_pct_high_from_open_by_interval_5k7p5k.png", "#2ca25f")
    bar_chart(L, summary["new_low_count"],
              f"New-low events by interval — exit day ({VNAME})",
              "count of new-low events",
              OUTDIR / "new_low_count_by_interval_5k7p5k.png", "#b2182b")
    bar_chart(L, summary["avg_pct_low_from_open"],
              f"Avg candle-low % from 09:15 open by interval — exit day ({VNAME})",
              "avg pct_low_from_open (%)",
              OUTDIR / "avg_pct_low_from_open_by_interval_5k7p5k.png", "#e08214")

    print(f"  4 charts saved -> {OUTDIR}")
    pd.set_option("display.width", 160)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
