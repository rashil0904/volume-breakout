# -*- coding: utf-8 -*-
"""
exit_day_high_low_distribution.py
=================================
Examines intraday price behaviour on EXIT DAYS (the trading day after entry,
when the exit price(s) occur) for two backtest variants, SEPARATELY:

  (1) baseline : fixed exit, 100% at next-day 3pm open   -> sheet "1_Standard trades"
  (2) split    : 50% at 9:45am + 50% at 11am next day     -> sheet "2_Split trades"

Both variants' exit day is the SAME single next trading day after entry (the
split's two legs both fall on that day), so each trade contributes exactly one
stock-exit-day instance.

For every stock-exit-day instance, over its 09:15..15:15 15-min candles:
  day_open              = 09:15 candle's open
  running max/min       = updated candle by candle
  "new high" event      = candle_high > running_max BEFORE updating this candle
  "new low"  event      = candle_low  < running_min BEFORE updating this candle
  pct_high_from_open    = (candle_high - day_open)/day_open*100   (own high)
  pct_low_from_open     = (candle_low  - day_open)/day_open*100   (own low)

NOTE on the first candle: with running max/min initialised to -/+inf, the 09:15
candle always counts as both a new high and a new low (no earlier candle to beat).
This is the literal "before updating" reading; the 09:15 bar therefore equals n
in the count charts by construction.

Produces, PER VARIANT, 4 bar charts (8 PNGs total) + a summary CSV, in
results/exit_day_analysis/. Does NOT recompute the backtest — trades are read
from the existing results/backtest_results.xlsx.
"""

import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE       = Path(__file__).resolve().parent.parent
MASTER_DIR = BASE / "master_data"
RESULTS    = BASE / "results"
XLSX       = RESULTS / "backtest_results.xlsx"
OUTDIR     = RESULTS / "exit_day_analysis"
IST        = "Asia/Kolkata"

HM_START, HM_END, STEP = 555, 915, 15               # 09:15 .. 15:15
HMS     = list(range(HM_START, HM_END + 1, STEP))    # 25 intervals
LABELS  = [f"{hm // 60:02d}:{hm % 60:02d}" for hm in HMS]

VARIANTS = [
    ("baseline", "1_Standard trades"),
    ("split",    "2_Split trades"),
]


def _empty_agg():
    return {hm: {"new_high": 0, "new_low": 0,
                 "sum_high_pct": 0.0, "sum_low_pct": 0.0, "n": 0} for hm in HMS}


def process_variant(variant, sheet):
    """Read this variant's trades and aggregate exit-day behaviour by interval."""
    trades = pd.read_excel(XLSX, sheet_name=sheet)
    trades["date"] = pd.to_datetime(trades["date"])

    agg = _empty_agg()
    n_instances = 0
    n_skipped_no_exit_day = 0

    for sym, tr in trades.groupby("symbol"):
        pq = MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            n_skipped_no_exit_day += len(tr)
            continue

        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"]   = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[(raw["hm"] >= HM_START) & (raw["hm"] <= HM_END)]

        by_day       = {d: g.sort_values("hm") for d, g in raw.groupby("date")}
        dates_sorted = sorted(by_day.keys())

        for _, t in tr.iterrows():
            ed  = t["date"].date()
            idx = bisect.bisect_right(dates_sorted, ed)      # first trading day AFTER entry
            if idx >= len(dates_sorted):
                n_skipped_no_exit_day += 1
                continue
            exit_day = dates_sorted[idx]
            g = by_day[exit_day]

            hms   = g["hm"].values
            highs = g["high"].values
            lows  = g["low"].values
            opens = g["open"].values

            # day_open = 09:15 open (fallback to earliest candle if 09:15 missing)
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
                is_new_low  = l < run_min
                if h > run_max:
                    run_max = h
                if l < run_min:
                    run_min = l
                a = agg[hm]
                a["new_high"]     += int(is_new_high)
                a["new_low"]      += int(is_new_low)
                a["sum_high_pct"] += (h - day_open) / day_open * 100
                a["sum_low_pct"]  += (l - day_open) / day_open * 100
                a["n"]            += 1
            n_instances += 1

    # Build summary table
    rows = []
    for hm, lab in zip(HMS, LABELS):
        a = agg[hm]
        n = a["n"]
        rows.append({
            "interval":              lab,
            "new_high_count":        a["new_high"],
            "new_low_count":         a["new_low"],
            "avg_pct_high_from_open": round(a["sum_high_pct"] / n, 6) if n else np.nan,
            "avg_pct_low_from_open":  round(a["sum_low_pct"]  / n, 6) if n else np.nan,
            "n":                     n,
        })
    summary = pd.DataFrame(rows)
    return summary, n_instances, n_skipped_no_exit_day


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
    results = {}

    for variant, sheet in VARIANTS:
        summary, n_inst, n_skip = process_variant(variant, sheet)
        results[variant] = (summary, n_inst, n_skip)

        # Save CSV
        csv_path = OUTDIR / f"summary_{variant}.csv"
        summary.to_csv(csv_path, index=False)

        labels = summary["interval"].tolist()
        vname  = variant.upper()

        bar_chart(labels, summary["new_high_count"],
                  f"New-High Events by Interval — {vname} (exit day)",
                  "Count of new-high events",
                  OUTDIR / f"new_high_count_by_interval_{variant}.png", "#2166ac")
        bar_chart(labels, summary["avg_pct_high_from_open"],
                  f"Avg % High from Open by Interval — {vname} (exit day)",
                  "Average (candle_high - day_open)/day_open %",
                  OUTDIR / f"avg_pct_high_from_open_by_interval_{variant}.png", "#4393c3")
        bar_chart(labels, summary["new_low_count"],
                  f"New-Low Events by Interval — {vname} (exit day)",
                  "Count of new-low events",
                  OUTDIR / f"new_low_count_by_interval_{variant}.png", "#b2182b")
        bar_chart(labels, summary["avg_pct_low_from_open"],
                  f"Avg % Low from Open by Interval — {vname} (exit day)",
                  "Average (candle_low - day_open)/day_open %",
                  OUTDIR / f"avg_pct_low_from_open_by_interval_{variant}.png", "#d6604d")

        print(f"\n[{vname}] exit-day instances (n) = {n_inst:,}  "
              f"(skipped, no exit day / no parquet = {n_skip:,})")
        print(summary.to_string(index=False))

    # Sample-size comparison
    nb = results["baseline"][1]
    ns = results["split"][1]
    print("\n" + "=" * 66)
    print(f"SAMPLE SIZE COMPARISON — baseline n = {nb:,}  |  split n = {ns:,}  |  diff = {nb - ns:+,}")
    print("=" * 66)
    print(f"\nPNGs + CSVs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
