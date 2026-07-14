#!/usr/bin/env python3
"""
fade_analysis.py — Sweep fade-from-high filter thresholds
==========================================================
Loads the existing diagnostic_table.csv and applies a fade filter:

  fade_pct = (intraday_high_before_entry - entry_price) / intraday_high_before_entry × 100

  For 3:15pm entry: intraday_high = today_cumhigh_15  (max high 9:15–15:00)
  For 3:00pm entry: intraday_high = today_cumhigh_14  (max high 9:15–14:00)

Signals where fade_pct > threshold are REMOVED (stock has faded too much).

Sweeps thresholds: 0% (no filter), 1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%, 9%, 10%

Output: results/fade_analysis.xlsx
  - Sheet "Summary"       — one row per (combo × threshold)
  - Sheet "Signals Kept"  — how many signals survive each threshold
  - Sheet per combo       — revenue / max-dd / win% across thresholds
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "results"

DAILY_POOL    = 500_000
MAX_PER_STOCK = 100_000

THRESHOLDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # percent

COMBOS = [
    ("3:15pm / Split (9:45+11am)", "entry_price_315pm", "split",  ("exit_945_open", "exit_1100_open"), "315pm_split"),
    ("3:15pm / 3pm exit",          "entry_price_315pm", "single", "exit_3pm_open",                    "315pm_3pm"),
    ("3:15pm / 2:30pm exit",       "entry_price_315pm", "single", "exit_1430_open",                   "315pm_230pm"),
    ("3:15pm / 2:45pm exit",       "entry_price_315pm", "single", "exit_1445_open",                   "315pm_245pm"),
    ("3pm / Split (9:45+11am)",    "entry_price_3pm",   "split",  ("exit_945_open", "exit_1100_open"), "3pm_split"),
    ("3pm / 3pm exit",             "entry_price_3pm",   "single", "exit_3pm_open",                    "3pm_3pm"),
    ("3pm / 2:30pm exit",          "entry_price_3pm",   "single", "exit_1430_open",                   "3pm_230pm"),
    ("3pm / 2:45pm exit",          "entry_price_3pm",   "single", "exit_1445_open",                   "3pm_245pm"),
]


def run_backtest(signals, entry_col, exit_type, exit_col_or_cols):
    if exit_type == "split":
        exit_cols = list(exit_col_or_cols)
    else:
        exit_cols = [exit_col_or_cols]

    valid = signals.dropna(subset=[entry_col] + exit_cols).copy()
    by_date = defaultdict(list)
    for _, row in valid.iterrows():
        by_date[row["date"]].append(row)

    exec_rows = []
    for entry_date in sorted(by_date):
        day_sigs = by_date[entry_date]
        n        = len(day_sigs)
        target   = MAX_PER_STOCK if n <= 5 else DAILY_POOL / n

        for row in day_sigs:
            ep     = float(row[entry_col])
            shares = int(target // ep)
            if shares == 0:
                continue
            actual_cap = shares * ep

            if exit_type == "split":
                col_945, col_1100 = exit_col_or_cols
                x945  = float(row[col_945])
                x1100 = float(row[col_1100])
                s1    = shares // 2
                s2    = shares - s1
                pnl   = s1 * (x945 - ep) + s2 * (x1100 - ep)
                ret   = ((x945 + x1100) / 2 - ep) / ep * 100
            else:
                xp  = float(row[exit_col_or_cols])
                pnl = shares * (xp - ep)
                ret = (xp - ep) / ep * 100

            exec_rows.append({
                "date":               entry_date,
                "pnl":                pnl,
                "return_pct":         ret,
                "capital_allocated":  actual_cap,
            })

    if not exec_rows:
        return pd.DataFrame(columns=["date", "pnl", "return_pct", "capital_allocated"])
    trades = pd.DataFrame(exec_rows)
    trades["date"] = pd.to_datetime(trades["date"])
    return trades


def compute_stats(trades):
    if trades.empty:
        return dict(revenue=0, total_return_pct=0, max_dd_pct=0, max_dd_rs=0,
                    win_pct=0, avg_ret=0, trades=0)

    daily = (trades.groupby("date")
             .agg(cap=("capital_allocated", "sum"), pnl=("pnl", "sum"))
             .reset_index())
    daily["dr"] = daily["pnl"] / daily["cap"] * 100

    total_ret = daily["dr"].sum()
    cumulative = daily["dr"].cumsum()
    running_max = cumulative.cummax()
    dd_pct = (cumulative - running_max)
    max_dd_pct = dd_pct.min()

    # Max drawdown in ₹ (absolute rupee drawdown from peak PnL)
    cum_pnl = trades.sort_values("date").groupby("date")["pnl"].sum().cumsum()
    peak_pnl = cum_pnl.cummax()
    max_dd_rs = (cum_pnl - peak_pnl).min()

    wins = (trades["pnl"] > 0).sum()
    return dict(
        revenue      = round(trades["pnl"].sum(), 0),
        total_return_pct = round(total_ret, 4),
        max_dd_pct   = round(max_dd_pct, 4),
        max_dd_rs    = round(max_dd_rs, 0),
        win_pct      = round(wins / len(trades) * 100, 2),
        avg_ret      = round(trades["return_pct"].mean(), 4),
        trades       = len(trades),
    )


def main():
    diag_path = RESULTS_DIR / "diagnostic_table.csv"
    print(f"Loading {diag_path} …")
    diag = pd.read_csv(diag_path, parse_dates=["date"])
    signals_all = diag[diag["passes_all_three"]].copy()
    print(f"  {len(diag):,} rows  |  {len(signals_all):,} signals")

    # Compute fade_pct for each entry type
    # For 3:15pm entry: fade vs cumhigh_15 (9:15–15:00)
    # For 3:00pm entry: fade vs cumhigh_14 (9:15–14:00)
    with np.errstate(invalid="ignore", divide="ignore"):
        signals_all["fade_315pm"] = np.where(
            signals_all["today_cumhigh_15"] > 0,
            (signals_all["today_cumhigh_15"] - signals_all["entry_price_315pm"])
            / signals_all["today_cumhigh_15"] * 100,
            np.nan,
        )
        signals_all["fade_3pm"] = np.where(
            signals_all["today_cumhigh_14"] > 0,
            (signals_all["today_cumhigh_14"] - signals_all["entry_price_3pm"])
            / signals_all["today_cumhigh_14"] * 100,
            np.nan,
        )

    print(f"\nFade stats (3:15pm entry):")
    fd = signals_all["fade_315pm"].dropna()
    print(f"  min={fd.min():.2f}%  median={fd.median():.2f}%  mean={fd.mean():.2f}%  max={fd.max():.2f}%")
    print(f"\nFade stats (3pm entry):")
    fd2 = signals_all["fade_3pm"].dropna()
    print(f"  min={fd2.min():.2f}%  median={fd2.median():.2f}%  mean={fd2.mean():.2f}%  max={fd2.max():.2f}%")

    print(f"\nRunning sweep: {THRESHOLDS} % thresholds × {len(COMBOS)} combos …")
    print("=" * 78)

    all_rows = []
    signal_counts = []

    for thresh in THRESHOLDS:
        if thresh == 0:
            filtered = signals_all.copy()
            label = "No filter"
        else:
            # Remove signals that have faded more than threshold from intraday high
            mask_315 = (signals_all["fade_315pm"] > thresh) | signals_all["fade_315pm"].isna()
            mask_3pm = (signals_all["fade_3pm"]   > thresh) | signals_all["fade_3pm"].isna()
            # We'll apply per-combo the appropriate fade column
            # But for signal count, use 315pm fade as primary
            filtered = signals_all[~mask_315].copy()
            label = f"≤{thresh}% fade"

        kept_315 = (signals_all["fade_315pm"] <= thresh).sum() if thresh > 0 else len(signals_all)
        kept_3pm = (signals_all["fade_3pm"]   <= thresh).sum() if thresh > 0 else len(signals_all)
        removed  = len(signals_all) - kept_315

        signal_counts.append({
            "threshold_pct": thresh,
            "label":         label,
            "total_signals": len(signals_all),
            "kept_315pm_entry": kept_315,
            "kept_3pm_entry":   kept_3pm,
            "removed_315pm":    len(signals_all) - kept_315,
            "removed_3pm":      len(signals_all) - kept_3pm,
            "pct_removed_315pm": round((len(signals_all) - kept_315) / len(signals_all) * 100, 2),
        })

        print(f"\n[{thresh:>2}% fade filter]  kept={kept_315:,}  removed={removed:,}  ({removed/len(signals_all)*100:.1f}%)")

        for combo_label, entry_col, exit_type, exit_cols, file_label in COMBOS:
            # Apply the appropriate fade column per entry type
            if thresh == 0:
                sig = signals_all.copy()
            elif "315pm" in entry_col:
                sig = signals_all[signals_all["fade_315pm"] <= thresh].copy()
            else:
                sig = signals_all[signals_all["fade_3pm"] <= thresh].copy()

            trades = run_backtest(sig, entry_col, exit_type, exit_cols)
            stats  = compute_stats(trades)

            row = {
                "threshold_pct": thresh,
                "label":         label,
                "combo":         combo_label,
                "file_label":    file_label,
                **stats,
            }
            all_rows.append(row)
            print(f"  {combo_label:<35}  {stats['trades']:>5,} trades  "
                  f"rev={stats['revenue']:>+12,.0f}  dd={stats['max_dd_pct']:>+7.2f}%  "
                  f"win={stats['win_pct']:>5.1f}%  avg={stats['avg_ret']:>+7.4f}%")

    # ── Build Excel output ────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "fade_analysis.xlsx"
    print(f"\nWriting {out_path} …")

    summary_df = pd.DataFrame(all_rows)
    counts_df  = pd.DataFrame(signal_counts)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # Sheet 1: Signal counts per threshold
        counts_df.to_excel(writer, sheet_name="Signal Counts", index=False)

        # Sheet 2: Full summary (all combos × all thresholds)
        cols_order = ["threshold_pct", "label", "combo", "trades", "revenue",
                      "total_return_pct", "max_dd_pct", "max_dd_rs",
                      "win_pct", "avg_ret"]
        summary_df[cols_order].to_excel(writer, sheet_name="Summary", index=False)

        # Per-combo sheets — pivot: threshold as rows, metrics as cols
        for combo_label, _, _, _, file_label in COMBOS:
            sub = summary_df[summary_df["file_label"] == file_label].copy()
            sub = sub[["threshold_pct", "label", "trades", "revenue",
                        "total_return_pct", "max_dd_pct", "max_dd_rs",
                        "win_pct", "avg_ret"]].reset_index(drop=True)
            sheet_name = file_label[:31]  # Excel sheet name limit
            sub.to_excel(writer, sheet_name=sheet_name, index=False)

        # Apply column widths
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                max_len = max((len(str(c.value)) for c in col if c.value), default=10)
                sheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)

    print(f"Saved: {out_path}")

    # ── Print best-threshold summary ─────────────────────────────────────────
    print("\n" + "=" * 90)
    print("FADE FILTER IMPACT — Best combo (3:15pm / Split) across thresholds")
    print("=" * 90)
    best = summary_df[summary_df["file_label"] == "315pm_split"].sort_values("threshold_pct")
    print(f"{'Threshold':>10}  {'Trades':>7}  {'Revenue (₹)':>14}  {'MaxDD%':>8}  {'Win%':>7}  {'Avg%':>8}  {'Removed':>8}")
    print("─" * 90)
    for _, r in best.iterrows():
        cnt = counts_df[counts_df["threshold_pct"] == r["threshold_pct"]]
        removed = int(cnt["removed_315pm"].values[0]) if len(cnt) else 0
        print(f"  {r['threshold_pct']:>4}%      {r['trades']:>7,}  {r['revenue']:>+14,.0f}  "
              f"{r['max_dd_pct']:>+7.2f}%  {r['win_pct']:>6.1f}%  {r['avg_ret']:>+8.4f}%  {removed:>7,}")


if __name__ == "__main__":
    main()
