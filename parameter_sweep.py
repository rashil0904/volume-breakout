#!/usr/bin/env python3
"""
parameter_sweep.py — Full 24×8×8 grid sweep
==============================================
Grid:
  lookback_days : 22–45 (24 values)
  vol_mult      : 4–11  (8 values: 4,5,6,7,8,9,10,11)
  entry/exit    : 8 unique combinations (duplicate 3pm-exit rows excluded)
  Total         : 24 × 8 × 8 = 1,536 backtest runs

Efficiency:
  • build_diagnostic_table() called ONCE per lookback (24 times, ~53s first,
    ~10-15s subsequent via OS page cache). vol_mult=0.0 keeps all rows;
    the volume_ratio column is used for cheap per-mult filtering in memory.
  • run_single_backtest() called for each (mult, combo) pair — fast.

Train/test split:
  TRAIN = 2022-01-01 → 2025-05-31
  TEST  = 2025-06-01 → present

Output:
  results/full_sweep_results.xlsx  (1,536 rows, one per run)
  Console: top 15 by full-period return, train/test table, split-exit analysis.

Run:
  python parameter_sweep.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from prepare_data import build_diagnostic_table
from run_backtest import run_single_backtest

BASE         = Path(__file__).parent
RESULTS_DIR  = BASE / "results"
TRAIN_CUTOFF = pd.Timestamp("2025-05-31")

LOOKBACKS = list(range(22, 46))          # 22, 23, …, 45  (24 values)
VOL_MULTS = [4, 5, 6, 7, 8, 9, 10, 11]  # 8 values

# 8 unique entry/exit combos — no duplicate 3pm-exit rows
UNIQUE_COMBOS = [
    ("3pm / split",    "entry_price_3pm",   "split",  ("exit_945_open", "exit_1100_open")),
    ("3pm / 3pm",      "entry_price_3pm",   "single", "exit_3pm_open"),
    ("3pm / 2:30pm",   "entry_price_3pm",   "single", "exit_1430_open"),
    ("3pm / 2:45pm",   "entry_price_3pm",   "single", "exit_1445_open"),
    ("315pm / split",  "entry_price_315pm", "split",  ("exit_945_open", "exit_1100_open")),
    ("315pm / 3pm",    "entry_price_315pm", "single", "exit_3pm_open"),
    ("315pm / 2:30pm", "entry_price_315pm", "single", "exit_1430_open"),
    ("315pm / 2:45pm", "entry_price_315pm", "single", "exit_1445_open"),
]


def compute_stats(trades):
    """Non-compounding return stats for a trade log."""
    if trades.empty:
        return {
            "total_trades": 0, "total_return_pct": 0.0,
            "win_rate_pct": 0.0, "avg_return_per_trade": 0.0,
        }
    daily = (trades.groupby("date")
             .agg(capital=("capital_allocated", "sum"), pnl=("pnl", "sum"))
             .reset_index())
    daily["dr"] = daily["pnl"] / daily["capital"] * 100
    return {
        "total_trades":         len(trades),
        "total_return_pct":     round(float(daily["dr"].sum()), 4),
        "win_rate_pct":         round(float((trades["pnl"] > 0).mean() * 100), 2),
        "avg_return_per_trade": round(float(trades["return_pct"].mean()), 4),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    n_total = len(LOOKBACKS) * len(VOL_MULTS) * len(UNIQUE_COMBOS)

    print("=" * 72)
    print(f"parameter_sweep.py")
    print(f"  Lookbacks  : {LOOKBACKS[0]}–{LOOKBACKS[-1]}  ({len(LOOKBACKS)} values)")
    print(f"  Vol mults  : {VOL_MULTS[0]}–{VOL_MULTS[-1]}  ({len(VOL_MULTS)} values)")
    print(f"  Combos     : {len(UNIQUE_COMBOS)} unique entry/exit (no duplicate 3pm-exit rows)")
    print(f"  Total runs : {n_total:,}  ({len(LOOKBACKS)} data builds × {len(VOL_MULTS)} mults × {len(UNIQUE_COMBOS)} combos)")
    print(f"  Train      : 2022-01-01 → 2025-05-31")
    print(f"  Test       : 2025-06-01 → present")
    print("=" * 72 + "\n")

    all_rows = []
    t_total  = time.time()

    for lb_idx, lb in enumerate(LOOKBACKS):
        t_lb = time.time()
        print(f"[{lb_idx + 1:>2}/{len(LOOKBACKS)}] lookback={lb}d  building diagnostic …", flush=True)

        # vol_mult=0.0 → passes_volume=True for all valid rows; we filter via
        # volume_ratio column below, avoiding 192 separate prepare_data calls.
        diag = build_diagnostic_table(
            vol_window = lb,
            vol_mult   = 0.0,
            save_csv   = False,
            verbose    = False,
        )

        if diag.empty:
            print(f"  !! empty diagnostic table for lb={lb}, skipping")
            continue

        diag["date"] = pd.to_datetime(diag["date"])
        build_elapsed = time.time() - t_lb
        n_pass_ret    = int(diag["passes_return"].sum())
        print(f"  {build_elapsed:.1f}s  |  {len(diag):,} rows  |  {n_pass_ret:,} pass_return", flush=True)

        for mult in VOL_MULTS:
            # Cheap in-memory filter — no recomputation needed
            signals = diag[(diag["volume_ratio"] >= mult) & diag["passes_return"]].copy()

            for combo_label, entry_col, exit_type, exit_col in UNIQUE_COMBOS:
                trades = run_single_backtest(signals, entry_col, exit_type, exit_col)

                if not trades.empty:
                    tr_train = trades[trades["date"] <= TRAIN_CUTOFF]
                    tr_test  = trades[trades["date"] >  TRAIN_CUTOFF]
                else:
                    tr_train = tr_test = trades

                sf = compute_stats(trades)
                st = compute_stats(tr_train)
                sv = compute_stats(tr_test)

                all_rows.append({
                    "lookback_days":        lb,
                    "vol_mult":             mult,
                    "combo":                combo_label,
                    # Full period
                    "total_trades":         sf["total_trades"],
                    "total_return_pct":     sf["total_return_pct"],
                    "win_rate_pct":         sf["win_rate_pct"],
                    "avg_return_per_trade": sf["avg_return_per_trade"],
                    # Train period
                    "train_trades":         st["total_trades"],
                    "train_return_pct":     st["total_return_pct"],
                    "train_win_rate_pct":   st["win_rate_pct"],
                    "train_avg_return":     st["avg_return_per_trade"],
                    # Test period
                    "test_trades":          sv["total_trades"],
                    "test_return_pct":      sv["total_return_pct"],
                    "test_win_rate_pct":    sv["win_rate_pct"],
                    "test_avg_return":      sv["avg_return_per_trade"],
                })

        done    = (lb_idx + 1) * len(VOL_MULTS) * len(UNIQUE_COMBOS)
        elapsed = time.time() - t_total
        rate    = done / elapsed if elapsed > 0 else 1
        eta     = (n_total - done) / rate
        print(f"  {done:>5}/{n_total} runs done  |  ETA {eta / 60:.1f} min", flush=True)

    # ── Assemble results ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)

    # ── Save to disk ──────────────────────────────────────────────────────────
    out_xlsx    = RESULTS_DIR / "full_sweep_results.xlsx"
    out_parquet = RESULTS_DIR / "full_sweep_results.parquet"
    try:
        results_df.to_excel(out_xlsx, index=False)
        print(f"\nSaved: {out_xlsx}  ({len(results_df):,} rows)")
    except Exception as e:
        print(f"\nExcel save failed ({e}), saving as parquet instead.")
        results_df.to_parquet(out_parquet, index=False)
        print(f"Saved: {out_parquet}  ({len(results_df):,} rows)")

    results_df.to_parquet(out_parquet, index=False)  # always save parquet copy

    # ── Full-period rank column ───────────────────────────────────────────────
    results_df["full_rank"] = (results_df["total_return_pct"]
                               .rank(ascending=False, method="min")
                               .astype(int))

    # ── TOP 15 — full-period total return ─────────────────────────────────────
    top15 = results_df.nlargest(15, "total_return_pct")
    print("\n" + "=" * 88)
    print("TOP 15  —  Full-period total return  (2022-01-01 → present)")
    print("=" * 88)
    print(f"{'#':>2}  {'Combo':<22}  {'LB':>3}  {'Mult':>4}  {'Trades':>7}  "
          f"{'Total%':>9}  {'Win%':>7}  {'Avg/trade%':>11}")
    print("    " + "─" * 79)
    for i, (_, r) in enumerate(top15.iterrows(), 1):
        print(f"{i:>2}  {r['combo']:<22}  {int(r['lookback_days']):>3}  "
              f"{int(r['vol_mult']):>4}  {int(r['total_trades']):>7,}  "
              f"{r['total_return_pct']:>+9.2f}  {r['win_rate_pct']:>6.2f}%  "
              f"{r['avg_return_per_trade']:>+10.4f}")

    # ── TRAIN/TEST — top 10 by train return ───────────────────────────────────
    top10_train = results_df.nlargest(10, "train_return_pct").reset_index(drop=True)
    print("\n" + "=" * 110)
    print("TRAIN/TEST  —  Top 10 selected by TRAIN return only (no look-ahead)")
    print("  Train: 2022-01-01 → 2025-05-31    Test: 2025-06-01 → present")
    print("=" * 110)
    print(f"{'#':>2}  {'Combo':<22}  {'LB':>3}  {'M':>2}  "
          f"{'Train%':>8}  {'Tr.Trades':>9}  "
          f"{'Test%':>8}  {'Te.Trades':>9}  "
          f"{'Full%':>8}  {'FullRank':>8}")
    print("    " + "─" * 99)
    for i, r in top10_train.iterrows():
        mask = (results_df["lookback_days"].eq(r["lookback_days"]) &
                results_df["vol_mult"].eq(r["vol_mult"]) &
                results_df["combo"].eq(r["combo"]))
        full_rank = int(results_df.loc[mask, "full_rank"].iloc[0])
        print(f"{i + 1:>2}  {r['combo']:<22}  {int(r['lookback_days']):>3}  "
              f"{int(r['vol_mult']):>2}  "
              f"{r['train_return_pct']:>+8.2f}  {int(r['train_trades']):>9,}  "
              f"{r['test_return_pct']:>+8.2f}  {int(r['test_trades']):>9,}  "
              f"{r['total_return_pct']:>+8.2f}  #{full_rank:>6}")

    # ── SWEET SPOT CHECK ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SWEET SPOT vs NOISE — top-train spread across 1,536 combos")
    print("=" * 72)
    full_spread  = results_df["total_return_pct"].max() - results_df["total_return_pct"].min()
    train_spread = results_df["train_return_pct"].max() - results_df["train_return_pct"].min()
    test_spread  = results_df["test_return_pct"].max() - results_df["test_return_pct"].min()
    print(f"  Full-period spread (max-min): {full_spread:+.2f}%")
    print(f"  Train spread               : {train_spread:+.2f}%")
    print(f"  Test spread                : {test_spread:+.2f}%")

    # Spearman correlation: do train top-ranked combos hold in test?
    corr = results_df[["train_return_pct", "test_return_pct"]].corr(method="spearman")
    spearman_r = corr.iloc[0, 1]
    print(f"  Spearman rank corr (train vs test): {spearman_r:.3f}  "
          f"({'stable ✓' if spearman_r > 0.5 else 'weak — overfitting risk ⚠' if spearman_r > 0 else 'negative — overfit ✗'})")

    # Parameter stability: what LB/mult range do the top 10% of full-period combos cluster in?
    p90 = results_df["total_return_pct"].quantile(0.90)
    top10pct = results_df[results_df["total_return_pct"] >= p90]
    lb_range  = f"{int(top10pct['lookback_days'].min())}–{int(top10pct['lookback_days'].max())}"
    mult_range = f"{int(top10pct['vol_mult'].min())}–{int(top10pct['vol_mult'].max())}"
    combo_freq = top10pct["combo"].value_counts()
    print(f"\n  Top 10% full-period runs ({len(top10pct)} combos, total% ≥ {p90:.2f}%):")
    print(f"    Lookback range  : {lb_range}d")
    print(f"    Vol mult range  : {mult_range}×")
    print(f"    Dominant combos : {combo_freq.index[0]} ({combo_freq.iloc[0]}/{len(top10pct)} runs)"
          + (f", {combo_freq.index[1]} ({combo_freq.iloc[1]})" if len(combo_freq) > 1 else ""))

    # ── SPLIT EXIT ADVANTAGE ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SPLIT EXIT ADVANTAGE — 9:45+11am split vs 3pm exit across full grid")
    print("=" * 72)
    for entry_tag, split_combo, pm3_combo in [
        ("3pm entry",    "3pm / split",   "3pm / 3pm"),
        ("3:15pm entry", "315pm / split", "315pm / 3pm"),
    ]:
        s_df = results_df[results_df["combo"] == split_combo][
            ["lookback_days", "vol_mult", "total_return_pct"]].copy()
        p_df = results_df[results_df["combo"] == pm3_combo][
            ["lookback_days", "vol_mult", "total_return_pct"]].copy()
        m = s_df.merge(p_df, on=["lookback_days", "vol_mult"], suffixes=("_split", "_3pm"))
        if m.empty:
            continue
        pct_wins  = (m["total_return_pct_split"] > m["total_return_pct_3pm"]).mean() * 100
        avg_adv   = (m["total_return_pct_split"] - m["total_return_pct_3pm"]).mean()
        med_split = m["total_return_pct_split"].median()
        med_pm3   = m["total_return_pct_3pm"].median()
        print(f"\n  {entry_tag}:")
        print(f"    Split wins in {pct_wins:.0f}% of 192 lb×mult combos")
        print(f"    Avg split advantage : {avg_adv:+.2f}% total return")
        print(f"    Median — split: {med_split:+.2f}%   vs   3pm exit: {med_pm3:+.2f}%")

    # ── BY-COMBO SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("AVERAGE TOTAL% BY COMBO  (avg across all 192 lb×mult combos)")
    print("=" * 72)
    by_combo = (results_df.groupby("combo", sort=False)["total_return_pct"]
                .agg(["mean", "median", "min", "max"])
                .round(2))
    by_combo.columns = ["avg%", "med%", "min%", "max%"]
    by_combo = by_combo.sort_values("avg%", ascending=False)
    print(f"  {'Combo':<22}  {'Avg%':>8}  {'Med%':>8}  {'Min%':>8}  {'Max%':>8}")
    print("  " + "─" * 62)
    for combo_name, row in by_combo.iterrows():
        print(f"  {combo_name:<22}  {row['avg%']:>+8.2f}  {row['med%']:>+8.2f}  "
              f"{row['min%']:>+8.2f}  {row['max%']:>+8.2f}")

    elapsed_total = time.time() - t_total
    print(f"\nTotal sweep time: {elapsed_total:.1f}s  ({elapsed_total / 60:.1f} min)")
    print(f"Full results    : {out_xlsx}")


if __name__ == "__main__":
    main()
