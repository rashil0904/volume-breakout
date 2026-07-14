#!/usr/bin/env python3
"""
run_backtest.py — Run all 10 entry × exit combinations from one diagnostic table
==================================================================================
Reads results/diagnostic_table.csv (built by prepare_data.py) and runs:

  ENTRIES (2):    3pm (15:00 open)  |  3:15pm (15:15 open)
  EXITS   (5):    split 9:45+11am   |  3pm   |  2:30pm  |  2:45pm  |  3pm*
                  (* 3pm repeated so the 2:30/2:45/3pm comparison sits in one block)

  = 10 result rows in summary, 8 unique trade files in results/

Position sizing (unchanged across all combinations):
  ≤5 signals/day → ₹1,00,000 per stock (≤₹5L total)
  6+ signals/day → ₹5,00,000 ÷ n per stock
  shares = floor(target ÷ entry_price); zero-share trades excluded

Returns (non-compounding):
  daily_return_pct = sum(pnl) / sum(capital) × 100 per day
  total_return     = arithmetic sum of all daily_return_pct values

CLI:
  python run_backtest.py [--diag results/diagnostic_table.csv]

Importable:
  from run_backtest import run_single_backtest, run_all_combos
  summary, trades = run_all_combos(diag_df)
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "results"
DAILY_POOL    = 500_000
MAX_PER_STOCK = 100_000

# ── Combination definitions ───────────────────────────────────────────────────
# Each tuple: (display_label, entry_col, exit_type, exit_col_or_cols, file_label)
# exit_type "single": exit_col_or_cols is a string column name
# exit_type "split":  exit_col_or_cols is (col_945, col_1100)
COMBOS = [
    # Entry 3pm
    ("3pm entry  / split exit (9:45 + 11am)",
     "entry_price_3pm",   "split",  ("exit_945_open", "exit_1100_open"), "3pm_split"),
    ("3pm entry  / 3pm exit",
     "entry_price_3pm",   "single", "exit_3pm_open",                    "3pm_3pm"),
    ("3pm entry  / 2:30pm exit",
     "entry_price_3pm",   "single", "exit_1430_open",                   "3pm_230pm"),
    ("3pm entry  / 2:45pm exit",
     "entry_price_3pm",   "single", "exit_1445_open",                   "3pm_245pm"),
    ("3pm entry  / 3pm exit",                         # duplicate row for comparison block
     "entry_price_3pm",   "single", "exit_3pm_open",                    "3pm_3pm"),
    # Entry 3:15pm
    ("3:15pm entry / split exit (9:45 + 11am)",
     "entry_price_315pm", "split",  ("exit_945_open", "exit_1100_open"), "315pm_split"),
    ("3:15pm entry / 3pm exit",
     "entry_price_315pm", "single", "exit_3pm_open",                    "315pm_3pm"),
    ("3:15pm entry / 2:30pm exit",
     "entry_price_315pm", "single", "exit_1430_open",                   "315pm_230pm"),
    ("3:15pm entry / 2:45pm exit",
     "entry_price_315pm", "single", "exit_1445_open",                   "315pm_245pm"),
    ("3:15pm entry / 3pm exit",                       # duplicate row for comparison block
     "entry_price_315pm", "single", "exit_3pm_open",                    "315pm_3pm"),
]

# Unique combinations to actually compute (dedup on file_label)
_UNIQUE = {}
for combo in COMBOS:
    label = combo[4]
    if label not in _UNIQUE:
        _UNIQUE[label] = combo
UNIQUE_COMBOS = list(_UNIQUE.values())


# ── Core backtest logic ───────────────────────────────────────────────────────

def _get_required_exit_cols(exit_type, exit_col_or_cols):
    if exit_type == "split":
        return list(exit_col_or_cols)
    return [exit_col_or_cols]


def run_single_backtest(signals, entry_col, exit_type, exit_col_or_cols,
                        daily_pool=DAILY_POOL, max_per_stock=MAX_PER_STOCK):
    """
    Run one backtest variant on a filtered signals DataFrame.

    Parameters
    ----------
    signals : pd.DataFrame
        Rows where passes_all_three is True.
    entry_col : str
        Column name for entry price.
    exit_type : str
        "single" or "split".
    exit_col_or_cols : str or tuple
        For "single": column name of exit price.
        For "split":  (col_945, col_1100) — 50% each, floor/ceil split.
    daily_pool, max_per_stock : int
        Position sizing constants.

    Returns
    -------
    pd.DataFrame
        Trade-level log with pnl, capital_allocated, etc.
    """
    exit_cols = _get_required_exit_cols(exit_type, exit_col_or_cols)

    # Drop rows missing entry or any required exit price
    valid = signals.dropna(subset=[entry_col] + exit_cols).copy()

    by_date = defaultdict(list)
    for _, row in valid.iterrows():
        by_date[row["date"]].append(row)

    exec_rows = []

    for entry_date in sorted(by_date):
        day_sigs = by_date[entry_date]
        n        = len(day_sigs)
        target   = max_per_stock if n <= 5 else daily_pool / n

        for row in day_sigs:
            ep     = float(row[entry_col])
            shares = int(target // ep)
            if shares == 0:
                continue

            actual_cap = shares * ep

            if exit_type == "split":
                col_945, col_1100   = exit_col_or_cols
                x945                = float(row[col_945])
                x1100               = float(row[col_1100])
                shares_first        = shares // 2
                shares_second       = shares - shares_first
                pnl                 = (shares_first  * (x945  - ep) +
                                       shares_second * (x1100 - ep))
                blended_exit        = (x945 + x1100) / 2
                ret_pct             = (blended_exit - ep) / ep * 100
                exit_detail         = {
                    "exit_945":          round(x945,   4),
                    "exit_1100":         round(x1100,  4),
                    "blended_exit":      round(blended_exit, 4),
                    "shares_first":      shares_first,
                    "shares_second":     shares_second,
                    "exit_price":        round(blended_exit, 4),
                }
            else:
                xp          = float(row[exit_col_or_cols])
                pnl         = shares * (xp - ep)
                ret_pct     = (xp - ep) / ep * 100
                exit_detail = {"exit_price": round(xp, 4)}

            # Enriched diagnostics pulled from diagnostic table
            o915  = row.get("today_open_915",   np.nan)
            c1515 = row.get("today_close_1515", np.nan)
            ed_h  = row.get("today_high",       np.nan)
            nd_h  = row.get("next_day_high",    np.nan)
            nd_o  = row.get("next_day_open_915",np.nan)
            nd_c  = row.get("next_day_close_1515", np.nan)

            def _pct(a, b):
                return round((a - b) / b * 100, 4) if (
                    pd.notna(a) and pd.notna(b) and b != 0) else np.nan

            TAGS = [("10",0),("11",0),("12",0),("13",0),("14",0),("15",0)]

            r = {
                "date":               entry_date,
                "symbol":             row["symbol"],
                "entry_price":        round(ep, 4),
                "shares_bought":      shares,
                "target_allocation":  round(target, 2),
                "capital_allocated":  round(actual_cap, 2),
                "pnl":                round(pnl, 2),
                "return_pct":         round(ret_pct, 4),
                "n_signals_day":      n,
                **exit_detail,
                "prev_day_vwap_close_used": round(float(row["prev_day_vwap_close"]), 4),
                "vol_ratio":          round(float(row["volume_ratio"]), 2),
                "mcap_cr":            round(float(row["market_cap_value"]), 1),
                "pre_entry_move_pct":     _pct(ep,   o915),
                "post_entry_move_pct":    _pct(c1515, ep),
                "entry_day_open_915":     round(float(o915),  4) if pd.notna(o915)  else np.nan,
                "entry_day_close_1515":   round(float(c1515), 4) if pd.notna(c1515) else np.nan,
                "entry_day_high_price":   round(float(ed_h),  4) if pd.notna(ed_h)  else np.nan,
                "entry_day_high_time":    row.get("today_high_time", ""),
                "entry_day_low_price":    round(float(row.get("today_low", np.nan)), 4) if pd.notna(row.get("today_low")) else np.nan,
                "entry_day_low_time":     row.get("today_low_time", ""),
                "exit_day_high_price":    round(float(nd_h),  4) if pd.notna(nd_h)  else np.nan,
                "exit_day_high_time":     row.get("next_day_high_time", ""),
                "exit_day_low_price":     round(float(row.get("next_day_low", np.nan)), 4) if pd.notna(row.get("next_day_low")) else np.nan,
                "exit_day_low_time":      row.get("next_day_low_time", ""),
                "entry_day_range_h_minus_o_pct": _pct(ed_h,  o915),
                "entry_day_range_c_minus_o_pct": _pct(c1515, o915),
                "exit_day_range_h_minus_o_pct":  _pct(nd_h,  nd_o),
                "exit_day_range_c_minus_o_pct":  _pct(nd_c,  nd_o),
                **{f"entry_day_cumlow_{t}":  round(float(row[f"today_cumlow_{t}"]),  4) if pd.notna(row.get(f"today_cumlow_{t}"))  else np.nan for t, _ in TAGS},
                **{f"entry_day_cumhigh_{t}": round(float(row[f"today_cumhigh_{t}"]), 4) if pd.notna(row.get(f"today_cumhigh_{t}")) else np.nan for t, _ in TAGS},
                **{f"exit_day_cumlow_{t}":   round(float(row[f"next_day_cumlow_{t}"]),  4) if pd.notna(row.get(f"next_day_cumlow_{t}"))  else np.nan for t, _ in TAGS},
                **{f"exit_day_cumhigh_{t}":  round(float(row[f"next_day_cumhigh_{t}"]), 4) if pd.notna(row.get(f"next_day_cumhigh_{t}")) else np.nan for t, _ in TAGS},
            }
            exec_rows.append(r)

    trades = (pd.DataFrame(exec_rows)
              .sort_values(["date", "symbol"])
              .reset_index(drop=True))
    if not trades.empty:
        trades["date"] = pd.to_datetime(trades["date"])
    return trades


def _compute_summary(trades, label):
    """Compute summary stats for one trade set."""
    if trades.empty:
        return {
            "combo": label, "total_trades": 0, "total_return_pct": 0,
            "win_rate_pct": 0, "positive_trades": 0, "negative_trades": 0,
            "avg_ret_per_trade_pct": 0, "median_ret_per_trade_pct": 0,
        }
    daily = (trades.groupby("date")
             .agg(capital_deployed=("capital_allocated", "sum"),
                  total_pnl       =("pnl",              "sum"))
             .reset_index())
    daily["daily_return_pct"] = daily["total_pnl"] / daily["capital_deployed"] * 100
    total_return = daily["daily_return_pct"].sum()
    wins         = (trades["pnl"] > 0).sum()
    return {
        "combo":                  label,
        "total_trades":           len(trades),
        "total_return_pct":       round(total_return, 4),
        "win_rate_pct":           round(wins / len(trades) * 100, 2),
        "positive_trades":        int(wins),
        "negative_trades":        int((trades["pnl"] <= 0).sum()),
        "avg_ret_per_trade_pct":  round(trades["return_pct"].mean(), 4),
        "median_ret_per_trade_pct": round(trades["return_pct"].median(), 4),
    }


def run_all_combos(diag_df, results_dir=None, save_trades=True, verbose=True):
    """
    Run all 10 entry × exit combinations.

    Parameters
    ----------
    diag_df : pd.DataFrame
        Full diagnostic table from build_diagnostic_table().
    results_dir : Path or str, optional
        Where to save trade CSVs. Defaults to results/.
    save_trades : bool
        Whether to write per-combination trade CSVs.
    verbose : bool
        Print progress.

    Returns
    -------
    summary_df : pd.DataFrame
        10-row comparison table.
    trades_dict : dict
        {file_label: trades_df} for each of the 8 unique combinations.
    """
    out_dir = Path(results_dir) if results_dir else RESULTS_DIR
    out_dir.mkdir(exist_ok=True)

    signals = diag_df[diag_df["passes_all_three"]].copy()
    if verbose:
        print(f"\n  passes_all_three signals: {len(signals):,}")

    # Run each unique combination once
    computed = {}
    for label, entry_col, exit_type, exit_cols, file_label in UNIQUE_COMBOS:
        if verbose:
            print(f"  Running: {label} …", end=" ", flush=True)
        trades = run_single_backtest(signals, entry_col, exit_type, exit_cols)
        computed[file_label] = trades
        if save_trades and not trades.empty:
            path = out_dir / f"trades_{file_label}.csv"
            trades.to_csv(path, index=False)
        if verbose:
            print(f"{len(trades):,} trades")

    # Build 10-row summary (COMBOS order, with 3pm exit repeated for each entry)
    summary_rows = []
    for label, entry_col, exit_type, exit_cols, file_label in COMBOS:
        trades = computed[file_label]
        summary_rows.append(_compute_summary(trades, label))

    summary_df = pd.DataFrame(summary_rows)
    if save_trades:
        summary_df.to_csv(out_dir / "summary.csv", index=False)

    return summary_df, computed


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all 10 entry×exit backtests from a diagnostic table")
    parser.add_argument("--diag", type=str,
                        default=str(RESULTS_DIR / "diagnostic_table.csv"),
                        help="Path to diagnostic_table.csv")
    args = parser.parse_args()

    diag_path = Path(args.diag)
    if not diag_path.exists():
        print(f"ERROR: diagnostic table not found: {diag_path}")
        print("Run prepare_data.py first.")
        raise SystemExit(1)

    print("=" * 72)
    print("run_backtest.py  —  10 entry × exit combinations")
    print(f"Reading  : {diag_path}")
    print(f"Output   : {RESULTS_DIR}/")
    print("=" * 72)

    print(f"\nLoading diagnostic table …")
    diag = pd.read_csv(diag_path, parse_dates=["date"])
    print(f"  {len(diag):,} rows  |  {diag['symbol'].nunique():,} symbols")
    print(f"  passes_all_three=True: {diag['passes_all_three'].sum():,}")

    print("\nRunning combinations:")
    summary, trades_dict = run_all_combos(diag, verbose=True)

    print("\n" + "=" * 72)
    print(f"{'#':>2}  {'Entry / Exit Combo':<40}  {'Trades':>7}  "
          f"{'Total%':>9}  {'WinRate':>8}  {'Avg%':>8}  {'Med%':>8}")
    print("    " + "─" * 68)
    for i, row in summary.iterrows():
        print(f"{i+1:>2}  {row['combo']:<40}  {int(row['total_trades']):>7,}  "
              f"{row['total_return_pct']:>+9.2f}  {row['win_rate_pct']:>7.2f}%  "
              f"{row['avg_ret_per_trade_pct']:>+8.4f}  {row['median_ret_per_trade_pct']:>+8.4f}")
    print("=" * 72)
    print(f"\nTrade files + summary saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
