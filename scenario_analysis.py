#!/usr/bin/env python3
"""
scenario_analysis.py — Reusable analysis functions for ad-hoc scenario testing
================================================================================
Reads trade logs from results/ (produced by run_backtest.py).
All functions are importable and accept a trades DataFrame as their first argument.

If a scenario needs a different vol_window/vol_mult or new exit time, call
build_diagnostic_table() and run_single_backtest() directly as imported
functions — no subprocess overhead.

Quick-start examples:
  from scenario_analysis import load_trades, filter_fade_from_high, plot_high_low_distribution
  trades = load_trades("315pm_split")
  filtered = filter_fade_from_high(trades, min_fade_pct=2.0)
  print(compute_summary(filtered))
  plot_high_low_distribution(trades, day="exit")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "results"

# Column groups used across functions
_CUMHIGH_COLS_ENTRY = [f"entry_day_cumhigh_{t}" for t in ["10","11","12","13","14","15"]]
_CUMLOW_COLS_ENTRY  = [f"entry_day_cumlow_{t}"  for t in ["10","11","12","13","14","15"]]
_CUMHIGH_COLS_EXIT  = [f"exit_day_cumhigh_{t}"  for t in ["10","11","12","13","14","15"]]
_CUMLOW_COLS_EXIT   = [f"exit_day_cumlow_{t}"   for t in ["10","11","12","13","14","15"]]
_CHECKPOINT_HOURS   = [10, 11, 12, 13, 14, 15]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_trades(label_or_path: str, results_dir: str | Path = RESULTS_DIR) -> pd.DataFrame:
    """
    Load a trade log by file_label (e.g. '315pm_split') or by full path.

    Available file labels:
      3pm_split, 3pm_3pm, 3pm_230pm, 3pm_245pm
      315pm_split, 315pm_3pm, 315pm_230pm, 315pm_245pm
    """
    path = Path(label_or_path)
    if not path.exists():
        path = Path(results_dir) / f"trades_{label_or_path}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Trade file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def compute_summary(trades: pd.DataFrame, label: str = "") -> dict:
    """
    Compute the standard backtest summary stats (non-compounding returns).

    Returns a dict with: total_trades, total_return_pct, win_rate_pct,
    positive_trades, negative_trades, avg_ret_per_trade_pct, median_ret_per_trade_pct
    """
    if trades.empty:
        return {"combo": label, "total_trades": 0, "total_return_pct": 0.0,
                "win_rate_pct": 0.0, "positive_trades": 0, "negative_trades": 0,
                "avg_ret_per_trade_pct": 0.0, "median_ret_per_trade_pct": 0.0}
    daily = (trades.groupby("date")
             .agg(capital_deployed=("capital_allocated", "sum"),
                  total_pnl       =("pnl",              "sum"))
             .reset_index())
    daily["daily_return_pct"] = daily["total_pnl"] / daily["capital_deployed"] * 100
    wins = (trades["pnl"] > 0).sum()
    return {
        "combo":                    label,
        "total_trades":             len(trades),
        "total_return_pct":         round(daily["daily_return_pct"].sum(), 4),
        "win_rate_pct":             round(wins / len(trades) * 100, 2),
        "positive_trades":          int(wins),
        "negative_trades":          int((trades["pnl"] <= 0).sum()),
        "avg_ret_per_trade_pct":    round(trades["return_pct"].mean(), 4),
        "median_ret_per_trade_pct": round(trades["return_pct"].median(), 4),
    }


def print_summary(trades: pd.DataFrame, label: str = "") -> None:
    s = compute_summary(trades, label)
    print(f"\n{'─'*50}")
    if label:
        print(f"  {label}")
    print(f"  Trades      : {s['total_trades']:,}")
    print(f"  Total return: {s['total_return_pct']:+.4f}%")
    print(f"  Win rate    : {s['win_rate_pct']:.2f}%  "
          f"({s['positive_trades']}W / {s['negative_trades']}L)")
    print(f"  Avg / Median: {s['avg_ret_per_trade_pct']:+.4f}% / "
          f"{s['median_ret_per_trade_pct']:+.4f}%")
    print(f"{'─'*50}")


# ── Filter: fade from entry-day high ─────────────────────────────────────────

def filter_fade_from_high(trades: pd.DataFrame, min_fade_pct: float = 2.0) -> pd.DataFrame:
    """
    Keep trades where the entry day's high is at least min_fade_pct% above the
    3:15pm close — i.e. the stock made a big intraday high and faded before entry.

    Fade = (entry_day_high - entry_day_close_1515) / entry_day_high × 100

    Requires columns: entry_day_high_price, entry_day_close_1515
    """
    df = trades.dropna(subset=["entry_day_high_price", "entry_day_close_1515"]).copy()
    df["fade_pct"] = ((df["entry_day_high_price"] - df["entry_day_close_1515"])
                      / df["entry_day_high_price"] * 100)
    return df[df["fade_pct"] >= min_fade_pct].copy()


def sweep_fade_from_high(trades: pd.DataFrame,
                         thresholds: list[float] | None = None) -> pd.DataFrame:
    """
    Sweep multiple fade_pct thresholds and return a comparison table.
    thresholds: list of % fade values to test (default 0–5% in 0.5% steps).
    """
    if thresholds is None:
        thresholds = [round(x * 0.5, 1) for x in range(0, 11)]
    rows = []
    for th in thresholds:
        filtered = filter_fade_from_high(trades, th)
        s = compute_summary(filtered, f"fade≥{th}%")
        rows.append(s)
    return pd.DataFrame(rows)


# ── Filter: post-entry drop on exit day ──────────────────────────────────────

def filter_max_exit_day_drawdown(trades: pd.DataFrame,
                                  max_drop_pct: float = 5.0) -> pd.DataFrame:
    """
    Keep trades where the exit day's cumulative low (at any checkpoint through 3pm)
    did NOT drop more than max_drop_pct% below the entry price.

    Uses exit_day_cumlow_* columns (10, 11, 12, 13, 14, 15).
    """
    available_cols = [c for c in _CUMLOW_COLS_EXIT if c in trades.columns]
    if not available_cols:
        raise ValueError("No exit_day_cumlow_* columns found in trade log.")
    df = trades.copy()
    df["_min_exit_low"] = df[available_cols].min(axis=1)
    df["_max_drop_pct"] = (df["entry_price"] - df["_min_exit_low"]) / df["entry_price"] * 100
    return df[df["_max_drop_pct"] <= max_drop_pct].drop(
        columns=["_min_exit_low", "_max_drop_pct"])


def sweep_exit_day_drawdown(trades: pd.DataFrame,
                             thresholds: list[float] | None = None) -> pd.DataFrame:
    """Sweep multiple max-drop thresholds on exit day."""
    if thresholds is None:
        thresholds = [round(x * 0.5, 1) for x in range(2, 21)]
    rows = []
    for th in thresholds:
        filtered = filter_max_exit_day_drawdown(trades, th)
        s = compute_summary(filtered, f"max_drop≤{th}%")
        rows.append(s)
    return pd.DataFrame(rows)


# ── Chart: time-of-day high/low distribution ──────────────────────────────────

def plot_high_low_distribution(trades: pd.DataFrame,
                                day: str = "entry",
                                save_path: str | Path | None = None) -> None:
    """
    Bar chart showing at which 15-min candle the intraday high and low occur.

    Parameters
    ----------
    trades   : trade log DataFrame
    day      : "entry" or "exit"
    save_path: path to save PNG; if None, saves to results/chart_{day}_day_hl_dist.png
    """
    if day == "entry":
        high_col, low_col = "entry_day_high_time", "entry_day_low_time"
        title = "Entry-day: time of intraday high / low"
    else:
        high_col, low_col = "exit_day_high_time", "exit_day_low_time"
        title = "Exit-day: time of intraday high / low"

    for col in [high_col, low_col]:
        if col not in trades.columns:
            print(f"Column {col!r} not found — skipping {day}-day HL distribution chart.")
            return

    time_labels = [f"{h:02d}:{m:02d}" for h in range(9, 16)
                   for m in (15, 30, 45) if not (h == 15 and m > 15)
                   if not (h == 9 and m < 15)]
    # Canonical 15-min session slots 09:15 → 15:15
    slots = [f"{h:02d}:{m:02d}" for h in range(9, 16)
             for m in range(0, 60, 15)
             if (h, m) >= (9, 15) and (h, m) <= (15, 15)]

    high_counts = trades[high_col].value_counts().reindex(slots, fill_value=0)
    low_counts  = trades[low_col ].value_counts().reindex(slots, fill_value=0)

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(slots))
    width = 0.4
    ax.bar(x - width/2, high_counts.values, width, label="High", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, low_counts.values,  width, label="Low",  color="#F44336", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(slots, rotation=90, fontsize=7)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("# trades")
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"chart_{day}_day_hl_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ── Chart: cumulative high/low build-up through the day ───────────────────────

def plot_cumulative_range_buildup(trades: pd.DataFrame,
                                   day: str = "entry",
                                   save_path: str | Path | None = None) -> None:
    """
    Line chart: average cumulative high and low (vs open) at each hourly
    checkpoint, showing how the intraday range builds through the session.

    Checkpoints: 10:00, 11:00, 12:00, 13:00, 14:00, 15:00
    """
    if day == "entry":
        high_cols = _CUMHIGH_COLS_ENTRY
        low_cols  = _CUMLOW_COLS_ENTRY
        open_col  = "entry_day_open_915"
        title     = "Entry-day: average cumulative high/low vs 09:15 open"
    else:
        high_cols = _CUMHIGH_COLS_EXIT
        low_cols  = _CUMLOW_COLS_EXIT
        open_col  = "next_day_open_915" if "next_day_open_915" in trades.columns else None
        title     = "Exit-day: average cumulative high/low vs 09:15 open"

    available_high = [c for c in high_cols if c in trades.columns]
    available_low  = [c for c in low_cols  if c in trades.columns]
    if not available_high:
        print("No cumulative high/low columns found — skipping chart.")
        return

    n_checkpoints = min(len(available_high), len(available_low), len(_CHECKPOINT_HOURS))
    hours = _CHECKPOINT_HOURS[:n_checkpoints]
    labels = [f"{h}:00" for h in hours]

    if open_col and open_col in trades.columns:
        ref = trades[open_col]
        avg_high_pct = [(trades[available_high[i]] - ref) / ref * 100 for i in range(n_checkpoints)]
        avg_low_pct  = [(trades[available_low [i]] - ref) / ref * 100 for i in range(n_checkpoints)]
        y_high = [s.mean() for s in avg_high_pct]
        y_low  = [s.mean() for s in avg_low_pct ]
        ylabel = "Avg % vs 09:15 open"
    else:
        y_high = [trades[available_high[i]].mean() for i in range(n_checkpoints)]
        y_low  = [trades[available_low [i]].mean() for i in range(n_checkpoints)]
        ylabel = "Avg price"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(labels, y_high, "o-", color="#2196F3", label="Cumulative High")
    ax.plot(labels, y_low,  "o-", color="#F44336", label="Cumulative Low")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time checkpoint")
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"chart_{day}_day_range_buildup.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ── Chart: volume crossing time ───────────────────────────────────────────────

def plot_volume_crossing_time(trades: pd.DataFrame,
                               day: str = "entry",
                               threshold_pcts: list[float] | None = None,
                               save_path: str | Path | None = None) -> None:
    """
    For each trade, estimate at which checkpoint hour the cumulative volume
    first exceeded a threshold percentage of the full-day volume.

    Uses entry_day_cumhigh_* or exit_day_cumhigh_* as a proxy for volume
    accumulation (strictly, cumhigh measures price but this function is
    a distribution-of-activity chart, not a volume cross).

    For actual volume crossing analysis, use the checkpoint CSVs directly.
    """
    if threshold_pcts is None:
        threshold_pcts = [50.0, 75.0, 90.0]

    if day == "entry":
        cumhigh_cols = [c for c in _CUMHIGH_COLS_ENTRY if c in trades.columns]
        vol_col      = "entry_day_fullday_vol" if "entry_day_fullday_vol" in trades.columns else None
        title        = "Entry-day: when does cumulative high exceed threshold?"
    else:
        cumhigh_cols = [c for c in _CUMHIGH_COLS_EXIT if c in trades.columns]
        vol_col      = "exit_day_fullday_vol"  if "exit_day_fullday_vol"  in trades.columns else None
        title        = "Exit-day: when does cumulative high exceed threshold?"

    if not cumhigh_cols:
        print("No cumulative high columns found — skipping volume crossing chart.")
        return

    n = min(len(cumhigh_cols), len(_CHECKPOINT_HOURS))
    hours = _CHECKPOINT_HOURS[:n]
    labels = [f"{h}:00" for h in hours]

    # For each threshold: find the first checkpoint where cumhigh >= threshold% of day high
    day_high_col = "entry_day_high_price" if day == "entry" else "exit_day_high_price"
    if day_high_col not in trades.columns:
        print(f"Column {day_high_col!r} missing — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for th_pct, color in zip(threshold_pcts, colors):
        crossing_hour = []
        for _, row in trades.iterrows():
            day_h = row[day_high_col]
            if not pd.notna(day_h) or day_h == 0:
                continue
            target = day_h * (th_pct / 100.0)
            crossed_at = None
            for i, col in enumerate(cumhigh_cols[:n]):
                if pd.notna(row.get(col)) and row[col] >= target:
                    crossed_at = hours[i]
                    break
            if crossed_at is not None:
                crossing_hour.append(crossed_at)

        counts = pd.Series(crossing_hour).value_counts().reindex(hours, fill_value=0)
        pcts   = counts / counts.sum() * 100
        ax.plot(labels, pcts.values, "o-", color=color, label=f"≥{th_pct:.0f}% of day high")

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("% of trades crossing threshold at this hour")
    ax.set_xlabel("Checkpoint hour")
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_DIR / f"chart_{day}_volume_crossing.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ── Custom scenario: on-the-fly backtest with different parameters ────────────

def run_custom_scenario(vol_window: int = 30,
                         vol_mult: float = 7.0,
                         entry: str = "315pm",
                         exit_strategy: str = "split",
                         filters: list[Callable] | None = None,
                         verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """
    Build a fresh diagnostic table and run one backtest combo, then apply
    any optional filter functions. Uses imported functions — no subprocess.

    Parameters
    ----------
    vol_window     : rolling lookback window
    vol_mult       : volume ratio threshold
    entry          : "3pm" or "315pm"
    exit_strategy  : "split", "3pm", "230pm", or "245pm"
    filters        : list of filter callables (trades_df → filtered_trades_df)
    verbose        : print progress

    Returns
    -------
    (summary_dict, trades_df)
    """
    from prepare_data import build_diagnostic_table
    from run_backtest import run_single_backtest, UNIQUE_COMBOS

    if verbose:
        print(f"Building diagnostic table: vol_window={vol_window}, vol_mult={vol_mult}x …")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        diag = build_diagnostic_table(
            vol_window=vol_window, vol_mult=vol_mult,
            output_path=tmp_path, verbose=verbose)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    entry_col_map = {
        "3pm":   "entry_price_3pm",
        "315pm": "entry_price_315pm",
    }
    exit_map = {
        "split": ("split",  ("exit_945_open", "exit_1100_open")),
        "3pm":   ("single", "exit_3pm_open"),
        "230pm": ("single", "exit_1430_open"),
        "245pm": ("single", "exit_1445_open"),
    }

    if entry not in entry_col_map:
        raise ValueError(f"entry must be one of {list(entry_col_map)}")
    if exit_strategy not in exit_map:
        raise ValueError(f"exit_strategy must be one of {list(exit_map)}")

    signals    = diag[diag["passes_all_three"]].copy()
    exit_type, exit_cols = exit_map[exit_strategy]
    label      = f"{entry} entry / {exit_strategy} exit"

    if verbose:
        print(f"Running: {label} …")

    trades = run_single_backtest(signals, entry_col_map[entry], exit_type, exit_cols)

    if filters:
        for fn in filters:
            trades = fn(trades)
            if verbose:
                print(f"  After {fn.__name__}: {len(trades):,} trades remain")

    summary = compute_summary(trades, label)
    if verbose:
        print_summary(trades, label)

    return summary, trades


# ── CLI: run all standard charts on a trade file ─────────────────────────────

def _cli_charts(args):
    trades = load_trades(args.trades)
    print(f"Loaded {len(trades):,} trades from {args.trades}")
    print_summary(trades)
    plot_high_low_distribution(trades, day="entry")
    plot_high_low_distribution(trades, day="exit")
    plot_cumulative_range_buildup(trades, day="entry")
    plot_cumulative_range_buildup(trades, day="exit")
    plot_volume_crossing_time(trades, day="entry")
    plot_volume_crossing_time(trades, day="exit")
    print(f"\nAll charts saved to {RESULTS_DIR}/")


def _cli_sweep_fade(args):
    trades = load_trades(args.trades)
    result = sweep_fade_from_high(trades)
    print(result.to_string(index=False))


def _cli_sweep_drawdown(args):
    trades = load_trades(args.trades)
    result = sweep_exit_day_drawdown(trades)
    print(result.to_string(index=False))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scenario analysis on backtest trade logs")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("charts", help="Generate all standard charts for a trade file")
    p.add_argument("trades", help="Trade file label (e.g. 315pm_split) or full path")

    p2 = sub.add_parser("fade-sweep", help="Sweep fade-from-high thresholds")
    p2.add_argument("trades", help="Trade file label or path")

    p3 = sub.add_parser("drawdown-sweep", help="Sweep exit-day max-drawdown thresholds")
    p3.add_argument("trades", help="Trade file label or path")

    p4 = sub.add_parser("custom", help="Run a backtest with custom parameters")
    p4.add_argument("--vol-window", type=int,   default=30)
    p4.add_argument("--vol-mult",   type=float, default=7.0)
    p4.add_argument("--entry",      type=str,   default="315pm",
                    choices=["3pm", "315pm"])
    p4.add_argument("--exit",       type=str,   default="split",
                    choices=["split", "3pm", "230pm", "245pm"])

    args = parser.parse_args()

    if args.cmd == "charts":
        _cli_charts(args)
    elif args.cmd == "fade-sweep":
        _cli_sweep_fade(args)
    elif args.cmd == "drawdown-sweep":
        _cli_sweep_drawdown(args)
    elif args.cmd == "custom":
        args.exit_strategy = args.exit
        run_custom_scenario(args.vol_window, args.vol_mult, args.entry, args.exit_strategy)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
