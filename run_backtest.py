#!/usr/bin/env python3
"""
run_backtest.py — Backtest 3 exit scenarios on LB=36 VM=6 signals
==================================================================
Reads results/diagnostic_table.csv (built by prepare_data.py).
Entry always = 3:15pm (entry_price_315pm).

Scenarios:
  1. Standard    — 100% exit at next-day 3:00pm open
  2. Split       — 50% at next-day 9:45am open + 50% at next-day 11:00am open
  3. Split+SL    — 50% at next-day 9:45am open (fixed)
                   remaining 50% protected by 5% stop-loss from entry_price:
                     sl_price = entry_price * 0.95
                   Scan candle lows from 15:15 entry day → next-day 15:00:
                     if low <= sl_price → exit at sl_price (zero slippage)
                     else              → exit at next-day 3:00pm open

Position sizing: ≤5 signals/day → ₹1L each; ≥6 → ₹5L ÷ n. Whole shares.
No compounding anywhere. Returns are non-compounding daily-weighted sums.

Max drawdown: one continuous equity curve (base ₹5L) across the full period,
never reset — a reset would understate multi-day losing streaks.

Outputs:
  results/backtest_results.xlsx   — daily/monthly/yearly tables for all 3 scenarios
  Console summary table

Usage:
  python run_backtest.py [--diag results/diagnostic_table.csv] [--three-conditions]

--three-conditions: filter on passes_all_three (original 3 conditions, no fade filter)
                    Default: passes_all_four (all 4 conditions including fade)
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE       = Path(__file__).parent
MASTER_DIR = BASE / "master_data"
RESULTS    = BASE / "results"
IST        = "Asia/Kolkata"

DAILY_POOL    = 500_000
MAX_PER_STOCK = 100_000
SL_PCT        = 0.05

HM_915  = 555
HM_1515 = 915
HM_1500 = 900


# ── Position sizing ───────────────────────────────────────────────────────────

def _day_target(n_signals):
    return MAX_PER_STOCK if n_signals <= 5 else DAILY_POOL / n_signals


# ── Scenario 1 & 2: pure diagnostic-table backtests ──────────────────────────

def run_standard(signals):
    """100% exit at next-day 3pm open."""
    valid   = signals.dropna(subset=["entry_price_315pm", "exit_3pm_open"]).copy()
    by_date = defaultdict(list)
    for _, row in valid.iterrows():
        by_date[row["date"]].append(row)

    rows = []
    for d in sorted(by_date):
        day = by_date[d]; n = len(day); tgt = _day_target(n)
        for row in day:
            ep = float(row["entry_price_315pm"])
            xp = float(row["exit_3pm_open"])
            sh = int(tgt // ep)
            if sh == 0: continue
            rows.append({
                "date": d, "symbol": row["symbol"],
                "entry": ep, "exit": xp, "shares": sh,
                "pnl": sh * (xp - ep),
                "ret": (xp - ep) / ep * 100,
                "cap": sh * ep, "sl_hit": False,
            })
    return pd.DataFrame(rows)


def run_split(signals):
    """50% at 9:45am, 50% at 11:00am."""
    valid   = signals.dropna(subset=["entry_price_315pm", "exit_945_open", "exit_1100_open"]).copy()
    by_date = defaultdict(list)
    for _, row in valid.iterrows():
        by_date[row["date"]].append(row)

    rows = []
    for d in sorted(by_date):
        day = by_date[d]; n = len(day); tgt = _day_target(n)
        for row in day:
            ep   = float(row["entry_price_315pm"])
            x945 = float(row["exit_945_open"])
            x110 = float(row["exit_1100_open"])
            sh   = int(tgt // ep)
            if sh == 0: continue
            s1   = sh // 2; s2 = sh - s1
            pnl  = s1 * (x945 - ep) + s2 * (x110 - ep)
            ret  = 0.5 * (x945 - ep) / ep * 100 + 0.5 * (x110 - ep) / ep * 100
            rows.append({
                "date": d, "symbol": row["symbol"],
                "entry": ep, "exit_945": x945, "exit_1100": x110,
                "shares": sh, "pnl": pnl, "ret": ret,
                "cap": sh * ep, "sl_hit": False,
            })
    return pd.DataFrame(rows)


# ── Scenario 3: Split + 5% SL on second leg ──────────────────────────────────

def run_split_sl(signals):
    """
    50% at next-day 9:45am open (fixed).
    Remaining 50%: 5% SL scanned candle-by-candle from 15:15 entry day
    through next-day 15:00. If low <= sl_price → exit at sl_price.
    Otherwise → exit at next-day 3:00pm open.
    """
    valid = signals.dropna(subset=["entry_price_315pm", "exit_945_open"]).copy()

    # Group by symbol to load each parquet once
    sig_by_sym = defaultdict(list)
    for _, row in valid.iterrows():
        sig_by_sym[row["symbol"]].append(row)

    all_rows = []

    for symbol, sig_list in sig_by_sym.items():
        pq = MASTER_DIR / f"{symbol}.parquet"
        if not pq.exists():
            continue

        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"]   = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw.sort_values("timestamp").reset_index(drop=True)

        all_dates = sorted(raw["date"].unique())

        for row in sig_list:
            entry_date  = pd.Timestamp(row["date"]).date()
            ep          = float(row["entry_price_315pm"])
            x945        = float(row["exit_945_open"])
            sl_price    = ep * (1 - SL_PCT)
            fallback_3pm = float(row["exit_3pm_open"]) if pd.notna(row.get("exit_3pm_open")) else np.nan

            if np.isnan(x945): continue

            try:
                idx = all_dates.index(entry_date)
            except ValueError:
                continue
            if idx + 1 >= len(all_dates):
                continue
            next_date = all_dates[idx + 1]

            # Candles to scan: 15:15 on entry day + all candles on next day up to 15:00
            scan = raw[
                ((raw["date"] == entry_date) & (raw["hm"] >= HM_1515)) |
                ((raw["date"] == next_date)  & (raw["hm"] <= HM_1500))
            ].sort_values("timestamp")

            leg2_exit = np.nan
            sl_hit    = False

            for _, c in scan.iterrows():
                c_date = c["date"]; c_hm = int(c["hm"])
                # At or past 15:00 on next day → use open as fallback
                if c_date == next_date and c_hm >= HM_1500:
                    leg2_exit = float(c["open"])
                    break
                if float(c["low"]) <= sl_price:
                    leg2_exit = sl_price
                    sl_hit    = True
                    break

            if np.isnan(leg2_exit):
                if not np.isnan(fallback_3pm):
                    leg2_exit = fallback_3pm
                else:
                    continue

            all_rows.append({
                "entry_date": entry_date,
                "symbol":     symbol,
                "ep":         ep,
                "x945":       x945,
                "leg2":       leg2_exit,
                "sl_hit":     sl_hit,
                "row":        row,
            })

    # Now build trades with correct per-day sizing
    by_date = defaultdict(list)
    for item in all_rows:
        by_date[item["entry_date"]].append(item)

    rows = []
    for d in sorted(by_date):
        day = by_date[d]; n = len(day); tgt = _day_target(n)
        for item in day:
            ep   = item["ep"]; x945 = item["x945"]; leg2 = item["leg2"]
            sh   = int(tgt // ep)
            if sh == 0: continue
            s1   = sh // 2; s2 = sh - s1
            pnl  = s1 * (x945 - ep) + s2 * (leg2 - ep)
            ret  = 0.5 * (x945 - ep) / ep * 100 + 0.5 * (leg2 - ep) / ep * 100
            rows.append({
                "date": d, "symbol": item["symbol"],
                "entry": ep, "exit_945": x945, "exit_leg2": leg2,
                "shares": sh, "pnl": pnl, "ret": ret,
                "cap": sh * ep, "sl_hit": item["sl_hit"],
            })

    return pd.DataFrame(rows)


# ── Stats & reporting ─────────────────────────────────────────────────────────

STARTING_EQUITY = 500_000   # ₹5L baseline for equity curve


def compute_stats(trades):
    """Return summary dict for a trade log."""
    if trades.empty:
        return dict(trades=0, total_ret_pct=0, total_pnl=0,
                    win_rate=0, wins=0, losses=0,
                    max_dd_pct=0, max_dd_rs=0,
                    avg_ret=0, median_ret=0, sl_hits=0)

    daily = (trades.groupby("date")
             .agg(cap=("cap", "sum"), pnl=("pnl", "sum"))
             .reset_index())
    daily["dr"] = daily["pnl"] / daily["cap"] * 100

    total_ret = daily["dr"].sum()

    # Continuous equity curve (never reset)
    cum_pnl   = daily["pnl"].cumsum()
    equity    = STARTING_EQUITY + cum_pnl
    peak      = equity.cummax()
    dd_rs     = equity - peak
    max_dd_rs = dd_rs.min()
    max_dd_pct = (dd_rs / peak).min() * 100

    wins   = int((trades["pnl"] > 0).sum())
    losses = int((trades["pnl"] <= 0).sum())
    sl_hits = int(trades["sl_hit"].sum()) if "sl_hit" in trades.columns else 0

    return dict(
        trades     = len(trades),
        total_ret_pct = round(total_ret, 2),
        total_pnl  = round(float(trades["pnl"].sum()), 0),
        win_rate   = round(wins / len(trades) * 100, 2),
        wins       = wins,
        losses     = losses,
        max_dd_pct = round(max_dd_pct, 2),
        max_dd_rs  = round(max_dd_rs, 0),
        avg_ret    = round(float(trades["ret"].mean()), 4),
        median_ret = round(float(trades["ret"].median()), 4),
        sl_hits    = sl_hits,
    )


def make_daily_table(trades):
    if trades.empty:
        return pd.DataFrame()
    daily = (trades.groupby("date")
             .agg(n_trades=("pnl", "count"),
                  pnl=("pnl", "sum"),
                  cap=("cap", "sum"))
             .reset_index())
    daily["return_pct"]  = (daily["pnl"] / daily["cap"] * 100).round(4)
    daily["cum_pnl"]     = daily["pnl"].cumsum().round(0)
    daily["equity"]      = (STARTING_EQUITY + daily["cum_pnl"]).round(0)
    return daily.sort_values("date").reset_index(drop=True)


def make_monthly_table(trades):
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["year_month"] = pd.to_datetime(t["date"]).dt.to_period("M")
    monthly = (t.groupby("year_month")
               .agg(n_trades=("pnl", "count"),
                    pnl=("pnl", "sum"),
                    cap=("cap", "sum"))
               .reset_index())
    monthly["return_pct"] = (monthly["pnl"] / monthly["cap"] * 100).round(4)
    monthly["cum_pnl"]    = monthly["pnl"].cumsum().round(0)
    return monthly.sort_values("year_month").reset_index(drop=True)


def make_yearly_table(trades):
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["year"] = pd.to_datetime(t["date"]).dt.year
    yearly = (t.groupby("year")
              .agg(n_trades=("pnl", "count"),
                   pnl=("pnl", "sum"),
                   cap=("cap", "sum"))
              .reset_index())
    yearly["return_pct"] = (yearly["pnl"] / yearly["cap"] * 100).round(4)
    return yearly.sort_values("year").reset_index(drop=True)


def save_excel(scenario_results, out_path):
    """Write all scenario results to a multi-sheet Excel file."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, trades in scenario_results.items():
            safe = name[:31]
            if not trades.empty:
                trades.to_excel(writer, sheet_name=f"{safe} trades", index=False)
            make_daily_table(trades).to_excel(
                writer, sheet_name=f"{safe} daily",   index=False)
            make_monthly_table(trades).to_excel(
                writer, sheet_name=f"{safe} monthly", index=False)
            make_yearly_table(trades).to_excel(
                writer, sheet_name=f"{safe} yearly",  index=False)

        # Auto-width
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                max_len = max(
                    (len(str(c.value)) for c in col if c.value is not None), default=10)
                sheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 28)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", default="results/diagnostic_table.csv")
    parser.add_argument("--three-conditions", action="store_true",
                        help="Filter on passes_all_three (no fade filter)")
    args = parser.parse_args()

    diag_path = BASE / args.diag
    if not diag_path.exists():
        print(f"ERROR: {diag_path} not found — run prepare_data.py first.")
        sys.exit(1)

    print(f"Loading {diag_path} …")
    diag = pd.read_csv(diag_path, parse_dates=["date"])

    # ── Validation: show 3-condition baseline first ───────────────────────────
    sig3 = diag[diag["passes_all_three"]].copy()
    print(f"\n  3-condition signals : {len(sig3):,}")

    t_std3  = run_standard(sig3)
    s_std3  = compute_stats(t_std3)
    t_spl3  = run_split(sig3)
    s_spl3  = compute_stats(t_spl3)

    print()
    print("=" * 72)
    print("  VALIDATION — Original 3 conditions (no fade filter)")
    print("  Expected: 3,497 trades, split exit +787.85%")
    print("=" * 72)
    print(f"  {'Scenario':<28}  {'Trades':>7}  {'Total%':>9}  {'Win%':>7}  {'Avg%':>8}")
    print("  " + "─" * 60)
    print(f"  {'Standard (3pm exit)':<28}  {s_std3['trades']:>7,}  "
          f"{s_std3['total_ret_pct']:>+9.2f}  {s_std3['win_rate']:>7.2f}  "
          f"{s_std3['avg_ret']:>+8.4f}")
    print(f"  {'Split (9:45+11am)':<28}  {s_spl3['trades']:>7,}  "
          f"{s_spl3['total_ret_pct']:>+9.2f}  {s_spl3['win_rate']:>7.2f}  "
          f"{s_spl3['avg_ret']:>+8.4f}")

    match = "✓  MATCH" if s_spl3["trades"] == 3497 and abs(s_spl3["total_ret_pct"] - 787.85) < 0.1 else "✗  MISMATCH"
    print(f"\n  Validation: {match}")

    # ── Main backtest: 4 conditions ───────────────────────────────────────────
    condition_col = "passes_all_three" if args.three_conditions else "passes_all_four"
    label         = "3 conditions" if args.three_conditions else "4 conditions (+ fade ≤5%)"
    signals       = diag[diag[condition_col]].copy()
    print(f"\n  4-condition signals : {len(signals):,}")

    print(f"\nRunning 3 scenarios on {label} …")

    t0 = time.time()
    t_std  = run_standard(signals)
    t_spl  = run_split(signals)
    print("  Standard + Split done. Running Split+SL (needs parquet scan) …")
    t_sl   = run_split_sl(signals)
    elapsed = time.time() - t0

    s_std  = compute_stats(t_std)
    s_spl  = compute_stats(t_spl)
    s_sl   = compute_stats(t_sl)

    sl_n    = int(t_sl["sl_hit"].sum()) if not t_sl.empty else 0
    sl_pct  = sl_n / len(t_sl) * 100 if not t_sl.empty else 0

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"  RESULTS — {label}  |  Entry 3:15pm")
    print("=" * 90)
    hdr = f"  {'Scenario':<28}  {'Trades':>7}  {'Total%':>9}  {'Win%':>7}  {'Avg%':>8}  {'MaxDD%':>8}  {'MaxDD₹':>10}  {'SL hits':>9}"
    print(hdr)
    print("  " + "─" * 86)

    def row(label, s, sl_info="N/A"):
        return (f"  {label:<28}  {s['trades']:>7,}  "
                f"{s['total_ret_pct']:>+9.2f}  {s['win_rate']:>7.2f}  "
                f"{s['avg_ret']:>+8.4f}  {s['max_dd_pct']:>+8.2f}  "
                f"{s['max_dd_rs']:>+10,.0f}  {sl_info:>9}")

    print(row("Standard (3pm exit)",      s_std))
    print(row("Split (9:45+11am)",         s_spl))
    print(row("Split+SL5% (fallback 3pm)", s_sl,
              f"{sl_n} ({sl_pct:.1f}%)"))
    print("  " + "─" * 86)
    print()

    # ── Impact of adding fade filter ──────────────────────────────────────────
    if not args.three_conditions:
        sig3_spl = s_spl3
        sig4_spl = s_spl
        removed  = len(sig3) - len(signals)
        print("  FADE FILTER IMPACT (split exit, 3 → 4 conditions):")
        print(f"  {'':28}  {'3 conditions':>14}  {'4 conditions':>14}")
        print(f"  {'Signals':28}  {len(sig3):>14,}  {len(signals):>14,}")
        print(f"  {'Trades':28}  {sig3_spl['trades']:>14,}  {sig4_spl['trades']:>14,}")
        print(f"  {'Total return %':28}  {sig3_spl['total_ret_pct']:>+14.2f}  {sig4_spl['total_ret_pct']:>+14.2f}")
        print(f"  {'Max drawdown %':28}  {sig3_spl['max_dd_pct']:>+14.2f}  {sig4_spl['max_dd_pct']:>+14.2f}")
        print(f"  {'Win rate %':28}  {sig3_spl['win_rate']:>14.2f}  {sig4_spl['win_rate']:>14.2f}")
        print(f"  Signals removed by fade filter: {removed:,} ({removed/len(sig3)*100:.1f}%)")
        print()

    # ── Save Excel ────────────────────────────────────────────────────────────
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "backtest_results.xlsx"
    scenario_results = {
        "1_Standard":  t_std,
        "2_Split":     t_spl,
        "3_SplitSL5":  t_sl,
    }
    save_excel(scenario_results, out_path)
    print(f"  Results saved → {out_path}")
    print(f"  Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
