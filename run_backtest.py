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
from collections import defaultdict, Counter
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

# Fixed capital base (₹5L) used as the denominator for monthly/quarterly/yearly
# returns — reuses the daily pool size so there is a single source of truth.
CAPITAL_BASE  = DAILY_POOL

HM_915  = 555
HM_1515 = 915
HM_1500 = 900

# ── Dynamic-window entry variant (additive; separate from the fixed 3pm/3:15pm path) ──
PHASE1_ALLOC   = 50_000                            # Rs reserved per stock on first qualify
DYN_VOL_MULT   = 6                                 # volume multiple for the dynamic checks
DYN_RET_THRESH = 0.05                              # 5% vs previous-day VWAP close
DYN_CHECK_HMS  = [840, 855, 870, 885, 900, 915]    # 14:00,14:15,14:30,14:45,15:00,15:15


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
    monthly["return_pct"] = (monthly["pnl"] / CAPITAL_BASE * 100).round(4)
    monthly["cum_pnl"]    = monthly["pnl"].cumsum().round(0)
    return monthly.sort_values("year_month").reset_index(drop=True)


def make_quarterly_table(trades):
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["quarter"] = pd.to_datetime(t["date"]).dt.to_period("Q")
    quarterly = (t.groupby("quarter")
                 .agg(n_trades=("pnl", "count"),
                      pnl=("pnl", "sum"),
                      cap=("cap", "sum"))
                 .reset_index())
    quarterly["return_pct"] = (quarterly["pnl"] / CAPITAL_BASE * 100).round(4)
    quarterly["cum_pnl"]    = quarterly["pnl"].cumsum().round(0)
    return quarterly.sort_values("quarter").reset_index(drop=True)


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
    yearly["return_pct"] = (yearly["pnl"] / CAPITAL_BASE * 100).round(4)
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
            make_quarterly_table(trades).to_excel(
                writer, sheet_name=f"{safe} quarterly", index=False)
            make_yearly_table(trades).to_excel(
                writer, sheet_name=f"{safe} yearly",  index=False)

        # Auto-width
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                max_len = max(
                    (len(str(c.value)) for c in col if c.value is not None), default=10)
                sheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 28)


# ── Dynamic-window entry variant ──────────────────────────────────────────────
# Multi-check entry (14:00–15:15) with two-phase capital allocation and a 15:15
# top-up. Entirely separate from the fixed 3pm/3:15pm entry functions above.

def _hm_str(hm):
    return f"{hm // 60:02d}:{hm % 60:02d}"


def run_dynamic_window(diag):
    """
    Multi-check dynamic-window entry with capital top-up.

    Universe = every (symbol, date) row in the diagnostic table (a row existing
    there already means the mcap ₹1,500–5,000 Cr band condition is satisfied).

    For each trading day, six check times (14:00–15:15) are evaluated in order.
    A stock enters at the OPEN of the FIRST check time T where all 3 conditions hold:
      1. mcap in band          (row present in diag)
      2. cum vol 09:15→(T-1 candle) ≥ DYN_VOL_MULT × avg full-day volume
                                (avg reused from the existing avg_30day_fullday_volume)
      3. (open_T − prev_vwap_close) / prev_vwap_close ≥ DYN_RET_THRESH
    One entry per stock per day.

    Phase 1 (as-you-go): each newly-qualifying stock reserves ₹50,000 from a shared
      ₹5L daily pool and buys whole shares at that candle's open. If the remaining
      pool < ₹50,000, the stock is skipped entirely.
    Phase 2 (15:15 top-up): leftover = 500000 − 50000×n_entered. Re-check condition 3
      at the 15:15 open for every entered stock; split the leftover equally among
      those still passing and buy extra whole shares at the 15:15 open. A blended
      (share-weighted) entry price represents the combined position.

    Exit: reuses the split-exit P&L (50% at next-day 9:45, 50% at next-day 11:00),
      taking the exit prices from the diagnostic table.

    Returns (trades_df, meta) where meta reports top-up statistics.
    """
    need = ["avg_30day_fullday_volume", "prev_day_vwap_close",
            "exit_945_open", "exit_1100_open"]
    missing = [c for c in need if c not in diag.columns]
    if missing:
        raise KeyError(f"diagnostic table missing columns for dynamic window: {missing}")

    qual_rows = []
    for sym, diag_sym in diag.groupby("symbol"):
        pq = MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"]   = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[(raw["hm"] >= HM_915) & (raw["hm"] <= HM_1515)]

        opens_by_T    = {T: raw[raw["hm"] == T].groupby("date")["open"].last().to_dict()
                         for T in DYN_CHECK_HMS}
        cumvol_before = {T: raw[raw["hm"] < T].groupby("date")["volume"].sum().to_dict()
                         for T in DYN_CHECK_HMS}

        for _, dr in diag_sym.iterrows():
            d          = dr["date"].date() if hasattr(dr["date"], "date") else dr["date"]
            avg_vol    = dr["avg_30day_fullday_volume"]
            prev_close = dr["prev_day_vwap_close"]
            x945       = dr["exit_945_open"]
            x110       = dr["exit_1100_open"]
            if pd.isna(avg_vol) or avg_vol <= 0 or pd.isna(prev_close) or prev_close <= 0:
                continue
            if pd.isna(x945) or pd.isna(x110):
                continue  # no valid next-day exit → not backtestable (mirror dropna)

            first_T, entry_open = None, None
            for T in DYN_CHECK_HMS:
                cv = cumvol_before[T].get(d)
                op = opens_by_T[T].get(d)
                if cv is None or op is None:
                    continue
                if cv >= DYN_VOL_MULT * avg_vol and (op - prev_close) / prev_close >= DYN_RET_THRESH:
                    first_T, entry_open = T, op
                    break
            if first_T is None:
                continue

            op1515 = opens_by_T[HM_1515].get(d, np.nan)
            qual_rows.append({
                "symbol": sym, "date": d, "first_T": first_T,
                "entry_open": float(entry_open),
                "open_1515": float(op1515) if op1515 == op1515 else np.nan,
                "prev_close": float(prev_close),
                "x945": float(x945), "x110": float(x110),
            })

    if not qual_rows:
        return pd.DataFrame(), {"n_days": 0, "avg_topups": 0.0, "total_topups": 0,
                                "trades": 0}

    quals = pd.DataFrame(qual_rows)
    out, topups_per_day = [], []

    for d, day in quals.groupby("date"):
        day = day.sort_values(["first_T", "symbol"])   # chronological, then FCFS by symbol

        # ── Phase 1: as-you-go ₹50k allocation from the shared pool ──
        pool, entered = DAILY_POOL, []
        for _, q in day.iterrows():
            if pool < PHASE1_ALLOC:
                continue
            sh1 = int(PHASE1_ALLOC // q["entry_open"])
            if sh1 == 0:
                continue  # entry price > ₹50k → can't buy a whole share; no pool reserved
            pool -= PHASE1_ALLOC
            entered.append({"q": q, "sh1": sh1, "entry1": q["entry_open"],
                            "sh2": 0, "entry2": 0.0})

        # ── Phase 2: 15:15 top-up of the leftover pool ──
        leftover = pool  # == DAILY_POOL − PHASE1_ALLOC × len(entered)
        passers = [e for e in entered
                   if not np.isnan(e["q"]["open_1515"])
                   and (e["q"]["open_1515"] - e["q"]["prev_close"]) / e["q"]["prev_close"] >= DYN_RET_THRESH]
        if leftover > 0 and passers:
            per_stock = leftover / len(passers)
            for e in passers:
                op15 = e["q"]["open_1515"]
                sh2  = int(per_stock // op15)
                if sh2 > 0:
                    e["sh2"], e["entry2"] = sh2, op15

        topups_per_day.append(sum(1 for e in entered if e["sh2"] > 0))

        # ── Build combined-position trades (reuse split-exit P&L) ──
        for e in entered:
            q, sh1, sh2 = e["q"], e["sh1"], e["sh2"]
            total = sh1 + sh2
            if total == 0:
                continue
            cap     = sh1 * e["entry1"] + sh2 * e["entry2"]
            blended = cap / total
            s_first, s_second = total // 2, total - total // 2
            pnl = s_first * (q["x945"] - blended) + s_second * (q["x110"] - blended)
            ret = 0.5 * (q["x945"] - blended) / blended * 100 + \
                  0.5 * (q["x110"] - blended) / blended * 100
            out.append({
                "date": d, "symbol": q["symbol"],
                "entry_time": _hm_str(int(q["first_T"])),
                "entry": round(blended, 4),
                "shares": total, "shares_p1": sh1, "shares_p2": sh2,
                "got_topup": bool(sh2 > 0),
                "exit_945": round(q["x945"], 4), "exit_1100": round(q["x110"], 4),
                "pnl": pnl, "ret": ret, "cap": cap, "sl_hit": False,
            })

    trades = pd.DataFrame(out)
    if not trades.empty:
        trades = trades.sort_values(["date", "symbol"]).reset_index(drop=True)
        trades["date"] = pd.to_datetime(trades["date"])

    n_days = len(topups_per_day)
    meta = {
        "n_days":       n_days,
        "avg_topups":   (sum(topups_per_day) / n_days) if n_days else 0.0,
        "total_topups": sum(topups_per_day),
        "trades":       len(trades),
    }
    return trades, meta


# ── Dynamic-window entry variant #2: same-day cull-and-redeploy ────────────────
# Same 6-check entry as run_dynamic_window, but at 15:15 it culls earlier entrants
# that no longer meet the 5% condition (same-day exit) and redeploys the freed +
# leftover capital into the survivors. Entirely separate from the fixed-entry path
# and from run_dynamic_window above.

def _gather_dynamic_quals(diag):
    """
    Shared 6-check qualification pass (14:00–15:15) used by the cull variant.
    Returns one row per (symbol, date) that first-qualifies, with the fields the
    allocation logic needs. Identical condition logic to run_dynamic_window;
    factored out so the cull variant can reuse it without touching that function.
    """
    need = ["avg_30day_fullday_volume", "prev_day_vwap_close",
            "exit_945_open", "exit_1100_open"]
    missing = [c for c in need if c not in diag.columns]
    if missing:
        raise KeyError(f"diagnostic table missing columns for dynamic window: {missing}")

    qual_rows = []
    for sym, diag_sym in diag.groupby("symbol"):
        pq = MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"]   = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[(raw["hm"] >= HM_915) & (raw["hm"] <= HM_1515)]

        opens_by_T    = {T: raw[raw["hm"] == T].groupby("date")["open"].last().to_dict()
                         for T in DYN_CHECK_HMS}
        cumvol_before = {T: raw[raw["hm"] < T].groupby("date")["volume"].sum().to_dict()
                         for T in DYN_CHECK_HMS}

        for _, dr in diag_sym.iterrows():
            d          = dr["date"].date() if hasattr(dr["date"], "date") else dr["date"]
            avg_vol    = dr["avg_30day_fullday_volume"]
            prev_close = dr["prev_day_vwap_close"]
            x945       = dr["exit_945_open"]
            x110       = dr["exit_1100_open"]
            if pd.isna(avg_vol) or avg_vol <= 0 or pd.isna(prev_close) or prev_close <= 0:
                continue
            if pd.isna(x945) or pd.isna(x110):
                continue  # no valid next-day exit → not backtestable

            first_T, entry_open = None, None
            for T in DYN_CHECK_HMS:
                cv = cumvol_before[T].get(d)
                op = opens_by_T[T].get(d)
                if cv is None or op is None:
                    continue
                if cv >= DYN_VOL_MULT * avg_vol and (op - prev_close) / prev_close >= DYN_RET_THRESH:
                    first_T, entry_open = T, op
                    break
            if first_T is None:
                continue

            op1515 = opens_by_T[HM_1515].get(d, np.nan)
            qual_rows.append({
                "symbol": sym, "date": d, "first_T": first_T,
                "entry_open": float(entry_open),
                "open_1515": float(op1515) if op1515 == op1515 else np.nan,
                "prev_close": float(prev_close),
                "x945": float(x945), "x110": float(x110),
            })

    return pd.DataFrame(qual_rows)


def run_dynamic_window_cull(diag):
    """
    Multi-check dynamic-window entry with a same-day cull-and-redeploy at 15:15.

    Phase 1 (14:00–15:15): each newly-qualifying stock reserves ₹50,000 from the
      shared ₹5L pool and buys whole shares at its first-qualifying candle's open
      (skipped if the pool has < ₹50,000 left, or if ₹50k buys < 1 share). A brand-
      new signal that first qualifies at 15:15 also gets this standard ₹50k tranche.

    Phase 2 at 15:15 (cull-and-redeploy):
      (b) Every stock that entered at 14:00–15:00 is re-checked against condition 3
          using the 15:15 open. Failures are EXITED same-day at the 15:15 open
          (exit_type="same_day_cull"). A brand-new 15:15 entrant is NOT culled
          (it just passed at that same instant) and stays active.
      (c) redeploy pool = leftover Phase-1 pool + Σ(shares × entry) of culled stocks.
      (d) The redeploy pool is split equally among the still-active stocks and used
          to buy an extra tranche at the 15:15 open (floor division; skipped for a
          stock if its share cannot be afforded). A share-weighted blended entry
          price represents the combined position.

    Survivors exit overnight via the split exit (50% next-day 9:45, 50% 11:00,
    exit_type="overnight_split"). Both cull and overnight trades are returned,
    tagged by exit_type.

    Returns (trades_df, meta).
    """
    quals = _gather_dynamic_quals(diag)
    if quals.empty:
        return pd.DataFrame(), {"n_days": 0, "entries": 0, "culls": 0,
                                "cull_pct": 0.0, "avg_redeploy": 0.0, "trades": 0}

    out, culls_per_day, redeploy_used_per_day = [], [], []

    for d, day in quals.groupby("date"):
        day = day.sort_values(["first_T", "symbol"])   # chronological, then FCFS by symbol

        # ── Phase 1: as-you-go ₹50k allocation (all 6 checks, incl. new 15:15 entrants) ──
        pool, entered = DAILY_POOL, []
        for _, q in day.iterrows():
            if pool < PHASE1_ALLOC:
                continue
            sh1 = int(PHASE1_ALLOC // q["entry_open"])
            if sh1 == 0:
                continue
            pool -= PHASE1_ALLOC
            entered.append({"q": q, "sh1": sh1, "entry1": q["entry_open"],
                            "first_T": int(q["first_T"]), "sh2": 0, "entry2": 0.0,
                            "is_1515": int(q["first_T"]) == HM_1515})

        leftover = pool  # nominal leftover: DAILY_POOL − 50000 × len(entered)

        # ── Phase 2 (b): cull early entrants that fail the 5% check at 15:15 ──
        culled, active = [], []
        for e in entered:
            if e["is_1515"]:
                active.append(e)            # brand-new 15:15 entrant → never culled
                continue
            op15, pc = e["q"]["open_1515"], e["q"]["prev_close"]
            if np.isnan(op15):
                active.append(e)            # no 15:15 candle → cannot cull, keep active
            elif (op15 - pc) / pc >= DYN_RET_THRESH:
                active.append(e)
            else:
                culled.append(e)

        # ── Phase 2 (c): redeploy pool = leftover + capital freed by culls ──
        freed    = sum(e["sh1"] * e["entry1"] for e in culled)
        redeploy = leftover + freed

        # ── Phase 2 (d): split redeploy equally among survivors, top-up at 15:15 open ──
        active_priced = [e for e in active if not np.isnan(e["q"]["open_1515"])]
        if redeploy > 0 and active_priced:
            per_stock = redeploy / len(active_priced)
            for e in active_priced:
                op15 = e["q"]["open_1515"]
                sh2  = int(per_stock // op15)
                if sh2 > 0:
                    e["sh2"], e["entry2"] = sh2, op15

        culls_per_day.append(len(culled))
        redeploy_used_per_day.append(sum(e["sh2"] * e["entry2"] for e in active))

        # ── Same-day cull trades ──
        for e in culled:
            ep, sh, xp = e["entry1"], e["sh1"], e["q"]["open_1515"]
            out.append({
                "date": d, "symbol": e["q"]["symbol"],
                "entry_time": _hm_str(e["first_T"]), "entry": round(ep, 4),
                "shares": sh, "shares_p1": sh, "shares_p2": 0, "got_topup": False,
                "exit_type": "same_day_cull", "exit_price": round(xp, 4),
                "exit_945": np.nan, "exit_1100": np.nan,
                "pnl": sh * (xp - ep), "ret": (xp - ep) / ep * 100,
                "cap": sh * ep, "sl_hit": False,
            })

        # ── Overnight survivors (split exit) ──
        for e in active:
            q, sh1, sh2 = e["q"], e["sh1"], e["sh2"]
            total = sh1 + sh2
            if total == 0:
                continue
            cap     = sh1 * e["entry1"] + sh2 * e["entry2"]
            blended = cap / total
            s_first, s_second = total // 2, total - total // 2
            x945, x110 = q["x945"], q["x110"]
            pnl = s_first * (x945 - blended) + s_second * (x110 - blended)
            ret = 0.5 * (x945 - blended) / blended * 100 + \
                  0.5 * (x110 - blended) / blended * 100
            out.append({
                "date": d, "symbol": q["symbol"],
                "entry_time": _hm_str(e["first_T"]), "entry": round(blended, 4),
                "shares": total, "shares_p1": sh1, "shares_p2": sh2,
                "got_topup": bool(sh2 > 0),
                "exit_type": "overnight_split", "exit_price": np.nan,
                "exit_945": round(x945, 4), "exit_1100": round(x110, 4),
                "pnl": pnl, "ret": ret, "cap": cap, "sl_hit": False,
            })

    trades = pd.DataFrame(out)
    if not trades.empty:
        trades = trades.sort_values(["date", "symbol"]).reset_index(drop=True)
        trades["date"] = pd.to_datetime(trades["date"])

    n_days  = len(culls_per_day)
    n_culls = sum(culls_per_day)
    meta = {
        "n_days":       n_days,
        "entries":      len(trades),
        "culls":        n_culls,
        "cull_pct":     (n_culls / len(trades) * 100) if len(trades) else 0.0,
        "avg_redeploy": (sum(redeploy_used_per_day) / n_days) if n_days else 0.0,
        "trades":       len(trades),
    }
    return trades, meta


# ── Capital-allocation variant: day-high risk-capped sizing ───────────────────
# Only the position-sizing step changes. Entry/exit/qualification are untouched.
# Exit = split (9:45 + 11am). A trade is "day-high" if its 3:15pm entry price is
# >= the running max high from 09:15 through the 15:00 candle (today_cumhigh_15).

CAPITAL_TARGET_GRID = [100_000, 105_000, 110_000, 115_000, 120_000, 125_000]


def _day_alloc(entries, is_high, X):
    """Allocate the ₹5L pool for one day. Returns (alloc_array, category)."""
    n = len(entries)
    nh = int(is_high.sum()); no = n - nh
    base_eq = DAILY_POOL / n
    alloc = np.empty(n, dtype=float)
    if n < 5:
        alloc[is_high] = X
        if no > 0:
            # non-day-high trades are NEVER allocated more than ₹1L; leftover pool goes unused
            alloc[~is_high] = min((DAILY_POOL - nh * X) / no, MAX_PER_STOCK)
        return alloc, "unscaled_lt5"
    # n_total >= 5 : day-high get (equal split + ₹5,000); others share the remainder equally,
    # still capped at ₹1L. (X is not used here — the boost on busy days is a flat ₹5,000.)
    if no == 0:
        alloc[:] = base_eq                                   # all day-high → can't boost, flat
        return alloc, "flat_no_other"
    dh = base_eq + 5000
    remaining = DAILY_POOL - nh * dh
    if remaining <= 0:
        alloc[:] = base_eq                                   # boost unaffordable → flat
        return alloc, "flat_fallback"
    alloc[is_high] = dh
    alloc[~is_high] = min(remaining / no, MAX_PER_STOCK)
    return alloc, "boost5000_ge5"


def _capital_run(day_groups, mode):
    """mode = target X (int) or 'baseline' (pure equal split). Returns (trades_df, cats)."""
    rows, cats = [], Counter()
    for d, ent, ish, x945, x110 in day_groups:
        n = len(ent)
        if mode == "baseline":
            alloc, cat = np.full(n, DAILY_POOL / n), "baseline"
        else:
            alloc, cat = _day_alloc(ent, ish, mode)
        cats[cat] += 1
        shares = np.floor(alloc / ent).astype(np.int64)
        for i in range(n):
            sh = int(shares[i])
            if sh == 0:
                continue
            ep = ent[i]; s1 = sh // 2; s2 = sh - s1
            rows.append({
                "date": d, "cap": sh * ep,
                "pnl": s1 * (x945[i] - ep) + s2 * (x110[i] - ep),
                "ret": 0.5 * (x945[i] - ep) / ep * 100 + 0.5 * (x110[i] - ep) / ep * 100,
            })
    return pd.DataFrame(rows), cats


def run_capital_allocation_sweep(diag):
    """6 target-level runs + a pure-equal-split baseline row, split exit."""
    sig = diag[diag["passes_all_three"]].dropna(
        subset=["entry_price_315pm", "exit_945_open", "exit_1100_open"]).copy()
    ch = sig["today_cumhigh_15"].values.astype(float)
    sig["day_high"] = np.where(np.isnan(ch), False,
                               sig["entry_price_315pm"].values.astype(float) >= ch)

    day_groups, nh_list, no_list = [], [], []
    for d, g in sig.groupby("date"):
        ent  = g["entry_price_315pm"].values.astype(float)
        ish  = g["day_high"].values.astype(bool)
        x945 = g["exit_945_open"].values.astype(float)
        x110 = g["exit_1100_open"].values.astype(float)
        day_groups.append((d, ent, ish, x945, x110))
        nh_list.append(int(ish.sum())); no_list.append(len(ent) - int(ish.sum()))

    avg_nh, avg_no = float(np.mean(nh_list)), float(np.mean(no_list))
    days_ge5 = sum(1 for _, e, _, _, _ in day_groups if len(e) >= 5)
    days_lt5 = len(day_groups) - days_ge5

    def _metrics_row(label, trades, cats):
        s = compute_stats(trades)
        return {
            "target_capital": label,
            "n_trades": len(trades),
            "win_rate_pct": s["win_rate"],
            "avg_return_per_trade_pct": s["avg_ret"],
            "median_return_per_trade_pct": s["median_ret"],
            "total_return_fixedbase_pct": round(float(trades["pnl"].sum()) / CAPITAL_BASE * 100, 4),
            "total_return_sumofdaily_pct": s["total_ret_pct"],
            "days_unscaled_lt5": cats.get("unscaled_lt5", 0),
            "days_scaledown_ge5": cats.get("boost5000_ge5", 0) + cats.get("flat_fallback", 0) + cats.get("flat_no_other", 0),
            "days_flat_fallback": cats.get("flat_fallback", 0) + cats.get("flat_no_other", 0),
        }

    rows = []
    for X in CAPITAL_TARGET_GRID:
        trades, cats = _capital_run(day_groups, X)
        rows.append(_metrics_row(X, trades, cats))
    summary = pd.DataFrame(rows).sort_values(
        "total_return_fixedbase_pct", ascending=False).reset_index(drop=True)

    tb, cb = _capital_run(day_groups, "baseline")
    base_row = _metrics_row("baseline_equal_split", tb, cb)
    for k in ("days_unscaled_lt5", "days_scaledown_ge5", "days_flat_fallback"):
        base_row[k] = "-"

    full = pd.concat([summary, pd.DataFrame([base_row])], ignore_index=True)
    return full, avg_nh, avg_no, days_ge5, days_lt5


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", default="results/diagnostic_table.csv")
    parser.add_argument("--three-conditions", action="store_true",
                        help="Filter on passes_all_three (no fade filter)")
    parser.add_argument("--dynamic-window", action="store_true",
                        help="Run the multi-check dynamic-window entry variant "
                             "(separate output file; leaves the fixed-entry path untouched)")
    parser.add_argument("--dynamic-window-cull", action="store_true",
                        help="Run the dynamic-window entry with same-day cull-and-redeploy "
                             "at 15:15 (separate output file; fixed-entry path untouched)")
    parser.add_argument("--capital-sweep", action="store_true",
                        help="Day-high risk-capped capital-allocation sweep (6 target levels + "
                             "equal-split baseline; split exit; sizing-only change)")
    args = parser.parse_args()

    diag_path = BASE / args.diag
    if not diag_path.exists():
        print(f"ERROR: {diag_path} not found — run prepare_data.py first.")
        sys.exit(1)

    # ── Dynamic-window variant: additive, early-return; fixed path below is unchanged ──
    if args.dynamic_window:
        print(f"Loading {diag_path} …")
        diag = pd.read_csv(diag_path, parse_dates=["date"])
        print(f"  {len(diag):,} diagnostic rows  |  {diag['symbol'].nunique():,} symbols")
        print("\nRunning DYNAMIC-WINDOW entry variant "
              "(6 checks 14:00–15:15, ₹50k as-you-go + 15:15 top-up) …")
        t0 = time.time()
        dyn, meta = run_dynamic_window(diag)
        s = compute_stats(dyn)
        elapsed = time.time() - t0

        RESULTS.mkdir(exist_ok=True)
        out_path = RESULTS / "backtest_results_dynamic_window.xlsx"
        save_excel({"Dynamic": dyn}, out_path)

        print("\n" + "=" * 72)
        print("  DYNAMIC-WINDOW RESULTS  |  entry 14:00–15:15, split exit (9:45 + 11am)")
        print("=" * 72)
        print(f"  Total trades              : {meta['trades']:,}")
        print(f"  Trading days with entries : {meta['n_days']:,}")
        print(f"  Total Phase-2 top-ups     : {meta['total_topups']:,}")
        print(f"  Avg top-ups per day       : {meta['avg_topups']:.2f}")
        print(f"  Total return % (non-comp) : {s['total_ret_pct']:+.2f}")
        print(f"  Win rate %                : {s['win_rate']:.2f}")
        print(f"  Avg return / trade %      : {s['avg_ret']:+.4f}")
        print(f"  Max drawdown %            : {s['max_dd_pct']:+.2f}")
        print("=" * 72)
        print(f"  Results saved → {out_path}")
        print(f"  Runtime: {elapsed:.1f}s")
        return

    # ── Capital-allocation sweep (day-high risk-capped sizing): additive, early-return ──
    if args.capital_sweep:
        print(f"Loading {diag_path} …")
        diag = pd.read_csv(diag_path, parse_dates=["date"])
        print("\nRunning DAY-HIGH CAPITAL-ALLOCATION SWEEP (6 target levels + baseline, split exit) …")
        full, avg_nh, avg_no, days_ge5, days_lt5 = run_capital_allocation_sweep(diag)

        outdir = RESULTS / "capital_alloc_sweep"; outdir.mkdir(parents=True, exist_ok=True)
        full.to_csv(outdir / "capital_alloc_sweep.csv", index=False)
        with pd.ExcelWriter(outdir / "capital_alloc_sweep.xlsx", engine="openpyxl") as w:
            full.to_excel(w, sheet_name="Capital_sweep", index=False)

        print("\n" + "=" * 122)
        print("DAY-HIGH CAPITAL-ALLOCATION SWEEP — split exit (9:45+11am)  |  6 targets ranked + equal-split baseline")
        print("=" * 122)
        print(f"  Avg day-high trades/day : {avg_nh:.2f}   |   Avg other trades/day : {avg_no:.2f}")
        print(f"  Days n_total>=5 (scale-down path) : {days_ge5:,}   |   Days n_total<5 (full-X unscaled) : {days_lt5:,}")
        print("-" * 122)
        hdr = (f"  {'target':>20}  {'Trades':>7}  {'Win%':>7}  {'Avg%':>8}  {'Med%':>8}  "
               f"{'TotRet(5L)%':>12}  {'TotRet(daily)%':>15}  {'d<5':>5}  {'d>=5':>6}  {'flat':>5}")
        print(hdr); print("  " + "-" * 118)
        for _, r in full.iterrows():
            print(f"  {str(r['target_capital']):>20}  {r['n_trades']:>7,}  {r['win_rate_pct']:>7.2f}  "
                  f"{r['avg_return_per_trade_pct']:>8.4f}  {r['median_return_per_trade_pct']:>8.4f}  "
                  f"{r['total_return_fixedbase_pct']:>12.2f}  {r['total_return_sumofdaily_pct']:>15.2f}  "
                  f"{str(r['days_unscaled_lt5']):>5}  {str(r['days_scaledown_ge5']):>6}  {str(r['days_flat_fallback']):>5}")
        print("=" * 122)
        print(f"  Saved → {outdir}")
        return

    # ── Dynamic-window cull-and-redeploy variant: additive, early-return ──
    if args.dynamic_window_cull:
        print(f"Loading {diag_path} …")
        diag = pd.read_csv(diag_path, parse_dates=["date"])
        print(f"  {len(diag):,} diagnostic rows  |  {diag['symbol'].nunique():,} symbols")
        print("\nRunning DYNAMIC-WINDOW CULL-AND-REDEPLOY variant "
              "(6 checks 14:00–15:15, ₹50k as-you-go, 15:15 cull + redeploy) …")
        t0 = time.time()
        dyn, meta = run_dynamic_window_cull(diag)
        s = compute_stats(dyn)
        elapsed = time.time() - t0

        RESULTS.mkdir(exist_ok=True)
        out_path = RESULTS / "backtest_results_dynamic_cull.xlsx"
        save_excel({"Dynamic_cull": dyn}, out_path)

        print("\n" + "=" * 72)
        print("  DYNAMIC CULL-AND-REDEPLOY  |  entry 14:00–15:15")
        print("=" * 72)
        print(f"  Total entries               : {meta['entries']:,}")
        print(f"  Same-day culls              : {meta['culls']:,}  ({meta['cull_pct']:.1f}%)")
        print(f"  Overnight survivors         : {meta['entries'] - meta['culls']:,}")
        print(f"  Trading days with entries   : {meta['n_days']:,}")
        print(f"  Avg capital redeployed/day  : ₹{meta['avg_redeploy']:,.0f}")
        print(f"  Total return % (non-comp)   : {s['total_ret_pct']:+.2f}")
        print(f"  Win rate %                  : {s['win_rate']:.2f}")
        print(f"  Avg return / trade %        : {s['avg_ret']:+.4f}")
        print(f"  Max drawdown %              : {s['max_dd_pct']:+.2f}")
        print("=" * 72)
        print(f"  Results saved → {out_path}")
        print(f"  Runtime: {elapsed:.1f}s")
        return

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

    # ── Main backtest: 3 conditions (fade filter deleted) ─────────────────────
    condition_col = "passes_all_three"          # fade (4th) filter removed — always 3 conditions
    label         = "3 conditions (mcap + volume + return; no fade)"
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
