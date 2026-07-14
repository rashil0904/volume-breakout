#!/usr/bin/env python3
"""
prepare_data.py — Build diagnostic_table.csv
=============================================
Reads from master_data/<SYMBOL>.parquet (built by data_loading.py).
Outputs one row per (symbol, trading-day) with BOTH entry price candidates
and ALL exit-day opens needed by run_backtest.py.

Signal conditions (evaluated at 3:00 PM each trading day):
  1. Market cap ₹1,500–5,000 Cr  (closest preceding NSE semi-annual snapshot)
  2. Cumulative volume 09:15–14:45 ≥ vol_mult × rolling-avg of full-day volume
       · full-day window 09:15–15:15; zero-vol days excluded; strict min_periods
  3. OPEN of 15:00 candle ≥ 5% above prev_day_vwap_close
       · prev_day_vwap_close = VWAP of prev day's 15:00 + 15:15 candles
         (typical_price = (H+L+C)/3 per candle, weighted by volume)

CLI:
  python prepare_data.py [--vol-window 30] [--vol-mult 7] [--output path]

Importable:
  from prepare_data import build_diagnostic_table
  df = build_diagnostic_table(vol_window=36, vol_mult=6.0)
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from bisect import bisect_right
import time

BASE      = Path(__file__).parent
MASTER_DIR = BASE / "master_data"
MDIR      = BASE / "mcap_cache"
RESULTS   = BASE / "results"
IST       = "Asia/Kolkata"

# HM encoding: hour*60 + minute
HM_915  = 555   # 09:15
HM_930  = 570   # 09:30
HM_945  = 585   # 09:45
HM_1100 = 660   # 11:00
HM_1430 = 870   # 14:30
HM_1445 = 885   # 14:45
HM_1500 = 900   # 15:00
HM_1515 = 915   # 15:15

MCAP_MIN_CR = 1_500
MCAP_MAX_CR = 5_000

# Cumulative intraday checkpoints (for high/low tracking)
CHECKPOINTS = [("10", 600), ("11", 660), ("12", 720), ("13", 780), ("14", 840), ("15", 900)]

SNAPS = [
    ("2022-03-31", "mcap_2022-03-31.xlsx",  "MCAP31032022_1.xlsx"),
    ("2022-12-31", "mcap_2022-12-31.xlsx",  "MCAP31122022_3.xlsx"),
    ("2023-03-31", "mcap_2023-03-31.xlsx",  "MCAP31032023_0.xlsx"),
    ("2023-12-31", "mcap_2023-12-31.xlsx",  "MCAP31122023_0.xlsx"),
    ("2024-03-28", "mcap_2024-03-28.xlsx",  "MCAP28032024.xlsx"),
    ("2024-12-31", "mcap_2024-12-31.xlsx",  "Average MCAP_July2024ToDecember 2024 (1).xlsx"),
    ("2025-12-31", "mcap_2025-12-31.xlsx",  "Average_MCAP_July2025ToDecember2025_20260102201101.xlsx"),
]


def _load_mcap_snapshots():
    snap_dates    = []
    snap_eligible = []
    snap_labels   = []
    for date_str, cache_file, source_file in SNAPS:
        path = MDIR / cache_file
        if not path.exists():
            print(f"  MISSING mcap snapshot: {cache_file}")
            continue
        df = pd.read_excel(path, header=None, skiprows=1, usecols=[1, 3])
        df.columns = ["symbol", "mcap_lakhs"]
        df["symbol"]  = df["symbol"].astype(str).str.strip().str.upper()
        df["mcap_cr"] = pd.to_numeric(df["mcap_lakhs"], errors="coerce") / 100
        df = df.dropna(subset=["mcap_cr"])
        eligible  = df[(df["mcap_cr"] >= MCAP_MIN_CR) & (df["mcap_cr"] <= MCAP_MAX_CR)]
        elig_dict = eligible.set_index("symbol")["mcap_cr"].to_dict()
        snap_dates.append(pd.Timestamp(date_str))
        snap_eligible.append(elig_dict)
        snap_labels.append(f"{date_str} ({source_file})")
        print(f"  {date_str}: {len(df):,} total → {len(elig_dict):,} eligible (₹1,500–5,000 Cr)")
    return snap_dates, snap_eligible, snap_labels


def get_arr(series_dict, dates, default=np.nan):
    return np.array([series_dict.get(d, default) for d in dates], dtype=float)


def shift_next(arr):
    """shift_next[i] = arr[i+1]; last element = NaN."""
    out       = np.empty(len(arr), dtype=float)
    out[-1]   = np.nan
    out[:-1]  = arr[1:]
    return out


def hm_to_time(hm_float):
    if np.isnan(hm_float):
        return ""
    hm = int(hm_float)
    return f"{hm//60:02d}:{hm%60:02d}"


def build_diagnostic_table(vol_window=30, vol_mult=7.0, output_path=None, verbose=True, save_csv=True):
    """
    Build the diagnostic table from master_data/ parquet files.

    Parameters
    ----------
    vol_window : int
        Rolling lookback for full-day volume average (strict min_periods).
    vol_mult : float
        Minimum volume ratio threshold for the signal.
    output_path : str or Path, optional
        Where to write the CSV. Defaults to results/diagnostic_table.csv.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        The complete diagnostic table (all symbols, all days).
    """
    RESULTS.mkdir(exist_ok=True)
    out_path = Path(output_path) if output_path else RESULTS / "diagnostic_table.csv"

    if verbose:
        print("=" * 68)
        print(f"prepare_data.py  —  vol_window={vol_window}  vol_mult={vol_mult}x")
        print(f"Reading from : {MASTER_DIR}/")
        print(f"Output       : {out_path if save_csv else '(not saved to disk)'}")
        print("=" * 68)

    # ── STEP 1: MCAP snapshots ────────────────────────────────────────────────
    if verbose:
        print("\nSTEP 1 — Loading MCAP snapshots")
    snap_dates, snap_eligible, snap_labels = _load_mcap_snapshots()

    ever_eligible: set = set()
    for ed in snap_eligible:
        ever_eligible.update(ed.keys())

    parquet_files = {f.stem: f for f in sorted(MASTER_DIR.glob("*.parquet"))}
    available     = sorted(ever_eligible & parquet_files.keys())
    missing       = sorted(ever_eligible - parquet_files.keys())

    if verbose:
        print(f"\nUnique symbols eligible in ≥1 snapshot : {len(ever_eligible)}")
        print(f"Have parquet data : {len(available)}  |  No parquet : {len(missing)}")
        if missing:
            print(f"  Missing: {missing[:15]}{'…' if len(missing) > 15 else ''}")

    # ── STEP 2: Build diagnostic rows ─────────────────────────────────────────
    if verbose:
        print(f"\nSTEP 2 — Processing {len(available)} symbols")
        print("=" * 68)

    t0          = time.time()
    first_write = True
    total_rows  = 0
    total_sigs  = 0
    all_chunks  = []

    for sym_idx, symbol in enumerate(available):

        if verbose and sym_idx > 0 and sym_idx % 50 == 0:
            elapsed = time.time() - t0
            rate    = sym_idx / elapsed
            eta     = (len(available) - sym_idx) / rate if rate > 0 else 0
            print(f"  [{sym_idx:>4}/{len(available)}]  {total_rows:>9,} rows  "
                  f"{total_sigs:>5,} signals  ETA {eta:>4.0f}s")

        try:
            raw = pd.read_parquet(parquet_files[symbol])
        except Exception as e:
            print(f"  SKIP {symbol}: {e}")
            continue

        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"]   = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute

        all_dates = sorted(raw["date"].unique())
        if len(all_dates) < vol_window + 1:
            continue

        # ── Per-day volume aggregates ─────────────────────────────────────────
        fd_vol = (raw[(raw["hm"] >= HM_915) & (raw["hm"] <= HM_1515)]
                  .groupby("date")["volume"].sum())
        c3_vol = (raw[(raw["hm"] >= HM_915) & (raw["hm"] <= HM_1445)]
                  .groupby("date")["volume"].sum())

        # ── VWAP of last two candles of previous day ──────────────────────────
        last2      = raw[raw["hm"].isin([HM_1500, HM_1515])].copy()
        last2["tp"]     = (last2["high"] + last2["low"] + last2["close"]) / 3
        last2["tp_vol"] = last2["tp"] * last2["volume"]
        last2_grp  = last2.groupby("date")[["tp_vol", "volume"]].sum()
        vwap_raw   = (last2_grp["tp_vol"] / last2_grp["volume"]).where(
            last2_grp["volume"] > 0, np.nan)

        # ── 15:00 open ────────────────────────────────────────────────────────
        pm3_raw = raw[raw["hm"] == HM_1500].groupby("date")["open"].last()

        n        = len(all_dates)
        fd_arr   = np.array([fd_vol.get(d,  0)      for d in all_dates], dtype=float)
        c3_arr   = np.array([c3_vol.get(d,  0)      for d in all_dates], dtype=float)
        pm3_arr  = np.array([pm3_raw.get(d, np.nan) for d in all_dates], dtype=float)
        vwap_arr = np.array([vwap_raw.get(d, np.nan) for d in all_dates], dtype=float)

        # Rolling vol average: computed only over non-zero-volume trading days so that
        # zero-volume holidays (e.g. Diwali Muhurat) don't create NaN cascades.
        # shift(1) by position means each day uses the previous non-zero-volume day's window.
        # reindex+ffill maps the result back to all calendar dates in all_dates.
        _fd_nonzero = pd.Series(fd_arr, index=all_dates)[fd_arr > 0]
        avg_arr = (
            _fd_nonzero
            .rolling(vol_window, min_periods=vol_window)
            .mean()
            .shift(1)
            .reindex(all_dates, method="ffill")
            .values
        )

        prev_vwap_arr      = np.empty(n, dtype=float); prev_vwap_arr[0]  = np.nan
        prev_vwap_arr[1:]  = vwap_arr[:-1]
        next_pm3_arr       = np.empty(n, dtype=float); next_pm3_arr[-1]  = np.nan
        next_pm3_arr[:-1]  = pm3_arr[1:]

        prev_d_list = [None] + list(all_dates[:-1])
        ts_dates    = [pd.Timestamp(d) for d in all_dates]
        snap_idxs   = [max(bisect_right(snap_dates, t) - 1, 0) for t in ts_dates]
        mcap_arr    = np.array(
            [snap_eligible[i].get(symbol, np.nan) for i in snap_idxs], dtype=float)

        # ── Full intraday candle set ───────────────────────────────────────────
        full = raw[(raw["hm"] >= HM_915) & (raw["hm"] <= HM_1515)].copy()

        o915   = full[full["hm"] == HM_915 ].groupby("date")["open" ].last()
        c915   = full[full["hm"] == HM_915 ].groupby("date")["close"].last()
        o930   = full[full["hm"] == HM_930 ].groupby("date")["open" ].last()
        o945   = full[full["hm"] == HM_945 ].groupby("date")["open" ].last()
        o1100  = full[full["hm"] == HM_1100].groupby("date")["open" ].last()
        o1430  = full[full["hm"] == HM_1430].groupby("date")["open" ].last()
        o1445  = full[full["hm"] == HM_1445].groupby("date")["open" ].last()
        o1515  = full[full["hm"] == HM_1515].groupby("date")["open" ].last()
        c1515  = full[full["hm"] == HM_1515].groupby("date")["close"].last()

        dh = full.groupby("date")["high"  ].max()
        dl = full.groupby("date")["low"   ].min()
        dv = full.groupby("date")["volume"].sum()

        sorted_f      = full.sort_values(["date", "hm"]).copy()
        sorted_f["_dh"] = sorted_f.groupby("date")["high"].transform("max")
        sorted_f["_dl"] = sorted_f.groupby("date")["low" ].transform("min")
        hm_high = (sorted_f[sorted_f["high"] == sorted_f["_dh"]]
                   .groupby("date")["hm"].first())
        hm_low  = (sorted_f[sorted_f["low"]  == sorted_f["_dl"]]
                   .groupby("date")["hm"].first())

        cumlow  = {tag: full[full["hm"] <= mhm].groupby("date")["low" ].min()
                   for tag, mhm in CHECKPOINTS}
        cumhigh = {tag: full[full["hm"] <= mhm].groupby("date")["high"].max()
                   for tag, mhm in CHECKPOINTS}

        o915_arr   = get_arr(o915,   all_dates)
        c915_arr   = get_arr(c915,   all_dates)
        o930_arr   = get_arr(o930,   all_dates)
        o945_arr   = get_arr(o945,   all_dates)
        o1100_arr  = get_arr(o1100,  all_dates)
        o1430_arr  = get_arr(o1430,  all_dates)
        o1445_arr  = get_arr(o1445,  all_dates)
        o1515_arr  = get_arr(o1515,  all_dates)
        c1515_arr  = get_arr(c1515,  all_dates)
        dh_arr     = get_arr(dh,     all_dates)
        dl_arr     = get_arr(dl,     all_dates)
        dv_arr     = get_arr(dv,     all_dates, default=0)
        hm_high_arr = get_arr(hm_high, all_dates)
        hm_low_arr  = get_arr(hm_low,  all_dates)

        cumlow_arrs  = {tag: get_arr(cumlow[tag],  all_dates) for tag, _ in CHECKPOINTS}
        cumhigh_arrs = {tag: get_arr(cumhigh[tag], all_dates) for tag, _ in CHECKPOINTS}

        nd_o915_arr    = shift_next(o915_arr)
        nd_c915_arr    = shift_next(c915_arr)
        nd_o930_arr    = shift_next(o930_arr)
        nd_o945_arr    = shift_next(o945_arr)
        nd_o1100_arr   = shift_next(o1100_arr)
        nd_o1430_arr   = shift_next(o1430_arr)
        nd_o1445_arr   = shift_next(o1445_arr)
        nd_c1515_arr   = shift_next(c1515_arr)
        nd_dh_arr      = shift_next(dh_arr)
        nd_dl_arr      = shift_next(dl_arr)
        nd_dv_arr      = shift_next(dv_arr)
        nd_hm_high_arr = shift_next(hm_high_arr)
        nd_hm_low_arr  = shift_next(hm_low_arr)
        nd_cumlow_arrs  = {tag: shift_next(arr) for tag, arr in cumlow_arrs.items()}
        nd_cumhigh_arrs = {tag: shift_next(arr) for tag, arr in cumhigh_arrs.items()}

        # ── Assemble per-symbol DataFrame ─────────────────────────────────────
        sym_df = pd.DataFrame({
            "date":           all_dates,
            "snap_idx":       snap_idxs,
            "mcap_cr":        mcap_arr,
            "prev_date":      prev_d_list,
            "prev_vwap_close": prev_vwap_arr,
            "pm3_open":       pm3_arr,
            "next_pm3_open":  next_pm3_arr,
            "cum_vol":        c3_arr,
            "avg_vol":        avg_arr,
            "o915":   o915_arr,   "c915":  c915_arr,
            "o930":   o930_arr,   "o945":  o945_arr,
            "o1100":  o1100_arr,  "o1430": o1430_arr,
            "o1445":  o1445_arr,  "o1515": o1515_arr,  "c1515": c1515_arr,
            "dh":     dh_arr,     "dl":    dl_arr,      "dv":    dv_arr,
            "hm_high": hm_high_arr, "hm_low": hm_low_arr,
            **{f"cumlow_{t}":  cumlow_arrs[t]  for t, _ in CHECKPOINTS},
            **{f"cumhigh_{t}": cumhigh_arrs[t] for t, _ in CHECKPOINTS},
            "nd_o915":    nd_o915_arr,   "nd_c915":   nd_c915_arr,
            "nd_o930":    nd_o930_arr,   "nd_o945":   nd_o945_arr,
            "nd_o1100":   nd_o1100_arr,  "nd_o1430":  nd_o1430_arr,
            "nd_o1445":   nd_o1445_arr,  "nd_c1515":  nd_c1515_arr,
            "nd_dh":      nd_dh_arr,     "nd_dl":     nd_dl_arr,   "nd_dv": nd_dv_arr,
            "nd_hm_high": nd_hm_high_arr, "nd_hm_low": nd_hm_low_arr,
            **{f"nd_cumlow_{t}":  nd_cumlow_arrs[t]  for t, _ in CHECKPOINTS},
            **{f"nd_cumhigh_{t}": nd_cumhigh_arrs[t] for t, _ in CHECKPOINTS},
        })

        # Drop first vol_window rows (rolling avg not yet valid) and rows with no mcap
        sym_df = sym_df.iloc[vol_window:].copy()
        sym_df = sym_df[sym_df["mcap_cr"].notna()].copy()
        if sym_df.empty:
            continue

        # ── Signal criteria ───────────────────────────────────────────────────
        with np.errstate(invalid="ignore", divide="ignore"):
            ret_arr = ((sym_df["pm3_open"].values - sym_df["prev_vwap_close"].values)
                       / sym_df["prev_vwap_close"].values * 100)
            vol_arr = sym_df["cum_vol"].values / sym_df["avg_vol"].values

        passes_vol = np.where(np.isnan(vol_arr), False, vol_arr >= vol_mult)
        passes_ret = np.where(np.isnan(ret_arr), False, ret_arr >= 5.0)
        passes_all = passes_vol & passes_ret

        hm_high_vals    = sym_df["hm_high"].values
        hm_low_vals     = sym_df["hm_low"].values
        nd_hm_high_vals = sym_df["nd_hm_high"].values
        nd_hm_low_vals  = sym_df["nd_hm_low"].values

        # ── Build output rows ─────────────────────────────────────────────────
        out_df = pd.DataFrame({
            "symbol":                    symbol,
            "date":                      sym_df["date"].values,
            "mcap_snapshot_period_used": [snap_labels[i] for i in sym_df["snap_idx"].values],
            "market_cap_value":          np.round(sym_df["mcap_cr"].values, 2),
            "prev_trading_day":          sym_df["prev_date"].values,
            "prev_day_vwap_close":       np.round(sym_df["prev_vwap_close"].values, 4),
            # Entry price candidates
            "entry_price_3pm":           np.round(sym_df["pm3_open"].values, 4),
            "entry_price_315pm":         np.round(sym_df["o1515"].values, 4),
            # Signal diagnostics
            "return_pct_vs_prev_close":  np.round(ret_arr, 6),
            "cum_volume_to_3pm_today":   sym_df["cum_vol"].values.astype(np.int64),
            "avg_30day_fullday_volume":  np.round(sym_df["avg_vol"].values, 2),
            "volume_ratio":              np.round(vol_arr, 6),
            "passes_volume":             passes_vol,
            "passes_return":             passes_ret,
            "passes_all_three":          passes_all,
            # Exit prices — standard single exits
            "exit_3pm_open":             np.round(sym_df["next_pm3_open"].values, 4),
            "exit_945_open":             np.round(sym_df["nd_o945"].values, 4),
            "exit_1100_open":            np.round(sym_df["nd_o1100"].values, 4),
            "exit_1430_open":            np.round(sym_df["nd_o1430"].values, 4),
            "exit_1445_open":            np.round(sym_df["nd_o1445"].values, 4),
            # Enriched entry-day columns
            "today_open_915":            np.round(sym_df["o915"].values,  4),
            "today_close_1515":          np.round(sym_df["c1515"].values, 4),
            "today_high":                np.round(sym_df["dh"].values,    4),
            "today_high_time":           [hm_to_time(v) for v in hm_high_vals],
            "today_low":                 np.round(sym_df["dl"].values,    4),
            "today_low_time":            [hm_to_time(v) for v in hm_low_vals],
            "today_fullday_vol":         sym_df["dv"].values.astype(np.int64),
            "today_cumlow_10":           np.round(sym_df["cumlow_10"].values,  4),
            "today_cumlow_11":           np.round(sym_df["cumlow_11"].values,  4),
            "today_cumlow_12":           np.round(sym_df["cumlow_12"].values,  4),
            "today_cumlow_13":           np.round(sym_df["cumlow_13"].values,  4),
            "today_cumlow_14":           np.round(sym_df["cumlow_14"].values,  4),
            "today_cumlow_15":           np.round(sym_df["cumlow_15"].values,  4),
            "today_cumhigh_10":          np.round(sym_df["cumhigh_10"].values, 4),
            "today_cumhigh_11":          np.round(sym_df["cumhigh_11"].values, 4),
            "today_cumhigh_12":          np.round(sym_df["cumhigh_12"].values, 4),
            "today_cumhigh_13":          np.round(sym_df["cumhigh_13"].values, 4),
            "today_cumhigh_14":          np.round(sym_df["cumhigh_14"].values, 4),
            "today_cumhigh_15":          np.round(sym_df["cumhigh_15"].values, 4),
            # Enriched exit-day columns
            "next_day_915_close":        np.round(sym_df["nd_c915"].values,   4),
            "next_day_930_open":         np.round(sym_df["nd_o930"].values,   4),
            "next_day_open_915":         np.round(sym_df["nd_o915"].values,   4),
            "next_day_close_1515":       np.round(sym_df["nd_c1515"].values,  4),
            "next_day_high":             np.round(sym_df["nd_dh"].values,     4),
            "next_day_high_time":        [hm_to_time(v) for v in nd_hm_high_vals],
            "next_day_low":              np.round(sym_df["nd_dl"].values,     4),
            "next_day_low_time":         [hm_to_time(v) for v in nd_hm_low_vals],
            "next_day_fullday_vol":      np.where(np.isnan(sym_df["nd_dv"].values), 0, sym_df["nd_dv"].values).astype(np.int64),
            "next_day_cumlow_10":        np.round(sym_df["nd_cumlow_10"].values,  4),
            "next_day_cumlow_11":        np.round(sym_df["nd_cumlow_11"].values,  4),
            "next_day_cumlow_12":        np.round(sym_df["nd_cumlow_12"].values,  4),
            "next_day_cumlow_13":        np.round(sym_df["nd_cumlow_13"].values,  4),
            "next_day_cumlow_14":        np.round(sym_df["nd_cumlow_14"].values,  4),
            "next_day_cumlow_15":        np.round(sym_df["nd_cumlow_15"].values,  4),
            "next_day_cumhigh_10":       np.round(sym_df["nd_cumhigh_10"].values, 4),
            "next_day_cumhigh_11":       np.round(sym_df["nd_cumhigh_11"].values, 4),
            "next_day_cumhigh_12":       np.round(sym_df["nd_cumhigh_12"].values, 4),
            "next_day_cumhigh_13":       np.round(sym_df["nd_cumhigh_13"].values, 4),
            "next_day_cumhigh_14":       np.round(sym_df["nd_cumhigh_14"].values, 4),
            "next_day_cumhigh_15":       np.round(sym_df["nd_cumhigh_15"].values, 4),
        })

        if save_csv:
            out_df.to_csv(out_path, mode="w" if first_write else "a",
                          header=first_write, index=False)
            first_write = False
        total_rows  += len(out_df)
        total_sigs  += int(passes_all.sum())
        all_chunks.append(out_df)

    elapsed = time.time() - t0
    if verbose:
        print()
        print("=" * 68)
        print(f"DONE  {elapsed:.1f}s  ({elapsed/60:.1f} min)")
        print(f"  Total rows        : {total_rows:>10,}")
        print(f"  passes_all_three  : {total_sigs:>10,}")
        print(f"  Symbols processed : {len(available):>10,}")
        print(f"  Output: {out_path if save_csv else '(not saved)'}")
        print("=" * 68)

    if all_chunks:
        return pd.concat(all_chunks, ignore_index=True)
    return pd.DataFrame()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build diagnostic_table.csv from master_data/")
    parser.add_argument("--vol-window", type=int,   default=30,
                        help="Rolling lookback for volume average (default: 30)")
    parser.add_argument("--vol-mult",   type=float, default=7.0,
                        help="Minimum volume ratio threshold (default: 7.0)")
    parser.add_argument("--output",     type=str,   default=None,
                        help="Output CSV path (default: results/diagnostic_table.csv)")
    args = parser.parse_args()

    if not any(MASTER_DIR.glob("*.parquet")):
        print(f"ERROR: No parquet files found in {MASTER_DIR}/")
        print("Run data_loading.py first to build master_data/.")
        sys.exit(1)

    build_diagnostic_table(
        vol_window  = args.vol_window,
        vol_mult    = args.vol_mult,
        output_path = args.output,
    )


if __name__ == "__main__":
    main()
