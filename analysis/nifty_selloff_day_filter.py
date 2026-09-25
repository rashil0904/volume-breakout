# -*- coding: utf-8 -*-
"""
nifty_selloff_day_filter.py
===========================
Filter the conditional_split_best_t2 (t1=09:45, t2=12:00) + 14% profit-target
trade set by the exit day's Nifty behaviour, and compare metrics.

Two filters, both reported:
  (existing) Nifty intraday fall : nifty_pct_low_from_open <= -0.75
  (new)      Gap-down + further fall (BOTH):
                nifty_pct_gap        = (day_open - prev_close)/prev_close*100 <= -0.50
                nifty_pct_low_from_open = (day_low - day_open)/day_open*100  <= -0.75
             prev_close = prior trading day's 15:15 candle close.

Reuses (not reimplemented):
  - per-trade exit return/exit_type : profit_target_sweep.run_cond (X=14)
  - both total-return metrics        : exit_time_sweep._metrics
  - exit_type breakdown              : profit_target_sweep.etype_breakdown
"""

import sys, bisect
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import profit_target_sweep as pts

IST = ets.IST
NIFTY_CSV = rb.RESULTS.parent / "data" / "nifty_15min_ohlc.csv"
OUTDIR = rb.RESULTS / "nifty_selloff_filter"
FALL_THRESHOLD = -0.75             # intraday: day low <= -0.75% below day open
GAP_THRESHOLD  = -0.50             # gap-down: day open <= -0.50% below prev close
CATS = ["early_target_pre_t1", "positive_at_t1",
        "early_target_between_t1_t2", "exit_at_t2_no_target"]


def fetch_ohlc_and_exitday(base):
    n = len(base)
    opens = np.full((n, len(pts.CANDLE_HMS)), np.nan)
    highs = np.full((n, len(pts.CANDLE_HMS)), np.nan)
    exit_days = [None] * n
    for sym, grp in base.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        dates_sorted = sorted(raw["date"].unique())
        sub = raw[raw["hm"].isin(pts.CANDLE_HMS)]
        po = sub.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=pts.CANDLE_HMS)
        ph = sub.pivot_table(index="date", columns="hm", values="high", aggfunc="last").reindex(columns=pts.CANDLE_HMS)
        for pos_idx, ed in zip(grp.index, grp["date"]):
            j = bisect.bisect_right(dates_sorted, ed.date())
            if j >= len(dates_sorted):
                continue
            nd = dates_sorted[j]
            exit_days[pos_idx] = nd
            if nd in po.index:
                opens[pos_idx, :] = po.loc[nd].values
                highs[pos_idx, :] = ph.loc[nd].values
    return opens, highs, exit_days


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Reconstruct the target-14% trade set (same run_cond the sweep uses) ──
    base = ets.load_base_positions()
    entry = base["entry"].values.astype(float)
    cap   = base["cap"].values.astype(float)
    edate = base["date"].values
    print("Fetching next-day OHLC + exit days …")
    opens, highs, exit_days = fetch_ohlc_and_exitday(base)
    pct_high = (highs - entry[:, None]) / entry[:, None] * 100
    t1, t2 = pts.HM_0945, pts.HM_1200
    ot1, ot2 = opens[:, pts.HCOL[t1]], opens[:, pts.HCOL[t2]]
    ret_t1 = (ot1 - entry) / entry * 100
    p1max = pts._window_max(pct_high, [pts.HCOL[hm] for hm in pts.CANDLE_HMS if hm < t1])
    p3max = pts._window_max(pct_high, [pts.HCOL[hm] for hm in pts.CANDLE_HMS if t1 < hm < t2])
    _, ret, et, mask_full = pts.run_cond(14, entry, np.ones(len(base)), ot1, ot2, ret_t1, p1max, p3max)
    pnl = cap * ret / 100

    # ── Nifty per-day: open, low, 15:15 close, prev close, %gap, %low-from-open ──
    nf = pd.read_csv(NIFTY_CSV, parse_dates=["timestamp"])
    nts = nf["timestamp"].dt.tz_convert(IST)
    nf["d"] = nts.dt.date
    nf["hm"] = nts.dt.hour * 60 + nts.dt.minute
    sess = nf[(nf["hm"] >= 555) & (nf["hm"] <= 915)]
    per = pd.DataFrame({
        "open":  sess[sess["hm"] == 555].groupby("d")["open"].first(),
        "low":   sess.groupby("d")["low"].min(),
        "close": sess[sess["hm"] == 915].groupby("d")["close"].first(),   # 15:15 close
    }).sort_index()
    per["prev_close"] = per["close"].shift(1)                              # prior trading day close
    per["pct_low"] = (per["low"] - per["open"]) / per["open"] * 100
    per["pct_gap"] = (per["open"] - per["prev_close"]) / per["prev_close"] * 100
    open_map, low_map = per["open"].to_dict(), per["low"].to_dict()
    pctlow_map, prevclose_map, gap_map = per["pct_low"].to_dict(), per["prev_close"].to_dict(), per["pct_gap"].to_dict()

    # ── Per-trade exit-day Nifty values + the two masks ──
    ex_low = np.array([pctlow_map.get(d, np.nan) for d in exit_days])
    ex_gap = np.array([gap_map.get(d, np.nan) for d in exit_days])
    fall_ok = np.where(np.isnan(ex_low), False, ex_low <= FALL_THRESHOLD)
    gap_ok  = np.where(np.isnan(ex_gap), False, ex_gap <= GAP_THRESHOLD)
    mask_existing = mask_full & fall_ok                       # intraday fall only
    mask_new      = mask_full & fall_ok & gap_ok              # gap-down + further fall

    # ── Metrics (reuse _metrics): full / existing / new ──
    def line(label, m):
        return {"dataset": label, "n_trades": m["n_trades"], "win_rate_pct": m["win_rate_pct"],
                "avg_return_per_trade_pct": m["avg_return_per_trade_pct"],
                "median_return_per_trade_pct": m["median_return_per_trade_pct"],
                "total_return_fixedbase_pct": m["total_return_fixedbase_pct"],
                "total_return_sumofdaily_pct": m["total_return_sumofdaily_pct"]}
    comp = pd.DataFrame([
        line("All days (full dataset)",                       ets._metrics(edate, cap, pnl, ret, mask_full)),
        line("Nifty intraday fall only (existing)",           ets._metrics(edate, cap, pnl, ret, mask_existing)),
        line("Nifty gap-down + further intraday fall (new)",  ets._metrics(edate, cap, pnl, ret, mask_new)),
    ])
    bd_existing = pts.etype_breakdown(ret, et, mask_existing, CATS)
    bd_new      = pts.etype_breakdown(ret, et, mask_new, CATS)

    # ── Per-day flag table (a/b/c/d) over unique exit days present ──
    present = sorted(set(d for d, m in zip(exit_days, mask_full) if m and d is not None))
    day_rows = []
    for d in present:
        pl, gp = pctlow_map.get(d, np.nan), gap_map.get(d, np.nan)
        pe = (not np.isnan(pl)) and pl <= FALL_THRESHOLD
        pn = (not np.isnan(pl)) and (not np.isnan(gp)) and gp <= GAP_THRESHOLD and pl <= FALL_THRESHOLD
        day_rows.append({"exit_day": d,
                         "nifty_prev_close": round(prevclose_map.get(d, np.nan), 2),
                         "nifty_day_open": round(open_map.get(d, np.nan), 2),
                         "nifty_pct_gap": round(gp, 4) if not np.isnan(gp) else np.nan,
                         "nifty_day_low": round(low_map.get(d, np.nan), 2),
                         "nifty_pct_low_from_open": round(pl, 4) if not np.isnan(pl) else np.nan,
                         "pass_existing": pe, "pass_new": pn})
    day_df = pd.DataFrame(day_rows)
    n_ex = int(day_df["pass_existing"].sum()); n_nw = int(day_df["pass_new"].sum())
    n_ex_only = int((day_df["pass_existing"] & ~day_df["pass_new"]).sum())
    n_neither = int((~day_df["pass_existing"]).sum())

    # ── Trade-level saves (both filters) with Nifty columns ──
    def trade_df(mask):
        fi = np.where(mask)[0]
        return pd.DataFrame({
            "symbol": base["symbol"].values[fi],
            "entry_date": [pd.Timestamp(x).date() for x in base["date"].values[fi]],
            "exit_day": [exit_days[i] for i in fi],
            "exit_type": et[fi],
            "return_pct": np.round(ret[fi], 4),
            "capital_deployed": np.round(cap[fi], 2),
            "pnl": np.round(pnl[fi], 2),
            "nifty_prev_close": [round(prevclose_map.get(exit_days[i], np.nan), 2) for i in fi],
            "nifty_day_open": [round(open_map.get(exit_days[i], np.nan), 2) for i in fi],
            "nifty_pct_gap": [round(gap_map.get(exit_days[i], np.nan), 4) for i in fi],
            "nifty_day_low": [round(low_map.get(exit_days[i], np.nan), 2) for i in fi],
            "nifty_pct_low_from_open": [round(pctlow_map.get(exit_days[i], np.nan), 4) for i in fi],
        }).sort_values(["exit_day", "symbol"])

    trade_df(mask_existing).to_csv(OUTDIR / "nifty_intraday_fall_trades.csv", index=False)
    trade_df(mask_new).to_csv(OUTDIR / "nifty_gapdown_fall_trades.csv", index=False)
    comp.to_csv(OUTDIR / "nifty_selloff_comparison.csv", index=False)
    day_df.to_csv(OUTDIR / "nifty_exit_day_flags.csv", index=False)
    bd_existing.to_csv(OUTDIR / "breakdown_existing.csv", index=False)
    bd_new.to_csv(OUTDIR / "breakdown_new.csv", index=False)

    # ── Prints ──
    print("\n" + "=" * 96)
    print(f"NIFTY EXIT-DAY FILTERS   fall<={FALL_THRESHOLD}%  |  gap<={GAP_THRESHOLD}%")
    print("=" * 96)
    print(f"  Unique exit days (in dataset)           : {len(present):,}")
    print(f"  (a) pass existing (intraday fall)       : {n_ex:,} days")
    print(f"  (b) pass new (gap-down + fall)          : {n_nw:,} days")
    print(f"  (c) both (new implies existing)         : {n_nw:,} days")
    print(f"  (d) existing-only (fall but no gap-down): {n_ex_only:,} days")
    print(f"      neither                             : {n_neither:,} days")
    print(f"  Trades: existing {int(mask_existing.sum()):,}  |  new {int(mask_new.sum()):,}  "
          f"of {int(mask_full.sum()):,} total")

    print("\n" + "=" * 118)
    print("COMPARISON — full vs existing filter vs new (gap-down + fall) filter")
    print("=" * 118)
    print(f"  {'dataset':<44}{'Trades':>8}{'Win%':>8}{'Avg%':>9}{'Med%':>9}"
          f"{'TotRet(5L)%':>13}{'TotRet(daily)%':>16}")
    print("  " + "-" * 114)
    for _, r in comp.iterrows():
        print(f"  {r['dataset']:<44}{r['n_trades']:>8,}{r['win_rate_pct']:>8.2f}"
              f"{r['avg_return_per_trade_pct']:>9.4f}{r['median_return_per_trade_pct']:>9.4f}"
              f"{r['total_return_fixedbase_pct']:>13.2f}{r['total_return_sumofdaily_pct']:>16.2f}")

    for title, bd in [("existing (intraday fall)", bd_existing), ("new (gap-down + fall)", bd_new)]:
        print("\n" + "=" * 92)
        print(f"EXIT-TYPE BREAKDOWN — {title}")
        print("=" * 92)
        print(f"  {'exit_type':<30}{'count':>7}{'pct%':>8}{'avg_ret%':>11}{'median_ret%':>13}")
        print("  " + "-" * 88)
        for _, r in bd.iterrows():
            av = "—" if pd.isna(r["avg_return_pct"]) else f"{r['avg_return_pct']:.4f}"
            md = "—" if pd.isna(r["median_return_pct"]) else f"{r['median_return_pct']:.4f}"
            print(f"  {r['exit_type']:<30}{int(r['count']):>7,}{r['pct']:>8.2f}{av:>11}{md:>13}")
    print("=" * 92)
    print(f"\nSaved → {OUTDIR}")


if __name__ == "__main__":
    main()
