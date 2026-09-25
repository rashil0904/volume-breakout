# -*- coding: utf-8 -*-
"""
nifty_stoploss_overlay.py
=========================
conditional_split_best_t2 (t1=09:45, t2=12:00) + 14% profit target, WITH a
Nifty-triggered hard stop-loss overlay, compared against the same scenario
WITHOUT the stop (the existing target-14% baseline).

Stop trigger (per exit day, from Nifty):
  gap = (day_open - prev_close)/prev_close*100.  If gap > -0.5 -> no stop that day.
  If gap <= -0.5: trigger_time = first 15-min candle whose cumulative low reaches
     <= -0.75% below the day's open.  If never -> no stop.
Applying: any trade STILL OPEN at trigger_time (original exit time >= trigger_time,
  ties -> stop) is force-exited at the trigger candle's OPEN; exit_type="nifty_stoploss".
  Trades already exited strictly before trigger_time keep their original exit.

Reuses: exit_time_sweep._metrics, profit_target_sweep.{fetch,run_cond,etype_breakdown}.
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
OUTDIR = rb.RESULTS / "nifty_stoploss_overlay"
GAP_THRESHOLD  = -0.50
FALL_THRESHOLD = -0.75
TARGET = 14.0
CATS = ["early_target_pre_t1", "positive_at_t1", "early_target_between_t1_t2",
        "exit_at_t2_no_target", "nifty_stoploss"]


def hm_str(hm):
    return f"{int(hm)//60:02d}:{int(hm)%60:02d}" if hm is not None and not (isinstance(hm, float) and np.isnan(hm)) else ""


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


def full_metrics(edate, cap, pnl, ret, mask):
    m = ets._metrics(edate, cap, pnl, ret, mask)
    r, p = ret[mask], pnl[mask]
    win, los = r[p > 0], r[p <= 0]
    m["avg_return_winning_trades_pct"] = round(float(win.mean()), 4) if len(win) else np.nan
    m["avg_return_losing_trades_pct"]  = round(float(los.mean()), 4) if len(los) else np.nan
    return m


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
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
    pre_hms = [hm for hm in pts.CANDLE_HMS if hm < t1]
    bet_hms = [hm for hm in pts.CANDLE_HMS if t1 < hm < t2]

    # ── Reconstruct ORIGINAL per-trade exit (type, time, price) ──
    N = len(base)
    o_et = np.empty(N, dtype=object); o_hm = np.full(N, np.nan); o_px = np.full(N, np.nan)
    valid = np.zeros(N, dtype=bool)
    for i in range(N):
        e = entry[i]; ph = pct_high[i]
        et = hm = px = None
        for hm_ in pre_hms:
            v = ph[pts.HCOL[hm_]]
            if not np.isnan(v) and v >= TARGET:
                et, hm, px = "early_target_pre_t1", hm_, e * (1 + TARGET / 100); break
        if et is None:
            if not np.isnan(ot1[i]) and ret_t1[i] > 0:
                et, hm, px = "positive_at_t1", t1, ot1[i]
            else:
                for hm_ in bet_hms:
                    v = ph[pts.HCOL[hm_]]
                    if not np.isnan(v) and v >= TARGET:
                        et, hm, px = "early_target_between_t1_t2", hm_, e * (1 + TARGET / 100); break
                if et is None and not np.isnan(ot2[i]):
                    et, hm, px = "exit_at_t2_no_target", t2, ot2[i]
        if et is not None:
            o_et[i], o_hm[i], o_px[i], valid[i] = et, hm, px, True
    ret_base = (o_px - entry) / entry * 100
    pnl_base = cap * ret_base / 100

    # ── Nifty per-day: open, close(15:15), prev_close, gap, + stop trigger_time ──
    nf = pd.read_csv(NIFTY_CSV, parse_dates=["timestamp"])
    nts = nf["timestamp"].dt.tz_convert(IST)
    nf["d"] = nts.dt.date; nf["hm"] = nts.dt.hour * 60 + nts.dt.minute
    sess = nf[(nf["hm"] >= 555) & (nf["hm"] <= 915)]
    per = pd.DataFrame({
        "open":  sess[sess["hm"] == 555].groupby("d")["open"].first(),
        "close": sess[sess["hm"] == 915].groupby("d")["close"].first(),
    }).sort_index()
    per["prev_close"] = per["close"].shift(1)
    per["gap"] = (per["open"] - per["prev_close"]) / per["prev_close"] * 100
    open_map, gap_map, prevclose_map = per["open"].to_dict(), per["gap"].to_dict(), per["prev_close"].to_dict()

    trigger_map = {}
    for d, g in sess.groupby("d"):
        gp = gap_map.get(d, np.nan)
        if np.isnan(gp) or gp > GAP_THRESHOLD:
            continue
        do = open_map[d]; thr = do * (1 + FALL_THRESHOLD / 100)
        hit = g.sort_values("hm")[g.sort_values("hm")["low"] <= thr]
        if len(hit):
            trigger_map[d] = int(hit.iloc[0]["hm"])

    # ── Apply the stop ──
    f_et = o_et.copy(); f_px = o_px.copy(); trig_hm = np.full(N, np.nan)
    for i in range(N):
        if not valid[i]:
            continue
        d = exit_days[i]
        trg = trigger_map.get(d)
        if trg is None:
            continue
        trig_hm[i] = trg
        if o_hm[i] >= trg:                                   # still open at trigger (tie -> stop)
            so = opens[i, pts.HCOL[trg]] if trg in pts.HCOL else np.nan
            if not np.isnan(so):
                f_et[i], f_px[i] = "nifty_stoploss", so
    ret_stop = (f_px - entry) / entry * 100
    pnl_stop = cap * ret_stop / 100

    # ── Metrics ──
    m_base = full_metrics(edate, cap, pnl_base, ret_base, valid)
    m_stop = full_metrics(edate, cap, pnl_stop, ret_stop, valid)

    # sanity: baseline should match the existing target-14 result
    _, r_rc, et_rc, mask_rc = pts.run_cond(14, entry, np.ones(N), ot1, ot2, ret_t1,
        pts._window_max(pct_high, [pts.HCOL[hm] for hm in pts.CANDLE_HMS if hm < t1]),
        pts._window_max(pct_high, [pts.HCOL[hm] for hm in pts.CANDLE_HMS if t1 < hm < t2]))
    rc_fixed = round(float((cap * r_rc / 100)[mask_rc].sum()) / ets.CAPITAL_BASE * 100, 2)

    cols = ["dataset", "n_trades", "win_rate_pct", "avg_return_per_trade_pct",
            "median_return_per_trade_pct", "avg_return_winning_trades_pct",
            "avg_return_losing_trades_pct", "total_return_fixedbase_pct", "total_return_sumofdaily_pct"]
    def row(lbl, m): return {"dataset": lbl, **{k: m[k] for k in cols[1:]}}
    comp = pd.DataFrame([
        row("Without Nifty stop-loss (target-14% baseline)", m_base),
        row("With Nifty stop-loss overlay", m_stop),
    ])[cols]

    f_et_stopped = f_et.copy()
    bd = pts.etype_breakdown(ret_stop, f_et_stopped, valid, CATS)

    # ── Coverage ──
    stop_mask = valid & (f_et == "nifty_stoploss")
    present_days = set(d for d, m in zip(exit_days, valid) if m and d is not None)
    days_trig = sorted(d for d in present_days if d in trigger_map)
    days_forced = sorted(set(exit_days[i] for i in range(N) if stop_mask[i]))

    # ── Save with-stop trade list ──
    vi = np.where(valid)[0]
    tl = pd.DataFrame({
        "symbol": base["symbol"].values[vi],
        "entry_date": [pd.Timestamp(x).date() for x in base["date"].values[vi]],
        "exit_day": [exit_days[i] for i in vi],
        "orig_exit_type": o_et[vi],
        "orig_exit_time": [hm_str(o_hm[i]) for i in vi],
        "trigger_time": [hm_str(trig_hm[i]) for i in vi],
        "final_exit_type": f_et[vi],
        "return_pct": np.round(ret_stop[vi], 4),
        "capital_deployed": np.round(cap[vi], 2),
        "pnl": np.round(pnl_stop[vi], 2),
        "nifty_gap_pct": [round(gap_map.get(exit_days[i], np.nan), 4) for i in vi],
    }).sort_values(["exit_day", "symbol"])
    tl.to_csv(OUTDIR / "niftystop_trades.csv", index=False)
    comp.to_csv(OUTDIR / "niftystop_comparison.csv", index=False)
    bd.to_csv(OUTDIR / "niftystop_exit_type_breakdown.csv", index=False)

    # ── Prints ──
    print(f"\n  SANITY: reconstructed baseline fixedbase = {m_base['total_return_fixedbase_pct']:.2f}%  "
          f"| run_cond = {rc_fixed:.2f}%  → {'MATCH' if abs(m_base['total_return_fixedbase_pct']-rc_fixed)<0.05 else 'MISMATCH'}")
    print("\n" + "=" * 96)
    print("NIFTY STOP-LOSS OVERLAY")
    print("=" * 96)
    print(f"  Unique exit days that triggered a stop : {len(days_trig):,}")
    print(f"  Days where a trade was force-exited    : {len(days_forced):,}")
    print(f"  Trades force-exited (nifty_stoploss)   : {int(stop_mask.sum()):,}  of {int(valid.sum()):,} total")

    print("\n" + "=" * 130)
    print("COMPARISON — without vs with Nifty stop-loss")
    print("=" * 130)
    h = (f"  {'dataset':<46}{'Trades':>7}{'Win%':>7}{'Avg%':>8}{'Med%':>8}"
         f"{'AvgWin%':>9}{'AvgLoss%':>10}{'TotRet(5L)%':>13}{'TotRet(daily)%':>16}")
    print(h); print("  " + "-" * 126)
    for _, r in comp.iterrows():
        print(f"  {r['dataset']:<46}{r['n_trades']:>7,}{r['win_rate_pct']:>7.2f}"
              f"{r['avg_return_per_trade_pct']:>8.4f}{r['median_return_per_trade_pct']:>8.4f}"
              f"{r['avg_return_winning_trades_pct']:>9.4f}{r['avg_return_losing_trades_pct']:>10.4f}"
              f"{r['total_return_fixedbase_pct']:>13.2f}{r['total_return_sumofdaily_pct']:>16.2f}")

    print("\n" + "=" * 92)
    print("EXIT-TYPE BREAKDOWN — with Nifty stop-loss (5 categories)")
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
