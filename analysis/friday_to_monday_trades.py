# -*- coding: utf-8 -*-
"""
friday_to_monday_trades.py
==========================
Filter the FULL conditional_split_best_t2 (t1=09:45, t2=12:00) + 14% profit-target
trade set to trades that ENTERED on a Friday and EXITED on the immediately
following Monday. Uses every trade in the target-14% dataset (no Nifty filter).

Per-trade fields are regenerated with the same run_cond exit logic used by the
sweep (baseline reproduces the 550.88% target-14 result). Metrics reuse
exit_time_sweep._metrics (fixed ₹5L + sum-of-daily).
"""

import sys, bisect
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import profit_target_sweep as pts

IST = ets.IST
OUTDIR = rb.RESULTS / "friday_to_monday"
TARGET = 14.0


def hm_str(hm):
    return f"{int(hm)//60:02d}:{int(hm)%60:02d}" if not (isinstance(hm, float) and np.isnan(hm)) else ""


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
    base = ets.load_base_positions()
    entry = base["entry"].values.astype(float)
    cap   = base["cap"].values.astype(float)
    shares = base["shares"].values.astype(float)
    edate = base["date"].values
    print("Fetching next-day OHLC + exit days …")
    opens, highs, exit_days = fetch_ohlc_and_exitday(base)
    pct_high = (highs - entry[:, None]) / entry[:, None] * 100
    t1, t2 = pts.HM_0945, pts.HM_1200
    ot1, ot2 = opens[:, pts.HCOL[t1]], opens[:, pts.HCOL[t2]]
    ret_t1 = (ot1 - entry) / entry * 100
    pre_hms = [hm for hm in pts.CANDLE_HMS if hm < t1]
    bet_hms = [hm for hm in pts.CANDLE_HMS if t1 < hm < t2]

    # ── Regenerate per-trade exit (type, time, price) — same logic as the sweep ──
    N = len(base)
    o_et = np.empty(N, dtype=object); o_hm = np.full(N, np.nan); o_px = np.full(N, np.nan)
    valid = np.zeros(N, dtype=bool)
    for i in range(N):
        e = entry[i]; ph = pct_high[i]; et = hm = px = None
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
    ret = (o_px - entry) / entry * 100
    pnl = cap * ret / 100

    # ── Friday → following-Monday filter ──
    entry_dates = np.array([pd.Timestamp(x).date() for x in edate])
    is_friday = np.array([d.weekday() == 4 for d in entry_dates])
    following_monday = np.array([d + timedelta(days=3) for d in entry_dates])   # Fri + 3 = Mon
    exit_np = np.array([d if d is not None else None for d in exit_days], dtype=object)
    is_next_monday = np.array([
        (exit_days[i] is not None and exit_days[i] == following_monday[i] and exit_days[i].weekday() == 0)
        for i in range(N)])
    mask = valid & is_friday & is_next_monday

    # Friday entries whose exit was NOT the following Monday (holiday-shifted) — flag (b)
    fri_valid = valid & is_friday
    shifted = fri_valid & ~is_next_monday
    shifted_fridays = sorted(set(entry_dates[i] for i in range(N) if shifted[i]))

    # ── Metrics ──
    m = ets._metrics(edate, cap, pnl, ret, mask)
    n_fridays = len(set(entry_dates[i] for i in range(N) if mask[i]))
    summary = pd.DataFrame([{
        "n_trades": m["n_trades"],
        "n_fridays": n_fridays,
        "win_rate_pct": m["win_rate_pct"],
        "avg_return_per_trade_pct": m["avg_return_per_trade_pct"],
        "median_return_per_trade_pct": m["median_return_per_trade_pct"],
        "total_return_fixedbase_pct": m["total_return_fixedbase_pct"],
        "total_return_sumofdaily_pct": m["total_return_sumofdaily_pct"],
    }])

    # ── Trade-level detail ──
    fi = np.where(mask)[0]
    trades = pd.DataFrame({
        "symbol": base["symbol"].values[fi],
        "entry_date": [entry_dates[i] for i in fi],
        "entry_weekday": ["Friday" for _ in fi],
        "entry_price": np.round(entry[fi], 4),
        "shares": shares[fi].astype(int),
        "capital_deployed": np.round(cap[fi], 2),
        "exit_day": [exit_days[i] for i in fi],
        "exit_weekday": ["Monday" for _ in fi],
        "exit_type": o_et[fi],
        "exit_time": [hm_str(o_hm[i]) for i in fi],
        "exit_price": np.round(o_px[fi], 4),
        "trade_pnl": np.round(pnl[fi], 2),
        "trade_return_pct": np.round(ret[fi], 4),
    }).sort_values(["entry_date", "symbol"])

    xlsx = OUTDIR / "friday_to_monday_trades.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        trades.to_excel(w, sheet_name="trades", index=False)
        summary.to_excel(w, sheet_name="summary", index=False)

    # ── Fridays context ──
    all_valid_entry = sorted(set(entry_dates[i] for i in range(N) if valid[i]))
    dmin, dmax = all_valid_entry[0], all_valid_entry[-1]
    cal_fridays = pd.date_range(dmin, dmax, freq="W-FRI")
    trading_fridays = sorted(set(d for d in all_valid_entry if d.weekday() == 4))

    # ── Prints ──
    print("\n" + "=" * 118)
    print("FRIDAY → following-MONDAY trades  (target-14% conditional_split_best_t2, full dataset)")
    print("=" * 118)
    print(f"  {'n_trades':>9}{'n_fridays':>11}{'Win%':>8}{'Avg%':>9}{'Med%':>9}"
          f"{'TotRet(5L)%':>13}{'TotRet(daily)%':>16}")
    print("  " + "-" * 114)
    r = summary.iloc[0]
    print(f"  {int(r['n_trades']):>9,}{int(r['n_fridays']):>11,}{r['win_rate_pct']:>8.2f}"
          f"{r['avg_return_per_trade_pct']:>9.4f}{r['median_return_per_trade_pct']:>9.4f}"
          f"{r['total_return_fixedbase_pct']:>13.2f}{r['total_return_sumofdaily_pct']:>16.2f}")

    print("\n  CONTEXT:")
    print(f"    Dataset entry-date range          : {dmin} → {dmax}")
    print(f"    Total calendar Fridays in range   : {len(cal_fridays):,}")
    print(f"    Trading Fridays (had ≥1 entry)    : {len(trading_fridays):,}")
    print(f"    Fridays producing a qualifying trade: {n_fridays:,}  "
          f"({n_fridays/len(trading_fridays)*100:.1f}% of trading Fridays)")
    print(f"    Friday entries whose next day was NOT the following Monday "
          f"(holiday-shifted, excluded): {len(shifted_fridays)} Fridays, {int(shifted.sum())} trades")
    if shifted_fridays:
        print("      " + ", ".join(str(d) for d in shifted_fridays[:12]) + (" …" if len(shifted_fridays) > 12 else ""))
    print("=" * 118)
    print(f"\nSaved → {xlsx}")


if __name__ == "__main__":
    main()
