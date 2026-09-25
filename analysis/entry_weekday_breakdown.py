# -*- coding: utf-8 -*-
"""
entry_weekday_breakdown.py
==========================
Break the FULL conditional_split_best_t2 (t1=09:45, t2=12:00) + 14% profit-target
trade set down by ENTRY WEEKDAY (Mon–Fri). Every trade exits on the next trading
day; this groups them by the weekday they were entered on.

Per-trade fields regenerated with the same run_cond exit logic used by the sweep
(baseline reproduces the target-14 550.88% result). Metrics reuse
exit_time_sweep._metrics (fixed ₹5L + sum-of-daily).
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
OUTDIR = rb.RESULTS / "entry_weekday"
TARGET = 14.0
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def hm_str(hm):
    return f"{int(hm)//60:02d}:{int(hm)%60:02d}" if not (isinstance(hm, float) and np.isnan(hm)) else ""


def fetch_ohlc_and_exitday(base):
    n = len(base)
    opens = np.full((n, len(pts.CANDLE_HMS)), np.nan)
    highs = np.full((n, len(pts.CANDLE_HMS)), np.nan)
    exit_days = [None] * n
    for sym, grp in base.groupby("symbol"):
        pqf = rb.MASTER_DIR / f"{sym}.parquet"
        if not pqf.exists():
            continue
        raw = pd.read_parquet(pqf)
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

    entry_dates = np.array([pd.Timestamp(x).date() for x in edate])
    ewd = np.array([d.weekday() for d in entry_dates])           # 0=Mon … 4=Fri

    # ── Per-weekday metrics ──
    def row(label, mask):
        m = ets._metrics(edate, cap, pnl, ret, mask)
        n_days = len(set(entry_dates[i] for i in range(N) if mask[i]))
        return {"entry_weekday": label, "n_trades": m["n_trades"], "n_days": n_days,
                "win_rate_pct": m["win_rate_pct"],
                "avg_return_per_trade_pct": m["avg_return_per_trade_pct"],
                "median_return_per_trade_pct": m["median_return_per_trade_pct"],
                "total_return_fixedbase_pct": m["total_return_fixedbase_pct"],
                "total_return_sumofdaily_pct": m["total_return_sumofdaily_pct"]}
    rows = [row(WD[wd], valid & (ewd == wd)) for wd in range(7) if (valid & (ewd == wd)).any()]
    rows.append(row("ALL (full dataset)", valid))
    table = pd.DataFrame(rows)

    # ── Trade-level detail (all valid, with weekday) ──
    vi = np.where(valid)[0]
    trades = pd.DataFrame({
        "symbol": base["symbol"].values[vi],
        "entry_date": [entry_dates[i] for i in vi],
        "entry_weekday": [WD[ewd[i]] for i in vi],
        "entry_price": np.round(entry[vi], 4),
        "shares": shares[vi].astype(int),
        "capital_deployed": np.round(cap[vi], 2),
        "exit_day": [exit_days[i] for i in vi],
        "exit_weekday": [pd.Timestamp(exit_days[i]).strftime("%A") if exit_days[i] is not None else "" for i in vi],
        "exit_type": o_et[vi],
        "exit_time": [hm_str(o_hm[i]) for i in vi],
        "exit_price": np.round(o_px[vi], 4),
        "trade_pnl": np.round(pnl[vi], 2),
        "trade_return_pct": np.round(ret[vi], 4),
    }).sort_values(["entry_date", "symbol"])

    xlsx = OUTDIR / "entry_weekday_breakdown.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        table.to_excel(w, sheet_name="by_weekday", index=False)
        trades.to_excel(w, sheet_name="trades", index=False)
    table.to_csv(OUTDIR / "entry_weekday_breakdown.csv", index=False)

    print("\n" + "=" * 120)
    print("METRICS BY ENTRY WEEKDAY  (target-14% conditional_split_best_t2; exit = next trading day)")
    print("=" * 120)
    print(f"  {'entry_weekday':<20}{'Trades':>8}{'Days':>7}{'Win%':>8}{'Avg%':>9}{'Med%':>9}"
          f"{'TotRet(5L)%':>13}{'TotRet(daily)%':>16}")
    print("  " + "-" * 116)
    for _, r in table.iterrows():
        sep = "  " + "-" * 116 if r["entry_weekday"].startswith("ALL") else None
        if sep: print(sep)
        print(f"  {r['entry_weekday']:<20}{int(r['n_trades']):>8,}{int(r['n_days']):>7,}"
              f"{r['win_rate_pct']:>8.2f}{r['avg_return_per_trade_pct']:>9.4f}"
              f"{r['median_return_per_trade_pct']:>9.4f}{r['total_return_fixedbase_pct']:>13.2f}"
              f"{r['total_return_sumofdaily_pct']:>16.2f}")
    print("=" * 120)
    print(f"\nSaved → {xlsx}")


if __name__ == "__main__":
    main()
