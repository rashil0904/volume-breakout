# -*- coding: utf-8 -*-
"""
reversal_bounce_vol_alloc_costs.py
==================================
Base 2-day reversal bounce (NO high condition) + volume sweep (K 2-7) under ₹5L-pool allocation,
reporting BOTH net@0.23% and net@0.38% costs. Entry D2 close, next-day D3 15-min full-exit sweep.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "reversal_bounce"
BASE_POOL, PER_TRADE = 500_000, 100_000
EXP23, EXP38 = 0.0023, 0.0038
DAY_OPEN_HM, DAY_CLOSE_HM = 555, 915
EXIT_HMS = list(range(570, 901, 15))
SESSION_HMS = list(range(555, 916, 15))
KGRID = [2, 3, 4, 5, 6, 7]; SMALL = 30


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def day_alloc(entry, dates):
    d = pd.Series(dates); cnt = d.map(d.value_counts()).values
    target = np.where(cnt <= 5, float(PER_TRADE), BASE_POOL / cnt)
    return np.floor(target / entry), cnt


def sweep(entry, d3opens, shares, cap):
    rows = []
    for j, hm in enumerate(EXIT_HMS):
        px = d3opens[:, j]; v = ~np.isnan(px)
        e, s, c, x = entry[v], shares[v], cap[v], px[v]
        if len(e) == 0:
            continue
        pnl = s*(x-e); ret = (x-e)/e*100
        n23 = pnl - c*EXP23; r23 = ret - EXP23*100
        n38 = pnl - c*EXP38; r38 = ret - EXP38*100
        rows.append({"exit_time": hm_lbl(hm), "n_trades": len(e),
                     "gross_win_rate_pct": round(float((pnl>0).mean()*100), 2),
                     "gross_avg_return_pct": round(float(ret.mean()), 4),
                     "gross_total_return_fixedbase_pct": round(float(pnl.sum())/BASE_POOL*100, 4),
                     "net023_win_rate_pct": round(float((n23>0).mean()*100), 2),
                     "net023_avg_return_pct": round(float(r23.mean()), 4),
                     "net023_median_return_pct": round(float(np.median(r23)), 4),
                     "net023_total_return_pct": round(float(n23.sum())/BASE_POOL*100, 4),
                     "net038_win_rate_pct": round(float((n38>0).mean()*100), 2),
                     "net038_avg_return_pct": round(float(r38.mean()), 4),
                     "net038_median_return_pct": round(float(np.median(r38)), 4),
                     "net038_total_return_pct": round(float(n38.sum())/BASE_POOL*100, 4),
                     "avg_capital_deployed_per_trade": round(float(c.mean()), 0)})
    return pd.DataFrame(rows).sort_values("net023_total_return_pct", ascending=False).reset_index(drop=True)


def daystats(dates):
    dc = pd.Series(dates).value_counts()
    return round(float(dc.mean()), 2), int((dc > 5).sum()), int(dc.max())


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols …")

    entry_l, d3_l, vd1_l, vd2_l, dt_l = [], [], [], [], []
    t0 = time.time()
    for si, sym in enumerate(symbols, 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[raw["hm"].isin(SESSION_HMS)]
        op = raw.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=SESSION_HMS)
        cl = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        pv = raw.pivot_table(index="date", columns="hm", values="volume", aggfunc="sum").reindex(columns=SESSION_HMS).fillna(0)
        full_vol = pv.sum(axis=1)
        dates = sorted(op.index)
        day_open = op[DAY_OPEN_HM]
        day_close = cl[DAY_CLOSE_HM].where(cl[DAY_CLOSE_HM].notna(), cl.ffill(axis=1).iloc[:, -1])
        exit_cols = op[EXIT_HMS]
        for i in range(2, len(dates) - 1):
            d0, d1, d2, d3 = dates[i-2], dates[i-1], dates[i], dates[i+1]
            c0, c1 = day_close.get(d0, np.nan), day_close.get(d1, np.nan)
            o1 = day_open.get(d1, np.nan); c2, o2 = day_close.get(d2, np.nan), day_open.get(d2, np.nan)
            if not (c1 < c0):
                continue
            if not (c2 > o2 and c2 > o1):
                continue
            if (sym, d2) not in eligible:
                continue
            entry = c2
            if not (entry == entry and entry > 0):
                continue
            entry_l.append(entry)
            d3_l.append(exit_cols.loc[d3].values if d3 in exit_cols.index else np.full(len(EXIT_HMS), np.nan))
            vd1_l.append(full_vol.get(d1, np.nan)); vd2_l.append(full_vol.get(d2, np.nan)); dt_l.append(d2)
        if si % 250 == 0:
            print(f"  …{si}/{len(symbols)} ({len(entry_l)} signals, {time.time()-t0:.0f}s)")

    entry = np.array(entry_l, float); d3opens = np.vstack(d3_l).astype(float)
    vd1 = np.array(vd1_l, float); vd2 = np.array(vd2_l, float); dts = np.array(dt_l, dtype=object)
    vratio = np.where(vd1 > 0, vd2 / vd1, np.nan)
    print(f"\nBase bounce signals: {len(entry)}")

    full_grid, summary = [], []

    def run_set(tag, mask):
        e_m, d3_m, dt_m = entry[mask], d3opens[mask], dts[mask]
        sh, _ = day_alloc(e_m, dt_m); keep = sh > 0
        t = sweep(e_m[keep], d3_m[keep], sh[keep], sh[keep]*e_m[keep])
        b = t.iloc[0]; avg_sig, over5, maxday = daystats(dt_m)
        return t, {"config": tag, "n_trades": int(b["n_trades"]), "best_exit": b["exit_time"],
                   "net023_win_rate_pct": b["net023_win_rate_pct"], "net023_avg_return_pct": b["net023_avg_return_pct"],
                   "net023_total_return_pct": b["net023_total_return_pct"],
                   "net038_win_rate_pct": b["net038_win_rate_pct"], "net038_avg_return_pct": b["net038_avg_return_pct"],
                   "net038_total_return_pct": b["net038_total_return_pct"],
                   "avg_capital_deployed_per_trade": b["avg_capital_deployed_per_trade"],
                   "avg_signals_per_day": avg_sig, "signal_days_over_5": over5, "max_signals_one_day": maxday,
                   "flag": "n<30" if int(b["n_trades"]) < SMALL else ""}

    t, r = run_set("(base) no vol", np.ones(len(entry), bool))
    tt = t.copy(); tt.insert(0, "K", "none"); full_grid.append(tt); summary.append(r)
    for K in KGRID:
        m = (~np.isnan(vratio)) & (vratio >= K)
        t, r = run_set(f"vol K={K}", m)
        tt = t.copy(); tt.insert(0, "K", K); full_grid.append(tt); summary.append(r)

    grid = pd.concat(full_grid, ignore_index=True)
    summ = pd.DataFrame(summary)
    with pd.ExcelWriter(OUTDIR / "reversal_bounce_vol_alloc_costs.xlsx", engine="openpyxl") as w:
        grid.to_excel(w, sheet_name="full_grid_K_x_exit", index=False)
        summ.to_excel(w, sheet_name="best_per_K_0.23_0.38", index=False)

    pd.set_option("display.width", 260)
    print("\n=== BASE + VOLUME under ₹5L-POOL allocation — net@0.23% AND net@0.38% ===")
    print(summ.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
