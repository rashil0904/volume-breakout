# -*- coding: utf-8 -*-
"""
reversal_bounce_vol_d1decline_sweep.py
==================================
Add a swept D1 minimum-decline filter to the stacked reversal-bounce strategy.
Pattern: D2 green (close>open, close>open[D1]) + high[D2]<high[D1] + vol[D2]>=K*vol[D1],
D1 down at least D% (d1_decline_pct <= -D). Entry D2 close; conditional-split exit on D3
(positives at t1, rest at t2, swept pairs); ₹5L pool allocation.

Grid: D in {0(any down),1..10} x K {2..7} x 253 (t1,t2) pairs. D=0 = baseline "any down".
"""
import sys, time, itertools
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "reversal_bounce"
BASE_POOL, PER_TRADE, EXPENSE, EXP38 = 500_000, 100_000, 0.0023, 0.0038
DAY_OPEN_HM, DAY_CLOSE_HM = 555, 915
EXIT_HMS = list(range(570, 901, 15))
SESSION_HMS = list(range(555, 916, 15))
DGRID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                 # 0 = any-down baseline
KGRID = [2, 3, 4, 5, 6, 7]; SMALL = 30


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def day_alloc(entry, dates):
    d = pd.Series(dates); cnt = d.map(d.value_counts()).values
    return np.floor(np.where(cnt <= 5, float(PER_TRADE), BASE_POOL / cnt) / entry)


def split_metrics(e, s, c, o1, o2):
    ret1 = (o1 - e) / e * 100
    exitpx = np.where(np.isnan(o1), np.nan, np.where(ret1 > 0, o1, o2))
    v = ~np.isnan(exitpx)
    e, s, c, x = e[v], s[v], c[v], exitpx[v]
    if len(e) == 0:
        return None
    pnl = s*(x-e); ret = (x-e)/e*100; npnl = pnl - c*EXPENSE; nret = ret - EXPENSE*100
    n38 = pnl - c*EXP38
    return {"n_trades": len(e), "gross_win_rate_pct": round(float((pnl>0).mean()*100), 2),
            "gross_avg_return_pct": round(float(ret.mean()), 4),
            "gross_total_return_fixedbase_pct": round(float(pnl.sum())/BASE_POOL*100, 4),
            "net_win_rate_pct": round(float((npnl>0).mean()*100), 2),
            "net_avg_return_pct": round(float(nret.mean()), 4),
            "net_median_return_pct": round(float(np.median(nret)), 4),
            "net_total_return_fixedbase_pct": round(float(npnl.sum())/BASE_POOL*100, 4),
            "net038_total_return_fixedbase_pct": round(float(n38.sum())/BASE_POOL*100, 4),
            "net_total_pnl_inr": round(float(npnl.sum()), 0),
            "avg_capital_deployed_per_trade": round(float(c.mean()), 0)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols (green + high-contain + any-down) …")

    entry_l, d3_l, decl_l, vr1_l, vr2_l, dt_l = [], [], [], [], [], []
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
        ph = raw.pivot_table(index="date", columns="hm", values="high", aggfunc="max")
        pv = raw.pivot_table(index="date", columns="hm", values="volume", aggfunc="sum").reindex(columns=SESSION_HMS).fillna(0)
        full_vol = pv.sum(axis=1)
        dates = sorted(op.index)
        day_open = op[DAY_OPEN_HM]
        day_close = cl[DAY_CLOSE_HM].where(cl[DAY_CLOSE_HM].notna(), cl.ffill(axis=1).iloc[:, -1])
        day_high = ph.max(axis=1)
        exit_cols = op[EXIT_HMS]
        for i in range(2, len(dates) - 1):
            d0, d1, d2, d3 = dates[i-2], dates[i-1], dates[i], dates[i+1]
            c0, c1 = day_close.get(d0, np.nan), day_close.get(d1, np.nan)
            o1, h1 = day_open.get(d1, np.nan), day_high.get(d1, np.nan)
            c2, o2, h2 = day_close.get(d2, np.nan), day_open.get(d2, np.nan), day_high.get(d2, np.nan)
            if not (c1 < c0):                                  # any-down base (D=0)
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
            decl_l.append((c1 - c0) / c0 * 100)
            vr1_l.append(full_vol.get(d1, np.nan)); vr2_l.append(full_vol.get(d2, np.nan)); dt_l.append(d2)
        if si % 300 == 0:
            print(f"  …{si}/{len(symbols)} ({len(entry_l)}, {time.time()-t0:.0f}s)")

    entry = np.array(entry_l, float); d3opens = np.vstack(d3_l).astype(float)
    decl = np.array(decl_l, float); vd1 = np.array(vr1_l, float); vd2 = np.array(vr2_l, float)
    dts = np.array(dt_l, dtype=object)
    vratio = np.where(vd1 > 0, vd2 / vd1, np.nan)
    pairs = list(itertools.combinations(range(len(EXIT_HMS)), 2))
    print(f"\nStacked any-down signals: {len(entry)} | sweeping {len(DGRID)*len(KGRID)*len(pairs):,} combos …")

    grid_rows = []
    per_dk_best = []
    t1 = time.time()
    for D in DGRID:
        dmask = decl <= -D if D > 0 else decl < 0
        for K in KGRID:
            m = dmask & (~np.isnan(vratio)) & (vratio >= K)
            e_m, d3_m, dt_m = entry[m], d3opens[m], dts[m]
            if len(e_m) == 0:
                per_dk_best.append({"D": D, "K": K, "n_signals": 0}); continue
            sh = day_alloc(e_m, dt_m); keep = sh > 0
            e_k, d3_k, sh_k = e_m[keep], d3_m[keep], sh[keep]; cap_k = sh_k * e_k
            best = None
            for i1, i2 in pairs:
                mm = split_metrics(e_k, sh_k, cap_k, d3_k[:, i1], d3_k[:, i2])
                if mm is None:
                    continue
                row = {"d1_decline_threshold": D, "K": K, "exit_time_1": hm_lbl(EXIT_HMS[i1]),
                       "exit_time_2": hm_lbl(EXIT_HMS[i2]), **mm}
                grid_rows.append(row)
                if best is None or row["net_total_return_fixedbase_pct"] > best["net_total_return_fixedbase_pct"]:
                    best = row
            per_dk_best.append({"D": D, "K": K, "n_signals": len(e_k), **{k: best[k] for k in
                                ["exit_time_1", "exit_time_2", "n_trades", "gross_win_rate_pct",
                                 "net_win_rate_pct", "net_avg_return_pct", "net_median_return_pct",
                                 "net_total_return_fixedbase_pct", "net038_total_return_fixedbase_pct",
                                 "avg_capital_deployed_per_trade"]}})
        print(f"  D={D} done ({time.time()-t1:.0f}s)")

    grid = pd.DataFrame(grid_rows)
    grid.to_parquet(OUTDIR / "reversal_bounce_vol_d1decline_sweep.parquet", index=False)
    gsort = grid.sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    gsort["small_sample_flag"] = np.where(gsort["n_trades"] < SMALL, "n<30", "")

    dkbest = pd.DataFrame(per_dk_best)
    dkbest["flag"] = np.where(dkbest.get("n_trades", 0).fillna(0) < SMALL, "n<30", "")
    dk_pass = dkbest[(dkbest["D"] > 0) & (dkbest.get("n_trades", 0).fillna(0) >= SMALL)]
    top20 = gsort.head(20)

    # focused D x K grid at best-overall (t1,t2)
    bt1, bt2 = top20.iloc[0]["exit_time_1"], top20.iloc[0]["exit_time_2"]
    fixed = grid[(grid["exit_time_1"] == bt1) & (grid["exit_time_2"] == bt2)]
    net_grid = fixed.pivot_table(index="d1_decline_threshold", columns="K", values="net_total_return_fixedbase_pct")
    net38_grid = fixed.pivot_table(index="d1_decline_threshold", columns="K", values="net038_total_return_fixedbase_pct")
    win_grid = fixed.pivot_table(index="d1_decline_threshold", columns="K", values="net_win_rate_pct")
    n_by_D = dkbest.groupby("D")["n_signals"].max().reset_index().rename(columns={"n_signals": "max_n_over_K"})

    with pd.ExcelWriter(OUTDIR / "reversal_bounce_vol_d1decline_condensed.xlsx", engine="openpyxl") as w:
        dkbest.to_excel(w, sheet_name="best_per_D_K", index=False)
        top20.to_excel(w, sheet_name="top20", index=False)
        net_grid.to_excel(w, sheet_name=f"DxK_net023_{bt1}-{bt2}".replace(":", ""))
        net38_grid.to_excel(w, sheet_name=f"DxK_net038_{bt1}-{bt2}".replace(":", ""))
        win_grid.to_excel(w, sheet_name=f"DxK_win_{bt1}-{bt2}".replace(":", ""))
        n_by_D.to_excel(w, sheet_name="n_by_D", index=False)

    pd.set_option("display.width", 270)
    print(f"\ngrid rows: {len(grid):,} | (D,K) best rows: {len(dkbest)} | "
          f"(D>=1,K) surviving n>=30: {len(dk_pass)}/60")
    print("\n=== TOP 20 (D, K, t1, t2) by net total_return ===")
    print(top20[["d1_decline_threshold", "K", "exit_time_1", "exit_time_2", "n_trades",
                 "net_win_rate_pct", "net_avg_return_pct", "net_total_return_fixedbase_pct",
                 "net038_total_return_fixedbase_pct", "small_sample_flag"]].to_string(index=False))
    print(f"\n=== D x K net@0.23% total_return @ fixed exit {bt1}->{bt2} (best overall pair) ===")
    print(net_grid.round(2).to_string())
    print(f"\n=== D x K net@0.38% total_return @ {bt1}->{bt2} ===")
    print(net38_grid.round(2).to_string())
    print(f"\n=== D x K net WIN RATE @ {bt1}->{bt2} ===")
    print(win_grid.round(1).to_string())
    print("\n=== n_signals by D (max across K) ===")
    print(n_by_D.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
