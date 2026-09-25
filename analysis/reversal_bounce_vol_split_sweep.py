# -*- coding: utf-8 -*-
"""
reversal_bounce_vol_split_sweep.py
==================================
Base 2-day reversal bounce + volume filter (K 2-7, NO high-containment) with CONDITIONAL-SPLIT
exit on D3 and a full (t1,t2) time-pair sweep. ₹5L-pool allocation. Reports net@0.23% and 0.38%,
and compares the best split to a 100% full exit at 09:30 for the same K.

Conditional split (t1<t2): return_at_t1>0 -> exit 100% at t1 open; else exit 100% at t2 open.
"""
import sys, time, itertools
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
I0930 = 0                                                   # 09:30 is EXIT_HMS[0]


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def day_alloc(entry, dates):
    d = pd.Series(dates); cnt = d.map(d.value_counts()).values
    return np.floor(np.where(cnt <= 5, float(PER_TRADE), BASE_POOL / cnt) / entry)


def metrics(e, s, c, exitpx):
    v = ~np.isnan(exitpx)
    e, s, c, x = e[v], s[v], c[v], exitpx[v]
    if len(e) == 0:
        return None
    pnl = s*(x-e); ret = (x-e)/e*100
    n23 = pnl - c*EXP23; r23 = ret - EXP23*100; n38 = pnl - c*EXP38
    return {"n_trades": len(e), "gross_win_rate_pct": round(float((pnl > 0).mean()*100), 2),
            "gross_avg_return_pct": round(float(ret.mean()), 4),
            "net_win_rate_pct": round(float((n23 > 0).mean()*100), 2),
            "net_avg_return_pct": round(float(r23.mean()), 4),
            "net_median_return_pct": round(float(np.median(r23)), 4),
            "net_total_return_fixedbase_pct": round(float(n23.sum())/BASE_POOL*100, 4),
            "net038_total_return_fixedbase_pct": round(float(n38.sum())/BASE_POOL*100, 4),
            "avg_capital_deployed_per_trade": round(float(c.mean()), 0)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols (base + volume, NO high) …")

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
        if si % 300 == 0:
            print(f"  …{si}/{len(symbols)} ({len(entry_l)}, {time.time()-t0:.0f}s)")

    entry = np.array(entry_l, float); d3opens = np.vstack(d3_l).astype(float)
    vd1 = np.array(vd1_l, float); vd2 = np.array(vd2_l, float); dts = np.array(dt_l, dtype=object)
    vratio = np.where(vd1 > 0, vd2 / vd1, np.nan)
    pairs = list(itertools.combinations(range(len(EXIT_HMS)), 2))
    print(f"\nVolume-set signals: {len(entry)} | sweeping {len(KGRID)*len(pairs)} split combos …")

    grid_rows, cmp_rows = [], []
    for K in KGRID:
        m = (~np.isnan(vratio)) & (vratio >= K)
        e_m, d3_m, dt_m = entry[m], d3opens[m], dts[m]
        sh = day_alloc(e_m, dt_m); keep = sh > 0
        e_k, d3_k, sh_k = e_m[keep], d3_m[keep], sh[keep]; cap_k = sh_k * e_k
        # 100% full exit at 09:30
        single = metrics(e_k, sh_k, cap_k, d3_k[:, I0930])
        # conditional-split sweep
        best = None
        for i1, i2 in pairs:
            o1 = d3_k[:, i1]; ret1 = (o1 - e_k) / e_k * 100
            exitpx = np.where(ret1 > 0, o1, d3_k[:, i2])
            exitpx = np.where(np.isnan(o1), np.nan, exitpx)
            mm = metrics(e_k, sh_k, cap_k, exitpx)
            if mm is None:
                continue
            row = {"K": K, "exit_time_1": hm_lbl(EXIT_HMS[i1]), "exit_time_2": hm_lbl(EXIT_HMS[i2]), **mm}
            grid_rows.append(row)
            if best is None or row["net_total_return_fixedbase_pct"] > best["net_total_return_fixedbase_pct"]:
                best = row
        cmp_rows.append({
            "K": K, "n_trades": single["n_trades"],
            "single0930_net_win": single["net_win_rate_pct"], "single0930_net_avg": single["net_avg_return_pct"],
            "single0930_net_med": single["net_median_return_pct"], "single0930_net_total": single["net_total_return_fixedbase_pct"],
            "single0930_net038_total": single["net038_total_return_fixedbase_pct"],
            "split_t1": best["exit_time_1"], "split_t2": best["exit_time_2"],
            "split_net_win": best["net_win_rate_pct"], "split_net_avg": best["net_avg_return_pct"],
            "split_net_med": best["net_median_return_pct"], "split_net_total": best["net_total_return_fixedbase_pct"],
            "split_net038_total": best["net038_total_return_fixedbase_pct"],
            "split_minus_single_pts": round(best["net_total_return_fixedbase_pct"] - single["net_total_return_fixedbase_pct"], 4),
            "flag": "n<30" if single["n_trades"] < SMALL else ""})

    grid = pd.DataFrame(grid_rows).sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    grid["small_sample_flag"] = np.where(grid["n_trades"] < SMALL, "n<30", "")
    cmp = pd.DataFrame(cmp_rows)

    bK = int(grid.iloc[0]["K"]); sub = grid[grid["K"] == bK]
    M = np.full((len(EXIT_HMS), len(EXIT_HMS)), np.nan)
    idx = {hm_lbl(h): k for k, h in enumerate(EXIT_HMS)}
    for _, r in sub.iterrows():
        M[idx[r["exit_time_1"]], idx[r["exit_time_2"]]] = r["net_total_return_fixedbase_pct"]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(EXIT_HMS))); ax.set_xticklabels([hm_lbl(h) for h in EXIT_HMS], rotation=90, fontsize=6)
    ax.set_yticks(range(len(EXIT_HMS))); ax.set_yticklabels([hm_lbl(h) for h in EXIT_HMS], fontsize=6)
    ax.set_xlabel("t2 (rest exit)"); ax.set_ylabel("t1 (positives exit)")
    ax.set_title(f"Volume-only conditional-split net total_return — best K={bK}")
    fig.colorbar(im, ax=ax, label="net total_return_fixedbase (%)")
    fig.tight_layout(); fig.savefig(OUTDIR / "reversal_bounce_vol_split_heatmap.png", dpi=120); plt.close(fig)

    grid.to_parquet(OUTDIR / "reversal_bounce_vol_split_sweep.parquet", index=False)
    with pd.ExcelWriter(OUTDIR / "reversal_bounce_vol_split_sweep.xlsx", engine="openpyxl") as w:
        grid.to_excel(w, sheet_name="full_grid", index=False)
        cmp.to_excel(w, sheet_name="split_vs_single0930", index=False)
        grid.head(15).to_excel(w, sheet_name="top15", index=False)

    pd.set_option("display.width", 270)
    print("\n=== TOP 15 (K, t1, t2) by net@0.23% total_return ===")
    print(grid.head(15)[["K", "exit_time_1", "exit_time_2", "n_trades", "net_win_rate_pct",
                         "net_avg_return_pct", "net_median_return_pct", "net_total_return_fixedbase_pct",
                         "net038_total_return_fixedbase_pct", "small_sample_flag"]].to_string(index=False))
    print("\n=== BEST SPLIT vs 100% FULL EXIT @ 09:30 (volume-only, per K) ===")
    print(cmp.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
