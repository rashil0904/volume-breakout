# -*- coding: utf-8 -*-
"""
reversal_bounce_split_sweep.py
==============================
2-day reversal (stacked: base + high[D2]<high[D1] + vol[D2]>=K*vol[D1], K 2-7) with
CONDITIONAL-SPLIT exit on D3 and a full (t1,t2) time-pair sweep.
Entry D2 close; ₹5L pool (₹1L/trade if <=5 signals else ₹5L/n split).

Conditional split per (t1<t2): return_at_t1 = (open_t1 - entry)/entry*100.
  return_at_t1 > 0  -> exit 100% at t1 open  (positives out first)
  return_at_t1 <= 0 -> exit 100% at t2 open  (rest out later)
C(23,2)=253 pairs x 6 K = 1518 combos. gross + net@0.23%.
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
BASE_POOL, PER_TRADE, EXPENSE = 500_000, 100_000, 0.0023
DAY_OPEN_HM, DAY_CLOSE_HM = 555, 915
EXIT_HMS = list(range(570, 901, 15))                      # 23 D3 times 09:30..15:00
SESSION_HMS = list(range(555, 916, 15))
KGRID = [2, 3, 4, 5, 6, 7]; SMALL = 30
# prior SINGLE full-exit best (stacked + allocation) for comparison:
PRIOR_SINGLE = {2: (1.2768, "10:00"), 3: (5.4902, "09:30"), 4: (2.4415, "15:00"),
                5: (0.9296, "15:00"), 6: (-0.8443, "09:30"), 7: (-0.3698, "15:00")}


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def day_alloc(entry, dates):
    d = pd.Series(dates); cnt = d.map(d.value_counts()).values
    return np.floor(np.where(cnt <= 5, float(PER_TRADE), BASE_POOL / cnt) / entry)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols (stacked base+high-contain) …")

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
            if not (c1 < c0):
                continue
            if not (c2 > o2 and c2 > o1):
                continue
            if not (h2 < h1):
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
    print(f"\nStacked signals: {len(entry)} | sweeping {len(KGRID)*len(pairs)} (K,t1,t2) combos …")

    rows = []
    for K in KGRID:
        m = (~np.isnan(vratio)) & (vratio >= K)
        e_m, d3_m, dt_m = entry[m], d3opens[m], dts[m]
        sh = day_alloc(e_m, dt_m); keep = sh > 0
        e_k, d3_k, sh_k = e_m[keep], d3_m[keep], sh[keep]
        cap_k = sh_k * e_k
        for i1, i2 in pairs:
            o1 = d3_k[:, i1]; o2 = d3_k[:, i2]
            ret1 = (o1 - e_k) / e_k * 100
            pos = ret1 > 0
            exitpx = np.where(pos, o1, o2)
            v = (~np.isnan(o1)) & (~np.isnan(exitpx))
            e, s, c, x = e_k[v], sh_k[v], cap_k[v], exitpx[v]
            nv = len(e)
            if nv == 0:
                continue
            pnl = s * (x - e); ret = (x - e) / e * 100
            npnl = pnl - c * EXPENSE; nret = ret - EXPENSE * 100
            rows.append({"K": K, "exit_time_1": hm_lbl(EXIT_HMS[i1]), "exit_time_2": hm_lbl(EXIT_HMS[i2]),
                         "n_trades": nv, "gross_win_rate_pct": round(float((pnl > 0).mean()*100), 2),
                         "gross_avg_return_pct": round(float(ret.mean()), 4),
                         "gross_median_return_pct": round(float(np.median(ret)), 4),
                         "gross_total_return_fixedbase_pct": round(float(pnl.sum())/BASE_POOL*100, 4),
                         "gross_total_pnl_inr": round(float(pnl.sum()), 0),
                         "net_win_rate_pct": round(float((npnl > 0).mean()*100), 2),
                         "net_avg_return_pct": round(float(nret.mean()), 4),
                         "net_median_return_pct": round(float(np.median(nret)), 4),
                         "net_total_return_fixedbase_pct": round(float(npnl.sum())/BASE_POOL*100, 4),
                         "net_total_pnl_inr": round(float(npnl.sum()), 0),
                         "avg_capital_deployed_per_trade": round(float(c.mean()), 0)})
    grid = pd.DataFrame(rows).sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    grid["small_sample_flag"] = np.where(grid["n_trades"] < SMALL, "n<30", "")

    # per-K best + comparison vs prior single-exit
    best_per_K = []
    for K in KGRID:
        sub = grid[grid["K"] == K]
        if len(sub) == 0:
            continue
        b = sub.iloc[0]; ps, pe = PRIOR_SINGLE[K]
        best_per_K.append({"K": K, "best_t1": b["exit_time_1"], "best_t2": b["exit_time_2"],
                           "n_trades": int(b["n_trades"]), "net_win_rate_pct": b["net_win_rate_pct"],
                           "net_avg_return_pct": b["net_avg_return_pct"], "net_median_return_pct": b["net_median_return_pct"],
                           "split_net_total_pct": b["net_total_return_fixedbase_pct"],
                           "prior_single_net_total_pct": ps, "prior_single_exit": pe,
                           "split_minus_single_pts": round(b["net_total_return_fixedbase_pct"] - ps, 4),
                           "flag": "n<30" if int(b["n_trades"]) < SMALL else ""})
    bestK = pd.DataFrame(best_per_K)
    top15 = grid.head(15)

    # heatmap for the best K overall
    bK = int(grid.iloc[0]["K"])
    sub = grid[grid["K"] == bK]
    M = np.full((len(EXIT_HMS), len(EXIT_HMS)), np.nan)
    idx = {hm_lbl(h): k for k, h in enumerate(EXIT_HMS)}
    for _, r in sub.iterrows():
        M[idx[r["exit_time_1"]], idx[r["exit_time_2"]]] = r["net_total_return_fixedbase_pct"]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(EXIT_HMS))); ax.set_xticklabels([hm_lbl(h) for h in EXIT_HMS], rotation=90, fontsize=6)
    ax.set_yticks(range(len(EXIT_HMS))); ax.set_yticklabels([hm_lbl(h) for h in EXIT_HMS], fontsize=6)
    ax.set_xlabel("t2 (rest exit)"); ax.set_ylabel("t1 (positives exit)")
    ax.set_title(f"Conditional-split net total_return_fixedbase — best K={bK}")
    fig.colorbar(im, ax=ax, label="net total_return_fixedbase (%)")
    fig.tight_layout(); fig.savefig(OUTDIR / "reversal_bounce_split_heatmap.png", dpi=120); plt.close(fig)

    grid.to_parquet(OUTDIR / "reversal_bounce_split_sweep.parquet", index=False)
    with pd.ExcelWriter(OUTDIR / "reversal_bounce_split_sweep.xlsx", engine="openpyxl") as w:
        grid.to_excel(w, sheet_name="full_grid_1518", index=False)
        bestK.to_excel(w, sheet_name="best_per_K_vs_single", index=False)
        top15.to_excel(w, sheet_name="top15", index=False)

    pd.set_option("display.width", 260)
    show = ["K", "exit_time_1", "exit_time_2", "n_trades", "gross_win_rate_pct", "net_win_rate_pct",
            "net_avg_return_pct", "net_median_return_pct", "net_total_return_fixedbase_pct", "small_sample_flag"]
    print("\n=== TOP 15 (K, t1, t2) by net total_return_fixedbase_pct ===")
    print(top15[show].to_string(index=False))
    print("\n=== BEST SPLIT per K vs prior SINGLE full-exit (stacked+alloc) ===")
    print(bestK.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
