# -*- coding: utf-8 -*-
"""
reversal_bounce_contain_vol.py
==============================
2-day reversal bounce + high-containment (high[D2]<high[D1]) + SWEPT D2/D1 volume filter.
Universe mcap ₹1,500-5,000 Cr, entry D2 close, next-day D3 15-min full-exit sweep, ₹5L/₹1L.

Pattern (all must hold): D1 down (close[D1]<close[D0]); D2 green (close[D2]>open[D2] and
close[D2]>open[D1]); high[D2]<high[D1]; volume[D2] >= K x volume[D1] (K in 2..7, full-day vol).
"""
import sys, time
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
EXIT_HMS = list(range(570, 901, 15))
SESSION_HMS = list(range(555, 916, 15))
KGRID = [2, 3, 4, 5, 6, 7]
CHART_EXIT = "09:30"; SMALL = 30
# base pattern only (no high, no vol), best exit 09:30 — from prior run:
BASE_ONLY = {"n": 41131, "best_exit": "09:30", "gross_win": 54.27, "net_win": 46.57,
             "gross_avg": 0.2423, "net_avg": 0.0123, "net_med": -0.0968, "net_total": 102.7232}


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def sweep(entry, d3opens, shares, cap):
    rows = []
    for j, hm in enumerate(EXIT_HMS):
        px = d3opens[:, j]; v = ~np.isnan(px)
        e, s, c, x = entry[v], shares[v], cap[v], px[v]
        if len(e) == 0:
            continue
        pnl = s*(x-e); ret = (x-e)/e*100; npnl = pnl - c*EXPENSE; nret = ret - EXPENSE*100
        rows.append({"exit_time": hm_lbl(hm), "n_trades": len(e),
                     "gross_win_rate_pct": round(float((pnl>0).mean()*100), 2),
                     "gross_avg_return_pct": round(float(ret.mean()), 4),
                     "gross_median_return_pct": round(float(np.median(ret)), 4),
                     "gross_total_return_fixedbase_pct": round(float(pnl.sum())/BASE_POOL*100, 4),
                     "gross_total_pnl_inr": round(float(pnl.sum()), 0),
                     "net_win_rate_pct": round(float((npnl>0).mean()*100), 2),
                     "net_avg_return_pct": round(float(nret.mean()), 4),
                     "net_median_return_pct": round(float(np.median(nret)), 4),
                     "net_total_return_fixedbase_pct": round(float(npnl.sum())/BASE_POOL*100, 4),
                     "net_total_pnl_inr": round(float(npnl.sum()), 0),
                     "avg_capital_deployed_per_trade": round(float(c.mean()), 0)})
    return pd.DataFrame(rows).sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)


def brow(tag, t):
    b = t.iloc[0]
    return {"config": tag, "n_trades": int(b["n_trades"]), "best_exit": b["exit_time"],
            "gross_win_rate_pct": b["gross_win_rate_pct"], "net_win_rate_pct": b["net_win_rate_pct"],
            "gross_avg_return_pct": b["gross_avg_return_pct"], "net_avg_return_pct": b["net_avg_return_pct"],
            "net_median_return_pct": b["net_median_return_pct"],
            "net_total_return_fixedbase_pct": b["net_total_return_fixedbase_pct"],
            "flag": "n<30" if int(b["n_trades"]) < SMALL else ""}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols (base + high[D2]<high[D1]) …")

    entry_l, d3_l, vd1_l, vd2_l = [], [], [], []
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
            vd1_l.append(full_vol.get(d1, np.nan)); vd2_l.append(full_vol.get(d2, np.nan))
        if si % 250 == 0:
            print(f"  …{si}/{len(symbols)} ({len(entry_l)} signals, {time.time()-t0:.0f}s)")

    entry = np.array(entry_l, float); d3opens = np.vstack(d3_l).astype(float)
    vd1 = np.array(vd1_l, float); vd2 = np.array(vd2_l, float)
    shares = np.floor(PER_TRADE / entry); ok = shares > 0
    entry, d3opens, vd1, vd2, shares = entry[ok], d3opens[ok], vd1[ok], vd2[ok], shares[ok]
    cap = shares * entry
    vratio = np.where(vd1 > 0, vd2 / vd1, np.nan)
    print(f"\nBase+high-contain signals: {len(entry)} | valid vol ratio: {int((~np.isnan(vratio)).sum())}")

    full_grid = []
    contain_tab = sweep(entry, d3opens, shares, cap)                 # (b) base + high-contain, no vol
    contain_tab.insert(0, "K", "none (contain only)"); full_grid.append(contain_tab)

    summary = [{"config": "(a) base only", **{k: BASE_ONLY[m] for k, m in [
                ("n_trades", "n"), ("best_exit", "best_exit"), ("gross_win_rate_pct", "gross_win"),
                ("net_win_rate_pct", "net_win"), ("gross_avg_return_pct", "gross_avg"),
                ("net_avg_return_pct", "net_avg"), ("net_median_return_pct", "net_med"),
                ("net_total_return_fixedbase_pct", "net_total")]}, "flag": ""},
               {**brow("(b) base + high-contain", contain_tab)}]

    for K in KGRID:
        m = (~np.isnan(vratio)) & (vratio >= K)
        t = sweep(entry[m], d3opens[m], shares[m], cap[m])
        tt = t.copy(); tt.insert(0, "K", K); full_grid.append(tt)
        if len(t):
            summary.append({**brow(f"(c) contain + vol K={K}", t)})
        else:
            summary.append({"config": f"(c) contain + vol K={K}", "n_trades": 0, "flag": "no trades"})

    grid = pd.concat(full_grid, ignore_index=True)
    summ = pd.DataFrame(summary)

    # chart: net total @ 09:30 vs K, with (a) and (b) baselines
    def at(tab):
        r = tab[tab["exit_time"] == CHART_EXIT]
        return float(r["net_total_return_fixedbase_pct"].iloc[0]) if len(r) else np.nan
    base_a = BASE_ONLY["net_total"]; base_b = at(contain_tab)
    ys = []
    for K in KGRID:
        sub = grid[(grid["K"] == K) & (grid["exit_time"] == CHART_EXIT)]
        ys.append(float(sub["net_total_return_fixedbase_pct"].iloc[0]) if len(sub) else np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(KGRID, ys, "o-", color="#1f77b4", label=f"contain + vol K @ {CHART_EXIT}")
    ax.axhline(base_a, ls="--", color="#d62728", label=f"(a) base only ({base_a:.1f}%)")
    ax.axhline(base_b, ls=":", color="#ff7f0e", label=f"(b) contain only ({base_b:.1f}%)")
    ax.set_xlabel("volume threshold K"); ax.set_ylabel("net total_return_fixedbase (%)")
    ax.set_title(f"Reversal bounce: high-contain + volume K vs baselines (exit {CHART_EXIT})")
    ax.set_xticks(KGRID); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "reversal_bounce_contain_vol.png", dpi=120); plt.close(fig)

    with pd.ExcelWriter(OUTDIR / "reversal_bounce_contain_vol.xlsx", engine="openpyxl") as w:
        grid.to_excel(w, sheet_name="full_grid_K_x_exit", index=False)
        summ.to_excel(w, sheet_name="summary_stacked", index=False)

    pd.set_option("display.width", 240)
    cols = ["config", "n_trades", "best_exit", "gross_win_rate_pct", "net_win_rate_pct",
            "gross_avg_return_pct", "net_avg_return_pct", "net_median_return_pct",
            "net_total_return_fixedbase_pct", "flag"]
    print("\n=== STACKED-FILTER SUMMARY (best exit each) ===")
    print(summ[cols].to_string(index=False))
    print(f"\n  n by K (contain+vol): " +
          ", ".join(f"K{K}:{int((~np.isnan(vratio) & (vratio>=K)).sum())}" for K in KGRID))
    print(f"\n  chart @ {CHART_EXIT}: (a) base {base_a:.1f}% | (b) contain {base_b:.1f}%")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
