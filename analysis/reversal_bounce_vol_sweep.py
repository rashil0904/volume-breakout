# -*- coding: utf-8 -*-
"""
reversal_bounce_vol_sweep.py
============================
Add a D2-vs-D1 full-day VOLUME filter to the 2-day reversal-bounce strategy and sweep the
threshold K in {2,3,4,5,6,7}. Base pattern + D3 exit sweep unchanged (from
reversal_bounce_exit_sweep.py); only vol_ratio = volume[D2]/volume[D1] >= K is added.

Universe mcap ₹1,500-5,000 Cr, entry = D2 close, D3 exit grid 09:30..15:00 (23 times),
₹5L/₹1L, cost 0.23%. Full-day volume both days; zero/missing-volume days excluded from ratios.
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
EXIT_HMS = list(range(570, 901, 15))                 # 23 exit times 09:30..15:00
SESSION_HMS = list(range(555, 916, 15))
KGRID = [2, 3, 4, 5, 6, 7]
CHART_EXIT = "09:30"                                 # fixed exit for the K chart (base best)
SMALL = 30


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def sweep(entry, d3opens, shares, cap):
    rows = []
    for j, hm in enumerate(EXIT_HMS):
        px = d3opens[:, j]; v = ~np.isnan(px)
        e, s, c, x = entry[v], shares[v], cap[v], px[v]
        nv = len(e)
        if nv == 0:
            continue
        pnl = s * (x - e); ret = (x - e) / e * 100
        npnl = pnl - c * EXPENSE; nret = ret - EXPENSE * 100
        rows.append({"exit_time": hm_lbl(hm), "n_trades": nv,
                     "gross_win_rate_pct": round(float((pnl > 0).mean()*100), 2),
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
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols …")

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
            if not (c2 > o1 and c2 > o2):
                continue
            if (sym, d2) not in eligible:
                continue
            entry = c2
            if not (entry == entry and entry > 0):
                continue
            entry_l.append(entry)
            d3_l.append(exit_cols.loc[d3].values if d3 in exit_cols.index else np.full(len(EXIT_HMS), np.nan))
            vd1_l.append(full_vol.get(d1, np.nan)); vd2_l.append(full_vol.get(d2, np.nan))
        if si % 200 == 0:
            print(f"  …{si}/{len(symbols)} ({len(entry_l)} signals, {time.time()-t0:.0f}s)")

    entry = np.array(entry_l, float); d3opens = np.vstack(d3_l).astype(float)
    vd1 = np.array(vd1_l, float); vd2 = np.array(vd2_l, float)
    shares = np.floor(PER_TRADE / entry); ok = shares > 0
    entry, d3opens, vd1, vd2, shares = entry[ok], d3opens[ok], vd1[ok], vd2[ok], shares[ok]
    cap = shares * entry
    vratio = np.where((vd1 > 0), vd2 / vd1, np.nan)
    print(f"\nBase pattern signals (valid): {len(entry)} | with valid volume ratio: {int((~np.isnan(vratio)).sum())}")

    full_grid, summary = [], []
    # baseline (no volume filter)
    base_tab = sweep(entry, d3opens, shares, cap)
    base_tab.insert(0, "K", "none (base)")
    full_grid.append(base_tab)
    bb = base_tab.sort_values("net_total_return_fixedbase_pct", ascending=False).iloc[0]
    summary.append({"K": "none (base)", "n_trades": int(bb["n_trades"]), "best_exit_time": bb["exit_time"],
                    "gross_win_rate_pct": bb["gross_win_rate_pct"], "net_win_rate_pct": bb["net_win_rate_pct"],
                    "gross_avg_return_pct": bb["gross_avg_return_pct"], "net_avg_return_pct": bb["net_avg_return_pct"],
                    "net_median_return_pct": bb["net_median_return_pct"],
                    "net_total_return_fixedbase_pct": bb["net_total_return_fixedbase_pct"],
                    "flag": ""})

    for K in KGRID:
        m = (~np.isnan(vratio)) & (vratio >= K)
        nK = int(m.sum())
        t = sweep(entry[m], d3opens[m], shares[m], cap[m])
        t.insert(0, "K", K)
        full_grid.append(t)
        if len(t) == 0:
            summary.append({"K": K, "n_trades": 0, "best_exit_time": "", "flag": "no trades"}); continue
        b = t.sort_values("net_total_return_fixedbase_pct", ascending=False).iloc[0]
        summary.append({"K": K, "n_trades": int(b["n_trades"]), "best_exit_time": b["exit_time"],
                        "gross_win_rate_pct": b["gross_win_rate_pct"], "net_win_rate_pct": b["net_win_rate_pct"],
                        "gross_avg_return_pct": b["gross_avg_return_pct"], "net_avg_return_pct": b["net_avg_return_pct"],
                        "net_median_return_pct": b["net_median_return_pct"],
                        "net_total_return_fixedbase_pct": b["net_total_return_fixedbase_pct"],
                        "flag": "n<30 (untrustworthy)" if int(b["n_trades"]) < SMALL else ""})
    grid = pd.concat(full_grid, ignore_index=True)
    summ = pd.DataFrame(summary)

    # chart: net total at fixed exit (09:30) vs K, baseline line
    def at_exit(tab):
        r = tab[tab["exit_time"] == CHART_EXIT]
        return float(r["net_total_return_fixedbase_pct"].iloc[0]) if len(r) else np.nan
    base_line = at_exit(base_tab)
    ks, ys = [], []
    for K in KGRID:
        sub = grid[(grid["K"] == K) & (grid["exit_time"] == CHART_EXIT)]
        ks.append(K); ys.append(float(sub["net_total_return_fixedbase_pct"].iloc[0]) if len(sub) else np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(ks, ys, "o-", color="#1f77b4", label=f"net total @ {CHART_EXIT} exit")
    ax.axhline(base_line, ls="--", color="#d62728", label=f"no-filter baseline ({base_line:.1f}%)")
    ax.set_xlabel("volume threshold K (D2 vol >= K × D1 vol)"); ax.set_ylabel("net total_return_fixedbase (%)")
    ax.set_title(f"Reversal bounce: volume filter K vs net total return (exit {CHART_EXIT})")
    ax.set_xticks(KGRID); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "reversal_bounce_vol_sweep.png", dpi=120); plt.close(fig)

    with pd.ExcelWriter(OUTDIR / "reversal_bounce_vol_sweep.xlsx", engine="openpyxl") as w:
        grid.to_excel(w, sheet_name="full_grid_K_x_exit", index=False)
        summ.to_excel(w, sheet_name="best_per_K", index=False)

    pd.set_option("display.width", 240)
    print("\n=== BEST EXIT per K (+ no-filter baseline) — net-sorted metrics ===")
    cols = ["K", "n_trades", "best_exit_time", "gross_win_rate_pct", "net_win_rate_pct",
            "gross_avg_return_pct", "net_avg_return_pct", "net_median_return_pct",
            "net_total_return_fixedbase_pct", "flag"]
    print(summ[cols].to_string(index=False))
    print(f"\n  n_trades by K: base {len(entry)} -> " +
          ", ".join(f"K{K}:{int((~np.isnan(vratio) & (vratio>=K)).sum())}" for K in KGRID))
    print(f"\n  chart exit fixed at {CHART_EXIT}; baseline net total there = {base_line:.2f}%")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
