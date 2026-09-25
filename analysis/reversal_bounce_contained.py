# -*- coding: utf-8 -*-
"""
reversal_bounce_contained.py
============================
2-day reversal bounce + NEW condition high[D2] < high[D1] (contained bounce). NO volume filter.
Universe mcap ₹1,500-5,000 Cr, entry at D2 close, next-day D3 15-min full-exit sweep, ₹5L/₹1L.

Pattern (all must hold, daily candles, 3 consecutive days D0,D1,D2):
  D1 down     : close[D1] < close[D0]
  D2 green    : close[D2] > open[D2] AND close[D2] > open[D1]
  D2 contained: high[D2] < high[D1]      (strict; bounce stays below the down-day peak)
  mcap on D2 in band. Enter at close[D2]. Exit D3 09:30..15:00 (23 times), cost 0.23%.
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
# base-pattern (no high condition) reference, from the prior run:
BASE_REF = {"n": 41131, "best_exit": "09:30", "gross_win": 54.27, "net_win": 46.57,
            "gross_avg": 0.2423, "net_avg": 0.0123, "net_med": -0.0968, "net_total": 102.7232}


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols (contained bounce: high[D2] < high[D1]) …")

    entry_l, d3_l = [], []
    sig_dates = []
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
            if not (h2 < h1):                              # NEW contained-bounce condition
                continue
            if (sym, d2) not in eligible:
                continue
            entry = c2
            if not (entry == entry and entry > 0):
                continue
            entry_l.append(entry)
            d3_l.append(exit_cols.loc[d3].values if d3 in exit_cols.index else np.full(len(EXIT_HMS), np.nan))
            sig_dates.append(d2)
        if si % 200 == 0:
            print(f"  …{si}/{len(symbols)} ({len(entry_l)} signals, {time.time()-t0:.0f}s)")

    entry = np.array(entry_l, float); d3opens = np.vstack(d3_l).astype(float)
    shares = np.floor(PER_TRADE / entry); ok = shares > 0
    entry, d3opens, shares = entry[ok], d3opens[ok], shares[ok]
    cap = shares * entry
    n_sig = len(entry)
    print(f"\nContained-bounce signals (valid): {n_sig}")

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
    tab = pd.DataFrame(rows).sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    best = tab.iloc[0]

    # comparison vs base pattern (no high condition)
    cmp = pd.DataFrame([
        {"pattern": "base (no high cond)", "n_trades": BASE_REF["n"], "best_exit": BASE_REF["best_exit"],
         "gross_win_rate_pct": BASE_REF["gross_win"], "net_win_rate_pct": BASE_REF["net_win"],
         "gross_avg_return_pct": BASE_REF["gross_avg"], "net_avg_return_pct": BASE_REF["net_avg"],
         "net_median_return_pct": BASE_REF["net_med"], "net_total_return_fixedbase_pct": BASE_REF["net_total"]},
        {"pattern": "contained (high[D2]<high[D1])", "n_trades": int(best["n_trades"]), "best_exit": best["exit_time"],
         "gross_win_rate_pct": best["gross_win_rate_pct"], "net_win_rate_pct": best["net_win_rate_pct"],
         "gross_avg_return_pct": best["gross_avg_return_pct"], "net_avg_return_pct": best["net_avg_return_pct"],
         "net_median_return_pct": best["net_median_return_pct"],
         "net_total_return_fixedbase_pct": best["net_total_return_fixedbase_pct"]},
    ])

    ch = tab.set_index("exit_time").reindex([hm_lbl(h) for h in EXIT_HMS])
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ch["net_total_return_fixedbase_pct"]]
    ax.bar(ch.index, ch["net_total_return_fixedbase_pct"], color=colors)
    ax.axhline(0, color="#333", lw=.8)
    ax.set_xlabel("D3 exit time"); ax.set_ylabel("net total_return_fixedbase (%)")
    ax.set_title("Contained reversal bounce (high[D2]<high[D1]): D3 exit-time sweep — net total return")
    plt.xticks(rotation=60); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "reversal_bounce_contained.png", dpi=120); plt.close(fig)

    with pd.ExcelWriter(OUTDIR / "reversal_bounce_contained.xlsx", engine="openpyxl") as w:
        tab.to_excel(w, sheet_name="exit_time_sweep", index=False)
        cmp.to_excel(w, sheet_name="vs_base_pattern", index=False)

    n_days = pd.Series(sig_dates).nunique()
    pd.set_option("display.width", 240)
    show = ["exit_time", "n_trades", "gross_win_rate_pct", "gross_avg_return_pct",
            "gross_total_return_fixedbase_pct", "net_win_rate_pct", "net_avg_return_pct",
            "net_median_return_pct", "net_total_return_fixedbase_pct", "net_total_pnl_inr"]
    print("\n=== EXIT-TIME SWEEP (contained bounce, sorted by net total_return_fixedbase_pct) ===")
    print(tab[show].to_string(index=False))
    print(f"\nBEST EXIT: {best['exit_time']} | n={int(best['n_trades'])} | "
          f"net total {best['net_total_return_fixedbase_pct']}% (gross {best['gross_total_return_fixedbase_pct']}%) | "
          f"net win {best['net_win_rate_pct']}% | net avg {best['net_avg_return_pct']}% | net med {best['net_median_return_pct']}%")
    print(f"\nPATTERN FREQUENCY: {n_sig} signals over {n_days} days ({n_sig/n_days:.2f}/day) "
          f"vs base {BASE_REF['n']} ({BASE_REF['n']/n_days:.1f}/day) -> "
          f"{(1-n_sig/BASE_REF['n'])*100:.1f}% fewer trades")
    print("\n=== vs BASE PATTERN (best exit each) ===")
    print(cmp.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
