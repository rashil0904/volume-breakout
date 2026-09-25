# -*- coding: utf-8 -*-
"""
reversal_bounce_contain_vol_alloc.py
====================================
Stacked reversal-bounce (base + high[D2]<high[D1] + vol[D2]>=K*vol[D1], K 2-7) with the main
strategy's ₹5L-POOL daily allocation applied per D2 signal day:
  n_signals that day <=5 -> ₹1L each (leftover uninvested); >5 -> ₹5L/n_signals equal split.
  shares = alloc // entry (D2 close); capital_deployed = shares*entry (actual).
Allocation uses each K's OWN daily qualifying count. D3 15-min full-exit sweep unchanged.
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
BASE_POOL, PER_TRADE, EXPENSE = 500_000, 100_000, 0.0023
DAY_OPEN_HM, DAY_CLOSE_HM = 555, 915
EXIT_HMS = list(range(570, 901, 15))
SESSION_HMS = list(range(555, 916, 15))
KGRID = [2, 3, 4, 5, 6, 7]; SMALL = 30
# prior flat-₹1L stacked totals (net total_return_fixedbase @ best exit) to flag overstatement:
PRIOR_FLAT = {2: 1.7470, 3: 5.4902, 4: 2.4415, 5: 0.9296, 6: -0.8443, 7: -0.3698}


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def day_alloc(entry, dates):
    """Per-trade allocation from the ₹5L pool given that day's signal count."""
    d = pd.Series(dates)
    cnt = d.map(d.value_counts()).values
    target = np.where(cnt <= 5, float(PER_TRADE), BASE_POOL / cnt)
    shares = np.floor(target / entry)
    return shares, cnt


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


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols (base + high-contain, capturing D2 date + vol) …")

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
        if si % 250 == 0:
            print(f"  …{si}/{len(symbols)} ({len(entry_l)} signals, {time.time()-t0:.0f}s)")

    entry = np.array(entry_l, float); d3opens = np.vstack(d3_l).astype(float)
    vd1 = np.array(vd1_l, float); vd2 = np.array(vd2_l, float); dts = np.array(dt_l, dtype=object)
    vratio = np.where(vd1 > 0, vd2 / vd1, np.nan)
    print(f"\nBase+high-contain signals: {len(entry)}")

    full_grid, summary = [], []
    for K in KGRID:
        m = (~np.isnan(vratio)) & (vratio >= K)
        e_m, d3_m, dt_m = entry[m], d3opens[m], dts[m]
        shares_m, cnt_m = day_alloc(e_m, dt_m)                    # ₹5L-pool allocation
        keep = shares_m > 0
        e_k, d3_k, sh_k = e_m[keep], d3_m[keep], shares_m[keep]
        cap_k = sh_k * e_k
        # daily-count stats (on qualifying signals that day)
        dc = pd.Series(dt_m).value_counts()
        n_days = len(dc); over5 = int((dc > 5).sum()); maxday = int(dc.max()) if len(dc) else 0
        avg_sig = round(float(dc.mean()), 2) if len(dc) else 0.0
        t = sweep(e_k, d3_k, sh_k, cap_k)
        tt = t.copy(); tt.insert(0, "K", K); full_grid.append(tt)
        if len(t) == 0:
            summary.append({"K": K, "n_trades": 0, "flag": "no trades"}); continue
        b = t.iloc[0]
        summary.append({"K": K, "n_trades": int(b["n_trades"]), "best_exit": b["exit_time"],
                        "net_win_rate_pct": b["net_win_rate_pct"], "net_avg_return_pct": b["net_avg_return_pct"],
                        "net_median_return_pct": b["net_median_return_pct"],
                        "net_total_return_capped_pct": b["net_total_return_fixedbase_pct"],
                        "prior_net_total_flat1L_pct": PRIOR_FLAT.get(K, np.nan),
                        "avg_capital_deployed_per_trade": b["avg_capital_deployed_per_trade"],
                        "avg_signals_per_signal_day": avg_sig, "signal_days_over_5": over5,
                        "max_signals_one_day": maxday,
                        "flag": "n<30" if int(b["n_trades"]) < SMALL else ""})

    grid = pd.concat(full_grid, ignore_index=True)
    summ = pd.DataFrame(summary)
    with pd.ExcelWriter(OUTDIR / "reversal_bounce_contain_vol_alloc.xlsx", engine="openpyxl") as w:
        grid.to_excel(w, sheet_name="full_grid_K_x_exit", index=False)
        summ.to_excel(w, sheet_name="best_per_K_capped", index=False)

    pd.set_option("display.width", 250)
    cols = ["K", "n_trades", "best_exit", "net_win_rate_pct", "net_avg_return_pct", "net_median_return_pct",
            "net_total_return_capped_pct", "prior_net_total_flat1L_pct", "avg_capital_deployed_per_trade",
            "avg_signals_per_signal_day", "signal_days_over_5", "max_signals_one_day", "flag"]
    print("\n=== BEST EXIT per K under ₹5L-POOL allocation (vs prior flat-₹1L total) ===")
    print(summ[cols].to_string(index=False))
    tot_over5 = summ["signal_days_over_5"].sum(); mx = summ["max_signals_one_day"].max()
    print(f"\n  Across K: total signal-days triggering >5 equal-split: {int(tot_over5)} | "
          f"max signals on any single day: {int(mx)}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
