# -*- coding: utf-8 -*-
"""
reversal_bounce_vol_alloc.py
============================
Base 2-day reversal bounce (NO high-containment) + swept D2/D1 volume filter (K 2-7) under the
main strategy's ₹5L-POOL daily allocation (₹1L each if <=5 signals that day, else ₹5L/n split).
Entry D2 close, next-day D3 15-min full-exit sweep. Universe mcap ₹1,500-5,000 Cr.

Allocation uses each K's OWN daily qualifying count; capped totals compared to prior flat-₹1L.
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
# prior FLAT-₹1L best-exit net totals (base-only + volume-only), to flag overstatement:
PRIOR_FLAT = {"none": 102.7232, 2: 326.7775, 3: 270.9187, 4: 273.6087, 5: 234.6419, 6: 214.6801, 7: 201.7043}


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def day_alloc(entry, dates):
    d = pd.Series(dates)
    cnt = d.map(d.value_counts()).values
    target = np.where(cnt <= 5, float(PER_TRADE), BASE_POOL / cnt)
    return np.floor(target / entry), cnt


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


def daystats(dates):
    dc = pd.Series(dates).value_counts()
    return round(float(dc.mean()), 2), int((dc > 5).sum()), int(dc.max()), len(dc)


def row_for(tag, t, dates, prior):
    b = t.iloc[0]
    avg_sig, over5, maxday, ndays = daystats(dates)
    return {"config": tag, "n_trades": int(b["n_trades"]), "best_exit": b["exit_time"],
            "net_win_rate_pct": b["net_win_rate_pct"], "net_avg_return_pct": b["net_avg_return_pct"],
            "net_median_return_pct": b["net_median_return_pct"],
            "net_total_return_capped_pct": b["net_total_return_fixedbase_pct"],
            "prior_net_total_flat1L_pct": prior,
            "overstatement_pts": round(prior - b["net_total_return_fixedbase_pct"], 2) if prior == prior else np.nan,
            "avg_capital_deployed_per_trade": b["avg_capital_deployed_per_trade"],
            "avg_signals_per_day": avg_sig, "signal_days_over_5": over5, "max_signals_one_day": maxday,
            "flag": "n<30" if int(b["n_trades"]) < SMALL else ""}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Scanning {len(symbols)} symbols (base bounce, NO high condition) …")

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

    def run_set(tag, mask, prior):
        e_m, d3_m, dt_m = entry[mask], d3opens[mask], dts[mask]
        sh, _ = day_alloc(e_m, dt_m)
        keep = sh > 0
        t = sweep(e_m[keep], d3_m[keep], sh[keep], (sh[keep] * e_m[keep]))
        return t, row_for(tag, t, dt_m, prior)

    # base-only (no volume) under allocation
    base_t, base_row = run_set("(base) no vol", np.ones(len(entry), bool), PRIOR_FLAT["none"])
    bt = base_t.copy(); bt.insert(0, "K", "none"); full_grid.append(bt); summary.append(base_row)
    for K in KGRID:
        m = (~np.isnan(vratio)) & (vratio >= K)
        t, r = run_set(f"vol K={K}", m, PRIOR_FLAT.get(K))
        tt = t.copy(); tt.insert(0, "K", K); full_grid.append(tt); summary.append(r)

    grid = pd.concat(full_grid, ignore_index=True)
    summ = pd.DataFrame(summary)
    with pd.ExcelWriter(OUTDIR / "reversal_bounce_vol_alloc.xlsx", engine="openpyxl") as w:
        grid.to_excel(w, sheet_name="full_grid_K_x_exit", index=False)
        summ.to_excel(w, sheet_name="best_per_K_capped", index=False)

    pd.set_option("display.width", 260)
    cols = ["config", "n_trades", "best_exit", "net_win_rate_pct", "net_avg_return_pct",
            "net_total_return_capped_pct", "prior_net_total_flat1L_pct", "overstatement_pts",
            "avg_capital_deployed_per_trade", "avg_signals_per_day", "signal_days_over_5", "max_signals_one_day", "flag"]
    print("\n=== BASE + VOLUME under ₹5L-POOL allocation (capped) vs prior flat-₹1L ===")
    print(summ[cols].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
