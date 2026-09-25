# -*- coding: utf-8 -*-
"""
intraday_short_1315_cover_sweep.py
==================================
Fixed-entry intraday short + EOD cover sweep. Entry (from the full sweep's best 13:15
config): lookback=14, volume_multiple=3, net_change<=-1%, short opened at the 13:15 open.
mcap ₹1,500-5,000 Cr, ₹1L/trade (shares = 100000 // entry_open), ₹5L base. Cover grid:
14:30 / 14:45 / 15:00 / 15:15 (all same day, after 13:15). Cost 0.10% intraday.

Scenario 1: full cover at each of the 4 times (4 rows). Scenario 2: split cover on the
SHORT's profitability at t1 (short_ret_t1>0 -> cover early @t1, else hold to t2), 6 pairs.
Reuses cached features.npz from intraday_short_sweep (no recompute).

Flags: (a) split conditioned on short profitability at t1 (price fell below 13:15 entry ->
cover early). (b) 0.10% intraday short cost. (c) 4 cover times all after the 13:15 entry.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "intraday_short_sweep"
CACHE = OUTDIR / "features.npz"
BASE_POOL, PER_TRADE, SHORT_COST = 500_000, 100_000, 0.0010
LB, VM, NET_THR = 14, 3, -1
ENTRY_HMS = list(range(570, 886, 15)); COVER_HMS_CACHE = [900, 915]
LOOKBACKS = list(range(3, 16))
COVER_GRID = [(870, "14:30"), (885, "14:45"), (900, "15:00"), (915, "15:15")]
SMALL = 30


def metrics(ctype, t1, t2, entry_open, shares, cover_price):
    v = ~np.isnan(cover_price)
    nv = int(v.sum())
    if nv == 0:
        return None
    e, s, c = entry_open[v], shares[v], cover_price[v]
    pnl = s * (e - c); ret = (e - c) / e * 100
    cost = SHORT_COST * s * e
    npnl = pnl - cost; nret = ret - SHORT_COST * 100
    return {"cover_type": ctype, "cover_time_1": t1, "cover_time_2": t2 or "", "n_trades": nv,
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
            "avg_capital_deployed_per_trade": round(float((s*e).mean()), 0)}


def main():
    if not CACHE.exists():
        raise SystemExit(f"Missing {CACHE} — run intraday_short_sweep.py first.")
    z = np.load(CACHE, allow_pickle=True)
    AVG, OE, OC, CV, NC = z["avg"], z["oe"], z["oc"], z["cv"], z["nc"]

    ei = ENTRY_HMS.index(795)                             # 13:15
    li = LOOKBACKS.index(LB)
    avgL = AVG[:, li]
    entry_open = OE[:, ei]
    q = (~np.isnan(avgL)) & (avgL > 0) & (CV[:, ei] >= VM * avgL) & (NC[:, ei] <= NET_THR) & (~np.isnan(entry_open))
    qi = np.where(q)[0]
    eo = entry_open[qi]; sh = np.floor(PER_TRADE / eo); oks = sh > 0
    qi = qi[oks]; eo = eo[oks]; sh = sh[oks]
    n = len(eo)
    print(f"Fixed entry (lb={LB}, vol={VM}, net<={NET_THR}%, 13:15): {n:,} shorts")

    # cover opens (n, 4)
    def cover_open(hm):
        if hm in COVER_HMS_CACHE:
            return OC[qi, COVER_HMS_CACHE.index(hm)]
        return OE[qi, ENTRY_HMS.index(hm)]
    cov = {lab: cover_open(hm) for hm, lab in COVER_GRID}

    rows = []
    # Scenario 1: full cover
    for hm, lab in COVER_GRID:
        m = metrics("full", lab, None, eo, sh, cov[lab])
        if m:
            rows.append(m)
    # Scenario 2: split cover (t1<t2)
    labels = [lab for _, lab in COVER_GRID]
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            l1, l2 = labels[i], labels[j]
            o1, o2 = cov[l1], cov[l2]
            sret_t1 = (eo - o1) / eo * 100
            cover_px = np.where(sret_t1 > 0, o1, o2)
            m = metrics("split", l1, l2, eo, sh, cover_px)
            if m:
                rows.append(m)
    tbl = pd.DataFrame(rows).sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    tbl["small_sample_flag"] = np.where(tbl["n_trades"] < SMALL, "n_trades<30", "")
    tbl.to_csv(OUTDIR / "short_1315_cover_sweep.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "short_1315_cover_sweep.xlsx", engine="openpyxl") as w:
        tbl.to_excel(w, sheet_name="cover_sweep", index=False)

    # avg short return by full-cover time (fade vs bounce toward close)
    fade = []
    for hm, lab in COVER_GRID:
        r = (eo - cov[lab]) / eo * 100
        fade.append({"cover_time": lab, "avg_short_return_pct": round(np.nanmean(r), 4),
                     "median_short_return_pct": round(np.nanmedian(r), 4),
                     "win_rate_pct": round((sh*(eo-cov[lab]) > 0).mean()*100, 2)})
    fade_tbl = pd.DataFrame(fade)

    pd.set_option("display.width", 220)
    show = ["cover_type","cover_time_1","cover_time_2","n_trades","gross_win_rate_pct",
            "gross_avg_return_pct","gross_median_return_pct","gross_total_return_fixedbase_pct",
            "net_win_rate_pct","net_avg_return_pct","net_total_return_fixedbase_pct","net_total_pnl_inr","small_sample_flag"]
    print("\n" + "="*130 + "\n10 EXIT COMBINATIONS (sorted by net total_return_fixedbase_pct)\n" + "="*130)
    print(tbl[show].to_string(index=False))
    bf = tbl[tbl.cover_type == "full"].iloc[0]; bs = tbl[tbl.cover_type == "split"].iloc[0]
    print(f"\nBEST FULL cover : {bf['cover_time_1']} -> net {bf['net_total_return_fixedbase_pct']}% "
          f"(gross {bf['gross_total_return_fixedbase_pct']}%)")
    print(f"BEST SPLIT cover: {bs['cover_time_1']}/{bs['cover_time_2']} -> net {bs['net_total_return_fixedbase_pct']}% "
          f"(gross {bs['gross_total_return_fixedbase_pct']}%)")
    print("\n--- avg short return by cover time (fade toward close?) ---")
    print(fade_tbl.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
