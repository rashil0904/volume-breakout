# -*- coding: utf-8 -*-
"""
intraday_short_dynamic_sweep.py
===============================
Standalone intraday short with DYNAMIC entry: scan every 15-min candle 09:30 -> 15:00;
short at the FIRST candle where all conditions hold (one short per stock-day, no re-entry);
cover fixed at 15:15. mcap ₹1,500-5,000 Cr, ₹1L/trade (100000 // entry_open), ₹5L base,
0.10% intraday cost. Sweep: lookback 3-15 (13) x volume 3-7 (5) x net_change -1..-10 (10)
= 650 combos.

Entry at candle T (checked in chronological order): net_change=(T open - prev close)/prev
close x100 <= threshold AND cumvol(09:15->candle before T) >= vol_mult x trailing L-day avg
full-day volume AND mcap in band. Features (avg per lookback, opens/cumvol/net_change at the
23 candles 09:30-15:00, cover open @15:15) extracted once per stock and cached.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "intraday_short_dynamic"
CACHE = OUTDIR / "features_dyn.npz"
BASE_POOL, PER_TRADE, SHORT_COST = 500_000, 100_000, 0.0010
LOOKBACKS = list(range(3, 16))
VOL_MULTS = list(range(3, 8))
NET_THRS = list(range(-1, -11, -1))
ENTRY_HMS = list(range(570, 901, 15))              # 09:30 .. 15:00 (23 scan candles)
COVER_HM = 915                                     # 15:15
SESSION_HMS = list(range(555, 916, 15))
SMALL = 30


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def build_features():
    if CACHE.exists():
        print(f"  reusing cached features -> {CACHE.name}")
        z = np.load(CACHE, allow_pickle=True)
        return z["avg"], z["oe"], z["cv"], z["nc"], z["cover"]
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    elig = diag.drop_duplicates().reset_index(drop=True)
    n = len(elig)
    print(f"  eligible symbol-days: {n:,} over {elig['symbol'].nunique():,} symbols")
    AVG = np.full((n, len(LOOKBACKS)), np.nan)
    OE = np.full((n, len(ENTRY_HMS)), np.nan)
    CV = np.full((n, len(ENTRY_HMS)), np.nan)
    NC = np.full((n, len(ENTRY_HMS)), np.nan)
    COVER = np.full(n, np.nan)
    key = {(s, d): i for i, (s, d) in enumerate(zip(elig["symbol"].values, elig["date"].values))}
    ecol = [SESSION_HMS.index(h) for h in ENTRY_HMS]; ccol = SESSION_HMS.index(COVER_HM)
    t0 = time.time(); nsym = elig["symbol"].nunique()
    for si, (sym, grp) in enumerate(elig.groupby("symbol"), 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[raw["hm"].isin(SESSION_HMS)]
        ds = sorted(raw["date"].unique())
        po = raw.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=SESSION_HMS)
        pv = raw.pivot_table(index="date", columns="hm", values="volume", aggfunc="sum").reindex(columns=SESSION_HMS).fillna(0)
        pc = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        full_vol = pv.sum(axis=1); last_close = pc.ffill(axis=1).iloc[:, -1]
        cumvol = pv.cumsum(axis=1); nz = full_vol[full_vol > 0]
        avg_maps = {L: nz.rolling(L, min_periods=L).mean().shift(1).reindex(ds, method="ffill").to_dict() for L in LOOKBACKS}
        prev_close = {ds[k]: last_close.get(ds[k-1], np.nan) for k in range(1, len(ds))}
        for d in grp["date"].values:
            row = key.get((sym, d))
            if row is None or d not in po.index:
                continue
            opens = po.loc[d].values; cvr = cumvol.loc[d].values; pcl = prev_close.get(d, np.nan)
            for li, L in enumerate(LOOKBACKS):
                v = avg_maps[L].get(d, np.nan)
                if v is not None and v == v:
                    AVG[row, li] = v
            for ei, col in enumerate(ecol):
                OE[row, ei] = opens[col]
                CV[row, ei] = cvr[col-1] if col >= 1 else 0.0
                if pcl == pcl and pcl:
                    NC[row, ei] = (opens[col] - pcl) / pcl * 100
            COVER[row] = opens[ccol]
        if si % 150 == 0:
            el = time.time()-t0
            print(f"    …{si}/{nsym} ({el:.0f}s, ETA {el/si*(nsym-si):.0f}s)")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, avg=AVG, oe=OE, cv=CV, nc=NC, cover=COVER)
    print(f"  cached -> {CACHE.name}")
    return AVG, OE, CV, NC, COVER


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Extracting dynamic-entry features …")
    AVG, OE, CV, NC, COVER = build_features()
    ent_hm = np.array(ENTRY_HMS)

    print(f"Sweeping {len(LOOKBACKS)*len(VOL_MULTS)*len(NET_THRS)} combos …")
    rows = []
    for li, L in enumerate(LOOKBACKS):
        avgL = AVG[:, li]; okavg = (~np.isnan(avgL)) & (avgL > 0)
        for vm in VOL_MULTS:
            volq = okavg[:, None] & (CV >= vm * avgL[:, None]) & (~np.isnan(OE))    # (n, 23)
            for nt in NET_THRS:
                qual = volq & (NC <= nt)
                has = qual.any(axis=1)
                if not has.any():
                    continue
                fi = qual.argmax(axis=1)                       # first qualifying candle
                ri = np.where(has)[0]; fic = fi[ri]
                eo = OE[ri, fic]; cov = COVER[ri]
                sh = np.floor(PER_TRADE / eo)
                ok = (sh > 0) & (~np.isnan(cov))
                ri, fic, eo, cov, sh = ri[ok], fic[ok], eo[ok], cov[ok], sh[ok]
                nv = len(eo)
                if nv == 0:
                    continue
                pnl = sh*(eo-cov); ret = (eo-cov)/eo*100
                npnl = pnl - SHORT_COST*sh*eo; nret = ret - SHORT_COST*100
                avg_et = ent_hm[fic].mean()
                rows.append({
                    "lookback_days": L, "volume_multiple": vm, "net_change_threshold": nt,
                    "n_trades": nv, "gross_win_rate_pct": round(float((pnl>0).mean()*100),2),
                    "gross_avg_return_pct": round(float(ret.mean()),4),
                    "gross_median_return_pct": round(float(np.median(ret)),4),
                    "gross_total_return_fixedbase_pct": round(float(pnl.sum())/BASE_POOL*100,4),
                    "gross_total_pnl_inr": round(float(pnl.sum()),0),
                    "net_win_rate_pct": round(float((npnl>0).mean()*100),2),
                    "net_avg_return_pct": round(float(nret.mean()),4),
                    "net_median_return_pct": round(float(np.median(nret)),4),
                    "net_total_return_fixedbase_pct": round(float(npnl.sum())/BASE_POOL*100,4),
                    "net_total_pnl_inr": round(float(npnl.sum()),0),
                    "avg_capital_deployed_per_trade": round(float((sh*eo).mean()),0),
                    "avg_entry_time": hm_lbl(avg_et), "avg_entry_hm": round(float(avg_et),1)})
    full = pd.DataFrame(rows).sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    full.to_parquet(OUTDIR / "intraday_short_dynamic_sweep.parquet", index=False)

    MIN = 30
    el = full[full["n_trades"] >= MIN]
    top20 = el.head(20)
    best = full.iloc[0]

    # best-combo entry-time distribution
    bL, bV, bN = int(best.lookback_days), int(best.volume_multiple), int(best.net_change_threshold)
    li = LOOKBACKS.index(bL); avgL = AVG[:, li]
    qual = (~np.isnan(avgL))[:,None] & (avgL[:,None] > 0) & (CV >= bV*avgL[:,None]) & (NC <= bN) & (~np.isnan(OE))
    has = qual.any(axis=1); fi = qual.argmax(axis=1)[has]
    cov_ok = ~np.isnan(COVER[np.where(has)[0]])
    et_hms = ent_hm[fi][cov_ok]
    et_dist = pd.Series(et_hms).map(hm_lbl).value_counts().reindex([hm_lbl(h) for h in ENTRY_HMS]).fillna(0).astype(int)
    et_dist = et_dist[et_dist > 0]

    # view: net total vs net_change_threshold at best lookback/volume
    v = full[(full.lookback_days==bL)&(full.volume_multiple==bV)].sort_values("net_change_threshold", ascending=False)
    view = v[["net_change_threshold","n_trades","net_win_rate_pct","net_avg_return_pct",
              "net_total_return_fixedbase_pct","avg_entry_time"]]

    with pd.ExcelWriter(OUTDIR / "intraday_short_dynamic_sweep.xlsx", engine="openpyxl") as w:
        full.to_excel(w, sheet_name="all_650", index=False)
        top20.to_excel(w, sheet_name="top20_by_net", index=False)
        view.to_excel(w, sheet_name="netchg_view_best_lb_vol", index=False)
        et_dist.rename("n_shorts").to_frame().to_excel(w, sheet_name="best_entry_time_dist")

    pd.set_option("display.width", 230)
    show = ["lookback_days","volume_multiple","net_change_threshold","n_trades","gross_win_rate_pct",
            "gross_avg_return_pct","gross_total_return_fixedbase_pct","net_total_return_fixedbase_pct","avg_entry_time"]
    print(f"\ncombos with n>=30: {len(el)} of {len(full)} ({len(el)/len(full)*100:.1f}%); "
          f"dropped {len(full)-len(el)} small-sample")
    print("\n=== TOP 20 by NET total_return (n>=30) ===")
    print(top20[show].to_string(index=False))
    print(f"\nBEST COMBO: lb={bL} vol={bV} net<={bN}% | n={int(best.n_trades)} | "
          f"net {best.net_total_return_fixedbase_pct}% (gross {best.gross_total_return_fixedbase_pct}%) | "
          f"win {best.net_win_rate_pct}% | avg_entry {best.avg_entry_time}")
    print("\n--- BEST-COMBO entry-time distribution (when do shorts fire?) ---")
    print(et_dist.to_string())
    print("\n--- NET total_return vs net_change_threshold (best lb/vol) ---")
    print(view.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
