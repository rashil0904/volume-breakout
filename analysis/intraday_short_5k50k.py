# -*- coding: utf-8 -*-
"""
intraday_short_5k50k.py
=======================
Intraday short (short a down-on-volume stock, cover at 15:15 same day, NO target) on the
LARGER-CAP universe ₹5,000-50,000 Cr. Produces BOTH the fixed-entry and dynamic-entry best
combos, and compares to the ₹1,500-5,000 Cr results.

Universe : diagnostic_table_mcap5k50k.csv rows = in-band symbol-days (₹5,000-50,000 Cr).
Sizing   : ₹5L base, ₹1L/trade (shares = 100000 // entry_open), 0.10% intraday short cost.
Grid     : lookback 3-15 (13) x volume 3-7 (5) x net_change -1..-10 (10) = 650 per strategy.
FIXED    : short at the 13:15 open (same fixed time selected in the ₹1,500-5,000 fixed work).
DYNAMIC  : scan 09:30 -> 15:00 candles; short at FIRST candle all conds hold; cover 15:15.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
DIAG = rb.RESULTS / "diagnostic_table_mcap5k50k.csv"
OUTDIR = rb.RESULTS / "intraday_short_5k50k"
CACHE = OUTDIR / "features_5k50k.npz"
BASE_POOL, PER_TRADE, SHORT_COST = 500_000, 100_000, 0.0010
LOOKBACKS = list(range(3, 16))
VOL_MULTS = list(range(3, 8))
NET_THRS = list(range(-1, -11, -1))
ENTRY_HMS = list(range(570, 901, 15))              # 09:30 .. 15:00 (23 scan candles)
FIXED_HM = 795                                      # 13:15
COVER_HM = 915                                      # 15:15
SESSION_HMS = list(range(555, 916, 15))
SMALL = 30


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def build_features():
    if CACHE.exists():
        print(f"  reusing cached features -> {CACHE.name}")
        z = np.load(CACHE, allow_pickle=True)
        return z["avg"], z["oe"], z["cv"], z["nc"], z["cover"]
    diag = pd.read_csv(DIAG, usecols=["symbol", "date"], parse_dates=["date"])
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


def _metrics(pnl, ret, sh, eo, avg_et=None):
    npnl = pnl - SHORT_COST*sh*eo; nret = ret - SHORT_COST*100
    d = {"n_trades": len(pnl),
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
         "avg_capital_deployed_per_trade": round(float((sh*eo).mean()), 0)}
    if avg_et is not None:
        d["avg_entry_time"] = hm_lbl(avg_et); d["avg_entry_hm"] = round(float(avg_et), 1)
    return d


def sweep_fixed(AVG, OE, CV, NC, COVER):
    ei = ENTRY_HMS.index(FIXED_HM)
    eo_all, cv_all, nc_all = OE[:, ei], CV[:, ei], NC[:, ei]
    rows = []
    for li, L in enumerate(LOOKBACKS):
        avgL = AVG[:, li]; okavg = (~np.isnan(avgL)) & (avgL > 0)
        for vm in VOL_MULTS:
            volq = okavg & (cv_all >= vm*avgL) & (~np.isnan(eo_all))
            for nt in NET_THRS:
                q = volq & (nc_all <= nt)
                qi = np.where(q)[0]
                if len(qi) == 0:
                    continue
                eo = eo_all[qi]; cov = COVER[qi]; sh = np.floor(PER_TRADE/eo)
                ok = (sh > 0) & (~np.isnan(cov))
                eo, cov, sh = eo[ok], cov[ok], sh[ok]
                if len(eo) == 0:
                    continue
                pnl = sh*(eo-cov); ret = (eo-cov)/eo*100
                rows.append({"lookback_days": L, "volume_multiple": vm, "net_change_threshold": nt,
                             **_metrics(pnl, ret, sh, eo)})
    return pd.DataFrame(rows).sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)


def sweep_dynamic(AVG, OE, CV, NC, COVER):
    ent_hm = np.array(ENTRY_HMS)
    rows = []
    for li, L in enumerate(LOOKBACKS):
        avgL = AVG[:, li]; okavg = (~np.isnan(avgL)) & (avgL > 0)
        for vm in VOL_MULTS:
            volq = okavg[:, None] & (CV >= vm*avgL[:, None]) & (~np.isnan(OE))
            for nt in NET_THRS:
                qual = volq & (NC <= nt)
                has = qual.any(axis=1)
                if not has.any():
                    continue
                fi = qual.argmax(axis=1)
                ri = np.where(has)[0]; fic = fi[ri]
                eo = OE[ri, fic]; cov = COVER[ri]; sh = np.floor(PER_TRADE/eo)
                ok = (sh > 0) & (~np.isnan(cov)) & (~np.isnan(eo))
                ri, fic, eo, cov, sh = ri[ok], fic[ok], eo[ok], cov[ok], sh[ok]
                if len(eo) == 0:
                    continue
                pnl = sh*(eo-cov); ret = (eo-cov)/eo*100
                rows.append({"lookback_days": L, "volume_multiple": vm, "net_change_threshold": nt,
                             **_metrics(pnl, ret, sh, eo, avg_et=ent_hm[fic].mean())})
    return pd.DataFrame(rows).sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)


def dyn_entry_dist(AVG, OE, CV, NC, COVER, bL, bV, bN):
    li = LOOKBACKS.index(bL); avgL = AVG[:, li]
    qual = ((~np.isnan(avgL))[:, None] & (avgL[:, None] > 0) & (CV >= bV*avgL[:, None])
            & (NC <= bN) & (~np.isnan(OE)))
    has = qual.any(axis=1); fi = qual.argmax(axis=1)[has]
    cov_ok = ~np.isnan(COVER[np.where(has)[0]])
    et = np.array(ENTRY_HMS)[fi][cov_ok]
    dist = pd.Series(et).map(hm_lbl).value_counts().reindex([hm_lbl(h) for h in ENTRY_HMS]).fillna(0).astype(int)
    return dist[dist > 0]


def small_sample(df):
    return (df.groupby("net_change_threshold")
              .apply(lambda g: pd.Series({"n_combos": len(g), "n_pass_ge30": int((g.n_trades >= SMALL).sum()),
                                          "pct_pass": round((g.n_trades >= SMALL).mean()*100, 1),
                                          "max_n_trades": int(g.n_trades.max())}))
              .reset_index())


def ref_best(parquet, extra_filter=None):
    df = pd.read_parquet(parquet)
    df = df[df["n_trades"] >= SMALL]
    if extra_filter is not None:
        df = extra_filter(df)
    return df.sort_values("net_total_return_fixedbase_pct", ascending=False).iloc[0]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Extracting features (₹5,000-50,000 Cr) …")
    AVG, OE, CV, NC, COVER = build_features()

    print("Sweeping FIXED (650) …")
    fx = sweep_fixed(AVG, OE, CV, NC, COVER)
    print("Sweeping DYNAMIC (650) …")
    dy = sweep_dynamic(AVG, OE, CV, NC, COVER)
    fx.to_parquet(OUTDIR / "fixed_sweep_5k50k.parquet", index=False)
    dy.to_parquet(OUTDIR / "dynamic_sweep_5k50k.parquet", index=False)

    fxe = fx[fx.n_trades >= SMALL]; dye = dy[dy.n_trades >= SMALL]
    fx_top20 = fxe.head(20); dy_top20 = dye.head(20)
    fb = fxe.iloc[0]; db = dye.iloc[0]

    ss_fx = small_sample(fx); ss_dy = small_sample(dy)
    et_dist = dyn_entry_dist(AVG, OE, CV, NC, COVER, int(db.lookback_days), int(db.volume_multiple), int(db.net_change_threshold))

    # ── reference: ₹1,500-5,000 Cr best combos ──
    fx_15 = ref_best(rb.RESULTS / "intraday_short_sweep" / "intraday_short_sweep.parquet",
                     lambda d: d[(d.entry_time == "13:15") & (d.exit_time == "15:15")])
    dy_15 = ref_best(rb.RESULTS / "intraday_short_dynamic" / "intraday_short_dynamic_sweep.parquet")

    cmp_rows = [
        {"universe": "₹1,500-5,000 Cr", "entry_style": "fixed 13:15",
         "combo": f"lb{int(fx_15.lookback_days)}/vol{int(fx_15.volume_multiple)}/net{int(fx_15.net_change_threshold)}",
         "n_trades": int(fx_15.n_trades), "net_win_rate_pct": fx_15.net_win_rate_pct,
         "net_avg_return_pct": fx_15.net_avg_return_pct, "net_median_return_pct": fx_15.net_median_return_pct,
         "net_total_return_fixedbase_pct": fx_15.net_total_return_fixedbase_pct,
         "avg_capital_deployed_per_trade": fx_15.avg_capital_deployed_per_trade},
        {"universe": "₹5,000-50,000 Cr", "entry_style": "fixed 13:15",
         "combo": f"lb{int(fb.lookback_days)}/vol{int(fb.volume_multiple)}/net{int(fb.net_change_threshold)}",
         "n_trades": int(fb.n_trades), "net_win_rate_pct": fb.net_win_rate_pct,
         "net_avg_return_pct": fb.net_avg_return_pct, "net_median_return_pct": fb.net_median_return_pct,
         "net_total_return_fixedbase_pct": fb.net_total_return_fixedbase_pct,
         "avg_capital_deployed_per_trade": fb.avg_capital_deployed_per_trade},
        {"universe": "₹1,500-5,000 Cr", "entry_style": "dynamic",
         "combo": f"lb{int(dy_15.lookback_days)}/vol{int(dy_15.volume_multiple)}/net{int(dy_15.net_change_threshold)}",
         "n_trades": int(dy_15.n_trades), "net_win_rate_pct": dy_15.net_win_rate_pct,
         "net_avg_return_pct": dy_15.net_avg_return_pct, "net_median_return_pct": dy_15.net_median_return_pct,
         "net_total_return_fixedbase_pct": dy_15.net_total_return_fixedbase_pct,
         "avg_capital_deployed_per_trade": dy_15.avg_capital_deployed_per_trade},
        {"universe": "₹5,000-50,000 Cr", "entry_style": "dynamic",
         "combo": f"lb{int(db.lookback_days)}/vol{int(db.volume_multiple)}/net{int(db.net_change_threshold)}",
         "n_trades": int(db.n_trades), "net_win_rate_pct": db.net_win_rate_pct,
         "net_avg_return_pct": db.net_avg_return_pct, "net_median_return_pct": db.net_median_return_pct,
         "net_total_return_fixedbase_pct": db.net_total_return_fixedbase_pct,
         "avg_capital_deployed_per_trade": db.avg_capital_deployed_per_trade},
    ]
    cmp = pd.DataFrame(cmp_rows)

    with pd.ExcelWriter(OUTDIR / "intraday_short_5k50k.xlsx", engine="openpyxl") as w:
        fx.to_excel(w, sheet_name="fixed_all_650", index=False)
        fx_top20.to_excel(w, sheet_name="fixed_top20", index=False)
        dy.to_excel(w, sheet_name="dynamic_all_650", index=False)
        dy_top20.to_excel(w, sheet_name="dynamic_top20", index=False)
        et_dist.rename("n_shorts").to_frame().to_excel(w, sheet_name="dyn_best_entry_dist")
        ss_fx.to_excel(w, sheet_name="fixed_smallsample", index=False)
        ss_dy.to_excel(w, sheet_name="dynamic_smallsample", index=False)
        cmp.to_excel(w, sheet_name="vs_1500_5000", index=False)

    pd.set_option("display.width", 240)
    fshow = ["lookback_days","volume_multiple","net_change_threshold","n_trades","gross_win_rate_pct",
             "gross_avg_return_pct","gross_total_return_fixedbase_pct","net_total_return_fixedbase_pct"]
    dshow = fshow + ["avg_entry_time"]

    print(f"\n{'='*120}\nFIXED-ENTRY (13:15) — ₹5,000-50,000 Cr | pass n>=30: {len(fxe)}/{len(fx)}\n{'='*120}")
    print(fx_top20[fshow].to_string(index=False))
    print(f"\nBEST FIXED: lb={int(fb.lookback_days)} vol={int(fb.volume_multiple)} net<={int(fb.net_change_threshold)}% "
          f"| n={int(fb.n_trades)} | net {fb.net_total_return_fixedbase_pct}% (gross {fb.gross_total_return_fixedbase_pct}%) "
          f"| win {fb.net_win_rate_pct}% | avg_ret {fb.net_avg_return_pct}% | med {fb.net_median_return_pct}% "
          f"| avg_cap ₹{fb.avg_capital_deployed_per_trade:,.0f}")

    print(f"\n{'='*120}\nDYNAMIC-ENTRY — ₹5,000-50,000 Cr | pass n>=30: {len(dye)}/{len(dy)}\n{'='*120}")
    print(dy_top20[dshow].to_string(index=False))
    print(f"\nBEST DYNAMIC: lb={int(db.lookback_days)} vol={int(db.volume_multiple)} net<={int(db.net_change_threshold)}% "
          f"| n={int(db.n_trades)} | net {db.net_total_return_fixedbase_pct}% (gross {db.gross_total_return_fixedbase_pct}%) "
          f"| win {db.net_win_rate_pct}% | avg_ret {db.net_avg_return_pct}% | med {db.net_median_return_pct}% "
          f"| avg_entry {db.avg_entry_time} | avg_cap ₹{db.avg_capital_deployed_per_trade:,.0f}")
    print("\n--- BEST DYNAMIC entry-time distribution ---")
    print(et_dist.to_string())

    print(f"\n{'='*120}\nHEAD-TO-HEAD (₹5,000-50,000 Cr): fixed vs dynamic\n{'='*120}")
    print(cmp[cmp.universe == "₹5,000-50,000 Cr"].to_string(index=False))

    print(f"\n{'='*120}\nUNIVERSE COMPARISON: ₹1,500-5,000 vs ₹5,000-50,000 (best combo each style)\n{'='*120}")
    print(cmp.to_string(index=False))

    print(f"\n{'='*120}\nSMALL-SAMPLE by net_change_threshold\n{'='*120}")
    print("FIXED:\n" + ss_fx.to_string(index=False))
    print("\nDYNAMIC:\n" + ss_dy.to_string(index=False))

    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
