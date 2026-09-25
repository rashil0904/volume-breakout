# -*- coding: utf-8 -*-
"""
intraday_short_sweep.py
=======================
Standalone INTRADAY SHORT strategy — downside volume-breakout analogue of the long.
Short a down-on-volume stock at entry_time, cover same-day at 3:00 or 3:15 (strictly
intraday). Full 28,600-combo sweep: lookback 3-15 (13) x volume_multiple 3-7 (5) x
net_change_threshold -1..-10 (10) x entry_time 09:30-14:45 (22) x exit 15:00/15:15 (2).

Universe: mcap ₹1,500-5,000 Cr (diagnostic_table.csv rows = mcap-eligible symbol-days).
Entry: cumvol(09:15 -> candle BEFORE entry_time) >= vol_mult x trailing L-day avg full-day
volume, AND net_change = (entry open - prev close)/prev close x100 <= threshold. Short at
entry open; cover at exit open. shares = 100000 // entry_open; cost 0.10% intraday.

Perf: per-(stock,lookback) trailing-avg computed once; each stock-day's opens/cumvol/
net_change cached once (npz); sweep batched by lookback (13 parquet checkpoints), resumable.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "intraday_short_sweep"
CACHE = OUTDIR / "features.npz"
PARTS = OUTDIR / "parts"
BASE_POOL, PER_TRADE, SHORT_COST = 500_000, 100_000, 0.0010
LOOKBACKS = list(range(3, 16))                     # 13
VOL_MULTS = list(range(3, 8))                       # 5
NET_THRS = list(range(-1, -11, -1))                 # -1..-10 (10)
ENTRY_HMS = list(range(570, 886, 15))               # 09:30..14:45 (22)
COVER_HMS = [900, 915]                              # 15:00, 15:15
SESSION_HMS = list(range(555, 916, 15))             # 09:15..15:15
ELABEL = [f"{h//60:02d}:{h%60:02d}" for h in ENTRY_HMS]
CLABEL = {900: "15:00", 915: "15:15"}


def build_features():
    if CACHE.exists():
        print(f"  reusing cached features -> {CACHE.name}")
        z = np.load(CACHE, allow_pickle=True)
        return z["avg"], z["oe"], z["oc"], z["cv"], z["nc"]
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    elig = diag.drop_duplicates()
    print(f"  eligible symbol-days: {len(elig):,} over {elig['symbol'].nunique():,} symbols")

    n = len(elig)
    AVG = np.full((n, len(LOOKBACKS)), np.nan)
    OE = np.full((n, len(ENTRY_HMS)), np.nan)          # opens at entry times
    OC = np.full((n, len(COVER_HMS)), np.nan)          # opens at cover times
    CV = np.full((n, len(ENTRY_HMS)), np.nan)          # cumvol before entry time
    NC = np.full((n, len(ENTRY_HMS)), np.nan)          # net_change at entry time
    key_to_row = {(s, d): i for i, (s, d) in enumerate(zip(elig["symbol"].values, elig["date"].values))}

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
        dates_sorted = sorted(raw["date"].unique())
        po = raw.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=SESSION_HMS)
        pv = raw.pivot_table(index="date", columns="hm", values="volume", aggfunc="sum").reindex(columns=SESSION_HMS).fillna(0)
        pc = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        full_vol = pv.sum(axis=1)
        last_close = pc.ffill(axis=1).iloc[:, -1]
        cumvol = pv.cumsum(axis=1)                       # inclusive cumulative volume by candle
        nz = full_vol[full_vol > 0]
        avg_maps = {L: nz.rolling(L, min_periods=L).mean().shift(1).reindex(dates_sorted, method="ffill").to_dict()
                    for L in LOOKBACKS}
        prev_close = {dates_sorted[k]: last_close.get(dates_sorted[k-1], np.nan) for k in range(1, len(dates_sorted))}
        oe_hmcol = {hm: po.columns.get_loc(hm) for hm in ENTRY_HMS}
        for d in grp["date"].values:
            row = key_to_row.get((sym, d))
            if row is None or d not in po.index:
                continue
            opens = po.loc[d].values; cvrow = cumvol.loc[d].values
            pcl = prev_close.get(d, np.nan)
            for li, L in enumerate(LOOKBACKS):
                v = avg_maps[L].get(d, np.nan)
                if v is not None and v == v:
                    AVG[row, li] = v
            for ei, hm in enumerate(ENTRY_HMS):
                col = SESSION_HMS.index(hm)
                OE[row, ei] = opens[col]
                CV[row, ei] = cvrow[col - 1] if col >= 1 else 0.0     # cumvol BEFORE entry candle
                if pcl == pcl and pcl:
                    NC[row, ei] = (opens[col] - pcl) / pcl * 100
            for ci, hm in enumerate(COVER_HMS):
                OC[row, ci] = opens[SESSION_HMS.index(hm)]
        if si % 100 == 0:
            el = time.time() - t0
            print(f"    …{si}/{nsym} symbols ({el:.0f}s, ETA {el/si*(nsym-si):.0f}s)")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, avg=AVG, oe=OE, oc=OC, cv=CV, nc=NC)
    print(f"  features cached -> {CACHE.name}")
    return AVG, OE, OC, CV, NC


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True); PARTS.mkdir(parents=True, exist_ok=True)
    print("Extracting intraday features (cumvol / net_change / opens) …")
    AVG, OE, OC, CV, NC = build_features()

    print(f"Sweeping {len(LOOKBACKS)*len(VOL_MULTS)*len(NET_THRS)*len(ENTRY_HMS)*len(COVER_HMS):,} combos "
          f"in {len(LOOKBACKS)} lookback batches …")
    t0 = time.time()
    for li, L in enumerate(LOOKBACKS, 1):
        part = PARTS / f"lb_{L:02d}.parquet"
        if part.exists():
            print(f"  [{li}/{len(LOOKBACKS)}] lookback={L}: checkpoint exists, skip"); continue
        avgL = AVG[:, li-1]; okavg = (~np.isnan(avgL)) & (avgL > 0)
        recs = []
        for vm in VOL_MULTS:
            volpass = okavg[:, None] & (CV >= vm * avgL[:, None]) # (n, 22) per entry time
            for nt in NET_THRS:
                netpass = NC <= nt                                # (n, 22)
                for ei, ehm in enumerate(ENTRY_HMS):
                    q = volpass[:, ei] & netpass[:, ei] & ~np.isnan(OE[:, ei])
                    if not q.any():
                        continue
                    eo = OE[q, ei]
                    sh = np.floor(PER_TRADE / eo); oks = sh > 0
                    eo, sh = eo[oks], sh[oks]
                    if len(eo) == 0:
                        continue
                    for ci, chm in enumerate(COVER_HMS):
                        co = OC[q, ci][oks]
                        v = ~np.isnan(co)
                        nv = int(v.sum())
                        if nv == 0:
                            continue
                        e2, s2, c2 = eo[v], sh[v], co[v]
                        pnl = s2 * (e2 - c2); ret = (e2 - c2) / e2 * 100
                        cost = SHORT_COST * s2 * e2
                        npnl = pnl - cost; nret = ret - SHORT_COST * 100
                        recs.append((L, vm, nt, ELABEL[ei], CLABEL[chm], nv,
                                     round(float((pnl > 0).mean()*100), 2),
                                     round(float(ret.mean()), 4), round(float(np.median(ret)), 4),
                                     round(float(pnl.sum())/BASE_POOL*100, 4), round(float(pnl.sum()), 0),
                                     round(float((npnl > 0).mean()*100), 2),
                                     round(float(nret.mean()), 4), round(float(np.median(nret)), 4),
                                     round(float(npnl.sum())/BASE_POOL*100, 4), round(float(npnl.sum()), 0),
                                     round(float((s2*e2).mean()), 0)))
        cols = ["lookback_days", "volume_multiple", "net_change_threshold", "entry_time", "exit_time",
                "n_trades", "gross_win_rate_pct", "gross_avg_return_pct", "gross_median_return_pct",
                "gross_total_return_fixedbase_pct", "gross_total_pnl_inr",
                "net_win_rate_pct", "net_avg_return_pct", "net_median_return_pct",
                "net_total_return_fixedbase_pct", "net_total_pnl_inr", "avg_capital_deployed_per_trade"]
        pd.DataFrame(recs, columns=cols).to_parquet(part, index=False)
        el = time.time()-t0
        print(f"  [{li}/{len(LOOKBACKS)}] lookback={L}: {len(recs):,} rows ({el:.0f}s, ETA {el/li*(len(LOOKBACKS)-li):.0f}s)")

    full = pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("lb_*.parquet"))], ignore_index=True)
    full.to_parquet(OUTDIR / "intraday_short_sweep.parquet", index=False)
    print(f"\nFull sweep: {len(full):,} rows -> intraday_short_sweep.parquet")

    # ── condensed outputs ──
    MIN = 30
    elig = full[full["n_trades"] >= MIN]
    print(f"  combos with n_trades >= {MIN}: {len(elig):,} of {len(full):,} "
          f"({len(elig)/len(full)*100:.1f}%); dropped {len(full)-len(elig):,} small-sample")
    top20 = elig.sort_values("net_total_return_fixedbase_pct", ascending=False).head(20)
    best = top20.iloc[0] if len(top20) else None

    # small-sample transparency by net_change_threshold
    ss = full.groupby("net_change_threshold").apply(
        lambda g: pd.Series({"n_combos": len(g), "n_pass_ge30": int((g["n_trades"] >= MIN).sum()),
                             "pct_pass": round((g["n_trades"] >= MIN).mean()*100, 1),
                             "max_n_trades": int(g["n_trades"].max())})).reset_index()

    # entry-time diagnostic at best (lookback, volume, net_change, exit), net return vs entry_time
    et_diag = None
    if best is not None:
        sub = full[(full.lookback_days == best.lookback_days) & (full.volume_multiple == best.volume_multiple)
                   & (full.net_change_threshold == best.net_change_threshold) & (full.exit_time == best.exit_time)]
        et_diag = sub.sort_values("entry_time")[["entry_time", "n_trades", "net_total_return_fixedbase_pct",
                                                 "net_avg_return_pct", "net_win_rate_pct"]]
    # entry-time x net_change interaction (best net return per (net_thr, entry_time), best lb/vm/exit fixed to overall best)
    inter = None
    if best is not None:
        s2 = full[(full.lookback_days == best.lookback_days) & (full.volume_multiple == best.volume_multiple)
                  & (full.exit_time == best.exit_time)]
        inter = s2.pivot_table(index="net_change_threshold", columns="entry_time",
                               values="net_total_return_fixedbase_pct")

    with pd.ExcelWriter(OUTDIR / "intraday_short_sweep_summary.xlsx", engine="openpyxl") as w:
        top20.to_excel(w, sheet_name="top20_by_net", index=False)
        ss.to_excel(w, sheet_name="small_sample_by_netchg", index=False)
        if et_diag is not None:
            et_diag.to_excel(w, sheet_name="entrytime_diag_best", index=False)
        if inter is not None:
            inter.to_excel(w, sheet_name="entrytime_x_netchg_net")

    pd.set_option("display.width", 240)
    print("\n=== TOP 20 by NET total_return_fixedbase_pct (n>=30) ===")
    show = ["lookback_days","volume_multiple","net_change_threshold","entry_time","exit_time","n_trades",
            "gross_win_rate_pct","gross_avg_return_pct","gross_total_return_fixedbase_pct",
            "net_total_return_fixedbase_pct"]
    print(top20[show].to_string(index=False))
    if best is not None:
        print(f"\nBEST COMBO: lb={int(best.lookback_days)} vol={int(best.volume_multiple)} "
              f"netchg<={int(best.net_change_threshold)}% entry={best.entry_time} cover={best.exit_time} "
              f"| n={int(best.n_trades)} | net {best.net_total_return_fixedbase_pct}% (gross {best.gross_total_return_fixedbase_pct}%)")
    print("\n=== SMALL-SAMPLE by net_change_threshold ===")
    print(ss.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
