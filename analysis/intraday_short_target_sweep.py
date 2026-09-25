# -*- coding: utf-8 -*-
"""
intraday_short_target_sweep.py
==============================
Profit-target (early-cover) sweep applied ONLY to the single best-net combo of each
intraday short strategy:
  - FIXED  : top net@0.1% combo from results/intraday_short_sweep/intraday_short_sweep.parquet
  - DYNAMIC: top net@0.1% combo from results/intraday_short_dynamic/intraday_short_dynamic_sweep.parquet

TARGET LOGIC (per trade, per X in 2..20):
  target_level = short_entry_price * (1 - X/100)
  scan candle LOWS from AFTER the short-entry candle up to (not incl.) 15:15:
    - if any LOW <= target_level -> cover at target_level (limit fill), return = +X% exactly.
    - else                       -> cover at 15:15 open, return = (entry-open_1515)/entry*100.
  The target caps a WINNING short (locks +X). It is NOT a stop-loss: a short going
  against you (price rising) has no target cover and rides to 3:15.

Cost 0.10% intraday round-trip (net@0.1%). Fixed ₹5L base, ₹1L/trade.
Perf: candle lows cached once (lows.npz, aligned to the same eligible symbol-day order as
both feature caches); every X just re-tests the target against per-trade min post-entry low.
"""
import sys, time, warnings
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
BASE_POOL, PER_TRADE, SHORT_COST = 500_000, 100_000, 0.0010
LOOKBACKS = list(range(3, 16))                    # both strategies share this axis
SESSION_HMS = list(range(555, 916, 15))           # 09:15 .. 15:15 (25 candles)
COVER_HM = 915                                    # 15:15
COVER_IDX = SESSION_HMS.index(COVER_HM)           # 24
TARGETS = list(range(2, 21))                      # 2% .. 20%

FIXED_ENTRY_HMS = list(range(570, 886, 15))       # cache grid of intraday_short_sweep
FIXED_OC_HMS = [900, 915]
DYN_ENTRY_HMS = list(range(570, 901, 15))         # cache grid of intraday_short_dynamic

FIXED_DIR = rb.RESULTS / "intraday_short_sweep"
DYN_DIR = rb.RESULTS / "intraday_short_dynamic"
OUTDIR = rb.RESULTS / "intraday_short_target"
LOWS_CACHE = OUTDIR / "lows.npz"


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


# ------------------------------------------------------------------ lows cache
def build_lows():
    if LOWS_CACHE.exists():
        print(f"  reusing cached lows -> {LOWS_CACHE.name}")
        return np.load(LOWS_CACHE)["lows"]
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    elig = diag.drop_duplicates().reset_index(drop=True)     # SAME order as both feature caches
    n = len(elig)
    print(f"  eligible symbol-days: {n:,} over {elig['symbol'].nunique():,} symbols")
    LOWS = np.full((n, len(SESSION_HMS)), np.nan)
    key = {(s, d): i for i, (s, d) in enumerate(zip(elig["symbol"].values, elig["date"].values))}
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
        pl = raw.pivot_table(index="date", columns="hm", values="low", aggfunc="min").reindex(columns=SESSION_HMS)
        for d in grp["date"].values:
            row = key.get((sym, d))
            if row is None or d not in pl.index:
                continue
            LOWS[row] = pl.loc[d].values
        if si % 150 == 0:
            el = time.time() - t0
            print(f"    …{si}/{nsym} ({el:.0f}s, ETA {el/si*(nsym-si):.0f}s)")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LOWS_CACHE, lows=LOWS)
    print(f"  cached -> {LOWS_CACHE.name}")
    return LOWS


# ------------------------------------------------------------------ metrics
def metric_row(label, ret, pnl, hit, sh, entry):
    net_ret = ret - SHORT_COST * 100
    net_pnl = pnl - SHORT_COST * sh * entry
    return {
        "target_X": label,
        "n_trades": len(ret),
        "pct_target_hit": round(float(hit.mean() * 100), 2),
        "gross_win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
        "net_win_rate_pct": round(float((net_pnl > 0).mean() * 100), 2),
        "gross_avg_return_pct": round(float(ret.mean()), 4),
        "net_avg_return_pct": round(float(net_ret.mean()), 4),
        "gross_median_return_pct": round(float(np.median(ret)), 4),
        "net_median_return_pct": round(float(np.median(net_ret)), 4),
        "gross_total_return_fixedbase_pct": round(float(pnl.sum()) / BASE_POOL * 100, 4),
        "net_total_return_fixedbase_pct": round(float(net_pnl.sum()) / BASE_POOL * 100, 4),
        "gross_total_pnl_inr": round(float(pnl.sum()), 0),
        "net_total_pnl_inr": round(float(net_pnl.sum()), 0),
    }


def run_targets(entry, cover, sh, min_low):
    """min_low = per-trade min LOW over post-entry, pre-1515 candles (nan if none)."""
    ride_ret = (entry - cover) / entry * 100.0
    ride_pnl = sh * (entry - cover)
    no_hit = np.zeros(len(entry), dtype=bool)
    rows = [metric_row("none (ride 3:15)", ride_ret, ride_pnl, no_hit, sh, entry)]
    for X in TARGETS:
        tgt = entry * (1 - X / 100.0)
        hit = (~np.isnan(min_low)) & (min_low <= tgt)
        ret = np.where(hit, float(X), ride_ret)
        pnl = np.where(hit, sh * entry * X / 100.0, ride_pnl)
        rows.append(metric_row(X, ret, pnl, hit, sh, entry))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ base-combo selection
def pick_fixed():
    df = pd.read_parquet(FIXED_DIR / "intraday_short_sweep.parquet")
    df = df[df["n_trades"] >= 30]
    b = df.sort_values("net_total_return_fixedbase_pct", ascending=False).iloc[0]
    return dict(lb=int(b.lookback_days), vm=int(b.volume_multiple), nt=int(b.net_change_threshold),
               entry=str(b.entry_time), exit=str(b.exit_time), n=int(b.n_trades),
               net=float(b.net_total_return_fixedbase_pct))


def pick_dyn():
    df = pd.read_parquet(DYN_DIR / "intraday_short_dynamic_sweep.parquet")
    df = df[df["n_trades"] >= 30]
    b = df.sort_values("net_total_return_fixedbase_pct", ascending=False).iloc[0]
    return dict(lb=int(b.lookback_days), vm=int(b.volume_multiple), nt=int(b.net_change_threshold),
               n=int(b.n_trades), net=float(b.net_total_return_fixedbase_pct))


# ------------------------------------------------------------------ trade reconstruction
def fixed_trades(LOWS):
    z = np.load(FIXED_DIR / "features.npz", allow_pickle=True)
    AVG, OE, OC, CV, NC = z["avg"], z["oe"], z["oc"], z["cv"], z["nc"]
    c = pick_fixed()
    entry_hm = int(c["entry"][:2]) * 60 + int(c["entry"][3:])       # e.g. 13:15 -> 795
    ei = FIXED_ENTRY_HMS.index(entry_hm)
    li = LOOKBACKS.index(c["lb"])
    avgL = AVG[:, li]; eo_all = OE[:, ei]
    cov_all = OC[:, FIXED_OC_HMS.index(COVER_HM)]
    q = ((~np.isnan(avgL)) & (avgL > 0) & (CV[:, ei] >= c["vm"] * avgL)
         & (NC[:, ei] <= c["nt"]) & (~np.isnan(eo_all)))
    qi = np.where(q)[0]
    eo = eo_all[qi]; cov = cov_all[qi]; sh = np.floor(PER_TRADE / eo)
    ok = (sh > 0) & (~np.isnan(cov))
    qi, eo, cov, sh = qi[ok], eo[ok], cov[ok], sh[ok]
    # post-entry, pre-1515 candles are the SAME for all fixed trades
    ent_sidx = SESSION_HMS.index(entry_hm)
    sub = LOWS[qi][:, ent_sidx + 1:COVER_IDX]                        # cols after entry, before 15:15
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_low = np.nanmin(sub, axis=1) if sub.shape[1] else np.full(len(eo), np.nan)
    return c, eo, cov, sh, min_low


def dyn_trades(LOWS):
    z = np.load(DYN_DIR / "features_dyn.npz", allow_pickle=True)
    AVG, OE, CV, NC, COVER = z["avg"], z["oe"], z["cv"], z["nc"], z["cover"]
    c = pick_dyn()
    li = LOOKBACKS.index(c["lb"]); avgL = AVG[:, li]
    okavg = (~np.isnan(avgL)) & (avgL > 0)
    volq = okavg[:, None] & (CV >= c["vm"] * avgL[:, None]) & (~np.isnan(OE))
    qual = volq & (NC <= c["nt"])
    has = qual.any(axis=1)
    fi = qual.argmax(axis=1)                                         # first qualifying candle idx
    ri = np.where(has)[0]; fic = fi[ri]
    entry_hm = np.array(DYN_ENTRY_HMS)[fic]
    eo = OE[ri, fic]; cov = COVER[ri]; sh = np.floor(PER_TRADE / eo)
    ok = (sh > 0) & (~np.isnan(cov)) & (~np.isnan(eo))
    ri, fic, entry_hm, eo, cov, sh = ri[ok], fic[ok], entry_hm[ok], eo[ok], cov[ok], sh[ok]
    # per-trade post-entry window: session cols in (ent_sidx, COVER_IDX)
    ent_sidx = np.array([SESSION_HMS.index(int(h)) for h in entry_hm])
    Lsub = LOWS[ri]
    cols = np.arange(len(SESSION_HMS))
    mask = (cols[None, :] > ent_sidx[:, None]) & (cols[None, :] < COVER_IDX)
    masked = np.where(mask, Lsub, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_low = np.nanmin(masked, axis=1)
    return c, eo, cov, sh, min_low


# ------------------------------------------------------------------ report
SHOW = ["target_X", "n_trades", "pct_target_hit", "gross_win_rate_pct",
        "gross_avg_return_pct", "net_avg_return_pct", "net_median_return_pct",
        "gross_total_return_fixedbase_pct", "net_total_return_fixedbase_pct", "net_total_pnl_inr"]


def chart(tbl, name, base_net, best_X, path):
    t = tbl[tbl.target_X != "none (ride 3:15)"].copy()
    t["X"] = t["target_X"].astype(int)
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(t["X"], t["net_total_return_fixedbase_pct"], "o-", color="#1f77b4", label="net total return")
    ax1.axhline(base_net, ls="--", color="#d62728", label=f"no-target baseline ({base_net:.2f}%)")
    if best_X is not None:
        ax1.axvline(best_X, ls=":", color="#2ca02c", alpha=.7)
    ax1.set_xlabel("profit-target X (%)"); ax1.set_ylabel("net total_return_fixedbase (%)", color="#1f77b4")
    ax1.set_xticks(TARGETS); ax1.grid(alpha=.3)
    ax2 = ax1.twinx()
    ax2.plot(t["X"], t["pct_target_hit"], "s-", color="#ff7f0e", alpha=.55, label="% target hit")
    ax2.set_ylabel("% of trades hitting target", color="#ff7f0e"); ax2.set_ylim(0, 100)
    ax1.set_title(f"{name}: profit-target sweep (net return & hit-rate vs X)")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def summarise(name, tbl):
    base = tbl[tbl.target_X == "none (ride 3:15)"].iloc[0]
    base_net = float(base.net_total_return_fixedbase_pct)
    tX = tbl[tbl.target_X != "none (ride 3:15)"].copy()
    tX["Xi"] = tX["target_X"].astype(int)
    best = tX.sort_values("net_total_return_fixedbase_pct", ascending=False).iloc[0]
    best_net = float(best.net_total_return_fixedbase_pct); best_X = int(best.Xi)
    print(f"\n{'='*140}\n{name}\n{'='*140}")
    print(tbl[SHOW].to_string(index=False))
    print(f"\n  no-target baseline net = {base_net:.4f}%")
    print(f"  BEST target X = {best_X}%  ->  net {best_net:.4f}%  ({best.pct_target_hit:.1f}% hit)")
    delta = best_net - base_net
    verdict = "BEATS" if delta > 0 else "does NOT beat"
    print(f"  target {verdict} no-target by {delta:+.4f} pts "
          f"({'add a target' if delta > 0 else 'keep riding to 3:15'})")
    return base_net, best_X


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Building candle-low cache …")
    LOWS = build_lows()

    fc, f_eo, f_cov, f_sh, f_ml = fixed_trades(LOWS)
    dc, d_eo, d_cov, d_sh, d_ml = dyn_trades(LOWS)

    print("\n### SELECTED BASE COMBOS ###")
    print(f"  FIXED  : lookback={fc['lb']}, volume={fc['vm']}x, net_change<={fc['nt']}%, "
          f"entry={fc['entry']}, exit={fc['exit']}  | sweep n={fc['n']}, net={fc['net']:.4f}%  "
          f"| reconstructed trades={len(f_eo)}")
    print(f"  DYNAMIC: lookback={dc['lb']}, volume={dc['vm']}x, net_change<={dc['nt']}%, "
          f"entry=first-qualifying candle, cover=15:15  | sweep n={dc['n']}, net={dc['net']:.4f}%  "
          f"| reconstructed trades={len(d_eo)}")

    ftab = run_targets(f_eo, f_cov, f_sh, f_ml)
    dtab = run_targets(d_eo, d_cov, d_sh, d_ml)

    f_base, f_bX = summarise("FIXED-ENTRY (lb14 / vol3 / net<=-1 / 13:15 -> 15:15)", ftab)
    d_base, d_bX = summarise("DYNAMIC-ENTRY (lb8 / vol7 / net<=-2 / first-candle -> 15:15)", dtab)

    chart(ftab, "FIXED-ENTRY short", f_base, f_bX, OUTDIR / "fixed_target_sweep.png")
    chart(dtab, "DYNAMIC-ENTRY short", d_base, d_bX, OUTDIR / "dynamic_target_sweep.png")

    with pd.ExcelWriter(OUTDIR / "intraday_short_target_sweep.xlsx", engine="openpyxl") as w:
        ftab.to_excel(w, sheet_name="fixed_target_sweep", index=False)
        dtab.to_excel(w, sheet_name="dynamic_target_sweep", index=False)
    ftab.to_csv(OUTDIR / "fixed_target_sweep.csv", index=False)
    dtab.to_csv(OUTDIR / "dynamic_target_sweep.csv", index=False)
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
