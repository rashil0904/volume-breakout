# -*- coding: utf-8 -*-
"""
full_param_sweep.py
===================
Full combinatorial sweep for the 3:15pm-entry BTST strategy.

  lookback_days   7..40 step 1   (34)  -- volume trailing-avg window AND RSI period
  volume_multiple 3..10 step 1   (8)
  rsi_threshold   50..80 step 5  (7)
                                 -> 1,904 entry combos
  x forward_day 1..7  x  23 intraday exit times   -> 306,544 result rows

Entry (per combo): mcap band + cum-vol(09:15..15:00) >= M x trailing L-day avg
full-day volume + return-vs-prev-close >= 5% + Wilder RSI(L) on entry-day close >= R.
Entry price = 15:15 candle open.

Per (forward_day, intraday_time) the stock return is measured against CNXMIDCAP100
at the same moment, giving relative strength = stock_return - index_return.

Perf: per-symbol candle scan runs ONCE and is cached to an .npz; RSI/volume series
are computed once per lookback_days and reused across all 56 (M, R) pairs sharing it.
Sweep is batched by lookback_days (34 batches) with per-batch parquet checkpoints,
so an interrupted run resumes where it stopped.
"""

import sys, bisect, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
from lookback_volume_forward_sweep import wilder_rsi

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "full_param_sweep"
PARTS = OUTDIR / "parts"
CACHE = OUTDIR / "scan_cache.npz"

LOOKBACKS = list(range(7, 41))                       # 34
VOL_MULTS = list(range(3, 11))                       # 8
RSI_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80]        # 7
FWD_DAYS = [1, 2, 3, 4, 5, 6, 7]
TIMES_HM = list(range(570, 901, 15))                 # 09:30..15:00 (23)
TLABEL = [f"{h//60:02d}:{h%60:02d}" for h in TIMES_HM]
NT = len(TIMES_HM)

HM_915, HM_1515 = 555, 915
RET_MIN = 5.0
MIN_TRADES = 30                                      # eligibility cutoff for ranking
BASE_POOL, MAX_PER_STOCK = 500_000, 100_000
INDEX_CSV = Path(__file__).resolve().parent.parent / "data" / "cnxmidcap100_15min_ohlc.csv"


def load_index():
    """CNXMIDCAP100: entry-moment (15:15 open) and the 23-time forward grid, by date."""
    d = pd.read_csv(INDEX_CSV)
    ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(IST)
    d["date"] = ts.dt.date
    d["hm"] = ts.dt.hour * 60 + ts.dt.minute
    entry = d[d["hm"] == HM_1515].groupby("date")["open"].last()
    grid = (d[d["hm"].isin(TIMES_HM)]
            .pivot_table(index="date", columns="hm", values="open", aggfunc="last")
            .reindex(columns=TIMES_HM))
    return entry.to_dict(), grid


def build_scan(cand):
    """Per-candidate trailing volume averages, Wilder RSI (one per lookback), forward
    stock prices and matched index prices. Cached to .npz — the expensive step."""
    if CACHE.exists():
        print(f"  reusing cached scan -> {CACHE.name}")
        z = np.load(CACHE)
        return z["avgs"], z["rsis"], z["fwd"], z["idx_entry"], z["idx_fwd"]

    n = len(cand)
    avgs = np.full((n, len(LOOKBACKS)), np.nan)
    rsis = np.full((n, len(LOOKBACKS)), np.nan)
    fwd = np.full((n, len(FWD_DAYS) * NT), np.nan)
    idx_entry = np.full(n, np.nan)
    idx_fwd = np.full((n, len(FWD_DAYS) * NT), np.nan)

    ie, igrid = load_index()
    igrid_d = {d: v for d, v in zip(igrid.index, igrid.values)}

    t0 = time.time()
    nsym = cand["symbol"].nunique()
    for si, (sym, grp) in enumerate(cand.groupby("symbol"), 1):
        pqf = rb.MASTER_DIR / f"{sym}.parquet"
        if not pqf.exists():
            continue
        raw = pd.read_parquet(pqf)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        sess = raw[(raw["hm"] >= HM_915) & (raw["hm"] <= HM_1515)]
        dates_sorted = sorted(sess["date"].unique())

        fd = sess.groupby("date")["volume"].sum().reindex(dates_sorted).fillna(0)
        nz = fd[fd > 0]
        dclose = sess.groupby("date")["close"].last().reindex(dates_sorted)

        avg_maps, rsi_maps = {}, {}
        for L in LOOKBACKS:                       # computed ONCE per lookback per symbol
            avg_maps[L] = (nz.rolling(L, min_periods=L).mean().shift(1)
                             .reindex(dates_sorted, method="ffill")).to_dict()
            rsi_maps[L] = wilder_rsi(dclose, L).to_dict()

        po = (sess[sess["hm"].isin(TIMES_HM)]
              .pivot_table(index="date", columns="hm", values="open", aggfunc="last")
              .reindex(columns=TIMES_HM))
        po_d = {d: v for d, v in zip(po.index, po.values)}

        for ridx, d in zip(grp.index, grp["date"]):
            dd = pd.Timestamp(d).date()
            for li, L in enumerate(LOOKBACKS):
                v = avg_maps[L].get(dd, np.nan)
                if v is not None and v == v:
                    avgs[ridx, li] = v
                r = rsi_maps[L].get(dd, np.nan)
                if r is not None and r == r:
                    rsis[ridx, li] = r
            idx_entry[ridx] = ie.get(dd, np.nan)
            j = bisect.bisect_right(dates_sorted, dd)
            for k in range(len(FWD_DAYS)):
                if j + k >= len(dates_sorted):
                    break
                fdate = dates_sorted[j + k]
                sl = slice(k * NT, (k + 1) * NT)
                if fdate in po_d:
                    fwd[ridx, sl] = po_d[fdate]
                if fdate in igrid_d:
                    idx_fwd[ridx, sl] = igrid_d[fdate]
        if si % 100 == 0:
            el = time.time() - t0
            print(f"    …{si}/{nsym} symbols  ({el:.0f}s elapsed, "
                  f"ETA {el / si * (nsym - si):.0f}s)")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, avgs=avgs, rsis=rsis, fwd=fwd,
                        idx_entry=idx_entry, idx_fwd=idx_fwd)
    print(f"  scan cached -> {CACHE.name}")
    return avgs, rsis, fwd, idx_entry, idx_fwd


def sweep_lookback(L, qi_base, entry, cdates, fwd, idx_entry, idx_fwd, rsiL):
    """All 56 (volume_multiple, rsi_threshold) combos for one lookback."""
    out = []
    for M in VOL_MULTS:
        qm = qi_base[M]
        if qm is None or not qm.any():
            continue
        for R in RSI_THRESHOLDS:
            q = qm & (~np.isnan(rsiL)) & (rsiL >= R)
            if not q.any():
                continue
            qi = np.where(q)[0]
            e_q, d_q = entry[qi], cdates[qi]
            f_q, ie_q, if_q = fwd[qi], idx_entry[qi], idx_fwd[qi]

            cnt = pd.Series(d_q).map(pd.Series(d_q).value_counts()).values
            tgt = np.where(cnt <= 5, MAX_PER_STOCK, BASE_POOL / cnt)
            sh = np.floor(tgt / e_q)
            ok = sh > 0
            e_q, f_q, sh, ie_q, if_q = e_q[ok], f_q[ok], sh[ok], ie_q[ok], if_q[ok]
            if len(e_q) == 0:
                continue

            ret_m = (f_q - e_q[:, None]) / e_q[:, None] * 100
            pnl_m = sh[:, None] * (f_q - e_q[:, None])
            idx_m = (if_q - ie_q[:, None]) / ie_q[:, None] * 100
            rs_m = ret_m - idx_m
            valid = ~np.isnan(f_q)

            for k, fd_off in enumerate(FWD_DAYS):
                for t in range(NT):
                    c = k * NT + t
                    v = valid[:, c]
                    nv = int(v.sum())
                    if nv == 0:
                        continue
                    r, p, rs = ret_m[v, c], pnl_m[v, c], rs_m[v, c]
                    out.append((L, M, R, fd_off, TLABEL[t], nv,
                                round(float((p > 0).mean() * 100), 2),
                                round(float(r.mean()), 4),
                                round(float(np.median(r)), 4),
                                round(float(np.nanmean(rs)), 4),
                                round(float(p.sum()) / BASE_POOL * 100, 4)))
    return out


COLS = ["lookback_days", "volume_multiple", "rsi_threshold", "forward_day",
        "intraday_time", "n_trades", "win_rate_pct", "avg_return_pct",
        "median_return_pct", "avg_relative_strength_pct", "total_return_fixedbase_pct"]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)

    print("Loading diagnostic table …")
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "entry_price_315pm",
                                "return_pct_vs_prev_close", "cum_volume_to_3pm_today"],
                       parse_dates=["date"])
    cand = diag[(diag["return_pct_vs_prev_close"] >= RET_MIN)
                & diag["entry_price_315pm"].notna()
                & diag["cum_volume_to_3pm_today"].notna()].reset_index(drop=True)
    print(f"  candidates (mcap + >=5% move): {len(cand):,} over {cand['symbol'].nunique():,} symbols")

    entry = cand["entry_price_315pm"].values.astype(float)
    cumvol = cand["cum_volume_to_3pm_today"].values.astype(float)
    cdates = np.array([pd.Timestamp(d).date() for d in cand["date"].values], dtype=object)

    print("Scanning candles (volume history, RSI, forward stock + index prices) …")
    avgs, rsis, fwd, idx_entry, idx_fwd = build_scan(cand)

    total_combos = len(LOOKBACKS) * len(VOL_MULTS) * len(RSI_THRESHOLDS)
    print(f"\nSweeping {total_combos:,} entry combos in {len(LOOKBACKS)} lookback batches …")
    t0 = time.time()
    for li, L in enumerate(LOOKBACKS, 1):
        part = PARTS / f"lb_{L:02d}.parquet"
        if part.exists():
            print(f"  [{li}/{len(LOOKBACKS)}] lookback={L}: checkpoint exists, skipping")
            continue
        avgL, rsiL = avgs[:, li - 1], rsis[:, li - 1]
        ok_avg = (~np.isnan(avgL)) & (avgL > 0)
        qi_base = {M: ok_avg & (cumvol >= M * avgL) for M in VOL_MULTS}
        recs = sweep_lookback(L, qi_base, entry, cdates, fwd, idx_entry, idx_fwd, rsiL)
        pd.DataFrame(recs, columns=COLS).to_parquet(part, index=False)
        el = time.time() - t0
        print(f"  [{li}/{len(LOOKBACKS)}] lookback={L}: {len(recs):,} rows "
              f"({el:.0f}s elapsed, ETA {el / li * (len(LOOKBACKS) - li):.0f}s)")

    print("\nConcatenating checkpoints …")
    full = pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("lb_*.parquet"))],
                     ignore_index=True)
    full.to_parquet(OUTDIR / "full_param_sweep.parquet", index=False)
    print(f"  full result set: {len(full):,} rows -> full_param_sweep.parquet")

    # ── condensed summary: one row per entry combo (best fwd_day x time by total return) ──
    grp = ["lookback_days", "volume_multiple", "rsi_threshold"]
    best = full.loc[full.groupby(grp)["total_return_fixedbase_pct"].idxmax()].copy()
    best = best.rename(columns={"forward_day": "best_forward_day",
                                "intraday_time": "best_intraday_time"})
    n_all = len(best)
    eligible = best[best["n_trades"] >= MIN_TRADES].copy()
    n_excl = n_all - len(eligible)

    summary = (eligible[grp + ["best_forward_day", "best_intraday_time", "n_trades",
                               "win_rate_pct", "avg_return_pct",
                               "avg_relative_strength_pct", "total_return_fixedbase_pct"]]
               .sort_values("total_return_fixedbase_pct", ascending=False)
               .reset_index(drop=True))
    with pd.ExcelWriter(OUTDIR / "top_combos_summary.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="top_combos", index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 130)
    print("TOP 20 ENTRY COMBOS (collapsed to best forward_day x intraday_time, n_trades >= "
          f"{MIN_TRADES})")
    print("=" * 130)
    print(summary.head(20).to_string(index=False))
    print(f"\nCombos evaluated : {n_all:,} of {total_combos:,} possible")
    print(f"Excluded by n_trades >= {MIN_TRADES}: {n_excl:,}  "
          f"({n_excl / n_all * 100:.2f}% of evaluated)")
    print(f"Eligible for ranking          : {len(summary):,}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
