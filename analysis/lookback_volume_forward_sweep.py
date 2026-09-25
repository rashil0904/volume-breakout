# -*- coding: utf-8 -*-
"""
lookback_volume_forward_sweep.py
================================
Sweeps the ENTRY condition's lookback window and volume multiple (entry TIME fixed
at 3:15pm), then tracks forward returns T+1..T+7 with a full 15-min exit sweep
(09:30..15:00, 23 times) on each forward day.

Grid: lookback_days [10..45] x volume_multiple [3..10] = 288 combos.
Entry = mcap band (from diagnostic table) + cum-vol(09:15-14:45) >= M x trailing
L-day avg full-day volume + return-vs-prev-close >= 5% (kept). Entry px = 15:15 open.

Per combo the qualifying trade set is computed ONCE and reused across all 7x23
forward points; each candidate's forward prices are fetched once.
"""

import sys, bisect
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
OUTDIR = rb.RESULTS / "lookback_volume_sweep"
LOOKBACKS = list(range(10, 46))                      # every lookback 10..45
VOL_MULTS = [3, 4, 5, 6, 7, 8, 9, 10]
FWD_DAYS = [1, 2, 3, 4, 5, 6, 7]
TIMES_HM = list(range(570, 901, 15))                 # 09:30 .. 15:00 (23)
TLABEL = [f"{h//60:02d}:{h%60:02d}" for h in TIMES_HM]
HM_915, HM_1445, HM_1515 = 555, 885, 915
RET_MIN = 5.0
RSI_MINS = [0.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]   # inclusive; 0 == no RSI filter
BASE_POOL, MAX_PER_STOCK = 500_000, 100_000


def day_target(n):
    return MAX_PER_STOCK if n <= 5 else BASE_POOL / n


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI on a daily-close series. Value at index i uses closes through i
    (entry day inclusive). Seed = SMA of the first `period` changes; thereafter
    avg = (prior*(period-1) + current) / period."""
    out = pd.Series(np.nan, index=close.index, dtype=float)
    d = close.diff().values
    if len(d) < period + 1:
        return out
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    g[0] = l[0] = 0.0                                 # d[0] is NaN
    # Wilder smoothing == ewm(alpha=1/period, adjust=False) seeded with the SMA
    gs, ls = g[period:].copy(), l[period:].copy()
    gs[0], ls[0] = g[1:period + 1].mean(), l[1:period + 1].mean()
    ag = pd.Series(gs).ewm(alpha=1.0 / period, adjust=False).mean().values
    al = pd.Series(ls).ewm(alpha=1.0 / period, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(al > 0, 100.0 - 100.0 / (1.0 + ag / al), 100.0)
    out.iloc[period:] = rsi
    return out


def _sweep_combo(rows, L, M, R, qi, entry, cdates, fwd):
    """Size + evaluate one (lookback, volume_multiple, rsi_min) trade set across all
    forward-day x intraday-time points, appending metric rows."""
    e_q, d_q, f_q = entry[qi], cdates[qi], fwd[qi]
    # standard per-day whole-share sizing on this combo's own signal counts
    cnt = pd.Series(d_q).map(pd.Series(d_q).value_counts()).values
    tgt = np.where(cnt <= 5, MAX_PER_STOCK, BASE_POOL / cnt)
    sh = np.floor(tgt / e_q)
    ok = sh > 0
    e_q, f_q, sh = e_q[ok], f_q[ok], sh[ok]
    if len(e_q) == 0:
        return
    ret_m = (f_q - e_q[:, None]) / e_q[:, None] * 100
    pnl_m = sh[:, None] * (f_q - e_q[:, None])
    valid = ~np.isnan(f_q)
    for k, fd_off in enumerate(FWD_DAYS):
        for t in range(len(TIMES_HM)):
            c = k * len(TIMES_HM) + t
            v = valid[:, c]
            nv = int(v.sum())
            if nv == 0:
                continue
            r, p = ret_m[v, c], pnl_m[v, c]
            rows.append({
                "lookback_days": L, "volume_multiple": M, "rsi_min": R,
                "forward_day": fd_off, "intraday_time": TLABEL[t],
                "n_trades": nv,
                "win_rate_pct": round(float((p > 0).mean() * 100), 2),
                "avg_return_pct": round(float(r.mean()), 4),
                "median_return_pct": round(float(np.median(r)), 4),
                "total_return_fixedbase_pct": round(float(p.sum()) / BASE_POOL * 100, 4),
            })


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── candidates: mcap-eligible days with >=5% move (volume applied per combo) ──
    print("Loading diagnostic table …")
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "entry_price_315pm",
                                "return_pct_vs_prev_close", "cum_volume_to_3pm_today"],
                       parse_dates=["date"])
    cand = diag[(diag["return_pct_vs_prev_close"] >= RET_MIN)
                & diag["entry_price_315pm"].notna()
                & diag["cum_volume_to_3pm_today"].notna()].reset_index(drop=True)
    print(f"  candidates (mcap + >=5% move): {len(cand):,} over {cand['symbol'].nunique():,} symbols")

    n = len(cand)
    entry = cand["entry_price_315pm"].values.astype(float)
    cumvol = cand["cum_volume_to_3pm_today"].values.astype(float)
    cdates = np.array([pd.Timestamp(d).date() for d in cand["date"].values], dtype=object)
    avgs = np.full((n, len(LOOKBACKS)), np.nan)                 # trailing avg per lookback
    rsis = np.full((n, len(LOOKBACKS)), np.nan)                 # Wilder RSI(lookback) on entry day
    fwd = np.full((n, len(FWD_DAYS) * len(TIMES_HM)), np.nan)   # 7 x 23 forward opens

    print("Scanning parquets for volume history + forward prices …")
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

        # full-day volume series -> trailing averages (non-zero days, min_periods=L, shift 1)
        fd = sess.groupby("date")["volume"].sum().reindex(dates_sorted).fillna(0)
        nz = fd[fd > 0]
        avg_maps = {}
        for L in LOOKBACKS:
            a = (nz.rolling(L, min_periods=L).mean().shift(1)
                   .reindex(dates_sorted, method="ffill"))
            avg_maps[L] = a.to_dict()

        # daily closes -> Wilder RSI, one series per lookback (period == lookback_days)
        dclose = sess.groupby("date")["close"].last().reindex(dates_sorted)
        rsi_maps = {L: wilder_rsi(dclose, L).to_dict() for L in LOOKBACKS}

        # forward-day opens at the 23 grid times
        po = (sess[sess["hm"].isin(TIMES_HM)]
              .pivot_table(index="date", columns="hm", values="open", aggfunc="last")
              .reindex(columns=TIMES_HM))

        for ridx, d in zip(grp.index, grp["date"]):
            dd = pd.Timestamp(d).date()
            for li, L in enumerate(LOOKBACKS):
                v = avg_maps[L].get(dd, np.nan)
                if v is not None and v == v:
                    avgs[ridx, li] = v
                r = rsi_maps[L].get(dd, np.nan)
                if r is not None and r == r:
                    rsis[ridx, li] = r
            j = bisect.bisect_right(dates_sorted, dd)
            for k in range(len(FWD_DAYS)):
                if j + k >= len(dates_sorted):
                    break
                fdate = dates_sorted[j + k]
                if fdate in po.index:
                    fwd[ridx, k * len(TIMES_HM):(k + 1) * len(TIMES_HM)] = po.loc[fdate].values
        if si % 100 == 0:
            print(f"    …{si} symbols")

    # ── sweep 56 combos ──
    print(f"Sweeping {len(LOOKBACKS) * len(VOL_MULTS) * len(RSI_MINS):,} "
          f"(lookback, volume_multiple, rsi_min) combos …")
    rows, selrows = [], []
    for li, L in enumerate(LOOKBACKS):
        avgL, rsiL = avgs[:, li], rsis[:, li]
        for M in VOL_MULTS:
            q_pre = (~np.isnan(avgL)) & (avgL > 0) & (cumvol >= M * avgL)
            n_pre = int(q_pre.sum())
            for R in RSI_MINS:
                q = q_pre if R <= 0 else (q_pre & (~np.isnan(rsiL)) & (rsiL >= R))
                n_post = int(q.sum())
                selrows.append({
                    "lookback_days": L, "volume_multiple": M, "rsi_min": R,
                    "n_before_rsi": n_pre, "n_after_rsi": n_post,
                    "n_dropped": n_pre - n_post,
                    "pct_dropped": round((n_pre - n_post) / n_pre * 100, 2) if n_pre else np.nan,
                    "median_rsi_before": round(float(np.nanmedian(rsiL[q_pre])), 2) if n_pre else np.nan,
                })
                if not q.any():
                    continue
                _sweep_combo(rows, L, M, R, np.where(q)[0],
                             entry, cdates, fwd)
    curve = pd.DataFrame(rows)
    curve.to_csv(OUTDIR / "event_time_curve.csv", index=False)
    print(f"  event-time curve rows: {len(curve):,}")

    sel = pd.DataFrame(selrows)
    sel.to_csv(OUTDIR / "rsi_filter_selectivity.csv", index=False)

    # ── summary view: best intraday_time per (combo, forward_day) by avg_return_pct ──
    idx = curve.groupby(["lookback_days", "volume_multiple", "rsi_min",
                         "forward_day"])["avg_return_pct"].idxmax()
    summary = curve.loc[idx].rename(columns={"intraday_time": "best_intraday_time"}).reset_index(drop=True)
    summary = summary[["lookback_days", "volume_multiple", "rsi_min", "forward_day",
                       "best_intraday_time", "n_trades", "win_rate_pct", "avg_return_pct",
                       "median_return_pct", "total_return_fixedbase_pct"]]
    summary.to_csv(OUTDIR / "summary_view.csv", index=False)

    # ── RSI threshold profile: best T+1 combo at each threshold ──
    t1 = summary[summary["forward_day"] == 1]
    rsi_profile = (t1.loc[t1.groupby("rsi_min")["total_return_fixedbase_pct"].idxmax()]
                     .sort_values("rsi_min").reset_index(drop=True))
    rsi_profile.to_csv(OUTDIR / "rsi_threshold_profile.csv", index=False)

    with pd.ExcelWriter(OUTDIR / "lookback_volume_forward_sweep.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary_view", index=False)
        sel.to_excel(w, sheet_name="rsi_filter_selectivity", index=False)
        rsi_profile.to_excel(w, sheet_name="rsi_threshold_profile", index=False)
        # event_time_curve is ~370k rows -> CSV only (openpyxl write would be very slow)

    # ── best combo by max total_return_fixedbase_pct across all points ──
    best_row = curve.loc[curve["total_return_fixedbase_pct"].idxmax()]
    bL, bM = int(best_row["lookback_days"]), int(best_row["volume_multiple"])
    bR = float(best_row["rsi_min"])
    sub = curve[(curve["lookback_days"] == bL) & (curve["volume_multiple"] == bM)
                & (curve["rsi_min"] == bR)]

    fig, ax = plt.subplots(figsize=(11, 6))
    for t in TLABEL:
        s = sub[sub["intraday_time"] == t].sort_values("forward_day")
        ax.plot(s["forward_day"], s["avg_return_pct"], color="#b0c4de", lw=0.8, alpha=0.7)
    bestline = sub.loc[sub.groupby("forward_day")["avg_return_pct"].idxmax()].sort_values("forward_day")
    ax.plot(bestline["forward_day"], bestline["avg_return_pct"], color="#b2182b", lw=2.5,
            marker="o", label="best intraday time each day")
    for _, r in bestline.iterrows():
        ax.annotate(r["intraday_time"], (r["forward_day"], r["avg_return_pct"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("forward day (T+d)"); ax.set_ylabel("avg return %")
    rlab = "no RSI filter" if bR <= 0 else f"RSI({bL})>={bR:.0f}"
    ax.set_title(f"Event-time curve — lookback={bL}d, volume_multiple={bM}x, {rlab} "
                 f"(best combo by total_return_fixedbase_pct)", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "event_time_curve_best_combo.png", dpi=130); plt.close(fig)

    # ── prints ──
    pd.set_option("display.width", 200)
    print("\n" + "=" * 110)
    print("TOP 10 SUMMARY ROWS (by total_return_fixedbase_pct)")
    print("=" * 110)
    print(summary.sort_values("total_return_fixedbase_pct", ascending=False).head(10).to_string(index=False))

    print("\n" + "=" * 110)
    print("RSI THRESHOLD PROFILE — best T+1 combo at each threshold (rsi_min=0 is the unfiltered baseline)")
    print("=" * 110)
    print(rsi_profile.to_string(index=False))

    print("\n" + "=" * 110)
    print("RSI FILTER SELECTIVITY — % of qualifying events dropped (pre-sizing)")
    print("=" * 110)
    print("  by threshold x lookback (averaged over volume multiples):")
    piv = (sel[sel.rsi_min > 0]
           .pivot_table(index="lookback_days", columns="rsi_min", values="pct_dropped"))
    print(piv.round(1).to_string())
    print("\n  overall surviving events by threshold:")
    agg = sel.groupby("rsi_min")[["n_before_rsi", "n_after_rsi"]].sum()
    agg["pct_dropped"] = ((1 - agg.n_after_rsi / agg.n_before_rsi) * 100).round(2)
    print(agg.to_string())
    print(f"\nBEST COMBO: lookback={bL}d, volume_multiple={bM}x, {rlab}  "
          f"(peak total_return_fixedbase_pct={best_row['total_return_fixedbase_pct']:.2f} at "
          f"T+{int(best_row['forward_day'])} {best_row['intraday_time']})")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
