# -*- coding: utf-8 -*-
"""
breach_no_reclaim_sweep.py
==========================
Breach-and-no-reclaim sweep on the main strategy's pure 12:00-exit bucket
(exit_type == exit_at_t2_no_target: non-positive at 09:45, no 14% target, exited at the
12:00 open). Reuses canonical base positions + next-day OHLC; works only on this bucket.

For x in 1..10 (%): level L_x = entry * (1 - x/100).
  BREACHED      = min low over candles AFTER 09:45 .. candle BEFORE 12:00  <= L_x
  RECLAIMED     = 12:00 return > -x   (judged ONLY by the 12:00 open price, per spec)
  NOT_RECLAIMED = BREACHED and 12:00 return <= -x

Flags confirmed: (a) breach uses intraday LOWS, reclaim uses the 12:00 OPEN — a wick
breach that closes back above L_x by 12:00 counts as reclaimed (asymmetry intended).
(b) breach window = after 09:45 through the candle before 12:00 (10:00..11:45), matching
the earlier Bucket N stop windows.
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
import exit_time_sweep as ets

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "breach_no_reclaim"
TARGET = 14.0
CANDLE_HMS = list(range(555, 901, 15))
HCOL = {hm: k for k, hm in enumerate(CANDLE_HMS)}
HM_0945, HM_1200 = 585, 720
PRE945 = [hm for hm in CANDLE_HMS if hm < HM_0945]
SCAN = [hm for hm in CANDLE_HMS if HM_0945 < hm < HM_1200]     # 10:00 .. 11:45
XS = list(range(1, 11))


def fetch_ohlc(base):
    n = len(base)
    O = np.full((n, len(CANDLE_HMS)), np.nan)
    H = np.full((n, len(CANDLE_HMS)), np.nan)
    Lo = np.full((n, len(CANDLE_HMS)), np.nan)
    for sym, grp in base.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        ds = sorted(raw["date"].unique())
        sub = raw[raw["hm"].isin(CANDLE_HMS)]
        po = sub.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=CANDLE_HMS)
        ph = sub.pivot_table(index="date", columns="hm", values="high", aggfunc="last").reindex(columns=CANDLE_HMS)
        pl = sub.pivot_table(index="date", columns="hm", values="low", aggfunc="last").reindex(columns=CANDLE_HMS)
        for idx, ed in zip(grp.index, grp["date"]):
            j = bisect.bisect_right(ds, ed.date())
            if j >= len(ds):
                continue
            nd = ds[j]
            if nd in po.index:
                O[idx, :] = po.loc[nd].values
                H[idx, :] = ph.loc[nd].values
                Lo[idx, :] = pl.loc[nd].values
    return O, H, Lo


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = ets.load_base_positions()
    e = base["entry"].values.astype(float)
    print(f"Base positions: {len(base):,} | fetching next-day OHLC …")
    O, H, Lo = fetch_ohlc(base)

    scan_cols = [HCOL[hm] for hm in SCAN]
    pre_cols = [HCOL[hm] for hm in PRE945]
    o945 = O[:, HCOL[HM_0945]]
    o1200 = O[:, HCOL[HM_1200]]
    ret945 = (o945 - e) / e * 100
    ret1200 = (o1200 - e) / e * 100
    tgt = e * (1 + TARGET / 100)

    # ── reconstruct the pure exit_at_t2_no_target bucket ──
    # non-positive at 09:45, no pre-945 target, no target in 09:45..12:00, valid 12:00 open
    pre_hit = np.nanmax(np.where(np.isnan(H[:, pre_cols]), -np.inf, H[:, pre_cols]), axis=1) >= tgt
    scan_hit = np.nanmax(np.where(np.isnan(H[:, scan_cols]), -np.inf, H[:, scan_cols]), axis=1) >= tgt
    nonpos = ~(~np.isnan(o945) & (ret945 > 0))
    bucket = nonpos & ~pre_hit & ~scan_hit & ~np.isnan(o1200)
    nb = int(bucket.sum())
    print(f"exit_at_t2_no_target bucket: {nb:,} trades  "
          f"(validate vs report's 1,548)")

    bmin_low = np.nanmin(np.where(np.isnan(Lo[:, scan_cols]), np.inf, Lo[:, scan_cols]), axis=1)  # min low in window
    e_b, min_low_b, r1200_b = e[bucket], bmin_low[bucket], ret1200[bucket]

    rows = []
    for x in XS:
        Lx = e_b * (1 - x / 100)
        breached = min_low_b <= Lx
        reclaimed = r1200_b > -x                       # judged by 12:00 open only
        not_recl = breached & ~reclaimed
        n_br = int(breached.sum())
        n_nr = int(not_recl.sum())
        br_recl = breached & reclaimed
        rows.append({
            "x_pct": x,
            "n_breached": n_br,
            "pct_of_bucket_breached": round(n_br / nb * 100, 2),
            "n_breached_and_not_reclaimed": n_nr,
            "pct_never_reclaimed": round(n_nr / n_br * 100, 2) if n_br else np.nan,
            "avg_return_not_reclaimed_group": round(r1200_b[not_recl].mean(), 4) if n_nr else np.nan,
            "avg_return_reclaimed_group": round(r1200_b[br_recl].mean(), 4) if br_recl.any() else np.nan,
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUTDIR / "breach_no_reclaim_sweep.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "breach_no_reclaim_sweep.xlsx", engine="openpyxl") as w:
        tbl.to_excel(w, sheet_name="breach_no_reclaim", index=False)

    # ── chart ──
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(tbl["x_pct"], tbl["pct_never_reclaimed"], "o-", color="#b2182b", lw=2,
             label="% never reclaimed (of breached)")
    ax1.set_xlabel("breach level  x  (−x% from entry)")
    ax1.set_ylabel("% never reclaimed by 12:00", color="#b2182b")
    ax1.tick_params(axis="y", labelcolor="#b2182b")
    ax1.set_ylim(0, 105); ax1.axhline(90, color="#b2182b", ls=":", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(tbl["x_pct"], tbl["pct_of_bucket_breached"], "s--", color="#1f4e79", lw=2,
             label="% of bucket that breaches −x%")
    ax2.set_ylabel("% of bucket breaching −x%", color="#1f4e79")
    ax2.tick_params(axis="y", labelcolor="#1f4e79")
    ax1.set_xticks(XS)
    ax1.set_title("Breach & no-reclaim by −x% level — exit_at_t2_no_target bucket\n"
                  "(red: reliability of no-recovery | blue: how often −x% is even hit)",
                  fontweight="bold")
    lines = ax1.get_lines()[:1] + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
    ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "breach_no_reclaim.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 120)
    print("BREACH & NO-RECLAIM SWEEP — exit_at_t2_no_target bucket (n=%d)" % nb)
    print("=" * 120)
    print(tbl.to_string(index=False))

    # ── plain read: x with high no-reclaim AND meaningful breach frequency ──
    cand = tbl[(tbl["pct_never_reclaimed"] >= 90) & (tbl["pct_of_bucket_breached"] >= 20)]
    print("\n--- CANDIDATE STOP LEVELS (pct_never_reclaimed >= 90% AND pct_of_bucket_breached >= 20%) ---")
    if len(cand):
        print(cand[["x_pct", "pct_of_bucket_breached", "pct_never_reclaimed",
                    "avg_return_not_reclaimed_group", "avg_return_reclaimed_group"]].to_string(index=False))
    else:
        print("  NONE — no x satisfies both thresholds simultaneously (see read below).")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
