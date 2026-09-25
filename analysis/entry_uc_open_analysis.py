# -*- coding: utf-8 -*-
"""
entry_uc_open_analysis.py
=========================
Among the main strategy's entry-day UC-hitters, how many had the circuit OPEN (unlock) later.
Reuses final_performance_report.build_trades() (NOT recomputed).

uc_level = prev_close * 1.1995 (plain prior-day 15:15 close; ~+20% band).
Hit UC          = some entry-day candle high >= uc_level.
Circuit OPENED  = after first UC hit, a later candle trades BELOW uc_level (low < uc_level; default).
Stayed locked   = after first hit, every candle stays pinned >= uc_level to the close.
time_locked     = first-open candle time - first-UC-hit candle time (minutes).
3:15 subset     = 15:15 candle pinned at UC (low >= uc*0.999); check if it opened the NEXT day.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "entry_day_uc"
UC_MULT = 1.1995
ENTRY_HM, DAY_CLOSE_HM = 915, 915
SESSION_HMS = list(range(555, 916, 15))
PIN = 0.999


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Building main-strategy trades (reused) …")
    T = fpr.build_trades()[["symbol", "entry_date"]].copy().reset_index(drop=True)
    T["entry_date"] = pd.to_datetime(T["entry_date"]).dt.date
    print(f"  trades: {len(T)}")

    recs = []
    t0 = time.time()
    for si, (sym, grp) in enumerate(T.groupby("symbol"), 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[raw["hm"].isin(SESSION_HMS)]
        ph = raw.pivot_table(index="date", columns="hm", values="high", aggfunc="max").reindex(columns=SESSION_HMS)
        pl = raw.pivot_table(index="date", columns="hm", values="low", aggfunc="min").reindex(columns=SESSION_HMS)
        pc = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        day_close = pc[DAY_CLOSE_HM].where(pc[DAY_CLOSE_HM].notna(), pc.ffill(axis=1).iloc[:, -1])
        dates = sorted(ph.index)
        pos = {d: k for k, d in enumerate(dates)}
        prev_close = {dates[k]: day_close.get(dates[k-1], np.nan) for k in range(1, len(dates))}
        for ed in grp["entry_date"].values:
            pcl = prev_close.get(ed, np.nan)
            if not (pcl == pcl and pcl > 0) or ed not in ph.index:
                continue
            uc = pcl * UC_MULT
            hi = ph.loc[ed].values; lo = pl.loc[ed].values
            hit_mask = hi >= uc
            if not hit_mask.any():
                continue
            fi = int(np.argmax(hit_mask))                        # first UC-hit candle idx
            post_lo = lo[fi+1:]                                    # candles after first hit
            below = post_lo < uc
            opened = bool(np.nansum(below) > 0)
            oi = fi + 1 + int(np.argmax(below)) if opened else None
            first_hit_hm = SESSION_HMS[fi]
            open_hm = SESSION_HMS[oi] if opened else np.nan
            time_locked = (open_hm - first_hit_hm) if opened else np.nan
            # multi-touch: does it re-lock (high>=uc) after opening?
            multi = bool(opened and np.nansum(hi[oi+1:] >= uc) > 0)
            # 3:15pm locked: 15:15 candle pinned at UC (low >= uc*0.999 & high>=uc)
            locked_1515 = bool(lo[-1] >= uc * PIN and hi[-1] >= uc)
            # if locked at 15:15, did it open NEXT day (any next-day low < uc_level)?
            opened_next = None
            if locked_1515:
                k = pos.get(ed)
                if k is not None and k + 1 < len(dates):
                    nd = dates[k+1]
                    opened_next = bool(np.nansum(pl.loc[nd].values < uc) > 0)
            recs.append({"symbol": sym, "entry_date": ed, "uc_level": round(float(uc), 2),
                         "uc_first_hit_time": hm_lbl(first_hit_hm), "opened": opened,
                         "uc_open_time": hm_lbl(open_hm) if opened else "",
                         "time_locked_min": int(time_locked) if opened else np.nan,
                         "multi_touch": multi, "locked_at_1515": locked_1515,
                         "opened_next_day": opened_next})
        if si % 100 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s)")

    H = pd.DataFrame(recs)                                        # UC-hitters only
    nh = len(H)
    opened = H[H["opened"]]; locked = H[~H["opened"]]
    no, nl = len(opened), len(locked)
    tl = opened["time_locked_min"].dropna()

    l1515 = H[H["locked_at_1515"]]
    l_open_next = int((l1515["opened_next_day"] == True).sum())
    l_frozen = int((l1515["opened_next_day"] == False).sum())
    l_na = int(l1515["opened_next_day"].isna().sum())
    n_multi = int(H["multi_touch"].sum())

    summary = pd.DataFrame([{
        "n_uc_hitters": nh,
        "n_circuit_opened_sameday": no, "pct_opened": round(no / nh * 100, 2),
        "n_stayed_locked_to_close": nl, "pct_stayed_locked": round(nl / nh * 100, 2),
        "avg_time_locked_min": round(float(tl.mean()), 1) if len(tl) else np.nan,
        "median_time_locked_min": round(float(tl.median()), 1) if len(tl) else np.nan,
        "n_multi_touch": n_multi, "pct_multi_touch": round(n_multi / nh * 100, 2),
        "n_locked_at_1515": len(l1515),
        "of_1515locked_opened_next_day": l_open_next,
        "of_1515locked_stayed_frozen_next_day": l_frozen,
        "of_1515locked_no_nextday_data": l_na,
    }])

    # chart: distribution of time_locked (minutes) for opened stocks
    if len(tl):
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = list(range(0, int(tl.max()) + 30, 15))
        ax.hist(tl, bins=bins, color="#1f77b4", edgecolor="white")
        ax.axvline(tl.median(), ls="--", color="#d62728", label=f"median {tl.median():.0f} min")
        ax.set_xlabel("time locked before circuit opened (minutes)"); ax.set_ylabel("count of stocks")
        ax.set_title("How long UC-hitters stayed locked before the circuit opened")
        ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(OUTDIR / "uc_time_locked_hist.png", dpi=120); plt.close(fig)

    H.to_csv(OUTDIR / "uc_open_detail.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "entry_uc_open_analysis.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        H.to_excel(w, sheet_name="uc_hitter_detail", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 68 + "\nENTRY-DAY UC — CIRCUIT OPEN (UNLOCK) ANALYSIS\n" + "=" * 68)
    print(summary.T.to_string(header=False))
    print(f"\n  Of {nh} UC-hitters: {no} ({no/nh*100:.1f}%) had the circuit OPEN same day; "
          f"{nl} ({nl/nh*100:.1f}%) stayed locked to the close.")
    print(f"  time locked before opening: avg {tl.mean():.0f} min, median {tl.median():.0f} min.")
    print(f"  3:15pm-locked subset ({len(l1515)}): {l_open_next} opened next day, "
          f"{l_frozen} stayed frozen next day, {l_na} no next-day data.")
    print(f"  multi-touch (opened then re-locked): {n_multi} ({n_multi/nh*100:.1f}%).")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
