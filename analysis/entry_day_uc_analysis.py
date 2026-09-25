# -*- coding: utf-8 -*-
"""
entry_day_uc_analysis.py
========================
Upper-circuit (UC) behavior on the ENTRY DAY for the main strategy's trades.
Reuses final_performance_report.build_trades() (NOT recomputed).

UC def (touched): a stock "hit UC" on entry day if any 15-min candle HIGH >= uc_level,
  uc_level = prev_close * 1.1995   (~20% band; 19.95% practical threshold).
prev_close = plain prior-day 15:15 close (exchange circuit convention; note this differs from
the strategy's VWAP-based prev-close — circuits are set off the plain close).
uc_hit_time = first candle whose high reached uc_level.
Fill-realism flag: entry (15:15 open) >= uc_level -> assumed entry sits at/above the circuit
(a UC-locked stock can't be bought there).
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
DAY_CLOSE_HM, ENTRY_HM = 915, 915
SESSION_HMS = list(range(555, 916, 15))


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Building main-strategy trades (reused) …")
    T = fpr.build_trades()[["symbol", "entry_date", "entry_price"]].copy().reset_index(drop=True)
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
        pc = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        day_close = pc[DAY_CLOSE_HM].where(pc[DAY_CLOSE_HM].notna(), pc.ffill(axis=1).iloc[:, -1])
        dates = sorted(ph.index)
        prev_close = {dates[k]: day_close.get(dates[k-1], np.nan) for k in range(1, len(dates))}
        for ed, epx in zip(grp["entry_date"], grp["entry_price"]):
            pcl = prev_close.get(ed, np.nan)
            if not (pcl == pcl and pcl > 0) or ed not in ph.index:
                recs.append({"symbol": sym, "entry_date": ed, "has_data": False}); continue
            uc = pcl * UC_MULT
            highs = ph.loc[ed]
            hit_mask = highs.values >= uc
            hit = bool(np.nansum(hit_mask) > 0)
            first_hm = SESSION_HMS[int(np.argmax(hit_mask))] if hit else np.nan
            entry_high = ph.loc[ed, ENTRY_HM]
            recs.append({"symbol": sym, "entry_date": ed, "has_data": True,
                         "prev_close": round(float(pcl), 2), "uc_level": round(float(uc), 2),
                         "entry_price": round(float(epx), 2), "hit_uc": hit,
                         "uc_hit_hm": first_hm, "uc_hit_time": hm_lbl(first_hm) if hit else "",
                         "uc_at_entry_high": bool(entry_high >= uc) if entry_high == entry_high else False,
                         "fill_at_or_above_uc": bool(epx >= uc)})
        if si % 100 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(recs)
    valid = df[df["has_data"]].copy()
    n = len(valid)
    hitters = valid[valid["hit_uc"]].copy()
    nh = len(hitters)
    fill_above = int(valid["fill_at_or_above_uc"].sum())
    uc_at_entry = int(valid["uc_at_entry_high"].sum())

    mins = (hitters["uc_hit_hm"] - 555).astype(float)          # minutes from 09:15
    avg_min = float(mins.mean()); med_min = float(mins.median())
    def m2t(m): return hm_lbl(555 + int(round(m)))

    # time-of-day distribution (15-min bins = candle times)
    dist = (hitters["uc_hit_time"].value_counts()
            .reindex([hm_lbl(h) for h in SESSION_HMS]).fillna(0).astype(int))
    dist = dist[dist > 0]
    dist_tbl = pd.DataFrame({"uc_hit_time_bin": dist.index, "count": dist.values,
                             "pct_of_uc_hitters": (dist.values / nh * 100).round(2)})

    # per-day count of UC-hitters
    by_day = hitters.groupby("entry_date").size()

    # ── charts ──
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(dist.index, dist.values, color="#1f77b4")
    ax.set_xlabel("time of day UC first hit (15-min candle)"); ax.set_ylabel("count of stocks")
    ax.set_title("Entry-day UC: when during the day was upper circuit first tagged")
    plt.xticks(rotation=60); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "uc_hit_time_histogram.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bd = by_day.copy(); bd.index = pd.to_datetime(bd.index)
    ax.plot(bd.index, bd.values, lw=0.8, color="#2ca02c")
    ax.set_xlabel("entry date"); ax.set_ylabel("entry-day UC stocks / day")
    ax.set_title("Entry-day UC-hitters over time")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "uc_hitters_timeseries.png", dpi=120); plt.close(fig)

    summary = pd.DataFrame([{
        "n_trades_with_data": n, "n_hit_uc": nh, "pct_hit_uc": round(nh / n * 100, 2),
        "avg_uc_hit_time": m2t(avg_min), "median_uc_hit_time": m2t(med_min),
        "avg_uc_hit_min_from_open": round(avg_min, 1), "median_uc_hit_min_from_open": round(med_min, 1),
        "n_uc_at_entry_candle_high": uc_at_entry, "pct_uc_at_entry_candle": round(uc_at_entry / n * 100, 2),
        "n_fill_at_or_above_uc": fill_above, "pct_fill_at_or_above_uc": round(fill_above / n * 100, 2),
    }])

    with pd.ExcelWriter(OUTDIR / "entry_day_uc_analysis.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        dist_tbl.to_excel(w, sheet_name="uc_hit_time_distribution", index=False)
        by_day.rename("n_uc_hitters").rename_axis("date").reset_index().to_excel(w, sheet_name="uc_hitters_by_day", index=False)
        hitters[["symbol", "entry_date", "prev_close", "uc_level", "entry_price", "uc_hit_time",
                 "uc_at_entry_high", "fill_at_or_above_uc"]].to_excel(w, sheet_name="uc_trades", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 70 + "\nENTRY-DAY UPPER-CIRCUIT ANALYSIS (main strategy trades)\n" + "=" * 70)
    print(summary.T.to_string(header=False))
    print("\n--- UC-hit-time distribution (15-min bins) ---")
    print(dist_tbl.to_string(index=False))
    print(f"\n[fill-realism] {fill_above} trades ({fill_above/n*100:.1f}%) have the 3:15pm entry AT/ABOVE the UC "
          f"level -> the assumed 15:15-open fill may be unrealistic (circuit-locked, can't buy there).")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
