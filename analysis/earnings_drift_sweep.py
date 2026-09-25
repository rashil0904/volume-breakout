# -*- coding: utf-8 -*-
"""earnings_drift_sweep.py — post-earnings drift event study on BSE F&O result filings, SWEEPING the
reaction threshold K in {3,4,5,6,7}%. Reaction events: post->next-day T0; pre->same-day T0;
during->same-day AND next-day (two events). reaction_return=(T0 close - prev close)/prev*100.
K: >=+K long, <=-K short. Drift T+1..T+10 cumulative from T0 close, directional. SL=T0 low/high tracked.
Dedup to earliest Result filing per (symbol, quarter). Daily cache = daily_ohlcv_all (plain close).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
OUTDIR = rb.RESULTS / "earnings_drift_sweep"
KS = [3, 4, 5, 6, 7]
HOR = 10


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(ANN)
    a["ann_date"] = pd.to_datetime(a["announcement_date"], format="%d-%m-%Y", errors="coerce").dt.date
    a["dtm"] = pd.to_datetime(a["announcement_datetime"], errors="coerce")
    a = a.dropna(subset=["ann_date", "dtm"])
    # dedup: earliest Result filing per (symbol, quarter)
    a = a.sort_values("dtm").drop_duplicates(["symbol", "quarter"], keep="first").reset_index(drop=True)
    print(f"result events (dedup per symbol-quarter): {len(a):,} | symbols {a['symbol'].nunique()}")

    d = pd.read_parquet(DAILY, columns=["symbol", "date", "high", "low", "close"])
    d["date"] = pd.to_datetime(d["date"]).dt.date
    sym_data = {}
    for sym, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date")
        sym_data[sym] = (g["date"].values, g["close"].values.astype(float), g["low"].values.astype(float),
                         g["high"].values.astype(float), {dt: i for i, dt in enumerate(g["date"].values)})

    events = []
    for r in a.itertuples():
        sd = sym_data.get(r.symbol)
        if sd is None:
            continue
        dates, cl, lo, hi, dmap = sd
        sess = r.announcement_session
        # derive (reaction_type, T0 position)
        cands = []
        pos_same = dmap.get(r.ann_date)                                   # same-day position (if trading day)
        pos_next = int(np.searchsorted(dates, r.ann_date, "right"))       # first index with date > ann_date
        pos_next = pos_next if pos_next < len(dates) else None
        if sess == "post_market":
            if pos_next is not None:
                cands.append(("post", pos_next))
        elif sess == "pre_market":
            cands.append(("pre", pos_same if pos_same is not None else pos_next))
        elif sess == "during_market":
            if pos_same is not None:
                cands.append(("during_sameday", pos_same))
            if pos_next is not None:
                cands.append(("during_nextday", pos_next))
        for rtype, p in cands:
            if p is None or p < 1 or p + 1 >= len(dates):
                continue
            entry = cl[p]; prevc = cl[p - 1]
            if not (entry > 0 and prevc > 0):
                continue
            rr = (entry - prevc) / prevc * 100.0
            nf = min(HOR, len(dates) - 1 - p)                             # available forward days
            fc = cl[p + 1:p + 1 + nf]; fl = lo[p + 1:p + 1 + nf]; fh = hi[p + 1:p + 1 + nf]
            rawd = (fc - entry) / entry * 100.0                           # raw close drift (long-sense)
            rec = {"symbol": r.symbol, "quarter": r.quarter, "reaction_type": rtype,
                   "T0_date": str(dates[p]), "reaction_return_pct": round(rr, 3),
                   "entry": round(entry, 2), "t0_low": lo[p], "t0_high": hi[p], "nf": nf}
            for n in range(1, HOR + 1):
                rec[f"raw_drift_T{n}"] = round(rawd[n - 1], 4) if n <= nf else np.nan
            rec["min_fwd_low"] = float(np.min(fl)) if nf else np.nan
            rec["max_fwd_high"] = float(np.max(fh)) if nf else np.nan
            events.append(rec)
    E = pd.DataFrame(events)
    print(f"reaction events: {len(E):,} | by type: {E['reaction_type'].value_counts().to_dict()}")

    # ── per-K aggregation ──
    drift_rows = []; xcut = []; per_event_out = []
    curves = {"long": {}, "short": {}}
    for K in KS:
        for direction in ["long", "short"]:
            if direction == "long":
                sub = E[E["reaction_return_pct"] >= K].copy(); sign = 1.0
                sub["sl_hit"] = sub["min_fwd_low"] <= sub["t0_low"]
            else:
                sub = E[E["reaction_return_pct"] <= -K].copy(); sign = -1.0
                sub["sl_hit"] = sub["max_fwd_high"] >= sub["t0_high"]
            if len(sub) == 0:
                continue
            # directional drift
            dd = pd.DataFrame({f"d{n}": sign * sub[f"raw_drift_T{n}"] for n in range(1, HOR + 1)})
            row = {"K": K, "direction": direction, "reaction_type": "ALL", "n_events": len(sub),
                   "sl_hit_rate_pct": round(sub["sl_hit"].mean() * 100, 1)}
            for n in range(1, HOR + 1):
                x = dd[f"d{n}"].dropna()
                row[f"avg_T{n}"] = round(x.mean(), 3) if len(x) else np.nan
                row[f"med_T{n}"] = round(x.median(), 3) if len(x) else np.nan
                row[f"win_T{n}"] = round((x > 0).mean() * 100, 1) if len(x) else np.nan
            drift_rows.append(row)
            curves[direction][K] = [row[f"avg_T{n}"] for n in range(1, HOR + 1)]
            xcut.append({"K": K, "direction": direction, "n_events": len(sub),
                         "avg_T5": row["avg_T5"], "win_T5": row["win_T5"], "avg_T10": row["avg_T10"],
                         "win_T10": row["win_T10"], "sl_hit_rate_pct": row["sl_hit_rate_pct"]})
            # by reaction_type
            for rt, g in sub.groupby("reaction_type"):
                ddr = pd.DataFrame({f"d{n}": sign * g[f"raw_drift_T{n}"] for n in range(1, HOR + 1)})
                rr = {"K": K, "direction": direction, "reaction_type": rt, "n_events": len(g),
                      "sl_hit_rate_pct": round(g["sl_hit"].mean() * 100, 1)}
                for n in range(1, HOR + 1):
                    x = ddr[f"d{n}"].dropna()
                    rr[f"avg_T{n}"] = round(x.mean(), 3) if len(x) else np.nan
                    rr[f"win_T{n}"] = round((x > 0).mean() * 100, 1) if len(x) else np.nan
                drift_rows.append(rr)
            # per-event (K=3 union captures all; tag which K each qualifies for once)
        # per-event K membership (once)
    DRIFT = pd.DataFrame(drift_rows)
    XCUT = pd.DataFrame(xcut).sort_values(["direction", "K"])

    # per-event detail with K-membership + directional drift at its own reaction sign
    Edet = E.copy()
    Edet["qualifies_K"] = Edet["reaction_return_pct"].abs().apply(lambda v: ",".join(str(k) for k in KS if v >= k))
    Edet["trade_dir"] = np.where(Edet["reaction_return_pct"] > 0, "long", "short")
    for n in range(1, HOR + 1):
        Edet[f"drift_T{n}"] = np.where(Edet["reaction_return_pct"] > 0, Edet[f"raw_drift_T{n}"], -Edet[f"raw_drift_T{n}"]).round(3)
    Edet["sl_hit"] = np.where(Edet["reaction_return_pct"] > 0, Edet["min_fwd_low"] <= Edet["t0_low"], Edet["max_fwd_high"] >= Edet["t0_high"])
    keepcols = ["symbol", "quarter", "reaction_type", "T0_date", "reaction_return_pct", "trade_dir", "qualifies_K",
                "sl_hit"] + [f"drift_T{n}" for n in range(1, HOR + 1)]
    Edet[keepcols].to_csv(OUTDIR / "per_event_drift.csv", index=False)

    with pd.ExcelWriter(OUTDIR / "earnings_drift_sweep.xlsx", engine="openpyxl") as w:
        XCUT.to_excel(w, sheet_name="T5_T10_by_K", index=False)
        DRIFT[DRIFT.reaction_type == "ALL"].to_excel(w, sheet_name="drift_curves_ALL", index=False)
        DRIFT.to_excel(w, sheet_name="drift_by_type", index=False)

    # charts
    for direction in ["long", "short"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for K in KS:
            if K in curves[direction]:
                ax.plot(range(1, HOR + 1), curves[direction][K], marker="o", label=f"K={K}% (n={XCUT[(XCUT.K==K)&(XCUT.direction==direction)]['n_events'].values[0]})")
        ax.axhline(0, color="grey", ls=":"); ax.set_xlabel("trading days after T0"); ax.set_ylabel("avg directional drift %")
        ax.set_title(f"Post-earnings drift ({direction}) by reaction threshold K"); ax.legend()
        fig.tight_layout(); fig.savefig(OUTDIR / f"drift_curves_{direction}.png", dpi=120); plt.close(fig)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 92 + "\nPOST-EARNINGS DRIFT — THRESHOLD SWEEP (K=3..7%)\n" + "=" * 92)
    print("\n--- T+5 & T+10 avg drift + win-rate + n_events by K (ALL reaction types) ---")
    print(XCUT.to_string(index=False))
    print("\n--- full drift curves (ALL types, avg) ---")
    cc = ["K", "direction", "n_events"] + [f"avg_T{n}" for n in [1, 2, 3, 5, 7, 10]] + ["sl_hit_rate_pct"]
    print(DRIFT[DRIFT.reaction_type == "ALL"][cc].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
