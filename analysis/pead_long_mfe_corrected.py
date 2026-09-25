# -*- coding: utf-8 -*-
"""pead_long_mfe_corrected.py — MFE (max favorable excursion) for F&O LONGS on the CORRECTED full event
set (fixes the DD-MM datetime parse). ADDITIVE: does not touch pead_mfe.py. Window = T+1..T+15 (events
with >=15 fwd days). Answers: does extra return from longer holds come from real drift (peak day extends)
or give-back (peak early, exit surrenders it)? Reports mfe_pct/day distribution, give-back vs no-target
close-SL realized per hold T+10..T+15, and target-capture rate at 8/10/13%. IS 2022-24 / OOS 2025+.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
VWAPD = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "pead_long_mfe_corrected"
K, COST, NMAX = 4.0, 0.20, 15
IS_YEARS = {2022, 2023, 2024}


def build():
    a = pd.read_csv(ANN, dtype=str)
    a["ann_date"] = pd.to_datetime(a["announcement_date"], format="%d-%m-%Y", errors="coerce").dt.date
    a["dtm"] = pd.to_datetime(a["announcement_datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    a = a.dropna(subset=["ann_date", "dtm"]).sort_values("dtm").drop_duplicates(["symbol", "quarter"], keep="first")
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "high", "low", "close"]); d["date"] = pd.to_datetime(d["date"]).dt.date
    vw = pd.read_parquet(VWAPD, columns=["symbol", "date", "close"]); vw["date"] = pd.to_datetime(vw["date"]).dt.date
    vmap = {(s, dt): c for s, dt, c in zip(vw["symbol"], vw["date"], vw["close"])}
    sd = {}
    for sym, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date")
        sd[sym] = (g["date"].values, g["high"].values.astype(float), g["low"].values.astype(float),
                   g["close"].values.astype(float), {dt: i for i, dt in enumerate(g["date"].values)})
    ev = []; seen = set()
    for r in a.itertuples():
        s = sd.get(r.symbol)
        if s is None:
            continue
        dates, hi, lo, cl, dmap = s
        ps = dmap.get(r.ann_date); pn = int(np.searchsorted(dates, r.ann_date, "right")); pn = pn if pn < len(dates) else None
        p = pn if r.announcement_session in ("post_market", "during_market") else (ps if ps is not None else pn)
        if p is None or p < 1 or p + NMAX >= len(dates):
            continue
        if (dates[p] - dates[p - 1]).days > 6:                # GUARD1: stale prev-VWAP (data gap)
            continue
        key = (r.symbol, int(p))                              # GUARD2: de-dup by (symbol, T0)
        if key in seen:
            continue
        seen.add(key)
        entry = cl[p]; pv = vmap.get((r.symbol, dates[p - 1]), np.nan)
        if not (entry > 0 and pv == pv and pv > 0):
            continue
        rr = (entry - pv) / pv * 100.0
        if abs(rr) > 50.0:                                    # GUARD3: corporate-action / bad-print
            continue
        if rr < K:
            continue
        ev.append({"symbol": r.symbol, "year": int(str(dates[p])[:4]), "entry": entry, "sl": lo[p],
                   "fh": hi[p + 1:p + 1 + NMAX], "fl": lo[p + 1:p + 1 + NMAX], "fc": cl[p + 1:p + 1 + NMAX]})
    return ev


def realized_no_target(e, n):                                # close-basis SL, no target (the no-tgt column)
    entry, sl, fc = e["entry"], e["sl"], e["fc"]
    for j in range(n):
        if fc[j] <= sl:
            return (fc[j] - entry) / entry * 100.0
    return (fc[n - 1] - entry) / entry * 100.0


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ev = build()
    rows = []
    for e in ev:
        entry = e["entry"]; fh = e["fh"]; fl = e["fl"]
        favor = (fh - entry) / entry * 100.0; adverse = (fl - entry) / entry * 100.0
        mfe = float(favor.max()); mfe_day = int(favor.argmax()) + 1
        mae = float(adverse.min()); mae_day = int(adverse.argmin()) + 1
        rows.append({"symbol": e["symbol"], "year": e["year"], "mfe_pct": round(mfe, 3), "mfe_day": mfe_day,
                     "mae_pct": round(mae, 3), "mae_day": mae_day,
                     **{f"real_T{n}": round(realized_no_target(e, n), 3) for n in range(10, 16)}})
    T = pd.DataFrame(rows)
    n = len(T)
    print(f"F&O LONG events (corrected, >=15 fwd): {n} | IS {int(T.year.isin(IS_YEARS).sum())} OOS {int((~T.year.isin(IS_YEARS)).sum())}")

    def blk(df, lbl):
        peak_buckets = {"peak<=T3": (df.mfe_day <= 3).mean(), "T4-5": df.mfe_day.between(4, 5).mean(),
                        "T6-10": df.mfe_day.between(6, 10).mean(), "T11-15": (df.mfe_day >= 11).mean()}
        return {"seg": lbl, "n": len(df), "avg_mfe%": round(df.mfe_pct.mean(), 2), "med_mfe%": round(df.mfe_pct.median(), 2),
                "avg_mfe_day": round(df.mfe_day.mean(), 1), "med_mfe_day": int(df.mfe_day.median()),
                "avg_mae%": round(df.mae_pct.mean(), 2),
                **{k: round(v * 100, 1) for k, v in peak_buckets.items()},
                "mfe>=8%": round((df.mfe_pct >= 8).mean() * 100, 1), "mfe>=10%": round((df.mfe_pct >= 10).mean() * 100, 1),
                "mfe>=13%": round((df.mfe_pct >= 13).mean() * 100, 1)}
    SUM = pd.DataFrame([blk(T, "ALL"), blk(T[T.year.isin(IS_YEARS)], "IS 22-24"), blk(T[~T.year.isin(IS_YEARS)], "OOS 25+")])

    # give-back: avg MFE vs avg realized(no-target close-SL) per hold; give-back = MFE - realized
    gb = []
    for nn in range(10, 16):
        rz = T[f"real_T{nn}"]
        gb.append({"hold": f"T+{nn}", "avg_realized%": round(rz.mean(), 3), "avg_mfe%(full15)": round(T.mfe_pct.mean(), 2),
                   "avg_giveback%": round(T.mfe_pct.mean() - rz.mean(), 2), "capture_ratio": round(rz.mean() / T.mfe_pct.mean(), 2),
                   "IS_real%": round(T[T.year.isin(IS_YEARS)][f"real_T{nn}"].mean(), 3),
                   "OOS_real%": round(T[~T.year.isin(IS_YEARS)][f"real_T{nn}"].mean(), 3)})
    GB = pd.DataFrame(gb)
    dist = T["mfe_day"].value_counts().reindex(range(1, 16), fill_value=0).rename("n_trades")

    with pd.ExcelWriter(OUTDIR / "pead_long_mfe_corrected.xlsx", engine="openpyxl") as w:
        SUM.to_excel(w, sheet_name="summary", index=False)
        GB.to_excel(w, sheet_name="giveback_by_hold", index=False)
        dist.reset_index().to_excel(w, sheet_name="mfe_day_distribution", index=False)
        T.to_excel(w, sheet_name="per_trade", index=False)
    T.to_csv(OUTDIR / "pead_long_mfe_corrected_per_trade.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 92 + "\nF&O LONG MFE (corrected full set), window T+1..T+15\n" + "=" * 92)
    print(SUM.to_string(index=False))
    print("\n--- MFE-day distribution (when the favorable peak lands) ---")
    print(dist.to_string())
    print("\n--- GIVE-BACK: no-target close-SL realized vs full-15 MFE, per hold ---")
    print(GB.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
