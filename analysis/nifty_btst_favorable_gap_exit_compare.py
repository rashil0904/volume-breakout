# -*- coding: utf-8 -*-
"""nifty_btst_favorable_gap_exit_compare.py — ANALYSIS-ONLY favorable-gap exit-timing comparison on the FINAL
NIFTY BTST strategy's actual trade set (385 trades, VIX 17-19 excluded, DTE-1 fix applied). For each trade,
checks whether next-day spot OPEN moved >=X% in the trade's favor vs the entry-day 15:20 reference spot
(RED: open <= ref*(1-X%) ; GREEN: open >= ref*(1+X%)), swept X in {1,1.5,2,2.5,3}%. For qualifying trades,
compares combined (2xlong-1xshort) exit value at 09:15 OPEN / 09:16 OPEN / 09:17 CLOSE (current convention).
Does not change entry logic or the strategy itself — pure exit-timing diagnostic on a gap subset.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "btst_close_direction_FINAL"; OUTDIR.mkdir(parents=True, exist_ok=True)
ENTRY_MOD = 15 * 60 + 20
THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]


def leg_series(folder, strike, otype):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    return o.set_index("ts").sort_index()


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot1520 = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    next_open = sp.groupby("date")["open"].first()          # 09:15 spot open
    spot916_open = sp[sp["mod"] == 9 * 60 + 16].groupby("date")["open"].last()
    spot917_close = sp[sp["mod"] == 9 * 60 + 17].groupby("date")["close"].last()

    F = pd.read_csv(OUTDIR / "btst_close_direction_FINAL_v2_trades.csv")
    F["entry_date"] = pd.to_datetime(F["entry_date"]); F["exit_date"] = pd.to_datetime(F["exit_date"])

    rows = []; skipped = 0
    for _, r in F.iterrows():
        D = r.entry_date; Dn = r.exit_date
        if D not in spot1520.index or Dn not in next_open.index: skipped += 1; continue
        ref = float(spot1520.loc[D]); n_open = float(next_open.loc[Dn])
        gap_pct = (n_open - ref) / ref * 100

        ot = "PE" if r.direction == "RED" else "CE"
        long_K = int(r.ATM); short_K = int(r.short_leg.split(" ")[1])
        folder = pd.Timestamp(r.expiry_used).strftime("%Y%m%d")
        Lo = leg_series(folder, long_K, ot); So = leg_series(folder, short_K, ot)
        if Lo is None or So is None: skipped += 1; continue

        t915 = Dn + pd.Timedelta(hours=9, minutes=15); t916 = Dn + pd.Timedelta(hours=9, minutes=16); t917 = Dn + pd.Timedelta(hours=9, minutes=17)
        try:
            L915 = Lo["open"].asof(t915); L916 = Lo["open"].asof(t916); L917 = Lo["close"].asof(t917)
            S915 = So["open"].asof(t915); S916 = So["open"].asof(t916); S917 = So["close"].asof(t917)
        except Exception:
            skipped += 1; continue
        if any(pd.isna(v) for v in (L915, L916, L917, S915, S916, S917)): skipped += 1; continue

        entry_cost = float(r.entry_cost)
        ev_915 = 2 * L915 - S915; ev_916 = 2 * L916 - S916; ev_917 = 2 * L917 - S917
        pnl_915 = ev_915 - entry_cost; pnl_916 = ev_916 - entry_cost; pnl_917 = ev_917 - entry_cost

        nifty_916 = spot916_open.get(Dn, np.nan); nifty_917 = spot917_close.get(Dn, np.nan)
        rows.append({"entry_date": D.date(), "direction": r.direction, "entry_vix": r.entry_vix, "ref_spot_1520": round(ref, 2),
                     "gap_pct": round(gap_pct, 3),
                     "nifty_915_open": round(n_open, 2), "exit_915_open_pnl": round(pnl_915, 2),
                     "nifty_916_open": round(float(nifty_916), 2) if pd.notna(nifty_916) else None, "exit_916_open_pnl": round(pnl_916, 2),
                     "nifty_917_close": round(float(nifty_917), 2) if pd.notna(nifty_917) else None, "exit_917_close_pnl": round(pnl_917, 2)})

    T = pd.DataFrame(rows)
    print(f"total trades processed: {len(T)} | skipped (missing data) {skipped}", flush=True)

    def qualifies(row, X):
        if row.direction == "RED": return row.gap_pct <= -X
        else: return row.gap_pct >= X

    def stats(df, col):
        p = df[col]
        return {"trades": len(p), "win_%": round((p > 0).mean() * 100, 1) if len(p) else 0,
                "total_pnl": round(p.sum(), 1) if len(p) else 0, "avg_pnl": round(p.mean(), 2) if len(p) else 0}

    summary_rows = []; detail_sheets = {}
    for X in THRESHOLDS:
        mask = T.apply(lambda r: qualifies(r, X), axis=1)
        sub = T[mask]
        s915 = stats(sub, "exit_915_open_pnl"); s916 = stats(sub, "exit_916_open_pnl"); s917 = stats(sub, "exit_917_close_pnl")
        best = max([("09:15 OPEN", s915["avg_pnl"]), ("09:16 OPEN", s916["avg_pnl"]), ("09:17 CLOSE (current)", s917["avg_pnl"])], key=lambda x: x[1])
        summary_rows.append({"threshold_X%": X, "qualifying_trades": len(sub),
                             "09:15_OPEN_total": s915["total_pnl"], "09:15_OPEN_avg": s915["avg_pnl"], "09:15_OPEN_win%": s915["win_%"],
                             "09:16_OPEN_total": s916["total_pnl"], "09:16_OPEN_avg": s916["avg_pnl"], "09:16_OPEN_win%": s916["win_%"],
                             "09:17_CLOSE_total": s917["total_pnl"], "09:17_CLOSE_avg": s917["avg_pnl"], "09:17_CLOSE_win%": s917["win_%"],
                             "best_avg_exit": best[0], "small_sample_flag": "YES (<15 trades)" if len(sub) < 15 else ""})
        detail_sheets[X] = sub.copy()

    SUM = pd.DataFrame(summary_rows)
    with pd.ExcelWriter(OUTDIR / "favorable_gap_exit_compare.xlsx", engine="openpyxl") as w:
        SUM.to_excel(w, sheet_name="Summary", index=False)
        for X in THRESHOLDS:
            detail_sheets[X].to_excel(w, sheet_name=f"Detail_{str(X).replace('.', '_')}pct", index=False)
        T.to_excel(w, sheet_name="All_Trades_GapCalc", index=False)
    T.to_csv(OUTDIR / "favorable_gap_all_trades.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 100 + "\nFAVORABLE-GAP EXIT-PRICE COMPARISON\n" + "=" * 100)
    print(SUM.to_string(index=False))
    print("\n--- Detail @ 2% threshold ---")
    print(detail_sheets[2.0].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/favorable_gap_exit_compare.xlsx")


if __name__ == "__main__":
    main()
