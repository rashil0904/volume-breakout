# -*- coding: utf-8 -*-
"""nifty_btst_monday_expiry_compare.py — For MONDAY-entry NIFTY BTST trades only (enter Mon 15:20, exit next
trading day 09:17), compare realized P&L using the strategy's NORMAL contract choice ("current expiry" = the
nearest expiry strictly after entry day, per the standard DTE-0 rule) vs deliberately using the FOLLOWING
week's contract instead ("next expiry" = the cycle right after current). Same direction/ATM/strikes (spot-
based, unaffected); only which expiry's option premiums are used differs. Entry/exit fills = option CLOSE
(matching the base strategy's convention exactly). OFFSET=300, ENTRY 15:20, EXIT 09:17. Read-only comparison,
does not alter the base strategy.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "btst_close_direction_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)
OFFSET = 300; ENTRY_MOD = 15 * 60 + 20; EXIT_MOD = 9 * 60 + 17


def leg_px(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None, None
    o = pd.read_parquet(fs[0], columns=["timestamp", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    s = o[(o.ts >= t0 - pd.Timedelta(minutes=5)) & (o.ts <= t1 + pd.Timedelta(minutes=5))].set_index("ts")["close"].sort_index()
    if s.empty: return None, None
    return s.asof(t0), s.asof(t1)


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    day_open = sp.groupby("date")["open"].first(); spot1520 = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    tdays = sorted(day_open.index)
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); first_exp = expiries[0]

    rows = []; skipped = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if pd.Timestamp(D).day_name() != "Monday": continue
        if D < first_exp or Dn > spot_end or D not in spot1520.index: continue
        j = bisect.bisect_right(expiries, D)
        if j + 1 >= len(expiries): continue                # need BOTH current and next-next to exist
        E_cur = expiries[j]; E_nxt = expiries[j + 1]
        espot = float(spot1520.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / 50) * 50
        direction = "RED" if espot < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        long_K = atm; short_K = atm - OFFSET if direction == "RED" else atm + OFFSET
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=20); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=17)

        res = {}
        for label, E in [("current_expiry", E_cur), ("next_expiry", E_nxt)]:
            folder = E.strftime("%Y%m%d")
            Le, Lx = leg_px(folder, long_K, ot, t_en, t_ex); Se, Sx = leg_px(folder, short_K, ot, t_en, t_ex)
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (Le, Lx, Se, Sx)):
                res[label] = None; continue
            entry_cost = 2 * Le - Se; exit_val = 2 * Lx - Sx; pnl = exit_val - entry_cost
            dte = (E.normalize() - pd.Timestamp(D).normalize()).days
            res[label] = {"expiry": E.date(), "DTE_at_entry": dte, "entry_cost": round(float(entry_cost), 2), "exit_value": round(float(exit_val), 2), "pnl": round(float(pnl), 2)}

        if res["current_expiry"] is None or res["next_expiry"] is None:
            skipped += 1; continue
        rows.append({"entry_date": D, "exit_date": Dn, "direction": direction, "ATM": int(atm),
                     "cur_expiry": res["current_expiry"]["expiry"], "cur_DTE": res["current_expiry"]["DTE_at_entry"],
                     "cur_entry_cost": res["current_expiry"]["entry_cost"], "cur_exit_value": res["current_expiry"]["exit_value"], "cur_pnl": res["current_expiry"]["pnl"],
                     "next_expiry": res["next_expiry"]["expiry"], "next_DTE": res["next_expiry"]["DTE_at_entry"],
                     "next_entry_cost": res["next_expiry"]["entry_cost"], "next_exit_value": res["next_expiry"]["exit_value"], "next_pnl": res["next_expiry"]["pnl"]})

    T = pd.DataFrame(rows)
    T["oos"] = pd.to_datetime(T["entry_date"]) >= pd.Timestamp("2025-08-29")

    def blk(col):
        return {"trades": len(T), "win_%": round((T[col] > 0).mean() * 100, 1), "total_pnl": round(T[col].sum(), 1),
                "avg_pnl": round(T[col].mean(), 2), "median_pnl": round(T[col].median(), 2)}
    CMP = pd.DataFrame([{"variant": "CURRENT expiry (strategy's normal choice)", **blk("cur_pnl")},
                        {"variant": "NEXT expiry (following week's contract)", **blk("next_pnl")}])

    isoos = pd.DataFrame([
        {"variant": "current_expiry", "IS_pnl": round(T[~T.oos].cur_pnl.sum(), 1), "OOS_pnl": round(T[T.oos].cur_pnl.sum(), 1)},
        {"variant": "next_expiry", "IS_pnl": round(T[~T.oos].next_pnl.sum(), 1), "OOS_pnl": round(T[T.oos].next_pnl.sum(), 1)},
    ])

    with pd.ExcelWriter(OUTDIR / "monday_current_vs_next_expiry.xlsx", engine="openpyxl") as w:
        CMP.to_excel(w, sheet_name="Comparison", index=False); isoos.to_excel(w, sheet_name="IS_OOS", index=False)
        T.to_excel(w, sheet_name="Monday_Trades", index=False)
    T.to_csv(OUTDIR / "monday_current_vs_next_expiry_trades.csv", index=False)

    pd.set_option("display.width", 230)
    print("=" * 96 + "\nNIFTY BTST — MONDAY ENTRIES: CURRENT vs NEXT EXPIRY (exit Tuesday/next trading day 09:17)\n" + "=" * 96)
    print(f"Monday trades compared: {len(T)} | skipped (missing data either side) {skipped}\n")
    print(CMP.to_string(index=False))
    print("\n--- IS/OOS ---"); print(isoos.to_string(index=False))
    print(f"\navg DTE: current={round(T.cur_DTE.mean(),1)} | next={round(T.next_DTE.mean(),1)}")
    print(f"avg entry_cost (credit basis): current={round(T.cur_entry_cost.mean(),1)} | next={round(T.next_entry_cost.mean(),1)}")
    print("\nfirst 8 rows:")
    print(T.head(8)[["entry_date", "direction", "cur_expiry", "cur_DTE", "cur_pnl", "next_expiry", "next_DTE", "next_pnl"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
