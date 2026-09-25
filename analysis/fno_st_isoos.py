# -*- coding: utf-8 -*-
"""fno_st_isoos.py — IS vs OOS consistency check for the 15min & 1hr Supertrend(10,3) legs on the 204 F&O
stocks (same legs as fno_3leg_stocks). Split at 2024-07-01. GROSS and NET (15 pts/trade). Does the NIFTY
sweep's IS->OOS sign-flip also appear across the stock universe?"""
import sys, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from supertrend_dual_tf import supertrend

CK = rb.BASE / "checkpoints"; FO_CSV = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
SPLIT = pd.Timestamp("2024-07-01"); WARM = 11; COST = 15.0


def leg_isoos(direction, close, isend):
    n = len(close); dc = np.diff(close)
    g_is = float(np.sum(direction[WARM:isend - 1] * dc[WARM:isend - 1])); g_oo = float(np.sum(direction[isend:n - 1] * dc[isend:n - 1]))
    t_is = sum(1 for i in range(max(WARM, 1), isend) if direction[i] != direction[i - 1])
    t_oo = sum(1 for i in range(isend, n) if direction[i] != direction[i - 1])
    return g_is, g_oo, g_is - COST * t_is, g_oo - COST * t_oo


def worker(sym):
    try:
        d = pd.read_csv(CK / f"{sym}.csv", usecols=["timestamp", "open", "high", "low", "close"])
        ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True)
        c15 = d["close"].values; d15dir = supertrend(d["high"].values, d["low"].values, c15, 10, 3)
        ise15 = int((d["ts"] < SPLIT).sum())
        mod = d["ts"].dt.hour * 60 + d["ts"].dt.minute
        d["hb"] = pd.factorize(d["ts"].dt.strftime("%Y-%m-%d") + "_" + ((mod - 555) // 60).astype(int).astype(str))[0]
        hourly = d.groupby("hb").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"), ts=("ts", "last")).sort_index()
        d1dir = supertrend(hourly["h"].values, hourly["l"].values, hourly["c"].values, 10, 3)
        ise1h = int((hourly["ts"] < SPLIT).sum())
        g15is, g15oo, n15is, n15oo = leg_isoos(d15dir, c15, ise15)
        g1is, g1oo, n1is, n1oo = leg_isoos(d1dir, hourly["c"].values, ise1h)
        return {"symbol": sym, "st15_gross_is": g15is, "st15_gross_oos": g15oo, "st15_net_is": n15is, "st15_net_oos": n15oo,
                "st1h_gross_is": g1is, "st1h_gross_oos": g1oo, "st1h_net_is": n1is, "st1h_net_oos": n1oo}
    except Exception as e:
        return {"symbol": sym, "error": str(e)}


def main():
    syms = sorted(pd.read_csv(FO_CSV, usecols=["symbol"])["symbol"].unique())
    syms = [s for s in syms if (CK / f"{s}.csv").exists()]
    res = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for f in as_completed([ex.submit(worker, s) for s in syms]):
            res.append(f.result())
    ok = [r for r in res if "error" not in r]; T = pd.DataFrame(ok)
    IS_Y = (SPLIT - pd.Timestamp("2022-01-03")).days / 365.25; OO_Y = (pd.Timestamp("2026-07-31") - SPLIT).days / 365.25

    def report(pref, label):
        cis, coo = f"{pref}_is", f"{pref}_oos"
        mis, moo = T[cis].mean(), T[coo].mean()
        both_pos = ((T[cis] > 0) & (T[coo] > 0)).mean() * 100
        is_pos = (T[cis] > 0).mean() * 100; oo_pos = (T[coo] > 0).mean() * 100
        # sign consistency: same sign in both halves
        consist = (np.sign(T[cis]) == np.sign(T[coo])).mean() * 100
        return {"leg/metric": label, "mean_IS": round(mis, 0), "mean_OOS": round(moo, 0),
                "mean_IS/yr": round(mis / IS_Y, 0), "mean_OOS/yr": round(moo / OO_Y, 0),
                "%stocks_IS+": round(is_pos, 1), "%stocks_OOS+": round(oo_pos, 1),
                "%both_positive": round(both_pos, 1), "%same_sign": round(consist, 1)}

    R = pd.DataFrame([report("st15_gross", "15min ST GROSS"), report("st15_net", "15min ST NET(-15/tr)"),
                      report("st1h_gross", "1hr ST GROSS"), report("st1h_net", "1hr ST NET(-15/tr)")])
    OUT = rb.RESULTS / "fno_3leg_stocks"; OUT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT / "fno_st_isoos.xlsx", engine="openpyxl") as w:
        R.to_excel(w, sheet_name="IS_OOS_Consistency", index=False)
        T.round(0).to_excel(w, sheet_name="Per_Stock", index=False)
    pd.set_option("display.width", 220)
    print(f"F&O stocks {len(ok)} | split {SPLIT.date()} | IS {IS_Y:.2f}y OOS {OO_Y:.2f}y | points per stock (mean across stocks)")
    print("\n=== Supertrend (10,3) legs on F&O stocks — IS vs OOS ===")
    print(R.to_string(index=False))
    print(f"\nSaved -> {OUT/'fno_st_isoos.xlsx'}")


if __name__ == "__main__":
    main()
