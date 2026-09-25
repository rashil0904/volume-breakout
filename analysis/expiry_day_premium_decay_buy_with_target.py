# -*- coding: utf-8 -*-
"""expiry_day_premium_decay_buy_with_target.py — extends the premium-decay-buy CAS study (2-minute
15:20-15:21 entry window, the config already established as the better one) with a TARGET exit overlay
for the bought legs: target = entry_price * multiple, for multiple in 2..10. If the option's HIGH touches
the target level any time between entry and end-of-day before settlement, exit there (limit-style, exactly
at target); otherwise hold to expiry/settlement as before (unchanged fallback). Read-only diagnostic, reuses
already-computed triggered entries -- no changes to the original study's files.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "expiry_day_premium_decay_buy" / "expiry_day_premium_decay_buy_window_comparison.xlsx"
OUTDIR = rb.RESULTS / "expiry_day_premium_decay_buy"; OUTDIR.mkdir(parents=True, exist_ok=True)
TARGET_MULTIPLES = list(range(2, 11))
OPT_DIRS = {"NIFTY": rb.BASE / "data" / "options_intraday_full" / "NIFTY",
            "SENSEX": rb.BASE / "data" / "options_intraday_full" / "SENSEX"}


def find_file(index_name, expiry, strike, otype):
    fs = glob.glob(str(OPT_DIRS[index_name] / str(expiry) / f"{index_name}_{int(strike)}_{otype}_*.parquet"))
    return fs[0] if fs else None


def main():
    T = pd.read_excel(SRC, sheet_name="Triggered_Entries_Only")
    T = T[T["window"] == "2min_1520to1521"].reset_index(drop=True)
    print(f"triggered entries (2-min window): {len(T)}", flush=True)

    rows = []
    for _, r in T.iterrows():
        fn = find_file(r["index"], r["expiry"], r["strike"], r["type"])
        if fn is None:
            print(f"  MISSING contract file: {r['index']} {r['expiry']} {r['strike']} {r['type']}"); continue
        df = pd.read_parquet(fn, columns=["timestamp", "high", "close"])
        df = df[df["timestamp"].dt.normalize() == pd.Timestamp(r["entry_time"]).normalize()].sort_values("timestamp")
        window = df[df["timestamp"] > r["entry_time"]]

        for mult in TARGET_MULTIPLES:
            target_px = r["entry_price"] * mult
            hit = window[window["high"] >= target_px]
            if len(hit):
                exit_price = target_px; exit_time = hit["timestamp"].iloc[0]; hit_target = True
            else:
                exit_price = r["settlement"]; exit_time = r["settlement_time"]; hit_target = False
            pnl = exit_price - r["entry_price"]
            rows.append({"index": r["index"], "expiry": r["expiry"], "strike": r["strike"], "type": r["type"],
                         "tag": r["tag"], "threshold_pct": r["threshold_pct"], "entry_price": r["entry_price"],
                         "target_multiple": mult, "target_price": round(target_px, 2), "hit_target": hit_target,
                         "exit_price": round(exit_price, 2), "exit_time": exit_time, "pnl": round(pnl, 2)})

    R = pd.DataFrame(rows)

    # ---- summary per index, threshold, target multiple ----
    SUMM = R.groupby(["index", "threshold_pct", "target_multiple"]).agg(
        n=("pnl", "size"), n_hit_target=("hit_target", "sum"), total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"), win_rate=("pnl", lambda s: (s > 0).mean() * 100)).reset_index()
    SUMM["total_pnl"] = SUMM["total_pnl"].round(2); SUMM["avg_pnl"] = SUMM["avg_pnl"].round(2); SUMM["win_rate"] = SUMM["win_rate"].round(1)
    SUMM["pct_hit_target"] = (SUMM["n_hit_target"] / SUMM["n"] * 100).round(1)

    # ---- also the NO-TARGET baseline (hold to expiry) for comparison ----
    BASE = T.groupby(["index", "threshold_pct"]).agg(n=("pnl", "size"), total_pnl=("pnl", "sum"), avg_pnl=("pnl", "mean"),
                                                       win_rate=("pnl", lambda s: (s > 0).mean() * 100)).reset_index()
    BASE["total_pnl"] = BASE["total_pnl"].round(2); BASE["avg_pnl"] = BASE["avg_pnl"].round(2); BASE["win_rate"] = BASE["win_rate"].round(1)

    pd.set_option("display.width", 200)
    for idx in ["NIFTY", "SENSEX"]:
        print(f"\n=== {idx} -- baseline (no target, hold to expiry) ===")
        print(BASE[BASE["index"] == idx].drop(columns="index").to_string(index=False))
        print(f"\n=== {idx} -- with target overlay (by threshold x target multiple) ===")
        print(SUMM[SUMM["index"] == idx].drop(columns="index").to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "expiry_day_premium_decay_buy_with_target.xlsx", engine="openpyxl") as w:
        BASE.to_excel(w, sheet_name="Baseline_No_Target", index=False)
        SUMM.to_excel(w, sheet_name="Summary_by_Target", index=False)
        R.to_excel(w, sheet_name="Trade_Level_Detail", index=False)
    print(f"\nSaved -> {OUTDIR}/expiry_day_premium_decay_buy_with_target.xlsx")


if __name__ == "__main__":
    main()
