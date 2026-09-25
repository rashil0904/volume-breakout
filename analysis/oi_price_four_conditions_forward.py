# -*- coding: utf-8 -*-
"""
oi_price_four_conditions_forward.py
===================================
All four OI x price conditions, each with forward Nifty-futures directional returns T+1..T+5
(cumulative from signal close), in ONE table. Cleaned series (post_expiry signal days excluded).

Conditions (OI floor +/-3%, price floor +/-1%):
  long_buildup   : OI up   & price up   -> LONG   (positive = up-move followed)
  short_buildup  : OI up   & price down -> SHORT  (positive = down-move followed)
  short_covering : OI down & price up   -> LONG
  long_unwinding : OI down & price down -> SHORT
Directional return: long = (close[D+N]-close[D])/close[D]*100 ; short = -that.
Clean = horizons whose forward window does NOT cross a contract rollover (artifact-free).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"
OUT = REPO / "data" / "oi_price_four_conditions_forward.xlsx"
H = [1, 2, 3, 4, 5]

CONDITIONS = [
    ("long_buildup",   "LONG",  lambda d: (d["combined_oi_change_pct"] >= 3) & (d["close_change_pct"] >= 1)),
    ("short_buildup",  "SHORT", lambda d: (d["combined_oi_change_pct"] >= 3) & (d["close_change_pct"] <= -1)),
    ("short_covering", "LONG",  lambda d: (d["combined_oi_change_pct"] <= -3) & (d["close_change_pct"] >= 1)),
    ("long_unwinding", "SHORT", lambda d: (d["combined_oi_change_pct"] <= -3) & (d["close_change_pct"] <= -1)),
]


def main():
    df = pd.read_excel(SRC, sheet_name="combined_oi_price_daily").sort_values("date").reset_index(drop=True)
    n = len(df); close = df["futures_close"].values.astype(float); pe = df["post_expiry_day"].astype(bool).values
    for N in H:
        fwd = np.r_[close[N:], np.full(N, np.nan)]
        df[f"long_T{N}"] = (fwd - close) / close * 100                 # long-convention forward return
        cross = np.zeros(n, bool)
        for k in range(1, N + 1):
            cross |= np.r_[pe[k:], np.zeros(k, bool)]
        df[f"cross_T{N}"] = cross

    base = df[~df["post_expiry_day"]]
    rows = []
    for name, direction, cond in CONDITIONS:
        sig = base[cond(base)]
        nq = len(sig)
        for N in H:
            r = sig[f"long_T{N}"].copy()
            if direction == "SHORT":
                r = -r
            all_v = r.dropna()
            cl = r[~sig[f"cross_T{N}"]].dropna()
            rows.append({
                "condition": name, "direction": direction, "n_signal_days": nq, "horizon": f"T+{N}",
                "avg_pct": round(float(all_v.mean()), 4) if len(all_v) else np.nan,
                "median_pct": round(float(all_v.median()), 4) if len(all_v) else np.nan,
                "win_pct": round(float((all_v > 0).mean() * 100), 1) if len(all_v) else np.nan,
                "n": len(all_v),
                "avg_pct_clean": round(float(cl.mean()), 4) if len(cl) else np.nan,
                "median_pct_clean": round(float(cl.median()), 4) if len(cl) else np.nan,
                "win_pct_clean": round(float((cl > 0).mean() * 100), 1) if len(cl) else np.nan,
                "n_clean": len(cl),
            })
    tab = pd.DataFrame(rows)

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        tab.to_excel(w, sheet_name="four_conditions_forward", index=False)

    pd.set_option("display.width", 240)
    print("ALL FOUR CONDITIONS — forward Nifty futures directional return (cumulative from signal close)")
    print("positive = the condition's implied direction paid off | clean = rollover-free windows\n")
    for name, direction, _ in CONDITIONS:
        sub = tab[tab["condition"] == name]
        nq = int(sub["n_signal_days"].iloc[0])
        print(f"=== {name.upper()} ({direction}) — {nq} signal days ===")
        print(sub[["horizon", "avg_pct", "median_pct", "win_pct", "n",
                   "avg_pct_clean", "median_pct_clean", "win_pct_clean", "n_clean"]].to_string(index=False))
        print()
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
