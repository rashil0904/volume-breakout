# -*- coding: utf-8 -*-
"""
oi_price_four_quadrants.py
==========================
Finer-grid (3% steps) combined-OI x price cross-tab as FOUR separate quadrant tables, each
with two DIRECTIONAL next-day futures forward-return measures + win rates. Reuses the cleaned
combined-OI series (post_expiry excluded for SIGNAL days) from
data/nifty_futures_combined_oi_price_daily.xlsx. Near-month futures OHLC.

OI buckets (3%..15%): inc +3-6/+6-9/+9-12/+12-15/>=+15 ; dec -3-6/-6-9/-9-12/-12-15/<=-15.
Price dir (same-day close_change_pct): up>=+1, down<=-1, flat (between) EXCLUDED from quadrants.

Quadrants (direction = sign convention; +ve return = that direction paid off):
  1 LONG BUILDUP   : OI up  & price up   -> LONG
  2 SHORT BUILDUP  : OI up  & price down -> SHORT
  3 SHORT COVERING : OI down & price up  -> LONG
  4 LONG UNWINDING : OI down & price down-> SHORT
Returns on NEXT trading day:
  ret_close_to_next_close : signal close -> next close   (rollover-excluded when next day is post_expiry)
  ret_nextopen_to_nextclose: next open  -> next close    (clean single-contract session return)
long: (b-a)/a*100 ; short: (a-b)/a*100.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"
OUT = REPO / "data" / "oi_price_four_quadrants.xlsx"

INC = [("+3-6%", lambda x: (x >= 3) & (x < 6)), ("+6-9%", lambda x: (x >= 6) & (x < 9)),
       ("+9-12%", lambda x: (x >= 9) & (x < 12)), ("+12-15%", lambda x: (x >= 12) & (x < 15)),
       (">= +15%", lambda x: x >= 15)]
DEC = [("-3-6%", lambda x: (x <= -3) & (x > -6)), ("-6-9%", lambda x: (x <= -6) & (x > -9)),
       ("-9-12%", lambda x: (x <= -9) & (x > -12)), ("-12-15%", lambda x: (x <= -12) & (x > -15)),
       ("<= -15%", lambda x: x <= -15)]


def cell(sub, direction):
    n = len(sub)
    cc = sub["fwd_cc_long"].copy(); oc = sub["fwd_oc_long"].copy()
    if direction == "short":
        cc, oc = -cc, -oc
    ccv = cc.dropna(); ocv = oc.dropna()
    return {
        "n_days": n,
        "avg_oi_change_pct": round(float(sub["combined_oi_change_pct"].mean()), 4) if n else np.nan,
        "avg_oi_change_num": round(float(sub["combined_oi_change_num"].mean()), 0) if n else np.nan,
        "avg_close_change_pct": round(float(sub["close_change_pct"].mean()), 4) if n else np.nan,
        "ret_close_to_next_close": round(float(ccv.mean()), 4) if len(ccv) else np.nan,
        "winrate_close_to_next_close": round(float((ccv > 0).mean() * 100), 1) if len(ccv) else np.nan,
        "n_cc_pairs": len(ccv), "n_cc_excl_rollover": n - len(ccv),
        "ret_nextopen_to_nextclose": round(float(ocv.mean()), 4) if len(ocv) else np.nan,
        "winrate_nextopen_to_nextclose": round(float((ocv > 0).mean() * 100), 1) if len(ocv) else np.nan,
    }


def build_table(sig, buckets, price_dir, direction):
    rows = []
    for label, cond in buckets:
        sub = sig[cond(sig["combined_oi_change_pct"]) & (sig["price_dir"] == price_dir)]
        rows.append({"oi_bucket": label, **cell(sub, direction)})
    return pd.DataFrame(rows)


def flat_counts(sig, buckets):
    rows = []
    for label, cond in buckets:
        b = sig[cond(sig["combined_oi_change_pct"])]
        rows.append({"oi_bucket": label, "total_in_bucket": len(b),
                     "n_up": int((b["price_dir"] == "up").sum()),
                     "n_down": int((b["price_dir"] == "down").sum()),
                     "n_flat_excluded": int((b["price_dir"] == "flat").sum())})
    return pd.DataFrame(rows)


def main():
    df = pd.read_excel(SRC, sheet_name="combined_oi_price_daily").sort_values("date").reset_index(drop=True)
    # forward values from FULL series (real next trading day, even if it's a post_expiry day)
    df["next_close"] = df["futures_close"].shift(-1)
    df["next_open"] = df["futures_open"].shift(-1)
    df["next_is_post_expiry"] = df["post_expiry_day"].shift(-1).fillna(False)
    # long-convention forward returns
    df["fwd_cc_long"] = np.where(df["next_is_post_expiry"], np.nan,               # exclude contract-switch pairs
                                 (df["next_close"] - df["futures_close"]) / df["futures_close"] * 100)
    df["fwd_oc_long"] = (df["next_close"] - df["next_open"]) / df["next_open"] * 100   # clean, single contract

    cc = df["close_change_pct"]
    df["price_dir"] = np.where(cc >= 1, "up", np.where(cc <= -1, "down", "flat"))

    sig = df[~df["post_expiry_day"] & df["combined_oi_change_pct"].notna()].copy()   # cleaned signal days

    t1 = build_table(sig, INC, "up", "long")     # long buildup
    t2 = build_table(sig, INC, "down", "short")  # short buildup
    t3 = build_table(sig, DEC, "up", "long")     # short covering
    t4 = build_table(sig, DEC, "down", "short")  # long unwinding
    flat_inc = flat_counts(sig, INC); flat_dec = flat_counts(sig, DEC)

    def pooled(t):
        tot_n = t["n_days"].sum()
        # weighted avg of ret by n_cc_pairs / n
        w_cc = (t["ret_close_to_next_close"] * t["n_cc_pairs"]).sum() / max(t["n_cc_pairs"].sum(), 1)
        w_oc = (t["ret_nextopen_to_nextclose"] * t["n_days"]).sum() / max(tot_n, 1)
        return tot_n, round(w_cc, 4), round(w_oc, 4)

    quads = [("LONG BUILDUP (OI up & price up -> LONG)", t1),
             ("SHORT BUILDUP (OI up & price down -> SHORT)", t2),
             ("SHORT COVERING (OI down & price up -> LONG)", t3),
             ("LONG UNWINDING (OI down & price down -> SHORT)", t4)]
    srows = []
    for name, t in quads:
        n, wcc, woc = pooled(t)
        srows.append({"quadrant": name, "total_days": int(n),
                      "pooled_ret_close_to_next_close": wcc, "pooled_ret_nextopen_to_nextclose": woc})
    summ = pd.DataFrame(srows)

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        t1.to_excel(w, sheet_name="1_long_buildup", index=False)
        t2.to_excel(w, sheet_name="2_short_buildup", index=False)
        t3.to_excel(w, sheet_name="3_short_covering", index=False)
        t4.to_excel(w, sheet_name="4_long_unwinding", index=False)
        summ.to_excel(w, sheet_name="quadrant_summary", index=False)
        flat_inc.to_excel(w, sheet_name="flat_counts_increases", index=False)
        flat_dec.to_excel(w, sheet_name="flat_counts_decreases", index=False)

    pd.set_option("display.width", 240)
    show = ["oi_bucket", "n_days", "avg_oi_change_pct", "avg_oi_change_num", "avg_close_change_pct",
            "ret_close_to_next_close", "winrate_close_to_next_close", "n_cc_excl_rollover",
            "ret_nextopen_to_nextclose", "winrate_nextopen_to_nextclose"]
    for name, t in quads:
        print("\n" + "=" * 150 + f"\nTABLE — {name}\n" + "=" * 150)
        print(t[show].to_string(index=False))
    n_roll = int(df.loc[~df["post_expiry_day"], "next_is_post_expiry"].sum())
    print(f"\n[rollover] forward close-to-next-close pairs excluded (next day = post_expiry): {n_roll} signal days "
          f"(their nextopen->nextclose is still used — single-contract, clean)")
    print("\n--- FLAT-price days excluded per OI bucket (increases) ---\n" + flat_inc.to_string(index=False))
    print("\n--- FLAT-price days excluded per OI bucket (decreases) ---\n" + flat_dec.to_string(index=False))
    print("\n--- QUADRANT SUMMARY (pooled directional forward returns; +ve = direction paid off) ---")
    print(summ.to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
