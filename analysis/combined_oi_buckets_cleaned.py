# -*- coding: utf-8 -*-
"""
combined_oi_buckets_cleaned.py
==============================
Re-run the combined-OI change bucketing EXCLUDING the trading day immediately after each
monthly expiry (post_expiry_day), whose combined-OI total is understated because new
far-month contracts aren't always written the next day -> its day-over-day change is a
rollover artifact, not a genuine OI move.

Reuses data/nifty_futures_combined_oi_price_daily.xlsx (does NOT recompute OI). Expiry
detection is DATA-DRIVEN from the near-month contract transitions (the actual expiry the
F&O data reflects), so holiday-shifted expiries are handled automatically:
  is_expiry_day    = last trading day a contract is the near (contract_expiry about to change)
  post_expiry_day  = first trading day the near contract switched  (= day after expiry)
Both flags are added to the saved series; post_expiry_day rows are dropped from bucketing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "nifty_futures_combined_oi_price_daily.xlsx"
OUT_BUCKETS = REPO / "data" / "combined_oi_change_buckets_cleaned.xlsx"

BUCKETS = [
    ("+5% to +10%",  lambda x: (x >= 5) & (x < 10)),
    ("+10% to +15%", lambda x: (x >= 10) & (x < 15)),
    ("+15% to +20%", lambda x: (x >= 15) & (x < 20)),
    (">= +20%",      lambda x: x >= 20),
    ("-5% to -10%",  lambda x: (x <= -5) & (x > -10)),
    ("-10% to -15%", lambda x: (x <= -10) & (x > -15)),
    ("-15% to -20%", lambda x: (x <= -15) & (x > -20)),
    ("<= -20%",      lambda x: x <= -20),
    ("within +/-5%", lambda x: (x > -5) & (x < 5)),
]


def bucket_table(df, denom):
    oi = df["combined_oi_change_pct"]
    rows = []
    for label, cond in BUCKETS:
        sub = df[cond(oi) & oi.notna()]
        n = len(sub)
        rows.append({"bucket": label, "count": n, "pct_of_days": round(n / denom * 100, 2),
                     "avg_oi_change_pct": round(float(sub["combined_oi_change_pct"].mean()), 4) if n else np.nan,
                     "avg_oi_change_num": round(float(sub["combined_oi_change_num"].mean()), 0) if n else np.nan,
                     "avg_close_change_pct": round(float(sub["close_change_pct"].mean()), 4) if n else np.nan})
    return pd.DataFrame(rows)


def main():
    df = pd.read_excel(SRC, sheet_name="combined_oi_price_daily")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["contract_expiry"] = pd.to_datetime(df["contract_expiry"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    # ── flags from contract transitions ──
    df["post_expiry_day"] = df["contract_expiry"].ne(df["contract_expiry"].shift(1))
    df.loc[0, "post_expiry_day"] = False                      # first obs isn't post-expiry
    df["is_expiry_day"] = df["contract_expiry"].ne(df["contract_expiry"].shift(-1))
    df.loc[df.index[-1], "is_expiry_day"] = False             # final contract still active, not expired

    # sanity: post_expiry_day should equal the existing rollover_day flag
    match = bool((df["post_expiry_day"] == df["rollover_day"]).all())

    total = len(df)
    n_post = int(df["post_expiry_day"].sum())
    n_expiry = int(df["is_expiry_day"].sum())

    # ── expiry-date detection detail (holiday-shift check) ──
    exp_days = df[df["is_expiry_day"]].copy()
    def _last_thursday(d):
        import calendar
        c = calendar.monthcalendar(d.year, d.month)
        thu = max(week[calendar.THURSDAY] for week in c if week[calendar.THURSDAY])
        return pd.Timestamp(d.year, d.month, thu).date()
    exp_days["month_last_thursday"] = exp_days["contract_expiry"].map(_last_thursday)
    exp_days["holiday_shifted"] = exp_days["contract_expiry"] != exp_days["month_last_thursday"]
    n_shift = int(exp_days["holiday_shifted"].sum())

    # ── diagnostic 1: does the expiry DAY itself look artifact-like? ──
    exp_chg = df.loc[df["is_expiry_day"], "combined_oi_change_pct"].dropna()
    exp_big = int((exp_chg.abs() >= 5).sum())

    # ── diagnostic 2: multi-day depression? change on +1 (post, excluded) vs +2 vs +3 ──
    pos_post = df.index[df["post_expiry_day"]].tolist()
    off = {1: [], 2: [], 3: []}                               # combined_oi_change_pct at day+1/+2/+3 after expiry
    for p in pos_post:                                        # p = post_expiry_day row (offset +1)
        for k, row in ((1, p), (2, p + 1), (3, p + 2)):
            if 0 <= row < total:
                v = df.loc[row, "combined_oi_change_pct"]
                if pd.notna(v):
                    off[k].append(float(v))
    traj = pd.DataFrame([{"day_after_expiry": k, "n": len(off[k]),
                          "avg_oi_change_pct": round(np.mean(off[k]), 3),
                          "pct_increase_gt5": round(np.mean([x >= 5 for x in off[k]]) * 100, 1),
                          "pct_decrease_lt_-5": round(np.mean([x <= -5 for x in off[k]]) * 100, 1)}
                         for k in (1, 2, 3)])

    # ── cleaned series + bucket tables ──
    clean = df[~df["post_expiry_day"]].copy()
    denom_clean = len(clean)
    before = bucket_table(df, total)                          # original (all days, denom = all)
    after = bucket_table(clean, denom_clean)                  # cleaned

    cmp = before[["bucket", "count"]].rename(columns={"count": "count_before"}).merge(
        after[["bucket", "count"]].rename(columns={"count": "count_after"}), on="bucket")
    cmp["removed"] = cmp["count_before"] - cmp["count_after"]

    # ── save: updated series (with flags) + cleaned buckets ──
    series_out = df[["date", "contract_expiry", "futures_open", "futures_high", "futures_low",
                     "futures_close", "close_change_pct", "combined_oi", "combined_oi_change_num",
                     "combined_oi_change_pct", "rollover_day", "is_expiry_day", "post_expiry_day"]]
    with pd.ExcelWriter(OUT_BUCKETS, engine="openpyxl") as w:
        after.to_excel(w, sheet_name="buckets_cleaned", index=False)
        cmp.to_excel(w, sheet_name="before_after", index=False)
        traj.to_excel(w, sheet_name="post_expiry_trajectory", index=False)
        exp_days[["date", "contract_expiry", "month_last_thursday", "holiday_shifted"]].to_excel(
            w, sheet_name="expiry_dates", index=False)
        series_out.to_excel(w, sheet_name="series_with_flags", index=False)

    # also persist the flags back into the main series workbook (keep flags in the saved series)
    with pd.ExcelWriter(SRC, engine="openpyxl") as w:
        series_out.to_excel(w, sheet_name="combined_oi_price_daily", index=False)
        df[df["post_expiry_day"]][["date", "contract_expiry"]].to_excel(w, sheet_name="post_expiry_dates", index=False)

    pd.set_option("display.width", 200)
    print("=" * 108)
    print("COMBINED OI buckets — CLEANED (post-expiry day excluded)")
    print("=" * 108)
    print(f"  total trading days           : {total}")
    print(f"  post_expiry_day excluded     : {n_post}   (== existing rollover_day flag? {match})")
    print(f"  cleaned trading days (denom) : {denom_clean}")
    print(f"  expiry days detected         : {n_expiry}  | holiday-shifted (not last Thursday): {n_shift}")
    print("\n--- CLEANED bucket table ---")
    print(after.to_string(index=False))
    print("\n--- BEFORE vs AFTER (post-expiry removed) ---")
    print(cmp.to_string(index=False))
    tot_inc_a = int(after[after["bucket"].str.match(r"[+>]")]["count"].sum())
    tot_dec_a = int(after[after["bucket"].str.match(r"[-<]")]["count"].sum())
    print(f"\n  CLEANED: increase>=5% = {tot_inc_a} | decrease<=5% = {tot_dec_a}  "
          f"-> {'more INCREASES' if tot_inc_a>tot_dec_a else 'more DECREASES'} (ratio {tot_inc_a}:{tot_dec_a})")
    print("\n--- DIAGNOSTIC: is the EXPIRY DAY itself artifact-like? ---")
    print(f"  expiry-day |change|>=5%: {exp_big}/{len(exp_chg)} days | avg expiry-day change {exp_chg.mean():.3f}% "
          f"| median {exp_chg.median():.3f}%")
    print("  -> " + ("expiry-day changes look mostly NORMAL (keep expiry day; exclude only day-after)"
                     if exp_big <= len(exp_chg) * 0.25 else
                     "expiry-day changes ALSO look artifact-like (consider excluding expiry day too)"))
    print("\n--- DIAGNOSTIC: multi-day post-expiry depression? (change on day +1/+2/+3 after expiry) ---")
    print(traj.to_string(index=False))
    two_day = traj.loc[traj["day_after_expiry"] == 2, "pct_increase_gt5"].iloc[0]
    print("  -> " + (f"day+2 shows large rebounds ({two_day}% are >+5%) -> OI may stay depressed >1 day; "
                     "consider a 2-day exclusion window"
                     if two_day >= 30 else
                     f"day+2 largely normal ({two_day}% >+5%) -> single-day (day-after-only) exclusion is sufficient"))
    print(f"\nSaved buckets -> {OUT_BUCKETS}\nFlags persisted into series -> {SRC}")


if __name__ == "__main__":
    main()
