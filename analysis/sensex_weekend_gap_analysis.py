# -*- coding: utf-8 -*-
"""sensex_weekend_gap_analysis.py — SENSEX weekend gap analysis: for every calendar week transition, gap =
(first REGULAR Mon-Fri trading day of the NEXT week)'s OPEN - (last REGULAR Mon-Fri trading day of the
CURRENT week)'s CLOSE. Holiday-adjusted per week (last day of week = Fri unless holiday -> Thu etc.; first
day of next week = Mon unless holiday -> Tue etc.) via the ACTUAL trading calendar (no fixed weekday assumed).
Special weekend sessions (Union Budget Day, which can fall on Sat/Sun) are EXCLUDED from serving as reference
days (they are not part of the regular weekday last-close/first-open boundary) and reported separately.
Classify: gap-up >=+1000, gap-down <=-1000, else neither. Scope = full available data in
data/sensex_1min_ohlc.csv (2024-10-01..2026-07-31 - flagged as the actual available range, shorter than
full SENSEX history, per user instruction not to pull further).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "sensex_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "sensex_weekend_gap"; OUTDIR.mkdir(parents=True, exist_ok=True)
THRESH = 1000.0


def main():
    d = pd.read_csv(SPOT)
    ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d["date"] = ts.dt.normalize()
    daily = d.groupby("date").agg(open=("open", "first"), close=("close", "last")).reset_index()
    daily["dow"] = daily["date"].dt.day_name()
    daily["iso_year"] = daily["date"].dt.isocalendar().year
    daily["iso_week"] = daily["date"].dt.isocalendar().week

    special = daily[daily.dow.isin(["Saturday", "Sunday"])].copy()   # Budget-day special sessions etc.
    reg = daily[~daily.dow.isin(["Saturday", "Sunday"])].copy()       # regular Mon-Fri trading days only

    # last regular trading day per week, first regular trading day per week
    wk = reg.groupby(["iso_year", "iso_week"]).agg(last_date=("date", "max"), first_date=("date", "min")).reset_index()
    wk = wk.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
    close_by_date = reg.set_index("date")["close"]; open_by_date = reg.set_index("date")["open"]
    dow_by_date = reg.set_index("date")["dow"]

    rows = []; flags = []
    for i in range(len(wk) - 1):
        Wc = wk.iloc[i]; Wn = wk.iloc[i + 1]
        last_d = Wc["last_date"]; first_d = Wn["first_date"]
        # sanity: weeks should be consecutive (iso_week+1, or year rollover); flag any unexpected multi-week jump
        exp_next = (Wc["iso_year"], Wc["iso_week"] + 1) if Wc["iso_week"] < 52 else (Wc["iso_year"] + 1, 1)
        wk_gap_flag = (Wn["iso_year"], Wn["iso_week"]) != exp_next and not (Wc["iso_week"] >= 52 and Wn["iso_week"] == 1)
        c = float(close_by_date.get(last_d, np.nan)); o = float(open_by_date.get(first_d, np.nan))
        if pd.isna(c) or pd.isna(o):
            flags.append({"week_last": last_d, "week_first": first_d, "reason": "missing close/open on boundary day"}); continue
        gap = o - c; pct = round(gap / c * 100, 3)
        cls = "gap-up" if gap >= THRESH else ("gap-down" if gap <= -THRESH else "neither")
        rows.append({"last_close_date": last_d.date(), "last_close_dow": dow_by_date[last_d], "last_close": round(c, 2),
                     "weekend_open_date": first_d.date(), "weekend_open_dow": dow_by_date[first_d], "weekend_open": round(o, 2),
                     "gap_points": round(gap, 2), "gap_pct": pct, "classification": cls,
                     "days_between": (first_d - last_d).days})
        if wk_gap_flag:
            flags.append({"week_last": last_d, "week_first": first_d, "reason": "non-consecutive ISO week transition (possible longer data gap)"})

    G = pd.DataFrame(rows)
    n_total = len(G); n_up = int((G.classification == "gap-up").sum()); n_dn = int((G.classification == "gap-down").sum()); n_neither = n_total - n_up - n_dn
    up = G[G.classification == "gap-up"].sort_values("gap_points", ascending=False)
    dn = G[G.classification == "gap-down"].sort_values("gap_points")

    summary = pd.DataFrame([
        {"metric": "Data source", "value": "data/sensex_1min_ohlc.csv"},
        {"metric": "ACTUAL available range (flagged - not full SENSEX history)", "value": f"{daily.date.min().date()} .. {daily.date.max().date()} ({len(daily)} trading days incl. special sessions)"},
        {"metric": "Reference-day rule", "value": "last close = last REGULAR (Mon-Fri) trading day of the week (Fri unless holiday->Thu etc.); weekend open = first REGULAR trading day of next week (Mon unless holiday->Tue etc.)"},
        {"metric": "Special weekend sessions EXCLUDED from reference-day role", "value": f"{len(special)}: " + "; ".join(f"{r.date.date()} ({r.dow}, Union Budget Day)" for _, r in special.iterrows())},
        {"metric": "Threshold", "value": f"gap-up >= +{int(THRESH)} pts ; gap-down <= -{int(THRESH)} pts"},
        {"metric": "Total weeks analyzed", "value": n_total},
        {"metric": "Gap-up weeks (>= +1000)", "value": n_up},
        {"metric": "Gap-down weeks (<= -1000)", "value": n_dn},
        {"metric": "Normal weeks (within +/-1000)", "value": n_neither},
        {"metric": "Largest gap-up (points)", "value": round(G.gap_points.max(), 2) if n_total else 0},
        {"metric": "Largest gap-down (points)", "value": round(G.gap_points.min(), 2) if n_total else 0},
        {"metric": "Average ABSOLUTE gap size (all weeks, points)", "value": round(G.gap_points.abs().mean(), 2) if n_total else 0},
        {"metric": "Average ABSOLUTE gap size (% terms)", "value": round(G.gap_pct.abs().mean(), 3) if n_total else 0},
        {"metric": "Median gap (points)", "value": round(G.gap_points.median(), 2) if n_total else 0},
        {"metric": "Incomplete/flagged weeks (excluded, logged not silently skipped)", "value": len(flags)},
    ])

    with pd.ExcelWriter(OUTDIR / "sensex_weekend_gap_analysis.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        up.to_excel(w, sheet_name="GapUp_weeks", index=False)
        dn.to_excel(w, sheet_name="GapDown_weeks", index=False)
        G.to_excel(w, sheet_name="All_Weeks", index=False)
        special.assign(note="Union Budget Day special session - excluded from ref-day role").to_excel(w, sheet_name="Special_Sessions", index=False)
        (pd.DataFrame(flags) if flags else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="Flagged_Incomplete", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nSENSEX WEEKEND GAP ANALYSIS\n" + "=" * 96)
    print(f"ACTUAL data range: {daily.date.min().date()} .. {daily.date.max().date()} ({len(daily)} trading days)")
    print(f"weeks analyzed: {n_total} | gap-up: {n_up} | gap-down: {n_dn} | neither: {n_neither}")
    print(f"largest gap-up: {round(G.gap_points.max(),2) if n_total else 0} | largest gap-down: {round(G.gap_points.min(),2) if n_total else 0}")
    print(f"avg |gap|: {round(G.gap_points.abs().mean(),2) if n_total else 0} pts ({round(G.gap_pct.abs().mean(),3) if n_total else 0}%) | median gap: {round(G.gap_points.median(),2) if n_total else 0}")
    print(f"flagged/incomplete weeks: {len(flags)}")
    if len(flags): print(pd.DataFrame(flags).to_string(index=False))
    print("\n--- GAP-UP weeks ---"); print(up.to_string(index=False) if len(up) else "  NONE")
    print("\n--- GAP-DOWN weeks ---"); print(dn.to_string(index=False) if len(dn) else "  NONE")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
