# -*- coding: utf-8 -*-
"""breakout_margin_compare.py — NIFTY daily 2-day breakout + gap: Variant A (10-pt buffer, baseline) vs
Variant B (0-pt, exact-touch). ONLY the breakout margin differs; core logic, gap handling (RAW level, same
for both), first-trade rule all identical. GROSS index points. Reports side-by-side + divergence days."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DATA = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "breakout_margin_compare"; OUTDIR.mkdir(parents=True, exist_ok=True)
FIRST15_END = 570


def walk(dm, margin):
    """gap-breakout walk with a given breakout margin (gap always uses RAW level). returns trades, sig, gap_held."""
    up_a = (dm["p2h"] + margin).values; dn_a = (dm["p2l"] - margin).values
    dt_a = dm["date"].values; mod_a = dm["mod"].values; hi_a = dm["high"].values; lo_a = dm["low"].values; op_a = dm["open"].values; tsr = dm["ts"].values
    p2h_a, p2l_a, do_a, fh_a, fl_a = (dm[c].values for c in ["p2h", "p2l", "dopen", "fh15", "fl15"])
    pos = 0; entry = np.nan; entry_t = None; cur_type = None; trades = []; sig = 0
    cur_day = None; gap = None; gap_flipped = False; gap_held = []
    for k in range(len(dm)):
        if dt_a[k] != cur_day:
            if gap in ("down", "up") and not gap_flipped:
                gap_held.append(pd.Timestamp(cur_day).date())
            cur_day = dt_a[k]; gap = None; gap_flipped = False
            if not np.isnan(p2l_a[k]):
                if pos == 1 and do_a[k] < p2l_a[k]: gap = "down"
                elif pos == -1 and do_a[k] > p2h_a[k]: gap = "up"
        up, dn, h, l, o, t, m = up_a[k], dn_a[k], hi_a[k], lo_a[k], op_a[k], tsr[k], mod_a[k]
        if np.isnan(up):
            continue
        if gap in ("down", "up") and not gap_flipped:
            if m < FIRST15_END:
                continue
            if gap == "down" and pos == 1 and l <= fl_a[k]:
                trades.append(("Long", entry_t, entry, t, fl_a[k], fl_a[k] - entry, cur_type)); pos, entry, entry_t, cur_type = -1, fl_a[k], t, "gap"; gap = None; gap_flipped = True; sig += 1
            elif gap == "up" and pos == -1 and h >= fh_a[k]:
                trades.append(("Short", entry_t, entry, t, fh_a[k], entry - fh_a[k], cur_type)); pos, entry, entry_t, cur_type = 1, fh_a[k], t, "gap"; gap = None; gap_flipped = True; sig += 1
            continue
        if pos == 0:
            hu, hd = h >= up, l <= dn
            if hu and hd:
                pos, entry = (1, up) if abs(o - up) <= abs(o - dn) else (-1, dn); entry_t, cur_type = t, "normal"; sig += 1
            elif hu: pos, entry, entry_t, cur_type = 1, up, t, "normal"; sig += 1
            elif hd: pos, entry, entry_t, cur_type = -1, dn, t, "normal"; sig += 1
        elif pos == 1:
            if l <= dn:
                trades.append(("Long", entry_t, entry, t, dn, dn - entry, cur_type)); pos, entry, entry_t, cur_type = -1, dn, t, "normal"; sig += 1
        else:
            if h >= up:
                trades.append(("Short", entry_t, entry, t, up, entry - up, cur_type)); pos, entry, entry_t, cur_type = 1, up, t, "normal"; sig += 1
    if gap in ("down", "up") and not gap_flipped:
        gap_held.append(pd.Timestamp(cur_day).date())
    T = pd.DataFrame(trades, columns=["direction", "entry_time", "entry_price", "exit_time", "exit_price", "points_pnl", "entry_type"])
    T["hold_days"] = (pd.to_datetime(T["exit_time"]).dt.normalize() - pd.to_datetime(T["entry_time"]).dt.normalize()).dt.days
    return T, sig, gap_held


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    d["date"] = d["ts"].dt.normalize(); d["mod"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    daily = d.groupby("date").agg(dh=("high", "max"), dl=("low", "min"))
    daily["p2h"] = daily["dh"].shift(1).rolling(2).max(); daily["p2l"] = daily["dl"].shift(1).rolling(2).min()
    f15 = d[(d["mod"] >= 555) & (d["mod"] < FIRST15_END)].groupby("date").agg(fh15=("high", "max"), fl15=("low", "min"))
    dopen = d.groupby("date")["open"].first().rename("dopen")
    daily = daily.join(f15).join(dopen)
    dm = d.merge(daily[["p2h", "p2l", "fh15", "fl15", "dopen"]], left_on="date", right_index=True, how="left")
    period = f"{daily.index[0].date()} .. {daily.index[-1].date()}"
    span = (daily.index[-1] - daily.index[0]).days; weeks = span / 7; months = span / 30.44

    TA, sigA, ghA = walk(dm, 10.0); TB, sigB, ghB = walk(dm, 0.0)

    def stat(T, sig, gh, name):
        return {"variant": name, "trades": len(T), "win_%": round((T.points_pnl > 0).mean() * 100, 1),
                "total_pts": round(T.points_pnl.sum(), 1), "avg_pts": round(T.points_pnl.mean(), 2),
                "avg_hold_days": round(T.hold_days.mean(), 2), "signals/wk": round(sig / weeks, 2), "signals/mo": round(sig / months, 2),
                "gap_trades": int((T.entry_type == "gap").sum()), "normal_trades": int((T.entry_type == "normal").sum()), "gap_held_days": len(gh)}
    cmp = pd.DataFrame([stat(TA, sigA, ghA, "A: +10 buffer (baseline)"), stat(TB, sigB, ghB, "B: 0 buffer (exact touch)")])

    # ---- day-level divergence: days where B has a touch that A does NOT (in isolation) ----
    dd = daily.dropna(subset=["p2h", "p2l"]).copy()
    dd["B_only_long"] = (dd["dh"] >= dd["p2h"]) & (dd["dh"] < dd["p2h"] + 10)     # touches raw high but not +10
    dd["B_only_short"] = (dd["dl"] <= dd["p2l"]) & (dd["dl"] > dd["p2l"] - 10)     # touches raw low but not -10
    div = dd[dd["B_only_long"] | dd["B_only_short"]]
    divsheet = div.reset_index()[["date", "dh", "dl", "p2h", "p2l", "B_only_long", "B_only_short"]].round(1)

    with pd.ExcelWriter(OUTDIR / "breakout_margin_compare.xlsx", engine="openpyxl") as w:
        cmp.to_excel(w, sheet_name="A_vs_B", index=False)
        pd.DataFrame([
            {"metric": "Strategy", "value": "NIFTY daily 2-day breakout + gap; A=+10 buffer vs B=0 buffer (only margin differs)"},
            {"metric": "Period", "value": period}, {"metric": "P&L", "value": "GROSS index points (no cost); gap logic identical (RAW level) for both"},
            {"metric": "Divergence days (B touches, A doesn't)", "value": len(div)},
            {"metric": "  B-only LONG touches", "value": int(div["B_only_long"].sum())},
            {"metric": "  B-only SHORT touches", "value": int(div["B_only_short"].sum())},
            {"metric": "Trade-count check", "value": f"B {len(TB)} vs A {len(TA)} -> B {'MORE' if len(TB)>len(TA) else 'NOT more'} ({round(len(TB)/len(TA),2)}x)"},
            {"metric": "Return check", "value": f"B {round(TB.points_pnl.sum(),1)} vs A {round(TA.points_pnl.sum(),1)} -> extra trades {'ADD' if TB.points_pnl.sum()>TA.points_pnl.sum() else 'SUBTRACT'}"},
        ]).to_excel(w, sheet_name="Summary", index=False)
        divsheet.to_excel(w, sheet_name="Divergence_Days", index=False)
        TA.assign(entry_time=pd.to_datetime(TA.entry_time).dt.strftime("%Y-%m-%d %H:%M"), exit_time=pd.to_datetime(TA.exit_time).dt.strftime("%Y-%m-%d %H:%M")).to_excel(w, sheet_name="A_Trades", index=False)
        TB.assign(entry_time=pd.to_datetime(TB.entry_time).dt.strftime("%Y-%m-%d %H:%M"), exit_time=pd.to_datetime(TB.exit_time).dt.strftime("%Y-%m-%d %H:%M")).to_excel(w, sheet_name="B_Trades", index=False)

    pd.set_option("display.width", 200)
    print("=" * 96 + "\nNIFTY DAILY 2-DAY BREAKOUT + GAP — MARGIN COMPARISON (A: +10 buffer vs B: 0 buffer)\n" + "=" * 96)
    print(f"period {period} | GROSS index points | gap logic identical (RAW level)\n")
    print(cmp.to_string(index=False))
    print(f"\ndivergence days (B has a touch A doesn't): {len(div)} ({int(div['B_only_long'].sum())} long, {int(div['B_only_short'].sum())} short)")
    print(f"trade-count: B {len(TB)} vs A {len(TA)} = {round(len(TB)/len(TA),2)}x  ->  B {'MORE (expected)' if len(TB)>len(TA) else 'NOT more'}")
    dlt = round(TB.points_pnl.sum() - TA.points_pnl.sum(), 1)
    print(f"return delta B-A: {dlt} pts  ->  the extra B trades {'ADD to' if dlt>0 else 'SUBTRACT from'} total return")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
