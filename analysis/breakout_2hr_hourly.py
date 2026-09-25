# -*- coding: utf-8 -*-
"""breakout_2hr_hourly.py — HOURLY-timeframe version of the NIFTY 2-candle high/low breakout + gap handling
(daily version = breakout_2day_gap.py; NOT overwritten). Same core logic, translated to hourly bars.
STANDALONE — could later become an extra leg in the multi-TF system (like the daily leg alongside 15m/1h ST).

LEVELS: rolling previous_2hr_high/low = max/min of the last 2 COMPLETED hourly candles (window rolls every
hour, continuous across days). up = p2h+10, down = p2l-10 (same flat 10-pt margin as daily, NOT scaled).
BREAKOUT: intraday 1-min touch within the forming hourly candle -> high>=up long / low<=down short; flip at
the exact trigger; always-in (+1/-1). GAP (only at day's FIRST hourly candle): if day-open gaps beyond the
RAW prev-2hr level (= prior day's last 2 hourly candles, no margin) against the position, hold old position
through the first 15 min, then flip on break of that first-15-min low/high (no margin). Gap checked ONCE per
day (at open), never re-checked at later hourly boundaries. First trade waits for the first natural breakout.

Hourly candles built from 1-min, anchored 09:15: 6 full hours + a 15:15-15:29 15-min STUB = ~7/day (FLAGGED:
the 6h15m session doesn't divide evenly). Muhurat evening 1-min (outside 09:15-15:29) excluded (FLAGGED).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DATA = rb.BASE / "data" / "nifty_1min_ohlc.csv"
DAILY_CSV = rb.RESULTS / "breakout_2day_gap" / "breakout_2day_gap_trades.csv"     # for the daily-vs-hourly comparison
OUTDIR = rb.RESULTS / "breakout_2hr_hourly"; OUTDIR.mkdir(parents=True, exist_ok=True)
MARGIN = 10.0; FIRST15_END = 570        # 09:30 ; first 15-min bar = 09:15-09:29 (mod 555..569)


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    d["date"] = d["ts"].dt.normalize(); d["mod"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    n_out = int(((d["mod"] < 555) | (d["mod"] > 929)).sum())
    d = d[(d["mod"] >= 555) & (d["mod"] <= 929)].reset_index(drop=True)           # regular session only

    # ---- global sequential hourly candles (anchored 09:15; last is a 15-min stub) ----
    d["hbd"] = ((d["mod"] - 555) // 60).clip(0, 6)
    d["hour_id"] = pd.factorize(d["date"].dt.strftime("%Y%m%d") + "_" + d["hbd"].astype(str))[0]   # time-ordered
    hourly = d.groupby("hour_id").agg(hh=("high", "max"), hl=("low", "min")).sort_index()
    hourly["p2h"] = hourly["hh"].shift(1).rolling(2).max()                        # prev 2 COMPLETED hourly candles
    hourly["p2l"] = hourly["hl"].shift(1).rolling(2).min()
    hourly["up"] = hourly["p2h"] + MARGIN; hourly["down"] = hourly["p2l"] - MARGIN
    d = d.merge(hourly[["p2h", "p2l", "up", "down"]], left_on="hour_id", right_index=True, how="left")
    # day-level: first-15-min hi/lo and day open
    f15 = d[(d["mod"] >= 555) & (d["mod"] < FIRST15_END)].groupby("date").agg(fh15=("high", "max"), fl15=("low", "min"))
    dopen = d.groupby("date")["open"].first().rename("dopen")
    d = d.merge(f15, left_on="date", right_index=True, how="left").merge(dopen, left_on="date", right_index=True, how="left")
    period = f"{d.date.iloc[0].date()} .. {d.date.iloc[-1].date()}"

    dt_a = d["date"].values; mod_a = d["mod"].values; hi_a = d["high"].values; lo_a = d["low"].values; op_a = d["open"].values; tsr = d["ts"].values
    up_a, dn_a, p2h_a, p2l_a, do_a, fh_a, fl_a = (d[c].values for c in ["up", "down", "p2h", "p2l", "dopen", "fh15", "fl15"])

    pos = 0; entry = np.nan; entry_t = None; cur_type = None; trades = []; sig = 0
    cur_day = None; gap = None; gap_flipped = False; gap_held = []
    for k in range(len(d)):
        if dt_a[k] != cur_day:                                    # ---- new trading day: gap check at day open (first hourly candle) ----
            if gap in ("down", "up") and not gap_flipped:
                gap_held.append((pd.Timestamp(cur_day).date(), gap))
            cur_day = dt_a[k]; gap = None; gap_flipped = False
            if not np.isnan(p2l_a[k]):                            # first-hour RAW level (= prior day's last 2 hourly candles)
                if pos == 1 and do_a[k] < p2l_a[k]: gap = "down"
                elif pos == -1 and do_a[k] > p2h_a[k]: gap = "up"
        up, dn, h, l, o, t, m = up_a[k], dn_a[k], hi_a[k], lo_a[k], op_a[k], tsr[k], mod_a[k]
        if np.isnan(up):
            continue
        if gap in ("down", "up") and not gap_flipped:            # ---- GAP MODE (overnight gap only) ----
            if m < FIRST15_END:
                continue                                         # hold through first 15 min
            if gap == "down" and pos == 1 and l <= fl_a[k]:
                trades.append(("Long", entry_t, entry, t, fl_a[k], fl_a[k] - entry, cur_type)); pos, entry, entry_t, cur_type = -1, fl_a[k], t, "gap"; gap = None; gap_flipped = True; sig += 1
            elif gap == "up" and pos == -1 and h >= fh_a[k]:
                trades.append(("Short", entry_t, entry, t, fh_a[k], entry - fh_a[k], cur_type)); pos, entry, entry_t, cur_type = 1, fh_a[k], t, "gap"; gap = None; gap_flipped = True; sig += 1
            continue                                             # while gap active, hourly triggers suppressed
        # ---- NORMAL HOURLY BREAKOUT (level = current hour's prev-2hr window) ----
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
        gap_held.append((pd.Timestamp(cur_day).date(), gap))

    T = pd.DataFrame(trades, columns=["direction", "entry_time", "entry_price", "exit_time", "exit_price", "points_pnl", "entry_type"])
    T["hold_hours"] = (pd.to_datetime(T["exit_time"]) - pd.to_datetime(T["entry_time"])).dt.total_seconds() / 3600
    T["result"] = np.where(T["points_pnl"] > 0, "Win", np.where(T["points_pnl"] < 0, "Loss", "Flat"))
    for c in ("entry_time", "exit_time"): T[c] = pd.to_datetime(T[c]).dt.strftime("%Y-%m-%d %H:%M")
    T.insert(0, "trade_no", range(1, len(T) + 1)); T["cum_points"] = T["points_pnl"].cumsum().round(1)

    span = (d.date.iloc[-1] - d.date.iloc[0]).days; weeks = span / 7; months = span / 30.44
    def blk(df, lbl):
        return {"type": lbl, "trades": len(df), "win_%": round((df.points_pnl > 0).mean() * 100, 1) if len(df) else 0,
                "total_pts": round(df.points_pnl.sum(), 1), "avg_pts": round(df.points_pnl.mean(), 2) if len(df) else 0,
                "avg_hold_hrs": round(df.hold_hours.mean(), 2) if len(df) else 0}
    byt = pd.DataFrame([blk(T, "ALL"), blk(T[T.entry_type == "normal"], "NORMAL breakout"), blk(T[T.entry_type == "gap"], "GAP-day")])

    # ---- comparison vs the DAILY version ----
    cmp = None
    if DAILY_CSV.exists():
        dl = pd.read_csv(DAILY_CSV); dgross = dl["gross_points"].sum() if "gross_points" in dl else dl["points_pnl"].sum()
        dhold_d = (pd.to_datetime(dl["exit_time"]).dt.normalize() - pd.to_datetime(dl["entry_time"]).dt.normalize()).dt.days.mean()
        cmp = pd.DataFrame([
            {"metric": "trades", "HOURLY": len(T), "DAILY": len(dl)},
            {"metric": "total P&L (pts, gross)", "HOURLY": round(T.points_pnl.sum(), 1), "DAILY": round(dgross, 1)},
            {"metric": "win %", "HOURLY": round((T.points_pnl > 0).mean() * 100, 1), "DAILY": round((dl.get("gross_points", dl["points_pnl"]) > 0).mean() * 100, 1)},
            {"metric": "avg hold", "HOURLY": f"{round(T.hold_hours.mean(),2)} hrs", "DAILY": f"{round(dhold_d,2)} days"},
            {"metric": "trades / month", "HOURLY": round(len(T) / months, 2), "DAILY": round(len(dl) / months, 2)},
            {"metric": "trades / week", "HOURLY": round(len(T) / weeks, 2), "DAILY": round(len(dl) / weeks, 2)},
            {"metric": "freq multiple (hourly/daily)", "HOURLY": f"{round(len(T)/max(len(dl),1),1)}x", "DAILY": "1x"}])

    with pd.ExcelWriter(OUTDIR / "breakout_2hr_hourly.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Strategy", "value": "STANDALONE NIFTY 2-HOURLY-candle breakout + gap (index pts, always-in flip, intraday touch)"},
            {"metric": "MERGE NOTE", "value": "Standalone hourly leg — could later join the multi-TF system (like the daily leg with 15m/1h Supertrend)"},
            {"metric": "Period", "value": period},
            {"metric": "Levels", "value": "rolling max/min of last 2 completed hourly candles (continuous across days); up/down = +-10 (unchanged)"},
            {"metric": "Gap", "value": "checked ONLY at day's first hourly candle (overnight gap), RAW prev-2hr level; hold 15 min then 15-min break"},
            {"metric": "DATA FLAG", "value": "hourly = 6 full hours + 15:15-15:29 15-min STUB (~7/day; 6h15m session doesn't divide evenly)"},
            {"metric": "DATA FLAG", "value": f"{n_out} out-of-session 1-min rows (Muhurat evening) excluded from hourly bucketing"},
            {"metric": "Total trades", "value": len(T)}, {"metric": "Win rate %", "value": round((T.points_pnl > 0).mean() * 100, 1)},
            {"metric": "Total P&L (pts, gross)", "value": round(T.points_pnl.sum(), 1)}, {"metric": "Avg pts/trade", "value": round(T.points_pnl.mean(), 2)},
            {"metric": "Avg hold (hours)", "value": round(T.hold_hours.mean(), 2)},
            {"metric": "Signals (incl 1st entry)", "value": sig}, {"metric": "Signals/week", "value": round(sig / weeks, 2)}, {"metric": "Signals/month", "value": round(sig / months, 2)},
            {"metric": "GAP-day trades", "value": int((T.entry_type == "gap").sum())}, {"metric": "NORMAL trades", "value": int((T.entry_type == "normal").sum())},
            {"metric": "Gap fired but 15-min NOT broken (held)", "value": len(gap_held)},
        ]).to_excel(w, sheet_name="Summary", index=False)
        byt.to_excel(w, sheet_name="By_Type", index=False)
        if cmp is not None: cmp.to_excel(w, sheet_name="Hourly_vs_Daily", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        pd.DataFrame(gap_held, columns=["date", "gap_direction"]).to_excel(w, sheet_name="Gap_Held_No_Flip", index=False)
    T.to_csv(OUTDIR / "breakout_2hr_hourly_trades.csv", index=False)

    pd.set_option("display.width", 200)
    print("=" * 96 + "\nSTANDALONE NIFTY 2-HOURLY-CANDLE BREAKOUT + GAP (hourly TF; intraday-touch; always-in flip)\n" + "=" * 96)
    print(f"period {period} | *** STANDALONE hourly leg — mergeable into multi-TF system later ***")
    print(f"\ntrades {len(T)} | win {round((T.points_pnl>0).mean()*100,1)}% | TOTAL {round(T.points_pnl.sum(),1):,} pts | avg {round(T.points_pnl.mean(),2)} | avg hold {round(T.hold_hours.mean(),2)} hrs")
    print(f"signals/week {round(sig/weeks,2)} | /month {round(sig/months,2)}")
    print("\n--- BY TYPE ---"); print(byt.to_string(index=False))
    print(f"\ngap-fired-but-not-broken (held) days: {len(gap_held)}")
    if cmp is not None:
        print("\n--- HOURLY vs DAILY ---"); print(cmp.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
