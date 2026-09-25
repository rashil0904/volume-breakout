# -*- coding: utf-8 -*-
"""breakout_2day_daily.py — STANDALONE NIFTY daily rolling 2-day high/low breakout (10-pt margin),
intraday-touch based, always-in flip system. INDEX-point P&L (spot). SEPARATE module (to be merged with
the dual-TF Supertrend later — NOT merged here).

Levels (rolling, per day D): up = max(high[D-1],high[D-2]) + 10 ; down = min(low[D-1],low[D-2]) - 10.
Touch (intraday, 1-min): current day's high>=up -> long trigger; low<=down -> short trigger; first touch
of the day wins. Always +1/-1: long flips to short when price touches `down`; short flips to long at `up`;
exit/entry price = the exact trigger level. First trade waits for the first natural breakout (either side).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DATA = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "breakout_2day"; OUTDIR.mkdir(parents=True, exist_ok=True)
MARGIN = 10.0


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True); d["date"] = d["ts"].dt.normalize()
    daily = d.groupby("date").agg(dh=("high", "max"), dl=("low", "min"), do=("open", "first"), dc=("close", "last"))
    daily["p2h"] = daily["dh"].shift(1).rolling(2).max()          # max(high[D-1], high[D-2])
    daily["p2l"] = daily["dl"].shift(1).rolling(2).min()
    daily["up"] = daily["p2h"] + MARGIN; daily["down"] = daily["p2l"] - MARGIN
    both_days = daily[(daily["dh"] >= daily["up"]) & (daily["dl"] <= daily["down"])].index   # BOTH thresholds touchable that day
    period = f"{daily.index[0].date()} .. {daily.index[-1].date()}"

    dm = d.merge(daily[["up", "down"]], left_on="date", right_index=True, how="left")
    up_a, dn_a, hi_a, lo_a, op_a, tsr = dm["up"].values, dm["down"].values, dm["high"].values, dm["low"].values, dm["open"].values, dm["ts"].values

    pos = 0; entry = np.nan; entry_t = None; trades = []; sig = 0; same_candle = []
    for k in range(len(dm)):
        up, dn = up_a[k], dn_a[k]
        if np.isnan(up):
            continue
        h, l, o, t = hi_a[k], lo_a[k], op_a[k], tsr[k]
        if pos == 0:                                              # FIRST trade: watch both, first touch wins
            hu, hd = h >= up, l <= dn
            if hu and hd:
                first = "up" if abs(o - up) <= abs(o - dn) else "down"; same_candle.append((t, first))
                pos, entry = (1, up) if first == "up" else (-1, dn); entry_t = t; sig += 1
            elif hu:
                pos, entry, entry_t = 1, up, t; sig += 1
            elif hd:
                pos, entry, entry_t = -1, dn, t; sig += 1
        elif pos == 1:                                            # long -> flip short on down-break
            if l <= dn:
                trades.append(("Long", entry_t, entry, t, dn, dn - entry)); pos, entry, entry_t = -1, dn, t; sig += 1
        else:                                                    # short -> flip long on up-break
            if h >= up:
                trades.append(("Short", entry_t, entry, t, up, entry - up)); pos, entry, entry_t = 1, up, t; sig += 1

    T = pd.DataFrame(trades, columns=["direction", "entry_time", "entry_price", "exit_time", "exit_price", "points_pnl"])
    T["hold_hours"] = (pd.to_datetime(T["exit_time"]) - pd.to_datetime(T["entry_time"])).dt.total_seconds() / 3600
    T["hold_cal_days"] = (pd.to_datetime(T["exit_time"]).dt.normalize() - pd.to_datetime(T["entry_time"]).dt.normalize()).dt.days
    T["result"] = np.where(T["points_pnl"] > 0, "Win", np.where(T["points_pnl"] < 0, "Loss", "Flat"))
    for c in ("entry_time", "exit_time"):
        T[c] = pd.to_datetime(T[c]).dt.strftime("%Y-%m-%d %H:%M")
    T.insert(0, "trade_no", range(1, len(T) + 1))
    T["cum_points"] = T["points_pnl"].cumsum().round(1)

    span_days = (daily.index[-1] - daily.index[0]).days; weeks = span_days / 7; months = span_days / 30.44
    tot = round(T["points_pnl"].sum(), 1); win = round((T["points_pnl"] > 0).mean() * 100, 1)
    summ = pd.DataFrame([
        {"metric": "Strategy", "value": "STANDALONE NIFTY rolling 2-day high/low breakout, 10-pt margin, intraday-touch, always-in flip"},
        {"metric": "MERGE NOTE", "value": "Independent module — to be COMBINED with the dual-TF Supertrend LATER (not merged here)"},
        {"metric": "Period", "value": period}, {"metric": "Data", "value": "NIFTY 1-min INDEX (spot); touch resolved to the 1-min bar"},
        {"metric": "Total trades (closed flips)", "value": len(T)},
        {"metric": "Win rate %", "value": win}, {"metric": "Total P&L (index points)", "value": tot},
        {"metric": "Avg points / trade", "value": round(T["points_pnl"].mean(), 1)},
        {"metric": "Best / worst trade (pts)", "value": f"{round(T['points_pnl'].max(),1)} / {round(T['points_pnl'].min(),1)}"},
        {"metric": "Avg holding (calendar days)", "value": round(T["hold_cal_days"].mean(), 2)},
        {"metric": "Avg holding (hours)", "value": round(T["hold_hours"].mean(), 1)},
        {"metric": "Total breakout signals (incl. 1st entry)", "value": sig},
        {"metric": "Signals per week", "value": round(sig / weeks, 2)}, {"metric": "Signals per month", "value": round(sig / months, 2)},
        {"metric": "Long / Short trades", "value": f"{int((T.direction=='Long').sum())} / {int((T.direction=='Short').sum())}"},
        {"metric": "", "value": ""},
        {"metric": "Both-thresholds-touched days (wide range)", "value": len(both_days)},
        {"metric": "  -> resolution", "value": "intraday walk records whichever trigger is touched FIRST (candle-by-candle); position then watches the opposite trigger"},
        {"metric": "Same-1min-bar both-touch events (ambiguous)", "value": len(same_candle)},
        {"metric": "  -> tie rule", "value": "within one 1-min bar (tick order unknown), whichever trigger is CLOSER TO THE BAR OPEN is taken first"},
        {"metric": "", "value": ""},
        {"metric": "DATA-GRANULARITY FLAG", "value": "1-min bars available (good). Sub-minute tick order inside a bar is NOT known -> same-bar both-touch resolved by open-proximity heuristic (rare)."},
        {"metric": "Note", "value": "Entry/exit = exact trigger level (not candle O/C). Always-in: a final open position is left unrealized (not in trade list)."},
    ])

    with pd.ExcelWriter(OUTDIR / "breakout_2day.xlsx", engine="openpyxl") as w:
        summ.to_excel(w, sheet_name="Summary", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        pd.DataFrame({"both_threshold_day": [x.date() for x in both_days]}).to_excel(w, sheet_name="Both_Touched_Days", index=False)
        if same_candle:
            pd.DataFrame(same_candle, columns=["timestamp", "first_touch"]).to_excel(w, sheet_name="Same_Bar_Ties", index=False)
    T.to_csv(OUTDIR / "breakout_2day_trades.csv", index=False)

    pd.set_option("display.width", 200)
    print("=" * 92 + "\nSTANDALONE NIFTY 2-DAY HIGH/LOW BREAKOUT (10-pt margin, intraday-touch, always-in flip)\n" + "=" * 92)
    print(f"period: {period} | NIFTY 1-min index | *** STANDALONE — to be merged with dual-TF Supertrend later ***")
    print(f"\ntrades {len(T)} | win {win}% | TOTAL P&L {tot:,.0f} pts | avg {round(T.points_pnl.mean(),1)} pts/trade")
    print(f"long/short {int((T.direction=='Long').sum())}/{int((T.direction=='Short').sum())} | best {round(T.points_pnl.max(),1)} worst {round(T.points_pnl.min(),1)}")
    print(f"avg holding: {round(T.hold_cal_days.mean(),2)} calendar days ({round(T.hold_hours.mean(),1)} h)")
    print(f"signals: {sig} total | {round(sig/weeks,2)}/week | {round(sig/months,2)}/month")
    print(f"\nboth-thresholds-touched (wide) days: {len(both_days)} (first-touch recorded intraday) | same-1min-bar ties: {len(same_candle)} (resolved by open-proximity)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
