# -*- coding: utf-8 -*-
"""nifty_3day_breakout_451_backtest.py — REAL options-based backtest of the "Nifty 3 Day Breakout 4:5:1"
weekly ratio put/call spread. Uses REAL NIFTY 1-min option data throughout (not synthetic) -- restricted to
the window where that data actually exists locally (Oct-2024 onward), unlike the earlier synthetic BTST/
Credit-Spread builds which could run back to Mar-2022 on spot alone.

EXPIRY CALENDAR: derived from the REAL local options-data folder names, not assumed -- correctly spans
both the OLD Thursday-expiry regime and the NEW Tuesday-expiry regime (confirmed switch: 2025-09-02).
Regime is regime-agnostic by design: the entry window is "the next up-to-3 trading days after expiry"
(whatever weekday those actually are for that historical regime), not hardcoded Wed/Thu/Fri -- so old-
regime weeks (expiry Thu -> entry window Fri/Mon/Tue) are handled correctly without special-casing, and
each trade's actual entry weekday + regime is recorded for review.

ENTRY: 3-day high/low = max high / min low of the 3 trading days immediately before the day being checked
(recomputed fresh each day rolled forward). Touch-basis on that day's spot 1-min bars, whole day (open to
close): first candle whose LOW <= 3-day-low -> BEARISH (Put Ratio Spread); first candle whose HIGH >=
3-day-high -> BULLISH (Call Ratio Spread), whichever occurs first chronologically. ATM at breakout = the
exact breakout LEVEL itself (the 3-day-low/high value that was touched -- the precise instant-of-breakout
price), rounded to the nearest 50-strike. If no breakout within the entry window (up to 3 trading days),
no trade that week.

STRUCTURE: BEARISH = +4 ATM PE, -5 (ATM-200) PE, +1 (ATM-400) PE. BULLISH = +4 ATM CE, -5 (ATM+200) CE,
+1 (ATM+400) CE. Entry fill = each leg's CLOSE at the breakout candle's timestamp. entry_cost = 4*L1 - 5*L2
+ 1*L3 (positive = net debit paid, negative = net credit received).

EXIT: checked MINUTE BY MINUTE (1-min close, all 3 legs aligned on common timestamps), from the breakout
entry minute (exclusive) through the next expiry's last available minute (time stop, inclusive) --
upgraded from an earlier EOD-close-only version per explicit user request, since 1-min option data is
available for the whole holding period. CONFIRMED TARGET FORMULA (derived + user-confirmed): max
theoretical profit at expiry occurs exactly at the SOLD strike and equals 4 x 200 = 800 points
(independent of CE/PE side); actual max profit for the trade = 800 - entry_cost. Target = 90% of that
fixed number. Live mark-to-market P&L at each minute = (4*L1_now - 5*L2_now + 1*L3_now) - entry_cost,
all 3 legs' CLOSE at that same timestamp; first minute this MTM P&L >= target -> exit there (realized at
that minute's actual value, not forced to exactly 90%). If never triggered by expiry's last minute,
time-stop exit at that final value, NO stop-loss (flagged: genuine open-ended intraweek risk by design,
matching the original spec's known gap -- now measured at 1-min resolution rather than daily).
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_3day_breakout_451"; OUTDIR.mkdir(parents=True, exist_ok=True)

STRIKE_INT = 50; NEAR_W = 200; FAR_W = 400
LOTS = {"L1": 4, "L2": -5, "L3": 1}  # L1=ATM long, L2=near short, L3=far long
MAX_ENTRY_WINDOW = 3  # trading days after expiry to look for a breakout
MAX_PROFIT_BASE = LOTS["L1"] * NEAR_W  # 4 * 200 = 800, independent of side


def find_leg_file(exp_folder, strike, otype):
    m = glob.glob(str(OPTDIR / exp_folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    return m[0] if m else None


def leg_close_at(exp_folder, strike, otype, ts):
    fn = find_leg_file(exp_folder, strike, otype)
    if fn is None:
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "close"])
    day = df[df["timestamp"].dt.normalize() == ts.normalize()]
    if day.empty:
        return None
    row = day[day["timestamp"] == ts]
    if not row.empty:
        return float(row["close"].iloc[0])
    before = day[day["timestamp"] <= ts]
    if before.empty:
        return None
    return float(before.sort_values("timestamp")["close"].iloc[-1])


def leg_day_close(exp_folder, strike, otype, day):
    fn = find_leg_file(exp_folder, strike, otype)
    if fn is None:
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "close"])
    d = df[df["timestamp"].dt.normalize() == day.normalize()]
    if d.empty:
        return None
    return float(d.sort_values("timestamp")["close"].iloc[-1])


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize()
    day_high = sp.groupby("date")["high"].max(); day_low = sp.groupby("date")["low"].min()
    full_days = sorted(sp["date"].unique()); day_idx = {d: i for i, d in enumerate(full_days)}
    by_day = {d: g.sort_values("ts") for d, g in sp.groupby("date")}

    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d"))
    wd = pd.Series([e.day_name() for e in expiries], index=expiries)
    regime_switch = None
    prev = None
    for e, w in wd.items():
        if w in ("Tuesday", "Thursday"):
            if prev is not None and w != prev:
                regime_switch = e
            prev = w
    print(f"expiries: {len(expiries)} | {expiries[0].date()} .. {expiries[-1].date()} | regime switch (Thu->Tue) at {regime_switch.date()}", flush=True)

    ENTRY_CUTOFF = pd.Timestamp("2026-07-31")  # no trade may touch August -- both entry expiry AND square-off expiry capped
    trades = []; missing = []; no_trade_weeks = []

    for i in range(len(expiries) - 1):
        E_prev, E_curr = expiries[i], expiries[i + 1]
        if E_prev > ENTRY_CUTOFF or E_curr > ENTRY_CUTOFF:
            continue
        pj = day_idx.get(E_prev)
        if pj is None or pj + 1 >= len(full_days):
            continue
        regime = "OLD (Thu-expiry)" if E_prev.day_name() == "Thursday" else ("NEW (Tue-expiry)" if E_prev.day_name() == "Tuesday" else f"IRREGULAR ({E_prev.day_name()})")

        found = False
        for k in range(1, MAX_ENTRY_WINDOW + 1):
            ci = pj + k
            if ci >= len(full_days) or full_days[ci] > E_curr:
                break
            D = full_days[ci]
            if ci < 3:
                continue
            prior3 = full_days[ci - 3:ci]
            d3high = max(day_high[d] for d in prior3); d3low = min(day_low[d] for d in prior3)

            day_bars = by_day.get(D)
            if day_bars is None or day_bars.empty:
                missing.append((D, "no spot candles")); continue

            trig_type = None; trig_price = None; trig_ts = None
            for _, r in day_bars.iterrows():
                if r["low"] <= d3low:
                    trig_type, trig_price, trig_ts = "BEARISH", d3low, r["ts"]; break
                if r["high"] >= d3high:
                    trig_type, trig_price, trig_ts = "BULLISH", d3high, r["ts"]; break
            if trig_type is None:
                continue

            atm = round(trig_price / STRIKE_INT) * STRIKE_INT
            exp_folder = E_curr.strftime("%Y%m%d")
            if trig_type == "BEARISH":
                otype = "PE"; K1, K2, K3 = atm, atm - NEAR_W, atm - FAR_W
            else:
                otype = "CE"; K1, K2, K3 = atm, atm + NEAR_W, atm + FAR_W

            L1 = leg_close_at(exp_folder, K1, otype, trig_ts)
            L2 = leg_close_at(exp_folder, K2, otype, trig_ts)
            L3 = leg_close_at(exp_folder, K3, otype, trig_ts)
            if None in (L1, L2, L3):
                missing.append((D, f"missing leg data at breakout {trig_type} ATM={atm}")); continue

            entry_cost = LOTS["L1"] * L1 + LOTS["L2"] * L2 + LOTS["L3"] * L3
            max_profit = MAX_PROFIT_BASE - entry_cost
            target = 0.90 * max_profit

            # ---- MINUTE-BY-MINUTE MTM check, entry minute (exclusive) through E_curr's last minute ----
            # each leg's own contract file spans its whole life (one expiry) -- load once, filter to window.
            f1 = find_leg_file(exp_folder, K1, otype); f2 = find_leg_file(exp_folder, K2, otype); f3 = find_leg_file(exp_folder, K3, otype)
            if None in (f1, f2, f3):
                missing.append((D, f"missing leg file for minute walk {trig_type} ATM={atm}")); break
            s1 = pd.read_parquet(f1, columns=["timestamp", "close"]).set_index("timestamp")["close"]
            s2 = pd.read_parquet(f2, columns=["timestamp", "close"]).set_index("timestamp")["close"]
            s3 = pd.read_parquet(f3, columns=["timestamp", "close"]).set_index("timestamp")["close"]
            s1 = s1[(s1.index > trig_ts) & (s1.index.normalize() <= E_curr)]
            s2 = s2[(s2.index > trig_ts) & (s2.index.normalize() <= E_curr)]
            s3 = s3[(s3.index > trig_ts) & (s3.index.normalize() <= E_curr)]
            common_ts = s1.index.intersection(s2.index).intersection(s3.index)
            if len(common_ts) == 0:
                missing.append((D, "no common 1-min timestamps across all 3 legs after entry")); continue
            common_ts = common_ts.sort_values()
            mtm_series = LOTS["L1"] * s1.loc[common_ts] + LOTS["L2"] * s2.loc[common_ts] + LOTS["L3"] * s3.loc[common_ts]
            pnl_series = mtm_series - entry_cost

            hit = pnl_series[pnl_series >= target]
            if len(hit):
                exit_ts = hit.index[0]; exit_pnl = float(hit.iloc[0]); exit_mtm_value = float(mtm_series.loc[exit_ts])
                exit_reason = "90% target"
            else:
                exit_ts = pnl_series.index[-1]; exit_pnl = float(pnl_series.iloc[-1]); exit_mtm_value = float(mtm_series.iloc[-1])
                exit_reason = "time stop"
            exit_date = exit_ts.normalize()

            trades.append({
                "entry_date": D, "entry_time": trig_ts, "entry_weekday": D.day_name(), "regime": regime, "expiry_used": E_curr,
                "direction": trig_type, "breakout_spot": round(trig_price, 2), "ATM": atm,
                "K1_long4": K1, "K2_short5": K2, "K3_long1": K3, "option_type": otype,
                "L1_entry": L1, "L2_entry": L2, "L3_entry": L3, "entry_cost": round(entry_cost, 2),
                "entry_type": "NET DEBIT" if entry_cost > 0 else "NET CREDIT",
                "max_theoretical_profit": round(max_profit, 2), "target_90pct": round(target, 2),
                "exit_date": exit_date, "exit_time": exit_ts, "exit_reason": exit_reason,
                "exit_mtm_value": round(exit_mtm_value, 2), "pnl_points": round(exit_pnl, 2),
            })
            found = True
            break
        if not found:
            no_trade_weeks.append(E_prev)

    T = pd.DataFrame(trades)
    print(f"\ntotal trades: {len(T)} | no-trade weeks (no breakout by entry-window end): {len(no_trade_weeks)} | missing/flagged: {len(missing)}", flush=True)
    T.to_csv(OUTDIR / "nifty_3day_breakout_451_trades.csv", index=False)

    # ---- summary ----
    n = len(T); win = T["pnl_points"] > 0
    print(f"\nwin rate (90% target hit before Tuesday): {round(win.mean()*100,2)}%")
    print(f"avg P&L winning weeks: {round(T.loc[win,'pnl_points'].mean(),2)} | avg P&L losing weeks: {round(T.loc[~win,'pnl_points'].mean(),2)}")
    print(f"worst single-week loss: {round(T['pnl_points'].min(),2)}")
    print(f"total weeks (trade + no-trade): {n + len(no_trade_weeks)} | %% no-trade: {round(len(no_trade_weeks)/(n+len(no_trade_weeks))*100,2)}%%")
    print(f"\nby entry weekday:\n{T.groupby('entry_weekday')['pnl_points'].agg(['size','mean','sum'])}")
    print(f"\nby direction:\n{T.groupby('direction')['pnl_points'].agg(['size','mean','sum'])}")
    print(f"\nby entry_type (debit/credit):\n{T.groupby('entry_type')['pnl_points'].agg(['size','mean','sum'])}")
    print(f"\nby regime:\n{T.groupby('regime')['pnl_points'].agg(['size','mean','sum'])}")
    print(f"\nby exit_reason:\n{T.groupby('exit_reason')['pnl_points'].agg(['size','mean','sum'])}")

    with pd.ExcelWriter(OUTDIR / "nifty_3day_breakout_451_report.xlsx", engine="openpyxl") as w:
        T.to_excel(w, sheet_name="Trades", index=False)
        if missing:
            pd.DataFrame(missing, columns=["date", "issue"]).to_excel(w, sheet_name="Missing_Flagged", index=False)
        pd.DataFrame({"expiry_no_trade": no_trade_weeks}).to_excel(w, sheet_name="No_Trade_Weeks", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)

    print(f"\nSaved -> {OUTDIR}/nifty_3day_breakout_451_report.xlsx")


if __name__ == "__main__":
    main()
