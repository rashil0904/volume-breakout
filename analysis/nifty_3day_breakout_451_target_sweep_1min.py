# -*- coding: utf-8 -*-
"""nifty_3day_breakout_451_target_sweep_1min.py — sweeps the profit-target percentage (70-90%) using
MINUTE-LEVEL MTM paths (matching the upgraded 1-min main backtest), same entries as the main backtest.
Each trade's full 1-min MTM P&L series is computed ONCE and cached, then evaluated against all 5 target
levels. Capped at the 2026-07-28 expiry (no August 2026), same as the main backtest.
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
LOTS = {"L1": 4, "L2": -5, "L3": 1}
MAX_ENTRY_WINDOW = 3
MAX_PROFIT_BASE = LOTS["L1"] * NEAR_W
ENTRY_CUTOFF = pd.Timestamp("2026-07-31")
TARGET_PCTS = [0.70, 0.75, 0.80, 0.85, 0.90]


def find_leg_file(exp_folder, strike, otype):
    m = glob.glob(str(OPTDIR / exp_folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    return m[0] if m else None


def leg_close_at(exp_folder, strike, otype, tsv):
    fn = find_leg_file(exp_folder, strike, otype)
    if fn is None:
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "close"])
    day = df[df["timestamp"].dt.normalize() == tsv.normalize()]
    if day.empty:
        return None
    row = day[day["timestamp"] == tsv]
    if not row.empty:
        return float(row["close"].iloc[0])
    before = day[day["timestamp"] <= tsv]
    return float(before.sort_values("timestamp")["close"].iloc[-1]) if len(before) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize()
    day_high = sp.groupby("date")["high"].max(); day_low = sp.groupby("date")["low"].min()
    full_days = sorted(sp["date"].unique()); day_idx = {d: i for i, d in enumerate(full_days)}
    by_day = {d: g.sort_values("ts") for d, g in sp.groupby("date")}

    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d"))

    trade_paths = []
    for i in range(len(expiries) - 1):
        E_prev, E_curr = expiries[i], expiries[i + 1]
        if E_prev > ENTRY_CUTOFF or E_curr > ENTRY_CUTOFF:
            continue
        pj = day_idx.get(E_prev)
        if pj is None or pj + 1 >= len(full_days):
            continue

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
                continue
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

            L1 = leg_close_at(exp_folder, K1, otype, trig_ts); L2 = leg_close_at(exp_folder, K2, otype, trig_ts); L3 = leg_close_at(exp_folder, K3, otype, trig_ts)
            if None in (L1, L2, L3):
                break
            entry_cost = LOTS["L1"] * L1 + LOTS["L2"] * L2 + LOTS["L3"] * L3
            max_profit = MAX_PROFIT_BASE - entry_cost

            f1 = find_leg_file(exp_folder, K1, otype); f2 = find_leg_file(exp_folder, K2, otype); f3 = find_leg_file(exp_folder, K3, otype)
            if None in (f1, f2, f3):
                break
            s1 = pd.read_parquet(f1, columns=["timestamp", "close"]).set_index("timestamp")["close"]
            s2 = pd.read_parquet(f2, columns=["timestamp", "close"]).set_index("timestamp")["close"]
            s3 = pd.read_parquet(f3, columns=["timestamp", "close"]).set_index("timestamp")["close"]
            s1 = s1[(s1.index > trig_ts) & (s1.index.normalize() <= E_curr)]
            s2 = s2[(s2.index > trig_ts) & (s2.index.normalize() <= E_curr)]
            s3 = s3[(s3.index > trig_ts) & (s3.index.normalize() <= E_curr)]
            common_ts = s1.index.intersection(s2.index).intersection(s3.index)
            if len(common_ts) == 0:
                break
            common_ts = common_ts.sort_values()
            mtm_series = LOTS["L1"] * s1.loc[common_ts] + LOTS["L2"] * s2.loc[common_ts] + LOTS["L3"] * s3.loc[common_ts]
            pnl_series = (mtm_series - entry_cost).values

            trade_paths.append({"entry_date": D, "direction": trig_type, "max_profit": max_profit, "pnl_series": pnl_series})
            break

    print(f"total qualifying trades (with valid 1-min MTM path): {len(trade_paths)}", flush=True)

    rows = []
    for pct in TARGET_PCTS:
        pnls = []
        for tp in trade_paths:
            target = pct * tp["max_profit"]
            series = tp["pnl_series"]
            hit_idx = np.argmax(series >= target) if np.any(series >= target) else None
            pnls.append(series[hit_idx] if hit_idx is not None else series[-1])
        pnls = pd.Series(pnls)
        win = pnls > 0
        rows.append({
            "target_pct": int(pct * 100), "n_trades": len(pnls), "total_return": round(pnls.sum(), 1),
            "avg_pnl": round(pnls.mean(), 2), "win_rate_pct": round(win.mean() * 100, 2),
            "avg_win": round(pnls[win].mean(), 2) if win.any() else None,
            "avg_loss": round(pnls[~win].mean(), 2) if (~win).any() else None,
            "worst_loss": round(pnls.min(), 1),
        })

    SWEEP = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== TARGET SWEEP (70-90%), 1-MIN GRANULARITY ===")
    print(SWEEP.to_string(index=False))

    SWEEP.to_csv(OUTDIR / "target_sweep_70to90_1min.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "target_sweep_70to90_1min.xlsx", engine="openpyxl") as w:
        SWEEP.to_excel(w, sheet_name="Target_Sweep_1min", index=False)
    print(f"\nSaved -> {OUTDIR}/target_sweep_70to90_1min.xlsx")


if __name__ == "__main__":
    main()
