# -*- coding: utf-8 -*-
"""nifty_3day_breakout_451_target_sweep.py — sweeps the profit-target percentage (70/75/80/85/90% of the
theoretical max profit) for the Nifty 3-Day Breakout 4:5:1 strategy, reusing the EXACT SAME entry logic
(breakout detection, ATM/leg selection, entry cost) as the main backtest -- only the exit target threshold
varies. Computes each trade's full daily MTM P&L path ONCE, then evaluates all 5 target levels against
that cached path (no re-fetching option data per target). Capped at the 2026-07-28 expiry (no August 2026),
same as the main backtest.
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


def leg_daily_closes(exp_folder, strike, otype):
    fn = find_leg_file(exp_folder, strike, otype)
    if fn is None:
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "close"])
    df["date"] = df["timestamp"].dt.normalize()
    return df.groupby("date")["close"].last()


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

            s1 = leg_daily_closes(exp_folder, K1, otype); s2 = leg_daily_closes(exp_folder, K2, otype); s3 = leg_daily_closes(exp_folder, K3, otype)
            if s1 is None or s2 is None or s3 is None:
                break
            path_days = [d for d in full_days[ci:] if d <= E_curr]
            mtm_path = []
            for Dchk in path_days:
                c1 = s1.get(Dchk); c2 = s2.get(Dchk); c3 = s3.get(Dchk)
                if None in (c1, c2, c3) or pd.isna(c1) or pd.isna(c2) or pd.isna(c3):
                    mtm_path.append(np.nan); continue
                mtm_path.append(LOTS["L1"] * c1 + LOTS["L2"] * c2 + LOTS["L3"] * c3 - entry_cost)

            trade_paths.append({"entry_date": D, "direction": trig_type, "max_profit": max_profit,
                                 "path_days": path_days, "mtm_path": mtm_path})
            break

    print(f"total qualifying trades (with valid daily MTM path): {len(trade_paths)}", flush=True)

    rows = []
    for pct in TARGET_PCTS:
        pnls = []
        for tp in trade_paths:
            target = pct * tp["max_profit"]
            pnl_series = tp["mtm_path"]
            hit_idx = next((idx for idx, v in enumerate(pnl_series) if not np.isnan(v) and v >= target), None)
            if hit_idx is not None:
                pnls.append(pnl_series[hit_idx])
            else:
                valid = [v for v in pnl_series if not np.isnan(v)]
                pnls.append(valid[-1] if valid else np.nan)
        pnls = pd.Series(pnls).dropna()
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
    print("\n=== TARGET SWEEP (70-90%) ===")
    print(SWEEP.to_string(index=False))

    SWEEP.to_csv(OUTDIR / "target_sweep_70to90.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "target_sweep_70to90.xlsx", engine="openpyxl") as w:
        SWEEP.to_excel(w, sheet_name="Target_Sweep", index=False)
    print(f"\nSaved -> {OUTDIR}/target_sweep_70to90.xlsx")


if __name__ == "__main__":
    main()
