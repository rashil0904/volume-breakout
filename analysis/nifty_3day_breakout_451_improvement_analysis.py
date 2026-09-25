# -*- coding: utf-8 -*-
"""nifty_3day_breakout_451_improvement_analysis.py — systematic improvement-test suite for the Nifty 3-Day
Breakout 4:5:1 strategy, built on the 1-min MTM backtest. Tests filters/parameters that are all knowable
AT ENTRY TIME (no look-ahead) -- direction, entry weekday, prior 3-day range size, regime -- plus the
target-pct finding from the earlier sweep (80% beat 90%). Each trade's full 1-min MTM path is cached once;
all filter/target combinations are evaluated against that same cache, no re-fetching option data.

IMPORTANT FLAG: the earlier "trending vs range-bound week" cut (from the original backtest report) used
the week's ACTUAL range from entry to exit -- that's only known in hindsight and is NOT a valid ex-ante
entry filter (look-ahead bias). This script instead tests the PRIOR 3-day range (the same range used to
detect the breakout itself) as a valid, entry-time-knowable proxy for "already-elevated volatility" --
a legitimate, non-look-ahead alternative.
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
        regime = "OLD (Thu-expiry)" if E_prev.day_name() == "Thursday" else ("NEW (Tue-expiry)" if E_prev.day_name() == "Tuesday" else f"IRREGULAR ({E_prev.day_name()})")
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
            prior_range = d3high - d3low
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

            trade_paths.append({"entry_date": D, "entry_weekday": D.day_name(), "direction": trig_type,
                                 "regime": regime, "prior_3day_range": prior_range, "max_profit": max_profit,
                                 "pnl_series": pnl_series})
            break

    print(f"total qualifying trades: {len(trade_paths)}", flush=True)
    TP = pd.DataFrame(trade_paths)
    median_range = TP["prior_3day_range"].median()
    print(f"median prior-3-day range (entry-time-knowable): {round(median_range,1)}", flush=True)

    def eval_combo(mask, target_pct, label):
        sub = TP[mask]
        if len(sub) == 0:
            return {"filter": label, "target_pct": int(target_pct*100), "n": 0}
        pnls = []
        for _, tp in sub.iterrows():
            target = target_pct * tp["max_profit"]; series = tp["pnl_series"]
            hit = np.argmax(series >= target) if np.any(series >= target) else None
            pnls.append(series[hit] if hit is not None else series[-1])
        pnls = pd.Series(pnls); win = pnls > 0
        return {"filter": label, "target_pct": int(target_pct*100), "n": len(pnls),
                "total_return": round(pnls.sum(), 1), "avg_pnl": round(pnls.mean(), 2),
                "win_rate_pct": round(win.mean()*100, 2),
                "avg_win": round(pnls[win].mean(), 2) if win.any() else None,
                "avg_loss": round(pnls[~win].mean(), 2) if (~win).any() else None,
                "worst_loss": round(pnls.min(), 1)}

    all_mask = pd.Series(True, index=TP.index)
    bearish_mask = TP["direction"] == "BEARISH"
    not_wed_mask = TP["entry_weekday"] != "Wednesday"
    high_range_mask = TP["prior_3day_range"] >= median_range
    old_regime_mask = TP["regime"] == "OLD (Thu-expiry)"

    combos = [
        (all_mask, 0.90, "BASELINE: all trades, 90% target"),
        (all_mask, 0.80, "all trades, 80% target"),
        (bearish_mask, 0.90, "BEARISH only, 90% target"),
        (bearish_mask, 0.80, "BEARISH only, 80% target"),
        (not_wed_mask, 0.90, "exclude Wednesday entries, 90% target"),
        (not_wed_mask, 0.80, "exclude Wednesday entries, 80% target"),
        (high_range_mask, 0.90, "prior-3day-range >= median, 90% target"),
        (high_range_mask, 0.80, "prior-3day-range >= median, 80% target"),
        (old_regime_mask, 0.90, "OLD (Thu-expiry) regime only, 90% target"),
        (bearish_mask & not_wed_mask, 0.80, "BEARISH + exclude Wed, 80% target"),
        (bearish_mask & high_range_mask, 0.80, "BEARISH + high prior-range, 80% target"),
        (bearish_mask & not_wed_mask & high_range_mask, 0.80, "BEARISH + exclude Wed + high prior-range, 80% target"),
        (bearish_mask & not_wed_mask & high_range_mask, 0.85, "BEARISH + exclude Wed + high prior-range, 85% target"),
    ]

    rows = [eval_combo(m, t, lbl) for m, t, lbl in combos]
    RES = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print("\n=== IMPROVEMENT TEST SUITE (all filters knowable at entry time -- no look-ahead) ===")
    print(RES.to_string(index=False))

    baseline_total = RES.iloc[0]["total_return"]; baseline_avg = RES.iloc[0]["avg_pnl"]
    RES["improvement_vs_baseline_total"] = (RES["total_return"] - baseline_total).round(1)
    RES["improvement_vs_baseline_avg_pct"] = ((RES["avg_pnl"] / baseline_avg - 1) * 100).round(1)

    with pd.ExcelWriter(OUTDIR / "improvement_analysis.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "All filters tested here are knowable AT ENTRY TIME (direction, entry weekday, regime, "
                      "prior-3-day range) -- no look-ahead. The earlier 'trending vs range-bound WEEK' cut from "
                      "the original report used the week's ACTUAL entry-to-exit range, which is only known in "
                      "hindsight and is NOT reproduced here as an entry filter for that reason."},
            {"note": f"median prior-3-day range across all {len(TP)} trades: {round(median_range,1)} points -- "
                      "used as the entry-time-knowable proxy for an already-elevated-volatility setup."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        RES.to_excel(w, sheet_name="Improvement_Tests", index=False)
        TP.drop(columns=["pnl_series"]).to_excel(w, sheet_name="Trade_Metadata", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 40)

    print("\n=== IMPROVEMENT vs BASELINE ===")
    print(RES[["filter", "target_pct", "n", "total_return", "improvement_vs_baseline_total", "avg_pnl", "improvement_vs_baseline_avg_pct", "win_rate_pct"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/improvement_analysis.xlsx")


if __name__ == "__main__":
    main()
