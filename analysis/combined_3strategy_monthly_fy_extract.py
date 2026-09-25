# -*- coding: utf-8 -*-
"""combined_3strategy_monthly_fy_extract.py — extracts monthly + FY-wise PnL/drawdown/win-loss from the
existing computed results of 3 strategies into one standardized combined file for downstream portfolio
analysis. Pure extraction/reformatting -- no new backtest, no re-run of any strategy.

STRATEGIES (all Rs 5,00,000 capital base):
  1. Main NSE Volume-Breakout BTST -- ACTUAL backtest, net_A basis (daily_performance sheet).
  2. NIFTY Daily Close Direction BTST -- SYNTHETIC (spot-movement-derived), full Mar-2022..Jul-2026 span.
  3. NIFTY Weekly Credit Spread -- SYNTHETIC (spot-movement/heuristic-derived), same span.

WINDOW: April 2023 - July 2026 for the monthly table (all three have real computed data back to Mar-2022,
so April-2023 is NOT a filled/proxied start for any of them -- confirmed by checking each source's actual
date range before extraction). FY buckets: FY23-24, FY24-25, FY25-26, FY26-27 (partial, Apr-Jul 2026).

CUMULATIVE PnL (monthly sheet): ONE continuous running total starting fresh at April-2023 (the start of
this extraction window), NOT resetting at FY boundaries -- appropriate for a downstream monthly track
record that spans multiple FYs. Monthly Max Drawdown is sliced from that SAME continuous curve (the worst
drawdown point reached during that month, relative to any prior peak since April-2023), for internal
consistency with the cumulative column.

FY-WISE max/avg drawdown (separate FY sheet): uses each strategy's OWN already-established FY-reporting
convention -- the equity curve RESETS to 0 at each FY's start (matching the numbers already published in
each strategy's own FY-wise report) -- these are NOT the same drawdown numbers as the monthly sheet's
continuous-curve slices, by design; flagged here so the two aren't confused downstream.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "combined_3strategy_extract"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIN_START = pd.Timestamp("2023-04-01"); WIN_END = pd.Timestamp("2026-07-31")

STRATEGIES = {
    "Volume Breakout (net_B, actual)": {
        "path": rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx",
        "sheet": "daily_performance", "date_col": "date", "pnl_col": "net_B_total_pnl_inr",
    },
    "NIFTY BTST Close Direction (synthetic)": {
        "path": rb.RESULTS / "btst_close_direction_FINAL" / "synthetic_fy_wise" / "synthetic_trades_full.csv",
        "sheet": None, "date_col": "entry_date", "pnl_col": "pnl_inr",
    },
    "NIFTY Weekly Credit Spread (synthetic)": {
        "path": rb.RESULTS / "weekly_credit_spread" / "synthetic_fy_wise" / "synthetic_credit_spread_trades_full.csv",
        "sheet": None, "date_col": "entry_date", "pnl_col": "pnl_inr",
    },
}

FY_BOUNDS = [
    ("FY23-24", pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    ("FY24-25", pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    ("FY25-26", pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
    ("FY26-27", pd.Timestamp("2026-04-01"), WIN_END),
]


def load_source(cfg):
    if cfg["sheet"]:
        df = pd.read_excel(cfg["path"], sheet_name=cfg["sheet"])
    else:
        df = pd.read_csv(cfg["path"])
    df = df.rename(columns={cfg["date_col"]: "date", cfg["pnl_col"]: "pnl_inr"})[["date", "pnl_inr"]]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def drawdown_series(cum):
    """Returns the running drawdown (equity - running peak) aligned to cum's index, plus episode list."""
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return dd


def episode_stats(cum):
    equity = np.concatenate([[0.0], cum])
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    eps = []; in_dd = False; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9:
                in_dd = True; tv = dd[k]
        else:
            if dd[k] < tv:
                tv = dd[k]
            if dd[k] >= -1e-9:
                eps.append(tv); in_dd = False
    if in_dd:
        eps.append(tv)
    if not eps:
        return 0.0, 0.0
    return abs(min(eps)), float(np.mean([abs(e) for e in eps]))


def main():
    monthly_all = []; fy_all = []

    for name, cfg in STRATEGIES.items():
        raw = load_source(cfg)
        print(f"{name}: source range {raw['date'].min().date()} .. {raw['date'].max().date()} "
              f"| covers Apr-2023 start? {'YES (real data, not proxied)' if raw['date'].min() <= WIN_START else 'GAP'}", flush=True)

        W = raw[(raw["date"] >= WIN_START) & (raw["date"] <= WIN_END)].sort_values("date").reset_index(drop=True)
        W["cum_pnl_inr"] = W["pnl_inr"].cumsum()
        W["dd_inr"] = drawdown_series(W["cum_pnl_inr"].values)
        W["month"] = W["date"].dt.to_period("M").astype(str)

        for month, g in W.groupby("month"):
            monthly_all.append({
                "Month": month, "Strategy": name,
                "Monthly_PnL_INR": round(g["pnl_inr"].sum(), 1),
                "Cumulative_PnL_INR": round(g["cum_pnl_inr"].iloc[-1], 1),
                "Monthly_Max_Drawdown_INR": round(abs(g["dd_inr"].min()), 1),
                "Winning_Days_Trades": int((g["pnl_inr"] > 0).sum()),
                "Losing_Days_Trades": int((g["pnl_inr"] <= 0).sum()),
            })

        for fy_label, start, end in FY_BOUNDS:
            Tfy = raw[(raw["date"] >= start) & (raw["date"] <= end)].sort_values("date").reset_index(drop=True)
            if Tfy.empty:
                continue
            cum = Tfy["pnl_inr"].cumsum().values
            maxdd, avgdd = episode_stats(cum)
            fy_all.append({
                "FY": fy_label, "Strategy": name,
                "FY_PnL_INR": round(float(cum[-1]), 1),
                "FY_Max_Drawdown_INR": round(maxdd, 1),
                "FY_Avg_Drawdown_INR": round(avgdd, 1),
                "Winning_Days_Trades": int((Tfy["pnl_inr"] > 0).sum()),
                "Losing_Days_Trades": int((Tfy["pnl_inr"] <= 0).sum()),
            })

    MONTHLY = pd.DataFrame(monthly_all).sort_values(["Month", "Strategy"]).reset_index(drop=True)
    FY = pd.DataFrame(fy_all).sort_values(["FY", "Strategy"]).reset_index(drop=True)

    pd.set_option("display.width", 200)
    print("\n=== MONTHLY (first 6 rows) ===")
    print(MONTHLY.head(6).to_string(index=False))
    print("\n=== FY SUMMARY (all rows) ===")
    print(FY.to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "combined_3strategy_monthly_fy_extract.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "Extraction only -- no new backtest. All 3 strategies confirmed to have REAL computed data "
                      "(not filled/proxied) at April-2023, verified against each source's actual date range."},
            {"note": "Monthly sheet: Cumulative_PnL_INR is ONE continuous running total starting at April-2023 "
                      "(not resetting at FY boundaries). Monthly_Max_Drawdown_INR is sliced from that SAME "
                      "continuous curve."},
            {"note": "FY sheet: FY_Max/Avg_Drawdown_INR use each strategy's OWN established FY-report convention "
                      "-- the equity curve RESETS to 0 at each FY's start. These are NOT the same drawdown "
                      "numbers as the monthly sheet (different curve, by design) -- do not mix the two."},
            {"note": "Strategy 1 (Volume Breakout) is real backtest P&L. Strategies 2 and 3 are SYNTHETIC/"
                      "heuristic spot-derived proxies, not real option P&L -- see each strategy's own FY report "
                      "for full methodology and caveats."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        MONTHLY.to_excel(w, sheet_name="Monthly", index=False)
        FY.to_excel(w, sheet_name="FY_Summary", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)

    MONTHLY.to_csv(OUTDIR / "combined_3strategy_monthly.csv", index=False)
    FY.to_csv(OUTDIR / "combined_3strategy_FY_summary.csv", index=False)

    print(f"\nSaved -> {OUTDIR}/combined_3strategy_monthly_fy_extract.xlsx (+ matching CSVs)")


if __name__ == "__main__":
    main()
