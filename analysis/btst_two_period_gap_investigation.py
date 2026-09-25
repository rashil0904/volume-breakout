# -*- coding: utf-8 -*-
"""btst_two_period_gap_investigation.py — investigates WHY the NIFTY BTST strategy's Period 1
(Mar2022-Sep2024, spot-only proxy) and Period 2 (Oct2024-Jul2026, actual options strategy) perform
differently, using the DTE-1-EXCLUDED + VIX[17,19]-FILTERED config confirmed better for both. Diagnostic
only -- read-only against all source data, writes to its own new results folder.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "btst_spot_proxy_pre2024"; OUTDIR.mkdir(parents=True, exist_ok=True)
SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
ENTRY_MOD = 15 * 60 + 20; EXIT_MOD = 9 * 60 + 17
P1_START, P1_END = pd.Timestamp("2022-03-01"), pd.Timestamp("2024-09-30")
P2_START, P2_END = pd.Timestamp("2024-10-01"), pd.Timestamp("2026-07-31")


def abs_move_bucket(v):
    if v < 50: return "<50"
    if v < 100: return "50-100"
    if v < 200: return "100-200"
    return "200+"


def main():
    # ---- source data ----
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    daily_close = sp[sp["mod"] == sp.groupby("date")["mod"].transform("max")].groupby("date")["close"].last()  # last candle of day (~15:29)

    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    print(f"VIX data availability check: {vx['date'].min().date()} .. {vx['date'].max().date()}")
    vix_p1_ok = vx["date"].min() <= P1_START
    print(f"VIX covers Period 1 start ({P1_START.date()})? {'YES' if vix_p1_ok else 'NO -- FLAG'}")

    # ---- Period 1: spot proxy, all 635 days (with exclusion flags already computed) ----
    ALL1 = pd.read_excel(OUTDIR / "nifty_btst_spot_proxy_pre2024.xlsx", sheet_name="Per_Day_Full_Log")
    ALL1["entry_date"] = pd.to_datetime(ALL1["entry_date"])
    Q1 = ALL1[(~ALL1["excluded_vix_filter"]) & (~ALL1["excluded_dte1_filter"])].copy()   # confirmed-better config
    Q1["abs_move"] = (Q1["exit_spot_0917"] - Q1["entry_spot_1520"]).abs()

    # ---- Period 2: actual strategy, DTE-1 excluded + VIX filtered (= FINAL v3 exactly) ----
    T2 = pd.read_csv(rb.RESULTS / "btst_close_direction_FINAL" / "btst_close_direction_FINAL_v3_trades.csv")
    T2["entry_date"] = pd.to_datetime(T2["entry_date"]); T2["exit_date"] = pd.to_datetime(T2["exit_date"])
    spot_entry_map = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    spot_exit_map = sp[sp["mod"] == EXIT_MOD].groupby("date")["close"].last()
    T2["entry_spot"] = T2["entry_date"].map(spot_entry_map); T2["exit_spot"] = T2["exit_date"].map(spot_exit_map)
    T2 = T2.dropna(subset=["entry_spot", "exit_spot"]).copy()
    T2["abs_move"] = (T2["exit_spot"] - T2["entry_spot"]).abs()
    T2["dir_consistent_spot_move"] = np.where(T2["direction"] == "GREEN", T2["exit_spot"] - T2["entry_spot"], -(T2["exit_spot"] - T2["entry_spot"]))
    T2["favorable"] = T2["dir_consistent_spot_move"] > 0

    # ---- full (unfiltered) pools for filter-impact % ----
    total_days_p1 = len(ALL1)
    vix_excl_p1 = int(ALL1["excluded_vix_filter"].sum()); dte1_excl_p1 = int(ALL1["excluded_dte1_filter"].sum())

    T2_unf = pd.read_csv(rb.RESULTS / "btst_dte1_included_variant" / "nifty_btst_dte1_included_UNFILTERED_trades.csv")
    total_days_p2 = len(T2_unf)
    vix_excl_p2 = int(((T2_unf["entry_vix"] >= 17.0) & (T2_unf["entry_vix"] <= 19.0)).sum())
    dte1_excl_p2 = int((T2_unf["DTE"] == 1).sum())

    # ---- 1. avg absolute movement ----
    def abs_move_stats(df, col="abs_move"):
        s = df[col]
        dist = s.apply(abs_move_bucket).value_counts(normalize=True).reindex(["<50", "50-100", "100-200", "200+"]).fillna(0) * 100
        return round(s.mean(), 2), round(s.median(), 2), dist.round(1)

    p1_avg_abs, p1_med_abs, p1_dist = abs_move_stats(Q1)
    p2_avg_abs, p2_med_abs, p2_dist = abs_move_stats(T2)

    # ---- 2. win rate ----
    p1_win = round(Q1["favorable"].mean() * 100, 2)
    p2_win = round(T2["favorable"].mean() * 100, 2)

    # ---- 3. avg VIX ----
    p1_avg_vix = round(Q1["entry_vix"].mean(), 2)
    p2_avg_vix = round(T2["entry_vix"].mean(), 2)

    # ---- 4. trend/drift: NIFTY cumulative % change, period start to end ----
    def trend(start, end):
        window = daily_close[(daily_close.index >= start) & (daily_close.index <= end)]
        return round((window.iloc[-1] / window.iloc[0] - 1) * 100, 2), window.index.min(), window.index.max(), window.iloc[0], window.iloc[-1]

    p1_trend, p1_td0, p1_td1, p1_c0, p1_c1 = trend(P1_START, P1_END)
    p2_trend, p2_td0, p2_td1, p2_c0, p2_c1 = trend(P2_START, P2_END)

    # ---- 5. day-of-week / month pattern ----
    Q1["weekday"] = Q1["entry_date"].dt.day_name(); Q1["month_num"] = Q1["entry_date"].dt.month
    T2["weekday"] = T2["entry_date"].dt.day_name(); T2["month_num"] = T2["entry_date"].dt.month
    dow_p1 = Q1.groupby("weekday")["favorable"].agg(["mean", "size"]); dow_p1["mean"] = (dow_p1["mean"] * 100).round(1)
    dow_p2 = T2.groupby("weekday")["favorable"].agg(["mean", "size"]); dow_p2["mean"] = (dow_p2["mean"] * 100).round(1)
    moy_p1 = Q1.groupby("month_num")["favorable"].agg(["mean", "size"]); moy_p1["mean"] = (moy_p1["mean"] * 100).round(1)
    moy_p2 = T2.groupby("month_num")["favorable"].agg(["mean", "size"]); moy_p2["mean"] = (moy_p2["mean"] * 100).round(1)

    # ---- summary table ----
    SUMMARY = pd.DataFrame([
        {"metric": "n_trades (qualifying config)", "Period 1 (spot proxy)": len(Q1), "Period 2 (actual strategy)": len(T2)},
        {"metric": "avg_abs_movement_pts", "Period 1 (spot proxy)": p1_avg_abs, "Period 2 (actual strategy)": p2_avg_abs},
        {"metric": "median_abs_movement_pts", "Period 1 (spot proxy)": p1_med_abs, "Period 2 (actual strategy)": p2_med_abs},
        {"metric": "win_rate_pct (favorable spot move)", "Period 1 (spot proxy)": p1_win, "Period 2 (actual strategy)": p2_win},
        {"metric": "avg_entry_VIX", "Period 1 (spot proxy)": p1_avg_vix, "Period 2 (actual strategy)": p2_avg_vix},
        {"metric": "NIFTY_period_trend_pct", "Period 1 (spot proxy)": p1_trend, "Period 2 (actual strategy)": p2_trend},
        {"metric": "NIFTY_start_close", "Period 1 (spot proxy)": round(p1_c0, 1), "Period 2 (actual strategy)": round(p2_c0, 1)},
        {"metric": "NIFTY_end_close", "Period 1 (spot proxy)": round(p1_c1, 1), "Period 2 (actual strategy)": round(p2_c1, 1)},
        {"metric": "total_trading_day_pairs_in_window", "Period 1 (spot proxy)": total_days_p1, "Period 2 (actual strategy)": total_days_p2},
        {"metric": "n_excluded_by_VIX_filter", "Period 1 (spot proxy)": vix_excl_p1, "Period 2 (actual strategy)": vix_excl_p2},
        {"metric": "pct_excluded_by_VIX_filter", "Period 1 (spot proxy)": round(vix_excl_p1 / total_days_p1 * 100, 2), "Period 2 (actual strategy)": round(vix_excl_p2 / total_days_p2 * 100, 2)},
        {"metric": "n_excluded_by_DTE1_filter", "Period 1 (spot proxy)": dte1_excl_p1, "Period 2 (actual strategy)": dte1_excl_p2},
        {"metric": "pct_excluded_by_DTE1_filter", "Period 1 (spot proxy)": round(dte1_excl_p1 / total_days_p1 * 100, 2), "Period 2 (actual strategy)": round(dte1_excl_p2 / total_days_p2 * 100, 2)},
        {"metric": "total_points (Period1=spot pts proxy, Period2=REAL option P&L)", "Period 1 (spot proxy)": round(Q1["direction_consistent_move_pts"].sum(), 1), "Period 2 (actual strategy)": round(T2["pnl_points"].sum(), 1)},
        {"metric": "avg_points_per_trade (same caveat)", "Period 1 (spot proxy)": round(Q1["direction_consistent_move_pts"].mean(), 2), "Period 2 (actual strategy)": round(T2["pnl_points"].mean(), 2)},
    ])

    pd.set_option("display.width", 260)
    print("\n" + "=" * 110 + "\nSIDE-BY-SIDE SUMMARY\n" + "=" * 110)
    print(SUMMARY.to_string(index=False))

    print("\n--- ABS MOVEMENT DISTRIBUTION ---")
    print("Period 1:", p1_dist.to_dict())
    print("Period 2:", p2_dist.to_dict())

    print("\n--- DAY-OF-WEEK favorable% (n) ---")
    print("Period 1:\n", dow_p1.to_string())
    print("Period 2:\n", dow_p2.to_string())

    print("\n--- MONTH-OF-YEAR favorable% (n) ---")
    print("Period 1:\n", moy_p1.to_string())
    print("Period 2:\n", moy_p2.to_string())

    with pd.ExcelWriter(OUTDIR / "btst_two_period_gap_investigation.xlsx", engine="openpyxl") as w:
        SUMMARY.to_excel(w, sheet_name="Summary_Comparison", index=False)
        pd.DataFrame({"Period_1": p1_dist, "Period_2": p2_dist}).to_excel(w, sheet_name="Abs_Movement_Distribution")
        dow_p1.reset_index().to_excel(w, sheet_name="DOW_Period1", index=False)
        dow_p2.reset_index().to_excel(w, sheet_name="DOW_Period2", index=False)
        moy_p1.reset_index().to_excel(w, sheet_name="MOY_Period1", index=False)
        moy_p2.reset_index().to_excel(w, sheet_name="MOY_Period2", index=False)
        Q1.to_excel(w, sheet_name="Period1_Trade_Detail", index=False)
        T2.to_excel(w, sheet_name="Period2_Trade_Detail", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)

    print(f"\nSaved -> {OUTDIR}/btst_two_period_gap_investigation.xlsx")


if __name__ == "__main__":
    main()
