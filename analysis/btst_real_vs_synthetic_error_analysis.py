# -*- coding: utf-8 -*-
"""btst_real_vs_synthetic_error_analysis.py — validates the 0.7x spot-to-option synthetic conversion
against the REAL FINAL v3 options-based P&L, day by day, Oct-2024..Jul-2026. Compares at the RAW OPTION-
POINTS level (no 1.5pt expense, no 455 INR multiplier) -- isolates the accuracy of the 0.7x factor itself.
Diagnostic only -- no changes to either strategy's logic.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "btst_close_direction_FINAL" / "real_vs_synthetic_validation"; OUTDIR.mkdir(parents=True, exist_ok=True)
NEAR_ZERO_THRESH = 1.0  # |real_pnl_points| below this -> flag instead of computing error %


def main():
    T = pd.read_csv(rb.RESULTS / "btst_close_direction_FINAL" / "btst_close_direction_FINAL_v3_trades.csv")
    T["entry_date"] = pd.to_datetime(T["entry_date"]); T["exit_date"] = pd.to_datetime(T["exit_date"])

    sp = pd.read_csv(rb.BASE / "data" / "nifty_1min_ohlc.csv", parse_dates=["timestamp"])
    ts = sp["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_entry = sp[sp["mod"] == 15 * 60 + 20].groupby("date")["close"].last()
    spot_exit = sp[sp["mod"] == 9 * 60 + 17].groupby("date")["close"].last()

    T["entry_spot"] = T["entry_date"].map(spot_entry); T["exit_spot"] = T["exit_date"].map(spot_exit)
    missing = T[T["entry_spot"].isna() | T["exit_spot"].isna()]
    print(f"total FINAL v3 trades: {len(T)} | unmatched to spot data (excluded): {len(missing)}", flush=True)
    T = T.dropna(subset=["entry_spot", "exit_spot"]).reset_index(drop=True)

    T["spot_move"] = T["exit_spot"] - T["entry_spot"]
    T["dir_consistent_spot_move"] = np.where(T["direction"] == "GREEN", T["spot_move"], -T["spot_move"])
    T["synthetic_pnl_points"] = T["dir_consistent_spot_move"] * 0.7
    T["real_pnl_points"] = T["pnl_points"]
    T["abs_error"] = (T["real_pnl_points"] - T["synthetic_pnl_points"]).abs()
    T["signed_error"] = T["synthetic_pnl_points"] - T["real_pnl_points"]  # + = synthetic overestimates

    near_zero = T["real_pnl_points"].abs() < NEAR_ZERO_THRESH
    T["near_zero_flag"] = near_zero
    T["error_pct"] = np.where(near_zero, np.nan, T["abs_error"] / T["real_pnl_points"].abs() * 100)

    DAY = T[["entry_date", "direction", "DTE_original", "real_pnl_points", "synthetic_pnl_points",
             "abs_error", "error_pct", "signed_error", "near_zero_flag"]].sort_values("entry_date")
    DAY.to_csv(OUTDIR / "real_vs_synthetic_daily.csv", index=False)

    valid = T[~near_zero]
    print(f"\nvalid (non-near-zero) trades for error%: {len(valid)} | near-zero flagged: {near_zero.sum()}")

    def dist_bucket(pct):
        if pct < 10: return "<10%"
        if pct < 25: return "10-25%"
        if pct < 50: return "25-50%"
        return "50%+"

    valid = valid.copy(); valid["bucket"] = valid["error_pct"].apply(dist_bucket)
    order = ["<10%", "10-25%", "25-50%", "50%+"]

    def summarize(df, label):
        n = len(df)
        if n == 0:
            print(f"{label}: no trades"); return {}
        avg_pct = round(df["error_pct"].mean(), 2); med_pct = round(df["error_pct"].median(), 2)
        std_pct = round(df["error_pct"].std(), 2)
        avg_signed = round(df["signed_error"].mean(), 2)
        dist = df["bucket"].value_counts(normalize=True).reindex(order).fillna(0) * 100
        print(f"\n--- {label} (n={n}) ---")
        print(f"avg error%={avg_pct} | median error%={med_pct} | std error%={std_pct} | avg signed error={avg_signed}")
        print(f"distribution: " + " | ".join(f"{b}: {round(dist[b],1)}%" for b in order))
        return {"label": label, "n": n, "avg_error_pct": avg_pct, "median_error_pct": med_pct,
                "std_error_pct": std_pct, "avg_signed_error": avg_signed,
                **{f"pct_{b}": round(dist[b], 1) for b in order}}

    overall = summarize(valid, "OVERALL")
    red = summarize(valid[valid.direction == "RED"], "RED")
    green = summarize(valid[valid.direction == "GREEN"], "GREEN")

    dte_rows = []
    for dte in sorted(valid["DTE_original"].unique()):
        dte_rows.append(summarize(valid[valid.DTE_original == dte], f"DTE={dte}"))

    SUMM = pd.DataFrame([overall, red, green] + dte_rows)

    with pd.ExcelWriter(OUTDIR / "real_vs_synthetic_error_analysis.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "Validates the 0.7x spot-to-option synthetic conversion against REAL FINAL v3 option P&L, "
                      "at raw option-points level (no 1.5pt expense, no 455 INR multiplier)."},
            {"note": f"n_trades_matched={len(T)} | n_unmatched_excluded={len(missing)} | "
                      f"n_near_zero_flagged (|real|<{NEAR_ZERO_THRESH}pt, error% not computed)={near_zero.sum()}"},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        DAY.to_excel(w, sheet_name="Day_By_Day", index=False)
        SUMM.to_excel(w, sheet_name="Summary_Stats", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)

    pd.set_option("display.width", 220)
    print("\n=== FULL SUMMARY TABLE ===")
    print(SUMM.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/real_vs_synthetic_error_analysis.xlsx")


if __name__ == "__main__":
    main()
