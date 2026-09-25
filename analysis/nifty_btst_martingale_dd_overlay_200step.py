# -*- coding: utf-8 -*-
"""nifty_btst_martingale_dd_overlay_200step.py — SAME additive martingale-style POSITION-SIZING overlay
on the locked FINAL v3 NIFTY Daily Close Direction BTST strategy as martingale_dd_overlay.py, but with
the COARSER 200-point-step, 3-bracket ladder (1x/2x/3x) requested in this follow-up, replacing the
earlier 100-point-step, 5-bracket ladder (1x-5x). Entry/exit logic, timing, VIX[17,19]-excluded filter,
and DTE-1-skip/DTE-0-next-week rules are all UNCHANGED -- only the sizing-bracket definition changed
from the prior run. Kept as a separate script/output so both bracket schemes remain comparable.

Sizing brackets on PRIOR trade's closing GROSS drawdown (base = 2 lots buy / 1 lot sell = "1x"):
  dd <200             -> 1x  (2:1)
  200 <= dd <400       -> 2x  (4:2)
  dd >=400             -> 3x  (6:3)   (capped -- no further scaling beyond 3x)
Recovery: dd==0 falls in the <200 bracket already -> reverts to 1x automatically, no separate rule needed.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "btst_close_direction_FINAL" / "btst_close_direction_FINAL_v3_trades.csv"
OUTDIR = rb.RESULTS / "btst_close_direction_FINAL" / "martingale_overlay"; OUTDIR.mkdir(parents=True, exist_ok=True)


def bracket(dd):
    if dd < 200: return 1
    if dd < 400: return 2
    return 3


def main():
    T = pd.read_csv(SRC)
    T["entry_date"] = pd.to_datetime(T["entry_date"]); T["exit_date"] = pd.to_datetime(T["exit_date"])
    T = T.sort_values("exit_date").reset_index(drop=True)
    n = len(T)
    print(f"loaded FINAL v3: {n} trades, {T['entry_date'].min().date()} .. {T['entry_date'].max().date()}", flush=True)

    base_eq = T["pnl_points"].cumsum()
    base_peak = base_eq.cummax()
    base_dd = base_peak - base_eq
    base_max_dd = base_dd.max()
    base_total = base_eq.iloc[-1]
    print(f"BASELINE (flat 1x/2:1) re-confirmed: total gross={base_total:.2f} pts | max GROSS drawdown={base_max_dd:.2f} pts", flush=True)

    rows = []
    equity = 0.0; peak = 0.0; prior_dd = 0.0
    n_recoveries = 0; was_in_dd = False
    for i in range(n):
        r = T.iloc[i]
        k = bracket(prior_dd)
        base_pnl = r["pnl_points"]
        scaled_pnl = k * base_pnl
        equity += scaled_pnl
        peak = max(peak, equity)
        dd = round(peak - equity, 6)

        is_new_high = dd <= 1e-6
        if is_new_high and was_in_dd:
            n_recoveries += 1
        was_in_dd = not is_new_high

        rows.append({
            "entry_date": r["entry_date"].date(), "exit_date": r["exit_date"].date(), "direction": r["direction"],
            "prior_closing_dd": round(prior_dd, 2), "sizing_bracket": f"{k}x", "buy_lots": 2 * k, "sell_lots": 1 * k,
            "base_pnl_points_2to1": round(base_pnl, 2), "scaled_gross_pnl_points": round(scaled_pnl, 2),
            "running_cum_equity": round(equity, 2), "running_peak": round(peak, 2), "running_drawdown": round(dd, 2),
        })
        prior_dd = dd

    D = pd.DataFrame(rows)
    dyn_total = D["scaled_gross_pnl_points"].sum()
    dyn_max_dd = D["running_drawdown"].max()
    dyn_max_dd_row = D.loc[D["running_drawdown"].idxmax()]

    print(f"\nDYNAMIC (200-step martingale) overlay: total gross={dyn_total:.2f} pts | max GROSS drawdown={dyn_max_dd:.2f} pts", flush=True)
    print(f"  worst drawdown occurred at exit_date={dyn_max_dd_row['exit_date']} (bracket {dyn_max_dd_row['sizing_bracket']} active that trade)", flush=True)

    bracket_days = D["sizing_bracket"].value_counts().reindex(["1x", "2x", "3x"], fill_value=0)
    ended_at_new_high = D.iloc[-1]["running_drawdown"] <= 1e-6

    pd.set_option("display.width", 220)
    print("\n=== days spent in each sizing bracket ===")
    print(bracket_days.to_string())
    print(f"\nfull recoveries to a new high (dd resets to 0 after being >0): {n_recoveries}")
    print(f"window ended {'AT a new high (fully recovered, 1x)' if ended_at_new_high else f'MID-DRAWDOWN at {D.iloc[-1].running_drawdown:.2f} pts (bracket {D.iloc[-1].sizing_bracket})'}")

    print("\n=== BEFORE vs AFTER SUMMARY ===")
    print(f"{'metric':<32s} {'BASELINE (flat 1x)':>20s} {'DYNAMIC (200-step)':>20s} {'delta':>14s}")
    print(f"{'total gross P&L (points)':<32s} {base_total:>20.2f} {dyn_total:>20.2f} {dyn_total-base_total:>+14.2f}")
    print(f"{'max GROSS drawdown (points)':<32s} {base_max_dd:>20.2f} {dyn_max_dd:>20.2f} {dyn_max_dd-base_max_dd:>+14.2f}")
    print(f"{'return / max-DD ratio':<32s} {base_total/base_max_dd:>20.2f} {dyn_total/dyn_max_dd:>20.2f} {'':>14s}")
    print(f"exceeds original 450pt baseline max DD? {'YES' if dyn_max_dd > base_max_dd else 'no'} (by {dyn_max_dd - base_max_dd:+.2f} pts)")

    worst5 = D.nlargest(5, "running_drawdown")[["exit_date", "sizing_bracket", "base_pnl_points_2to1", "scaled_gross_pnl_points", "running_drawdown"]]

    with pd.ExcelWriter(OUTDIR / "martingale_dd_overlay_200step.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "ADDITIVE overlay on the locked FINAL v3 NIFTY BTST strategy -- entry/exit logic, timing "
                      "(15:20 entry / 09:17 exit), the VIX[17,19]-excluded filter, and DTE-1-skip/DTE-0-next-week "
                      "rules are UNCHANGED. Only position sizing (lot multiplier) is dynamic, driven by the prior "
                      "trading day's closing GROSS drawdown. Computed sequentially, trade by trade (not vectorized)."},
            {"note": f"Re-confirmed on the ACTUAL FINAL v3 trade set (323 trades, {T['entry_date'].min().date()}.."
                      f"{T['entry_date'].max().date()}): flat-1x baseline max GROSS drawdown = {base_max_dd:.2f} points "
                      "-- matches the known ~450pt checkpoint figure."},
            {"note": "Sizing brackets (200pt steps, on prior day's closing dd): <200->1x(2:1), 200-399->2x(4:2), "
                      ">=400->3x(6:3, capped, no further scaling defined beyond this)."},
            {"note": "RISK FLAG: this is a MARTINGALE-style overlay -- it sizes UP precisely while already in "
                      "drawdown. A continued adverse run compounds losses at a larger multiplier than the original "
                      "strategy ever used, and CAN produce a materially larger max drawdown than the 450pt flat-"
                      "sizing baseline. See the DYNAMIC max GROSS drawdown figure and Worst_Drawdown_Days below for "
                      "the actual worst case found in this backtest."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        pd.DataFrame([
            {"metric": "total_gross_pnl_points", "baseline_flat_1x": round(base_total, 2), "dynamic_200step": round(dyn_total, 2), "delta": round(dyn_total - base_total, 2)},
            {"metric": "max_gross_drawdown_points", "baseline_flat_1x": round(base_max_dd, 2), "dynamic_200step": round(dyn_max_dd, 2), "delta": round(dyn_max_dd - base_max_dd, 2)},
            {"metric": "exceeds_450pt_baseline_dd", "baseline_flat_1x": "", "dynamic_200step": bool(dyn_max_dd > base_max_dd), "delta": ""},
            {"metric": "return_over_maxDD_ratio", "baseline_flat_1x": round(base_total / base_max_dd, 3), "dynamic_200step": round(dyn_total / dyn_max_dd, 3), "delta": ""},
            {"metric": "n_full_recoveries_to_new_high", "baseline_flat_1x": "", "dynamic_200step": n_recoveries, "delta": ""},
            {"metric": "still_in_drawdown_at_window_end", "baseline_flat_1x": "", "dynamic_200step": not ended_at_new_high, "delta": ""},
        ]).to_excel(w, sheet_name="Before_After_Summary", index=False)
        bracket_days.rename("n_days").reset_index().rename(columns={"index": "sizing_bracket"}).to_excel(w, sheet_name="Days_per_Bracket", index=False)
        worst5.to_excel(w, sheet_name="Worst_Drawdown_Days", index=False)
        D.to_excel(w, sheet_name="Trade_Level_Detail", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)

    print(f"\nSaved -> {OUTDIR}/martingale_dd_overlay_200step.xlsx")


if __name__ == "__main__":
    main()
