# -*- coding: utf-8 -*-
"""combined_4strategy_95L_basket.py — combines 4 already-computed strategies into one Rs 95L basket:
  1. Volume Breakout (actual, net_A) -- Rs 5L basis -> SCALED x5 to Rs 25L
  2. NIFTY BTST Close Direction (synthetic) -- Rs 5L basis -> SCALED x5 to Rs 25L
  3. NIFTY Weekly Credit Spread (synthetic) -- Rs 5L basis -> SCALED x5 to Rs 25L
  4. AlgoTest selling strategy -- ALREADY on Rs 20L basis -- used AS-IS, no further scaling
Total: 25+25+25+20 = Rs 95L. Window: April 2023 - July 2026. Pure extraction/combination -- no new
backtesting, no re-scaling of anything already on its target basis.

April 2023's AlgoTest figure is a PROXY (filled with Oct-2023's actual value) -- flagged explicitly
throughout. Strategies 1-3's April 2023 figures are genuine computed data (confirmed in the earlier
3-strategy extraction).

DRAWDOWN CONVENTION: combined drawdown is computed from the ACTUAL COMBINED equity curve (sum of all 4
strategies' scaled cumulative curves, month by month), NOT by summing each strategy's own individual max
drawdown -- since drawdowns don't necessarily occur in the same month across strategies, and one
strategy's gain can offset another's loss in the same month.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC_3STRAT = rb.RESULTS / "combined_3strategy_extract" / "combined_3strategy_monthly_fy_extract.xlsx"
SRC_ALGOTEST = rb.RESULTS / "algotest_selling_strategy" / "algotest_selling_strategy_report.xlsx"
OUTDIR = rb.RESULTS / "combined_4strategy_95L_basket"; OUTDIR.mkdir(parents=True, exist_ok=True)
SCALE = 5

FY_BOUNDS = [
    ("FY23-24", "2023-04", "2024-03"),
    ("FY24-25", "2024-04", "2025-03"),
    ("FY25-26", "2025-04", "2026-03"),
    ("FY26-27", "2026-04", "2026-07"),
]


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
    THREE = pd.read_excel(SRC_3STRAT, sheet_name="Monthly")
    ALGO = pd.read_excel(SRC_ALGOTEST, sheet_name="Monthly_Data")

    piv = THREE.pivot_table(index="Month", columns="Strategy", values="Monthly_PnL_INR", aggfunc="first")
    piv = piv.rename(columns={
        "Volume Breakout (net_B, actual)": "S1_VolumeBreakout",
        "NIFTY BTST Close Direction (synthetic)": "S2_BTST_synthetic",
        "NIFTY Weekly Credit Spread (synthetic)": "S3_CreditSpread_synthetic",
    })
    for c in ["S1_VolumeBreakout", "S2_BTST_synthetic", "S3_CreditSpread_synthetic"]:
        piv[c] = piv[c] * SCALE  # scale to Rs 25L each

    algo_s = ALGO.set_index("Month")["PnL_INR"].rename("S4_AlgoTest_20L")
    algo_proxy = ALGO.set_index("Month")["Is_Proxy"].rename("S4_Is_Proxy")

    COMB = piv.join(algo_s).join(algo_proxy).reset_index()
    COMB = COMB.sort_values("Month").reset_index(drop=True)
    missing = COMB[COMB.isna().any(axis=1)]
    if len(missing):
        print("WARNING -- months with missing data in one or more sources (excluded from combined sums):")
        print(missing.to_string(index=False))
    COMB = COMB.dropna(subset=["S1_VolumeBreakout", "S2_BTST_synthetic", "S3_CreditSpread_synthetic", "S4_AlgoTest_20L"]).reset_index(drop=True)

    COMB["Combined_Total_PnL_INR"] = (COMB["S1_VolumeBreakout"] + COMB["S2_BTST_synthetic"] +
                                        COMB["S3_CreditSpread_synthetic"] + COMB["S4_AlgoTest_20L"])
    COMB["Combined_Cumulative_PnL_INR"] = COMB["Combined_Total_PnL_INR"].cumsum()
    peak = np.maximum.accumulate(COMB["Combined_Cumulative_PnL_INR"].values)
    COMB["Combined_Running_Drawdown_INR"] = COMB["Combined_Cumulative_PnL_INR"].values - peak

    for c in ["S1_VolumeBreakout", "S2_BTST_synthetic", "S3_CreditSpread_synthetic", "S4_AlgoTest_20L",
              "Combined_Total_PnL_INR", "Combined_Cumulative_PnL_INR", "Combined_Running_Drawdown_INR"]:
        COMB[c] = COMB[c].round(1)

    pd.set_option("display.width", 220)
    print("=== COMBINED MONTHLY (first 6 rows) ===")
    print(COMB.head(6).to_string(index=False))
    print(f"\ntotal months: {len(COMB)} | final cumulative: {COMB['Combined_Cumulative_PnL_INR'].iloc[-1]:,.0f}")

    proxy_row = COMB[COMB["S4_Is_Proxy"] == "Y"]
    print(f"\nPROXY flag: {len(proxy_row)} month(s) contain a proxied input -> {proxy_row['Month'].tolist()}")

    # ---- FY summary ----
    fy_rows = []
    for fy_label, start, end in FY_BOUNDS:
        Mfy = COMB[(COMB["Month"] >= start) & (COMB["Month"] <= end)].reset_index(drop=True)
        if Mfy.empty:
            continue
        cum = Mfy["Combined_Total_PnL_INR"].cumsum().values
        maxdd, avgdd = episode_stats(cum)
        n_win = int((Mfy["Combined_Total_PnL_INR"] > 0).sum()); n_lose = int((Mfy["Combined_Total_PnL_INR"] <= 0).sum())
        has_proxy = (Mfy["S4_Is_Proxy"] == "Y").any()
        fy_rows.append({"FY": fy_label, "n_months": len(Mfy), "Total_Combined_Return_INR": round(float(cum[-1]), 1),
                         "Max_Drawdown_INR": round(maxdd, 1), "Avg_Drawdown_INR": round(avgdd, 1),
                         "Winning_Months": n_win, "Losing_Months": n_lose,
                         "Contains_Proxy_Month": "YES (Apr-23)" if has_proxy else "No"})
    FY = pd.DataFrame(fy_rows)
    print("\n=== COMBINED FY SUMMARY ===")
    print(FY.to_string(index=False))

    # ---- save Excel ----
    out_xlsx = OUTDIR / "combined_4strategy_95L_basket.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "Combined Rs 95L basket: Volume Breakout (Rs 25L, actual net_A x5), NIFTY BTST synthetic "
                      "(Rs 25L, x5), NIFTY Weekly Credit Spread synthetic (Rs 25L, x5), AlgoTest selling strategy "
                      "(Rs 20L, used AS-IS, no further scaling). Window: April 2023 - July 2026."},
            {"note": "*** April 2023 (2023-04) contains a PROXY input from the AlgoTest strategy (filled with "
                      "Oct-2023's actual figure) -- see S4_Is_Proxy column. Strategies 1-3's April 2023 figures "
                      "are genuine computed data, not proxied. ***"},
            {"note": "Combined_Running_Drawdown_INR is computed from the ACTUAL COMBINED equity curve (sum of all "
                      "4 strategies month by month), NOT by summing each strategy's own individual max drawdown."},
            {"note": "FY-level Max/Avg_Drawdown_INR reset the equity curve to 0 at each FY's start (same convention "
                      "as each individual strategy's own FY report)."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        COMB.to_excel(w, sheet_name="Combined_Monthly", index=False)
        FY.to_excel(w, sheet_name="Combined_FY_Summary", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    print(f"\nSaved -> {out_xlsx}")

    COMB.to_csv(OUTDIR / "combined_4strategy_monthly.csv", index=False)
    FY.to_csv(OUTDIR / "combined_4strategy_FY_summary.csv", index=False)

    # ---- combined cumulative equity curve graph, full window ----
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(COMB["Month"], COMB["Combined_Cumulative_PnL_INR"], color="#1f6feb", linewidth=1.8)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    proxy_pts = COMB[COMB["S4_Is_Proxy"] == "Y"]
    if len(proxy_pts):
        ax.scatter(proxy_pts["Month"], proxy_pts["Combined_Cumulative_PnL_INR"], color="red", s=90, zorder=5,
                   label="Month contains AlgoTest PROXY input (Apr'23)")
        ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Combined 4-Strategy Portfolio — Rs 95L Basket\n"
                  "(Volume Breakout Rs25L + NIFTY BTST synthetic Rs25L + Credit Spread synthetic Rs25L + AlgoTest Rs20L)\n"
                  "April 2023 - July 2026", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Month", fontsize=11); ax.set_ylabel("Cumulative Return (INR)", fontsize=11)
    ax.grid(True, alpha=0.3)
    step = max(1, len(COMB) // 18)
    ax.set_xticks(COMB["Month"][::step])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out_png = OUTDIR / "combined_4strategy_95L_equity_curve.png"
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
