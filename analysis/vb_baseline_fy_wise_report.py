# -*- coding: utf-8 -*-
"""vb_baseline_fy_wise_report.py — FY-wise (Indian FY, Apr-Mar) returns tables + graphs for the main
NSE Volume-Breakout BTST strategy's LOCKED baseline, all 3 cost bases (gross/net_A/net_B). Uses the
EXISTING baseline_final_performance.xlsx (daily_performance for INR pnl/drawdown, all_trades for
trade-level win rate) -- no backtest re-run, read-only against the locked strategy file.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
OUTDIR = rb.RESULTS / "baseline_and_cross_final"; OUTDIR.mkdir(parents=True, exist_ok=True)
BASES = ["gross", "net_A", "net_B"]
COLORS = {"gross": "#1f6feb", "net_A": "#e0742a", "net_B": "#2ba84a"}


def fy_bounds(d):
    D = pd.read_excel(SRC, sheet_name="daily_performance")
    D["date"] = pd.to_datetime(D["date"])
    data_start = D["date"].min(); data_end = D["date"].max()
    bounds = [
        ("FY21-22", data_start, pd.Timestamp("2022-03-31")),
        ("FY22-23", pd.Timestamp("2022-04-01"), pd.Timestamp("2023-03-31")),
        ("FY23-24", pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
        ("FY24-25", pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
        ("FY25-26", pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
        ("FY26-27", pd.Timestamp("2026-04-01"), data_end),
    ]
    return D, bounds, data_start, data_end


def drawdown_stats(cum):
    """cum: numpy array of cumulative INR pnl, restarting at 0 at series start. Returns (max_dd, avg_dd, n_episodes)."""
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
        return 0.0, 0.0, 0
    return abs(min(eps)), float(np.mean([abs(e) for e in eps])), len(eps)


def main():
    D, bounds, data_start, data_end = fy_bounds(None)
    ALL = pd.read_excel(SRC, sheet_name="all_trades")
    ALL["entry_date"] = pd.to_datetime(ALL["entry_date"])

    rows = []
    for fy_label, start, end in bounds:
        Dfy = D[(D["date"] >= start) & (D["date"] <= end)].sort_values("date").reset_index(drop=True)
        Tfy = ALL[(ALL["entry_date"] >= start) & (ALL["entry_date"] <= end)]
        if Dfy.empty:
            continue
        row = {"FY": fy_label, "period": f"{Dfy['date'].min().date()} to {Dfy['date'].max().date()}", "n_days": len(Dfy)}
        for base in BASES:
            pnl_col = f"{base}_pnl_inr" if base != "gross" else "gross_total_pnl_inr"
            if base == "gross":
                pnl_col = "gross_total_pnl_inr"
            elif base == "net_A":
                pnl_col = "net_A_total_pnl_inr"
            else:
                pnl_col = "net_B_total_pnl_inr"
            daily = Dfy[pnl_col].values
            cum = np.cumsum(daily)
            total_return = round(float(cum[-1]) if len(cum) else 0.0, 0)
            n_win_days = int((daily > 0).sum()); n_lose_days = int((daily <= 0).sum())
            maxdd, avgdd, n_eps = drawdown_stats(cum)

            trade_pnl_col = "gross_pnl" if base == "gross" else ("netA_pnl" if base == "net_A" else "netB_pnl")
            twin = Tfy[trade_pnl_col]
            trade_win_rate = round((twin > 0).mean() * 100, 2) if len(twin) else 0.0

            row.update({
                f"{base}_total_return_inr": total_return,
                f"{base}_win_rate_pct": trade_win_rate,
                f"{base}_n_winning_days": n_win_days,
                f"{base}_n_losing_days": n_lose_days,
                f"{base}_max_dd_inr": round(maxdd, 0),
                f"{base}_avg_dd_inr": round(avgdd, 0),
            })
        rows.append(row)

    SUMMARY = pd.DataFrame(rows)
    metric_order = ["total_return_inr", "win_rate_pct", "n_winning_days", "n_losing_days", "max_dd_inr", "avg_dd_inr"]
    col_order = ["FY", "period", "n_days"] + [f"{b}_{m}" for m in metric_order for b in BASES]
    SUMMARY = SUMMARY[col_order]

    pd.set_option("display.width", 260)
    print("=== FY-WISE SUMMARY (all 3 cost bases) ===")
    print(SUMMARY.to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "baseline_FY_wise_report.xlsx", engine="openpyxl") as w:
        SUMMARY.to_excel(w, sheet_name="FY_Summary", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 22)

    # ---- per-FY graphs ----
    for fy_label, start, end in bounds:
        Dfy = D[(D["date"] >= start) & (D["date"] <= end)].sort_values("date").reset_index(drop=True)
        if Dfy.empty:
            continue
        fig, ax = plt.subplots(figsize=(13, 6))
        for base in BASES:
            pnl_col = "gross_total_pnl_inr" if base == "gross" else (f"{base}_total_pnl_inr")
            cum = Dfy[pnl_col].cumsum()
            ax.plot(Dfy["date"], cum, color=COLORS[base], linewidth=1.5, label=base)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(f"NSE Volume-Breakout BTST — Locked Baseline — {fy_label}\n"
                      f"({Dfy['date'].min().date()} to {Dfy['date'].max().date()})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11); ax.set_ylabel("Cumulative Return (INR)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        fig.autofmt_xdate()
        fig.tight_layout()
        out_png = OUTDIR / f"volume_breakout_{fy_label}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"saved -> {out_png}")

    print(f"\nSaved -> {OUTDIR}/baseline_FY_wise_report.xlsx")


if __name__ == "__main__":
    main()
