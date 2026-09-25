# -*- coding: utf-8 -*-
"""algotest_selling_strategy_report.py — builds the Excel + 4 FY PNGs for the AlgoTest selling strategy
(D.SENSEX1/2/3 + D.NIFTY1/2 + D.SENSEX/NIFTY POSITIONAL combined portfolio), using USER-PROVIDED monthly
PnL figures already extracted and scaled to the 2-lot/Rs 20L basis. No recomputation, no rescaling --
figures used exactly as given. April 2023 is a PROXY value (filled with Oct-2023's actual figure, since
the platform showed 0 for Apr'23) -- flagged explicitly in the data itself and in outputs.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "algotest_selling_strategy"; OUTDIR.mkdir(parents=True, exist_ok=True)

MONTHLY = [
    # (Month, FY, PnL, Is_Proxy)
    ("2023-04", "FY23-24", -3803, "Y"),
    ("2023-05", "FY23-24", 934020, "N"),
    ("2023-06", "FY23-24", 113010, "N"),
    ("2023-07", "FY23-24", 4384, "N"),
    ("2023-08", "FY23-24", 62628, "N"),
    ("2023-09", "FY23-24", 661, "N"),
    ("2023-10", "FY23-24", -3803, "N"),
    ("2023-11", "FY23-24", -2648, "N"),
    ("2023-12", "FY23-24", 34576, "N"),
    ("2024-01", "FY23-24", 29696, "N"),
    ("2024-02", "FY23-24", 114477, "N"),
    ("2024-03", "FY23-24", 92772, "N"),

    ("2024-04", "FY24-25", 88943, "N"),
    ("2024-05", "FY24-25", 148581, "N"),
    ("2024-06", "FY24-25", 146761, "N"),
    ("2024-07", "FY24-25", 117968, "N"),
    ("2024-08", "FY24-25", 52472, "N"),
    ("2024-09", "FY24-25", 79299, "N"),
    ("2024-10", "FY24-25", 21462, "N"),
    ("2024-11", "FY24-25", 98181, "N"),
    ("2024-12", "FY24-25", 150111, "N"),
    ("2025-01", "FY24-25", 53194, "N"),
    ("2025-02", "FY24-25", 182263, "N"),
    ("2025-03", "FY24-25", 167038, "N"),

    ("2025-04", "FY25-26", 178537, "N"),
    ("2025-05", "FY25-26", 139682, "N"),
    ("2025-06", "FY25-26", 175330, "N"),
    ("2025-07", "FY25-26", 164716, "N"),
    ("2025-08", "FY25-26", 79896, "N"),
    ("2025-09", "FY25-26", 132565, "N"),
    ("2025-10", "FY25-26", 145037, "N"),
    ("2025-11", "FY25-26", 98257, "N"),
    ("2025-12", "FY25-26", 86746, "N"),
    ("2026-01", "FY25-26", 184768, "N"),
    ("2026-02", "FY25-26", 153374, "N"),
    ("2026-03", "FY25-26", 271220, "N"),

    ("2026-04", "FY26-27", 450678, "N"),
    ("2026-05", "FY26-27", 249213, "N"),
    ("2026-06", "FY26-27", 230465, "N"),
    ("2026-07", "FY26-27", 126558, "N"),
]

FY_SUMMARY = [
    ("FY23-24", 1375970, -50637, 9, 3),
    ("FY24-25", 1306273, -44028, 12, 0),
    ("FY25-26", 1810128, -23148, 12, 0),
    ("FY26-27", 1056914, -35779, 4, 0),
]


def main():
    M = pd.DataFrame(MONTHLY, columns=["Month", "FY", "PnL_INR", "Is_Proxy"])
    M["Cumulative_PnL_INR"] = M["PnL_INR"].cumsum()

    FY = pd.DataFrame(FY_SUMMARY, columns=["FY", "Total_Return_INR", "Max_Drawdown_INR", "Winning_Months", "Losing_Months"])

    # sanity check: monthly sums should match FY summary totals
    check = M.groupby("FY")["PnL_INR"].sum()
    print("Cross-check: sum of monthly PnL per FY vs given FY totals")
    for fy, total in FY.set_index("FY")["Total_Return_INR"].items():
        s = check.get(fy, 0)
        flag = "OK" if abs(s - total) < 1 else f"MISMATCH (monthly sum={s})"
        print(f"  {fy}: given={total} | {flag}")

    out_xlsx = OUTDIR / "algotest_selling_strategy_report.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "AlgoTest selling strategy (D.SENSEX1/2/3 + D.NIFTY1/2 + D.SENSEX/NIFTY POSITIONAL combined), "
                      "2-lot / Rs 20L basis (already scaled by 2/9 from the original 9-lot/Rs 90L reporting)."},
            {"note": "*** April 2023 (2023-04) is a PROXY value -- filled with October 2023's actual figure "
                      "(-3,803), since the AlgoTest platform reported 0 activity for Apr'23. See Is_Proxy column. ***"},
            {"note": "Figures provided pre-computed and pre-scaled -- not recomputed or rescaled in this script."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        M.to_excel(w, sheet_name="Monthly_Data", index=False)
        FY.to_excel(w, sheet_name="FY_Summary", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)

    print(f"\nSaved -> {out_xlsx}")

    # ---- 4 FY graphs, cumulative resetting to 0 at each FY's start ----
    png_paths = []
    for fy in ["FY23-24", "FY24-25", "FY25-26", "FY26-27"]:
        Mfy = M[M["FY"] == fy].reset_index(drop=True)
        Mfy["fy_cum"] = Mfy["PnL_INR"].cumsum()

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(Mfy["Month"], Mfy["fy_cum"], color="#1f6feb", linewidth=1.8, marker="o", markersize=5)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        proxy_months = Mfy[Mfy["Is_Proxy"] == "Y"]
        if len(proxy_months):
            ax.scatter(proxy_months["Month"], proxy_months["fy_cum"], color="red", s=80, zorder=5, label="PROXY value (Apr'23 = Oct'23 actual)")
            ax.legend(loc="upper left", fontsize=9)
        title = f"AlgoTest Selling Strategy — {fy} (Rs 20L basis)"
        if len(proxy_months):
            title += "\n(red marker = proxy-filled month, not genuine reported data)"
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Month", fontsize=11); ax.set_ylabel("Cumulative Return (INR)", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        out_png = OUTDIR / f"{fy}_returns.png"
        fig.savefig(out_png, dpi=150); plt.close(fig)
        png_paths.append(out_png)
        print(f"saved -> {out_png}")

    print("\nAll files:")
    print(f"  {out_xlsx}")
    for p in png_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
