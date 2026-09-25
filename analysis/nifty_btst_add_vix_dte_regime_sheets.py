# -*- coding: utf-8 -*-
"""nifty_btst_add_vix_dte_regime_sheets.py — ADDITIVE update to btst_close_direction_FINAL_v2.xlsx: appends
3 new sheets (VIX_Summary, DTE_Summary_OldRegime, DTE_Summary_NewRegime) WITHOUT touching the existing
Summary/By_Group/Trades/Monthly_PnL/Drawdowns sheets (opened in append mode, existing sheets untouched).
DTE bucketed by ORIGINAL/nominal classification (nearest expiry strictly after entry day, before any DTE-1
shift) so the DTE=1 bucket remains visible with contract_used=next-week reflecting the fix. Regime boundary
derived from the actual expiry calendar's weekday change point (not assumed).
"""
import sys, os, bisect
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "btst_close_direction_FINAL"; OUTF = OUTDIR / "btst_close_direction_FINAL_v2.xlsx"
TRADES_CSV = OUTDIR / "btst_close_direction_FINAL_v2_trades.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]


def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


def main():
    T = pd.read_csv(TRADES_CSV)
    T["entry_date"] = pd.to_datetime(T["entry_date"]); T["exit_date"] = pd.to_datetime(T["exit_date"])

    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d"))
    wd_all = pd.Series([e.day_name() for e in expiries], index=expiries)
    # filter to Tue/Thu ONLY before scanning for the change point - holiday-shifted Mon/Wed/Fri weeks
    # sit BETWEEN two same-regime Tue or Thu expiries and must not be mistaken for a regime change
    tue_thu = wd_all[wd_all.isin(["Tuesday", "Thursday"])]
    regime_start_new = None
    for i in range(1, len(tue_thu)):
        if tue_thu.iloc[i] != tue_thu.iloc[i - 1]:
            regime_start_new = tue_thu.index[i]; break   # first (and expected only) genuine change
    print(f"regime boundary (from actual calendar, Tue/Thu-only scan): NEW regime starts at expiry {regime_start_new.date()}")

    def dte_original_and_regime(d):
        j = bisect.bisect_right(expiries, d)
        E0 = expiries[j]
        dte0 = (E0.normalize() - d.normalize()).days
        # regime by CHRONOLOGICAL POSITION vs the confirmed boundary (not weekday name) - a holiday-shifted
        # week's expiry may not literally fall on Tue/Thu but still belongs to that regime by date
        regime = "NEW (Tue-expiry)" if E0 >= regime_start_new else "OLD (Thu-expiry)"
        return dte0, regime

    res = T["entry_date"].map(dte_original_and_regime)
    T["DTE_original"] = [r[0] for r in res]; T["regime"] = [r[1] for r in res]

    t_en = T["entry_date"] + pd.Timedelta(hours=15, minutes=20)
    t_ex = T["exit_date"] + pd.Timedelta(hours=9, minutes=17)
    T["holding_hours"] = (t_ex - t_en).dt.total_seconds() / 3600

    T["vix_bucket"] = T["entry_vix"].map(vbucket)

    # ---- VIX_Summary ----
    def blk(df, extra=None):
        d = {"trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
             "total_pnl": round(df.pnl_points.sum(), 1) if len(df) else 0, "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
             "avg_holding_hours": round(df.holding_hours.mean(), 2) if len(df) else 0}
        if extra: d.update(extra)
        return d

    VIX = pd.DataFrame([{"vix_bucket": b, **blk(T[T.vix_bucket == b])} for b in VIX_BUCKETS])

    # ---- DTE_Summary per regime ----
    def dte_summary(regime_label):
        sub = T[T.regime == regime_label]
        rows = []
        for d in range(1, 7):
            g = sub[sub.DTE_original == d]
            contract_counts = g["contract"].value_counts().to_dict() if len(g) else {}
            rows.append({"DTE": d, **blk(g), "contract_used": str(contract_counts)})
        return pd.DataFrame(rows)

    DTE_OLD = dte_summary("OLD (Thu-expiry)")
    DTE_NEW = dte_summary("NEW (Tue-expiry)")

    with pd.ExcelWriter(OUTF, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        VIX.to_excel(w, sheet_name="VIX_Summary", index=False)
        DTE_OLD.to_excel(w, sheet_name="DTE_Summary_OldRegime", index=False)
        DTE_NEW.to_excel(w, sheet_name="DTE_Summary_NewRegime", index=False)

    print(f"\nsheets in workbook now: {pd.ExcelFile(OUTF).sheet_names}")
    print("\n--- VIX_Summary ---"); print(VIX.to_string(index=False))
    print("\n--- DTE_Summary_OldRegime ---"); print(DTE_OLD.to_string(index=False))
    print("\n--- DTE_Summary_NewRegime ---"); print(DTE_NEW.to_string(index=False))


if __name__ == "__main__":
    main()
