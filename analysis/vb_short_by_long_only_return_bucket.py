# -*- coding: utf-8 -*-
"""vb_short_by_long_only_return_bucket.py — CORRECTED version of short_mae_by_long_return_bucket.py.
The original bucketed trades by `gross_ret` (long+short COMBINED return / capital) but mislabeled it
"long_gross_ret" -- circular, since a trade's bucket already baked in its own short's outcome. This
version buckets by the LONG LEG'S OWN return only (long_pnl / capital_deployed * 100 -- equivalently
(exit_price/avg_entry - 1)*100), the only return actually knowable at the moment the double-down short
would be opened. short_pnl / MAE values reused from the already-computed
short_mae_by_long_return_bucket.xlsx (Trade_Level_Detail) to avoid re-scanning parquet files -- joined
back to all_trades on (symbol, exit_date) to pull long_pnl/capital_deployed. Read-only diagnostic.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
PRIOR = rb.RESULTS / "baseline_and_cross_final" / "long8pct_short_dd_diagnostic" / "short_mae_by_long_return_bucket.xlsx"
OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "long8pct_short_dd_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    T = pd.read_excel(SRC, sheet_name="all_trades")
    T = T[T["short_exit_type"] != "no_short"].copy()
    T["exit_date"] = pd.to_datetime(T["exit_date"]).dt.date
    T["long_only_ret"] = T["long_pnl"] / T["capital_deployed"] * 100

    prior = pd.read_excel(PRIOR, sheet_name="Trade_Level_Detail")
    prior["date"] = pd.to_datetime(prior["date"]).dt.date
    prior = prior.rename(columns={"date": "exit_date"})

    R = T[["symbol", "exit_date", "long_only_ret", "short_pnl"]].merge(
        prior[["symbol", "exit_date", "short_mae_pct"]], on=["symbol", "exit_date"], how="left")
    # short_pnl comes from T directly (authoritative); prior's short_pnl is identical, only MAE reused
    missing_mae = int(R["short_mae_pct"].isna().sum())
    print(f"trades: {len(R)} | joined MAE for {len(R) - missing_mae} | missing MAE (dropped join): {missing_mae}", flush=True)

    edges = list(np.arange(-10, 15.001, 1))
    def bucket_label(v):
        if v < -10: return "<-10%"
        if v > 15: return ">15%"
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if lo <= v < hi:
                return f"{int(lo)} to {int(hi)}%"
        return ">15%"
    R["bucket"] = R["long_only_ret"].apply(bucket_label)
    order = ["<-10%"] + [f"{int(edges[i])} to {int(edges[i+1])}%" for i in range(len(edges) - 1)] + [">15%"]

    def agg(g):
        return pd.Series({
            "n_trades": len(g), "total_short_pnl": round(g.short_pnl.sum(), 1), "avg_short_pnl": round(g.short_pnl.mean(), 1),
            "short_win_rate_pct": round((g.short_pnl > 0).mean() * 100, 1),
            "avg_mae_pct": round(g.short_mae_pct.mean(), 3) if g.short_mae_pct.notna().any() else np.nan,
            "max_mae_pct": round(g.short_mae_pct.max(), 3) if g.short_mae_pct.notna().any() else np.nan,
            "small_sample_flag": "YES (<15)" if len(g) < 15 else ""})

    TAB = R.groupby("bucket").apply(agg, include_groups=False).reindex(order)
    TAB = TAB.reset_index().rename(columns={"index": "long_only_return_bucket"})
    TAB = TAB.dropna(subset=["n_trades"])
    TAB["n_trades"] = TAB["n_trades"].astype(int)

    with pd.ExcelWriter(OUTDIR / "short_pnl_by_LONG_ONLY_return_bucket.xlsx", engine="openpyxl") as w:
        TAB.to_excel(w, sheet_name="Bucket_Summary", index=False)
        R.to_excel(w, sheet_name="Trade_Level_Detail", index=False)

    pd.set_option("display.width", 220)
    print("\n=== BUCKET SUMMARY (bucketed by LONG-ONLY return, i.e. what's knowable BEFORE the short opens) ===")
    print(TAB.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/short_pnl_by_LONG_ONLY_return_bucket.xlsx")


if __name__ == "__main__":
    main()
