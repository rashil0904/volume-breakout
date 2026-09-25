# -*- coding: utf-8 -*-
"""vb_short_by_long_bucket.py — diagnostic: segment the SHORT leg's performance by the corresponding LONG leg's
realized return, in 1% buckets, across every trade (Categories A/B/C) in the main NSE Volume-Breakout BTST
strategy's locked baseline. Pure segmentation of the EXISTING, unmodified backtest -- byte-identical engine
(BC.build_cache/BC.run_config, full history), no strategy-logic changes.

long_ret_pct  = long_pnl / capital_deployed * 100         (the long leg's own return on its own capital)
short_ret_pct = short_pnl / (shares * exit_price) * 100   (short opens at the long-exit price xp with the same
                share count -- shares*exit_price is exactly the short's entry notional; only computed for trades
                that actually HAD a short, i.e. cover_price is not null)
Bucket = 1%-wide, floor-based (e.g. a long_ret_pct of 4.37 falls in the "4-5%" bucket; -2.10 falls in "-3--2%").
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC

OUT = rb.RESULTS / "vb_short_by_long_bucket"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    cache = BC.build_cache()
    T, _ = BC.run_config("baseline", cache)
    print(f"total trades: {len(T)}", flush=True)

    T["long_ret_pct"] = T["long_pnl"] / T["capital_deployed"] * 100
    T["has_short"] = T["cover_price"].notna()
    short_notional = T["shares"] * T["exit_price"]
    T["short_ret_pct"] = np.where(T["has_short"], T["short_pnl"] / short_notional * 100, np.nan)

    n_short = int(T["has_short"].sum())
    print(f"trades with an actual short leg: {n_short} of {len(T)} ({n_short/len(T)*100:.1f}%)", flush=True)
    print(f"long_ret_pct range: min={T['long_ret_pct'].min():.2f}% max={T['long_ret_pct'].max():.2f}% "
          f"| long target = {(BC.LONG_TGT-1)*100:.1f}%", flush=True)

    T["bucket_lo"] = np.floor(T["long_ret_pct"]).astype(int)
    T["bucket"] = T["bucket_lo"].astype(str) + "-" + (T["bucket_lo"] + 1).astype(str) + "%"

    S = T[T["has_short"]]     # short-leg stats computed only over trades that actually had a short
    def agg(g):
        sret = g["short_ret_pct"]
        return pd.Series({
            "n_short_trades": len(g),
            "avg_short_ret_pct": round(sret.mean(), 3),
            "median_short_ret_pct": round(sret.median(), 3),
            "total_short_pnl_inr": round(g["short_pnl"].sum(), 0),
            "short_win_rate_pct": round((g["short_pnl"] > 0).mean() * 100, 1),
            "std_short_ret_pct": round(sret.std(), 3) if len(g) > 1 else np.nan})
    G = S.groupby(["bucket_lo", "bucket"]).apply(agg, include_groups=False).reset_index().sort_values("bucket_lo")

    # also carry the LONG-trade count per bucket (all trades, incl. those with no short) for context
    n_long_all = T.groupby(["bucket_lo"]).size().rename("n_long_trades_all")
    G = G.merge(n_long_all, left_on="bucket_lo", right_index=True, how="left")

    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 60); pd.set_option("display.max_columns", 20)
    print("\n=== SHORT-LEG PERFORMANCE BY LONG-LEG RETURN BUCKET (1% steps) ===\n" +
          G[["bucket", "n_long_trades_all", "n_short_trades", "avg_short_ret_pct", "median_short_ret_pct",
             "total_short_pnl_inr", "short_win_rate_pct", "std_short_ret_pct"]].to_string(index=False), flush=True)

    thin = G[G["n_short_trades"] < 15]
    print(f"\nBUCKETS WITH THIN SAMPLE (<15 short trades) -- {len(thin)} of {len(G)} buckets:", flush=True)
    print(thin[["bucket", "n_short_trades", "avg_short_ret_pct"]].to_string(index=False), flush=True)

    # correlation check: does long-leg strength predict short-leg outcome?
    valid = S.dropna(subset=["long_ret_pct", "short_ret_pct"])
    corr = valid["long_ret_pct"].corr(valid["short_ret_pct"])
    corr_spear = valid["long_ret_pct"].corr(valid["short_ret_pct"], method="spearman")
    # regression slope for context
    slope, intercept = np.polyfit(valid["long_ret_pct"], valid["short_ret_pct"], 1)
    print(f"\ncorrelation (long_ret_pct vs short_ret_pct), pearson: {corr:.4f} | spearman: {corr_spear:.4f}", flush=True)
    print(f"OLS fit: short_ret_pct = {intercept:.3f} + {slope:.4f} * long_ret_pct  (n={len(valid)})", flush=True)

    # weighted (by trade count) view restricted to buckets with reasonable sample, for the "clear relationship?" call
    solid = G[G["n_short_trades"] >= 15].sort_values("bucket_lo")
    print(f"\nsolid-sample buckets (>=15 trades), avg_short_ret_pct trend:\n" +
          solid[["bucket", "n_short_trades", "avg_short_ret_pct"]].to_string(index=False), flush=True)

    # ---- chart ----
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = G["bucket_lo"]
    ax1.plot(x, G["avg_short_ret_pct"], marker="o", color="tab:blue", label="avg short return % (all buckets)")
    solid_x = solid["bucket_lo"]
    ax1.plot(solid_x, solid["avg_short_ret_pct"], marker="o", color="tab:red", linewidth=2.2, label="avg short return % (n>=15 only)")
    ax1.axhline(0, color="grey", linewidth=0.8)
    ax1.set_xlabel("Long-leg return bucket (1% steps, floor of bucket)")
    ax1.set_ylabel("Average short-leg return %")
    ax1.set_title("Short-leg return vs long-leg return bucket — Volume BTST, full history")
    ax2 = ax1.twinx()
    ax2.bar(x, G["n_short_trades"], alpha=0.15, color="grey", width=0.9, label="n short trades (right axis)")
    ax2.set_ylabel("Trade count in bucket")
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    chart_fn = OUT / "short_by_long_bucket.png"
    fig.savefig(chart_fn, dpi=150)
    print(f"\nChart saved -> {chart_fn}", flush=True)

    fn = OUT / "vb_short_by_long_bucket.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": "Diagnostic segmentation of the EXISTING locked baseline (full history, BC.build_cache/BC.run_config, "
                                "unmodified). long_ret_pct = long_pnl/capital_deployed*100. short_ret_pct computed only for trades that "
                                f"actually had a short leg ({n_short} of {len(T)}, {n_short/len(T)*100:.1f}%) -- short opens at the long "
                                "exit price with the same share count, so shares*exit_price is the short's entry notional. Buckets with "
                                "fewer than 15 short trades are flagged as thin."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        G.to_excel(w, sheet_name="Bucket_summary", index=False)
        pd.DataFrame([{"pearson_corr": corr, "spearman_corr": corr_spear, "ols_slope": slope, "ols_intercept": intercept, "n": len(valid)}]).to_excel(w, sheet_name="Correlation", index=False)
        T.to_excel(w, sheet_name="All_trades_with_buckets", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 32)
    print(f"Saved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
