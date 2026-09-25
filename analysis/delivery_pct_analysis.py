# -*- coding: utf-8 -*-
"""
delivery_pct_analysis.py
========================
Part 4: attach NSE delivery % (from fetch_delivery_data.py cache) to each main-strategy
trade by (symbol, entry_date), then test whether delivery % (positional conviction proxy)
predicts trade quality.

Outputs: per_trade table (+ missing flag/count), delivery_pct summary + bucket
distribution, and a bucket cross-tab (n, win_rate, avg/median return).
Flags T2T-series (BE/BZ/BT) trades, where delivery is ~100% by rule and would skew buckets.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

CACHE = rb.RESULTS / "delivery_cache"
OUTDIR = rb.RESULTS / "delivery_analysis"
T2T_SERIES = {"BE", "BZ", "BT"}                     # trade-to-trade: delivery ~100% by rule
BUCKETS = [(-0.01, 20, "<20%"), (20, 40, "20-40%"), (40, 60, "40-60%"),
           (60, 80, "60-80%"), (80, 100.01, ">80%")]


def load_delivery():
    files = sorted(CACHE.glob("*.csv"))
    if not files:
        raise SystemExit(f"No delivery cache in {CACHE} — run fetch_delivery_data.py first.")
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if len(d):
            dfs.append(d)
    D = pd.concat(dfs, ignore_index=True)
    D["date"] = pd.to_datetime(D["date"]).dt.date
    D["SYMBOL"] = D["SYMBOL"].astype(str).str.upper().str.strip()
    D["SERIES"] = D["SERIES"].astype(str).str.strip()
    D["delivery_pct"] = pd.to_numeric(D["DELIV_PER"], errors="coerce")
    return D


def bucket_of(v):
    if pd.isna(v):
        return "missing"
    for lo, hi, name in BUCKETS:
        if lo < v <= hi:
            return name
    return "missing"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    D = load_delivery()
    print(f"Delivery rows loaded: {len(D):,} over {D['date'].nunique():,} dates")

    # ── trades (+ entry_return from diagnostic) ──
    T = fpr.build_trades()
    T["entry_date_d"] = pd.to_datetime(T["entry_date"]).dt.date
    T["symbol_u"] = T["symbol"].str.upper().str.strip()
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "return_pct_vs_prev_close"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    diag["symbol"] = diag["symbol"].str.upper().str.strip()
    er = diag.set_index(["symbol", "date"])["return_pct_vs_prev_close"].to_dict()
    T["entry_return_pct"] = [round(er.get((s, d), np.nan), 4)
                             for s, d in zip(T["symbol_u"], T["entry_date_d"])]

    dmap = D.set_index(["SYMBOL", "date"])[["delivery_pct", "SERIES", "TTL_TRD_QNTY", "DELIV_QTY"]]
    dmap = dmap[~dmap.index.duplicated()]
    def lk(s, d, col):
        try:
            return dmap.loc[(s, d), col]
        except KeyError:
            return np.nan
    T["delivery_pct"] = [lk(s, d, "delivery_pct") for s, d in zip(T["symbol_u"], T["entry_date_d"])]
    T["series"] = [lk(s, d, "SERIES") for s, d in zip(T["symbol_u"], T["entry_date_d"])]
    T["delivery_missing"] = T["delivery_pct"].isna()
    T["is_t2t"] = T["series"].isin(T2T_SERIES)

    n_missing = int(T["delivery_missing"].sum())
    n_t2t = int(T["is_t2t"].sum())
    print(f"Trades: {len(T):,} | delivery attached: {len(T)-n_missing:,} | missing: {n_missing:,} "
          f"({n_missing/len(T)*100:.2f}%) | T2T-series: {n_t2t}")

    # ── 1. per-trade table ──
    per = T[["symbol", "entry_date_d", "entry_return_pct", "delivery_pct", "series", "is_t2t",
             "gross_ret", "net_ret", "exit_type", "delivery_missing"]].copy()
    per.columns = ["symbol", "entry_date", "entry_return_pct", "delivery_pct", "series", "is_t2t",
                   "trade_return_pct", "net_trade_return_pct", "exit_type", "delivery_missing"]
    per = per.sort_values("entry_date")

    # ── 2. summary + distribution ──
    dl = T.loc[~T["delivery_missing"], "delivery_pct"]
    summary = pd.DataFrame([
        {"metric": "n_trades_total", "value": len(T)},
        {"metric": "n_trades_with_delivery", "value": len(dl)},
        {"metric": "n_trades_missing_delivery", "value": n_missing},
        {"metric": "n_trades_t2t_series", "value": n_t2t},
        {"metric": "delivery_pct_mean", "value": round(dl.mean(), 2)},
        {"metric": "delivery_pct_median", "value": round(dl.median(), 2)},
        {"metric": "delivery_pct_min", "value": round(dl.min(), 2)},
        {"metric": "delivery_pct_max", "value": round(dl.max(), 2)},
    ])

    # ── 3. bucket cross-tab (exclude T2T + missing so buckets aren't skewed) ──
    core = T[~T["delivery_missing"] & ~T["is_t2t"]].copy()
    core["bucket"] = core["delivery_pct"].map(bucket_of)
    order = [b[2] for b in BUCKETS]
    rows = []
    for b in order:
        g = core[core["bucket"] == b]
        rows.append({
            "delivery_bucket": b, "n_trades": len(g),
            "pct_of_trades": round(len(g) / len(core) * 100, 2) if len(core) else 0,
            "win_rate_pct": round((g["gross_pnl"] > 0).mean() * 100, 2) if len(g) else np.nan,
            "avg_return_per_trade_pct": round(g["gross_ret"].mean(), 4) if len(g) else np.nan,
            "median_return_per_trade_pct": round(g["gross_ret"].median(), 4) if len(g) else np.nan,
            "net_avg_return_per_trade_pct": round(g["net_ret"].mean(), 4) if len(g) else np.nan,
        })
    crosstab = pd.DataFrame(rows)
    # T2T shown separately for transparency
    t2t = T[T["is_t2t"]]
    if len(t2t):
        crosstab = pd.concat([crosstab, pd.DataFrame([{
            "delivery_bucket": "T2T-series (excluded above)", "n_trades": len(t2t),
            "pct_of_trades": np.nan,
            "win_rate_pct": round((t2t["gross_pnl"] > 0).mean() * 100, 2),
            "avg_return_per_trade_pct": round(t2t["gross_ret"].mean(), 4),
            "median_return_per_trade_pct": round(t2t["gross_ret"].median(), 4),
            "net_avg_return_per_trade_pct": round(t2t["net_ret"].mean(), 4)}])], ignore_index=True)

    with pd.ExcelWriter(OUTDIR / "delivery_pct_analysis.xlsx", engine="openpyxl") as w:
        per.to_excel(w, sheet_name="per_trade", index=False)
        summary.to_excel(w, sheet_name="summary", index=False)
        crosstab.to_excel(w, sheet_name="delivery_bucket_crosstab", index=False)

    pd.set_option("display.width", 200)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== DELIVERY-% BUCKET CROSS-TAB (EQ only; T2T & missing excluded from buckets) ===")
    print(crosstab.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
