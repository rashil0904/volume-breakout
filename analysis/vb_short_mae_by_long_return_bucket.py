# -*- coding: utf-8 -*-
"""vb_short_mae_by_long_return_bucket.py — extends the long8pct short-leg diagnostic to the FULL Volume-
Breakout trade set: every long trade (gross_ret) bucketed into 1%-step buckets, with the corresponding
double-down short's P&L, win rate, and MAE (max adverse excursion -- highest price reached after the short
opens, before it's covered via the 5% target or 14:39 fallback) aggregated per bucket. Read-only diagnostic,
no strategy changes. Per-symbol 1-min data cached (loaded once per symbol, not per trade) for performance
across ~3,400 trades.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
MD = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "long8pct_short_dd_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)
COVER_HM = 14 * 60 + 39


def hm(t):
    h, m = t.split(":"); return int(h) * 60 + int(m)


def main():
    T = pd.read_excel(SRC, sheet_name="all_trades")
    T = T[T["short_exit_type"] != "no_short"].copy()
    print(f"trades with a short leg: {len(T)} (excluded {15} 'no_short' rows)", flush=True)

    rows = []; missing = []
    symbols = sorted(T["symbol"].unique())
    print(f"unique symbols to load: {len(symbols)}", flush=True)

    for si, sym in enumerate(symbols, 1):
        fn = MD / f"{sym}.parquet"
        if not fn.exists():
            for _, r in T[T.symbol == sym].iterrows():
                missing.append((sym, r["exit_date"], "no master_data file"))
            continue
        df = pd.read_parquet(fn, columns=["timestamp", "high", "low"])
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.assign(ts=ts, date=ts.dt.normalize(), mod=ts.dt.hour * 60 + ts.dt.minute).sort_values("ts")
        by_date = {d: g for d, g in df.groupby("date")}

        for _, r in T[T.symbol == sym].iterrows():
            ex_date = pd.Timestamp(r["exit_date"])
            open_time_hm = hm(r["exit_time"]); open_price = float(r["exit_price"])
            cover_price = float(r["cover_price"]); short_exit_type = r["short_exit_type"]
            day = by_date.get(ex_date.normalize())
            if day is None:
                missing.append((sym, ex_date.date(), "no candles that day")); continue
            window = day[(day["mod"] > open_time_hm) & (day["mod"] <= COVER_HM)]
            if window.empty:
                missing.append((sym, ex_date.date(), "no candles after short-open")); continue

            if short_exit_type == "short_target_5pct":
                target_px = open_price * 0.95
                hit = window[window["low"] <= target_px]
                cover_mod = hit["mod"].iloc[0] if len(hit) else window["mod"].iloc[-1]
            else:
                cover_mod = COVER_HM

            mae_window = window[window["mod"] <= cover_mod]
            mae_price = float(mae_window["high"].max()) if len(mae_window) else open_price
            mae_pct = (mae_price - open_price) / open_price * 100

            rows.append({"symbol": sym, "date": ex_date.date(), "long_gross_ret": float(r["gross_ret"]),
                         "short_pnl": float(r["short_pnl"]), "short_mae_pct": mae_pct})

        if si % 200 == 0:
            print(f"  {si}/{len(symbols)} symbols processed | rows so far {len(rows)}", flush=True)

    R = pd.DataFrame(rows)
    print(f"\nprocessed: {len(R)} | missing/flagged: {len(missing)}", flush=True)

    # ---- 1%-step buckets ----
    edges = list(np.arange(-10, 15.001, 1))  # -10 ... 15
    def bucket_label(v):
        if v < -10: return "<-10%"
        if v > 15: return ">15%"
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if lo <= v < hi:
                return f"{int(lo)} to {int(hi)}%"
        return ">15%"
    R["bucket"] = R["long_gross_ret"].apply(bucket_label)

    order = ["<-10%"] + [f"{int(edges[i])} to {int(edges[i+1])}%" for i in range(len(edges) - 1)] + [">15%"]

    def agg(g):
        return pd.Series({
            "n_trades": len(g), "total_short_pnl": round(g.short_pnl.sum(), 1), "avg_short_pnl": round(g.short_pnl.mean(), 1),
            "short_win_rate_pct": round((g.short_pnl > 0).mean() * 100, 1), "avg_mae_pct": round(g.short_mae_pct.mean(), 3),
            "max_mae_pct": round(g.short_mae_pct.max(), 3), "small_sample_flag": "YES (<15)" if len(g) < 15 else ""})

    TAB = R.groupby("bucket").apply(agg, include_groups=False).reindex(order)
    TAB = TAB.reset_index().rename(columns={"index": "long_return_bucket"})

    with pd.ExcelWriter(OUTDIR / "short_mae_by_long_return_bucket.xlsx", engine="openpyxl") as w:
        TAB.to_excel(w, sheet_name="Bucket_Summary", index=False)
        R.to_excel(w, sheet_name="Trade_Level_Detail", index=False)
        if missing: pd.DataFrame(missing, columns=["symbol", "date", "issue"]).to_excel(w, sheet_name="Missing_Flagged", index=False)

    pd.set_option("display.width", 220)
    print("\n=== BUCKET SUMMARY ===")
    print(TAB.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/short_mae_by_long_return_bucket.xlsx")


if __name__ == "__main__":
    main()
