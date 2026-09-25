# -*- coding: utf-8 -*-
"""
variant_52whigh_ret3pct.py
==========================
Parallel variant "_52whigh_ret3pct": main strategy with TWO stacked entry changes —
(1) daily return >= +3% (loosened from +5%) AND (2) entry day makes a NEW 52-week high.
All else identical: mcap ₹1,500-5,000 Cr, LB 36, VM 6, 3:15pm entry, 09:45/12:00 split +
14% target, ₹5L pool / ₹1L per trade. Reuses the diagnostic table + fpr report machinery;
does not modify existing strategies.

52w high = max daily HIGH over the 252 trading days ending the day BEFORE entry (excl.
entry day). Breakout = entry-day high >= that. Sizing = pool-split on THIS variant's own
daily signal counts. The full report is produced by reusing fpr.main() (exact mirror:
summary/all_trades/daily/monthly/quarterly/half_yearly/yearly, gross|net@0.23%|net@0.38%,
compounding). Report compounding uses fpr's CURRENT bake-in-loss rule (flagged).
"""
import sys, bisect
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import final_performance_report as fpr

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "variant_52whigh_ret3pct"
WIN_52 = 252


def breakout_mask(sig):
    """For each +3% signal row, True if entry-day high >= trailing-252d high (excl. entry
    day). Returns (mask over sig.index, n_dropped_no_history)."""
    keep = pd.Series(False, index=sig.index)
    n_no_hist = 0
    for sym, grp in sig.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            n_no_hist += len(grp); continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        dh = raw.groupby("date")["high"].max()
        ds = sorted(dh.index); dh = dh.reindex(ds)
        r52 = dh.rolling(WIN_52, min_periods=WIN_52).max().shift(1)   # ends day BEFORE entry
        r52_map, dh_map = r52.to_dict(), dh.to_dict()
        for idx, d in zip(grp.index, grp["date"]):
            dd = pd.Timestamp(d).date()
            hi = r52_map.get(dd, np.nan)
            if hi is None or hi != hi:
                n_no_hist += 1; continue
            if dh_map.get(dd, np.nan) >= hi:
                keep.loc[idx] = True
    return keep, n_no_hist


def build_base(diag, ret_thr, require_52w):
    """Pool-split base positions for signals: passes_volume & return>=ret_thr [& new 52w
    high]. Sizing on this set's own daily counts."""
    sig = diag[diag["passes_volume"] & (diag["return_pct_vs_prev_close"] >= ret_thr)] \
        .dropna(subset=["entry_price_315pm"]).copy()
    n_signals = len(sig)
    n_no_hist = 0
    if require_52w:
        mask, n_no_hist = breakout_mask(sig)
        sig = sig[mask]
    rows = []
    for d, day in sig.groupby("date"):
        tgt = rb._day_target(len(day))
        for _, r in day.iterrows():
            ep = float(r["entry_price_315pm"]); sh = int(tgt // ep)
            if sh == 0:
                continue
            rows.append({"date": d, "symbol": r["symbol"], "entry": ep, "shares": sh, "cap": sh * ep})
    base = pd.DataFrame(rows)
    base["date"] = pd.to_datetime(base["date"])
    return base.sort_values(["date", "symbol"]).reset_index(drop=True), n_signals, n_no_hist


def cmp_metrics(T, label):
    n = len(T)
    g, npl = T["gross_pnl"].sum(), T["net_pnl"].sum()
    return {
        "strategy": label, "n_trades": n,
        "gross_win_rate_pct": round((T["gross_pnl"] > 0).mean() * 100, 2),
        "net_win_rate_pct": round((T["net_pnl"] > 0).mean() * 100, 2),
        "gross_total_return_fixedbase_pct": round(g / fpr.BASE_POOL * 100, 4),
        "net_total_return_fixedbase_pct": round(npl / fpr.BASE_POOL * 100, 4),
        "gross_total_pnl_inr": round(g, 0), "net_total_pnl_inr": round(npl, 0),
        "gross_avg_return_per_trade_pct": round(T["gross_ret"].mean(), 4),
        "net_avg_return_per_trade_pct": round(T["net_ret"].mean(), 4),
        "gross_median_return_per_trade_pct": round(T["gross_ret"].median(), 4),
        "net_median_return_per_trade_pct": round(T["net_ret"].median(), 4),
        "avg_capital_deployed_per_trade": round(T["capital_deployed"].mean(), 0),
    }


def trades_for(base):
    orig = ets.load_base_positions
    ets.load_base_positions = lambda: base
    try:
        return fpr.build_trades()
    finally:
        ets.load_base_positions = orig


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "entry_price_315pm",
                                "return_pct_vs_prev_close", "passes_volume"], parse_dates=["date"])

    print("Building B (+3% only) and C (+3% AND new 52w high) base positions …")
    base_B, nsig_B, _ = build_base(diag, 3.0, require_52w=False)
    base_C, nsig_C, n_no_hist = build_base(diag, 3.0, require_52w=True)
    print(f"  +3% signals: {nsig_B:,} | +3% base positions: {len(base_B):,}")
    print(f"  +3% & 52w-high: dropped {n_no_hist:,} for <252d history | "
          f"base positions: {len(base_C):,}")

    # ── full report for variant C (mirror fpr exactly, via monkeypatch) ──
    orig_load, orig_xlsx = ets.load_base_positions, fpr.XLSX
    ets.load_base_positions = lambda: base_C
    fpr.XLSX = OUTDIR / "main_strategy_52whigh_ret3pct_performance.xlsx"
    print("\nGenerating full performance report for variant C …")
    fpr.main()
    ets.load_base_positions, fpr.XLSX = orig_load, orig_xlsx

    # ── three-way comparison A / B / C ──
    T_A = fpr.build_trades()                       # canonical +5% baseline (deployed sheet)
    T_B = trades_for(base_B)
    T_C = trades_for(base_C)
    comp = pd.DataFrame([cmp_metrics(T_A, "A_main_+5pct"),
                         cmp_metrics(T_B, "B_+3pct_only"),
                         cmp_metrics(T_C, "C_+3pct_AND_52whigh")])

    # ── flag thin periods in variant C ──
    thin = []
    for lvl in ["month", "quarter", "year"]:
        cnt = T_C.groupby(lvl).size()
        for k, v in cnt.items():
            if v < 10:
                thin.append({"period_level": lvl, "period": str(k), "n_trades": int(v)})
    thin_tbl = pd.DataFrame(thin)

    with pd.ExcelWriter(OUTDIR / "three_way_comparison.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="A_B_C_comparison", index=False)
        if len(thin_tbl):
            thin_tbl.to_excel(w, sheet_name="variantC_thin_periods", index=False)

    pd.set_option("display.width", 220)
    gcols = ["strategy", "n_trades", "gross_win_rate_pct", "gross_total_return_fixedbase_pct",
             "gross_total_pnl_inr", "gross_avg_return_per_trade_pct",
             "gross_median_return_per_trade_pct", "avg_capital_deployed_per_trade"]
    ncols = ["strategy", "net_win_rate_pct", "net_total_return_fixedbase_pct",
             "net_total_pnl_inr", "net_avg_return_per_trade_pct", "net_median_return_per_trade_pct"]
    print("\n" + "=" * 120 + "\nTHREE-WAY COMPARISON — GROSS\n" + "=" * 120)
    print(comp[gcols].to_string(index=False))
    print("\n" + "=" * 120 + "\nTHREE-WAY COMPARISON — NET @ 0.23%\n" + "=" * 120)
    print(comp[ncols].to_string(index=False))
    print(f"\nVariant C monthly/quarterly/yearly periods with < 10 trades (unreliable): {len(thin_tbl)}")
    if len(thin_tbl):
        print(thin_tbl.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
