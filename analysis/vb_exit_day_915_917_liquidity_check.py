# -*- coding: utf-8 -*-
"""vb_exit_day_915_917_liquidity_check.py — diagnostic/data-quality check (NO strategy-logic changes) on
the main NSE Volume-Breakout BTST strategy's locked baseline (baseline_final_performance.xlsx, 3,436
trades). For every trade, measures traded volume in the 9:15-9:17 window (3 one-min candles, hm 555/556/
557) on the EXIT day, normalizes it against the same 36-day average full-day volume already used as the
strategy's entry filter (diagnostic_table.csv::avg_nday_fullday_volume, joined on symbol+entry_date -- the
value used AT ENTRY time, since that's the only pre-computed ADV reference the strategy itself uses), and
checks the trade's own exit quantity as a % of that 3-min window's volume (reusing the exact
shares_pct_of_volume convention from liquidity_proxies.py, applied here to the EXIT window instead of the
entry window). Also builds a same-day, same-minute OPEN-vs-CLOSE slippage proxy at the trade's own
exit_time, to check whether thin 9:15-9:17 liquidity correlates with worse fills.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

TRADES = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
DIAG = rb.RESULTS / "diagnostic_table.csv"
OUTDIR = rb.RESULTS / "exit_day_915_917_liquidity"; OUTDIR.mkdir(parents=True, exist_ok=True)
IST = "Asia/Kolkata"
WIN_HM = [555, 556, 557]   # 9:15-9:16, 9:16-9:17, 9:17-9:18 candles


def hm_to_lbl(h):
    return f"{int(h)//60:02d}:{int(h)%60:02d}" if h == h else ""


def main():
    tr = pd.read_excel(TRADES, sheet_name="all_trades",
                        usecols=["symbol", "entry_date", "exit_date", "category", "long_exit_type",
                                 "exit_time", "exit_price", "shares"])
    tr["entry_date"] = pd.to_datetime(tr["entry_date"]).dt.date
    tr["exit_date"] = pd.to_datetime(tr["exit_date"]).dt.date
    print(f"backtest trades: {len(tr):,} | distinct symbols: {tr['symbol'].nunique():,}", flush=True)

    diag = pd.read_csv(DIAG, usecols=["symbol", "date", "avg_nday_fullday_volume"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    tr = tr.merge(diag.rename(columns={"date": "entry_date", "avg_nday_fullday_volume": "adv_36day"}),
                  on=["symbol", "entry_date"], how="left")
    print(f"joined 36-day ADV: {tr['adv_36day'].notna().sum():,} of {len(tr):,} trades", flush=True)

    # exit_time ("HH:MM" string) -> hm minutes, for the exit-minute open/close slippage check
    def hhmm_to_hm(s):
        if not isinstance(s, str) or ":" not in s:
            return np.nan
        h, m = s.split(":"); return int(h) * 60 + int(m)
    tr["exit_hm"] = tr["exit_time"].apply(hhmm_to_hm)

    rows, missing = [], []
    for sym, gtr in tr.groupby("symbol", sort=False):
        p = rb.MASTER_DIR / f"{sym}.parquet"
        if not p.exists():
            for r in gtr.itertuples():
                missing.append({"symbol": sym, "exit_date": str(r.exit_date), "reason": "no_1min_symbol"})
            continue
        df = pd.read_parquet(p, columns=["timestamp", "open", "close", "volume"]).astype(
            {"open": float, "close": float, "volume": float})
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
        hm = ts.dt.hour * 60 + ts.dt.minute
        d = ts.dt.date
        win_mask = hm.isin(WIN_HM)
        win = df[win_mask].assign(d=d[win_mask])
        wagg = win.groupby("d")["volume"].agg(vol_915_917="sum", n_candles="size").to_dict("index")
        by_hm_close = {(dd, hh): cc for dd, hh, cc in zip(d, hm, df["close"])}
        by_hm_open = {(dd, hh): oo for dd, hh, oo in zip(d, hm, df["open"])}

        for r in gtr.itertuples():
            rec = wagg.get(r.exit_date)
            if rec is None:
                missing.append({"symbol": sym, "exit_date": str(r.exit_date), "reason": "no_915_917_candles_on_exit_date"})
                continue
            vol_w = float(rec["vol_915_917"]); n_c = int(rec["n_candles"])
            adv36 = r.adv_36day
            pct_of_adv = round(vol_w / adv36 * 100, 4) if (adv36 == adv36 and adv36 > 0) else np.nan
            shares_pct_of_win = round(r.shares / vol_w * 100, 3) if vol_w > 0 else np.nan

            close_at_exit = by_hm_close.get((r.exit_date, r.exit_hm)) if r.exit_hm == r.exit_hm else None
            slippage_pct = round((r.exit_price - close_at_exit) / close_at_exit * 100, 3) \
                if close_at_exit not in (None, 0) and close_at_exit == close_at_exit else np.nan

            rows.append({
                "symbol": sym, "entry_date": r.entry_date, "exit_date": r.exit_date, "category": r.category,
                "long_exit_type": r.long_exit_type, "exit_time": r.exit_time, "exit_price": r.exit_price,
                "exit_shares": int(r.shares), "vol_915_917_shares": int(vol_w), "n_candles_present": n_c,
                "adv_36day": round(adv36, 0) if adv36 == adv36 else np.nan,
                "vol_915_917_pct_of_adv36": pct_of_adv,
                "exit_shares_pct_of_915_917_vol": shares_pct_of_win,
                "close_at_exit_minute": round(close_at_exit, 4) if close_at_exit == close_at_exit and close_at_exit is not None else np.nan,
                "slippage_pct_vs_exit_minute_close": slippage_pct,
            })
    R = pd.DataFrame(rows)
    MISS = pd.DataFrame(missing)
    print(f"per-trade rows computed: {len(R):,} | missing/excluded: {len(MISS):,}", flush=True)

    # ---- summary stats ----
    valid_pct = R["vol_915_917_pct_of_adv36"].dropna()
    summary = {
        "n_trades_with_valid_liquidity_data": len(valid_pct),
        "avg_vol_915_917_shares": round(R["vol_915_917_shares"].mean(), 0),
        "median_vol_915_917_shares": round(R["vol_915_917_shares"].median(), 0),
        "avg_pct_of_adv36": round(valid_pct.mean(), 3),
        "median_pct_of_adv36": round(valid_pct.median(), 3),
        "avg_exit_shares_pct_of_win_vol": round(R["exit_shares_pct_of_915_917_vol"].mean(), 3),
        "median_exit_shares_pct_of_win_vol": round(R["exit_shares_pct_of_915_917_vol"].median(), 3),
    }
    bins = [-np.inf, 5, 10, 20, np.inf]
    labels = ["<5%", "5-10%", "10-20%", ">20%"]
    R["adv36_bucket"] = pd.cut(R["vol_915_917_pct_of_adv36"], bins=bins, labels=labels)
    DIST = R["adv36_bucket"].value_counts(normalize=True).reindex(labels).mul(100).round(2).rename("pct_of_trades").reset_index().rename(columns={"index": "adv36_bucket"})
    DIST_n = R["adv36_bucket"].value_counts().reindex(labels).rename("n_trades").reset_index().rename(columns={"index": "adv36_bucket"})
    DIST = DIST.merge(DIST_n, on="adv36_bucket")

    # ---- bottom 5-10% least-liquid trades (by pct_of_adv36) ----
    valid_R = R.dropna(subset=["vol_915_917_pct_of_adv36"]).copy()
    n_bottom5 = max(1, int(round(len(valid_R) * 0.05)))
    n_bottom10 = max(1, int(round(len(valid_R) * 0.10)))
    worst5 = valid_R.nsmallest(n_bottom5, "vol_915_917_pct_of_adv36")
    worst10 = valid_R.nsmallest(n_bottom10, "vol_915_917_pct_of_adv36")

    # ---- liquidity vs slippage cross-check ----
    v = R.dropna(subset=["vol_915_917_pct_of_adv36", "slippage_pct_vs_exit_minute_close"]).copy()
    v["abs_slip"] = v["slippage_pct_vs_exit_minute_close"].abs()
    corr_pct_adv = float(np.corrcoef(v["vol_915_917_pct_of_adv36"], v["abs_slip"])[0, 1]) if len(v) > 2 else np.nan
    v2 = R.dropna(subset=["exit_shares_pct_of_915_917_vol", "slippage_pct_vs_exit_minute_close"]).copy()
    v2["abs_slip"] = v2["slippage_pct_vs_exit_minute_close"].abs()
    corr_shares_pct = float(np.corrcoef(v2["exit_shares_pct_of_915_917_vol"], v2["abs_slip"])[0, 1]) if len(v2) > 2 else np.nan
    thresh = valid_R["vol_915_917_pct_of_adv36"].quantile(0.10)
    low_liq_mask = v["vol_915_917_pct_of_adv36"] <= thresh
    avg_slip_low = round(v.loc[low_liq_mask, "abs_slip"].mean(), 3) if low_liq_mask.any() else np.nan
    avg_slip_rest = round(v.loc[~low_liq_mask, "abs_slip"].mean(), 3) if (~low_liq_mask).any() else np.nan

    pd.set_option("display.width", 240)
    print("\n=== SUMMARY ===")
    for k, val in summary.items():
        print(f"  {k}: {val}")
    print("\n=== DISTRIBUTION: 9:15-9:17 volume as % of 36-day ADV ===")
    print(DIST.to_string(index=False))
    print(f"\n=== LIQUIDITY vs SLIPPAGE (n={len(v)} trades with both liquidity & slippage data) ===")
    print(f"  corr(vol_915_917_pct_of_adv36, |slippage_pct|)       = {corr_pct_adv:+.3f}")
    print(f"  corr(exit_shares_pct_of_915_917_vol, |slippage_pct|) = {corr_shares_pct:+.3f}")
    print(f"  bottom-10%-liquidity trades: avg |slippage| = {avg_slip_low}%  |  rest: avg |slippage| = {avg_slip_rest}%")
    print(f"\n=== BOTTOM 5% least-liquid exits (n={len(worst5)}) ===")
    print(worst5[["symbol", "entry_date", "exit_date", "long_exit_type", "vol_915_917_shares", "adv_36day",
                  "vol_915_917_pct_of_adv36", "exit_shares", "exit_shares_pct_of_915_917_vol"]].to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "exit_day_915_917_liquidity_check.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "Diagnostic/data-quality check on the LOCKED baseline strategy's existing trades (no logic "
                      "changes). Measures traded volume in the 9:15-9:17 window (3 one-min candles) on each "
                      "trade's EXIT day, normalized against the 36-day average full-day volume already used as "
                      "the entry filter (diagnostic_table.csv::avg_nday_fullday_volume, as of the trade's ENTRY "
                      "date -- the only pre-computed ADV reference the strategy itself uses)."},
            {"note": "exit_shares_pct_of_915_917_vol reuses the exact shares_pct_of_volume fill-realism "
                      "convention from liquidity_proxies.py (originally applied to the 15:00-15:22 entry window), "
                      "applied here to the 9:15-9:17 exit window instead."},
            {"note": "slippage_pct_vs_exit_minute_close compares the backtest's assumed exit_price (an open price "
                      "or the 17% target limit price, per the locked exit logic) against that SAME MINUTE's close "
                      "price on the exit day -- a same-day, same-minute open/close divergence proxy, not a live-"
                      "tradebook comparison (no live exit tradebook exists for this cross-check)."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        pd.DataFrame([summary]).to_excel(w, sheet_name="Summary", index=False)
        DIST.to_excel(w, sheet_name="ADV36_Bucket_Distribution", index=False)
        pd.DataFrame([{
            "corr_pct_of_adv36_vs_abs_slippage": round(corr_pct_adv, 3),
            "corr_exit_shares_pct_of_win_vol_vs_abs_slippage": round(corr_shares_pct, 3),
            "bottom10pct_liquidity_avg_abs_slippage": avg_slip_low,
            "rest_avg_abs_slippage": avg_slip_rest,
            "n_trades_in_correlation": len(v),
        }]).to_excel(w, sheet_name="Liquidity_vs_Slippage", index=False)
        worst5.to_excel(w, sheet_name="Bottom_5pct_Least_Liquid", index=False)
        worst10.to_excel(w, sheet_name="Bottom_10pct_Least_Liquid", index=False)
        R.to_excel(w, sheet_name="per_trade_full", index=False)
        if not MISS.empty:
            MISS.to_excel(w, sheet_name="missing", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)

    R.to_csv(OUTDIR / "per_trade_exit_liquidity.csv", index=False)
    print(f"\nSaved -> {OUTDIR}/exit_day_915_917_liquidity_check.xlsx")


if __name__ == "__main__":
    main()
