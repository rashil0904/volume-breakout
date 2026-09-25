# -*- coding: utf-8 -*-
"""
liquidity_proxies.py — OHLCV-based fill-realism / liquidity proxies for every backtest trade
(baseline_final all_trades, ~3,436 trades) using 1-min entry-day data.

(a) ALL measures are OHLCV proxies. True liquidity (bid-ask spread, order-book depth) is NOT in
    candle data — these correlate with execution cost but do not measure it.
(b) Two windows, both computed: W = 15:00-15:22 (entry window, hm 900..922, 23 candles) and
    D = full session 09:15-15:29 (hm 555..929). The entry window is the one that matters for the
    3:15 fill; the full day is the classic %-of-ADV context.
(c) trade-size-to-volume uses the trade's ACTUAL shares / capital_deployed vs the window volume/turnover.

Per trade & window:
  turnover        = SUM(close*volume)                      (rupee traded value)   [primary proxy #1]
  avg_candle_turnover = turnover / n_candles
  shares_pct_of_volume   = shares  / total_volume  * 100                          [#2 key executability]
  capital_pct_of_turnover= capital / turnover      * 100
  avg_candle_range_pct   = mean((high-low)/low)*100 per candle                    [#3 spread/impact proxy]
  n_zero_vol_candles     = count(volume==0)                                       [#4 sporadic-trade flag]
  window_low, avg_vol_per_candle, product = window_low*avg_vol_per_candle         [#5 continuity]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

TRADES = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
MASTER = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "liquidity_proxies"
SLIP = rb.RESULTS / "entry_slippage" / "entry_slippage_per_trade.csv"
TB = Path.home() / "Downloads" / "tradebook.xlsx"
IST = "Asia/Kolkata"
W_LO, W_HI = 900, 922       # 15:00..15:22 inclusive
D_LO, D_HI = 555, 929       # 09:15..15:29 inclusive


def agg_window(df, ts, hm, lo_m, hi_m):
    """Per-date aggregates over [lo_m, hi_m] inclusive. Returns dict date->metrics."""
    mask = (hm >= lo_m) & (hm <= hi_m)
    sub = df[mask].copy()
    sub["d"] = ts[mask].dt.date
    sub["cv"] = sub["close"] * sub["volume"]
    sub["rng"] = np.where(sub["low"] > 0, (sub["high"] - sub["low"]) / sub["low"] * 100.0, np.nan)
    sub["zv"] = (sub["volume"] == 0).astype(int)
    g = sub.groupby("d")
    out = pd.DataFrame({
        "turnover": g["cv"].sum(), "volume": g["volume"].sum(), "n": g["volume"].size(),
        "low": g["low"].min(), "range_pct": g["rng"].mean(), "zero_vol": g["zv"].sum()})
    return out.to_dict("index")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tr = pd.read_excel(TRADES, sheet_name="all_trades", usecols=["symbol", "entry_date", "shares", "capital_deployed"])
    tr["entry_date"] = pd.to_datetime(tr["entry_date"]).dt.date
    print(f"backtest trades: {len(tr):,} | distinct symbols: {tr['symbol'].nunique():,}")

    rows, missing = [], []
    for sym, gtr in tr.groupby("symbol", sort=False):
        p = MASTER / f"{sym}.parquet"
        if not p.exists():
            for r in gtr.itertuples():
                missing.append({"symbol": sym, "entry_date": str(r.entry_date), "reason": "no_1min_symbol"})
            continue
        df = pd.read_parquet(p, columns=["timestamp", "high", "low", "close", "volume"]).astype(
            {"high": float, "low": float, "close": float, "volume": float})
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
        hm = ts.dt.hour * 60 + ts.dt.minute
        W = agg_window(df, ts, hm, W_LO, W_HI)
        D = agg_window(df, ts, hm, D_LO, D_HI)
        for r in gtr.itertuples():
            w = W.get(r.entry_date)
            if w is None:
                missing.append({"symbol": sym, "entry_date": str(r.entry_date), "reason": "no_1500_1522_candles"})
                continue
            d = D.get(r.entry_date, {})
            sh, cap = float(r.shares), float(r.capital_deployed)
            nw = int(w["n"]); tw = float(w["turnover"]); vw = float(w["volume"])
            nd = int(d.get("n", 0)); td = float(d.get("turnover", np.nan)); vd = float(d.get("volume", np.nan))
            avgvw = vw / nw if nw else np.nan
            rows.append({
                "symbol": sym, "entry_date": str(r.entry_date), "shares": int(sh), "capital_deployed": round(cap, 0),
                # --- entry window 15:00-15:22 (primary) ---
                "window_turnover": round(tw, 0), "avg_candle_turnover": round(tw / nw, 0) if nw else np.nan,
                "shares_pct_of_volume": round(sh / vw * 100, 3) if vw else np.nan,
                "capital_pct_of_turnover": round(cap / tw * 100, 3) if tw else np.nan,
                "avg_candle_range_pct": round(float(w["range_pct"]), 4),
                "n_zero_vol_candles": int(w["zero_vol"]), "n_candles_window": nw,
                "window_low": round(float(w["low"]), 4), "avg_vol_per_candle": round(avgvw, 2),
                "product": round(float(w["low"]) * avgvw, 2),
                # --- full day 09:15-15:29 (context / %ADV) ---
                "day_turnover": round(td, 0), "shares_pct_of_DAY_volume": round(sh / vd * 100, 3) if vd else np.nan,
                "capital_pct_of_DAY_turnover": round(cap / td * 100, 3) if td else np.nan,
                "day_avg_candle_range_pct": round(float(d.get("range_pct", np.nan)), 4),
                "day_n_zero_vol_candles": int(d.get("zero_vol", 0)), "n_candles_day": nd})
    R = pd.DataFrame(rows)
    MISS = pd.DataFrame(missing)

    # ── per-symbol liquidity profile (repeat-traded get a profile) ──
    g = R.groupby("symbol")
    PS = pd.DataFrame({
        "n_trades": g.size(),
        "avg_window_turnover": g["window_turnover"].mean().round(0),
        "avg_day_turnover": g["day_turnover"].mean().round(0),
        "avg_shares_pct_of_volume": g["shares_pct_of_volume"].mean().round(3),
        "avg_capital_pct_of_turnover": g["capital_pct_of_turnover"].mean().round(3),
        "avg_candle_range_pct": g["avg_candle_range_pct"].mean().round(4),
        "avg_n_zero_vol_candles": g["n_zero_vol_candles"].mean().round(2),
        "avg_shares_pct_of_DAY_volume": g["shares_pct_of_DAY_volume"].mean().round(3),
    }).reset_index().sort_values("avg_shares_pct_of_volume", ascending=False).reset_index(drop=True)

    # ── least-liquid trades (fills least trustworthy) ──
    worst = R.sort_values(["shares_pct_of_volume", "window_turnover"], ascending=[False, True]).head(50)

    # ── distribution of shares_pct_of_volume (danger zone) ──
    def dist(col, label):
        s = R[col].dropna()
        return {"basis": label, "n": len(s),
                "median_pct": round(float(s.median()), 3), "mean_pct": round(float(s.mean()), 3),
                "p90": round(float(s.quantile(0.90)), 3), "p99": round(float(s.quantile(0.99)), 3),
                "pct_trades_gt_1pct": round(float((s > 1).mean() * 100), 1),
                "pct_trades_gt_5pct": round(float((s > 5).mean() * 100), 1),
                "pct_trades_gt_10pct": round(float((s > 10).mean() * 100), 1),
                "pct_trades_gt_25pct": round(float((s > 25).mean() * 100), 1)}
    DIST = pd.DataFrame([dist("shares_pct_of_volume", "vs 15:00-15:22 window volume"),
                         dist("shares_pct_of_DAY_volume", "vs full-day volume (%ADV-like)")])

    # ── informal liquidity <-> live slippage tie-in ──
    corr_note = "live tradebook slippage file not found — skipped"
    try:
        sl = pd.read_csv(SLIP)
        tb = pd.read_excel(TB, sheet_name="trade_book")
        tb["stock"] = tb["Stock Name"].astype(str).str.strip().str.upper()
        tb["entry_date"] = pd.to_datetime(tb["Position entry date"]).dt.date.astype(str)
        m = sl.merge(tb[["stock", "entry_date", "No of shares"]], on=["stock", "entry_date"], how="left")
        # window volume for those trades = avg_1min_vol_1500_1522 * 23 (from the slippage output)
        m["win_vol"] = m["avg_1min_vol_1500_1522"] * 23
        m["shares_pct_win"] = m["No of shares"] / m["win_vol"] * 100
        m["abs_slip"] = m["slippage_pct"].abs()
        v = m.dropna(subset=["shares_pct_win", "abs_slip"])
        c1 = float(np.corrcoef(v["shares_pct_win"], v["abs_slip"])[0, 1])
        c2 = float(np.corrcoef(m.dropna(subset=["avg_1min_vol_1500_1522"])["avg_1min_vol_1500_1522"],
                               m.dropna(subset=["avg_1min_vol_1500_1522"])["abs_slip"])[0, 1])
        corr_note = (f"n={len(v)} live trades | corr(shares_%_of_window_vol, |slippage|)={c1:+.2f} | "
                     f"corr(window_vol, |slippage|)={c2:+.2f}")
        m[["stock", "entry_date", "No of shares", "win_vol", "shares_pct_win", "slippage_pct", "abs_slip"]] \
            .to_csv(OUTDIR / "live_slippage_vs_liquidity.csv", index=False)
    except Exception as e:
        corr_note = f"tie-in skipped: {e}"

    R.to_parquet(OUTDIR / "per_trade_liquidity.parquet", index=False)
    R.to_csv(OUTDIR / "per_trade_liquidity.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "liquidity_proxies.xlsx", engine="openpyxl") as w:
        R.to_excel(w, sheet_name="per_trade", index=False)
        PS.to_excel(w, sheet_name="per_symbol_profile", index=False)
        worst.to_excel(w, sheet_name="least_liquid_trades_top50", index=False)
        DIST.to_excel(w, sheet_name="shares_pct_distribution", index=False)
        if not MISS.empty:
            MISS.to_excel(w, sheet_name="missing", index=False)

    pd.set_option("display.width", 240)
    print("=" * 100)
    print("LIQUIDITY / FILL-REALISM PROXIES (OHLCV-based; no order-book/spread data)")
    print("=" * 100)
    print(f"per-trade rows: {len(R):,} | symbols: {R['symbol'].nunique():,} | missing: {len(MISS)}")
    print("\n--- shares_pct_of_volume DISTRIBUTION (danger zone for fill realism) ---")
    print(DIST.to_string(index=False))
    print("\n--- LEAST-LIQUID TRADES (top 12 by shares_%_of_window_volume) ---")
    print(worst[["symbol", "entry_date", "shares", "shares_pct_of_volume", "window_turnover",
                 "capital_pct_of_turnover", "avg_candle_range_pct", "n_zero_vol_candles"]].head(12).to_string(index=False))
    print("\n--- PER-SYMBOL PROFILE — WORST 12 liquidity (highest avg shares_%_of_window_vol) ---")
    print(PS.head(12)[["symbol", "n_trades", "avg_shares_pct_of_volume", "avg_window_turnover",
                       "avg_candle_range_pct", "avg_n_zero_vol_candles"]].to_string(index=False))
    print("\n--- PER-SYMBOL PROFILE — BEST 8 liquidity (most liquid) ---")
    print(PS.tail(8)[["symbol", "n_trades", "avg_shares_pct_of_volume", "avg_window_turnover", "avg_candle_range_pct"]].to_string(index=False))
    print(f"\n--- INFORMAL liquidity <-> live slippage tie-in ---\n  {corr_note}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
