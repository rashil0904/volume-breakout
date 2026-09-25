# -*- coding: utf-8 -*-
"""run_long_breakout_k25_n20_allstocks.py — same as run_long_breakout_k25_n20.py but on the FULL
master_data universe (~1609 symbols, all NSE stocks with data), not just the reconciled F&O baseline.
Same default config (K=2.5, N=20, 15% stop, T+20, liquidity filter disabled, 10bps cost) — the strategy's
own docstring flags that non-F&O names are more prone to illiquidity/circuit-freeze artifacts, so results
here should be read with that caveat rather than assumed to carry the same track record as the F&O run.
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import long_breakout_k25_n20 as strat

MD = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "long_breakout_k25_n20_allstocks"; OUTDIR.mkdir(parents=True, exist_ok=True)


def load_daily(symbol):
    df = pd.read_parquet(MD / f"{symbol}.parquet", columns=["timestamp", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.assign(ts=ts).sort_values("ts")
    df["date"] = df["ts"].dt.normalize()
    daily = df.groupby("date").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                    close=("close", "last"), volume=("volume", "sum")).reset_index()
    return daily


def main():
    files = sorted(MD.glob("*.parquet"))
    syms = [f.stem for f in files]
    print(f"full universe: {len(syms)} symbols in master_data/", flush=True)

    price_data = {}; errs = []
    for i, s in enumerate(syms, 1):
        try:
            d = load_daily(s)
            if len(d) >= 25:            # need at least LOOKBACK+few rows to ever fire a signal
                price_data[s] = d
        except Exception as e:
            errs.append((s, str(e)[:80]))
        if i % 200 == 0:
            print(f"  loaded {i}/{len(syms)}", flush=True)
    print(f"loaded {len(price_data)} symbols (usable) | errors {len(errs)}", flush=True)

    trades = strat.run_backtest(price_data)
    print(f"total trades: {len(trades)}", flush=True)

    overall = strat.summarize(trades)
    print("\n=== OVERALL (ALL STOCKS) ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    per_symbol = trades.groupby("symbol").apply(
        lambda g: pd.Series({"n_trades": len(g), "win_rate_%": round((g.net_return > 0).mean() * 100, 1),
                              "mean_net_return_%": round(g.net_return.mean() * 100, 3),
                              "total_net_return_%": round(g.net_return.sum() * 100, 2)}),
        include_groups=False).reset_index().sort_values("total_net_return_%", ascending=False)

    yearly = trades.copy(); yearly["year"] = pd.to_datetime(yearly["entry_date"]).dt.year
    yearly_stats = yearly.groupby("year").apply(
        lambda g: pd.Series({"n_trades": len(g), "win_rate_%": round((g.net_return > 0).mean() * 100, 1),
                              "mean_net_return_%": round(g.net_return.mean() * 100, 3)}),
        include_groups=False).reset_index()

    # flag suspicious extreme-return trades (possible circuit-freeze / thin-data artifacts)
    extreme = trades[(trades.net_return.abs() > 0.5)].sort_values("net_return", ascending=False)

    with pd.ExcelWriter(OUTDIR / "long_breakout_k25_n20_allstocks.xlsx", engine="openpyxl") as w:
        pd.DataFrame([overall]).to_excel(w, sheet_name="Overall", index=False)
        yearly_stats.to_excel(w, sheet_name="By_Year", index=False)
        per_symbol.to_excel(w, sheet_name="By_Symbol", index=False)
        extreme.to_excel(w, sheet_name="Extreme_Trades_gt50pct", index=False)
        trades.to_excel(w, sheet_name="Trades", index=False)
        if errs:
            pd.DataFrame(errs, columns=["symbol", "error"]).to_excel(w, sheet_name="Load_Errors", index=False)
    trades.to_csv(OUTDIR / "long_breakout_k25_n20_allstocks_trades.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== BY YEAR ===")
    print(yearly_stats.to_string(index=False))
    print("\n=== TOP 15 SYMBOLS BY TOTAL NET RETURN ===")
    print(per_symbol.head(15).to_string(index=False))
    print("\n=== BOTTOM 15 SYMBOLS BY TOTAL NET RETURN ===")
    print(per_symbol.tail(15).to_string(index=False))
    print(f"\n=== EXTREME TRADES (|net return| > 50%): {len(extreme)} ===")
    print(extreme.head(15).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/long_breakout_k25_n20_allstocks.xlsx")


if __name__ == "__main__":
    main()
