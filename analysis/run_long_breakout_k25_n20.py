# -*- coding: utf-8 -*-
"""run_long_breakout_k25_n20.py — runs long_breakout_k25_n20.run_backtest() on this project's real data:
daily OHLC aggregated from master_data/{SYM}.parquet (1-min) for the reconciled F&O universe (206 symbols,
204 w/ data), matching the universe convention used by fno_3leg_stocks.py elsewhere in this repo. Default
config as pasted: K=2.5, LOOKBACK=20, STOP_PCT=0.15, TIME_EXIT_DAYS=20, liquidity filter disabled.
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import long_breakout_k25_n20 as strat

MD = rb.BASE / "master_data"
FO_CSV = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
OUTDIR = rb.RESULTS / "long_breakout_k25_n20"; OUTDIR.mkdir(parents=True, exist_ok=True)


def load_daily(symbol):
    df = pd.read_parquet(MD / f"{symbol}.parquet", columns=["timestamp", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.assign(ts=ts).sort_values("ts")
    df["date"] = df["ts"].dt.normalize()
    daily = df.groupby("date").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                    close=("close", "last"), volume=("volume", "sum")).reset_index()
    return daily


def main():
    syms = sorted(pd.read_csv(FO_CSV, usecols=["symbol"])["symbol"].unique())
    syms = [s for s in syms if (MD / f"{s}.parquet").exists()]
    print(f"F&O universe with data: {len(syms)} stocks", flush=True)

    price_data = {}; errs = []
    for i, s in enumerate(syms, 1):
        try:
            price_data[s] = load_daily(s)
        except Exception as e:
            errs.append((s, str(e)[:80]))
        if i % 50 == 0:
            print(f"  loaded {i}/{len(syms)}", flush=True)
    print(f"loaded {len(price_data)} symbols | errors {len(errs)}", flush=True)

    trades = strat.run_backtest(price_data)
    print(f"total trades: {len(trades)}", flush=True)

    overall = strat.summarize(trades)
    print("\n=== OVERALL ===")
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

    with pd.ExcelWriter(OUTDIR / "long_breakout_k25_n20.xlsx", engine="openpyxl") as w:
        pd.DataFrame([overall]).to_excel(w, sheet_name="Overall", index=False)
        yearly_stats.to_excel(w, sheet_name="By_Year", index=False)
        per_symbol.to_excel(w, sheet_name="By_Symbol", index=False)
        trades.to_excel(w, sheet_name="Trades", index=False)
        if errs:
            pd.DataFrame(errs, columns=["symbol", "error"]).to_excel(w, sheet_name="Load_Errors", index=False)
    trades.to_csv(OUTDIR / "long_breakout_k25_n20_trades.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== BY YEAR ===")
    print(yearly_stats.to_string(index=False))
    print("\n=== TOP 10 SYMBOLS BY TOTAL NET RETURN ===")
    print(per_symbol.head(10).to_string(index=False))
    print("\n=== BOTTOM 10 SYMBOLS BY TOTAL NET RETURN ===")
    print(per_symbol.tail(10).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}/long_breakout_k25_n20.xlsx")


if __name__ == "__main__":
    main()
