# -*- coding: utf-8 -*-
"""long_breakout_holdperiod_sweep_lt20kcr.py — for the long-breakout (K=2.5, N=20, 15% stop) signals
whose entry-day market cap is <=20,000 Cr (the 1500-5000 + 5000-10000 + 10000-20000 bands from the mcap
segregation), sweeps the TIME_EXIT_DAYS holding period from 1 to 20 to see if a shorter/longer hold beats
the original 20-day config. Signal generation (K/N band crossover) is unchanged and identical across the
sweep -- only the exit day (and hence stop/time-exit outcome) differs per H. Reuses each trade's own
entry_date/entry_price from the already-computed mcap1500cr trade set to relocate entry_idx in that
symbol's daily series, then resimulates the stop-or-time exit for every H in 1..20.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb

MD = rb.BASE / "master_data"
TRADES = rb.RESULTS / "long_breakout_k25_n20_allstocks" / "long_breakout_k25_n20_mcap1500cr_trades.csv"
OUTDIR = rb.RESULTS / "long_breakout_k25_n20_allstocks"; OUTDIR.mkdir(parents=True, exist_ok=True)
STOP_PCT = 0.15
COST_BPS = 10
MAX_H = 20
MCAP_MAX_CR = 20_000


def load_daily(symbol):
    df = pd.read_parquet(MD / f"{symbol}.parquet", columns=["timestamp", "open", "high", "low", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.assign(ts=ts).sort_values("ts")
    df["date"] = df["ts"].dt.normalize()
    daily = df.groupby("date").agg(open=("open", "first"), close=("close", "last")).reset_index()
    return daily


def main():
    T = pd.read_csv(TRADES, parse_dates=["signal_date", "entry_date", "exit_date"])
    T = T[T["entry_mcap_cr"] <= MCAP_MAX_CR].copy()
    print(f"signals with entry mcap <= {MCAP_MAX_CR:,} Cr: {len(T)}", flush=True)

    syms = sorted(T["symbol"].unique())
    print(f"symbols involved: {len(syms)}", flush=True)

    daily_cache = {}
    for i, s in enumerate(syms, 1):
        try:
            daily_cache[s] = load_daily(s)
        except Exception:
            pass
        if i % 100 == 0:
            print(f"  loaded {i}/{len(syms)}", flush=True)

    # for each trade, locate entry_idx in that symbol's daily series
    rows = []
    skipped = 0
    for _, r in T.iterrows():
        sym = r["symbol"]
        d = daily_cache.get(sym)
        if d is None:
            skipped += 1; continue
        idx = d.index[d["date"] == r["entry_date"]]
        if len(idx) == 0:
            skipped += 1; continue
        entry_idx = idx[0]
        entry_price = float(d["close"].iat[entry_idx - 1]) if entry_idx > 0 else float(r["entry_price"])
        # use the SAVED entry_price directly (T+1 open, already correct) instead of re-deriving
        entry_price = float(r["entry_price"])
        stop_level = entry_price * (1 - STOP_PCT)
        max_h_avail = len(d) - 1 - entry_idx
        rows.append((sym, entry_idx, entry_price, stop_level, max_h_avail, d))

    print(f"usable signals: {len(rows)} | skipped (no date match): {skipped}", flush=True)

    sweep_results = []
    for H in range(1, MAX_H + 1):
        rets = []
        for sym, entry_idx, entry_price, stop_level, max_h_avail, d in rows:
            max_idx = min(entry_idx + H, len(d) - 1)
            if max_idx <= entry_idx and max_h_avail < 0:
                continue
            exit_idx = None; reason = None
            close_vals = d["close"].values
            for j in range(entry_idx, max_idx + 1):
                hd = j - entry_idx
                if close_vals[j] <= stop_level:
                    exit_idx = j; reason = "stop"; break
                if hd >= H:
                    exit_idx = j; reason = "time"; break
            if exit_idx is None:
                exit_idx = max_idx; reason = "time" if (max_idx - entry_idx) >= H else "no_forward_data"
            exit_price = float(close_vals[exit_idx])
            gross = exit_price / entry_price - 1
            net = gross - COST_BPS / 10000.0
            rets.append((net, reason))
        R = pd.DataFrame(rets, columns=["net_return", "exit_reason"])
        sweep_results.append({
            "holding_days": H, "n_trades": len(R),
            "win_rate_%": round((R.net_return > 0).mean() * 100, 2),
            "mean_net_return_%": round(R.net_return.mean() * 100, 3),
            "median_net_return_%": round(R.net_return.median() * 100, 3),
            "total_net_return_%": round(R.net_return.sum() * 100, 1),
            "avg_winner_%": round(R.loc[R.net_return > 0, "net_return"].mean() * 100, 2),
            "avg_loser_%": round(R.loc[R.net_return <= 0, "net_return"].mean() * 100, 2),
            "pct_exit_stop": round((R.exit_reason == "stop").mean() * 100, 1),
            "pct_exit_time": round((R.exit_reason == "time").mean() * 100, 1),
        })
        print(f"  H={H:2d} | trades {len(R)} | win% {sweep_results[-1]['win_rate_%']:.1f} | "
              f"mean {sweep_results[-1]['mean_net_return_%']:.3f}% | median {sweep_results[-1]['median_net_return_%']:.3f}%", flush=True)

    SW = pd.DataFrame(sweep_results)
    best_mean = SW.loc[SW["mean_net_return_%"].idxmax()]
    best_median = SW.loc[SW["median_net_return_%"].idxmax()]

    with pd.ExcelWriter(OUTDIR / "holdperiod_sweep_lt20kcr.xlsx", engine="openpyxl") as w:
        SW.to_excel(w, sheet_name="Sweep_H1_to_H20", index=False)
    SW.to_csv(OUTDIR / "holdperiod_sweep_lt20kcr.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(SW.to_string(index=False))
    print(f"\nBest by mean return  : H={int(best_mean.holding_days)} ({best_mean['mean_net_return_%']:.3f}%)")
    print(f"Best by median return: H={int(best_median.holding_days)} ({best_median['median_net_return_%']:.3f}%)")
    print(f"Current default H=20 : mean {SW[SW.holding_days==20]['mean_net_return_%'].values[0]:.3f}% | median {SW[SW.holding_days==20]['median_net_return_%'].values[0]:.3f}%")
    print(f"\nSaved -> {OUTDIR}/holdperiod_sweep_lt20kcr.xlsx")


if __name__ == "__main__":
    main()
