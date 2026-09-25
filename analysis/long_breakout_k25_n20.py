# -*- coding: utf-8 -*-
"""
Long Breakout Strategy — K=2.5, N=20 lookback, 15% fixed stop, 20-day time exit
================================================================================

Signal:  Enter LONG when a stock's daily close crosses from at/below its
         upper Bollinger-style band to above it, where:
             SMA_N   = rolling N-day simple moving average of close
             STD_N   = rolling N-day standard deviation of close (ddof=1)
             Upper_N = SMA_N + K * STD_N

Entry:   Next trading day's OPEN after the signal close (T+1 open).

Exit:    Whichever happens first:
           1. Close falls to or below entry_price * (1 - stop_pct)   -> "stop"
           2. holding_days reaches time_exit_days                     -> "time"

Re-arming: after a signal fires for a stock, no new signal can fire for
that stock until its close is back at/below SMA_N at least once.

This is a reference implementation matching the spec used throughout the
2sigma/2.5sigma breakout research. Feed it a dict of {symbol: DataFrame}
with daily OHLC data (columns: 'date', 'open', 'high', 'low', 'close').

NOTE ON PARAMETERS: K=2.5 / N=14 tested slightly better in the full sweep
(2.08% mean return vs 1.98% for N=20 at otherwise identical settings).
N=20 is still close to the best result, not a mistake — just not the top
one. Change LOOKBACK below to 14 if you want the single best-tested config
instead.

NOTE ON NON-F&O UNIVERSE: every prior backtest in this research (the
2.08% mean return, 54.5% win rate, etc.) was tested ONLY on the ~208
F&O stock universe. This script's logic is universe-agnostic and will
run on any set of OHLC data you give it, including non-F&O stocks — but
those results do NOT carry an existing track record and need to be
evaluated fresh. Non-F&O names are typically less liquid, more prone to
circuit-filter freezes (no trading for a day+), stale/thin price data,
and lower-float spikes that can look like a genuine breakout but aren't.
A basic average-turnover liquidity filter is included below
(MIN_AVG_TURNOVER_INR) — set it to 0 to disable, or raise it if you want
to exclude thinner names before running the backtest.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration — this is the exact config requested: K=2.5, N=20, 15% stop, T+20
# ---------------------------------------------------------------------------
K = 2.5                # band width multiplier (std devs above the mean)
LOOKBACK = 20           # rolling window (days) for SMA and STD
STOP_PCT = 0.15         # fixed stop-loss, 15% below entry
TIME_EXIT_DAYS = 20     # max holding period in trading days
COST_BPS_ROUNDTRIP = 10 # placeholder round-trip cost assumption, in bps
                        # (consider raising this for non-F&O / thinner names —
                        # 10 bps was already a generous assumption for F&O stocks)

MIN_AVG_TURNOVER_INR = 0   # optional liquidity filter — minimum average daily
                           # turnover (price * volume, in INR) over the same
                           # LOOKBACK window, required to allow a signal to fire.
                           # 0 = disabled. Needs a 'volume' column in the input
                           # data if enabled. Useful when running on non-F&O
                           # stocks to filter out illiquid/thinly-traded names.


@dataclass
class Trade:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str
    holding_days: int
    gross_return: float
    net_return: float


def compute_bands(df: pd.DataFrame, lookback: int = LOOKBACK, k: float = K) -> pd.DataFrame:
    """Add SMA, STD, and upper band columns to a daily OHLC dataframe."""
    df = df.copy()
    df["sma"] = df["close"].rolling(lookback).mean()
    df["std"] = df["close"].rolling(lookback).std(ddof=1)
    df["upper_band"] = df["sma"] + k * df["std"]
    if "volume" in df.columns:
        df["avg_turnover"] = (df["close"] * df["volume"]).rolling(lookback).mean()
    return df


def find_signals(df: pd.DataFrame, min_avg_turnover: float = MIN_AVG_TURNOVER_INR) -> list[int]:
    """
    Return the integer row-indices (into df) where a long entry signal fires:
    close crosses from <= upper_band to > upper_band, with re-arming
    (must close back at/below SMA before the next signal can fire).

    If min_avg_turnover > 0 and an 'avg_turnover' column is present
    (requires 'volume' in the input data), signals on days where average
    turnover is below the threshold are skipped — a basic liquidity filter
    useful for non-F&O / thinner names.
    """
    signals = []
    armed = True  # can a new signal fire?
    has_liquidity_filter = min_avg_turnover > 0 and "avg_turnover" in df.columns

    for i in range(1, len(df)):
        prev_close, prev_upper = df["close"].iat[i - 1], df["upper_band"].iat[i - 1]
        cur_close, cur_upper, cur_sma = df["close"].iat[i], df["upper_band"].iat[i], df["sma"].iat[i]

        if pd.isna(prev_upper) or pd.isna(cur_upper):
            continue

        crossed_above = (prev_close <= prev_upper) and (cur_close > cur_upper)

        if crossed_above and armed:
            liquidity_ok = True
            if has_liquidity_filter:
                turnover = df["avg_turnover"].iat[i]
                liquidity_ok = not pd.isna(turnover) and turnover >= min_avg_turnover
            if liquidity_ok:
                signals.append(i)
                armed = False  # disarm until price closes back at/below SMA

        # re-arm once price closes back at/below its own SMA
        if not armed and not pd.isna(cur_sma) and cur_close <= cur_sma:
            armed = True

    return signals


def simulate_trade(df: pd.DataFrame, signal_idx: int, symbol: str,
                    stop_pct: float = STOP_PCT,
                    time_exit_days: int = TIME_EXIT_DAYS,
                    cost_bps: float = COST_BPS_ROUNDTRIP) -> "Trade | None":
    """
    Simulate a single trade starting the day after signal_idx (T+1 open entry).
    Returns None if there isn't enough forward data to enter.
    """
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None  # no next-day data available

    signal_date = df["date"].iat[signal_idx]
    entry_date = df["date"].iat[entry_idx]
    entry_price = df["open"].iat[entry_idx]
    stop_level = entry_price * (1 - stop_pct)

    exit_idx = None
    exit_reason = None

    max_idx = min(entry_idx + time_exit_days, len(df) - 1)
    for j in range(entry_idx, max_idx + 1):
        close_j = df["close"].iat[j]
        holding_days = j - entry_idx

        if close_j <= stop_level:
            exit_idx = j
            exit_reason = "stop"
            break

        if holding_days >= time_exit_days:
            exit_idx = j
            exit_reason = "time"
            break
    else:
        # loop completed without break -> ran out of data before time exit
        exit_idx = max_idx
        exit_reason = "time" if (max_idx - entry_idx) >= time_exit_days else "no_forward_data"

    if exit_idx is None:
        return None

    exit_date = df["date"].iat[exit_idx]
    exit_price = df["close"].iat[exit_idx]
    holding_days = exit_idx - entry_idx

    gross_return = (exit_price / entry_price) - 1
    net_return = gross_return - (cost_bps / 10000.0)

    return Trade(
        symbol=symbol,
        signal_date=signal_date,
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=exit_reason,
        holding_days=holding_days,
        gross_return=gross_return,
        net_return=net_return,
    )


def run_backtest(price_data: "dict[str, pd.DataFrame]",
                  k: float = K, lookback: int = LOOKBACK,
                  stop_pct: float = STOP_PCT,
                  time_exit_days: int = TIME_EXIT_DAYS,
                  min_avg_turnover: float = MIN_AVG_TURNOVER_INR) -> pd.DataFrame:
    """
    price_data: dict mapping symbol -> daily OHLC DataFrame with columns
                ['date','open','high','low','close'], sorted by date ascending.
                Include a 'volume' column and set min_avg_turnover > 0 to
                apply the liquidity filter (recommended for non-F&O stocks).

    Returns a DataFrame of all trades across all symbols.
    """
    all_trades = []

    for symbol, raw_df in price_data.items():
        df = compute_bands(raw_df, lookback=lookback, k=k)
        signal_indices = find_signals(df, min_avg_turnover=min_avg_turnover)

        for sig_idx in signal_indices:
            trade = simulate_trade(df, sig_idx, symbol,
                                    stop_pct=stop_pct,
                                    time_exit_days=time_exit_days)
            if trade is not None:
                all_trades.append(trade)

    if not all_trades:
        return pd.DataFrame(columns=[
            "symbol", "signal_date", "entry_date", "entry_price",
            "exit_date", "exit_price", "exit_reason", "holding_days",
            "gross_return", "net_return",
        ])

    return pd.DataFrame([t.__dict__ for t in all_trades])


def summarize(trades: pd.DataFrame) -> dict:
    """Quick headline stats matching the format used throughout this research."""
    if trades.empty:
        return {"n_trades": 0}
    return {
        "n_trades": len(trades),
        "win_rate_%": round((trades["net_return"] > 0).mean() * 100, 2),
        "mean_net_return_%": round(trades["net_return"].mean() * 100, 3),
        "median_net_return_%": round(trades["net_return"].median() * 100, 3),
        "best_trade_%": round(trades["net_return"].max() * 100, 2),
        "worst_trade_%": round(trades["net_return"].min() * 100, 2),
        "avg_winner_%": round(trades.loc[trades["net_return"] > 0, "net_return"].mean() * 100, 2),
        "avg_loser_%": round(trades.loc[trades["net_return"] <= 0, "net_return"].mean() * 100, 2),
        "avg_holding_days": round(trades["holding_days"].mean(), 1),
        "pct_exit_stop": round((trades["exit_reason"] == "stop").mean() * 100, 1),
        "pct_exit_time": round((trades["exit_reason"] == "time").mean() * 100, 1),
    }


if __name__ == "__main__":
    print("Import this module and call run_backtest(price_data) with your own OHLC data.")
    print(f"Configured for: K={K}, lookback={LOOKBACK}, stop={STOP_PCT*100:.0f}%, "
          f"time_exit={TIME_EXIT_DAYS} days")
