# -*- coding: utf-8 -*-
"""breakout_2day_pure.py — PURE NIFTY daily 2-day high/low breakout: 0-point margin (exact touch of the raw
previous_2day_high/low), NO gap handling. Always-in flip (+1/-1), intraday 1-min touch, entry/exit at the
exact level. GROSS index points. Saves trades + cumulative-return equity curve (x=time, y=NIFTY points)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.dates as mdates
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DATA = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "breakout_2day_pure"; OUTDIR.mkdir(parents=True, exist_ok=True)
MARGIN = 0.0                                       # PURE: exact-touch, no buffer


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True); d["date"] = d["ts"].dt.normalize()
    daily = d.groupby("date").agg(dh=("high", "max"), dl=("low", "min"))
    daily["p2h"] = daily["dh"].shift(1).rolling(2).max(); daily["p2l"] = daily["dl"].shift(1).rolling(2).min()
    daily["up"] = daily["p2h"] + MARGIN; daily["down"] = daily["p2l"] - MARGIN          # margin 0 -> up=p2h, down=p2l
    dm = d.merge(daily[["up", "down"]], left_on="date", right_index=True, how="left")
    up_a, dn_a, hi_a, lo_a, op_a, tsr = dm["up"].values, dm["down"].values, dm["high"].values, dm["low"].values, dm["open"].values, dm["ts"].values
    period = f"{daily.index[0].date()} .. {daily.index[-1].date()}"

    pos = 0; entry = np.nan; entry_t = None; trades = []; sig = 0
    for k in range(len(dm)):
        up, dn = up_a[k], dn_a[k]
        if np.isnan(up):
            continue
        h, l, o, t = hi_a[k], lo_a[k], op_a[k], tsr[k]
        # FILL REALISM: fill at the trigger level only if the level actually traded in this candle;
        # if the candle GAPPED THROUGH it (level outside [low,high]), fill at the candle OPEN (the real price).
        fill_up = up if l <= up else o                            # long-side fill (gap-open if price gapped above the level)
        fill_dn = dn if h >= dn else o                            # short-side fill (gap-open if price gapped below the level)
        if pos == 0:                                              # first trade: wait for first breakout
            hu, hd = h >= up, l <= dn
            if hu and hd:
                pos, entry = (1, fill_up) if abs(o - up) <= abs(o - dn) else (-1, fill_dn); entry_t = t; sig += 1
            elif hu: pos, entry, entry_t = 1, fill_up, t; sig += 1
            elif hd: pos, entry, entry_t = -1, fill_dn, t; sig += 1
        elif pos == 1:
            if l <= dn:
                trades.append(("Long", entry_t, entry, t, fill_dn, fill_dn - entry)); pos, entry, entry_t = -1, fill_dn, t; sig += 1
        else:
            if h >= up:
                trades.append(("Short", entry_t, entry, t, fill_up, entry - fill_up)); pos, entry, entry_t = 1, fill_up, t; sig += 1

    T = pd.DataFrame(trades, columns=["direction", "entry_time", "entry_price", "exit_time", "exit_price", "points_pnl"])
    T["hold_cal_days"] = (pd.to_datetime(T["exit_time"]).dt.normalize() - pd.to_datetime(T["entry_time"]).dt.normalize()).dt.days
    T["cum_points"] = T["points_pnl"].cumsum().round(1)
    win = round((T.points_pnl > 0).mean() * 100, 1); tot = round(T.points_pnl.sum(), 1)
    span = (daily.index[-1] - daily.index[0]).days
    for c in ("entry_time", "exit_time"): T[c] = pd.to_datetime(T[c])

    # ---- cumulative-return curve (start at 0 on first entry) ----
    x = [T.entry_time.iloc[0]] + list(T.exit_time); y = [0.0] + list(T.cum_points)
    peak = np.maximum.accumulate(y); maxdd = float((np.array(y) - peak).min())
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, y, color="#2E8B57", lw=1.6); ax.fill_between(x, y, 0, color="#2E8B57", alpha=.10); ax.axhline(0, color="#888", lw=.8)
    ax.scatter([x[-1]], [y[-1]], color="#2E8B57", zorder=5)
    ax.annotate(f"  +{y[-1]:,.0f} pts", (x[-1], y[-1]), color="#2E8B57", fontweight="bold", va="center")
    ax.set_title(f"NIFTY Daily 2-Day High/Low Breakout — PURE (no buffer, no gap)\nCumulative return, index points (gross), {x[0].date()} to {T.exit_time.iloc[-1].date()}", fontsize=12)
    ax.set_xlabel("Time"); ax.set_ylabel("Cumulative NIFTY points"); ax.grid(alpha=.25)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(0.015, 0.97, f"trades {len(T)} | final +{y[-1]:,.0f} pts | max drawdown {maxdd:,.0f} pts | win {win}%",
            transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=.9))
    fig.tight_layout(); fig.savefig(OUTDIR / "equity_curve.png", dpi=130); plt.close(fig)
    for c in ("entry_time", "exit_time"): T[c] = T[c].dt.strftime("%Y-%m-%d %H:%M")
    T.insert(0, "trade_no", range(1, len(T) + 1)); T.to_csv(OUTDIR / "breakout_2day_pure_trades.csv", index=False)

    print(f"PURE 2-day breakout (0 buffer, no gap) | {period}")
    print(f"trades {len(T)} | win {win}% | TOTAL {tot:,} pts | avg {round(T.points_pnl.mean(),2)} | avg hold {round(T.hold_cal_days.mean(),2)} d | max DD {round(maxdd,1)}")
    print(f"signals/wk {round(sig/(span/7),2)} | /mo {round(sig/(span/30.44),2)}")
    print(f"saved -> {OUTDIR/'equity_curve.png'} , {OUTDIR/'breakout_2day_pure_trades.csv'}")


if __name__ == "__main__":
    main()
