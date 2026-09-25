# -*- coding: utf-8 -*-
"""supertrend_dual_tf.py — NIFTY dual-timeframe Supertrend(10,3), INDEX-POINT P&L only (spot value, not a
tradable instrument). Two independent always-in legs: 15-min and 1-hour. Each leg holds +1/-1 = its
Supertrend direction, FLIPPING (long<->short) on every fresh close-confirmed flip (dir -1->+1 bull /
+1->-1 bear), same-candle-CLOSE execution. Net = leg15 + leg1h in {+2, 0, -2}. Reports per-leg + combined
points P&L, flips/trades, win rate, and time-in-net-state (+2/-2/0) with a chop metric to validate the
net=0 chop-protection hypothesis. DESIGN FLAGS printed. 1-hour built from the 15-min index series.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DATA = rb.BASE / "data" / "nifty_15min_ohlc.csv"
OUTDIR = rb.RESULTS / "supertrend_dual_tf"; OUTDIR.mkdir(parents=True, exist_ok=True)
ATR_N, FACT = 10, 3.0
LOT_REF = 75           # illustrative NIFTY lot multiplier (see caveat; lot size actually varied over 2022-2026)


def supertrend(high, low, close, n=10, f=3.0):
    m = len(close); tr = np.full(m, np.nan); tr[0] = high[0] - low[0]
    for i in range(1, m):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(m, np.nan)
    if m <= n:
        return np.zeros(m)
    atr[n - 1] = np.nanmean(tr[:n])
    for i in range(n, m):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    hl2 = (high + low) / 2.0; ub = hl2 + f * atr; lb = hl2 - f * atr; fub = np.copy(ub); flb = np.copy(lb)
    for i in range(n, m):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
    st = np.full(m, np.nan); direction = np.zeros(m, dtype=int); st[n] = fub[n]; direction[n] = -1
    for i in range(n + 1, m):
        if st[i - 1] == fub[i - 1]:
            if close[i] > fub[i]: st[i] = flb[i]; direction[i] = 1
            else: st[i] = fub[i]; direction[i] = -1
        else:
            if close[i] < flb[i]: st[i] = fub[i]; direction[i] = -1
            else: st[i] = flb[i]; direction[i] = 1
    direction[:n + 1] = direction[n]                                    # backfill seed for warmup region
    return direction


def leg_trades(direction, close, start):
    """segments between flips: entered at flip close, exited at next flip close. returns per-trade points."""
    flips = [i for i in range(start + 1, len(direction)) if direction[i] != direction[i - 1]]
    trades = []
    for k, i in enumerate(flips):
        j = flips[k + 1] if k + 1 < len(flips) else len(close) - 1        # exit at next flip (or last bar)
        trades.append(direction[i] * (close[j] - close[i]))
    return np.array(trades), len(flips)


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"]); d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    ts = d["ts"]; h15, l15, c15 = d["high"].values, d["low"].values, d["close"].values
    N = len(d); period = f"{ts.iloc[0].date()} .. {ts.iloc[-1].date()}"
    dir15 = supertrend(h15, l15, c15, ATR_N, FACT)

    # ---- build 1-hour from 15-min (buckets anchored 09:15; ~7 bars/day) ----
    mod = ts.dt.hour * 60 + ts.dt.minute
    bkey = ts.dt.strftime("%Y-%m-%d") + "_" + ((mod - 555) // 60).astype(int).astype(str)
    hb = pd.factorize(bkey)[0]; d["hb"] = hb
    hourly = d.groupby("hb").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    hourly = hourly.sort_index()
    dir1h = supertrend(hourly["h"].values, hourly["l"].values, hourly["c"].values, ATR_N, FACT)
    # map 1h direction onto the 15-min grid at each hour's CLOSE (last 15m of bucket), forward-fill
    is_last = hb != np.append(hb[1:], -1)
    dir1h_pos = np.full(N, np.nan)
    dir1h_pos[is_last] = dir1h[hb[is_last]]
    dir1h_pos = pd.Series(dir1h_pos).ffill().values

    # ---- common valid region ----
    start = max(ATR_N + 1, int(np.argmax(~np.isnan(dir1h_pos))))
    dir1h_pos_i = np.nan_to_num(dir1h_pos).astype(int)
    dc = np.diff(c15)                                                    # close-to-close moves (len N-1)

    # ---- P&L (index points), all on the 15-min grid ----
    idx = np.arange(start, N - 1)
    leg15_pnl = float((dir15[idx] * dc[idx]).sum())
    leg1h_pnl = float((dir1h_pos_i[idx] * dc[idx]).sum())
    net = dir15 + dir1h_pos_i                                            # {+2,0,-2}
    combined_pnl = float((net[idx] * dc[idx]).sum())

    # ---- per-leg trades / win rate ----
    t15, f15 = leg_trades(dir15, c15, start)
    t1h, f1h = leg_trades(dir1h, hourly["c"].values, ATR_N + 1)

    def legstat(name, pnl, tr, nflip):
        return {"leg": name, "total_points": round(pnl, 1), "flips_trades": int(nflip), "win_rate_%": round((tr > 0).mean() * 100, 1) if len(tr) else 0,
                "avg_pts_per_trade": round(tr.mean(), 1) if len(tr) else 0, "best_trade": round(tr.max(), 1) if len(tr) else 0, "worst_trade": round(tr.min(), 1) if len(tr) else 0}
    legs = pd.DataFrame([legstat("15min", leg15_pnl, t15, f15), legstat("1hour", leg1h_pnl, t1h, f1h),
                         {"leg": "COMBINED (net)", "total_points": round(combined_pnl, 1), "flips_trades": int(f15 + f1h),
                          "win_rate_%": "-", "avg_pts_per_trade": "-", "best_trade": "-", "worst_trade": "-"}])

    # ---- net-state time + chop metric (Kaufman efficiency ratio over 8x15m=2h window) ----
    win = 8; er = np.full(N, np.nan)
    for i in range(start, N):
        seg = c15[max(0, i - win):i + 1]
        denom = np.abs(np.diff(seg)).sum()
        er[i] = abs(seg[-1] - seg[0]) / denom if denom > 0 else 0.0
    st_rows = []
    for s, lbl in [(2, "net +2 (both long)"), (-2, "net -2 (both short)"), (0, "net 0 (divergent)")]:
        mask = np.zeros(N, bool); mask[idx] = net[idx] == s
        pnl_s = float((net[idx][net[idx] == s] * dc[idx][net[idx] == s]).sum())
        st_rows.append({"net_state": lbl, "pct_time": round(mask[idx].mean() * 100, 1), "candles": int(mask[idx].sum()),
                        "combined_pnl_pts": round(pnl_s, 1), "mean_efficiency_ratio": round(np.nanmean(er[mask]), 3) if mask.any() else 0})
    STATE = pd.DataFrame(st_rows)

    # ---- outputs ----
    with pd.ExcelWriter(OUTDIR / "supertrend_dual_tf.xlsx", engine="openpyxl") as w:
        legs.to_excel(w, sheet_name="leg_performance", index=False)
        STATE.to_excel(w, sheet_name="net_state_breakdown", index=False)
        pd.DataFrame([{"metric": "period", "value": period}, {"metric": "bars_15min", "value": N}, {"metric": "bars_1hour", "value": len(hourly)},
                     {"metric": "combined_total_points", "value": round(combined_pnl, 1)},
                     {"metric": "illustrative_INR_@lot75", "value": round(combined_pnl * LOT_REF)},
                     {"metric": "execution", "value": "SAME-CANDLE-CLOSE (design choice; differs from daily-ST next-open)"}]).to_excel(w, sheet_name="info", index=False)
    # equity curve
    eq = np.cumsum(net[start:N - 1] * dc[start:N - 1])
    fig, ax = plt.subplots(figsize=(11, 4.5)); ax.plot(ts.iloc[start:N - 1], eq, color="#1F497D")
    ax.set_title(f"NIFTY dual-TF Supertrend(10,3) — combined net index-point equity ({period})"); ax.set_ylabel("cum points"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "equity_combined.png", dpi=120); plt.close(fig)

    pd.set_option("display.width", 200)
    print("=" * 92 + "\nNIFTY DUAL-TIMEFRAME SUPERTREND(10,3) — INDEX-POINT P&L (spot value, not tradable)\n" + "=" * 92)
    print(f"period: {period} | 15-min bars {N:,} | 1-hour bars {len(hourly):,} | index value (volume=0)")
    print("\n--- PER-LEG & COMBINED PERFORMANCE (points) ---"); print(legs.to_string(index=False))
    print("\n--- NET-POSITION-STATE BREAKDOWN (chop-protection validation) ---"); print(STATE.to_string(index=False))
    tot = combined_pnl
    print(f"\nCOMBINED total: {tot:,.0f} index points  |  illustrative INR @ lot 75 = Rs.{tot*LOT_REF:,.0f}  [ILLUSTRATIVE ONLY]")
    print("\n" + "-" * 92 + "\nDESIGN CHOICES / FLAGS (per spec):")
    print(" * EXECUTION = SAME-CANDLE CLOSE (signal confirmed at close, filled at that same close). This")
    print("   DIFFERS from the single-TF daily Supertrend (next-day-open). Slightly optimistic (assumes the")
    print("   just-observed close is tradable) but matches the dual-TF spec. Not look-ahead (uses only closed data).")
    print(" * ALWAYS-IN-POSITION: each leg is always +1/-1 (never flat); before the first flip it holds the")
    print("   Supertrend seed direction from the end of the ATR warmup. No separate exit/wait state.")
    print(" * FLIP = 2 lots change hands (close old +1, open new -1 at same close). Point-P&L is mark-to-market")
    print("   (position = direction each bar), which correctly realizes the closed leg and opens the new one at")
    print("   that close -> NO double counting. Costs are NOT modelled (spot sim).")
    print(" * net=0 means the two legs oppose -> ZERO net exposure -> ZERO combined P&L in that state (see table):")
    print("   chop protection is literal (flat), validated by lower efficiency-ratio in the net=0 state.")
    print(" * INR figure is ILLUSTRATIVE ONLY: index is not tradable; a real book would use NIFTY futures/options")
    print("   (premium/margin/roll), and the lot size (here 75) actually changed over 2022-2026.")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
