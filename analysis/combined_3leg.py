# -*- coding: utf-8 -*-
"""combined_3leg.py — COMBINE the 3 independent NIFTY legs into one net backtest (summation + reporting
layer ONLY; each leg's own logic is untouched):
  LEG 1 = daily 2-day breakout + gap handling  (from results/breakout_2day_gap trades)
  LEG 2 = 15-min Supertrend(10,3)  (close-confirmed flip, same-close)
  LEG 3 = 1-hour Supertrend(10,3)
Each leg always +1/-1. Net = L1+L2+L3 in {+3,+1,-1,-3} — ODD ONLY (never 0, unlike the 2-leg case).
Alignment: daily leg flips at its exact intraday touch time; all legs mapped to the 15-min grid for the
net-position-state time breakdown. Index-point P&L per leg + combined net. Reports per-leg stats, net-
state time %, and efficiency ratio per state (chop-vs-trend check).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from supertrend_dual_tf import supertrend, ATR_N, FACT

D15 = rb.BASE / "data" / "nifty_15min_ohlc.csv"
LEG1_CSV = rb.RESULTS / "breakout_2day_gap" / "breakout_2day_gap_trades.csv"
OUTDIR = rb.RESULTS / "combined_3leg"; OUTDIR.mkdir(parents=True, exist_ok=True)
COST_PER_TRADE = 15.0; LOT = 65        # FLAT futures cost = 15 index points per trade (round-trip); LOT=65 for INR only


def st_trades(direction, close, start):
    flips = [i for i in range(start + 1, len(direction)) if direction[i] != direction[i - 1]]
    gross = []
    for k, i in enumerate(flips):
        j = flips[k + 1] if k + 1 < len(flips) else len(close) - 1
        gross.append(direction[i] * (close[j] - close[i]))
    gross = np.array(gross); cost = np.full(len(gross), COST_PER_TRADE)            # flat 15 pts/trade
    return gross, cost, len(flips)


COLS = ["leg", "direction", "entry_time", "entry_price", "exit_time", "exit_price", "gross_points", "cost_points", "net_points", "result"]


def st_trade_df(direction, ts, close, start, leg):
    """full flip-to-flip trade list for a Supertrend leg (entered/exited at candle close)."""
    flips = [i for i in range(start + 1, len(direction)) if direction[i] != direction[i - 1]]
    rows = []
    for k, i in enumerate(flips):
        j = flips[k + 1] if k + 1 < len(flips) else len(close) - 1
        gross = direction[i] * (close[j] - close[i]); net = gross - COST_PER_TRADE
        rows.append({"leg": leg, "direction": "Long" if direction[i] == 1 else "Short",
                     "entry_time": pd.Timestamp(ts[i]).strftime("%Y-%m-%d %H:%M"), "entry_price": round(float(close[i]), 2),
                     "exit_time": pd.Timestamp(ts[j]).strftime("%Y-%m-%d %H:%M"), "exit_price": round(float(close[j]), 2),
                     "gross_points": round(float(gross), 2), "cost_points": COST_PER_TRADE, "net_points": round(float(net), 2),
                     "result": "Win" if net > 0 else ("Loss" if net < 0 else "Flat")})
    return pd.DataFrame(rows, columns=COLS)


def main():
    # ---------- LEG 1 (daily breakout + gap) from saved trades ----------
    l1 = pd.read_csv(LEG1_CSV)
    l1_ft = pd.to_datetime(l1["entry_time"]).values                 # each trade entry = a flip time
    l1_fd = l1["direction"].map({"Long": 1, "Short": -1}).values
    last_exit = pd.to_datetime(l1["exit_time"].iloc[-1]); last_dir = -l1_fd[-1]  # final open position after last close
    l1_ft = np.append(l1_ft, np.datetime64(last_exit)); l1_fd = np.append(l1_fd, last_dir)
    g1 = round(l1["gross_points"].sum(), 1); c1 = round(l1["cost_points"].sum(), 1)          # leg1 gross/cost/net (from its own trades)
    pnl1 = round(l1["points_pnl"].sum(), 1); nt1 = len(l1); wr1 = round((l1["points_pnl"] > 0).mean() * 100, 1)

    # ---------- LEG 2/3 (Supertrend on 15-min / 1-hour) ----------
    d = pd.read_csv(D15); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True); N = len(d); c15 = d["close"].values
    gt = d["ts"].values                                             # 15-min grid timestamps (tz-naive)
    dir15 = supertrend(d["high"].values, d["low"].values, c15, ATR_N, FACT)
    mod = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    d["hb"] = pd.factorize(d["ts"].dt.strftime("%Y-%m-%d") + "_" + ((mod - 555) // 60).astype(int).astype(str))[0]
    hourly = d.groupby("hb").agg(h=("high", "max"), l=("low", "min"), c=("close", "last")).sort_index()
    hts = d.groupby("hb")["ts"].last().sort_index().values          # each hour bucket's close timestamp
    dir1h = supertrend(hourly["h"].values, hourly["l"].values, hourly["c"].values, ATR_N, FACT)
    is_last = d["hb"].values != np.append(d["hb"].values[1:], -1)
    dir1h_pos = np.full(N, np.nan); dir1h_pos[is_last] = dir1h[d["hb"].values[is_last]]; dir1h_pos = pd.Series(dir1h_pos).ffill().values
    dp = np.nan_to_num(dir1h_pos).astype(int)
    g2a, c2a, n2 = st_trades(dir15, c15, ATR_N + 1); net2a = g2a - c2a
    g2, c2, pnl2, wr2 = round(g2a.sum(), 1), round(c2a.sum(), 1), round(net2a.sum(), 1), round((net2a > 0).mean() * 100, 1)
    g3a, c3a, n3 = st_trades(dir1h, hourly["c"].values, ATR_N + 1); net3a = g3a - c3a
    g3, c3, pnl3, wr3 = round(g3a.sum(), 1), round(c3a.sum(), 1), round(net3a.sum(), 1), round((net3a > 0).mean() * 100, 1)

    # ---------- align LEG 1 onto the 15-min grid ----------
    ii = np.searchsorted(l1_ft, gt, side="right") - 1
    leg1 = np.where((ii >= 0) & (gt >= l1_ft[0]), l1_fd[ii.clip(0)], 0)   # 0 = flat (before leg1's first breakout)
    leg2, leg3 = dir15, dp
    start = max(ATR_N + 1, int(np.argmax(~np.isnan(dir1h_pos))), int(np.searchsorted(gt, l1_ft[0], "right")))
    reg = np.arange(start, N)
    reg = reg[leg1[reg] != 0]                                        # all-3-in-position region only
    net = leg1 + leg2 + leg3

    # efficiency ratio (Kaufman, 8x15m=2h) for chop-vs-trend per state
    win = 8; er = np.full(N, np.nan)
    for i in range(start, N):
        seg = c15[max(0, i - win):i + 1]; den = np.abs(np.diff(seg)).sum(); er[i] = abs(seg[-1] - seg[0]) / den if den > 0 else 0.0
    states = []
    for s, lbl, kind in [(3, "net +3 (all long)", "aligned"), (1, "net +1 (2 long,1 short)", "divergent"),
                         (-1, "net -1 (2 short,1 long)", "divergent"), (-3, "net -3 (all short)", "aligned")]:
        m = np.zeros(N, bool); m[reg] = net[reg] == s
        states.append({"net_state": lbl, "type": kind, "pct_time": round(m[reg].mean() * 100, 1), "candles": int(m[reg].sum()),
                       "mean_efficiency_ratio": round(np.nanmean(er[m]), 3) if m.any() else 0})
    STATE = pd.DataFrame(states)
    er_aligned = round(np.nanmean(er[reg][np.isin(net[reg], [3, -3])]), 3)
    er_diverg = round(np.nanmean(er[reg][np.isin(net[reg], [1, -1])]), 3)

    gT, cT, nT = round(g1 + g2 + g3, 1), round(c1 + c2 + c3, 1), round(pnl1 + pnl2 + pnl3, 1)
    legs = pd.DataFrame([
        {"leg": "LEG1 daily breakout+gap", "gross_pts": g1, "cost_pts": c1, "NET_pts": pnl1, "trades": nt1, "win_%_net": wr1},
        {"leg": "LEG2 15-min Supertrend", "gross_pts": g2, "cost_pts": c2, "NET_pts": pnl2, "trades": n2, "win_%_net": wr2},
        {"leg": "LEG3 1-hour Supertrend", "gross_pts": g3, "cost_pts": c3, "NET_pts": pnl3, "trades": n3, "win_%_net": wr3},
        {"leg": "COMBINED (sum of legs)", "gross_pts": gT, "cost_pts": cT, "NET_pts": nT, "trades": nt1 + n2 + n3, "win_%_net": "-"}])

    # ---------- per-leg trade lists + combined all-trades ----------
    L1 = l1.assign(leg="LEG1 daily breakout+gap", net_points=l1["points_pnl"])[COLS]
    L2 = st_trade_df(dir15, gt, c15, ATR_N + 1, "LEG2 15-min ST")
    L3 = st_trade_df(dir1h, hts, hourly["c"].values, ATR_N + 1, "LEG3 1-hour ST")
    ALL = pd.concat([L1, L2, L3], ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    ALL.insert(0, "seq", range(1, len(ALL) + 1)); ALL["cum_net_combined"] = ALL["net_points"].cumsum().round(1)
    for df in (L1, L2, L3): df.insert(0, "trade_no", range(1, len(df) + 1))

    # ---------- SUMMARY (single sheet: leg performance + combined + net-state + notes folded in) ----------
    summ = [
        {"metric": "Strategy", "value": "3-LEG COMBINED NIFTY (daily 2-day breakout+gap | 15-min Supertrend | 1-hour Supertrend) — index points"},
        {"metric": "Period", "value": f"{d['ts'].iloc[0].date()} .. {d['ts'].iloc[-1].date()}"},
        {"metric": "Structure", "value": "3 INDEPENDENT legs, each always +1/-1; net = sum in {+3,+1,-1,-3} — ODD ONLY, NEVER 0 (2-leg case could net 0)"},
        {"metric": "Daily-leg timing", "value": "flips at its EXACT intraday touch time; mapped to 15-min grid for state timing"},
        {"metric": "Futures cost model", "value": "FLAT 15 index points per trade (round-trip); independent of price/lot"},
        {"metric": "", "value": ""},
        {"metric": "--- PER-LEG (gross / cost / NET pts | trades | win% net) ---", "value": ""},
        {"metric": "LEG1 daily breakout+gap", "value": f"{g1} / {c1} / {pnl1}  |  {nt1} trades  |  {wr1}%"},
        {"metric": "LEG2 15-min Supertrend", "value": f"{g2} / {c2} / {pnl2}  |  {n2} trades  |  {wr2}%"},
        {"metric": "LEG3 1-hour Supertrend", "value": f"{g3} / {c3} / {pnl3}  |  {n3} trades  |  {wr3}%"},
        {"metric": "", "value": ""},
        {"metric": "COMBINED GROSS points", "value": gT},
        {"metric": "COMBINED COST points (15/trade)", "value": cT},
        {"metric": "COMBINED NET points", "value": nT},
        {"metric": "COMBINED cost as % of gross", "value": f"{round(cT/gT*100,1)}%"},
        {"metric": "COMBINED total trades", "value": nt1 + n2 + n3},
        {"metric": "Illustrative COMBINED NET INR @ lot 65", "value": f"Rs.{round(nT*LOT):,} (ILLUSTRATIVE; lot only scales INR, not pts)"},
        {"metric": "", "value": ""},
        {"metric": "--- NET-POSITION-STATE TIME (odd only; never 0) ---", "value": "pct_time | mean efficiency-ratio"},
        *[{"metric": r["net_state"] + f" [{r['type']}]", "value": f"{r['pct_time']}% | ER {r['mean_efficiency_ratio']}"} for r in states],
        {"metric": "chop check", "value": f"aligned(+3/-3) ER {er_aligned} vs divergent(+1/-1) ER {er_diverg} -> divergence {'IS' if er_diverg<er_aligned else 'is NOT'} choppier"},
        {"metric": "", "value": ""},
        {"metric": "P&L note", "value": "combined = sum of each leg's own-convention P&L (daily=trigger levels, ST=candle close)."},
        {"metric": "Sheets", "value": "Summary | All_Trades (3 legs merged, time-sorted) | Leg1_Trades | Leg2_Trades | Leg3_Trades"},
    ]
    with pd.ExcelWriter(OUTDIR / "combined_3leg.xlsx", engine="openpyxl") as w:
        pd.DataFrame(summ).to_excel(w, sheet_name="Summary", index=False)
        ALL.to_excel(w, sheet_name="All_Trades", index=False)
        L1.to_excel(w, sheet_name="Leg1_Trades", index=False)
        L2.to_excel(w, sheet_name="Leg2_Trades", index=False)
        L3.to_excel(w, sheet_name="Leg3_Trades", index=False)

    pd.set_option("display.width", 200)
    print("=" * 92 + "\n3-LEG COMBINED NIFTY BACKTEST (daily breakout+gap | 15m ST | 1h ST) — index points\n" + "=" * 92)
    print(f"period {d['ts'].iloc[0].date()} .. {d['ts'].iloc[-1].date()}")
    print("\n--- PER-LEG & COMBINED ---"); print(legs.to_string(index=False))
    print("\n--- NET-POSITION-STATE (odd only: +3/+1/-1/-3; never 0) ---"); print(STATE.to_string(index=False))
    print(f"\nchop check: aligned(+3/-3) eff-ratio {er_aligned} vs divergent(+1/-1) {er_diverg} -> "
          f"divergence {'IS choppier (hypothesis holds)' if er_diverg<er_aligned else 'NOT choppier'}")
    print(f"\nSTRUCTURAL NOTE: 3 legs each +-1 -> net in {{+3,+1,-1,-3}}, NEVER 0 (differs from 2-leg case which could net 0)")
    print(f"Saved -> {OUTDIR}")


if __name__ == "__main__":
    main()
