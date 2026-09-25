# -*- coding: utf-8 -*-
"""breakout_2day_gap.py — STANDALONE NIFTY rolling 2-day high/low breakout (10-pt margin, intraday-touch,
always-in flip) WITH special GAP-DAY handling. INDEX-point P&L (spot). SEPARATE module (merge with dual-TF
Supertrend later — NOT here).

NORMAL day: up=max(H[D-1],H[D-2])+10 ; down=min(L[D-1],L[D-2])-10 ; long flips short on touch of `down`,
short flips long on touch of `up`, at the exact trigger level.
GAP day (overrides normal): if at OPEN the day's open is beyond the RAW 2-day level OPPOSITE the current
position (long & open<rawP2L = gap-down ; short & open>rawP2H = gap-up), do NOT flip at open — hold the OLD
position through the first 15-min candle (09:15-09:29). After it closes, watch for a break of that first
15-min candle's LOW (long/gap-down) or HIGH (short/gap-up); on break, flip AT THAT EXACT LEVEL (no +10
margin). If never broken that day, hold (no flip) — flagged as its own category.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DATA = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "breakout_2day_gap"; OUTDIR.mkdir(parents=True, exist_ok=True)
MARGIN = 10.0; FIRST15_END = 570      # 09:30 (minute-of-day); first 15-min bar = 09:15-09:29 (mod 555..569)
COST_PER_TRADE = 15.0; LOT = 65       # FLAT futures cost = 15 index points per trade (round-trip); LOT=65 for INR illustration only


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    d["date"] = d["ts"].dt.normalize(); d["mod"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    daily = d.groupby("date").agg(dh=("high", "max"), dl=("low", "min"), do=("open", "first"))
    daily["p2h"] = daily["dh"].shift(1).rolling(2).max(); daily["p2l"] = daily["dl"].shift(1).rolling(2).min()
    daily["up"] = daily["p2h"] + MARGIN; daily["down"] = daily["p2l"] - MARGIN
    f15 = d[(d["mod"] >= 555) & (d["mod"] < FIRST15_END)].groupby("date").agg(fh15=("high", "max"), fl15=("low", "min"))
    daily = daily.join(f15)
    dm = d.merge(daily[["p2h", "p2l", "up", "down", "do", "fh15", "fl15"]], left_on="date", right_index=True, how="left")
    period = f"{daily.index[0].date()} .. {daily.index[-1].date()}"

    dt_a = dm["date"].values; mod_a = dm["mod"].values; hi_a = dm["high"].values; lo_a = dm["low"].values; op_a = dm["open"].values; tsr = dm["ts"].values
    up_a, dn_a, p2h_a, p2l_a, do_a, fh_a, fl_a = (dm[c].values for c in ["up", "down", "p2h", "p2l", "do", "fh15", "fl15"])

    pos = 0; entry = np.nan; entry_t = None; cur_type = None; trades = []; sig = 0
    cur_day = None; gap = None; gap_flipped = False; gap_held = []
    for k in range(len(dm)):
        if dt_a[k] != cur_day:                                    # ---- new day ----
            if gap in ("down", "up") and not gap_flipped:         # prior day: gap fired but never broke
                gap_held.append((pd.Timestamp(cur_day).date(), gap))
            cur_day = dt_a[k]; gap = None; gap_flipped = False
            if not np.isnan(up_a[k]):                             # gap detection at open (needs a position + valid window)
                if pos == 1 and do_a[k] < p2l_a[k]: gap = "down"
                elif pos == -1 and do_a[k] > p2h_a[k]: gap = "up"
        up, dn, h, l, o, t, m = up_a[k], dn_a[k], hi_a[k], lo_a[k], op_a[k], tsr[k], mod_a[k]
        if np.isnan(up):
            continue
        if gap in ("down", "up") and not gap_flipped:            # ---- GAP MODE ----
            if m < FIRST15_END:                                  # hold through first 15 min
                continue
            if gap == "down" and pos == 1 and l <= fl_a[k]:      # break first-15 LOW -> flip short at that low
                trades.append(("Long", entry_t, entry, t, fl_a[k], fl_a[k] - entry, cur_type)); pos, entry, entry_t, cur_type = -1, fl_a[k], t, "gap"; gap = None; gap_flipped = True; sig += 1
            elif gap == "up" and pos == -1 and h >= fh_a[k]:     # break first-15 HIGH -> flip long at that high
                trades.append(("Short", entry_t, entry, t, fh_a[k], entry - fh_a[k], cur_type)); pos, entry, entry_t, cur_type = 1, fh_a[k], t, "gap"; gap = None; gap_flipped = True; sig += 1
            continue                                              # while gap active: ignore normal triggers
        # ---- NORMAL MODE ----
        if pos == 0:
            hu, hd = h >= up, l <= dn
            if hu and hd:
                first = "up" if abs(o - up) <= abs(o - dn) else "down"
                pos, entry = (1, up) if first == "up" else (-1, dn); entry_t, cur_type = t, "normal"; sig += 1
            elif hu: pos, entry, entry_t, cur_type = 1, up, t, "normal"; sig += 1
            elif hd: pos, entry, entry_t, cur_type = -1, dn, t, "normal"; sig += 1
        elif pos == 1:
            if l <= dn:
                trades.append(("Long", entry_t, entry, t, dn, dn - entry, cur_type)); pos, entry, entry_t, cur_type = -1, dn, t, "normal"; sig += 1
        else:
            if h >= up:
                trades.append(("Short", entry_t, entry, t, up, entry - up, cur_type)); pos, entry, entry_t, cur_type = 1, up, t, "normal"; sig += 1
    if gap in ("down", "up") and not gap_flipped:
        gap_held.append((pd.Timestamp(cur_day).date(), gap))

    T = pd.DataFrame(trades, columns=["direction", "entry_time", "entry_price", "exit_time", "exit_price", "gross_points", "entry_type"])
    T["cost_points"] = COST_PER_TRADE                                              # FLAT 15 pts/trade round-trip futures cost
    T["points_pnl"] = (T["gross_points"] - T["cost_points"]).round(2)              # NET points (after cost); win% below is on NET
    T["hold_cal_days"] = (pd.to_datetime(T["exit_time"]).dt.normalize() - pd.to_datetime(T["entry_time"]).dt.normalize()).dt.days
    T["result"] = np.where(T["points_pnl"] > 0, "Win", np.where(T["points_pnl"] < 0, "Loss", "Flat"))
    for c in ("entry_time", "exit_time"): T[c] = pd.to_datetime(T[c]).dt.strftime("%Y-%m-%d %H:%M")
    T.insert(0, "trade_no", range(1, len(T) + 1)); T["cum_points"] = T["points_pnl"].cumsum().round(1)

    def blk(df, lbl):
        return {"type": lbl, "trades": len(df), "win_%_net": round((df.points_pnl > 0).mean() * 100, 1) if len(df) else 0,
                "gross_pts": round(df.gross_points.sum(), 1), "cost_pts": round(df.cost_points.sum(), 1),
                "NET_pts": round(df.points_pnl.sum(), 1), "avg_net_pts": round(df.points_pnl.mean(), 1) if len(df) else 0,
                "avg_hold_days": round(df.hold_cal_days.mean(), 2) if len(df) else 0}
    byt = pd.DataFrame([blk(T, "ALL"), blk(T[T.entry_type == "normal"], "NORMAL breakout"), blk(T[T.entry_type == "gap"], "GAP-day")])
    span = (daily.index[-1] - daily.index[0]).days

    with pd.ExcelWriter(OUTDIR / "breakout_2day_gap.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Strategy", "value": "STANDALONE NIFTY 2-day breakout + GAP-day handling (index pts, always-in flip)"},
            {"metric": "MERGE NOTE", "value": "Independent module — combine with dual-TF Supertrend LATER (not here)"},
            {"metric": "Period", "value": period}, {"metric": "Data", "value": "NIFTY 1-min INDEX; first-15-min bar = 09:15-09:29; gap check vs RAW 2-day level"},
            {"metric": "Total trades", "value": len(T)}, {"metric": "Win rate % (on NET, after cost)", "value": round((T.points_pnl > 0).mean() * 100, 1)},
            {"metric": "GROSS P&L (pts)", "value": round(T.gross_points.sum(), 1)},
            {"metric": "Futures cost FLAT 15 pts/trade (pts)", "value": round(T.cost_points.sum(), 1)},
            {"metric": "  -> cost model", "value": "FLAT 15 index points per trade (round-trip), independent of price/lot"},
            {"metric": "NET P&L (pts, after cost)", "value": round(T.points_pnl.sum(), 1)},
            {"metric": "Cost as % of gross", "value": f"{round(T.cost_points.sum()/T.gross_points.sum()*100,1)}%"},
            {"metric": "Avg NET pts/trade", "value": round(T.points_pnl.mean(), 1)},
            {"metric": "Illustrative NET INR @ lot 65", "value": f"Rs.{round(T.points_pnl.sum()*LOT):,} (ILLUSTRATIVE; lot only scales INR, not pts)"},
            {"metric": "Avg holding (cal days)", "value": round(T.hold_cal_days.mean(), 2)},
            {"metric": "Signals (incl 1st entry)", "value": sig}, {"metric": "Signals/month", "value": round(sig / (span / 30.44), 2)},
            {"metric": "", "value": ""},
            {"metric": "GAP-day trades", "value": int((T.entry_type == "gap").sum())},
            {"metric": "NORMAL breakout trades", "value": int((T.entry_type == "normal").sum())},
            {"metric": "Gap fired but first-15 NOT broken (held, no flip)", "value": len(gap_held)},
            {"metric": "  -> note", "value": "these days: gap detected at open, stayed in old position, first-15 low/high never broke -> no trade"},
        ]).to_excel(w, sheet_name="Summary", index=False)
        byt.to_excel(w, sheet_name="By_Type", index=False)
        T.to_excel(w, sheet_name="Trades", index=False)
        pd.DataFrame(gap_held, columns=["date", "gap_direction"]).to_excel(w, sheet_name="Gap_Held_No_Flip", index=False)
    T.to_csv(OUTDIR / "breakout_2day_gap_trades.csv", index=False)

    pd.set_option("display.width", 200)
    print("=" * 92 + "\nSTANDALONE NIFTY 2-DAY BREAKOUT + GAP-DAY HANDLING\n" + "=" * 92)
    print(f"period {period} | *** STANDALONE — merge with dual-TF Supertrend later ***")
    print(f"\ntrades {len(T)} | win(net) {round((T.points_pnl>0).mean()*100,1)}% | GROSS {round(T.gross_points.sum(),1):,} - COST {round(T.cost_points.sum(),1):,} ({round(T.cost_points.sum()/T.gross_points.sum()*100,1)}%) = NET {round(T.points_pnl.sum(),1):,} pts | avg net {round(T.points_pnl.mean(),1)} | hold {round(T.hold_cal_days.mean(),2)}d")
    print("\n--- BY TYPE (gap-handling impact) ---"); print(byt.to_string(index=False))
    print(f"\nGAP-fired-but-not-broken (held, no flip) days: {len(gap_held)}")
    if gap_held: print("  e.g.:", gap_held[:6])
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
