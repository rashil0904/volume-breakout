# -*- coding: utf-8 -*-
"""breakout_2day_gap0.py — CANONICAL daily 2-day high/low breakout: 0-point buffer (exact touch) + 15-min
GAP HANDLING. Always-in flip. GROSS index points. Realistic fills: gap trades fill at the first-15-min
low/high; normal flips fill at the level, or at the candle OPEN if price gapped through it (fill-realism
guard). Outputs equity curve + Excel (Summary w/ drawdown + MFE/MAE, By_Type, Drawdown_Episodes, Trades)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.dates as mdates
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DATA = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "breakout_2day_gap0"; OUTDIR.mkdir(parents=True, exist_ok=True)
MARGIN = 0.0; FIRST15_END = 570


def drawdown_episodes(equity, times):
    peak = np.maximum.accumulate(equity); dd = equity - peak; eps = []; in_dd = False; s = t = 0; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9: in_dd = True; s = k - 1; t = k; tv = dd[k]
        else:
            if dd[k] < tv: tv = dd[k]; t = k
            if dd[k] >= -1e-9: eps.append((s, t, k, tv)); in_dd = False
    if in_dd: eps.append((s, t, len(equity) - 1, tv))
    rows = []
    for (s, t, r, v) in eps:
        rec = (r != len(equity) - 1) or (equity[r] >= peak[s] - 1e-9)
        rows.append({"peak_date": pd.Timestamp(times[s]).date(), "trough_date": pd.Timestamp(times[t]).date(),
                     "recovery_date": (pd.Timestamp(times[r]).date() if rec else "NOT RECOVERED"), "drawdown_points": round(abs(v), 1),
                     "days_peak_to_trough": int((pd.Timestamp(times[t]) - pd.Timestamp(times[s])).days),
                     "days_peak_to_recovery": int((pd.Timestamp(times[r]) - pd.Timestamp(times[s])).days)})
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    d["date"] = d["ts"].dt.normalize(); d["mod"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    daily = d.groupby("date").agg(dh=("high", "max"), dl=("low", "min"))
    daily["p2h"] = daily["dh"].shift(1).rolling(2).max(); daily["p2l"] = daily["dl"].shift(1).rolling(2).min()
    daily["up"] = daily["p2h"] + MARGIN; daily["down"] = daily["p2l"] - MARGIN
    f15 = d[(d["mod"] >= 555) & (d["mod"] < FIRST15_END)].groupby("date").agg(fh15=("high", "max"), fl15=("low", "min"))
    dopen = d.groupby("date")["open"].first().rename("dopen")
    daily = daily.join(f15).join(dopen)
    dm = d.merge(daily[["p2h", "p2l", "up", "down", "fh15", "fl15", "dopen"]], left_on="date", right_index=True, how="left")
    period = f"{daily.index[0].date()} .. {daily.index[-1].date()}"
    dt_a = dm["date"].values; mod_a = dm["mod"].values; hi_a = dm["high"].values; lo_a = dm["low"].values; op_a = dm["open"].values; tsr = dm["ts"].values
    up_a, dn_a, p2h_a, p2l_a, do_a, fh_a, fl_a = (dm[c].values for c in ["up", "down", "p2h", "p2l", "dopen", "fh15", "fl15"])

    pos = 0; entry = np.nan; entry_t = None; cur_type = None; trades = []; sig = 0
    cur_day = None; gap = None; gap_flipped = False; gap_held = []
    for k in range(len(dm)):
        if dt_a[k] != cur_day:
            if gap in ("down", "up") and not gap_flipped: gap_held.append((pd.Timestamp(cur_day).date(), gap))
            cur_day = dt_a[k]; gap = None; gap_flipped = False
            if not np.isnan(up_a[k]):
                if pos == 1 and do_a[k] < p2l_a[k]: gap = "down"
                elif pos == -1 and do_a[k] > p2h_a[k]: gap = "up"
        up, dn, h, l, o, t, m = up_a[k], dn_a[k], hi_a[k], lo_a[k], op_a[k], tsr[k], mod_a[k]
        if np.isnan(up):
            continue
        fill_up = up if l <= up else o; fill_dn = dn if h >= dn else o          # fill-realism guard (gap-open if gapped through)
        if gap in ("down", "up") and not gap_flipped:                          # ---- GAP MODE: fill at first-15-min level ----
            if m < FIRST15_END:
                continue
            if gap == "down" and pos == 1 and l <= fl_a[k]:
                trades.append(("Long", entry_t, entry, t, fl_a[k], fl_a[k] - entry, cur_type)); pos, entry, entry_t, cur_type = -1, fl_a[k], t, "gap"; gap = None; gap_flipped = True; sig += 1
            elif gap == "up" and pos == -1 and h >= fh_a[k]:
                trades.append(("Short", entry_t, entry, t, fh_a[k], entry - fh_a[k], cur_type)); pos, entry, entry_t, cur_type = 1, fh_a[k], t, "gap"; gap = None; gap_flipped = True; sig += 1
            continue
        if pos == 0:
            hu, hd = h >= up, l <= dn
            if hu and hd:
                pos, entry = (1, fill_up) if abs(o - up) <= abs(o - dn) else (-1, fill_dn); entry_t, cur_type = t, "normal"; sig += 1
            elif hu: pos, entry, entry_t, cur_type = 1, fill_up, t, "normal"; sig += 1
            elif hd: pos, entry, entry_t, cur_type = -1, fill_dn, t, "normal"; sig += 1
        elif pos == 1:
            if l <= dn:
                trades.append(("Long", entry_t, entry, t, fill_dn, fill_dn - entry, cur_type)); pos, entry, entry_t, cur_type = -1, fill_dn, t, "normal"; sig += 1
        else:
            if h >= up:
                trades.append(("Short", entry_t, entry, t, fill_up, entry - fill_up, cur_type)); pos, entry, entry_t, cur_type = 1, fill_up, t, "normal"; sig += 1
    if gap in ("down", "up") and not gap_flipped: gap_held.append((pd.Timestamp(cur_day).date(), gap))

    T = pd.DataFrame(trades, columns=["direction", "entry_time", "entry_price", "exit_time", "exit_price", "points_pnl", "entry_type"])
    T["hold_cal_days"] = (pd.to_datetime(T["exit_time"]).dt.normalize() - pd.to_datetime(T["entry_time"]).dt.normalize()).dt.days
    # MFE/MAE from 1-min path
    tsv = dm["ts"].values; hv = dm["high"].values; lv = dm["low"].values; mfe = []; mae = []
    for r in T.itertuples():
        i0 = np.searchsorted(tsv, np.datetime64(pd.Timestamp(r.entry_time)), "left"); i1 = np.searchsorted(tsv, np.datetime64(pd.Timestamp(r.exit_time)), "right")
        if i1 <= i0: mfe.append(0.0); mae.append(0.0); continue
        hi = hv[i0:i1].max(); lo = lv[i0:i1].min()
        if r.direction == "Long": mfe.append(hi - r.entry_price); mae.append(lo - r.entry_price)
        else: mfe.append(r.entry_price - lo); mae.append(r.entry_price - hi)
    T["mfe_points"] = np.round(mfe, 1); T["mae_points"] = np.round(mae, 1)
    T["cum_points"] = T["points_pnl"].cumsum().round(1)
    entdt = pd.to_datetime(T["entry_time"]); extdt = pd.to_datetime(T["exit_time"])
    equity = np.concatenate([[0.0], T["cum_points"].values]); times = np.concatenate([[entdt.iloc[0]], extdt.values])
    DE = drawdown_episodes(equity, times)
    win = round((T.points_pnl > 0).mean() * 100, 1); tot = round(T.points_pnl.sum(), 1); span = (daily.index[-1] - daily.index[0]).days
    maxdd = DE.drawdown_points.max() if len(DE) else 0

    def blk(df, lbl):
        return {"type": lbl, "trades": len(df), "win_%": round((df.points_pnl > 0).mean() * 100, 1) if len(df) else 0,
                "total_pts": round(df.points_pnl.sum(), 1), "avg_pts": round(df.points_pnl.mean(), 2) if len(df) else 0}
    byt = pd.DataFrame([blk(T, "ALL"), blk(T[T.entry_type == "normal"], "NORMAL"), blk(T[T.entry_type == "gap"], "GAP")])

    # equity curve
    x = [entdt.iloc[0]] + list(extdt); y = [0.0] + list(T.cum_points)
    fig, ax = plt.subplots(figsize=(12, 5.5)); ax.plot(x, y, color="#6A3D9A", lw=1.6); ax.fill_between(x, y, 0, color="#6A3D9A", alpha=.10); ax.axhline(0, color="#888", lw=.8)
    ax.scatter([x[-1]], [y[-1]], color="#6A3D9A", zorder=5); ax.annotate(f"  +{y[-1]:,.0f} pts", (x[-1], y[-1]), color="#6A3D9A", fontweight="bold", va="center")
    ax.set_title(f"NIFTY Daily 2-Day Breakout — 0 buffer + GAP HANDLING (realistic fills)\nCumulative return, index points (gross), {x[0].date()} to {extdt.iloc[-1].date()}", fontsize=12)
    ax.set_xlabel("Time"); ax.set_ylabel("Cumulative NIFTY points"); ax.grid(alpha=.25); ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(0.015, 0.97, f"trades {len(T)} | final +{y[-1]:,.0f} pts | max DD {maxdd:,.0f} pts | win {win}%", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=.9))
    fig.tight_layout(); fig.savefig(OUTDIR / "equity_curve.png", dpi=130); plt.close(fig)

    Tout = T.copy()
    for c in ("entry_time", "exit_time"): Tout[c] = pd.to_datetime(Tout[c]).dt.strftime("%Y-%m-%d %H:%M")
    Tout.insert(0, "trade_no", range(1, len(Tout) + 1))
    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY daily 2-day breakout — 0 buffer + 15-min GAP HANDLING (always-in flip, realistic fills)"},
        {"metric": "Period", "value": period}, {"metric": "P&L basis", "value": "GROSS index points"},
        {"metric": "Total trades", "value": len(T)}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot}, {"metric": "Avg points / trade", "value": round(T.points_pnl.mean(), 2)},
        {"metric": "Best / worst REALIZED trade", "value": f"{round(T.points_pnl.max(),1)} / {round(T.points_pnl.min(),1)}"},
        {"metric": "Max PROFIT reached in a trade (MFE)", "value": round(T.mfe_points.max(), 1)},
        {"metric": "Max LOSS reached in a trade (MAE)", "value": round(T.mae_points.min(), 1)},
        {"metric": "Avg MFE / MAE", "value": f"{round(T.mfe_points.mean(),1)} / {round(T.mae_points.mean(),1)}"},
        {"metric": "Avg holding (cal days)", "value": round(T.hold_cal_days.mean(), 2)},
        {"metric": "Signals / month", "value": round(len(T) / (span / 30.44), 2)},
        {"metric": "GAP / NORMAL trades", "value": f"{int((T.entry_type=='gap').sum())} / {int((T.entry_type=='normal').sum())}"},
        {"metric": "Gap fired but 15-min not broken (held)", "value": len(gap_held)},
        {"metric": "", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)},
        {"metric": "Max drawdown (points)", "value": round(maxdd, 1)},
        {"metric": "Avg drawdown (points)", "value": round(DE.drawdown_points.mean(), 1) if len(DE) else 0},
        {"metric": "Max DD duration (days)", "value": int(DE.days_peak_to_recovery.max()) if len(DE) else 0},
        {"metric": "Return / Max-DD", "value": round(tot / maxdd, 2) if maxdd else "-"},
    ])
    out = OUTDIR / "breakout_2day_gap0.xlsx"
    try: wr = pd.ExcelWriter(out, engine="openpyxl")
    except PermissionError: out = OUTDIR / "breakout_2day_gap0_new.xlsx"; wr = pd.ExcelWriter(out, engine="openpyxl")
    with wr as w:
        summary.to_excel(w, sheet_name="Summary", index=False); byt.to_excel(w, sheet_name="By_Type", index=False)
        DE.sort_values("drawdown_points", ascending=False).to_excel(w, sheet_name="Drawdown_Episodes", index=False)
        Tout.to_excel(w, sheet_name="Trades", index=False)
    Tout.to_csv(OUTDIR / "breakout_2day_gap0_trades.csv", index=False)

    print(f"0-BUFFER + GAP HANDLING (realistic fills) | {period}")
    print(f"trades {len(T)} | win {win}% | TOTAL {tot:,} pts | avg {round(T.points_pnl.mean(),2)} | avg hold {round(T.hold_cal_days.mean(),2)} d")
    print(f"drawdown: {len(DE)} eps | max {round(maxdd,1)} pts | ret/maxDD {round(tot/maxdd,2) if maxdd else '-'}")
    print("--- BY TYPE ---"); print(byt.to_string(index=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
