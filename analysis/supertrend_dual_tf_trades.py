# -*- coding: utf-8 -*-
"""supertrend_dual_tf_trades.py — NIFTY dual-TF Supertrend(10,3) TRADE LIST to Excel with FUTURES COST.
Sheets: Summary + Combined_Trades + 15min_Trades + 1hour_Trades. Each trade = one flip-to-flip segment
(entered at flip close, exited at next flip close). Cost = 0.03% round-trip on notional = 0.015% x price
per side (entry + exit), charged in index points (cost_side = 0.00015 x price). Reports gross, cost, net
points, win rate (on NET), and gross + NET drawdown (max/avg pts + duration). Index-point P&L, spot sim.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from supertrend_dual_tf import supertrend, ATR_N, FACT, LOT_REF, DATA

OUT = rb.RESULTS / "supertrend_dual_tf" / "supertrend_dual_tf_trades_withcost.xlsx"
SIDE = 0.00015                     # 0.015% per side (0.03% round-trip); cost_points = SIDE * price


def extract(direction, ts, close, start, tf):
    flips = [i for i in range(start + 1, len(direction)) if direction[i] != direction[i - 1]]
    rows = []; cumn = 0.0
    for k, i in enumerate(flips):
        openend = k + 1 >= len(flips); j = flips[k + 1] if not openend else len(close) - 1
        pts = direction[i] * (close[j] - close[i]); cost = SIDE * (close[i] + close[j]); netp = pts - cost; cumn += netp
        rows.append({"timeframe": tf, "direction": "Long" if direction[i] == 1 else "Short",
                     "entry_time": pd.Timestamp(ts[i]).strftime("%Y-%m-%d %H:%M"), "entry_price": round(close[i], 2),
                     "exit_time": pd.Timestamp(ts[j]).strftime("%Y-%m-%d %H:%M"), "exit_price": round(close[j], 2),
                     "bars_held": int(j - i), "gross_points": round(pts, 2), "cost_points": round(cost, 2), "net_points": round(netp, 2),
                     "result": "Win" if netp > 0 else ("Loss" if netp < 0 else "Flat"), "open_at_end": openend, "cum_net_leg": round(cumn, 1)})
    df = pd.DataFrame(rows); df.insert(0, "trade_no", range(1, len(df) + 1)); return df


def dd_stats(equity, times):
    peak = np.maximum.accumulate(equity); dd = equity - peak; eps = []; in_dd = False; s = t = 0; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9: in_dd = True; s = max(k - 1, 0); t = k; tv = dd[k]
        else:
            if dd[k] < tv: tv = dd[k]; t = k
            if dd[k] >= -1e-9: eps.append((s, t, k, tv)); in_dd = False
    if in_dd: eps.append((s, t, len(equity) - 1, tv))
    depths = [abs(e[3]) for e in eps] or [0]; durs = [int((times[e[2]] - times[e[0]]).astype("timedelta64[D]").astype(int)) for e in eps] or [0]
    mi = int(np.argmin([e[3] for e in eps])) if eps else 0; me = eps[mi] if eps else (0, 0, 0, 0)
    return {"max": round(max(depths), 1), "avg": round(float(np.mean(depths)), 1), "maxdur": max(durs), "avgdur": round(float(np.mean(durs)), 1),
            "n": len(eps), "from": pd.Timestamp(times[me[0]]).strftime("%Y-%m-%d") if eps else "-", "trough": pd.Timestamp(times[me[1]]).strftime("%Y-%m-%d") if eps else "-"}


def main():
    d = pd.read_csv(DATA); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True); ts = d["ts"]; ts_ist = ts.dt.tz_localize(None); d["ts_ist"] = ts_ist
    N = len(d); period = f"{ts.iloc[0].date()} .. {ts.iloc[-1].date()}"; c15 = d["close"].values
    dir15 = supertrend(d["high"].values, d["low"].values, c15, ATR_N, FACT)
    mod = ts.dt.hour * 60 + ts.dt.minute; d["hb"] = pd.factorize(ts.dt.strftime("%Y-%m-%d") + "_" + ((mod - 555) // 60).astype(int).astype(str))[0]
    hourly = d.groupby("hb").agg(h=("high", "max"), l=("low", "min"), c=("close", "last")).sort_index()
    hts = d.groupby("hb")["ts_ist"].last().sort_index().values
    dir1h = supertrend(hourly["h"].values, hourly["l"].values, hourly["c"].values, ATR_N, FACT)

    t15 = extract(dir15, ts_ist.values, c15, ATR_N + 1, "15min")
    t1h = extract(dir1h, hts, hourly["c"].values, ATR_N + 1, "1hour")
    comb = pd.concat([t15, t1h], ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    comb["cum_net_combined"] = comb["net_points"].cumsum().round(1); comb.insert(0, "seq", range(1, len(comb) + 1))
    comb = comb.drop(columns=["trade_no", "cum_net_leg"])

    # ---- time-series equity (gross + net) for drawdown ----
    is_last = d["hb"].values != np.append(d["hb"].values[1:], -1)
    dir1h_pos = np.full(N, np.nan); dir1h_pos[is_last] = dir1h[d["hb"].values[is_last]]
    dir1h_pos = pd.Series(dir1h_pos).ffill().values; dp = np.nan_to_num(dir1h_pos).astype(int)
    start_c = max(ATR_N + 1, int(np.argmax(~np.isnan(dir1h_pos)))); dc = np.diff(c15); idx = np.arange(start_c, N - 1); etimes = ts_ist.values[start_c + 1:N]
    cost_ts = np.zeros(N)
    for tf_dir in (dir15, dp):                                          # 0.03% round-trip charged at each flip (close old + open new)
        fl = np.where(tf_dir[1:] != tf_dir[:-1])[0] + 1
        cost_ts[fl] += 2 * SIDE * c15[fl]
    gross_eq = np.cumsum((dir15[idx] + dp[idx]) * dc[idx]); net_eq = gross_eq - np.cumsum(cost_ts[idx + 1])
    DDg = dd_stats(gross_eq, etimes); DDn = dd_stats(net_eq, etimes)

    def stat(df):
        return dict(trades=len(df), gross=round(df.gross_points.sum(), 1), cost=round(df.cost_points.sum(), 1), net=round(df.net_points.sum(), 1),
                    win=round((df.net_points > 0).mean() * 100, 1), avg=round(df.net_points.mean(), 1), best=round(df.net_points.max(), 1), worst=round(df.net_points.min(), 1))
    s15, s1 = stat(t15), stat(t1h); nettot = s15["net"] + s1["net"]; grosstot = s15["gross"] + s1["gross"]; costtot = s15["cost"] + s1["cost"]
    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY dual-TF Supertrend(10,3) — index-point P&L (spot sim); FUTURES cost 0.03% round-trip"},
        {"metric": "Period", "value": period}, {"metric": "Cost model", "value": "0.03% round-trip on notional = 0.015% x price each side (entry+exit), in index points"},
        {"metric": "Execution", "value": "same-candle close (design choice; differs from daily-ST next-open)"}, {"metric": "", "value": ""},
        {"metric": "15min — gross / cost / NET points", "value": f"{s15['gross']} / {s15['cost']} / {s15['net']}"},
        {"metric": "15min — trades / win% (net)", "value": f"{s15['trades']} / {s15['win']}"},
        {"metric": "15min — avg/best/worst net pts", "value": f"{s15['avg']} / {s15['best']} / {s15['worst']}"}, {"metric": "", "value": ""},
        {"metric": "1hour — gross / cost / NET points", "value": f"{s1['gross']} / {s1['cost']} / {s1['net']}"},
        {"metric": "1hour — trades / win% (net)", "value": f"{s1['trades']} / {s1['win']}"},
        {"metric": "1hour — avg/best/worst net pts", "value": f"{s1['avg']} / {s1['best']} / {s1['worst']}"}, {"metric": "", "value": ""},
        {"metric": "COMBINED — GROSS points", "value": grosstot}, {"metric": "COMBINED — total COST points", "value": costtot},
        {"metric": "COMBINED — NET points", "value": round(nettot, 1)}, {"metric": "COMBINED — cost as % of gross", "value": f"{round(costtot/grosstot*100,1)}%"},
        {"metric": "COMBINED — total trades", "value": s15["trades"] + s1["trades"]},
        {"metric": "COMBINED — illustrative NET INR @ lot 75", "value": f"Rs.{round(nettot*LOT_REF):,} (ILLUSTRATIVE ONLY)"}, {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (per-candle MTM equity, points) ---", "value": "GROSS  ->  NET"},
        {"metric": "COMBINED — max drawdown (pts)", "value": f"{DDg['max']}  ->  {DDn['max']}"},
        {"metric": "COMBINED — avg drawdown (pts)", "value": f"{DDg['avg']}  ->  {DDn['avg']}"},
        {"metric": "COMBINED — max DD duration (days)", "value": f"{DDg['maxdur']}  ->  {DDn['maxdur']}"},
        {"metric": "COMBINED — avg DD duration (days)", "value": f"{DDg['avgdur']}  ->  {DDn['avgdur']}"},
        {"metric": "COMBINED — # DD episodes", "value": f"{DDg['n']}  ->  {DDn['n']}"},
        {"metric": "COMBINED — NET max DD period", "value": f"{DDn['from']} -> {DDn['trough']} (peak->trough)"},
        {"metric": "COMBINED — NET return / NET max-DD", "value": round(nettot / DDn['max'], 2)}, {"metric": "", "value": ""},
        {"metric": "NOTE", "value": "result/win% now on NET (after cost). Combined_Trades net_points sum to COMBINED NET."},
        {"metric": "NOTE", "value": "Cost is illustrative futures cost; index itself not tradable. 1,101 flips make cost material."},
    ])

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        comb.to_excel(w, sheet_name="Combined_Trades", index=False)
        t15.to_excel(w, sheet_name="15min_Trades", index=False)
        t1h.to_excel(w, sheet_name="1hour_Trades", index=False)
    print(f"15min: gross {s15['gross']} cost {s15['cost']} NET {s15['net']} (win {s15['win']}%)")
    print(f"1hour: gross {s1['gross']} cost {s1['cost']} NET {s1['net']} (win {s1['win']}%)")
    print(f"COMBINED: gross {grosstot} cost {costtot} ({round(costtot/grosstot*100,1)}%) NET {round(nettot,1)}")
    print(f"NET drawdown: max {DDn['max']} pts, avg {DDn['avg']} | NET return/maxDD = {round(nettot/DDn['max'],2)}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
