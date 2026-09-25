# -*- coding: utf-8 -*-
"""fno_3leg_stocks.py — the 3-leg NIFTY strategy (daily 2-day-hilo-breakout+gap | 15-min Supertrend |
1-hour Supertrend) extended to the F&O STOCK universe. GROSS-ONLY: legs are NEVER netted against each
other, and returns are NEVER summed across stocks — every (stock, leg) is reported independently.

ONLY change vs the Nifty version: the DAILY leg's breakout margin is 0.1% of price (up = p2h*1.001,
down = p2l*0.999) instead of a flat 10 pts. Gap-day handling identical (RAW 2-day level for the gap check;
hold first 15 min; flip on the first-15-min low/high with NO extra margin). Supertrend(10,3), close-
confirmed same-candle-close flips — identical to Nifty. No futures cost (gross).

Data per stock: 1-min = master_data/{SYM}.parquet ; 15-min = checkpoints/{SYM}.csv ; 1-hour resampled
from 15-min ; daily aggregated from 1-min. Universe = reconciled BSE F&O baseline (206 syms, 204 w/ data).
"""
import sys, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from supertrend_dual_tf import supertrend, ATR_N, FACT

MD = rb.BASE / "master_data"; CK = rb.BASE / "checkpoints"
FO_CSV = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
OUTDIR = rb.RESULTS / "fno_3leg_stocks"; OUTDIR.mkdir(parents=True, exist_ok=True)
MARGIN_PCT = 0.001; FIRST15_END = 570        # 0.1% daily-leg margin ; first 15-min bar = 09:15-09:29
NOTIONAL = 1_000_000                          # Rs.10 Lakh deployed per trade (fixed notional) -> INR P&L = ret_fraction x NOTIONAL
COST_PER_TRADE_INR = 650                       # flat Rs.650 expense per trade (round-trip)


def _ist(ts):
    return pd.to_datetime(ts, utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)   # handles tz-aware parquet + +05:30 CSV strings


def daily_gap_leg(d):
    """DAILY 2-day breakout + gap, 0.1% margin. d has ts(naive IST), open/high/low/close (1-min). -> trades."""
    d = d.copy(); d["date"] = d["ts"].dt.normalize(); d["mod"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    daily = d.groupby("date").agg(dh=("high", "max"), dl=("low", "min"), do=("open", "first"))
    daily["p2h"] = daily["dh"].shift(1).rolling(2).max(); daily["p2l"] = daily["dl"].shift(1).rolling(2).min()
    daily["up"] = daily["p2h"] * (1 + MARGIN_PCT); daily["down"] = daily["p2l"] * (1 - MARGIN_PCT)   # 0.1% margin
    f15 = d[(d["mod"] >= 555) & (d["mod"] < FIRST15_END)].groupby("date").agg(fh15=("high", "max"), fl15=("low", "min"))
    daily = daily.join(f15)
    dm = d.merge(daily[["p2h", "p2l", "up", "down", "do", "fh15", "fl15"]], left_on="date", right_index=True, how="left")
    dt_a = dm["date"].values; mod_a = dm["mod"].values; hi_a = dm["high"].values; lo_a = dm["low"].values; op_a = dm["open"].values; tsr = dm["ts"].values
    up_a, dn_a, p2h_a, p2l_a, do_a, fh_a, fl_a = (dm[c].values for c in ["up", "down", "p2h", "p2l", "do", "fh15", "fl15"])
    pos = 0; entry = np.nan; entry_t = None; trades = []; cur_day = None; gap = None; gap_flipped = False
    for k in range(len(dm)):
        if dt_a[k] != cur_day:
            cur_day = dt_a[k]; gap = None; gap_flipped = False
            if not np.isnan(up_a[k]):
                if pos == 1 and do_a[k] < p2l_a[k]: gap = "down"
                elif pos == -1 and do_a[k] > p2h_a[k]: gap = "up"
        up, dn, h, l, o, t, m = up_a[k], dn_a[k], hi_a[k], lo_a[k], op_a[k], tsr[k], mod_a[k]
        if np.isnan(up):
            continue
        if gap in ("down", "up") and not gap_flipped:
            if m < FIRST15_END:
                continue
            if gap == "down" and pos == 1 and l <= fl_a[k]:
                trades.append(("Long", entry_t, entry, t, fl_a[k], fl_a[k] - entry)); pos, entry, entry_t = -1, fl_a[k], t; gap = None; gap_flipped = True
            elif gap == "up" and pos == -1 and h >= fh_a[k]:
                trades.append(("Short", entry_t, entry, t, fh_a[k], entry - fh_a[k])); pos, entry, entry_t = 1, fh_a[k], t; gap = None; gap_flipped = True
            continue
        if pos == 0:
            hu, hd = h >= up, l <= dn
            if hu and hd:
                pos, entry = (1, up) if abs(o - up) <= abs(o - dn) else (-1, dn); entry_t = t
            elif hu: pos, entry, entry_t = 1, up, t
            elif hd: pos, entry, entry_t = -1, dn, t
        elif pos == 1:
            if l <= dn:
                trades.append(("Long", entry_t, entry, t, dn, dn - entry)); pos, entry, entry_t = -1, dn, t
        else:
            if h >= up:
                trades.append(("Short", entry_t, entry, t, up, entry - up)); pos, entry, entry_t = 1, up, t
    return trades


def st_flip_trades(direction, ts, close, start):
    """Supertrend flip-to-flip: enter/exit at candle close. -> trades [(dir,entry_t,entry,exit_t,exit,pts)]."""
    flips = [i for i in range(start + 1, len(direction)) if direction[i] != direction[i - 1]]
    out = []
    for k, i in enumerate(flips):
        j = flips[k + 1] if k + 1 < len(flips) else len(close) - 1
        pts = direction[i] * (close[j] - close[i])
        out.append(("Long" if direction[i] == 1 else "Short", ts[i], close[i], ts[j], close[j], pts))
    return out


def leg_stats(trades):
    if not trades:
        return {"trades": 0, "win_pct": 0.0, "avg_hold_days": 0.0, "total_pts": 0.0, "total_ret_pct": 0.0, "avg_ret_pct": 0.0, "avg_pts": 0.0, "gross_inr": 0.0, "cost_inr": 0.0, "net_inr": 0.0}
    pts = np.array([t[5] for t in trades]); entry = np.array([t[2] for t in trades]); n = len(trades)
    ed = pd.to_datetime([t[1] for t in trades]).normalize(); xd = pd.to_datetime([t[3] for t in trades]).normalize()
    hold = (xd - ed).days; ret_pct = pts / entry * 100
    gross_inr = float((pts / entry * NOTIONAL).sum())            # each trade Rs.10L notional; sum of INR P&L
    cost_inr = n * COST_PER_TRADE_INR                            # Rs.650 per trade
    return {"trades": n, "win_pct": round((pts > 0).mean() * 100, 1), "avg_hold_days": round(float(np.mean(hold)), 2),
            "total_pts": round(float(pts.sum()), 1), "total_ret_pct": round(float(ret_pct.sum()), 1),
            "avg_ret_pct": round(float(ret_pct.mean()), 3), "avg_pts": round(float(pts.mean()), 2),
            "gross_inr": round(gross_inr), "cost_inr": cost_inr, "net_inr": round(gross_inr - cost_inr)}   # TOTAL INR @ Rs.10L/trade, minus Rs.650/trade


def process(symbol):
    try:
        d1 = pd.read_parquet(MD / f"{symbol}.parquet", columns=["timestamp", "open", "high", "low", "close"])
        d1 = d1.assign(ts=_ist(d1["timestamp"])).sort_values("ts").reset_index(drop=True)
        d15 = pd.read_csv(CK / f"{symbol}.csv", usecols=["timestamp", "open", "high", "low", "close"])
        d15 = d15.assign(ts=_ist(d15["timestamp"])).sort_values("ts").reset_index(drop=True)
        # LEG1 daily breakout + gap
        daily = daily_gap_leg(d1[["ts", "open", "high", "low", "close"]])
        # LEG2 15-min Supertrend
        c15 = d15["close"].values; dir15 = supertrend(d15["high"].values, d15["low"].values, c15, ATR_N, FACT)
        t15 = st_flip_trades(dir15, d15["ts"].values, c15, ATR_N + 1)
        # LEG3 1-hour Supertrend (resample 15-min, anchored 09:15)
        mod = d15["ts"].dt.hour * 60 + d15["ts"].dt.minute
        d15 = d15.assign(hb=pd.factorize(d15["ts"].dt.strftime("%Y-%m-%d") + "_" + ((mod - 555) // 60).astype(int).astype(str))[0])
        hourly = d15.groupby("hb").agg(h=("high", "max"), l=("low", "min"), c=("close", "last")).sort_index()
        hts = d15.groupby("hb")["ts"].last().sort_index().values
        dir1h = supertrend(hourly["h"].values, hourly["l"].values, hourly["c"].values, ATR_N, FACT)
        t1h = st_flip_trades(dir1h, hts, hourly["c"].values, ATR_N + 1)
        cov = f"{d1['ts'].iloc[0].date()}..{d1['ts'].iloc[-1].date()}"
        return {"symbol": symbol, "n_1min": len(d1), "n_15min": len(d15), "coverage": cov,
                "daily": leg_stats(daily), "15min": leg_stats(t15), "1hr": leg_stats(t1h)}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def main():
    syms = sorted(pd.read_csv(FO_CSV, usecols=["symbol"])["symbol"].unique())
    syms = [s for s in syms if (MD / f"{s}.parquet").exists() and (CK / f"{s}.csv").exists()]
    print(f"F&O universe with data: {len(syms)} stocks — running 3 legs each (gross-only, no netting)...")
    res = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futs = {ex.submit(process, s): s for s in syms}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result(); res.append(r)
            if n % 25 == 0 or n == len(syms): print(f"  {n}/{len(syms)} done")
    ok = [r for r in res if "error" not in r]; err = [r for r in res if "error" in r]
    ok.sort(key=lambda r: r["symbol"])

    # ---- Summary_Returns: rows=stock, cols=leg -> return% AND points side by side ----
    srows = []
    for r in ok:
        srows.append({"symbol": r["symbol"],
                      "daily_avgret%": r["daily"]["avg_ret_pct"], "daily_net_inr": r["daily"]["net_inr"],
                      "15min_avgret%": r["15min"]["avg_ret_pct"], "15min_net_inr": r["15min"]["net_inr"],
                      "1hr_avgret%": r["1hr"]["avg_ret_pct"], "1hr_net_inr": r["1hr"]["net_inr"]})
    SUMM = pd.DataFrame(srows)
    # per-stock: avg-per-trade% and NET INR (Rs.10L/trade, minus Rs.650/trade), each = mean of its 3 timeframes
    SUMM["avg_3tf_ret%"] = SUMM[["daily_avgret%", "15min_avgret%", "1hr_avgret%"]].mean(axis=1).round(3)
    SUMM["avg_3tf_net_inr"] = SUMM[["daily_net_inr", "15min_net_inr", "1hr_net_inr"]].mean(axis=1).round()

    # OVERALL = per timeframe: GROSS / COST / NET INR across ALL stocks combined, + avg NET per stock.  (NET = Rs.10L/trade gross minus Rs.650/trade)
    nstk = len(SUMM)
    def col(leg, key): return sum(r[leg][key] for r in ok)
    OVERALL = pd.DataFrame([
        {"timeframe": leg_lbl, "avg_NET_inr_per_stock": round(col(leg, "net_inr") / nstk),
         "TOTAL_gross_inr_all_stocks": round(col(leg, "gross_inr")), "TOTAL_cost_inr_all_stocks": col(leg, "cost_inr"),
         "TOTAL_NET_inr_all_stocks": round(col(leg, "net_inr"))}
        for leg, leg_lbl in [("daily", "daily"), ("15min", "15min"), ("1hr", "1hr")]]
        + [{"timeframe": f"ALL-3-LEGS SUMMED ({nstk} stocks)", "avg_NET_inr_per_stock": round(sum(col(l, "net_inr") for l in ("daily", "15min", "1hr")) / nstk),
            "TOTAL_gross_inr_all_stocks": round(sum(col(l, "gross_inr") for l in ("daily", "15min", "1hr"))),
            "TOTAL_cost_inr_all_stocks": sum(col(l, "cost_inr") for l in ("daily", "15min", "1hr")),
            "TOTAL_NET_inr_all_stocks": round(sum(col(l, "net_inr") for l in ("daily", "15min", "1hr")))}])

    # ---- Per_Leg_Detail: one row per (stock, leg) ----
    drows = []
    for r in ok:
        for leg in ("daily", "15min", "1hr"):
            s = r[leg]; drows.append({"symbol": r["symbol"], "leg": leg, "trades": s["trades"], "win_%": s["win_pct"],
                                      "avg_hold_days": s["avg_hold_days"], "avg_ret_%_per_trade": s["avg_ret_pct"], "avg_pts_per_trade": s["avg_pts"],
                                      "total_ret_%": s["total_ret_pct"], "total_pts": s["total_pts"],
                                      "gross_inr@10L": s["gross_inr"], "cost_inr@650/trade": s["cost_inr"], "net_inr": s["net_inr"],
                                      "n_1min": r["n_1min"], "n_15min": r["n_15min"], "coverage": r["coverage"]})
    DETAIL = pd.DataFrame(drows)

    # ---- Low-trade flags (data-quality / participation check) ----
    med = DETAIL.groupby("leg")["trades"].median().to_dict()
    flags = []
    for _, row in DETAIL.iterrows():
        thr = max(0.5 * med[row["leg"]], 8)                    # < 50% of leg median (or <8 absolute) => flag
        if row["trades"] < thr:
            flags.append({"symbol": row["symbol"], "leg": row["leg"], "trades": row["trades"],
                          "leg_median_trades": med[row["leg"]], "flag_threshold": round(thr, 1),
                          "n_1min": row["n_1min"], "n_15min": row["n_15min"], "coverage": row["coverage"]})
    FLAGS = pd.DataFrame(flags).sort_values(["leg", "trades"]) if flags else pd.DataFrame(columns=["symbol", "leg", "trades"])

    info = pd.DataFrame([
        {"metric": "Strategy", "value": "3-LEG (daily 2d-breakout+gap | 15m Supertrend | 1h Supertrend) on F&O STOCKS — GROSS-ONLY"},
        {"metric": "GROSS-ONLY", "value": "legs NOT netted vs each other; returns NOT summed across stocks; each (stock,leg) independent"},
        {"metric": "Daily-leg margin", "value": "0.1% of price (up=p2h*1.001, down=p2l*0.999) — ONLY change vs Nifty; gap uses RAW 2-day level"},
        {"metric": "Supertrend", "value": f"({ATR_N},{FACT}) identical to Nifty; close-confirmed same-candle-close flips"},
        {"metric": "Sizing / cost", "value": "Rs.10,00,000 notional per trade ; expense Rs.650 flat per trade ; NET = gross INR - 650 x trades"},
        {"metric": "Return convention", "value": "Summary_Returns & Overall_Averages = AVG PER TRADE (mean of trade_pts/entry x 100, and mean pts/trade). Per_Leg_Detail also keeps totals."},
        {"metric": "Averaging", "value": "per stock: mean of its 3 TFs' avg-per-trade; overall: mean across stocks per TF, plus grand avg of the 3-TF number"},
        {"metric": "Universe (with data)", "value": f"{len(ok)} F&O stocks" + (f"; {len(err)} errored" if err else "")},
        {"metric": "Period", "value": ok[0]["coverage"] if ok else "-"},
        {"metric": "Low-trade flag rule", "value": "leg trades < 50% of that leg's cross-stock median (or < 8) -> possible data gap / low participation"},
        {"metric": "Data", "value": "1-min master_data/*.parquet | 15-min checkpoints/*.csv | 1-hr resampled | daily from 1-min"},
    ])
    if err:
        info = pd.concat([info, pd.DataFrame([{"metric": f"ERROR {e['symbol']}", "value": e["error"][:80]} for e in err])], ignore_index=True)

    with pd.ExcelWriter(OUTDIR / "fno_3leg_stocks.xlsx", engine="openpyxl") as w:
        info.to_excel(w, sheet_name="Info", index=False)
        OVERALL.to_excel(w, sheet_name="Overall_Averages", index=False)
        SUMM.to_excel(w, sheet_name="Summary_Returns", index=False)
        DETAIL.to_excel(w, sheet_name="Per_Leg_Detail", index=False)
        FLAGS.to_excel(w, sheet_name="Low_Trade_Flags", index=False)
    SUMM.to_csv(OUTDIR / "fno_3leg_summary_returns.csv", index=False)
    DETAIL.to_csv(OUTDIR / "fno_3leg_per_leg_detail.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 96 + "\n3-LEG on F&O STOCKS (GROSS-ONLY; per stock, per leg — no netting)\n" + "=" * 96)
    print(f"stocks {len(ok)} | period {ok[0]['coverage'] if ok else '-'} | daily-leg margin 0.1%")
    print("\nmedian trades/leg:", {k: int(v) for k, v in med.items()})
    print("\nleg AVG-PER-TRADE (across stocks):")
    for leg in ("daily", "15min", "1hr"):
        sub = DETAIL[DETAIL.leg == leg]
        print(f"  {leg:6s}: avg ret%/trade {sub['avg_ret_%_per_trade'].mean():7.3f} | avg pts/trade {sub['avg_pts_per_trade'].mean():7.2f} | "
              f"mean trades {sub['trades'].mean():5.1f} | mean win% {sub['win_%'].mean():.1f} | positive stocks {round((sub['avg_ret_%_per_trade']>0).mean()*100,1)}%")
    print("\n--- TOTAL INR PnL @ Rs.10L/trade, minus Rs.650/trade expense: combined (all stocks) + avg NET per stock ---")
    ov = OVERALL.copy()
    for c in ("avg_NET_inr_per_stock", "TOTAL_gross_inr_all_stocks", "TOTAL_cost_inr_all_stocks", "TOTAL_NET_inr_all_stocks"):
        ov[c] = ov[c].map(lambda x: f"Rs.{x:,}")
    print(ov.to_string(index=False))
    print(f"\nlow-trade flags: {len(FLAGS)} (stock,leg) cells below threshold")
    if len(FLAGS): print(FLAGS.head(12).to_string(index=False))
    if err: print(f"\nERRORS ({len(err)}):", [e["symbol"] for e in err])
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
