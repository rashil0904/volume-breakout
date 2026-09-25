# -*- coding: utf-8 -*-
"""stock_rsi_divergence.py — hourly RSI-14 divergence strategy on individual stocks (mcap>1500Cr).
Pipeline: hourly candles -> Wilder RSI-14 -> exact-TradingView divergence (pivots 5/5, valuewhen, _inRange
5-60) + 30/70 filter -> trades 3% SL / 6% target (2:1), bullish->LONG bearish->SHORT, entry at
confirmation-bar hourly close, exits first-touch on 1-min candles (stop-first on same-1min-span),
hold-till-hit. mcap point-in-time. cost 0.20% round-trip.

FLAGS: (a) no 9:07 pre-open in the data -> candle-1 open = 09:15 for EVERY stock/day (universal fallback);
7 buckets [09:07-10 (=09:15),10-11,11-12,12-13,13-14,14-15,15-15:30). (b) Wilder RSI, continuous cross-day.
(c) exact TV + 30/70. (d) entry at confirmation bar T (pivot at T-5), no look-ahead. (e) 3/6% 1-min
first-touch, stop-first same-candle. (f) hold-till-hit. (g) mcap point-in-time; universe = current
master_data constituents -> SURVIVORSHIP-BIAS. (h) same spot instrument for RSI & exits; cost 0.20%.
SHORTS on cash stocks held across days are NOT deliverable in India -> bearish leg is a spot-price PROXY
(futures in practice) -> flagged.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

MASTER = rb.MASTER_DIR
MDIR = rb.BASE / "mcap_cache"
OUTDIR = rb.RESULTS / "stock_rsi_divergence"
IST = "Asia/Kolkata"
COST, MCAP_MIN = 0.20, 1500.0
POOL, PER, MAXPOS = 1_000_000, 100_000, 10
LBL, LBR, RLO, RHI = 5, 5, 5, 60
BUCKETS = [(547, 600, 1), (600, 660, 2), (660, 720, 3), (720, 780, 4), (780, 840, 5), (840, 900, 6), (900, 930, 7)]
SNAPS = [("2022-03-31", "mcap_2022-03-31.xlsx"), ("2022-12-31", "mcap_2022-12-31.xlsx"),
         ("2023-03-31", "mcap_2023-03-31.xlsx"), ("2023-12-31", "mcap_2023-12-31.xlsx"),
         ("2024-03-28", "mcap_2024-03-28.xlsx"), ("2024-12-31", "mcap_2024-12-31.xlsx"),
         ("2025-12-31", "mcap_2025-12-31.xlsx")]


def load_mcap():
    dates, dicts = [], []
    for ds, fn in SNAPS:
        d = pd.read_excel(MDIR / fn, header=None, skiprows=1, usecols=[1, 3])
        d.columns = ["symbol", "mcap_lakhs"]; d["symbol"] = d["symbol"].astype(str).str.strip().str.upper()
        d["mcap_cr"] = pd.to_numeric(d["mcap_lakhs"], errors="coerce") / 100
        dates.append(np.datetime64(ds)); dicts.append(d.dropna(subset=["mcap_cr"]).set_index("symbol")["mcap_cr"].to_dict())
    return np.array(dates), dicts


def wilder_rsi(close, n=14):
    c = np.asarray(close, float); m = len(c); rsi = np.full(m, np.nan)
    if m < n + 1:
        return rsi
    d = np.diff(c); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    ag, al = g[:n].mean(), l[:n].mean()
    def rs(a, b):
        return 100.0 if (b == 0 and a > 0) else (0.0 if a == 0 and b > 0 else (50.0 if b == 0 else 100 - 100 / (1 + a / b)))
    rsi[n] = rs(ag, al)
    for i in range(n + 1, m):
        ag = (ag * (n - 1) + g[i - 1]) / n; al = (al * (n - 1) + l[i - 1]) / n; rsi[i] = rs(ag, al)
    return rsi


def pivots(v, lbl, lbr, kind):
    n = len(v); found = []
    for t in range(lbl + lbr, n):
        c = t - lbr; cand = v[c]
        if not (cand == cand):
            continue
        left = v[c - lbl:c]; right = v[c + 1:c + lbr + 1]
        if np.isnan(left).any() or np.isnan(right).any():
            continue
        if kind == "low" and np.all(cand < left) and np.all(cand < right):
            found.append(t)
        elif kind == "high" and np.all(cand > left) and np.all(cand > right):
            found.append(t)
    return found


def divergences(rsi, low, high):
    """Exact TV regular bull/bear + 30/70. Returns list of (type, T_confirm, c_pivot)."""
    out = []
    pl = pivots(rsi, LBL, LBR, "low"); ph = pivots(rsi, LBL, LBR, "high")
    for pt, t in zip(pl, pl[1:]):
        c, cp = t - LBR, pt - LBR
        if not (RLO <= (t - pt) <= RHI):
            continue
        if low[c] < low[cp] and rsi[c] > rsi[cp] and rsi[c] > 30:
            out.append(("bullish", t, c))
    for pt, t in zip(ph, ph[1:]):
        c, cp = t - LBR, pt - LBR
        if not (RLO <= (t - pt) <= RHI):
            continue
        if high[c] > high[cp] and rsi[c] < rsi[cp] and rsi[c] < 70:
            out.append(("bearish", t, c))
    out.sort(key=lambda x: x[1])
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    snap_dates, snap_dicts = load_mcap()
    def mcap_of(sym, d64):
        idx = int(np.searchsorted(snap_dates, d64, "right")) - 1
        return snap_dicts[max(idx, 0)].get(sym, np.nan)

    files = sorted(MASTER.glob("*.parquet"))
    trades = []; n_amb = 0; n_halfday_stocks = 0; n_preopen_missing_days = 0; n_total_days = 0
    t0 = time.time()
    for fi, f in enumerate(files, 1):
        sym = f.stem
        try:
            raw = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception:
            continue
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        hm = ts.dt.hour * 60 + ts.dt.minute
        keep = (hm >= 547) & (hm < 930)
        raw = raw[keep].copy(); ts = ts[keep]; hm = hm[keep]
        if len(raw) < 500:
            continue
        raw = raw.assign(date=ts.dt.date, hm=hm.values)
        raw = raw.sort_values("timestamp").reset_index(drop=True)
        bidx = np.zeros(len(raw), int)
        hmv = raw["hm"].values
        for loo, hio, bi in BUCKETS:
            bidx[(hmv >= loo) & (hmv < hio)] = bi
        raw["bidx"] = bidx; raw["pos"] = np.arange(len(raw))
        lo1 = raw["low"].values.astype(float); hi1 = raw["high"].values.astype(float); cl1 = raw["close"].values.astype(float)
        ts1 = raw["timestamp"].values
        # hourly bars — integer bar-id on contiguous (date,bidx) blocks (fast; already time-sorted)
        dcode = pd.factorize(raw["date"])[0]
        change = np.empty(len(raw), bool); change[0] = True
        change[1:] = (dcode[1:] != dcode[:-1]) | (bidx[1:] != bidx[:-1])
        raw["barid"] = np.cumsum(change)
        g = raw.groupby("barid", sort=True)
        hb = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                           "close": g["close"].last(), "date": g["date"].first(), "bidx": g["bidx"].first(),
                           "close_pos": g["pos"].max()}).reset_index(drop=True)
        if len(hb) < 60:
            continue
        # data quality: candle-1 always fallback (no 9:07); half-days
        per_day = hb.groupby("date").size(); n_total_days += len(per_day)
        if (per_day != 7).any():
            n_halfday_stocks += 1
        n_preopen_missing_days += len(per_day)          # every day: no 9:07 print
        hc = hb["close"].values.astype(float); hl = hb["low"].values.astype(float); hh = hb["high"].values.astype(float)
        cpos = hb["close_pos"].values.astype(int); hdate = hb["date"].values
        rsi = wilder_rsi(hc, 14)
        divs = divergences(rsi, hl, hh)
        if not divs:
            continue
        M = len(raw); last_exit_pos = -1
        for typ, T, c in divs:
            entry = hc[T]; ps = cpos[T] + 1
            if ps >= M or ps <= last_exit_pos or entry <= 0:
                continue
            mc = mcap_of(sym, np.datetime64(pd.Timestamp(hdate[T])))
            if not (mc == mc and mc > MCAP_MIN):
                continue
            long = (typ == "bullish")
            stop = entry * (0.97 if long else 1.03); tgt = entry * (1.06 if long else 0.94)
            # vectorized first-touch on 1-min slice [ps:]
            shit = (lo1[ps:] <= stop) if long else (hi1[ps:] >= stop)
            thit = (hi1[ps:] >= tgt) if long else (lo1[ps:] <= tgt)
            js = int(np.argmax(shit)) if shit.any() else -1
            jt = int(np.argmax(thit)) if thit.any() else -1
            if js < 0 and jt < 0:
                xt, xp, xj, amb = "time", cl1[M - 1], M - 1, False
            elif js >= 0 and (jt < 0 or js < jt):
                xt, xp, xj, amb = "stop", stop, ps + js, False
            elif jt >= 0 and (js < 0 or jt < js):
                xt, xp, xj, amb = "target", tgt, ps + jt, False
            else:                                   # js == jt -> same 1-min candle spans both -> stop first
                xt, xp, xj, amb = "stop", stop, ps + js, True
            n_amb += int(amb)
            ret = (xp - entry) / entry * 100 if long else (entry - xp) / entry * 100
            trades.append({"stock": sym, "div_type": typ,
                           "confirmation_bar": f"{pd.Timestamp(hdate[T]).strftime('%Y-%m-%d')} b{hb['bidx'].values[T]}",
                           "pivot_bar": f"{pd.Timestamp(hdate[c]).strftime('%Y-%m-%d')} b{hb['bidx'].values[c]}",
                           "entry_price": round(entry, 2), "exit_price": round(float(xp), 2), "exit_type": xt,
                           "exit_timestamp": pd.Timestamp(ts1[xj]).tz_localize("UTC").tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
                           "holding_1min_bars": int(xj - ps + 1), "holding_hours": round((xj - ps + 1) / 60.0, 1),
                           "same_1min_ambiguity": amb, "return_pct": round(ret, 3), "net_return_pct": round(ret - COST, 3),
                           "entry_dt": pd.Timestamp(hdate[T]), "exit_dt": pd.Timestamp(ts1[xj]).tz_localize("UTC").tz_convert(IST).tz_localize(None),
                           "year": pd.Timestamp(hdate[T]).year})
            last_exit_pos = xj
        if fi % 300 == 0:
            print(f"  {fi}/{len(files)} stocks | {len(trades):,} trades | {time.time()-t0:.0f}s", flush=True)

    T = pd.DataFrame(trades).sort_values("entry_dt").reset_index(drop=True)
    print(f"stocks scanned {len(files):,} | trades {len(T):,} | same-1min-ambiguity {n_amb} | half-day stocks {n_halfday_stocks}")

    def block(df, lbl):
        n = len(df)
        if n == 0:
            return {"segment": lbl, "n_trades": 0}
        r = df["net_return_pct"]; rg = df["return_pct"]
        tg = int((df.exit_type == "target").sum()); sp = int((df.exit_type == "stop").sum()); tm = int((df.exit_type == "time").sum())
        wr = (r > 0).mean() * 100
        eq = df.sort_values("exit_dt")["net_return_pct"].cumsum().values
        dd = float((eq - np.maximum.accumulate(eq)).min())
        return {"segment": lbl, "n_trades": n, "n_target": tg, "n_stop": sp, "n_time": tm,
                "win_rate_pct": round(wr, 2), "clears_33.3_net": bool(wr > 33.333),
                "avg_ret_gross_pct": round(rg.mean(), 3), "avg_ret_net_pct": round(r.mean(), 3),
                "median_ret_net_pct": round(r.median(), 3), "total_ret_net_pct": round(r.sum(), 1),
                "expectancy_net_pct": round(r.mean(), 3), "avg_holding_hours": round(df["holding_hours"].mean(), 1),
                "max_holding_hours": round(df["holding_hours"].max(), 1), "max_dd_tradeseq_net_pct": round(dd, 1),
                "n_same1min_ambiguity": int(df["same_1min_ambiguity"].sum())}
    summary = pd.DataFrame([block(T[T.div_type == "bullish"], "bullish"), block(T[T.div_type == "bearish"], "bearish"), block(T, "combined")])

    # portfolio
    def portfolio(df):
        op = []; cash = POOL; realized = 0.0; taken = 0; skip = 0
        for r in df.sort_values("entry_dt").itertuples():
            keep = []
            for x, nt in op:
                if x <= r.entry_dt:
                    cash += PER * (1 + nt / 100); realized += PER * nt / 100
                else:
                    keep.append((x, nt))
            op = keep
            if len(op) < MAXPOS and cash >= PER:
                cash -= PER; op.append((r.exit_dt, r.net_return_pct)); taken += 1
            else:
                skip += 1
        for x, nt in op:
            realized += PER * nt / 100
        return round(realized / POOL * 100, 1), taken, skip
    pr, ptk, psk = portfolio(T)

    per_stock = T.groupby("stock").agg(n_trades=("stock", "size"), win_rate=("net_return_pct", lambda x: round((x > 0).mean() * 100, 1)),
                                       avg_net=("net_return_pct", "mean"), total_net=("net_return_pct", "sum")).round(3) \
        .reset_index().sort_values("total_net", ascending=False)
    yr = T.groupby(["year", "div_type"]).agg(n=("stock", "size"), win=("net_return_pct", lambda x: round((x > 0).mean() * 100, 1)),
                                             avg_net=("net_return_pct", "mean"), total_net=("net_return_pct", "sum")).round(2).reset_index()

    Tout = T.drop(columns=["entry_dt", "exit_dt"]); Tout["equity_net_cum"] = T["net_return_pct"].cumsum().round(2)
    with pd.ExcelWriter(OUTDIR / "stock_rsi_divergence.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        Tout.to_excel(w, sheet_name="all_trades", index=False)
        per_stock.to_excel(w, sheet_name="per_stock", index=False)
        yr.to_excel(w, sheet_name="per_year", index=False)
        pd.DataFrame([{"pool_inr": POOL, "per_trade_inr": PER, "max_concurrent": MAXPOS, "n_signals": len(T),
                       "n_taken": ptk, "n_skipped_full": psk, "portfolio_return_on_pool_pct": pr}]).to_excel(w, sheet_name="portfolio", index=False)
    Tout.to_csv(OUTDIR / "all_trades.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100 + "\nSTOCK HOURLY RSI-DIVERGENCE (3% SL / 6% target, 2:1; 1-min exits; net 0.20%)\n" + "=" * 100)
    cols = ["segment", "n_trades", "n_target", "n_stop", "n_time", "win_rate_pct", "clears_33.3_net", "avg_ret_net_pct",
            "median_ret_net_pct", "total_ret_net_pct", "avg_holding_hours", "max_dd_tradeseq_net_pct", "n_same1min_ambiguity"]
    print(summary[cols].to_string(index=False))
    print(f"\n  PORTFOLIO (Rs10L, Rs1L/trade, max 10): return {pr}% | taken {ptk} | skipped(full) {psk}")
    print(f"  DATA QUALITY: 9:07 pre-open absent for ALL stocks -> candle-1 open = 09:15 (universal fallback); "
          f"half-day stocks {n_halfday_stocks}")
    print("\n--- PER YEAR (net) ---"); print(yr.to_string(index=False))
    print("\n--- TOP 12 stocks by total net pnl ---"); print(per_stock.head(12).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
