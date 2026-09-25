# -*- coding: utf-8 -*-
"""
sma44_reclaim_strategy.py — DAILY swing/positional backtest: 44-SMA reclaim + RSI momentum band
+ volume expansion. Fresh build on daily candles (aggregated from the 1-min master_data). Separate
from the intraday volume-breakout strategy.

UNIVERSE: NSE stocks, mcap > 1,500 Cr (no upper bound) on the signal day (nearest-preceding of the
7 quarterly mcap snapshots; Rs-Lakhs/100 = Cr). ~1,609 symbols in master_data.

ENTRY — ALL on the SAME daily candle T:
  1. mcap_cr > 1500                                               (flag f-universe)
  2. 55 <= RSI-14(Wilder, daily) <= 62                            (flag e, inclusive)
  3. vol_1w > vol_1m : mean(vol,5d) > mean(vol,20d), trailing inclusive of T   (flag b: 20d, incl)
  4. bullish: close>open AND close>prev_close
  5. 44-SMA reclaim: close>=sma44 AND (close-sma44)/sma44 <= PROX AND low<=sma44   (flag a)
       (PROX default 2% = key tunable; low<=sma44 => candle dipped to/through the SMA intraday)
  6. rising SMA: sma44[T] > sma44[T-1]                            (flag c: day-over-day)
ENTRY EXECUTION: LONG at NEXT day's OPEN (flag d, realistic). Exits scanned from the entry day.
EXIT: stop=entry*0.92 (-8%), target=entry*1.17 (+17%); daily low<=stop / high>=target, first hit.
      Same-day both -> STOP first (flag g), counted. No time exit (hold till hit); report holding days.
COST: gross primary; net variant = 0.30% round-trip (cash delivery STT+charges ballpark).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
MDIR = rb.BASE / "mcap_cache"
OUTDIR = rb.RESULTS / "sma44_reclaim"
IST = "Asia/Kolkata"

SNAPS = [("2022-03-31", "mcap_2022-03-31.xlsx"), ("2022-12-31", "mcap_2022-12-31.xlsx"),
         ("2023-03-31", "mcap_2023-03-31.xlsx"), ("2023-12-31", "mcap_2023-12-31.xlsx"),
         ("2024-03-28", "mcap_2024-03-28.xlsx"), ("2024-12-31", "mcap_2024-12-31.xlsx"),
         ("2025-12-31", "mcap_2025-12-31.xlsx")]

MCAP_MIN = 1500.0
RSI_LO, RSI_HI = 55.0, 62.0
PROX = 0.02                 # (a) key tunable: close within 2% above the 44-SMA
VOL_WK, VOL_MO = 5, 20      # (b) 1-week=5d, 1-month=20d, trailing inclusive
SL, TGT = 0.08, 0.17        # -8% / +17%
COST = 0.30                 # % round-trip (net variant)
BE_WR = SL / (SL + TGT) * 100.0   # 32%


def load_mcap():
    dates, dicts = [], []
    for ds, fn in SNAPS:
        df = pd.read_excel(MDIR / fn, header=None, skiprows=1, usecols=[1, 3])
        df.columns = ["symbol", "mcap_lakhs"]
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df["mcap_cr"] = pd.to_numeric(df["mcap_lakhs"], errors="coerce") / 100
        df = df.dropna(subset=["mcap_cr"])
        dates.append(np.datetime64(ds)); dicts.append(df.set_index("symbol")["mcap_cr"].to_dict())
    return np.array(dates), dicts


def wilder_rsi(close, n=14):
    c = np.asarray(close, float); m = len(c); rsi = np.full(m, np.nan)
    if m < n + 1:
        return rsi
    d = np.diff(c); gain = np.where(d > 0, d, 0.0); loss = np.where(d < 0, -d, 0.0)
    ag = gain[:n].mean(); al = loss[:n].mean()
    def _r(ag, al):
        if al == 0:
            return 100.0 if ag > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + ag / al)
    rsi[n] = _r(ag, al)
    for i in range(n + 1, m):
        ag = (ag * (n - 1) + gain[i - 1]) / n
        al = (al * (n - 1) + loss[i - 1]) / n
        rsi[i] = _r(ag, al)
    return rsi


def run(prox=PROX, vol_mo=VOL_MO, collect=True):
    df = pd.read_parquet(DAILY)
    snap_dates, snap_dicts = load_mcap()

    def mcap_of(sym, d64):
        idx = int(np.searchsorted(snap_dates, d64, "right")) - 1
        if idx < 0:
            idx = 0                      # before first snapshot -> use earliest (fallback)
        return snap_dicts[idx].get(sym, np.nan)

    trades = []
    for sym, s in df.groupby("symbol", sort=False):
        if len(s) < 50:
            continue
        s = s.sort_values("date")
        o = s["open"].values.astype(float); h = s["high"].values.astype(float)
        lo = s["low"].values.astype(float); c = s["close"].values.astype(float)
        v = s["volume"].values.astype(float); dt = s["date"].values
        n = len(s)
        sma = pd.Series(c).rolling(44).mean().values
        rsi = wilder_rsi(c, 14)
        v1w = pd.Series(v).rolling(VOL_WK).mean().values
        v1m = pd.Series(v).rolling(vol_mo).mean().values
        prevc = np.concatenate([[np.nan], c[:-1]])
        sma_prev = np.concatenate([[np.nan], sma[:-1]])

        for t in range(n):
            if t + 1 >= n:                       # need next-day open to enter
                continue
            if not (sma[t] == sma[t] and rsi[t] == rsi[t] and v1m[t] == v1m[t] and prevc[t] == prevc[t]
                    and sma_prev[t] == sma_prev[t]):
                continue
            if not (RSI_LO <= rsi[t] <= RSI_HI):                       # (e)
                continue
            if not (v1w[t] > v1m[t]):                                  # (3)
                continue
            if not (c[t] > o[t] and c[t] > prevc[t]):                 # (4)
                continue
            if not (c[t] >= sma[t] and (c[t] - sma[t]) / sma[t] <= prox and lo[t] <= sma[t]):  # (5)
                continue
            if not (sma[t] > sma_prev[t]):                            # (6)
                continue
            mc = mcap_of(sym, dt[t])
            if not (mc == mc and mc > MCAP_MIN):                      # (1)
                continue
            if not collect:
                trades.append(1); continue
            # ── entry next-day open, exit scan from entry day inclusive ──
            ei = t + 1
            entry = o[ei]
            if not (entry == entry and entry > 0):
                continue
            stop = entry * (1 - SL); tgt = entry * (1 + TGT)
            xtype, xprice, xi = "open", c[n - 1], n - 1
            amb = False
            for j in range(ei, n):
                hs = lo[j] <= stop; ht = h[j] >= tgt
                if hs and ht:
                    xtype, xprice, xi, amb = "stop", stop, j, True; break
                if hs:
                    xtype, xprice, xi = "stop", stop, j; break
                if ht:
                    xtype, xprice, xi = "target", tgt, j; break
            ret = (xprice - entry) / entry * 100.0
            trades.append({
                "symbol": sym, "signal_date": pd.Timestamp(dt[t]).strftime("%Y-%m-%d"),
                "entry_date": pd.Timestamp(dt[ei]).strftime("%Y-%m-%d"), "entry_price": round(entry, 2),
                "sma_44": round(sma[t], 2), "rsi_14": round(rsi[t], 2),
                "close_vs_sma_pct": round((c[t] - sma[t]) / sma[t] * 100, 2),
                "vol_1w": int(v1w[t]), "vol_1m": int(v1m[t]), "vol_1w_over_1m": round(v1w[t] / v1m[t], 2),
                "mcap_cr": round(float(mc), 0),
                "stop": round(stop, 2), "target": round(tgt, 2),
                "exit_date": pd.Timestamp(dt[xi]).strftime("%Y-%m-%d"), "exit_type": xtype,
                "exit_price": round(xprice, 2), "same_day_ambiguity": amb,
                "holding_days": int(xi - ei), "return_pct": round(ret, 3),
                "net_return_pct": round(ret - COST, 3)})
    return trades


def summarize(T):
    n = len(T)
    tgt = int((T.exit_type == "target").sum()); stp = int((T.exit_type == "stop").sum())
    opn = int((T.exit_type == "open").sum()); wr = tgt / n * 100
    eq = T.sort_values("entry_date").net_return_pct.cumsum().values
    dd = float((eq - np.maximum.accumulate(eq)).min())
    return {"n_trades": n, "n_target": tgt, "n_stop": stp, "n_open": opn,
            "win_rate_pct": round(wr, 1), "breakeven_wr_pct": round(BE_WR, 1),
            "clears_breakeven": bool(wr > BE_WR),
            "avg_return_gross_pct": round(T.return_pct.mean(), 3), "median_return_pct": round(T.return_pct.median(), 3),
            "avg_return_net_pct": round(T.net_return_pct.mean(), 3),
            "total_return_gross_pct": round(T.return_pct.sum(), 1), "total_return_net_pct": round(T.net_return_pct.sum(), 1),
            "expectancy_gross_pct": round(wr / 100 * (TGT * 100) - (1 - wr / 100) * (SL * 100), 3),
            "avg_holding_days": round(T.holding_days.mean(), 1), "median_holding_days": int(T.holding_days.median()),
            "max_holding_days": int(T.holding_days.max()),
            "max_drawdown_seq_net_pct": round(dd, 1), "n_same_day_ambiguity": int(T.same_day_ambiguity.sum())}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = run(PROX, VOL_MO, collect=True)
    T = pd.DataFrame(rows)
    if T.empty:
        print("no trades"); return
    T["year"] = T.entry_date.str[:4]

    overall = summarize(T)
    summ = pd.DataFrame([{"segment": "ALL", **overall}])

    # per-year
    yr = []
    for y, g in T.groupby("year"):
        yr.append({"year": y, **summarize(g)})
    YR = pd.DataFrame(yr)

    # signal frequency (per year / per month)
    freq_y = T.groupby("year").size().rename("signals").reset_index()
    T["month"] = T.entry_date.str[:7]
    freq_m = T.groupby("month").size()

    # portfolio capacity: max concurrent open positions (allow-all, equal ₹1L)  (flag f)
    ev = []
    for r in T.itertuples():
        ev.append((r.entry_date, 1));
    # build (entry,exit) integer overlap via sorted dates
    ed = pd.to_datetime(T.entry_date).values.astype("datetime64[D]").astype("int64")
    xd = pd.to_datetime(T.exit_date).values.astype("datetime64[D]").astype("int64")
    evs = sorted([(d, +1) for d in ed] + [(d, -1) for d in xd], key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, delta in evs:
        cur += delta; mx = max(mx, cur)
    max_concurrent = mx

    # PROX sweep (key tunable) — counts + win rate + avg (fast: collect metrics)
    sweep = []
    for p in [0.01, 0.015, 0.02, 0.03, 0.05]:
        rp = run(p, VOL_MO, collect=True)
        if rp:
            tp = pd.DataFrame(rp)
            wr = (tp.exit_type == "target").mean() * 100
            sweep.append({"PROX_pct": p * 100, "n_trades": len(tp),
                          "win_rate_pct": round(wr, 1), "clears_32": bool(wr > BE_WR),
                          "avg_return_gross_pct": round(tp.return_pct.mean(), 3),
                          "avg_return_net_pct": round(tp.net_return_pct.mean(), 3),
                          "total_net_pct": round(tp.net_return_pct.sum(), 1)})
    SW = pd.DataFrame(sweep)

    # equity curve (trade-sequence, net)
    Tord = T.sort_values("entry_date").reset_index(drop=True)
    Tord["equity_net_cum"] = Tord.net_return_pct.cumsum().round(2)

    with pd.ExcelWriter(OUTDIR / "sma44_reclaim.xlsx", engine="openpyxl") as w:
        summ.to_excel(w, sheet_name="summary", index=False)
        YR.to_excel(w, sheet_name="by_year", index=False)
        SW.to_excel(w, sheet_name="PROX_sweep", index=False)
        freq_y.to_excel(w, sheet_name="signals_per_year", index=False)
        freq_m.reset_index(name="signals").to_excel(w, sheet_name="signals_per_month", index=False)
        Tord.to_excel(w, sheet_name="all_trades", index=False)
    Tord.to_csv(OUTDIR / "all_trades.csv", index=False)

    pd.set_option("display.width", 260)
    print("=" * 100)
    print("44-SMA RECLAIM + RSI 55-62 + VOL EXPANSION — DAILY POSITIONAL  (mcap>1500Cr; -8%/+17%; 2:1.06)")
    print("=" * 100)
    print(f"PROX={PROX*100:.1f}%  1-month={VOL_MO}d  entry=next-open  breakeven WR={BE_WR:.0f}%")
    print("\n--- SUMMARY (all trades) ---")
    for k, val in overall.items():
        print(f"  {k:28s}: {val}")
    print(f"\n  signals/year: {dict(zip(freq_y.year, freq_y.signals))}")
    print(f"  max concurrent open positions (allow-all, portfolio capacity): {max_concurrent}")
    print("\n--- BY YEAR ---")
    print(YR[["year", "n_trades", "n_target", "n_stop", "n_open", "win_rate_pct", "clears_breakeven",
              "avg_return_net_pct", "total_return_net_pct", "avg_holding_days"]].to_string(index=False))
    print("\n--- PROX SWEEP (the key tunable, (a)) ---")
    print(SW.to_string(index=False))
    print("\n--- first 10 trades ---")
    print(Tord[["symbol", "entry_date", "entry_price", "sma_44", "rsi_14", "vol_1w_over_1m",
                "exit_type", "exit_date", "holding_days", "return_pct", "equity_net_cum"]].head(10).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
