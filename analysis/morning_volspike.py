# -*- coding: utf-8 -*-
"""morning_volspike.py — intraday morning volume-spike strategy (mcap 1500-5000). Entry: first 1-min
candle in 09:15-11:00 where return>=+4% (vs prev VWAP-close) AND cumulative-vol-so-far >= 3x 36d-avg
full-day volume. Enter LONG at that candle's close. Exit: 15:20 1-min OPEN (pure time square-off).
One entry/stock/day. Capital Rs5L pool, min(Rs1L, 5L/N)/trade. Cost 0.15% round-trip.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DIAG = rb.RESULTS / "diagnostic_table.csv"
MASTER = rb.MASTER_DIR
OUTDIR = rb.RESULTS / "morning_volspike"
IST = "Asia/Kolkata"
RET_MIN, VM_MULT = 4.0, 3.0
ENTRY_LO, ENTRY_HI = 555, 660       # 09:15 .. 11:00 inclusive
EXIT_HM = 920                        # 15:20 square-off (1-min open)
COST = 0.15
BP, PER = 500_000, 100_000
RF = 0.075


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(DIAG, usecols=["symbol", "date", "prev_day_vwap_close", "avg_nday_fullday_volume"], parse_dates=["date"])
    d["date"] = d["date"].dt.date
    d = d.dropna(subset=["prev_day_vwap_close", "avg_nday_fullday_volume"])
    d = d[(d["prev_day_vwap_close"] > 0) & (d["avg_nday_fullday_volume"] > 0)]
    ref = {(s, dt): (pc, av) for s, dt, pc, av in zip(d["symbol"], d["date"], d["prev_day_vwap_close"], d["avg_nday_fullday_volume"])}
    syms = sorted(d["symbol"].unique())
    print(f"universe (mcap1500-5000) symbols: {len(syms):,} | reference stock-days: {len(d):,}")

    rows = []; t0 = time.time()
    for si, sym in enumerate(syms, 1):
        pq = MASTER / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq, columns=["timestamp", "open", "close", "volume"])
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        hm = ts.dt.hour * 60 + ts.dt.minute
        keep = (hm >= 555) & (hm < 930)
        raw = raw[keep].copy(); ts = ts[keep]; hm = hm[keep]
        raw = raw.assign(date=ts.dt.date, hm=hm.values, ts=ts.values)
        raw = raw.sort_values("timestamp")
        raw["cumvol"] = raw.groupby("date")["volume"].cumsum()
        dser = raw["date"].values; hmv = raw["hm"].values; cl = raw["close"].values.astype(float)
        op = raw["open"].values.astype(float); cv = raw["cumvol"].values.astype(float); tsv = raw["ts"].values
        # per-day exit (15:20 open)
        exitrow = raw[raw["hm"] == EXIT_HM].groupby("date")["open"].first()
        exit_map = exitrow.to_dict()
        # scan per date
        for dt0, idxs in raw.groupby("date").indices.items():
            r = ref.get((sym, dt0))
            if r is None:
                continue
            pc, av = r
            if dt0 not in exit_map:
                continue
            sl = idxs  # positional indices for this date (sorted)
            # morning window
            for k in sl:
                if hmv[k] < ENTRY_LO or hmv[k] > ENTRY_HI:
                    if hmv[k] > ENTRY_HI:
                        break
                    continue
                ret = (cl[k] - pc) / pc * 100.0
                vmult = cv[k] / av
                if ret >= RET_MIN and vmult >= VM_MULT:
                    entry = cl[k]; ex = float(exit_map[dt0])
                    gret = (ex - entry) / entry * 100.0
                    rows.append({"symbol": sym, "date": str(dt0),
                                 "trigger_time": f"{hmv[k]//60:02d}:{hmv[k]%60:02d}", "trigger_hm": int(hmv[k]),
                                 "entry_price": round(entry, 2), "return_at_entry_pct": round(ret, 2),
                                 "cum_vol_multiple_at_entry": round(vmult, 2),
                                 "exit_time": "15:20", "exit_price": round(ex, 2),
                                 "return_pct": round(gret, 3), "net_return_pct": round(gret - COST, 3),
                                 "year": pd.Timestamp(dt0).year})
                    break
        if si % 200 == 0:
            print(f"  {si}/{len(syms)} | {len(rows):,} trades | {time.time()-t0:.0f}s", flush=True)

    T = pd.DataFrame(rows).sort_values(["date", "trigger_hm"]).reset_index(drop=True)
    print(f"trades: {len(T):,} | stocks with >=1 trade: {T['symbol'].nunique():,}")

    # allocation: per day, min(1L, 5L/N); pnl in Rs
    nday = T.groupby("date")["symbol"].transform("size")
    T["alloc"] = np.minimum(PER, BP / nday)
    T["shares"] = (T["alloc"] / T["entry_price"]).astype(int)
    T["capital_deployed"] = T["shares"] * T["entry_price"]
    T["gross_pnl"] = T["shares"] * (T["exit_price"] - T["entry_price"])
    T["net_pnl"] = T["gross_pnl"] - COST / 100.0 * T["capital_deployed"]

    def summ(net):
        rc = "net_return_pct" if net else "return_pct"; pc = "net_pnl" if net else "gross_pnl"
        r = T[rc]; win = r > 0
        dpnl = T.groupby("date")[pc].sum().sort_index()
        eq = dpnl.cumsum().values; dd = float((eq - np.maximum.accumulate(eq)).min())
        dret = dpnl / BP * 100; rfd = RF / 252 * 100
        sharpe = round((dret.mean() - rfd) / dret.std(ddof=1) * np.sqrt(252), 3) if len(dret) > 1 else 0.0
        gp = r[r > 0].sum(); gl = -r[r < 0].sum()
        return {"basis": "net" if net else "gross", "n_trades": len(T), "win_rate_pct": round(win.mean() * 100, 2),
                "avg_return_pct": round(r.mean(), 3), "median_return_pct": round(r.median(), 3),
                "avg_winner_pct": round(r[win].mean(), 3), "avg_loser_pct": round(r[~win].mean(), 3),
                "profit_factor": round(gp / gl, 3) if gl > 0 else np.inf,
                "total_return_fixedbase_pct": round(T[pc].sum() / BP * 100, 2), "total_pnl_inr": round(T[pc].sum(), 0),
                "avg_capital_deployed": round(T["capital_deployed"].mean(), 0),
                "max_dd_daily_inr": round(dd, 0), "sharpe": sharpe}
    SUM = pd.DataFrame([summ(False), summ(True)])

    # trigger-time buckets + return-by-trigger
    def tb(hm):
        return "0915-0930" if hm < 570 else "0930-1000" if hm < 600 else "1000-1030" if hm < 630 else "1030-1100"
    T["trigger_bucket"] = T["trigger_hm"].map(tb)
    TT = T.groupby("trigger_bucket").agg(n=("symbol", "size"), pct_of_trades=("symbol", lambda x: round(len(x) / len(T) * 100, 1)),
                                         avg_ret_net=("net_return_pct", "mean"), median_ret_net=("net_return_pct", "median"),
                                         win_rate=("net_return_pct", lambda x: round((x > 0).mean() * 100, 1)),
                                         avg_vmult=("cum_vol_multiple_at_entry", "mean"),
                                         avg_ret_at_entry=("return_at_entry_pct", "mean")).round(3).reset_index()
    yr = T.groupby("year").agg(n=("symbol", "size"), win=("net_return_pct", lambda x: round((x > 0).mean() * 100, 1)),
                               avg_net=("net_return_pct", "mean"), total_net_pnl=("net_pnl", "sum")).round(2).reset_index()
    per_day = T.groupby("date").size()
    freq = {"n_trading_days_with_signal": int(per_day.shape[0]), "avg_trades_per_signal_day": round(per_day.mean(), 2),
            "max_trades_in_a_day": int(per_day.max()), "total_trades": len(T)}

    Tout = T.drop(columns=["trigger_hm"]); Tout["equity_net_cum_inr"] = T["net_pnl"].cumsum().round(0)
    with pd.ExcelWriter(OUTDIR / "morning_volspike.xlsx", engine="openpyxl") as w:
        SUM.to_excel(w, sheet_name="summary", index=False)
        Tout.to_excel(w, sheet_name="all_trades", index=False)
        TT.to_excel(w, sheet_name="return_by_trigger_time", index=False)
        yr.to_excel(w, sheet_name="per_year", index=False)
        pd.DataFrame([freq]).to_excel(w, sheet_name="signal_frequency", index=False)
    Tout.to_csv(OUTDIR / "all_trades.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 96 + "\nMORNING VOLUME-SPIKE (>=+4% & cumvol>=3x36davg, 09:15-11:00; exit 15:20; net 0.15%)\n" + "=" * 96)
    print(SUM.to_string(index=False))
    print("\n--- RETURN BY TRIGGER TIME (earlier = bigger relative spike) ---")
    print(TT.to_string(index=False))
    print("\n--- PER YEAR (net) ---"); print(yr.to_string(index=False))
    print(f"\n--- SIGNAL FREQUENCY ---\n  {freq}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
