# -*- coding: utf-8 -*-
"""morning_volspike_capped.py — morning volume-spike with entry-return BAND +4% to +7% (inclusive).
Entry = FIRST 1-min candle in 09:15-11:00 where 4<=ret<=7 (vs prev VWAP-close) AND cumvol>=3x36d-avg.
A stock already >+7% when volume confirms is SKIPPED at that candle (not chased); first candle where
BOTH align & in-band wins (a fade-back into band with sustained 3x volume can still trigger later).
Exit 15:20 open, cost 0.15%. Compares vs the uncapped (>=+4%) run + isolates removed >+7% trades.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DIAG = rb.RESULTS / "diagnostic_table.csv"
MASTER = rb.MASTER_DIR
OUTDIR = rb.RESULTS / "morning_volspike_capped"
UNCAPPED = rb.RESULTS / "morning_volspike" / "all_trades.csv"
IST = "Asia/Kolkata"
RET_MIN, RET_MAX, VM_MULT = 4.0, 7.0, 3.0
ENTRY_LO, ENTRY_HI, EXIT_HM = 555, 660, 920
COST, BP, PER, RF = 0.15, 500_000, 100_000, 0.075


def build():
    d = pd.read_csv(DIAG, usecols=["symbol", "date", "prev_day_vwap_close", "avg_nday_fullday_volume"], parse_dates=["date"])
    d["date"] = d["date"].dt.date
    d = d.dropna(subset=["prev_day_vwap_close", "avg_nday_fullday_volume"])
    d = d[(d["prev_day_vwap_close"] > 0) & (d["avg_nday_fullday_volume"] > 0)]
    ref = {(s, dt): (pc, av) for s, dt, pc, av in zip(d["symbol"], d["date"], d["prev_day_vwap_close"], d["avg_nday_fullday_volume"])}
    syms = sorted(d["symbol"].unique())
    print(f"universe symbols: {len(syms):,}")
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
        raw = raw.assign(date=ts.dt.date, hm=hm.values).sort_values("timestamp")
        raw["cumvol"] = raw.groupby("date")["volume"].cumsum()
        hmv = raw["hm"].values; cl = raw["close"].values.astype(float); cv = raw["cumvol"].values.astype(float)
        exit_map = raw[raw["hm"] == EXIT_HM].groupby("date")["open"].first().to_dict()
        for dt0, idxs in raw.groupby("date").indices.items():
            r = ref.get((sym, dt0))
            if r is None or dt0 not in exit_map:
                continue
            pc, av = r
            for k in idxs:
                if hmv[k] > ENTRY_HI:
                    break
                if hmv[k] < ENTRY_LO:
                    continue
                ret = (cl[k] - pc) / pc * 100.0; vmult = cv[k] / av
                if RET_MIN <= ret <= RET_MAX and vmult >= VM_MULT:
                    entry = cl[k]; ex = float(exit_map[dt0]); gret = (ex - entry) / entry * 100.0
                    rows.append({"symbol": sym, "date": str(dt0), "trigger_time": f"{hmv[k]//60:02d}:{hmv[k]%60:02d}",
                                 "trigger_hm": int(hmv[k]), "entry_price": round(entry, 2),
                                 "return_at_entry_pct": round(ret, 2), "cum_vol_multiple_at_entry": round(vmult, 2),
                                 "exit_time": "15:20", "exit_price": round(ex, 2), "return_pct": round(gret, 3),
                                 "net_return_pct": round(gret - COST, 3), "year": pd.Timestamp(dt0).year})
                    break
        if si % 250 == 0:
            print(f"  {si}/{len(syms)} | {len(rows):,} trades | {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows).sort_values(["date", "trigger_hm"]).reset_index(drop=True)


def summ(T, net):
    rc = "net_return_pct" if net else "return_pct"; pc = "net_pnl" if net else "gross_pnl"
    r = T[rc]; win = r > 0
    dpnl = T.groupby("date")[pc].sum().sort_index(); eq = dpnl.cumsum().values
    dd = float((eq - np.maximum.accumulate(eq)).min()); dret = dpnl / BP * 100; rfd = RF / 252 * 100
    sharpe = round((dret.mean() - rfd) / dret.std(ddof=1) * np.sqrt(252), 3) if len(dret) > 1 else 0.0
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    return {"basis": "net" if net else "gross", "n_trades": len(T), "win_rate_pct": round(win.mean() * 100, 2),
            "avg_return_pct": round(r.mean(), 3), "median_return_pct": round(r.median(), 3),
            "avg_winner_pct": round(r[win].mean(), 3), "avg_loser_pct": round(r[~win].mean(), 3),
            "profit_factor": round(gp / gl, 3) if gl > 0 else np.inf,
            "total_return_fixedbase_pct": round(T[pc].sum() / BP * 100, 2), "total_pnl_inr": round(T[pc].sum(), 0),
            "max_dd_daily_inr": round(dd, 0), "sharpe": sharpe}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = build()
    print(f"CAPPED (+4..+7%) trades: {len(T):,}")
    nday = T.groupby("date")["symbol"].transform("size")
    T["alloc"] = np.minimum(PER, BP / nday); T["shares"] = (T["alloc"] / T["entry_price"]).astype(int)
    T["capital_deployed"] = T["shares"] * T["entry_price"]; T["gross_pnl"] = T["shares"] * (T["exit_price"] - T["entry_price"])
    T["net_pnl"] = T["gross_pnl"] - COST / 100.0 * T["capital_deployed"]
    SUM = pd.DataFrame([summ(T, False), summ(T, True)])

    def tb(hm):
        return "0915-0930" if hm < 570 else "0930-1000" if hm < 600 else "1000-1030" if hm < 630 else "1030-1100"
    T["trigger_bucket"] = T["trigger_hm"].map(tb)
    TT = T.groupby("trigger_bucket").agg(n=("symbol", "size"), avg_ret_net=("net_return_pct", "mean"),
                                         win_rate=("net_return_pct", lambda x: round((x > 0).mean() * 100, 1)),
                                         avg_ret_at_entry=("return_at_entry_pct", "mean")).round(3).reset_index()
    yr = T.groupby("year").agg(n=("symbol", "size"), win=("net_return_pct", lambda x: round((x > 0).mean() * 100, 1)),
                               avg_net=("net_return_pct", "mean"), total_net_pnl=("net_pnl", "sum")).round(2).reset_index()

    # comparison vs uncapped
    cmp_rows = []; removed = None
    if UNCAPPED.exists():
        U = pd.read_csv(UNCAPPED)
        kept = U[U["return_at_entry_pct"] <= 7.0]; removed = U[U["return_at_entry_pct"] > 7.0]
        def hl(df, lbl):
            r = df["net_return_pct"]
            return {"set": lbl, "n_trades": len(df), "win_rate_pct": round((r > 0).mean() * 100, 2),
                    "avg_net_ret_pct": round(r.mean(), 3), "total_net_ret_on_5L_pct": round(df["net_pnl"].sum() / BP * 100, 2),
                    "total_net_pnl_inr": round(df["net_pnl"].sum(), 0)}
        cmp_rows = [hl(U, "UNCAPPED (>=+4%)"), hl(kept, "  of which <=+7% (kept)"),
                    hl(removed, "  REMOVED (>+7%, cap drops)"),
                    {"set": "CAPPED run (+4..+7, this)", "n_trades": len(T), "win_rate_pct": round((T["net_return_pct"] > 0).mean() * 100, 2),
                     "avg_net_ret_pct": round(T["net_return_pct"].mean(), 3), "total_net_ret_on_5L_pct": round(T["net_pnl"].sum() / BP * 100, 2),
                     "total_net_pnl_inr": round(T["net_pnl"].sum(), 0)}]
    CMP = pd.DataFrame(cmp_rows)

    Tout = T.drop(columns=["trigger_hm"]); Tout["equity_net_cum_inr"] = T["net_pnl"].cumsum().round(0)
    with pd.ExcelWriter(OUTDIR / "morning_volspike_capped.xlsx", engine="openpyxl") as w:
        SUM.to_excel(w, sheet_name="summary", index=False)
        Tout.to_excel(w, sheet_name="all_trades", index=False)
        TT.to_excel(w, sheet_name="return_by_trigger_time", index=False)
        yr.to_excel(w, sheet_name="per_year", index=False)
        if len(CMP):
            CMP.to_excel(w, sheet_name="capped_vs_uncapped", index=False)
    Tout.to_csv(OUTDIR / "all_trades.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 96 + "\nMORNING VOLUME-SPIKE — CAPPED +4% to +7% BAND (exit 15:20; net 0.15%)\n" + "=" * 96)
    print(SUM.to_string(index=False))
    print("\n--- CAPPED vs UNCAPPED (and the removed >+7% trades) ---")
    print(CMP.to_string(index=False) if len(CMP) else "  (uncapped run not found)")
    print("\n--- RETURN BY TRIGGER TIME (capped) ---"); print(TT.to_string(index=False))
    print("\n--- PER YEAR (net, capped) ---"); print(yr.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
