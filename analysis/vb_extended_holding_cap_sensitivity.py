# -*- coding: utf-8 -*-
"""vb_extended_holding_cap_sensitivity.py — sensitivity sweep on the extended-holding BB overlay
(vb_extended_holding_bb_overlay.py): re-evaluates BOTH variants (V1: SL=BB(36,2.5) basis/mean;
V2: SL=BB(36,1.5) upper) for every max-hold-day cap from 2 to 20 (step 1), to see how performance
degrades/improves as the forced-exit cap shortens -- directly addresses the flag that V1's headline
number leans heavily on the 20-day cap rather than genuine SL triggers. Read-only diagnostic, additive,
does not alter the baseline or the existing overlay files.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
MD = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "extended_holding_bb_overlay"; OUTDIR.mkdir(parents=True, exist_ok=True)

BB_LEN = 36; BB_STD_QUALIFY = 2.5; BB_STD_V2 = 1.5
MAX_WINDOW = 20   # widest cap tested
R023, SR = 0.0023, 0.0010
BASE_POOL = 500_000
CAPS = list(range(2, 21))


def daily_close_series(sym):
    fn = MD / f"{sym}.parquet"
    if not fn.exists():
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    daily = df.groupby(ts.dt.normalize())["close"].last().sort_index()
    daily.index.name = "date"
    return daily


def bollinger(daily, length, nstd):
    basis = daily.rolling(length).mean()
    std = daily.rolling(length).std(ddof=1)
    return basis, basis + nstd * std


def main():
    T = pd.read_excel(SRC, sheet_name="all_trades")
    T["entry_date"] = pd.to_datetime(T["entry_date"])
    symbols = sorted(T["symbol"].unique())
    print(f"total baseline trades: {len(T)} | symbols: {len(symbols)}", flush=True)

    trade_paths = []   # one row per qualifying trade, with day-by-day close/trigger arrays

    for si, sym in enumerate(symbols, 1):
        daily = daily_close_series(sym)
        if daily is None or len(daily) < BB_LEN:
            continue
        basis25, upper25 = bollinger(daily, BB_LEN, BB_STD_QUALIFY)
        _, upper15 = bollinger(daily, BB_LEN, BB_STD_V2)
        closes_arr = daily.values

        for _, r in T[T.symbol == sym].iterrows():
            ed = r["entry_date"].normalize()
            if ed not in daily.index or pd.isna(basis25.loc[ed]):
                continue
            if not (closes_arr[daily.index.get_loc(ed)] > upper25.loc[ed]):
                continue   # not qualifying

            i0 = daily.index.get_loc(ed)
            fwd_idx = list(range(i0 + 1, min(i0 + 1 + MAX_WINDOW, len(daily))))
            if not fwd_idx:
                continue
            shares = float(r["shares"]); avg = float(r["avg_entry"]); cap = float(r["capital_deployed"])

            path = {"symbol": sym, "entry_date": ed, "shares": shares, "avg": avg, "cap": cap,
                    "n_fwd_available": len(fwd_idx)}
            v1_trig = []; v2_trig = []; close_by_day = []
            for k, i in enumerate(fwd_idx, start=1):
                d = daily.index[i]; c = closes_arr[i]
                close_by_day.append(c)
                b25 = basis25.loc[d]; u15 = upper15.loc[d]
                v1_trig.append(bool(c < b25) if not pd.isna(b25) else False)
                v2_trig.append(bool(c < u15) if not pd.isna(u15) else False)
            path["close_by_day"] = close_by_day; path["v1_trig"] = v1_trig; path["v2_trig"] = v2_trig
            trade_paths.append(path)

        if si % 200 == 0:
            print(f"  {si}/{len(symbols)} symbols done, {len(trade_paths)} qualifying trades so far", flush=True)

    n_qual = len(trade_paths)
    print(f"\ntotal qualifying trades (entry-day close > BB(36,2.5) upper): {n_qual}", flush=True)

    def exit_for_cap(path, trig_list, cap):
        avail = path["n_fwd_available"]
        upto = min(cap, avail)
        trig_days = [k for k in range(1, upto + 1) if trig_list[k - 1]]
        if trig_days:
            day = trig_days[0]; reason = "sl_triggered"
        elif avail < cap:
            day = avail; reason = "open_at_data_end"
        else:
            day = cap; reason = "max_hold_cap"
        exit_price = path["close_by_day"][day - 1]
        pnl = path["shares"] * (exit_price - path["avg"])
        return day, reason, pnl

    rows = []
    for cap in CAPS:
        for variant, trig_key in [("V1", "v1_trig"), ("V2", "v2_trig")]:
            days = []; pnls = []; reasons = []
            for p in trade_paths:
                d, rsn, pnl = exit_for_cap(p, p[trig_key], cap)
                days.append(d); pnls.append(pnl); reasons.append(rsn)
            pnls = np.array(pnls); days = np.array(days)
            n_capped = sum(1 for x in reasons if x == "max_hold_cap")
            n_sl = sum(1 for x in reasons if x == "sl_triggered")
            n_end = sum(1 for x in reasons if x == "open_at_data_end")
            rows.append({
                "variant": variant, "cap_days": cap, "n_trades": n_qual,
                "total_pnl": round(pnls.sum(), 1), "avg_pnl": round(pnls.mean(), 1),
                "win_rate_pct": round((pnls > 0).mean() * 100, 1),
                "fixedbase_pct": round(pnls.sum() / BASE_POOL * 100, 2),
                "avg_hold_days": round(days.mean(), 2),
                "pct_sl_triggered": round(n_sl / n_qual * 100, 1),
                "pct_hit_cap": round(n_capped / n_qual * 100, 1),
                "pct_open_at_data_end": round(n_end / n_qual * 100, 1),
            })

    RES = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n=== VARIANT 1 (SL = basis/mean) across cap 2..20 ===")
    print(RES[RES.variant == "V1"].drop(columns="variant").to_string(index=False))
    print("\n=== VARIANT 2 (SL = BB(36,1.5) upper) across cap 2..20 ===")
    print(RES[RES.variant == "V2"].drop(columns="variant").to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "cap_sensitivity_2to20.xlsx", engine="openpyxl") as w:
        RES.to_excel(w, sheet_name="Cap_Sensitivity", index=False)
        RES.pivot(index="cap_days", columns="variant", values="total_pnl").to_excel(w, sheet_name="Total_PnL_by_Cap")
        RES.pivot(index="cap_days", columns="variant", values="win_rate_pct").to_excel(w, sheet_name="WinRate_by_Cap")
    print(f"\nSaved -> {OUTDIR}/cap_sensitivity_2to20.xlsx")


if __name__ == "__main__":
    main()
