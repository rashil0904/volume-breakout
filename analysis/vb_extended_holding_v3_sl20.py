# -*- coding: utf-8 -*-
"""vb_extended_holding_v3_sl20.py — third SL variant for the extended-holding BB overlay. QUALIFICATION
unchanged (entry-day daily close > BB(36,2.5) upper band, same 1,904 qualifying trades as V1/V2). Only the
exit SL changes: V3 = basis (SMA) of a SHORTER length-20 rolling window (vs V1's length-36 basis), i.e. a
faster-reacting mean. Same 20-trading-day max hold cap as V1/V2 for direct comparability. Read-only,
additive, does not touch the baseline or the existing overlay files.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
MD = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "extended_holding_bb_overlay"; OUTDIR.mkdir(parents=True, exist_ok=True)

QUAL_LEN, QUAL_STD = 36, 2.5     # entry-day qualification check, unchanged
SL_LEN = 20                       # NEW: basis length for the SL (was 36 in V1)
MAX_HOLD_DAYS = 20
R023, R038, SR = 0.0023, 0.0038, 0.0010
BASE_POOL = 500_000


def daily_close_series(sym):
    fn = MD / f"{sym}.parquet"
    if not fn.exists():
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    daily = df.groupby(ts.dt.normalize())["close"].last().sort_index()
    daily.index.name = "date"
    return daily


def main():
    T = pd.read_excel(SRC, sheet_name="all_trades")
    T["entry_date"] = pd.to_datetime(T["entry_date"])
    symbols = sorted(T["symbol"].unique())
    print(f"total baseline trades: {len(T)} | symbols: {len(symbols)}", flush=True)

    v3_rows = []; n_qual = 0

    for si, sym in enumerate(symbols, 1):
        daily = daily_close_series(sym)
        if daily is None or len(daily) < QUAL_LEN:
            continue
        basis_qual = daily.rolling(QUAL_LEN).mean()
        std_qual = daily.rolling(QUAL_LEN).std(ddof=1)
        upper_qual = basis_qual + QUAL_STD * std_qual
        basis_sl = daily.rolling(SL_LEN).mean()
        closes_arr = daily.values

        for _, r in T[T.symbol == sym].iterrows():
            ed = r["entry_date"].normalize()
            if ed not in daily.index or pd.isna(upper_qual.loc[ed]):
                continue
            if not (closes_arr[daily.index.get_loc(ed)] > upper_qual.loc[ed]):
                continue
            n_qual += 1

            i0 = daily.index.get_loc(ed)
            fwd_idx = list(range(i0 + 1, len(daily)))
            if not fwd_idx:
                continue
            shares = float(r["shares"]); avg = float(r["avg_entry"]); cap = float(r["capital_deployed"])

            exit_i = None; exit_reason = "sl_triggered"
            for k, i in enumerate(fwd_idx[:MAX_HOLD_DAYS], start=1):
                d = daily.index[i]; c = closes_arr[i]
                sl_level = basis_sl.loc[d]
                if pd.isna(sl_level):
                    continue
                if c < sl_level:
                    exit_i = i; break
            if exit_i is None:
                if len(fwd_idx) >= MAX_HOLD_DAYS:
                    exit_i = fwd_idx[MAX_HOLD_DAYS - 1]; exit_reason = "max_hold_cap_20d"
                else:
                    exit_i = fwd_idx[-1]; exit_reason = "open_at_data_end"

            exit_date = daily.index[exit_i]; exit_price = float(closes_arr[exit_i])
            hold_days = exit_i - i0
            gross_pnl = shares * (exit_price - avg)
            v3_rows.append({"symbol": sym, "entry_date": ed, "exit_date": exit_date, "exit_reason": exit_reason,
                            "shares": shares, "avg_entry": avg, "capital_deployed": cap, "exit_price": exit_price,
                            "hold_days": int(hold_days), "gross_pnl": gross_pnl,
                            "netA_pnl": gross_pnl - R023 * cap, "netB_pnl": gross_pnl - R038 * cap})

        if si % 200 == 0:
            print(f"  {si}/{len(symbols)} symbols done", flush=True)

    V3 = pd.DataFrame(v3_rows)
    print(f"\nqualifying trades: {n_qual} (resolved: {len(V3)})")
    print(f"\n=== VARIANT 3 (SL = basis of BB(20, std irrelevant) -- 20-day SMA) ===")
    print(f"n={len(V3)} total_pnl={V3.gross_pnl.sum():,.1f} avg_pnl={V3.gross_pnl.mean():,.1f} "
          f"win_rate={round((V3.gross_pnl>0).mean()*100,1)}% fixedbase%={V3.gross_pnl.sum()/BASE_POOL*100:.2f}")
    print(f"avg hold days={V3.hold_days.mean():.2f} median={V3.hold_days.median():.1f} max={V3.hold_days.max()}")
    print(f"\nexit reason counts:\n{V3.exit_reason.value_counts().to_string()}")

    win = V3[V3.gross_pnl > 0]; lose = V3[V3.gross_pnl <= 0]
    print(f"\nwinners: n={len(win)} avg={win.gross_pnl.mean():,.1f} median={win.gross_pnl.median():,.1f} "
          f"avg_hold={win.hold_days.mean():.2f} median_hold={win.hold_days.median():.1f}")
    print(f"losers : n={len(lose)} avg={lose.gross_pnl.mean():,.1f} median={lose.gross_pnl.median():,.1f} "
          f"avg_hold={lose.hold_days.mean():.2f} median_hold={lose.hold_days.median():.1f}")

    with pd.ExcelWriter(OUTDIR / "variant3_sl20_basis.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "sl_definition", "value": "basis (SMA) of a 20-day rolling window, qualification unchanged (BB(36,2.5) upper on entry day)"},
            {"metric": "n_trades", "value": len(V3)},
            {"metric": "total_pnl", "value": round(V3.gross_pnl.sum(), 1)},
            {"metric": "avg_pnl", "value": round(V3.gross_pnl.mean(), 1)},
            {"metric": "win_rate_pct", "value": round((V3.gross_pnl > 0).mean() * 100, 1)},
            {"metric": "fixedbase_pct", "value": round(V3.gross_pnl.sum() / BASE_POOL * 100, 2)},
            {"metric": "avg_hold_days", "value": round(V3.hold_days.mean(), 2)},
            {"metric": "median_hold_days", "value": V3.hold_days.median()},
            {"metric": "max_hold_days", "value": int(V3.hold_days.max())},
            {"metric": "winners_avg_pnl", "value": round(win.gross_pnl.mean(), 1)},
            {"metric": "winners_median_pnl", "value": round(win.gross_pnl.median(), 1)},
            {"metric": "losers_avg_pnl", "value": round(lose.gross_pnl.mean(), 1)},
            {"metric": "losers_median_pnl", "value": round(lose.gross_pnl.median(), 1)},
        ]).to_excel(w, sheet_name="Summary", index=False)
        V3.to_excel(w, sheet_name="V3_All_Trades", index=False)

    print(f"\nSaved -> {OUTDIR}/variant3_sl20_basis.xlsx")


if __name__ == "__main__":
    main()
