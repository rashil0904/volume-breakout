# -*- coding: utf-8 -*-
"""vb_extended_holding_no_overlap.py — fixes the overlap gap found in the extended-holding BB overlay:
previously every qualifying trade was priced independently, even when a same-symbol qualifying re-entry
fired while an earlier extended-hold position on that symbol was still open (425/1904 = 22.3% of cases).
That silently double-counted the same underlying move and assumed capital that was still tied up.

FIX (per user instruction): process each symbol's qualifying candidates in chronological order and greedily
SKIP (drop entirely -- no position, no capital, not counted anywhere) any candidate whose entry_date falls
on/before the exit_date of that symbol's still-open extended-hold position, for THAT SAME VARIANT (each
variant holds for a different duration, so the blocking window differs by variant and is computed
independently for V1/V2/V3). Non-qualifying baseline trades are NOT touched, per the original "additive,
do not alter baseline for non-qualifying trades" instruction -- this fix only prunes overlaps WITHIN the
qualifying/extended-hold universe.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
MD = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "extended_holding_bb_overlay"; OUTDIR.mkdir(parents=True, exist_ok=True)

QUAL_LEN, QUAL_STD = 36, 2.5
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


def walk_exit(daily, closes_arr, i0, shares, avg, sl_series, max_days=MAX_HOLD_DAYS):
    fwd_idx = list(range(i0 + 1, len(daily)))
    if not fwd_idx:
        return None
    exit_i = None; exit_reason = "sl_triggered"
    for k, i in enumerate(fwd_idx[:max_days], start=1):
        d = daily.index[i]; c = closes_arr[i]
        sl_level = sl_series.loc[d]
        if pd.isna(sl_level):
            continue
        if c < sl_level:
            exit_i = i; break
    if exit_i is None:
        if len(fwd_idx) >= max_days:
            exit_i = fwd_idx[max_days - 1]; exit_reason = "max_hold_cap_20d"
        else:
            exit_i = fwd_idx[-1]; exit_reason = "open_at_data_end"
    exit_date = daily.index[exit_i]; exit_price = float(closes_arr[exit_i])
    hold_days = exit_i - i0
    gross_pnl = shares * (exit_price - avg)
    return {"exit_date": exit_date, "exit_price": exit_price, "exit_reason": exit_reason,
            "hold_days": int(hold_days), "gross_pnl": gross_pnl}


def main():
    T = pd.read_excel(SRC, sheet_name="all_trades")
    T["entry_date"] = pd.to_datetime(T["entry_date"])
    symbols = sorted(T["symbol"].unique())
    print(f"total baseline trades: {len(T)} | symbols: {len(symbols)}", flush=True)

    variants = {"V1": {"sl_len": 36, "sl_kind": "basis"}, "V2": {"sl_len": 36, "sl_kind": "upper15"},
                "V3": {"sl_len": 20, "sl_kind": "basis"}}
    accepted = {k: [] for k in variants}; dropped_overlap = {k: 0 for k in variants}
    n_qual_total = 0

    for si, sym in enumerate(symbols, 1):
        daily = daily_close_series(sym)
        if daily is None or len(daily) < QUAL_LEN:
            continue
        basis36 = daily.rolling(QUAL_LEN).mean()
        std36 = daily.rolling(QUAL_LEN).std(ddof=1)
        upper25 = basis36 + QUAL_STD * std36
        upper15 = basis36 + 1.5 * std36
        basis20 = daily.rolling(20).mean()
        sl_map = {"V1": basis36, "V2": upper15, "V3": basis20}
        closes_arr = daily.values

        cand = []
        for _, r in T[T.symbol == sym].sort_values("entry_date").iterrows():
            ed = r["entry_date"].normalize()
            if ed not in daily.index or pd.isna(upper25.loc[ed]):
                continue
            if not (closes_arr[daily.index.get_loc(ed)] > upper25.loc[ed]):
                continue
            cand.append((ed, float(r["shares"]), float(r["avg_entry"]), float(r["capital_deployed"])))
        n_qual_total += len(cand)

        for vname, sl_series in sl_map.items():
            blocked_until = None
            for ed, shares, avg, cap in cand:
                if blocked_until is not None and ed <= blocked_until:
                    dropped_overlap[vname] += 1
                    continue
                i0 = daily.index.get_loc(ed)
                res = walk_exit(daily, closes_arr, i0, shares, avg, sl_series, MAX_HOLD_DAYS)
                if res is None:
                    continue
                blocked_until = res["exit_date"]
                accepted[vname].append({"symbol": sym, "entry_date": ed, "shares": shares, "avg_entry": avg,
                                         "capital_deployed": cap, **res})

        if si % 200 == 0:
            print(f"  {si}/{len(symbols)} symbols done", flush=True)

    print(f"\ntotal qualifying-trade instances (before overlap fix): {n_qual_total}", flush=True)

    results = {}
    for vname in variants:
        df = pd.DataFrame(accepted[vname])
        df["netA_pnl"] = df["gross_pnl"] - R023 * df["capital_deployed"]
        df["netB_pnl"] = df["gross_pnl"] - R038 * df["capital_deployed"]
        results[vname] = df
        win = df[df.gross_pnl > 0]; lose = df[df.gross_pnl <= 0]
        print(f"\n=== {vname} AFTER overlap fix ===")
        print(f"n_trades={len(df)} (dropped {dropped_overlap[vname]} overlapping re-entries, "
              f"{dropped_overlap[vname]/n_qual_total*100:.1f}% of {n_qual_total})")
        print(f"total_pnl={df.gross_pnl.sum():,.1f}  avg_pnl={df.gross_pnl.mean():,.1f}  "
              f"win_rate={round((df.gross_pnl>0).mean()*100,1)}%  fixedbase%={df.gross_pnl.sum()/BASE_POOL*100:.2f}")
        print(f"avg_hold_days={df.hold_days.mean():.2f}  median={df.hold_days.median():.1f}  max={df.hold_days.max()}")
        print(f"winners: n={len(win)} avg={win.gross_pnl.mean():,.1f} median={win.gross_pnl.median():,.1f}")
        print(f"losers : n={len(lose)} avg={lose.gross_pnl.mean():,.1f} median={lose.gross_pnl.median():,.1f}")

    print("\n=== BEFORE vs AFTER overlap fix (totals) ===")
    before = {"V1": (1904, 3691981.0), "V2": (1904, 1495581.6), "V3": (1904, 3184922.4)}
    for vname in variants:
        b_n, b_pnl = before[vname]
        a_n, a_pnl = len(results[vname]), results[vname].gross_pnl.sum()
        print(f"{vname}: n {b_n} -> {a_n} ({a_n-b_n:+d})  |  total_pnl {b_pnl:,.1f} -> {a_pnl:,.1f} ({a_pnl-b_pnl:+,.1f})")

    with pd.ExcelWriter(OUTDIR / "extended_holding_no_overlap_fixed.xlsx", engine="openpyxl") as w:
        summary_rows = []
        for vname in variants:
            df = results[vname]; win = df[df.gross_pnl > 0]; lose = df[df.gross_pnl <= 0]
            b_n, b_pnl = before[vname]
            summary_rows.append({
                "variant": vname, "n_before_fix": b_n, "n_after_fix": len(df),
                "n_dropped_overlap": dropped_overlap[vname],
                "pct_dropped": round(dropped_overlap[vname]/n_qual_total*100, 1),
                "total_pnl_before": b_pnl, "total_pnl_after": round(df.gross_pnl.sum(), 1),
                "pnl_delta": round(df.gross_pnl.sum() - b_pnl, 1),
                "avg_pnl_after": round(df.gross_pnl.mean(), 1),
                "win_rate_after_pct": round((df.gross_pnl > 0).mean() * 100, 1),
                "fixedbase_pct_after": round(df.gross_pnl.sum() / BASE_POOL * 100, 2),
                "avg_hold_days_after": round(df.hold_days.mean(), 2),
                "winners_avg": round(win.gross_pnl.mean(), 1), "winners_median": round(win.gross_pnl.median(), 1),
                "losers_avg": round(lose.gross_pnl.mean(), 1), "losers_median": round(lose.gross_pnl.median(), 1),
            })
        pd.DataFrame(summary_rows).to_excel(w, sheet_name="Summary_Before_After", index=False)
        for vname in variants:
            results[vname].to_excel(w, sheet_name=f"{vname}_no_overlap_trades", index=False)

    print(f"\nSaved -> {OUTDIR}/extended_holding_no_overlap_fixed.xlsx")


if __name__ == "__main__":
    main()
