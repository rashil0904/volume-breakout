# -*- coding: utf-8 -*-
"""vb_extended_holding_v4_entrylow_sl.py — fourth SL variant: SL = a FIXED level set on the entry day
itself (that day's daily LOW), not a recomputed rolling indicator like V1/V2/V3. Exit triggers (close-basis,
same convention as the others) the first subsequent day the daily CLOSE falls below the entry day's LOW.
Same qualification (BB(36,2.5) upper on entry day), same 20-day max-hold cap, and the SAME no-overlap fix
already applied (same-symbol re-entries while a position is still open are dropped, per the correction
made to V1/V2/V3). Read-only, additive, does not touch the baseline or other locked files.
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


def daily_ohlc(sym):
    fn = MD / f"{sym}.parquet"
    if not fn.exists():
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "low", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = ts.dt.normalize()
    g = df.groupby(d)
    daily = pd.DataFrame({"low": g["low"].min(), "close": g["close"].last()}).sort_index()
    daily.index.name = "date"
    return daily


def main():
    T = pd.read_excel(SRC, sheet_name="all_trades")
    T["entry_date"] = pd.to_datetime(T["entry_date"])
    symbols = sorted(T["symbol"].unique())
    print(f"total baseline trades: {len(T)} | symbols: {len(symbols)}", flush=True)

    accepted = []; dropped_overlap = 0; n_qual_total = 0

    for si, sym in enumerate(symbols, 1):
        daily = daily_ohlc(sym)
        if daily is None or len(daily) < QUAL_LEN:
            continue
        closes = daily["close"]
        basis36 = closes.rolling(QUAL_LEN).mean()
        std36 = closes.rolling(QUAL_LEN).std(ddof=1)
        upper25 = basis36 + QUAL_STD * std36
        closes_arr = closes.values; lows_arr = daily["low"].values

        cand = []
        for _, r in T[T.symbol == sym].sort_values("entry_date").iterrows():
            ed = r["entry_date"].normalize()
            if ed not in daily.index or pd.isna(upper25.loc[ed]):
                continue
            if not (closes.loc[ed] > upper25.loc[ed]):
                continue
            entry_low = float(daily.loc[ed, "low"])
            cand.append((ed, float(r["shares"]), float(r["avg_entry"]), float(r["capital_deployed"]), entry_low))
        n_qual_total += len(cand)

        blocked_until = None
        for ed, shares, avg, cap, entry_low in cand:
            if blocked_until is not None and ed <= blocked_until:
                dropped_overlap += 1
                continue
            i0 = daily.index.get_loc(ed)
            fwd_idx = list(range(i0 + 1, len(daily)))
            if not fwd_idx:
                continue
            exit_i = None; exit_reason = "sl_triggered"
            for k, i in enumerate(fwd_idx[:MAX_HOLD_DAYS], start=1):
                if closes_arr[i] < entry_low:
                    exit_i = i; break
            if exit_i is None:
                if len(fwd_idx) >= MAX_HOLD_DAYS:
                    exit_i = fwd_idx[MAX_HOLD_DAYS - 1]; exit_reason = "max_hold_cap_20d"
                else:
                    exit_i = fwd_idx[-1]; exit_reason = "open_at_data_end"
            exit_date = daily.index[exit_i]; exit_price = float(closes_arr[exit_i])
            hold_days = exit_i - i0
            gross_pnl = shares * (exit_price - avg)
            blocked_until = exit_date
            accepted.append({"symbol": sym, "entry_date": ed, "exit_date": exit_date, "exit_reason": exit_reason,
                              "shares": shares, "avg_entry": avg, "capital_deployed": cap, "entry_day_low": entry_low,
                              "exit_price": exit_price, "hold_days": int(hold_days), "gross_pnl": gross_pnl,
                              "netA_pnl": gross_pnl - R023 * cap, "netB_pnl": gross_pnl - R038 * cap})

        if si % 200 == 0:
            print(f"  {si}/{len(symbols)} symbols done", flush=True)

    V4 = pd.DataFrame(accepted)
    print(f"\nqualifying-trade instances (before overlap fix): {n_qual_total}")
    print(f"V4 n_trades={len(V4)} (dropped {dropped_overlap} overlapping re-entries, {dropped_overlap/n_qual_total*100:.1f}%)")

    win = V4[V4.gross_pnl > 0]; lose = V4[V4.gross_pnl <= 0]
    print(f"\n=== VARIANT 4 (SL = entry day's LOW, fixed level, close-basis trigger) ===")
    print(f"total_pnl={V4.gross_pnl.sum():,.1f}  avg_pnl={V4.gross_pnl.mean():,.1f}  "
          f"win_rate={round((V4.gross_pnl>0).mean()*100,1)}%  fixedbase%={V4.gross_pnl.sum()/BASE_POOL*100:.2f}")
    print(f"avg_hold_days={V4.hold_days.mean():.2f}  median={V4.hold_days.median():.1f}  max={V4.hold_days.max()}")
    print(f"\nexit reason counts:\n{V4.exit_reason.value_counts().to_string()}")
    print(f"\nwinners: n={len(win)} avg={win.gross_pnl.mean():,.1f} median={win.gross_pnl.median():,.1f} "
          f"avg_hold={win.hold_days.mean():.2f} median_hold={win.hold_days.median():.1f}")
    print(f"losers : n={len(lose)} avg={lose.gross_pnl.mean():,.1f} median={lose.gross_pnl.median():,.1f} "
          f"avg_hold={lose.hold_days.mean():.2f} median_hold={lose.hold_days.median():.1f}")

    with pd.ExcelWriter(OUTDIR / "variant4_entrylow_sl.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "sl_definition", "value": "fixed level = entry day's daily LOW; exit on first close < that level"},
            {"metric": "n_qualifying_before_overlap_fix", "value": n_qual_total},
            {"metric": "n_dropped_overlap", "value": dropped_overlap},
            {"metric": "n_trades", "value": len(V4)},
            {"metric": "total_pnl", "value": round(V4.gross_pnl.sum(), 1)},
            {"metric": "avg_pnl", "value": round(V4.gross_pnl.mean(), 1)},
            {"metric": "win_rate_pct", "value": round((V4.gross_pnl > 0).mean() * 100, 1)},
            {"metric": "fixedbase_pct", "value": round(V4.gross_pnl.sum() / BASE_POOL * 100, 2)},
            {"metric": "avg_hold_days", "value": round(V4.hold_days.mean(), 2)},
            {"metric": "median_hold_days", "value": V4.hold_days.median()},
            {"metric": "max_hold_days", "value": int(V4.hold_days.max())},
            {"metric": "winners_avg_pnl", "value": round(win.gross_pnl.mean(), 1)},
            {"metric": "winners_median_pnl", "value": round(win.gross_pnl.median(), 1)},
            {"metric": "losers_avg_pnl", "value": round(lose.gross_pnl.mean(), 1)},
            {"metric": "losers_median_pnl", "value": round(lose.gross_pnl.median(), 1)},
        ]).to_excel(w, sheet_name="Summary", index=False)
        V4.to_excel(w, sheet_name="V4_no_overlap_trades", index=False)

    print(f"\nSaved -> {OUTDIR}/variant4_entrylow_sl.xlsx")


if __name__ == "__main__":
    main()
