# -*- coding: utf-8 -*-
"""vb_extended_holding_bb_overlay.py — ADDITIVE overlay on the locked baseline Volume-Breakout strategy.
Does NOT alter baseline_and_cross_final.py or its output file. Reads the existing all_trades (sizing,
shares, avg_entry, capital_deployed already computed there) and, for trades that QUALIFY (stock's daily
close on entry day is above its own daily BB(36, 2.5) upper band), replaces the standard next-day exit +
double-down-short with a multi-day hold, exited on the first subsequent trading day the daily close falls
below a stop-loss level. Two independent SL variants are tested (V1: BB(36,2.5) basis/mean; V2: BB(36,1.5)
upper band), each capped at a 20-trading-day maximum hold (confirmed with user) if the SL never triggers.
Non-qualifying trades are entirely untouched (not reproduced here -- comparison against baseline pulls
them from the original all_trades unchanged).

BB CONVENTION (flagged): pandas rolling(36).mean()/.std(ddof=1) on daily closes, window ending on (and
including) the day being evaluated -- i.e. "today's" BB uses today's own close plus the prior 35 closes.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
MD = rb.BASE / "master_data"
OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "extended_holding_bb_overlay"; OUTDIR.mkdir(parents=True, exist_ok=True)

BB_LEN = 36
BB_STD_QUALIFY = 2.5
BB_STD_V2 = 1.5
MAX_HOLD_DAYS = 20          # confirmed with user
R023, R038, SR = 0.0023, 0.0038, 0.0010   # same cost constants as baseline (long round-trip only, no short here)


def daily_close_series(sym):
    fn = MD / f"{sym}.parquet"
    if not fn.exists():
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = ts.dt.normalize()
    daily = df.groupby(d)["close"].last().sort_index()
    daily.index.name = "date"
    return daily


def bollinger(daily, length, nstd):
    basis = daily.rolling(length).mean()
    std = daily.rolling(length).std(ddof=1)
    upper = basis + nstd * std
    lower = basis - nstd * std
    return basis, upper, lower


def main():
    T = pd.read_excel(SRC, sheet_name="all_trades")
    T["entry_date"] = pd.to_datetime(T["entry_date"]); T["exit_date"] = pd.to_datetime(T["exit_date"])
    print(f"total baseline trades: {len(T)}", flush=True)

    symbols = sorted(T["symbol"].unique())
    print(f"unique symbols: {len(symbols)}", flush=True)

    qual_rows = []          # qualification result per trade
    v1_rows = []; v2_rows = []
    insuff_hist = []
    missing_data = []

    for si, sym in enumerate(symbols, 1):
        daily = daily_close_series(sym)
        if daily is None or len(daily) < BB_LEN:
            for _, r in T[T.symbol == sym].iterrows():
                missing_data.append((sym, r["entry_date"].date(), "no/short daily history"))
            continue
        basis25, upper25, _ = bollinger(daily, BB_LEN, BB_STD_QUALIFY)
        basis15, upper15, _ = bollinger(daily, BB_LEN, BB_STD_V2)
        dates_arr = daily.index.values
        closes_arr = daily.values

        for _, r in T[T.symbol == sym].iterrows():
            ed = r["entry_date"].normalize()
            if ed not in daily.index:
                missing_data.append((sym, ed.date(), "entry_date not in daily series")); continue
            if pd.isna(basis25.loc[ed]):
                insuff_hist.append((sym, ed.date(), "insufficient prior daily bars for BB(36)")); continue

            qualifies = bool(closes_arr[daily.index.get_loc(ed)] > upper25.loc[ed])
            qual_rows.append({"symbol": sym, "entry_date": ed, "qualifies": qualifies,
                               "entry_day_close": float(daily.loc[ed]), "entry_day_upper25": float(upper25.loc[ed])})
            if not qualifies:
                continue

            # forward walk starting at exit_date (the existing next trading day)
            i0 = daily.index.get_loc(ed)
            fwd_idx = list(range(i0 + 1, len(daily)))
            if not fwd_idx:
                missing_data.append((sym, ed.date(), "no forward daily data after entry")); continue

            shares = float(r["shares"]); avg = float(r["avg_entry"]); cap = float(r["capital_deployed"])

            for variant, basis_s, upper_s, out in [(1, basis25, None, v1_rows), (2, None, upper15, v2_rows)]:
                exit_i = None; exit_reason = "sl_triggered"
                for k, i in enumerate(fwd_idx[:MAX_HOLD_DAYS], start=1):
                    d = daily.index[i]; c = closes_arr[i]
                    sl_level = basis_s.loc[d] if variant == 1 else upper_s.loc[d]
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
                long_pnl = shares * (exit_price - avg)
                gross_pnl = long_pnl
                netA_pnl = gross_pnl - R023 * cap
                netB_pnl = gross_pnl - R038 * cap
                out.append({"symbol": sym, "entry_date": ed, "exit_date": exit_date, "exit_reason": exit_reason,
                            "shares": shares, "avg_entry": avg, "capital_deployed": cap, "exit_price": exit_price,
                            "hold_days": int(hold_days), "gross_pnl": gross_pnl, "netA_pnl": netA_pnl, "netB_pnl": netB_pnl})

        if si % 200 == 0:
            print(f"  {si}/{len(symbols)} symbols done", flush=True)

    QUAL = pd.DataFrame(qual_rows)
    V1 = pd.DataFrame(v1_rows); V2 = pd.DataFrame(v2_rows)
    n_qual = int(QUAL["qualifies"].sum()) if len(QUAL) else 0
    print(f"\nqualification check resolved for {len(QUAL)} trades ({len(insuff_hist)} skipped: insufficient BB history; "
          f"{len(missing_data)} skipped: missing data)", flush=True)
    print(f"QUALIFYING trades (entry-day close > BB(36,2.5) upper): {n_qual} "
          f"({n_qual/len(QUAL)*100:.2f}% of {len(QUAL)} checkable trades)")

    # ---- baseline comparison for the SAME qualifying trades ----
    qual_keys = set(zip(QUAL.loc[QUAL.qualifies, "symbol"], QUAL.loc[QUAL.qualifies, "entry_date"]))
    T_idx = T.set_index(["symbol", "entry_date"])
    base_qual = T_idx.loc[T_idx.index.isin(qual_keys)].reset_index()

    def summarize(df, pnl_col, label):
        if len(df) == 0:
            print(f"{label}: no trades"); return {}
        n = len(df); win = (df[pnl_col] > 0).mean() * 100
        total = df[pnl_col].sum(); avg = df[pnl_col].mean()
        print(f"{label:35s} | n={n:5d} | win_rate={win:5.1f}% | total_pnl={total:>13,.1f} | avg_pnl/trade={avg:>9,.1f}")
        return {"n_trades": n, "win_rate_pct": round(win, 1), "total_pnl": round(total, 1), "avg_pnl": round(avg, 1)}

    print("\n=== BASELINE (standard exit) on the SAME qualifying trades ===")
    base_summary = summarize(base_qual, "gross_pnl", "BASELINE gross (long+short combined)")

    print("\n=== VARIANT 1 (SL = BB(36,2.5) basis/mean) ===")
    v1_summary = summarize(V1, "gross_pnl", "V1 gross")
    if len(V1):
        print(f"  avg hold days = {V1.hold_days.mean():.2f} | max hold days = {V1.hold_days.max()}")
        print(f"  exit reason counts:\n{V1.exit_reason.value_counts().to_string()}")
        print(f"  holding-period distribution:\n{V1.hold_days.describe().to_string()}")

    print("\n=== VARIANT 2 (SL = BB(36,1.5) upper) ===")
    v2_summary = summarize(V2, "gross_pnl", "V2 gross")
    if len(V2):
        print(f"  avg hold days = {V2.hold_days.mean():.2f} | max hold days = {V2.hold_days.max()}")
        print(f"  exit reason counts:\n{V2.exit_reason.value_counts().to_string()}")
        print(f"  holding-period distribution:\n{V2.hold_days.describe().to_string()}")

    # ---- out-of-sample: chronological half split of qualifying trades ----
    print("\n=== OUT-OF-SAMPLE (chronological half split of qualifying entry dates) ===")
    if n_qual >= 30:
        cutoff = QUAL.loc[QUAL.qualifies, "entry_date"].median()
        print(f"split cutoff (median qualifying entry_date): {cutoff.date()}")
        for out, label in [(V1, "V1"), (V2, "V2")]:
            if len(out) == 0:
                continue
            early = out[out.entry_date <= cutoff]; late = out[out.entry_date > cutoff]
            print(f"\n-- {label} EARLY half --"); summarize(early, "gross_pnl", f"{label} early")
            print(f"-- {label} LATE half --"); summarize(late, "gross_pnl", f"{label} late")
    else:
        print(f"FLAG: only {n_qual} qualifying trades -- too small for a meaningful out-of-sample split (recommend n>=30 per half); skipping OOS split.")

    if insuff_hist:
        print(f"\nFLAG: {len(insuff_hist)} trades skipped for insufficient BB(36) history (mostly earliest trades in "
              f"2022, since master_data daily history starts 2022-01-03 -- fewer than 36 prior trading days existed yet).")
    if missing_data:
        print(f"FLAG: {len(missing_data)} trades skipped for missing/short master_data.")

    with pd.ExcelWriter(OUTDIR / "extended_holding_bb_overlay.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "total_baseline_trades", "value": len(T)},
            {"metric": "trades_checkable_for_qualification", "value": len(QUAL)},
            {"metric": "n_qualifying", "value": n_qual},
            {"metric": "pct_qualifying_of_checkable", "value": round(n_qual/len(QUAL)*100, 2) if len(QUAL) else 0},
            {"metric": "pct_qualifying_of_all_baseline", "value": round(n_qual/len(T)*100, 2)},
            {"metric": "max_hold_days_cap", "value": MAX_HOLD_DAYS},
            {"metric": "--- BASELINE (same qual. trades) ---", "value": ""},
            *[{"metric": f"baseline_{k}", "value": v} for k, v in base_summary.items()],
            {"metric": "--- VARIANT 1 (SL=basis) ---", "value": ""},
            *[{"metric": f"v1_{k}", "value": v} for k, v in v1_summary.items()],
            {"metric": "v1_avg_hold_days", "value": round(V1.hold_days.mean(), 2) if len(V1) else ""},
            {"metric": "v1_max_hold_days", "value": int(V1.hold_days.max()) if len(V1) else ""},
            {"metric": "--- VARIANT 2 (SL=BB1.5 upper) ---", "value": ""},
            *[{"metric": f"v2_{k}", "value": v} for k, v in v2_summary.items()],
            {"metric": "v2_avg_hold_days", "value": round(V2.hold_days.mean(), 2) if len(V2) else ""},
            {"metric": "v2_max_hold_days", "value": int(V2.hold_days.max()) if len(V2) else ""},
        ]).to_excel(w, sheet_name="Summary", index=False)
        QUAL.to_excel(w, sheet_name="Qualification_Check", index=False)
        base_qual.to_excel(w, sheet_name="Baseline_on_Qual_Trades", index=False)
        V1.to_excel(w, sheet_name="Variant1_Trades", index=False)
        V2.to_excel(w, sheet_name="Variant2_Trades", index=False)
        if V1.exit_reason.eq("max_hold_cap_20d").any() or (len(V1) and True):
            pd.concat([V1.exit_reason.value_counts().rename("V1"), V2.exit_reason.value_counts().rename("V2")], axis=1
                      ).reset_index().rename(columns={"index": "exit_reason"}).to_excel(w, sheet_name="Exit_Reason_Counts", index=False)
        if insuff_hist:
            pd.DataFrame(insuff_hist, columns=["symbol", "entry_date", "issue"]).to_excel(w, sheet_name="Insufficient_History", index=False)
        if missing_data:
            pd.DataFrame(missing_data, columns=["symbol", "entry_date", "issue"]).to_excel(w, sheet_name="Missing_Flagged", index=False)

    print(f"\nSaved -> {OUTDIR}/extended_holding_bb_overlay.xlsx")


if __name__ == "__main__":
    main()
