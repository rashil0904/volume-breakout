# -*- coding: utf-8 -*-
"""banknifty_btst_close_direction.py — BankNifty Daily Close-Direction BTST, FUTURES-referenced, MONTHLY cycle.
Entry 15:15 daily: direction = current-month FUTURES close(15:15) vs futures day-OPEN. RED(fut<open): +2 ATM
PE / -1 (ATM-500) PE ; GREEN: +2 ATM CE / -1 (ATM+500) CE. ATM=round(fut_price/100)*100 (BankNifty step 100).
Never DTE-0: contract cycle E = nearest MONTHLY expiry STRICTLY AFTER entry day (bisect_right on the derived
22-monthly list) -> automatically next-month on an expiry day itself, current-month otherwise. Both futures
AND options are sourced from the SAME cycle E. Exit 09:17 next trading day, OPTION-LEG fill = that minute's
OPEN (futures not touched at exit - overnight hold, same E). Entry option fill = CLOSE at 15:15.

CONFIRMED DATA GAPS (skip-and-log, NOT approximated, per explicit user decision):
  GAP1: March-2025 BankNifty futures contract has ZERO data (confirmed permanent Upstox archive gap, verified
        across all timeframes). Any entry day whose cycle E == '2025-03-27' is skipped (~19-20 trading days,
        2025-02-27..2025-03-26 inclusive - Feb-27 itself needs March futures via its own next-month fallback).
  GAP2: Options pull stops at the July-2026 monthly expiry (2026-07-28); no Aug-2026 options chain exists, so
        entry days needing E beyond 2026-07-28 (i.e. D >= 2026-07-28) are skipped - naturally bounds the window.
GROSS OPTION PREMIUM POINTS (2xlong-1xshort), no lot-size multiplier applied.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

FUTDIR = rb.BASE / "data" / "futures_intraday_full" / "BANKNIFTY"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "BANKNIFTY"
OUTDIR = rb.RESULTS / "banknifty_btst_close_direction"; OUTDIR.mkdir(parents=True, exist_ok=True)
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]


def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


def drawdown_episodes(equity, times):
    peak = np.maximum.accumulate(equity); dd = equity - peak; eps = []; in_dd = False; s = t = 0; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9: in_dd = True; s = k - 1; t = k; tv = dd[k]
        else:
            if dd[k] < tv: tv = dd[k]; t = k
            if dd[k] >= -1e-9: eps.append((s, t, k, tv)); in_dd = False
    if in_dd: eps.append((s, t, len(equity) - 1, tv))
    rows = []
    for (s, t, r, v) in eps:
        rec = (r != len(equity) - 1) or (equity[r] >= peak[s] - 1e-9)
        rows.append({"peak_date": pd.Timestamp(times[s]).date(), "trough_date": pd.Timestamp(times[t]).date(),
                     "recovery_date": (pd.Timestamp(times[r]).date() if rec else "NOT RECOVERED"), "drawdown_points": round(abs(v), 1),
                     "days_peak_to_trough": int((pd.Timestamp(times[t]) - pd.Timestamp(times[s])).days),
                     "days_peak_to_recovery": int((pd.Timestamp(times[r]) - pd.Timestamp(times[s])).days)})
    return pd.DataFrame(rows)
ENTRY_MOD = 15 * 60 + 15; EXIT_MOD = 9 * 60 + 17; STEP = 100; OFFSET = 500
FLOOR = pd.Timestamp("2024-10-01")
MONTHLY = ["2024-10-30", "2024-11-27", "2024-12-24", "2025-01-30", "2025-02-27", "2025-03-27", "2025-04-24",
           "2025-05-29", "2025-06-26", "2025-07-31", "2025-08-28", "2025-09-30", "2025-10-28", "2025-11-25",
           "2025-12-30", "2026-01-27", "2026-02-24", "2026-03-30", "2026-04-28", "2026-05-26", "2026-06-30", "2026-07-28"]
EXPSET = set(MONTHLY)
GAP1_EXPIRY = "2025-03-27"   # cycle whose FUTURES contract has zero data
GAP3_EXPIRY = "2024-12-24"   # cycle whose OPTIONS have a total blackout in the near-ATM band 49600-56900 (both
                             # CE&PE; verified live against the API, not a pull bug) - covers every strike this
                             # strategy could ever need for that cycle (BankNifty traded ~50800-54300 throughout)


def futures_day(exp):
    """load the single futures parquet for expiry `exp`; returns df or None if missing (GAP1)."""
    edir = FUTDIR / exp.replace("-", "")
    fs = glob.glob(str(edir / "*.parquet"))
    if not fs: return None
    d = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"])
    return d


def opt_series(exp, strike, ot):
    edir = OPTDIR / exp.replace("-", "")
    fs = glob.glob(str(edir / f"BANKNIFTY_{int(strike)}_{ot}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]); return o.set_index("timestamp").sort_index()


def main():
    d = pd.read_parquet(DAILY, columns=["date"]); tdays = sorted(d["date"].dt.date.unique())
    tdays = [t for t in tdays if t >= FLOOR.date()]

    fut_cache = {}   # exp -> df (or None if genuinely absent)
    def get_fut(exp):
        if exp not in fut_cache:
            fut_cache[exp] = futures_day(exp)
        return fut_cache[exp]

    trades = []; gap_log = []; skipped_other = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        j = bisect.bisect_right(MONTHLY, str(D))
        if j >= len(MONTHLY):
            gap_log.append({"date": D, "reason": "GAP2: no monthly cycle available beyond 2026-07-28 (no Aug-2026 options)"}); continue
        E = MONTHLY[j]; is_exp = str(D) in EXPSET
        if E == GAP1_EXPIRY:
            gap_log.append({"date": D, "reason": f"GAP1: cycle={E} needs March-2025 futures (confirmed missing)"}); continue
        if E == GAP3_EXPIRY:
            gap_log.append({"date": D, "reason": f"GAP3: cycle={E} options blackout in near-ATM band 49600-56900 (confirmed, verified live)"}); continue

        FT = get_fut(E)
        if FT is None:
            gap_log.append({"date": D, "reason": f"futures data missing for cycle {E} (unexpected)"}); continue
        fday = FT[FT["timestamp"].dt.normalize() == pd.Timestamp(D)]
        if fday.empty:
            skipped_other += 1; continue
        dopen = float(fday["open"].iloc[0])
        f1515 = fday[fday["timestamp"].dt.hour * 60 + fday["timestamp"].dt.minute == ENTRY_MOD]
        if f1515.empty:
            skipped_other += 1; continue
        fprice = float(f1515["close"].iloc[-1]); atm = int(round(fprice / STEP) * STEP)
        direction = "RED" if fprice < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        long_K = atm; short_K = atm - OFFSET if direction == "RED" else atm + OFFSET
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=17)

        Lo = opt_series(E, long_K, ot); So = opt_series(E, short_K, ot)
        if Lo is None or So is None:
            gap_log.append({"date": D, "reason": f"missing option series {long_K}/{short_K} {ot} cycle {E}"}); continue
        Le = Lo["close"].asof(t_en); Se = So["close"].asof(t_en)
        Lx = Lo["open"].reindex([t_ex], method="ffill").iloc[0]; Sx = So["open"].reindex([t_ex], method="ffill").iloc[0]
        if any(pd.isna(v) for v in (Le, Se, Lx, Sx)):
            gap_log.append({"date": D, "reason": f"nan option price {long_K}/{short_K} {ot} cycle {E}"}); continue

        entry_cost = 2 * Le - Se; exit_val = 2 * Lx - Sx; pnl = exit_val - entry_cost
        trades.append({"entry_date": D, "direction": direction, "contract_cycle": "NEXT-month (expiry-day switch)" if is_exp else "current-month",
                       "expiry_used": E, "is_expiry_day": is_exp, "futures_ref_price": round(fprice, 2), "futures_day_open": round(dopen, 2),
                       "ATM": long_K, "long_leg": f"2x {long_K} {ot}", "short_leg": f"1x {short_K} {ot}",
                       "long_entry": round(float(Le), 2), "short_entry": round(float(Se), 2), "entry_cost_debit": round(float(entry_cost), 2),
                       "exit_date": Dn, "exit_time": "09:17", "long_exit": round(float(Lx), 2), "short_exit": round(float(Sx), 2),
                       "exit_value": round(float(exit_val), 2), "pnl_points": round(float(pnl), 2)})

    T = pd.DataFrame(trades); G = pd.DataFrame(gap_log)
    # India VIX at entry (15:15) — informational segregation only, no filtering
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vser = vx.assign(ts=vts)[["ts", "close"]].dropna().sort_values("ts").set_index("ts")["close"]
    ent_ts = pd.to_datetime(T["entry_date"].astype(str)) + pd.Timedelta(hours=15, minutes=15)
    T["entry_vix"] = [round(float(vser.asof(t)), 2) if pd.notna(vser.asof(t)) else np.nan for t in ent_ts]
    T["vix_bucket"] = T["entry_vix"].map(vbucket)

    win = round((T.pnl_points > 0).mean() * 100, 1); tot = round(T.pnl_points.sum(), 1)
    red = T[T.direction == "RED"]; grn = T[T.direction == "GREEN"]; nrm = T[~T.is_expiry_day]; exp = T[T.is_expiry_day]

    def blk(df, lbl):
        return {"group": lbl, "trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0}
    BY = pd.DataFrame([blk(T, "ALL"), blk(red, "RED"), blk(grn, "GREEN"), blk(nrm, "normal-day (current-month)"), blk(exp, "expiry-day (next-month)")])

    Tc = T.sort_values("entry_date").reset_index(drop=True); cum = Tc["pnl_points"].cumsum().values
    equity = np.concatenate([[0.0], cum]); dtimes = np.concatenate([[pd.Timestamp(Tc.entry_date.iloc[0])], pd.to_datetime(Tc.entry_date).values])
    DE = drawdown_episodes(equity, dtimes)
    maxdd = DE.drawdown_points.max() if len(DE) else 0.0; avgdd = round(DE.drawdown_points.mean(), 1) if len(DE) else 0.0
    maxddp = int(DE.days_peak_to_recovery.max()) if len(DE) else 0; avgddp = round(DE.days_peak_to_recovery.mean(), 1) if len(DE) else 0

    present = [b for b in VIX_BUCKETS if (T.vix_bucket == b).any()]
    VIX = pd.DataFrame([{"vix_bucket": b, "trades": int((T.vix_bucket == b).sum()),
                        "win_%": round((T[T.vix_bucket == b].pnl_points > 0).mean() * 100, 1),
                        "total_pnl": round(T[T.vix_bucket == b].pnl_points.sum(), 1),
                        "avg_pnl": round(T[T.vix_bucket == b].pnl_points.mean(), 2)} for b in present])

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "BankNifty Close-Direction BTST — FUTURES-referenced, MONTHLY cycle (15:15 in, 09:17-next-day out)"},
        {"metric": "Structure", "value": f"RED: +2 ATM PE / -1 (ATM-{OFFSET}) PE ; GREEN: +2 ATM CE / -1 (ATM+{OFFSET}) CE ; 1 trade/day"},
        {"metric": "Reference", "value": "current/next-MONTH BankNifty FUTURES (not spot) for direction+ATM; ATM step 100"},
        {"metric": "Never DTE-0", "value": "cycle E = nearest monthly expiry STRICTLY after entry day; on expiry day itself, uses NEXT month for both futures and options"},
        {"metric": "Fills", "value": "entry = option CLOSE @15:15 ; exit = next-day 09:17 option OPEN"},
        {"metric": "P&L unit", "value": "GROSS option premium points (2xlong-1xshort). No lot-size multiplier."},
        {"metric": "GAP1 (skipped, confirmed)", "value": f"March-2025 futures missing -> cycle=2025-03-27 entry days skipped ({int((G.reason.str.startswith('GAP1')).sum()) if len(G) else 0} days)"},
        {"metric": "GAP2 (skipped, confirmed)", "value": f"no Aug-2026 options -> entries on/after 2026-07-28 skipped ({int((G.reason.str.startswith('GAP2')).sum()) if len(G) else 0} days)"},
        {"metric": "GAP3 (skipped, confirmed)", "value": f"Dec-2024 options blackout, near-ATM strikes 49600-56900 (both CE/PE) entirely missing from Upstox archive, verified live -> ENTIRE Dec-2024 cycle skipped ({int((G.reason.str.startswith('GAP3')).sum()) if len(G) else 0} days, 0 trades possible that month)"},
        {"metric": "Window", "value": f"{T.entry_date.min()} .. {T.exit_date.max()}"},
        {"metric": "Total trades", "value": len(T)}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (points)", "value": tot}, {"metric": "Avg P&L / trade", "value": round(T.pnl_points.mean(), 2)},
        {"metric": "Median P&L / trade", "value": round(T.pnl_points.median(), 2)},
        {"metric": "RED / GREEN trades", "value": f"{len(red)} / {len(grn)}"},
        {"metric": "RED avg / GREEN avg P&L", "value": f"{round(red.pnl_points.mean(),2)} / {round(grn.pnl_points.mean(),2)}"},
        {"metric": "Normal-day trades / avg P&L", "value": f"{len(nrm)} / {round(nrm.pnl_points.mean(),2)}"},
        {"metric": "Expiry-day (next-month) trades / avg P&L", "value": f"{len(exp)} / {round(exp.pnl_points.mean(),2) if len(exp) else 0}"},
        {"metric": "Other skipped (missing minute/series, unflagged elsewhere)", "value": skipped_other},
        {"metric": "Total gap/skip rows logged", "value": len(G)},
        {"metric": "", "value": ""},
        {"metric": "--- DRAWDOWN (realized cumulative equity, premium points) ---", "value": ""},
        {"metric": "Number of drawdown episodes", "value": len(DE)},
        {"metric": "Average drawdown (points)", "value": avgdd},
        {"metric": "Maximum drawdown (points)", "value": round(maxdd, 1)},
        {"metric": "Average drawdown period (days, peak->recovery)", "value": avgddp},
        {"metric": "Maximum drawdown period (days, peak->recovery)", "value": maxddp},
        {"metric": "Return / Max-DD", "value": round(tot / maxdd, 2) if maxdd else "-"},
    ])
    with pd.ExcelWriter(OUTDIR / "banknifty_btst_close_direction.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False); BY.to_excel(w, sheet_name="By_Group", index=False)
        VIX.to_excel(w, sheet_name="VIX_Segregation", index=False)
        DE.sort_values("drawdown_points", ascending=False).to_excel(w, sheet_name="Drawdowns", index=False)
        T.to_excel(w, sheet_name="Trades", index=False); (G if len(G) else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="Gap_Log", index=False)
    T.to_csv(OUTDIR / "banknifty_btst_close_direction_trades.csv", index=False)

    pd.set_option("display.width", 230)
    print("=" * 96 + "\nBANKNIFTY DAILY CLOSE-DIRECTION BTST (futures-referenced, monthly cycle)\n" + "=" * 96)
    print(f"window {T.entry_date.min()} .. {T.exit_date.max()} | trades {len(T)} | gap-skipped {len(G)} | other-skipped {skipped_other}")
    print(f"win {win}% | TOTAL {tot:,} pts | avg {round(T.pnl_points.mean(),2)} | median {round(T.pnl_points.median(),2)}")
    print(f"DD: episodes {len(DE)} | avg {avgdd} | MAX {round(maxdd,1)} | avg period {avgddp}d | max period {maxddp}d | ret/maxDD {round(tot/maxdd,2) if maxdd else '-'}")
    print("\n--- BY GROUP ---"); print(BY.to_string(index=False))
    print("\n--- VIX SEGREGATION (informational, entry @15:15) ---"); print(VIX.to_string(index=False))
    if len(G):
        print("\n--- GAP LOG reasons ---"); print(G["reason"].str.split(":").str[0].value_counts().to_string())
    print("\nfirst 8 trades:")
    print(T.head(8)[["entry_date", "direction", "contract_cycle", "ATM", "entry_cost_debit", "exit_date", "exit_value", "pnl_points"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
