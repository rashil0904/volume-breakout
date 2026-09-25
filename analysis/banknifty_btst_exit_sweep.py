# -*- coding: utf-8 -*-
"""banknifty_btst_exit_sweep.py — EXIT-TIME SWEEP (09:16-10:00, 1-min steps, 45 values) on the BankNifty
Daily Close-Direction BTST. Entry logic UNCHANGED from banknifty_btst_close_direction.py (futures-referenced
15:15 direction/ATM, monthly cycle, DTE-0 next-month switch, 500-pt offset). Only the exit time varies; exit
fill = that minute's option OPEN on the next trading day. Confirmed gaps (skip-and-log, same as base):
GAP1 March-2025 futures missing, GAP2 no Aug-2026 options, GAP3 Dec-2024 options near-ATM blackout.
Entry computed ONCE per day; exit price vectorized across all 45 minutes per leg (single file read/leg/day).
GROSS option premium points. OOS split 2025-08-29 (project convention).
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
ENTRY_MOD = 15 * 60 + 15; STEP = 100; OFFSET = 500; FLOOR = pd.Timestamp("2024-10-01")
EXIT_MODS = list(range(9 * 60 + 16, 10 * 60 + 1))          # 09:16..10:00 inclusive (45)
OOS_SPLIT = pd.Timestamp("2025-08-29")
MONTHLY = ["2024-10-30", "2024-11-27", "2024-12-24", "2025-01-30", "2025-02-27", "2025-03-27", "2025-04-24",
           "2025-05-29", "2025-06-26", "2025-07-31", "2025-08-28", "2025-09-30", "2025-10-28", "2025-11-25",
           "2025-12-30", "2026-01-27", "2026-02-24", "2026-03-30", "2026-04-28", "2026-05-26", "2026-06-30", "2026-07-28"]
EXPSET = set(MONTHLY); GAP1_EXPIRY = "2025-03-27"; GAP3_EXPIRY = "2024-12-24"


def futures_day(exp):
    fs = glob.glob(str(FUTDIR / exp.replace("-", "") / "*.parquet"))
    if not fs: return None
    return pd.read_parquet(fs[0], columns=["timestamp", "open", "close"])


def opt_series(exp, strike, ot):
    fs = glob.glob(str(OPTDIR / exp.replace("-", "") / f"BANKNIFTY_{int(strike)}_{ot}_*.parquet"))
    if not fs: return None
    return pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]).set_index("timestamp").sort_index()


def main():
    d = pd.read_parquet(DAILY, columns=["date"]); tdays = sorted(d["date"].dt.date.unique())
    tdays = [t for t in tdays if t >= FLOOR.date()]
    fut_cache = {}
    def get_fut(exp):
        if exp not in fut_cache: fut_cache[exp] = futures_day(exp)
        return fut_cache[exp]

    rows = []; gap_log = []; skipped_other = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        j = bisect.bisect_right(MONTHLY, str(D))
        if j >= len(MONTHLY): gap_log.append({"date": D, "reason": "GAP2"}); continue
        E = MONTHLY[j]; is_exp = str(D) in EXPSET
        if E == GAP1_EXPIRY: gap_log.append({"date": D, "reason": "GAP1"}); continue
        if E == GAP3_EXPIRY: gap_log.append({"date": D, "reason": "GAP3"}); continue

        FT = get_fut(E)
        if FT is None: gap_log.append({"date": D, "reason": f"futures missing cycle {E}"}); continue
        fday = FT[FT["timestamp"].dt.normalize() == pd.Timestamp(D)]
        if fday.empty: skipped_other += 1; continue
        dopen = float(fday["open"].iloc[0])
        f1515 = fday[fday["timestamp"].dt.hour * 60 + fday["timestamp"].dt.minute == ENTRY_MOD]
        if f1515.empty: skipped_other += 1; continue
        fprice = float(f1515["close"].iloc[-1]); atm = int(round(fprice / STEP) * STEP)
        direction = "RED" if fprice < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        long_K = atm; short_K = atm - OFFSET if direction == "RED" else atm + OFFSET
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=15)
        exit_ts = [pd.Timestamp(Dn) + pd.Timedelta(minutes=m) for m in EXIT_MODS]

        Lo = opt_series(E, long_K, ot); So = opt_series(E, short_K, ot)
        if Lo is None or So is None: gap_log.append({"date": D, "reason": f"missing option series {long_K}/{short_K} {ot} cycle {E}"}); continue
        Le = Lo["close"].asof(t_en); Se = So["close"].asof(t_en)
        if pd.isna(Le) or pd.isna(Se): gap_log.append({"date": D, "reason": f"nan entry premium cycle {E}"}); continue
        Lx = Lo["open"].reindex(exit_ts, method="ffill").values; Sx = So["open"].reindex(exit_ts, method="ffill").values
        entry_cost = 2 * Le - Se; pnl = (2 * Lx - Sx) - entry_cost              # vector over 45 exit minutes
        for k, m in enumerate(EXIT_MODS):
            if np.isnan(pnl[k]): continue
            rows.append({"entry_date": D, "exit_mod": m, "direction": direction, "is_expiry_day": is_exp, "pnl": round(float(pnl[k]), 2)})

    R = pd.DataFrame(rows); R["exit_time"] = R["exit_mod"].map(lambda m: f"{m//60:02d}:{m%60:02d}")
    R["oos"] = pd.to_datetime(R["entry_date"]) >= OOS_SPLIT

    def blk(df):
        return {"trades": len(df), "win_%": round((df.pnl > 0).mean() * 100, 1) if len(df) else 0,
                "total_pnl": round(df.pnl.sum(), 1), "avg_pnl": round(df.pnl.mean(), 2) if len(df) else 0}

    per_exit = []
    for m in EXIT_MODS:
        sub = R[R.exit_mod == m]; red = sub[sub.direction == "RED"]; grn = sub[sub.direction == "GREEN"]
        nrm = sub[~sub.is_expiry_day]; exp = sub[sub.is_expiry_day]
        row = {"exit_time": f"{m//60:02d}:{m%60:02d}"}
        b = blk(sub); row.update({f"ALL_{k}": v for k, v in b.items()})
        row.update({f"RED_{k}": v for k, v in blk(red).items()}); row.update({f"GREEN_{k}": v for k, v in blk(grn).items()})
        row.update({f"normal_{k}": v for k, v in blk(nrm).items()}); row.update({f"expiry_{k}": v for k, v in blk(exp).items()})
        per_exit.append(row)
    PE = pd.DataFrame(per_exit)

    # comparison table: exit time vs total P&L, best stable region (3-neighbour mean)
    v = PE["ALL_total_pnl"].values; nb = np.array([np.nanmean(v[max(0, k - 1):k + 2]) for k in range(len(v))])
    bi = int(np.nanargmax(nb)); region = [PE.exit_time.iloc[x] for x in range(max(0, bi - 1), min(len(v), bi + 2))]
    best_single = PE.exit_time.iloc[int(np.nanargmax(v))]
    baseline_row = PE[PE.exit_time == "09:17"].iloc[0]

    CMP = PE[["exit_time", "ALL_trades", "ALL_win_%", "ALL_total_pnl", "ALL_avg_pnl"]].copy()
    CMP["baseline"] = CMP.exit_time.map(lambda t: "<== 09:17 BASELINE" if t == "09:17" else "")
    CMP["best_region"] = CMP.exit_time.map(lambda t: "*" if t in region else "")

    # IS/OOS on ALL_total_pnl per exit time
    isoos = R.pivot_table(index="exit_mod", columns="oos", values="pnl", aggfunc="sum").reindex(EXIT_MODS)
    isoos.columns = ["IS_total_pnl" if not c else "OOS_total_pnl" for c in isoos.columns]
    isoos = isoos.reset_index(); isoos["exit_time"] = isoos["exit_mod"].map(lambda m: f"{m//60:02d}:{m%60:02d}")
    isoos_cnt = R.pivot_table(index="exit_mod", columns="oos", values="pnl", aggfunc="size").reindex(EXIT_MODS)
    isoos_cnt.columns = ["IS_trades" if not c else "OOS_trades" for c in isoos_cnt.columns]
    isoos = isoos.merge(isoos_cnt.reset_index(), on="exit_mod")
    isoos = isoos[["exit_time", "IS_trades", "IS_total_pnl", "OOS_trades", "OOS_total_pnl"]].round(1)
    # IS-best-region validated OOS
    is_v = isoos["IS_total_pnl"].values; is_nb = np.array([np.nanmean(is_v[max(0, k - 1):k + 2]) for k in range(len(is_v))])
    is_bi = int(np.nanargmax(is_nb)); is_region = [isoos.exit_time.iloc[x] for x in range(max(0, is_bi - 1), min(len(is_v), is_bi + 2))]

    summary = pd.DataFrame([
        {"metric": "Sweep", "value": "exit time 09:16-10:00 (1-min steps, 45 values); entry logic UNCHANGED (15:15 futures-referenced, monthly cycle, DTE-0 next-month switch, offset 500)"},
        {"metric": "Exit fill", "value": "swept minute's option OPEN on next trading day (even if expiry day)"},
        {"metric": "Confirmed gaps excluded (same as base)", "value": f"GAP1 (Mar-2025 futures) + GAP3 (Dec-2024 options) + GAP2 (no Aug-2026 options); {len(gap_log)} days skipped, other-skipped {skipped_other}"},
        {"metric": "Trading days used", "value": R.entry_date.nunique()},
        {"metric": "09:17 BASELINE total P&L", "value": baseline_row.ALL_total_pnl}, {"metric": "09:17 BASELINE win %", "value": baseline_row["ALL_win_%"]},
        {"metric": "Best stable region (3-min nbhd)", "value": f"{region} (center {PE.exit_time.iloc[bi]})"},
        {"metric": "Best single exit time", "value": f"{best_single} = {round(v[int(np.nanargmax(v))],1)} pts"},
        {"metric": "IS best stable region (fit on IS)", "value": f"{is_region}"},
        {"metric": "OOS split", "value": str(OOS_SPLIT.date()) + " (IS<split / OOS>=split)"},
    ])
    with pd.ExcelWriter(OUTDIR / "banknifty_btst_exit_sweep.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        CMP.to_excel(w, sheet_name="Comparison", index=False)
        PE.to_excel(w, sheet_name="Per_ExitTime_FullSplit", index=False)
        isoos.to_excel(w, sheet_name="IS_OOS", index=False)
        pd.DataFrame(gap_log).to_excel(w, sheet_name="Gap_Log", index=False) if gap_log else pd.DataFrame([{"note": "none"}]).to_excel(w, sheet_name="Gap_Log", index=False)

    pd.set_option("display.width", 260)
    print("=" * 100 + "\nBANKNIFTY BTST — EXIT-TIME SWEEP (09:16-10:00)\n" + "=" * 100)
    print(f"days {R.entry_date.nunique()} | 09:17 baseline: {baseline_row.ALL_total_pnl} pts, {baseline_row['ALL_win_%']}% win")
    print(f"\nBest stable region: {region} (center {PE.exit_time.iloc[bi]}) | best single {best_single} = {round(v[int(np.nanargmax(v))],1)}")
    print("\n--- COMPARISON (every 3rd minute) ---"); print(CMP.iloc[::3].to_string(index=False))
    print("\n--- IS/OOS (every 3rd minute) ---"); print(isoos.iloc[::3].to_string(index=False))
    print(f"IS best region: {is_region}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
