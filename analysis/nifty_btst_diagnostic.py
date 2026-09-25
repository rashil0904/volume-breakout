# -*- coding: utf-8 -*-
"""nifty_btst_diagnostic.py — ANALYSIS-ONLY diagnostic dissection of the NIFTY Close-Direction BTST strategy
(existing finalized spec: entry 15:20, exit 09:17 next day, offset 300, never DTE-0). Rebuilds the FULL
unfiltered trade set (no VIX 17-19 exclusion — needed to properly diagnose cut 4) enriched with: weekday,
entry-day move size, overnight-vs-intraday P&L split (position value at 3:15pm entry -> next-day open ->
09:17 exit), and peak-vs-actual P&L (best net position value reached between entry and exit vs actual exit).
Does NOT change strategy logic — read-only diagnostic. 9 cuts + ranked improvement candidates + IS/OOS check.
"""
import sys, glob, os, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "btst_close_direction_diagnostic"; OUTDIR.mkdir(parents=True, exist_ok=True)
OFFSET = 300; ENTRY_MOD = 15 * 60 + 20; EXIT_MOD = 9 * 60 + 17
VIXF = rb.BASE / "data" / "india_vix_1min.csv"; VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]
OOS_SPLIT = pd.Timestamp("2025-08-29")


def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


def leg_full(folder, strike, otype, t0, t1):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    o = o[(o.ts >= t0) & (o.ts <= t1)].set_index("ts").sort_index()
    return o if len(o) else None


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    spot_end = sp["date"].max()
    day_open = sp.groupby("date")["open"].first()
    spot1520 = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    vix_ent = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    tdays = sorted(day_open.index)
    expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")); expset = set(expiries); first_exp = expiries[0]

    trades = []; skipped = 0
    for i in range(len(tdays) - 1):
        D = tdays[i]; Dn = tdays[i + 1]
        if D < first_exp or Dn > spot_end or D not in spot1520.index: continue
        j = bisect.bisect_right(expiries, D)
        if j >= len(expiries): continue
        E = expiries[j]; is_exp = D in expset
        espot = float(spot1520.loc[D]); dopen = float(day_open.loc[D]); atm = round(espot / 50) * 50
        direction = "RED" if espot < dopen else "GREEN"; ot = "PE" if direction == "RED" else "CE"
        long_K = atm; short_K = atm - OFFSET if direction == "RED" else atm + OFFSET
        t_en = pd.Timestamp(D) + pd.Timedelta(hours=15, minutes=20); t_ex = pd.Timestamp(Dn) + pd.Timedelta(hours=9, minutes=17)
        folder = E.strftime("%Y%m%d")
        Lo = leg_full(folder, long_K, ot, t_en - pd.Timedelta(minutes=5), t_ex + pd.Timedelta(minutes=5))
        So = leg_full(folder, short_K, ot, t_en - pd.Timedelta(minutes=5), t_ex + pd.Timedelta(minutes=5))
        if Lo is None or So is None: skipped += 1; continue
        Le = Lo["close"].asof(t_en); Se = So["close"].asof(t_en)
        Lx = Lo["close"].asof(t_ex); Sx = So["close"].asof(t_ex)
        if any(pd.isna(v) for v in (Le, Se, Lx, Sx)): skipped += 1; continue
        entry_cost = 2 * Le - Se; exit_val = 2 * Lx - Sx; pnl = exit_val - entry_cost

        # --- peak (best net value reached between entry and exit, on CLOSE, matching fill convention) ---
        both = pd.concat([Lo["close"].rename("l"), So["close"].rename("s")], axis=1).ffill().dropna()
        win = both[(both.index > t_en) & (both.index <= t_ex)]
        if len(win):
            netv = 2 * win["l"] - win["s"]; peak_net = float(netv.max())
        else:
            peak_net = float(exit_val)
        peak_pnl = peak_net - entry_cost; given_up = max(0.0, peak_pnl - pnl)

        # --- overnight vs intraday split (position value at 3:15pm entry -> next-day OPEN -> 09:17 exit) ---
        nd = both[both.index.normalize() == pd.Timestamp(Dn)]
        if len(nd):
            first_row = nd.iloc[0]; overnight_boundary = 2 * float(first_row["l"]) - float(first_row["s"])
        else:
            overnight_boundary = float(exit_val)
        overnight_pnl = overnight_boundary - entry_cost; intraday_pnl = pnl - overnight_pnl

        move_pts = espot - dopen; move_pct = move_pts / dopen * 100
        trades.append({"entry_date": D, "weekday": pd.Timestamp(D).day_name(), "entry_vix": round(float(vix_ent.get(D, np.nan)), 2),
                       "direction": direction, "is_expiry_day": is_exp, "ATM": int(atm),
                       "day_open": round(dopen, 2), "entry_spot": round(espot, 2), "move_pts": round(move_pts, 2), "move_pct": round(move_pct, 3),
                       "entry_cost": round(float(entry_cost), 2), "exit_value": round(float(exit_val), 2), "pnl_points": round(float(pnl), 2),
                       "peak_pnl": round(peak_pnl, 2), "pnl_given_up": round(given_up, 2),
                       "overnight_pnl": round(overnight_pnl, 2), "intraday_pnl": round(intraday_pnl, 2),
                       "exit_date": Dn})
    T = pd.DataFrame(trades)
    T["vix_bucket"] = T["entry_vix"].map(vbucket)
    T["abs_move_pct"] = T["move_pct"].abs()
    T["move_bucket"] = pd.cut(T["abs_move_pct"], [0, 0.5, 1.0, 2.0, np.inf], labels=["<0.5%", "0.5-1%", "1-2%", ">2%"], right=False)
    T["oos"] = pd.to_datetime(T["entry_date"]) >= OOS_SPLIT
    T["win"] = T["pnl_points"] > 0
    T.to_csv(OUTDIR / "diagnostic_trades_full.csv", index=False)

    def blk(df, extra_avg_cols=()):
        d = {"trades": len(df), "win_%": round((df.pnl_points > 0).mean() * 100, 1) if len(df) else 0,
             "total_pnl": round(df.pnl_points.sum(), 1), "avg_pnl": round(df.pnl_points.mean(), 2) if len(df) else 0,
             "median_pnl": round(df.pnl_points.median(), 2) if len(df) else 0}
        for c in extra_avg_cols: d[f"avg_{c}"] = round(df[c].mean(), 2) if len(df) else 0
        return d

    # ===== CUT 1: day of week =====
    WD_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    C1 = pd.DataFrame([{"weekday": w, **blk(T[T.weekday == w])} for w in WD_ORDER if (T.weekday == w).any()])

    # ===== CUT 2: RED vs GREEN =====
    C2 = pd.DataFrame([{"direction": g, **blk(T[T.direction == g])} for g in ["RED", "GREEN"]])

    # ===== CUT 3: normal vs expiry-day =====
    C3 = pd.DataFrame([{"group": g, **blk(T[T.is_expiry_day == v])} for g, v in [("normal-day", False), ("expiry-day (next-week)", True)]])

    # ===== CUT 4: VIX buckets =====
    present = [b for b in VIX_BUCKETS if (T.vix_bucket == b).any()]
    C4 = pd.DataFrame([{"vix_bucket": b, **blk(T[T.vix_bucket == b])} for b in present])

    # ===== CUT 5: move-size buckets =====
    MB_ORDER = ["<0.5%", "0.5-1%", "1-2%", ">2%"]
    C5 = pd.DataFrame([{"move_bucket": m, **blk(T[T.move_bucket == m])} for m in MB_ORDER if (T.move_bucket == m).any()])

    # ===== CUT 6: overnight vs intraday P&L attribution =====
    C6 = pd.DataFrame([
        {"component": "overnight (entry->next-day open)", "total_pnl": round(T.overnight_pnl.sum(), 1), "avg": round(T.overnight_pnl.mean(), 2), "win_%": round((T.overnight_pnl > 0).mean() * 100, 1)},
        {"component": "intraday (next-day open->09:17 exit)", "total_pnl": round(T.intraday_pnl.sum(), 1), "avg": round(T.intraday_pnl.mean(), 2), "win_%": round((T.intraday_pnl > 0).mean() * 100, 1)},
        {"component": "TOTAL (check = overnight+intraday)", "total_pnl": round((T.overnight_pnl + T.intraday_pnl).sum(), 1), "avg": round((T.overnight_pnl + T.intraday_pnl).mean(), 2), "win_%": round(((T.overnight_pnl + T.intraday_pnl) > 0).mean() * 100, 1)},
    ])
    corr_on_intra = round(T["overnight_pnl"].corr(T["intraday_pnl"]), 3)

    # ===== CUT 7: peak vs actual (P&L left on the table) =====
    C7 = pd.DataFrame([
        {"metric": "Total actual P&L", "value": round(T.pnl_points.sum(), 1)},
        {"metric": "Total peak (best-possible) P&L", "value": round(T.peak_pnl.sum(), 1)},
        {"metric": "Total P&L given up (peak - actual, sum of positive gaps)", "value": round(T.pnl_given_up.sum(), 1)},
        {"metric": "Avg P&L given up / trade", "value": round(T.pnl_given_up.mean(), 2)},
        {"metric": "% trades where exit == peak (given_up<=0.5)", "value": round((T.pnl_given_up <= 0.5).mean() * 100, 1)},
        {"metric": "% trades leaving >50 pts on the table", "value": round((T.pnl_given_up > 50).mean() * 100, 1)},
    ])

    # ===== CUT 8: loss analysis =====
    L = T[~T.win]
    C8_vix = L.groupby("vix_bucket").size().reindex(present).fillna(0).astype(int).rename("loss_count").reset_index()
    C8_vix["loss_%_of_bucket"] = [round((T[(T.vix_bucket == b) & (~T.win)].shape[0] / max((T.vix_bucket == b).sum(), 1)) * 100, 1) for b in C8_vix.vix_bucket]
    C8_move = L.groupby("move_bucket", observed=True).size().rename("loss_count").reset_index()
    C8_wd = L.groupby("weekday").size().reindex(WD_ORDER).fillna(0).astype(int).rename("loss_count").reset_index()
    C8_dir = L.groupby("direction").size().rename("loss_count").reset_index()
    C8_summary = pd.DataFrame([
        {"metric": "Total losing trades", "value": len(L)}, {"metric": "Total loss (sum of negative pnl)", "value": round(L.pnl_points.sum(), 1)},
        {"metric": "Avg loss size", "value": round(L.pnl_points.mean(), 2)}, {"metric": "Worst loss", "value": round(L.pnl_points.min(), 1)},
        {"metric": "Median entry-day |move%| on losers", "value": round(L.abs_move_pct.median(), 3)},
        {"metric": "Median entry-day |move%| on winners", "value": round(T[T.win].abs_move_pct.median(), 3)},
    ])

    # ===== CUT 9: streaks =====
    T9 = T.sort_values("entry_date").reset_index(drop=True)
    T9["dir_code"] = (T9.direction == "GREEN").astype(int)
    T9["streak_id"] = (T9["dir_code"] != T9["dir_code"].shift()).cumsum()
    streak_len = T9.groupby("streak_id").size()
    T9["pos_in_streak"] = T9.groupby("streak_id").cumcount() + 1
    T9["streak_len_total"] = T9["streak_id"].map(streak_len)
    C9_by_pos = T9.groupby("pos_in_streak").apply(lambda g: pd.Series(blk(g)), include_groups=False).reset_index()
    C9_by_pos = C9_by_pos[C9_by_pos.pos_in_streak <= 5]
    # win/loss streaks (consecutive winning or losing trades)
    T9["win_code"] = T9["win"].astype(int)
    T9["wstreak_id"] = (T9["win_code"] != T9["win_code"].shift()).cumsum()
    wstreaks = T9.groupby("wstreak_id").agg(is_win=("win", "first"), length=("win", "size"))
    max_win_streak = wstreaks[wstreaks.is_win].length.max() if wstreaks.is_win.any() else 0
    max_loss_streak = wstreaks[~wstreaks.is_win].length.max() if (~wstreaks.is_win).any() else 0
    C9_summary = pd.DataFrame([
        {"metric": "Longest same-direction streak (days)", "value": int(streak_len.max())},
        {"metric": "Longest winning streak (trades)", "value": int(max_win_streak)},
        {"metric": "Longest losing streak (trades)", "value": int(max_loss_streak)},
        {"metric": "Avg win/loss on 1st day of a direction streak vs 3rd+", "value": f"{round(T9[T9.pos_in_streak==1].pnl_points.mean(),2)} vs {round(T9[T9.pos_in_streak>=3].pnl_points.mean(),2)}"},
    ])

    # ===== IS/OOS validation of top candidate rules =====
    def rule_impact(mask_bad, label):
        keep = T[~mask_bad]; bad = T[mask_bad]
        is_bad = bad[~bad.oos]; oos_bad = bad[bad.oos]
        return {"rule": label, "trades_removed": len(bad), "total_pnl_of_removed": round(bad.pnl_points.sum(), 1),
                "IS_removed_pnl": round(is_bad.pnl_points.sum(), 1), "OOS_removed_pnl": round(oos_bad.pnl_points.sum(), 1),
                "new_total_if_applied": round(keep.pnl_points.sum(), 1), "delta_vs_baseline": round(keep.pnl_points.sum() - T.pnl_points.sum(), 1)}

    candidates = []
    candidates.append(rule_impact(T.vix_bucket == "17-18", "Exclude VIX 17-18"))
    candidates.append(rule_impact(T.vix_bucket == "18-19", "Exclude VIX 18-19"))
    candidates.append(rule_impact(T.vix_bucket.isin(["17-18", "18-19"]), "Exclude VIX 17-19 (current filter)"))
    candidates.append(rule_impact(T.is_expiry_day, "Exclude expiry-day (next-week) trades"))
    candidates.append(rule_impact(T.direction == "GREEN", "Trade RED only (exclude GREEN)"))
    candidates.append(rule_impact(T.abs_move_pct > 1.5, "Skip entries where entry-day |move| > 1.5%"))
    candidates.append(rule_impact(T.abs_move_pct > 2.0, "Skip entries where entry-day |move| > 2.0%"))
    candidates.append(rule_impact(T.weekday == "Friday", "Exclude Friday entries"))
    CAND = pd.DataFrame(candidates).sort_values("delta_vs_baseline", ascending=False)

    baseline_total = round(T.pnl_points.sum(), 1)

    with pd.ExcelWriter(OUTDIR / "nifty_btst_diagnostic.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Scope", "value": "ANALYSIS-ONLY diagnostic on existing NIFTY BTST (entry 15:20, exit 09:17, offset 300). Full UNFILTERED set (no VIX 17-19 exclusion) used here so cut 4/8 can see that bucket."},
            {"metric": "Total trades (unfiltered)", "value": len(T)}, {"metric": "Baseline total P&L (unfiltered)", "value": baseline_total},
            {"metric": "Win rate % (unfiltered)", "value": round(T.win.mean()*100,1)},
            {"metric": "Skipped (missing leg data)", "value": skipped}, {"metric": "OOS split", "value": str(OOS_SPLIT.date())},
        ]).to_excel(w, sheet_name="Scope", index=False)
        C1.to_excel(w, sheet_name="Cut1_DayOfWeek", index=False); C2.to_excel(w, sheet_name="Cut2_RedGreen", index=False)
        C3.to_excel(w, sheet_name="Cut3_Normal_vs_Expiry", index=False); C4.to_excel(w, sheet_name="Cut4_VIX", index=False)
        C5.to_excel(w, sheet_name="Cut5_MoveSize", index=False); C6.to_excel(w, sheet_name="Cut6_Overnight_Intraday", index=False)
        C7.to_excel(w, sheet_name="Cut7_PeakVsActual", index=False)
        C8_summary.to_excel(w, sheet_name="Cut8_Losses", index=False, startrow=0)
        C8_vix.to_excel(w, sheet_name="Cut8_Losses", index=False, startrow=len(C8_summary)+2)
        C8_move.to_excel(w, sheet_name="Cut8_Losses", index=False, startrow=len(C8_summary)+2+len(C8_vix)+2)
        C8_wd.to_excel(w, sheet_name="Cut8_Losses", index=False, startrow=len(C8_summary)+2+len(C8_vix)+2+len(C8_move)+2)
        C9_summary.to_excel(w, sheet_name="Cut9_Streaks", index=False, startrow=0); C9_by_pos.to_excel(w, sheet_name="Cut9_Streaks", index=False, startrow=len(C9_summary)+2)
        CAND.to_excel(w, sheet_name="Improvement_Candidates", index=False)
        T.to_excel(w, sheet_name="Full_Trade_Data", index=False)

    pd.set_option("display.width", 220)
    print("=" * 100 + "\nNIFTY BTST DIAGNOSTIC (analysis-only, unfiltered set)\n" + "=" * 100)
    print(f"trades {len(T)} | baseline total {baseline_total} | win {round(T.win.mean()*100,1)}% | skipped {skipped}\n")
    print("--- CUT 1: Day of week ---"); print(C1.to_string(index=False))
    print("\n--- CUT 2: RED vs GREEN ---"); print(C2.to_string(index=False))
    print("\n--- CUT 3: Normal vs Expiry-day ---"); print(C3.to_string(index=False))
    print("\n--- CUT 4: VIX buckets ---"); print(C4.to_string(index=False))
    print("\n--- CUT 5: Move-size buckets ---"); print(C5.to_string(index=False))
    print("\n--- CUT 6: Overnight vs Intraday ---"); print(C6.to_string(index=False)); print(f"corr(overnight,intraday) = {corr_on_intra}")
    print("\n--- CUT 7: Peak vs Actual ---"); print(C7.to_string(index=False))
    print("\n--- CUT 8: Loss analysis ---"); print(C8_summary.to_string(index=False)); print(C8_vix.to_string(index=False)); print(C8_move.to_string(index=False)); print(C8_wd.to_string(index=False))
    print("\n--- CUT 9: Streaks ---"); print(C9_summary.to_string(index=False)); print(C9_by_pos.to_string(index=False))
    print("\n--- IMPROVEMENT CANDIDATES (ranked by delta vs baseline) ---"); print(CAND.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
