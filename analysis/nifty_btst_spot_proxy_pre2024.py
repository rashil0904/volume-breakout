# -*- coding: utf-8 -*-
"""nifty_btst_spot_proxy_pre2024.py — EXPLORATORY SPOT-ONLY PROXY for the NIFTY Close-Direction BTST
strategy, March 2022 - September 2024 (before any options data exists locally -- options data in this
project starts Oct-2024). This is NOT a real P&L backtest. No option premiums, IV, delta, theta, or decay
are modeled here -- it tracks NIFTY SPOT's own movement under the same entry/exit timing as the finalized
options strategy, purely to see whether the directional premise (red day -> further downside into next
morning, green day -> further upside) held up historically. Any "illustrative P&L" figure is spot points,
explicitly NOT option P&L -- flagged throughout.

ENTRY/EXIT (matches FINAL v3 timing exactly): entry = spot CLOSE @ 15:20 on day D; direction RED if that
close < day D's OPEN, else GREEN. Exit = spot CLOSE @ 09:17 on next trading day Dn.

FILTERS (for day-selection consistency with the options FINAL v3 strategy):
  - VIX[17,19] exclusion: India VIX close @ 15:20 in [17,19] -> excluded. VIX 1-min data confirmed
    available for this whole window (data/india_vix_1min.csv, 2022-01-03 onward).
  - DTE-1 exclusion: entry day where the current week's NIFTY expiry is exactly 1 calendar day away ->
    excluded entirely (no trade), matching FINAL v3. EXPIRY CALENDAR NOTE: this project's other BTST
    scripts derive the expiry calendar from local options-data folder names, which do NOT exist before
    Oct-2024 -- there is no local source for this period. Confirmed via web search (NSE/exchange
    circulars): NIFTY 50 weekly options expired on THURSDAY throughout this entire window with NO
    mid-window change (Bank Nifty's Wed-shift in Sep-2023 did not apply to Nifty; Nifty's own Thu->Tue
    shift did not happen until Sep-2025, well after this window ends). So the expiry calendar here is
    every Thursday, holiday-adjusted to the prior actual trading day when Thursday itself isn't one
    (derived from the real spot trading-day calendar) -- standard NSE convention, single regime, no
    external file needed.
  - DTE-0 (expiry-day entry): FLAGGED, not excluded. The real strategy's DTE-0 rule exists to avoid
    pricing an already-expired current-week contract at next-morning exit (so it switches to the next
    week's contract). Spot itself has no such expiry mechanic -- there's nothing for spot to "switch" to
    -- so DTE-0 days are kept in the proxy pool but marked in a column for visibility, per the explicit
    instruction to flag rather than force an exclusion that doesn't have a real spot-analog.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
OUTDIR = rb.RESULTS / "btst_spot_proxy_pre2024"; OUTDIR.mkdir(parents=True, exist_ok=True)

ENTRY_MOD = 15 * 60 + 20; EXIT_MOD = 9 * 60 + 17
VIX_LO, VIX_HI = 17.0, 19.0
WIN_START = pd.Timestamp("2022-03-01"); WIN_END = pd.Timestamp("2024-09-30")

# reference: FINAL v3 options-based strategy (Oct-2024..Jul-2026), for comparison
FINAL_V3 = {"trades": 323, "win_pct": 55.42, "total_pnl_pts": 9662.0,
            "red_n": 182, "red_win_pct": 53.8, "green_n": 141, "green_win_pct": 57.4}


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    day_open = sp.groupby("date")["open"].first()
    spot_entry = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    spot_exit = sp[sp["mod"] == EXIT_MOD].groupby("date")["close"].last()

    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    vix_entry = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()

    tdays = sorted(d for d in day_open.index if WIN_START <= d <= WIN_END)
    all_tdays_set = set(sp["date"].unique())

    # ---- expiry calendar: every Thursday, holiday-adjusted to the prior actual trading day ----
    thursdays = pd.date_range(WIN_START - pd.Timedelta(days=30), WIN_END + pd.Timedelta(days=10), freq="W-THU")
    expiries = []
    for th in thursdays:
        d = th
        while d not in all_tdays_set and d >= th - pd.Timedelta(days=6):
            d -= pd.Timedelta(days=1)
        if d in all_tdays_set:
            expiries.append(d)
    expiries = sorted(set(expiries))
    expset = set(expiries)

    rows = []
    # build next-trading-day map from the full spot calendar (not just the windowed tdays)
    full_days = sorted(sp["date"].unique())
    next_day_map = {full_days[i]: full_days[i + 1] for i in range(len(full_days) - 1)}

    for D in tdays:
        Dn = next_day_map.get(D)
        if Dn is None or D not in spot_entry.index or Dn not in spot_exit.index:
            continue
        entry_spot = float(spot_entry.loc[D]); dopen = float(day_open.loc[D]); exit_spot = float(spot_exit.loc[Dn])
        direction = "RED" if entry_spot < dopen else "GREEN"
        vix_val = float(vix_entry.get(D, np.nan))

        # current-week expiry: smallest expiry >= D
        fut_exp = [e for e in expiries if e >= D]
        if not fut_exp:
            continue
        E = fut_exp[0]; dte = (E.normalize() - D.normalize()).days
        is_expiry_day = D in expset

        vix_excluded = (not np.isnan(vix_val)) and (VIX_LO <= vix_val <= VIX_HI)
        dte1_excluded = (dte == 1)

        spot_move = exit_spot - entry_spot
        spot_move_pct = spot_move / entry_spot * 100
        dir_consistent_move = spot_move if direction == "GREEN" else -spot_move
        favorable = dir_consistent_move > 0

        rows.append({
            "entry_date": D.date(), "direction": direction, "day_open": round(dopen, 2),
            "entry_spot_1520": round(entry_spot, 2), "exit_date": Dn.date(), "exit_spot_0917": round(exit_spot, 2),
            "spot_move_pts": round(spot_move, 2), "spot_move_pct": round(spot_move_pct, 4),
            "direction_consistent_move_pts": round(dir_consistent_move, 2), "favorable": favorable,
            "entry_vix": round(vix_val, 2) if not np.isnan(vix_val) else None,
            "excluded_vix_filter": vix_excluded, "expiry_used": E.date(), "DTE": dte,
            "excluded_dte1_filter": dte1_excluded, "is_expiry_day_DTE0": is_expiry_day,
            "qualifies": (not vix_excluded) and (not dte1_excluded),
        })

    ALL = pd.DataFrame(rows)
    Q = ALL[ALL["qualifies"]].reset_index(drop=True)

    n_all = len(ALL); n_vix_excl = int(ALL["excluded_vix_filter"].sum()); n_dte1_excl = int(ALL["excluded_dte1_filter"].sum())
    n_dte0_flagged = int(Q["is_expiry_day_DTE0"].sum())
    n_q = len(Q)
    n_fav = int(Q["favorable"].sum()); n_unfav = n_q - n_fav
    pct_fav = round(n_fav / n_q * 100, 2) if n_q else 0.0
    avg_fav = round(Q.loc[Q["favorable"], "direction_consistent_move_pts"].mean(), 2) if n_fav else 0.0
    avg_unfav = round(Q.loc[~Q["favorable"], "direction_consistent_move_pts"].mean(), 2) if n_unfav else 0.0
    illustrative_total = round(Q["direction_consistent_move_pts"].sum(), 1)

    red = Q[Q.direction == "RED"]; grn = Q[Q.direction == "GREEN"]
    red_fav_pct = round(red["favorable"].mean() * 100, 1) if len(red) else 0.0
    grn_fav_pct = round(grn["favorable"].mean() * 100, 1) if len(grn) else 0.0

    print("=" * 100)
    print("NIFTY BTST SPOT-ONLY PROXY (Mar-2022 .. Sep-2024) -- EXPLORATORY, NOT A REAL BACKTEST")
    print("No option premiums/IV/theta modeled. Directional spot-move proxy only.")
    print("=" * 100)
    print(f"trading-day pairs in window: {n_all}")
    print(f"  excluded by VIX[17,19] filter: {n_vix_excl}")
    print(f"  excluded by DTE-1 filter: {n_dte1_excl}")
    print(f"  QUALIFYING days: {n_q}  (of which {n_dte0_flagged} are DTE-0/expiry-day entries -- flagged, NOT excluded, see note)")
    print()
    print(f"favorable (direction-consistent) moves: {n_fav}/{n_q} = {pct_fav}%")
    print(f"unfavorable moves: {n_unfav}/{n_q} = {round(100-pct_fav,2)}%")
    print(f"avg favorable move: +{avg_fav} pts | avg unfavorable move: {avg_unfav} pts")
    print(f"illustrative 'directional P&L' (SUM of direction-consistent spot points, NOT real option P&L): {illustrative_total} pts")
    print()
    print(f"RED days: {len(red)}, favorable {red_fav_pct}%  |  GREEN days: {len(grn)}, favorable {grn_fav_pct}%")

    print("\n" + "=" * 100)
    print("COMPARISON vs FINAL v3 options-based strategy (Oct-2024..Jul-2026)")
    print("=" * 100)
    CMP = pd.DataFrame([
        {"metric": "n_days/trades", "Spot proxy (Mar22-Sep24)": n_q, "FINAL v3 options (Oct24-Jul26)": FINAL_V3["trades"]},
        {"metric": "favorable/win rate %", "Spot proxy (Mar22-Sep24)": pct_fav, "FINAL v3 options (Oct24-Jul26)": FINAL_V3["win_pct"]},
        {"metric": "RED n", "Spot proxy (Mar22-Sep24)": len(red), "FINAL v3 options (Oct24-Jul26)": FINAL_V3["red_n"]},
        {"metric": "RED favorable/win %", "Spot proxy (Mar22-Sep24)": red_fav_pct, "FINAL v3 options (Oct24-Jul26)": FINAL_V3["red_win_pct"]},
        {"metric": "GREEN n", "Spot proxy (Mar22-Sep24)": len(grn), "FINAL v3 options (Oct24-Jul26)": FINAL_V3["green_n"]},
        {"metric": "GREEN favorable/win %", "Spot proxy (Mar22-Sep24)": grn_fav_pct, "FINAL v3 options (Oct24-Jul26)": FINAL_V3["green_win_pct"]},
    ])
    print(CMP.to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "nifty_btst_spot_proxy_pre2024.xlsx", engine="openpyxl") as w:
        summary = pd.DataFrame([
            {"metric": "LABEL", "value": "EXPLORATORY SPOT-ONLY PROXY -- NOT A REAL BACKTEST. No option premiums/IV/theta/decay modeled."},
            {"metric": "window", "value": f"{WIN_START.date()} .. {WIN_END.date()}"},
            {"metric": "expiry_calendar_source", "value": "Every Thursday (holiday-adjusted to prior trading day), confirmed via web search: Nifty weekly expiry was Thursday throughout this entire window, no mid-window change (unlike Bank Nifty Sep-2023 and Nifty's own later Sep-2025 Thu->Tue shift)"},
            {"metric": "n_trading_day_pairs_in_window", "value": n_all},
            {"metric": "n_excluded_vix_filter", "value": n_vix_excl},
            {"metric": "n_excluded_dte1_filter", "value": n_dte1_excl},
            {"metric": "n_qualifying", "value": n_q},
            {"metric": "n_dte0_expiry_day_flagged_not_excluded", "value": n_dte0_flagged},
            {"metric": "n_favorable", "value": n_fav}, {"metric": "n_unfavorable", "value": n_unfav},
            {"metric": "pct_favorable", "value": pct_fav},
            {"metric": "avg_favorable_move_pts", "value": avg_fav}, {"metric": "avg_unfavorable_move_pts", "value": avg_unfav},
            {"metric": "illustrative_directional_pnl_pts_SUM (NOT real option P&L)", "value": illustrative_total},
            {"metric": "RED_n", "value": len(red)}, {"metric": "RED_favorable_pct", "value": red_fav_pct},
            {"metric": "GREEN_n", "value": len(grn)}, {"metric": "GREEN_favorable_pct", "value": grn_fav_pct},
        ])
        summary.to_excel(w, sheet_name="Summary", index=False)
        CMP.to_excel(w, sheet_name="Comparison_vs_FINAL_v3", index=False)
        ALL.to_excel(w, sheet_name="Per_Day_Full_Log", index=False)
        Q.to_excel(w, sheet_name="Qualifying_Days_Only", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)

    print(f"\nSaved -> {OUTDIR}/nifty_btst_spot_proxy_pre2024.xlsx")


if __name__ == "__main__":
    main()
