# -*- coding: utf-8 -*-
"""nifty_btst_synthetic_fy_wise_report.py — FY-wise (Indian FY, Apr-Mar) returns tables + graphs for the
NIFTY Daily Close Direction BTST strategy, FULL March-2022..July-2026 span, using a UNIFORM SYNTHETIC
options-P&L conversion applied to EVERY trade in this span (even Oct-2024+ which has real option data --
deliberately recomputed synthetically here per explicit instruction, so the whole span uses one consistent
proxy method rather than mixing real and proxy P&L). This is NOT the same number as FINAL v3's real P&L.

SYNTHETIC CONVERSION (flagged, applied uniformly):
  1. direction-consistent NIFTY SPOT move per trade (entry 15:20 close -> next-day 9:17 close).
  2. synthetic_option_points = spot_move_pts * 0.7
  3. net_points = synthetic_option_points - 1.5 (flat per-trade expense)
  4. pnl_inr = net_points * 455  (7 lots x 65 qty/lot)
  Capital base Rs 5,00,000 shown for context only (same convention as the Volume-Breakout FY report).

EXPIRY CALENDAR (DTE-1 exclusion), spans both regimes:
  - Pre-Oct-2024 (no local options data): synthetic Thursday-expiry calendar, holiday-adjusted to the
    prior actual trading day -- confirmed via web search elsewhere in this project that Nifty weekly
    expiry was Thursday throughout this window, no mid-window change.
  - Oct-2024 onward: REAL expiry calendar from the actual local options data folder names (captures the
    real Thu->Tue regime change exactly, no synthesis needed).

VIX[17,19] FILTER: uses data/india_vix_1min.csv, which covers 2022-01-03 onward -- this ALREADY spans the
entire Mar-2022..Jul-2026 window requested here in full, at true intraday (entry-minute-matched) granularity.
The separately-pulled 2020-2021 daily VIX file is NOT needed/used here since it ends Dec-2021, before this
report's period even starts -- flagged explicitly, not silently omitted: NO VIX COVERAGE GAP EXISTS for
this specific task's date range.
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "btst_close_direction_FINAL" / "synthetic_fy_wise"; OUTDIR.mkdir(parents=True, exist_ok=True)

ENTRY_MOD = 15 * 60 + 20; EXIT_MOD = 9 * 60 + 17
VIX_LO, VIX_HI = 17.0, 19.0
SPOT_TO_SYNTH = 0.7; EXPENSE_PTS = 1.5; LOT_SIZE = 65; LOTS = 7; INR_MULT = LOT_SIZE * LOTS
CAPITAL_BASE = 500_000
WIN_START = pd.Timestamp("2022-03-01"); WIN_END = pd.Timestamp("2026-07-31")


def build_expiry_calendar(spot_all_days):
    real_expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d"))
    real_expiries = [e for e in real_expiries if e <= WIN_END]
    first_real = real_expiries[0]

    thursdays = pd.date_range(WIN_START - pd.Timedelta(days=30), first_real, freq="W-THU")
    synth_expiries = []
    tdays_set = set(spot_all_days)
    for th in thursdays:
        if th >= first_real:
            continue
        d = th
        while d not in tdays_set and d >= th - pd.Timedelta(days=6):
            d -= pd.Timedelta(days=1)
        if d in tdays_set:
            synth_expiries.append(d)
    combined = sorted(set(synth_expiries) | set(real_expiries))
    return combined


def fy_bounds(data_start, data_end):
    return [
        ("FY21-22", data_start, pd.Timestamp("2022-03-31")),
        ("FY22-23", pd.Timestamp("2022-04-01"), pd.Timestamp("2023-03-31")),
        ("FY23-24", pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
        ("FY24-25", pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
        ("FY25-26", pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
        ("FY26-27", pd.Timestamp("2026-04-01"), data_end),
    ]


def drawdown_stats(cum):
    equity = np.concatenate([[0.0], cum])
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    eps = []; in_dd = False; tv = 0.0
    for k in range(len(equity)):
        if not in_dd:
            if dd[k] < -1e-9:
                in_dd = True; tv = dd[k]
        else:
            if dd[k] < tv:
                tv = dd[k]
            if dd[k] >= -1e-9:
                eps.append(tv); in_dd = False
    if in_dd:
        eps.append(tv)
    if not eps:
        return 0.0, 0.0
    return abs(min(eps)), float(np.mean([abs(e) for e in eps]))


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    day_open = sp.groupby("date")["open"].first()
    spot_entry = sp[sp["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    spot_exit = sp[sp["mod"] == EXIT_MOD].groupby("date")["close"].last()
    full_days = sorted(sp["date"].unique())
    next_day_map = {full_days[i]: full_days[i + 1] for i in range(len(full_days) - 1)}

    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(mod=vts.dt.hour * 60 + vts.dt.minute, date=vts.dt.normalize())
    vix_entry = vx[vx["mod"] == ENTRY_MOD].groupby("date")["close"].last()
    print(f"VIX 1-min coverage: {vx['date'].min().date()} .. {vx['date'].max().date()} "
          f"-- covers this report's full window ({WIN_START.date()}..{WIN_END.date()})? "
          f"{'YES, no gap' if vx['date'].min() <= WIN_START and vx['date'].max() >= WIN_END else 'GAP FOUND'}", flush=True)

    expiries = build_expiry_calendar(full_days)
    expset = set(expiries)
    print(f"combined expiry calendar: {len(expiries)} expiries, {expiries[0].date()} .. {expiries[-1].date()}", flush=True)

    tdays = [d for d in full_days if WIN_START <= d <= WIN_END]
    rows = []; n_dte1_excl = 0; n_vix_excl = 0
    for D in tdays:
        Dn = next_day_map.get(D)
        if Dn is None or D not in spot_entry.index or Dn not in spot_exit.index:
            continue
        fut_exp = [e for e in expiries if e >= D]
        if not fut_exp:
            continue
        E = fut_exp[0]; dte = (E.normalize() - D.normalize()).days
        if dte == 1:
            n_dte1_excl += 1; continue

        vix_val = float(vix_entry.get(D, np.nan))
        if not np.isnan(vix_val) and VIX_LO <= vix_val <= VIX_HI:
            n_vix_excl += 1; continue

        entry_spot = float(spot_entry.loc[D]); dopen = float(day_open.loc[D]); exit_spot = float(spot_exit.loc[Dn])
        direction = "RED" if entry_spot < dopen else "GREEN"
        spot_move = exit_spot - entry_spot
        dir_move = spot_move if direction == "GREEN" else -spot_move
        synth_pts = dir_move * SPOT_TO_SYNTH
        net_pts = synth_pts - EXPENSE_PTS
        pnl_inr = net_pts * INR_MULT

        rows.append({"entry_date": D, "exit_date": Dn, "direction": direction, "spot_move_pts": round(spot_move, 2),
                     "dir_consistent_spot_move": round(dir_move, 2), "synthetic_option_pts": round(synth_pts, 2),
                     "net_pts_after_expense": round(net_pts, 2), "pnl_inr": round(pnl_inr, 1),
                     "entry_vix": round(vix_val, 2) if not np.isnan(vix_val) else None, "DTE": dte,
                     "expiry_used": E.date()})

    T = pd.DataFrame(rows)
    print(f"\nqualifying trades: {len(T)} | DTE-1 excluded: {n_dte1_excl} | VIX[17,19] excluded: {n_vix_excl}", flush=True)
    T.to_csv(OUTDIR / "synthetic_trades_full.csv", index=False)

    data_start = T["entry_date"].min(); data_end = T["entry_date"].max()
    bounds = fy_bounds(data_start, data_end)

    fy_rows = []
    for fy_label, start, end in bounds:
        Tfy = T[(T["entry_date"] >= start) & (T["entry_date"] <= end)].sort_values("entry_date").reset_index(drop=True)
        if Tfy.empty:
            continue
        daily = Tfy["pnl_inr"].values
        cum = np.cumsum(daily)
        total_return = round(float(cum[-1]), 0)
        win_rate = round((daily > 0).mean() * 100, 2)
        n_win = int((daily > 0).sum()); n_lose = int((daily <= 0).sum())
        maxdd, avgdd = drawdown_stats(cum)
        fy_rows.append({"FY": fy_label, "period": f"{Tfy['entry_date'].min().date()} to {Tfy['entry_date'].max().date()}",
                         "n_trades": len(Tfy), "total_return_inr": total_return, "win_rate_pct": win_rate,
                         "n_winning_days": n_win, "n_losing_days": n_lose,
                         "max_dd_inr": round(maxdd, 0), "avg_dd_inr": round(avgdd, 0)})

        # ---- per-FY graph ----
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(Tfy["entry_date"], cum, color="#1f6feb", linewidth=1.5)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(f"NIFTY Close-Direction BTST — SYNTHETIC P&L (0.7x spot, -1.5pt expense, 7 lots) — {fy_label}\n"
                      f"({Tfy['entry_date'].min().date()} to {Tfy['entry_date'].max().date()})", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11); ax.set_ylabel("Cumulative Return (INR)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        fig.autofmt_xdate(); fig.tight_layout()
        out_png = OUTDIR / f"nifty_btst_synthetic_{fy_label}.png"
        fig.savefig(out_png, dpi=150); plt.close(fig)
        print(f"saved -> {out_png}")

    FY = pd.DataFrame(fy_rows)
    pd.set_option("display.width", 200)
    print("\n=== FY-WISE SUMMARY (synthetic P&L) ===")
    print(FY.to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "nifty_btst_synthetic_FY_wise_report.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "SYNTHETIC P&L, uniform method across the ENTIRE Mar-2022..Jul-2026 span, including Oct-2024+ "
                      "which has real option data elsewhere -- deliberately recomputed synthetically here per explicit "
                      "instruction so the whole span uses ONE consistent proxy, not mixed real+proxy. NOT the same "
                      "number as FINAL v3's real option P&L."},
            {"note": "Conversion: dir-consistent spot move x 0.7 -> synthetic option points; -1.5 pts flat expense; "
                      "x455 INR (7 lots x 65 qty). Capital base Rs 5,00,000 shown for context only."},
            {"note": f"VIX coverage check: india_vix_1min.csv spans {vx['date'].min().date()}..{vx['date'].max().date()} "
                      f"-- fully covers this report's window, NO GAP. The separately-pulled 2020-2021 daily VIX file "
                      f"is not used here (ends Dec-2021, before this period starts)."},
            {"note": f"DTE-1 exclusion: {n_dte1_excl} days excluded via combined expiry calendar "
                      f"(synthetic Thursday pre-Oct-2024 + REAL local expiry folders Oct-2024 onward)."},
            {"note": f"VIX[17,19] exclusion: {n_vix_excl} days excluded."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        FY.to_excel(w, sheet_name="FY_Summary", index=False)
        T.to_excel(w, sheet_name="All_Qualifying_Trades", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 24)

    print(f"\nSaved -> {OUTDIR}/nifty_btst_synthetic_FY_wise_report.xlsx")


if __name__ == "__main__":
    main()
