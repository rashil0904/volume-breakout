# -*- coding: utf-8 -*-
"""nifty_credit_spread_synthetic_fy_wise_report.py — FY-wise (Indian FY, Apr-Mar) returns for the NIFTY
Weekly Credit Spread strategy, FULL March-2022..22-July-2026 span, using a UNIFORM SYNTHETIC/heuristic
P&L methodology applied to EVERY week in this span -- including Oct-2024+, which has real option data
elsewhere in this project (deliberately recomputed synthetically here for one uniform historical
comparison, per explicit instruction; NOT a substitute for the real FINAL strategy's actual results).

Entirely SPOT-ONLY: no options data used anywhere. Net credit is a FIXED ASSUMPTION (87 pts CCS / 67 pts
PCS, 200-pt width), not derived from real premiums.

ENTRY: one trade per week, entered the first trading day after the prior week's expiry. 3-Day High/Low =
highest high / lowest low of the 3 completed trading days before entry. Touch-basis breakout check on
entry day's 1-min bars from open through 14:35: first candle whose LOW <= 3-day-low -> Call Credit Spread
(trigger '3d-low'); first candle whose HIGH >= 3-day-high -> Put Credit Spread (trigger '3d-high'),
whichever occurs first in time. If neither by 14:35, FALLBACK: 14:35 candle's open vs day's open -- red ->
CCS, green -> PCS (trigger 'fallback-2:35'). entry_spot = the open of the triggering/fallback candle.
ATM = round(entry_spot/50)*50.

EXIT (checked once per day, entry_date through expiry, in this priority order):
  1. EOD-CLOSE SL, 3d-high-triggered PCS trades ONLY: exit if that day's close < the 3-day-high level
     that triggered entry (terminal payoff evaluated at that close).
  2. EARLY-TARGET HEURISTIC, DTE==1 or DTE==0 only: favorable move since entry (down for CCS, up for PCS)
     >= 1.7% (DTE=1) or >= 0.7% (DTE=0) -> synthetic 90%-of-credit P&L.
  3. DTE=0 SETTLEMENT fallback: terminal payoff at the mean of the day's last 30 one-min closes (proxy
     for the real 30-min settlement VWAP -- flagged, not the official exchange-computed value).

TERMINAL PAYOFF: CCS (net credit 87) -> spot<=ATM: +87 ; ATM<spot<=ATM+200: 87-(spot-ATM) ; spot>ATM+200: -113.
                 PCS (net credit 67) -> spot>=ATM: +67 ; ATM-200<=spot<ATM: 67-(ATM-spot) ; spot<ATM-200: -133.
INR: points x 455 (7 lots x 65 qty). Capital base Rs 5,00,000 shown for context only.
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
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "weekly_credit_spread" / "synthetic_fy_wise"; OUTDIR.mkdir(parents=True, exist_ok=True)

WIDTH = 200; CCS_CREDIT = 87; PCS_CREDIT = 67
INR_MULT = 455  # 7 lots x 65 qty
EXPENSE_PTS = 1.5  # flat per-trade expense, matching the BTST synthetic convention -- added per explicit user request
WIN_START = pd.Timestamp("2022-03-01"); ENTRY_CUTOFF = pd.Timestamp("2026-07-22")
FALLBACK_HM = 14 * 60 + 35


def build_expiry_calendar(spot_all_days, win_end):
    real_expiries = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d"))
    real_expiries = [e for e in real_expiries if e <= win_end + pd.Timedelta(days=10)]
    first_real = real_expiries[0]
    thursdays = pd.date_range(WIN_START - pd.Timedelta(days=45), first_real, freq="W-THU")
    tdays_set = set(spot_all_days)
    synth = []
    for th in thursdays:
        if th >= first_real:
            continue
        d = th
        while d not in tdays_set and d >= th - pd.Timedelta(days=6):
            d -= pd.Timedelta(days=1)
        if d in tdays_set:
            synth.append(d)
    return sorted(set(synth) | set(real_expiries))


def terminal_payoff(ttype, spot, atm):
    if ttype == "Call Credit Spread":
        if spot <= atm:
            return CCS_CREDIT
        if spot <= atm + WIDTH:
            return CCS_CREDIT - (spot - atm)
        return -(WIDTH - CCS_CREDIT)
    else:
        if spot >= atm:
            return PCS_CREDIT
        if spot >= atm - WIDTH:
            return PCS_CREDIT - (atm - spot)
        return -(WIDTH - PCS_CREDIT)


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
    day_high = sp.groupby("date")["high"].max()
    day_low = sp.groupby("date")["low"].min()
    day_close = sp.groupby("date")["close"].last()
    full_days = sorted(sp["date"].unique())
    day_idx = {d: i for i, d in enumerate(full_days)}
    by_day = {d: g.sort_values("mod") for d, g in sp.groupby("date")}

    expiries = build_expiry_calendar(full_days, ENTRY_CUTOFF)
    print(f"combined expiry calendar: {len(expiries)} expiries, {expiries[0].date()} .. {expiries[-1].date()}", flush=True)

    trades = []; missing = []; ambiguous = []
    for i in range(len(expiries) - 1):
        E_prev, E_curr = expiries[i], expiries[i + 1]
        j = day_idx.get(E_prev)
        if j is None or j + 1 >= len(full_days):
            continue
        entry_date = full_days[j + 1]
        if entry_date < WIN_START or entry_date > ENTRY_CUTOFF:
            continue
        # 3 completed trading days before entry_date
        ei = day_idx[entry_date]
        if ei < 3:
            continue
        prior3 = full_days[ei - 3:ei]
        d3high = max(day_high[d] for d in prior3)
        d3low = min(day_low[d] for d in prior3)

        eday = by_day.get(entry_date)
        if eday is None or eday.empty:
            missing.append((entry_date, "no candles on entry day")); continue
        pre_fallback = eday[eday["mod"] <= FALLBACK_HM]
        trig_type = None; trig_label = None; trig_row = None
        for _, r in pre_fallback.iterrows():
            low_break = r["low"] <= d3low; high_break = r["high"] >= d3high
            if low_break and high_break:
                ambiguous.append((entry_date, r["ts"]))
            if low_break:
                trig_type, trig_label, trig_row = "Call Credit Spread", "3d-low", r; break
            if high_break:
                trig_type, trig_label, trig_row = "Put Credit Spread", "3d-high", r; break
        if trig_type is None:
            fb = eday[eday["mod"] == FALLBACK_HM]
            if fb.empty:
                missing.append((entry_date, "no 14:35 candle for fallback")); continue
            fb_open = float(fb["open"].iloc[0]); d_open = float(day_open[entry_date])
            trig_type = "Call Credit Spread" if fb_open < d_open else "Put Credit Spread"
            trig_label = "fallback-2:35"; entry_spot = fb_open; entry_time = fb["ts"].iloc[0]
        else:
            entry_spot = float(trig_row["open"]); entry_time = trig_row["ts"]

        atm = round(entry_spot / 50) * 50
        net_credit = CCS_CREDIT if trig_type == "Call Credit Spread" else PCS_CREDIT
        entry_dte = (E_curr.normalize() - entry_date.normalize()).days

        # ---- day-by-day exit walk ----
        exit_date = None; exit_reason = None; exit_pts = None; exit_spot_used = None
        idx = ei
        while full_days[idx] <= E_curr:
            D = full_days[idx]; dte = (E_curr.normalize() - D.normalize()).days
            close_d = float(day_close[D])

            if trig_label == "3d-high" and close_d < d3high:
                exit_date, exit_reason = D, "EOD-close SL"
                exit_pts = terminal_payoff(trig_type, close_d, atm); exit_spot_used = close_d
                break

            if dte in (1, 0):
                if trig_type == "Call Credit Spread":
                    fav_pct = (entry_spot - close_d) / entry_spot * 100
                else:
                    fav_pct = (close_d - entry_spot) / entry_spot * 100
                thresh = 1.7 if dte == 1 else 0.7
                if fav_pct >= thresh:
                    exit_date, exit_reason = D, "90% target (synthetic)"
                    exit_pts = 0.90 * net_credit; exit_spot_used = close_d
                    break

            if D == E_curr:
                dday = by_day.get(D)
                if dday is None or dday.empty:
                    missing.append((entry_date, "no candles on expiry day for settlement")); break
                last30 = dday.sort_values("mod").tail(30)
                settlement = float(last30["close"].mean())
                exit_date, exit_reason = D, "DTE0 settlement"
                exit_pts = terminal_payoff(trig_type, settlement, atm); exit_spot_used = settlement
                break
            idx += 1
        if exit_date is None:
            missing.append((entry_date, "no exit resolved")); continue

        net_pts_after_expense = exit_pts - EXPENSE_PTS
        trades.append({"entry_date": entry_date, "entry_time": entry_time, "type": trig_type, "trigger": trig_label,
                        "entry_DTE": entry_dte, "ATM": atm, "entry_spot": round(entry_spot, 2), "net_credit": net_credit,
                        "d3_high": round(d3high, 2), "d3_low": round(d3low, 2),
                        "exit_date": exit_date, "exit_reason": exit_reason, "exit_spot_used": round(exit_spot_used, 2),
                        "exit_DTE": (E_curr.normalize() - exit_date.normalize()).days,
                        "pnl_points_before_expense": round(exit_pts, 2), "pnl_points": round(net_pts_after_expense, 2),
                        "pnl_inr": round(net_pts_after_expense * INR_MULT, 1)})

    T = pd.DataFrame(trades)
    print(f"\ntotal synthetic trades: {len(T)} | missing/flagged: {len(missing)} | ambiguous same-candle triggers: {len(ambiguous)}", flush=True)
    T.to_csv(OUTDIR / "synthetic_credit_spread_trades_full.csv", index=False)

    data_start = T["entry_date"].min(); data_end = T["entry_date"].max()
    bounds = [
        ("FY21-22", data_start, pd.Timestamp("2022-03-31")),
        ("FY22-23", pd.Timestamp("2022-04-01"), pd.Timestamp("2023-03-31")),
        ("FY23-24", pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
        ("FY24-25", pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
        ("FY25-26", pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
        ("FY26-27", pd.Timestamp("2026-04-01"), data_end),
    ]

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
                         "n_weeks": len(Tfy), "total_return_inr": total_return, "win_rate_pct": win_rate,
                         "n_winning_weeks": n_win, "n_losing_weeks": n_lose,
                         "max_dd_inr": round(maxdd, 0), "avg_dd_inr": round(avgdd, 0)})

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(Tfy["entry_date"], cum, color="#1f6feb", linewidth=1.5, marker="o", markersize=3)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(f"NIFTY Weekly Credit Spread — SYNTHETIC P&L (heuristic, spot-only) — {fy_label}\n"
                      f"({Tfy['entry_date'].min().date()} to {Tfy['entry_date'].max().date()})", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Date (weekly entries)", fontsize=11); ax.set_ylabel("Cumulative Return (INR)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        fig.autofmt_xdate(); fig.tight_layout()
        out_png = OUTDIR / f"credit_spread_synthetic_{fy_label}.png"
        fig.savefig(out_png, dpi=150); plt.close(fig)
        print(f"saved -> {out_png}")

    FY = pd.DataFrame(fy_rows)
    pd.set_option("display.width", 200)
    print("\n=== FY-WISE SUMMARY (synthetic, spot-only heuristic P&L) ===")
    print(FY.to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "credit_spread_synthetic_FY_wise_report.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "SYNTHETIC/HEURISTIC P&L, spot-only, uniform across the ENTIRE Mar-2022..22-Jul-2026 span, "
                      "including Oct-2024+ which has REAL option data elsewhere in this project -- deliberately "
                      "recomputed synthetically here for one uniform historical comparison. NOT a substitute for "
                      "the real FINAL weekly credit spread strategy's actual results."},
            {"note": "Net credit is a FIXED ASSUMPTION (87 pts CCS / 67 pts PCS, 200-pt width) -- not derived from "
                      "any real option premium anywhere in this report."},
            {"note": "DTE0 settlement = mean of that day's last 30 one-min spot closes (proxy for the real 30-min "
                      "VWAP settlement calculation, not the official exchange value)."},
            {"note": f"n_trades={len(T)} | n_missing_flagged={len(missing)} | n_ambiguous_same_candle_triggers={len(ambiguous)}"},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        FY.to_excel(w, sheet_name="FY_Summary", index=False)
        T.to_excel(w, sheet_name="All_Synthetic_Trades", index=False)
        if missing:
            pd.DataFrame(missing, columns=["entry_date", "issue"]).to_excel(w, sheet_name="Missing_Flagged", index=False)
        if ambiguous:
            pd.DataFrame(ambiguous, columns=["entry_date", "candle_ts"]).to_excel(w, sheet_name="Ambiguous_Triggers", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 24)

    print(f"\nSaved -> {OUTDIR}/credit_spread_synthetic_FY_wise_report.xlsx")


if __name__ == "__main__":
    main()
