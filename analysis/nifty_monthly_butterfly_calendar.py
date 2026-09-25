# -*- coding: utf-8 -*-
"""nifty_monthly_butterfly_calendar.py — NIFTY Monthly Butterfly + Calendar combined strategy.
Entry: first Friday after a monthly expiry's settlement (Thursday of that week if Friday is a holiday),
15:16 using spot/option OPEN. ATM = nearest 50-pt strike to spot. MONTHLY-ONLY cycle (derived: last real
expiry per calendar month, distinct from NIFTY's weekly expiries).
  PE Butterfly @ CURRENT (front) monthly expiry E_cur: +1 ATM PE / -2 (ATM-400) PE / +1 (ATM-800) PE.
  CE Calendar: -1 (ATM+200) CE @ E_cur / +1 (ATM+200) CE @ NEXT monthly expiry E_next.
Combined (all 5 legs) tracked as ONE position: touch-based +-Rs3000 target/SL, else full square-off (ALL
legs, confirmed) at E_cur's expiry 15:15. Lot size sourced PER EXPIRY from live contract metadata (verified
changed across the window: 25 -> 75 -> 65 - not assumed fixed). GROSS Rs P&L.
"""
import sys, glob, os, time, bisect
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import opt_pull_nifty_full as op

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"; OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "nifty_monthly_butterfly_calendar"; OUTDIR.mkdir(parents=True, exist_ok=True)
ENTRY_MOD = 15 * 60 + 16; EXIT_MOD = 15 * 60 + 15; STEP = 50
TARGET, SL = 3000.0, -3000.0
NIFTY_KEY = "NSE_INDEX|Nifty 50"


def lot_size_for(exp_dt):
    e = exp_dt.strftime("%Y-%m-%d")
    cons, sc = op.get(f"{op.BASE}/option/contract?instrument_key={op.enc(NIFTY_KEY)}&expiry_date={e}")
    if cons: return int(cons[0]["lot_size"])
    return None


def leg_series(folder, strike, otype):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{otype}_*.parquet"))
    if not fs: return None
    o = pd.read_parquet(fs[0], columns=["timestamp", "open", "close"]); o["ts"] = pd.to_datetime(o["timestamp"])
    return o.set_index("ts").sort_index()


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True); sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    tdays = sorted(sp["date"].unique()); tdset = set(tdays); spot_end = sp["date"].max()

    all_exp = sorted(pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d"))
    dfm = pd.DataFrame({"date": all_exp}); dfm["month"] = dfm["date"].dt.to_period("M")
    monthly = sorted(dfm.groupby("month")["date"].max().tolist())
    print(f"monthly expiries derived: {len(monthly)}", flush=True)

    lot_cache = {}
    def lots(e):
        if e not in lot_cache:
            lot_cache[e] = lot_size_for(e); time.sleep(0.1)
        return lot_cache[e]

    trades = []; skipped = []
    for i in range(len(monthly) - 2):
        E_prev, E_cur, E_next = monthly[i], monthly[i + 1], monthly[i + 2]
        if E_cur > spot_end or E_next > spot_end: skipped.append((E_prev.date(), "beyond pulled data window")); continue

        # entry day: first Friday on/after E_prev+1day; if that Friday is a holiday, use Thursday of that week
        d0 = pd.Timestamp(E_prev) + pd.Timedelta(days=1)
        days_to_fri = (4 - d0.weekday()) % 7  # Mon=0..Sun=6, Friday=4
        target_fri = d0 + pd.Timedelta(days=days_to_fri)
        if target_fri.normalize() in tdset:
            entry_day = target_fri.normalize()
        else:
            thu = (target_fri - pd.Timedelta(days=1)).normalize()
            if thu in tdset: entry_day = thu
            else: skipped.append((E_prev.date(), f"neither target Friday {target_fri.date()} nor Thursday {thu.date()} is a trading day")); continue

        ed = sp[sp["date"] == entry_day]
        e1516 = ed[ed["mod"] == ENTRY_MOD]
        if e1516.empty: skipped.append((entry_day.date(), "no 15:16 spot candle")); continue
        espot = float(e1516["open"].iloc[0]); atm = round(espot / STEP) * STEP
        t_en = pd.Timestamp(entry_day) + pd.Timedelta(hours=15, minutes=16); t_ex_cap = pd.Timestamp(E_cur) + pd.Timedelta(hours=15, minutes=15)

        lot_cur = lots(E_cur); lot_next = lots(E_next)
        if lot_cur is None or lot_next is None: skipped.append((entry_day.date(), "lot size lookup failed")); continue

        legs_def = [  # (label, strike, otype, expiry, signed_qty, lot)
            ("BFLY buy 1x ATM PE", atm, "PE", E_cur, 1, lot_cur),
            ("BFLY sell 2x (ATM-400) PE", atm - 400, "PE", E_cur, -2, lot_cur),
            ("BFLY buy 1x (ATM-800) PE", atm - 800, "PE", E_cur, 1, lot_cur),
            ("CAL sell 1x (ATM+200) CE @cur", atm + 200, "CE", E_cur, -1, lot_cur),
            ("CAL buy 1x (ATM+200) CE @next", atm + 200, "CE", E_next, 1, lot_next),
        ]
        series = {}; ok = True
        for lbl, K, ot, E, q, lot in legs_def:
            folder = E.strftime("%Y%m%d"); s = leg_series(folder, K, ot)
            if s is None: skipped.append((entry_day.date(), f"missing series {K}{ot}@{E.date()}")); ok = False; break
            series[lbl] = s
        if not ok: continue

        entry_prices = {}
        for lbl, K, ot, E, q, lot in legs_def:
            v = series[lbl]["open"].asof(t_en)
            if pd.isna(v): skipped.append((entry_day.date(), f"nan entry price {lbl}")); ok = False; break
            entry_prices[lbl] = float(v)
        if not ok: continue
        entry_value = sum(q * entry_prices[lbl] * lot for lbl, K, ot, E, q, lot in legs_def)

        # build combined 1-min close series across entry -> t_ex_cap, touch-based check
        idx = None
        for lbl, K, ot, E, q, lot in legs_def:
            s = series[lbl][(series[lbl].index > t_en) & (series[lbl].index <= t_ex_cap)]["close"]
            idx = s.index if idx is None else idx.union(s.index)
        idx = idx.sort_values() if idx is not None else pd.DatetimeIndex([])
        combined = pd.Series(0.0, index=idx)
        for lbl, K, ot, E, q, lot in legs_def:
            px = series[lbl]["close"].reindex(idx).ffill()
            combined = combined + q * px * lot
        pnl_path = combined - entry_value

        exit_time = None; exit_reason = None; exit_value = None
        hit = pnl_path[(pnl_path >= TARGET) | (pnl_path <= SL)]
        if len(hit):
            exit_time = hit.index[0]; exit_value = float(combined.loc[exit_time])
            exit_reason = "target hit" if pnl_path.loc[exit_time] >= TARGET else "SL hit"
        else:
            if len(pnl_path) == 0: skipped.append((entry_day.date(), "no post-entry candles")); continue
            exit_time = pnl_path.index[-1]; exit_value = float(combined.loc[exit_time]); exit_reason = "3:15pm expiry square-off"
        pnl = exit_value - entry_value

        row = {"entry_date": entry_day.date(), "entry_time": "15:16", "ATM": int(atm), "E_prev": E_prev.date(), "E_cur": E_cur.date(), "E_next": E_next.date(),
               "lot_cur": lot_cur, "lot_next": lot_next, "entry_cost_value": round(entry_value, 2),
               "exit_date": pd.Timestamp(exit_time).date(), "exit_time": pd.Timestamp(exit_time).strftime("%H:%M"), "exit_reason": exit_reason,
               "exit_value": round(exit_value, 2), "pnl_rs": round(pnl, 2), "days_held": (pd.Timestamp(exit_time).normalize() - entry_day).days}
        for lbl, K, ot, E, q, lot in legs_def:
            key = lbl.split(" ")[0] + "_" + lbl.split(" ")[-1].replace("@", "")
            row[f"leg[{lbl}]_strike"] = K; row[f"leg[{lbl}]_qty"] = q; row[f"leg[{lbl}]_entry_px"] = round(entry_prices[lbl], 2)
        trades.append(row)
        print(f"  {entry_day.date()} ATM={atm} E_cur={E_cur.date()} E_next={E_next.date()} -> {exit_reason} pnl={round(pnl,2)}", flush=True)

    T = pd.DataFrame(trades); S = pd.DataFrame(skipped, columns=["date", "reason"]) if skipped else pd.DataFrame(columns=["date", "reason"])
    win = round((T.pnl_rs > 0).mean() * 100, 1) if len(T) else 0; tot = round(T.pnl_rs.sum(), 1) if len(T) else 0
    tgt = T[T.exit_reason == "target hit"]; sl = T[T.exit_reason == "SL hit"]; texp = T[T.exit_reason == "3:15pm expiry square-off"]

    summary = pd.DataFrame([
        {"metric": "Strategy", "value": "NIFTY Monthly Butterfly + Calendar (combined single position); entry 15:16 (spot/option OPEN) first Friday after monthly expiry"},
        {"metric": "Structure", "value": "PE Butterfly @E_cur: +1 ATM / -2 (ATM-400) / +1 (ATM-800) ; CE Calendar: -1 (ATM+200)@E_cur / +1 (ATM+200)@E_next"},
        {"metric": "Target / SL", "value": "+Rs3000 / -Rs3000 combined, touch-based every 1-min, closes ALL 5 legs"},
        {"metric": "Time exit", "value": "3:15pm on E_cur's expiry day; ALL legs closed incl. the next-month CE long leg (confirmed, no tail carried)"},
        {"metric": "Lot size", "value": "sourced per-expiry from live contract metadata (confirmed CHANGED across window: 25->75->65, not assumed fixed)"},
        {"metric": "Window", "value": f"{T.entry_date.min() if len(T) else '-'} .. {T.exit_date.max() if len(T) else '-'}"},
        {"metric": "Total trades", "value": len(T)}, {"metric": "Win rate %", "value": win},
        {"metric": "Total P&L (Rs)", "value": tot}, {"metric": "Avg P&L / trade (Rs)", "value": round(T.pnl_rs.mean(), 2) if len(T) else 0},
        {"metric": "Median P&L / trade (Rs)", "value": round(T.pnl_rs.median(), 2) if len(T) else 0},
        {"metric": "Target hit / SL hit / expiry square-off", "value": f"{len(tgt)} / {len(sl)} / {len(texp)}"},
        {"metric": "% target / SL / expiry", "value": f"{round(len(tgt)/len(T)*100,1) if len(T) else 0} / {round(len(sl)/len(T)*100,1) if len(T) else 0} / {round(len(texp)/len(T)*100,1) if len(T) else 0}"},
        {"metric": "Avg days held", "value": round(T.days_held.mean(), 2) if len(T) else 0},
        {"metric": "Skipped cycles", "value": len(S)},
    ])
    with pd.ExcelWriter(OUTDIR / "nifty_monthly_butterfly_calendar.xlsx", engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        (T if len(T) else pd.DataFrame([{"note": "no trades"}])).to_excel(w, sheet_name="Trades", index=False)
        (S if len(S) else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="Skipped", index=False)
    if len(T): T.to_csv(OUTDIR / "nifty_monthly_butterfly_calendar_trades.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 96 + "\nNIFTY MONTHLY BUTTERFLY + CALENDAR\n" + "=" * 96)
    print(f"trades {len(T)} | skipped {len(S)} | win {win}% | TOTAL Rs {tot:,} | avg {round(T.pnl_rs.mean(),2) if len(T) else 0}")
    print(f"target/SL/expiry: {len(tgt)}/{len(sl)}/{len(texp)} | avg days held {round(T.days_held.mean(),2) if len(T) else 0}")
    if len(S): print("\n--- skipped ---"); print(S.to_string(index=False))
    if len(T): print("\n--- trades ---"); print(T[["entry_date", "ATM", "E_cur", "E_next", "exit_date", "exit_reason", "pnl_rs", "days_held"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
