# -*- coding: utf-8 -*-
"""nifty_btst_daily_proxy_2020_2021.py — DAILY-GRANULARITY PROXY of the NIFTY BTST Close Direction
strategy for Jan 2020 - Dec 2021, using data/nifty_daily_2020_2021.csv (no intraday data exists for this
period). This is NOT the same mechanics as the intraday 3:20pm entry / 9:17am exit version used for
later periods -- entry uses the DAILY CLOSE (proxy for 3:20pm), exit uses the NEXT DAY'S DAILY OPEN
(proxy for ~9:15am, NOT the 9:17 close reference used elsewhere). Purely exploratory/directional --
not a tradeable backtest, no options pricing involved at all (there isn't even a spot-1-min layer here,
just daily bars).

FILTERS: India VIX DAILY data now pulled for this period (data/india_vix_daily_2020_2021.csv) -- VIX[17,19]
filter IS applied, using that day's VIX DAILY CLOSE as the entry-VIX proxy (matching this proxy's own
entry reference = daily close). NOTE this is NOT the same convention as the later intraday-matched periods
(which use the exact 1-min VIX candle at the true entry minute) -- flagged, not equivalent, since only
daily VIX granularity exists for 2020-2021. DTE-1 exclusion applied via a Thursday-expiry ANALOG calendar
(holiday-adjusted, no real options chain involved -- there isn't one for this period).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DAILY = rb.BASE / "data" / "nifty_daily_2020_2021.csv"
VIXF = rb.BASE / "data" / "india_vix_daily_2020_2021.csv"
VIX_LO, VIX_HI = 17.0, 19.0
OUTDIR = rb.RESULTS / "btst_daily_proxy_2020_2021"; OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    d = pd.read_csv(DAILY, parse_dates=["timestamp"])
    d["date"] = d["timestamp"].dt.normalize()
    d = d.sort_values("date").reset_index(drop=True)

    print(f"daily bars loaded: {len(d)} | {d['date'].min().date()} .. {d['date'].max().date()}")

    vx = pd.read_csv(VIXF, parse_dates=["timestamp"])
    vx["date"] = vx["timestamp"].dt.normalize()
    vix_close = vx.set_index("date")["close"]

    # ---- DTE-1 ANALOG: Thursday-expiry calendar, holiday-adjusted to the prior actual trading day,
    # built from this daily dataset's own trading days (same convention as the Mar22-Sep24 proxy;
    # confirmed via web search elsewhere in this project that Nifty expiry was Thursday throughout
    # 2020-2021 too, no mid-window change -- that only happened 2025+). ----
    all_tdays_set = set(d["date"])
    thursdays = pd.date_range(d["date"].min() - pd.Timedelta(days=30), d["date"].max() + pd.Timedelta(days=10), freq="W-THU")
    expiries = []
    for th in thursdays:
        dd = th
        while dd not in all_tdays_set and dd >= th - pd.Timedelta(days=6):
            dd -= pd.Timedelta(days=1)
        if dd in all_tdays_set:
            expiries.append(dd)
    expiries = sorted(set(expiries))

    rows = []
    for i in range(len(d) - 1):
        D = d.iloc[i]; Dn = d.iloc[i + 1]
        entry_close = float(D["close"]); day_open = float(D["open"]); exit_open = float(Dn["open"])
        direction = "RED" if entry_close < day_open else "GREEN"
        raw_move = exit_open - entry_close
        dir_consistent = raw_move if direction == "GREEN" else -raw_move

        fut_exp = [e for e in expiries if e >= D["date"]]
        dte1_excluded = False
        if fut_exp:
            dte = (fut_exp[0].normalize() - D["date"].normalize()).days
            dte1_excluded = (dte == 1)

        entry_vix = float(vix_close.get(D["date"], np.nan))
        vix_excluded = (not np.isnan(entry_vix)) and (VIX_LO <= entry_vix <= VIX_HI)

        rows.append({
            "entry_date": D["date"].date(), "direction": direction, "day_open": round(day_open, 2),
            "entry_close": round(entry_close, 2), "exit_date": Dn["date"].date(), "exit_open": round(exit_open, 2),
            "points_move": round(dir_consistent, 2), "favorable": dir_consistent > 0,
            "dte1_analog_excluded": dte1_excluded,
            "entry_vix_daily_close": round(entry_vix, 2) if not np.isnan(entry_vix) else None,
            "vix_excluded": vix_excluded,
        })

    T_all = pd.DataFrame(rows)
    n_dte1_excluded = int(T_all["dte1_analog_excluded"].sum())
    n_vix_excluded = int(T_all["vix_excluded"].sum())
    T = T_all[(~T_all["dte1_analog_excluded"]) & (~T_all["vix_excluded"])].reset_index(drop=True)
    T["cum_points"] = T["points_move"].cumsum().round(2)
    T["month"] = pd.to_datetime(T["entry_date"]).dt.to_period("M").astype(str)

    n = len(T); n_fav = int(T["favorable"].sum()); n_unfav = n - n_fav
    pct_fav = round(n_fav / n * 100, 2)
    avg_fav = round(T.loc[T["favorable"], "points_move"].mean(), 2)
    avg_unfav = round(T.loc[~T["favorable"], "points_move"].mean(), 2)
    total_pts = round(T["points_move"].sum(), 1)
    avg_pts = round(T["points_move"].mean(), 2)
    median_pts = round(T["points_move"].median(), 2)

    red = T[T.direction == "RED"]; grn = T[T.direction == "GREEN"]

    print("\n" + "=" * 100)
    print("NIFTY BTST DAILY-GRANULARITY PROXY (Jan-2020..Dec-2021) -- EXPLORATORY, NOT A REAL BACKTEST")
    print("Entry = daily CLOSE (proxy for 3:20pm) | Exit = next-day daily OPEN (proxy for ~9:15am, NOT 9:17 close)")
    print(f"VIX filter: APPLIED -- VIX[{VIX_LO:.0f},{VIX_HI:.0f}] excluded using DAILY CLOSE (only granularity available for 2020-2021, NOT the intraday-matched convention used elsewhere) -- {n_vix_excluded} days excluded")
    print(f"DTE-1 filter: APPLIED via Thursday-expiry ANALOG calendar (holiday-adjusted) -- {n_dte1_excluded} days excluded")
    print("=" * 100)
    print(f"n trades (before either filter): {len(T_all)}")
    print(f"n trades (after both filters): {n}")
    print(f"favorable: {n_fav} ({pct_fav}%) | unfavorable: {n_unfav} ({round(100-pct_fav,2)}%)")
    print(f"avg favorable move: +{avg_fav} pts | avg unfavorable move: {avg_unfav} pts")
    print(f"total points: {total_pts} | avg/trade: {avg_pts} | median/trade: {median_pts}")
    print(f"\nRED: n={len(red)}, favorable {round(red.favorable.mean()*100,1)}% | GREEN: n={len(grn)}, favorable {round(grn.favorable.mean()*100,1)}%")

    MON = T.groupby("month").agg(n_trades=("points_move", "size"), points=("points_move", "sum")).reset_index()
    MON["points"] = MON["points"].round(1); MON["cum_points"] = MON["points"].cumsum().round(1)
    print("\n--- MONTHLY ---")
    pd.set_option("display.width", 200)
    print(MON.to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "nifty_btst_daily_proxy_2020_2021_DTE1_VIX_excluded.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "LABEL: DAILY-GRANULARITY PROXY, Jan-2020..Dec-2021. NOT the intraday 3:20pm/9:17am mechanics used for later periods -- entry=daily CLOSE, exit=next-day daily OPEN. Exploratory/directional only, no options pricing, not a tradeable backtest."},
            {"note": f"VIX filter: APPLIED using india_vix_daily_2020_2021.csv DAILY CLOSE as the entry-VIX proxy (matches this proxy's own daily-close entry reference). NOT the same convention as later periods, which use the true intraday 1-min VIX candle at the exact entry minute -- flagged, not equivalent. {n_vix_excluded} of {len(T_all)} days excluded."},
            {"note": f"DTE-1 filter: APPLIED as an ANALOG -- Thursday-expiry calendar (holiday-adjusted to prior trading day, same convention as the Mar22-Sep24 proxy), no real options chain involved. {n_dte1_excluded} of {len(T_all)} days excluded."},
            {"note": f"n_trades={n} (of {len(T_all)} before filters) | favorable={n_fav} ({pct_fav}%) | total_points={total_pts} | avg_pts_per_trade={avg_pts} | median_pts_per_trade={median_pts}"},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        summary = pd.DataFrame([
            {"metric": "n_trades", "value": n}, {"metric": "n_favorable", "value": n_fav}, {"metric": "n_unfavorable", "value": n_unfav},
            {"metric": "pct_favorable", "value": pct_fav}, {"metric": "avg_favorable_move", "value": avg_fav}, {"metric": "avg_unfavorable_move", "value": avg_unfav},
            {"metric": "total_points", "value": total_pts}, {"metric": "avg_points_per_trade", "value": avg_pts}, {"metric": "median_points_per_trade", "value": median_pts},
            {"metric": "RED_n", "value": len(red)}, {"metric": "RED_favorable_pct", "value": round(red.favorable.mean()*100,1)},
            {"metric": "GREEN_n", "value": len(grn)}, {"metric": "GREEN_favorable_pct", "value": round(grn.favorable.mean()*100,1)},
        ])
        summary.to_excel(w, sheet_name="Summary", index=False)
        T.to_excel(w, sheet_name="Per_Trade_Log", index=False)
        MON.to_excel(w, sheet_name="Monthly", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)

    print(f"\nSaved -> {OUTDIR}/nifty_btst_daily_proxy_2020_2021_DTE1_VIX_excluded.xlsx")


if __name__ == "__main__":
    main()
