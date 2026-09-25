# -*- coding: utf-8 -*-
"""pead_sl_compare.py — TRIGGER SL vs CLOSE-BASED SL on the finalized PEAD event set (VWAP-ref reaction,
K=4, LONG T+10 / SHORT T+3, during->next-day, entry=T0 close, SL=T0 low/high). Only the stop mechanic
differs. Config1 trigger: intraday touch, fill at SL (gap-through at open). Config2 close-based: fires
only on a CLOSE beyond SL, fills at that breaching close. Reports the 'saved by close-based' set.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
VWAPD = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "pead_sl_compare"
K, N_LONG, N_SHORT, COST = 4.0, 10, 3, 0.20


def build():
    a = pd.read_csv(ANN)
    a["ann_date"] = pd.to_datetime(a["announcement_date"], format="%d-%m-%Y", errors="coerce").dt.date
    a["dtm"] = pd.to_datetime(a["announcement_datetime"], errors="coerce")
    a = a.dropna(subset=["ann_date", "dtm"]).sort_values("dtm").drop_duplicates(["symbol", "quarter"], keep="first")
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "open", "high", "low", "close"]); d["date"] = pd.to_datetime(d["date"]).dt.date
    vw = pd.read_parquet(VWAPD, columns=["symbol", "date", "close"]); vw["date"] = pd.to_datetime(vw["date"]).dt.date
    vmap = {(s, dt): c for s, dt, c in zip(vw["symbol"], vw["date"], vw["close"])}
    sd = {}
    for sym, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date")
        sd[sym] = (g["date"].values, g["open"].values.astype(float), g["high"].values.astype(float),
                   g["low"].values.astype(float), g["close"].values.astype(float), {dt: i for i, dt in enumerate(g["date"].values)})

    rows = []
    for r in a.itertuples():
        s = sd.get(r.symbol)
        if s is None:
            continue
        dates, op, hi, lo, cl, dmap = s
        ps = dmap.get(r.ann_date); pn = int(np.searchsorted(dates, r.ann_date, "right")); pn = pn if pn < len(dates) else None
        p = pn if r.announcement_session in ("post_market", "during_market") else (ps if ps is not None else pn)
        if p is None or p < 1:
            continue
        entry = cl[p]; prev_vwap = vmap.get((r.symbol, dates[p - 1]), np.nan)
        if not (entry > 0 and prev_vwap == prev_vwap and prev_vwap > 0):
            continue
        rr = (entry - prev_vwap) / prev_vwap * 100.0
        if rr >= K:
            direction, n, sl = "long", N_LONG, lo[p]
        elif rr <= -K:
            direction, n, sl = "short", N_SHORT, hi[p]
        else:
            continue
        if p + n >= len(dates):
            continue

        def exit_trigger():
            for j in range(1, n + 1):
                k = p + j
                if direction == "long" and lo[k] <= sl:
                    return (min(sl, op[k]), "gap_through" if op[k] < sl else "sl_trigger", dates[k], j)
                if direction == "short" and hi[k] >= sl:
                    return (max(sl, op[k]), "gap_through" if op[k] > sl else "sl_trigger", dates[k], j)
            return (cl[p + n], "holding_end", dates[p + n], n)

        def exit_close():
            for j in range(1, n + 1):
                k = p + j
                if direction == "long" and cl[k] <= sl:
                    return (cl[k], "close_sl", dates[k], j)
                if direction == "short" and cl[k] >= sl:
                    return (cl[k], "close_sl", dates[k], j)
            return (cl[p + n], "holding_end", dates[p + n], n)

        exp_t, xr_t, xd_t, dh_t = exit_trigger()
        exp_c, xr_c, xd_c, dh_c = exit_close()
        rt = (exp_t - entry) / entry * 100 if direction == "long" else (entry - exp_t) / entry * 100
        rc = (exp_c - entry) / entry * 100 if direction == "long" else (entry - exp_c) / entry * 100
        rows.append({"symbol": r.symbol, "quarter": r.quarter, "T0_date": str(dates[p]), "direction": direction,
                     "reaction_return_pct": round(rr, 2), "entry_price": round(entry, 2), "sl_price": round(sl, 2), "holding_target": f"T+{n}",
                     "trig_exit_date": str(xd_t), "trig_exit_price": round(exp_t, 2), "trig_exit_reason": xr_t, "trig_days": dh_t,
                     "trig_realized_pct": round(rt, 3), "trig_net_pct": round(rt - COST, 3),
                     "close_exit_date": str(xd_c), "close_exit_price": round(exp_c, 2), "close_exit_reason": xr_c, "close_days": dh_c,
                     "close_realized_pct": round(rc, 3), "close_net_pct": round(rc - COST, 3),
                     "stopped_trigger": xr_t in ("sl_trigger", "gap_through"), "stopped_close": xr_c == "close_sl",
                     "year": int(str(dates[p])[:4])})
    return pd.DataFrame(rows).sort_values("T0_date").reset_index(drop=True)


def block(df, cfg, lbl):
    r = df[f"{cfg}_realized_pct"]; nr = df[f"{cfg}_net_pct"]; win = r > 0
    eq = df.sort_values("T0_date")[f"{cfg}_net_pct"].cumsum().values
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    stop = df["stopped_trigger"] if cfg == "trig" else df["stopped_close"]
    days = df["trig_days"] if cfg == "trig" else df["close_days"]
    return {"config": "trigger" if cfg == "trig" else "close_based", "segment": lbl, "n_trades": len(df),
            "win_rate_pct": round(win.mean() * 100, 1), "avg_realized_pct": round(r.mean(), 3), "median_realized_pct": round(r.median(), 3),
            "total_return_sum_pct": round(r.sum(), 1), "avg_net_pct": round(nr.mean(), 3), "expectancy_net_pct": round(nr.mean(), 3),
            "sl_exit_rate_pct": round(stop.mean() * 100, 1), "avg_days_held": round(days.mean(), 1), "max_dd_seq_net_pct": round(dd, 1)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = build()
    print(f"events {len(T)} | long {int((T.direction=='long').sum())} short {int((T.direction=='short').sum())}")

    comp = []
    for lbl, sub in [("LONG", T[T.direction == "long"]), ("SHORT", T[T.direction == "short"]), ("COMBINED", T)]:
        c1 = block(sub, "trig", lbl); c2 = block(sub, "close", lbl)
        dl = {"config": "delta(close-trig)", "segment": lbl, "n_trades": 0}
        for kk in c1:
            if kk not in ("config", "segment") and isinstance(c1[kk], (int, float)):
                dl[kk] = round(c2[kk] - c1[kk], 3)
        comp += [c1, c2, dl]
    COMP = pd.DataFrame(comp)

    # mechanic breakdown
    saved = T[T["stopped_trigger"] & ~T["stopped_close"]]           # trigger stopped, close-based held on
    both = T[T["stopped_trigger"] & T["stopped_close"]]             # both stopped -> fill-quality diff
    mech = {
        "n_stopped_trigger": int(T["stopped_trigger"].sum()), "n_stopped_close": int(T["stopped_close"].sum()),
        "n_saved_by_closebased(trig_stop_only)": len(saved),
        "saved_avg_trig_realized": round(saved["trig_realized_pct"].mean(), 3) if len(saved) else np.nan,
        "saved_avg_closebased_realized": round(saved["close_realized_pct"].mean(), 3) if len(saved) else np.nan,
        "saved_pct_profitable_under_closebased": round((saved["close_realized_pct"] > 0).mean() * 100, 1) if len(saved) else np.nan,
        "n_stopped_both": len(both),
        "both_avg_trig_exit_price": round(both["trig_exit_price"].mean(), 2) if len(both) else np.nan,
        "both_avg_closebased_exit_price": round(both["close_exit_price"].mean(), 2) if len(both) else np.nan,
        "both_avg_trig_realized": round(both["trig_realized_pct"].mean(), 3) if len(both) else np.nan,
        "both_avg_closebased_realized": round(both["close_realized_pct"].mean(), 3) if len(both) else np.nan}
    MECH = pd.DataFrame([mech]).T.reset_index(); MECH.columns = ["metric", "value"]

    trig_cols = ["symbol", "quarter", "T0_date", "direction", "reaction_return_pct", "entry_price", "sl_price", "holding_target",
                 "trig_exit_date", "trig_exit_price", "trig_exit_reason", "trig_days", "trig_realized_pct", "trig_net_pct"]
    close_cols = ["symbol", "quarter", "T0_date", "direction", "reaction_return_pct", "entry_price", "sl_price", "holding_target",
                  "close_exit_date", "close_exit_price", "close_exit_reason", "close_days", "close_realized_pct", "close_net_pct"]
    with pd.ExcelWriter(OUTDIR / "pead_sl_compare.xlsx", engine="openpyxl") as w:
        COMP.to_excel(w, sheet_name="comparison_summary", index=False)
        MECH.to_excel(w, sheet_name="mechanic_breakdown", index=False)
        T[trig_cols].to_excel(w, sheet_name="trigger_trades", index=False)
        T[close_cols].to_excel(w, sheet_name="closebased_trades", index=False)
        saved[["symbol", "T0_date", "direction", "sl_price", "trig_exit_price", "trig_realized_pct", "close_exit_price", "close_realized_pct"]].to_excel(w, sheet_name="saved_by_closebased", index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 100 + "\nTRIGGER SL vs CLOSE-BASED SL  (net 0.20%)\n" + "=" * 100)
    print(COMP[["segment", "config", "n_trades", "win_rate_pct", "avg_realized_pct", "total_return_sum_pct", "avg_net_pct",
                "sl_exit_rate_pct", "avg_days_held", "max_dd_seq_net_pct"]].to_string(index=False))
    print("\n--- MECHANIC BREAKDOWN ---")
    print(MECH.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
