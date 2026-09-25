# -*- coding: utf-8 -*-
"""pead_final.py — FINALIZED post-earnings drift trade list + summary. TRIGGER SL (intraday touch fills
at SL level; gap-through fills at open, flagged). LONG K=4 hold T+10 SL=T0 low; SHORT K=4 hold T+3
SL=T0 high. during->next-day (post->next, pre->same). Entry=T0 close. Cost 0.20% round-trip; nominal
Rs1L/trade for pnl_inr (equal-weight; real book faces ~16-22 concurrent positions in earnings season).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
OUTDIR = rb.RESULTS / "pead_final"
K, N_LONG, N_SHORT = 4.0, 10, 3
COST, NOTIONAL = 0.20, 100_000


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(ANN)
    a["ann_date"] = pd.to_datetime(a["announcement_date"], format="%d-%m-%Y", errors="coerce").dt.date
    a["dtm"] = pd.to_datetime(a["announcement_datetime"], errors="coerce")
    a = a.dropna(subset=["ann_date", "dtm"]).sort_values("dtm").drop_duplicates(["symbol", "quarter"], keep="first")
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "open", "high", "low", "close"])
    d["date"] = pd.to_datetime(d["date"]).dt.date
    sd = {}
    for sym, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date")
        sd[sym] = (g["date"].values, g["open"].values.astype(float), g["high"].values.astype(float),
                   g["low"].values.astype(float), g["close"].values.astype(float), {dt: i for i, dt in enumerate(g["date"].values)})

    rows = []; n_excl_incomplete = 0
    for r in a.itertuples():
        s = sd.get(r.symbol)
        if s is None:
            continue
        dates, op, hi, lo, cl, dmap = s
        ps = dmap.get(r.ann_date); pn = int(np.searchsorted(dates, r.ann_date, "right")); pn = pn if pn < len(dates) else None
        if r.announcement_session == "post_market":
            p = pn
        elif r.announcement_session == "pre_market":
            p = ps if ps is not None else pn
        elif r.announcement_session == "during_market":
            p = pn                                              # next-day
        else:
            continue
        if p is None or p < 1:
            continue
        entry = cl[p]; prevc = cl[p - 1]
        if not (entry > 0 and prevc > 0):
            continue
        rr = (entry - prevc) / prevc * 100.0
        if rr >= K:
            direction, n, sl = "long", N_LONG, lo[p]
        elif rr <= -K:
            direction, n, sl = "short", N_SHORT, hi[p]
        else:
            continue
        if p + n >= len(dates):                                 # incomplete forward window -> exclude
            n_excl_incomplete += 1; continue
        exit_price = exit_reason = exit_date = days = None
        for j in range(1, n + 1):
            k = p + j
            if direction == "long" and lo[k] <= sl:
                exit_price = min(sl, op[k]); exit_reason = "gap_through" if op[k] < sl else "sl_trigger"
                exit_date = dates[k]; days = j; break
            if direction == "short" and hi[k] >= sl:
                exit_price = max(sl, op[k]); exit_reason = "gap_through" if op[k] > sl else "sl_trigger"
                exit_date = dates[k]; days = j; break
        if exit_price is None:
            exit_price = cl[p + n]; exit_reason = "holding_end"; exit_date = dates[p + n]; days = n
        realized = (exit_price - entry) / entry * 100 if direction == "long" else (entry - exit_price) / entry * 100
        shares = int(NOTIONAL // entry); cap = shares * entry
        gross_pnl = shares * (exit_price - entry) if direction == "long" else shares * (entry - exit_price)
        net_pnl = gross_pnl - COST / 100 * cap
        rows.append({"symbol": r.symbol, "quarter": r.quarter, "announcement_date": str(r.ann_date),
                     "announcement_time": r.announcement_time, "announcement_session": r.announcement_session,
                     "T0_date": str(dates[p]), "prev_close": round(prevc, 2), "T0_close": round(entry, 2),
                     "reaction_return_pct": round(rr, 2), "direction": direction, "entry_price": round(entry, 2),
                     "sl_price": round(sl, 2), "holding_target": f"T+{n}", "exit_date": str(exit_date),
                     "exit_price": round(exit_price, 2), "exit_reason": exit_reason, "days_held": days,
                     "realized_return_pct": round(realized, 3), "net_return_pct": round(realized - COST, 3),
                     "shares": shares, "capital_deployed": round(cap, 0), "net_pnl_inr": round(net_pnl, 0),
                     "year": int(str(dates[p])[:4])})
    T = pd.DataFrame(rows).sort_values("T0_date").reset_index(drop=True)
    T["equity_net_cum_inr"] = T["net_pnl_inr"].cumsum().round(0)
    print(f"trades {len(T)} | long {int((T.direction=='long').sum())} short {int((T.direction=='short').sum())} | excluded incomplete-window {n_excl_incomplete}")

    def block(df, lbl):
        if len(df) == 0:
            return {"segment": lbl, "n_trades": 0}
        r = df["realized_return_pct"]; nr = df["net_return_pct"]; win = r > 0
        eq = df.sort_values("T0_date")["net_return_pct"].cumsum().values
        dd = float((eq - np.maximum.accumulate(eq)).min())
        return {"segment": lbl, "n_trades": len(df), "win_rate_pct": round(win.mean() * 100, 1),
                "avg_realized_pct": round(r.mean(), 3), "median_realized_pct": round(r.median(), 3),
                "total_return_sum_pct": round(r.sum(), 1), "avg_net_pct": round(nr.mean(), 3),
                "expectancy_net_pct": round(nr.mean(), 3), "total_net_pnl_inr": round(df["net_pnl_inr"].sum(), 0),
                "sl_trigger_rate_pct": round((df["exit_reason"].isin(["sl_trigger", "gap_through"])).mean() * 100, 1),
                "n_gap_through": int((df["exit_reason"] == "gap_through").sum()), "avg_days_held": round(df["days_held"].mean(), 1),
                "avg_winner_pct": round(r[win].mean(), 3) if win.any() else 0.0,
                "avg_loser_pct": round(r[~win].mean(), 3) if (~win).any() else 0.0,
                "largest_win_pct": round(r.max(), 2), "largest_loss_pct": round(r.min(), 2),
                "max_dd_seq_net_pct": round(dd, 1)}
    L, S = T[T.direction == "long"], T[T.direction == "short"]
    SUM = pd.DataFrame([block(L, "LONG (T+10)"), block(S, "SHORT (T+3)"), block(T, "COMBINED")])
    by_sess = pd.DataFrame([{**block(T[(T.direction == dr) & (T.announcement_session == se)], f"{dr}/{se}")}
                            for dr in ["long", "short"] for se in ["post_market", "during_market", "pre_market"]])
    yr = T.groupby(["year", "direction"]).agg(n=("symbol", "size"), avg_net=("net_return_pct", "mean"),
                                              total_net=("net_return_pct", "sum"), win=("realized_return_pct", lambda x: round((x > 0).mean() * 100, 1))).round(3).reset_index()

    with pd.ExcelWriter(OUTDIR / "pead_final_trades.xlsx", engine="openpyxl") as w:
        T.to_excel(w, sheet_name="all_trades", index=False)
        SUM.to_excel(w, sheet_name="summary", index=False)
        by_sess.to_excel(w, sheet_name="by_session", index=False)
        yr.to_excel(w, sheet_name="per_year", index=False)
    T.to_csv(OUTDIR / "pead_final_trades.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 96 + "\nPEAD FINAL — trigger SL (K=4; long T+10 / short T+3; next-day; net 0.20%)\n" + "=" * 96)
    cols = ["segment", "n_trades", "win_rate_pct", "avg_realized_pct", "median_realized_pct", "total_return_sum_pct",
            "avg_net_pct", "total_net_pnl_inr", "sl_trigger_rate_pct", "n_gap_through", "avg_days_held",
            "avg_winner_pct", "avg_loser_pct", "max_dd_seq_net_pct"]
    print(SUM[cols].to_string(index=False))
    print("\n--- BY SESSION ---"); print(by_sess[["segment", "n_trades", "avg_net_pct", "total_return_sum_pct", "sl_trigger_rate_pct"]].to_string(index=False))
    print("\n--- PER YEAR ---"); print(yr.to_string(index=False))
    print("\n--- first 8 trades ---")
    print(T[["symbol", "T0_date", "direction", "reaction_return_pct", "entry_price", "sl_price", "exit_date",
             "exit_price", "exit_reason", "days_held", "net_return_pct"]].head(8).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
