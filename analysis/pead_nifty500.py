# -*- coding: utf-8 -*-
"""pead_nifty500.py — PEAD backtest over the Nifty-500 universe, ADDITIVE to the F&O pipeline. Finalized
logic UNCHANGED: reaction = (T0 close - prev-day VWAP-close)/prev_vwap*100; K=4%; during/post -> next-day
T0, pre -> same-day; entry = T0 close; hold T+10 long / T+3 short; CLOSE-BASIS SL at T0 low/high; SHORT
profit target 8% (trigger); LONG no target. Direction gate: F&O names -> long+short; NON-F&O names ->
LONG ONLY (shorts are never generated, not filtered post-hoc). Reports F&O / non-F&O / combined books.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements_combined.csv"
FO_CSV = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
UNIV = rb.BASE / "data" / "nifty500_universe.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
VWAPD = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "pead_nifty500"
K, N_LONG, N_SHORT, COST, SHORT_TGT = 4.0, 10, 3, 0.20, 8.0
IS_YEARS = {2022, 2023, 2024}


def build(fno_set):
    a = pd.read_csv(ANN, dtype=str)
    a["ann_date"] = pd.to_datetime(a["announcement_date"], format="%d-%m-%Y", errors="coerce").dt.date
    a["dtm"] = pd.to_datetime(a["announcement_datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
    a = a.dropna(subset=["ann_date", "dtm"]).sort_values("dtm").drop_duplicates(["symbol", "quarter"], keep="first")
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "open", "high", "low", "close"]); d["date"] = pd.to_datetime(d["date"]).dt.date
    vw = pd.read_parquet(VWAPD, columns=["symbol", "date", "close"]); vw["date"] = pd.to_datetime(vw["date"]).dt.date
    vmap = {(s, dt): c for s, dt, c in zip(vw["symbol"], vw["date"], vw["close"])}
    sd = {}
    for sym, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date")
        sd[sym] = (g["date"].values, g["open"].values.astype(float), g["high"].values.astype(float),
                   g["low"].values.astype(float), g["close"].values.astype(float), {dt: i for i, dt in enumerate(g["date"].values)})
    ev = []; skipped_no_price = set(); seen = set()
    for r in a.itertuples():
        s = sd.get(r.symbol)
        if s is None:
            skipped_no_price.add(r.symbol); continue
        dates, op, hi, lo, cl, dmap = s
        ps = dmap.get(r.ann_date); pn = int(np.searchsorted(dates, r.ann_date, "right")); pn = pn if pn < len(dates) else None
        p = pn if r.announcement_session in ("post_market", "during_market") else (ps if ps is not None else pn)
        if p is None or p < 1:
            continue
        if (dates[p] - dates[p - 1]).days > 6:                # GUARD1: stale prev-VWAP (data gap)
            continue
        key = (r.symbol, int(p))                              # GUARD2: de-dup by (symbol, T0)
        if key in seen:
            continue
        seen.add(key)
        entry = cl[p]; prev_vwap = vmap.get((r.symbol, dates[p - 1]), np.nan)
        if not (entry > 0 and prev_vwap == prev_vwap and prev_vwap > 0):
            continue
        rr = (entry - prev_vwap) / prev_vwap * 100.0
        if abs(rr) > 50.0:                                    # GUARD3: corporate-action / bad-print
            continue
        is_fno = r.symbol in fno_set
        if rr >= K:
            direction, n, sl = "long", N_LONG, lo[p]
        elif rr <= -K and is_fno:                               # SHORTS ONLY for F&O names (never generated for non-F&O)
            direction, n, sl = "short", N_SHORT, hi[p]
        else:
            continue
        if p + n >= len(dates):
            continue
        ev.append({"symbol": r.symbol, "quarter": r.quarter, "T0_date": str(dates[p]), "direction": direction,
                   "book": "F&O" if is_fno else "non-F&O", "reaction_return": round(rr, 2), "entry": entry, "sl": sl, "n": n,
                   "year": int(str(dates[p])[:4]),
                   "fo": op[p + 1:p + 1 + n], "fh": hi[p + 1:p + 1 + n], "fl": lo[p + 1:p + 1 + n], "fc": cl[p + 1:p + 1 + n]})
    return ev, skipped_no_price


def exit_trade(e):
    entry, sl, n, dr = e["entry"], e["sl"], e["n"], e["direction"]
    fo, fh, fl, fc = e["fo"], e["fh"], e["fl"], e["fc"]
    if dr == "long":                                            # close-basis SL, no target
        for j in range(n):
            if fc[j] <= sl:
                return fc[j], "close_sl", j + 1
        return fc[n - 1], "holding_end", n
    tgt = entry * (1 - SHORT_TGT / 100.0)                        # short: trigger target 8% then close-basis SL
    for j in range(n):
        if fl[j] <= tgt:                                        # intraday target touch (gap-through fills at open)
            return (min(tgt, fo[j]), "target", j + 1)
        if fc[j] >= sl:                                        # close beyond T0 high -> stop
            return fc[j], "close_sl", j + 1
    return fc[n - 1], "holding_end", n


def book_stats(df, label):
    r = df["net_return"]; g = df["gross_return"]
    eq = df.sort_values("T0_date")["net_return"].cumsum().values
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    ism = df["year"].isin(IS_YEARS)
    return {"book": label, "n_trades": len(df), "n_long": int((df.direction == "long").sum()), "n_short": int((df.direction == "short").sum()),
            "total_return_net": round(r.sum(), 1), "avg_net": round(r.mean(), 3) if len(df) else np.nan,
            "win_rate_pct": round((g > 0).mean() * 100, 1) if len(df) else np.nan, "avg_days_held": round(df.days_held.mean(), 1) if len(df) else np.nan,
            "sl_rate_pct": round((df.exit_reason == "close_sl").mean() * 100, 1) if len(df) else np.nan,
            "tgt_rate_pct": round((df.exit_reason == "target").mean() * 100, 1) if len(df) else np.nan, "max_dd_net": round(dd, 1),
            "IS_total": round(r[ism].sum(), 1), "OOS_total": round(r[~ism].sum(), 1), "n_symbols": df.symbol.nunique()}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fno_set = set(pd.read_csv(FO_CSV, usecols=["symbol"])["symbol"].unique())
    ev, skipped = build(fno_set)
    rows = []
    for e in ev:
        exp, reason, dh = exit_trade(e)
        g = (exp - e["entry"]) / e["entry"] * 100 if e["direction"] == "long" else (e["entry"] - exp) / e["entry"] * 100
        rows.append({**{k: e[k] for k in ("symbol", "quarter", "T0_date", "direction", "book", "reaction_return", "entry", "sl", "year")},
                     "exit_price": round(exp, 2), "exit_reason": reason, "days_held": dh,
                     "gross_return": round(g, 3), "net_return": round(g - COST, 3)})
    T = pd.DataFrame(rows).sort_values("T0_date").reset_index(drop=True)

    books = pd.DataFrame([book_stats(T[T.book == "F&O"], "F&O (long+short)"),
                          book_stats(T[T.book == "non-F&O"], "non-F&O (long only)"),
                          book_stats(T, "COMBINED")])
    # long-only cross-cut (comparable across books)
    L = T[T.direction == "long"]
    longbooks = pd.DataFrame([book_stats(L[L.book == "F&O"], "F&O longs"),
                              book_stats(L[L.book == "non-F&O"], "non-F&O longs"),
                              book_stats(L, "ALL longs")])

    # coverage caveat: announcements per symbol, thin small-caps
    u = pd.read_csv(UNIV)
    n500 = set(u["symbol"]); traded = set(T.symbol)
    n500_traded = len(n500 & traded)
    covq = pd.read_csv(ANN, dtype=str).groupby("symbol")["quarter"].nunique()
    nonfo_syms = set(u[~u.fno_eligible]["symbol"])
    thin = sorted([s for s in nonfo_syms if covq.get(s, 0) <= 4])

    with pd.ExcelWriter(OUTDIR / "pead_nifty500.xlsx", engine="openpyxl") as w:
        books.to_excel(w, sheet_name="books_summary", index=False)
        longbooks.to_excel(w, sheet_name="longs_only_compare", index=False)
        T.to_excel(w, sheet_name="trades", index=False)
        pd.DataFrame({"thin_coverage_nonfo(<=4q)": thin}).to_excel(w, sheet_name="coverage_caveat", index=False)
    T.to_csv(OUTDIR / "pead_nifty500_trades.csv", index=False)

    pd.set_option("display.width", 240)
    print("=" * 96 + "\nPEAD NIFTY-500 (additive; F&O long+short, non-F&O LONG-only)\n" + "=" * 96)
    print(books[["book", "n_trades", "n_long", "n_short", "total_return_net", "avg_net", "win_rate_pct",
                 "sl_rate_pct", "tgt_rate_pct", "avg_days_held", "max_dd_net", "IS_total", "OOS_total", "n_symbols"]].to_string(index=False))
    print("\n--- LONGS-ONLY comparison (apples-to-apples across books) ---")
    print(longbooks[["book", "n_trades", "total_return_net", "avg_net", "win_rate_pct", "avg_days_held", "max_dd_net", "n_symbols"]].to_string(index=False))
    print(f"\nCOVERAGE: Nifty500 names that produced >=1 trade: {n500_traded}/500 | skipped (no price data): {len(skipped)} {sorted(skipped)[:12]}")
    print(f"DATA-QUALITY CAVEAT: {len(thin)} non-F&O names have <=4 quarters of BSE results (thin/illiquid coverage) -> their signal counts are understated.")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
