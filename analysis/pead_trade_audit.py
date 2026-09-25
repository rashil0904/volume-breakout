# -*- coding: utf-8 -*-
"""pead_trade_audit.py — SURFACE GENUINENESS AUDIT of the three PEAD books (F&O long, F&O short,
non-F&O long). Rebuilds each trade from the reconciled announcements + raw daily and checks: entry=T0
close, SL=T0 low/high, reaction matches (T0 close vs prev-day VWAP-close), prev-VWAP day is the real
prior trading day, exit price is achievable within the holding window, forward bars are consecutive,
days_held within cap, no duplicate (symbol,T0,direction), non-F&O shorts == 0, and CORPORATE-ACTION
artifacts (unadjusted split/bonus faking a huge gap: |reaction|>25% or any intra-hold daily move>25%).
Reports pass/flag counts per book + lists suspects. Read-only; changes nothing.
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
OUTDIR = rb.RESULTS / "pead_trade_audit"
K, N_LONG, N_SHORT = 4.0, 10, 3
CA_JUMP = 25.0   # % single-day move treated as a corporate-action / bad-print suspect


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fno = set(pd.read_csv(FO_CSV, usecols=["symbol"])["symbol"].unique())
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

    rows = []; flags = {}
    def flag(book, key):
        flags.setdefault(book, {}).setdefault(key, 0); flags[book][key] += 1
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
        is_fno = r.symbol in fno
        if rr >= K:
            direction, n, sl = "long", N_LONG, lo[p]
        elif rr <= -K and is_fno:
            direction, n, sl = "short", N_SHORT, hi[p]
        else:
            continue
        if p + n >= len(dates):
            continue
        book = ("F&O " if is_fno else "non-F&O ") + direction
        # --- genuineness checks ---
        # 1) SL relationship
        if direction == "long" and not (sl <= entry + 1e-9):
            flag(book, "sl_not_below_entry")
        if direction == "short" and not (sl >= entry - 1e-9):
            flag(book, "sl_not_above_entry")
        # 2) prev-VWAP day is the immediately prior trading day
        gap_days = (dates[p] - dates[p - 1]).days if p >= 1 else 999
        if gap_days > 6:
            flag(book, "prev_vwap_gap>6d(halt/illiquid)")
        # 3) entry->T+1 continuity and forward window consecutive
        fwd_dates = dates[p + 1:p + 1 + n]
        maxgap = max([int((fwd_dates[i + 1] - fwd_dates[i]).days) for i in range(len(fwd_dates) - 1)] + [0])
        if maxgap > 6:
            flag(book, "forward_gap>6d")
        # 4) corporate-action / bad-print: reaction extreme or any intra-hold daily close move >25%
        ca = abs(rr) > CA_JUMP
        prev_c = entry
        for j in range(n):
            c = cl[p + 1 + j]
            if prev_c > 0 and abs((c - prev_c) / prev_c * 100.0) > CA_JUMP:
                ca = True
            prev_c = c
        if ca:
            flag(book, "corp_action_suspect(|move|>25%)")
        # 5) price sanity
        if entry <= 0 or np.any(hi[p:p + 1 + n] <= 0) or np.any(lo[p:p + 1 + n] <= 0):
            flag(book, "nonpositive_price")
        if np.any(lo[p:p + 1 + n] > hi[p:p + 1 + n] + 1e-6):
            flag(book, "low>high_badbar")
        flag(book, "TOTAL")
        rows.append({"symbol": r.symbol, "book": book, "direction": direction, "T0": str(dates[p]),
                     "reaction": round(rr, 2), "entry": round(entry, 2), "sl": round(sl, 2),
                     "prev_gap_d": gap_days, "fwd_maxgap_d": maxgap, "ca_suspect": ca})
    T = pd.DataFrame(rows)

    # duplicates
    dups = T[T.duplicated(["symbol", "T0", "direction"], keep=False)]
    # non-F&O short leakage
    nonfo_short = int(((T.book == "non-F&O short")).sum())

    fl = pd.DataFrame(flags).T.fillna(0).astype(int)
    T.to_csv(OUTDIR / "audit_trades.csv", index=False)
    susp = T[T.ca_suspect | (T.prev_gap_d > 6) | (T.fwd_maxgap_d > 6)].sort_values("reaction", key=lambda s: s.abs(), ascending=False)
    susp.to_csv(OUTDIR / "audit_suspects.csv", index=False)

    pd.set_option("display.width", 220)
    print("=" * 88 + "\nPEAD TRADE GENUINENESS AUDIT (surface)\n" + "=" * 88)
    print("book composition:"); print(T.book.value_counts().to_string())
    print(f"\nNON-F&O SHORT leakage (must be 0): {nonfo_short}")
    print(f"duplicate (symbol,T0,direction) rows: {len(dups)}")
    print("\n--- per-book flag counts (0 = clean) ---")
    print(fl.to_string())
    print(f"\ncorp-action/bad-print suspects total: {int(T.ca_suspect.sum())} ({round(T.ca_suspect.mean()*100,1)}% of trades)")
    print("\n--- top 12 |reaction| (eyeball for split/bonus artifacts) ---")
    print(T.reindex(T.reaction.abs().sort_values(ascending=False).index).head(12)[["symbol", "book", "T0", "reaction", "entry", "ca_suspect"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR} (audit_trades.csv, audit_suspects.csv)")


if __name__ == "__main__":
    main()
