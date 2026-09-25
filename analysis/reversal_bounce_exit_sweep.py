# -*- coding: utf-8 -*-
"""
reversal_bounce_exit_sweep.py
=============================
STANDALONE strategy (separate from volume-breakout): 2-day reversal (down-then-green-bounce)
long entry, next-day 15-min exit-time sweep. Universe mcap ₹1,500-5,000 Cr (entry-day cap),
₹5L pool / ₹1L per trade. NO volume/lookback/daily-return filters — pure price pattern + mcap.

ENTRY (daily candles, needs 3 consecutive trading days D0,D1,D2):
  D1 down  : close[D1] < close[D0]
  D2 green : close[D2] > open[D1]  AND  close[D2] > open[D2]
  -> ENTER LONG at D2 CLOSE (entry_price = close[D2]); mcap on D2 in band.
EXIT next trading day D3: 100% at each 15-min open, 09:30..15:00 (23 times).
  pnl = shares × (open_T_D3 − entry); shares = 100000 // entry; cost 0.23%.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "reversal_bounce"
BASE_POOL, PER_TRADE, EXPENSE = 500_000, 100_000, 0.0023
DAY_OPEN_HM, DAY_CLOSE_HM = 555, 915                 # 09:15 open, 15:15 close
EXIT_HMS = list(range(570, 901, 15))                 # 09:30 .. 15:00 (23 exit times on D3)
SESSION_HMS = list(range(555, 916, 15))


def hm_lbl(hm): return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv", usecols=["symbol", "date"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    eligible = set(zip(diag["symbol"].values, diag["date"].values))
    symbols = sorted(diag["symbol"].unique())
    print(f"Eligible (mcap ₹1,500-5,000 Cr) symbol-days: {len(eligible):,} over {len(symbols)} symbols")

    entries = []            # (entry_price, D3_opens[23])
    n_signals = 0
    signal_dates = []       # D2 dates (for signals/day)
    dup_pairs = 0           # same stock, consecutive-D2 overlap
    t0 = time.time()
    for si, sym in enumerate(symbols, 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        raw = raw[raw["hm"].isin(SESSION_HMS)]
        op = raw.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=SESSION_HMS)
        cl = raw.pivot_table(index="date", columns="hm", values="close", aggfunc="last").reindex(columns=SESSION_HMS)
        dates = sorted(op.index)
        day_open = op[DAY_OPEN_HM]
        day_close = cl[DAY_CLOSE_HM].where(cl[DAY_CLOSE_HM].notna(), cl.ffill(axis=1).iloc[:, -1])
        exit_cols = op[EXIT_HMS]
        last_sig_i = -10
        for i in range(2, len(dates) - 1):                 # D2=i, need D3=i+1
            d0, d1, d2, d3 = dates[i-2], dates[i-1], dates[i], dates[i+1]
            c0, c1 = day_close.get(d0, np.nan), day_close.get(d1, np.nan)
            o1 = day_open.get(d1, np.nan)
            c2, o2 = day_close.get(d2, np.nan), day_open.get(d2, np.nan)
            if not (c1 < c0):                              # D1 down
                continue
            if not (c2 > o1 and c2 > o2):                  # D2 green bounce over D1 open
                continue
            if (sym, d2) not in eligible:                  # mcap band on entry day
                continue
            entry = c2
            if not (entry == entry and entry > 0):
                continue
            d3open = exit_cols.loc[d3].values if d3 in exit_cols.index else np.full(len(EXIT_HMS), np.nan)
            entries.append((entry, d3open))
            n_signals += 1
            signal_dates.append(d2)
            if i - last_sig_i == 1:
                dup_pairs += 1
            last_sig_i = i
        if si % 150 == 0:
            el = time.time() - t0
            print(f"  …{si}/{len(symbols)} ({n_signals} signals, {el:.0f}s)")

    if not entries:
        raise SystemExit("No signals generated.")
    entry_price = np.array([e[0] for e in entries], float)
    d3opens = np.vstack([e[1] for e in entries]).astype(float)   # (n, 23)
    shares = np.floor(PER_TRADE / entry_price)
    ok_sh = shares > 0
    entry_price, d3opens, shares = entry_price[ok_sh], d3opens[ok_sh], shares[ok_sh]
    cap = shares * entry_price
    print(f"\nTotal signals: {n_signals} | valid (shares>0): {len(entry_price)}")

    rows = []
    for j, hm in enumerate(EXIT_HMS):
        px = d3opens[:, j]
        v = ~np.isnan(px)
        e, s, c, x = entry_price[v], shares[v], cap[v], px[v]
        nv = len(e)
        if nv == 0:
            continue
        pnl = s * (x - e); ret = (x - e) / e * 100
        npnl = pnl - c * EXPENSE; nret = ret - EXPENSE * 100
        rows.append({
            "exit_time": hm_lbl(hm), "n_trades": nv,
            "gross_win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "gross_avg_return_pct": round(float(ret.mean()), 4),
            "gross_median_return_pct": round(float(np.median(ret)), 4),
            "gross_total_return_fixedbase_pct": round(float(pnl.sum()) / BASE_POOL * 100, 4),
            "gross_total_pnl_inr": round(float(pnl.sum()), 0),
            "net_win_rate_pct": round(float((npnl > 0).mean() * 100), 2),
            "net_avg_return_pct": round(float(nret.mean()), 4),
            "net_median_return_pct": round(float(np.median(nret)), 4),
            "net_total_return_fixedbase_pct": round(float(npnl.sum()) / BASE_POOL * 100, 4),
            "net_total_pnl_inr": round(float(npnl.sum()), 0),
            "avg_capital_deployed_per_trade": round(float(c.mean()), 0)})
    tab = pd.DataFrame(rows).sort_values("net_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)

    # capacity stats
    sig_by_day = pd.Series(signal_dates).value_counts()
    n_days = sig_by_day.shape[0]
    best = tab.iloc[0]

    # chart (exit_time chronological)
    ch = tab.set_index("exit_time").reindex([hm_lbl(h) for h in EXIT_HMS])
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ch["net_total_return_fixedbase_pct"]]
    ax.bar(ch.index, ch["net_total_return_fixedbase_pct"], color=colors)
    ax.axhline(0, color="#333", lw=.8)
    ax.axvline(list(ch.index).index(best["exit_time"]), ls=":", color="#1f77b4", alpha=.6)
    ax.set_xlabel("D3 exit time"); ax.set_ylabel("net total_return_fixedbase (%)")
    ax.set_title("2-day reversal bounce: next-day (D3) exit-time sweep — net total return")
    plt.xticks(rotation=60); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "reversal_bounce_exit_sweep.png", dpi=120); plt.close(fig)

    with pd.ExcelWriter(OUTDIR / "reversal_bounce_exit_sweep.xlsx", engine="openpyxl") as w:
        tab.to_excel(w, sheet_name="exit_time_sweep", index=False)
        sig_by_day.rename("n_signals").rename_axis("date").reset_index().to_excel(w, sheet_name="signals_by_day", index=False)

    pd.set_option("display.width", 240)
    show = ["exit_time", "n_trades", "gross_win_rate_pct", "gross_avg_return_pct",
            "gross_total_return_fixedbase_pct", "net_win_rate_pct", "net_avg_return_pct",
            "net_total_return_fixedbase_pct", "net_total_pnl_inr", "avg_capital_deployed_per_trade"]
    print("\n=== EXIT-TIME SWEEP (sorted by net total_return_fixedbase_pct) ===")
    print(tab[show].to_string(index=False))
    print(f"\nBEST EXIT: {best['exit_time']} | n={int(best['n_trades'])} | "
          f"net total {best['net_total_return_fixedbase_pct']}% (gross {best['gross_total_return_fixedbase_pct']}%) | "
          f"net win {best['net_win_rate_pct']}% | net avg {best['net_avg_return_pct']}% | "
          f"net med {best['net_median_return_pct']}% | avg_cap ₹{best['avg_capital_deployed_per_trade']:,.0f}")
    print(f"\nPATTERN FREQUENCY: {n_signals} signals over {n_days} distinct entry days "
          f"({n_signals/n_days:.2f} signals/day avg) | consecutive-D2 overlaps on same stock: {dup_pairs} "
          f"(each taken, no dedup)")
    print(f"  ₹5L/₹1L capacity: {n_signals/n_days:.1f} signals/day vs 5 slots -> "
          f"{'EXCEEDS 5-slot pool often' if n_signals/n_days > 5 else 'within pool most days'}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
