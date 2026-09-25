# -*- coding: utf-8 -*-
"""supertrend_strategy.py — daily Supertrend(10,3) long-only trend-following on NSE stocks (mcap>1500Cr).
Enter next-day open after a bullish flip (dir -1->+1), exit next-day open after a bearish flip (+1->-1).
One position per stock at a time. Standard Supertrend (Wilder ATR, final-band carry-forward, flip logic).
Reports per-trade stats (universe-wide) AND an equal-weight portfolio (Rs10L pool, Rs1L/trade, max 10
concurrent, skip when full). Cost 0.30% round-trip.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
MDIR = rb.BASE / "mcap_cache"
OUTDIR = rb.RESULTS / "supertrend"
ATR_N, FACT = 10, 3.0
COST = 0.30
MCAP_MIN = 1500.0
POOL, PER_TRADE, MAX_POS = 1_000_000, 100_000, 10
SNAPS = [("2022-03-31", "mcap_2022-03-31.xlsx"), ("2022-12-31", "mcap_2022-12-31.xlsx"),
         ("2023-03-31", "mcap_2023-03-31.xlsx"), ("2023-12-31", "mcap_2023-12-31.xlsx"),
         ("2024-03-28", "mcap_2024-03-28.xlsx"), ("2024-12-31", "mcap_2024-12-31.xlsx"),
         ("2025-12-31", "mcap_2025-12-31.xlsx")]


def load_mcap():
    dates, dicts = [], []
    for ds, fn in SNAPS:
        df = pd.read_excel(MDIR / fn, header=None, skiprows=1, usecols=[1, 3])
        df.columns = ["symbol", "mcap_lakhs"]
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df["mcap_cr"] = pd.to_numeric(df["mcap_lakhs"], errors="coerce") / 100
        dates.append(np.datetime64(ds)); dicts.append(df.dropna(subset=["mcap_cr"]).set_index("symbol")["mcap_cr"].to_dict())
    return np.array(dates), dicts


def supertrend(high, low, close, n=10, f=3.0):
    m = len(close)
    tr = np.full(m, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, m):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(m, np.nan)
    if m <= n:
        return np.zeros(m)
    atr[n - 1] = np.nanmean(tr[:n])
    for i in range(n, m):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    hl2 = (high + low) / 2.0
    ub = hl2 + f * atr; lb = hl2 - f * atr
    fub = np.copy(ub); flb = np.copy(lb)
    for i in range(n, m):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
    st = np.full(m, np.nan); direction = np.zeros(m, dtype=int)
    st[n] = fub[n]; direction[n] = -1                                    # seed downtrend
    for i in range(n + 1, m):
        if st[i - 1] == fub[i - 1]:                                      # prior downtrend
            if close[i] > fub[i]:
                st[i] = flb[i]; direction[i] = 1
            else:
                st[i] = fub[i]; direction[i] = -1
        else:                                                            # prior uptrend
            if close[i] < flb[i]:
                st[i] = fub[i]; direction[i] = -1
            else:
                st[i] = flb[i]; direction[i] = 1
    return direction


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DAILY)
    df["date"] = pd.to_datetime(df["date"])
    snap_dates, snap_dicts = load_mcap()

    def mcap_of(sym, d64):
        idx = int(np.searchsorted(snap_dates, d64, "right")) - 1
        return snap_dicts[max(idx, 0)].get(sym, np.nan)

    trades = []
    n_flips = 0
    for sym, s in df.groupby("symbol", sort=False):
        if len(s) < ATR_N + 5:
            continue
        s = s.sort_values("date")
        o = s["open"].values.astype(float); h = s["high"].values.astype(float)
        lo = s["low"].values.astype(float); c = s["close"].values.astype(float)
        dt = s["date"].values
        d = supertrend(h, lo, c, ATR_N, FACT)
        n = len(s)
        i = ATR_N + 1
        while i < n:
            if d[i] == 1 and d[i - 1] == -1:                            # bullish flip at bar i
                n_flips += 1
                if i + 1 >= n:
                    break
                ei = i + 1                                              # enter next open
                # find next bearish flip j>i
                j = ei
                while j < n and not (d[j] == -1 and d[j - 1] == 1):
                    j += 1
                if j < n:                                               # exit next open after bear flip
                    xi = min(j + 1, n - 1); xtype = "bear_flip" if j + 1 < n else "bear_flip_lastbar"
                else:
                    xi = n - 1; xtype = "open_end"                      # still in trend at data end
                mc = mcap_of(sym, dt[i])
                if mc == mc and mc > MCAP_MIN and o[ei] > 0:
                    entry = o[ei]; exit_p = o[xi]
                    ret = (exit_p - entry) / entry * 100
                    trades.append({"symbol": sym, "flip_date": pd.Timestamp(dt[i]).strftime("%Y-%m-%d"),
                                   "entry_date": pd.Timestamp(dt[ei]).strftime("%Y-%m-%d"), "entry_price": round(entry, 2),
                                   "exit_flip_date": pd.Timestamp(dt[j]).strftime("%Y-%m-%d") if j < n else "",
                                   "exit_date": pd.Timestamp(dt[xi]).strftime("%Y-%m-%d"), "exit_price": round(exit_p, 2),
                                   "exit_type": xtype, "holding_days": int(xi - ei), "mcap_cr": round(float(mc), 0),
                                   "return_pct": round(ret, 3), "net_return_pct": round(ret - COST, 3),
                                   "year": pd.Timestamp(dt[ei]).year})
                i = j if j < n else n                                   # continue after this trend
            else:
                i += 1
    T = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    T["entry_dt"] = pd.to_datetime(T["entry_date"]); T["exit_dt"] = pd.to_datetime(T["exit_date"])
    print(f"universe symbols traded: {T['symbol'].nunique():,} | bullish flips (mcap-eligible entries): {len(T):,} "
          f"| total flips scanned: {n_flips:,}")

    # ── per-trade metrics ──
    def block(df, net=True):
        col = "net_return_pct" if net else "return_pct"
        r = df[col]; win = r > 0
        gp = r[r > 0].sum(); gl = -r[r < 0].sum()
        eq = r.cumsum().values; dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
        return {"basis": "net" if net else "gross", "n_trades": len(df), "win_rate_pct": round(win.mean() * 100, 2),
                "avg_return_pct": round(r.mean(), 3), "median_return_pct": round(r.median(), 3),
                "avg_winner_pct": round(r[win].mean(), 3) if win.any() else 0.0,
                "avg_loser_pct": round(r[~win].mean(), 3) if (~win).any() else 0.0,
                "expectancy_pct": round(r.mean(), 3), "profit_factor": round(gp / gl, 3) if gl > 0 else np.inf,
                "largest_win_pct": round(r.max(), 2), "largest_loss_pct": round(r.min(), 2),
                "avg_holding_days": round(df["holding_days"].mean(), 1), "median_holding_days": int(df["holding_days"].median()),
                "total_return_sum_pct": round(r.sum(), 1), "max_dd_tradeseq_pct": round(dd, 1)}
    summ = pd.DataFrame([block(T, False), block(T, True)])

    # per-year
    yr = []
    for y, g in T.groupby("year"):
        b = block(g, True); yr.append({"year": y, **{k: b[k] for k in ["n_trades", "win_rate_pct", "avg_return_pct",
                                                                        "median_return_pct", "avg_holding_days", "profit_factor",
                                                                        "total_return_sum_pct"]}})
    YR = pd.DataFrame(yr)
    freq = T.groupby("year").size().rename("entries").reset_index()

    # ── portfolio (Rs10L, Rs1L/trade, max 10 concurrent, skip when full) ──
    events = []
    for r in T.itertuples():
        events.append((r.entry_dt, "entry", r))
    ev = sorted(events, key=lambda x: x[0])
    open_pos = []; cash = POOL; equity_dates = []; realized = 0.0
    n_taken = n_skip = 0; peak_open = 0
    taken_flags = []
    for edt, _, r in ev:
        # free positions that exited on/before this entry date
        still = []
        for p in open_pos:
            if p["exit_dt"] <= edt:
                cash += PER_TRADE * (1 + p["net"] / 100)
                realized += PER_TRADE * p["net"] / 100
            else:
                still.append(p)
        open_pos = still
        if len(open_pos) < MAX_POS and cash >= PER_TRADE:
            cash -= PER_TRADE
            open_pos.append({"exit_dt": r.exit_dt, "net": r.net_return_pct})
            n_taken += 1; taken_flags.append(True)
        else:
            n_skip += 1; taken_flags.append(False)
        peak_open = max(peak_open, len(open_pos))
    # close any remaining
    for p in open_pos:
        cash += PER_TRADE * (1 + p["net"] / 100); realized += PER_TRADE * p["net"] / 100
    port_total_pnl = realized
    port_ret_on_pool = realized / POOL * 100
    port = {"pool_inr": POOL, "per_trade_inr": PER_TRADE, "max_concurrent_cap": MAX_POS,
            "n_signals": len(T), "n_taken": n_taken, "n_skipped_full": n_skip, "max_concurrent_used": peak_open,
            "total_pnl_inr": round(port_total_pnl, 0), "total_return_on_pool_pct": round(port_ret_on_pool, 1)}

    # ── outputs / plots ──
    T2 = T.drop(columns=["entry_dt", "exit_dt"])
    T2["equity_net_cum_pct"] = T["net_return_pct"].cumsum().round(2)
    with pd.ExcelWriter(OUTDIR / "supertrend.xlsx", engine="openpyxl") as w:
        summ.to_excel(w, sheet_name="summary", index=False)
        YR.to_excel(w, sheet_name="by_year", index=False)
        pd.DataFrame([port]).to_excel(w, sheet_name="portfolio", index=False)
        freq.to_excel(w, sheet_name="signals_per_year", index=False)
        T2.to_excel(w, sheet_name="all_trades", index=False)
    T2.to_csv(OUTDIR / "all_trades.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    a1.plot(range(len(T2)), T2["equity_net_cum_pct"], lw=0.8, color="#2E74B5")
    a1.set_title("Supertrend(10,3) trade-sequence net equity (sum of per-trade %)"); a1.set_xlabel("trade #"); a1.set_ylabel("cum net return %")
    a2.hist(np.clip(T["net_return_pct"], -30, 100), bins=80, color="#5B9BD5", edgecolor="white", lw=0.3)
    a2.axvline(0, color="grey", ls=":"); a2.set_title("Net return distribution (clipped -30..100) — trend-following fat tail")
    a2.set_xlabel("net return %"); a2.set_ylabel("n trades")
    fig.tight_layout(); fig.savefig(OUTDIR / "supertrend_overview.png", dpi=120); plt.close(fig)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 100 + "\nSUPERTREND(10,3) LONG-ONLY — daily trend-following (mcap>1500Cr; next-open entry/exit)\n" + "=" * 100)
    print("\n--- SUMMARY (gross vs net 0.30%) ---")
    print(summ.to_string(index=False))
    print("\n--- BY YEAR (net) ---")
    print(YR.to_string(index=False))
    print("\n--- SIGNAL FREQUENCY (bullish-flip entries/yr) ---")
    print(freq.to_string(index=False))
    print("\n--- PORTFOLIO (Rs10L pool, Rs1L/trade, max 10 concurrent) ---")
    for k, v in port.items():
        print(f"  {k:26s}: {v}")
    print("\n--- first 10 trades ---")
    print(T2[["symbol", "flip_date", "entry_date", "entry_price", "exit_date", "exit_price",
              "holding_days", "return_pct", "net_return_pct"]].head(10).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
