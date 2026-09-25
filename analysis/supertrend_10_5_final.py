# -*- coding: utf-8 -*-
"""supertrend_10_5_final.py — FINALIZED Supertrend(10,5) long-only daily strategy across mcap>1500Cr
stocks. Exports a rich ALL-TRADES Excel (one row per bull-flip->bear-flip cycle) + a SUMMARY sheet.

(a) Supertrend(10,5) standard; CLOSE = VWAP of 15:00-15:29 (NSE official-close), Wilder ATR;
(b) entry = next-day open after bull flip, exit = next-day open after bear flip;
(c) one position per stock at a time, buy only on the flip;
(d) mcap>1500 point-in-time via nearest-preceding quarterly snapshot, BUT the symbol universe is
    current master_data constituents -> SURVIVORSHIP-BIAS caveat (delisted names absent);
(e) open-at-end trades flagged is_open=True (marked to last close, unrealized);
(f) per-trade cost 0.23% round-trip (delivery) -> net_return_pct.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DAILY = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"   # close = VWAP of 15:00-15:29 (NSE official-close)
MDIR = rb.BASE / "mcap_cache"
OUTDIR = rb.RESULTS / "supertrend_10_5_vwapclose"
ATR_N, FACT = 10, 5.0
COST = 0.23
MCAP_MIN = 1500.0
SNAPS = [("2022-03-31", "mcap_2022-03-31.xlsx"), ("2022-12-31", "mcap_2022-12-31.xlsx"),
         ("2023-03-31", "mcap_2023-03-31.xlsx"), ("2023-12-31", "mcap_2023-12-31.xlsx"),
         ("2024-03-28", "mcap_2024-03-28.xlsx"), ("2024-12-31", "mcap_2024-12-31.xlsx"),
         ("2025-12-31", "mcap_2025-12-31.xlsx")]


def load_mcap():
    dates, dicts = [], []
    for ds, fn in SNAPS:
        d = pd.read_excel(MDIR / fn, header=None, skiprows=1, usecols=[1, 3])
        d.columns = ["symbol", "mcap_lakhs"]; d["symbol"] = d["symbol"].astype(str).str.strip().str.upper()
        d["mcap_cr"] = pd.to_numeric(d["mcap_lakhs"], errors="coerce") / 100
        dates.append(np.datetime64(ds)); dicts.append(d.dropna(subset=["mcap_cr"]).set_index("symbol")["mcap_cr"].to_dict())
    return np.array(dates), dicts


def supertrend_full(h, lo, c, n=10, f=5.0):
    """Return (direction, st_line, atr) — standard Supertrend."""
    m = len(c); tr = np.empty(m); tr[0] = h[0] - lo[0]
    for i in range(1, m):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
    atr = np.full(m, np.nan)
    if m <= n:
        return np.zeros(m, int), np.full(m, np.nan), atr
    atr[n - 1] = tr[:n].mean()
    for i in range(n, m):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    hl2 = (h + lo) * 0.5
    ub = hl2 + f * atr; lb = hl2 - f * atr
    fub = ub.copy(); flb = lb.copy()
    for i in range(n, m):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
    st = np.full(m, np.nan); direction = np.zeros(m, int)
    st[n] = fub[n]; direction[n] = -1
    for i in range(n + 1, m):
        if st[i - 1] == fub[i - 1]:                       # prior downtrend
            if c[i] > fub[i]:
                st[i] = flb[i]; direction[i] = 1
            else:
                st[i] = fub[i]; direction[i] = -1
        else:                                             # prior uptrend
            if c[i] < flb[i]:
                st[i] = fub[i]; direction[i] = -1
            else:
                st[i] = flb[i]; direction[i] = 1
    return direction, st, atr


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DAILY); df["date"] = pd.to_datetime(df["date"])
    snap_dates, snap_dicts = load_mcap()
    def mcap_of(sym, d64):
        idx = int(np.searchsorted(snap_dates, d64, "right")) - 1
        return snap_dicts[max(idx, 0)].get(sym, np.nan)

    rows = []
    for sym, s in df.groupby("symbol", sort=False):
        if len(s) < ATR_N + 5:
            continue
        s = s.sort_values("date")
        o = s["open"].values.astype(float); h = s["high"].values.astype(float)
        lo = s["low"].values.astype(float); c = s["close"].values.astype(float); dt = s["date"].values
        d, st, atr = supertrend_full(h, lo, c, ATR_N, FACT)
        n = len(s); i = ATR_N + 1
        while i < n:
            if d[i] == 1 and d[i - 1] == -1:                       # bull flip at i
                if i + 1 >= n:
                    break
                ei = i + 1
                j = ei
                while j < n and not (d[j] == -1 and d[j - 1] == 1):
                    j += 1
                is_open = j >= n
                if not is_open:
                    xi = min(j + 1, n - 1)                          # next open after bear flip
                    exit_price = o[xi]; exit_date = dt[xi]; bear_flip = dt[j]; exit_st = st[j]
                else:
                    xi = n - 1                                     # open trade -> mark to last close
                    exit_price = c[xi]; exit_date = dt[xi]; bear_flip = None; exit_st = st[xi]
                mc = mcap_of(sym, dt[i])
                if mc == mc and mc > MCAP_MIN and o[ei] > 0:
                    ret = (exit_price - o[ei]) / o[ei] * 100
                    rows.append({
                        "symbol": sym,
                        "bull_flip_date": pd.Timestamp(dt[i]).strftime("%Y-%m-%d"),
                        "entry_date": pd.Timestamp(dt[ei]).strftime("%Y-%m-%d"), "entry_price": round(o[ei], 2),
                        "bear_flip_date": pd.Timestamp(bear_flip).strftime("%Y-%m-%d") if bear_flip is not None else "",
                        "exit_date": pd.Timestamp(exit_date).strftime("%Y-%m-%d"), "exit_price": round(exit_price, 2),
                        "holding_days": int(xi - ei),
                        "return_pct": round(ret, 3), "net_return_pct": round(ret - COST, 3),
                        "entry_supertrend_value": round(float(st[i]), 2), "exit_supertrend_value": round(float(exit_st), 2),
                        "atr_at_entry": round(float(atr[i]), 3),
                        "year": pd.Timestamp(dt[ei]).year, "is_open": bool(is_open),
                        "mcap_at_entry": round(float(mc), 0)})
                i = j if j < n else n
            else:
                i += 1
    T = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
    print(f"symbols traded: {T['symbol'].nunique():,} | trades: {len(T):,} | open-at-end: {int(T['is_open'].sum())}")

    # ── summary ──
    def summ(net):
        col = "net_return_pct" if net else "return_pct"
        r = T[col]; win = r > 0
        gp = r[r > 0].sum(); gl = -r[r < 0].sum()
        eq = r.cumsum().values; dd = float((eq - np.maximum.accumulate(eq)).min())
        return {"basis": "net" if net else "gross", "n_trades": len(T), "win_rate_pct": round(win.mean() * 100, 2),
                "avg_return_pct": round(r.mean(), 3), "median_return_pct": round(r.median(), 3),
                "avg_winner_pct": round(r[win].mean(), 3), "avg_loser_pct": round(r[~win].mean(), 3),
                "profit_factor": round(gp / gl, 3), "avg_holding_days": round(T["holding_days"].mean(), 1),
                "median_holding_days": int(T["holding_days"].median()),
                "largest_win_pct": round(r.max(), 2), "largest_loss_pct": round(r.min(), 2),
                "total_return_sum_pct": round(r.sum(), 1), "max_drawdown_tradeseq_pct": round(dd, 1),
                "n_open_at_end": int(T["is_open"].sum())}
    SUM = pd.DataFrame([summ(False), summ(True)])
    yr = T.groupby("year").agg(n_trades=("symbol", "size"), avg_return_gross_pct=("return_pct", "mean"),
                               avg_return_net_pct=("net_return_pct", "mean"),
                               win_rate_pct=("return_pct", lambda x: (x > 0).mean() * 100),
                               avg_holding_days=("holding_days", "mean")).round(3).reset_index()

    with pd.ExcelWriter(OUTDIR / "supertrend_10_5_all_trades.xlsx", engine="openpyxl") as w:
        T.to_excel(w, sheet_name="all_trades", index=False)
        SUM.to_excel(w, sheet_name="summary", index=False)
        yr.to_excel(w, sheet_name="by_year", index=False)
        for sh in w.sheets.values():
            for col in sh.columns:
                width = max((len(str(x.value)) for x in col if x.value is not None), default=10)
                sh.column_dimensions[col[0].column_letter].width = min(width + 2, 24)
    T.to_csv(OUTDIR / "supertrend_10_5_all_trades.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n=== SUMMARY (gross vs net 0.30%) ===")
    print(SUM.to_string(index=False))
    print("\n=== BY YEAR ===")
    print(yr.to_string(index=False))
    print("\n=== first 8 trades ===")
    print(T[["symbol", "bull_flip_date", "entry_date", "entry_price", "bear_flip_date", "exit_date", "exit_price",
             "holding_days", "return_pct", "net_return_pct", "atr_at_entry", "is_open"]].head(8).to_string(index=False))
    print(f"\nSaved -> {OUTDIR / 'supertrend_10_5_all_trades.xlsx'}")


if __name__ == "__main__":
    main()
