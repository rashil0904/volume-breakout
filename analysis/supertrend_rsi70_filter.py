# -*- coding: utf-8 -*-
"""supertrend_rsi70_filter.py — add an RSI-14>=70 signal-day filter to finalized Supertrend(10,5)
long-only (VWAP-close), compare vs unfiltered baseline. On the bullish-flip day require RSI-14>=70
(Wilder, daily close) else skip. Config1=all flips, Config2=RSI>=70 flips. Removed-set validation +
per-year + IS/OOS. cost 0.23%. Portfolio: Rs10L pool, Rs1L/trade, max 10 concurrent, skip when full.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import supertrend_10_5_final as ST         # reuse supertrend_full, load_mcap, constants

DAILY = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "supertrend_rsi70"
COST, MCAP_MIN = 0.23, 1500.0
POOL, PER, MAXPOS = 1_000_000, 100_000, 10
IS_YEARS = {2022, 2023, 2024}


def wilder_rsi(c, n=14):
    c = np.asarray(c, float); m = len(c); r = np.full(m, np.nan)
    if m < n + 1:
        return r
    d = np.diff(c); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    ag, al = g[:n].mean(), l[:n].mean()
    def rs(ag, al):
        return 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 else 100 - 100 / (1 + ag / al))
    r[n] = rs(ag, al)
    for i in range(n + 1, m):
        ag = (ag * (n - 1) + g[i - 1]) / n; al = (al * (n - 1) + l[i - 1]) / n; r[i] = rs(ag, al)
    return r


def portfolio(T):
    open_pos = []; cash = POOL; realized = 0.0; taken = skipped = maxc = 0
    for r in T.sort_values("entry_dt").itertuples():
        keep = []
        for xdt, net in open_pos:
            if xdt <= r.entry_dt:
                cash += PER * (1 + net / 100); realized += PER * net / 100
            else:
                keep.append((xdt, net))
        open_pos = keep
        if len(open_pos) < MAXPOS and cash >= PER:
            cash -= PER; open_pos.append((r.exit_dt, r.net_return_pct)); taken += 1
        else:
            skipped += 1
        maxc = max(maxc, len(open_pos))
    for xdt, net in open_pos:
        realized += PER * net / 100
    return round(realized / POOL * 100, 1), taken, skipped, maxc


def build():
    df = pd.read_parquet(DAILY); df["date"] = pd.to_datetime(df["date"])
    snap_dates, snap_dicts = ST.load_mcap()
    def mcap_of(sym, d64):
        idx = int(np.searchsorted(snap_dates, d64, "right")) - 1
        return snap_dicts[max(idx, 0)].get(sym, np.nan)
    rows = []
    for sym, s in df.groupby("symbol", sort=False):
        if len(s) < 30:
            continue
        s = s.sort_values("date")
        o = s["open"].values.astype(float); h = s["high"].values.astype(float)
        lo = s["low"].values.astype(float); c = s["close"].values.astype(float); dt = s["date"].values
        d, st, atr = ST.supertrend_full(h, lo, c, 10, 5.0)
        rsi = wilder_rsi(c, 14)
        n = len(s); i = 11
        while i < n:
            if d[i] == 1 and d[i - 1] == -1:
                if i + 1 >= n:
                    break
                ei = i + 1; j = ei
                while j < n and not (d[j] == -1 and d[j - 1] == 1):
                    j += 1
                is_open = j >= n
                xi = min(j + 1, n - 1) if not is_open else n - 1
                exit_price = o[xi] if not is_open else c[xi]
                mc = mcap_of(sym, dt[i])
                if mc == mc and mc > MCAP_MIN and o[ei] > 0:
                    ret = (exit_price - o[ei]) / o[ei] * 100
                    rows.append({"symbol": sym, "signal_date": pd.Timestamp(dt[i]).strftime("%Y-%m-%d"),
                                 "entry_date": pd.Timestamp(dt[ei]).strftime("%Y-%m-%d"),
                                 "exit_date": pd.Timestamp(dt[xi]).strftime("%Y-%m-%d"),
                                 "signal_day_rsi": round(float(rsi[i]), 2) if rsi[i] == rsi[i] else np.nan,
                                 "holding_days": int(xi - ei), "return_pct": round(ret, 3),
                                 "net_return_pct": round(ret - COST, 3), "is_open": bool(is_open),
                                 "year": pd.Timestamp(dt[ei]).year})
                i = j if j < n else n
            else:
                i += 1
    T = pd.DataFrame(rows)
    T["entry_dt"] = pd.to_datetime(T["entry_date"]); T["exit_dt"] = pd.to_datetime(T["exit_date"])
    return T


def metrics(T, label):
    r = T["net_return_pct"]; rg = T["return_pct"]; win = r > 0
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    eq = T.sort_values("exit_dt")["net_return_pct"].cumsum().values
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    port, taken, skipped, maxc = portfolio(T)
    whip = ((T["holding_days"] <= 3) & (r < 0)).mean() * 100
    return {"config": label, "n_trades": len(T), "win_rate_pct": round(win.mean() * 100, 2),
            "avg_ret_gross_pct": round(rg.mean(), 3), "avg_ret_net_pct": round(r.mean(), 3),
            "median_ret_net_pct": round(r.median(), 3), "avg_holding_days": round(T["holding_days"].mean(), 1),
            "profit_factor_net": round(gp / gl, 3) if gl > 0 else np.inf, "expectancy_net_pct": round(r.mean(), 3),
            "avg_winner_pct": round(r[win].mean(), 3) if win.any() else 0.0,
            "avg_loser_pct": round(r[~win].mean(), 3) if (~win).any() else 0.0,
            "total_return_sum_net_pct": round(r.sum(), 1),
            "portfolio_return_on_10L_pct": port, "port_taken": taken, "port_skipped": skipped,
            "max_dd_tradeseq_net_pct": round(dd, 1), "whipsaw_frac_pct": round(whip, 1)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = build()
    base = T
    filt = T[T["signal_day_rsi"] >= 70]
    removed = T[T["signal_day_rsi"] < 70]
    print(f"total flips {len(T):,} | RSI>=70 kept {len(filt):,} | RSI<70 removed {len(removed):,} | "
          f"RSI NaN {int(T['signal_day_rsi'].isna().sum())}")

    comp = pd.DataFrame([metrics(base, "1_baseline_all_flips"), metrics(filt, "2_RSI>=70_filtered")])
    delta = {"config": "delta (2-1)"}
    for c in comp.columns:
        if c != "config":
            delta[c] = round(comp.iloc[1][c] - comp.iloc[0][c], 4)
    comp = pd.concat([comp, pd.DataFrame([delta])], ignore_index=True)

    # removed-set validation
    def agg(df, lbl):
        r = df["net_return_pct"]; gl = -r[r < 0].sum()
        return {"set": lbl, "n": len(df), "total_net_pct": round(r.sum(), 1), "avg_net_pct": round(r.mean(), 3),
                "median_net_pct": round(r.median(), 3), "win_rate_pct": round((r > 0).mean() * 100, 1),
                "profit_factor": round(r[r > 0].sum() / gl, 3) if gl > 0 else np.inf,
                "avg_holding_days": round(df["holding_days"].mean(), 1)}
    VAL = pd.DataFrame([agg(filt, "KEPT (RSI>=70)"), agg(removed, "REMOVED (RSI<70)"), agg(base, "ALL")])

    # per-year both configs
    py = []
    for lbl, dd0 in [("baseline", base), ("rsi70", filt)]:
        for y, g in dd0.groupby("year"):
            r = g["net_return_pct"]; gl = -r[r < 0].sum()
            py.append({"config": lbl, "year": y, "n": len(g), "win_rate": round((r > 0).mean() * 100, 1),
                       "avg_net": round(r.mean(), 3), "profit_factor": round(r[r > 0].sum() / gl, 3) if gl > 0 else np.inf})
    PY = pd.DataFrame(py)

    # IS/OOS
    def pf_block(df):
        r = df["net_return_pct"]; gl = -r[r < 0].sum()
        return {"n": len(df), "avg_net": round(r.mean(), 3), "total_net": round(r.sum(), 1),
                "win_rate": round((r > 0).mean() * 100, 1), "profit_factor": round(r[r > 0].sum() / gl, 3) if gl > 0 else np.inf}
    oos = []
    for lbl, dd0 in [("baseline", base), ("rsi70_kept", filt), ("removed_rsi<70", removed)]:
        oos.append({"set": lbl, "period": "IS 2022-24", **pf_block(dd0[dd0["year"].isin(IS_YEARS)])})
        oos.append({"set": lbl, "period": "OOS 2025+", **pf_block(dd0[~dd0["year"].isin(IS_YEARS)])})
    OOS = pd.DataFrame(oos)

    with pd.ExcelWriter(OUTDIR / "supertrend_rsi70_filter.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="headline_compare", index=False)
        VAL.to_excel(w, sheet_name="removed_set_validation", index=False)
        PY.to_excel(w, sheet_name="per_year", index=False)
        OOS.to_excel(w, sheet_name="IS_OOS", index=False)
        filt.drop(columns=["entry_dt", "exit_dt"]).to_excel(w, sheet_name="rsi70_trades", index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100 + "\nSUPERTREND(10,5) + RSI>=70 SIGNAL-DAY FILTER vs BASELINE (net 0.23%)\n" + "=" * 100)
    show = ["config", "n_trades", "win_rate_pct", "avg_ret_net_pct", "median_ret_net_pct", "avg_holding_days",
            "profit_factor_net", "avg_winner_pct", "avg_loser_pct", "total_return_sum_net_pct",
            "portfolio_return_on_10L_pct", "max_dd_tradeseq_net_pct", "whipsaw_frac_pct"]
    print(comp[show].to_string(index=False))
    print("\n--- REMOVED-SET VALIDATION (were RSI<70 flips losers or winners?) ---")
    print(VAL.to_string(index=False))
    print("\n--- PER YEAR ---")
    print(PY.pivot(index="year", columns="config", values=["n", "avg_net", "profit_factor"]).to_string())
    print("\n--- IS / OOS ---")
    print(OOS.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
