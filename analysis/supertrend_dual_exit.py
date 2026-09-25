# -*- coding: utf-8 -*-
"""supertrend_dual_exit.py — Supertrend(10,5)+RSI>=70 entry, DUAL EXIT (RSI-14 close<70 OR ST bear flip,
whichever FIRST; fill next-day open) vs Supertrend-flip-only exit. Same entries both configs. VWAP-close,
mcap>1500, cost 0.23%. Records exit_reason. Per-year + IS/OOS.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import supertrend_10_5_final as ST

DAILY = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "supertrend_dual_exit"
COST, MCAP_MIN = 0.23, 1500.0
POOL, PER, MAXPOS = 1_000_000, 100_000, 10
IS_YEARS = {2022, 2023, 2024}


def wilder_rsi(c, n=14):
    c = np.asarray(c, float); m = len(c); r = np.full(m, np.nan)
    if m < n + 1:
        return r
    dd = np.diff(c); g = np.where(dd > 0, dd, 0.0); l = np.where(dd < 0, -dd, 0.0)
    ag, al = g[:n].mean(), l[:n].mean()
    def rs(ag, al):
        return 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 else 100 - 100 / (1 + ag / al))
    r[n] = rs(ag, al)
    for i in range(n + 1, m):
        ag = (ag * (n - 1) + g[i - 1]) / n; al = (al * (n - 1) + l[i - 1]) / n; r[i] = rs(ag, al)
    return r


def portfolio(T):
    open_pos = []; cash = POOL; realized = 0.0; taken = 0
    for r in T.sort_values("entry_dt").itertuples():
        keep = []
        for x, nt in open_pos:
            if x <= r.entry_dt:
                cash += PER * (1 + nt / 100); realized += PER * nt / 100
            else:
                keep.append((x, nt))
        open_pos = keep
        if len(open_pos) < MAXPOS and cash >= PER:
            cash -= PER; open_pos.append((r.exit_dt, r.net_return_pct)); taken += 1
    for x, nt in open_pos:
        realized += PER * nt / 100
    return round(realized / POOL * 100, 1), taken


def metrics(T, label):
    r = T["net_return_pct"]; rg = T["return_pct"]; win = r > 0
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    eq = T.sort_values("exit_dt")["net_return_pct"].cumsum().values
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    port, taken = portfolio(T)
    return {"config": label, "n_trades": len(T), "win_rate_pct": round(win.mean() * 100, 2),
            "avg_ret_gross_pct": round(rg.mean(), 3), "avg_ret_net_pct": round(r.mean(), 3),
            "median_ret_net_pct": round(r.median(), 3), "avg_holding_days": round(T["holding_days"].mean(), 1),
            "profit_factor_net": round(gp / gl, 3) if gl > 0 else np.inf, "expectancy_net_pct": round(r.mean(), 3),
            "avg_winner_pct": round(r[win].mean(), 3) if win.any() else 0.0,
            "avg_loser_pct": round(r[~win].mean(), 3) if (~win).any() else 0.0,
            "total_return_sum_net_pct": round(r.sum(), 1), "portfolio_return_on_10L_pct": port,
            "max_dd_tradeseq_net_pct": round(dd, 1)}


def build():
    df = pd.read_parquet(DAILY); df["date"] = pd.to_datetime(df["date"])
    snap_dates, snap_dicts = ST.load_mcap()
    def mcap_of(sym, d64):
        idx = int(np.searchsorted(snap_dates, d64, "right")) - 1
        return snap_dicts[max(idx, 0)].get(sym, np.nan)
    t1, t2 = [], []
    for sym, s in df.groupby("symbol", sort=False):
        if len(s) < 30:
            continue
        s = s.sort_values("date")
        o = s["open"].values.astype(float); h = s["high"].values.astype(float)
        lo = s["low"].values.astype(float); c = s["close"].values.astype(float); dt = s["date"].values
        d, st, atr = ST.supertrend_full(h, lo, c, 10, 5.0); rsi = wilder_rsi(c, 14)
        n = len(s); i = 11
        while i < n:
            if d[i] == 1 and d[i - 1] == -1:
                if i + 1 >= n:
                    break
                ei = i + 1
                # next ST bear flip
                j = ei
                while j < n and not (d[j] == -1 and d[j - 1] == 1):
                    j += 1
                mc = mcap_of(sym, dt[i])
                take = (mc == mc and mc > MCAP_MIN and o[ei] > 0 and rsi[i] == rsi[i] and rsi[i] >= 70)
                if take:
                    base = {"symbol": sym, "signal_date": pd.Timestamp(dt[i]).strftime("%Y-%m-%d"),
                            "entry_date": pd.Timestamp(dt[ei]).strftime("%Y-%m-%d"), "entry_price": round(o[ei], 2),
                            "signal_day_rsi": round(float(rsi[i]), 2), "year": pd.Timestamp(dt[ei]).year}
                    # CONFIG 1: ST-only exit
                    if j < n:
                        xi = min(j + 1, n - 1); ep = o[xi]; xo1 = False
                    else:
                        xi = n - 1; ep = c[xi]; xo1 = True
                    r1 = (ep - o[ei]) / o[ei] * 100
                    t1.append({**base, "exit_date": pd.Timestamp(dt[xi]).strftime("%Y-%m-%d"), "exit_price": round(ep, 2),
                               "exit_reason": "supertrend_flip" if not xo1 else "open_end", "holding_days": int(xi - ei),
                               "return_pct": round(r1, 3), "net_return_pct": round(r1 - COST, 3), "is_open": xo1})
                    # CONFIG 2: dual exit — first of RSI<70 (from ei) or ST bear flip (j)
                    k = ei
                    while k < n and not (rsi[k] == rsi[k] and rsi[k] < 70):
                        k += 1
                    sig = min(k, j)
                    if sig >= n:
                        xi2 = n - 1; ep2 = c[xi2]; reason = "open_end"; xo2 = True
                    else:
                        reason = "rsi_below_70" if k <= j else "supertrend_flip"
                        xi2 = min(sig + 1, n - 1); ep2 = o[xi2]; xo2 = False
                    r2 = (ep2 - o[ei]) / o[ei] * 100
                    t2.append({**base, "exit_date": pd.Timestamp(dt[xi2]).strftime("%Y-%m-%d"), "exit_price": round(ep2, 2),
                               "exit_reason": reason, "holding_days": int(xi2 - ei),
                               "return_pct": round(r2, 3), "net_return_pct": round(r2 - COST, 3), "is_open": xo2})
                i = j if j < n else n
            else:
                i += 1
    T1 = pd.DataFrame(t1); T2 = pd.DataFrame(t2)
    for T in (T1, T2):
        T["entry_dt"] = pd.to_datetime(T["entry_date"]); T["exit_dt"] = pd.to_datetime(T["exit_date"])
    return T1, T2


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T1, T2 = build()
    print(f"entries (both configs): C1={len(T1):,}  C2={len(T2):,}")

    comp = pd.DataFrame([metrics(T1, "1_supertrend_only_exit"), metrics(T2, "2_dual_exit_RSI<70_or_STflip")])
    delta = {"config": "delta (2-1)"}
    for c in comp.columns:
        if c != "config":
            delta[c] = round(comp.iloc[1][c] - comp.iloc[0][c], 4)
    comp = pd.concat([comp, pd.DataFrame([delta])], ignore_index=True)

    # exit-reason split (config 2)
    er = []
    for reason, g in T2.groupby("exit_reason"):
        r = g["net_return_pct"]
        er.append({"exit_reason": reason, "n": len(g), "pct_of_trades": round(len(g) / len(T2) * 100, 1),
                   "avg_net_ret_pct": round(r.mean(), 3), "median_net_ret_pct": round(r.median(), 3),
                   "win_rate_pct": round((r > 0).mean() * 100, 1), "avg_holding_days": round(g["holding_days"].mean(), 1)})
    ER = pd.DataFrame(er).sort_values("n", ascending=False)

    # per-year
    py = []
    for lbl, dd0 in [("ST_only", T1), ("dual", T2)]:
        for y, g in dd0.groupby("year"):
            r = g["net_return_pct"]; gl = -r[r < 0].sum()
            py.append({"config": lbl, "year": y, "n": len(g), "win_rate": round((r > 0).mean() * 100, 1),
                       "avg_net": round(r.mean(), 3), "avg_hold": round(g["holding_days"].mean(), 1),
                       "profit_factor": round(r[r > 0].sum() / gl, 3) if gl > 0 else np.inf})
    PY = pd.DataFrame(py)

    def pf_block(df):
        r = df["net_return_pct"]; gl = -r[r < 0].sum()
        return {"n": len(df), "avg_net": round(r.mean(), 3), "total_net": round(r.sum(), 1),
                "win_rate": round((r > 0).mean() * 100, 1), "avg_hold": round(df["holding_days"].mean(), 1),
                "profit_factor": round(r[r > 0].sum() / gl, 3) if gl > 0 else np.inf}
    oos = []
    for lbl, dd0 in [("ST_only", T1), ("dual", T2)]:
        oos.append({"config": lbl, "period": "IS 2022-24", **pf_block(dd0[dd0["year"].isin(IS_YEARS)])})
        oos.append({"config": lbl, "period": "OOS 2025+", **pf_block(dd0[~dd0["year"].isin(IS_YEARS)])})
    OOS = pd.DataFrame(oos)

    with pd.ExcelWriter(OUTDIR / "supertrend_dual_exit.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="headline_compare", index=False)
        ER.to_excel(w, sheet_name="exit_reason_split_C2", index=False)
        PY.to_excel(w, sheet_name="per_year", index=False)
        OOS.to_excel(w, sheet_name="IS_OOS", index=False)
        T2.drop(columns=["entry_dt", "exit_dt"]).to_excel(w, sheet_name="dual_exit_all_trades", index=False)
        T1.drop(columns=["entry_dt", "exit_dt"]).to_excel(w, sheet_name="ST_only_all_trades", index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100 + "\nDUAL EXIT (RSI<70 or ST flip) vs SUPERTREND-ONLY EXIT  (net 0.23%)\n" + "=" * 100)
    show = ["config", "n_trades", "win_rate_pct", "avg_ret_net_pct", "median_ret_net_pct", "avg_holding_days",
            "profit_factor_net", "avg_winner_pct", "avg_loser_pct", "total_return_sum_net_pct",
            "portfolio_return_on_10L_pct", "max_dd_tradeseq_net_pct"]
    print(comp[show].to_string(index=False))
    print("\n--- CONFIG 2 EXIT-REASON SPLIT ---")
    print(ER.to_string(index=False))
    print("\n--- PER YEAR ---")
    print(PY.pivot(index="year", columns="config", values=["n", "avg_net", "avg_hold", "profit_factor"]).to_string())
    print("\n--- IS / OOS ---")
    print(OOS.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
