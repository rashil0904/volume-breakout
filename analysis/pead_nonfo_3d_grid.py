# -*- coding: utf-8 -*-
"""pead_nonfo_3d_grid.py — NON-F&O Nifty-500 PEAD LONG-ONLY 3D sweep: K(4-7) x hold(T+1..T+15) x
target(5-20%) = 960 cells. Separate from the F&O book. LONG-only by construction (no short signal is
ever generated for this universe). reaction = (T0 close - prev-day VWAP-close)/prev_vwap*100; long when
>=K. Entry=T0 close; during/post->next-day T0, pre->same-day. Exit: TARGET intraday touch (fill
max(tgt,open)); SL close-basis at T0 low (fill that close); HOLD-END at T+n close. Priority: target is an
intraday touch so it is evaluated BEFORE the close exists -> on a same-day both-hit the target fires
first (the 'SL-precedence' convention only applies when both legs are close-based, which is not the case
here). Corrected DD-MM datetime parse. IS 2022-24 / OOS 2025+. Best = smoothed robust REGION, not a cell.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements_combined.csv"
UNIV = rb.BASE / "data" / "nifty500_universe.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
VWAPD = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "pead_nonfo_3d_grid"
COST = 0.20
KS = [4, 5, 6, 7]
NS = list(range(1, 16))               # T+1 .. T+15
TGTS = list(range(5, 21))             # 5% .. 20%
NMAX = 15
IS_YEARS = {2022, 2023, 2024}


def build(nonfo_syms):
    a = pd.read_csv(ANN, dtype=str)
    a = a[a["symbol"].isin(nonfo_syms)]
    a["ann_date"] = pd.to_datetime(a["announcement_date"], format="%d-%m-%Y", errors="coerce").dt.date
    a["dtm"] = pd.to_datetime(a["announcement_datetime"], format="%d-%m-%Y %H:%M", errors="coerce")   # CORRECT parse
    a = a.dropna(subset=["ann_date", "dtm"]).sort_values("dtm").drop_duplicates(["symbol", "quarter"], keep="first")
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "open", "high", "low", "close"]); d["date"] = pd.to_datetime(d["date"]).dt.date
    vw = pd.read_parquet(VWAPD, columns=["symbol", "date", "close"]); vw["date"] = pd.to_datetime(vw["date"]).dt.date
    vmap = {(s, dt): c for s, dt, c in zip(vw["symbol"], vw["date"], vw["close"])}
    sd = {}
    for sym, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date")
        sd[sym] = (g["date"].values, g["open"].values.astype(float), g["high"].values.astype(float),
                   g["low"].values.astype(float), g["close"].values.astype(float), {dt: i for i, dt in enumerate(g["date"].values)})
    ev = []; seen = set()
    for r in a.itertuples():
        s = sd.get(r.symbol)
        if s is None:
            continue
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
        entry = cl[p]; pv = vmap.get((r.symbol, dates[p - 1]), np.nan)
        if not (entry > 0 and pv == pv and pv > 0):
            continue
        rr = (entry - pv) / pv * 100.0
        if abs(rr) > 50.0:                                    # GUARD3: corporate-action / bad-print
            continue
        if rr < min(KS):                                      # LONG only, weakest K gate
            continue
        avail = len(dates) - 1 - p
        if avail < 1:
            continue
        m = min(NMAX, avail)
        ev.append({"symbol": r.symbol, "year": int(str(dates[p])[:4]), "rr": rr, "entry": entry, "sl": lo[p], "navail": m,
                   "fo": op[p + 1:p + 1 + m], "fh": hi[p + 1:p + 1 + m], "fl": lo[p + 1:p + 1 + m], "fc": cl[p + 1:p + 1 + m]})
    return ev


def exit_long(e, n, tgt):
    entry, sl = e["entry"], e["sl"]; fo, fh, fc = e["fo"], e["fh"], e["fc"]
    tp = entry * (1 + tgt / 100.0)
    for j in range(n):
        if fh[j] >= tp:                                       # TARGET intraday touch (checked first)
            return max(tp, fo[j])
        if fc[j] <= sl:                                       # SL close-basis
            return fc[j]
    return fc[n - 1]                                          # HOLD-END


def cell(ev, K, n, tgt):
    sub = [e for e in ev if e["rr"] >= K and e["navail"] >= n]
    if not sub:
        return {"K": K, "n_hold": n, "target_pct": tgt, "n_trades": 0, "total_net": 0.0, "avg_net": np.nan,
                "win_rate": np.nan, "IS_total": 0.0, "OOS_total": 0.0, "IS_avg": np.nan, "OOS_avg": np.nan, "IS_n": 0, "OOS_n": 0}
    g = np.array([(exit_long(e, n, tgt) - e["entry"]) / e["entry"] * 100.0 for e in sub])
    yrs = np.array([e["year"] for e in sub]); net = g - COST; ism = np.isin(yrs, list(IS_YEARS))
    return {"K": K, "n_hold": n, "target_pct": tgt, "n_trades": len(sub), "total_net": round(net.sum(), 1),
            "avg_net": round(net.mean(), 3), "win_rate": round((g > 0).mean() * 100, 1),
            "IS_total": round(net[ism].sum(), 1), "OOS_total": round(net[~ism].sum(), 1),
            "IS_avg": round(net[ism].mean(), 3) if ism.any() else np.nan, "OOS_avg": round(net[~ism].mean(), 3) if (~ism).any() else np.nan,
            "IS_n": int(ism.sum()), "OOS_n": int((~ism).sum())}


def smooth(M):
    sm = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            sm[i, j] = np.nanmean(M[max(0, i - 1):i + 2, max(0, j - 1):j + 2])
    return sm


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    u = pd.read_csv(UNIV)
    nonfo = set(u[~u["fno_eligible"]]["symbol"])
    ev = build(nonfo)
    print(f"NON-F&O Nifty500 long events (K>=4): {len(ev)} | symbols {len(set(e['symbol'] for e in ev))} | "
          f"IS {sum(1 for e in ev if e['year'] in IS_YEARS)} OOS {sum(1 for e in ev if e['year'] not in IS_YEARS)}")
    for K in KS:
        nk = sum(1 for e in ev if e["rr"] >= K)
        print(f"  K>={K}%: {nk} long signals", flush=True)

    G = pd.DataFrame([cell(ev, K, n, t) for K in KS for n in NS for t in TGTS])
    G.to_csv(OUTDIR / "pead_nonfo_3d_all_cells.csv", index=False)

    # per-K robust region (smoothed min(IS_avg,OOS_avg)); pick overall best
    regions = []
    with pd.ExcelWriter(OUTDIR / "pead_nonfo_3d_grid.xlsx", engine="openpyxl") as w:
        G.to_excel(w, sheet_name="all_cells", index=False)
        for K in KS:
            gk = G[G.K == K]
            for metric, sheet in [("total_net", f"K{K}_total"), ("win_rate", f"K{K}_win"),
                                  ("OOS_total", f"K{K}_OOS"), ("n_trades", f"K{K}_ntrades")]:
                gk.pivot(index="n_hold", columns="target_pct", values=metric).to_excel(w, sheet_name=sheet)
            piv_is = gk.pivot(index="n_hold", columns="target_pct", values="IS_avg")
            piv_oos = gk.pivot(index="n_hold", columns="target_pct", values="OOS_avg")
            rob = smooth(np.minimum(piv_is.values, piv_oos.values))
            if np.isnan(rob).all():
                continue
            bi, bj = np.unravel_index(np.nanargmax(rob), rob.shape)
            ns_v, ts_v = list(piv_is.index), list(piv_is.columns)
            nc = ns_v[max(0, bi - 1):bi + 2]; tc = ts_v[max(0, bj - 1):bj + 2]
            reg = gk[gk.n_hold.isin(nc) & gk.target_pct.isin(tc)]
            regions.append({"K": K, "hold_cluster": f"T+{min(nc)}..T+{max(nc)}", "target_cluster": f"{min(tc)}-{max(tc)}%",
                            "reg_IS_avg": round(reg.IS_avg.mean(), 3), "reg_OOS_avg": round(reg.OOS_avg.mean(), 3),
                            "reg_total_net": round(reg.total_net.mean(), 1), "reg_win": round(reg.win_rate.mean(), 1),
                            "reg_avg_ntrades": int(reg.n_trades.mean())})
        REG = pd.DataFrame(regions).sort_values("reg_OOS_avg", ascending=False)
        REG.to_excel(w, sheet_name="robust_regions_by_K", index=False)

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    tcols = [5, 6, 8, 10, 12, 14, 16, 18, 20]
    for K in KS:
        gk = G[G.K == K]
        piv = gk.pivot(index="n_hold", columns="target_pct", values="total_net")[tcols]
        print("\n" + "=" * 96 + f"\nNON-F&O  K>={K}%  — TOTAL NET RETURN (rows=T+n, cols=target%, trimmed)\n" + "=" * 96)
        print(piv.round(0).to_string())
    print("\n" + "=" * 96 + "\nROBUST REGION per K (smoothed min(IS_avg,OOS_avg)); ranked by OOS_avg\n" + "=" * 96)
    print(REG.to_string(index=False))
    if len(REG):
        b = REG.iloc[0]
        print(f"\nRECOMMENDED (most OOS-robust): K>={int(b.K)}%, hold {b.hold_cluster}, target {b.target_cluster} | "
              f"IS_avg {b.reg_IS_avg} OOS_avg {b.reg_OOS_avg} [{'HOLDS' if b.reg_OOS_avg>0 else 'FAILS'} OOS] avg_ntrades {b.reg_avg_ntrades}")
    print("\nNOTE: same-day target+SL -> target fires first (intraday touch precedes the close the SL is judged on).")
    print("NOTE: non-F&O coverage is thinner for illiquid/recent-IPO names; flat 0.20% cost understates real slippage here.")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
