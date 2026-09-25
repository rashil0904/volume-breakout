# -*- coding: utf-8 -*-
"""pead_long_hold_target_grid.py — ADDITIVE F&O LONG extension: holding-period T+10..T+15 x profit-target
5-20% (1% steps) grid, plus a no-target reference column. F&O LONGS ONLY (K=4% reaction vs prev-day
VWAP-close, entry=T0 close, during/post->next-day T0). SHORTS NOT TOUCHED. Runs on the CORRECTED full
event set (fixes the DD-MM datetime parse that silently dropped 63% of events in the earlier sweep).
Exit mechanics (mixed, per spec): TARGET touch/trigger (intraday high>=tgt, fill max(tgt,open)); SL
close-basis (close<=T0 low, fill that close); HOLD-END at T+n close. Priority per day: target(intraday)
BEFORE SL(close); else next day; else period-end. IS 2022-2024 / OOS 2025+. Best region = smoothed
neighborhood peak (not a single cell).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"   # reconciled F&O
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
VWAPD = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "pead_long_hold_target_grid"
K, COST = 4.0, 0.20
NS = list(range(10, 16))                 # T+10 .. T+15
TGTS = list(range(5, 21))                # 5% .. 20%
NMAX = max(NS)
IS_YEARS = {2022, 2023, 2024}


def build():
    a = pd.read_csv(ANN, dtype=str)
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
        if (dates[p] - dates[p - 1]).days > 6:                # GUARD1: stale prev-VWAP (data gap) -> skip
            continue
        key = (r.symbol, int(p))                              # GUARD2: de-dup by (symbol, T0)
        if key in seen:
            continue
        seen.add(key)
        entry = cl[p]; prev_vwap = vmap.get((r.symbol, dates[p - 1]), np.nan)
        if not (entry > 0 and prev_vwap == prev_vwap and prev_vwap > 0):
            continue
        rr = (entry - prev_vwap) / prev_vwap * 100.0
        if abs(rr) > 50.0:                                    # GUARD3: corporate-action / bad-print -> skip
            continue
        if rr < K:                                            # LONGS ONLY
            continue
        avail = len(dates) - 1 - p                            # forward days available
        if avail < 1:
            continue
        m = min(NMAX, avail)
        ev.append({"symbol": r.symbol, "year": int(str(dates[p])[:4]), "entry": entry, "sl": lo[p], "navail": m,
                   "fo": op[p + 1:p + 1 + m], "fh": hi[p + 1:p + 1 + m], "fl": lo[p + 1:p + 1 + m], "fc": cl[p + 1:p + 1 + m]})
    return ev


def exit_long(e, n, tgt_pct):
    entry, sl = e["entry"], e["sl"]; fo, fh, fc = e["fo"], e["fh"], e["fc"]
    tgt = None if tgt_pct is None else entry * (1 + tgt_pct / 100.0)
    for j in range(n):
        if tgt is not None and fh[j] >= tgt:                  # TARGET: intraday touch, checked BEFORE close-SL
            return max(tgt, fo[j]), "target", j + 1
        if fc[j] <= sl:                                       # SL: close-basis
            return fc[j], "close_sl", j + 1
    return fc[n - 1], "holding_end", n


def cell(ev, n, tgt_pct):
    sub = [e for e in ev if e["navail"] >= n]
    g = np.empty(len(sub)); yrs = np.empty(len(sub), int); reasons = []
    for i, e in enumerate(sub):
        exp, reason, dh = exit_long(e, n, tgt_pct)
        g[i] = (exp - e["entry"]) / e["entry"] * 100.0; yrs[i] = e["year"]; reasons.append(reason)
    net = g - COST; ism = np.isin(yrs, list(IS_YEARS)); reasons = np.array(reasons)
    return {"n_hold": n, "target_pct": tgt_pct if tgt_pct is not None else "none", "n_trades": len(sub),
            "total_net": round(net.sum(), 1), "avg_net": round(net.mean(), 3) if len(sub) else np.nan,
            "win_rate": round((g > 0).mean() * 100, 1) if len(sub) else np.nan,
            "pct_target": round((reasons == "target").mean() * 100, 1) if len(sub) else np.nan,
            "IS_total": round(net[ism].sum(), 1), "OOS_total": round(net[~ism].sum(), 1),
            "IS_avg": round(net[ism].mean(), 3) if ism.any() else np.nan,
            "OOS_avg": round(net[~ism].mean(), 3) if (~ism).any() else np.nan, "IS_n": int(ism.sum()), "OOS_n": int((~ism).sum())}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ev = build()
    print(f"F&O LONG events (corrected full set): {len(ev)} | with >=15 fwd days: {sum(1 for e in ev if e['navail']>=15)}", flush=True)

    recs = [cell(ev, n, None) for n in NS] + [cell(ev, n, t) for n in NS for t in TGTS]
    G = pd.DataFrame(recs)
    swept = G[G.target_pct != "none"].copy(); swept["target_pct"] = swept["target_pct"].astype(int)

    piv_tot = swept.pivot(index="n_hold", columns="target_pct", values="total_net")
    piv_win = swept.pivot(index="n_hold", columns="target_pct", values="win_rate")
    piv_is = swept.pivot(index="n_hold", columns="target_pct", values="IS_total")
    piv_oos = swept.pivot(index="n_hold", columns="target_pct", values="OOS_total")
    piv_isavg = swept.pivot(index="n_hold", columns="target_pct", values="IS_avg")
    piv_oosavg = swept.pivot(index="n_hold", columns="target_pct", values="OOS_avg")
    notgt = G[G.target_pct == "none"][["n_hold", "n_trades", "total_net", "avg_net", "win_rate", "IS_total", "OOS_total"]]

    def smooth(P):
        M = P.values.astype(float); sm = np.full_like(M, np.nan)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                sm[i, j] = np.nanmean(M[max(0, i - 1):i + 2, max(0, j - 1):j + 2])
        return sm
    ns_v = list(piv_isavg.index); ts_v = list(piv_isavg.columns)
    def region_of(sm):
        bi, bj = np.unravel_index(np.nanargmax(sm), sm.shape)
        nc = ns_v[max(0, bi - 1):bi + 2]; tc = ts_v[max(0, bj - 1):bj + 2]
        r = swept[(swept.n_hold.isin(nc)) & (swept.target_pct.isin(tc))]
        return nc, tc, round(r["IS_avg"].mean(), 3), round(r["OOS_avg"].mean(), 3), round(r["total_net"].mean(), 1)
    # (a) IS-peak region (overfit-prone reference); (b) ROBUST region = smoothed min(IS_avg, OOS_avg)
    ncluster, tcluster, reg_is_avg, reg_oos_avg, reg_tot = region_of(smooth(piv_isavg))
    robust = smooth(pd.DataFrame(np.minimum(piv_isavg.values, piv_oosavg.values), index=piv_isavg.index, columns=piv_isavg.columns))
    rnc, rtc, r_is, r_oos, r_tot = region_of(robust)

    with pd.ExcelWriter(OUTDIR / "pead_long_hold_target_grid.xlsx", engine="openpyxl") as w:
        G.to_excel(w, sheet_name="all_cells", index=False)
        piv_tot.to_excel(w, sheet_name="grid_total_net")
        piv_win.to_excel(w, sheet_name="grid_win_rate")
        piv_is.to_excel(w, sheet_name="grid_IS_total")
        piv_oos.to_excel(w, sheet_name="grid_OOS_total")
        notgt.to_excel(w, sheet_name="no_target_reference", index=False)

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 100 + "\nF&O LONG — TOTAL NET RETURN grid  (rows=T+n hold, cols=target%)\n" + "=" * 100)
    print(piv_tot.round(0).to_string())
    print("\n--- WIN RATE % grid ---"); print(piv_win.to_string())
    print("\n--- NO-TARGET reference (hold-cap + close-SL only) ---"); print(notgt.to_string(index=False))
    print("\n--- IS (2022-24) TOTAL grid ---"); print(piv_is.round(0).to_string())
    print("\n--- OOS (2025+) TOTAL grid ---"); print(piv_oos.round(0).to_string())
    print("\n" + "=" * 100)
    print(f"(a) IS-PEAK region (OVERFIT-PRONE, reference only): T+{ncluster} x target {tcluster}%")
    print(f"     IS_avg={reg_is_avg}  OOS_avg={reg_oos_avg}  -> OOS degrades {round((1-reg_oos_avg/reg_is_avg)*100)}% vs IS  mean total_net={reg_tot}")
    print(f"(b) ROBUST region [smoothed min(IS,OOS)] -> RECOMMENDED: T+{rnc} x target {rtc}%")
    print(f"     IS_avg={r_is}  OOS_avg={r_oos}  [{'HOLDS' if r_oos>0 else 'FAILS'} OOS]  mean total_net={r_tot}")
    best_row = swept.sort_values("total_net", ascending=False).iloc[0]
    print(f"  single best cell (do NOT trade on this): T+{int(best_row.n_hold)} tgt{int(best_row.target_pct)}% total={best_row.total_net} OOS_avg={best_row.OOS_avg}")
    nt10 = notgt[notgt.n_hold == 10].iloc[0]
    print(f"  no-target: OOS_total is NEGATIVE for every hold ({notgt['OOS_total'].min()}..{notgt['OOS_total'].max()}) -> in 2025+ an untargeted long gives back its drift")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
