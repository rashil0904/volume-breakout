# -*- coding: utf-8 -*-
"""vb_exit_time_grid_sweep_apr_jul2026.py — Full-grid sweep of the LONG EXIT's conditional-split
timing (t1/t2) on the main NSE Volume-Breakout BTST strategy, RESTRICTED to April-July 2026 only.
Category A/B/C entry logic, the 17% target overlay, the short's 14:39/5%-target cover rule, and
capital allocation are all byte-identical to the locked baseline_and_cross_final.py -- UNCHANGED.
Only the two long-exit timestamps (t1: "exit now if positive", t2: "force exit the rest") are swept.

Reports BOTH:
  - LONG-ONLY: the double-down short stripped out entirely; net_A/net_B use only the long round-trip
    cost (R023/R038 * capital_deployed), no short-side cost term -- the natural extension of the
    existing net_A/net_B cost convention with the short leg removed.
  - LONG+SHORT (combined): identical to the current locked strategy's full structure.

New diagnostic script; new output folder; the locked baseline_and_cross_final.py is not modified.
"""
import sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC

OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "exit_time_grid_sweep_apr_jul2026"
OUTDIR.mkdir(parents=True, exist_ok=True)
WIN_START = pd.Timestamp("2026-04-01").date(); WIN_END = pd.Timestamp("2026-07-31").date()

OPEN_LO, OPEN_HI = 556, 720                 # 9:16am .. 12:00pm -- the t1/t2 sweep grid
DATA_LO_HI, DATA_HI_HI = 555, 720           # window for target-hit-time + open-price lookups
DATA_LO_LO, DATA_HI_LO = 555, BC.COVER_HM   # window for short 5%-target low-price checks (.. 14:39)
BASE_T1, BASE_T2 = BC.T1, BC.T2             # 565 (9:25) / 719 (11:59) -- current locked baseline


def build_wide_cache():
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                        usecols=["symbol", "date", "passes_all_three", "prev_day_vwap_close"],
                        parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True]
    cache = []; t0 = time.time()
    for si, sym in enumerate(sorted(Q["symbol"].unique()), 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        rows_here = Q[Q["symbol"] == sym]
        rows_here = rows_here[(rows_here["date"] >= WIN_START) & (rows_here["date"] <= WIN_END)]
        if rows_here.empty:
            continue
        raw = pd.read_parquet(pq)
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(BC.IST)
        raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
        bd = {d: g for d, g in raw.groupby("date")}
        dts = sorted(bd)
        for _, r in rows_here.iterrows():
            ed = r["date"]; pc = float(r["prev_day_vwap_close"])
            if ed not in bd or not (pc == pc and pc > 0):
                continue
            j = dts.index(ed)
            nd = dts[j + 1] if j < len(dts) - 1 else None
            if nd is None:
                continue
            g = bd[ed]
            eg = {"hm": g["hm"].values, "hi": g["high"].values.astype(float),
                  "lo": g["low"].values.astype(float), "op": g["open"].values.astype(float)}
            ng = bd[nd]
            nhm = ng["hm"].values; nhi = ng["high"].values.astype(float)
            nlo = ng["low"].values.astype(float); nop = ng["open"].values.astype(float)
            hi_mask = (nhm >= DATA_LO_HI) & (nhm <= DATA_HI_HI)
            lo_mask = (nhm >= DATA_LO_LO) & (nhm <= DATA_HI_LO)
            no_dict = dict(zip(nhm, nop))
            cache.append({"symbol": sym, "entry_date": ed, "pc": pc, "eg": eg,
                          "nhm_hi": nhm[hi_mask], "nhi_hi": nhi[hi_mask], "nop_hi": nop[hi_mask],
                          "nhm_lo": nhm[lo_mask], "nlo_lo": nlo[lo_mask],
                          "o879": no_dict.get(BC.COVER_HM, np.nan)})
        if si % 300 == 0:
            print(f"  ...{si} symbols ({time.time()-t0:.0f}s), {len(cache):,} stock-days in window so far", flush=True)
    return cache


def entry_positions(cache):
    """UNCHANGED Category A/B/C classify() + capital sequencing/sizing -- stops right before exit-time
    logic, since none of that depends on t1/t2."""
    recs = []
    for c in cache:
        cat, entered, legs, meta = BC.classify("baseline", c["pc"], c["eg"]["hm"], c["eg"]["hi"], c["eg"]["lo"], c["eg"]["op"])
        recs.append({**{k: c[k] for k in ("symbol", "entry_date", "nhm_hi", "nhi_hi", "nop_hi", "nhm_lo", "nlo_lo", "o879")},
                     "category": cat, "entered": entered, "legs": legs})
    ent = [r for r in recs if r["entered"] and r["legs"]]
    for r in ent:
        r["_a"] = [0.0] * len(r["legs"])
    by_day = defaultdict(list)
    for r in ent:
        by_day[r["entry_date"]].append(r)
    for d, rs in by_day.items():
        pool = BC.BASE_POOL; p1, p2, cc = [], [], []
        for r in rs:
            for li, (frac, price, thm, ph) in enumerate(r["legs"]):
                (cc if ph == 3 else p2 if ph == 2 else p1).append(
                    (r, li, price) if ph == 3 else (r, li, 0.5 * BC.BASE_ALLOC) if ph == 2
                    else (thm, r, li, BC.BASE_ALLOC if frac >= 1.0 else 0.5 * BC.BASE_ALLOC))
        for thm, r, li, intd in sorted(p1, key=lambda x: (x[0] if x[0] is not None else 99999)):
            give = min(intd, pool); r["_a"][li] = give; pool -= give
        for r, li, intd in p2:
            give = min(intd, pool); r["_a"][li] = give; pool -= give
        nC = len(cc)
        per_c = min(BC.BASE_ALLOC, max(0.0, pool) / nC) if nC else 0.0
        for r, li, price in cc:
            r["_a"][li] = per_c

    rows = []
    for r in ent:
        shares = cap = 0.0
        for li, (frac, price, thm, ph) in enumerate(r["legs"]):
            s = np.floor(r["_a"][li] / price) if r["_a"][li] > 0 else 0.0
            if s > 0:
                shares += s; cap += s * price
        if shares <= 0:
            continue
        avg = cap / shares
        rows.append({"symbol": r["symbol"], "entry_date": r["entry_date"], "avg": avg, "shares": shares, "cap": cap,
                     "nhm_hi": r["nhm_hi"], "nhi_hi": r["nhi_hi"], "nop_hi": r["nop_hi"],
                     "nhm_lo": r["nhm_lo"], "nlo_lo": r["nlo_lo"], "o879": r["o879"]})
    return rows


def build_matrices(rows):
    n = len(rows)
    avg = np.array([r["avg"] for r in rows]); shares = np.array([r["shares"] for r in rows])
    cap = np.array([r["cap"] for r in rows]); o879 = np.array([r["o879"] for r in rows])
    tgt = avg * BC.LONG_TGT

    open_cols = np.arange(DATA_LO_HI, DATA_HI_HI + 1)
    OPEN_M = np.full((n, len(open_cols)), np.nan)
    th = np.full(n, np.nan)
    low_cols = np.arange(DATA_LO_LO, DATA_HI_LO + 1)
    LOW_M = np.full((n, len(low_cols)), np.nan)

    for i, r in enumerate(rows):
        s_open = pd.Series(r["nop_hi"], index=r["nhm_hi"])
        s_open = s_open[~s_open.index.duplicated(keep="first")]
        OPEN_M[i, :] = s_open.reindex(open_cols).values

        hit = r["nhi_hi"] >= tgt[i]
        if hit.any():
            th[i] = r["nhm_hi"][hit].min()

        s_low = pd.Series(r["nlo_lo"], index=r["nhm_lo"])
        s_low = s_low[~s_low.index.duplicated(keep="first")]
        LOW_M[i, :] = s_low.reindex(low_cols).values

    return dict(avg=avg, shares=shares, cap=cap, o879=o879, tgt=tgt, th=th,
                open_cols=open_cols, OPEN_M=OPEN_M, low_cols=low_cols, LOW_M=LOW_M)


def run_grid(M):
    TIMES = list(range(OPEN_LO, OPEN_HI + 1))
    total_combos = len(TIMES) * (len(TIMES) - 1) // 2
    print(f"time grid: {len(TIMES)} minutes ({BC.lbl(OPEN_LO)}..{BC.lbl(OPEN_HI)}) -> {total_combos:,} valid (t1<t2) combinations", flush=True)

    avg, shares, cap, o879, tgt, th = M["avg"], M["shares"], M["cap"], M["o879"], M["tgt"], M["th"]
    low_cols, LOW_M = M["low_cols"], M["LOW_M"]
    results = []; t0 = time.time()
    for t1 in TIMES:
        ot1 = M["OPEN_M"][:, t1 - DATA_LO_HI]
        m1 = th <= t1
        m2 = (~m1) & (ot1 == ot1) & (ot1 > avg)
        base_valid = m1 | m2
        xp12 = np.where(m1, tgt, np.where(m2, ot1, np.nan))
        xhm12 = np.where(m1, th, np.where(m2, t1, np.nan))
        for t2 in TIMES:
            if t2 <= t1:
                continue
            ot2 = M["OPEN_M"][:, t2 - DATA_LO_HI]
            m3 = (~base_valid) & (th <= t2)
            m4 = (~base_valid) & (~m3) & (ot2 == ot2)
            xp = np.where(base_valid, xp12, np.where(m3, tgt, np.where(m4, ot2, np.nan)))
            xhm = np.where(base_valid, xhm12, np.where(m3, th, np.where(m4, t2, np.nan)))
            valid = ~np.isnan(xp)
            n_trades = int(valid.sum())
            if n_trades == 0:
                continue

            long_pnl = shares * (xp - avg)
            lg = float(np.nansum(long_pnl))
            la = float(np.nansum(long_pnl - BC.R023 * cap))
            lb = float(np.nansum(long_pnl - BC.R038 * cap))
            lw = float((long_pnl[valid] > 0).mean() * 100)

            stgt = xp * BC.SHORT_TGT
            sw_any = ((low_cols[None, :] > xhm[:, None]) & (LOW_M <= stgt[:, None])).any(axis=1)
            cover = np.where(sw_any, stgt, np.where(o879 == o879, o879, np.nan))
            has_short = cover == cover
            short_pnl = np.where(has_short, shares * (xp - cover), 0.0)
            snotl = np.where(has_short, shares * xp, 0.0)
            comb = long_pnl + short_pnl
            cg = float(np.nansum(comb))
            ca = float(np.nansum(comb - BC.R023 * cap - BC.SR * snotl))
            cb_ = float(np.nansum(comb - BC.R038 * cap - BC.SR * snotl))
            cw = float((comb[valid] > 0).mean() * 100)

            results.append({"t1": t1, "t2": t2, "t1_lbl": BC.lbl(t1), "t2_lbl": BC.lbl(t2), "n_trades": n_trades,
                             "lo_gross": round(lg, 0), "lo_netA": round(la, 0), "lo_netB": round(lb, 0), "lo_win": round(lw, 2),
                             "cb_gross": round(cg, 0), "cb_netA": round(ca, 0), "cb_netB": round(cb_, 0), "cb_win": round(cw, 2)})
        if (t1 - OPEN_LO) % 20 == 0:
            print(f"  t1={BC.lbl(t1)} done ({time.time()-t0:.0f}s elapsed, {len(results):,} combos so far)", flush=True)
    print(f"grid complete: {len(results):,} combos in {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(results)


def smooth3x3(pivot):
    """Simple NaN-aware 3x3 neighborhood mean over a (t1 x t2) pivot table, for stable-region ID."""
    vals = pivot.values.astype(float)
    out = np.full_like(vals, np.nan)
    nrow, ncol = vals.shape
    for i in range(nrow):
        for j in range(ncol):
            if np.isnan(vals[i, j]):
                continue
            i0, i1 = max(0, i - 1), min(nrow, i + 2)
            j0, j1 = max(0, j - 1), min(ncol, j + 2)
            block = vals[i0:i1, j0:j1]
            out[i, j] = np.nanmean(block) if np.any(~np.isnan(block)) else np.nan
    return pd.DataFrame(out, index=pivot.index, columns=pivot.columns)


def main():
    print("Building wide-window cache (Apr-Jul 2026 only)...", flush=True)
    cache = build_wide_cache()
    print(f"cache: {len(cache):,} stock-days in window", flush=True)

    rows = entry_positions(cache)
    print(f"entered & sized positions: {len(rows):,}", flush=True)

    M = build_matrices(rows)
    RES = run_grid(M)

    base = RES[(RES["t1"] == BASE_T1) & (RES["t2"] == BASE_T2)]
    if base.empty:
        raise RuntimeError("baseline (t1=9:25, t2=11:59) not found in grid results")
    base = base.iloc[0]
    print(f"\nBASELINE (t1={base['t1_lbl']}, t2={base['t2_lbl']}): "
          f"long-only netA={base['lo_netA']:.0f} | combined netA={base['cb_netA']:.0f} | n={base['n_trades']}", flush=True)

    for pfx in ["lo", "cb"]:
        for s in ["gross", "netA", "netB"]:
            RES[f"delta_{pfx}_{s}"] = (RES[f"{pfx}_{s}"] - base[f"{pfx}_{s}"]).round(0)

    LO_sorted = RES.sort_values("delta_lo_netA", ascending=False).reset_index(drop=True)
    CB_sorted = RES.sort_values("delta_cb_netA", ascending=False).reset_index(drop=True)

    pd.set_option("display.width", 260)
    print("\n=== TOP 15 LONG-ONLY combos (by delta net_A vs baseline) ===")
    print(LO_sorted[["t1_lbl", "t2_lbl", "n_trades", "lo_gross", "lo_netA", "lo_netB", "delta_lo_netA", "lo_win"]].head(15).to_string(index=False))
    print("\n=== TOP 15 LONG+SHORT combos (by delta net_A vs baseline) ===")
    print(CB_sorted[["t1_lbl", "t2_lbl", "n_trades", "cb_gross", "cb_netA", "cb_netB", "delta_cb_netA", "cb_win"]].head(15).to_string(index=False))

    best_lo = LO_sorted.iloc[0]; best_cb = CB_sorted.iloc[0]
    print(f"\nBest LONG-ONLY combo:   t1={best_lo['t1_lbl']} t2={best_lo['t2_lbl']}  (delta netA {best_lo['delta_lo_netA']:+.0f})")
    print(f"Best LONG+SHORT combo:  t1={best_cb['t1_lbl']} t2={best_cb['t2_lbl']}  (delta netA {best_cb['delta_cb_netA']:+.0f})")
    same_pair = (best_lo["t1"] == best_cb["t1"]) and (best_lo["t2"] == best_cb["t2"])
    print(f"Same (t1,t2) pair for both: {same_pair}")

    # pivots + smoothed "stable region" maps
    piv_lo = RES.pivot(index="t1_lbl", columns="t2_lbl", values="delta_lo_netA")
    piv_cb = RES.pivot(index="t1_lbl", columns="t2_lbl", values="delta_cb_netA")
    # keep chronological (minute) row/col order, not alphabetical string order
    order_lbls = [BC.lbl(t) for t in range(OPEN_LO, OPEN_HI + 1)]
    piv_lo = piv_lo.reindex(index=order_lbls, columns=order_lbls)
    piv_cb = piv_cb.reindex(index=order_lbls, columns=order_lbls)
    smooth_lo = smooth3x3(piv_lo)
    smooth_cb = smooth3x3(piv_cb)

    def top_smoothed(smooth_df, n=10):
        s = smooth_df.stack().rename("smoothed_delta").reset_index()
        s.columns = ["t1_lbl", "t2_lbl", "smoothed_delta"]
        return s.sort_values("smoothed_delta", ascending=False).head(n)

    top_lo_region = top_smoothed(smooth_lo)
    top_cb_region = top_smoothed(smooth_cb)
    print("\n=== LONG-ONLY: top smoothed (3x3 neighborhood avg) regions ===")
    print(top_lo_region.to_string(index=False))
    print("\n=== LONG+SHORT: top smoothed (3x3 neighborhood avg) regions ===")
    print(top_cb_region.to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "exit_time_grid_sweep_apr_jul2026.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": f"Full-grid sweep of the long exit's conditional-split t1/t2 timestamps, RESTRICTED to "
                      f"{WIN_START}..{WIN_END}. Category A/B/C entry logic, the 17% target overlay, and the "
                      "short's 14:39/5%-target cover rule are all UNCHANGED, byte-identical to the locked "
                      "baseline_and_cross_final.py -- only t1 (positive-exit time) and t2 (force-exit time) move."},
            {"note": f"Sweep grid: every minute {BC.lbl(OPEN_LO)}-{BC.lbl(OPEN_HI)} for both t1 and t2, t1<t2 only "
                      f"({len(range(OPEN_LO, OPEN_HI+1))} times -> {len(RES):,} valid combinations)."},
            {"note": "Exit fills use the OPEN price of the swept minute's candle, matching the strategy's own "
                      "existing convention (o565/o719 = opens at the fixed 9:25/11:59 times in the locked code)."},
            {"note": "LONG-ONLY net_A/net_B = long_pnl minus ONLY the long round-trip cost (R023/R038 x capital "
                      "deployed) -- no short-side cost term, since there is no short in this version. This is the "
                      "natural extension of the existing cost convention with the double-down short removed."},
            {"note": f"BASELINE (current locked setting) = t1={base['t1_lbl']}, t2={base['t2_lbl']}. Long-only netA "
                      f"= {base['lo_netA']:.0f}; combined netA = {base['cb_netA']:.0f}; n={int(base['n_trades'])} trades."},
            {"note": f"SAMPLE SIZE FLAG: {WIN_START}..{WIN_END} is a ~4-month window ({int(base['n_trades'])} trades "
                      "at baseline). Treat all findings here as a narrow-period observation, not a generalizable "
                      "conclusion -- look at the smoothed/stable region, not a single best cell."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        LO_sorted.to_excel(w, sheet_name="Long_Only_Full_Grid", index=False)
        CB_sorted.to_excel(w, sheet_name="Combined_Full_Grid", index=False)
        piv_lo.to_excel(w, sheet_name="Long_Only_Heatmap_deltaNetA")
        piv_cb.to_excel(w, sheet_name="Combined_Heatmap_deltaNetA")
        smooth_lo.to_excel(w, sheet_name="Long_Only_Smoothed3x3")
        smooth_cb.to_excel(w, sheet_name="Combined_Smoothed3x3")
        top_lo_region.to_excel(w, sheet_name="Long_Only_Top_Regions", index=False)
        top_cb_region.to_excel(w, sheet_name="Combined_Top_Regions", index=False)
        pd.DataFrame([
            {"metric": "best_long_only_t1", "value": best_lo["t1_lbl"]},
            {"metric": "best_long_only_t2", "value": best_lo["t2_lbl"]},
            {"metric": "best_long_only_delta_netA", "value": best_lo["delta_lo_netA"]},
            {"metric": "best_combined_t1", "value": best_cb["t1_lbl"]},
            {"metric": "best_combined_t2", "value": best_cb["t2_lbl"]},
            {"metric": "best_combined_delta_netA", "value": best_cb["delta_cb_netA"]},
            {"metric": "same_pair_for_both", "value": same_pair},
            {"metric": "baseline_t1", "value": base["t1_lbl"]}, {"metric": "baseline_t2", "value": base["t2_lbl"]},
            {"metric": "baseline_n_trades", "value": int(base["n_trades"])},
        ]).to_excel(w, sheet_name="Best_Combo_Comparison", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 22)

    print(f"\nSaved -> {OUTDIR}/exit_time_grid_sweep_apr_jul2026.xlsx")


if __name__ == "__main__":
    main()
