# -*- coding: utf-8 -*-
"""vb_t1t2_cover_full_grid_sweep.py — Full 3-D sweep (long-exit t1 x long-exit t2 x short cover time) on
the main NSE Volume-Breakout BTST strategy's locked baseline. Category A/B/C entry logic, the 17% target
overlay, and the short's entry-tied-to-long-exit rule are all UNCHANGED, byte-identical to the locked
baseline_and_cross_final.py -- only t1, t2, and the short's cover-time ceiling are swept. Runs on the FULL
backtest history, plus a 70/30-by-calendar-date out-of-sample check (fit on the first 70% of the date
span, confirm the fit-period's best region on the held-out final 30%).

GRID: t1/t2 in [9:16am, 12:00pm] (165 minutes, 13,530 valid t1<t2 pairs) x cover in [2:00pm, 3:00pm]
(61 minutes) = 825,330 combinations -- small enough to materialize IN FULL (unlike the earlier 4-D sweep
that also included a Category C dimension), so every combo's gross/net_A/net_B/win/n is saved to disk.
LONG-ONLY (short stripped out, cost = long round-trip only, no short-side cost term) is computed
alongside LONG+SHORT for every (t1,t2) pair -- long-only has no cover dependency, so it is a 13,530-row
grid, reported next to the 825,330-row combined grid for direct comparison at the best cells.
"""
import sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC

OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "t1t2_cover_full_grid_sweep"
OUTDIR.mkdir(parents=True, exist_ok=True)

T_RANGE = list(range(556, 721))       # 9:16am-12:00pm, 165 values (t1/t2)
COVER_RANGE = list(range(840, 901))   # 2:00pm-3:00pm, 61 values
DATA_LO, DATA_HI = 555, 900           # unified next-day window for hi/lo/open lookups
BASE_T1, BASE_T2, BASE_COVER = BC.T1, BC.T2, BC.COVER_HM   # 9:25 / 11:59 / 14:39
NBHD = 2


VWAP_LO, VWAP_HI = 900, 929   # 15:00-15:29, for the entry-day's OWN VWAP-close (bucket segmentation reuse)


def _vwap_by_date(raw):
    """Same formula as prepare_data.py's prev_day_vwap_close: typical price (H+L+C)/3 over 15:00-15:29."""
    last30 = raw[(raw["hm"] >= VWAP_LO) & (raw["hm"] <= VWAP_HI)].copy()
    last30["tp"] = (last30["high"] + last30["low"] + last30["close"]) / 3.0
    last30["tp_vol"] = last30["tp"] * last30["volume"]
    g = last30.groupby("date")[["tp_vol", "volume"]].sum()
    vwap = (g["tp_vol"] / g["volume"]).where(g["volume"] > 0, np.nan)
    return vwap.to_dict()


def build_wide_cache(entry_lo=None, entry_hi=None):
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
        raw = pd.read_parquet(pq)
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(BC.IST)
        raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
        bd = {d: g for d, g in raw.groupby("date")}
        dts = sorted(bd)
        vwap_map = _vwap_by_date(raw)   # entry-day's OWN VWAP-close, per date (for bucket segmentation)
        for _, r in Q[Q["symbol"] == sym].iterrows():
            ed = r["date"]
            if entry_lo is not None and not (entry_lo <= ed <= entry_hi):
                continue
            pc = float(r["prev_day_vwap_close"])
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
            wmask = (nhm >= DATA_LO) & (nhm <= DATA_HI)
            entry_vwap = vwap_map.get(ed, np.nan)
            vwap_move_pct = (entry_vwap - pc) / pc * 100 if entry_vwap == entry_vwap else np.nan
            cache.append({"symbol": sym, "entry_date": ed, "pc": pc, "eg": eg,
                          "nhm_w": nhm[wmask], "nhi_w": nhi[wmask], "nlo_w": nlo[wmask], "nop_w": nop[wmask],
                          "vwap_move_pct": vwap_move_pct})
        if si % 300 == 0:
            print(f"  ...{si} symbols ({time.time()-t0:.0f}s), {len(cache):,} stock-days so far", flush=True)
    print(f"cache built: {len(cache):,} stock-days ({time.time()-t0:.0f}s)", flush=True)
    return cache


def entry_positions(cache):
    """UNCHANGED Category A/B/C classify() (standard "baseline" config) + capital sequencing/sizing."""
    recs = []
    for c in cache:
        cat, entered, legs, meta = BC.classify("baseline", c["pc"], c["eg"]["hm"], c["eg"]["hi"], c["eg"]["lo"], c["eg"]["op"])
        recs.append({"symbol": c["symbol"], "entry_date": c["entry_date"], "entered": entered, "legs": legs,
                     "nhm_w": c["nhm_w"], "nhi_w": c["nhi_w"], "nlo_w": c["nlo_w"], "nop_w": c["nop_w"],
                     "vwap_move_pct": c.get("vwap_move_pct", np.nan)})
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
        rows.append({"avg": avg, "shares": shares, "cap": cap,
                     "nhm_w": r["nhm_w"], "nhi_w": r["nhi_w"], "nlo_w": r["nlo_w"], "nop_w": r["nop_w"],
                     "symbol": r["symbol"], "entry_date": r["entry_date"], "vwap_move_pct": r["vwap_move_pct"]})
    return rows


def build_matrices(rows):
    n = len(rows)
    avg = np.array([r["avg"] for r in rows]); shares = np.array([r["shares"] for r in rows]); cap = np.array([r["cap"] for r in rows])
    tgt = avg * BC.LONG_TGT
    cols = np.arange(DATA_LO, DATA_HI + 1)
    OPEN_M = np.full((n, len(cols)), np.nan)
    LOW_M = np.full((n, len(cols)), np.nan)
    th = np.full(n, np.nan)
    for i, r in enumerate(rows):
        s_open = pd.Series(r["nop_w"], index=r["nhm_w"]); s_open = s_open[~s_open.index.duplicated(keep="first")]
        OPEN_M[i, :] = s_open.reindex(cols).values
        s_low = pd.Series(r["nlo_w"], index=r["nhm_w"]); s_low = s_low[~s_low.index.duplicated(keep="first")]
        LOW_M[i, :] = s_low.reindex(cols).values
        hi_mask = r["nhm_w"] <= 720
        if hi_mask.any():
            hit = r["nhi_w"][hi_mask] >= tgt[i]
            if hit.any():
                th[i] = r["nhm_w"][hi_mask][hit].min()
    vwap_move_pct = np.array([r.get("vwap_move_pct", np.nan) for r in rows])
    return dict(avg=avg, shares=shares, cap=cap, tgt=tgt, th=th, cols=cols, OPEN_M=OPEN_M, LOW_M=LOW_M,
                vwap_move_pct=vwap_move_pct)


def run_grid(M, label):
    avg, shares, cap, tgt, th, cols, OPEN_M, LOW_M = M["avg"], M["shares"], M["cap"], M["tgt"], M["th"], M["cols"], M["OPEN_M"], M["LOW_M"]
    cover_arr = np.array(COVER_RANGE)
    cover_col_idx = cover_arr - DATA_LO
    lo_rows, cb_rows = [], []
    t0 = time.time()
    for t1 in T_RANGE:
        ot1 = OPEN_M[:, t1 - DATA_LO]
        m1 = th <= t1
        m2 = (~m1) & (ot1 == ot1) & (ot1 > avg)
        base_valid = m1 | m2
        xp12 = np.where(m1, tgt, np.where(m2, ot1, np.nan))
        xhm12 = np.where(m1, th, np.where(m2, t1, np.nan))
        for t2 in T_RANGE:
            if t2 <= t1:
                continue
            ot2 = OPEN_M[:, t2 - DATA_LO]
            m3 = (~base_valid) & (th <= t2)
            m4 = (~base_valid) & (~m3) & (ot2 == ot2)
            xp = np.where(base_valid, xp12, np.where(m3, tgt, np.where(m4, ot2, np.nan)))
            xhm = np.where(base_valid, xhm12, np.where(m3, th, np.where(m4, t2, np.nan)))
            valid = ~np.isnan(xp)
            n_trades = int(valid.sum())
            if n_trades == 0:
                continue
            long_pnl = shares * (xp - avg)

            lg = float(np.nansum(long_pnl)); la = float(np.nansum(long_pnl - BC.R023 * cap)); lb = float(np.nansum(long_pnl - BC.R038 * cap))
            lw = float((long_pnl[valid] > 0).mean() * 100)
            lo_rows.append({"t1": t1, "t2": t2, "n_trades": n_trades, "lo_gross": round(lg, 0), "lo_netA": round(la, 0), "lo_netB": round(lb, 0), "lo_win": round(lw, 2)})

            stgt = xp * BC.SHORT_TGT
            mask = (cols[None, :] > xhm[:, None]) & (LOW_M <= stgt[:, None])
            has_hit = mask.any(axis=1)
            idx = np.argmax(mask, axis=1)
            short_hit_time = np.where(has_hit, cols[idx], np.inf)

            cover_open_blk = OPEN_M[:, cover_col_idx]
            cover_from_target = short_hit_time[:, None] <= cover_arr[None, :]
            cover_price = np.where(cover_from_target, stgt[:, None],
                                    np.where(cover_open_blk == cover_open_blk, cover_open_blk, np.nan))
            has_short = cover_price == cover_price
            short_pnl = np.where(has_short, shares[:, None] * (xp[:, None] - cover_price), 0.0)
            snotl = np.where(has_short, shares[:, None] * xp[:, None], 0.0)
            comb = long_pnl[:, None] + short_pnl

            gross_arr = np.nansum(comb, axis=0)
            netA_arr = np.nansum(comb - BC.R023 * cap[:, None] - BC.SR * snotl, axis=0)
            netB_arr = np.nansum(comb - BC.R038 * cap[:, None] - BC.SR * snotl, axis=0)
            win_arr = (comb[valid] > 0).mean(axis=0) * 100

            for k, cover_t in enumerate(cover_arr):
                cb_rows.append({"t1": t1, "t2": t2, "cover": int(cover_t), "n_trades": n_trades,
                                "cb_gross": round(float(gross_arr[k]), 0), "cb_netA": round(float(netA_arr[k]), 0),
                                "cb_netB": round(float(netB_arr[k]), 0), "cb_win": round(float(win_arr[k]), 2)})
    print(f"  [{label}] grid done in {time.time()-t0:.0f}s -> {len(lo_rows):,} (t1,t2) pairs, {len(cb_rows):,} (t1,t2,cover) combos", flush=True)
    return pd.DataFrame(lo_rows), pd.DataFrame(cb_rows)


def eval_baseline(M):
    """Exact baseline (t1=9:25,t2=11:59,cover=14:39) -- long-only and combined."""
    avg, shares, cap, tgt, th, cols, OPEN_M, LOW_M = M["avg"], M["shares"], M["cap"], M["tgt"], M["th"], M["cols"], M["OPEN_M"], M["LOW_M"]
    ot1 = OPEN_M[:, BASE_T1 - DATA_LO]; ot2 = OPEN_M[:, BASE_T2 - DATA_LO]
    m1 = th <= BASE_T1
    m2 = (~m1) & (ot1 == ot1) & (ot1 > avg)
    base_valid = m1 | m2
    m3 = (~base_valid) & (th <= BASE_T2)
    m4 = (~base_valid) & (~m3) & (ot2 == ot2)
    xp = np.where(base_valid, np.where(m1, tgt, ot1), np.where(m3, tgt, np.where(m4, ot2, np.nan)))
    xhm = np.where(base_valid, np.where(m1, th, BASE_T1), np.where(m3, th, np.where(m4, BASE_T2, np.nan)))
    valid = ~np.isnan(xp)
    n_trades = int(valid.sum())
    long_pnl = shares * (xp - avg)
    lo = {"gross": float(np.nansum(long_pnl)), "netA": float(np.nansum(long_pnl - BC.R023 * cap)),
          "netB": float(np.nansum(long_pnl - BC.R038 * cap)), "win": float((long_pnl[valid] > 0).mean() * 100), "n": n_trades}
    stgt = xp * BC.SHORT_TGT
    sw = (cols[None, :] > xhm[:, None]) & (LOW_M <= stgt[:, None])
    has_hit = sw.any(axis=1); idx = np.argmax(sw, axis=1)
    short_hit_time = np.where(has_hit, cols[idx], np.inf)
    cover_open = OPEN_M[:, BASE_COVER - DATA_LO]
    cover_price = np.where(short_hit_time <= BASE_COVER, stgt, np.where(cover_open == cover_open, cover_open, np.nan))
    has_short = cover_price == cover_price
    short_pnl = np.where(has_short, shares * (xp - cover_price), 0.0)
    snotl = np.where(has_short, shares * xp, 0.0)
    comb = long_pnl + short_pnl
    cb = {"gross": float(np.nansum(comb)), "netA": float(np.nansum(comb - BC.R023 * cap - BC.SR * snotl)),
          "netB": float(np.nansum(comb - BC.R038 * cap - BC.SR * snotl)), "win": float((comb[valid] > 0).mean() * 100), "n": n_trades}
    return lo, cb


def analyze_run(label, LO, CB, base_lo, base_cb, outdir):
    LO = LO.copy(); CB = CB.copy()
    LO["delta_netA"] = (LO["lo_netA"] - base_lo["netA"]).round(0)
    CB["delta_netA"] = (CB["cb_netA"] - base_cb["netA"]).round(0)
    best_lo = LO.loc[LO["lo_netA"].idxmax()]
    best_cb = CB.loc[CB["cb_netA"].idxmax()]

    nb = CB[(CB["t1"].between(best_cb["t1"] - NBHD, best_cb["t1"] + NBHD)) &
            (CB["t2"].between(best_cb["t2"] - NBHD, best_cb["t2"] + NBHD)) &
            (CB["cover"].between(best_cb["cover"] - NBHD, best_cb["cover"] + NBHD))]
    nb_stats = {"n_combos": len(nb), "mean_netA": round(nb["cb_netA"].mean(), 0), "min_netA": round(nb["cb_netA"].min(), 0),
                "max_netA": round(nb["cb_netA"].max(), 0), "pct_above_baseline": round((nb["cb_netA"] > base_cb["netA"]).mean() * 100, 1)}

    # long-only at the best COMBINED (t1,t2), to see if short cover timing changes the picture
    lo_at_best_cb = LO[(LO["t1"] == best_cb["t1"]) & (LO["t2"] == best_cb["t2"])]
    lo_at_best_cb_netA = float(lo_at_best_cb["lo_netA"].iloc[0]) if len(lo_at_best_cb) else np.nan

    print(f"\n[{label}] BASELINE: long-only netA={base_lo['netA']:.0f} (n={base_lo['n']}) | combined netA={base_cb['netA']:.0f} (win={base_cb['win']:.1f}%)")
    print(f"[{label}] BEST LONG-ONLY:  t1={BC.lbl(best_lo['t1'])} t2={BC.lbl(best_lo['t2'])} netA={best_lo['lo_netA']:.0f} (delta {best_lo['delta_netA']:+.0f}), win={best_lo['lo_win']:.1f}%, n={int(best_lo['n_trades'])}")
    print(f"[{label}] BEST COMBINED:   t1={BC.lbl(best_cb['t1'])} t2={BC.lbl(best_cb['t2'])} cover={BC.lbl(best_cb['cover'])} netA={best_cb['cb_netA']:.0f} (delta {best_cb['delta_netA']:+.0f}), win={best_cb['cb_win']:.1f}%, n={int(best_cb['n_trades'])}")
    print(f"[{label}]   long-only netA AT that same (t1,t2): {lo_at_best_cb_netA:.0f}  (short cover adds {best_cb['cb_netA']-lo_at_best_cb_netA:+.0f})")
    print(f"[{label}] STABLE REGION (+/-{NBHD}min all 3 dims, {nb_stats['n_combos']} combos): mean={nb_stats['mean_netA']:.0f} min={nb_stats['min_netA']:.0f} max={nb_stats['max_netA']:.0f} | {nb_stats['pct_above_baseline']:.1f}% beat baseline")

    LO.to_parquet(outdir / f"{label}_long_only_grid.parquet", index=False)
    CB.to_parquet(outdir / f"{label}_combined_grid.parquet", index=False)
    summary = {
        "run": label, "n_trades_baseline": base_cb["n"],
        "baseline_lo_netA": round(base_lo["netA"], 0), "baseline_cb_netA": round(base_cb["netA"], 0), "baseline_cb_win": round(base_cb["win"], 2),
        "best_lo_t1": BC.lbl(best_lo["t1"]), "best_lo_t2": BC.lbl(best_lo["t2"]), "best_lo_netA": round(best_lo["lo_netA"], 0),
        "best_lo_delta_netA": round(best_lo["delta_netA"], 0), "best_lo_win": round(best_lo["lo_win"], 2),
        "best_cb_t1": BC.lbl(best_cb["t1"]), "best_cb_t2": BC.lbl(best_cb["t2"]), "best_cb_cover": BC.lbl(best_cb["cover"]),
        "best_cb_netA": round(best_cb["cb_netA"], 0), "best_cb_delta_netA": round(best_cb["delta_netA"], 0),
        "best_cb_win": round(best_cb["cb_win"], 2), "best_cb_n_trades": int(best_cb["n_trades"]),
        "long_only_netA_at_best_combined_t1t2": round(lo_at_best_cb_netA, 0),
        "short_cover_contribution": round(best_cb["cb_netA"] - lo_at_best_cb_netA, 0),
        "nbhd_n_combos": nb_stats["n_combos"], "nbhd_mean_netA": nb_stats["mean_netA"],
        "nbhd_min_netA": nb_stats["min_netA"], "nbhd_max_netA": nb_stats["max_netA"],
        "nbhd_pct_above_baseline": nb_stats["pct_above_baseline"],
    }
    return summary, best_cb


def main():
    total_combos = len(T_RANGE) * (len(T_RANGE) - 1) // 2 * len(COVER_RANGE)
    print(f"Grid: {len(T_RANGE)} t1/t2 minutes -> {len(T_RANGE)*(len(T_RANGE)-1)//2:,} valid (t1,t2) pairs "
          f"x {len(COVER_RANGE)} cover minutes = {total_combos:,} combinations", flush=True)

    print("\n=== FULL HISTORY ===", flush=True)
    cache_full = build_wide_cache()
    dates = sorted(c["entry_date"] for c in cache_full)
    d_min, d_max = dates[0], dates[-1]
    split_date = d_min + (d_max - d_min) * 0.70
    print(f"date span: {d_min} .. {d_max} | 70/30 split date: {split_date}", flush=True)

    rows_full = entry_positions(cache_full)
    M_full = build_matrices(rows_full)
    LO_full, CB_full = run_grid(M_full, "FULL_HISTORY")
    base_lo_full, base_cb_full = eval_baseline(M_full)
    summary_full, best_full = analyze_run("FULL_HISTORY", LO_full, CB_full, base_lo_full, base_cb_full, OUTDIR)

    print("\n=== OUT-OF-SAMPLE: FIT (first 70% by date) ===", flush=True)
    cache_fit = [c for c in cache_full if c["entry_date"] <= split_date]
    rows_fit = entry_positions(cache_fit)
    M_fit = build_matrices(rows_fit)
    LO_fit, CB_fit = run_grid(M_fit, "FIT_70PCT")
    base_lo_fit, base_cb_fit = eval_baseline(M_fit)
    summary_fit, best_fit = analyze_run("FIT_70PCT", LO_fit, CB_fit, base_lo_fit, base_cb_fit, OUTDIR)

    print("\n=== OUT-OF-SAMPLE: CONFIRM (final 30% by date) -- evaluating FIT's best combo out-of-sample ===", flush=True)
    cache_confirm = [c for c in cache_full if c["entry_date"] > split_date]
    rows_confirm = entry_positions(cache_confirm)
    M_confirm = build_matrices(rows_confirm)
    LO_confirm, CB_confirm = run_grid(M_confirm, "CONFIRM_30PCT")
    base_lo_confirm, base_cb_confirm = eval_baseline(M_confirm)
    summary_confirm, best_confirm = analyze_run("CONFIRM_30PCT", LO_confirm, CB_confirm, base_lo_confirm, base_cb_confirm, OUTDIR)

    # the actual OOS check: does FIT's best (t1,t2,cover) still beat baseline on CONFIRM's own data?
    fit_combo_on_confirm = CB_confirm[(CB_confirm["t1"] == best_fit["t1"]) & (CB_confirm["t2"] == best_fit["t2"]) & (CB_confirm["cover"] == best_fit["cover"])]
    oos_check = fit_combo_on_confirm.iloc[0].to_dict() if len(fit_combo_on_confirm) else None
    if oos_check:
        print(f"\n=== OOS VALIDATION: FIT's best combo (t1={BC.lbl(best_fit['t1'])}, t2={BC.lbl(best_fit['t2'])}, cover={BC.lbl(best_fit['cover'])}) evaluated on CONFIRM data ===")
        print(f"  CONFIRM netA at FIT's chosen combo = {oos_check['cb_netA']:.0f}  vs  CONFIRM's own baseline netA = {base_cb_confirm['netA']:.0f}  (delta {oos_check['cb_netA']-base_cb_confirm['netA']:+.0f})")
        print(f"  CONFIRM's own independently-found best combo netA = {best_confirm['cb_netA']:.0f} (t1={BC.lbl(best_confirm['t1'])}, t2={BC.lbl(best_confirm['t2'])}, cover={BC.lbl(best_confirm['cover'])})")
    else:
        print("\n=== OOS VALIDATION: FIT's best combo has 0 trades in the CONFIRM period (cannot evaluate) ===")

    with pd.ExcelWriter(OUTDIR / "t1t2_cover_full_grid_sweep_SUMMARY.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": f"3-D sweep (t1 x t2 x short cover time) on the locked baseline_and_cross_final.py strategy. "
                      f"Category A/B/C entry logic, 17% target overlay, and the short's entry-tied-to-long-exit "
                      f"rule are UNCHANGED. Grid: 165 t1/t2 minutes (9:16am-12:00pm) -> 13,530 valid pairs x 61 "
                      f"cover minutes (2:00pm-3:00pm) = {total_combos:,} combinations."},
            {"note": f"Full backtest history date span: {d_min} .. {d_max}. 70/30-by-calendar-date OOS split at "
                      f"{split_date} (not a 70/30 split of trade COUNT) -- FIT = on/before split date, CONFIRM = after."},
            {"note": "LONG-ONLY grid (13,530 rows, no cover dependency) and LONG+SHORT combined grid (825,330 rows) "
                      "both fully materialized and saved per run (FULL_HISTORY / FIT_70PCT / CONFIRM_30PCT) as "
                      "separate parquet files -- see *_long_only_grid.parquet / *_combined_grid.parquet."},
            {"note": "Best combo = argmax net_A. Stable region = exact +/-2-minute neighborhood in all 3 dimensions "
                      "around the best cell, read directly off the materialized grid (no smoothing approximation "
                      "needed this time, since the full grid is small enough to query directly)."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        pd.DataFrame([summary_full, summary_fit, summary_confirm]).to_excel(w, sheet_name="Run_Summaries", index=False)
        if oos_check:
            pd.DataFrame([{
                "fit_best_t1": BC.lbl(best_fit["t1"]), "fit_best_t2": BC.lbl(best_fit["t2"]), "fit_best_cover": BC.lbl(best_fit["cover"]),
                "confirm_netA_at_fit_combo": round(oos_check["cb_netA"], 0), "confirm_baseline_netA": round(base_cb_confirm["netA"], 0),
                "delta_vs_confirm_baseline": round(oos_check["cb_netA"] - base_cb_confirm["netA"], 0),
                "confirm_own_best_netA": round(best_confirm["cb_netA"], 0),
                "confirm_own_best_t1": BC.lbl(best_confirm["t1"]), "confirm_own_best_t2": BC.lbl(best_confirm["t2"]), "confirm_own_best_cover": BC.lbl(best_confirm["cover"]),
            }]).to_excel(w, sheet_name="OOS_Validation", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)

    print(f"\nSaved -> {OUTDIR}/t1t2_cover_full_grid_sweep_SUMMARY.xlsx (+ 6 grid parquet files)")


if __name__ == "__main__":
    main()
