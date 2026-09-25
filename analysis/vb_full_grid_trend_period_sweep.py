# -*- coding: utf-8 -*-
"""vb_full_grid_trend_period_sweep.py — Comprehensive 4-D sweep (Category C entry time x long-exit t1 x
long-exit t2 x short cover time) on the main NSE Volume-Breakout BTST strategy, segmented by specific
trend periods (6 uptrend + combined-uptrend, 4 downtrend + combined-downtrend = 12 runs). Category A/B
logic, the 17% target overlay, and the short's entry-tied-to-long-exit rule are all UNCHANGED, byte-
identical to the locked baseline_and_cross_final.py -- only Category C's fill time, t1, t2, and the
short's cover time are swept, independently per period/group.

SCALE: per period, Category C (29) x valid (t1,t2) pairs (13,530) x cover time (91, narrowed from the
originally-proposed 12:00pm-3:00pm to 1:30pm-3:00pm per user confirmation) = ~35.7M combinations; x12
runs. This is NOT materialized as a full grid -- for each (CatC, t1, t2) triple the cover dimension is
fully vectorized (all 91 cover values via one broadcasted numpy op reusing a single "first short-target-
hit-time" computation per row, since that value doesn't depend on the cover ceiling), and only the single
running best-by-net_A combo is tracked during the bulk sweep. After the bulk sweep finds the best cell, a
small targeted neighborhood re-evaluation (+/-2 minutes in each of the 4 swept dimensions) is run via an
exact evaluator to check for a stable plateau vs an overfit spike, and the exact locked baseline
(15:21 / 9:25 / 11:59 / 14:39) is evaluated the same way for the required comparison.
"""
import sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC

OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "full_grid_trend_period_sweep"
OUTDIR.mkdir(parents=True, exist_ok=True)

CATC_RANGE = list(range(900, 929))          # 3:00pm-3:28pm, 29 values
T_RANGE = list(range(556, 721))             # 9:16am-12:00pm, 165 values (t1/t2)
COVER_RANGE = list(range(810, 901))         # 1:30pm-3:00pm, 91 values (NARROWED per user confirmation)
DATA_LO, DATA_HI = 555, 900                 # unified next-day window for hi/lo/open lookups
BASE_CATC, BASE_T1, BASE_T2, BASE_COVER = BC.HM_1521, BC.T1, BC.T2, BC.COVER_HM
NBHD = 2                                    # +/- minutes for the stable-region neighborhood check

UPTREND_PERIODS = {
    "UP1_2022-06-21_2022-09-16": [("2022-06-21", "2022-09-16")],
    "UP2_2023-03-28_2024-02-08": [("2023-03-28", "2024-02-08")],
    "UP3_2024-03-20_2024-07-15": [("2024-03-20", "2024-07-15")],
    "UP4_2025-04-08_2025-06-12": [("2025-04-08", "2025-06-12")],
    "UP5_2026-04-02_2026-05-08": [("2026-04-02", "2026-05-08")],
    "UP6_2026-06-10_2026-07-31": [("2026-06-10", "2026-07-31")],
}
DOWNTREND_PERIODS = {
    "DN1_2022-04-06_2022-06-20": [("2022-04-06", "2022-06-20")],
    "DN2_2022-12-20_2023-06-30": [("2022-12-20", "2023-06-30")],
    "DN3_2024-12-12_2025-03-04": [("2024-12-12", "2025-03-04")],
    "DN4_2025-11-03_2026-03-27": [("2025-11-03", "2026-03-27")],
}
ALL_PERIODS = dict(UPTREND_PERIODS)
ALL_PERIODS["COMBINED_UPTREND"] = [r for v in UPTREND_PERIODS.values() for r in v]
ALL_PERIODS.update(DOWNTREND_PERIODS)
ALL_PERIODS["COMBINED_DOWNTREND"] = [r for v in DOWNTREND_PERIODS.values() for r in v]


def in_any_range(d, ranges):
    for lo, hi in ranges:
        if lo <= d <= hi:
            return True
    return False


def build_master_cache():
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                        usecols=["symbol", "date", "passes_all_three", "prev_day_vwap_close"],
                        parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True]
    print(f"qualifying (all history): {len(Q):,}", flush=True)
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
        for _, r in Q[Q["symbol"] == sym].iterrows():
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
            wmask = (nhm >= DATA_LO) & (nhm <= DATA_HI)
            cache.append({"symbol": sym, "entry_date": ed, "pc": pc, "eg": eg,
                          "nhm_w": nhm[wmask], "nhi_w": nhi[wmask], "nlo_w": nlo[wmask], "nop_w": nop[wmask]})
        if si % 300 == 0:
            print(f"  ...{si} symbols ({time.time()-t0:.0f}s), {len(cache):,} stock-days so far", flush=True)
    print(f"master cache built: {len(cache):,} stock-days total ({time.time()-t0:.0f}s)", flush=True)
    return cache


def classify_catC(pc, hm, hi, lo, op, catc_time):
    """Category A/B UNCHANGED. Category C's own fill uses the swept catc_time's open (px_C) instead of
    the fixed 15:21 open (px) -- px/locked_321 (Cat A leg2 fallback + eligibility gate) stay fixed."""
    uc, l19, l17 = pc * BC.UC_M, pc * BC.L19_M, pc * BC.L17_M
    o = dict(zip(hm, op)); px = o.get(BC.HM_1521, np.nan)
    px_C = o.get(catc_time, np.nan)
    lo_of = dict(zip(hm, lo)); lo_1521 = lo_of.get(BC.HM_1521, np.nan)
    locked_321 = lo_1521 == lo_1521 and lo_1521 >= uc
    uc_in_230_300 = BC.fhm(hm, (hm >= BC.A_START) & (hm < BC.HM_1500) & (hi >= uc))
    first_uc = BC.fhm(hm, hi >= uc)

    def catB_or_C():
        inB = (hm >= BC.HM_1500) & (hm < BC.HM_1521)
        touch_B = BC.fhm(hm, inB & (hi >= uc))
        opened_in_B = bool((lo[inB] < uc).any()) if inB.any() else False
        locked_before_3 = first_uc is not None and first_uc < BC.HM_1500
        if locked_321 and locked_before_3 and not opened_in_B:
            return "C", False, []
        if touch_B is not None:
            return "B", True, [(1.0, uc * 0.999, touch_B, 1)]
        if px_C == px_C and px_C > 0 and not locked_321:
            return "C", True, [(1.0, px_C, catc_time, 3)]
        return "C", False, []

    if uc_in_230_300 is not None:
        w = (hm > uc_in_230_300) & (hm < BC.HM_1521)
        wl, wh = lo[w], hm[w]
        i19 = np.where(wl <= l19)[0]
        if len(i19):
            legs = [(0.5, l19, int(wh[i19[0]]), 1)]
            i17 = np.where(wl <= l17)[0]
            if len(i17):
                legs.append((0.5, l17, int(wh[i17[0]]), 1))
            elif px == px and not locked_321:
                legs.append((0.5, px, BC.HM_1521, 2))
            return "A", True, legs
        return catB_or_C()
    return catB_or_C()


def entry_positions_for_catc(cache, catc_time):
    """UNCHANGED Category A/B/C classification (with the swept Cat C price) + capital sequencing/sizing."""
    recs = []
    for c in cache:
        cat, entered, legs = classify_catC(c["pc"], c["eg"]["hm"], c["eg"]["hi"], c["eg"]["lo"], c["eg"]["op"], catc_time)
        recs.append({"entry_date": c["entry_date"], "entered": entered, "legs": legs,
                     "nhm_w": c["nhm_w"], "nhi_w": c["nhi_w"], "nlo_w": c["nlo_w"], "nop_w": c["nop_w"]})
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
                     "nhm_w": r["nhm_w"], "nhi_w": r["nhi_w"], "nlo_w": r["nlo_w"], "nop_w": r["nop_w"]})
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
    return dict(avg=avg, shares=shares, cap=cap, tgt=tgt, th=th, cols=cols, OPEN_M=OPEN_M, LOW_M=LOW_M)


def eval_small_grid(cache, catc_values, t1_values, t2_values, cover_values):
    """Exact (non-bulk-optimized) evaluator for a SMALL set of (catC,t1,t2,cover) combos -- used for the
    baseline point and the post-hoc neighborhood/stable-region check. Reuses the same vectorized-cover-
    sweep trick, just over a tiny grid instead of the full sweep."""
    cover_arr = np.array(sorted(set(cover_values)))
    out = []
    for catc_time in catc_values:
        rows = entry_positions_for_catc(cache, catc_time)
        if not rows:
            continue
        M = build_matrices(rows)
        avg, shares, cap, tgt, th, cols, OPEN_M, LOW_M = M["avg"], M["shares"], M["cap"], M["tgt"], M["th"], M["cols"], M["OPEN_M"], M["LOW_M"]
        cover_col_idx = cover_arr - DATA_LO
        for t1 in t1_values:
            ot1 = OPEN_M[:, t1 - DATA_LO]
            m1 = th <= t1
            m2 = (~m1) & (ot1 == ot1) & (ot1 > avg)
            base_valid = m1 | m2
            xp12 = np.where(m1, tgt, np.where(m2, ot1, np.nan))
            xhm12 = np.where(m1, th, np.where(m2, t1, np.nan))
            for t2 in t2_values:
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
                    out.append({"catc": catc_time, "t1": t1, "t2": t2, "cover": int(cover_t),
                                "gross": round(float(gross_arr[k]), 0), "netA": round(float(netA_arr[k]), 0),
                                "netB": round(float(netB_arr[k]), 0), "win": round(float(win_arr[k]), 2), "n": n_trades})
    return pd.DataFrame(out)


def run_period_bulk(name, ranges, master_cache):
    date_ranges = [(pd.Timestamp(lo).date(), pd.Timestamp(hi).date()) for lo, hi in ranges]
    cache = [c for c in master_cache if in_any_range(c["entry_date"], date_ranges)]
    print(f"\n{'='*78}\n{name}: {len(cache):,} stock-days in period", flush=True)
    if len(cache) < 5:
        print(f"  SKIPPED -- too few stock-days ({len(cache)}) for a meaningful sweep", flush=True)
        return None

    best = {"netA": -np.inf}
    best_gross = -np.inf; best_netB = -np.inf
    cover_arr = np.array(COVER_RANGE)
    t0 = time.time()
    for ci, catc_time in enumerate(CATC_RANGE):
        rows = entry_positions_for_catc(cache, catc_time)
        if not rows:
            continue
        M = build_matrices(rows)
        avg, shares, cap, tgt, th, cols, OPEN_M, LOW_M = M["avg"], M["shares"], M["cap"], M["tgt"], M["th"], M["cols"], M["OPEN_M"], M["LOW_M"]
        cover_col_idx = cover_arr - DATA_LO

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

                bi = int(np.argmax(netA_arr))
                if netA_arr[bi] > best["netA"]:
                    win_bi = float((comb[valid, bi] > 0).mean() * 100)
                    best = {"netA": float(netA_arr[bi]), "catc": catc_time, "t1": t1, "t2": t2,
                            "cover": int(cover_arr[bi]), "gross": float(gross_arr[bi]),
                            "netB": float(netB_arr[bi]), "win": win_bi, "n": n_trades}
                gmax = float(gross_arr.max()); bmax = float(netB_arr.max())
                if gmax > best_gross: best_gross = gmax
                if bmax > best_netB: best_netB = bmax
        if (ci + 1) % 10 == 0 or ci == len(CATC_RANGE) - 1:
            print(f"  catC {BC.lbl(catc_time)} done ({ci+1}/{len(CATC_RANGE)}, {time.time()-t0:.0f}s elapsed)", flush=True)

    elapsed = time.time() - t0
    print(f"  bulk sweep done in {elapsed:.0f}s. best netA = {best['netA']:.0f} @ catC={BC.lbl(best['catc'])} "
          f"t1={BC.lbl(best['t1'])} t2={BC.lbl(best['t2'])} cover={BC.lbl(best['cover'])} (n={best['n']}, win={best['win']:.1f}%)", flush=True)

    # ---- baseline (exact) ----
    base_df = eval_small_grid(cache, [BASE_CATC], [BASE_T1], [BASE_T2], [BASE_COVER])
    base = base_df.iloc[0].to_dict() if len(base_df) else {"gross": np.nan, "netA": np.nan, "netB": np.nan, "win": np.nan, "n": 0}

    # ---- neighborhood (+/-2 min in each dim) around the best cell, for stable-region check ----
    catc_nbhd = [c for c in range(best["catc"] - NBHD, best["catc"] + NBHD + 1) if c in CATC_RANGE]
    t1_nbhd = [t for t in range(best["t1"] - NBHD, best["t1"] + NBHD + 1) if t in T_RANGE]
    t2_nbhd = [t for t in range(best["t2"] - NBHD, best["t2"] + NBHD + 1) if t in T_RANGE]
    cover_nbhd = [c for c in range(best["cover"] - NBHD, best["cover"] + NBHD + 1) if c in COVER_RANGE]
    NB = eval_small_grid(cache, catc_nbhd, t1_nbhd, t2_nbhd, cover_nbhd)
    nb_stats = {"n_combos": len(NB), "mean_netA": round(NB["netA"].mean(), 0) if len(NB) else np.nan,
                "min_netA": round(NB["netA"].min(), 0) if len(NB) else np.nan,
                "max_netA": round(NB["netA"].max(), 0) if len(NB) else np.nan,
                "pct_above_baseline": round((NB["netA"] > base["netA"]).mean() * 100, 1) if len(NB) else np.nan}

    result = {
        "period": name, "n_stock_days": len(cache), "n_trades_baseline": int(base["n"]),
        "best_catC": BC.lbl(best["catc"]), "best_t1": BC.lbl(best["t1"]), "best_t2": BC.lbl(best["t2"]), "best_cover": BC.lbl(best["cover"]),
        "best_gross": round(best["gross"], 0), "best_netA": round(best["netA"], 0), "best_netB": round(best["netB"], 0),
        "best_win_pct": round(best["win"], 2), "best_n_trades": best["n"],
        "baseline_gross": round(base["gross"], 0) if base["gross"] == base["gross"] else np.nan,
        "baseline_netA": round(base["netA"], 0) if base["netA"] == base["netA"] else np.nan,
        "baseline_netB": round(base["netB"], 0) if base["netB"] == base["netB"] else np.nan,
        "baseline_win_pct": round(base["win"], 2) if base["win"] == base["win"] else np.nan,
        "delta_gross": round(best["gross"] - base["gross"], 0) if base["gross"] == base["gross"] else np.nan,
        "delta_netA": round(best["netA"] - base["netA"], 0) if base["netA"] == base["netA"] else np.nan,
        "delta_netB": round(best["netB"] - base["netB"], 0) if base["netB"] == base["netB"] else np.nan,
        "nbhd_n_combos": nb_stats["n_combos"], "nbhd_mean_netA": nb_stats["mean_netA"],
        "nbhd_min_netA": nb_stats["min_netA"], "nbhd_max_netA": nb_stats["max_netA"],
        "nbhd_pct_above_baseline": nb_stats["pct_above_baseline"],
        "sweep_seconds": round(elapsed, 1), "best_gross_any_combo": round(best_gross, 0), "best_netB_any_combo": round(best_netB, 0),
    }
    pd.DataFrame([result]).to_csv(OUTDIR / f"{name}_summary.csv", index=False)
    NB.to_csv(OUTDIR / f"{name}_neighborhood.csv", index=False)
    print(f"  -> saved {name}_summary.csv / {name}_neighborhood.csv", flush=True)
    return result


def main():
    t_start = time.time()
    print("Building master wide cache (full history, all periods)...", flush=True)
    master_cache = build_master_cache()

    all_results = []
    for name, ranges in ALL_PERIODS.items():
        res = run_period_bulk(name, ranges, master_cache)
        if res:
            all_results.append(res)
        # checkpoint the cross-period summary after every period so partial progress is never lost
        if all_results:
            pd.DataFrame(all_results).to_csv(OUTDIR / "CROSS_PERIOD_SUMMARY.csv", index=False)

    SUMMARY = pd.DataFrame(all_results)
    with pd.ExcelWriter(OUTDIR / "full_grid_trend_period_sweep_SUMMARY.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "4-D sweep (Category C entry time, long-exit t1, long-exit t2, short cover time) on the "
                      "locked baseline_and_cross_final.py strategy, segmented by trend period. Category A/B "
                      "logic, the 17% target overlay, and the short's entry-tied-to-long-exit rule are UNCHANGED."},
            {"note": f"Grid per period: CatC 29 x (t1,t2) 13,530 x cover {len(COVER_RANGE)} "
                      f"(narrowed to 1:30pm-3:00pm per explicit confirmation) = ~{29*13530*len(COVER_RANGE):,} combos."},
            {"note": "Best combo = argmax net_A during the bulk sweep (gross/net_B reported AT that same cell, "
                      "not independently re-optimized -- consistent with the prior t1/t2-only sweep's convention)."},
            {"note": f"Stable-region check = exact re-evaluation of a +/-{NBHD}-minute neighborhood in all 4 "
                      "dimensions around the best cell (up to 5^4 combos) -- see the per-period *_neighborhood.csv "
                      "files and the nbhd_* summary columns (mean/min/max net_A, % of neighbors beating baseline)."},
            {"note": "SAMPLE SIZE: several periods are only 2-4 months. Treat single-period results as narrow-"
                      "window observations; the combined-uptrend/combined-downtrend runs pool more data and are "
                      "more reliable indicators of a genuine regime-dependent pattern (if one exists)."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        SUMMARY.to_excel(w, sheet_name="Cross_Period_Summary", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)

    print(f"\nALL {len(all_results)} PERIODS COMPLETE in {(time.time()-t_start)/60:.1f} min -> {OUTDIR}/full_grid_trend_period_sweep_SUMMARY.xlsx", flush=True)


if __name__ == "__main__":
    main()
