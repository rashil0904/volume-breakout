# -*- coding: utf-8 -*-
"""vb_catBC_boundary_sweep.py — Category B/C BOUNDARY sweep on the main NSE Volume-Breakout BTST strategy's locked
baseline. Instead of moving Category C's fill time independently, this LINKS Category B's window end and Category
C's fill time to a single swept boundary B_HM (3:13pm-3:21pm, 1-min steps, 9 values):
  Category B window becomes 3:00pm -> B_HM (same "about to lock UC" entry logic, at uc*0.999, just a shorter/longer
      window when B_HM != 3:21).
  Category C fills at B_HM instead of the fixed 3:21 open (same mechanic: single fill for the day's remainder).
  Category A is BYTE-IDENTICAL to the locked classify() -- 2:30-3pm pullback window, its own "3:21" fallback
      references stay FIXED at 15:21 regardless of B_HM (see ASSUMPTION note below). Long exit t1/t2, short cover
      mechanics (14:39 / 5% target), and capital-allocation SEQUENCING (phase1 intraday -> phase2 Cat-A-3:21-pending
      -> phase3 Cat-C-remainder) are all UNCHANGED, per instruction -- only the B/C boundary itself moves.

FLAGGED ASSUMPTION (Category A fallback timing, as the brief asked to confirm): Category A's own 2nd-leg fallback
(fills at the 15:21 open if the stock never re-locks UC) is disjoint from B/C -- a stock that is Category A never
reaches catB_or_C(), so there's no direct double-classification conflict. There IS a second-order inconsistency:
when B_HM < 15:21, Category C's capital claim happens chronologically BEFORE Category A's still-pending 15:21 leg2
claim in the real market, but the LOCKED capital-sequencing code still gives Category A's phase-2 pool priority over
Category C's phase-3 remainder regardless of clock time (it was an arbitrary tie-break when both were simultaneous
at 15:21; it's no longer simultaneous once B_HM shifts earlier). Per the "do not change capital allocation"
instruction this ordering is left AS-IS -- flagged, not fixed.

At B_HM = 15:21 (921) this reduces to the exact byte-identical locked baseline (checked programmatically below).
"""
import sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC

OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "catBC_boundary_sweep"; OUTDIR.mkdir(parents=True, exist_ok=True)
BOUNDARIES = list(range(913, 922))     # 15:13 .. 15:21, 1-min steps (9 values)
A_START, HM_1500, HM_1521 = BC.A_START, BC.HM_1500, BC.HM_1521
UC_M, L19_M, L17_M = BC.UC_M, BC.L19_M, BC.L17_M
fhm = BC.fhm


def classify_boundary(pc, hm, hi, lo, op, B_HM):
    """classify(), Category A section BYTE-IDENTICAL to locked code; catB_or_C parameterised on B_HM."""
    uc, l19, l17 = pc * UC_M, pc * L19_M, pc * L17_M
    o = dict(zip(hm, op)); px = o.get(HM_1521, np.nan)      # Category A's OWN fixed 15:21 fallback price
    lo_of = dict(zip(hm, lo)); lo_1521 = lo_of.get(HM_1521, np.nan)
    locked_321 = lo_1521 == lo_1521 and lo_1521 >= uc        # Category A's OWN fixed lock check (unchanged)
    uc_in_230_300 = fhm(hm, (hm >= A_START) & (hm < HM_1500) & (hi >= uc))
    first_uc = fhm(hm, hi >= uc)
    meta = {"hit_uc_ever": first_uc is not None}

    def catB_or_C():
        px_b = o.get(B_HM, np.nan)
        lo_b = lo_of.get(B_HM, np.nan)
        locked_b = lo_b == lo_b and lo_b >= uc
        inB = (hm >= HM_1500) & (hm < B_HM)
        touch_B = fhm(hm, inB & (hi >= uc))
        opened_in_B = bool((lo[inB] < uc).any()) if inB.any() else False
        locked_before_3 = first_uc is not None and first_uc < HM_1500
        if locked_b and locked_before_3 and not opened_in_B:
            return "C", False, [], {**meta, "no_entry": "locked_from_before3_never_opened"}
        if touch_B is not None:
            return "B", True, [(1.0, uc * 0.999, touch_B, 1)], {**meta, "entry_kind": "B_uc0.999", "B_HM": B_HM}
        if px_b == px_b and px_b > 0 and not locked_b:
            return "C", True, [(1.0, px_b, B_HM, 3)], {**meta, "entry_kind": "C_boundary", "B_HM": B_HM}
        return "C", False, [], {**meta, "no_entry": "locked_or_no_boundary_candle"}

    if uc_in_230_300 is not None:
        w = (hm > uc_in_230_300) & (hm < HM_1521)
        wl, wh = lo[w], hm[w]
        i19 = np.where(wl <= l19)[0]
        if len(i19):
            legs = [(0.5, l19, int(wh[i19[0]]), 1)]
            i17 = np.where(wl <= l17)[0]
            if len(i17):
                legs.append((0.5, l17, int(wh[i17[0]]), 1)); full = True; k = "17%"
            elif px == px and not locked_321:
                legs.append((0.5, px, HM_1521, 2)); full = True; k = "3:21"
            else:
                full = False; k = "half"
            return "A", True, legs, {**meta, "cat_a_full": full, "leg2": k, "uc_after_first_fill": True}
        return catB_or_C()
    return catB_or_C()


def run_config_boundary(cache, B_HM):
    """run_config(), byte-identical exit/sequencing logic; only the classify() call differs (boundary)."""
    recs = []
    for c in cache:
        cat, entered, legs, meta = classify_boundary(c["pc"], c["eg"]["hm"], c["eg"]["hi"], c["eg"]["lo"], c["eg"]["op"], B_HM)
        recs.append({**{k: c[k] for k in ("symbol", "entry_date", "exit_date", "o565", "o719", "o879", "nhm", "nhi", "lhm", "nlo")},
                     "category": cat, "entered": entered, "legs": legs, "meta": meta})
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
        nC = len(cc); per_c = min(BC.BASE_ALLOC, max(0.0, pool) / nC) if nC else 0.0
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
        tgt = avg * BC.LONG_TGT
        hit = r["nhi"] >= tgt
        th = int(r["nhm"][hit].min()) if hit.any() else 10 ** 9
        o565, o719 = r["o565"], r["o719"]
        if th <= BC.T1: xp, xt, xhm = tgt, "target_pre_0925", th
        elif o565 == o565 and o565 > avg: xp, xt, xhm = o565, "positive_0925", BC.T1
        elif th <= BC.T2: xp, xt, xhm = tgt, "target_0925_1159", th
        elif o719 == o719: xp, xt, xhm = o719, "exit_1159", BC.T2
        else: continue
        long_pnl = shares * (xp - avg)
        stgt = xp * BC.SHORT_TGT
        sw = (r["lhm"] > xhm) & (r["nlo"] <= stgt)
        if sw.any(): cover = stgt
        elif r["o879"] == r["o879"]: cover = r["o879"]
        else: cover = np.nan
        has_short = cover == cover
        short_pnl = shares * (xp - cover) if has_short else 0.0
        snotl = shares * xp if has_short else 0.0
        comb = long_pnl + short_pnl
        rows.append({"symbol": r["symbol"], "entry_date": r["entry_date"], "exit_date": r["exit_date"], "category": r["category"],
                     "shares": int(shares), "capital_deployed": cap, "avg_entry": avg, "long_pnl": long_pnl, "short_pnl": short_pnl,
                     "gross_pnl": comb, "netA_pnl": comb - BC.R023 * cap - BC.SR * snotl, "netB_pnl": comb - BC.R038 * cap - BC.SR * snotl,
                     "gross_long_only": long_pnl, "netA_long_only": long_pnl - BC.R023 * cap, "netB_long_only": long_pnl - BC.R038 * cap})
    return pd.DataFrame(rows)


def summarize(T, label):
    if T.empty:
        return {"boundary": label, "n_trades": 0}
    n_b = int((T["category"] == "B").sum()); n_c = int((T["category"] == "C").sum())
    o = {"boundary": label, "n_trades": len(T), "n_catA": int((T["category"] == "A").sum()), "n_catB": n_b, "n_catC": n_c}
    for s in ["gross", "netA", "netB"]:
        p = T[f"{s}_pnl"]
        o[f"{s}_total_pnl_inr"] = round(p.sum())
        o[f"{s}_return_pct_of_5L"] = round(p.sum() / BC.BASE_POOL * 100, 2)
        o[f"{s}_win_rate_pct"] = round((p > 0).mean() * 100, 2)
        pl = T[f"{s}_long_only"]
        o[f"{s}_LONGONLY_total_pnl_inr"] = round(pl.sum())
        o[f"{s}_LONGONLY_return_pct_of_5L"] = round(pl.sum() / BC.BASE_POOL * 100, 2)
        o[f"{s}_LONGONLY_win_rate_pct"] = round((pl > 0).mean() * 100, 2)
    return o


def hm2s(m):
    return f"{int(m)//60:02d}:{int(m)%60:02d}"


def main():
    cache = BC.build_cache()
    print(f"cache: {len(cache):,} stock-days", flush=True)

    results = {}
    for B in BOUNDARIES:
        T = run_config_boundary(cache, B)
        results[B] = T
        print(f"  boundary {hm2s(B)}: {len(T)} trades (A={int((T['category']=='A').sum())} B={int((T['category']=='B').sum())} C={int((T['category']=='C').sum())}) "
              f"netA={T['netA_pnl'].sum():,.0f}", flush=True)

    # sanity check: boundary=921 must reproduce the exact locked baseline
    T_base_locked, _ = BC.run_config("baseline", cache)
    T_921 = results[921]
    key = ["symbol", "entry_date"]
    m = T_base_locked.set_index(key)[["category", "gross_pnl", "netA_pnl", "netB_pnl", "shares"]].sort_index()
    m2 = T_921.set_index(key)[["category", "gross_pnl", "netA_pnl", "netB_pnl", "shares"]].sort_index()
    same_n = len(m) == len(m2)
    same_idx = same_n and m.index.equals(m2.index)
    max_diff = float((m["gross_pnl"] - m2["gross_pnl"]).abs().max()) if same_idx else float("nan")
    print(f"\n=== SANITY CHECK: boundary=15:21 vs locked baseline === n_locked={len(m)} n_921={len(m2)} "
          f"same_trade_set={same_idx} max|gross diff|={max_diff}", flush=True)
    assert same_idx and max_diff < 1e-6, "boundary=15:21 does NOT reproduce the locked baseline -- STOP, logic bug"
    print("PASSED: boundary=15:21 exactly reproduces the locked baseline.", flush=True)

    S = pd.DataFrame([summarize(results[B], hm2s(B)) for B in BOUNDARIES])
    S.insert(1, "boundary_hm", BOUNDARIES)
    base_row = S[S["boundary_hm"] == 921].iloc[0]
    for s in ["gross", "netA", "netB"]:
        S[f"delta_{s}_pnl_vs_baseline"] = (S[f"{s}_total_pnl_inr"] - base_row[f"{s}_total_pnl_inr"]).round(0)
        S[f"delta_{s}_LONGONLY_pnl_vs_baseline"] = (S[f"{s}_LONGONLY_total_pnl_inr"] - base_row[f"{s}_LONGONLY_total_pnl_inr"]).round(0)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print("\n=== FULL SWEEP TABLE ===\n" + S.to_string(index=False), flush=True)

    # stable region: best 3-consecutive-boundary window by netA_return_pct_of_5L
    vals = S.set_index("boundary_hm")["netA_return_pct_of_5L"]
    best_win, best_mean = None, -1e18
    for i in range(len(BOUNDARIES) - 2):
        w = BOUNDARIES[i:i + 3]
        m3 = vals.loc[w].mean()
        if m3 > best_mean:
            best_mean, best_win = m3, w
    print(f"\n=== BEST STABLE 3-CONSECUTIVE-MINUTE REGION (by netA return %) ===\n"
          f"  boundaries {[hm2s(b) for b in best_win]} -> mean netA {best_mean:.2f}% "
          f"(individual: {[round(vals.loc[b],2) for b in best_win]})", flush=True)
    single_best = vals.idxmax()
    print(f"  single best boundary (reference only, overfit risk): {hm2s(single_best)} -> netA {vals.loc[single_best]:.2f}%", flush=True)

    best_B = best_win[1]           # centre of the stable window
    Tb = results[best_B]
    print(f"\n=== LONG-ONLY vs LONG+SHORT at best boundary {hm2s(best_B)} ===")
    for s in ["gross", "netA", "netB"]:
        print(f"  {s}: LONG+SHORT total={Tb[f'{s}_pnl'].sum():,.0f} ({Tb[f'{s}_pnl'].sum()/BC.BASE_POOL*100:.2f}%) win={((Tb[f'{s}_pnl']>0).mean()*100):.1f}%  |  "
              f"LONG-ONLY total={Tb[f'{s}_long_only'].sum():,.0f} ({Tb[f'{s}_long_only'].sum()/BC.BASE_POOL*100:.2f}%) win={((Tb[f'{s}_long_only']>0).mean()*100):.1f}%", flush=True)

    fn = OUTDIR / "vb_catBC_boundary_sweep.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": "Category B window becomes 3:00pm->boundary; Category C fills at boundary (was fixed 3:21pm). Category A "
                                "unchanged, its own 3:21 fallback stays fixed. ASSUMPTION FLAGGED: capital-sequencing phase order (Cat-A-pending "
                                "before Cat-C-remainder) is left as-is though it is no longer strictly chronological once boundary<15:21 -- see "
                                "module docstring. Sanity check: boundary=15:21 reproduces the locked baseline exactly (0.0000 diff, same trade set)."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        S.to_excel(w, sheet_name="Sweep_summary", index=False)
        for B in BOUNDARIES:
            results[B].to_excel(w, sheet_name=f"trades_{hm2s(B).replace(':','')}", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
