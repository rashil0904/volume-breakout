# -*- coding: utf-8 -*-
"""vb_catC_entry_time_sweep_apr_jul2026.py — Category C entry-TIME sweep (3:00pm-3:28pm, 1-min steps) for
the main NSE Volume-Breakout BTST strategy's locked baseline, RESTRICTED to April-July 2026 only. Category
A and B logic, exit rules, capital allocation, and every other part of the strategy are byte-identical
copies of the locked baseline_and_cross_final.py -- NOT modified, NOT touched. Diagnostic variant only;
new script, new output folder.

KEY DESIGN DECISION (per explicit instruction, confirmed no logical conflict for any swept time incl.
before 3:21): Category A's leg2 3:21pm fallback price/time and the "locked_321" circuit-breaker validity
check (both keyed off the REAL 15:21 candle) stay COMPLETELY FIXED regardless of where Category C's own
sweep time is set -- only Category C's OWN fill price is read from the swept time's candle instead of the
15:21 candle. Category C's eligibility condition becomes "swept-time candle has a valid positive open, AND
the stock is not circuit-locked per the unchanged 15:21-based check" -- the same logical structure as the
original ("px==px and px>0 and not locked_321"), just with px replaced by the swept-time price for
Category C's OWN entry only. No conflict arises for swept times before 15:21 because locked_321 is a
backtest-only historical eligibility gate (evaluated with full-day hindsight), not something Category C's
own price needs to "already know" live.
"""
import sys, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC
import uc_staggered_dd_report as R

OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "catC_entry_time_sweep_apr_jul2026"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIN_START = pd.Timestamp("2026-04-01").date(); WIN_END = pd.Timestamp("2026-07-31").date()
BASELINE_HM = 921  # 3:21pm, current fixed Category C fill time
SWEEP_RANGE = list(range(900, 929))  # 3:00pm .. 3:28pm inclusive, 29 values


def classify_swept(config, pc, hm, hi, lo, op, sweep_hm):
    """Byte-identical to BC.classify() for config=='baseline' EXCEPT: Category C's own fill uses
    sweep_hm's open instead of the fixed 15:21 open. Category A's leg2 3:21 fallback and the
    locked_321 circuit check are UNTOUCHED (still keyed to BC.HM_1521 exactly as the original)."""
    uc, l19, l17, l195 = pc * BC.UC_M, pc * BC.L19_M, pc * BC.L17_M, pc * BC.L195_M
    o = dict(zip(hm, op)); px = o.get(BC.HM_1521, np.nan)          # UNCHANGED: for Cat A leg2 fallback only
    px_C = o.get(sweep_hm, np.nan)                                  # NEW: Category C's own swept-time price
    lo_of = dict(zip(hm, lo)); lo_1521 = lo_of.get(BC.HM_1521, np.nan)
    locked_321 = lo_1521 == lo_1521 and lo_1521 >= uc               # UNCHANGED circuit check (real 15:21 candle)
    uc_in_230_300 = BC.fhm(hm, (hm >= BC.A_START) & (hm < BC.HM_1500) & (hi >= uc))
    first_uc = BC.fhm(hm, hi >= uc)
    meta = {"hit_uc_ever": first_uc is not None}

    def catB_or_C():
        inB = (hm >= BC.HM_1500) & (hm < BC.HM_1521)
        touch_B = BC.fhm(hm, inB & (hi >= uc))
        opened_in_B = bool((lo[inB] < uc).any()) if inB.any() else False
        locked_before_3 = first_uc is not None and first_uc < BC.HM_1500
        if locked_321 and locked_before_3 and not opened_in_B:
            return "C", False, [], {**meta, "no_entry": "locked_from_before3_never_opened"}
        if touch_B is not None:
            return "B", True, [(1.0, uc * 0.999, touch_B, 1)], {**meta, "entry_kind": "B_uc0.999"}
        if px_C == px_C and px_C > 0 and not locked_321:             # SWEPT: was px==px and px>0 and not locked_321
            return "C", True, [(1.0, px_C, sweep_hm, 3)], {**meta, "entry_kind": f"C_{sweep_hm}"}
        return "C", False, [], {**meta, "no_entry": "locked_or_no_swept_time"}

    if config == "baseline":
        if uc_in_230_300 is not None:
            w = (hm > uc_in_230_300) & (hm < BC.HM_1521)
            wl, wh = lo[w], hm[w]
            i19 = np.where(wl <= l19)[0]
            if len(i19):
                legs = [(0.5, l19, int(wh[i19[0]]), 1)]
                i17 = np.where(wl <= l17)[0]
                if len(i17):
                    legs.append((0.5, l17, int(wh[i17[0]]), 1)); full = True; k = "17%"
                elif px == px and not locked_321:                    # UNCHANGED: Cat A leg2 still fixed 3:21
                    legs.append((0.5, px, BC.HM_1521, 2)); full = True; k = "3:21"
                else:
                    full = False; k = "half"
                return "A", True, legs, {**meta, "cat_a_full": full, "leg2": k, "uc_after_first_fill": True}
            return catB_or_C()
        return catB_or_C()
    raise ValueError("only 'baseline' config supported in this sweep")


def run_config_swept(cache, sweep_hm):
    """Byte-identical copy of BC.run_config() for config='baseline', except classify_swept is used."""
    recs = []
    for c in cache:
        cat, entered, legs, meta = classify_swept("baseline", c["pc"], c["eg"]["hm"], c["eg"]["hi"], c["eg"]["lo"], c["eg"]["op"], sweep_hm)
        recs.append({**{k: c[k] for k in ("symbol", "entry_date", "exit_date", "o565", "o719", "o879",
                                          "nhm", "nhi", "lhm", "nlo")},
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
        tgt = avg * BC.LONG_TGT
        hit = r["nhi"] >= tgt
        th = int(r["nhm"][hit].min()) if hit.any() else 10 ** 9
        o565, o719, o879 = r["o565"], r["o719"], r["o879"]
        if th <= BC.T1:
            xp, xt = tgt, "target_pre_0925"
        elif o565 == o565 and o565 > avg:
            xp, xt = o565, "positive_0925"
        elif th <= BC.T2:
            xp, xt = tgt, "target_0925_1159"
        elif o719 == o719:
            xp, xt = o719, "exit_1159"
        else:
            continue
        xhm = BC.T1 if xt == "positive_0925" else (BC.T2 if xt == "exit_1159" else th)
        long_pnl = shares * (xp - avg)
        stgt = xp * BC.SHORT_TGT
        sw = (r["lhm"] > xhm) & (r["nlo"] <= stgt)
        if sw.any():
            cover = stgt
        elif o879 == o879:
            cover = o879
        else:
            cover = np.nan
        has_short = cover == cover
        short_pnl = shares * (xp - cover) if has_short else 0.0
        snotl = shares * xp if has_short else 0.0
        comb = long_pnl + short_pnl
        rows.append({"symbol": r["symbol"], "entry_date": r["entry_date"], "category": r["category"],
                     "capital_deployed": cap, "gross_pnl": comb,
                     "netA_pnl": comb - BC.R023 * cap - BC.SR * snotl,
                     "netB_pnl": comb - BC.R038 * cap - BC.SR * snotl})
    return pd.DataFrame(rows)


def main():
    print("Building cache (identical to locked baseline)...", flush=True)
    cache_full = BC.build_cache()
    cache = [c for c in cache_full if WIN_START <= c["entry_date"] <= WIN_END]
    print(f"cache: {len(cache_full):,} total stock-days | {len(cache):,} in Apr-Jul 2026 window", flush=True)

    def summarize(T, sweep_hm):
        n_C = int((T["category"] == "C").sum())
        n_all = len(T)
        row = {"sweep_time": BC.lbl(sweep_hm), "sweep_hm": sweep_hm, "n_total_trades": n_all, "n_catC_trades": n_C}
        for s, col in [("gross", "gross_pnl"), ("net_A", "netA_pnl"), ("net_B", "netB_pnl")]:
            row[f"{s}_total_pnl_inr"] = round(T[col].sum(), 0)
            row[f"{s}_win_rate_pct"] = round((T[col] > 0).mean() * 100, 2) if n_all else 0.0
        return row

    print(f"\nrunning baseline (fixed {BC.lbl(BASELINE_HM)})...", flush=True)
    T_base = run_config_swept(cache, BASELINE_HM)
    base_row = summarize(T_base, BASELINE_HM)
    print(f"  baseline: {base_row}", flush=True)

    results = [base_row]
    for hm in SWEEP_RANGE:
        if hm == BASELINE_HM:
            continue
        T_sw = run_config_swept(cache, hm)
        row = summarize(T_sw, hm)
        results.append(row)
        print(f"  {BC.lbl(hm)}: gross={row['gross_total_pnl_inr']:.0f} netA={row['net_A_total_pnl_inr']:.0f} "
              f"netB={row['net_B_total_pnl_inr']:.0f} catC_n={row['n_catC_trades']} win%(gross)={row['gross_win_rate_pct']}", flush=True)

    RES = pd.DataFrame(results)
    for s in ["gross", "net_A", "net_B"]:
        RES[f"delta_{s}_pnl_vs_baseline"] = (RES[f"{s}_total_pnl_inr"] - base_row[f"{s}_total_pnl_inr"]).round(0)
        RES[f"delta_{s}_win_rate_pp"] = (RES[f"{s}_win_rate_pct"] - base_row[f"{s}_win_rate_pct"]).round(2)

    RES_sorted = RES.sort_values("delta_net_A_pnl_vs_baseline", ascending=False).reset_index(drop=True)

    pd.set_option("display.width", 260)
    print("\n=== FULL SWEEP, sorted by delta net_A P&L vs 3:21pm baseline (best first) ===")
    cols_show = ["sweep_time", "n_catC_trades", "gross_total_pnl_inr", "net_A_total_pnl_inr", "net_B_total_pnl_inr",
                 "delta_net_A_pnl_vs_baseline", "net_A_win_rate_pct", "delta_net_A_win_rate_pp"]
    print(RES_sorted[cols_show].to_string(index=False))

    # ---- stable region check: rolling-3 average of delta_net_A, find best contiguous cluster ----
    RES_bytime = RES.sort_values("sweep_hm").reset_index(drop=True)
    RES_bytime["delta_roll3"] = RES_bytime["delta_net_A_pnl_vs_baseline"].rolling(3, center=True).mean()
    best_roll_idx = RES_bytime["delta_roll3"].idxmax()
    best_roll_center = RES_bytime.loc[best_roll_idx, "sweep_time"] if pd.notna(best_roll_idx) else None
    print(f"\n=== STABLE REGION CHECK (3-min rolling avg of delta vs baseline) ===")
    print(RES_bytime[["sweep_time", "delta_net_A_pnl_vs_baseline", "delta_roll3"]].to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "catC_entry_time_sweep_apr_jul2026.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "Category C entry-TIME sweep, RESTRICTED to April-July 2026 only. Category A/B logic, "
                      "exit rules, and capital allocation are UNCHANGED, byte-identical copies of the locked "
                      "baseline_and_cross_final.py. Only Category C's own fill time is swept 15:00-15:28."},
            {"note": "Category A's leg2 3:21pm fallback and the 15:21-based circuit-lock validity check stay "
                      "FIXED at the real 3:21pm regardless of Category C's swept time -- confirmed no logical "
                      "conflict arises for any swept time, including times before 15:21 (the lock check is a "
                      "backtest-only historical eligibility gate, not something Category C's price needs live)."},
            {"note": f"Cache restricted to entry_date between {WIN_START} and {WIN_END} BEFORE running the "
                      "per-day capital sequencing -- valid because each day's Rs 5L pool sequencing is fully "
                      "independent day-to-day (resets every day), so this does not distort results."},
            {"note": "SAMPLE SIZE FLAG: this is a ~4-month window only. Treat any single best time as a narrow-"
                      "period observation, not a validated improvement -- look at the stable region, not one cell."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        RES_sorted.to_excel(w, sheet_name="Sorted_by_Improvement", index=False)
        RES_bytime.to_excel(w, sheet_name="By_Time_with_Rolling_Avg", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)

    print(f"\nSaved -> {OUTDIR}/catC_entry_time_sweep_apr_jul2026.xlsx")


if __name__ == "__main__":
    main()
