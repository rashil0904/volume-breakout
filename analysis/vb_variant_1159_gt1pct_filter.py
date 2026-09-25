# -*- coding: utf-8 -*-
"""vb_variant_1159_gt1pct_filter.py — diagnostic backtest VARIANT of the locked baseline Volume-
Breakout strategy. Motivated by short_mae_by_long_return_bucket.py's finding: the exit_1159 tranche
(the half of each position held to 11:59am) loses money net of its double-down short when the long's
gross return at 11:59 is <=1%, but is profitable above that.

RULE CHANGE (this variant only -- NOT applied to the locked baseline_and_cross_final.py or its output
file): for the exit_1159 tranche only,
  - if gross_ret at 11:59 (o719 price) > 1%  -> exit long + open double-down short exactly as today
    (unchanged: cover at 14:39 or 5% target, whichever first).
  - if gross_ret at 11:59 <= 1%              -> do NOT exit yet. Hold the long until 14:39 (o879 price)
    and exit there. NO short is opened for this group at all (per explicit user confirmation -- there's
    no meaningful runway left between 14:39 and the 15:15/15:29 close for a double-down short to work).

All other tranches (positive_0925, target_pre_0925, target_0925_1159) are byte-for-byte unchanged.
Only the "baseline" config is run (the one used in baseline_final_performance.xlsx). Read-only
diagnostic -- writes its own output file, does not touch the locked strategy files.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC
import uc_staggered_dd_report as R

OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "variant_1159_gt1pct_filter"
OUTDIR.mkdir(parents=True, exist_ok=True)
GATE_PCT = 1.0   # gross_ret threshold at 11:59, per user spec


def run_config_variant(config, cache):
    recs = []
    for c in cache:
        cat, entered, legs, meta = BC.classify(config, c["pc"], c["eg"]["hm"], c["eg"]["hi"], c["eg"]["lo"], c["eg"]["op"])
        recs.append({**{k: c[k] for k in ("symbol", "entry_date", "exit_date", "o565", "o719", "o879",
                                          "nhm", "nhi", "lhm", "nlo")},
                     "category": cat, "entered": entered, "legs": legs, "meta": meta})
    ent = [r for r in recs if r["entered"] and r["legs"]]
    for r in ent:
        r["_a"] = [0.0] * len(r["legs"])
    from collections import defaultdict
    by_day = defaultdict(list)
    for r in ent:
        by_day[r["entry_date"]].append(r)
    diag_days = {}
    for d, rs in by_day.items():
        pool = BC.BASE_POOL; X = 0.0; p1, p2, cc = [], [], []
        for r in rs:
            for li, (frac, price, thm, ph) in enumerate(r["legs"]):
                (cc if ph == 3 else p2 if ph == 2 else p1).append(
                    (r, li, price) if ph == 3 else (r, li, 0.5 * BC.BASE_ALLOC) if ph == 2
                    else (thm, r, li, BC.BASE_ALLOC if frac >= 1.0 else 0.5 * BC.BASE_ALLOC))
        for thm, r, li, intd in sorted(p1, key=lambda x: (x[0] if x[0] is not None else 99999)):
            give = min(intd, pool); r["_a"][li] = give; pool -= give; X += give
        p2_sum = 0.0
        for r, li, intd in p2:
            give = min(intd, pool); r["_a"][li] = give; pool -= give; X += give; p2_sum += give
        nC = len(cc)
        per_c = min(BC.BASE_ALLOC, max(0.0, pool) / nC) if nC else 0.0
        for r, li, price in cc:
            r["_a"][li] = per_c
        diag_days[d] = {"X": X, "nC": nC, "per_c": per_c, "p2": p2_sum}

    rows = []
    for r in ent:
        shares = cap = 0.0; lbls = []; pxs = []
        for li, (frac, price, thm, ph) in enumerate(r["legs"]):
            s = np.floor(r["_a"][li] / price) if r["_a"][li] > 0 else 0.0
            if s > 0:
                shares += s; cap += s * price
                lbls.append(f"{r['meta'].get('entry_kind', r['category'])}:{round(price,2)}" if r["category"] != "A"
                            else f"{['19%','17%/3:21'][min(li,1)]}({int(s)})")
                pxs.append(round(price, 2))
        if shares <= 0:
            continue
        avg = cap / shares
        tgt = avg * BC.LONG_TGT
        hit = r["nhi"] >= tgt
        th = int(r["nhm"][hit].min()) if hit.any() else 10 ** 9
        o565, o719, o879 = r["o565"], r["o719"], r["o879"]

        variant_late_exit = False
        if th <= BC.T1:
            xp, xt, xhm = tgt, "target_pre_0925", th
        elif o565 == o565 and o565 > avg:
            xp, xt, xhm = o565, "positive_0925", BC.T1
        elif th <= BC.T2:
            xp, xt, xhm = tgt, "target_0925_1159", th
        elif o719 == o719:
            gross_ret_1159 = (o719 - avg) / avg * 100
            if gross_ret_1159 > GATE_PCT:
                xp, xt, xhm = o719, "exit_1159", BC.T2
            elif o879 == o879:
                xp, xt, xhm = o879, "exit_1439_held_lowmom", BC.COVER_HM
                variant_late_exit = True
            else:
                # no 14:39 price available next day (rare) -> fall back to the original 11:59 exit
                # so the trade isn't silently dropped from the backtest
                xp, xt, xhm = o719, "exit_1159", BC.T2
        else:
            continue

        long_pnl = shares * (xp - avg)
        if variant_late_exit:
            cover, sxt = np.nan, "no_short_variant"
        else:
            stgt = xp * BC.SHORT_TGT
            sw = (r["lhm"] > xhm) & (r["nlo"] <= stgt)
            if sw.any():
                cover, sxt = stgt, "short_target_5pct"
            elif o879 == o879:
                cover, sxt = o879, "short_cover_1439"
            else:
                cover, sxt = np.nan, "no_short"
        has_short = cover == cover
        short_pnl = shares * (xp - cover) if has_short else 0.0
        snotl = shares * xp if has_short else 0.0
        comb = long_pnl + short_pnl
        rows.append({"symbol": r["symbol"], "entry_date": r["entry_date"], "exit_date": r["exit_date"],
                     "category": r["category"], "cat_a_full": r["meta"].get("cat_a_full"),
                     "legs_filled": "+".join(lbls), "leg_prices": "+".join(map(str, pxs)),
                     "n_legs": len(lbls), "shares": int(shares), "capital_deployed": cap, "avg_entry": avg,
                     "long_exit_type": xt, "exit_time": BC.lbl(xhm), "exit_price": xp,
                     "short_exit_type": sxt, "cover_price": cover,
                     "long_pnl": long_pnl, "short_pnl": short_pnl, "combined_pnl": comb,
                     "long_cost_023": BC.R023 * cap, "long_cost_038": BC.R038 * cap, "short_cost": BC.SR * snotl,
                     "gross_pnl": comb, "netA_pnl": comb - BC.R023 * cap - BC.SR * snotl,
                     "netB_pnl": comb - BC.R038 * cap - BC.SR * snotl,
                     "uc_after_first_fill": r["meta"].get("uc_after_first_fill"),
                     "hit_uc_ever": r["meta"].get("hit_uc_ever")})
    T = pd.DataFrame(rows)
    for s in ["gross", "netA", "netB"]:
        T[f"{s}_ret"] = T[f"{s}_pnl"] / T["capital_deployed"] * 100
    ts = pd.to_datetime(T["entry_date"])
    T["year"] = ts.dt.year; T["month"] = ts.dt.strftime("%Y-%m")
    T["quarter"] = ts.dt.year.astype(str) + "Q" + ts.dt.quarter.astype(str)
    T["half_year"] = ts.dt.year.astype(str) + "H" + np.where(ts.dt.month <= 6, "1", "2")
    return T.sort_values(["entry_date", "symbol"]).reset_index(drop=True), diag_days


def main():
    print("Building cache (identical to locked baseline)...", flush=True)
    cache = BC.build_cache()
    print(f"cache: {len(cache):,} stock-days", flush=True)

    T_old = pd.read_excel(rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx", sheet_name="all_trades")
    T_new, diag_days = run_config_variant("baseline", cache)
    print(f"OLD trades: {len(T_old)} | NEW trades: {len(T_new)}", flush=True)

    changed_mask_old = T_old["long_exit_type"] == "exit_1159"
    n_1159_old = int(changed_mask_old.sum())
    n_moved = int((T_new["long_exit_type"] == "exit_1439_held_lowmom").sum())
    n_kept = int((T_new["long_exit_type"] == "exit_1159").sum())
    print(f"\nexit_1159 tranche (old): {n_1159_old}")
    print(f"  kept at 11:59 (>1% gate, unchanged): {n_kept}")
    print(f"  moved to 14:39 hold, no short (<=1% gate): {n_moved}")
    assert n_kept + n_moved == n_1159_old, "gate split doesn't add up to original tranche size"

    def totals(T, label):
        g, a, b = T["gross_pnl"].sum(), T["netA_pnl"].sum(), T["netB_pnl"].sum()
        print(f"{label:>22s} | gross={g:>14,.1f} | net_A={a:>14,.1f} | net_B={b:>14,.1f} | "
              f"fixedbase_gross%={g/BC.BASE_POOL*100:.2f}  net_A%={a/BC.BASE_POOL*100:.2f}  net_B%={b/BC.BASE_POOL*100:.2f}")
        return g, a, b

    print("\n=== WHOLE-STRATEGY TOTALS ===")
    old_g, old_a, old_b = totals(T_old, "BASELINE (current)")
    new_g, new_a, new_b = totals(T_new, "VARIANT (>1% gate)")
    print(f"{'DELTA':>22s} | gross={new_g-old_g:>+14,.1f} | net_A={new_a-old_a:>+14,.1f} | net_B={new_b-old_b:>+14,.1f}")

    print("\n=== exit_1159 TRANCHE ONLY (old vs. the same trades' fate in the variant) ===")
    old_1159 = T_old[changed_mask_old]
    print(f"  OLD (all forced to 11:59): long={old_1159.long_pnl.sum():,.1f}  short={old_1159.short_pnl.sum():,.1f}  "
          f"combined_gross={old_1159.gross_pnl.sum():,.1f}")
    new_1159_group = T_new[T_new["long_exit_type"].isin(["exit_1159", "exit_1439_held_lowmom"])]
    print(f"  NEW (gated): long={new_1159_group.long_pnl.sum():,.1f}  short={new_1159_group.short_pnl.sum():,.1f}  "
          f"combined_gross={new_1159_group.gross_pnl.sum():,.1f}")
    held = T_new[T_new["long_exit_type"] == "exit_1439_held_lowmom"]
    print(f"\n  of the {len(held)} trades moved to 14:39-hold-no-short:")
    print(f"    long_pnl total = {held.long_pnl.sum():,.1f}  (avg {held.long_pnl.mean():,.1f}/trade)  win_rate={round((held.long_pnl>0).mean()*100,1)}%")
    same_trades_old = old_1159.set_index(["symbol", "entry_date"])
    held_idx = held.set_index(["symbol", "entry_date"])
    common = same_trades_old.index.intersection(held_idx.index)
    print(f"    same {len(common)} trades' OLD combined (long+short forced @11:59) = {same_trades_old.loc[common, 'gross_pnl'].sum():,.1f}")

    with pd.ExcelWriter(OUTDIR / "variant_1159_gt1pct_filter.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "n_trades", "old": len(T_old), "new": len(T_new)},
            {"metric": "total_gross_pnl", "old": round(old_g, 1), "new": round(new_g, 1)},
            {"metric": "total_netA_pnl", "old": round(old_a, 1), "new": round(new_a, 1)},
            {"metric": "total_netB_pnl", "old": round(old_b, 1), "new": round(new_b, 1)},
            {"metric": "fixedbase_gross_pct", "old": round(old_g/BC.BASE_POOL*100, 2), "new": round(new_g/BC.BASE_POOL*100, 2)},
            {"metric": "fixedbase_netA_pct", "old": round(old_a/BC.BASE_POOL*100, 2), "new": round(new_a/BC.BASE_POOL*100, 2)},
            {"metric": "fixedbase_netB_pct", "old": round(old_b/BC.BASE_POOL*100, 2), "new": round(new_b/BC.BASE_POOL*100, 2)},
            {"metric": "n_exit_1159_original_tranche", "old": n_1159_old, "new": n_1159_old},
            {"metric": "n_kept_at_1159_gt1pct_gate", "old": "", "new": n_kept},
            {"metric": "n_moved_to_1439_hold_no_short", "old": "", "new": n_moved},
        ]).to_excel(w, sheet_name="Summary", index=False)
        T_new.to_excel(w, sheet_name="all_trades_variant", index=False)
        held.to_excel(w, sheet_name="moved_to_1439_trades", index=False)

    print(f"\nSaved -> {OUTDIR}/variant_1159_gt1pct_filter.xlsx")


if __name__ == "__main__":
    main()
