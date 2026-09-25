# -*- coding: utf-8 -*-
"""vb_catBC_boundary_sweep_postCAS.py — SAME linked Category B/C boundary sweep as vb_catBC_boundary_sweep.py
(classify_boundary/run_config_boundary reused unmodified), but:
  - FULL range 3:00pm-3:21pm, 1-min steps (22 values, was 3:13-3:21 / 9 values in the first run).
  - Restricted to entry dates 2026-08-03 (CAS start) through the last date available in the data.
  - Signal source = the NEW daily market-cap scan (vb_baseline_newmcap_aug18.py), NOT the locked
    diagnostic_table.csv, because that locked table stops at 2026-07-31 and has no rows at all in this window.

CAS CAVEAT: since 2026-08-03, F&O-segment stocks freeze (zero volume, static price) from 15:15 to 15:28, with
the auction print at 15:29 (see project memory). Any boundary in [15:15, 15:21] therefore reads a FROZEN,
non-tradable candle for F&O-segment stocks -- both Category B's touch-detection in that sub-window and a
Category C fill priced at that boundary would be unrealistic for those names. Flagged and quantified below,
not silently backtested as if real.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC
import vb_baseline_newmcap_aug18 as V
import vb_catBC_boundary_sweep as SW      # classify_boundary / run_config_boundary / hm2s (reused, unmodified)

OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "catBC_boundary_sweep_postCAS"; OUTDIR.mkdir(parents=True, exist_ok=True)
CAS_START = pd.Timestamp("2026-08-03").date()
BOUNDARIES = list(range(900, 922))     # 15:00 .. 15:21 inclusive, 1-min steps (22 values)
CAS_FREEZE_LO, CAS_FREEZE_HI = 915, 921    # 15:15-15:21 (freeze window extends to 15:28, but Cat C never fills beyond 15:21)


def main():
    S = pd.read_parquet(V.SCAN_FN); S["date"] = pd.to_datetime(S["date"]).dt.date
    last_date = S["date"].max()
    print(f"post-CAS window: {CAS_START} .. {last_date}", flush=True)
    new_mcap = V.load_new_mcap()

    win = S[(S["date"] >= CAS_START) & (S["date"] <= last_date)].copy()
    cand = win[win["passes_vol"] & win["passes_ret"]].copy()
    cand["in_new"] = [(s in set(new_mcap[d]["symbol"])) if d in new_mcap else False for s, d in zip(cand["symbol"], cand["date"])]
    sig = cand[cand["in_new"]][["symbol", "folder", "date", "pc"]].reset_index(drop=True)
    print(f"vol+ret signals in window: {len(cand)} | in new-mcap band: {len(sig)}", flush=True)

    cache, extra = V.build_cache(sig)
    print(f"cache (have next-day data): {len(cache):,} stock-days", flush=True)

    # CAS-frozen flag per entry: 15:21 candle volume == 0 (proxy for the whole 15:15-15:28 freeze on that name)
    frozen_keys = {(s, d) for (s, d), e in extra.items() if e["vol_1521"] == 0}
    print(f"entries with a FROZEN (zero-volume) 15:21 candle: {len(frozen_keys)} of {len(cache)}", flush=True)

    fno = V.fno_underlyings()
    if fno is not None:
        fno_keys = {(s, d) for s, d in zip(sig["symbol"], sig["date"]) if s in fno}
        print(f"in-band signals on F&O-segment underlyings: {len(fno_keys)} of {len(sig)}", flush=True)

    results = {}
    for B in BOUNDARIES:
        T = SW.run_config_boundary(cache, B)
        results[B] = T
        print(f"  boundary {SW.hm2s(B)}: {len(T)} trades (A={int((T['category']=='A').sum())} B={int((T['category']=='B').sum())} C={int((T['category']=='C').sum())}) "
              f"netA={T['netA_pnl'].sum():,.0f}", flush=True)

    # sanity check: boundary=921 must reproduce the locked-engine baseline run on this SAME post-CAS cache/signal set
    T_base_locked, _ = BC.run_config("baseline", cache)
    key = ["symbol", "entry_date"]
    m = T_base_locked.set_index(key)[["gross_pnl"]].sort_index()
    m2 = results[921].set_index(key)[["gross_pnl"]].sort_index()
    same_idx = len(m) == len(m2) and m.index.equals(m2.index)
    max_diff = float((m["gross_pnl"] - m2["gross_pnl"]).abs().max()) if same_idx else float("nan")
    print(f"\n=== SANITY CHECK: boundary=15:21 vs locked engine on this cache === same_trade_set={same_idx} max|gross diff|={max_diff}", flush=True)
    assert same_idx and max_diff < 1e-6, "boundary=15:21 does NOT reproduce the locked engine on this data -- STOP"
    print("PASSED.", flush=True)

    def summarize(T, label, B):
        if T.empty:
            return {"boundary": label, "n_trades": 0}
        Tk = set(zip(T["symbol"], T["entry_date"]))
        n_frozen = len(Tk & frozen_keys) if CAS_FREEZE_LO <= B <= CAS_FREEZE_HI else 0
        o = {"boundary": label, "boundary_hm": B, "n_trades": len(T), "n_catA": int((T["category"] == "A").sum()),
             "n_catB": int((T["category"] == "B").sum()), "n_catC": int((T["category"] == "C").sum()),
             "n_trades_on_CAS_frozen_candle": n_frozen}
        for s in ["gross", "netA", "netB"]:
            p = T[f"{s}_pnl"]
            o[f"{s}_total_pnl_inr"] = round(p.sum()); o[f"{s}_return_pct_of_5L"] = round(p.sum() / BC.BASE_POOL * 100, 2)
            o[f"{s}_win_rate_pct"] = round((p > 0).mean() * 100, 2)
            pl = T[f"{s}_long_only"]
            o[f"{s}_LONGONLY_total_pnl_inr"] = round(pl.sum()); o[f"{s}_LONGONLY_return_pct_of_5L"] = round(pl.sum() / BC.BASE_POOL * 100, 2)
        return o

    Srows = [summarize(results[B], SW.hm2s(B), B) for B in BOUNDARIES]
    Sdf = pd.DataFrame(Srows)
    base_row = Sdf[Sdf["boundary_hm"] == 921].iloc[0]
    for s in ["gross", "netA", "netB"]:
        Sdf[f"delta_{s}_pnl_vs_baseline"] = (Sdf[f"{s}_total_pnl_inr"] - base_row[f"{s}_total_pnl_inr"]).round(0)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print("\n=== FULL SWEEP TABLE (post-CAS, 2026-08-03 .. %s) ===\n" % last_date + Sdf.to_string(index=False), flush=True)

    vals = Sdf.set_index("boundary_hm")["netA_return_pct_of_5L"]
    best_win_, best_mean = None, -1e18
    for i in range(len(BOUNDARIES) - 2):
        w = BOUNDARIES[i:i + 3]
        m3 = vals.loc[w].mean()
        if m3 > best_mean:
            best_mean, best_win_ = m3, w
    single_best = vals.idxmax()
    print(f"\n=== BEST STABLE 3-CONSECUTIVE-MINUTE REGION (by netA %) ===\n  boundaries {[SW.hm2s(b) for b in best_win_]} -> mean netA {best_mean:.2f}% "
          f"(individual: {[round(vals.loc[b],2) for b in best_win_]})\n  single best (reference only): {SW.hm2s(single_best)} -> netA {vals.loc[single_best]:.2f}%", flush=True)

    best_B = best_win_[1]
    Tb = results[best_B]
    print(f"\n=== LONG-ONLY vs LONG+SHORT at best-region centre {SW.hm2s(best_B)} ===")
    for s in ["gross", "netA", "netB"]:
        print(f"  {s}: LONG+SHORT total={Tb[f'{s}_pnl'].sum():,.0f} ({Tb[f'{s}_pnl'].sum()/BC.BASE_POOL*100:.2f}%) win={((Tb[f'{s}_pnl']>0).mean()*100):.1f}%  |  "
              f"LONG-ONLY total={Tb[f'{s}_long_only'].sum():,.0f} ({Tb[f'{s}_long_only'].sum()/BC.BASE_POOL*100:.2f}%) win={((Tb[f'{s}_long_only']>0).mean()*100):.1f}%", flush=True)

    fn = OUTDIR / "vb_catBC_boundary_sweep_postCAS.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": f"Post-CAS window {CAS_START} .. {last_date}. Signal source = new daily market-cap scan (vb_baseline_newmcap_aug18.py), "
                                "not the locked diagnostic_table.csv (which stops 2026-07-31 and has zero rows here). Boundary range widened to the full "
                                "3:00pm-3:21pm (22 values). CAS: since 2026-08-03, F&O-segment stocks freeze 15:15-15:28 (zero volume); any boundary in "
                                "[15:15,15:21] reads a frozen candle for those names -- n_trades_on_CAS_frozen_candle column flags this per boundary row. "
                                "Sanity check: boundary=15:21 reproduces the locked engine's own trade set/P&L exactly on this same data."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        Sdf.to_excel(w, sheet_name="Sweep_summary", index=False)
        for B in BOUNDARIES:
            results[B].to_excel(w, sheet_name=f"trades_{SW.hm2s(B).replace(':','')}", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
