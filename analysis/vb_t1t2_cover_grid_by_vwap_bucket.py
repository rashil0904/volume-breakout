# -*- coding: utf-8 -*-
"""vb_t1t2_cover_grid_by_vwap_bucket.py — runs the SAME t1/t2/cover full-grid sweep as
vb_t1t2_cover_full_grid_sweep.py, separately within each entry-day VWAP-close-move bucket (5-10%, 10-15%,
15-20%) established in vb_vwap_close_move_segmentation.py. Category A/B/C entry logic and capital
sequencing are computed EXACTLY as in the full/unrestricted run (the real historical ₹5L pool was shared
with ALL that day's qualifying stocks, not just the ones in a given bucket) -- bucket membership is applied
ONLY as a post-hoc filter on which rows' P&L feeds into the aggregate grid metrics, by slicing the row-
level matrices before handing them to the UNCHANGED run_grid()/eval_baseline()/analyze_run() functions
from the full-grid sweep module. No sweep logic is duplicated or reimplemented.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC
import vb_t1t2_cover_full_grid_sweep as sweep

OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "t1t2_cover_grid_by_vwap_bucket"
OUTDIR.mkdir(parents=True, exist_ok=True)
BUCKETS = {"5-10pct": (5, 10), "10-15pct": (10, 15), "15-20pct": (15, 20)}


def slice_M(M, mask):
    return dict(avg=M["avg"][mask], shares=M["shares"][mask], cap=M["cap"][mask], tgt=M["tgt"][mask],
                th=M["th"][mask], cols=M["cols"], OPEN_M=M["OPEN_M"][mask, :], LOW_M=M["LOW_M"][mask, :],
                vwap_move_pct=M["vwap_move_pct"][mask])


def main():
    total_combos = len(sweep.T_RANGE) * (len(sweep.T_RANGE) - 1) // 2 * len(sweep.COVER_RANGE)
    print(f"Grid per bucket: {total_combos:,} (t1,t2,cover) combinations x {len(BUCKETS)} buckets", flush=True)

    print("Building full-history wide cache (with entry-day VWAP-move attached per stock-day)...", flush=True)
    cache_full = sweep.build_wide_cache()
    rows_full = sweep.entry_positions(cache_full)
    print(f"entered & sized positions (full, unrestricted): {len(rows_full):,}", flush=True)
    M_full = sweep.build_matrices(rows_full)

    vmove = M_full["vwap_move_pct"]
    n_missing_vwap = int(np.isnan(vmove).sum())
    print(f"rows missing a resolvable VWAP-move value: {n_missing_vwap:,} of {len(vmove):,}", flush=True)

    summaries = []
    for label, (lo, hi) in BUCKETS.items():
        mask = (vmove >= lo) & (vmove < hi)
        n_in_bucket = int(mask.sum())
        print(f"\n{'='*80}\nBUCKET {label}: {n_in_bucket:,} entered positions in range [{lo},{hi})%", flush=True)
        if n_in_bucket < 5:
            print(f"  SKIPPED -- too few positions ({n_in_bucket}) for a meaningful grid sweep", flush=True)
            continue
        M_b = slice_M(M_full, mask)

        LO_b, CB_b = sweep.run_grid(M_b, label)
        base_lo_b, base_cb_b = sweep.eval_baseline(M_b)
        summary_b, best_cb_b = sweep.analyze_run(label, LO_b, CB_b, base_lo_b, base_cb_b, OUTDIR)
        summary_b["n_entered_positions_in_bucket"] = n_in_bucket
        summaries.append(summary_b)

    SUMMARY = pd.DataFrame(summaries)
    pd.set_option("display.width", 240)
    print("\n=== CROSS-BUCKET SUMMARY (best combined combo per bucket) ===")
    cols_show = [c for c in ["run", "n_entered_positions_in_bucket", "best_cb_t1", "best_cb_t2", "best_cb_cover",
                             "best_cb_netA", "best_cb_delta_netA", "best_cb_win", "best_cb_n_trades",
                             "nbhd_pct_above_baseline"] if c in SUMMARY.columns]
    print(SUMMARY[cols_show].to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "t1t2_cover_grid_by_vwap_bucket_SUMMARY.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "t1/t2/short-cover full-grid sweep run SEPARATELY within each entry-day VWAP-close-move "
                      "bucket (5-10%, 10-15%, 15-20%). Category A/B/C entry logic and capital sequencing are "
                      "UNCHANGED and computed on the FULL, unrestricted day (the real ₹5L pool was shared with "
                      "ALL qualifying stocks that day, not just the bucket's own stocks) -- bucket membership is "
                      "applied only as a post-hoc filter on which positions' P&L feeds the grid's aggregate "
                      "metrics. This reuses the exact run_grid/eval_baseline/analyze_run logic from the full-"
                      "history sweep, applied to a row-sliced subset -- no sweep logic was duplicated."},
            {"note": f"{n_missing_vwap:,} of {len(vmove):,} entered positions could not get a resolvable VWAP-"
                      "move value (e.g. missing raw 1-min data for that stock/date) and are excluded from every "
                      "bucket (not just misassigned) -- same standing exclusion for all three runs."},
            {"note": "SAMPLE SIZE: restricting to a VWAP-move range shrinks the trade count substantially versus "
                      "the full-history sweep -- treat any bucket with a notably smaller n as a narrower, less "
                      "reliable observation, per the project's standing overfit caution."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        SUMMARY.to_excel(w, sheet_name="Cross_Bucket_Summary", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)

    print(f"\nSaved -> {OUTDIR}/t1t2_cover_grid_by_vwap_bucket_SUMMARY.xlsx (+ per-bucket grid parquet files)")


if __name__ == "__main__":
    main()
