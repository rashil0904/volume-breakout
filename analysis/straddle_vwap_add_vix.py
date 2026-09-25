# -*- coding: utf-8 -*-
"""straddle_vwap_add_vix.py — ADDITIVE: add India VIX context to the ATM Straddle VWAP report.
Adds entry-time VIX to each segment, a VIX-bucket summary sheet, and a VIX x DTE breakdown — preserving all
existing sheets. VIX = India VIX 1-min close at each segment's entry minute (intraday, matches the strategy)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "straddle_vwap"; XLSX = OUTDIR / "straddle_vwap.xlsx"; CSV = OUTDIR / "straddle_vwap_segments.csv"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]


def vbucket(v):
    if pd.isna(v): return "n/a"
    if v < 9: return "<9"
    if v >= 20: return ">20"
    return f"{int(v)}-{int(v)+1}"


def main():
    T = pd.read_csv(CSV)
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vser = vx.assign(ts=vts).set_index("ts")["close"].sort_index()
    ent_ts = pd.to_datetime(T["date"].astype(str) + " " + T["entry_time"].astype(str))
    T["entry_vix"] = [round(float(vser.asof(t)), 2) if pd.notna(vser.asof(t)) else np.nan for t in ent_ts]
    T["vix_bucket"] = T["entry_vix"].map(vbucket)
    em = pd.to_datetime(T["entry_time"], format="%H:%M"); xm = pd.to_datetime(T["exit_time"], format="%H:%M")
    T["hold_min"] = ((xm - em).dt.total_seconds() / 60).round(0)

    def bmet(sub):
        return {"segments": len(sub), "win_%": round((sub.pnl_points > 0).mean() * 100, 1) if len(sub) else 0,
                "total_points": round(sub.pnl_points.sum(), 1), "avg_points": round(sub.pnl_points.mean(), 2) if len(sub) else 0,
                "avg_hold_min": round(sub.hold_min.mean(), 1) if len(sub) else 0,
                "leg_shift_exits": int((sub.exit_reason == "leg-shift").sum()), "leg_shift_%": round((sub.exit_reason == "leg-shift").mean() * 100, 1) if len(sub) else 0,
                "shift_entries": int((sub.segment_type == "shift").sum()), "avg_entry_vix": round(sub.entry_vix.mean(), 2) if len(sub) else np.nan}

    present = [b for b in VIX_BUCKETS if (T.vix_bucket == b).any()]
    VB = pd.DataFrame([{"vix_bucket": b, **bmet(T[T.vix_bucket == b])} for b in present])
    # VIX x DTE: total P&L matrix + segment counts
    pnl_mat = T.pivot_table(index="vix_bucket", columns="DTE", values="pnl_points", aggfunc="sum").reindex(present).round(1)
    cnt_mat = T.pivot_table(index="vix_bucket", columns="DTE", values="pnl_points", aggfunc="size").reindex(present)
    pnl_mat = pnl_mat.reset_index(); cnt_mat = cnt_mat.reset_index()
    VXD = pd.concat([pnl_mat.assign(_metric="total_points"), cnt_mat.assign(_metric="n_segments")], ignore_index=True)
    cols = ["_metric", "vix_bucket"] + [c for c in VXD.columns if c not in ("_metric", "vix_bucket")]
    VXD = VXD[cols].sort_values(["_metric", "vix_bucket"])

    # preserve existing sheets, add VIX column to Trades_Segments + new sheets
    keep = {s: pd.read_excel(XLSX, sheet_name=s) for s in ["Summary", "DTE_Buckets", "By_Exit_Reason", "Daily_Summary"]}
    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        keep["Summary"].to_excel(w, sheet_name="Summary", index=False)
        keep["DTE_Buckets"].to_excel(w, sheet_name="DTE_Buckets", index=False)
        VB.to_excel(w, sheet_name="VIX_Buckets", index=False)
        VXD.to_excel(w, sheet_name="VIX_by_DTE", index=False)
        keep["By_Exit_Reason"].to_excel(w, sheet_name="By_Exit_Reason", index=False)
        keep["Daily_Summary"].to_excel(w, sheet_name="Daily_Summary", index=False)
        T.to_excel(w, sheet_name="Trades_Segments", index=False)     # now includes entry_vix + vix_bucket
    T.to_csv(CSV, index=False)

    pd.set_option("display.width", 200)
    print("STRADDLE VWAP — India VIX context added (entry-minute VIX)")
    print(f"segments {len(T)} | VIX coverage: {int(T.entry_vix.notna().sum())}/{len(T)} | period {T.date.min()}..{T.date.max()}")
    print("\n--- VIX BUCKET BREAKDOWN ---"); print(VB.to_string(index=False))
    print("\n--- VIX x DTE total P&L (points) ---"); print(pnl_mat.to_string(index=False))
    print(f"\nSaved -> {XLSX} (added VIX_Buckets, VIX_by_DTE + entry_vix/vix_bucket on Trades_Segments; existing sheets preserved)")


if __name__ == "__main__":
    main()
