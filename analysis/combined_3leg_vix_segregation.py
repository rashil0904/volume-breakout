# -*- coding: utf-8 -*-
"""combined_3leg_vix_segregation.py — ADDS India VIX segregation on top of the ORIGINAL, UNCHANGED 3-leg
combined NIFTY strategy (analysis/combined_3leg.py: daily 2-day breakout+gap, 15-min Supertrend(10,3),
1-hour Supertrend(10,3)). Does NOT alter any entry/exit logic, Supertrend parameters, or the daily leg's
breakout/gap mechanics -- reuses combined_3leg.py's own functions/results directly, only adding a VIX
join + bucketed summary layer on top.

PER-LEG VIX BUCKETS: each leg's own trade list (from combined_3leg.xlsx), VIX at that trade's entry_time
(India VIX 1-min, exact-or-nearest-prior candle), bucketed into the project's standard 1-point bands.

COMBINED/NET VIX BUCKETS: reconstructs the same 15-min net-position array {+3,+1,-1,-3} the original
script computes (via its own supertrend()/st_trades() functions, byte-identical logic, not reimplemented
differently). Whenever net changes value, the PRIOR segment's P&L = prev_net * (price_at_this_change -
price_at_segment_start) is realized and attributed to the VIX value at that segment's OWN start (its
"entry VIX") -- this segment-level P&L attribution is a NEW derived calculation (the original script only
tracked %-time and efficiency ratio per state, not P&L per state-episode), built for this VIX-segregation
purpose only, added on top rather than replacing anything.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from supertrend_dual_tf import supertrend, ATR_N, FACT

D15 = rb.BASE / "data" / "nifty_15min_ohlc.csv"
LEG1_CSV = rb.RESULTS / "breakout_2day_gap" / "breakout_2day_gap_trades.csv"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
COMBINED_XLSX = rb.RESULTS / "combined_3leg" / "combined_3leg.xlsx"
OUTDIR = rb.RESULTS / "combined_3leg"; OUTDIR.mkdir(parents=True, exist_ok=True)
COST_PER_TRADE = 15.0


def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


def load_vix():
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vx = vx.assign(ts=vts).sort_values("ts").reset_index(drop=True)
    return vx[["ts", "close"]].rename(columns={"close": "vix"})


def vix_at(vx, ts_array):
    """VIX at each timestamp in ts_array, exact-or-nearest-PRIOR 1-min VIX candle (asof join)."""
    s = pd.Series(vx["vix"].to_numpy(), index=pd.to_datetime(vx["ts"].to_numpy())).sort_index()
    idx = pd.to_datetime(ts_array)
    return s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx).values


def bucket_summary(df, pnl_col, vix_col, label):
    order = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]
    df = df.copy(); df["vix_bucket"] = df[vix_col].apply(vbucket)
    g = df.groupby("vix_bucket").agg(n=(pnl_col, "size"), total_pnl=(pnl_col, "sum"), avg_pnl=(pnl_col, "mean"),
                                       win_rate=(pnl_col, lambda s: round((s > 0).mean() * 100, 1))).reindex(order)
    g["total_pnl"] = g["total_pnl"].round(1); g["avg_pnl"] = g["avg_pnl"].round(2)
    g = g.dropna(how="all").reset_index().rename(columns={"index": "vix_bucket"})
    g.insert(0, "leg", label)
    return g


def main():
    vx = load_vix()
    print(f"VIX data: {vx['ts'].min()} .. {vx['ts'].max()}", flush=True)

    # ---------- reload the ORIGINAL script's exact per-leg trade lists (already generated, unchanged) ----------
    L1 = pd.read_excel(COMBINED_XLSX, sheet_name="Leg1_Trades")
    L2 = pd.read_excel(COMBINED_XLSX, sheet_name="Leg2_Trades")
    L3 = pd.read_excel(COMBINED_XLSX, sheet_name="Leg3_Trades")
    for L in (L1, L2, L3):
        L["entry_time"] = pd.to_datetime(L["entry_time"])
        L["vix_entry"] = vix_at(vx, L["entry_time"].values)

    B1 = bucket_summary(L1, "net_points", "vix_entry", "LEG1 daily breakout+gap")
    B2 = bucket_summary(L2, "net_points", "vix_entry", "LEG2 15-min Supertrend")
    B3 = bucket_summary(L3, "net_points", "vix_entry", "LEG3 1-hour Supertrend")

    pd.set_option("display.width", 200)
    print("\n=== LEG1 (daily breakout+gap) VIX buckets ===\n" + B1.drop(columns="leg").to_string(index=False))
    print("\n=== LEG2 (15-min Supertrend) VIX buckets ===\n" + B2.drop(columns="leg").to_string(index=False))
    print("\n=== LEG3 (1-hour Supertrend) VIX buckets ===\n" + B3.drop(columns="leg").to_string(index=False))

    # ---------- reconstruct the ORIGINAL script's exact net-position array (byte-identical logic) ----------
    l1 = pd.read_csv(LEG1_CSV)
    l1_ft = pd.to_datetime(l1["entry_time"]).values
    l1_fd = l1["direction"].map({"Long": 1, "Short": -1}).values
    last_exit = pd.to_datetime(l1["exit_time"].iloc[-1]); last_dir = -l1_fd[-1]
    l1_ft = np.append(l1_ft, np.datetime64(last_exit)); l1_fd = np.append(l1_fd, last_dir)

    d = pd.read_csv(D15); ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d = d.assign(ts=ts).sort_values("ts").reset_index(drop=True); N = len(d); c15 = d["close"].values
    gt = d["ts"].values
    dir15 = supertrend(d["high"].values, d["low"].values, c15, ATR_N, FACT)
    mod = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    d["hb"] = pd.factorize(d["ts"].dt.strftime("%Y-%m-%d") + "_" + ((mod - 555) // 60).astype(int).astype(str))[0]
    hourly = d.groupby("hb").agg(h=("high", "max"), l=("low", "min"), c=("close", "last")).sort_index()
    dir1h = supertrend(hourly["h"].values, hourly["l"].values, hourly["c"].values, ATR_N, FACT)
    is_last = d["hb"].values != np.append(d["hb"].values[1:], -1)
    dir1h_pos = np.full(N, np.nan); dir1h_pos[is_last] = dir1h[d["hb"].values[is_last]]; dir1h_pos = pd.Series(dir1h_pos).ffill().values
    dp = np.nan_to_num(dir1h_pos).astype(int)

    ii = np.searchsorted(l1_ft, gt, side="right") - 1
    leg1 = np.where((ii >= 0) & (gt >= l1_ft[0]), l1_fd[ii.clip(0)], 0)
    leg2, leg3 = dir15, dp
    start = max(ATR_N + 1, int(np.argmax(~np.isnan(dir1h_pos))), int(np.searchsorted(gt, l1_ft[0], "right")))
    reg = np.arange(start, N)
    reg = reg[leg1[reg] != 0]
    net = leg1 + leg2 + leg3

    # ---------- NEW: segment-level P&L attribution for the net position, VIX-bucketed ----------
    seg_net = net[reg]; seg_ts = gt[reg]; seg_close = c15[reg]
    change_idx = np.where(np.diff(seg_net) != 0)[0] + 1
    bounds = np.concatenate([[0], change_idx, [len(seg_net)]])
    seg_rows = []
    for si in range(len(bounds) - 1):
        s0, s1_ = bounds[si], bounds[si + 1] - 1
        if s1_ <= s0:
            continue
        seg_pnl = seg_net[s0] * (seg_close[s1_] - seg_close[s0])
        seg_rows.append({"state": int(seg_net[s0]), "start_ts": pd.Timestamp(seg_ts[s0]), "end_ts": pd.Timestamp(seg_ts[s1_]),
                          "pnl_points": seg_pnl})
    SEG = pd.DataFrame(seg_rows)
    SEG["vix_entry"] = vix_at(vx, SEG["start_ts"].values)
    print(f"\ntotal net-position segments (state-episodes): {len(SEG)}")

    BC = bucket_summary(SEG, "pnl_points", "vix_entry", "COMBINED (net position)")
    print("\n=== COMBINED (net position) VIX buckets ===\n" + BC.drop(columns="leg").to_string(index=False))

    ALL_BUCKETS = pd.concat([B1, B2, B3, BC], ignore_index=True)

    with pd.ExcelWriter(OUTDIR / "combined_3leg_vix_segregation.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "VIX segregation ADDED ON TOP of the ORIGINAL, UNCHANGED 3-leg combined strategy "
                      "(analysis/combined_3leg.py) -- no entry/exit logic, Supertrend params, or daily-leg "
                      "breakout/gap mechanics were altered. Per-leg tables use the original strategy's own "
                      "trade lists with VIX joined at each entry_time (India VIX 1-min, asof/nearest-prior)."},
            {"note": "COMBINED/net-position table: whenever the net position {+3,+1,-1,-3} changes, the prior "
                      "segment's P&L = net_state x (15-min close at change - 15-min close at segment start) is "
                      "attributed to the VIX value at that segment's OWN start. This segment-level P&L-by-state "
                      "attribution is a NEW calculation built for this VIX task -- the original script only "
                      "tracked %-time and efficiency ratio per state, not P&L per episode."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        ALL_BUCKETS.to_excel(w, sheet_name="All_VIX_Buckets", index=False)
        B1.to_excel(w, sheet_name="Leg1_VIX_Buckets", index=False)
        B2.to_excel(w, sheet_name="Leg2_VIX_Buckets", index=False)
        B3.to_excel(w, sheet_name="Leg3_VIX_Buckets", index=False)
        BC.to_excel(w, sheet_name="Combined_VIX_Buckets", index=False)
        L1[["entry_time", "vix_entry", "net_points"]].to_excel(w, sheet_name="Leg1_Trades_with_VIX", index=False)
        L2[["entry_time", "vix_entry", "net_points"]].to_excel(w, sheet_name="Leg2_Trades_with_VIX", index=False)
        L3[["entry_time", "vix_entry", "net_points"]].to_excel(w, sheet_name="Leg3_Trades_with_VIX", index=False)
        SEG.to_excel(w, sheet_name="Combined_Segments_with_VIX", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)

    print(f"\nSaved -> {OUTDIR}/combined_3leg_vix_segregation.xlsx")


if __name__ == "__main__":
    main()
