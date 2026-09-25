# -*- coding: utf-8 -*-
"""vb_t1t2_cover_bucket_detailed_stats.py — for each VWAP-move bucket (5-10%, 10-15%, 15-20%) and each of
3 specific (t1,t2,cover) combos (current baseline 9:25/11:59/14:39, the full-history-optimal alt
9:16/11:00/14:57, and that bucket's own best combo from the grid sweep), computes per-trade combined
(long+short, net_A basis) P&L and derives avg/median overall plus avg/median of winners and losers
separately. Reuses build_wide_cache/entry_positions/build_matrices from the full-grid sweep module
unchanged -- only adds a per-trade (not aggregate-only) evaluator for a handful of exact cells.
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
BUCKETS = {"5-10pct": (5, 10), "10-15pct": (10, 15), "15-20pct": (15, 20)}
ALT_COMBO = (556, 660, 897)   # 9:16 / 11:00 / 14:57
BASE_COMBO = (BC.T1, BC.T2, BC.COVER_HM)   # 9:25 / 11:59 / 14:39
BUCKET_BEST = {   # from the earlier grid sweep's per-bucket best-combined result
    "5-10pct": (556, 635, 868),     # 9:16 / 10:35 / 14:28
    "10-15pct": (558, 659, 897),    # 9:18 / 10:59 / 14:57
    "15-20pct": (584, 708, 841),    # 9:44 / 11:48 / 14:01
}


def per_trade_pnl(M, t1, t2, cover):
    avg, shares, cap, tgt, th, cols, OPEN_M, LOW_M = M["avg"], M["shares"], M["cap"], M["tgt"], M["th"], M["cols"], M["OPEN_M"], M["LOW_M"]
    ot1 = OPEN_M[:, t1 - sweep.DATA_LO]; ot2 = OPEN_M[:, t2 - sweep.DATA_LO]
    m1 = th <= t1
    m2 = (~m1) & (ot1 == ot1) & (ot1 > avg)
    base_valid = m1 | m2
    m3 = (~base_valid) & (th <= t2)
    m4 = (~base_valid) & (~m3) & (ot2 == ot2)
    xp = np.where(base_valid, np.where(m1, tgt, ot1), np.where(m3, tgt, np.where(m4, ot2, np.nan)))
    xhm = np.where(base_valid, np.where(m1, th, t1), np.where(m3, th, np.where(m4, t2, np.nan)))
    valid = ~np.isnan(xp)
    long_pnl = shares * (xp - avg)
    stgt = xp * BC.SHORT_TGT
    sw = (cols[None, :] > xhm[:, None]) & (LOW_M <= stgt[:, None])
    has_hit = sw.any(axis=1); idx = np.argmax(sw, axis=1)
    short_hit_time = np.where(has_hit, cols[idx], np.inf)
    cover_open = OPEN_M[:, cover - sweep.DATA_LO]
    cover_price = np.where(short_hit_time <= cover, stgt, np.where(cover_open == cover_open, cover_open, np.nan))
    has_short = cover_price == cover_price
    short_pnl = np.where(has_short, shares * (xp - cover_price), 0.0)
    snotl = np.where(has_short, shares * xp, 0.0)
    comb = long_pnl + short_pnl
    netA = comb - BC.R023 * cap - BC.SR * snotl
    return netA[valid]   # per-trade net_A pnl, valid trades only


def stats_block(pnl):
    win = pnl[pnl > 0]; loss = pnl[pnl <= 0]
    return {
        "n_trades": len(pnl), "total_netA": round(pnl.sum(), 0),
        "avg_netA": round(pnl.mean(), 2), "median_netA": round(np.median(pnl), 2),
        "win_rate_pct": round((pnl > 0).mean() * 100, 2),
        "n_winners": len(win), "avg_winner": round(win.mean(), 2) if len(win) else np.nan, "median_winner": round(np.median(win), 2) if len(win) else np.nan,
        "n_losers": len(loss), "avg_loser": round(loss.mean(), 2) if len(loss) else np.nan, "median_loser": round(np.median(loss), 2) if len(loss) else np.nan,
    }


def main():
    print("Building full-history wide cache...", flush=True)
    cache_full = sweep.build_wide_cache()
    rows_full = sweep.entry_positions(cache_full)
    M_full = sweep.build_matrices(rows_full)
    vmove = M_full["vwap_move_pct"]
    print(f"entered positions: {len(vmove):,}", flush=True)

    all_rows = []
    for label, (lo, hi) in BUCKETS.items():
        mask = (vmove >= lo) & (vmove < hi)
        M_b = {k: (v[mask] if k not in ("cols",) else v) for k, v in M_full.items() if k != "OPEN_M" and k != "LOW_M"}
        M_b["OPEN_M"] = M_full["OPEN_M"][mask, :]; M_b["LOW_M"] = M_full["LOW_M"][mask, :]

        for combo_name, (t1, t2, cover) in [("baseline_9:25_11:59_14:39", BASE_COMBO),
                                             ("alt_9:16_11:00_14:57", ALT_COMBO),
                                             ("bucket_best", BUCKET_BEST[label])]:
            pnl = per_trade_pnl(M_b, t1, t2, cover)
            s = stats_block(pnl)
            s.update({"bucket": label, "combo": combo_name, "t1": BC.lbl(t1), "t2": BC.lbl(t2), "cover": BC.lbl(cover)})
            all_rows.append(s)

    R = pd.DataFrame(all_rows)
    cols = ["bucket", "combo", "t1", "t2", "cover", "n_trades", "total_netA", "avg_netA", "median_netA",
            "win_rate_pct", "n_winners", "avg_winner", "median_winner", "n_losers", "avg_loser", "median_loser"]
    R = R[cols]
    pd.set_option("display.width", 260)
    print("\n=== DETAILED PER-TRADE STATS (net_A basis) ===")
    print(R.to_string(index=False))

    out_fn = OUTDIR / "detailed_avg_median_stats.xlsx"
    with pd.ExcelWriter(out_fn, engine="openpyxl") as w:
        R.to_excel(w, sheet_name="Detailed_Stats", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 24)
    print(f"\nSaved -> {out_fn}")


if __name__ == "__main__":
    main()
