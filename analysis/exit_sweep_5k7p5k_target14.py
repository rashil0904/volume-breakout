# -*- coding: utf-8 -*-
"""
exit_sweep_5k7p5k_target14.py
=============================
Exit-timing sweep for the ₹5,000–7,500 Cr variant with a FIXED 14% profit-target overlay
on both exit types. Entry fixed: mcap 5,000–7,500 Cr, lookback 36, volume 6x, 3:15pm,
+5% daily return. Qualifying trade set built ONCE; each combo slices a shared candle cache.

  EXIT TYPE 1 — full 100% exit at time T (target scan 09:15 → before T):        23 combos
  EXIT TYPE 2 — conditional split (positive→t1, rest→t2) with 4-phase target:  253 combos
                                                                        total = 276 combos

Exit logic (run_A / run_cond) and the candle cache (fetch_nextday_ohlc) are reused verbatim
from profit_target_sweep — no re-implementation. Sizing = pool-split (cap ₹1L/trade), the
established variant's sizing, via mcap_band_comparison.base_from_diag.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import profit_target_sweep as pts
import mcap_band_comparison as MC

OUTDIR = rb.RESULTS / "exit_sweep_5k7p5k_target14"
DIAG = rb.RESULTS / "diagnostic_table_mcap5k7p5k.csv"
BASE_POOL = fpr_base = 500_000
TARGET = 14
TIMES_HM = list(range(570, 901, 15))                 # 09:30 .. 15:00 (23)
SMALL = 30


def tlabel(hm):
    return f"{hm // 60:02d}:{hm % 60:02d}"


def combo_metrics(exit_type, t1, t2, pnl, ret, et, mask, cap, target_cats):
    m = mask
    n = int(m.sum())
    if n == 0:
        return None
    p, r, e = pnl[m], ret[m], et[m]
    hit = np.isin(e, target_cats)
    return {
        "exit_type": exit_type,
        "exit_time_1": tlabel(t1),
        "exit_time_2": tlabel(t2) if t2 is not None else "",
        "n_trades": n,
        "win_rate_pct": round(float((p > 0).mean() * 100), 2),
        "avg_return_per_trade_pct": round(float(r.mean()), 4),
        "median_return_per_trade_pct": round(float(np.median(r)), 4),
        "total_return_fixedbase_pct": round(float(p.sum()) / BASE_POOL * 100, 4),
        "total_pnl_inr": round(float(p.sum()), 0),
        "avg_capital_deployed_per_trade": round(float(cap[m].mean()), 0),
        "pct_of_trades_hitting_target": round(float(hit.mean() * 100), 2),
        "small_sample_flag": "n_trades<30" if n < SMALL else "",
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not DIAG.exists():
        raise SystemExit(f"Missing {DIAG.name}")

    # ── qualifying trade set (once) — +5% fixed, pool-split sizing ──
    base = MC.base_from_diag(DIAG)
    print(f"Qualifying base positions (₹5k–7.5k, +5%): {len(base):,}")

    # ── candle cache (once): next-day 15-min opens + highs ──
    print("Caching next-day 15-min opens + highs …")
    opens, highs = pts.fetch_nextday_ohlc(base)
    entry = base["entry"].values.astype(float)
    shares = base["shares"].values.astype(float)
    cap = base["cap"].values.astype(float)
    pct_high = (highs - entry[:, None]) / entry[:, None] * 100

    rows = []

    # ── EXIT TYPE 1: full exit @ T with 14% target scan 09:15 → before T ──
    FULL_CATS = ["target_hit"]
    for T in TIMES_HM:
        o_T = opens[:, pts.HCOL[T]]
        p1max = pts._window_max(pct_high, [pts.HCOL[hm] for hm in pts.CANDLE_HMS if hm < T])
        pnl, ret, et, mask = pts.run_A(TARGET, entry, shares, o_T, p1max)
        m = combo_metrics("full_exit", T, None, pnl, ret, et, mask, cap, FULL_CATS)
        if m:
            rows.append(m)

    # ── EXIT TYPE 2: conditional split (t1 < t2) with 4-phase 14% target ──
    COND_CATS = ["early_target_pre_t1", "early_target_between_t1_t2"]   # target-exit types
    for i, t1 in enumerate(TIMES_HM):
        ot1 = opens[:, pts.HCOL[t1]]
        ret_t1 = (ot1 - entry) / entry * 100
        p1max = pts._window_max(pct_high, [pts.HCOL[hm] for hm in pts.CANDLE_HMS if hm < t1])
        for t2 in TIMES_HM[i + 1:]:
            ot2 = opens[:, pts.HCOL[t2]]
            p3max = pts._window_max(pct_high, [pts.HCOL[hm] for hm in pts.CANDLE_HMS if t1 < hm < t2])
            pnl, ret, et, mask = pts.run_cond(TARGET, entry, shares, ot1, ot2, ret_t1, p1max, p3max)
            m = combo_metrics("conditional_split", t1, t2, pnl, ret, et, mask, cap, COND_CATS)
            if m:
                rows.append(m)

    df = pd.DataFrame(rows).sort_values("total_return_fixedbase_pct",
                                        ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(OUTDIR / "exit_sweep_5k7p5k_target14.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "exit_sweep_5k7p5k_target14.xlsx", engine="openpyxl") as w:
        df.to_excel(w, sheet_name="all_combos", index=False)

    pd.set_option("display.width", 230)
    show = ["rank", "exit_type", "exit_time_1", "exit_time_2", "n_trades", "win_rate_pct",
            "avg_return_per_trade_pct", "median_return_per_trade_pct",
            "total_return_fixedbase_pct", "total_pnl_inr", "avg_capital_deployed_per_trade",
            "pct_of_trades_hitting_target", "small_sample_flag"]
    best_full = df[df["exit_type"] == "full_exit"].iloc[0]
    best_cond = df[df["exit_type"] == "conditional_split"].iloc[0]
    print(f"\nTotal combinations: {len(df)}  (full_exit {(df.exit_type=='full_exit').sum()}, "
          f"conditional_split {(df.exit_type=='conditional_split').sum()})")
    print("\n--- BEST full_exit ---")
    print(best_full[show[1:]].to_string())
    print("\n--- BEST conditional_split ---")
    print(best_cond[show[1:]].to_string())
    print("\n" + "=" * 130)
    print("TOP 15 ACROSS BOTH EXIT TYPES (by total_return_fixedbase_pct)")
    print("=" * 130)
    print(df[show].head(15).to_string(index=False))

    # ── heatmap: conditional_split t1 × t2, upper triangle, total return ──
    cond = df[df["exit_type"] == "conditional_split"]
    lab = [tlabel(h) for h in TIMES_HM]
    grid = np.full((len(TIMES_HM), len(TIMES_HM)), np.nan)
    pos = {tlabel(h): k for k, h in enumerate(TIMES_HM)}
    for _, r in cond.iterrows():
        grid[pos[r["exit_time_1"]], pos[r["exit_time_2"]]] = r["total_return_fixedbase_pct"]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=90, fontsize=7)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("t2 (second exit)"); ax.set_ylabel("t1 (first exit)")
    ax.set_title("Conditional split (14% target) — total_return_fixedbase_pct\n"
                 "₹5,000–7,500 Cr variant, t1 < t2 upper triangle", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="total return %")
    br, bc = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.plot(bc, br, "r*", markersize=18, markeredgecolor="white")
    fig.tight_layout(); fig.savefig(OUTDIR / "conditional_split_heatmap.png", dpi=130); plt.close(fig)

    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
