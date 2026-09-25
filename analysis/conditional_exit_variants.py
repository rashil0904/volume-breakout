# -*- coding: utf-8 -*-
"""
conditional_exit_variants.py
============================
Conditional exit scenarios on the EXISTING fixed 3:15pm-entry split universe.
Entries/shares are taken AS-IS from results/backtest_results.xlsx sheet
"2_Split trades" (which already carries the next-day 9:45 open = exit_945 and
11:00 open = exit_1100). Entries are never recomputed.

Variants compared (all 100%-position exits unless noted):
  U) Unconditional split (9:45 + 11am)  — 50% @9:45 + 50% @11am        (the existing split)
  C) Conditional split (positive->945, rest->11am)  — 2-way:
        return_at_945 >= 0  -> full exit @9:45
        else                -> full exit @11am
  D) Conditional deep-loss-cut (NEW) — 3-way:
        Bucket A (full exit @9:45): return_at_945 >= 0  OR  return_at_945 <= -3
        Bucket B (full exit @11am): -3 < return_at_945 < 0   (mild-loss band only)

return_at_945 = (open_945 - entry) / entry * 100.

total_ret is reported BOTH ways (no prior table to match):
  total_ret_fixedbase_pct  = Σpnl / CAPITAL_BASE * 100        (fixed ₹5L)
  total_ret_sumofdaily_pct = Σ(daily Σpnl/Σcap * 100)          (compute_stats)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

CB     = rb.CAPITAL_BASE
XLSX   = rb.RESULTS / "backtest_results.xlsx"
OUTDIR = rb.RESULTS / "conditional_exit"


def load_base():
    b = pd.read_excel(XLSX, sheet_name="2_Split trades")
    b = b[["date", "symbol", "entry", "shares", "exit_945", "exit_1100", "cap"]].copy()
    b["date"] = pd.to_datetime(b["date"])
    return b.reset_index(drop=True)


def overall_metrics(date, cap, pnl, ret):
    """Both total-return conventions + win/avg/median/n, reusing compute_stats."""
    df = pd.DataFrame({"date": date, "cap": cap, "pnl": pnl, "ret": ret})
    s  = rb.compute_stats(df)
    return dict(
        n_trades                 = len(df),
        win_rate_pct             = round(s["win_rate"], 2),
        total_ret_fixedbase_pct  = round(float(pnl.sum() / CB * 100), 2),
        total_ret_sumofdaily_pct = round(s["total_ret_pct"], 2),
        avg_return_per_trade_pct = round(s["avg_ret"], 4),
        median_return_per_trade_pct = round(s["median_ret"], 4),
    )


def subgroup(name, pnl, ret):
    if len(pnl) == 0:
        return dict(group=name, n=0, win_rate_pct=0.0,
                    avg_return_per_trade_pct=0.0, median_return_per_trade_pct=0.0,
                    total_pnl_rs=0.0)
    return dict(
        group=name, n=len(pnl),
        win_rate_pct=round(float((pnl > 0).mean() * 100), 2),
        avg_return_per_trade_pct=round(float(ret.mean()), 4),
        median_return_per_trade_pct=round(float(np.median(ret)), 4),
        total_pnl_rs=round(float(pnl.sum()), 0),
    )


THRESHOLDS = [-3, -4, -5, -6, -7, -8, -9, -10]


def three_bucket_sweep(ep, sh, o945, o110, ret945, ret110, s1, s2, date, cap):
    """
    Run the 3-bucket v2 logic across the deep-loss threshold grid.
    Only the A/C boundary moves; Bucket B (ret945>0, 50/50 split) is computed
    ONCE and reused for every threshold (never recomputed).
    Returns (summary_df, bucket_ac_df, bucket_B_once_dict).
    """
    pnl_B  = s1 * (o945 - ep) + s2 * (o110 - ep)   # Bucket B — unchanged across thresholds
    ret_B  = 0.5 * ret945 + 0.5 * ret110
    B_mask = ret945 > 0

    summary, bucket_rows = [], []
    for X in THRESHOLDS:
        A = ret945 <= X
        C = (ret945 > X) & (ret945 <= 0)
        pnl = np.select([A, B_mask, C], [sh * (o945 - ep), pnl_B, sh * (o110 - ep)])
        ret = np.select([A, B_mask, C], [ret945,           ret_B,  ret110])

        summary.append({"deep_loss_threshold": X, **overall_metrics(date, cap, pnl, ret)})
        sA = subgroup("A", pnl[A], ret[A])
        sC = subgroup("C", pnl[C], ret[C])
        bucket_rows.append({
            "threshold": X,
            "A_n": sA["n"], "A_win%": sA["win_rate_pct"], "A_avg": sA["avg_return_per_trade_pct"],
            "A_med": sA["median_return_per_trade_pct"], "A_totpnl_rs": sA["total_pnl_rs"],
            "C_n": sC["n"], "C_win%": sC["win_rate_pct"], "C_avg": sC["avg_return_per_trade_pct"],
            "C_med": sC["median_return_per_trade_pct"], "C_totpnl_rs": sC["total_pnl_rs"],
        })

    B_once = subgroup("Bucket B (positive, 50/50 split) — same for all thresholds",
                      pnl_B[B_mask], ret_B[B_mask])
    return pd.DataFrame(summary), pd.DataFrame(bucket_rows), B_once


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    b = load_base()
    ep   = b["entry"].values.astype(float)
    sh   = b["shares"].values.astype(float)
    o945 = b["exit_945"].values.astype(float)
    o110 = b["exit_1100"].values.astype(float)
    cap  = b["cap"].values.astype(float)
    date = b["date"].values

    ret945 = (o945 - ep) / ep * 100
    ret110 = (o110 - ep) / ep * 100

    # ── U) Unconditional split (50/50, run_split convention) ──
    s1 = np.floor(sh / 2); s2 = sh - s1
    pnl_u = s1 * (o945 - ep) + s2 * (o110 - ep)
    ret_u = 0.5 * ret945 + 0.5 * ret110

    # ── C) Conditional split (positive->945, rest->11am), 2-way full exits ──
    a_pos = ret945 >= 0
    pnl_c = np.where(a_pos, sh * (o945 - ep), sh * (o110 - ep))
    ret_c = np.where(a_pos, ret945, ret110)

    # ── D) Conditional deep-loss-cut (NEW), 3-way full exits ──
    bucketA = (ret945 >= 0) | (ret945 <= -3)     # exit @9:45
    bucketB = ~bucketA                            # -3 < ret945 < 0 -> exit @11am
    pnl_d = np.where(bucketA, sh * (o945 - ep), sh * (o110 - ep))
    ret_d = np.where(bucketA, ret945, ret110)

    # sub-cases of bucket A
    subA_pos  = ret945 >= 0
    subA_deep = ret945 <= -3

    # ── E) Conditional cut-ONLY-deep-losers (NEW), 2-way full exits ──
    #   deep loss (<=-3) -> exit @9:45 ; everything else (winners + mild losses) -> 11am
    E_cut  = ret945 <= -3
    E_hold = ~E_cut
    pnl_e = np.where(E_cut, sh * (o945 - ep), sh * (o110 - ep))
    ret_e = np.where(E_cut, ret945, ret110)

    # ── F) Conditional three-bucket v2 (NEW) ──
    #   A: ret945 <= -3       -> full exit @9:45 (deep losers)
    #   B: ret945 >  0        -> 50/50 split @9:45 + @11am (positive)
    #   C: -3 < ret945 <= 0   -> full exit @11am (mid-negatives; flat/0% falls here)
    F_A = ret945 <= -3
    F_B = ret945 > 0
    F_C = (ret945 > -3) & (ret945 <= 0)
    pnl_fA = sh * (o945 - ep)
    pnl_fB = s1 * (o945 - ep) + s2 * (o110 - ep)     # reuse run_split odd-share split
    pnl_fC = sh * (o110 - ep)
    pnl_f  = np.select([F_A, F_B, F_C], [pnl_fA, pnl_fB, pnl_fC])
    ret_fB = 0.5 * ret945 + 0.5 * ret110
    ret_f  = np.select([F_A, F_B, F_C], [ret945, ret_fB, ret110])

    # ── Comparison table ──
    comp = pd.DataFrame([
        {"strategy": "Unconditional split (9:45+11am)",           **overall_metrics(date, cap, pnl_u, ret_u)},
        {"strategy": "Conditional split (positive->945, rest->11am)", **overall_metrics(date, cap, pnl_c, ret_c)},
        {"strategy": "Conditional deep-loss-cut (>=0 or <=-3 ->945; mid->11am)", **overall_metrics(date, cap, pnl_d, ret_d)},
        {"strategy": "Conditional cut-ONLY-deep-losers (<=-3 ->945; rest->11am)", **overall_metrics(date, cap, pnl_e, ret_e)},
        {"strategy": "Conditional 3-bucket v2 (<=-3->945; >0 split; mid->11am)", **overall_metrics(date, cap, pnl_f, ret_f)},
    ])

    # ── Sub-group breakdown for variant D (3-way) ──
    sub = pd.DataFrame([
        subgroup("Bucket A — exit @9:45 (ret945>=0 OR <=-3)", pnl_d[bucketA], ret_d[bucketA]),
        subgroup("   A1: positive/flat @9:45 (ret945>=0)",    pnl_d[subA_pos & bucketA], ret_d[subA_pos & bucketA]),
        subgroup("   A2: deep loss @9:45 (ret945<=-3)",       pnl_d[subA_deep & bucketA], ret_d[subA_deep & bucketA]),
        subgroup("Bucket B — hold to 11am (-3<ret945<0)",     pnl_d[bucketB], ret_d[bucketB]),
    ])

    # ── Sub-group breakdown for the NEW variant E (cut-only-deep-losers) ──
    sub_e = pd.DataFrame([
        subgroup("Bucket A — exit @9:45 (ret945<=-3, deep loss only)",  pnl_e[E_cut],  ret_e[E_cut]),
        subgroup("Bucket B — hold to 11am (ret945>-3, everything else)", pnl_e[E_hold], ret_e[E_hold]),
    ])

    # ── Sub-group breakdown for the NEW variant F (3-bucket v2) ──
    sub_f = pd.DataFrame([
        subgroup("Bucket A — deep losers, exit @9:45 (ret945<=-3)", pnl_f[F_A], ret_f[F_A]),
        subgroup("Bucket B — positive, 50/50 split (ret945>0)",     pnl_f[F_B], ret_f[F_B]),
        subgroup("Bucket C — mid-neg, hold @11am (-3<ret945<=0)",   pnl_f[F_C], ret_f[F_C]),
    ])

    # ── THRESHOLD SWEEP of the 3-bucket v2 logic ──
    sweep, buckets_tbl, B_once = three_bucket_sweep(
        ep, sh, o945, o110, ret945, ret110, s1, s2, date, cap)
    sweep_sorted = sweep.sort_values(
        "total_ret_fixedbase_pct", ascending=False).reset_index(drop=True)
    sweep_sorted.insert(0, "rank", range(1, len(sweep_sorted) + 1))   # all 8 rows kept, ranked

    comp.to_csv(OUTDIR / "conditional_exit_comparison.csv", index=False)
    sub.to_csv(OUTDIR / "conditional_exit_subgroups.csv", index=False)
    sub_e.to_csv(OUTDIR / "conditional_exit_subgroups_E.csv", index=False)
    sub_f.to_csv(OUTDIR / "conditional_exit_subgroups_F.csv", index=False)
    sweep_sorted.to_csv(OUTDIR / "threshold_sweep_summary.csv", index=False)
    buckets_tbl.to_csv(OUTDIR / "threshold_sweep_buckets.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "conditional_exit_variants.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="Comparison", index=False)
        sub.to_excel(w, sheet_name="Subgroup_breakdown_D", index=False)
        sub_e.to_excel(w, sheet_name="Subgroup_breakdown_E", index=False)
        sub_f.to_excel(w, sheet_name="Subgroup_breakdown_F", index=False)
        sweep_sorted.to_excel(w, sheet_name="Threshold_sweep", index=False)
        buckets_tbl.to_excel(w, sheet_name="Threshold_buckets", index=False)

    # ── Chart: total return vs threshold (both conventions) ──
    xs = sweep["deep_loss_threshold"]
    fig, ax = plt.subplots(figsize=(10, 5))
    l1, = ax.plot(xs, sweep["total_ret_fixedbase_pct"], marker="o", color="#2166ac",
                  label="Total return (fixed ₹5L base)")
    ax2 = ax.twinx()
    l2, = ax2.plot(xs, sweep["total_ret_sumofdaily_pct"], marker="s", color="#b2182b",
                   label="Total return (sum-of-daily)")
    ax.set_xlabel("deep_loss_threshold (%)")
    ax.set_ylabel("Total return % (fixed ₹5L)", color="#2166ac")
    ax2.set_ylabel("Total return % (sum-of-daily)", color="#b2182b")
    ax.invert_xaxis()            # -3 on the left, -10 on the right
    ax.set_xticks(xs)
    ax.set_title("3-bucket v2 — total return vs deep-loss threshold", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(handles=[l1, l2], loc="lower center")
    fig.tight_layout()
    fig.savefig(OUTDIR / "threshold_sweep_totret.png", dpi=130)
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(f"Base universe: {len(b):,} trades (from '2_Split trades'; entries/shares as-is)\n")

    def _print_sub(title, table):
        print("=" * 120)
        print(title)
        print("=" * 120)
        print(f"{'Group':<52}{'n':>7}{'Win%':>9}{'Avg%':>10}{'Median%':>11}{'TotP&L(Rs)':>16}")
        print("-" * 120)
        for _, r in table.iterrows():
            print(f"{r['group']:<52}{r['n']:>7,}{r['win_rate_pct']:>9.2f}"
                  f"{r['avg_return_per_trade_pct']:>10.4f}{r['median_return_per_trade_pct']:>11.4f}"
                  f"{r['total_pnl_rs']:>16,.0f}")

    _print_sub("SUB-GROUP BREAKDOWN — Conditional 3-bucket v2 (variant F, NEW)", sub_f)
    print()
    _print_sub("SUB-GROUP BREAKDOWN — Conditional cut-ONLY-deep-losers (variant E)", sub_e)
    print()
    _print_sub("SUB-GROUP BREAKDOWN — Conditional deep-loss-cut (variant D)", sub)

    print("\n" + "=" * 118)
    print("COMPARISON TABLE — total return shown BOTH ways")
    print("=" * 118)
    print(f"{'Strategy':<50}{'Trades':>8}{'TotRet(5L)%':>13}{'TotRet(daily)%':>16}"
          f"{'Win%':>8}{'Avg%':>9}{'Med%':>9}")
    print("-" * 118)
    for _, r in comp.iterrows():
        print(f"{r['strategy']:<50}{r['n_trades']:>8,}{r['total_ret_fixedbase_pct']:>13.2f}"
              f"{r['total_ret_sumofdaily_pct']:>16.2f}{r['win_rate_pct']:>8.2f}"
              f"{r['avg_return_per_trade_pct']:>9.4f}{r['median_return_per_trade_pct']:>9.4f}")
    print("=" * 118)

    # ── THRESHOLD SWEEP output ──
    print("\n" + "=" * 96)
    print("THRESHOLD SWEEP — 3-bucket v2, deep_loss_threshold in [-3 … -10]  (sorted best-first)")
    print("=" * 96)
    print(f"{'thr%':>5}{'Trades':>8}{'TotRet(5L)%':>13}{'TotRet(daily)%':>16}"
          f"{'Win%':>8}{'Avg%':>9}{'Med%':>9}")
    print("-" * 96)
    for _, r in sweep_sorted.iterrows():
        print(f"{r['deep_loss_threshold']:>5.0f}{r['n_trades']:>8,}"
              f"{r['total_ret_fixedbase_pct']:>13.2f}{r['total_ret_sumofdaily_pct']:>16.2f}"
              f"{r['win_rate_pct']:>8.2f}{r['avg_return_per_trade_pct']:>9.4f}"
              f"{r['median_return_per_trade_pct']:>9.4f}")

    print("\n" + "=" * 118)
    print("BUCKET A (cut @9:45) vs BUCKET C (hold @11am) by threshold  (grid order -3 → -10)")
    print("=" * 118)
    print(f"{'thr%':>5} | {'A_n':>5}{'A_win%':>8}{'A_avg':>9}{'A_med':>9}{'A_totP&L':>13}"
          f"  | {'C_n':>5}{'C_win%':>8}{'C_avg':>9}{'C_med':>9}{'C_totP&L':>13}")
    print("-" * 118)
    for _, r in buckets_tbl.iterrows():
        print(f"{r['threshold']:>5.0f} | {r['A_n']:>5,}{r['A_win%']:>8.2f}{r['A_avg']:>9.4f}"
              f"{r['A_med']:>9.4f}{r['A_totpnl_rs']:>13,.0f}"
              f"  | {r['C_n']:>5,}{r['C_win%']:>8.2f}{r['C_avg']:>9.4f}"
              f"{r['C_med']:>9.4f}{r['C_totpnl_rs']:>13,.0f}")
    print("-" * 118)
    print(f"Bucket B (unchanged, reported once): n={B_once['n']:,}  win={B_once['win_rate_pct']:.2f}%  "
          f"avg={B_once['avg_return_per_trade_pct']:.4f}%  med={B_once['median_return_per_trade_pct']:.4f}%  "
          f"totP&L=₹{B_once['total_pnl_rs']:,.0f}")

    best = sweep_sorted.iloc[0]
    print("\n" + "=" * 96)
    print(f"BEST THRESHOLD: {best['deep_loss_threshold']:.0f}%  →  "
          f"fixedbase {best['total_ret_fixedbase_pct']:.2f}%  |  "
          f"sumofdaily {best['total_ret_sumofdaily_pct']:.2f}%  |  "
          f"win {best['win_rate_pct']:.2f}%  |  avg {best['avg_return_per_trade_pct']:.4f}%  |  "
          f"med {best['median_return_per_trade_pct']:.4f}%")
    row3 = sweep[sweep["deep_loss_threshold"] == -3].iloc[0]
    fF = comp[comp["strategy"].str.startswith("Conditional 3-bucket v2")].iloc[0]
    print(f"SANITY @-3 vs existing variant F: "
          f"sweep {row3['total_ret_fixedbase_pct']:.2f}/{row3['total_ret_sumofdaily_pct']:.2f}  "
          f"vs F {fF['total_ret_fixedbase_pct']:.2f}/{fF['total_ret_sumofdaily_pct']:.2f}  "
          f"→ {'MATCH' if abs(row3['total_ret_fixedbase_pct']-fF['total_ret_fixedbase_pct'])<0.01 else 'MISMATCH'}")
    print("=" * 96)

    # ── FINAL ANSWER: TOP 5 thresholds (ranked by total_ret_pct) ──
    bmap = {r["threshold"]: r for _, r in buckets_tbl.iterrows()}
    top5 = sweep_sorted.head(5)
    print("\n" + "#" * 92)
    print("#  FINAL ANSWER — TOP 5 DEEP-LOSS THRESHOLDS (ranked by total_ret_pct, fixed ₹5L base)")
    print("#" * 92)
    print(f"{'rank':>4}{'thr%':>7}{'n_trades':>10}{'win_rate%':>11}"
          f"{'total_ret%':>12}{'avg_ret%':>10}{'median_ret%':>13}")
    print("-" * 92)
    for _, r in top5.iterrows():
        print(f"{int(r['rank']):>4}{r['deep_loss_threshold']:>7.0f}{r['n_trades']:>10,}"
              f"{r['win_rate_pct']:>11.2f}{r['total_ret_fixedbase_pct']:>12.2f}"
              f"{r['avg_return_per_trade_pct']:>10.4f}{r['median_return_per_trade_pct']:>13.4f}")
    print("-" * 92)
    print("Bucket A (cut deep losers @9:45)  vs  Bucket C (hold mid-negs @11am) — per top-5 threshold:")
    for _, r in top5.iterrows():
        b = bmap[r["deep_loss_threshold"]]
        print(f"  #{int(r['rank'])}  thr {r['deep_loss_threshold']:>3.0f}%  |  "
              f"Bucket A: n={b['A_n']:>4,}  P&L ₹{b['A_totpnl_rs']:>13,.0f}  |  "
              f"Bucket C: n={b['C_n']:>5,}  P&L ₹{b['C_totpnl_rs']:>13,.0f}")
    print("#" * 92)

    print(f"\nSaved → {OUTDIR}  (full 8-row CSV + conditional_exit_variants.xlsx + threshold_sweep_totret.png)")


if __name__ == "__main__":
    main()
