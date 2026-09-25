# -*- coding: utf-8 -*-
"""
exit_time_sweep.py
==================
Sweeps exit-timing variations on the EXISTING fixed 3:15pm-entry positions.
Entries/shares/capital are taken AS-IS from results/backtest_results.xlsx
(sheet "1_Standard trades") — entries are never regenerated. Only the exit
price(s) vary.

Metrics per scenario row:
  win_rate_pct              = % of trades with pnl > 0
  avg_return_per_trade_pct  = mean of per-trade return %
  median_return_per_trade_pct = median of per-trade return %
  total_return_pct          = Σ(trade pnl, ₹) / CAPITAL_BASE × 100
                              (FIXED ₹5,00,000 denominator — a constant, NOT the
                               capital actually deployed)
  n_trades                  = trade count

Per-trade math (share-weighted pnl; split rounding identical to run_split):
  full  : pnl = shares × (exit − entry);   ret% = (exit − entry)/entry × 100
  split : s1 = shares//2 (t1), s2 = shares − s1 (t2, odd share → later leg)
          pnl = s1×(e1 − entry) + s2×(e2 − entry)
          ret% = 0.5×(e1−entry)/entry×100 + 0.5×(e2−entry)/entry×100

Scenario 1 — full exit at each of 23 next-day times (23 rows, ranked).
Scenario 2 — two-leg 50/50 split for all 253 (t1<t2) pairs (ranked).

Each stock's next-day candle opens are fetched ONCE per trade and cached, then
sliced per scenario/combo.
"""

import sys
import bisect
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb   # reuse CAPITAL_BASE, MASTER_DIR, RESULTS, IST

CAPITAL_BASE = rb.CAPITAL_BASE          # ₹5,00,000 fixed denominator (== run_backtest.DAILY_POOL)
XLSX   = rb.RESULTS / "backtest_results.xlsx"
OUTDIR = rb.RESULTS / "exit_time_sweep"
IST    = rb.IST

# 23 next-day exit times: 09:30 … 15:00
TIMES_HM = list(range(570, 901, 15))
LABELS   = [f"{hm // 60:02d}:{hm % 60:02d}" for hm in TIMES_HM]
NT       = len(TIMES_HM)


def load_base_positions():
    """Fixed 3:15pm entries as-is from the existing standard trade sheet."""
    base = pd.read_excel(XLSX, sheet_name="1_Standard trades")
    base = base[["date", "symbol", "entry", "shares", "cap"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    return base.reset_index(drop=True)


def fetch_nextday_opens(base):
    """
    For each base position, fetch the next trading day's candle opens at the 23
    grid times. Reads each symbol's parquet ONCE. Returns an (N × 23) matrix.
    """
    opens = np.full((len(base), NT), np.nan, dtype=float)
    for sym, grp in base.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"]   = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute

        dates_sorted = sorted(raw["date"].unique())
        sub  = raw[raw["hm"].isin(TIMES_HM)]
        piv  = (sub.pivot_table(index="date", columns="hm", values="open", aggfunc="last")
                   .reindex(columns=TIMES_HM))

        for pos_idx, ed in zip(grp.index, grp["date"]):
            j = bisect.bisect_right(dates_sorted, ed.date())   # first trading day AFTER entry
            if j >= len(dates_sorted):
                continue
            nd = dates_sorted[j]
            if nd in piv.index:
                opens[pos_idx, :] = piv.loc[nd].values
    return opens


def _metrics(date_arr, cap_arr, pnl_vec, ret_vec, mask):
    """
    The metrics on the valid subset, with BOTH total-return conventions:
      total_return_fixedbase_pct   = Σpnl / CAPITAL_BASE × 100        (fixed ₹5L)
      total_return_sumofdaily_pct  = Σ(daily Σpnl/Σcap × 100)          (compute_stats)
    """
    if not mask.any():
        return dict(n_trades=0, win_rate_pct=0.0, avg_return_per_trade_pct=0.0,
                    median_return_per_trade_pct=0.0,
                    total_return_fixedbase_pct=0.0, total_return_sumofdaily_pct=0.0)
    pnl = pnl_vec[mask]
    ret = ret_vec[mask]
    # sum-of-daily total via the existing pipeline function
    sub = pd.DataFrame({"date": date_arr[mask], "cap": cap_arr[mask],
                        "pnl": pnl, "ret": ret})
    sumofdaily = rb.compute_stats(sub)["total_ret_pct"]
    return dict(
        n_trades                    = int(mask.sum()),
        win_rate_pct                = round(float((pnl > 0).mean() * 100), 2),
        avg_return_per_trade_pct    = round(float(ret.mean()), 4),
        median_return_per_trade_pct = round(float(np.median(ret)), 4),
        total_return_fixedbase_pct  = round(float(pnl.sum() / CAPITAL_BASE * 100), 4),
        total_return_sumofdaily_pct = round(float(sumofdaily), 4),
    )


def scenario1(base, opens):
    date_arr = base["date"].values
    cap_arr  = base["cap"].values.astype(float)
    entry    = base["entry"].values.astype(float)
    shares   = base["shares"].values.astype(float)
    rows = []
    for j, lab in enumerate(LABELS):
        e    = opens[:, j]
        mask = ~np.isnan(e)
        pnl  = shares * (e - entry)
        ret  = (e - entry) / entry * 100
        rows.append({"exit_time": lab, **_metrics(date_arr, cap_arr, pnl, ret, mask)})
    df = pd.DataFrame(rows).sort_values(
        "total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    return df


def scenario2(base, opens):
    date_arr = base["date"].values
    cap_arr  = base["cap"].values.astype(float)
    entry    = base["entry"].values.astype(float)
    shares   = base["shares"].values.astype(float)
    s1 = np.floor(shares / 2)      # earlier leg t1 (floor)  — matches run_split
    s2 = shares - s1               # later leg t2 gets the odd share
    rows = []
    for i, j in combinations(range(NT), 2):     # i < j → 253 pairs
        e1, e2 = opens[:, i], opens[:, j]
        mask   = ~np.isnan(e1) & ~np.isnan(e2)
        pnl    = s1 * (e1 - entry) + s2 * (e2 - entry)
        ret    = 0.5 * (e1 - entry) / entry * 100 + 0.5 * (e2 - entry) / entry * 100
        rows.append({"exit_time_1": LABELS[i], "exit_time_2": LABELS[j],
                     **_metrics(date_arr, cap_arr, pnl, ret, mask)})
    df = pd.DataFrame(rows).sort_values(
        "total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    return df


# t1 grid for the conditional scenario: 09:30 … 12:00 only (indices 0..10)
COND_T1_MAX_IDX = 10   # TIMES_HM[10] == 720 == 12:00


def scenario_conditional_best_t2(base, opens):
    """
    conditional_split_best_t2 — for each t1 (09:30..12:00):
      Bucket P (ret@t1 > 0): exit 100% at t1.
      Bucket N (ret@t1 <= 0): exit 100% at the single t2 (t2>t1, full 09:30..15:00
        grid) that MAXIMISES the whole scenario's total_return_fixedbase_pct.
        Since Bucket P's pnl is fixed once t1 is chosen, this == maximising Bucket
        N's Σpnl. Ties broken to the EARLIEST t2 (strict-greater replacement).
    Metrics reuse _metrics (both total-return conventions). One row per t1.
    """
    date_arr = base["date"].values
    cap_arr  = base["cap"].values.astype(float)
    entry    = base["entry"].values.astype(float)
    shares   = base["shares"].values.astype(float)

    rows = []
    for i in range(COND_T1_MAX_IDX + 1):          # t1 = 09:30 … 12:00
        e1     = opens[:, i]
        ret_t1 = (e1 - entry) / entry * 100
        valid1 = ~np.isnan(e1)
        P = valid1 & (ret_t1 > 0)                 # positive at t1 → exit @t1
        N = valid1 & (ret_t1 <= 0)                # non-positive at t1 → exit @best t2

        # search best t2 (> t1) over the FULL grid, by Bucket N's Σpnl
        best_j, best_sumN = None, None
        for j in range(i + 1, NT):
            e2 = opens[:, j]
            Nv = N & ~np.isnan(e2)
            sumN = float(np.sum(shares[Nv] * (e2[Nv] - entry[Nv])))
            if best_sumN is None or sumN > best_sumN:   # strict > → earliest wins ties
                best_sumN, best_j = sumN, j

        # build the combined scenario at the winning t2
        e2 = opens[:, best_j]
        Nv = N & ~np.isnan(e2)
        mask = P | Nv
        pnl = np.full(len(base), np.nan)
        ret = np.full(len(base), np.nan)
        pnl[P]  = shares[P] * (e1[P] - entry[P])
        ret[P]  = ret_t1[P]
        pnl[Nv] = shares[Nv] * (e2[Nv] - entry[Nv])
        ret[Nv] = (e2[Nv] - entry[Nv]) / entry[Nv] * 100

        rows.append({"scenario": "conditional_split_best_t2",
                     "exit_time_1": LABELS[i], "exit_time_2": LABELS[best_j],
                     **_metrics(date_arr, cap_arr, pnl, ret, mask)})
    return pd.DataFrame(rows)


def chart_scenario1(df, out_png):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    colors = ["#2166ac" if v >= 0 else "#b2182b" for v in df["total_return_fixedbase_pct"]]
    ax.bar(x, df["total_return_fixedbase_pct"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(df["exit_time"], rotation=90, fontsize=8)
    ax.set_xlabel("Next-day full-exit time (ranked best → worst)")
    ax.set_ylabel("Total return %  (Σ pnl / ₹5,00,000)")
    ax.set_title("Scenario 1 — Full Exit Sweep (fixed 3:15pm entry, fixed ₹5L base)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def heatmap_scenario2(df, out_png):
    M = np.full((NT, NT), np.nan)
    idx = {lab: k for k, lab in enumerate(LABELS)}
    for _, r in df.iterrows():
        M[idx[r["exit_time_1"]], idx[r["exit_time_2"]]] = r["total_return_fixedbase_pct"]

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("#e9e9e9")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, aspect="equal")
    ax.set_xticks(range(NT)); ax.set_xticklabels(LABELS, rotation=90, fontsize=7)
    ax.set_yticks(range(NT)); ax.set_yticklabels(LABELS, fontsize=7)
    ax.set_xlabel("t2 (later exit)"); ax.set_ylabel("t1 (earlier exit)")
    ax.set_title("Scenario 2 — Two-Leg Split Sweep: total return %  (Σ pnl / ₹5L)\n"
                 "(upper triangle, t1 < t2; fixed 3:15pm entry)",
                 fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Total return %")
    br, bc = np.unravel_index(np.nanargmax(M), M.shape)
    ax.scatter([bc], [br], marker="*", s=200, c="black")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"CAPITAL_BASE = ₹{CAPITAL_BASE:,} (fixed denominator)")
    print("Loading base positions (fixed 3:15pm entries, as-is) …")
    base = load_base_positions()
    print(f"  base positions: {len(base):,}")

    print("Fetching next-day opens (once per trade, cached) …")
    opens = fetch_nextday_opens(base)
    print(f"  positions with ≥1 next-day candle: {int((~np.isnan(opens)).any(axis=1).sum()):,}")

    print("\nScenario 1 — full exit sweep (23 times) …")
    s1 = scenario1(base, opens)
    s1.to_csv(OUTDIR / "scenario1_full_exit.csv", index=False)
    chart_scenario1(s1, OUTDIR / "scenario1_full_exit.png")

    print("Scenario 2 — two-leg split sweep (253 pairs) …")
    s2 = scenario2(base, opens)
    s2.to_csv(OUTDIR / "scenario2_split_pairs.csv", index=False)
    heatmap_scenario2(s2, OUTDIR / "scenario2_split_heatmap.png")

    print("Scenario 3 — conditional_split_best_t2 (11 t1 values, best t2 per t1) …")
    s3 = scenario_conditional_best_t2(base, opens)
    s3.to_csv(OUTDIR / "scenario3_conditional_best_t2.csv", index=False)

    # ── Combine BOTH scenarios into ONE ranked table ──
    metric_cols = ["n_trades", "win_rate_pct",
                   "avg_return_per_trade_pct", "median_return_per_trade_pct",
                   "total_return_fixedbase_pct", "total_return_sumofdaily_pct"]
    a = s1.rename(columns={"exit_time": "exit_time_1"}).copy()
    a.insert(0, "scenario", "full_exit")
    a["exit_time_2"] = ""                      # blank second leg for full exits
    b = s2.copy()
    b.insert(0, "scenario", "split")
    c = s3.copy()                              # already has scenario/exit_time_1/exit_time_2 + metrics
    combined_cols = ["scenario", "exit_time_1", "exit_time_2"] + metric_cols
    combined = (pd.concat([a[combined_cols], b[combined_cols], c[combined_cols]], ignore_index=True)
                  .sort_values("total_return_fixedbase_pct", ascending=False)
                  .reset_index(drop=True))
    combined.insert(0, "rank", range(1, len(combined) + 1))

    combined.to_csv(OUTDIR / "combined_all.csv", index=False)

    xlsx_out = OUTDIR / "exit_time_sweep_combined.xlsx"
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="Combined_all", index=False)
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                sheet.column_dimensions[col[0].column_letter].width = min(width + 2, 30)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 118)
    print(f"COMBINED ranked table — all scenarios ({len(combined)} rows) — TOP 15")
    print("=" * 118)
    print(combined.head(15).to_string(index=False))
    print("\n  conditional_split_best_t2 rows (all 11, best t2 per t1):")
    print(combined[combined["scenario"] == "conditional_split_best_t2"].to_string(index=False))

    # Sanity anchor (15:00 full exit)
    anc = s1[s1["exit_time"] == "15:00"].iloc[0]
    print("\n  SANITY @15:00 full exit: "
          f"fixedbase {anc['total_return_fixedbase_pct']:+.2f}%  (expect 393.08)  |  "
          f"sumofdaily {anc['total_return_sumofdaily_pct']:+.2f}%  (expect 683.94)")
    print("=" * 112)
    print(f"Saved combined Excel (1 sheet) → {xlsx_out}")
    print(f"Saved combined CSV + per-scenario CSVs + charts → {OUTDIR}")


if __name__ == "__main__":
    main()
