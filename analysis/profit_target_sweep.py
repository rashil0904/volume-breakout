# -*- coding: utf-8 -*-
"""
profit_target_sweep.py
======================
Profit-target (intraday limit-fill) overlays on the existing fixed-3:15pm-entry
trade set, for three base scenarios, swept over target_pct in [2..20].

Reuses (does NOT reimplement):
  - entries/shares/cap           -> exit_time_sweep.load_base_positions() (1_Standard trades)
  - both total-return metrics     -> exit_time_sweep._metrics()  (fixed-₹5L + sum-of-daily
                                     via run_backtest.compute_stats)
  - next-day candle fetch pattern -> extended here to fetch HIGHS + OPENS incl. 09:15

TARGET RULE (shared): at a 15-min candle, pct_from_entry = (HIGH - entry)/entry*100.
Target triggers the FIRST candle in the applicable window where pct_from_entry >= X.
On trigger: exit_price = entry*(1 + X/100) (limit fill, NOT the actual high);
so a target-hit trade's return is exactly X%.

Scenario A — full_exit@11:00 (2 exit_types):
  Phase 1: candles 09:15 .. before 11:00. Trigger -> "target_hit".
  Else: exit at 11:00 open -> "time_exit".

Scenarios B & C — conditional_split_best_t2 (t1,t2) (4 exit_types):
  Phase 1 (09:15 .. before t1): trigger -> "early_target_pre_t1".
  Phase 2 (at t1, survivors): ret_at_t1>0 -> exit @t1 open "positive_at_t1"; else continue.
  Phase 3 (after t1 .. before t2, non-positive survivors): trigger -> "early_target_between_t1_t2".
  Phase 4 (at t2, remainder): exit @t2 open "exit_at_t2_no_target".

  B: t1=09:45, t2=12:00   |   C: t1=09:30, t2=10:30
"""

import sys
import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))          # analysis/  (exit_time_sweep)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (run_backtest)
import exit_time_sweep as ets
import run_backtest as rb

CB       = ets.CAPITAL_BASE                       # ₹500,000 fixed base
IST      = ets.IST
OUTDIR   = rb.RESULTS / "target_pct_sweep"
CANDLE_HMS = list(range(555, 901, 15))            # 09:15 .. 15:00 (24 candles)
HCOL     = {hm: k for k, hm in enumerate(CANDLE_HMS)}
TARGETS  = list(range(2, 21))                     # 2 .. 20, 1% steps

HM_1100, HM_0945, HM_1200, HM_0930, HM_1030 = 660, 585, 720, 570, 630


def hm_label(hm):
    return f"{hm // 60:02d}:{hm % 60:02d}"


def fetch_nextday_ohlc(base):
    """Next-day 15-min OPENS and HIGHS at CANDLE_HMS. One parquet read per symbol."""
    n = len(base)
    opens = np.full((n, len(CANDLE_HMS)), np.nan)
    highs = np.full((n, len(CANDLE_HMS)), np.nan)
    for sym, grp in base.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"]   = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        dates_sorted = sorted(raw["date"].unique())
        sub = raw[raw["hm"].isin(CANDLE_HMS)]
        po = sub.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=CANDLE_HMS)
        ph = sub.pivot_table(index="date", columns="hm", values="high", aggfunc="last").reindex(columns=CANDLE_HMS)
        for pos_idx, ed in zip(grp.index, grp["date"]):
            j = bisect.bisect_right(dates_sorted, ed.date())
            if j >= len(dates_sorted):
                continue
            nd = dates_sorted[j]
            if nd in po.index:
                opens[pos_idx, :] = po.loc[nd].values
                highs[pos_idx, :] = ph.loc[nd].values
    return opens, highs


def _window_max(pct_high, cols):
    """Per-trade max pct_from_entry over the given candle columns (NaN-safe)."""
    if not cols:
        return np.full(pct_high.shape[0], -np.inf)
    sub = pct_high[:, cols]
    return np.where(np.isnan(sub), -np.inf, sub).max(axis=1)


def run_A(X, entry, shares, o1100, p1max):
    n = len(entry)
    pnl = np.full(n, np.nan); ret = np.full(n, np.nan); et = np.full(n, "", dtype=object)
    trig = p1max >= X
    pnl[trig] = shares[trig] * entry[trig] * X / 100; ret[trig] = X; et[trig] = "target_hit"
    nt = ~trig
    v = nt & ~np.isnan(o1100)
    pnl[v] = shares[v] * (o1100[v] - entry[v])
    ret[v] = (o1100[v] - entry[v]) / entry[v] * 100
    et[v] = "time_exit"
    return pnl, ret, et, (trig | v)


def run_cond(X, entry, shares, ot1, ot2, ret_t1, p1max, p3max):
    n = len(entry)
    pnl = np.full(n, np.nan); ret = np.full(n, np.nan); et = np.full(n, "", dtype=object)

    M1 = p1max >= X                                          # early target pre-t1
    pnl[M1] = shares[M1] * entry[M1] * X / 100; ret[M1] = X; et[M1] = "early_target_pre_t1"

    surv = ~M1
    vt1  = ~np.isnan(ot1)
    M2 = surv & vt1 & (ret_t1 > 0)                           # positive at t1 -> exit @t1
    pnl[M2] = shares[M2] * (ot1[M2] - entry[M2]); ret[M2] = ret_t1[M2]; et[M2] = "positive_at_t1"

    rest = surv & ~M2                                        # non-positive survivors
    M3 = rest & (p3max >= X)                                 # early target between t1,t2
    pnl[M3] = shares[M3] * entry[M3] * X / 100; ret[M3] = X; et[M3] = "early_target_between_t1_t2"

    M4 = rest & ~M3
    vt2 = ~np.isnan(ot2)
    M4v = M4 & vt2                                           # exit at t2
    pnl[M4v] = shares[M4v] * (ot2[M4v] - entry[M4v])
    ret[M4v] = (ot2[M4v] - entry[M4v]) / entry[M4v] * 100
    et[M4v] = "exit_at_t2_no_target"

    return pnl, ret, et, (M1 | M2 | M3 | M4v)


def etype_breakdown(ret, et, mask, categories):
    rvalid = ret[mask]; evalid = et[mask]
    total = len(rvalid)
    rows = []
    for cat in categories:
        sel = evalid == cat
        n = int(sel.sum())
        if n == 0:
            rows.append({"exit_type": cat, "count": 0, "pct": 0.0,
                         "avg_return_pct": np.nan, "median_return_pct": np.nan})
        else:
            rows.append({"exit_type": cat, "count": n, "pct": round(n / total * 100, 2),
                         "avg_return_pct": round(float(rvalid[sel].mean()), 4),
                         "median_return_pct": round(float(np.median(rvalid[sel])), 4)})
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Loading base positions (fixed 3:15pm entries, as-is) …")
    base = ets.load_base_positions()
    print(f"  base positions: {len(base):,}")
    print("Fetching next-day OHLC (once, shared across all 57 combos) …")
    opens, highs = fetch_nextday_ohlc(base)

    entry  = base["entry"].values.astype(float)
    shares = base["shares"].values.astype(float)
    date   = base["date"].values
    cap    = base["cap"].values.astype(float)
    pct_high = (highs - entry[:, None]) / entry[:, None] * 100
    o1100 = opens[:, HCOL[HM_1100]]

    scenarios = [
        {"name": "full_exit@11:00", "kind": "A",
         "cats": ["target_hit", "time_exit"]},
        {"name": "cond_best_t2 t1=09:45 t2=12:00", "kind": "cond", "t1": HM_0945, "t2": HM_1200,
         "cats": ["early_target_pre_t1", "positive_at_t1",
                  "early_target_between_t1_t2", "exit_at_t2_no_target"]},
        {"name": "cond_best_t2 t1=09:30 t2=10:30", "kind": "cond", "t1": HM_0930, "t2": HM_1030,
         "cats": ["early_target_pre_t1", "positive_at_t1",
                  "early_target_between_t1_t2", "exit_at_t2_no_target"]},
    ]
    for s in scenarios:
        if s["kind"] == "A":
            s["p1max"] = _window_max(pct_high, [HCOL[hm] for hm in CANDLE_HMS if hm < HM_1100])
        else:
            t1, t2 = s["t1"], s["t2"]
            s["ot1"] = opens[:, HCOL[t1]]
            s["ot2"] = opens[:, HCOL[t2]]
            s["ret_t1"] = (s["ot1"] - entry) / entry * 100
            s["p1max"] = _window_max(pct_high, [HCOL[hm] for hm in CANDLE_HMS if hm < t1])
            s["p3max"] = _window_max(pct_high, [HCOL[hm] for hm in CANDLE_HMS if t1 < hm < t2])

    # ── Sweep 3 scenarios × 19 targets ──
    print("Running 57 combinations …")
    rows, store = [], {}
    for s in scenarios:
        for X in TARGETS:
            if s["kind"] == "A":
                pnl, ret, et, mask = run_A(X, entry, shares, o1100, s["p1max"])
            else:
                pnl, ret, et, mask = run_cond(X, entry, shares, s["ot1"], s["ot2"],
                                              s["ret_t1"], s["p1max"], s["p3max"])
            rows.append({"scenario": s["name"], "target_pct": X,
                         **ets._metrics(date, cap, pnl, ret, mask)})
            store[(s["name"], X)] = (ret, et, mask)

    summary = pd.DataFrame(rows).sort_values(
        "total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))

    cols = ["rank", "scenario", "target_pct", "n_trades", "win_rate_pct",
            "avg_return_per_trade_pct", "median_return_per_trade_pct",
            "total_return_fixedbase_pct", "total_return_sumofdaily_pct"]
    summary = summary[cols]
    top10 = summary.head(10)

    # ── Excel: all 57 + top-10 ──
    xlsx = OUTDIR / "target_pct_sweep_all_combos.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="All_57", index=False)
        top10.to_excel(w, sheet_name="Top_10", index=False)
        for sh in w.sheets.values():
            for col in sh.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                sh.column_dimensions[col[0].column_letter].width = min(width + 2, 34)
    summary.to_csv(OUTDIR / "target_pct_sweep_all_combos.csv", index=False)

    # ── Per-scenario line charts ──
    for s in scenarios:
        sub = summary[summary["scenario"] == s["name"]].sort_values("target_pct")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(sub["target_pct"], sub["total_return_fixedbase_pct"], marker="o", color="#2166ac")
        ax.set_xlabel("target_pct (%)"); ax.set_ylabel("Total return % (fixed ₹5L)")
        ax.set_title(f"Profit-target sweep — {s['name']}", fontweight="bold")
        ax.set_xticks(TARGETS); ax.grid(alpha=0.3)
        slug = (s["name"].replace("@", "_").replace(":", "").replace(" ", "_")
                .replace("=", "").replace("t1", "t1").replace("t2", "t2"))
        fig.tight_layout(); fig.savefig(OUTDIR / f"target_sweep_{slug}.png", dpi=130); plt.close(fig)

    # ── Console: pooled top-10 ──
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("TOP 10 (pooled across all 57 combinations, by total_return_fixedbase_pct)")
    print("=" * 100)
    show = top10[["rank", "scenario", "target_pct", "total_return_fixedbase_pct",
                  "win_rate_pct", "avg_return_per_trade_pct", "median_return_per_trade_pct"]]
    print(show.to_string(index=False))

    # ── Console: exit_type breakdown at each scenario's best target ──
    print("\n" + "=" * 100)
    print("EXIT_TYPE BREAKDOWN — each scenario at its BEST target_pct (by fixed-base)")
    print("=" * 100)
    for s in scenarios:
        srows = summary[summary["scenario"] == s["name"]]
        best = srows.loc[srows["total_return_fixedbase_pct"].idxmax()]
        bX = int(best["target_pct"])
        ret, et, mask = store[(s["name"], bX)]
        bd = etype_breakdown(ret, et, mask, s["cats"])
        print(f"\n  {s['name']}  |  best target = {bX}%  |  "
              f"total {best['total_return_fixedbase_pct']:.2f}% (5L) / "
              f"{best['total_return_sumofdaily_pct']:.2f}% (daily)  |  win {best['win_rate_pct']:.2f}%")
        print(f"    {'exit_type':<30}{'count':>7}{'pct%':>8}{'avg_ret%':>11}{'median_ret%':>13}")
        for _, r in bd.iterrows():
            av = f"{r['avg_return_pct']:.4f}" if pd.notna(r['avg_return_pct']) else "—"
            md = f"{r['median_return_pct']:.4f}" if pd.notna(r['median_return_pct']) else "—"
            print(f"    {r['exit_type']:<30}{int(r['count']):>7,}{r['pct']:>8.2f}{av:>11}{md:>13}")

    print("\n" + "=" * 100)
    ovr = summary.iloc[0]
    print(f"OVERALL BEST: {ovr['scenario']}  @ target {int(ovr['target_pct'])}%  →  "
          f"{ovr['total_return_fixedbase_pct']:.2f}% (5L) / {ovr['total_return_sumofdaily_pct']:.2f}% (daily)")
    print(f"Saved → {xlsx}  (+ CSV + 3 charts in {OUTDIR})")


if __name__ == "__main__":
    main()
