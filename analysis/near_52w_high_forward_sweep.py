# -*- coding: utf-8 -*-
"""
near_52w_high_forward_sweep.py
==============================
Filters the main strategy's trades (mcap ₹1,500-5,000 Cr, lookback 36, volume 6x, 3:15pm
entry, +5% day, ₹5L pool / ₹1L per trade) to stocks within 10% of their 52-WEEK HIGH at
entry, then sweeps forward exit day T+1..T+10 with a conditional split exit across all
intraday (t1,t2) pairs.

Reuses the canonical trade set (fpr.build_trades) — entries/sizing NOT recomputed. Each
trade's forward opens (T+1..T+10 x 23 times) are cached once; every (d,t1,t2) combo slices
the cache. 52w high = max daily HIGH over the trailing 252 trading days incl. entry day.

Exit (per d, per t1<t2 on day T+d): if (open@t1 - entry)/entry > 0 -> exit 100% at t1;
else exit 100% at t2. No 14% target overlay (plain split, per spec).
"""
import sys, bisect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import final_performance_report as fpr

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "near_52w_high_sweep"
BASE_POOL = fpr.BASE_POOL
FWD_DAYS = list(range(1, 21))                          # T+1..T+20
TIMES_HM = list(range(570, 901, 15))                   # 09:30..15:00 (23)
TLABEL = [f"{h//60:02d}:{h%60:02d}" for h in TIMES_HM]
NT = len(TIMES_HM)
WIN_52 = 252
PCT_BELOW_MAX = 10.0                                   # keep if entry within 10% of 52w high


def build_cache(trades):
    """Per-trade 52w-high and forward opens (T+1..T+10 x 23 times). Returns:
       pct_below (n,), fwd (n, 10*23), n_no_history."""
    n = len(trades)
    pct_below = np.full(n, np.nan)
    fwd = np.full((n, len(FWD_DAYS) * NT), np.nan)
    n_no_hist = 0

    for sym, grp in trades.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        dates_sorted = sorted(raw["date"].unique())

        daily_high = raw.groupby("date")["high"].max().reindex(dates_sorted)
        r52 = daily_high.rolling(WIN_52, min_periods=WIN_52).max()   # incl. current day
        r52_map = r52.to_dict()

        sub = raw[raw["hm"].isin(TIMES_HM)]
        po = (sub.pivot_table(index="date", columns="hm", values="open", aggfunc="last")
              .reindex(columns=TIMES_HM))
        po_map = {d: v for d, v in zip(po.index, po.values)}

        for ridx, ed, ep in zip(grp.index, grp["entry_date"], grp["entry_price"]):
            dd = pd.Timestamp(ed).date()
            hi = r52_map.get(dd, np.nan)
            if hi is None or hi != hi:            # no full 252-day history -> can't apply filter
                n_no_hist += 1                    # (pct_below stays NaN; still cache forward opens)
            else:
                pct_below[ridx] = (hi - ep) / hi * 100
            idx0 = bisect.bisect_right(dates_sorted, dd)     # T+1 position
            for k, d in enumerate(FWD_DAYS):
                j = idx0 + d - 1
                if j >= len(dates_sorted):
                    break
                fdate = dates_sorted[j]
                if fdate in po_map:
                    fwd[ridx, k * NT:(k + 1) * NT] = po_map[fdate]
    return pct_below, fwd, n_no_hist


def combo_metrics(entry, shares, cap, fwd, d, i1, i2):
    c1 = (d - 1) * NT + i1
    c2 = (d - 1) * NT + i2
    o1, o2 = fwd[:, c1], fwd[:, c2]
    ret1 = (o1 - entry) / entry * 100
    positive = ret1 > 0                        # NaN -> False -> falls to t2
    exit_px = np.where(positive, o1, o2)
    valid = ~np.isnan(exit_px)
    nv = int(valid.sum())
    if nv == 0:
        return None
    e, sh, cp, xp = entry[valid], shares[valid], cap[valid], exit_px[valid]
    pnl = sh * (xp - e)
    ret = (xp - e) / e * 100
    return {
        "forward_day": d, "exit_time_1": TLABEL[i1], "exit_time_2": TLABEL[i2],
        "n_trades": nv,
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
        "avg_return_per_trade_pct": round(float(ret.mean()), 4),
        "median_return_per_trade_pct": round(float(np.median(ret)), 4),
        "total_return_fixedbase_pct": round(float(pnl.sum()) / BASE_POOL * 100, 4),
        "total_pnl_inr": round(float(pnl.sum()), 0),
        "avg_capital_deployed_per_trade": round(float(cp.mean()), 0),
    }


def sweep(entry, shares, cap, fwd):
    rows = []
    for d in FWD_DAYS:
        for i1 in range(NT):
            for i2 in range(i1 + 1, NT):
                m = combo_metrics(entry, shares, cap, fwd, d, i1, i2)
                if m:
                    rows.append(m)
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    T = fpr.build_trades().reset_index(drop=True)
    entry = T["entry_price"].values.astype(float)
    shares = T["shares"].values.astype(float)
    cap = T["capital_deployed"].values.astype(float)

    print("Caching 52w-high + forward opens (once) …")
    pct_below, fwd, n_no_hist = build_cache(T)

    has_hist = ~np.isnan(pct_below)
    keep = has_hist & (pct_below <= PCT_BELOW_MAX)
    print(f"  total trades: {len(T):,}")
    print(f"  dropped (no full 252-day history): {n_no_hist:,}")
    print(f"  with history: {int(has_hist.sum()):,} | within 10% of 52w high: {int(keep.sum()):,} "
          f"({int(keep.sum())/len(T)*100:.1f}% of total)")

    ki = np.where(keep)[0]
    e_f, sh_f, cp_f, fwd_f = entry[ki], shares[ki], cap[ki], fwd[ki]

    # ── forward-data availability per day (filtered subset): trades with a usable
    #    open on day T+d (else the day ran past dataset end / missing candles) ──
    n_base = len(ki)
    avail_rows = []
    for k, d in enumerate(FWD_DAYS):
        blk = fwd_f[:, k * NT:(k + 1) * NT]
        have = int((~np.isnan(blk)).any(axis=1).sum())
        avail_rows.append({"forward_day": d, "n_trades_with_data": have,
                           "n_dropped_vs_Tplus1": n_base - have,
                           "pct_available": round(have / n_base * 100, 2)})
    avail = pd.DataFrame(avail_rows)
    avail.to_csv(OUTDIR / "forward_data_availability.csv", index=False)
    mat_drop = avail[avail["pct_available"] < 95]

    # ── full sweep on the filtered (near-52w-high) subset ──
    print(f"Sweeping {len(FWD_DAYS)*253:,} combos on filtered subset …")
    res = sweep(e_f, sh_f, cp_f, fwd_f).sort_values(
        "total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    res.insert(0, "rank", range(1, len(res) + 1))
    res.to_csv(OUTDIR / "near_52w_high_forward_sweep.csv", index=False)

    # ── per-forward-day best ──
    best_by_day = (res.loc[res.groupby("forward_day")["total_return_fixedbase_pct"].idxmax()]
                   .sort_values("forward_day").reset_index(drop=True))

    # ── comparison: overall-best combo re-computed on the UNFILTERED set ──
    br = res.iloc[0]
    bd, bi1, bi2 = int(br["forward_day"]), TLABEL.index(br["exit_time_1"]), TLABEL.index(br["exit_time_2"])
    unf = combo_metrics(entry, shares, cap, fwd, bd, bi1, bi2)
    cmp_tbl = pd.DataFrame([
        {"set": "near_52w_high (filtered)", **{k: br[k] for k in
         ["forward_day", "exit_time_1", "exit_time_2", "n_trades", "win_rate_pct",
          "avg_return_per_trade_pct", "median_return_per_trade_pct",
          "total_return_fixedbase_pct", "total_pnl_inr", "avg_capital_deployed_per_trade"]}},
        {"set": "all_trades (unfiltered)", **unf},
    ])

    with pd.ExcelWriter(OUTDIR / "near_52w_high_forward_sweep.xlsx", engine="openpyxl") as w:
        res.to_excel(w, sheet_name="all_combos", index=False)
        best_by_day.to_excel(w, sheet_name="best_per_forward_day", index=False)
        res.head(20).to_excel(w, sheet_name="top_20", index=False)
        cmp_tbl.to_excel(w, sheet_name="filter_vs_unfiltered", index=False)
        avail.to_excel(w, sheet_name="forward_data_availability", index=False)

    # ── heatmaps for the top-3 best-performing forward days ──
    pos = {lab: k for k, lab in enumerate(TLABEL)}
    top_days = (best_by_day.sort_values("total_return_fixedbase_pct", ascending=False)
                ["forward_day"].head(3).astype(int).tolist())
    best_d = top_days[0]
    for dd in top_days:
        sub = res[res["forward_day"] == dd]
        grid = np.full((NT, NT), np.nan)
        for _, r in sub.iterrows():
            grid[pos[r["exit_time_1"]], pos[r["exit_time_2"]]] = r["total_return_fixedbase_pct"]
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_xticks(range(NT)); ax.set_xticklabels(TLABEL, rotation=90, fontsize=7)
        ax.set_yticks(range(NT)); ax.set_yticklabels(TLABEL, fontsize=7)
        ax.set_xlabel("t2 (second exit)"); ax.set_ylabel("t1 (first exit)")
        ax.set_title(f"Near-52w-high conditional split — total_return_fixedbase_pct, T+{dd} "
                     f"(t1<t2 upper triangle)", fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="total return %")
        br2, bc2 = np.unravel_index(np.nanargmax(grid), grid.shape)
        ax.plot(bc2, br2, "r*", markersize=18, markeredgecolor="white")
        fig.tight_layout(); fig.savefig(OUTDIR / f"heatmap_T+{dd}.png", dpi=130); plt.close(fig)

    # ── decay chart: best-achievable total return by forward day ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(best_by_day["forward_day"], best_by_day["total_return_fixedbase_pct"],
            "o-", color="#1f4e79", lw=2)
    for _, r in best_by_day.iterrows():
        ax.annotate(f"{r['exit_time_1']}/{r['exit_time_2']}",
                    (r["forward_day"], r["total_return_fixedbase_pct"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("forward day (T+d)"); ax.set_ylabel("best total_return_fixedbase_pct")
    ax.set_title("Momentum decay — best-achievable total return by forward day\n"
                 "(near-52w-high subset; label = best t1/t2)", fontweight="bold")
    ax.set_xticks(FWD_DAYS); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "best_total_return_by_forward_day.png", dpi=130); plt.close(fig)

    # ── prints ──
    pd.set_option("display.width", 230)
    cols = ["rank", "forward_day", "exit_time_1", "exit_time_2", "n_trades", "win_rate_pct",
            "avg_return_per_trade_pct", "median_return_per_trade_pct",
            "total_return_fixedbase_pct", "total_pnl_inr", "avg_capital_deployed_per_trade"]
    print("\n" + "=" * 130)
    print("FORWARD-DATA AVAILABILITY  (filtered subset, n=%d at T+1)" % n_base)
    print("=" * 130)
    print(avail.to_string(index=False))
    if len(mat_drop):
        print(f"  MATERIAL DROP (<95% available) at: "
              + ", ".join(f"T+{int(r.forward_day)} ({r.pct_available}%)" for _, r in mat_drop.iterrows()))
    else:
        print("  No forward day loses >5% of trades — n_trades decline is negligible.")

    print("\n" + "=" * 130)
    print("BEST (t1,t2) PER FORWARD DAY  (near-52w-high subset)")
    print("=" * 130)
    print(best_by_day[cols[1:]].to_string(index=False))
    print("\n" + "=" * 130)
    print("TOP 20 COMBINATIONS OVERALL")
    print("=" * 130)
    print(res[cols].head(20).to_string(index=False))
    print("\n" + "=" * 130)
    print("FILTER vs UNFILTERED — overall-best combo on both sets")
    print("=" * 130)
    print(cmp_tbl.to_string(index=False))
    print(f"\nBest forward day = T+{best_d}  |  Saved -> {OUTDIR}")


if __name__ == "__main__":
    main()
