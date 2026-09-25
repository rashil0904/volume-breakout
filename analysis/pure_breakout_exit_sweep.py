# -*- coding: utf-8 -*-
"""
pure_breakout_exit_sweep.py  ("_pure_breakout")
===============================================
A PURE 52-week-high breakout strategy — NO volume filter, NO lookback, NO profit target.
Entry (all, and only): mcap ₹1,500-5,000 Cr + NEW 52-week high on entry day + daily
return >= +3% (vs prev close) + 3:15pm entry (15:15 open). ₹5L pool / ₹1L per trade.

52w high = max daily HIGH over the 252 trading days ending the day BEFORE entry (excl.
entry day). Trades lacking full 252-day history are dropped and counted.

Exit sweep (plain time-based, next trading day, NO target): 23 full-exits + 253 conditional
splits (positive-at-t1 -> t1 open; else -> t2 open) = 276 combos. Trade set built once;
each trade's next-day 15-min opens cached once; every combo slices the cache.
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

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "pure_breakout"
BASE_POOL, MAX_PER_STOCK, DAILY_POOL = 500_000, 100_000, 500_000
EXP023, EXP038 = 0.0023, 0.0038          # expense as % of capital deployed, per trade
WIN_52, RET_MIN, SMALL = 252, 3.0, 30
TIMES_HM = list(range(570, 901, 15))                # 09:30..15:00 (23)
TLABEL = [f"{h//60:02d}:{h%60:02d}" for h in TIMES_HM]
NT = len(TIMES_HM)


def day_target(n):
    return MAX_PER_STOCK if n <= 5 else DAILY_POOL / n


def build():
    """Pure-breakout qualifying trades + cached next-day opens.
    Returns (breakout DataFrame with entry/shares/cap, opens (n,23), counts)."""
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "entry_price_315pm",
                                "return_pct_vs_prev_close"], parse_dates=["date"])
    sig = diag[(diag["return_pct_vs_prev_close"] >= RET_MIN)
               & diag["entry_price_315pm"].notna()].reset_index(drop=True)   # NO volume filter
    n_ret3 = len(sig)

    rows = []                       # breakout signals with cached T+1 opens
    opens_cache = []
    n_no_hist = 0
    for sym, grp in sig.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            n_no_hist += len(grp); continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        dh = raw.groupby("date")["high"].max()
        ds = sorted(dh.index); dh = dh.reindex(ds)
        r52 = dh.rolling(WIN_52, min_periods=WIN_52).max().shift(1)      # ends day BEFORE entry
        r52_map, dh_map = r52.to_dict(), dh.to_dict()
        po = (raw[raw["hm"].isin(TIMES_HM)]
              .pivot_table(index="date", columns="hm", values="open", aggfunc="last")
              .reindex(columns=TIMES_HM))
        po_map = {d: v for d, v in zip(po.index, po.values)}
        for d, ep in zip(grp["date"], grp["entry_price_315pm"]):
            dd = pd.Timestamp(d).date()
            hi = r52_map.get(dd, np.nan)
            if hi is None or hi != hi:                  # < 252-day history
                n_no_hist += 1; continue
            if not (dh_map.get(dd, np.nan) >= hi):      # not a new 52w high
                continue
            j = bisect.bisect_right(ds, dd)             # T+1
            o = po_map.get(ds[j]) if j < len(ds) else None
            rows.append({"date": dd, "symbol": sym, "entry": float(ep)})
            opens_cache.append(o if o is not None else np.full(NT, np.nan))

    bk = pd.DataFrame(rows)
    opens = np.array(opens_cache, dtype=float)
    # pool-split sizing on breakout signals' own daily counts
    cnt = bk.groupby("date")["symbol"].transform("size").values
    tgt = np.where(cnt <= 5, MAX_PER_STOCK, DAILY_POOL / cnt)
    sh = np.floor(tgt / bk["entry"].values)
    ok = sh > 0
    bk = bk[ok].reset_index(drop=True); opens = opens[ok]; sh = sh[ok]
    bk["shares"] = sh.astype(int); bk["cap"] = sh * bk["entry"].values
    return bk, opens, n_ret3, n_no_hist


def combo_metrics(exit_type, i1, i2, e, sh, cap, exit_px):
    valid = ~np.isnan(exit_px)
    nv = int(valid.sum())
    if nv == 0:
        return None
    ev, shv, cpv, xv = e[valid], sh[valid], cap[valid], exit_px[valid]
    pnl = shv * (xv - ev); ret = (xv - ev) / ev * 100
    n023 = pnl - cpv * EXP023; r023 = ret - EXP023 * 100      # net @ 0.23%
    n038 = pnl - cpv * EXP038; r038 = ret - EXP038 * 100      # net @ 0.38%
    return {"exit_type": exit_type, "exit_time_1": TLABEL[i1],
            "exit_time_2": TLABEL[i2] if i2 is not None else "", "n_trades": nv,
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "avg_return_per_trade_pct": round(float(ret.mean()), 4),
            "median_return_per_trade_pct": round(float(np.median(ret)), 4),
            "total_return_fixedbase_pct": round(float(pnl.sum()) / BASE_POOL * 100, 4),
            "total_pnl_inr": round(float(pnl.sum()), 0),
            "net023_win_rate_pct": round(float((n023 > 0).mean() * 100), 2),
            "net023_avg_return_per_trade_pct": round(float(r023.mean()), 4),
            "net023_median_return_per_trade_pct": round(float(np.median(r023)), 4),
            "net023_total_return_fixedbase_pct": round(float(n023.sum()) / BASE_POOL * 100, 4),
            "net023_total_pnl_inr": round(float(n023.sum()), 0),
            "net038_win_rate_pct": round(float((n038 > 0).mean() * 100), 2),
            "net038_avg_return_per_trade_pct": round(float(r038.mean()), 4),
            "net038_median_return_per_trade_pct": round(float(np.median(r038)), 4),
            "net038_total_return_fixedbase_pct": round(float(n038.sum()) / BASE_POOL * 100, 4),
            "net038_total_pnl_inr": round(float(n038.sum()), 0),
            "avg_capital_deployed_per_trade": round(float(cpv.mean()), 0),
            "small_sample_flag": "n_trades<30" if nv < SMALL else ""}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Building pure-breakout trade set (no volume filter) …")
    bk, opens, n_ret3, n_no_hist = build()
    e = bk["entry"].values; sh = bk["shares"].values.astype(float); cap = bk["cap"].values
    n = len(bk); ndays = bk["date"].nunique()
    print(f"  +3% signals (no volume): {n_ret3:,}")
    print(f"  dropped (<252d history / missing parquet): {n_no_hist:,}")
    print(f"  pure-breakout trades: {n:,} over {ndays:,} days | avg {n/ndays:.2f} signals/day")
    print(f"  (main volume-breakout strategy: ~3,494 trades)")

    rows = []
    for i1 in range(NT):                                # EXIT 1: full exit @ T
        m = combo_metrics("full_exit", i1, None, e, sh, cap, opens[:, i1])
        if m:
            rows.append(m)
    for i1 in range(NT):                                # EXIT 2: conditional split
        o1 = opens[:, i1]; ret1 = (o1 - e) / e * 100
        for i2 in range(i1 + 1, NT):
            o2 = opens[:, i2]
            exit_px = np.where(ret1 > 0, o1, o2)
            m = combo_metrics("conditional_split", i1, i2, e, sh, cap, exit_px)
            if m:
                rows.append(m)
    df = pd.DataFrame(rows).sort_values("total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(OUTDIR / "pure_breakout_exit_sweep.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "pure_breakout_exit_sweep.xlsx", engine="openpyxl") as w:
        df.to_excel(w, sheet_name="all_combos", index=False)

    best_full = df[df["exit_type"] == "full_exit"].iloc[0]
    best_cond = df[df["exit_type"] == "conditional_split"].iloc[0]

    # heatmap (conditional_split)
    cond = df[df["exit_type"] == "conditional_split"]
    pos = {lab: k for k, lab in enumerate(TLABEL)}
    grid = np.full((NT, NT), np.nan)
    for _, r in cond.iterrows():
        grid[pos[r["exit_time_1"]], pos[r["exit_time_2"]]] = r["total_return_fixedbase_pct"]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(NT)); ax.set_xticklabels(TLABEL, rotation=90, fontsize=7)
    ax.set_yticks(range(NT)); ax.set_yticklabels(TLABEL, fontsize=7)
    ax.set_xlabel("t2"); ax.set_ylabel("t1")
    ax.set_title("Pure 52w-high breakout — conditional split total_return_fixedbase_pct\n(t1<t2 upper triangle)",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="total return %")
    br, bc = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.plot(bc, br, "r*", markersize=18, markeredgecolor="white")
    fig.tight_layout(); fig.savefig(OUTDIR / "conditional_split_heatmap.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 230)
    cols = ["rank", "exit_type", "exit_time_1", "exit_time_2", "n_trades", "win_rate_pct",
            "avg_return_per_trade_pct", "median_return_per_trade_pct",
            "total_return_fixedbase_pct", "net023_total_return_fixedbase_pct",
            "net038_total_return_fixedbase_pct", "avg_capital_deployed_per_trade",
            "small_sample_flag"]
    print(f"\nTotal combos: {len(df)} (full_exit {(df.exit_type=='full_exit').sum()}, "
          f"conditional_split {(df.exit_type=='conditional_split').sum()})")
    print("\n--- BEST full_exit ---"); print(best_full[cols[1:]].to_string())
    print("\n--- BEST conditional_split ---"); print(best_cond[cols[1:]].to_string())
    print("\n" + "=" * 130 + "\nTOP 15 ACROSS BOTH TYPES\n" + "=" * 130)
    print(df[cols].head(15).to_string(index=False))

    print("\n" + "=" * 130)
    print("COMPARISON — pure-breakout best exit vs main volume-breakout strategy")
    print("=" * 130)
    b = df.iloc[0]
    cmp = pd.DataFrame([
        {"strategy": "pure_breakout (best exit)", "exit": f"{b.exit_type} {b.exit_time_1}/{b.exit_time_2}",
         "n_trades": int(b.n_trades), "win_rate_pct": b.win_rate_pct,
         "avg_return_per_trade_pct": b.avg_return_per_trade_pct,
         "median_return_per_trade_pct": b.median_return_per_trade_pct,
         "total_return_fixedbase_pct": b.total_return_fixedbase_pct},
        {"strategy": "main volume-breakout (9:45/12:00 +14% target)", "exit": "conditional_split_best_t2",
         "n_trades": 3494, "win_rate_pct": 63.48, "avg_return_per_trade_pct": 0.9140,
         "median_return_per_trade_pct": 0.7351, "total_return_fixedbase_pct": 550.8827},
    ])
    print(cmp.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
