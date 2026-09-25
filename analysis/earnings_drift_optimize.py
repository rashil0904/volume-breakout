# -*- coding: utf-8 -*-
"""earnings_drift_optimize.py — 3-D best-parameter search on post-earnings drift: (K, holding n,
during-timing) maximizing TOTAL RETURN with SL active, LONG and SHORT independently. SL=T0 low/high,
exit early if breached T+1..T+n else exit at T+n close. IS(2022-24)/OOS(2025+) validation. Heatmaps.
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

ANN = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
OUTDIR = rb.RESULTS / "earnings_drift_optimize"
KS = [3, 4, 5, 6, 7]; NS = list(range(1, 11)); IS_YEARS = {2022, 2023, 2024}


def build_events():
    a = pd.read_csv(ANN)
    a["ann_date"] = pd.to_datetime(a["announcement_date"], format="%d-%m-%Y", errors="coerce").dt.date
    a["dtm"] = pd.to_datetime(a["announcement_datetime"], errors="coerce")
    a = a.dropna(subset=["ann_date", "dtm"]).sort_values("dtm").drop_duplicates(["symbol", "quarter"], keep="first")
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "high", "low", "close"])
    d["date"] = pd.to_datetime(d["date"]).dt.date
    sd = {}
    for sym, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date")
        sd[sym] = (g["date"].values, g["close"].values.astype(float), g["low"].values.astype(float),
                   g["high"].values.astype(float), {dt: i for i, dt in enumerate(g["date"].values)})
    meta = []; FC, FL, FH = [], [], []
    for r in a.itertuples():
        s = sd.get(r.symbol)
        if s is None:
            continue
        dates, cl, lo, hi, dmap = s
        ps = dmap.get(r.ann_date)
        pn = int(np.searchsorted(dates, r.ann_date, "right")); pn = pn if pn < len(dates) else None
        cands = []
        if r.announcement_session == "post_market" and pn is not None:
            cands = [("post", pn)]
        elif r.announcement_session == "pre_market":
            cands = [("pre", ps if ps is not None else pn)]
        elif r.announcement_session == "during_market":
            if ps is not None:
                cands.append(("during_sameday", ps))
            if pn is not None:
                cands.append(("during_nextday", pn))
        for rtype, p in cands:
            if p is None or p < 1 or p + 1 >= len(dates):
                continue
            entry = cl[p]; prevc = cl[p - 1]
            if not (entry > 0 and prevc > 0):
                continue
            nf = min(10, len(dates) - 1 - p)
            fc = np.full(10, np.nan); fl = np.full(10, np.nan); fh = np.full(10, np.nan)
            fc[:nf] = cl[p + 1:p + 1 + nf]; fl[:nf] = lo[p + 1:p + 1 + nf]; fh[:nf] = hi[p + 1:p + 1 + nf]
            meta.append({"symbol": r.symbol, "reaction_type": rtype, "year": int(str(dates[p])[:4]),
                         "rr": (entry - prevc) / prevc * 100.0, "entry": entry, "t0_low": lo[p], "t0_high": hi[p], "nf": nf})
            FC.append(fc); FL.append(fl); FH.append(fh)
    M = pd.DataFrame(meta)
    return M, np.array(FC), np.array(FL), np.array(FH)


def realized(idx, M, FC, FL, FH, direction, n):
    """SL-active realized return for a horizon n over event indices idx (that have nf>=n)."""
    entry = M["entry"].values[idx]; t0l = M["t0_low"].values[idx]; t0h = M["t0_high"].values[idx]
    fc = FC[idx, :n]; fl = FL[idx, :n]; fh = FH[idx, :n]
    if direction == "long":
        breach = fl <= t0l[:, None]
        stopped = breach.any(axis=1)
        first = np.where(stopped, breach.argmax(axis=1) + 1, n)
        exit_px = np.where(stopped, t0l, fc[:, n - 1])
        ret = (exit_px - entry) / entry * 100.0
    else:
        breach = fh >= t0h[:, None]
        stopped = breach.any(axis=1)
        first = np.where(stopped, breach.argmax(axis=1) + 1, n)
        exit_px = np.where(stopped, t0h, fc[:, n - 1])
        ret = (entry - exit_px) / entry * 100.0
    return ret, first, stopped


def metrics(ret, first, stopped, yearsub):
    if len(ret) == 0:
        return None
    eq = np.cumsum(ret); dd = float((eq - np.maximum.accumulate(eq)).min())
    return {"n_events": len(ret), "total_return": round(float(ret.sum()), 1),
            "avg_return": round(float(ret.mean()), 3), "win_rate_pct": round(float((ret > 0).mean() * 100), 1),
            "sl_hit_rate_pct": round(float(stopped.mean() * 100), 1), "avg_holding": round(float(first.mean()), 1),
            "max_dd": round(dd, 1)}


def event_subset(M, K, timing, direction):
    """indices of events in the set for (K, timing): post(nextday)+pre(sameday)+during(timing), sign-filtered."""
    rt = M["reaction_type"].values; rr = M["rr"].values
    during = "during_sameday" if timing == "sameday" else "during_nextday"
    inset = (rt == "post") | (rt == "pre") | (rt == during)
    sign = (rr >= K) if direction == "long" else (rr <= -K)
    return np.where(inset & sign)[0]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    M, FC, FL, FH = build_events()
    print(f"events {len(M):,} | types {M['reaction_type'].value_counts().to_dict()}")
    yr = M["year"].values

    grids = {}
    for direction in ["long", "short"]:
        rows = []
        for K in KS:
            for timing in ["sameday", "nextday"]:
                base = event_subset(M, K, timing, direction)
                for n in NS:
                    idx = base[M["nf"].values[base] >= n]
                    if len(idx) == 0:
                        continue
                    ret, first, stopped = realized(idx, M, FC, FL, FH, direction, n)
                    m = metrics(ret, first, stopped, yr[idx])
                    # IS / OOS
                    ismask = np.isin(yr[idx], list(IS_YEARS)); oosmask = ~ismask
                    mis = metrics(ret[ismask], first[ismask], stopped[ismask], None) if ismask.any() else None
                    moos = metrics(ret[oosmask], first[oosmask], stopped[oosmask], None) if oosmask.any() else None
                    rows.append({"direction": direction, "K": K, "timing": timing, "n": n, **m,
                                 "IS_total": mis["total_return"] if mis else np.nan, "IS_avg": mis["avg_return"] if mis else np.nan,
                                 "IS_n": mis["n_events"] if mis else 0,
                                 "OOS_total": moos["total_return"] if moos else np.nan, "OOS_avg": moos["avg_return"] if moos else np.nan,
                                 "OOS_n": moos["n_events"] if moos else 0})
        grids[direction] = pd.DataFrame(rows)

    with pd.ExcelWriter(OUTDIR / "earnings_drift_optimize.xlsx", engine="openpyxl") as w:
        for direction in ["long", "short"]:
            g = grids[direction].sort_values("total_return", ascending=False)
            g.to_excel(w, sheet_name=f"{direction}_by_total_return", index=False)

    # heatmaps (K x n) at each direction's best timing
    def heat(direction):
        g = grids[direction]
        best_timing = g.groupby("timing")["total_return"].max().idxmax()
        piv = g[g.timing == best_timing].pivot(index="K", columns="n", values="total_return")
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(NS))); ax.set_xticklabels(NS); ax.set_yticks(range(len(KS))); ax.set_yticklabels(KS)
        ax.set_xlabel("holding n (T+n)"); ax.set_ylabel("K %"); ax.set_title(f"{direction.upper()} total return (K x n) @ timing={best_timing}")
        for yi in range(len(KS)):
            for xi in range(len(NS)):
                v = piv.values[yi, xi]
                if v == v:
                    ax.text(xi, yi, f"{v:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(OUTDIR / f"heat_{direction}.png", dpi=110); plt.close(fig)
        return best_timing
    bt_long = heat("long"); bt_short = heat("short")

    pd.set_option("display.width", 240)
    for direction in ["long", "short"]:
        g = grids[direction].sort_values("total_return", ascending=False)
        print("\n" + "=" * 96 + f"\n{direction.upper()} — top 10 by TOTAL RETURN (SL-active)\n" + "=" * 96)
        cols = ["K", "timing", "n", "n_events", "total_return", "avg_return", "win_rate_pct", "sl_hit_rate_pct",
                "avg_holding", "max_dd", "OOS_avg", "OOS_n"]
        print(g[cols].head(10).to_string(index=False))
        b = g.iloc[0]
        print(f"  BEST {direction}: K={int(b.K)} n=T+{int(b.n)} timing={b.timing} | total {b.total_return} avg {b.avg_return} "
              f"win {b.win_rate_pct}% SLhit {b.sl_hit_rate_pct}% | IS_avg {b.IS_avg} OOS_avg {b.OOS_avg} (OOS n={int(b.OOS_n)})")
        # during-timing head-to-head (best n per timing)
        for tm in ["sameday", "nextday"]:
            gt = g[g.timing == tm]
            if len(gt):
                bb = gt.iloc[0]
                print(f"    timing={tm}: best total {bb.total_return} @ K={int(bb.K)} n=T+{int(bb.n)} (avg {bb.avg_return}, n={int(bb.n_events)})")
        # best combo on IS, its OOS
        gis = grids[direction].dropna(subset=["IS_avg"]).copy()
        gis = gis[gis["IS_n"] >= 30].sort_values("IS_total", ascending=False)
        if len(gis):
            bi = gis.iloc[0]
            print(f"  BEST-on-IS: K={int(bi.K)} n=T+{int(bi.n)} {bi.timing} | IS avg {bi.IS_avg} (n={int(bi.IS_n)}) -> OOS avg {bi.OOS_avg} (n={int(bi.OOS_n)})  "
                  f"[{'HOLDS' if (bi.OOS_avg==bi.OOS_avg and bi.OOS_avg>0) else 'FAILS'} OOS]")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
