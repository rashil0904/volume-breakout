# -*- coding: utf-8 -*-
"""pead_target_sweep.py — trigger PROFIT-TARGET sweep on finalized PEAD trades (VWAP-ref reaction, K=4,
LONG hold-cap T+10 / SHORT T+3, trigger SL=T0 low/high, during->next-day, entry=T0 close). Three exits:
target(trigger)/SL(trigger)/hold-cap, whichever first; same-day-both -> SL first. Gap-through fills at
open (flagged). LONG targets 2-7% (0.5 step), SHORT 1-4%. Rank by total return. vs no-target. IS/OOS.
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
VWAPD = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "pead_target_sweep"
K, N_LONG, N_SHORT, COST = 4.0, 10, 3, 0.20
LONG_TGTS = sorted(set([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7] + list(range(2, 21))))     # 0.5 steps low + 1% to 20
SHORT_TGTS = sorted(set([1, 1.5, 2, 2.5, 3, 3.5, 4] + list(range(1, 21))))
IS_YEARS = {2022, 2023, 2024}


def build():
    a = pd.read_csv(ANN)
    a["ann_date"] = pd.to_datetime(a["announcement_date"], format="%d-%m-%Y", errors="coerce").dt.date
    a["dtm"] = pd.to_datetime(a["announcement_datetime"], errors="coerce")
    a = a.dropna(subset=["ann_date", "dtm"]).sort_values("dtm").drop_duplicates(["symbol", "quarter"], keep="first")
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "open", "high", "low", "close"]); d["date"] = pd.to_datetime(d["date"]).dt.date
    vw = pd.read_parquet(VWAPD, columns=["symbol", "date", "close"]); vw["date"] = pd.to_datetime(vw["date"]).dt.date
    vmap = {(s, dt): c for s, dt, c in zip(vw["symbol"], vw["date"], vw["close"])}
    sd = {}
    for sym, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date")
        sd[sym] = (g["date"].values, g["open"].values.astype(float), g["high"].values.astype(float),
                   g["low"].values.astype(float), g["close"].values.astype(float), {dt: i for i, dt in enumerate(g["date"].values)})
    ev = []
    for r in a.itertuples():
        s = sd.get(r.symbol)
        if s is None:
            continue
        dates, op, hi, lo, cl, dmap = s
        ps = dmap.get(r.ann_date); pn = int(np.searchsorted(dates, r.ann_date, "right")); pn = pn if pn < len(dates) else None
        p = pn if r.announcement_session in ("post_market", "during_market") else (ps if ps is not None else pn)
        if p is None or p < 1:
            continue
        entry = cl[p]; prev_vwap = vmap.get((r.symbol, dates[p - 1]), np.nan)
        if not (entry > 0 and prev_vwap == prev_vwap and prev_vwap > 0):
            continue
        rr = (entry - prev_vwap) / prev_vwap * 100.0
        if rr >= K:
            direction, n, sl = "long", N_LONG, lo[p]
        elif rr <= -K:
            direction, n, sl = "short", N_SHORT, hi[p]
        else:
            continue
        if p + n >= len(dates):
            continue
        ev.append({"symbol": r.symbol, "direction": direction, "entry": entry, "sl": sl, "n": n,
                   "year": int(str(dates[p])[:4]),
                   "fo": op[p + 1:p + 1 + n], "fh": hi[p + 1:p + 1 + n], "fl": lo[p + 1:p + 1 + n], "fc": cl[p + 1:p + 1 + n]})
    return ev


def exit_trade(e, tgt_pct):
    entry, sl, n, dr = e["entry"], e["sl"], e["n"], e["direction"]
    fo, fh, fl, fc = e["fo"], e["fh"], e["fl"], e["fc"]
    tgt = None if tgt_pct is None else (entry * (1 + tgt_pct / 100) if dr == "long" else entry * (1 - tgt_pct / 100))
    for j in range(n):
        o, h, l, c = fo[j], fh[j], fl[j], fc[j]
        if dr == "long":
            hit_sl = l <= sl; hit_t = (tgt is not None) and (h >= tgt)
            if hit_sl and hit_t:
                return min(sl, o), "sl", j + 1, o < sl, True
            if hit_sl:
                return min(sl, o), "sl", j + 1, o < sl, False
            if hit_t:
                return max(tgt, o), "target", j + 1, o > tgt, False
        else:
            hit_sl = h >= sl; hit_t = (tgt is not None) and (l <= tgt)
            if hit_sl and hit_t:
                return max(sl, o), "sl", j + 1, o > sl, True
            if hit_sl:
                return max(sl, o), "sl", j + 1, o > sl, False
            if hit_t:
                return min(tgt, o), "target", j + 1, o < tgt, False
    return fc[n - 1], "holding_end", n, False, False


def agg(events, tgt_pct):
    rr = []; days = []; reasons = []; ngap = 0; nboth = 0
    for e in events:
        exp, reason, dh, gap, both = exit_trade(e, tgt_pct)
        r = (exp - e["entry"]) / e["entry"] * 100 if e["direction"] == "long" else (e["entry"] - exp) / e["entry"] * 100
        rr.append(r); days.append(dh); reasons.append(reason); ngap += int(gap); nboth += int(both)
    rr = np.array(rr); nr = rr - COST; reasons = np.array(reasons); yrs = np.array([e["year"] for e in events])
    eqn = np.cumsum(nr); dd = float((eqn - np.maximum.accumulate(eqn)).min()) if len(eqn) else 0.0
    ism = np.isin(yrs, list(IS_YEARS))
    def sub(mask):
        return (round(rr[mask].sum(), 1), round(rr[mask].mean(), 3), int(mask.sum())) if mask.any() else (0, np.nan, 0)
    is_t, is_a, is_n = sub(ism); oos_t, oos_a, oos_n = sub(~ism)
    return {"target_pct": tgt_pct if tgt_pct is not None else "NONE", "n_trades": len(rr),
            "total_return": round(rr.sum(), 1), "avg_return": round(rr.mean(), 3), "median_return": round(np.median(rr), 3),
            "avg_net": round(nr.mean(), 3), "win_rate_pct": round((rr > 0).mean() * 100, 1),
            "pct_target": round((reasons == "target").mean() * 100, 1), "pct_sl": round((reasons == "sl").mean() * 100, 1),
            "pct_holding_end": round((reasons == "holding_end").mean() * 100, 1), "avg_days_held": round(np.mean(days), 1),
            "n_gap_through": ngap, "n_same_day_both": nboth, "max_dd_net": round(dd, 1),
            "IS_total": is_t, "IS_avg": is_a, "IS_n": is_n, "OOS_total": oos_t, "OOS_avg": oos_a, "OOS_n": oos_n}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ev = build()
    L = [e for e in ev if e["direction"] == "long"]; S = [e for e in ev if e["direction"] == "short"]
    print(f"events {len(ev)} | long {len(L)} short {len(S)}")

    grids = {}
    for dr, evs, tgts in [("long", L, LONG_TGTS), ("short", S, SHORT_TGTS)]:
        rows = [agg(evs, None)] + [agg(evs, t) for t in tgts]
        grids[dr] = pd.DataFrame(rows)

    with pd.ExcelWriter(OUTDIR / "pead_target_sweep.xlsx", engine="openpyxl") as w:
        for dr in ["long", "short"]:
            grids[dr].to_excel(w, sheet_name=f"{dr}_sweep", index=False)

    # charts
    for dr in ["long", "short"]:
        g = grids[dr][grids[dr].target_pct != "NONE"]
        base = grids[dr][grids[dr].target_pct == "NONE"]["total_return"].values[0]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(g["target_pct"], g["total_return"], marker="o", color="#4472C4", label="with target")
        ax.axhline(base, color="grey", ls="--", label=f"no-target (hold-cap+SL) = {base:.0f}")
        ax.set_xlabel("profit target %"); ax.set_ylabel("total return (Σ realized %)"); ax.set_title(f"{dr.upper()} — total return vs profit target")
        ax.legend(); fig.tight_layout(); fig.savefig(OUTDIR / f"target_curve_{dr}.png", dpi=120); plt.close(fig)

    pd.set_option("display.width", 250)
    for dr in ["long", "short"]:
        g = grids[dr]
        gt = g[g.target_pct != "NONE"].sort_values("total_return", ascending=False)
        base = g[g.target_pct == "NONE"].iloc[0]
        print("\n" + "=" * 100 + f"\n{dr.upper()} TARGET SWEEP (ranked by total return) — hold-cap T+{N_LONG if dr=='long' else N_SHORT}\n" + "=" * 100)
        cols = ["target_pct", "n_trades", "total_return", "avg_return", "win_rate_pct", "pct_target", "pct_sl", "pct_holding_end",
                "avg_days_held", "max_dd_net", "n_same_day_both", "OOS_avg", "OOS_n"]
        print(g[cols].to_string(index=False))
        b = gt.iloc[0]
        print(f"  NO-TARGET (hold-cap+SL): total {base.total_return} avg {base.avg_return} win {base.win_rate_pct}% maxDD {base.max_dd_net}")
        print(f"  BEST TARGET {b.target_pct}%: total {b.total_return} (Δ{round(b.total_return-base.total_return,1)} vs no-target) "
              f"avg {b.avg_return} win {b.win_rate_pct}% maxDD {b.max_dd_net} | %target {b.pct_target} %sl {b.pct_sl} | "
              f"IS_avg {b.IS_avg} OOS_avg {b.OOS_avg} (OOS n={int(b.OOS_n)})")
        # OOS-robust: best on IS
        gis = g[g.target_pct != "NONE"].sort_values("IS_total", ascending=False).iloc[0]
        print(f"  BEST-on-IS {gis.target_pct}%: IS_avg {gis.IS_avg} -> OOS_avg {gis.OOS_avg} "
              f"[{'HOLDS' if (gis.OOS_avg==gis.OOS_avg and gis.OOS_avg>0) else 'FAILS'} OOS]")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
