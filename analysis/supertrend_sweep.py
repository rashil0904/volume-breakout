# -*- coding: utf-8 -*-
"""supertrend_sweep.py — sweep Supertrend (ATR period x factor) on the daily long-only strategy.
period in [7,10,14,21], factor in [1,1.5,2,2.5,3,3.5,4,5] = 32 combos. ATR computed ONCE per period and
reused across the 8 factors. Same conventions as supertrend_strategy.py (bull flip->enter next open,
bear flip->exit next open, mcap>1500, cost 0.30%). Primary ranking = profit factor. IS/OOS validation.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
MDIR = rb.BASE / "mcap_cache"
OUTDIR = rb.RESULTS / "supertrend_sweep"
PERIODS = [7, 10, 14, 21]
FACTORS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
COST = 0.30
MCAP_MIN = 1500.0
POOL, PER_TRADE, MAX_POS = 1_000_000, 100_000, 10
IS_YEARS = {2022, 2023, 2024}
SNAPS = [("2022-03-31", "mcap_2022-03-31.xlsx"), ("2022-12-31", "mcap_2022-12-31.xlsx"),
         ("2023-03-31", "mcap_2023-03-31.xlsx"), ("2023-12-31", "mcap_2023-12-31.xlsx"),
         ("2024-03-28", "mcap_2024-03-28.xlsx"), ("2024-12-31", "mcap_2024-12-31.xlsx"),
         ("2025-12-31", "mcap_2025-12-31.xlsx")]


def load_mcap():
    dates, dicts = [], []
    for ds, fn in SNAPS:
        d = pd.read_excel(MDIR / fn, header=None, skiprows=1, usecols=[1, 3])
        d.columns = ["symbol", "mcap_lakhs"]; d["symbol"] = d["symbol"].astype(str).str.strip().str.upper()
        d["mcap_cr"] = pd.to_numeric(d["mcap_lakhs"], errors="coerce") / 100
        dates.append(np.datetime64(ds)); dicts.append(d.dropna(subset=["mcap_cr"]).set_index("symbol")["mcap_cr"].to_dict())
    return np.array(dates), dicts


def wilder_atr(h, lo, c, n):
    m = len(c); tr = np.empty(m); tr[0] = h[0] - lo[0]
    for i in range(1, m):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
    atr = np.full(m, np.nan)
    if m <= n:
        return atr
    atr[n - 1] = tr[:n].mean()
    for i in range(n, m):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def st_direction(h, lo, c, atr, n, f):
    m = len(c); hl2 = (h + lo) * 0.5
    ub = hl2 + f * atr; lb = hl2 - f * atr
    fub = ub.copy(); flb = lb.copy(); direction = np.zeros(m, dtype=np.int8)
    for i in range(n, m):
        if ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]:
            fub[i] = ub[i]
        else:
            fub[i] = fub[i - 1]
        if lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]:
            flb[i] = lb[i]
        else:
            flb[i] = flb[i - 1]
    up = True                                       # track ST=lower(up) vs upper(down); seed down at n
    stu = False
    prev_upper = True
    direction[n] = -1
    for i in range(n + 1, m):
        if prev_upper:                              # prior downtrend (ST=upper)
            if c[i] > fub[i]:
                direction[i] = 1; prev_upper = False
            else:
                direction[i] = -1; prev_upper = True
        else:                                       # prior uptrend (ST=lower)
            if c[i] < flb[i]:
                direction[i] = -1; prev_upper = True
            else:
                direction[i] = 1; prev_upper = False
    return direction


def extract_trades(sym, o, dt, direction, n, mcap_of):
    m = len(o); out = []; i = n + 1
    while i < m:
        if direction[i] == 1 and direction[i - 1] == -1:
            if i + 1 >= m:
                break
            ei = i + 1
            j = ei
            while j < m and not (direction[j] == -1 and direction[j - 1] == 1):
                j += 1
            xi = min(j + 1, m - 1) if j < m else m - 1
            mc = mcap_of(sym, dt[i])
            if mc == mc and mc > MCAP_MIN and o[ei] > 0:
                ret = (o[xi] - o[ei]) / o[ei] * 100
                out.append((pd.Timestamp(dt[ei]).year, ret, int(xi - ei)))
            i = j if j < m else m
        else:
            i += 1
    return out


def metrics(arr):
    r = arr["net"]; win = r > 0
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    hold = arr["hold"]
    whip = ((hold <= 3) & (r < 0)).mean() * 100
    return {"n_trades": len(r), "win_rate_pct": round(win.mean() * 100, 2),
            "avg_return_pct": round(r.mean(), 3), "median_return_pct": round(np.median(r), 3),
            "avg_winner_pct": round(r[win].mean(), 3) if win.any() else 0.0,
            "avg_loser_pct": round(r[~win].mean(), 3) if (~win).any() else 0.0,
            "expectancy_pct": round(r.mean(), 3), "profit_factor": round(gp / gl, 3) if gl > 0 else np.inf,
            "avg_holding_days": round(hold.mean(), 1), "total_return_sum_pct": round(r.sum(), 1),
            "whipsaw_frac_pct": round(whip, 1)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DAILY); df["date"] = pd.to_datetime(df["date"])
    snap_dates, snap_dicts = load_mcap()
    def mcap_of(sym, d64):
        idx = int(np.searchsorted(snap_dates, d64, "right")) - 1
        return snap_dicts[max(idx, 0)].get(sym, np.nan)

    # preload arrays + ATR per period (once)
    print("preloading symbols + ATR per period ...", flush=True)
    syms = {}
    for sym, s in df.groupby("symbol", sort=False):
        if len(s) < 30:
            continue
        s = s.sort_values("date")
        h = s["high"].values.astype(float); lo = s["low"].values.astype(float)
        c = s["close"].values.astype(float); o = s["open"].values.astype(float); dt = s["date"].values
        atrs = {p: wilder_atr(h, lo, c, p) for p in PERIODS}
        syms[sym] = (h, lo, c, o, dt, atrs)
    print(f"  {len(syms):,} symbols cached", flush=True)

    grid = []; peryear = []; t0 = time.time()
    for p in PERIODS:
        for f in FACTORS:
            recs = []
            for sym, (h, lo, c, o, dt, atrs) in syms.items():
                if len(c) <= p + 2:
                    continue
                d = st_direction(h, lo, c, atrs[p], p, f)
                recs.extend([(sym, *t) for t in extract_trades(sym, o, dt, d, p, mcap_of)])
            if not recs:
                continue
            R = pd.DataFrame(recs, columns=["symbol", "year", "ret", "hold"])
            R["net"] = R["ret"] - COST
            m = metrics(R)
            # IS/OOS profit factor + expectancy
            def pf_exp(sub):
                r = sub["net"]; gl = -r[r < 0].sum()
                return (round(r[r > 0].sum() / gl, 3) if gl > 0 else np.inf), round(r.mean(), 3), len(sub)
            pf_is, ex_is, n_is = pf_exp(R[R["year"].isin(IS_YEARS)])
            pf_oos, ex_oos, n_oos = pf_exp(R[~R["year"].isin(IS_YEARS)])
            grid.append({"period": p, "factor": f, **m, "pf_IS": pf_is, "expectancy_IS": ex_is, "n_IS": n_is,
                         "pf_OOS": pf_oos, "expectancy_OOS": ex_oos, "n_OOS": n_oos})
            for y, g in R.groupby("year"):
                rr = g["net"]; gl = -rr[rr < 0].sum()
                peryear.append({"period": p, "factor": f, "year": y, "n": len(g),
                                "win_rate": round((rr > 0).mean() * 100, 1), "avg_ret": round(rr.mean(), 3),
                                "profit_factor": round(rr[rr > 0].sum() / gl, 3) if gl > 0 else np.inf})
            print(f"  ({p},{f}): n={m['n_trades']:5d} PF={m['profit_factor']:.2f} exp={m['expectancy_pct']:+.2f} "
                  f"win={m['win_rate_pct']:.1f}% hold={m['avg_holding_days']:.0f} whip={m['whipsaw_frac_pct']:.0f}% "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    G = pd.DataFrame(grid)
    PY = pd.DataFrame(peryear)

    with pd.ExcelWriter(OUTDIR / "supertrend_sweep.xlsx", engine="openpyxl") as w:
        G.sort_values("profit_factor", ascending=False).to_excel(w, sheet_name="grid_by_PF", index=False)
        G.sort_values("total_return_sum_pct", ascending=False).to_excel(w, sheet_name="grid_by_totret", index=False)
        PY.to_excel(w, sheet_name="per_year", index=False)

    # heatmaps
    def heat(metric, title, fmt=".2f"):
        piv = G.pivot(index="period", columns="factor", values=metric)
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(FACTORS))); ax.set_xticklabels(FACTORS); ax.set_yticks(range(len(PERIODS))); ax.set_yticklabels(PERIODS)
        ax.set_xlabel("factor"); ax.set_ylabel("ATR period"); ax.set_title(f"Supertrend sweep — {title}")
        for yi in range(len(PERIODS)):
            for xi in range(len(FACTORS)):
                v = piv.values[yi, xi]
                ax.text(xi, yi, format(v, fmt), ha="center", va="center", fontsize=7)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(OUTDIR / f"heat_{metric}.png", dpi=110); plt.close(fig)
    for mtr, ttl, fmt in [("total_return_sum_pct", "total return (sum of per-trade %)", ".0f"),
                          ("profit_factor", "profit factor", ".2f"), ("win_rate_pct", "win rate %", ".0f"),
                          ("avg_holding_days", "avg holding days", ".0f"), ("n_trades", "n trades", ".0f"),
                          ("expectancy_pct", "expectancy %", ".2f")]:
        heat(mtr, ttl, fmt)

    pd.set_option("display.width", 260)
    dflt = G[(G["period"] == 10) & (G["factor"] == 3.0)].iloc[0]
    print("\n" + "=" * 110 + "\nSUPERTREND SWEEP — 32 combos (ranked by PROFIT FACTOR)\n" + "=" * 110)
    cols = ["period", "factor", "n_trades", "win_rate_pct", "avg_return_pct", "median_return_pct", "avg_holding_days",
            "profit_factor", "expectancy_pct", "total_return_sum_pct", "whipsaw_frac_pct", "pf_IS", "pf_OOS"]
    print(G.sort_values("profit_factor", ascending=False)[cols].to_string(index=False))
    print(f"\n  DEFAULT (10,3): PF={dflt['profit_factor']:.3f} exp={dflt['expectancy_pct']:+.3f} totret_sum={dflt['total_return_sum_pct']:.0f} "
          f"win={dflt['win_rate_pct']:.1f}% | pf_IS={dflt['pf_IS']:.2f} pf_OOS={dflt['pf_OOS']:.2f}")
    best_pf = G.sort_values("profit_factor", ascending=False).iloc[0]
    best_tot = G.sort_values("total_return_sum_pct", ascending=False).iloc[0]
    print(f"  BEST by PF        : ({int(best_pf['period'])},{best_pf['factor']}) PF={best_pf['profit_factor']:.3f} "
          f"totret_sum={best_pf['total_return_sum_pct']:.0f} pf_IS={best_pf['pf_IS']:.2f} pf_OOS={best_pf['pf_OOS']:.2f}")
    print(f"  BEST by total_ret : ({int(best_tot['period'])},{best_tot['factor']}) totret_sum={best_tot['total_return_sum_pct']:.0f} "
          f"PF={best_tot['profit_factor']:.3f} pf_IS={best_tot['pf_IS']:.2f} pf_OOS={best_tot['pf_OOS']:.2f}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
