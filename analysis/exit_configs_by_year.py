# -*- coding: utf-8 -*-
"""
exit_configs_by_year.py — year-wise performance for specific exit configs (net_A combined
long+short), to test out-of-sample stability like the 15:21 entry-time check.

Configs (Category C entry FIXED 15:21, A/B unchanged, short covers 3pm):
  F1  split  t1=09:25  t2=11:59  target 17%   (File-1 best, 09:20-12:00 grid)
  F2  split  t1=09:48  t2=11:59  target 10%   (File-2 best, 09:30-12:00 grid)
  BL  split  t1=09:45  t2=12:00  target 14%   (incumbent baseline)
All on the SAME trade set (complete exit-day grid) so year-by-year is apples-to-apples.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import uc_staggered_dd_report as R
import exit_sweep_wide as XW

OUTDIR = rb.RESULTS / "exit_sweep_wide"
BASE_POOL, R023, R038, SR = XW.BASE_POOL, XW.R023, XW.R038, XW.SR
EXIT_HMS = XW.EXIT_HMS
CONFIGS = [("F1_0925_1159_17", "09:25", "11:59", 17),
           ("F2_0948_1159_10", "09:48", "11:59", 10),
           ("BL_0945_1200_14", "09:45", "12:00", 14)]


def build_cache_year():
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_all_three",
                                "prev_day_vwap_close", "entry_price_315pm"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True]
    print("Scanning engine …", flush=True)
    records = []
    for s in sorted(Q["symbol"].unique()):
        records += R.scan_symbol(s, Q[Q["symbol"] == s])
    T_def, day_diag = R.size_and_price(records)
    per_C = {d: v["per_C"] for d, v in day_diag.items()}
    trades = []
    for _, r in T_def[T_def["category"].isin(["A", "B"])].iterrows():
        trades.append(dict(symbol=r["symbol"], entry_date=r["entry_date"], exit_date=r["exit_date"],
                           entry_price=r["avg_entry"], shares=float(r["shares"]),
                           cap=float(r["capital_deployed"]), per_C=None))
    for r in records:
        if r["category"] == "C" and r["entered"] and per_C.get(r["entry_date"], 0) > 0:
            trades.append(dict(symbol=r["symbol"], entry_date=r["entry_date"], exit_date=r["next_date"],
                               entry_price=None, shares=None, cap=None, per_C=per_C[r["entry_date"]]))
    by_sym = {}
    for t in trades:
        by_sym.setdefault(t["symbol"], []).append(t)
    rows, dropped = [], 0
    for sym, ts in by_sym.items():
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            dropped += len(ts); continue
        raw = pd.read_parquet(pq)
        tstamp = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(XW.IST)
        raw = raw.assign(date=tstamp.dt.date, hm=tstamp.dt.hour * 60 + tstamp.dt.minute)
        by_date = {d: g for d, g in raw.groupby("date")}
        for t in ts:
            ed, xd = t["entry_date"], t["exit_date"]
            if xd is None or xd not in by_date:
                dropped += 1; continue
            xg = by_date[xd]
            xo = dict(zip(xg["hm"].values, xg["open"].values.astype(float)))
            xh = dict(zip(xg["hm"].values, xg["high"].values.astype(float)))
            opens = np.array([xo.get(h, np.nan) for h in EXIT_HMS])
            highs = np.array([xh.get(h, np.nan) for h in XW.HIGH_HMS])
            o3pm = xo.get(XW.HM_3PM, np.nan)
            if t["per_C"] is not None:
                eg = by_date.get(ed)
                p = dict(zip(eg["hm"].values, eg["open"].values.astype(float))).get(XW.HM_C_ENTRY, np.nan) if eg is not None else np.nan
                if not (p == p and p > 0):
                    dropped += 1; continue
                shares = float(np.floor(t["per_C"] / p))
                if shares <= 0:
                    dropped += 1; continue
                entry_price, cap = p, shares * p
            else:
                entry_price, shares, cap = t["entry_price"], t["shares"], t["cap"]
            if np.isnan(opens).any() or not (o3pm == o3pm):
                dropped += 1; continue
            rows.append((ed.year, entry_price, shares, cap, o3pm, opens, highs))
    C = dict(year=np.array([r[0] for r in rows]),
             entry=np.array([r[1] for r in rows], float), shares=np.array([r[2] for r in rows], float),
             cap=np.array([r[3] for r in rows], float), o3pm=np.array([r[4] for r in rows], float),
             hs=np.ones(len(rows)), opens=np.vstack([r[5] for r in rows]), highs=np.vstack([r[6] for r in rows]))
    print(f"  kept {len(rows):,} trades; dropped {dropped:,}.", flush=True)
    return C


def combined_pnls(C, t1, t2, tgt):
    E, _ = XW.E_matrix(C, tgt)
    i = int(np.where(EXIT_HMS == XW.hm_of(t1))[0][0]); j = int(np.where(EXIT_HMS == XW.hm_of(t2))[0][0])
    lep = np.where(C["opens"][:, i] > C["entry"], E[:, i], E[:, j])
    long_pnl = C["shares"] * (lep - C["entry"]); short_pnl = C["shares"] * (lep - C["o3pm"])
    comb = long_pnl + short_pnl
    netA = comb - R023 * C["cap"] - SR * C["shares"] * lep
    netB = comb - R038 * C["cap"] - SR * C["shares"] * lep
    return comb, netA, netB


def main():
    C = build_cache_year()
    years = sorted(set(C["year"].tolist()))
    per_cfg = {}
    for name, t1, t2, tgt in CONFIGS:
        g, a, b = combined_pnls(C, t1, t2, tgt)
        rows = []
        for y in years:
            m = C["year"] == y
            n = int(m.sum())
            ra = a[m] / C["cap"][m] * 100
            rows.append({"year": y, "n_trades": n,
                         "gross_total_return_fixedbase_pct": round(g[m].sum() / BASE_POOL * 100, 2),
                         "net_A_total_return_fixedbase_pct": round(a[m].sum() / BASE_POOL * 100, 2),
                         "net_B_total_return_fixedbase_pct": round(b[m].sum() / BASE_POOL * 100, 2),
                         "net_A_win_rate_pct": round(float((a[m] > 0).mean() * 100), 2),
                         "net_A_avg_return_per_trade_pct": round(float(ra.mean()), 4),
                         "net_A_median_return_per_trade_pct": round(float(np.median(ra)), 4)})
        df = pd.DataFrame(rows)
        tot = {"year": "ALL", "n_trades": len(C["year"]),
               "gross_total_return_fixedbase_pct": round(g.sum() / BASE_POOL * 100, 2),
               "net_A_total_return_fixedbase_pct": round(a.sum() / BASE_POOL * 100, 2),
               "net_B_total_return_fixedbase_pct": round(b.sum() / BASE_POOL * 100, 2),
               "net_A_win_rate_pct": round(float((a > 0).mean() * 100), 2),
               "net_A_avg_return_per_trade_pct": round(float((a / C["cap"] * 100).mean()), 4),
               "net_A_median_return_per_trade_pct": round(float(np.median(a / C["cap"] * 100)), 4)}
        per_cfg[name] = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)

    # comparison matrices: year × config
    def matrix(metric):
        d = {name: per_cfg[name].set_index("year")[metric] for name, *_ in CONFIGS}
        return pd.DataFrame(d)
    cmp_tot = matrix("net_A_total_return_fixedbase_pct")
    cmp_avg = matrix("net_A_avg_return_per_trade_pct")
    cmp_win = matrix("net_A_win_rate_pct")

    with pd.ExcelWriter(OUTDIR / "exit_configs_by_year.xlsx", engine="openpyxl") as w:
        cmp_tot.to_excel(w, sheet_name="net_A_total_return_by_year")
        cmp_avg.to_excel(w, sheet_name="net_A_avg_return_by_year")
        cmp_win.to_excel(w, sheet_name="net_A_win_rate_by_year")
        for name, *_ in CONFIGS:
            per_cfg[name].to_excel(w, sheet_name=name[:31], index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print("YEAR-WISE net_A TOTAL RETURN (fixedbase %) — combined long+short")
    print("=" * 90)
    print(cmp_tot.to_string())
    print("\nYEAR-WISE net_A AVG RETURN PER TRADE (%):")
    print(cmp_avg.to_string())
    print("\nYEAR-WISE net_A WIN RATE (%):")
    print(cmp_win.to_string())
    yrs = [y for y in cmp_avg.index if y != "ALL"]
    wins = cmp_avg.loc[yrs].idxmax(axis=1)
    print("\nBest config each year (by net_A avg return/trade):")
    print("  ", dict(wins))
    from collections import Counter
    print("  win counts:", dict(Counter(wins)))
    print(f"\nSaved -> {OUTDIR / 'exit_configs_by_year.xlsx'}")


if __name__ == "__main__":
    main()
