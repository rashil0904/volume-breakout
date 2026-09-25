# -*- coding: utf-8 -*-
"""
joint_exit_cover_sweep.py — cross the TOP-100 long-exit configs with a 14:30-15:00 short-cover
sweep to find the best JOINT long+short exit scenario. Composite strategy: Category C entry
FIXED at 15:21, A/B unchanged, double-down short on filled qty, gross/net_A/net_B costs.

STEP 1  long configs = top 100 by net_A total_return (n>=30) from exit_sweep_3d + the incumbent
        baseline (09:45/12:00 split, 14% target)  = 101 configs.
STEP 2  short-cover grid = 14:30..15:00, 1-min = 31 cover times (hm 870..900), full 100% cover.
STEP 3  cross: 101 x 31 = 3,131 joint combos. For each trade:
          long_exit_price (= short-open price) from the long config;
          short_pnl = shares*(long_exit_price - open_at_cover);  combined = long + short.
        Costs: net_A = long 0.23% + short 0.10%; net_B = long 0.38% + short 0.10%. Short notional
        = shares*long_exit_price, so short cost is fixed per long config (cover time only moves the
        cover price). All long exits are <=12:00 < 14:30, so every short is eligible at every cover.

Reuses exit_sweep_3d.E_matrix + entry logic; caches each trade's 14:30-15:00 opens once.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import uc_staggered_dd_report as R
import exit_sweep_3d as X

OUTDIR = rb.RESULTS / "joint_exit_cover_sweep"
IST = R.IST
EXIT_HMS = X.EXIT_HMS                                   # 556..720 (long grid)
HIGH_HMS = X.HIGH_HMS                                   # 555..720
COVER_HMS = np.arange(870, 901)                         # 14:30..15:00 (31)
HM_C_ENTRY = X.HM_C_ENTRY                               # 921 (15:21)
BASE_POOL, R023, R038, SR = X.BASE_POOL, X.R023, X.R038, X.SR
N_MIN = 30
BASELINE = ("split", "09:45", "12:00", 14)             # incumbent long config


def hm_of(s):
    return int(s[:2]) * 60 + int(s[3:])


def lbl(hm):
    return f"{hm // 60:02d}:{hm % 60:02d}"


def build_cache2():
    """Like exit_sweep_3d.build_cache but ALSO caches 14:30-15:00 cover opens (same trade set
    rules: A/B fixed, C@15:21, require complete long grid + complete cover grid)."""
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
                           category=r["category"], entry_price=r["avg_entry"], shares=float(r["shares"]),
                           cap=float(r["capital_deployed"]), per_C=None))
    for r in records:
        if r["category"] == "C" and r["entered"] and per_C.get(r["entry_date"], 0) > 0:
            trades.append(dict(symbol=r["symbol"], entry_date=r["entry_date"], exit_date=r["next_date"],
                               category="C", entry_price=None, shares=None, cap=None, per_C=per_C[r["entry_date"]]))
    by_sym = {}
    for t in trades:
        by_sym.setdefault(t["symbol"], []).append(t)
    rows, dropped = [], 0
    t0 = time.time()
    for si, (sym, ts) in enumerate(by_sym.items(), 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            dropped += len(ts); continue
        raw = pd.read_parquet(pq)
        tstamp = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
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
            highs = np.array([xh.get(h, np.nan) for h in HIGH_HMS])
            covers = np.array([xo.get(h, np.nan) for h in COVER_HMS])
            if t["category"] == "C":
                eg = by_date.get(ed)
                p = dict(zip(eg["hm"].values, eg["open"].values.astype(float))).get(HM_C_ENTRY, np.nan) if eg is not None else np.nan
                if not (p == p and p > 0):
                    dropped += 1; continue
                shares = float(np.floor(t["per_C"] / p))
                if shares <= 0:
                    dropped += 1; continue
                entry_price, cap = p, shares * p
            else:
                entry_price, shares, cap = t["entry_price"], t["shares"], t["cap"]
            if np.isnan(opens).any() or np.isnan(covers).any():
                dropped += 1; continue
            rows.append((entry_price, shares, cap, opens, highs, covers))
        if si % 300 == 0:
            print(f"  …{si}/{len(by_sym)} ({time.time()-t0:.0f}s), kept {len(rows):,}", flush=True)
    C = dict(entry=np.array([r[0] for r in rows], float),
             shares=np.array([r[1] for r in rows], float),
             cap=np.array([r[2] for r in rows], float),
             hs=np.ones(len(rows)),
             opens=np.vstack([r[3] for r in rows]),
             highs=np.vstack([r[4] for r in rows]),
             covers=np.vstack([r[5] for r in rows]))
    print(f"  kept {len(rows):,} trades; dropped {dropped:,} (missing long/cover grid).", flush=True)
    return C


def long_exit_price(C, structure, t1, t2, tgt, Ecache):
    """Per-trade long exit price (= short-open price) for a long config."""
    E = Ecache[tgt]
    i = int(np.where(EXIT_HMS == hm_of(t1))[0][0])
    if structure == "full":
        return E[:, i]
    j = int(np.where(EXIT_HMS == hm_of(t2))[0][0])
    pos = C["opens"][:, i] > C["entry"]
    return np.where(pos, E[:, i], E[:, j])


def metrics(pnl_g, pnl_a, pnl_b, cap):
    N = len(pnl_g)
    out = {}
    for tag, p in [("gross", pnl_g), ("net_A", pnl_a), ("net_B", pnl_b)]:
        r = p / cap * 100
        out[f"{tag}_total_return_fixedbase_pct"] = round(float(p.sum() / BASE_POOL * 100), 4)
        out[f"{tag}_total_pnl_inr"] = round(float(p.sum()), 0)
        out[f"{tag}_win_rate_pct"] = round(float((p > 0).mean() * 100), 2)
        out[f"{tag}_avg_return_per_trade_pct"] = round(float(r.mean()), 4)
        out[f"{tag}_median_return_per_trade_pct"] = round(float(np.median(r)), 4)
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # STEP 1 — long configs
    ls = pd.read_parquet(rb.RESULTS / "exit_sweep_3d" / "exit_sweep_3d_full.parquet")
    ls = ls[ls["n_trades"] >= N_MIN].sort_values("net_A_total_return_fixedbase_pct", ascending=False)
    top = ls.head(100)[["structure", "exit_time_1", "exit_time_2", "target_pct"]].copy()
    configs = [(r["structure"], r["exit_time_1"], r["exit_time_2"] or "", int(r["target_pct"]), False)
               for _, r in top.iterrows()]
    if not any(c[:4] == BASELINE for c in configs):
        configs.append((*BASELINE, True))
    else:
        configs = [(*c[:4], c[:4] == BASELINE) for c in configs]
    print(f"long configs: {len(configs)} (top100 + baseline)")

    C = build_cache2()
    N = len(C["entry"])
    Ecache = {t: X.E_matrix(C, t)[0] for t in sorted({c[3] for c in configs})}
    shares, cap, entry = C["shares"], C["cap"], C["entry"]
    covers = C["covers"]                                # N × 31

    # eligibility: every long exit <= 12:00 < 14:30 -> all shorts eligible
    print("short eligibility: all long exits <=12:00, all covers 14:30-15:00 -> 0 ineligible.")

    rows, long_best = [], {}
    t0 = time.time()
    for ci, (structure, t1, t2, tgt, is_base) in enumerate(configs, 1):
        lep = long_exit_price(C, structure, t1, t2, tgt, Ecache)     # short-open price
        long_pnl = shares * (lep - entry)
        short_notl = shares * lep
        long_cost_A, long_cost_B = R023 * cap, R038 * cap
        short_cost = SR * short_notl                                 # fixed across cover times
        cfg_key = f"{structure}|{t1}|{t2}|{tgt}"
        best_cov = None
        for ki, tc in enumerate(COVER_HMS):
            short_pnl = shares * (lep - covers[:, ki])
            comb = long_pnl + short_pnl
            pnl_g = comb
            pnl_a = comb - long_cost_A - short_cost
            pnl_b = comb - long_cost_B - short_cost
            m = metrics(pnl_g, pnl_a, pnl_b, cap)
            row = {"structure": structure, "exit_time_1": t1, "exit_time_2": t2, "target_pct": tgt,
                   "short_cover_time": lbl(int(tc)), "is_baseline": is_base, "n_trades": N,
                   "long_leg_total_pnl_inr": round(float(long_pnl.sum()), 0),
                   "long_leg_return_fixedbase_pct": round(float(long_pnl.sum() / BASE_POOL * 100), 4),
                   "short_leg_total_pnl_inr": round(float(short_pnl.sum()), 0),
                   "short_leg_return_fixedbase_pct": round(float(short_pnl.sum() / BASE_POOL * 100), 4),
                   **m}
            rows.append(row)
            key = m["net_A_total_return_fixedbase_pct"]
            if best_cov is None or key > best_cov[1]:
                best_cov = (lbl(int(tc)), key)
        long_best[cfg_key] = {"structure": structure, "t1": t1, "t2": t2, "target": tgt,
                              "is_baseline": is_base, "best_cover": best_cov[0], "best_netA": best_cov[1]}
        if ci % 25 == 0:
            print(f"  …{ci}/{len(configs)} configs ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values("net_A_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_parquet(OUTDIR / "joint_exit_cover_sweep.parquet", index=False)

    KEY = "net_A_total_return_fixedbase_pct"
    top20 = df.head(20)
    best = df.iloc[0]
    base_rows = df[df["is_baseline"]]
    base_best = base_rows.sort_values(KEY, ascending=False).iloc[0] if len(base_rows) else None

    # STEP 5 — cover-time stability across the top-20 LONG configs (by their best-cover net_A)
    lb = pd.DataFrame(long_best.values()).sort_values("best_netA", ascending=False).head(20)
    cov_counts = lb["best_cover"].value_counts()

    with pd.ExcelWriter(OUTDIR / "joint_exit_cover_sweep.xlsx", engine="openpyxl") as w:
        df.to_excel(w, sheet_name="all_3131", index=False)
        top20.to_excel(w, sheet_name="top20_joint", index=False)
        lb.to_excel(w, sheet_name="top20_longcfg_bestcover", index=False)
        if base_best is not None:
            pd.DataFrame([base_best]).to_excel(w, sheet_name="baseline_best_cover", index=False)
        for sht in w.sheets.values():
            for col in sht.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                sht.column_dimensions[col[0].column_letter].width = min(width + 2, 26)

    # STEP 6 — cover curve at the best long config
    bcfg = df[(df["structure"] == best["structure"]) & (df["exit_time_1"] == best["exit_time_1"]) &
              (df["exit_time_2"] == best["exit_time_2"]) & (df["target_pct"] == best["target_pct"])]
    bcfg = bcfg.sort_values("short_cover_time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bcfg["short_cover_time"], bcfg[KEY], "-o", color="#2E74B5")
    ax.set_xlabel("short cover time"); ax.set_ylabel("combined net_A total_return %")
    ax.set_title(f"Short-cover curve at best long config "
                 f"{best['exit_time_1']}->{best['exit_time_2']} @{best['target_pct']}% (C@15:21)", fontweight="bold")
    ax.tick_params(axis="x", rotation=90); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTDIR / "cover_curve_best_config.png", dpi=130); plt.close(fig)

    # ── console report ──
    pd.set_option("display.width", 240)
    print("\n" + "=" * 100 + "\nJOINT LONG-EXIT x SHORT-COVER SWEEP — RESULTS (net_A; full in Parquet/Excel)\n" + "=" * 100)
    print(f"trades: {N:,} | joint combos: {len(df):,} (101 long x 31 cover)")
    print(f"\nBEST JOINT: {best['structure']} {best['exit_time_1']}"
          f"{'/' + best['exit_time_2'] if best['exit_time_2'] else ''} @ {best['target_pct']}%  +  cover {best['short_cover_time']}")
    for tag in ["gross", "net_A", "net_B"]:
        print(f"   {tag:6s}: total_return {best[f'{tag}_total_return_fixedbase_pct']:.2f}%  "
              f"win {best[f'{tag}_win_rate_pct']}%  avg {best[f'{tag}_avg_return_per_trade_pct']}%  "
              f"median {best[f'{tag}_median_return_per_trade_pct']}%")
    print(f"   LEG DECOMP (gross): long {best['long_leg_return_fixedbase_pct']:.2f}%  "
          f"short {best['short_leg_return_fixedbase_pct']:.2f}%")
    if base_best is not None:
        print(f"\nBASELINE (09:45/12:00 @14%): best cover {base_best['short_cover_time']} -> "
              f"net_A {base_best[KEY]:.2f}%  (short leg {base_best['short_leg_return_fixedbase_pct']:.2f}%)")
        print(f"  best joint beats baseline-best-cover by {best[KEY]-base_best[KEY]:+.2f} pts (net_A).")
    print("\nTOP 10 JOINT (net_A):")
    print(top20.head(10)[["rank", "exit_time_1", "exit_time_2", "target_pct", "short_cover_time",
                          "net_A_total_return_fixedbase_pct", "net_A_win_rate_pct",
                          "long_leg_return_fixedbase_pct", "short_leg_return_fixedbase_pct"]].to_string(index=False))
    print(f"\nSTEP 5 — short-cover stability across top-20 long configs:")
    print("  individually-best cover time counts:", dict(cov_counts))
    if cov_counts.iloc[0] >= 16:
        print(f"  -> STABLE: {cov_counts.index[0]} is best for {cov_counts.iloc[0]}/20 long configs; "
              "short cover is ~independent of the long exit (separable).")
    else:
        print("  -> VARIES: best cover time depends on the long config; the joint grid mattered.")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
