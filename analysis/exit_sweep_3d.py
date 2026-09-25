# -*- coding: utf-8 -*-
"""
exit_sweep_3d.py — 3-D exit sweep on the composite UC + double-down strategy.
================================================================================
Category C entry FIXED at 15:21 (hm 921). Categories A (UC ladder 19%/17%) and B
(UC 3:00-3:15), the ₹5L/₹1L day-level capital sequencing, and the double-down short
(cover at 3:00pm on the exit day) are UNCHANGED. Costs: gross / net_A (long 0.23% +
short 0.10%) / net_B (long 0.38% + short 0.10%), proportional to filled value.

SWEEP (≈150,645 combos):
  EXIT TIME grid = 09:16..12:00, 1-min (hm 556..720) = 165 times (exit-day candle opens).
  TARGET %      = 10..20, 1% steps = 11 (limit-fill cover of the LONG at entry×(1+t/100)
                  when a 1-min HIGH reaches it, before the scheduled time exit).
  STRUCTURE 1 FULL EXIT : 165 × 11 = 1,815  (target if hit ≤ T, else 100% at T's open).
  STRUCTURE 2 SPLIT     : C(165,2)=13,530 pairs × 11 = 148,830  (positives at t1, rest at t2;
                          phased target-scan before t1 and t1→t2).

KEY PERFORMANCE IDEA — combined P&L is LINEAR in the exit price:
  combined_t = shares_t·(exit-entry) + [short]·shares_t·(exit-o3pm)
             = ω_t·exit − κ_t     (ω = share weight, κ = constant per trade)
  net_A/net_B differ only by fixed costs → also linear. So Σ(pnl), Σ(return) and the count
  of winners over ALL 13,530 pairs are obtained from two matmuls per target (notpos.T @ WE
  and notpos.T @ B), not a per-pair loop. Median is the only order-statistic → computed for
  leaderboard rows only. Entry side + each trade's exit-day 1-min series are cached ONCE.

Flags (all confirmed): (a) target 10-20 applies to both structures, limit-fill at
entry×(1+t/100) on 1-min high; (b) split = 100% positives@t1 / 100% rest@t2 (sign at t1),
phased target before t1 and t1→t2; (c) exit grid 09:16-12:00 1-min (165); (d) short unchanged
(covers 3pm), combined pnl throughout; (e) entry side fixed (15:21 Cat C), cached once.
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

OUTDIR = rb.RESULTS / "exit_sweep_3d"
IST = R.IST
HM_C_ENTRY = 921                                  # 15:21 Category-C entry
HM_3PM = 900                                      # short cover (exit day)
EXIT_HMS = np.arange(556, 721)                    # 09:16..12:00 (165) — exit/time grid
HIGH_HMS = np.arange(555, 721)                    # 09:15..12:00 (166) — target scan
TARGETS = list(range(4, 21))                      # 4..20 %  (extended below 10 to probe the boundary)
BASE_POOL = R.BASE_POOL
R023, R038, SR = R.LONG_023, R.LONG_038, R.SHORT_RATE
BASE_T1, BASE_T2, BASE_TGT = 585, 720, 14         # incumbent: 09:45 / 12:00 / 14%
N_MIN = 30                                         # small-sample filter


# ════════════════════════════════════════════════════════════════════════════
# ENTRY CACHE — trades (A/B fixed, C at 15:21) + exit-day 1-min series, built ONCE
# ════════════════════════════════════════════════════════════════════════════
def build_cache():
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_all_three",
                                "prev_day_vwap_close", "entry_price_315pm"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True]
    print("Scanning engine (A/B + per-day C allocation) …", flush=True)
    records = []
    for s in sorted(Q["symbol"].unique()):
        records += R.scan_symbol(s, Q[Q["symbol"] == s])
    T_def, day_diag = R.size_and_price(records)
    per_C = {d: v["per_C"] for d, v in day_diag.items()}

    # trade specs (entry side): A/B from the engine; C rebuilt at 15:21
    trades = []
    for _, r in T_def[T_def["category"].isin(["A", "B"])].iterrows():
        trades.append(dict(symbol=r["symbol"], entry_date=r["entry_date"], exit_date=r["exit_date"],
                           category=r["category"], entry_price=r["avg_entry"], shares=float(r["shares"]),
                           cap=float(r["capital_deployed"]), per_C=None))
    for r in records:
        if r["category"] == "C" and r["entered"] and per_C.get(r["entry_date"], 0) > 0:
            trades.append(dict(symbol=r["symbol"], entry_date=r["entry_date"], exit_date=r["next_date"],
                               category="C", entry_price=None, shares=None, cap=None,
                               per_C=per_C[r["entry_date"]]))
    print(f"  trade specs: {len(trades):,} (A/B fixed + C@15:21)", flush=True)

    # fetch exit-day 1-min series (+ 15:21 entry for C) — one parquet read per symbol
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
            o3pm = xo.get(HM_3PM, np.nan)
            if t["category"] == "C":                      # set 15:21 entry
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
            # require a COMPLETE exit-day time grid + 3pm (clean sweep; illiquid gaps dropped)
            if np.isnan(opens).any() or not (o3pm == o3pm):
                dropped += 1; continue
            has_short = 1.0
            rows.append((t["category"], entry_price, shares, cap, o3pm, has_short, opens, highs))
        if si % 300 == 0:
            print(f"  …{si}/{len(by_sym)} symbols ({time.time()-t0:.0f}s), kept {len(rows):,}", flush=True)

    cat = np.array([r[0] for r in rows])
    entry = np.array([r[1] for r in rows], float)
    shares = np.array([r[2] for r in rows], float)
    cap = np.array([r[3] for r in rows], float)
    o3pm = np.array([r[4] for r in rows], float)
    hs = np.array([r[5] for r in rows], float)
    opens = np.vstack([r[6] for r in rows])                 # N × 165
    highs = np.vstack([r[7] for r in rows])                 # N × 166
    print(f"  kept {len(rows):,} trades for the sweep; dropped {dropped:,} (missing exit grid / entry).", flush=True)
    return dict(cat=cat, entry=entry, shares=shares, cap=cap, o3pm=o3pm, hs=hs, opens=opens, highs=highs)


# ════════════════════════════════════════════════════════════════════════════
# per-target E matrix (exit price if scheduled at each grid time, target-if-hit)
# ════════════════════════════════════════════════════════════════════════════
def E_matrix(C, tgt):
    entry, opens, highs = C["entry"], C["opens"], C["highs"]
    level = entry * (1 + tgt / 100.0)                       # N,
    crossed = highs >= level[:, None]                       # N × 166
    ever = crossed.any(axis=1)
    first = np.argmax(crossed, axis=1)                      # first True (0 if none)
    hit_hm = np.where(ever, HIGH_HMS[first], 10 ** 9)       # minute of target hit (or huge)
    E = np.where(hit_hm[:, None] <= EXIT_HMS[None, :], level[:, None], opens)   # N × 165
    return E, level


def weights(C):
    w = C["shares"] * (1 + C["hs"])                         # gross exit weight ω
    wnet = w - SR * C["hs"] * C["shares"]                   # net exit weight (short cost part)
    base = C["shares"] * C["entry"] + C["hs"] * C["shares"] * C["o3pm"]     # κ_gross
    return {
        "gross": (w, base),
        "net_A": (wnet, base + R023 * C["cap"]),
        "net_B": (wnet, base + R038 * C["cap"]),
    }


def series_sums_full(E, ω, κ, cap, N):
    """Structure-1 (full exit): per grid time k, exit = E[:,k]. Returns arrays over 165 times."""
    WE = ω[:, None] * E                                     # N × 165
    tot = WE.sum(axis=0) - κ.sum()                          # 165,
    ret = ((ω / cap * 100)[:, None] * E).sum(axis=0) - (κ / cap * 100).sum()
    thr = κ / ω
    win = (E > thr[:, None]).sum(axis=0)                    # 165,
    return tot, ret / N, win / N * 100


def series_matrices_split(E, pos, ω, κ, cap, N):
    """Structure-2 (split): exit = pos? E[:,i] : E[:,j]. Returns 165×165 matrices [i,j]."""
    notp = (~pos).astype(np.float64)
    WE = ω[:, None] * E
    CPw = (pos * WE).sum(axis=0)                            # i-part
    tot = CPw[:, None] + notp.T @ WE - κ.sum()             # 165×165
    ωr = (ω / cap * 100)[:, None] * E
    CPr = (pos * ωr).sum(axis=0)
    sret = CPr[:, None] + notp.T @ ωr - (κ / cap * 100).sum()
    thr = κ / ω
    B = (E > thr[:, None]).astype(np.float64)
    CPb = (pos * B).sum(axis=0)
    cwin = CPb[:, None] + notp.T @ B                        # winners count
    return tot, sret / N, cwin / N * 100


def median_for(C, tgt, structure, i, j):
    """Exact median return per trade for ONE combo (leaderboard only)."""
    E, level = E_matrix(C, tgt)
    if structure == "full":
        exit_p = E[:, i]
    else:
        pos = C["opens"][:, i] > C["entry"]
        exit_p = np.where(pos, E[:, i], E[:, j])
    out = {}
    for s, (ω, κ) in weights(C).items():
        pnl = ω * exit_p - κ
        out[s] = round(float(np.median(pnl / C["cap"] * 100)), 4)
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    C = build_cache()
    N = len(C["entry"])
    W = weights(C)
    pos_all = C["opens"] > C["entry"][:, None]              # N × 165 (positive at each grid time)
    iu, ju = np.triu_indices(165, k=1)                     # 13,530 upper-tri pairs
    tlbl = {int(h): f"{h//60:02d}:{h%60:02d}" for h in np.concatenate([EXIT_HMS, HIGH_HMS])}

    full_rows, split_chunks = [], []
    best_split_ret = {}                                    # target -> 165×165 net_A total_return (for heatmap)
    t0 = time.time()
    for ti, tgt in enumerate(TARGETS, 1):
        E, _ = E_matrix(C, tgt)
        # STRUCTURE 1
        f = {}
        for s, (ω, κ) in W.items():
            tot, avg, win = series_sums_full(E, ω, κ, C["cap"], N)
            f[s] = (tot, avg, win)
        for k in range(len(EXIT_HMS)):
            full_rows.append({"structure": "full", "exit_time_1": tlbl[int(EXIT_HMS[k])],
                              "exit_time_2": "", "target_pct": tgt, "n_trades": N,
                              **{f"{s}_total_return_fixedbase_pct": round(f[s][0][k] / BASE_POOL * 100, 4) for s in W},
                              **{f"{s}_total_pnl_inr": round(f[s][0][k], 0) for s in W},
                              **{f"{s}_avg_return_per_trade_pct": round(f[s][1][k], 4) for s in W},
                              **{f"{s}_win_rate_pct": round(f[s][2][k], 2) for s in W}})
        # STRUCTURE 2
        mats = {}
        for s, (ω, κ) in W.items():
            mats[s] = series_matrices_split(E, pos_all, ω, κ, C["cap"], N)
        best_split_ret[tgt] = mats["net_A"][0] / BASE_POOL * 100
        chunk = {"structure": "split",
                 "exit_time_1": [tlbl[int(EXIT_HMS[i])] for i in iu],
                 "exit_time_2": [tlbl[int(EXIT_HMS[j])] for j in ju],
                 "target_pct": tgt, "n_trades": N}
        for s in W:
            tot, avg, win = mats[s]
            chunk[f"{s}_total_return_fixedbase_pct"] = np.round(tot[iu, ju] / BASE_POOL * 100, 4)
            chunk[f"{s}_total_pnl_inr"] = np.round(tot[iu, ju], 0)
            chunk[f"{s}_avg_return_per_trade_pct"] = np.round(avg[iu, ju], 4)
            chunk[f"{s}_win_rate_pct"] = np.round(win[iu, ju], 2)
        cdf = pd.DataFrame(chunk)
        cdf.to_parquet(OUTDIR / f"split_target_{tgt}.parquet", index=False)   # checkpoint per target
        split_chunks.append(cdf)
        el = time.time() - t0
        print(f"  [target {ti}/{len(TARGETS)}] {tgt}% done | {el:.0f}s | ETA {el/ti*(len(TARGETS)-ti):.0f}s", flush=True)

    full_df = pd.DataFrame(full_rows)
    split_df = pd.concat(split_chunks, ignore_index=True)
    all_df = pd.concat([full_df, split_df], ignore_index=True)
    all_df.to_parquet(OUTDIR / "exit_sweep_3d_full.parquet", index=False)

    # ── small-sample flag ──
    n_small = int((all_df["n_trades"] < N_MIN).sum())
    KEY = "net_A_total_return_fixedbase_pct"

    def add_median(df, structure):
        meds = []
        for _, r in df.iterrows():
            i = int(np.where(EXIT_HMS == int(r["exit_time_1"][:2]) * 60 + int(r["exit_time_1"][3:]))[0][0])
            j = 0 if structure == "full" else int(np.where(EXIT_HMS == int(r["exit_time_2"][:2]) * 60 + int(r["exit_time_2"][3:]))[0][0])
            meds.append(median_for(C, int(r["target_pct"]), structure, i, j))
        for s in W:
            df[f"{s}_median_return_per_trade_pct"] = [m[s] for m in meds]
        return df

    full_lb = add_median(full_df[full_df["n_trades"] >= N_MIN].sort_values(KEY, ascending=False).head(30).copy(), "full")
    split_lb = add_median(split_df[split_df["n_trades"] >= N_MIN].sort_values(KEY, ascending=False).head(30).copy(), "split")

    # ── baseline (09:45 / 12:00 / 14%, split) ──
    bl = split_df[(split_df["exit_time_1"] == "09:45") & (split_df["exit_time_2"] == "12:00")
                  & (split_df["target_pct"] == BASE_TGT)]
    bl = add_median(bl.copy(), "split").iloc[0] if len(bl) else None

    # ── overall best across both structures ──
    overall_best = all_df[all_df["n_trades"] >= N_MIN].sort_values(KEY, ascending=False).iloc[0]

    # ── save leaderboards ──
    with pd.ExcelWriter(OUTDIR / "exit_sweep_3d_leaderboards.xlsx", engine="openpyxl") as w:
        full_lb.to_excel(w, sheet_name="full_exit_top30", index=False)
        split_lb.to_excel(w, sheet_name="split_top30", index=False)
        if bl is not None:
            pd.DataFrame([bl]).to_excel(w, sheet_name="baseline_0945_1200_14", index=False)
        for sht in w.sheets.values():
            for col in sht.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                sht.column_dimensions[col[0].column_letter].width = min(width + 2, 30)

    # ── structure cuts: heatmap at best target, target-curve at best pair ──
    bsp = split_lb.iloc[0]
    bt = int(bsp["target_pct"])
    M = best_split_ret[bt].copy()
    M[np.tril_indices(165)] = np.nan                        # upper triangle only
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(M, origin="upper", cmap="RdYlGn", aspect="equal")
    tick = list(range(0, 165, 15))
    ax.set_xticks(tick); ax.set_xticklabels([tlbl[int(EXIT_HMS[k])] for k in tick], rotation=90, fontsize=7)
    ax.set_yticks(tick); ax.set_yticklabels([tlbl[int(EXIT_HMS[k])] for k in tick], fontsize=7)
    ax.set_xlabel("t2 (later exit)"); ax.set_ylabel("t1 (earlier exit)")
    ax.set_title(f"Split net_A total_return %  (target {bt}%, C@15:21)", fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="net_A total_return %")
    bi = int(np.where(EXIT_HMS == int(bsp["exit_time_1"][:2]) * 60 + int(bsp["exit_time_1"][3:]))[0][0])
    bj = int(np.where(EXIT_HMS == int(bsp["exit_time_2"][:2]) * 60 + int(bsp["exit_time_2"][3:]))[0][0])
    ax.scatter([bj], [bi], marker="*", s=180, c="black")
    fig.tight_layout(); fig.savefig(OUTDIR / "split_t1_t2_heatmap.png", dpi=130); plt.close(fig)

    # target curve at best pair
    tcurve = [float(split_df[(split_df["exit_time_1"] == bsp["exit_time_1"]) &
                             (split_df["exit_time_2"] == bsp["exit_time_2"]) &
                             (split_df["target_pct"] == t)][KEY].iloc[0]) for t in TARGETS]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(TARGETS, tcurve, "-o", color="#2E74B5")
    ax.axvline(bt, color="grey", ls=":"); ax.set_xlabel("target %"); ax.set_ylabel("net_A total_return %")
    ax.set_title(f"Target sweep at best pair {bsp['exit_time_1']}→{bsp['exit_time_2']} (C@15:21)", fontweight="bold")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUTDIR / "target_curve_best_pair.png", dpi=130); plt.close(fig)

    # ── console report ──
    pd.set_option("display.width", 240)
    print("\n" + "=" * 100 + "\n3-D EXIT SWEEP — RESULTS (net_A; full gross/net_A/net_B in Parquet/Excel)\n" + "=" * 100)
    print(f"trades in sweep: {N:,} | combos: full {len(full_df):,} + split {len(split_df):,} = {len(all_df):,}"
          f" | combos with n<{N_MIN}: {n_small}")
    fb = full_lb.iloc[0]
    print(f"\nBEST FULL-EXIT : {fb['exit_time_1']} @ target {fb['target_pct']}%  ->  "
          f"net_A total_return {fb[KEY]:.2f}%  (win {fb['net_A_win_rate_pct']}%, avg {fb['net_A_avg_return_per_trade_pct']}%)")
    print(f"BEST SPLIT     : {bsp['exit_time_1']} / {bsp['exit_time_2']} @ target {bsp['target_pct']}%  ->  "
          f"net_A total_return {bsp[KEY]:.2f}%  (win {bsp['net_A_win_rate_pct']}%, avg {bsp['net_A_avg_return_per_trade_pct']}%)")
    print(f"OVERALL BEST   : {overall_best['structure']}  {overall_best['exit_time_1']}"
          f"{'/' + overall_best['exit_time_2'] if overall_best['exit_time_2'] else ''} @ {overall_best['target_pct']}%  ->  "
          f"net_A {overall_best[KEY]:.2f}%")
    if bl is not None:
        print(f"\nBASELINE (09:45/12:00 @14%): net_A total_return {bl[KEY]:.2f}%  "
              f"(win {bl['net_A_win_rate_pct']}%)")
        print(f"  best split beats baseline by {bsp[KEY]-bl[KEY]:+.2f} pts (net_A total return).")
    print("\nTOP 8 SPLIT (net_A):")
    print(split_lb.head(8)[["exit_time_1", "exit_time_2", "target_pct", "net_A_total_return_fixedbase_pct",
                            "net_A_win_rate_pct", "net_A_avg_return_per_trade_pct"]].to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
