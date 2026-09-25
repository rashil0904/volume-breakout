# -*- coding: utf-8 -*-
"""
exit_sweep_wide.py — long-exit sweep over 09:20-12:00 with targets 4-20% + NO-TARGET.
Runs ONCE over the wider window, then writes TWO Excel files (no re-run):
  FILE 1  exit_sweep_920_1200.xlsx  — full 09:20-12:00 grid (161 times).
  FILE 2  exit_sweep_930_1200.xlsx  — same results filtered to combos whose exit times are ALL
                                      >= 09:30 (full T>=09:30; split both t1,t2>=09:30), re-ranked.
Composite strategy: Category C entry FIXED 15:21, A/B unchanged, double-down short on filled qty
covered at 3:00pm (exit day), gross / net_A (long0.23%+short0.10%) / net_B (long0.38%+short0.10%).

Combined P&L is LINEAR in the long exit price -> total/avg/win over all 12,880 split pairs come
from two matmuls per target (median = leaderboard only). Entry side + per-trade exit-day 1-min
series cached ONCE. Checkpoint per target. Full 234,738-row results saved to Parquet.
"""
import sys, time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import uc_staggered_dd_report as R

OUTDIR = rb.RESULTS / "exit_sweep_wide"
IST = R.IST
EXIT_HMS = np.arange(560, 721)                    # 09:20..12:00 (161)
HIGH_HMS = np.arange(555, 721)                     # 09:15..12:00 (166) target scan
TARGETS = list(range(4, 21))                       # 4..20
SETTINGS = [("%d%%" % t, t) for t in TARGETS] + [("no_target", None)]   # 18 settings
HM_C_ENTRY, HM_3PM = 921, 900                      # 15:21 entry ; 3pm short cover
RESTRICT_HM = 570                                  # 09:30 for File 2
BASE_POOL, R023, R038, SR = R.BASE_POOL, R.LONG_023, R.LONG_038, R.SHORT_RATE
BASE = ("split", "09:45", "12:00", "14%")          # incumbent
N_MIN = 30
KEY = "net_A_total_return_fixedbase_pct"


def lbl(hm):
    return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def hm_of(s):
    return int(s[:2]) * 60 + int(s[3:])


# ── entry side + exit-day series cache (short cover 3pm) ─────────────────────
def build_cache():
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
            if t["per_C"] is not None:
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
            if np.isnan(opens).any() or not (o3pm == o3pm):
                dropped += 1; continue
            rows.append((entry_price, shares, cap, o3pm, opens, highs))
        if si % 300 == 0:
            print(f"  …{si}/{len(by_sym)} ({time.time()-t0:.0f}s), kept {len(rows):,}", flush=True)
    C = dict(entry=np.array([r[0] for r in rows], float), shares=np.array([r[1] for r in rows], float),
             cap=np.array([r[2] for r in rows], float), o3pm=np.array([r[3] for r in rows], float),
             hs=np.ones(len(rows)), opens=np.vstack([r[4] for r in rows]), highs=np.vstack([r[5] for r in rows]))
    print(f"  kept {len(rows):,} trades; dropped {dropped:,}.", flush=True)
    return C


def E_matrix(C, tgt):
    if tgt is None:                                    # no-target: pure time exit
        return C["opens"], None
    level = C["entry"] * (1 + tgt / 100.0)
    crossed = C["highs"] >= level[:, None]
    ever = crossed.any(axis=1); first = np.argmax(crossed, axis=1)
    hit_hm = np.where(ever, HIGH_HMS[first], 10 ** 9)
    E = np.where(hit_hm[:, None] <= EXIT_HMS[None, :], level[:, None], C["opens"])
    return E, level


def weights(C):
    w = C["shares"] * (1 + C["hs"]); wnet = w - SR * C["hs"] * C["shares"]
    base = C["shares"] * C["entry"] + C["hs"] * C["shares"] * C["o3pm"]
    return {"gross": (w, base), "net_A": (wnet, base + R023 * C["cap"]), "net_B": (wnet, base + R038 * C["cap"])}


def full_series(E, ω, κ, cap, N):
    tot = (ω[:, None] * E).sum(axis=0) - κ.sum()
    ret = ((ω / cap * 100)[:, None] * E).sum(axis=0) - (κ / cap * 100).sum()
    win = (E > (κ / ω)[:, None]).sum(axis=0)
    return tot, ret / N, win / N * 100


def split_matrices(E, pos, ω, κ, cap, N):
    notp = (~pos).astype(np.float64); WE = ω[:, None] * E
    tot = (pos * WE).sum(axis=0)[:, None] + notp.T @ WE - κ.sum()
    ωr = (ω / cap * 100)[:, None] * E
    sret = (pos * ωr).sum(axis=0)[:, None] + notp.T @ ωr - (κ / cap * 100).sum()
    B = (E > (κ / ω)[:, None]).astype(np.float64)
    cwin = (pos * B).sum(axis=0)[:, None] + notp.T @ B
    return tot, sret / N, cwin / N * 100


def leg_decomp(C, structure, i, j, tgt):
    """(long_return%, short_return%) fixedbase for one combo (gross)."""
    E, _ = E_matrix(C, tgt)
    exit_p = E[:, i] if structure == "full" else np.where(C["opens"][:, i] > C["entry"], E[:, i], E[:, j])
    lp = C["shares"] * (exit_p - C["entry"]); sp = C["shares"] * (exit_p - C["o3pm"])
    return round(float(lp.sum() / BASE_POOL * 100), 4), round(float(sp.sum() / BASE_POOL * 100), 4)


def median_for(C, structure, i, j, tgt):
    E, _ = E_matrix(C, tgt)
    exit_p = E[:, i] if structure == "full" else np.where(C["opens"][:, i] > C["entry"], E[:, i], E[:, j])
    out = {}
    for s, (ω, κ) in weights(C).items():
        out[s] = round(float(np.median((ω * exit_p - κ) / C["cap"] * 100)), 4)
    return out


def idx(t):
    return int(np.where(EXIT_HMS == hm_of(t))[0][0])


def enrich(df, C):
    """Add median + leg decomposition to a small leaderboard frame."""
    meds, lg, sh = [], [], []
    for _, r in df.iterrows():
        tgt = None if r["target_label"] == "no_target" else int(r["target_label"][:-1])
        i = idx(r["exit_time_1"]); j = idx(r["exit_time_2"]) if r["exit_time_2"] else 0
        meds.append(median_for(C, r["structure"], i, j, tgt))
        L, S = leg_decomp(C, r["structure"], i, j, tgt); lg.append(L); sh.append(S)
    for s in ["gross", "net_A", "net_B"]:
        df[f"{s}_median_return_per_trade_pct"] = [m[s] for m in meds]
    df["long_leg_return_fixedbase_pct"] = lg; df["short_leg_return_fixedbase_pct"] = sh
    return df


def style(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, d in sheets.items():
            d.to_excel(w, sheet_name=name[:31], index=(name.startswith("view_t1t2")))
            sh = w.sheets[name[:31]]
            for c in sh[1]:
                c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="2E74B5")
                c.alignment = Alignment(horizontal="center", wrap_text=True)
            for col in sh.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                sh.column_dimensions[col[0].column_letter].width = min(width + 2, 24)


def make_file(df, mats, C, path, label):
    sub = df.copy()
    full = sub[sub["structure"] == "full"]
    split = sub[sub["structure"] == "split"]
    full_lb = enrich(full[full["n_trades"] >= N_MIN].sort_values(KEY, ascending=False).head(30).copy(), C)
    split_lb = enrich(split[split["n_trades"] >= N_MIN].sort_values(KEY, ascending=False).head(30).copy(), C)
    overall = enrich(sub[sub["n_trades"] >= N_MIN].sort_values(KEY, ascending=False).head(1).copy(), C)

    # target vs no-target (best split per setting)
    tvn = (split.groupby("target_label")[KEY].max().reset_index()
           .rename(columns={KEY: "best_split_net_A_total_return_pct"}))
    tvn["order"] = tvn["target_label"].map(lambda s: 999 if s == "no_target" else int(s[:-1]))
    tvn = tvn.sort_values("order").drop(columns="order")

    # baseline
    bl = split[(split["exit_time_1"] == BASE[1]) & (split["exit_time_2"] == BASE[2]) & (split["target_label"] == BASE[3])]
    bl = enrich(bl.copy(), C) if len(bl) else pd.DataFrame()
    bcmp = pd.concat([overall.assign(which="overall_best"),
                      (bl.assign(which="baseline_0945_1200_14") if len(bl) else pd.DataFrame())], ignore_index=True)

    # structure views: t1×t2 heatmap DATA at the file's best split target + return-vs-target
    bt = split_lb.iloc[0]["target_label"]
    M = mats[bt].copy()                                 # 161×161 net_A total_return, full-grid indices
    keep = [k for k, h in enumerate(EXIT_HMS) if (label == "full" or h >= RESTRICT_HM)]
    times = [lbl(EXIT_HMS[k]) for k in keep]
    Msub = np.full((len(keep), len(keep)), np.nan)
    for a, i in enumerate(keep):
        for b, j in enumerate(keep):
            if j > i:
                Msub[a, b] = M[i, j]
    heat = pd.DataFrame(Msub, index=[f"t1={t}" for t in times], columns=[f"t2={t}" for t in times])

    sheets = {"full_exit_leaderboard": full_lb, "split_leaderboard": split_lb, "overall_best": overall,
              "target_vs_notarget": tvn, "baseline_comparison": bcmp,
              "view_return_vs_target": tvn, "view_t1t2_heatmap_bestT": heat}
    style(path, sheets)
    return overall.iloc[0], (bl.iloc[0] if len(bl) else None)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    C = build_cache()
    N = len(C["entry"])
    W = weights(C)
    pos_all = C["opens"] > C["entry"][:, None]
    iu, ju = np.triu_indices(len(EXIT_HMS), k=1)         # 12,880 pairs
    full_rows, split_chunks, mats = [], [], {}
    t0 = time.time()
    for si, (slabel, tgt) in enumerate(SETTINGS, 1):
        E, _ = E_matrix(C, tgt)
        fmet = {s: full_series(E, ω, κ, C["cap"], N) for s, (ω, κ) in W.items()}
        for k in range(len(EXIT_HMS)):
            row = {"structure": "full", "exit_time_1": lbl(EXIT_HMS[k]), "exit_time_2": "",
                   "target_label": slabel, "n_trades": N}
            for s in W:
                tot, avg, win = fmet[s]
                row[f"{s}_total_return_fixedbase_pct"] = round(tot[k] / BASE_POOL * 100, 4)
                row[f"{s}_total_pnl_inr"] = round(tot[k], 0)
                row[f"{s}_avg_return_per_trade_pct"] = round(avg[k], 4)
                row[f"{s}_win_rate_pct"] = round(win[k], 2)
            full_rows.append(row)
        chunk = {"structure": "split", "exit_time_1": [lbl(EXIT_HMS[i]) for i in iu],
                 "exit_time_2": [lbl(EXIT_HMS[j]) for j in ju], "target_label": slabel, "n_trades": N}
        for s, (ω, κ) in W.items():
            tot, avg, win = split_matrices(E, pos_all, ω, κ, C["cap"], N)
            if s == "net_A":
                mats[slabel] = tot / BASE_POOL * 100
            chunk[f"{s}_total_return_fixedbase_pct"] = np.round(tot[iu, ju] / BASE_POOL * 100, 4)
            chunk[f"{s}_total_pnl_inr"] = np.round(tot[iu, ju], 0)
            chunk[f"{s}_avg_return_per_trade_pct"] = np.round(avg[iu, ju], 4)
            chunk[f"{s}_win_rate_pct"] = np.round(win[iu, ju], 2)
        cdf = pd.DataFrame(chunk)
        cdf.to_parquet(OUTDIR / f"split_{slabel.replace('%','pct')}.parquet", index=False)
        split_chunks.append(cdf)
        el = time.time() - t0
        print(f"  [{si}/{len(SETTINGS)}] target {slabel} | {el:.0f}s | ETA {el/si*(len(SETTINGS)-si):.0f}s", flush=True)

    full_df = pd.DataFrame(full_rows)
    all_df = pd.concat([full_df, pd.concat(split_chunks, ignore_index=True)], ignore_index=True)
    all_df.to_parquet(OUTDIR / "exit_sweep_wide_full.parquet", index=False)
    print(f"combos: full {len(full_df):,} + split {len(all_df)-len(full_df):,} = {len(all_df):,}")

    # FILE 1 — full 09:20-12:00
    b1, bl1 = make_file(all_df, mats, C, OUTDIR / "exit_sweep_920_1200.xlsx", "full")
    # FILE 2 — restricted to all exit times >= 09:30
    def ok(r):
        if hm_of(r["exit_time_1"]) < RESTRICT_HM:
            return False
        return not (r["structure"] == "split" and hm_of(r["exit_time_2"]) < RESTRICT_HM)
    df930 = all_df[all_df.apply(ok, axis=1)].reset_index(drop=True)
    b2, bl2 = make_file(df930, mats, C, OUTDIR / "exit_sweep_930_1200.xlsx", "930")

    pd.set_option("display.width", 240)
    print("\n" + "=" * 100 + "\nWIDER EXIT SWEEP — BEST OF EACH FILE (net_A combined long+short)\n" + "=" * 100)
    def show(tag, b):
        pair = f"{b['exit_time_1']}" + (f"/{b['exit_time_2']}" if b["exit_time_2"] else "")
        print(f"{tag:26s}: {b['structure']:5s} {pair:12s} @ {b['target_label']:9s} -> "
              f"net_A {b[KEY]:.2f}%  (win {b['net_A_win_rate_pct']}%, avg {b['net_A_avg_return_per_trade_pct']}%, "
              f"long {b['long_leg_return_fixedbase_pct']:.1f}%/short {b['short_leg_return_fixedbase_pct']:.1f}%)")
    show("FILE1 best (09:20-12:00)", b1)
    show("FILE2 best (09:30-12:00)", b2)
    same = (b1["structure"], b1["exit_time_1"], b1["exit_time_2"], b1["target_label"]) == \
           (b2["structure"], b2["exit_time_1"], b2["exit_time_2"], b2["target_label"])
    uses_early = hm_of(b1["exit_time_1"]) < RESTRICT_HM or (b1["exit_time_2"] and hm_of(b1["exit_time_2"]) < RESTRICT_HM)
    print(f"\n-> File1 best {'USES' if uses_early else 'does NOT use'} a 09:20-09:29 exit; "
          f"bests {'COINCIDE' if same else 'DIFFER'} by {b1[KEY]-b2[KEY]:+.2f} net_A pts.")
    if bl1 is not None:
        print(f"baseline (09:45/12:00 @14%): net_A {bl1[KEY]:.2f}%  |  File1 best beats it by {b1[KEY]-bl1[KEY]:+.2f} pts.")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
