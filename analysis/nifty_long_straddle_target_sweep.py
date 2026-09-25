# -*- coding: utf-8 -*-
"""nifty_long_straddle_target_sweep.py — per-DTE + per-entry-premium PROFIT-TARGET sweep on BOTH VWAP sub-
variants of the Long ATM Straddle VWAP (same_strike, rolling_atm). Target layered on existing exits (VWAP-
close-below / 3:20 forced); whichever fires first. TARGET = CLOSE-BASED (straddle close >= entry*(1+X%);
fill at the target level) - chosen over touch-based because the straddle's intraday high isn't cleanly
derivable from 1-min leg OHLC (legs peak at opposite moments). Sweep X=10..100% step 5 (+ baseline). Grids
DTE x target and premium-bucket x target, per variant; best region per row; same vs rolling at each best;
India VIX context; IS(<2025-09)/OOS split. GROSS premium points."""
import sys, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
from straddle_vwap_backtest import build_front_map, DTE_MAX

OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"; VIXF = rb.BASE / "data" / "india_vix_1min.csv"
OUTDIR = rb.RESULTS / "long_straddle_target_sweep"; OUTDIR.mkdir(parents=True, exist_ok=True)
FORCE_MOD = 15 * 60 + 20; TPCT = list(range(10, 101, 5)); TARGETS = [None] + [t / 100.0 for t in TPCT]
PREM_BUCKETS = [f"{i}-{i+20}" for i in range(0, 300, 20)] + [">300"]; OOS_SPLIT = pd.Timestamp("2025-08-29")
VIX_BUCKETS = ["<9"] + [f"{i}-{i+1}" for i in range(9, 20)] + [">20"]


def pbucket(p): return ">300" if p >= 300 else f"{int(p // 20) * 20}-{int(p // 20) * 20 + 20}"
def vbucket(v):
    if pd.isna(v): return "n/a"
    return "<9" if v < 9 else (">20" if v >= 20 else f"{int(v)}-{int(v)+1}")


def run_day_sweep(day_df):
    ce = day_df[day_df.option_type == "CE"]; pe = day_df[day_df.option_type == "PE"]
    times = np.array(sorted(day_df["timestamp"].unique())); strikes = np.array(sorted(day_df["strike"].unique()))
    ti = {t: i for i, t in enumerate(times)}; si = {s: j for j, s in enumerate(strikes)}; nT, nS = len(times), len(strikes)
    ri = ce["timestamp"].map(ti).values; ci = ce["strike"].map(si).values; rj = pe["timestamp"].map(ti).values; cj = pe["strike"].map(si).values
    dte = int(day_df["DTE"].iloc[0]); dat = pd.Timestamp(times[0]).date()

    def mat(sub, col, r, c):
        a = np.full((nT, nS), np.nan); a[r, c] = sub[col].values; return a
    CEc = mat(ce, "close", ri, ci); CEv = mat(ce, "volume", ri, ci); PEc = mat(pe, "close", rj, cj); PEv = mat(pe, "volume", rj, cj)
    CEsyn = np.ones((nT, nS), bool); CEsyn[ri, ci] = ce["is_synthetic"].values; PEsyn = np.ones((nT, nS), bool); PEsyn[rj, cj] = pe["is_synthetic"].values
    real = (~CEsyn) & (~PEsyn) & ~np.isnan(CEc) & ~np.isnan(PEc); Sc = CEc + PEc; Sv = np.nan_to_num(CEv) + np.nan_to_num(PEv); absdiff = np.abs(CEc - PEc)
    mod = np.array([pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute for t in times]); _w = np.where(mod <= FORCE_MOD)[0]; force_i = int(_w[-1]) if len(_w) else nT - 1

    def chain_atm(i):
        cand = real[i]
        return int(np.argmin(np.where(cand, absdiff[i], np.inf))) if cand.any() else None
    a0 = chain_atm(0)
    if a0 is None:
        for i in range(1, nT):
            a0 = chain_atm(i)
            if a0 is not None: break

    out = []
    for rolling in (False, True):
        for tf in TARGETS:
            cumPV = 0.0; cumV = 0.0; atm = a0; held = None; pos = 0; entry = np.nan; et = None
            for i in range(nT):
                feed = chain_atm(i) if rolling else atm
                if feed is not None and not np.isnan(Sc[i, feed]): cumPV += Sc[i, feed] * Sv[i, feed]; cumV += Sv[i, feed]
                vwap = (cumPV / cumV) if cumV > 0 else (Sc[i, feed] if feed is not None else np.nan)
                if i == force_i:
                    if pos == 1:
                        xp = Sc[i, held] if not np.isnan(Sc[i, held]) else entry
                        out.append((("rolling_atm" if rolling else "same_strike"), tf, dat, dte, round(entry, 2), round(xp - entry, 2), "3:20pm forced", int(mod[et])))
                    break
                if pos == 0:
                    if feed is not None and not np.isnan(Sc[i, feed]) and cumV > 0 and Sc[i, feed] > vwap:
                        k = feed if rolling else chain_atm(i)
                        if k is not None and not np.isnan(Sc[i, k]): held = k; atm = k; entry = Sc[i, k]; et = i; pos = 1
                else:
                    if tf is not None and not np.isnan(Sc[i, held]) and Sc[i, held] >= entry * (1 + tf):   # CLOSE-based target
                        out.append((("rolling_atm" if rolling else "same_strike"), tf, dat, dte, round(entry, 2), round(entry * tf, 2), "target", int(mod[et]))); pos = 0
                    elif feed is not None and not np.isnan(Sc[i, feed]) and Sc[i, feed] < vwap:
                        xp = Sc[i, held] if not np.isnan(Sc[i, held]) else entry
                        out.append((("rolling_atm" if rolling else "same_strike"), tf, dat, dte, round(entry, 2), round(xp - entry, 2), "VWAP-exit", int(mod[et]))); pos = 0
    return out


def process_expiry(args):
    exp, days = args; fs = list((OPTDIR / exp).glob("*.parquet")); cols = ["strike", "option_type", "timestamp", "DTE", "close", "volume", "is_synthetic"]; parts = []
    for f in fs:
        try:
            dd = pd.read_parquet(f, columns=cols); parts.append(dd[dd["DTE"].isin(range(0, DTE_MAX + 1))])
        except Exception: pass
    if not parts: return []
    chain = pd.concat(parts, ignore_index=True); chain["timestamp"] = pd.to_datetime(chain["timestamp"]); chain["day"] = chain["timestamp"].dt.normalize()
    out = []
    for day, dte in days:
        sub = chain[chain["day"] == pd.Timestamp(day)]
        if not sub.empty: out += run_day_sweep(sub.drop(columns=["day"]))
    return out


def main():
    front = build_front_map(); byexp = {}
    for day, (exp, dte) in front.items(): byexp.setdefault(exp, []).append((day, dte))
    print(f"days {len(front)} | expiries {len(byexp)} | targets {len(TARGETS)} x 2 variants")
    rows = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futs = {ex.submit(process_expiry, (e, d)): e for e, d in byexp.items()}; done = 0
        for f in as_completed(futs):
            rows += f.result(); done += 1
            if done % 20 == 0 or done == len(byexp): print(f"  {done}/{len(byexp)}")
    S = pd.DataFrame(rows, columns=["variant", "target", "date", "DTE", "entry_prem", "pnl", "exit_reason", "entry_mod"])
    S["target_pct"] = S["target"].map(lambda t: "baseline" if pd.isna(t) else int(round(t * 100)))
    S["prem_bucket"] = S["entry_prem"].map(pbucket); S["oos"] = pd.to_datetime(S["date"]) >= OOS_SPLIT
    # entry-minute India VIX per segment (asof on 1-min VIX, over unique date+minute pairs)
    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vser = vx.assign(ts=vts)[["ts", "close"]].dropna().sort_values("ts").rename(columns={"close": "entry_vix"})
    uniq = S[["date", "entry_mod"]].drop_duplicates().copy()
    uniq["ts"] = pd.to_datetime(uniq["date"].astype(str)) + pd.to_timedelta(uniq["entry_mod"], unit="m")
    uniq = pd.merge_asof(uniq.sort_values("ts"), vser, on="ts", direction="backward")
    S = S.merge(uniq[["date", "entry_mod", "entry_vix"]], on=["date", "entry_mod"], how="left")
    S["vix_bucket"] = S["entry_vix"].map(vbucket)
    # ---- FILTERS: VIX>=13 AND DTE in 2..6 ----
    VIX_MIN = 13.0; DTE_KEEP = [2, 3, 4, 5, 6]
    F = S[(S.entry_vix >= VIX_MIN) & (S.DTE.isin(DTE_KEEP))].copy()
    base = S[S.target_pct == "baseline"].copy()

    def grid(sub, rowkey, roworder):
        cols = ["baseline"] + TPCT
        piv = sub.pivot_table(index=rowkey, columns="target_pct", values="pnl", aggfunc="sum").round(1)
        piv = piv.reindex(index=[r for r in roworder if r in piv.index], columns=[c for c in cols if c in piv.columns])
        return piv.reset_index()

    def best_row(piv, rowkey):
        res = []
        for _, r in piv.iterrows():
            vals = r.drop(rowkey); v = vals.values.astype(float); nb = np.array([np.nanmean(v[max(0, i - 1):i + 2]) for i in range(len(v))])
            bi = int(np.nanargmax(nb)); res.append({rowkey: r[rowkey], "best_target": vals.index[bi], "best_pnl": round(float(v[bi]), 1),
                                                     "region": list(vals.index[max(0, bi - 1):bi + 2]), "baseline_pnl": round(float(vals.get("baseline", np.nan)), 1)})
        return pd.DataFrame(res)

    dord = list(range(DTE_MAX, -1, -1)); pord = PREM_BUCKETS
    grids = {}
    for v in ["same_strike", "rolling_atm"]:
        sv = S[S.variant == v]
        grids[f"{v}_DTE"] = grid(sv, "DTE", dord); grids[f"{v}_Prem"] = grid(sv, "prem_bucket", pord)

    # side-by-side best per DTE / per prem
    def compare(rowkey, roworder):
        out = []
        bss = {v: best_row(grids[f"{v}_" + ('DTE' if rowkey == 'DTE' else 'Prem')], rowkey).set_index(rowkey) for v in ["same_strike", "rolling_atm"]}
        for r in roworder:
            if r in bss["same_strike"].index and r in bss["rolling_atm"].index:
                a = bss["same_strike"].loc[r]; b = bss["rolling_atm"].loc[r]
                out.append({rowkey: r, "SS_best_target": a.best_target, "SS_best_pnl": a.best_pnl, "SS_baseline": a.baseline_pnl,
                            "RA_best_target": b.best_target, "RA_best_pnl": b.best_pnl, "RA_baseline": b.baseline_pnl})
        return pd.DataFrame(out)
    CMP_DTE = compare("DTE", dord); CMP_Prem = compare("prem_bucket", pord)

    # IS/OOS on DTE x target and Prem x target (total P&L) per variant
    def isoos(v, key, frame=None):
        sv = (S if frame is None else frame); sv = sv[sv.variant == v]; piv = sv.pivot_table(index=key, columns=["oos", "target_pct"], values="pnl", aggfunc="sum")
        cols = [(o, c) for o in (False, True) for c in (["baseline"] + TPCT) if (o, c) in piv.columns]
        return piv[cols].round(1)

    # VIX context from REAL entry-minute VIX (baseline trades)
    VIXsum = base.groupby(["variant", "vix_bucket"]).agg(trades=("pnl", "size"), win_pct=("pnl", lambda x: round((x > 0).mean() * 100, 1)),
                                                         total_pnl=("pnl", lambda x: round(x.sum(), 1)), avg=("pnl", lambda x: round(x.mean(), 2))).reset_index()

    # ================= FILTERED analysis (VIX>=13 & DTE 2-6) =================
    dord_f = [d for d in dord if d in DTE_KEEP]
    fgrids = {}
    for v in ["same_strike", "rolling_atm"]:
        fv = F[F.variant == v]
        fgrids[f"{v}_DTE"] = grid(fv, "DTE", dord_f); fgrids[f"{v}_Prem"] = grid(fv, "prem_bucket", pord)

    def rowstats(sub, rowkey, roworder):
        br = best_row(grid(sub, rowkey, roworder), rowkey).set_index(rowkey)
        cnt = sub[sub.target_pct == "baseline"].groupby(rowkey).size()
        return br, cnt

    def build_cmp(rowkey, ro_f):
        st = {}
        for v in ["same_strike", "rolling_atm"]:
            st[v] = (rowstats(S[S.variant == v], rowkey, dord if rowkey == "DTE" else pord),
                     rowstats(F[F.variant == v], rowkey, ro_f))
        out = []
        for r in ro_f:
            row = {rowkey: r}
            for v, p in [("same_strike", "SS"), ("rolling_atm", "RA")]:
                (brB, cB), (brA, cA) = st[v]
                gv = lambda br, r, c: (round(float(br.loc[r, c]), 1) if r in br.index else np.nan)
                row[f"{p}_base_before"] = gv(brB, r, "baseline_pnl"); row[f"{p}_base_after"] = gv(brA, r, "baseline_pnl")
                row[f"{p}_bestReg_before"] = gv(brB, r, "best_pnl"); row[f"{p}_bestReg_after"] = gv(brA, r, "best_pnl")
                row[f"{p}_bestT_after"] = (brA.loc[r, "best_target"] if r in brA.index else np.nan)
                row[f"{p}_n_before"] = int(cB.get(r, 0)); row[f"{p}_n_after"] = int(cA.get(r, 0))
            out.append(row)
        return pd.DataFrame(out)
    FCMP_DTE = build_cmp("DTE", dord_f); FCMP_Prem = build_cmp("prem_bucket", pord)
    FBEST_DTE = compareF = None

    def compare_f(rowkey, roworder):
        bss = {v: best_row(fgrids[f"{v}_" + ('DTE' if rowkey == 'DTE' else 'Prem')], rowkey).set_index(rowkey) for v in ["same_strike", "rolling_atm"]}
        out = []
        for r in roworder:
            if r in bss["same_strike"].index and r in bss["rolling_atm"].index:
                a = bss["same_strike"].loc[r]; b = bss["rolling_atm"].loc[r]
                out.append({rowkey: r, "SS_best_target": a.best_target, "SS_best_pnl": a.best_pnl, "SS_baseline": a.baseline_pnl,
                            "RA_best_target": b.best_target, "RA_best_pnl": b.best_pnl, "RA_baseline": b.baseline_pnl})
        return pd.DataFrame(out)
    FBEST_DTE = compare_f("DTE", dord_f); FBEST_Prem = compare_f("prem_bucket", pord)

    # filter-removal transparency (on baseline trade set)
    imp = []
    for v in ["same_strike", "rolling_atm", "BOTH"]:
        sub = base if v == "BOTH" else base[base.variant == v]
        tot = len(sub); vlow = int((sub.entry_vix < VIX_MIN).sum()); dlow = int(sub.DTE.isin([0, 1]).sum())
        ovl = int(((sub.entry_vix < VIX_MIN) & (sub.DTE.isin([0, 1]))).sum())
        ret = int(((sub.entry_vix >= VIX_MIN) & (sub.DTE.isin(DTE_KEEP))).sum())
        imp.append({"variant": v, "baseline_trades": tot, "removed_VIX<13": vlow, "removed_DTE_0_1": dlow,
                    "removed_overlap(both)": ovl, "removed_union": tot - ret, "retained": ret,
                    "retained_%": round(ret / tot * 100, 1) if tot else 0})
    FIMP = pd.DataFrame(imp)

    with pd.ExcelWriter(OUTDIR / "long_straddle_target_sweep_filtered.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Filters", "value": "entry India VIX >= 13 AND DTE in {2,3,4,5,6} (DTE 0,1 dropped). Applied to trade RESULTS only; strategy logic unchanged."},
            {"metric": "VIX source", "value": "entry-minute India VIX (asof 1-min), per segment"},
            {"metric": "Grids", "value": "*_DTE_f (DTE 2-6 x target total P&L) ; *_Prem_f (16 premium buckets x target). Compare_* = before(unfiltered) vs after(filtered)."},
            {"metric": "Target convention", "value": "CLOSE-based (unchanged from base sweep)"},
        ]).to_excel(w, sheet_name="About", index=False)
        FIMP.to_excel(w, sheet_name="Filter_Impact", index=False)
        FBEST_DTE.to_excel(w, sheet_name="Best_by_DTE_filt", index=False); FBEST_Prem.to_excel(w, sheet_name="Best_by_Prem_filt", index=False)
        FCMP_DTE.to_excel(w, sheet_name="Compare_DTE", index=False); FCMP_Prem.to_excel(w, sheet_name="Compare_Prem", index=False)
        for k, g in fgrids.items(): g.to_excel(w, sheet_name=("SS_" if k.startswith("same") else "RA_") + ("DTE_f" if k.endswith("DTE") else "Prem_f"), index=False)
        isoos("same_strike", "DTE", F).reindex(dord_f).to_excel(w, sheet_name="SS_DTE_IS_OOS_f"); isoos("rolling_atm", "DTE", F).reindex(dord_f).to_excel(w, sheet_name="RA_DTE_IS_OOS_f")
        VIXsum.to_excel(w, sheet_name="VIX_Context", index=False)

    with pd.ExcelWriter(OUTDIR / "long_straddle_target_sweep.xlsx", engine="openpyxl") as w:
        pd.DataFrame([
            {"metric": "Sweep", "value": "profit-target 10-100% step5 (+baseline) x 2 VWAP sub-variants x per-DTE & per-entry-premium buckets"},
            {"metric": "Target convention", "value": "CLOSE-based (straddle close>=entry*(1+X%); fill at target level). Chosen over touch-based - straddle intraday high not cleanly derivable from 1-min leg OHLC."},
            {"metric": "Whichever first", "value": "target / VWAP-close-below / 3:20 forced. Entry & existing exits unchanged."},
            {"metric": "GRIDS", "value": "*_DTE = DTE x target total P&L ; *_Prem = entry-premium bucket x target total P&L (per variant)"},
            {"metric": "OOS split", "value": str(OOS_SPLIT.date()) + " (IS<split, OOS>=split)"},
            {"metric": "VIX note", "value": "VIX context uses DAILY VIX close as a light proxy for the bucketed summary (entry-minute VIX available per trade in the base long-straddle report)."},
        ]).to_excel(w, sheet_name="About", index=False)
        CMP_DTE.to_excel(w, sheet_name="Best_by_DTE", index=False); CMP_Prem.to_excel(w, sheet_name="Best_by_Prem", index=False)
        for k, g in grids.items(): g.to_excel(w, sheet_name=("SS_" if k.startswith("same") else "RA_") + ("DTE" if k.endswith("DTE") else "Prem"), index=False)
        isoos("same_strike", "DTE").to_excel(w, sheet_name="SS_DTE_IS_OOS"); isoos("rolling_atm", "DTE").to_excel(w, sheet_name="RA_DTE_IS_OOS")
        isoos("same_strike", "prem_bucket").reindex(pord).to_excel(w, sheet_name="SS_Prem_IS_OOS"); isoos("rolling_atm", "prem_bucket").reindex(pord).to_excel(w, sheet_name="RA_Prem_IS_OOS")
        VIXsum.to_excel(w, sheet_name="VIX_Context", index=False)

    pd.set_option("display.width", 260)
    print("=" * 96 + "\nLONG STRADDLE VWAP - FILTERED (entry VIX>=13 & DTE 2-6)\n" + "=" * 96)
    print("\n--- FILTER IMPACT (on baseline trade set) ---"); print(FIMP.to_string(index=False))
    print("\n--- FILTERED best target per DTE (SS vs RA) ---"); print(FBEST_DTE.to_string(index=False))
    print("\n--- BEFORE vs AFTER per DTE (baseline P&L / best-region P&L / trades) ---")
    print(FCMP_DTE[["DTE", "SS_base_before", "SS_base_after", "SS_bestReg_before", "SS_bestReg_after", "SS_n_before", "SS_n_after",
                    "RA_base_before", "RA_base_after", "RA_bestReg_before", "RA_bestReg_after", "RA_n_before", "RA_n_after"]].to_string(index=False))
    print("\n--- FILTERED best target per PREMIUM bucket (SS vs RA) ---"); print(FBEST_Prem.to_string(index=False))
    print(f"\nSaved -> {OUTDIR} (long_straddle_target_sweep_filtered.xlsx)")


if __name__ == "__main__":
    main()
