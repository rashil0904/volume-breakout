# -*- coding: utf-8 -*-
"""
variant_sl_realistic_fills.py
=============================
CORRECTED −1% stop variants with REALISTIC fills (supersedes variant_sl_low_vs_close.py,
which fantasy-capped every stop at exactly −1%). Main strategy Bucket N only. Base:
mcap ₹1,500-5,000 Cr, LB 36, VM 6, 3:15pm entry, +5%, ₹5L/₹1L. Reuse canonical trades.
L = entry * 0.99.

STEP 0 (both stop variants): if return_at_945 <= -1 (already at/below -1% at the 09:45
  price) -> exit at the 09:45 OPEN, actual (uncapped) loss. "sl_1pct_already_breached_at_945".
SL-LOW  : Bucket N not-yet-breached -> first candle low<=L; fill = min(L, candle_open)
          (gap-through fills worse than -1%). "sl_1pct_low". Else 12:00.
SL-CLOSE: first candle close<=L; fill = candle close. "sl_1pct_close". Else 12:00.
14% target throughout, priority over stop. Positives exit @09:45 in all variants.

Flags: (a) already-breached exits at 09:45 open, full actual loss. (b) SL-LOW fill=min(L,
open). (c) SL-CLOSE fill=breaching candle close. (d) already-breached uses the 09:45 return.
"""
import sys, bisect
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets

IST = "Asia/Kolkata"
OUTDIR = rb.RESULTS / "variant_sl_realistic"
BASE_POOL, EXPENSE = 500_000, 0.0023
TARGET, SL_PCT = 14.0, 1.0
CANDLE_HMS = list(range(555, 901, 15))
HCOL = {hm: k for k, hm in enumerate(CANDLE_HMS)}
HM_0945, HM_1200 = 585, 720
PRE945 = [hm for hm in CANDLE_HMS if hm < HM_0945]
SCAN = [hm for hm in CANDLE_HMS if HM_0945 < hm < HM_1200]


def fetch_ohlc(base):
    n = len(base)
    A = {c: np.full((n, len(CANDLE_HMS)), np.nan) for c in ("O", "H", "L", "C")}
    for sym, grp in base.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        ds = sorted(raw["date"].unique())
        sub = raw[raw["hm"].isin(CANDLE_HMS)]
        piv = {k: sub.pivot_table(index="date", columns="hm", values=v, aggfunc="last").reindex(columns=CANDLE_HMS)
               for k, v in [("O", "open"), ("H", "high"), ("L", "low"), ("C", "close")]}
        for idx, ed in zip(grp.index, grp["date"]):
            j = bisect.bisect_right(ds, ed.date())
            if j >= len(ds):
                continue
            nd = ds[j]
            if nd in piv["O"].index:
                for k in A:
                    A[k][idx, :] = piv[k].loc[nd].values
    return A["O"], A["H"], A["L"], A["C"]


def _pre945(e, O, H):
    tgt = e * (1 + TARGET / 100)
    for hm in PRE945:
        v = H[HCOL[hm]]
        if not np.isnan(v) and v >= tgt:
            return "early_target_pre_945", tgt, None
    o945 = O[HCOL[HM_0945]]
    if not np.isnan(o945) and (o945 - e) / e * 100 > 0:
        return "positive_at_945", o945, None
    return None, None, o945          # Bucket N (return_at_945 <= 0, or o945 nan)


def variant_A(e, O, H):
    et, px, _ = _pre945(e, O, H)
    if et:
        return et, px
    tgt = e * (1 + TARGET / 100)
    for hm in SCAN:
        if not np.isnan(H[HCOL[hm]]) and H[HCOL[hm]] >= tgt:
            return "early_target_945_1200", tgt
    o = O[HCOL[HM_1200]]
    return ("exit_at_1200_no_trigger", o) if not np.isnan(o) else (None, np.nan)


def variant_stop(e, O, H, L_, C, basis):
    et, px, o945 = _pre945(e, O, H)
    if et:
        return et, px
    tgt = e * (1 + TARGET / 100)
    Lx = e * (1 - SL_PCT / 100)
    # STEP 0 — already breached at 09:45 -> exit at 09:45 open, actual loss (uncapped)
    if not np.isnan(o945) and (o945 - e) / e * 100 <= -SL_PCT:
        return "sl_1pct_already_breached_at_945", o945
    # not-yet-breached Bucket N -> scan
    for hm in SCAN:
        h = H[HCOL[hm]]
        if not np.isnan(h) and h >= tgt:                        # target priority
            return "early_target_945_1200", tgt
        if basis == "low":
            l, op = L_[HCOL[hm]], O[HCOL[hm]]
            if not np.isnan(l) and l <= Lx:
                fill = min(Lx, op) if not np.isnan(op) else Lx   # gap-through: worse of -1% / open
                return "sl_1pct_low", fill
        else:
            c = C[HCOL[hm]]
            if not np.isnan(c) and c <= Lx:
                return "sl_1pct_close", c
    o = O[HCOL[HM_1200]]
    return ("exit_at_1200_no_trigger", o) if not np.isnan(o) else (None, np.nan)


def blk(pnl, ret, cap):
    w, l = pnl > 0, pnl <= 0
    return dict(n_trades=len(pnl),
                total_return_fixedbase_pct=round(pnl.sum() / BASE_POOL * 100, 4),
                total_pnl_inr=round(pnl.sum(), 0),
                win_rate_pct=round((pnl > 0).mean() * 100, 2),
                avg_return_per_trade_pct=round(ret.mean(), 4),
                median_return_per_trade_pct=round(np.median(ret), 4),
                avg_return_winning_trades_pct=round(ret[w].mean(), 4) if w.any() else np.nan,
                avg_return_losing_trades_pct=round(ret[l].mean(), 4) if l.any() else np.nan,
                worst_single_trade_return_pct=round(ret.min(), 4),
                avg_capital_deployed_per_trade=round(cap.mean(), 0))


def metrics(name, e, sh, cap, px, valid):
    e, sh, cap, px = e[valid], sh[valid], cap[valid], px[valid]
    gp = sh * (px - e); gret = (px - e) / e * 100
    npl = gp - cap * EXPENSE; nret = gret - EXPENSE * 100
    return ({"variant": name, "basis": "gross", **blk(gp, gret, cap)},
            {"variant": name, "basis": "net", **blk(npl, nret, cap)})


def exit_table(name, et, e, sh, px, valid, cats):
    e, sh, px, et = e[valid], sh[valid], px[valid], et[valid]
    ret = (px - e) / e * 100
    rows = []
    for c in cats:
        m = et == c
        rows.append({"variant": name, "exit_type": c, "n": int(m.sum()),
                     "pct": round(m.mean() * 100, 2) if len(et) else 0,
                     "avg_return_pct": round(ret[m].mean(), 4) if m.any() else np.nan,
                     "median_return_pct": round(np.median(ret[m]), 4) if m.any() else np.nan,
                     "min_return_pct": round(ret[m].min(), 4) if m.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = ets.load_base_positions()
    e = base["entry"].values.astype(float)
    sh = base["shares"].values.astype(float)
    cap = base["cap"].values.astype(float)
    print(f"Base positions: {len(base):,} | fetching next-day OHLC …")
    O, H, L, C = fetch_ohlc(base)

    n = len(base)
    a_et = np.empty(n, dtype=object); a_px = np.full(n, np.nan)
    lo_et = np.empty(n, dtype=object); lo_px = np.full(n, np.nan)
    cl_et = np.empty(n, dtype=object); cl_px = np.full(n, np.nan)
    bucketN = np.zeros(n, dtype=bool)
    for i in range(n):
        a_et[i], a_px[i] = variant_A(e[i], O[i], H[i])
        lo_et[i], lo_px[i] = variant_stop(e[i], O[i], H[i], L[i], C[i], "low")
        cl_et[i], cl_px[i] = variant_stop(e[i], O[i], H[i], L[i], C[i], "close")
        pre, _, _ = _pre945(e[i], O[i], H[i])
        bucketN[i] = pre is None
    av = np.array([x is not None for x in a_et]) & ~np.isnan(a_px)
    lv = np.array([x is not None for x in lo_et]) & ~np.isnan(lo_px)
    cv = np.array([x is not None for x in cl_et]) & ~np.isnan(cl_px)

    # ── 4. three-way comparison ──
    comp = pd.DataFrame([*metrics("A_baseline", e, sh, cap, a_px, av),
                         *metrics("SL_LOW", e, sh, cap, lo_px, lv),
                         *metrics("SL_CLOSE", e, sh, cap, cl_px, cv)]).sort_values(
        ["basis", "variant"]).reset_index(drop=True)

    A_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200", "exit_at_1200_no_trigger"]
    L_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200",
              "sl_1pct_already_breached_at_945", "sl_1pct_low", "exit_at_1200_no_trigger"]
    C_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200",
              "sl_1pct_already_breached_at_945", "sl_1pct_close", "exit_at_1200_no_trigger"]
    exit_tbl = pd.concat([exit_table("A_baseline", a_et, e, sh, a_px, av, A_CATS),
                          exit_table("SL_LOW", lo_et, e, sh, lo_px, lv, L_CATS),
                          exit_table("SL_CLOSE", cl_et, e, sh, cl_px, cv, C_CATS)], ignore_index=True)

    # ── 5. Bucket-N paired ──
    pm = bucketN & av & lv & cv
    def rr(px): return (px[pm] - e[pm]) / e[pm] * 100
    aR, lR, cR = rr(a_px), rr(lo_px), rr(cl_px)
    aP = sh[pm]*(a_px[pm]-e[pm]); lP = sh[pm]*(lo_px[pm]-e[pm]); cP = sh[pm]*(cl_px[pm]-e[pm])
    paired = pd.DataFrame([
        {"treatment": "A_hold_to_1200", "n": int(pm.sum()), "avg_return_pct": round(aR.mean(),4),
         "median_return_pct": round(np.median(aR),4), "worst_return_pct": round(aR.min(),4),
         "win_rate_pct": round((aP>0).mean()*100,2), "total_pnl_inr": round(aP.sum(),0)},
        {"treatment": "SL_LOW_realistic", "n": int(pm.sum()), "avg_return_pct": round(lR.mean(),4),
         "median_return_pct": round(np.median(lR),4), "worst_return_pct": round(lR.min(),4),
         "win_rate_pct": round((lP>0).mean()*100,2), "total_pnl_inr": round(lP.sum(),0)},
        {"treatment": "SL_CLOSE_realistic", "n": int(pm.sum()), "avg_return_pct": round(cR.mean(),4),
         "median_return_pct": round(np.median(cR),4), "worst_return_pct": round(cR.min(),4),
         "win_rate_pct": round((cP>0).mean()*100,2), "total_pnl_inr": round(cP.sum(),0)},
    ])

    # ── CRITICAL: max-loss realism (points 1-3), for SL-LOW ──
    lo_ret = (lo_px - e) / e * 100
    Lx = e * (1 - SL_PCT / 100)
    already = bucketN & (lo_et == "sl_1pct_already_breached_at_945")
    stopped = bucketN & (lo_et == "sl_1pct_low")
    exact = stopped & (np.abs(lo_px - Lx) < 1e-6)
    gap = stopped & (lo_px < Lx - 1e-6)
    rode = bucketN & (lo_et == "exit_at_1200_no_trigger")
    tgt_hit = bucketN & (lo_et == "early_target_945_1200")
    worse = already | gap                       # any fill worse than -1%
    nb = int(bucketN.sum())

    realism = pd.DataFrame([
        {"metric": "WORST single-trade return % — A_baseline", "value": round(((a_px[av]-e[av])/e[av]*100).min(), 4)},
        {"metric": "WORST single-trade return % — SL_LOW", "value": round(lo_ret[lv].min(), 4)},
        {"metric": "WORST single-trade return % — SL_CLOSE", "value": round(((cl_px[cv]-e[cv])/e[cv]*100).min(), 4)},
        {"metric": "--- SL_LOW stopped-fill distribution ---", "value": ""},
        {"metric": "n filled at EXACTLY -1% (clean resting stop)", "value": int(exact.sum())},
        {"metric": "n filled WORSE than -1% (gap-through + already-breached)", "value": int(worse.sum())},
        {"metric": "  worse-group avg fill return %", "value": round(lo_ret[worse].mean(), 4) if worse.any() else np.nan},
        {"metric": "  worse-group median fill return %", "value": round(np.median(lo_ret[worse]), 4) if worse.any() else np.nan},
        {"metric": "  worse-group MIN fill return %", "value": round(lo_ret[worse].min(), 4) if worse.any() else np.nan},
        {"metric": "--- SL_LOW Bucket N breakdown (n=%d) ---" % nb, "value": ""},
        {"metric": "already-breached-at-9:45 (exit @9:45, actual loss)", "value": int(already.sum())},
        {"metric": "  their avg actual loss %", "value": round(lo_ret[already].mean(), 4) if already.any() else np.nan},
        {"metric": "  their MIN (worst) loss %", "value": round(lo_ret[already].min(), 4) if already.any() else np.nan},
        {"metric": "stopped later at EXACTLY -1%", "value": int(exact.sum())},
        {"metric": "stopped later WORSE than -1% (gap-through)", "value": int(gap.sum())},
        {"metric": "  gap-through avg fill %", "value": round(lo_ret[gap].mean(), 4) if gap.any() else np.nan},
        {"metric": "rode to 12:00 (never breached)", "value": int(rode.sum())},
        {"metric": "hit 14% target 9:45-12:00", "value": int(tgt_hit.sum())},
    ])

    with pd.ExcelWriter(OUTDIR / "variant_sl_realistic_fills.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        exit_tbl.to_excel(w, sheet_name="exit_type_breakdown", index=False)
        paired.to_excel(w, sheet_name="bucket_n_paired", index=False)
        realism.to_excel(w, sheet_name="max_loss_realism", index=False)

    pd.set_option("display.width", 240)
    show = ["variant","basis","n_trades","total_return_fixedbase_pct","total_pnl_inr","win_rate_pct",
            "avg_return_per_trade_pct","median_return_per_trade_pct","avg_return_winning_trades_pct",
            "avg_return_losing_trades_pct","worst_single_trade_return_pct"]
    print("\n"+"="*130+"\nTHREE-WAY COMPARISON (realistic fills; gross then net)\n"+"="*130)
    print(comp[show].to_string(index=False))
    print("\n--- EXIT-TYPE BREAKDOWN ---"); print(exit_tbl.to_string(index=False))
    print(f"\n--- BUCKET-N PAIRED (same {int(pm.sum())} trades) ---"); print(paired.to_string(index=False))
    print("\n--- MAX-LOSS REALISM (points 1-3) ---"); print(realism.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
