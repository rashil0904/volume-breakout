# -*- coding: utf-8 -*-
"""
variant_sl_n2only.py
====================
−1% stop that arms ONLY on Bucket N trades still above −1% at 09:45 (N2). Trades already
below −1% at 09:45 (N1) get NO stop and ride to 12:00. Main strategy: mcap ₹1,500-5,000 Cr,
LB 36, VM 6, 3:15pm entry, +5%, ₹5L/₹1L, 14% target throughout (priority). Positives
(>0% @9:45) exit @09:45. Reuse canonical trades. L = entry * 0.99.

Bucket N (return_at_945 <= 0):
  N1 (return_at_945 <= -1): NO STOP. 14% target still applies; else exit 12:00 open.
     exit_type "already_below_1pct_held_to_1200".
  N2 (-1 < return_at_945 <= 0): STOP ARMED.
     SL-LOW  : first candle low<=L -> fill min(L, candle_open). "sl_1pct_low"; else 12:00.
     SL-CLOSE: first candle close<=L -> fill candle close. "sl_1pct_close"; else 12:00.

Flags: (a) N1 rides to 12:00, no stop/other intervention (target still allowed). (b) N2 =
-1 < return_at_945 <= 0. (c) SL-LOW fill min(L,open); SL-CLOSE fill breaching close.
(d) 14% target applies to N1 and N2 with priority over the stop.
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
OUTDIR = rb.RESULTS / "variant_sl_n2only"
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
    return None, None, o945


def _target_then_1200(e, O, H, held_type):
    tgt = e * (1 + TARGET / 100)
    for hm in SCAN:
        if not np.isnan(H[HCOL[hm]]) and H[HCOL[hm]] >= tgt:
            return "early_target_945_1200", tgt
    o = O[HCOL[HM_1200]]
    return (held_type, o) if not np.isnan(o) else (None, np.nan)


def variant_A(e, O, H):
    et, px, _ = _pre945(e, O, H)
    if et:
        return et, px
    return _target_then_1200(e, O, H, "exit_at_1200_no_trigger")


def variant_stop(e, O, H, L_, C, basis):
    et, px, o945 = _pre945(e, O, H)
    if et:
        return et, px
    ret945 = (o945 - e) / e * 100 if not np.isnan(o945) else np.nan
    # N1 — already below -1% -> NO stop (target still applies), else 12:00
    if not np.isnan(ret945) and ret945 <= -SL_PCT:
        return _target_then_1200(e, O, H, "already_below_1pct_held_to_1200")
    # N2 — stop armed
    tgt = e * (1 + TARGET / 100)
    Lx = e * (1 - SL_PCT / 100)
    for hm in SCAN:
        h = H[HCOL[hm]]
        if not np.isnan(h) and h >= tgt:                        # target priority
            return "early_target_945_1200", tgt
        if basis == "low":
            l, op = L_[HCOL[hm]], O[HCOL[hm]]
            if not np.isnan(l) and l <= Lx:
                fill = min(Lx, op) if not np.isnan(op) else Lx
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
    bucketN = np.zeros(n, dtype=bool); ret945 = np.full(n, np.nan)
    for i in range(n):
        a_et[i], a_px[i] = variant_A(e[i], O[i], H[i])
        lo_et[i], lo_px[i] = variant_stop(e[i], O[i], H[i], L[i], C[i], "low")
        cl_et[i], cl_px[i] = variant_stop(e[i], O[i], H[i], L[i], C[i], "close")
        pre, _, o945 = _pre945(e[i], O[i], H[i])
        bucketN[i] = pre is None
        if pre is None and not np.isnan(o945):
            ret945[i] = (o945 - e[i]) / e[i] * 100
    av = np.array([x is not None for x in a_et]) & ~np.isnan(a_px)
    lv = np.array([x is not None for x in lo_et]) & ~np.isnan(lo_px)
    cv = np.array([x is not None for x in cl_et]) & ~np.isnan(cl_px)

    N1 = bucketN & (ret945 <= -SL_PCT)                 # already below -1%
    N2 = bucketN & ~N1                                 # above -1% (incl. rare o945-nan)

    # ── 1. three-way comparison ──
    comp = pd.DataFrame([*metrics("A_baseline", e, sh, cap, a_px, av),
                         *metrics("SL_LOW", e, sh, cap, lo_px, lv),
                         *metrics("SL_CLOSE", e, sh, cap, cl_px, cv)]).sort_values(
        ["basis", "variant"]).reset_index(drop=True)

    A_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200", "exit_at_1200_no_trigger"]
    L_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200",
              "already_below_1pct_held_to_1200", "sl_1pct_low", "exit_at_1200_no_trigger"]
    C_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200",
              "already_below_1pct_held_to_1200", "sl_1pct_close", "exit_at_1200_no_trigger"]
    exit_tbl = pd.concat([exit_table("A_baseline", a_et, e, sh, a_px, av, A_CATS),
                          exit_table("SL_LOW", lo_et, e, sh, lo_px, lv, L_CATS),
                          exit_table("SL_CLOSE", cl_et, e, sh, cl_px, cv, C_CATS)], ignore_index=True)

    # ── 2 & 4. Bucket N sub-case sizing (N1 vs N2), with N1 tail risk ──
    def rp(px, m): return (px[m] - e[m]) / e[m] * 100
    n1r = rp(lo_px, N1); n2r_base = rp(a_px, N2)
    subcases = pd.DataFrame([
        {"subcase": "N1 already<=-1% (HELD to 12:00, no stop)", "n": int(N1.sum()),
         "pct_of_bucketN": round(N1.sum()/bucketN.sum()*100, 2),
         "avg_return_pct": round(n1r.mean(), 4), "median_return_pct": round(np.median(n1r), 4),
         "WORST_return_pct": round(n1r.min(), 4)},
        {"subcase": "N2 -1%<ret<=0 (STOP-ELIGIBLE; baseline 12:00 return)", "n": int(N2.sum()),
         "pct_of_bucketN": round(N2.sum()/bucketN.sum()*100, 2),
         "avg_return_pct": round(n2r_base.mean(), 4), "median_return_pct": round(np.median(n2r_base), 4),
         "WORST_return_pct": round(n2r_base.min(), 4)},
    ])

    # ── 3. N2 composition under each stop variant ──
    def n2_compo(et, px, stop_type):
        stopped = N2 & (et == stop_type)
        rode = N2 & (et == "exit_at_1200_no_trigger")
        tgt = N2 & (et == "early_target_945_1200")
        fr = (px[stopped] - e[stopped]) / e[stopped] * 100
        return {"variant": stop_type, "n_N2": int(N2.sum()), "n_stopped": int(stopped.sum()),
                "n_rode_1200": int(rode.sum()), "n_hit_target": int(tgt.sum()),
                "avg_fill_return_stopped_pct": round(fr.mean(), 4) if stopped.any() else np.nan,
                "worst_return_in_stopped_pct": round(fr.min(), 4) if stopped.any() else np.nan}
    n2_tbl = pd.DataFrame([n2_compo(lo_et, lo_px, "sl_1pct_low"),
                           n2_compo(cl_et, cl_px, "sl_1pct_close")])

    # ── 5. whipsaw on N2 (SL-LOW): stopped but 12:00 > -1% ──
    ret1200 = (O[:, HCOL[HM_1200]] - e) / e * 100
    lo_stopped_n2 = N2 & (lo_et == "sl_1pct_low")
    whip = lo_stopped_n2 & (ret1200 > -SL_PCT)
    whip_tbl = pd.DataFrame([
        {"metric": "N2 SL-LOW stopped", "value": int(lo_stopped_n2.sum())},
        {"metric": "  of which 12:00 recovered >-1% (whipsaw)", "value": int(whip.sum())},
        {"metric": "  whipsaw % of N2 stops", "value": round(whip.sum()/lo_stopped_n2.sum()*100, 2) if lo_stopped_n2.sum() else np.nan},
        {"metric": "  avg 12:00 return of whipsaw trades (given up)", "value": round(ret1200[whip].mean(), 4) if whip.any() else np.nan},
        {"metric": "  avg SL-LOW fill of whipsaw trades", "value": round(((lo_px[whip]-e[whip])/e[whip]*100).mean(), 4) if whip.any() else np.nan},
    ])

    with pd.ExcelWriter(OUTDIR / "variant_sl_n2only.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        exit_tbl.to_excel(w, sheet_name="exit_type_breakdown", index=False)
        subcases.to_excel(w, sheet_name="bucket_n_subcases", index=False)
        n2_tbl.to_excel(w, sheet_name="n2_stop_composition", index=False)
        whip_tbl.to_excel(w, sheet_name="whipsaw", index=False)

    pd.set_option("display.width", 240)
    show = ["variant","basis","n_trades","total_return_fixedbase_pct","total_pnl_inr","win_rate_pct",
            "avg_return_per_trade_pct","median_return_per_trade_pct","avg_return_winning_trades_pct",
            "avg_return_losing_trades_pct","worst_single_trade_return_pct"]
    print("\n"+"="*130+"\nTHREE-WAY COMPARISON (stop on N2 only; gross then net)\n"+"="*130)
    print(comp[show].to_string(index=False))
    print("\n--- EXIT-TYPE BREAKDOWN ---"); print(exit_tbl.to_string(index=False))
    print("\n--- BUCKET N SUB-CASES (N1 held vs N2 stop-eligible; N1 carries the uncapped tail) ---")
    print(subcases.to_string(index=False))
    print("\n--- N2 STOP COMPOSITION ---"); print(n2_tbl.to_string(index=False))
    print("\n--- WHIPSAW ON N2 (SL-LOW) ---"); print(whip_tbl.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
