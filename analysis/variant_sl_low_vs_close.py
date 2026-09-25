# -*- coding: utf-8 -*-
"""
variant_sl_low_vs_close.py
==========================
Two −1% stop styles vs the 12:00-hold baseline, on the main strategy's Bucket N
(non-positive at 09:45) trades. Base: mcap ₹1,500-5,000 Cr, LB 36, VM 6, 3:15pm entry,
+5% day, ₹5L pool / ₹1L per trade, 14% target. Positives exit @09:45 in all variants;
14% target throughout, priority over any stop in the same candle. Reuses canonical base
positions; recomputed on the SAME trade set for the paired read.

Stop level L = entry * 0.99 (−1%). Scan = candle after 09:45 .. candle before 12:00.
  A         : Bucket N -> 12:00 open.
  SL-LOW    : first candle with LOW  <= L -> fill at L        (resting-stop). "sl_1pct_low"
  SL-CLOSE  : first candle with CLOSE<= L -> fill at CLOSE    (decide-at-close). "sl_1pct_close"
Fallback (no target, no stop) -> 12:00 open, "exit_at_1200_no_trigger".

Flags: (a) SL-LOW fills at L exactly; SL-CLOSE fills at the breaching candle's close
(asymmetry reflects resting-order vs decide-at-close). (b) target priority same-candle.
(c) scan window after-09:45 to before-12:00.
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
OUTDIR = rb.RESULTS / "variant_sl_low_vs_close"
BASE_POOL, EXPENSE = 500_000, 0.0023
TARGET = 14.0
SL_PCT = 1.0
CANDLE_HMS = list(range(555, 901, 15))
HCOL = {hm: k for k, hm in enumerate(CANDLE_HMS)}
HM_0945, HM_1200 = 585, 720
PRE945 = [hm for hm in CANDLE_HMS if hm < HM_0945]
SCAN = [hm for hm in CANDLE_HMS if HM_0945 < hm < HM_1200]     # 10:00..11:45


def fetch_ohlc(base):
    n = len(base)
    O = np.full((n, len(CANDLE_HMS)), np.nan)
    H = np.full((n, len(CANDLE_HMS)), np.nan)
    Lo = np.full((n, len(CANDLE_HMS)), np.nan)
    C = np.full((n, len(CANDLE_HMS)), np.nan)
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
        piv = {c: sub.pivot_table(index="date", columns="hm", values=c, aggfunc="last")
               .reindex(columns=CANDLE_HMS) for c in ("open", "high", "low", "close")}
        for idx, ed in zip(grp.index, grp["date"]):
            j = bisect.bisect_right(ds, ed.date())
            if j >= len(ds):
                continue
            nd = ds[j]
            if nd in piv["open"].index:
                O[idx, :] = piv["open"].loc[nd].values
                H[idx, :] = piv["high"].loc[nd].values
                Lo[idx, :] = piv["low"].loc[nd].values
                C[idx, :] = piv["close"].loc[nd].values
    return O, H, Lo, C


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


def variant_stop(e, O, H, Lo, C, basis):
    """basis='low' -> trigger low<=L, fill at L; basis='close' -> trigger close<=L, fill at close."""
    et, px, _ = _pre945(e, O, H)
    if et:
        return et, px
    tgt = e * (1 + TARGET / 100)
    L = e * (1 - SL_PCT / 100)
    for hm in SCAN:
        h = H[HCOL[hm]]
        if not np.isnan(h) and h >= tgt:                     # target priority
            return "early_target_945_1200", tgt
        if basis == "low":
            l = Lo[HCOL[hm]]
            if not np.isnan(l) and l <= L:
                return "sl_1pct_low", L
        else:
            c = C[HCOL[hm]]
            if not np.isnan(c) and c <= L:
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
                     "median_return_pct": round(np.median(ret[m]), 4) if m.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = ets.load_base_positions()
    e = base["entry"].values.astype(float)
    sh = base["shares"].values.astype(float)
    cap = base["cap"].values.astype(float)
    print(f"Base positions: {len(base):,} | fetching next-day OHLC …")
    O, H, Lo, C = fetch_ohlc(base)

    n = len(base)
    a_et = np.empty(n, dtype=object); a_px = np.full(n, np.nan)
    lo_et = np.empty(n, dtype=object); lo_px = np.full(n, np.nan)
    cl_et = np.empty(n, dtype=object); cl_px = np.full(n, np.nan)
    bucketN = np.zeros(n, dtype=bool)
    for i in range(n):
        a_et[i], a_px[i] = variant_A(e[i], O[i], H[i])
        lo_et[i], lo_px[i] = variant_stop(e[i], O[i], H[i], Lo[i], C[i], "low")
        cl_et[i], cl_px[i] = variant_stop(e[i], O[i], H[i], Lo[i], C[i], "close")
        pre, _, _ = _pre945(e[i], O[i], H[i])
        bucketN[i] = pre is None
    av = np.array([x is not None for x in a_et]) & ~np.isnan(a_px)
    lv = np.array([x is not None for x in lo_et]) & ~np.isnan(lo_px)
    cv = np.array([x is not None for x in cl_et]) & ~np.isnan(cl_px)

    # ── 1. three-way comparison ──
    comp = pd.DataFrame([*metrics("A_baseline", e, sh, cap, a_px, av),
                         *metrics("SL_LOW", e, sh, cap, lo_px, lv),
                         *metrics("SL_CLOSE", e, sh, cap, cl_px, cv)]).sort_values(
        ["basis", "variant"]).reset_index(drop=True)

    # ── 2. exit-type breakdown ──
    A_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200", "exit_at_1200_no_trigger"]
    L_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200", "sl_1pct_low", "exit_at_1200_no_trigger"]
    C_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200", "sl_1pct_close", "exit_at_1200_no_trigger"]
    exit_tbl = pd.concat([exit_table("A_baseline", a_et, e, sh, a_px, av, A_CATS),
                          exit_table("SL_LOW", lo_et, e, sh, lo_px, lv, L_CATS),
                          exit_table("SL_CLOSE", cl_et, e, sh, cl_px, cv, C_CATS)], ignore_index=True)

    # ── 3. Bucket-N paired ──
    pm = bucketN & av & lv & cv
    ret1200 = (O[:, HCOL[HM_1200]] - e) / e * 100
    def rr(px): return (px[pm] - e[pm]) / e[pm] * 100
    aR, lR, cR = rr(a_px), rr(lo_px), rr(cl_px)
    aP = sh[pm]*(a_px[pm]-e[pm]); lP = sh[pm]*(lo_px[pm]-e[pm]); cP = sh[pm]*(cl_px[pm]-e[pm])
    paired = pd.DataFrame([
        {"treatment": "A_hold_to_1200", "n": int(pm.sum()), "avg_return_pct": round(aR.mean(),4),
         "median_return_pct": round(np.median(aR),4), "win_rate_pct": round((aP>0).mean()*100,2), "total_pnl_inr": round(aP.sum(),0)},
        {"treatment": "SL_LOW_stop_on_low", "n": int(pm.sum()), "avg_return_pct": round(lR.mean(),4),
         "median_return_pct": round(np.median(lR),4), "win_rate_pct": round((lP>0).mean()*100,2), "total_pnl_inr": round(lP.sum(),0)},
        {"treatment": "SL_CLOSE_stop_on_close", "n": int(pm.sum()), "avg_return_pct": round(cR.mean(),4),
         "median_return_pct": round(np.median(cR),4), "win_rate_pct": round((cP>0).mean()*100,2), "total_pnl_inr": round(cP.sum(),0)},
    ])

    # ── 3b. stop-trigger composition + fill quality (Bucket N) ──
    def compo(et, px, label):
        m = bucketN
        stop_types = {"SL_LOW": "sl_1pct_low", "SL_CLOSE": "sl_1pct_close"}[label]
        stopped = m & (et == stop_types)
        rode = m & (et == "exit_at_1200_no_trigger")
        tgt = m & np.isin(et, ["early_target_945_1200"])
        fill_ret = (px[stopped] - e[stopped]) / e[stopped] * 100
        return {"variant": label, "n_bucketN": int(m.sum()),
                "n_stopped": int(stopped.sum()), "pct_stopped": round(stopped.sum()/m.sum()*100,2),
                "n_rode_1200": int(rode.sum()), "n_hit_target": int(tgt.sum()),
                "avg_fill_return_pct_of_stopped": round(fill_ret.mean(),4) if stopped.any() else np.nan}
    trigger_tbl = pd.DataFrame([compo(lo_et, lo_px, "SL_LOW"), compo(cl_et, cl_px, "SL_CLOSE")])

    # ── 4. whipsaw analysis ──
    lo_stopped = bucketN & (lo_et == "sl_1pct_low")
    cl_stopped = bucketN & (cl_et == "sl_1pct_close")
    # (i) SL-LOW stopped but recovered above -1% by 12:00 (whipsaw)
    whip = lo_stopped & (ret1200 > -SL_PCT)
    # (ii) SL-LOW stopped but SL-CLOSE did NOT (close never broke -1%) = whipsaw the close rule avoids
    avoided = lo_stopped & ~cl_stopped
    whip_summary = pd.DataFrame([
        {"metric": "SL_LOW_stopped_total", "value": int(lo_stopped.sum())},
        {"metric": "  of which recovered >-1% by 12:00 (whipsaw)", "value": int(whip.sum())},
        {"metric": "  whipsaw as % of SL_LOW stops", "value": round(whip.sum()/lo_stopped.sum()*100,2) if lo_stopped.sum() else np.nan},
        {"metric": "  avg 12:00 return of whipsaw trades (given up)", "value": round(ret1200[whip].mean(),4) if whip.any() else np.nan},
        {"metric": "  avg SL_LOW fill return of whipsaw trades", "value": round(((lo_px[whip]-e[whip])/e[whip]*100).mean(),4) if whip.any() else np.nan},
        {"metric": "SL_CLOSE_stopped_total", "value": int(cl_stopped.sum())},
        {"metric": "SL_LOW stopped but SL_CLOSE did NOT (whipsaw avoided by close-rule)", "value": int(avoided.sum())},
        {"metric": "  those trades' avg 12:00 return", "value": round(ret1200[avoided].mean(),4) if avoided.any() else np.nan},
        {"metric": "  those trades' avg SL_LOW fill return (what low-stop locked in)", "value": round(((lo_px[avoided]-e[avoided])/e[avoided]*100).mean(),4) if avoided.any() else np.nan},
    ])

    # ── per-trade SL-LOW export (all valid trades) ──
    L_level = e * (1 - SL_PCT / 100)
    g_pnl = sh * (lo_px - e); g_ret = (lo_px - e) / e * 100
    exp = cap * EXPENSE; n_pnl = g_pnl - exp; n_ret = g_ret - EXPENSE * 100
    sl_low_trades = pd.DataFrame({
        "symbol": base["symbol"].values, "entry_date": base["date"].values,
        "entry_price": np.round(e, 2), "shares": sh.astype(int),
        "capital_deployed": np.round(cap, 0), "stop_level_L_minus1pct": np.round(L_level, 2),
        "exit_type": lo_et, "exit_price": np.round(lo_px, 2),
        "gross_trade_return_pct": np.round(g_ret, 4), "gross_trade_pnl_inr": np.round(g_pnl, 0),
        "expense_inr": np.round(exp, 2), "net_trade_pnl_inr": np.round(n_pnl, 0),
        "net_trade_return_pct": np.round(n_ret, 4),
        "is_bucketN": bucketN, "ret_at_1200_pct": np.round(ret1200, 4),
        "is_whipsaw_stop": (lo_et == "sl_1pct_low") & (ret1200 > -SL_PCT),
    })[lv].sort_values("entry_date").reset_index(drop=True)

    with pd.ExcelWriter(OUTDIR / "variant_sl_low_vs_close.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        exit_tbl.to_excel(w, sheet_name="exit_type_breakdown", index=False)
        paired.to_excel(w, sheet_name="bucket_n_paired", index=False)
        trigger_tbl.to_excel(w, sheet_name="stop_trigger_composition", index=False)
        whip_summary.to_excel(w, sheet_name="whipsaw_analysis", index=False)
        sl_low_trades.to_excel(w, sheet_name="sl_low_trades", index=False)
    # standalone copy too
    sl_low_trades.to_csv(OUTDIR / "sl_low_trades.csv", index=False)
    print(f"  sl_low_trades sheet: {len(sl_low_trades):,} rows")

    pd.set_option("display.width", 240)
    show = ["variant","basis","n_trades","total_return_fixedbase_pct","total_pnl_inr","win_rate_pct",
            "avg_return_per_trade_pct","median_return_per_trade_pct","avg_return_winning_trades_pct","avg_return_losing_trades_pct"]
    print("\n"+"="*120+"\nTHREE-WAY COMPARISON (gross then net)\n"+"="*120)
    print(comp[show].to_string(index=False))
    print("\n--- EXIT-TYPE BREAKDOWN ---"); print(exit_tbl.to_string(index=False))
    print(f"\n--- BUCKET-N PAIRED (same {int(pm.sum())} non-positive trades) ---"); print(paired.to_string(index=False))
    print("\n--- STOP-TRIGGER COMPOSITION + FILL QUALITY (Bucket N) ---"); print(trigger_tbl.to_string(index=False))
    print("\n--- WHIPSAW ANALYSIS ---"); print(whip_summary.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
