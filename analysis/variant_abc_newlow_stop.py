# -*- coding: utf-8 -*-
"""
variant_abc_newlow_stop.py
==========================
Three-variant new-day-low stop comparison on the main strategy (mcap ₹1,500-5,000 Cr,
lookback 36, volume 6x, 3:15pm entry, +5% day, ₹5L pool / ₹1L per trade, 14% target).
Positives at 09:45 exit @09:45 in ALL variants; variants differ only in Bucket N
(non-positive at 09:45) handling. Reuses canonical base positions; A/B/C recomputed on
the SAME trade set (needed for the paired Bucket-N read).

  A (baseline)     : Bucket N -> 12:00 open (target scan 09:45->12:00 throughout).
  B (1st-low stop) : Bucket N -> exit at reference_low on the FIRST candle whose low
                     undercuts reference_low (target priority); else 12:00.
  C (2nd-low stop) : Bucket N -> 1st breach (low<reference_low) sets low_after_first_breach
                     but does NOT exit; exit only when a LATER candle undercuts
                     low_after_first_breach (2nd breach), filling at low_after_first_breach;
                     else 12:00. Target priority throughout.

reference_low = min low over 09:15,09:30,09:45 inclusive. Scan = candle after 09:45 to
candle before 12:00 (10:00..11:45).

Flags: (a) 2nd-breach fills at low_after_first_breach (the undercut level), NOT the 2nd
candle's own low. (b) the 2nd low must undercut low_after_first_breach (running low since
the 1st breach), not merely re-break the 09:45 reference_low. (c) target priority over
stop in the same candle, in both B and C.
NOTE on low_after_first_breach: per the spec's ordering it is set at the 1st breach and
only "updates" on candles whose low >= it (a no-op min), so it equals the 1st-breach
candle's low; a later candle going below it IS the 2nd breach and exits at that level.
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
OUTDIR = rb.RESULTS / "variant_abc_newlow_stop"
BASE_POOL, EXPENSE = 500_000, 0.0023
TARGET = 14.0
CANDLE_HMS = list(range(555, 901, 15))
HCOL = {hm: k for k, hm in enumerate(CANDLE_HMS)}
HM_0945, HM_1200 = 585, 720
PRE945 = [hm for hm in CANDLE_HMS if hm < HM_0945]
REFLOW_HMS = [hm for hm in CANDLE_HMS if hm <= HM_0945]
SCAN = [hm for hm in CANDLE_HMS if HM_0945 < hm < HM_1200]


def fetch_ohlc(base):
    n = len(base)
    O = np.full((n, len(CANDLE_HMS)), np.nan)
    H = np.full((n, len(CANDLE_HMS)), np.nan)
    L = np.full((n, len(CANDLE_HMS)), np.nan)
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
        po = sub.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=CANDLE_HMS)
        ph = sub.pivot_table(index="date", columns="hm", values="high", aggfunc="last").reindex(columns=CANDLE_HMS)
        pl = sub.pivot_table(index="date", columns="hm", values="low", aggfunc="last").reindex(columns=CANDLE_HMS)
        for idx, ed in zip(grp.index, grp["date"]):
            j = bisect.bisect_right(ds, ed.date())
            if j >= len(ds):
                continue
            nd = ds[j]
            if nd in po.index:
                O[idx, :] = po.loc[nd].values
                H[idx, :] = ph.loc[nd].values
                L[idx, :] = pl.loc[nd].values
    return O, H, L


def _pre_and_945(e, O, H):
    """Shared pre-945 target + positive-at-945; returns (early_et, px) if resolved,
    else (None, o945) meaning -> Bucket N with that o945."""
    tgt = e * (1 + TARGET / 100)
    for hm in PRE945:
        v = H[HCOL[hm]]
        if not np.isnan(v) and v >= tgt:
            return "early_target_pre_945", tgt, None
    o945 = O[HCOL[HM_0945]]
    if not np.isnan(o945) and (o945 - e) / e * 100 > 0:
        return "positive_at_945", o945, None
    return None, None, o945          # Bucket N


def variant_A(e, O, H):
    et, px, o945 = _pre_and_945(e, O, H)
    if et:
        return et, px
    tgt = e * (1 + TARGET / 100)
    for hm in SCAN:
        v = H[HCOL[hm]]
        if not np.isnan(v) and v >= tgt:
            return "early_target_945_1200", tgt
    o1200 = O[HCOL[HM_1200]]
    return ("exit_at_1200", o1200) if not np.isnan(o1200) else (None, np.nan)


def variant_B(e, O, H, L):
    et, px, o945 = _pre_and_945(e, O, H)
    if et:
        return et, px
    tgt = e * (1 + TARGET / 100)
    ref = [L[HCOL[hm]] for hm in REFLOW_HMS if not np.isnan(L[HCOL[hm]])]
    ref_low = min(ref) if ref else np.nan
    for hm in SCAN:
        h, l = H[HCOL[hm]], L[HCOL[hm]]
        if not np.isnan(h) and h >= tgt:
            return "early_target_945_1200", tgt
        if not np.isnan(ref_low) and not np.isnan(l) and l < ref_low:
            return "first_new_day_low_stop", ref_low
    o1200 = O[HCOL[HM_1200]]
    return ("exit_at_1200", o1200) if not np.isnan(o1200) else (None, np.nan)


def variant_C(e, O, H, L):
    """Returns (exit_type, exit_price, ref_low, low_afb, n_breach). n_breach: 0=no breach,
    1=only 1st breach (fell to 12:00), 2=2nd breach fired."""
    et, px, o945 = _pre_and_945(e, O, H)
    if et:
        return et, px, np.nan, np.nan, 0
    tgt = e * (1 + TARGET / 100)
    ref = [L[HCOL[hm]] for hm in REFLOW_HMS if not np.isnan(L[HCOL[hm]])]
    ref_low = min(ref) if ref else np.nan
    low_afb = np.nan
    first = False
    for hm in SCAN:
        h, l = H[HCOL[hm]], L[HCOL[hm]]
        if not np.isnan(h) and h >= tgt:                       # target priority
            return "early_target_945_1200", tgt, ref_low, low_afb, (1 if first else 0)
        if np.isnan(l) or np.isnan(ref_low):
            continue
        if not first:
            if l < ref_low:                                    # 1st breach — do NOT exit
                first = True
                low_afb = l
        else:
            if l < low_afb:                                    # 2nd breach — exit at undercut level
                return "second_new_day_low_stop", low_afb, ref_low, low_afb, 2
            low_afb = min(low_afb, l)                           # (no-op given ordering; per spec)
    o1200 = O[HCOL[HM_1200]]
    n_breach = 1 if first else 0
    if np.isnan(o1200):
        return None, np.nan, ref_low, low_afb, n_breach
    return "exit_at_1200_no_second_breach", o1200, ref_low, low_afb, n_breach


def blk(pnl, ret, cap):
    w, l = pnl > 0, pnl <= 0
    return dict(
        n_trades=len(pnl),
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


def exit_table(name, et_arr, e, sh, px, valid, cats):
    e, sh, px, et = e[valid], sh[valid], px[valid], et_arr[valid]
    ret = (px - e) / e * 100
    ntot = len(et)
    rows = []
    for c in cats:
        m = et == c
        rows.append({"variant": name, "exit_type": c, "n": int(m.sum()),
                     "pct": round(m.mean() * 100, 2) if ntot else 0,
                     "avg_return_pct": round(ret[m].mean(), 4) if m.any() else np.nan,
                     "median_return_pct": round(np.median(ret[m]), 4) if m.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = ets.load_base_positions()
    e = base["entry"].values.astype(float)
    sh = base["shares"].values.astype(float)
    cap = base["cap"].values.astype(float)
    print(f"Base positions: {len(base):,}")
    print("Fetching next-day OHLC (opens/highs/lows) …")
    O, H, L = fetch_ohlc(base)

    n = len(base)
    a_et = np.empty(n, dtype=object); a_px = np.full(n, np.nan)
    b_et = np.empty(n, dtype=object); b_px = np.full(n, np.nan)
    c_et = np.empty(n, dtype=object); c_px = np.full(n, np.nan)
    c_ref = np.full(n, np.nan); c_afb = np.full(n, np.nan); c_nb = np.zeros(n, dtype=int)
    bucketN = np.zeros(n, dtype=bool)
    for i in range(n):
        a_et[i], a_px[i] = variant_A(e[i], O[i], H[i])
        b_et[i], b_px[i] = variant_B(e[i], O[i], H[i], L[i])
        c_et[i], c_px[i], c_ref[i], c_afb[i], c_nb[i] = variant_C(e[i], O[i], H[i], L[i])
        pre, _, o945 = _pre_and_945(e[i], O[i], H[i])
        bucketN[i] = (pre is None)
    a_valid = np.array([x is not None for x in a_et]) & ~np.isnan(a_px)
    b_valid = np.array([x is not None for x in b_et]) & ~np.isnan(b_px)
    c_valid = np.array([x is not None for x in c_et]) & ~np.isnan(c_px)

    # ── 1. three-way comparison (gross + net) ──
    comp = pd.DataFrame([*metrics("A_baseline", e, sh, cap, a_px, a_valid),
                         *metrics("B_first_low_stop", e, sh, cap, b_px, b_valid),
                         *metrics("C_second_low_stop", e, sh, cap, c_px, c_valid)])
    comp = comp.sort_values(["basis", "variant"]).reset_index(drop=True)

    # ── 2. exit-type breakdown ──
    A_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200", "exit_at_1200"]
    B_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200",
              "first_new_day_low_stop", "exit_at_1200"]
    C_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_1200",
              "second_new_day_low_stop", "exit_at_1200_no_second_breach"]
    exit_tbl = pd.concat([exit_table("A_baseline", a_et, e, sh, a_px, a_valid, A_CATS),
                          exit_table("B_first_low_stop", b_et, e, sh, b_px, b_valid, B_CATS),
                          exit_table("C_second_low_stop", c_et, e, sh, c_px, c_valid, C_CATS)],
                         ignore_index=True)

    # ── 3. Bucket-N paired (same non-positive trades under A / B / C) ──
    pm = bucketN & a_valid & b_valid & c_valid
    def rr(px):
        return (px[pm] - e[pm]) / e[pm] * 100
    aR, bR, cR = rr(a_px), rr(b_px), rr(c_px)
    aP = sh[pm] * (a_px[pm] - e[pm]); bP = sh[pm] * (b_px[pm] - e[pm]); cP = sh[pm] * (c_px[pm] - e[pm])
    paired_summary = pd.DataFrame([
        {"treatment": "A_hold_to_1200", "n": int(pm.sum()), "avg_return_pct": round(aR.mean(), 4),
         "median_return_pct": round(np.median(aR), 4), "total_pnl_inr": round(aP.sum(), 0),
         "win_rate_pct": round((aP > 0).mean() * 100, 2)},
        {"treatment": "B_stop_on_1st_low", "n": int(pm.sum()), "avg_return_pct": round(bR.mean(), 4),
         "median_return_pct": round(np.median(bR), 4), "total_pnl_inr": round(bP.sum(), 0),
         "win_rate_pct": round((bP > 0).mean() * 100, 2)},
        {"treatment": "C_stop_on_2nd_low", "n": int(pm.sum()), "avg_return_pct": round(cR.mean(), 4),
         "median_return_pct": round(np.median(cR), 4), "total_pnl_inr": round(cP.sum(), 0),
         "win_rate_pct": round((cP > 0).mean() * 100, 2)},
    ])

    # ── 4. Variant C breach subgroups (Bucket N only) ──
    cN = bucketN & c_valid
    cNret = (c_px - e) / e * 100
    breach_rows = []
    for k, lbl in [(0, "0 breaches -> 12:00"), (1, "exactly 1 breach -> 12:00"),
                   (2, "2+ breaches -> 2nd-low stop")]:
        m = cN & (c_nb == k)
        breach_rows.append({"breach_group": lbl, "n": int(m.sum()),
                            "pct_of_bucketN": round(m.sum() / cN.sum() * 100, 2) if cN.sum() else 0,
                            "avg_return_pct": round(cNret[m].mean(), 4) if m.any() else np.nan,
                            "median_return_pct": round(np.median(cNret[m]), 4) if m.any() else np.nan})
    breach_tbl = pd.DataFrame(breach_rows)

    # ── 5. variant_c_trades (Bucket N) ──
    ci = np.where(cN)[0]
    c_trades = pd.DataFrame({
        "symbol": base["symbol"].values[ci], "entry_date": base["date"].values[ci],
        "entry": e[ci], "reference_low": np.round(c_ref[ci], 2),
        "low_after_first_breach": np.round(c_afb[ci], 2), "n_breach": c_nb[ci],
        "exit_type": c_et[ci], "exit_price": np.round(c_px[ci], 2),
        "gross_ret_pct": np.round((c_px[ci] - e[ci]) / e[ci] * 100, 4)})

    with pd.ExcelWriter(OUTDIR / "variant_abc_newlow_stop.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        exit_tbl.to_excel(w, sheet_name="exit_type_breakdown", index=False)
        paired_summary.to_excel(w, sheet_name="bucket_n_paired", index=False)
        breach_tbl.to_excel(w, sheet_name="variant_c_breach_subgroups", index=False)
        c_trades.to_excel(w, sheet_name="variant_c_trades", index=False)

    pd.set_option("display.width", 240)
    show = ["variant", "basis", "n_trades", "total_return_fixedbase_pct", "total_pnl_inr",
            "win_rate_pct", "avg_return_per_trade_pct", "median_return_per_trade_pct",
            "avg_return_winning_trades_pct", "avg_return_losing_trades_pct"]
    print("\n" + "=" * 120 + "\nTHREE-WAY COMPARISON (gross then net)\n" + "=" * 120)
    print(comp[show].to_string(index=False))
    print("\n--- EXIT-TYPE BREAKDOWN ---")
    print(exit_tbl.to_string(index=False))
    print(f"\n--- BUCKET-N PAIRED (same {int(pm.sum())} non-positive trades under A/B/C) ---")
    print(paired_summary.to_string(index=False))
    print("\n--- VARIANT C: breach subgroups (Bucket N) ---")
    print(breach_tbl.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
