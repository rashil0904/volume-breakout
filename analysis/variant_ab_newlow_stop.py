# -*- coding: utf-8 -*-
"""
variant_ab_newlow_stop.py
=========================
Variant A (baseline) vs Variant B (new-day-low stop with 12:00 fallback), both on the
main strategy (mcap ₹1,500-5,000 Cr, lookback 36, volume 6x, 3:15pm entry, +5% day,
₹5L pool / ₹1L per trade, 14% profit-target overlay). Reuses the canonical base
positions (ets.load_base_positions); exit logic recomputed from next-day 15-min OHLC so
A and B run on the IDENTICAL trade set (needed for the paired stopped-subset read).

VARIANT A: 14% target scan; positives@09:45 -> 09:45 open; non-positives -> 12:00 open
           (target scan continues 09:45->12:00). == the deployed conditional_split_best_t2.

VARIANT B: same, except the non-positive (Bucket N) leg adds a NEW-DAY-LOW STOP:
  reference_low = min low over 09:15,09:30,09:45. From the candle after 09:45 to the
  candle before 12:00, first trigger wins — target (high>=entry*1.14, priority) else
  stop (low<reference_low, fill at reference_low). No trigger -> 12:00 open.
  => B differs from A ONLY on trades where the stop fires before target/12:00.

Flags confirmed: (a) target priority over stop in the same candle; (b) stop fills at
reference_low exactly; (c) reference_low over 09:15-09:45 inclusive.
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
OUTDIR = rb.RESULTS / "variant_ab_newlow_stop"
BASE_POOL, EXPENSE = 500_000, 0.0023
TARGET = 14.0
CANDLE_HMS = list(range(555, 901, 15))          # 09:15 .. 15:00
HCOL = {hm: k for k, hm in enumerate(CANDLE_HMS)}
HM_0945, HM_1200 = 585, 720
PRE945 = [hm for hm in CANDLE_HMS if hm < HM_0945]              # 09:15, 09:30
REFLOW_HMS = [hm for hm in CANDLE_HMS if hm <= HM_0945]         # 09:15, 09:30, 09:45
SCAN = [hm for hm in CANDLE_HMS if HM_0945 < hm < HM_1200]      # 10:00 .. 11:45


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


def variant_A(e, O, H):
    tgt = e * (1 + TARGET / 100)
    for hm in PRE945:
        v = H[HCOL[hm]]
        if not np.isnan(v) and v >= tgt:
            return "early_target_pre_t1", tgt
    o945 = O[HCOL[HM_0945]]
    if not np.isnan(o945) and (o945 - e) / e * 100 > 0:
        return "positive_at_t1", o945
    for hm in SCAN:
        v = H[HCOL[hm]]
        if not np.isnan(v) and v >= tgt:
            return "early_target_between_t1_t2", tgt
    o1200 = O[HCOL[HM_1200]]
    if np.isnan(o1200):
        return None, np.nan
    return "exit_at_t2_no_target", o1200


def variant_B(e, O, H, L):
    tgt = e * (1 + TARGET / 100)
    for hm in PRE945:
        v = H[HCOL[hm]]
        if not np.isnan(v) and v >= tgt:
            return "early_target_pre_945", tgt, np.nan
    o945 = O[HCOL[HM_0945]]
    if not np.isnan(o945) and (o945 - e) / e * 100 > 0:
        return "positive_at_945", o945, np.nan
    # Bucket N
    ref_lows = [L[HCOL[hm]] for hm in REFLOW_HMS if not np.isnan(L[HCOL[hm]])]
    ref_low = min(ref_lows) if ref_lows else np.nan
    for hm in SCAN:
        h, l = H[HCOL[hm]], L[HCOL[hm]]
        if not np.isnan(h) and h >= tgt:                 # target priority
            return "early_target_945_to_1200", tgt, ref_low
        if not np.isnan(ref_low) and not np.isnan(l) and l < ref_low:
            return "new_day_low_stop", ref_low, ref_low
    o1200 = O[HCOL[HM_1200]]
    if np.isnan(o1200):
        return None, np.nan, ref_low
    return "exit_at_1200_no_trigger", o1200, ref_low


def metrics(name, e, sh, cap, px, valid):
    e, sh, cap, px = e[valid], sh[valid], cap[valid], px[valid]
    gp = sh * (px - e); gret = (px - e) / e * 100
    npl = gp - cap * EXPENSE; nret = gret - EXPENSE * 100
    def blk(pnl, ret):
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
    g = blk(gp, gret); n = blk(npl, nret)
    return ({"variant": name, "basis": "gross", **g}, {"variant": name, "basis": "net", **n})


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
    b_et = np.empty(n, dtype=object); b_px = np.full(n, np.nan); b_ref = np.full(n, np.nan)
    for i in range(n):
        a_et[i], a_px[i] = variant_A(e[i], O[i], H[i])
        b_et[i], b_px[i], b_ref[i] = variant_B(e[i], O[i], H[i], L[i])
    a_valid = np.array([x is not None for x in a_et]) & ~np.isnan(a_px)
    b_valid = np.array([x is not None for x in b_et]) & ~np.isnan(b_px)

    # ── comparison (gross + net) ──
    ma_g, ma_n = metrics("A_baseline", e, sh, cap, a_px, a_valid)
    mb_g, mb_n = metrics("B_newlow_stop", e, sh, cap, b_px, b_valid)
    comp = pd.DataFrame([ma_g, mb_g, ma_n, mb_n])

    A_CATS = ["early_target_pre_t1", "positive_at_t1", "early_target_between_t1_t2", "exit_at_t2_no_target"]
    B_CATS = ["early_target_pre_945", "positive_at_945", "early_target_945_to_1200",
              "new_day_low_stop", "exit_at_1200_no_trigger"]
    exit_tbl = pd.concat([exit_table("A_baseline", a_et, e, sh, a_px, a_valid, A_CATS),
                          exit_table("B_newlow_stop", b_et, e, sh, b_px, b_valid, B_CATS)],
                         ignore_index=True)

    # ── Bucket N sub-outcome split (Variant B) ──
    bucketN = np.isin(b_et, ["early_target_945_to_1200", "new_day_low_stop", "exit_at_1200_no_trigger"]) & b_valid
    nN = int(bucketN.sum())
    bret = (b_px - e) / e * 100
    subN = []
    for c in ["new_day_low_stop", "early_target_945_to_1200", "exit_at_1200_no_trigger"]:
        m = (b_et == c) & b_valid
        subN.append({"bucketN_outcome": c, "n": int(m.sum()),
                     "pct_of_bucketN": round(m.sum() / nN * 100, 2) if nN else 0,
                     "avg_return_pct": round(bret[m].mean(), 4) if m.any() else np.nan,
                     "median_return_pct": round(np.median(bret[m]), 4) if m.any() else np.nan})
    subN_tbl = pd.DataFrame(subN)

    # ── paired stopped-subset: B stop vs A (same trades held per A) ──
    stopped = (b_et == "new_day_low_stop") & b_valid & a_valid
    si = np.where(stopped)[0]
    aret_s = (a_px[si] - e[si]) / e[si] * 100
    bret_s = (b_px[si] - e[si]) / e[si] * 100
    a_pnl_s = sh[si] * (a_px[si] - e[si]); b_pnl_s = sh[si] * (b_px[si] - e[si])
    paired = pd.DataFrame({
        "symbol": base["symbol"].values[si], "entry_date": base["date"].values[si],
        "entry": e[si], "reference_low": b_ref[si],
        "B_exit_type": "new_day_low_stop", "B_exit_price": b_px[si], "B_return_pct": np.round(bret_s, 4),
        "A_exit_type": a_et[si], "A_exit_price": a_px[si], "A_return_pct": np.round(aret_s, 4),
        "B_pnl_inr": np.round(b_pnl_s, 0), "A_pnl_inr": np.round(a_pnl_s, 0),
        "stop_helped": b_pnl_s > a_pnl_s})
    paired_summary = {
        "n_stopped": len(si),
        "B_avg_return_pct": round(bret_s.mean(), 4), "B_median_return_pct": round(np.median(bret_s), 4),
        "A_avg_return_pct": round(aret_s.mean(), 4), "A_median_return_pct": round(np.median(aret_s), 4),
        "B_total_pnl_inr": round(b_pnl_s.sum(), 0), "A_total_pnl_inr": round(a_pnl_s.sum(), 0),
        "diff_total_pnl_B_minus_A_inr": round((b_pnl_s - a_pnl_s).sum(), 0),
        "pct_trades_stop_helped": round((b_pnl_s > a_pnl_s).mean() * 100, 2),
        "A_outcome_types": {k: int((a_et[si] == k).sum()) for k in set(a_et[si])},
    }

    # ── save ──
    b_out = pd.DataFrame({
        "symbol": base["symbol"].values, "entry_date": base["date"].values,
        "entry": e, "shares": sh.astype(int), "cap": cap,
        "exit_type": b_et, "exit_price": b_px, "reference_low": b_ref,
        "gross_ret_pct": np.round((b_px - e) / e * 100, 4)})[b_valid if False else slice(None)]
    b_out = b_out[b_valid]
    with pd.ExcelWriter(OUTDIR / "variant_ab_newlow_stop.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        exit_tbl.to_excel(w, sheet_name="exit_type_breakdown", index=False)
        subN_tbl.to_excel(w, sheet_name="variant_b_bucketN_split", index=False)
        b_out.to_excel(w, sheet_name="variant_b_trades", index=False)
        paired.to_excel(w, sheet_name="stopped_subset_paired", index=False)

    pd.set_option("display.width", 230)
    print("\n" + "=" * 120)
    print("VARIANT A vs B — comparison (gross then net)")
    print("=" * 120)
    show = ["variant", "basis", "n_trades", "total_return_fixedbase_pct", "total_pnl_inr",
            "win_rate_pct", "avg_return_per_trade_pct", "median_return_per_trade_pct",
            "avg_return_winning_trades_pct", "avg_return_losing_trades_pct", "avg_capital_deployed_per_trade"]
    print(comp[show].to_string(index=False))
    print("\n--- EXIT-TYPE BREAKDOWN ---")
    print(exit_tbl.to_string(index=False))
    print(f"\n--- VARIANT B: Bucket N sub-outcomes (non-positive at 09:45, n={nN}) ---")
    print(subN_tbl.to_string(index=False))
    print("\n--- PAIRED: trades STOPPED in B vs the SAME trades under A (held per A) ---")
    for k, v in paired_summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
