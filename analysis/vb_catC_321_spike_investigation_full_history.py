# -*- coding: utf-8 -*-
"""vb_catC_321_spike_investigation.py — diagnostic: what does the price actually do between 3:13pm and 3:21pm for
the stock-days that make up Category C, in the SAME universe as vb_catC_entry_time_sweep_apr_jul2026.py (the
locked BC.build_cache() cache, restricted to Apr-Jul 2026)? For each such stock-day, pull the 1-min OPEN at
3:13,3:14,...,3:20,3:21 and compute % diff of each earlier minute vs the 3:21 price. Diagnostic only -- no
strategy-logic changes.

POPULATION: stock-days classified Category C (entered=True) by the UNCHANGED, real BC.classify("baseline", ...)
-- i.e. the actual set of stock-days whose OWN fill price/time is what the entry-time sweep varies. By
construction (catB_or_C's touch_B check), no Category C stock-day ever touched UC in 15:00-15:21 -- that's what
makes it Category B instead. So "near-UC" here means got CLOSE to UC (by the day's own high before 15:21) without
actually touching it, not an actual circuit lock.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC

OUTDIR = rb.RESULTS / "baseline_and_cross_final" / "catC_321_spike_investigation_full_history"; OUTDIR.mkdir(parents=True, exist_ok=True)
WIN_START = pd.Timestamp("2022-03-01").date(); WIN_END = pd.Timestamp("2099-12-31").date()   # full history from Mar-2022 through whatever the locked cache covers
MINUTES = list(range(913, 922))    # 15:13 .. 15:21 inclusive
HM_1521 = BC.HM_1521
NEAR_UC_THRESH = 0.95               # day's own high before 15:21, as a fraction of the UC level


def main():
    print("Building cache (identical to locked baseline)...", flush=True)
    cache_full = BC.build_cache()
    cache = [c for c in cache_full if WIN_START <= c["entry_date"] <= WIN_END]
    print(f"cache: {len(cache_full):,} total stock-days | {len(cache):,} in Apr-Jul 2026 window", flush=True)

    rows = []
    n_catC_total = 0
    for c in cache:
        pc = c["pc"]; hm, hi, lo, op = c["eg"]["hm"], c["eg"]["hi"], c["eg"]["lo"], c["eg"]["op"]
        cat, entered, legs, meta = BC.classify("baseline", pc, hm, hi, lo, op)
        if cat != "C":
            continue
        n_catC_total += 1
        if not entered:
            continue          # circuit-locked / no valid 3:21 price -- excluded from the price-diff analysis
        o = dict(zip(hm, op))
        p1521 = o.get(HM_1521, np.nan)
        if not (p1521 == p1521 and p1521 > 0):
            continue
        uc = pc * BC.UC_M
        before1521 = hm < HM_1521
        high_before_1521 = float(hi[before1521].max()) if before1521.any() else np.nan
        proximity_to_uc = high_before_1521 / uc if (high_before_1521 == high_before_1521 and uc > 0) else np.nan
        near_uc = bool(proximity_to_uc >= NEAR_UC_THRESH) if proximity_to_uc == proximity_to_uc else False
        row = {"symbol": c["symbol"], "entry_date": c["entry_date"], "price_1521": p1521,
               "proximity_to_uc_pct": round(proximity_to_uc * 100, 2) if proximity_to_uc == proximity_to_uc else np.nan, "near_uc": near_uc}
        for m in MINUTES[:-1]:
            pm = o.get(m, np.nan)
            row[f"price_{BC.lbl(m).replace(':','')}"] = pm
            row[f"pctdiff_{BC.lbl(m).replace(':','')}"] = round((p1521 - pm) / pm * 100, 4) if (pm == pm and pm > 0) else np.nan
        rows.append(row)

    D = pd.DataFrame(rows)
    print(f"\nCategory C stock-days (all, incl. non-entered/circuit-locked): {n_catC_total}", flush=True)
    print(f"Category C stock-days WITH a valid 3:21 price (entered, used below): {len(D)}", flush=True)
    print(f"  of which near-UC (day-high before 15:21 >= {NEAR_UC_THRESH*100:.0f}% of the UC level): {int(D['near_uc'].sum())}", flush=True)
    print(f"  of which NOT near-UC: {int((~D['near_uc']).sum())}", flush=True)

    def summarize(df, label):
        recs = []
        for m in MINUTES[:-1]:
            col = f"pctdiff_{BC.lbl(m).replace(':','')}"
            x = df[col].dropna()
            recs.append({"scope": label, "minute": BC.lbl(m), "hm": m, "n": len(x),
                         "avg_pctdiff": round(x.mean(), 4) if len(x) else np.nan, "median_pctdiff": round(x.median(), 4) if len(x) else np.nan,
                         "max_pctdiff": round(x.max(), 3) if len(x) else np.nan, "min_pctdiff": round(x.min(), 3) if len(x) else np.nan,
                         "std_pctdiff": round(x.std(), 4) if len(x) else np.nan})
        return pd.DataFrame(recs)

    S_all = summarize(D, "ALL_catC")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n=== ALL Category C stock-days: 3:21pm price vs each earlier minute (%) ===\n" + S_all.to_string(index=False), flush=True)

    S_near = summarize(D[D["near_uc"]], "NEAR_UC")
    S_far = summarize(D[~D["near_uc"]], "NOT_near_UC")
    print(f"\n=== NEAR-UC (n={int(D['near_uc'].sum())}) ===\n" + S_near.to_string(index=False), flush=True)
    print(f"\n=== NOT near-UC (n={int((~D['near_uc']).sum())}) ===\n" + S_far.to_string(index=False), flush=True)

    # ---- chart ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [BC.lbl(m) for m in MINUTES[:-1]]
    ax.plot(x, S_all["avg_pctdiff"], marker="o", label=f"ALL Category C (n={len(D)})", linewidth=2)
    ax.plot(x, S_near["avg_pctdiff"], marker="s", linestyle="--", label=f"Near-UC (n={int(D['near_uc'].sum())})")
    ax.plot(x, S_far["avg_pctdiff"], marker="^", linestyle="--", label=f"Not near-UC (n={int((~D['near_uc']).sum())})")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Minute (vs 3:21pm price)"); ax.set_ylabel("Avg % diff: (3:21pm price - minute price) / minute price x 100")
    ax.set_title("Category C: price drift into 3:21pm, Mar 2022 - Jul 2026")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    chart_fn = OUTDIR / "catC_321_spike_price_drift_full_history.png"
    fig.savefig(chart_fn, dpi=150)
    print(f"\nChart saved -> {chart_fn}", flush=True)

    fn = OUTDIR / "vb_catC_321_spike_investigation_full_history.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": f"Population = Category C stock-days (entered=True, valid 3:21 price) in the SAME cache/window as "
                                "vb_catC_entry_time_sweep_apr_jul2026.py (Apr-Jul 2026). By construction no Cat-C stock-day ever "
                                "touched UC in 15:00-15:21 (that would make it Category B) -- 'near-UC' here means the day's own high "
                                f"before 15:21 reached >= {NEAR_UC_THRESH*100:.0f}% of the UC level without crossing it."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        S_all.to_excel(w, sheet_name="Summary_ALL", index=False)
        S_near.to_excel(w, sheet_name="Summary_NEAR_UC", index=False)
        S_far.to_excel(w, sheet_name="Summary_NOT_near_UC", index=False)
        D.to_excel(w, sheet_name="Per_stockday_detail", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    print(f"Saved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
