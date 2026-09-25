# -*- coding: utf-8 -*-
"""
uc_conversion_diag.py — of the strategy's qualifying stock-days that crossed +19% on the ENTRY
day, how many went on to hit the upper circuit (+19.95%/~+20%)? Entry-day 1-min path diagnostic
(not a trade recompute).

SCOPE (default a): the strategy's qualifying stock-days (passes_all_three: mcap ₹1,500-5,000 Cr,
LB 36, VM 6, +5% vs prev-day VWAP-close, 3:15pm entry). [Flag (a): broader ₹1,500-5,000 universe
not used here.]
LEVELS off prev_close (= prev_day_vwap_close, VWAP-close ref), detected on entry-day 1-min HIGHS:
  +19% = prev_close × 1.19 ;  UC = prev_close × 1.1995 (~+20%).
Note (flag d): any stock that reaches +19% necessarily has a >=20% circuit band, so the ~20% UC
assumption is valid for this crossed-19% subset (tighter-band stocks lock below +19%).
Entry day = full session 09:15-15:29.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = rb.IST
OUTDIR = rb.RESULTS / "uc_conversion_diag"
HM_OPEN, HM_CLOSE = 555, 929
UC_MULT, L19_MULT = 1.1995, 1.19


def hm_lbl(hm):
    return f"{int(hm)//60:02d}:{int(hm)%60:02d}" if hm == hm else ""


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_all_three", "prev_day_vwap_close"],
                       parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True].reset_index(drop=True)
    print(f"qualifying stock-days: {len(Q):,}")

    rows = []
    for sym, sub in Q.groupby("symbol"):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq, columns=["timestamp", "high"])
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
        raw = raw[(raw["hm"] >= HM_OPEN) & (raw["hm"] <= HM_CLOSE)]
        by_date = {d: g for d, g in raw.groupby("date")}
        for _, r in sub.iterrows():
            d = r["date"]; pc = float(r["prev_day_vwap_close"])
            g = by_date.get(d)
            if g is None or not (pc == pc and pc > 0):
                continue
            hm = g["hm"].values; hi = g["high"].values.astype(float)
            l19, uc = pc * L19_MULT, pc * UC_MULT
            max_ret = (np.nanmax(hi) - pc) / pc * 100
            c19 = hi >= l19
            if not c19.any():
                rows.append((sym, d, False, np.nan, False, np.nan, round(max_ret, 3))); continue
            f19 = int(hm[c19].min())
            cuc = hi >= uc
            fuc = int(hm[cuc].min()) if cuc.any() else np.nan
            rows.append((sym, d, True, f19, cuc.any(), fuc, round(max_ret, 3)))

    D = pd.DataFrame(rows, columns=["symbol", "date", "crossed_19", "cross_19_hm",
                                    "hit_uc", "uc_hm", "max_return_reached"])
    D["cross_19_time"] = D["cross_19_hm"].map(hm_lbl)
    D["uc_time"] = D["uc_hm"].map(hm_lbl)
    D["min_19_to_uc"] = D["uc_hm"] - D["cross_19_hm"]

    cross = D[D["crossed_19"]]
    conv = cross[cross["hit_uc"]]
    nocv = cross[~cross["hit_uc"]]
    n19, nuc = len(cross), len(conv)
    conv_rate = nuc / n19 * 100 if n19 else 0.0

    # max-return histogram among +19% crossers
    edges = [19.0, 19.2, 19.4, 19.6, 19.8, 19.95, 100]
    labels = ["19.0-19.2", "19.2-19.4", "19.4-19.6", "19.6-19.8", "19.8-19.95", ">=19.95 (UC)"]
    cross = cross.copy()
    cross["bucket"] = pd.cut(cross["max_return_reached"], bins=edges, labels=labels, right=False)
    hist = cross["bucket"].value_counts().reindex(labels).fillna(0).astype(int)
    hist_df = pd.DataFrame({"max_return_bucket": labels, "n": hist.values,
                            "pct_of_crossers": (hist.values / n19 * 100).round(2)})

    summary = pd.DataFrame([{
        "qualifying_stock_days": len(Q),
        "n_crossed_19": n19,
        "pct_of_qualifying_that_crossed_19": round(n19 / len(Q) * 100, 2),
        "n_hit_uc": nuc,
        "conversion_rate_19_to_uc_pct": round(conv_rate, 2),
        "n_crossed_19_but_no_uc": len(nocv),
        "pct_crossers_no_uc": round(len(nocv) / n19 * 100, 2) if n19 else 0.0,
        "avg_max_return_no_uc_group": round(float(nocv["max_return_reached"].mean()), 3) if len(nocv) else np.nan,
        "median_max_return_no_uc_group": round(float(nocv["max_return_reached"].median()), 3) if len(nocv) else np.nan,
        "avg_min_19_to_uc": round(float(conv["min_19_to_uc"].mean()), 2) if len(conv) else np.nan,
        "median_min_19_to_uc": round(float(conv["min_19_to_uc"].median()), 2) if len(conv) else np.nan,
        "avg_cross_19_time": hm_lbl(round(float(cross["cross_19_hm"].mean()))) if n19 else "",
        "median_cross_19_time": hm_lbl(float(cross["cross_19_hm"].median())) if n19 else "",
        "avg_cross_19_time_converters": hm_lbl(round(float(conv["cross_19_hm"].mean()))) if len(conv) else "",
        "avg_cross_19_time_no_uc": hm_lbl(round(float(nocv["cross_19_hm"].mean()))) if len(nocv) else "",
    }])

    D.to_csv(OUTDIR / "uc_conversion_detail.csv", index=False)
    summary.to_csv(OUTDIR / "uc_conversion_summary.csv", index=False)
    hist_df.to_csv(OUTDIR / "uc_conversion_maxret_histogram.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78 + "\nUC CONVERSION — of +19% crossers, how many hit UC (~+20%)?\n" + "=" * 78)
    print(f"  qualifying stock-days           : {len(Q):,}")
    print(f"  crossed +19% intraday           : {n19:,}  ({n19/len(Q)*100:.1f}% of qualifying)")
    print(f"  of those, hit UC (~+20%)        : {nuc:,}")
    print(f"  CONVERSION RATE (+19% -> UC)    : {conv_rate:.2f}%")
    print(f"\n  crossed +19% but NO UC          : {len(nocv):,}  ({len(nocv)/n19*100:.1f}% of crossers)")
    print(f"    avg / median max-return topped : {nocv['max_return_reached'].mean():.3f}% / {nocv['max_return_reached'].median():.3f}%")
    print(f"  time +19% -> UC (converters)    : avg {conv['min_19_to_uc'].mean():.1f} min / median {conv['min_19_to_uc'].median():.0f} min")
    print(f"  avg time crossing +19%          : all {hm_lbl(round(float(cross['cross_19_hm'].mean())))} | "
          f"converters {hm_lbl(round(float(conv['cross_19_hm'].mean())))} | no-UC {hm_lbl(round(float(nocv['cross_19_hm'].mean())))}")
    print("\n  MAX-RETURN-REACHED distribution among +19% crossers:")
    print(hist_df.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
