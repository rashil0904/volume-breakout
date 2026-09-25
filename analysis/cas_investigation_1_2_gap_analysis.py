# -*- coding: utf-8 -*-
"""cas_investigation_1_2_gap_analysis.py — CAS research, Investigations 1 & 2.
INV1: last-continuous-price (15:14 close) vs official CAS close (15:29 close) gap, every day 2026-08-03..08-25.
      Bias test + correlation of gap size with pre-15:15 realized vol / day's net move / VIX level.
INV2: does the gap predict next-day behavior? Tests mean-reversion vs momentum via correlation + sign hit-rate,
      against both the next-day OPEN reaction and the next-day's own continuous-session return.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
VIXF = rb.BASE / "data" / "india_vix_1min.csv"
OUTDIR = rb.RESULTS / "cas_investigation"; OUTDIR.mkdir(parents=True, exist_ok=True)
CAS_START = pd.Timestamp("2026-08-03")
CAS_END = pd.Timestamp("2026-08-25")


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute

    vx = pd.read_csv(VIXF); vts = pd.to_datetime(vx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    vser = vx.assign(ts=vts)[["ts", "close"]].dropna().sort_values("ts").set_index("ts")["close"]

    days = sorted(sp[(sp["date"] >= CAS_START) & (sp["date"] <= CAS_END)]["date"].unique())
    days = [pd.Timestamp(d) for d in days]
    print(f"CAS-era trading days: {len(days)}", flush=True)

    rows = []
    for d in days:
        day_data = sp[sp["date"] == d].sort_values("ts")
        pre15 = day_data[day_data["mod"] <= 555 + 359]  # 09:15 to 15:14 inclusive (mod 555..914)
        last_cont = day_data[day_data["mod"] == 914]  # 15:14 close
        cas_close_row = day_data[day_data["mod"] == 929]  # 15:29 close
        if last_cont.empty or cas_close_row.empty:
            continue
        last_price = float(last_cont["close"].iloc[0])
        cas_close = float(cas_close_row["close"].iloc[0])
        day_open = float(day_data["open"].iloc[0])

        gap_pts = cas_close - last_price
        gap_pct = gap_pts / last_price * 100

        # pre-15:15 realized vol: sum of squared 1-min log returns, 09:15->15:14
        px = pre15["close"].values
        logret = np.diff(np.log(px))
        rv = float(np.sqrt(np.sum(logret ** 2)) * 100)  # in %, realized-vol-style

        day_net_move_pct = (last_price - day_open) / day_open * 100
        vix_at_1514 = vser.asof(d + pd.Timedelta(hours=15, minutes=14))

        rows.append({"date": d.date(), "day_of_week": d.day_name(), "last_cont_1514": round(last_price, 2),
                     "cas_close_1529": round(cas_close, 2), "gap_pts": round(gap_pts, 2), "gap_pct": round(gap_pct, 4),
                     "day_open": round(day_open, 2), "day_net_move_pct_to_1514": round(day_net_move_pct, 3),
                     "realized_vol_pct_to_1514": round(rv, 3),
                     "vix_at_1514": round(float(vix_at_1514), 2) if pd.notna(vix_at_1514) else np.nan})

    G = pd.DataFrame(rows)
    G["next_day"] = G["date"].shift(-1)

    # ---- next-day metrics: overnight gap (CAS close -> next open) and next-day's own continuous return ----
    next_open = {}; next_last_cont = {}; next_cas_close = {}
    for d in days:
        dd = sp[sp["date"] == d]
        o = dd[dd["mod"] == 555]["open"]
        lc = dd[dd["mod"] == 914]["close"]
        cc = dd[dd["mod"] == 929]["close"]
        next_open[d.date()] = float(o.iloc[0]) if len(o) else np.nan
        next_last_cont[d.date()] = float(lc.iloc[0]) if len(lc) else np.nan
        next_cas_close[d.date()] = float(cc.iloc[0]) if len(cc) else np.nan

    def get_next(d, mapping):
        nd = None
        idx = days.index(pd.Timestamp(d))
        if idx + 1 < len(days):
            nd = days[idx + 1]
        return mapping.get(nd.date()) if nd is not None else np.nan

    G["next_day_open"] = G["date"].apply(lambda d: get_next(d, next_open))
    G["next_day_last_cont"] = G["date"].apply(lambda d: get_next(d, next_last_cont))
    G["next_day_cas_close"] = G["date"].apply(lambda d: get_next(d, next_cas_close))

    G["overnight_gap_pct"] = (G["next_day_open"] - G["cas_close_1529"]) / G["cas_close_1529"] * 100  # CAS close -> next open
    G["next_day_continuous_ret_pct"] = (G["next_day_last_cont"] - G["next_day_open"]) / G["next_day_open"] * 100  # next day's own session
    G["next_day_close_to_close_pct"] = (G["next_day_cas_close"] - G["cas_close_1529"]) / G["cas_close_1529"] * 100

    print("\n" + "=" * 100 + "\nINVESTIGATION 1 -- LAST-TRADE-TO-CAS-CLOSE GAP\n" + "=" * 100)
    pd.set_option("display.width", 220)
    print(G[["date", "day_of_week", "last_cont_1514", "cas_close_1529", "gap_pts", "gap_pct", "realized_vol_pct_to_1514", "day_net_move_pct_to_1514", "vix_at_1514"]].to_string(index=False))

    gaps = G["gap_pct"].dropna()
    print(f"\nn = {len(gaps)}")
    print(f"mean gap: {gaps.mean():.4f}%  | median: {gaps.median():.4f}%  | std: {gaps.std():.4f}%")
    print(f"positive gaps (up): {int((gaps>0).sum())}  | negative (down): {int((gaps<0).sum())}  | zero: {int((gaps==0).sum())}")
    tstat, pval = stats.ttest_1samp(gaps, 0)
    print(f"one-sample t-test vs 0: t={tstat:.3f}, p={pval:.4f}  {'(NOT significant, small n)' if pval>0.05 else '(significant, but n is tiny -- treat cautiously)'}")
    binom_p = stats.binomtest(int((gaps>0).sum()), len(gaps), 0.5).pvalue
    print(f"sign test (binomial, up vs down): p={binom_p:.4f}")
    print(f"abs gap range: {gaps.abs().min():.4f}% to {gaps.abs().max():.4f}%")

    print("\n--- correlation of |gap| with pre-15:15 observables ---")
    valid = G.dropna(subset=["gap_pct", "realized_vol_pct_to_1514", "day_net_move_pct_to_1514", "vix_at_1514"])
    for col in ["realized_vol_pct_to_1514", "day_net_move_pct_to_1514", "vix_at_1514"]:
        r_abs, p_abs = stats.pearsonr(valid[col], valid["gap_pct"].abs())
        r_signed, p_signed = stats.pearsonr(valid[col], valid["gap_pct"])
        print(f"  {col}: corr with |gap|  r={r_abs:+.3f} (p={p_abs:.3f})  |  corr with signed gap  r={r_signed:+.3f} (p={p_signed:.3f})   n={len(valid)}")

    print("\n" + "=" * 100 + "\nINVESTIGATION 2 -- DOES THE GAP PREDICT NEXT-DAY BEHAVIOR?\n" + "=" * 100)
    G2 = G.dropna(subset=["gap_pct", "overnight_gap_pct", "next_day_continuous_ret_pct"]).copy()
    print(f"n = {len(G2)} (gap -> next-day pairs)")
    print(G2[["date", "gap_pct", "overnight_gap_pct", "next_day_continuous_ret_pct", "next_day_close_to_close_pct"]].to_string(index=False))
    print()
    for target in ["overnight_gap_pct", "next_day_continuous_ret_pct", "next_day_close_to_close_pct"]:
        r, p = stats.pearsonr(G2["gap_pct"], G2[target])
        slope, intercept, r2, p2, se = stats.linregress(G2["gap_pct"], G2[target])
        same_sign = int(np.sign(G2["gap_pct"]) .eq(np.sign(G2[target])).sum())
        n = len(G2)
        print(f"  gap_pct -> {target}: r={r:+.3f} (p={p:.3f}) | slope={slope:+.3f} R^2={r2**2:.3f} | same-sign (momentum) hit-rate: {same_sign}/{n} = {same_sign/n*100:.1f}%")
        interp = "MOMENTUM (same sign)" if r > 0 else ("MEAN-REVERSION (opposite sign)" if r < 0 else "no relationship")
        print(f"     direction suggested by sign of r: {interp}  -- p={p:.3f}, {'NOT statistically significant (n=%d, expected with this sample size)'%n if p>0.05 else 'nominally significant but n=%d is very small, treat as suggestive only'%n}")
        print()

    with pd.ExcelWriter(OUTDIR / "cas_inv1_2_gap_analysis.xlsx", engine="openpyxl") as w:
        G.to_excel(w, sheet_name="Gap_Data", index=False)

    G.to_csv(OUTDIR / "cas_gap_data.csv", index=False)
    print(f"Saved -> {OUTDIR}/cas_inv1_2_gap_analysis.xlsx")


if __name__ == "__main__":
    main()
