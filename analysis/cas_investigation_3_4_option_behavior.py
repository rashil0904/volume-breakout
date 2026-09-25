# -*- coding: utf-8 -*-
"""cas_investigation_3_4_option_behavior.py — CAS research, Investigations 3 & 4.
INV3: ATM/near-ATM option premium behavior during 15:15-15:39 (the CAS window, options data extends 10min
      beyond spot). Uses is_synthetic flag as an objective stale-quote/no-real-trade proxy, plus high-low
      range as a bid-ask-width proxy. Compares against late-July pre-CAS equivalent (last 15 real minutes
      of continuous trading, 15:15-15:29).
INV4: ATM straddle premium behavior in the last hour before the cutoff (14:15-15:15) now vs pre-CAS, as a
      rough IV-proxy / front-running check.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
OUTDIR = rb.RESULTS / "cas_investigation"; OUTDIR.mkdir(parents=True, exist_ok=True)
CAS_START = pd.Timestamp("2026-08-03")
CAS_END = pd.Timestamp("2026-08-25")
PRE_START = pd.Timestamp("2026-07-14")  # last ~2 weeks pre-CAS for baseline
PRE_END = pd.Timestamp("2026-07-31")


def nearest_expiry(d, expiries):
    for e in expiries:
        if e >= d:
            return e
    return None


def leg(folder, strike, ot):
    fs = glob.glob(str(OPTDIR / folder / f"NIFTY_{int(strike)}_{ot}_*.parquet"))
    if not fs:
        return None
    return pd.read_parquet(fs[0])


def main():
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    sp["date"] = sp["ts"].dt.normalize(); sp["mod"] = sp["ts"].dt.hour * 60 + sp["ts"].dt.minute
    atm_1514 = sp[sp["mod"] == 914].groupby("date")["close"].last()  # 15:14 close for ATM basis

    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d") if False else None
    import os
    expiries = sorted(pd.Timestamp(x) for x in os.listdir(OPTDIR))

    def build_window_stats(days, window_start_m, window_end_m, tag):
        rows = []
        for d in days:
            if d.date() not in atm_1514.index.date if hasattr(atm_1514.index, 'date') else True:
                pass
            spot_ref = atm_1514.get(d, np.nan)
            if pd.isna(spot_ref):
                continue
            atm = round(spot_ref / 50) * 50
            exp = nearest_expiry(d, expiries)
            if exp is None:
                continue
            folder = exp.strftime("%Y%m%d")
            ce = leg(folder, atm, "CE"); pe = leg(folder, atm, "PE")
            if ce is None or pe is None:
                continue
            for name, df in [("CE", ce), ("PE", pe)]:
                dd = df[df["timestamp"].dt.normalize() == d]
                dd = dd[(dd["timestamp"].dt.hour * 60 + dd["timestamp"].dt.minute >= window_start_m) &
                        (dd["timestamp"].dt.hour * 60 + dd["timestamp"].dt.minute <= window_end_m)]
                if dd.empty:
                    continue
                has_synth_col = "is_synthetic" in dd.columns
                synth_frac = dd["is_synthetic"].mean() if has_synth_col else np.nan
                hl_range_pct = ((dd["high"] - dd["low"]) / dd["close"].replace(0, np.nan)).mean() * 100
                premium_std_pct = dd["close"].pct_change().std() * 100
                max_1min_move_pct = dd["close"].pct_change().abs().max() * 100
                rows.append({"date": d.date(), "window": tag, "leg": name, "strike": atm, "expiry": exp.date(),
                             "n_candles": len(dd), "synthetic_frac_%": round(synth_frac * 100, 1) if pd.notna(synth_frac) else np.nan,
                             "avg_hl_range_%": round(hl_range_pct, 3) if pd.notna(hl_range_pct) else np.nan,
                             "premium_1min_std_%": round(premium_std_pct, 3) if pd.notna(premium_std_pct) else np.nan,
                             "max_1min_move_%": round(max_1min_move_pct, 3) if pd.notna(max_1min_move_pct) else np.nan,
                             "avg_volume": round(dd["volume"].mean(), 1), "avg_premium": round(dd["close"].mean(), 2)})
        return pd.DataFrame(rows)

    days_cas = sorted(sp[(sp["date"] >= CAS_START) & (sp["date"] <= CAS_END)]["date"].unique())
    days_cas = [pd.Timestamp(d) for d in days_cas]
    days_pre = sorted(sp[(sp["date"] >= PRE_START) & (sp["date"] <= PRE_END)]["date"].unique())
    days_pre = [pd.Timestamp(d) for d in days_pre]

    print(f"CAS-era days: {len(days_cas)} | pre-CAS baseline days: {len(days_pre)}", flush=True)

    print("\n" + "=" * 100 + "\nINVESTIGATION 3 -- ATM OPTION BEHAVIOR DURING THE 15:15-15:39 CAS WINDOW\n" + "=" * 100)
    cas_window = build_window_stats(days_cas, 915, 939, "CAS_window_1515_1539")
    pre_window = build_window_stats(days_pre, 915, 929, "PRE_last15min_1515_1529")

    print("\n--- CAS window (post-Aug-3), by day, CE+PE combined avg ---")
    agg_cas = cas_window.groupby("date").agg(synthetic_frac_pct=("synthetic_frac_%", "mean"),
                                              avg_hl_range_pct=("avg_hl_range_%", "mean"),
                                              premium_1min_std_pct=("premium_1min_std_%", "mean"),
                                              max_1min_move_pct=("max_1min_move_%", "max"),
                                              avg_volume=("avg_volume", "mean")).reset_index()
    pd.set_option("display.width", 200)
    print(agg_cas.round(2).to_string(index=False))

    print("\n--- PRE-CAS baseline (late July), same aggregation ---")
    agg_pre = pre_window.groupby("date").agg(synthetic_frac_pct=("synthetic_frac_%", "mean"),
                                              avg_hl_range_pct=("avg_hl_range_%", "mean"),
                                              premium_1min_std_pct=("premium_1min_std_%", "mean"),
                                              max_1min_move_pct=("max_1min_move_%", "max"),
                                              avg_volume=("avg_volume", "mean")).reset_index()
    print(agg_pre.round(2).to_string(index=False))

    print("\n--- SUMMARY: CAS window vs pre-CAS baseline (means across days) ---")
    for col in ["synthetic_frac_pct", "avg_hl_range_pct", "premium_1min_std_pct", "max_1min_move_pct", "avg_volume"]:
        print(f"  {col}: CAS={agg_cas[col].mean():.2f}  |  PRE={agg_pre[col].mean():.2f}  |  ratio CAS/PRE={agg_cas[col].mean()/max(agg_pre[col].mean(),1e-9):.2f}x")

    print("\n--- Days with highest synthetic_frac (most stale/no-real-trade) in CAS window ---")
    print(agg_cas.sort_values("synthetic_frac_pct", ascending=False).head(6).round(2).to_string(index=False))

    print("\n--- Days with highest max_1min_move_pct (most abnormal single-candle premium swing) in CAS window ---")
    print(agg_cas.sort_values("max_1min_move_pct", ascending=False).head(6).round(2).to_string(index=False))

    # ---- INVESTIGATION 4 ----
    print("\n" + "=" * 100 + "\nINVESTIGATION 4 -- LAST-HOUR (14:15-15:15) OPTION BEHAVIOR: CAS-ERA vs PRE-CAS\n" + "=" * 100)
    last_hour_cas = build_window_stats(days_cas, 855, 915, "lasthour_1415_1515_CAS")
    last_hour_pre = build_window_stats(days_pre, 855, 915, "lasthour_1415_1515_PRE")
    agg_lh_cas = last_hour_cas.groupby("date").agg(avg_hl_range_pct=("avg_hl_range_%", "mean"),
                                                    premium_1min_std_pct=("premium_1min_std_%", "mean"),
                                                    avg_volume=("avg_volume", "mean")).reset_index()
    agg_lh_pre = last_hour_pre.groupby("date").agg(avg_hl_range_pct=("avg_hl_range_%", "mean"),
                                                    premium_1min_std_pct=("premium_1min_std_%", "mean"),
                                                    avg_volume=("avg_volume", "mean")).reset_index()
    print("CAS-era last-hour (mean across days):")
    print(agg_lh_cas[["avg_hl_range_pct", "premium_1min_std_pct", "avg_volume"]].mean().round(3))
    print("\nPre-CAS last-hour (mean across days):")
    print(agg_lh_pre[["avg_hl_range_pct", "premium_1min_std_pct", "avg_volume"]].mean().round(3))

    with pd.ExcelWriter(OUTDIR / "cas_inv3_4_option_behavior.xlsx", engine="openpyxl") as w:
        agg_cas.to_excel(w, sheet_name="CAS_Window_ByDay", index=False)
        agg_pre.to_excel(w, sheet_name="PreCAS_Baseline_ByDay", index=False)
        cas_window.to_excel(w, sheet_name="CAS_Window_Detail", index=False)
        agg_lh_cas.to_excel(w, sheet_name="LastHour_CAS", index=False)
        agg_lh_pre.to_excel(w, sheet_name="LastHour_PreCAS", index=False)
    print(f"\nSaved -> {OUTDIR}/cas_inv3_4_option_behavior.xlsx")


if __name__ == "__main__":
    import os
    main()
