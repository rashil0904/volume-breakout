# -*- coding: utf-8 -*-
"""spread_adverse_reversal.py — EXPLORATORY adverse-move-then-reversal study on NIFTY SPOT within each Weekly
Credit Spread trade's own entry->expiry window. Descriptive ONLY (no averaging rule implemented).

Adverse = spot moving against the trade (CCS/bearish -> UP ; PCS/bullish -> DOWN), measured % from entry spot,
1-min touch basis. Def1: after touching X% adverse, does spot revert to ENTRY before expiry (+ time). Def2:
2D grid X% adverse x Y% reversal-from-adverse-extreme (max favourable retracement from the running adverse
peak, measured after first touching X%). Split by direction (CCS/PCS) and IS/OOS.
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SPOT = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OPTDIR = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
TRADES = rb.RESULTS / "weekly_credit_spread" / "weekly_credit_spread_trades.csv"
OUTDIR = rb.RESULTS / "spread_adverse_reversal"; OUTDIR.mkdir(parents=True, exist_ok=True)
XS = [round(x * 0.5, 1) for x in range(1, 11)]        # 0.5 .. 5.0 adverse %
YS = [round(y * 0.5, 1) for y in range(1, 7)]         # 0.5 .. 3.0 reversal-from-extreme %


def per_trade(hi, lo, tsr, P, ccs):
    """returns per-X dicts: touched, revert_entry, revert_hours, rev_ext(max favourable retrace after touch)."""
    if ccs: adv = (hi - P) / P * 100.0                # CCS adverse = UP (use highs)
    else: adv = (P - lo) / P * 100.0                  # PCS adverse = DOWN (use lows)
    out = {}
    for X in XS:
        idx = np.where(adv >= X)[0]
        if len(idx) == 0:
            out[X] = (False, False, np.nan, 0.0); continue
        ti = idx[0]
        if ccs:                                       # revert to entry = low back to P ; retrace = peak(hi) - lo
            rev_entry = np.where(lo[ti:] <= P)[0]
            rp = np.maximum.accumulate(hi[ti:]); retr = (rp - lo[ti:]) / P * 100.0
        else:                                         # revert to entry = high back to P ; retrace = hi - trough(lo)
            rev_entry = np.where(hi[ti:] >= P)[0]
            rt = np.minimum.accumulate(lo[ti:]); retr = (hi[ti:] - rt) / P * 100.0
        reverted = len(rev_entry) > 0
        rhours = float((tsr[ti + rev_entry[0]] - tsr[ti]) / np.timedelta64(1, "h")) if reverted else np.nan
        out[X] = (True, reverted, rhours, float(np.max(retr)))
    return out


def main():
    T = pd.read_csv(TRADES)
    sp = pd.read_csv(SPOT); ts = pd.to_datetime(sp["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    sp = sp.assign(ts=ts).sort_values("ts").reset_index(drop=True)
    tsv = sp["ts"].values; hv = sp["high"].values; lv = sp["low"].values; cv = sp["close"].values
    expiries = pd.to_datetime(sorted(os.listdir(OPTDIR)), format="%Y%m%d")

    recs = []
    for r in T.itertuples():
        et = pd.Timestamp(r.entry_time); entry_day = pd.Timestamp(r.entry_date)
        exp = [e for e in expiries if e > entry_day][0]
        i0 = np.searchsorted(tsv, np.datetime64(et), "left")
        i1 = np.searchsorted(tsv, np.datetime64(exp + pd.Timedelta(hours=15, minutes=30)), "right")
        if i1 - i0 < 5:
            continue
        P = float(cv[i0])                              # entry spot = spot close at entry minute
        ccs = (r.type == "Call Credit Spread")
        d = per_trade(hv[i0:i1], lv[i0:i1], tsv[i0:i1], P, ccs)
        recs.append({"entry_date": entry_day, "ccs": ccs, "d": d})
    n = len(recs); split = sorted([x["entry_date"] for x in recs])[n // 2]

    def agg_def1(sub):
        rows = []
        for X in XS:
            tt = [x["d"][X] for x in sub if x["d"][X][0]]           # touched X
            nt = len(tt); rv = [t for t in tt if t[1]]
            rows.append({"adverse_X%": X, "n_touched": nt, "n_revert_to_entry": len(rv),
                         "pct_revert": round(len(rv) / nt * 100, 1) if nt else 0,
                         "avg_clockhrs_to_revert": round(np.mean([t[2] for t in rv]), 1) if rv else np.nan,
                         "avg_caldays_to_revert": round(np.mean([t[2] for t in rv]) / 24.0, 2) if rv else np.nan})
        return pd.DataFrame(rows)

    def agg_def2(sub):
        grid = {"adverse_X%": XS}
        for Y in YS:
            col = []
            for X in XS:
                tt = [x["d"][X] for x in sub if x["d"][X][0]]
                nt = len(tt); nrev = sum(1 for t in tt if t[3] >= Y)
                col.append(round(nrev / nt * 100, 1) if nt else 0)
            grid[f"revY>={Y}%"] = col
        return pd.DataFrame(grid)

    groups = {"ALL": recs, "CCS(bearish)": [x for x in recs if x["ccs"]], "PCS(bullish)": [x for x in recs if not x["ccs"]],
              "IS(early)": [x for x in recs if x["entry_date"] < split], "OOS(late)": [x for x in recs if x["entry_date"] >= split]}

    with pd.ExcelWriter(OUTDIR / "adverse_reversal.xlsx", engine="openpyxl") as w:
        d1all = agg_def1(recs)
        # highlights: X with high revert-to-entry% (min 15 touches) and grid cells with high reversal freq
        cand = d1all[d1all.n_touched >= 15].sort_values("pct_revert", ascending=False)
        pd.DataFrame([
            {"metric": "Study", "value": "EXPLORATORY adverse-then-reversal on NIFTY spot per credit-spread trade window (descriptive only)"},
            {"metric": "Trades analysed", "value": n}, {"metric": "Adverse X% sweep", "value": str(XS)}, {"metric": "Reversal Y% sweep", "value": str(YS)},
            {"metric": "Adverse defn", "value": "CCS->spot UP, PCS->spot DOWN, % from entry spot, 1-min touch"},
            {"metric": "Def1", "value": "after touching X% adverse, spot reverts to ENTRY before expiry (+time)"},
            {"metric": "Def2", "value": "2D grid: among X%-touchers, % where favourable retrace from running adverse peak (after touch) >= Y%"},
            {"metric": "IS/OOS split date", "value": str(pd.Timestamp(split).date())},
            {"metric": "FLAG", "value": "DESCRIPTIVE / EXPLORATORY only — NOT a validated averaging rule"},
            {"metric": "Candidate X (high revert-to-entry, >=15 touches)", "value": ", ".join(f"{r['adverse_X%']}%={r['pct_revert']}%" for _, r in cand.head(4).iterrows())},
        ]).to_excel(w, sheet_name="Summary", index=False)
        for name, sub in groups.items():
            agg_def1(sub).to_excel(w, sheet_name=f"Def1_{name[:22]}", index=False)
        for name in ["ALL", "CCS(bearish)", "PCS(bullish)", "IS(early)", "OOS(late)"]:
            agg_def2(groups[name]).to_excel(w, sheet_name=f"Def2_{name[:22]}", index=False)

    pd.set_option("display.width", 220)
    print("=" * 96 + "\nADVERSE-MOVE -> REVERSAL STUDY (NIFTY spot, per credit-spread window) — EXPLORATORY\n" + "=" * 96)
    print(f"trades {n} | IS/OOS split {pd.Timestamp(split).date()} | X {XS} | Y {YS}")
    print("\n--- DEF1: revert-to-ENTRY after touching X% adverse (ALL) ---"); print(agg_def1(recs).to_string(index=False))
    print("\n--- DEF1 by direction ---")
    print("CCS(bearish):"); print(agg_def1(groups["CCS(bearish)"])[["adverse_X%", "n_touched", "pct_revert", "avg_caldays_to_revert"]].to_string(index=False))
    print("PCS(bullish):"); print(agg_def1(groups["PCS(bullish)"])[["adverse_X%", "n_touched", "pct_revert", "avg_caldays_to_revert"]].to_string(index=False))
    print("\n--- DEF2 grid: % of X-touchers that retrace >=Y% from adverse peak (ALL) ---"); print(agg_def2(recs).to_string(index=False))
    print("\n--- DEF2 IS vs OOS (stability check, ALL dirs) ---")
    print("IS:"); print(agg_def2(groups["IS(early)"]).to_string(index=False)); print("OOS:"); print(agg_def2(groups["OOS(late)"]).to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
