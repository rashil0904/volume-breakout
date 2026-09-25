# -*- coding: utf-8 -*-
"""newly_listed_first_results_ath_filter.py — segments the "newly listed stocks - first quarterly result" trades
by whether the stock printed a NEW ALL-TIME HIGH (since its own listing/first available data -- these are newly
listed stocks, so "all-time" is necessarily "since listing") within 1 week (5 trading days) after the reaction day.

ASSUMPTION FLAGGED (not specified, stated explicitly): "within 1 week after posting result" is read as the 5
trading days starting from and including the reaction day itself (reaction_day .. reaction_day+5 trading days,
6 trading days spanning the window), since that's when the market first has a chance to react/trade the result.
A day counts as an ATH hit if its daily HIGH is >= the running maximum of every prior day's high (a genuine new
high, not a tie with an earlier day). Reuses entry/reaction-day logic + all three exit variants from
newly_listed_first_results_hold_sweep.py unmodified. New file; originals untouched.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import newly_listed_first_results_backtest as N
import newly_listed_first_results_hold_sweep as H
from ipo_consolidation_breakout_backtest import daily_ohlc      # reused unmodified, gives open/high/low/close

OUT = rb.RESULTS / "newly_listed_first_results_ath_filter"; OUT.mkdir(parents=True, exist_ok=True)
SRC = rb.RESULTS / "newly_listed_first_results"
ATH_WINDOW_DAYS = 5     # trading days after (and including) the reaction day


def ath_flag(sym, reaction_day):
    D = daily_ohlc(sym)
    if D is None or reaction_day not in D.index:
        return None
    tdays = list(D.index); hi = D["high"].values
    r_idx = tdays.index(reaction_day)
    cummax_before = np.maximum.accumulate(hi)          # cummax_before[j] = max(hi[0..j]) inclusive
    hit_day, hit_price = None, None
    for j in range(r_idx, min(r_idx + ATH_WINDOW_DAYS, len(tdays) - 1) + 1):
        prior_max = cummax_before[j - 1] if j > 0 else -np.inf
        if hi[j] >= prior_max - 1e-9:
            hit_day, hit_price = tdays[j], float(hi[j]); break
    return {"hit_ath": hit_day is not None, "ath_date": hit_day, "ath_price": hit_price,
            "days_reaction_to_ath": (hit_day - reaction_day).days if hit_day else None}


def stats(x):
    if len(x) == 0:
        return {"n": 0}
    return {"n": len(x), "avg_pct": round(x.mean(), 3), "median_pct": round(x.median(), 3), "win_pct": round((x > 0).mean() * 100, 1),
            "std_pct": round(x.std(), 3)}


def main():
    FR = pd.read_csv(SRC / "first_results_raw.csv", parse_dates=["announcement_datetime"])
    V1, V2, V3, DI = H.build(FR)
    print(f"variant1 (morning open): {len(V1)} | variant2 (closing): {len(V2)} | variant3 rows: {len(V3)}", flush=True)

    keys = V1[["symbol", "reaction_day"]].drop_duplicates()
    flags = []
    for r in keys.itertuples():
        f = ath_flag(r.symbol, r.reaction_day)
        if f is None:
            f = {"hit_ath": False, "ath_date": None, "ath_price": None, "days_reaction_to_ath": None}
        flags.append({"symbol": r.symbol, "reaction_day": r.reaction_day, **f})
    FL = pd.DataFrame(flags)
    n_hit = int(FL["hit_ath"].sum())
    print(f"\ntrades with a NEW ATH within {ATH_WINDOW_DAYS} trading days of the reaction day: {n_hit} of {len(FL)} ({n_hit/len(FL)*100:.1f}%)", flush=True)

    V1 = V1.merge(FL[["symbol", "reaction_day", "hit_ath", "ath_date", "ath_price", "days_reaction_to_ath"]], on=["symbol", "reaction_day"])
    V2 = V2.merge(FL[["symbol", "reaction_day", "hit_ath"]], on=["symbol", "reaction_day"])
    V3 = V3.merge(FL[["symbol", "reaction_day", "hit_ath"]], on=["symbol", "reaction_day"])

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("\n=== VARIANT 1 (morning open): ATH-hit vs not ===")
    R1 = pd.DataFrame([{"group": "ALL", **stats(V1["return_pct"])}, {"group": "HIT_ATH_within_1wk", **stats(V1[V1.hit_ath]["return_pct"])},
                       {"group": "NO_ATH", **stats(V1[~V1.hit_ath]["return_pct"])}])
    print(R1.to_string(index=False), flush=True)

    print("\n=== VARIANT 2 (closing): ATH-hit vs not ===")
    R2 = pd.DataFrame([{"group": "ALL", **stats(V2["return_pct"])}, {"group": "HIT_ATH_within_1wk", **stats(V2[V2.hit_ath]["return_pct"])},
                       {"group": "NO_ATH", **stats(V2[~V2.hit_ath]["return_pct"])}])
    print(R2.to_string(index=False), flush=True)

    print("\n=== VARIANT 3 (T+N hold sweep): ATH-hit vs not, selected N ===")
    R3 = []
    for Nn in [1, 5, 10, 15, 18, 20]:
        g = V3[V3["holding_period"] == Nn]
        R3.append({"N": Nn, "scope": "ALL", **stats(g["return_pct"])})
        R3.append({"N": Nn, "scope": "HIT_ATH", **stats(g[g.hit_ath]["return_pct"])})
        R3.append({"N": Nn, "scope": "NO_ATH", **stats(g[~g.hit_ath]["return_pct"])})
    R3 = pd.DataFrame(R3)
    print(R3.to_string(index=False), flush=True)

    print(f"\n=== the {n_hit} HIT_ATH trades themselves (variant 1, morning-open exit), sorted by return ===")
    detail = V1[V1.hit_ath][["symbol", "listing_date", "first_result_date", "session", "reaction_day", "entry_price",
                              "ath_date", "days_reaction_to_ath", "ath_price", "exit_price", "return_pct"]].sort_values("return_pct", ascending=False)
    print(detail.to_string(index=False), flush=True)

    fn = OUT / "newly_listed_first_results_ath_filter.xlsx"
    with pd.ExcelWriter(fn, engine="openpyxl") as w:
        pd.DataFrame([{"note": f"'Within 1 week' = the {ATH_WINDOW_DAYS} trading days starting from (and including) the reaction day. "
                                "A day counts as a new ATH if its daily HIGH >= the running max of every PRIOR day's high since the "
                                "stock's own first available data (these are newly listed stocks, so all-time = since listing)."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        R1.to_excel(w, sheet_name="V1_MorningOpen_by_ATH", index=False)
        R2.to_excel(w, sheet_name="V2_Closing_by_ATH", index=False)
        R3.to_excel(w, sheet_name="V3_HoldSweep_by_ATH", index=False)
        detail.to_excel(w, sheet_name="HIT_ATH_trades_detail", index=False)
        FL.to_excel(w, sheet_name="ATH_flags_all_trades", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
