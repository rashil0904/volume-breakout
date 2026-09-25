# -*- coding: utf-8 -*-
"""short_52wh_suppress.py — suppress the double-down SHORT on trades whose stock hit a 52-week high
on the entry day (entry-day intraday high == trailing-252-day max of daily high, incl entry day).
Long leg unchanged; trade not removed. Compare vs baseline + standalone table of the suppressed shorts.
(a) 52WH = trailing 252 daily-high max incl entry day; hit = entry_high >= that; (c) <252d history uses
shorter window, flagged; (d) combined gross/net_A/net_B; (e) fixed 5L base, n_trades identical.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

BASE_XLSX = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
OUTDIR = rb.RESULTS / "short_52wh_suppress"
BP = 500_000
RF = 0.075
R023, R038 = 0.0023, 0.0038
IS_YEARS = {2022, 2023, 2024}


def metrics(T, label):
    dd = T.groupby("exit_date").agg(g=("gross_pnl", "sum"), a=("netA_pnl", "sum"),
                                    b=("netB_pnl", "sum"), c=("capital_deployed", "sum")).reset_index()
    rfd = RF / 252 * 100
    def shp(x):
        return round((x.mean() - rfd) / x.std(ddof=1) * np.sqrt(252), 3) if len(x) > 1 else 0.0
    row = {"config": label, "n_trades": len(T)}
    for s, tag in [("gross", "gross"), ("netA", "net_A"), ("netB", "net_B")]:
        row[f"tot_ret_{tag}_pct"] = round(T[f"{s}_pnl"].sum() / BP * 100, 2)
        row[f"tot_pnl_{tag}_inr"] = round(T[f"{s}_pnl"].sum(), 0)
        row[f"win_{tag}_pct"] = round((T[f"{s}_pnl"] > 0).mean() * 100, 2)
        row[f"avg_ret_{tag}_pct"] = round(T[f"{s}_ret"].mean(), 4)
        row[f"med_ret_{tag}_pct"] = round(T[f"{s}_ret"].median(), 4)
    row["total_long_pnl_inr"] = round(T["long_pnl"].sum(), 0)
    row["total_short_pnl_inr"] = round(T["short_pnl"].sum(), 0)
    row["sharpe_gross"] = shp(dd["g"] / dd["c"] * 100)
    row["sharpe_net_A"] = shp(dd["a"] / dd["c"] * 100)
    row["sharpe_net_B"] = shp(dd["b"] / dd["c"] * 100)
    return row


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = pd.read_excel(BASE_XLSX, sheet_name="all_trades")
    T["entry_date"] = pd.to_datetime(T["entry_date"]); T["exit_date"] = pd.to_datetime(T["exit_date"])
    T["ed"] = T["entry_date"].dt.date

    # ── 52-week-high flag (trailing 252 daily-high max incl entry day) ──
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "high"]).sort_values(["symbol", "date"])
    d["date"] = pd.to_datetime(d["date"]).dt.date
    g = d.groupby("symbol")
    d["roll252_high"] = g["high"].transform(lambda s: s.rolling(252, min_periods=1).max())
    d["ndays"] = g.cumcount() + 1
    d["hit_52wh"] = d["high"] >= d["roll252_high"] - 1e-9          # entry-day high IS the 252-day max
    d["short_window"] = d["ndays"] < 252                          # <252 prior daily highs -> flagged
    dm = d.set_index(["symbol", "date"])
    T["entry_day_high"] = [float(dm["high"].get((s, e), np.nan)) for s, e in zip(T["symbol"], T["ed"])]
    T["wk52_high"] = [float(dm["roll252_high"].get((s, e), np.nan)) for s, e in zip(T["symbol"], T["ed"])]
    T["hit_52wh"] = [bool(dm["hit_52wh"].get((s, e), False)) for s, e in zip(T["symbol"], T["ed"])]
    T["short_window_flag"] = [bool(dm["short_window"].get((s, e), False)) for s, e in zip(T["symbol"], T["ed"])]

    has_short = T["short_exit_type"].isin(["short_target_5pct", "short_cover_1439"])
    # FULL-HISTORY only: exclude the 192 short-window (<252d) '52WH' trades from suppression (unreliable) —
    # they keep their shorts and run as normal baseline trades.
    suppress = T["hit_52wh"] & has_short & (~T["short_window_flag"])
    n_supp = int(suppress.sum())
    n_excluded_shortwin = int((T["hit_52wh"] & T["short_window_flag"]).sum())
    print(f"trades {len(T):,} | hit 52WH on entry day {int(T['hit_52wh'].sum()):,} "
          f"(full-252 history {int((T['hit_52wh']&~T['short_window_flag']).sum())}, short-window {n_excluded_shortwin}) | "
          f"shorts suppressed (full-history 52WH with a short) {n_supp} | "
          f"short-window '52WH' KEPT (not suppressed) {n_excluded_shortwin}")

    # ── CONFIG 2: suppress short on 52WH trades ──
    C2 = T.copy()
    m = suppress.values
    C2.loc[m, "short_pnl"] = 0.0
    C2.loc[m, "short_cost"] = 0.0
    C2["gross_pnl"] = C2["long_pnl"] + C2["short_pnl"]                       # combined
    C2["netA_pnl"] = C2["gross_pnl"] - R023 * C2["capital_deployed"] - C2["short_cost"]
    C2["netB_pnl"] = C2["gross_pnl"] - R038 * C2["capital_deployed"] - C2["short_cost"]
    for s in ["gross", "netA", "netB"]:
        C2[f"{s}_ret"] = C2[f"{s}_pnl"] / C2["capital_deployed"] * 100

    # ── TABLE 1 ──
    tbl1 = pd.DataFrame([metrics(T, "1_baseline_short_all"), metrics(C2, "2_52WH_short_suppressed")])
    delta = {"config": "delta (2-1)"}
    for c in tbl1.columns:
        if c != "config":
            delta[c] = round(tbl1.iloc[1][c] - tbl1.iloc[0][c], 4)
    tbl1 = pd.concat([tbl1, pd.DataFrame([delta])], ignore_index=True)
    tbl1["n_shorts_suppressed"] = n_supp

    # ── TABLE 2: the suppressed shorts (would-be baseline short) ──
    S = T[suppress].copy()
    swin = S["short_pnl"] > 0
    t2 = {"n_suppressed": len(S),
          "wouldbe_short_pnl_gross_inr": round(S["short_pnl"].sum(), 0),
          "wouldbe_short_pnl_net_inr": round(S["short_pnl"].sum() - S["short_cost"].sum(), 0),
          "short_win_rate_pct": round(swin.mean() * 100, 2), "n_short_win": int(swin.sum()), "n_short_loss": int((~swin).sum()),
          "avg_short_pnl_inr": round(S["short_pnl"].mean(), 0),
          "long_pnl_of_these_inr": round(S["long_pnl"].sum(), 0),
          "avg_long_pnl_inr": round(S["long_pnl"].mean(), 0),
          "short_ret_on_5L_pct": round(S["short_pnl"].sum() / BP * 100, 3)}
    TBL2 = pd.DataFrame([t2])
    det = S[["symbol", "entry_date", "entry_day_high", "wk52_high", "short_window_flag", "category",
             "long_pnl", "short_pnl", "short_cost"]].copy()
    det["entry_date"] = det["entry_date"].dt.strftime("%Y-%m-%d")
    det = det.rename(columns={"short_pnl": "short_pnl_baseline"})
    det["combined_baseline"] = (det["long_pnl"] + det["short_pnl_baseline"]).round(1)
    det["combined_suppressed"] = det["long_pnl"].round(1)
    det = det.sort_values("short_pnl_baseline").round(2)

    # ── OOS check ──
    S["yr"] = S["entry_date"].dt.year
    def agg(df):
        return {"n": len(df), "short_pnl_gross": round(df["short_pnl"].sum(), 0),
                "short_pnl_net": round(df["short_pnl"].sum() - df["short_cost"].sum(), 0),
                "short_win_rate": round((df["short_pnl"] > 0).mean() * 100, 1) if len(df) else np.nan}
    OOS = pd.DataFrame([{"period": "ALL", **agg(S)},
                        {"period": "IS 2022-24", **agg(S[S["yr"].isin(IS_YEARS)])},
                        {"period": "OOS 2025+", **agg(S[~S["yr"].isin(IS_YEARS)])}])

    with pd.ExcelWriter(OUTDIR / "short_52wh_suppress.xlsx", engine="openpyxl") as w:
        tbl1.to_excel(w, sheet_name="TABLE1_baseline_vs_suppress", index=False)
        TBL2.to_excel(w, sheet_name="TABLE2_suppressed_standalone", index=False)
        det.to_excel(w, sheet_name="suppressed_per_trade", index=False)
        OOS.to_excel(w, sheet_name="OOS_check", index=False)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 105 + "\nTABLE 1 — baseline (short all) vs 52WH-short-suppressed\n" + "=" * 105)
    show = ["config", "n_trades", "tot_ret_gross_pct", "tot_ret_net_A_pct", "tot_ret_net_B_pct",
            "win_net_A_pct", "avg_ret_net_A_pct", "total_long_pnl_inr", "total_short_pnl_inr",
            "sharpe_gross", "sharpe_net_A", "sharpe_net_B"]
    print(tbl1[show].to_string(index=False))
    print(f"\n  n_shorts_suppressed: {n_supp}")
    print("\n" + "=" * 105 + "\nTABLE 2 — the suppressed 52WH shorts (would-be baseline short leg)\n" + "=" * 105)
    for k, v in t2.items():
        print(f"  {k:28s}: {v}")
    print("\n--- OOS CHECK (were 52WH-stock shorts net-negative in both periods?) ---")
    print(OOS.to_string(index=False))
    print("\n--- PER-TRADE (worst would-be shorts first; first 20) ---")
    print(det[["symbol", "entry_date", "entry_day_high", "wk52_high", "category", "long_pnl",
               "short_pnl_baseline", "combined_baseline", "combined_suppressed"]].head(20).to_string(index=False))

    tot = S["short_pnl"].sum()
    print("\n" + "=" * 105 + "\nVERDICT\n" + "=" * 105)
    print(f"  suppressed shorts: {len(S)} | aggregate would-be short pnl Rs{tot:,.0f} "
          f"({'NET LOSER -> suppression ADDS it back (helps)' if tot < 0 else 'NET WINNER -> suppression COSTS you'})")
    print(f"  net_A total return: {tbl1.iloc[0]['tot_ret_net_A_pct']:.2f}% -> {tbl1.iloc[1]['tot_ret_net_A_pct']:.2f}% "
          f"(delta {tbl1.iloc[2]['tot_ret_net_A_pct']:+.2f} pp)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
