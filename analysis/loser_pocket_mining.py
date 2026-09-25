# -*- coding: utf-8 -*-
"""loser_pocket_mining.py — find filters that remove NET-LOSER subsets ONLY (negative total pnl),
to MAXIMIZE TOTAL RETURN (Σ pnl / 5L). A filter is accepted only if the REMOVED trades have negative
aggregate pnl in BOTH in-sample (2022-24) and out-of-sample (2025-26). Cutting merely-below-average
but still-profitable trades is REJECTED. Winner-concentration (size-tilt, no removal) reported separately.
Entry-time features only. Reuses results/pattern_mining/trade_features.parquet.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

FEAT = rb.RESULTS / "pattern_mining" / "trade_features.parquet"
LIQ = rb.RESULTS / "liquidity_proxies" / "per_trade_liquidity.csv"
OUTDIR = rb.RESULTS / "loser_pockets"
BP = 500_000


def totret(df):
    return round(df["gross_pnl"].sum() / BP * 100, 2)


def seg_pnl(df):
    return (round(df["gross_pnl"].sum(), 0),
            round(df[df["sample"] == "IS"]["gross_pnl"].sum(), 0),
            round(df[df["sample"] == "OOS"]["gross_pnl"].sum(), 0))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = pd.read_parquet(FEAT)
    lq = pd.read_csv(LIQ)[["symbol", "entry_date", "shares_pct_of_volume"]]
    lq["entry_date"] = pd.to_datetime(lq["entry_date"]).dt.date
    T["ekey"] = T["entry_date"].dt.date
    T = T.merge(lq.rename(columns={"entry_date": "ekey"}), on=["symbol", "ekey"], how="left")

    base_all, base_is, base_oos = seg_pnl(T)
    print(f"trades {len(T):,} | baseline total pnl Rs{base_all:,.0f} (IS {base_is:,.0f} / OOS {base_oos:,.0f}) "
          f"| total_return {totret(T)}%  (IS {totret(T[T['sample']=='IS'])} / OOS {totret(T[T['sample']=='OOS'])})")

    NUM = ["dist_sma36_pct", "intraday_range_pct", "atr14", "rsi14", "gap_pct", "vol_ratio",
           "ret_vs_prevclose", "fade", "shares_pct_to_3pm", "shares_pct_of_volume", "mcap_cr",
           "nifty_prevday_ret", "nifty_entryday_ret"]
    CAT = ["category", "above_sma36", "hit_uc", "uc_after", "cat_a_full", "dow", "month"]

    # ── STEP 2: bucket every feature; flag deciles with NEGATIVE total pnl (IS & OOS) ──
    pockets = []
    for f in NUM:
        s = T[[f, "gross_pnl", "sample", "ret", "win"]].dropna(subset=[f])
        try:
            s["b"] = pd.qcut(s[f], 10, labels=False, duplicates="drop")
        except Exception:
            continue
        for b, g in s.groupby("b"):
            tot, tis, toos = seg_pnl(g)
            pockets.append({"feature": f, "bucket": int(b), "lo": round(g[f].min(), 3), "hi": round(g[f].max(), 3),
                            "n": len(g), "win_rate": round(g["win"].mean() * 100, 1), "avg_ret": round(g["ret"].mean(), 3),
                            "total_pnl": tot, "pnl_IS": tis, "pnl_OOS": toos,
                            "net_negative_both": bool(tis < 0 and toos < 0)})
    for f in CAT:
        for lv, g in T.groupby(f):
            tot, tis, toos = seg_pnl(g)
            pockets.append({"feature": f, "bucket": lv, "lo": lv, "hi": lv, "n": len(g),
                            "win_rate": round(g["win"].mean() * 100, 1), "avg_ret": round(g["ret"].mean(), 3),
                            "total_pnl": tot, "pnl_IS": tis, "pnl_OOS": toos,
                            "net_negative_both": bool(tis < 0 and toos < 0)})
    P = pd.DataFrame(pockets).sort_values("total_pnl").reset_index(drop=True)
    losers = P[P["net_negative_both"]].copy()

    # ── STEP 3: tail-threshold sweeps — removed-subset pnl must be NEGATIVE in IS & OOS ──
    sweeps = []
    def sweep(feature, side, cuts):
        for c in cuts:
            rem = T[T[feature] > c] if side == "hi" else T[T[feature] < c]
            kept = T.drop(rem.index)
            if len(rem) == 0:
                continue
            rtot, ris, roos = seg_pnl(rem)
            sweeps.append({"filter": f"remove {feature} {'>' if side=='hi' else '<'} {c}", "n_removed": len(rem),
                           "removed_pnl_total": rtot, "removed_pnl_IS": ris, "removed_pnl_OOS": roos,
                           "removed_net_neg_both": bool(ris < 0 and roos < 0),
                           "new_total_return_pct": totret(kept), "delta_vs_base_pct": round(totret(kept) - totret(T), 2),
                           "new_n_trades": len(kept),
                           "removed_avg_shares_pct_vol": round(rem["shares_pct_of_volume"].mean(), 2)})
    sweep("dist_sma36_pct", "hi", [20, 25, 30, 35, 40, 50])
    sweep("intraday_range_pct", "hi", [8, 10, 12, 15, 20, 25])
    sweep("atr14", "hi", [4, 5, 6, 7, 8])
    sweep("rsi14", "hi", [65, 70, 72, 75])
    sweep("rsi14", "lo", [30, 35, 40])
    sweep("gap_pct", "hi", [3, 5, 8, 10])
    sweep("gap_pct", "lo", [-2, -1, 0])
    sweep("vol_ratio", "hi", [20, 30, 40, 50])
    sweep("ret_vs_prevclose", "hi", [9, 12, 15, 18])
    sweep("shares_pct_of_volume", "hi", [3, 5, 8, 10])
    SW = pd.DataFrame(sweeps)
    SW_ok = SW[SW["removed_net_neg_both"] & (SW["delta_vs_base_pct"] > 0)].sort_values("delta_vs_base_pct", ascending=False)

    # ── STEP 4: shallow tree -> leaf pockets with NEGATIVE total pnl in both ──
    fc = ["dist_sma36_pct", "intraday_range_pct", "atr14", "rsi14", "gap_pct", "vol_ratio", "ret_vs_prevclose",
          "fade", "shares_pct_to_3pm", "mcap_cr", "nifty_prevday_ret", "nifty_entryday_ret", "above_sma36",
          "hit_uc", "cat_a_full", "dow", "month"]
    Tt = T.dropna(subset=fc).copy()
    Xis = Tt[Tt["sample"] == "IS"][fc]; yis = Tt[Tt["sample"] == "IS"]["win"]
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=60, random_state=0).fit(Xis, yis)
    Tt["leaf"] = clf.apply(Tt[fc])
    from sklearn.tree import _tree
    tt = clf.tree_

    def path(leaf):
        # reconstruct decision path to a leaf id
        paths = {}
        def rec(node, cond):
            if tt.feature[node] != _tree.TREE_UNDEFINED:
                f = fc[tt.feature[node]]; th = tt.threshold[node]
                rec(tt.children_left[node], cond + [f"{f}<= {th:.3g}"])
                rec(tt.children_right[node], cond + [f"{f}> {th:.3g}"])
            else:
                paths[node] = " AND ".join(cond)
        rec(0, [])
        return paths
    pmap = path(0)
    leafrows = []
    for lf, g in Tt.groupby("leaf"):
        tot, tis, toos = seg_pnl(g)
        leafrows.append({"rule": pmap.get(lf, str(lf)), "n": len(g), "win_rate": round(g["win"].mean() * 100, 1),
                         "avg_ret": round(g["ret"].mean(), 3), "total_pnl": tot, "pnl_IS": tis, "pnl_OOS": toos,
                         "net_negative_both": bool(tis < 0 and toos < 0)})
    LEAF = pd.DataFrame(leafrows).sort_values("total_pnl").reset_index(drop=True)

    # ── STEP 6: winner-concentration (high-expectancy pockets; size-tilt, no removal) ──
    winners = P[(P["avg_ret"] > T["ret"].mean() * 1.4) & (P["pnl_IS"] > 0) & (P["pnl_OOS"] > 0) & (P["n"] >= 120)] \
        .sort_values("avg_ret", ascending=False).head(12)

    # ── REJECTED: filters that cut NET-POSITIVE pnl (win-rate up but total return down) ──
    rej = []
    for label, mask in [("keep above_sma36 (removes below-SMA)", T["above_sma36"] == 1),
                        ("keep category C (removes A+B)", T["category"] == "C"),
                        ("keep rsi14 53-67 (removes outside band)", T["rsi14"].between(53, 67)),
                        ("keep mcap<2300 (removes larger caps)", T["mcap_cr"] < 2300),
                        ("exclude hit_uc", T["hit_uc"] == 0)]:
        rem = T[~mask]; rtot, ris, roos = seg_pnl(rem)
        rej.append({"filter": label, "n_removed": len(rem), "removed_pnl_total": rtot, "removed_pnl_IS": ris,
                    "removed_pnl_OOS": roos, "removed_net_positive": bool(rtot > 0),
                    "new_total_return_pct": totret(T[mask]), "delta_vs_base_pct": round(totret(T[mask]) - totret(T), 2)})
    REJ = pd.DataFrame(rej)

    with pd.ExcelWriter(OUTDIR / "loser_pockets.xlsx", engine="openpyxl") as w:
        losers.to_excel(w, sheet_name="loser_pockets_neg_both", index=False)
        P.to_excel(w, sheet_name="all_buckets", index=False)
        SW.to_excel(w, sheet_name="threshold_sweeps", index=False)
        LEAF.to_excel(w, sheet_name="tree_leaf_pockets", index=False)
        winners.to_excel(w, sheet_name="winner_concentration", index=False)
        REJ.to_excel(w, sheet_name="rejected_cuts_winners", index=False)

    pd.set_option("display.width", 260)
    print("\n=== STEP 2: NET-NEGATIVE loser-pockets (neg total pnl in BOTH IS & OOS) ===")
    print(losers[["feature", "bucket", "lo", "hi", "n", "win_rate", "avg_ret", "total_pnl", "pnl_IS", "pnl_OOS"]].to_string(index=False)
          if len(losers) else "  NONE — no single-feature bucket loses money in both periods.")
    print("\n=== STEP 3: ACCEPTED tail filters (removed pnl NEGATIVE both & total return UP) ===")
    cols = ["filter", "n_removed", "removed_pnl_total", "removed_pnl_IS", "removed_pnl_OOS",
            "new_total_return_pct", "delta_vs_base_pct", "removed_avg_shares_pct_vol"]
    print(SW_ok[cols].to_string(index=False) if len(SW_ok) else "  NONE cleared (neg-both AND total-return-up).")
    print("\n  --- full sweep (see where removed pnl flips sign) ---")
    print(SW[cols].to_string(index=False))
    print("\n=== STEP 4: TREE leaf loser-pockets (neg total pnl both) ===")
    tl = LEAF[LEAF["net_negative_both"]]
    print(tl.to_string(index=False) if len(tl) else "  NONE — no depth-4 leaf loses in both periods.")
    print("\n=== STEP 6: WINNER-CONCENTRATION pockets (size-tilt candidates, keep all trades) ===")
    print(winners[["feature", "bucket", "lo", "hi", "n", "avg_ret", "total_pnl", "pnl_IS", "pnl_OOS"]].to_string(index=False))
    print("\n=== REJECTED cuts (remove NET-POSITIVE pnl -> total return DOWN) ===")
    print(REJ.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
