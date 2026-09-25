# -*- coding: utf-8 -*-
"""pattern_mining.py — mine ENTRY-TIME patterns that separate winners from losers in the baseline
trade set (combined long+short). Strict anti-look-ahead: every feature is knowable at/before the 15:15
entry. Mandatory in-sample (2022-2024) vs out-of-sample (2025-2026) validation.

Flags: (a) features strictly entry-time (diagnostic 'today_*'/prev-day only; NO 'next_day_*'); (b) OOS
time-split mandatory; (c) min-sample flagged; (d) interpretable patterns preferred; (e) combined pnl
primary outcome, leg-split secondary.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

BASE_XLSX = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
DIAG = rb.RESULTS / "diagnostic_table.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
NIFTY1M = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "pattern_mining"
IST = "Asia/Kolkata"
IS_YEARS = {2022, 2023, 2024}


def wilder_rsi(c, n=14):
    c = np.asarray(c, float); m = len(c); r = np.full(m, np.nan)
    if m < n + 1:
        return r
    d = np.diff(c); g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    ag, al = g[:n].mean(), l[:n].mean()
    def rs(ag, al):
        return 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 else 100 - 100 / (1 + ag / al))
    r[n] = rs(ag, al)
    for i in range(n + 1, m):
        ag = (ag * (n - 1) + g[i - 1]) / n; al = (al * (n - 1) + l[i - 1]) / n; r[i] = rs(ag, al)
    return r


def build_features():
    T = pd.read_excel(BASE_XLSX, sheet_name="all_trades")
    T["entry_date"] = pd.to_datetime(T["entry_date"]); ed = T["entry_date"].dt.date
    T["ed"] = ed
    # outcome (combined, primary)
    T["ret"] = T["gross_ret"]                      # combined pnl / capital *100
    T["win"] = (T["gross_pnl"] > 0).astype(int)
    T["dow"] = T["entry_date"].dt.dayofweek
    T["month"] = T["entry_date"].dt.month
    T["yr"] = T["entry_date"].dt.year
    T["uc_after"] = (T["uc_after_first_fill"] == 1.0).astype(int)
    T["hit_uc"] = (T["hit_uc_ever"] == True).astype(int)
    T["cat_a_full"] = ((T["category"] == "A") & (T["n_legs"] == 2)).astype(int)

    dg = pd.read_csv(DIAG, parse_dates=["date"])
    dg["ed"] = dg["date"].dt.date
    keep = ["symbol", "ed", "market_cap_value", "prev_day_vwap_close", "return_pct_vs_prev_close",
            "volume_ratio", "cum_volume_to_3pm_today", "avg_nday_fullday_volume", "fade_at_entry_pct",
            "today_open_915", "today_cumhigh_15", "today_cumlow_15"]
    T = T.merge(dg[keep], on=["symbol", "ed"], how="left")
    T["mcap_cr"] = T["market_cap_value"]
    T["ret_vs_prevclose"] = T["return_pct_vs_prev_close"]
    T["vol_ratio"] = T["volume_ratio"]
    T["fade"] = T["fade_at_entry_pct"]
    T["gap_pct"] = (T["today_open_915"] - T["prev_day_vwap_close"]) / T["prev_day_vwap_close"] * 100
    T["intraday_range_pct"] = (T["today_cumhigh_15"] - T["today_cumlow_15"]) / T["today_cumlow_15"] * 100
    T["shares_pct_to_3pm"] = T["shares"] / T["cum_volume_to_3pm_today"] * 100

    # daily-derived (through prev day): sma36, rsi14, atr14
    d = pd.read_parquet(DAILY, columns=["symbol", "date", "high", "low", "close"]).sort_values(["symbol", "date"])
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d["sma36"] = d.groupby("symbol")["close"].transform(lambda s: s.rolling(36).mean().shift(1))
    d["rsi14"] = d.groupby("symbol")["close"].transform(lambda s: pd.Series(wilder_rsi(s.values), index=s.index).shift(1))
    d["atr14"] = d.groupby("symbol").apply(
        lambda g: ((g["high"] - g["low"]) / g["close"] * 100).rolling(14).mean().shift(1)).reset_index(level=0, drop=True)
    dm = d.set_index(["symbol", "date"])[["sma36", "rsi14", "atr14"]]
    for col in ["sma36", "rsi14", "atr14"]:
        T[col] = [dm[col].get((s, e), np.nan) for s, e in zip(T["symbol"], T["ed"])]
    T["above_sma36"] = (T["avg_entry"] > T["sma36"]).astype(int)
    T["dist_sma36_pct"] = (T["avg_entry"] - T["sma36"]) / T["sma36"] * 100

    # nifty features
    nm = pd.read_csv(NIFTY1M)
    nts = pd.to_datetime(nm["timestamp"], utc=True).dt.tz_convert(IST)
    nm = nm.assign(d=nts.dt.date, hm=nts.dt.hour * 60 + nts.dt.minute)
    nm = nm[(nm["hm"] >= 555) & (nm["hm"] < 930)]
    nd = nm.groupby("d").agg(o=("open", "first"), c=("close", "last")).reset_index()
    nd["c1459"] = nm[nm["hm"] <= 899].groupby("d")["close"].last().values   # ~15:00 (pre-entry)
    nd["prevret"] = nd["c"].pct_change() * 100
    nd["entryret"] = (nd["c1459"] - nd["o"]) / nd["o"] * 100
    npre = dict(zip(nd["d"], nd["prevret"].shift(-0)))       # prevret aligned to that day; we want PRIOR day's ret
    # nifty prev-day return = return realized on the previous trading day (known at entry)
    nd["prevday_known"] = nd["prevret"].shift(1)
    pmap = dict(zip(nd["d"], nd["prevday_known"])); emap = dict(zip(nd["d"], nd["entryret"]))
    T["nifty_prevday_ret"] = [pmap.get(e, np.nan) for e in T["ed"]]
    T["nifty_entryday_ret"] = [emap.get(e, np.nan) for e in T["ed"]]

    T["sample"] = np.where(T["yr"].isin(IS_YEARS), "IS", "OOS")
    return T


NUMERIC = ["vol_ratio", "ret_vs_prevclose", "shares_pct_to_3pm", "intraday_range_pct", "gap_pct",
           "fade", "mcap_cr", "dist_sma36_pct", "rsi14", "atr14", "nifty_prevday_ret",
           "nifty_entryday_ret", "capital_deployed", "shares", "avg_entry", "month", "dow"]
CATVARS = ["category", "above_sma36", "hit_uc", "uc_after", "cat_a_full", "dow"]
BASE_POOL = 500_000


def univariate(T):
    rows, buckets = [], {}
    for f in NUMERIC:
        s = T[[f, "win", "ret"]].dropna()
        if len(s) < 100:
            continue
        try:
            s["b"] = pd.qcut(s[f], 5, labels=False, duplicates="drop")
        except Exception:
            continue
        bt = s.groupby("b").agg(n=("win", "size"), lo=(f, "min"), hi=(f, "max"),
                                win_rate=("win", "mean"), avg_ret=("ret", "mean"))
        bt["win_rate"] = (bt["win_rate"] * 100).round(1); bt["avg_ret"] = bt["avg_ret"].round(3)
        buckets[f] = bt.reset_index()
        sep = bt["win_rate"].max() - bt["win_rate"].min()
        corr = s[f].corr(s["ret"])
        rows.append({"feature": f, "type": "numeric", "n": len(s), "winrate_spread_pp": round(sep, 1),
                     "corr_with_ret": round(corr, 3),
                     "lo_bucket_winrate": bt["win_rate"].iloc[0], "hi_bucket_winrate": bt["win_rate"].iloc[-1]})
    for f in CATVARS:
        g = T.groupby(f).agg(n=("win", "size"), win_rate=("win", "mean"), avg_ret=("ret", "mean"))
        g["win_rate"] = (g["win_rate"] * 100).round(1); g["avg_ret"] = g["avg_ret"].round(3)
        buckets[f] = g.reset_index()
        rows.append({"feature": f, "type": "categorical", "n": len(T),
                     "winrate_spread_pp": round(g["win_rate"].max() - g["win_rate"].min(), 1),
                     "corr_with_ret": np.nan, "lo_bucket_winrate": g["win_rate"].min(), "hi_bucket_winrate": g["win_rate"].max()})
    R = pd.DataFrame(rows).sort_values("winrate_spread_pp", ascending=False).reset_index(drop=True)
    return R, buckets


def apply_filter(T, mask, label):
    """headline of the KEPT set for a boolean keep-mask, split IS/OOS."""
    out = {}
    for seg, sub in [("ALL", T), ("IS", T[T["sample"] == "IS"]), ("OOS", T[T["sample"] == "OOS"])]:
        kept = sub[mask.loc[sub.index]]
        rem = sub[~mask.loc[sub.index]]
        out[seg] = {"n_kept": len(kept), "n_removed": len(rem),
                    "win_kept": round(kept["win"].mean() * 100, 1) if len(kept) else np.nan,
                    "win_base": round(sub["win"].mean() * 100, 1),
                    "avg_ret_kept": round(kept["ret"].mean(), 3) if len(kept) else np.nan,
                    "avg_ret_removed": round(rem["ret"].mean(), 3) if len(rem) else np.nan,
                    "tot_ret_base_pct": round(sub["gross_pnl"].sum() / BASE_POOL * 100, 1),
                    "tot_ret_kept_pct": round(kept["gross_pnl"].sum() / BASE_POOL * 100, 1)}
    return {"filter": label, **{f"{seg}_{k}": v for seg in ["ALL", "IS", "OOS"] for k, v in out[seg].items()}}


def tree_rules(clf, names):
    t = clf.tree_; out = []
    def rec(node, depth, cond):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            f = names[t.feature[node]]; thr = t.threshold[node]
            rec(t.children_left[node], depth + 1, cond + [f"{f} <= {thr:.3g}"])
            rec(t.children_right[node], depth + 1, cond + [f"{f} > {thr:.3g}"])
        else:
            val = t.value[node][0]; n = int(t.n_node_samples[node]); wr = val[1] * 100   # value is normalized
            out.append({"rule": " AND ".join(cond) if cond else "(root)", "n": n, "win_rate_pct": round(wr, 1)})
    rec(0, 0, [])
    return pd.DataFrame(out).sort_values("win_rate_pct", ascending=False)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = build_features()
    print(f"trades {len(T):,} | IS {int((T['sample']=='IS').sum()):,} | OOS {int((T['sample']=='OOS').sum()):,} | "
          f"overall win {T['win'].mean()*100:.1f}% avg_ret {T['ret'].mean():.3f}%")
    T.to_parquet(OUTDIR / "trade_features.parquet", index=False)

    # STEP 2 — univariate
    RANK, buckets = univariate(T)
    print("\n--- UNIVARIATE RANKING (winner/loser separation) ---")
    print(RANK.to_string(index=False))

    # STEP 3 — candidate filters (IS/OOS)
    filters = []
    for X in [8, 10, 12, 15]:
        filters.append(apply_filter(T, T["vol_ratio"] > X, f"keep vol_ratio > {X}"))
    for X in [3, 5, 8]:
        filters.append(apply_filter(T, T["shares_pct_to_3pm"] <= X, f"exclude shares_pct_to_3pm > {X}"))
    filters.append(apply_filter(T, T["above_sma36"] == 1, "keep above_sma36 (uptrend)"))
    filters.append(apply_filter(T, T["hit_uc"] == 0, "exclude hit_uc_ever"))
    filters.append(apply_filter(T, T["category"] == "C", "keep category C only"))
    filters.append(apply_filter(T, T["category"] != "A", "exclude category A"))
    for X in [10, 15, 20]:
        filters.append(apply_filter(T, T["intraday_range_pct"] <= X, f"exclude intraday_range > {X}%"))
    for lo, hi in [(5, 9), (5, 12)]:
        filters.append(apply_filter(T, (T["ret_vs_prevclose"] >= lo) & (T["ret_vs_prevclose"] <= hi),
                                    f"keep ret_vs_prevclose {lo}-{hi}%"))
    for X in [40, 50, 60]:
        filters.append(apply_filter(T, T["rsi14"] <= X, f"exclude rsi14 > {X}"))
    FILT = pd.DataFrame(filters)

    # STEP 4 — shallow tree + RF importance (fit on IS only)
    feat_cols = ["vol_ratio", "ret_vs_prevclose", "shares_pct_to_3pm", "intraday_range_pct", "gap_pct",
                 "fade", "mcap_cr", "dist_sma36_pct", "rsi14", "atr14", "nifty_prevday_ret",
                 "nifty_entryday_ret", "above_sma36", "hit_uc", "cat_a_full", "dow", "month"]
    for c in ["cat_B", "cat_C"]:
        T[c] = (T["category"] == c[-1]).astype(int)
    feat_cols += ["cat_B", "cat_C"]
    X_is = T[T["sample"] == "IS"][feat_cols].fillna(T[feat_cols].median())
    y_is = T[T["sample"] == "IS"]["win"]
    X_oos = T[T["sample"] == "OOS"][feat_cols].fillna(T[feat_cols].median())
    y_oos = T[T["sample"] == "OOS"]["win"]

    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=80, random_state=0).fit(X_is, y_is)
    RULES = tree_rules(clf, feat_cols)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=40,
                                random_state=0, n_jobs=-1).fit(X_is, y_is)
    IMP = pd.DataFrame({"feature": feat_cols, "rf_importance": rf.feature_importances_.round(4)}) \
        .sort_values("rf_importance", ascending=False).reset_index(drop=True)
    tree_is_auc = clf.score(X_is, y_is); tree_oos_auc = clf.score(X_oos, y_oos)

    with pd.ExcelWriter(OUTDIR / "pattern_mining.xlsx", engine="openpyxl") as w:
        RANK.to_excel(w, sheet_name="univariate_ranking", index=False)
        FILT.to_excel(w, sheet_name="candidate_filters", index=False)
        RULES.to_excel(w, sheet_name="tree_rules", index=False)
        IMP.to_excel(w, sheet_name="rf_importance", index=False)
        for f, bt in buckets.items():
            bt.to_excel(w, sheet_name=("buk_" + f)[:31], index=False)

    print("\n--- CANDIDATE FILTERS (kept-set win% & total-return, IS vs OOS) ---")
    cols = ["filter", "ALL_n_kept", "ALL_win_kept", "ALL_avg_ret_removed", "ALL_tot_ret_kept_pct",
            "IS_win_kept", "IS_avg_ret_removed", "OOS_win_kept", "OOS_avg_ret_removed", "OOS_tot_ret_kept_pct"]
    print(FILT[cols].to_string(index=False))
    print(f"\n  baseline: ALL win {T['win'].mean()*100:.1f}% | tot_ret {T['gross_pnl'].sum()/BASE_POOL*100:.1f}% | "
          f"OOS win {T[T['sample']=='OOS']['win'].mean()*100:.1f}% tot {T[T['sample']=='OOS']['gross_pnl'].sum()/BASE_POOL*100:.1f}%")
    print("\n--- SHALLOW DECISION TREE (depth 3, fit on IS) — leaves by win-rate ---")
    print(RULES.to_string(index=False))
    print(f"  tree accuracy: IS {tree_is_auc:.3f}  OOS {tree_oos_auc:.3f}  (base rate {y_is.mean():.3f}/{y_oos.mean():.3f})")
    print("\n--- RANDOM-FOREST FEATURE IMPORTANCE (ranking only) ---")
    print(IMP.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
