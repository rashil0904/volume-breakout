# -*- coding: utf-8 -*-
"""supertrend_loser_pockets.py — entry-time pattern-mining on the Supertrend(10,5) long-only trade set
to find NET-LOSER pockets (negative TOTAL pnl in BOTH 2022-24 and 2025+). Objective = total return /
profit factor (only remove net-negative subsets, never profitable-but-below-average trades). Trend
strategy -> judge by profit factor. Features strictly knowable at the bull-flip signal (no look-ahead),
including NIFTY market-regime features. VWAP-close, cost 0.23%.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import supertrend_10_5_final as ST

DAILY = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
NIFTY1M = rb.BASE / "data" / "nifty_1min_ohlc.csv"
OUTDIR = rb.RESULTS / "supertrend_loser_pockets"
IST = "Asia/Kolkata"
COST, MCAP_MIN = 0.23, 1500.0
IS_YEARS = {2022, 2023, 2024}


def wilder_rsi(c, n=14):
    c = np.asarray(c, float); m = len(c); r = np.full(m, np.nan)
    if m < n + 1:
        return r
    dd = np.diff(c); g = np.where(dd > 0, dd, 0.0); l = np.where(dd < 0, -dd, 0.0)
    ag, al = g[:n].mean(), l[:n].mean()
    def rs(a, b):
        return 100.0 if b == 0 and a > 0 else (50.0 if b == 0 else 100 - 100 / (1 + a / b))
    r[n] = rs(ag, al)
    for i in range(n + 1, m):
        ag = (ag * (n - 1) + g[i - 1]) / n; al = (al * (n - 1) + l[i - 1]) / n; r[i] = rs(ag, al)
    return r


def nifty_regime():
    m = pd.read_csv(NIFTY1M)
    ts = pd.to_datetime(m["timestamp"], utc=True).dt.tz_convert(IST)
    m = m.assign(d=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
    m = m[(m["hm"] >= 555) & (m["hm"] < 930)]
    g = m.groupby("d")
    nd = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                       "close": g["close"].last()}).reset_index().rename(columns={"d": "date"})
    c = nd["close"].values.astype(float)
    nd["sma200"] = nd["close"].rolling(200).mean()
    nd["above200"] = (nd["close"] > nd["sma200"]).astype(int)
    nd["close_vs_200_pct"] = (nd["close"] - nd["sma200"]) / nd["sma200"] * 100
    ddir, _, _ = ST.supertrend_full(nd["high"].values.astype(float), nd["low"].values.astype(float), c, 10, 5.0)
    nd["nifty_st_dir"] = ddir
    nd["ret20"] = nd["close"].pct_change(20) * 100
    nd["ret50"] = nd["close"].pct_change(50) * 100
    return {r.date: r for r in nd.itertuples()}


def build():
    df = pd.read_parquet(DAILY); df["date"] = pd.to_datetime(df["date"])
    snap_dates, snap_dicts = ST.load_mcap()
    def mcap_of(sym, d64):
        idx = int(np.searchsorted(snap_dates, d64, "right")) - 1
        return snap_dicts[max(idx, 0)].get(sym, np.nan)
    NR = nifty_regime()
    rows = []
    for sym, s in df.groupby("symbol", sort=False):
        if len(s) < 40:
            continue
        s = s.sort_values("date")
        o = s["open"].values.astype(float); h = s["high"].values.astype(float)
        lo = s["low"].values.astype(float); c = s["close"].values.astype(float)
        v = s["volume"].values.astype(float); dt = s["date"].values
        d, st, atr = ST.supertrend_full(h, lo, c, 10, 5.0); rsi = wilder_rsi(c, 14)
        sma50 = pd.Series(c).rolling(50).mean().values; sma200 = pd.Series(c).rolling(200).mean().values
        roll252h = pd.Series(h).rolling(252, min_periods=20).max().values
        roll252l = pd.Series(lo).rolling(252, min_periods=20).min().values
        atr_pct_series = atr / c * 100
        prior20_atrpct = pd.Series(atr_pct_series).rolling(20).mean().values
        turn20 = pd.Series(c * v).rolling(20).mean().values
        n = len(s); i = 11; stock_trades = []
        while i < n:
            if d[i] == 1 and d[i - 1] == -1:
                if i + 1 >= n:
                    break
                ei = i + 1
                j = ei
                while j < n and not (d[j] == -1 and d[j - 1] == 1):
                    j += 1
                is_open = j >= n
                xi = min(j + 1, n - 1) if not is_open else n - 1
                exit_price = o[xi] if not is_open else c[xi]
                mc = mcap_of(sym, dt[i])
                if mc == mc and mc > MCAP_MIN and o[ei] > 0:
                    # prior downtrend length (consecutive -1 before i)
                    p = i - 1; cnt = 0
                    while p >= 0 and d[p] == -1:
                        cnt += 1; p -= 1
                    ret = (exit_price - o[ei]) / o[ei] * 100
                    nr = NR.get(pd.Timestamp(dt[i]).date())
                    rows_i = {"symbol": sym, "signal_date": pd.Timestamp(dt[i]).strftime("%Y-%m-%d"),
                              "year": pd.Timestamp(dt[ei]).year, "month": pd.Timestamp(dt[ei]).month,
                              "net_return_pct": round(ret - COST, 3), "holding_days": int(xi - ei), "is_open": bool(is_open),
                              # signal-time features
                              "atr_pct": round(float(atr_pct_series[i]), 3),
                              "flip_decisiveness_pct": round(float((c[i] - st[i]) / st[i] * 100), 3),
                              "prior_downtrend_len": int(cnt), "rsi14": round(float(rsi[i]), 2) if rsi[i] == rsi[i] else np.nan,
                              "close_vs_sma50_pct": round(float((c[i] - sma50[i]) / sma50[i] * 100), 3) if sma50[i] == sma50[i] else np.nan,
                              "close_vs_sma200_pct": round(float((c[i] - sma200[i]) / sma200[i] * 100), 3) if sma200[i] == sma200[i] else np.nan,
                              "above_sma200": int(c[i] > sma200[i]) if sma200[i] == sma200[i] else np.nan,
                              "dist_52wk_high_pct": round(float((c[i] - roll252h[i]) / roll252h[i] * 100), 3),
                              "dist_52wk_low_pct": round(float((c[i] - roll252l[i]) / roll252l[i] * 100), 3),
                              "mcap_cr": round(float(mc), 0), "prior20_atr_pct": round(float(prior20_atrpct[i]), 3) if prior20_atrpct[i] == prior20_atrpct[i] else np.nan,
                              "log_turnover20": round(float(np.log10(turn20[i])), 2) if (turn20[i] == turn20[i] and turn20[i] > 0) else np.nan,
                              "nifty_above_200dma": int(nr.above200) if nr is not None and nr.above200 == nr.above200 else np.nan,
                              "nifty_st_dir": int(nr.nifty_st_dir) if nr is not None else np.nan,
                              "nifty_close_vs_200_pct": round(float(nr.close_vs_200_pct), 2) if nr is not None and nr.close_vs_200_pct == nr.close_vs_200_pct else np.nan,
                              "nifty_ret20": round(float(nr.ret20), 2) if nr is not None and nr.ret20 == nr.ret20 else np.nan,
                              "nifty_ret50": round(float(nr.ret50), 2) if nr is not None and nr.ret50 == nr.ret50 else np.nan}
                    stock_trades.append(rows_i)
                i = j if j < n else n
            else:
                i += 1
        # stock's own prior Supertrend win-rate (cumulative, entry-time-valid)
        wins = 0
        for k, tr in enumerate(stock_trades):
            tr["stock_prior_ntrades"] = k
            tr["stock_prior_winrate"] = round(wins / k * 100, 1) if k > 0 else np.nan
            wins += 1 if tr["net_return_pct"] > 0 else 0
        rows.extend(stock_trades)
    T = pd.DataFrame(rows)
    T["sample"] = np.where(T["year"].isin(IS_YEARS), "IS", "OOS")
    return T


NUM = ["atr_pct", "flip_decisiveness_pct", "prior_downtrend_len", "rsi14", "close_vs_sma50_pct",
       "close_vs_sma200_pct", "dist_52wk_high_pct", "dist_52wk_low_pct", "mcap_cr", "prior20_atr_pct",
       "log_turnover20", "nifty_close_vs_200_pct", "nifty_ret20", "nifty_ret50", "stock_prior_winrate", "month"]
CAT = ["above_sma200", "nifty_above_200dma", "nifty_st_dir"]


def seg(df):
    a = df["net_return_pct"].sum()
    return round(a, 1), round(df[df.sample_ == "IS"]["net_return_pct"].sum(), 1), round(df[df.sample_ == "OOS"]["net_return_pct"].sum(), 1)


def pf(df):
    r = df["net_return_pct"]; gl = -r[r < 0].sum()
    return round(r[r > 0].sum() / gl, 3) if gl > 0 else np.inf


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = build(); T["sample_"] = T["sample"]
    tot, tis, toos = seg(T)
    print(f"trades {len(T):,} (open {int(T['is_open'].sum())}) | total_ret {tot} (IS {tis}/OOS {toos}) | PF {pf(T)} "
          f"| IS PF {pf(T[T.sample_=='IS'])} OOS PF {pf(T[T.sample_=='OOS'])}")

    # STEP2 univariate loser pockets
    pk = []
    for f in NUM:
        s = T[[f, "net_return_pct", "sample_"]].dropna(subset=[f])
        try:
            s["b"] = pd.qcut(s[f], 8, labels=False, duplicates="drop")
        except Exception:
            continue
        for b, g in s.groupby("b"):
            a, i2, o2 = g["net_return_pct"].sum(), g[g.sample_ == "IS"]["net_return_pct"].sum(), g[g.sample_ == "OOS"]["net_return_pct"].sum()
            pk.append({"feature": f, "bucket": int(b), "lo": round(g[f].min(), 3), "hi": round(g[f].max(), 3), "n": len(g),
                       "win_rate": round((g["net_return_pct"] > 0).mean() * 100, 1), "avg_ret": round(g["net_return_pct"].mean(), 3),
                       "total_pnl": round(a, 1), "pnl_IS": round(i2, 1), "pnl_OOS": round(o2, 1), "net_neg_both": bool(i2 < 0 and o2 < 0)})
    for f in CAT:
        for lv, g in T.dropna(subset=[f]).groupby(f):
            a, i2, o2 = g["net_return_pct"].sum(), g[g.sample_ == "IS"]["net_return_pct"].sum(), g[g.sample_ == "OOS"]["net_return_pct"].sum()
            pk.append({"feature": f, "bucket": lv, "lo": lv, "hi": lv, "n": len(g),
                       "win_rate": round((g["net_return_pct"] > 0).mean() * 100, 1), "avg_ret": round(g["net_return_pct"].mean(), 3),
                       "total_pnl": round(a, 1), "pnl_IS": round(i2, 1), "pnl_OOS": round(o2, 1), "net_neg_both": bool(i2 < 0 and o2 < 0)})
    P = pd.DataFrame(pk).sort_values("total_pnl").reset_index(drop=True)
    losers = P[P["net_neg_both"]]

    # STEP3 threshold sweeps (remove side)
    sw = []
    def sweep(feat, side, cuts):
        for cut in cuts:
            rem = T[T[feat] > cut] if side == "hi" else T[T[feat] < cut]
            kept = T.drop(rem.index)
            if len(rem) == 0 or rem[feat].isna().all():
                continue
            ra, ri, ro = seg(rem)
            sw.append({"filter": f"remove {feat} {'>' if side == 'hi' else '<'} {cut}", "n_removed": len(rem),
                       "removed_pnl_total": ra, "removed_pnl_IS": ri, "removed_pnl_OOS": ro,
                       "removed_neg_both": bool(ri < 0 and ro < 0), "new_total_return": round(kept["net_return_pct"].sum(), 1),
                       "delta_vs_base": round(kept["net_return_pct"].sum() - tot, 1), "new_PF": pf(kept), "new_n": len(kept)})
    sweep("nifty_close_vs_200_pct", "lo", [-2, 0, 2, 5])          # index below/near 200dma (regime)
    sweep("nifty_ret20", "lo", [-5, -2, 0, 2])
    sweep("atr_pct", "hi", [4, 5, 6, 8, 10])                      # choppy high-vol flips
    sweep("flip_decisiveness_pct", "hi", [8, 10, 12])            # marginal? actually decisiveness high=far
    sweep("close_vs_sma200_pct", "lo", [-20, -10, 0])           # counter long-term trend
    sweep("prior_downtrend_len", "hi", [30, 50, 80])
    sweep("dist_52wk_low_pct", "lo", [10, 20, 30])
    sweep("stock_prior_winrate", "lo", [20, 30, 40])
    SW = pd.DataFrame(sw)
    # regime categorical filters
    for feat, val, lbl in [("nifty_above_200dma", 0, "remove nifty_below_200dma"),
                           ("nifty_st_dir", -1, "remove nifty_ST_bearish"), ("above_sma200", 0, "remove stock_below_sma200")]:
        rem = T[T[feat] == val]; kept = T[T[feat] != val]
        ra, ri, ro = seg(rem)
        SW = pd.concat([SW, pd.DataFrame([{"filter": lbl, "n_removed": len(rem), "removed_pnl_total": ra, "removed_pnl_IS": ri,
                                           "removed_pnl_OOS": ro, "removed_neg_both": bool(ri < 0 and ro < 0),
                                           "new_total_return": round(kept["net_return_pct"].sum(), 1),
                                           "delta_vs_base": round(kept["net_return_pct"].sum() - tot, 1), "new_PF": pf(kept), "new_n": len(kept)}])], ignore_index=True)
    SW_ok = SW[SW["removed_neg_both"] & (SW["delta_vs_base"] > 0)].sort_values("delta_vs_base", ascending=False)

    # STEP4 tree
    fc = ["atr_pct", "flip_decisiveness_pct", "prior_downtrend_len", "rsi14", "close_vs_sma50_pct", "close_vs_sma200_pct",
          "dist_52wk_high_pct", "dist_52wk_low_pct", "mcap_cr", "prior20_atr_pct", "log_turnover20",
          "nifty_close_vs_200_pct", "nifty_ret20", "nifty_ret50", "month"]
    Tt = T.dropna(subset=fc).copy(); Tt["win"] = (Tt["net_return_pct"] > 0).astype(int)
    Xis = Tt[Tt.sample_ == "IS"][fc].fillna(Tt[fc].median()); yis = Tt[Tt.sample_ == "IS"]["win"]
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=0).fit(Xis, yis)
    Tt["leaf"] = clf.apply(Tt[fc].fillna(Tt[fc].median()))
    tt = clf.tree_; paths = {}
    def rec(node, cond):
        if tt.feature[node] != _tree.TREE_UNDEFINED:
            f = fc[tt.feature[node]]; th = tt.threshold[node]
            rec(tt.children_left[node], cond + [f"{f}<={th:.3g}"]); rec(tt.children_right[node], cond + [f"{f}>{th:.3g}"])
        else:
            paths[node] = " AND ".join(cond)
    rec(0, [])
    lf = []
    for leaf, g in Tt.groupby("leaf"):
        a, i2, o2 = g["net_return_pct"].sum(), g[g.sample_ == "IS"]["net_return_pct"].sum(), g[g.sample_ == "OOS"]["net_return_pct"].sum()
        lf.append({"rule": paths.get(leaf, str(leaf)), "n": len(g), "win_rate": round((g["net_return_pct"] > 0).mean() * 100, 1),
                   "avg_ret": round(g["net_return_pct"].mean(), 3), "total_pnl": round(a, 1), "pnl_IS": round(i2, 1),
                   "pnl_OOS": round(o2, 1), "net_neg_both": bool(i2 < 0 and o2 < 0)})
    LEAF = pd.DataFrame(lf).sort_values("total_pnl").reset_index(drop=True)

    # rejected (cut net-positive)
    rej = []
    for lbl, mask in [("keep nifty_above_200dma", T["nifty_above_200dma"] == 1),
                      ("keep nifty_ST_bullish", T["nifty_st_dir"] == 1),
                      ("keep stock above_sma200", T["above_sma200"] == 1),
                      ("keep rsi14>=70", T["rsi14"] >= 70)]:
        rem = T[~mask.fillna(False)]; ra, ri, ro = seg(rem)
        rej.append({"filter": lbl, "n_removed": len(rem), "removed_pnl_total": ra, "removed_pnl_IS": ri, "removed_pnl_OOS": ro,
                    "removed_net_positive": bool(ra > 0), "new_total_return": round(T[mask.fillna(False)]["net_return_pct"].sum(), 1),
                    "delta_vs_base": round(T[mask.fillna(False)]["net_return_pct"].sum() - tot, 1)})
    REJ = pd.DataFrame(rej)

    with pd.ExcelWriter(OUTDIR / "supertrend_loser_pockets.xlsx", engine="openpyxl") as w:
        losers.to_excel(w, sheet_name="loser_pockets_neg_both", index=False)
        P.to_excel(w, sheet_name="all_buckets", index=False)
        SW.to_excel(w, sheet_name="threshold_sweeps", index=False)
        LEAF.to_excel(w, sheet_name="tree_leaves", index=False)
        REJ.to_excel(w, sheet_name="rejected_cuts", index=False)

    pd.set_option("display.width", 260)
    print(f"\nBASELINE: n={len(T)} total_ret={tot} (IS {tis}/OOS {toos}) PF={pf(T)}")
    print("\n=== STEP2 NET-NEGATIVE loser-pockets (neg pnl in BOTH IS & OOS) ===")
    print(losers[["feature", "bucket", "lo", "hi", "n", "win_rate", "avg_ret", "total_pnl", "pnl_IS", "pnl_OOS"]].to_string(index=False) if len(losers) else "  NONE")
    print("\n=== STEP3 ACCEPTED filters (removed pnl NEG both & total return UP) ===")
    cols = ["filter", "n_removed", "removed_pnl_total", "removed_pnl_IS", "removed_pnl_OOS", "new_total_return", "delta_vs_base", "new_PF"]
    print(SW_ok[cols].to_string(index=False) if len(SW_ok) else "  NONE cleared")
    print("\n  --- full regime/vol sweep ---")
    print(SW[cols].to_string(index=False))
    print("\n=== STEP4 TREE leaf loser-pockets (neg both) ===")
    tl = LEAF[LEAF["net_neg_both"]]
    print(tl.to_string(index=False) if len(tl) else "  NONE")
    print("\n=== REJECTED cuts (remove NET-POSITIVE -> total return down) ===")
    print(REJ.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
