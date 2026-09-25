# -*- coding: utf-8 -*-
"""
baseline_and_cross_final.py
===========================
Final report for the BASELINE composite strategy + backtest of the two CROSS variants
(19% / 19.5%) + three-way comparison. ONLY Category A's first-leg entry differs across the
three; Category B, C, capital sequencing, exits, and the double-down short are IDENTICAL.

SHARED (all 3): mcap ₹1,500-5,000 Cr, LB36/VM6/+5% (1-min conventions). Cat-A trigger window
2:30-3:00pm (fills to 3:21); Cat B = UC approach in 3:00-3:21 not already Cat A, fill at
uc*0.999; Cat C = rest at 3:21 open. ₹5L pool / ₹1L cap, sequencing Phase1 A/B intraday ->
Phase2 A 3:21 pending -> Phase3 C remainder. Long exit: split 09:25/11:59 + 17% target.
Short: cover 14:39 OR earlier at short_open*0.95 (5% target). Costs gross/net_A(0.23+0.10)/
net_B(0.38+0.10).

CATEGORY A (the only difference):
  BASELINE  : in UC during 2:30-3pm, reopens; 1st 50% at +19% on post-reopen pullback,
              2nd 50% at +17% (else 3:21 if not re-locked, else half).
  CROSS_19  : 1st 50% at +19% on the upward CROSS (2:30-3pm), UC or not; 2nd 50% via +17%
              post-UC-reopen, else 3:21 fallback (never-UC), else half.
  CROSS_19_5: same, first leg at +19.5%.
"""
import sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import uc_staggered_dd_report as R          # reuse pmetrics / period_table / streaks / constants

IST = rb.IST
OUTDIR = rb.RESULTS / "baseline_and_cross_final"
A_START, HM_1500, HM_1521 = 870, 900, 921    # 2:30 / 3:00 / 3:21
T1, T2 = 565, 719                            # 09:25 / 11:59 long-exit legs
LONG_TGT, SHORT_TGT = 1.17, 0.95             # 17% long target ; 5% short target
COVER_HM = 879                               # 14:39 short cover
UC_M, L19_M, L17_M, L195_M = 1.1995, 1.19, 1.17, 1.195
BASE_POOL, BASE_ALLOC = 500_000, 100_000
R023, R038, SR = 0.0023, 0.0038, 0.0010
RF = 0.075
CONFIGS = ["baseline", "cross_19", "cross_19_5"]


def fhm(hm, mask):
    return int(hm[mask].min()) if mask.any() else None


def lbl(h):
    return f"{int(h)//60:02d}:{int(h)%60:02d}" if h == h and h is not None else ""


# ── entry-day classification (per config) ───────────────────────────────────
def classify(config, pc, hm, hi, lo, op):
    uc, l19, l17, l195 = pc * UC_M, pc * L19_M, pc * L17_M, pc * L195_M
    o = dict(zip(hm, op)); px = o.get(HM_1521, np.nan)
    lo_of = dict(zip(hm, lo)); lo_1521 = lo_of.get(HM_1521, np.nan)
    locked_321 = lo_1521 == lo_1521 and lo_1521 >= uc     # 15:21 candle traded ENTIRELY at/above circuit
    uc_in_230_300 = fhm(hm, (hm >= A_START) & (hm < HM_1500) & (hi >= uc))
    uc_in_300_321 = fhm(hm, (hm >= HM_1500) & (hm < HM_1521) & (hi >= uc))
    first_uc = fhm(hm, hi >= uc)
    meta = {"hit_uc_ever": first_uc is not None}

    def catB_or_C():
        inB = (hm >= HM_1500) & (hm < HM_1521)                       # 3:00-3:21 window
        touch_B = fhm(hm, inB & (hi >= uc))                          # any UC touch in 3:00-3:21
        opened_in_B = bool((lo[inB] < uc).any()) if inB.any() else False   # traded below UC in window (opened)
        locked_before_3 = first_uc is not None and first_uc < HM_1500
        # in circuit from BEFORE 3pm AND never opened through 3:21 -> un-fillable -> NO ENTRY
        if locked_321 and locked_before_3 and not opened_in_B:
            return "C", False, [], {**meta, "no_entry": "locked_from_before3_never_opened"}
        # UC touch in 3:00-3:21 (fresh lock OR re-approach after opening) -> Cat B at uc*0.999
        if touch_B is not None:
            return "B", True, [(1.0, uc * 0.999, touch_B, 1)], {**meta, "entry_kind": "B_uc0.999"}
        if px == px and px > 0 and not locked_321:                   # Cat C if fillable at 3:21
            return "C", True, [(1.0, px, HM_1521, 3)], {**meta, "entry_kind": "C_1521"}
        return "C", False, [], {**meta, "no_entry": "locked_or_no_1521"}

    if config == "baseline":
        if uc_in_230_300 is not None:                            # in UC during 2:30-3pm
            w = (hm > uc_in_230_300) & (hm < HM_1521)            # post-reopen pullback window
            wl, wh = lo[w], hm[w]
            i19 = np.where(wl <= l19)[0]
            if len(i19):
                legs = [(0.5, l19, int(wh[i19[0]]), 1)]
                i17 = np.where(wl <= l17)[0]
                if len(i17):
                    legs.append((0.5, l17, int(wh[i17[0]]), 1)); full = True; k = "17%"
                elif px == px and not locked_321:                # not re-locked at 3:21
                    legs.append((0.5, px, HM_1521, 2)); full = True; k = "3:21"
                else:
                    full = False; k = "half"
                return "A", True, legs, {**meta, "cat_a_full": full, "leg2": k, "uc_after_first_fill": True}
            return catB_or_C()                                   # never pulled to +19% -> not Cat A
        return catB_or_C()

    # ── cross configs ──
    L = l195 if "19_5" in config else l19
    first_L = fhm(hm, hi >= L)                                     # FIRST time all day the stock reaches L
    cross = first_L if (first_L is not None and A_START <= first_L < HM_1500) else None  # fresh cross in 2:30-3pm
    if cross is not None:
        legs = [(0.5, L, cross, 1)]
        uc_after = first_uc is not None and first_uc >= cross
        full = False; k = "half"
        if uc_after:
            retr = np.where((hm > first_uc) & (lo <= l17))[0]   # +17% only via post-UC-reopen
            if len(retr):
                legs.append((0.5, l17, int(hm[retr[0]]), 1)); full = True; k = "17%_postUC"
            elif px == px and not locked_321:
                legs.append((0.5, px, HM_1521, 2)); full = True; k = "3:21"
        else:                                                    # never-UC cross -> 3:21 fallback only
            if px == px and px > 0 and not locked_321:
                legs.append((0.5, px, HM_1521, 2)); full = True; k = "3:21_noUC"
        return "A", True, legs, {**meta, "cat_a_full": full, "leg2": k, "uc_after_first_fill": bool(uc_after)}
    return catB_or_C()


def build_cache(diag_path=None):
    diag = pd.read_csv(diag_path or (rb.RESULTS / "diagnostic_table.csv"),
                       usecols=["symbol", "date", "passes_all_three", "prev_day_vwap_close"],
                       parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True]
    print(f"qualifying: {len(Q):,}\nScanning …", flush=True)
    cache = []; t0 = time.time()
    for si, sym in enumerate(sorted(Q["symbol"].unique()), 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
        bd = {d: g for d, g in raw.groupby("date")}
        dts = sorted(bd)
        for _, r in Q[Q["symbol"] == sym].iterrows():
            ed = r["date"]; pc = float(r["prev_day_vwap_close"])
            if ed not in bd or not (pc == pc and pc > 0):
                continue
            j = dts.index(ed)
            nd = dts[j + 1] if j < len(dts) - 1 else None
            if nd is None:
                continue
            g = bd[ed]
            eg = {"hm": g["hm"].values, "hi": g["high"].values.astype(float),
                  "lo": g["low"].values.astype(float), "op": g["open"].values.astype(float)}
            ng = bd[nd]
            no = dict(zip(ng["hm"].values, ng["open"].values.astype(float)))
            nhm = ng["hm"].values
            nhi = ng["high"].values.astype(float); nlo = ng["low"].values.astype(float)
            hm_t = (nhm >= 555) & (nhm <= T2); lm_t = (nhm >= T1) & (nhm <= COVER_HM)
            cache.append({"symbol": sym, "entry_date": ed, "exit_date": nd, "pc": pc, "eg": eg,
                          "o565": no.get(T1, np.nan), "o719": no.get(T2, np.nan), "o879": no.get(COVER_HM, np.nan),
                          "nhm": nhm[hm_t], "nhi": nhi[hm_t], "lhm": nhm[lm_t], "nlo": nlo[lm_t]})
        if si % 300 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s), {len(cache):,} stock-days", flush=True)
    return cache


def run_config(config, cache, exclude=None, full_deploy=False):
    recs = []
    for c in cache:
        cat, entered, legs, meta = classify(config, c["pc"], c["eg"]["hm"], c["eg"]["hi"], c["eg"]["lo"], c["eg"]["op"])
        recs.append({**{k: c[k] for k in ("symbol", "entry_date", "exit_date", "o565", "o719", "o879",
                                          "nhm", "nhi", "lhm", "nlo")},
                     "category": cat, "entered": entered, "legs": legs, "meta": meta})
    # sequencing
    ent = [r for r in recs if r["entered"] and r["legs"]]
    if exclude:                                        # reallocation: drop these (sym,date) BEFORE sequencing
        ent = [r for r in ent if (r["symbol"], r["entry_date"]) not in exclude]
    for r in ent:
        r["_a"] = [0.0] * len(r["legs"])
    by_day = defaultdict(list)
    for r in ent:
        by_day[r["entry_date"]].append(r)
    diag_days = {}
    for d, rs in by_day.items():
        pool = BASE_POOL; X = 0.0; p1, p2, cc = [], [], []
        for r in rs:
            for li, (frac, price, thm, ph) in enumerate(r["legs"]):
                (cc if ph == 3 else p2 if ph == 2 else p1).append(
                    (r, li, price) if ph == 3 else (r, li, 0.5 * BASE_ALLOC) if ph == 2
                    else (thm, r, li, BASE_ALLOC if frac >= 1.0 else 0.5 * BASE_ALLOC))
        for thm, r, li, intd in sorted(p1, key=lambda x: (x[0] if x[0] is not None else 99999)):
            give = min(intd, pool); r["_a"][li] = give; pool -= give; X += give
        p2_sum = 0.0
        for r, li, intd in p2:
            give = min(intd, pool); r["_a"][li] = give; pool -= give; X += give; p2_sum += give
        # full_deploy: Cat C splits the FULL remainder uncapped (no 1L ceiling) -> deploys all idle capital
        nC = len(cc)
        if nC:
            per_c = (max(0.0, pool) / nC) if full_deploy else min(BASE_ALLOC, max(0.0, pool) / nC)
        else:
            per_c = 0.0
        for r, li, price in cc:
            r["_a"][li] = per_c
        diag_days[d] = {"X": X, "nC": nC, "per_c": per_c, "p2": p2_sum}
    # sizing + exit + short
    rows = []
    for r in ent:
        shares = cap = 0.0; lbls = []; pxs = []
        for li, (frac, price, thm, ph) in enumerate(r["legs"]):
            s = np.floor(r["_a"][li] / price) if r["_a"][li] > 0 else 0.0
            if s > 0:
                shares += s; cap += s * price
                lbls.append(f"{r['meta'].get('entry_kind', r['category'])}:{round(price,2)}" if r["category"] != "A"
                            else f"{['19%','17%/3:21'][min(li,1)]}({int(s)})")
                pxs.append(round(price, 2))
        if shares <= 0:
            continue
        avg = cap / shares
        tgt = avg * LONG_TGT
        hit = r["nhi"] >= tgt
        th = int(r["nhm"][hit].min()) if hit.any() else 10 ** 9
        o565, o719, o879 = r["o565"], r["o719"], r["o879"]
        if th <= T1:
            xp, xt, xhm = tgt, "target_pre_0925", th
        elif o565 == o565 and o565 > avg:
            xp, xt, xhm = o565, "positive_0925", T1
        elif th <= T2:
            xp, xt, xhm = tgt, "target_0925_1159", th
        elif o719 == o719:
            xp, xt, xhm = o719, "exit_1159", T2
        else:
            continue
        long_pnl = shares * (xp - avg)
        # short: open at xp @ xhm ; cover 14:39 OR 5% target (low<=xp*0.95 in (xhm,879])
        stgt = xp * SHORT_TGT
        sw = (r["lhm"] > xhm) & (r["nlo"] <= stgt)
        if sw.any():
            cover, sxt = stgt, "short_target_5pct"
        elif o879 == o879:
            cover, sxt = o879, "short_cover_1439"
        else:
            cover, sxt = np.nan, "no_short"
        has_short = cover == cover
        short_pnl = shares * (xp - cover) if has_short else 0.0
        snotl = shares * xp if has_short else 0.0
        comb = long_pnl + short_pnl
        rows.append({"symbol": r["symbol"], "entry_date": r["entry_date"], "exit_date": r["exit_date"],
                     "category": r["category"], "cat_a_full": r["meta"].get("cat_a_full"),
                     "legs_filled": "+".join(lbls), "leg_prices": "+".join(map(str, pxs)),
                     "n_legs": len(lbls), "shares": int(shares), "capital_deployed": cap, "avg_entry": avg,
                     "long_exit_type": xt, "exit_time": lbl(xhm), "exit_price": xp,
                     "short_exit_type": sxt, "cover_price": cover,
                     "long_pnl": long_pnl, "short_pnl": short_pnl, "combined_pnl": comb,
                     "long_cost_023": R023 * cap, "long_cost_038": R038 * cap, "short_cost": SR * snotl,
                     "gross_pnl": comb, "netA_pnl": comb - R023 * cap - SR * snotl,
                     "netB_pnl": comb - R038 * cap - SR * snotl,
                     "uc_after_first_fill": r["meta"].get("uc_after_first_fill"),
                     "hit_uc_ever": r["meta"].get("hit_uc_ever")})
    T = pd.DataFrame(rows)
    for s in ["gross", "netA", "netB"]:
        T[f"{s}_ret"] = T[f"{s}_pnl"] / T["capital_deployed"] * 100
    ts = pd.to_datetime(T["entry_date"])
    T["year"] = ts.dt.year; T["month"] = ts.dt.strftime("%Y-%m")
    T["quarter"] = ts.dt.year.astype(str) + "Q" + ts.dt.quarter.astype(str)
    T["half_year"] = ts.dt.year.astype(str) + "H" + np.where(ts.dt.month <= 6, "1", "2")
    return T.sort_values(["entry_date", "symbol"]).reset_index(drop=True), diag_days


# ── full report (baseline) ──────────────────────────────────────────────────
def full_report(T, diag_days, path):
    monthly = R.period_table(T, "month")
    quarterly = R.period_table(T, "quarter", compounded=True)
    halfy = R.period_table(T, "half_year")
    yearly = R.period_table(T, "year", order_fn=lambda x: int(x[0]), compounded=True)
    daily = pd.DataFrame([{"date": k, **R.pmetrics(g),
                           "gross_is_winning_day": bool(g["gross_pnl"].sum() > 0)}
                          for k, g in T.groupby("exit_date")]).sort_values("date")
    de = T.groupby("exit_date").agg(gross=("gross_pnl", "sum"), netA=("netA_pnl", "sum"),
                                    netB=("netB_pnl", "sum")).reset_index().sort_values("exit_date")
    exd = de["exit_date"].tolist()
    strk = {s: R.streaks(exd, de[s].values) for s in ["gross", "netA", "netB"]}
    dd = T.groupby("exit_date").agg(g=("gross_pnl", "sum"), a=("netA_pnl", "sum"),
                                    b=("netB_pnl", "sum"), c=("capital_deployed", "sum")).reset_index()
    rfd = RF / 252 * 100
    def shp(x): return round((x.mean() - rfd) / x.std(ddof=1) * np.sqrt(252), 3) if len(x) > 1 else 0.0

    n = len(T)
    def S(m, g, a, b): return {"metric": m, "gross": g, "net_A": a, "net_B": b}
    def col(m, fn): return S(m, fn("gross"), fn("netA"), fn("netB"))
    rows = [
        S("n_trades", n, n, n),
        col("total_return_fixedbase_pct", lambda s: round(T[f"{s}_pnl"].sum() / BASE_POOL * 100, 2)),
        col("total_pnl_inr", lambda s: round(T[f"{s}_pnl"].sum(), 0)),
        col("win_rate_pct", lambda s: round((T[f"{s}_pnl"] > 0).mean() * 100, 2)),
        col("n_winning_trades", lambda s: int((T[f"{s}_pnl"] > 0).sum())),
        col("n_losing_trades", lambda s: int((T[f"{s}_pnl"] <= 0).sum())),
        col("avg_return_per_trade_pct", lambda s: round(T[f"{s}_ret"].mean(), 4)),
        col("median_return_per_trade_pct", lambda s: round(T[f"{s}_ret"].median(), 4)),
        col("avg_return_winning_pct", lambda s: round(T.loc[T[f"{s}_pnl"] > 0, f"{s}_ret"].mean(), 4)),
        col("avg_return_losing_pct", lambda s: round(T.loc[T[f"{s}_pnl"] <= 0, f"{s}_ret"].mean(), 4)),
        col("avg_capital_deployed_inr", lambda s: round(T["capital_deployed"].mean(), 0)),
        S("sharpe_ratio", shp(dd["g"] / dd["c"] * 100), shp(dd["a"] / dd["c"] * 100), shp(dd["b"] / dd["c"] * 100)),
    ]
    def per(t): return f"{t[0]} -> {t[1]}" if t[0] else "—"
    rows += [
        S("n_winning_days", int((de["gross"] > 0).sum()), int((de["netA"] > 0).sum()), int((de["netB"] > 0).sum())),
        S("n_losing_days", int((de["gross"] <= 0).sum()), int((de["netA"] <= 0).sum()), int((de["netB"] <= 0).sum())),
        S("max_consec_win_days", strk["gross"][0], strk["netA"][0], strk["netB"][0]),
        S("max_consec_win_days_pnl", round(strk["gross"][1][2], 0), round(strk["netA"][1][2], 0), round(strk["netB"][1][2], 0)),
        S("max_consec_win_days_period", per(strk["gross"][1]), per(strk["netA"][1]), per(strk["netB"][1])),
        S("max_consec_lose_days", strk["gross"][2], strk["netA"][2], strk["netB"][2]),
        S("max_consec_lose_days_pnl", round(strk["gross"][3][2], 0), round(strk["netA"][3][2], 0), round(strk["netB"][3][2], 0)),
        S("max_consec_lose_days_period", per(strk["gross"][3]), per(strk["netA"][3]), per(strk["netB"][3])),
    ]
    SER = {"gross": "gross", "netA": "net_A", "netB": "net_B"}
    for lb2, d2 in [("month", monthly), ("quarter", quarterly), ("half_year", halfy), ("year", yearly)]:
        rows.append(col(f"avg_return_per_{lb2}_pct", lambda s, d=d2: round(d[f"{SER[s]}_total_return_fixedbase_pct"].mean(), 3)))
    # category + leg + allocation diagnostics
    def SC(m, v): return S(m, v, v, v)
    rows.append(SC("--- CATEGORY / LEG / ALLOCATION ---", ""))
    A = T[T["category"] == "A"]
    rows += [SC("n_cat_A", int((T["category"] == "A").sum())),
             SC("n_cat_A_full", int((A["n_legs"] == 2).sum())), SC("n_cat_A_half", int((A["n_legs"] == 1).sum())),
             SC("n_cat_B", int((T["category"] == "B").sum())), SC("n_cat_C", int((T["category"] == "C").sum()))]
    for c in ["A", "B", "C"]:
        g = T[T["category"] == c]
        rows.append(col(f"category_{c}_return_fixedbase_pct", lambda s, gg=g: round(gg[f"{s}_pnl"].sum() / BASE_POOL * 100, 3)))
    rows.append(SC("long_leg_total_pnl_inr", round(T["long_pnl"].sum(), 0)))
    rows.append(SC("long_leg_return_fixedbase_pct", round(T["long_pnl"].sum() / BASE_POOL * 100, 3)))
    rows.append(SC("short_leg_total_pnl_inr", round(T["short_pnl"].sum(), 0)))
    rows.append(SC("short_leg_return_fixedbase_pct", round(T["short_pnl"].sum() / BASE_POOL * 100, 3)))
    for c in ["A", "B", "C"]:                                    # per-category long vs short
        g = T[T["category"] == c]
        rows.append(SC(f"category_{c}_long_pnl_inr", round(g["long_pnl"].sum(), 0)))
        rows.append(SC(f"category_{c}_short_pnl_inr", round(g["short_pnl"].sum(), 0)))
    # ── allocation diagnostics ──
    xab = pd.Series([v["X"] for v in diag_days.values()])
    perc = pd.Series([v["per_c"] for v in diag_days.values() if v["nC"] > 0])
    p2c = int(sum(1 for v in diag_days.values() if v.get("p2", 0) > 0 and v["nC"] > 0 and v["per_c"] < BASE_ALLOC - 1))
    p2ser = pd.Series([v.get("p2", 0) for v in diag_days.values()])
    rows += [SC("avg_daily_AB_committed_inr", round(xab.mean(), 0)), SC("max_daily_AB_committed_inr", round(xab.max(), 0)),
             SC("n_days_C_squeezed_lt_1L", int((perc < BASE_ALLOC - 1).sum())),
             SC("n_days_C_got_zero", int((perc <= 0).sum())),
             SC("avg_C_allocation_inr", round(perc.mean(), 0) if len(perc) else 0),
             SC("n_days_3:21_Atopup_consumed_C", p2c),
             SC("avg_3:21_Atopup_committed_inr", round(p2ser[p2ser > 0].mean(), 0) if (p2ser > 0).any() else 0),
             SC("avg_cap_A", round(A["capital_deployed"].mean(), 0) if len(A) else 0),
             SC("avg_cap_B", round(T.loc[T["category"] == "B", "capital_deployed"].mean(), 0) if (T["category"] == "B").any() else 0),
             SC("avg_cap_C", round(T.loc[T["category"] == "C", "capital_deployed"].mean(), 0))]
    # ── period extremes (highest/lowest day/month/quarter/year; ₹ and %) ──
    def add_ext(df, name, lblcol):
        mx = df.loc[df["net_A_total_pnl_inr"].idxmax()]; mn = df.loc[df["net_A_total_pnl_inr"].idxmin()]
        rows.append(SC(f"--- {name.upper()} EXTREMES ---", ""))
        rows.append(S(f"highest_{name}_pnl_inr", round(mx["gross_total_pnl_inr"], 0), round(mx["net_A_total_pnl_inr"], 0), round(mx["net_B_total_pnl_inr"], 0)))
        rows.append(S(f"highest_{name}_return_pct", round(mx["gross_total_return_fixedbase_pct"], 3), round(mx["net_A_total_return_fixedbase_pct"], 3), round(mx["net_B_total_return_fixedbase_pct"], 3)))
        rows.append(SC(f"highest_{name}", str(mx[lblcol])))
        rows.append(S(f"lowest_{name}_pnl_inr", round(mn["gross_total_pnl_inr"], 0), round(mn["net_A_total_pnl_inr"], 0), round(mn["net_B_total_pnl_inr"], 0)))
        rows.append(S(f"lowest_{name}_return_pct", round(mn["gross_total_return_fixedbase_pct"], 3), round(mn["net_A_total_return_fixedbase_pct"], 3), round(mn["net_B_total_return_fixedbase_pct"], 3)))
        rows.append(SC(f"lowest_{name}", str(mn[lblcol])))
    add_ext(daily, "day", "date"); add_ext(monthly, "month", "month")
    add_ext(quarterly, "quarter", "quarter"); add_ext(yearly, "year", "year")
    summary = pd.DataFrame(rows)

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        T.to_excel(w, sheet_name="all_trades", index=False)
        daily.to_excel(w, sheet_name="daily_performance", index=False)
        monthly.to_excel(w, sheet_name="monthly_performance", index=False)
        quarterly.to_excel(w, sheet_name="quarterly_performance", index=False)
        halfy.to_excel(w, sheet_name="half_yearly_performance", index=False)
        yearly.to_excel(w, sheet_name="yearly_performance", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    return summary, yearly


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = build_cache()
    Ts = {}; diags = {}
    for cfg in CONFIGS:
        Ts[cfg], diags[cfg] = run_config(cfg, cache)
        print(f"  {cfg}: {len(Ts[cfg]):,} trades")

    summary, yearly = full_report(Ts["baseline"], diags["baseline"], OUTDIR / "baseline_final_performance.xlsx")

    # three-way comparison
    def line(T, name):
        d = {"config": name, "n_trades": len(T)}
        for s, tag in [("gross", "gross"), ("netA", "net_A"), ("netB", "net_B")]:
            p = T[f"{s}_pnl"]; r = T[f"{s}_ret"]
            d[f"{tag}_total_return_fixedbase_pct"] = round(p.sum() / BASE_POOL * 100, 2)
            d[f"{tag}_total_pnl_inr"] = round(p.sum(), 0)
            d[f"{tag}_win_rate_pct"] = round((p > 0).mean() * 100, 2)
            d[f"{tag}_avg_return_per_trade_pct"] = round(r.mean(), 4)
            d[f"{tag}_median_return_per_trade_pct"] = round(r.median(), 4)
        return d
    comp = pd.DataFrame([line(Ts[c], c) for c in CONFIGS])

    # Category A isolated
    aiso = []
    for c in CONFIGS:
        A = Ts[c][Ts[c]["category"] == "A"]
        aiso.append({"config": c, "n_A": len(A),
                     "n_full": int((A["n_legs"] == 2).sum()), "n_half": int((A["n_legs"] == 1).sum()),
                     "netA_avg_return_pct": round(A["netA_ret"].mean(), 4), "netA_median_pct": round(A["netA_ret"].median(), 4),
                     "catA_total_pnl_netA": round(A["netA_pnl"].sum(), 0),
                     "catA_return_fixedbase_pct": round(A["netA_pnl"].sum() / BASE_POOL * 100, 2),
                     "first_leg_DID_hit_UC": int((A["uc_after_first_fill"] == True).sum()),
                     "first_leg_NEVER_hit_UC": int((A["uc_after_first_fill"] == False).sum())})
    catA_iso = pd.DataFrame(aiso)

    # B+C identity check
    bc = []
    for c in CONFIGS:
        BC = Ts[c][Ts[c]["category"].isin(["B", "C"])]
        bc.append({"config": c, "n_B": int((Ts[c]["category"] == "B").sum()),
                   "n_C": int((Ts[c]["category"] == "C").sum()), "n_BC": len(BC),
                   "BC_total_pnl_netA": round(BC["netA_pnl"].sum(), 0),
                   "BC_return_fixedbase_pct": round(BC["netA_pnl"].sum() / BASE_POOL * 100, 2)})
    bc_check = pd.DataFrame(bc)

    with pd.ExcelWriter(OUTDIR / "three_way_comparison.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="three_way", index=False)
        catA_iso.to_excel(w, sheet_name="categoryA_isolated", index=False)
        bc_check.to_excel(w, sheet_name="BC_identity_check", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 110 + "\nTHREE-WAY COMPARISON (combined long+short; net_A leads)\n" + "=" * 110)
    print(comp[["config", "n_trades", "gross_total_return_fixedbase_pct", "net_A_total_return_fixedbase_pct",
                "net_B_total_return_fixedbase_pct", "net_A_win_rate_pct", "net_A_avg_return_per_trade_pct",
                "net_A_median_return_per_trade_pct"]].to_string(index=False))
    print("\n--- CATEGORY A ISOLATED ---"); print(catA_iso.to_string(index=False))
    print("\n--- B+C IDENTITY CHECK (should match if Cat A membership were identical) ---")
    print(bc_check.to_string(index=False))
    base = comp[comp["config"] == "baseline"]["net_A_total_return_fixedbase_pct"].iloc[0]
    best = comp.sort_values("net_A_total_return_fixedbase_pct", ascending=False).iloc[0]
    print(f"\nWINNER (net_A total return): {best['config']} at {best['net_A_total_return_fixedbase_pct']:.2f}% "
          f"({best['net_A_total_return_fixedbase_pct']-base:+.2f} pts vs baseline)")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
