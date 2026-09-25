# -*- coding: utf-8 -*-
"""
cat_a_cross_compare.py — compare 3 configs of the composite strategy that differ ONLY in
Category A's FIRST-leg entry:
  BASELINE   : Cat A first 50% at +19% on PULLBACK after UC (recent-final logic).
  CROSS_19   : Cat A first 50% at +19%  on the UPWARD CROSS (fills at L as soon as a 1-min high
               reaches L, NO time limit; Cat A = any L-crosser, UC or not).
  CROSS_19_5 : same but L = +19.5%.
Everything else identical: C entry 15:21, exits conditional split t1=09:25 t2=11:59 + 17% long
target, double-down short covered at 3:00pm, ₹5L day-level capital sequencing, gross/net_A/net_B.

Cat A (cross variants) 2nd leg: if UC hit after the first fill -> +17% if a 1-min low <= +17%
occurs after the UC hit (proxy for post-release retrace), else 3:21 open if in [+17%,+20%) (not
re-locked), else half; if NO UC after first fill -> 3:21 open (if not in UC), else half.
Cat B (UC 3:00-3:15) is ABSORBED into Cat A in the cross variants (every UC-hitter crosses L
first). Flags: first-leg fills at L on the way up (limit-fill as it rallies through L) — a
different fill assumption than the pullback buy; both are limit-fill approximations.
"""
import sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = rb.IST
OUTDIR = rb.RESULTS / "cat_a_cross_compare"
HM_915, HM_1430, HM_1500, HM_1515, HM_1521, HM_END = 555, 870, 900, 915, 921, 929
UC_M, L19_M, L17_M, L195_M = 1.1995, 1.19, 1.17, 1.195
T1, T2 = 565, 719                 # 09:25 / 11:59 long exit legs
LONG_TGT_M = 1.17                 # 17% long profit target
COVER_HM = 900                    # 3:00pm short cover
BASE_POOL, BASE_ALLOC = 500_000, 100_000
LONG_023, LONG_038, SHORT_RATE = 0.0023, 0.0038, 0.0010
RF_ANNUAL = 0.075
MODES = ["baseline", "cross_19", "cross_19_5", "cross_19_a230", "cross_19_5_a230"]


def first_hm(hm, mask):
    return int(hm[mask].min()) if mask.any() else None


def build_legs(mode, pc, hm, hi, lo, op):
    """Return (category, entered, legs, meta) for one stock-day under one mode.
    legs = list of (frac, price, trigger_hm, phase); meta = dict of diagnostic flags."""
    uc, l19, l17, l195 = pc * UC_M, pc * L19_M, pc * L17_M, pc * L195_M
    o = dict(zip(hm, op))
    px_1521 = o.get(HM_1521, np.nan)
    ret_1521 = (px_1521 - pc) / pc * 100 if px_1521 == px_1521 else np.nan
    first_uc = first_hm(hm, hi >= uc)
    meta = {"hit_uc": first_uc is not None}

    if mode == "baseline":
        # Cat A = first UC hit before 3pm ; pullback ladder
        if first_uc is not None and first_uc < HM_1500:
            w = (hm >= first_uc) & (hm < HM_1500)
            wl, whm = lo[w], hm[w]
            i19 = np.where(wl <= l19)[0]
            if len(i19):
                t19 = int(whm[i19[0]])
                legs = [(0.5, l19, t19, 1)]
                i17 = np.where(wl <= l17)[0]
                if len(i17):
                    legs.append((0.5, l17, int(whm[i17[0]]), 1)); full = True
                elif l17 <= px_1521 < uc:
                    legs.append((0.5, px_1521, HM_1521, 2)); full = True
                else:
                    full = False
                return "A", True, legs, {**meta, "cat_a_full": full, "leg2": "17%" if len(i17) else ("3:21" if full else "half")}
            return "A", False, [], {**meta, "no_entry": "no_pullback_to_19"}
        if first_uc is not None and first_uc < HM_1515:
            return "B", True, [(1.0, uc, first_uc, 1)], meta
        if px_1521 == px_1521 and px_1521 > 0:
            return "C", True, [(1.0, px_1521, HM_1521, 3)], meta
        return "C", False, [], {**meta, "no_entry": "no_1521"}

    # ── CROSS variants: Cat A = crossed L in [lo_hm, 3pm) (first leg at L on the cross) ──
    #   plain cross_*  -> lo_hm = market open (any cross before 3pm)
    #   *_a230 variant -> lo_hm = 2:30pm (first cross must be a late 2:30-3:00 breakout)
    L = l195 if "19_5" in mode else l19
    lo_hm = HM_1430 if "a230" in mode else 0
    cross = first_hm(hm, hi >= L)
    if cross is not None and lo_hm <= cross < HM_1500:              # crossed L in window -> Cat A
        legs = [(0.5, L, cross, 1)]
        uc_after = first_uc is not None and first_uc >= cross
        leg2 = "half"; full = False
        if uc_after:
            retr = np.where((hm > first_uc) & (lo <= l17))[0]      # low<=17% after UC hit
            if len(retr):
                legs.append((0.5, l17, int(hm[retr[0]]), 1)); full = True; leg2 = "17%_postUC"
            elif l17 <= px_1521 < uc:
                legs.append((0.5, px_1521, HM_1521, 2)); full = True; leg2 = "3:21"
        else:
            if (px_1521 == px_1521) and 0 < px_1521 < uc:
                legs.append((0.5, px_1521, HM_1521, 2)); full = True; leg2 = "3:21_noUC"
        return "A", True, legs, {**meta, "cat_a_full": full, "uc_after_first_fill": bool(uc_after), "leg2": leg2}
    # not a Cat-A crosser: UC specifically in 3:00-3:15 -> Cat B (like baseline); else Cat C
    if first_uc is not None and HM_1500 <= first_uc < HM_1515:
        return "B", True, [(1.0, uc, first_uc, 1)], meta
    if px_1521 == px_1521 and px_1521 > 0:
        return "C", True, [(1.0, px_1521, HM_1521, 3)], meta
    return "C", False, [], {**meta, "no_entry": "no_1521"}


def scan(symbols, Q):
    """Per symbol read once; per qualifying stock-day produce shared next-day exit data + the
    3 modes' entry legs."""
    recs = {m: [] for m in MODES}
    t0 = time.time()
    for si, sym in enumerate(sorted(symbols), 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
        by_date = {d: g for d, g in raw.groupby("date")}
        dates = sorted(by_date)
        sub = Q[Q["symbol"] == sym]
        for _, r in sub.iterrows():
            ed = r["date"]; pc = float(r["prev_day_vwap_close"])
            if ed not in by_date or not (pc == pc and pc > 0):
                continue
            g = by_date[ed]
            hm = g["hm"].values; hi = g["high"].values.astype(float)
            lo = g["low"].values.astype(float); op = g["open"].values.astype(float)
            j = dates.index(ed) if ed in dates else -1
            nd = dates[j + 1] if 0 <= j < len(dates) - 1 else None
            if nd is None:
                continue
            ng = by_date[nd]
            no = dict(zip(ng["hm"].values, ng["open"].values.astype(float)))
            nhm = ng["hm"].values; nhi = ng["high"].values.astype(float)
            hmask = (nhm >= HM_915) & (nhm <= T2)
            exitdata = {"o565": no.get(T1, np.nan), "o719": no.get(T2, np.nan),
                        "o900": no.get(COVER_HM, np.nan),
                        "nhm": nhm[hmask], "nhi": nhi[hmask]}
            for m in MODES:
                cat, entered, legs, meta = build_legs(m, pc, hm, hi, lo, op)
                recs[m].append({"symbol": sym, "entry_date": ed, "category": cat, "entered": entered,
                                "legs": legs, "meta": meta, "ex": exitdata})
        if si % 300 == 0:
            print(f"  …{si} symbols ({time.time()-t0:.0f}s)", flush=True)
    return recs


def size_and_exit(records):
    """Day-level sequencing (Phase1 A/B chronological FCFS -> Phase2 A 3:21 topups -> Phase3 C
    remainder, ₹1L cap) then fixed exit (split 09:25/11:59 + 17% target) + 3pm short cover."""
    entered = [r for r in records if r["entered"] and r["legs"]]
    for r in entered:
        r["_alloc"] = [0.0] * len(r["legs"])
    by_day = defaultdict(list)
    for r in entered:
        by_day[r["entry_date"]].append(r)
    for d, rs in by_day.items():
        pool = BASE_POOL; X = 0.0
        p1, p2, cc = [], [], []
        for r in rs:
            for li, (frac, price, thm, ph) in enumerate(r["legs"]):
                if ph == 3:
                    cc.append((r, li, price))
                elif ph == 2:
                    p2.append((r, li, 0.5 * BASE_ALLOC))
                else:
                    p1.append((thm, r, li, BASE_ALLOC if frac >= 1.0 else 0.5 * BASE_ALLOC))
        for thm, r, li, intd in sorted(p1, key=lambda x: (x[0] if x[0] is not None else 99999)):
            give = min(intd, pool); r["_alloc"][li] = give; pool -= give; X += give
        for r, li, intd in p2:
            give = min(intd, pool); r["_alloc"][li] = give; pool -= give; X += give
        nC = len(cc); per_c = min(BASE_ALLOC, max(0.0, pool) / nC) if nC else 0.0
        for r, li, price in cc:
            r["_alloc"][li] = per_c

    rows = []
    for r in entered:
        shares = cap = 0.0; nlegs = 0
        for li, (frac, price, thm, ph) in enumerate(r["legs"]):
            s = np.floor(r["_alloc"][li] / price) if r["_alloc"][li] > 0 else 0.0
            if s > 0:
                shares += s; cap += s * price; nlegs += 1
        if shares <= 0:
            continue
        avg = cap / shares
        ex = r["ex"]
        tgt = avg * LONG_TGT_M
        hit = ex["nhi"] >= tgt
        thit = int(ex["nhm"][hit].min()) if hit.any() else 10 ** 9
        o565, o719, o900 = ex["o565"], ex["o719"], ex["o900"]
        # conditional split with 17% target
        if thit <= T1:
            xp = tgt
        elif o565 == o565 and o565 > avg:
            xp = o565
        elif thit <= T2:
            xp = tgt
        elif o719 == o719:
            xp = o719
        else:
            continue
        long_pnl = shares * (xp - avg)
        has_short = o900 == o900
        short_pnl = shares * (xp - o900) if has_short else 0.0
        short_notl = shares * xp if has_short else 0.0
        comb = long_pnl + short_pnl
        rows.append({"symbol": r["symbol"], "entry_date": r["entry_date"], "category": r["category"],
                     "n_legs": nlegs, "shares": int(shares), "capital_deployed": cap, "avg_entry": avg,
                     "long_pnl": long_pnl, "short_pnl": short_pnl, "combined_pnl": comb,
                     "gross_pnl": comb, "netA_pnl": comb - LONG_023 * cap - SHORT_RATE * short_notl,
                     "netB_pnl": comb - LONG_038 * cap - SHORT_RATE * short_notl,
                     "meta": r["meta"]})
    T = pd.DataFrame(rows)
    for s in ["gross", "netA", "netB"]:
        T[f"{s}_ret"] = T[f"{s}_pnl"] / T["capital_deployed"] * 100
    return T


def metrics(T, label):
    out = {"config": label, "n_trades": len(T)}
    for s, tag in [("gross", "gross"), ("netA", "net_A"), ("netB", "net_B")]:
        p = T[f"{s}_pnl"]; r = T[f"{s}_ret"]
        out[f"{tag}_total_return_fixedbase_pct"] = round(p.sum() / BASE_POOL * 100, 2)
        out[f"{tag}_total_pnl_inr"] = round(p.sum(), 0)
        out[f"{tag}_win_rate_pct"] = round((p > 0).mean() * 100, 2)
        out[f"{tag}_avg_return_per_trade_pct"] = round(r.mean(), 4)
        out[f"{tag}_median_return_per_trade_pct"] = round(r.median(), 4)
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_all_three", "prev_day_vwap_close"],
                       parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True].reset_index(drop=True)
    print(f"qualifying stock-days: {len(Q):,}\nScanning (3 modes) …")
    recs = scan(Q["symbol"].unique(), Q)

    Ts = {m: size_and_exit(recs[m]) for m in MODES}
    comp = pd.DataFrame([metrics(Ts[m], m) for m in MODES])

    # Category-A(+absorbed-B) isolated
    catA_rows = []
    for m in MODES:
        T = Ts[m]
        cats = ["A", "B"]
        A = T[T["category"].isin(cats)]
        catA_rows.append({"config": m, "cat": "+".join(cats), "n": len(A),
                          "gross_total_pnl_inr": round(A["gross_pnl"].sum(), 0),
                          "gross_return_fixedbase_pct": round(A["gross_pnl"].sum() / BASE_POOL * 100, 2),
                          "netA_total_pnl_inr": round(A["netA_pnl"].sum(), 0),
                          "netA_return_fixedbase_pct": round(A["netA_pnl"].sum() / BASE_POOL * 100, 2),
                          "netA_win_rate_pct": round((A["netA_pnl"] > 0).mean() * 100, 2),
                          "netA_avg_return_per_trade_pct": round(A["netA_ret"].mean(), 4),
                          "avg_capital_deployed": round(A["capital_deployed"].mean(), 0)})
    catA = pd.DataFrame(catA_rows)

    # cross-variant fill-path breakdown
    fp = []
    for m in [x for x in MODES if x != "baseline"]:
        A = [r for r in recs[m] if r["category"] == "A" and r["entered"]]
        nuc = sum(1 for r in A if r["meta"].get("uc_after_first_fill"))
        nno = len(A) - nuc
        l2 = defaultdict(int)
        for r in A:
            l2[r["meta"].get("leg2", "?")] += 1
        fp.append({"config": m, "cat_A_entered": len(A),
                   "first_fill_then_hit_UC": nuc, "first_fill_no_UC (new vs baseline)": nno,
                   "2nd_leg_17pct": l2.get("17%_postUC", 0),
                   "2nd_leg_3:21": l2.get("3:21", 0) + l2.get("3:21_noUC", 0),
                   "2nd_leg_half(none)": l2.get("half", 0)})
    fillpath = pd.DataFrame(fp)

    with pd.ExcelWriter(OUTDIR / "cat_a_cross_compare.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="whole_strategy_compare", index=False)
        catA.to_excel(w, sheet_name="categoryA_isolated", index=False)
        fillpath.to_excel(w, sheet_name="cross_fill_path", index=False)
        for m in MODES:
            Ts[m][["entry_date", "symbol", "category", "n_legs", "shares", "capital_deployed",
                   "avg_entry", "long_pnl", "short_pnl", "combined_pnl", "gross_pnl", "netA_pnl",
                   "netB_pnl"]].to_excel(w, sheet_name=f"trades_{m}"[:31], index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 110 + "\nCAT-A: PULLBACK-after-UC (baseline) vs CROSS-before-UC (19% / 19.5%) — combined long+short\n" + "=" * 110)
    show = ["config", "n_trades", "gross_total_return_fixedbase_pct", "net_A_total_return_fixedbase_pct",
            "net_B_total_return_fixedbase_pct", "net_A_win_rate_pct", "net_A_avg_return_per_trade_pct",
            "net_A_median_return_per_trade_pct"]
    print(comp[show].to_string(index=False))
    print("\n--- CATEGORY A (+absorbed B in baseline) ISOLATED ---")
    print(catA.to_string(index=False))
    print("\n--- CROSS-VARIANT FILL-PATH BREAKDOWN ---")
    print(fillpath.to_string(index=False))
    b = comp[comp["config"] == "baseline"].iloc[0]
    for m in ["cross_19", "cross_19_5"]:
        c = comp[comp["config"] == m].iloc[0]
        print(f"  {m:11s}: net_A {c['net_A_total_return_fixedbase_pct']:.2f}% vs baseline {b['net_A_total_return_fixedbase_pct']:.2f}%  "
              f"-> {c['net_A_total_return_fixedbase_pct']-b['net_A_total_return_fixedbase_pct']:+.2f} pts")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
