# -*- coding: utf-8 -*-
"""
uc_staggered_dd_report.py
=========================
Full performance report for the MODIFIED main strategy: UC-specific staggered entries
(Categories A/B/C) + a double-down short overlay, on 1-MINUTE data.

Base signal set = passes_all_three from results/diagnostic_table.csv
(mcap ₹1,500-5,000 Cr, LB 36, VM 6, +5% vs prev-day VWAP-close, 3:15pm reference).
Every level (UC ±20%, +19%, +17%, +5%) is measured off the strategy's VWAP prev-close
(prev_day_vwap_close) for internal consistency with the +5% filter.

ENTRY (per qualifying stock-day; alloc = ₹1L, or ₹5L/n when >5 positions enter that day):
  UC level  = prev_close × 1.1995   (first 1-min high ≥ this = "hit UC")
  Category by FIRST UC-hit time:
    A  first hit  < 15:00  → staggered pullback ladder (after the UC hit, before 15:00):
         50% fills @ prev_close×1.19 if a 1-min low ≤ that level;
         50% fills @ prev_close×1.17 if a 1-min low ≤ that level;
         if +17% not reached before 15:00 but the +19% leg filled → fill the 2nd 50% at the
           15:15 candle open ONLY IF that open is in [+17%, +20%) (not re-locked), else stay half;
         if +19% never reached before 15:00 → NO ENTRY.
    B  15:00 ≤ first hit < 15:15 → enter FULL ₹1L at the UC price (the circuit price).
    C  else (no UC hit, or first hit ≥ 15:15) → normal FULL ₹1L at the 15:15 candle open.

EXITS (unchanged for ALL categories; scanned on next-day 1-min candles):
  14% target from the (blended) entry, scanned on 1-min highs 09:15→12:00;
  else positive at 09:45 (vs entry) → exit at the 09:45 open;
  else exit at the 12:00 open.

DOUBLE-DOWN SHORT (all categories): at each long exit, short the FILLED long qty
  (Cat-A half-fill shorts only the 50%); cover at the exit-day 15:00 candle open.
  short_pnl = filled_shares × (long_exit_price − open_1500);  combined = long_pnl + short_pnl.

COSTS (proportional to ACTUAL filled value):
  gross  — none
  net_A  — long 0.23% × capital_deployed + short 0.10% × short_notional
  net_B  — long 0.38% × capital_deployed + short 0.10% × short_notional

Flags (see module docstring / conversation): (a) Cat-A no-pullback-to-19% ⇒ no entry;
(b) Cat-A legs fill at the exact 1.19 / 1.17 limit levels; (c) UC + pullbacks on 1-min candles;
(d) Cat-B at the UC price; (e) short size = filled long qty; (f) exits & 14% target unchanged,
target on 1-min highs; (g) costs on actual filled value. UC/±% levels off VWAP prev-close.
"""

import sys, bisect
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

IST = rb.IST
OUTDIR = rb.RESULTS / "uc_staggered_dd_report"
XLSX = OUTDIR / "uc_staggered_dd_report.xlsx"

# ── 1-min session grid (minutes-of-day) ──
HM_915, HM_945, HM_1200, HM_1500, HM_1515, HM_END = 555, 585, 720, 900, 915, 929
UC_MULT, L19_MULT, L17_MULT = 1.1995, 1.19, 1.17
TARGET = 14.0

BASE_POOL, BASE_ALLOC, SPLIT_ABOVE = 500_000, 100_000, 5
LONG_023, LONG_038, SHORT_RATE = 0.0023, 0.0038, 0.0010
RF_ANNUAL = 0.075


def hm_str(hm):
    return f"{int(hm)//60:02d}:{int(hm)%60:02d}" if hm is not None and hm == hm else ""


# ════════════════════════════════════════════════════════════════════════════
# PASS 1 — per-symbol 1-min scan: classify category, decide fills + exit path
# ════════════════════════════════════════════════════════════════════════════
def scan_symbol(sym, sub):
    """sub: rows of diagnostic_table for this symbol (passes_all_three). Returns list of
    partial records (sizing-independent: category, fill legs, exit path, opens)."""
    pq = rb.MASTER_DIR / f"{sym}.parquet"
    if not pq.exists():
        return []
    raw = pd.read_parquet(pq)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
    raw["date"] = raw["timestamp"].dt.date
    raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
    raw = raw[(raw["hm"] >= HM_915) & (raw["hm"] <= HM_END)]
    dates = sorted(raw["date"].unique())
    # per-date candle dicts
    by_date = {d: g for d, g in raw.groupby("date")}

    out = []
    for _, r in sub.iterrows():
        ed = r["date"]
        pc = float(r["prev_day_vwap_close"])
        if not (pc == pc and pc > 0) or ed not in by_date:
            continue
        j = bisect.bisect_right(dates, ed)
        nd = dates[j] if j < len(dates) else None            # next trading day

        g = by_date[ed]
        hm = g["hm"].values
        hi = g["high"].values.astype(float)
        lo = g["low"].values.astype(float)
        op = g["open"].values.astype(float)
        o_by = dict(zip(hm, op))
        uc, l19, l17 = pc * UC_MULT, pc * L19_MULT, pc * L17_MULT
        px_315 = o_by.get(HM_1515, float(r["entry_price_315pm"]))

        # first UC-hit minute
        uc_mask = hi >= uc
        first_uc = int(hm[uc_mask].min()) if uc_mask.any() else None

        rec = {"symbol": sym, "entry_date": ed, "next_date": nd, "prev_close": pc,
               "uc": uc, "px_315": px_315, "uc_hit_hm": first_uc,
               "ret_315": (px_315 - pc) / pc * 100, "legs": None, "category": None,
               "entered": False, "cat_a_full": None, "no_entry_reason": None}

        # legs carry (frac, price, label, trigger_hm, phase) — phase drives day-level sequencing:
        #   phase 1 = intraday A/B commitments (ordered by trigger_hm); phase 2 = A 3:15 top-up;
        #   phase 3 = Category C (splits the day's remaining pool).
        if first_uc is not None and first_uc < HM_1500:
            rec["category"] = "A"
            # pullback window: from the UC-hit minute (incl.) up to but not incl. 15:00
            w = (hm >= first_uc) & (hm < HM_1500)
            wl = lo[w]; whm = hm[w]
            i19 = np.where(wl <= l19)[0]
            if len(i19):                                      # 1st 50% fills @ +19% (Phase 1, t19)
                t19 = int(whm[i19[0]])
                legs = [(0.5, l19, "19%", t19, 1)]
                i17 = np.where(wl <= l17)[0]
                if len(i17):                                  # 2nd 50% @ +17% intraday (Phase 1, t17)
                    t17 = int(whm[i17[0]])
                    legs.append((0.5, l17, "17%", t17, 1)); rec["cat_a_full"] = True
                elif l17 <= px_315 < uc:                      # 2nd 50% fallback @ 15:15 (Phase 2)
                    legs.append((0.5, px_315, "3:15open", HM_1515, 2)); rec["cat_a_full"] = True
                else:                                         # stays half
                    rec["cat_a_full"] = False
                rec["legs"] = legs; rec["entered"] = True
            else:
                rec["no_entry_reason"] = "no_pullback_to_19"
        elif first_uc is not None and first_uc < HM_1515:      # 15:00 ≤ hit < 15:15 (Phase 1)
            rec["category"] = "B"; rec["legs"] = [(1.0, uc, "UC", first_uc, 1)]; rec["entered"] = True
        else:                                                  # no hit, or hit ≥ 15:15 (Phase 3)
            rec["category"] = "C"
            if px_315 == px_315 and px_315 > 0:
                rec["legs"] = [(1.0, px_315, "3:15open", HM_1515, 3)]; rec["entered"] = True
            else:
                rec["no_entry_reason"] = "no_315_open"

        if not rec["entered"]:
            out.append(rec); continue

        # ── next-day exit path (nominal sizing to pick the path; size-independent otherwise) ──
        if nd is None or nd not in by_date:
            rec["entered"] = False; rec["no_entry_reason"] = "no_next_day"; out.append(rec); continue
        avg0 = (sum(f * p for f, p, *_ in rec["legs"]) / sum(f for f, p, *_ in rec["legs"]))  # ~fill price
        ng = by_date[nd]
        nhm = ng["hm"].values; nhi = ng["high"].values.astype(float)
        nop = dict(zip(nhm, ng["open"].values.astype(float)))
        o945, o1200, o1500 = nop.get(HM_945, np.nan), nop.get(HM_1200, np.nan), nop.get(HM_1500, np.nan)
        tgt = avg0 * (1 + TARGET / 100)

        exit_type = exit_hm = None; is_target = False
        pre = (nhm >= HM_915) & (nhm < HM_945)
        hit = pre & (nhi >= tgt)
        if hit.any():
            exit_type, exit_hm, is_target = "early_target_pre_0945", int(nhm[hit].min()), True
        elif o945 == o945 and o945 > avg0:
            exit_type, exit_hm = "positive_at_0945", HM_945
        else:
            bet = (nhm > HM_945) & (nhm < HM_1200)
            hit2 = bet & (nhi >= tgt)
            if hit2.any():
                exit_type, exit_hm, is_target = "early_target_0945_1200", int(nhm[hit2].min()), True
            elif o1200 == o1200:
                exit_type, exit_hm = "exit_at_1200", HM_1200
        if exit_type is None:
            rec["entered"] = False; rec["no_entry_reason"] = "no_valid_exit"; out.append(rec); continue

        rec.update({"exit_type": exit_type, "exit_hm": exit_hm, "is_target": is_target,
                    "o945": o945, "o1200": o1200, "o1500_next": o1500})
        out.append(rec)
    return out


# ════════════════════════════════════════════════════════════════════════════
# PASS 2 — pool-split sizing, blended entry, exit price, long/short/combined P&L
# ════════════════════════════════════════════════════════════════════════════
def size_and_price(records):
    """Day-level capital SEQUENCING (strict priority A/B intraday -> A 3:15 top-up -> C):
      Phase 1  chronological FCFS: A first leg ₹50k @+19%, A second leg ₹50k @+17%, B ₹1L @UC.
      Phase 2  at 3:15, before C: A second-leg fallback ₹50k top-ups (in [+17%,+20%), not re-locked).
      Phase 3  Category C splits the remainder: per_C = min(₹1L, max(0, 500000-X)/n_C).
    Hard ₹1L/position cap always. Pool exhaustion is FCFS (a boundary leg gets whatever remains,
    later legs get 0). Returns (T, day_diag) where day_diag holds per-day allocation diagnostics."""
    entered = [r for r in records if r["entered"]]
    for r in entered:
        r["_leg_alloc"] = [0.0] * len(r["legs"])
    by_day = defaultdict(list)
    for r in entered:
        by_day[r["entry_date"]].append(r)

    day_diag = {}
    for d, recs in by_day.items():
        pool = BASE_POOL; X = 0.0
        phase1, phase2, cat_c = [], [], []
        for r in recs:
            for li, (frac, price, lbl, thm, phase) in enumerate(r["legs"]):
                if phase == 3:
                    cat_c.append((r, li, price))
                elif phase == 2:
                    phase2.append((r, li, 0.5 * BASE_ALLOC))
                else:                                              # phase 1 (A leg = 50k, B = 1L)
                    phase1.append((thm, r, li, BASE_ALLOC if frac >= 1.0 else 0.5 * BASE_ALLOC))
        for thm, r, li, intended in sorted(phase1, key=lambda x: x[0]):   # chronological FCFS
            give = min(intended, pool); r["_leg_alloc"][li] = give; pool -= give; X += give
        for r, li, intended in phase2:                             # A top-ups, before C
            give = min(intended, pool); r["_leg_alloc"][li] = give; pool -= give; X += give
        n_C = len(cat_c); remaining = pool
        per_C = min(BASE_ALLOC, max(0.0, remaining) / n_C) if n_C else 0.0
        for r, li, price in cat_c:
            r["_leg_alloc"][li] = per_C
        cstat = ("no_C" if n_C == 0 else "zero" if per_C <= 0
                 else "full_1L" if per_C >= BASE_ALLOC - 1e-6 else "equal_split")
        day_diag[d] = {"X_ab": X, "n_C": n_C, "per_C": per_C, "remaining": remaining, "cstat": cstat}

    rows = []
    for r in entered:
        shares = 0.0; cap = 0.0; leg_lbls = []; leg_pxs = []
        for li, (frac, price, lbl, thm, phase) in enumerate(r["legs"]):
            a = r["_leg_alloc"][li]
            s = float(np.floor(a / price)) if a > 0 else 0.0
            if s <= 0:
                continue
            shares += s; cap += s * price
            leg_lbls.append(f"{lbl}({int(s)})"); leg_pxs.append(round(price, 2))
        if shares <= 0:
            continue
        avg_entry = cap / shares
        exit_price = avg_entry * (1 + TARGET / 100) if r["is_target"] else \
            (r["o945"] if r["exit_type"] == "positive_at_0945" else r["o1200"])
        long_pnl = shares * (exit_price - avg_entry)
        o1500 = r["o1500_next"]
        has_short = o1500 == o1500
        short_pnl = shares * (exit_price - o1500) if has_short else 0.0
        short_notl = shares * exit_price if has_short else 0.0
        combined = long_pnl + short_pnl

        rows.append({
            "symbol": r["symbol"], "entry_date": r["entry_date"], "exit_date": r["next_date"],
            "category": r["category"], "cat_a_full": r["cat_a_full"],
            "legs_filled": "+".join(leg_lbls), "leg_prices": "+".join(str(p) for p in leg_pxs),
            "n_legs": len(leg_lbls), "avg_entry": avg_entry, "shares": int(shares),
            "capital_deployed": cap, "alloc_intended": round(sum(r["_leg_alloc"]), 0),
            "exit_type": r["exit_type"], "exit_time": hm_str(r["exit_hm"]), "exit_price": exit_price,
            "cover_1500_open": o1500 if has_short else np.nan,
            "long_pnl": long_pnl, "short_pnl": short_pnl, "short_notional": short_notl,
            "combined_pnl": combined,
            "long_cost_023": LONG_023 * cap, "long_cost_038": LONG_038 * cap,
            "short_cost": SHORT_RATE * short_notl,
        })
    T = pd.DataFrame(rows)
    # P&L series (gross / net_A / net_B) on the COMBINED position
    T["gross_pnl"] = T["combined_pnl"]
    T["netA_pnl"] = T["combined_pnl"] - T["long_cost_023"] - T["short_cost"]
    T["netB_pnl"] = T["combined_pnl"] - T["long_cost_038"] - T["short_cost"]
    for s in ["gross", "netA", "netB"]:
        T[f"{s}_ret"] = T[f"{s}_pnl"] / T["capital_deployed"] * 100
    T["long_ret"] = T["long_pnl"] / T["capital_deployed"] * 100
    T["short_ret"] = np.where(T["short_notional"] > 0,
                              T["short_pnl"] / T["capital_deployed"] * 100, 0.0)
    ts = pd.to_datetime(T["entry_date"])
    T["year"] = ts.dt.year
    T["month"] = ts.dt.strftime("%Y-%m")
    T["quarter"] = ts.dt.year.astype(str) + "Q" + ts.dt.quarter.astype(str)
    T["half_year"] = ts.dt.year.astype(str) + "H" + np.where(ts.dt.month <= 6, "1", "2")
    return T.sort_values(["entry_date", "symbol"]).reset_index(drop=True), day_diag


# ════════════════════════════════════════════════════════════════════════════
# period metrics (3 series) + compounding
# ════════════════════════════════════════════════════════════════════════════
def pmetrics(df):
    n = len(df)
    out = {"n_trades": n}
    for s, tag in [("gross", "gross"), ("netA", "net_A"), ("netB", "net_B")]:
        pnl = df[f"{s}_pnl"]; ret = df[f"{s}_ret"]
        out[f"{tag}_total_pnl_inr"] = round(pnl.sum(), 0)
        out[f"{tag}_total_return_fixedbase_pct"] = round(pnl.sum() / BASE_POOL * 100, 4)
        out[f"{tag}_win_rate_pct"] = round((pnl > 0).mean() * 100, 2) if n else 0.0
        out[f"{tag}_avg_return_per_trade_pct"] = round(ret.mean(), 4) if n else 0.0
        out[f"{tag}_median_return_per_trade_pct"] = round(ret.median(), 4) if n else 0.0
    return out


def period_table(T, key, order_fn=None, compounded=False):
    groups = sorted(T.groupby(key), key=(order_fn or (lambda x: x[0])))
    pool = {"gross": BASE_POOL, "net_A": BASE_POOL, "net_B": BASE_POOL}
    rows = []
    for k, g in groups:
        m = {key: k, **pmetrics(g)}
        if compounded:
            for tag in ["gross", "net_A", "net_B"]:
                fb = m[f"{tag}_total_return_fixedbase_pct"]          # % on ₹5L fixed base
                cpnl = pool[tag] * fb / 100                          # ₹ earned on current pool
                m[f"{tag}_compounded_pool_value"] = round(pool[tag], 0)
                m[f"{tag}_compounded_pnl_inr"] = round(cpnl, 0)
                m[f"{tag}_compounded_return_pct"] = round(fb, 4)
                # carry-forward-on-loss RATCHET: scale pool up on a positive period return;
                # on a loss (<=0) carry the prior allocation level UNCHANGED (never scale down).
                if fb > 0:
                    pool[tag] *= (1 + fb / 100)
        rows.append(m)
    return pd.DataFrame(rows)


# ── daily win/loss streaks on a chosen series ──
def streaks(dates, pnls):
    bw_len = bl_len = 0; bw = bl = (None, None, 0.0)
    ct = None; cl = 0; cs = ce = None; cp = 0.0
    for d, p in zip(dates, pnls):
        typ = "win" if p > 0 else "los"
        if typ == ct:
            cl += 1; cp += p; ce = d
        else:
            ct, cl, cp, cs, ce = typ, 1, float(p), d, d
        if ct == "win" and cl > bw_len:
            bw_len, bw = cl, (cs, ce, cp)
        if ct == "los" and cl > bl_len:
            bl_len, bl = cl, (cs, ce, cp)
    return bw_len, bw, bl_len, bl


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_all_three",
                                "prev_day_vwap_close", "entry_price_315pm"],
                       parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True].reset_index(drop=True)
    print(f"qualifying signals (passes_all_three): {len(Q):,}")

    print("Scanning 1-min candles per symbol (entry-day classify + next-day exits) …")
    records = []
    syms = sorted(Q["symbol"].unique())
    for i, sym in enumerate(syms, 1):
        records += scan_symbol(sym, Q[Q["symbol"] == sym])
        if i % 200 == 0:
            print(f"  …{i}/{len(syms)} symbols, {sum(r['entered'] for r in records):,} entries so far")

    # ── entry funnel ──
    cat = pd.Series([r["category"] for r in records])
    n_A = int((cat == "A").sum()); n_B = int((cat == "B").sum()); n_C = int((cat == "C").sum())
    a_noentry = sum(1 for r in records if r["category"] == "A" and not r["entered"]
                    and r["no_entry_reason"] == "no_pullback_to_19")

    T, day_diag = size_and_price(records)
    # A full/half based on legs ACTUALLY funded under the sequenced allocation
    A = T[T["category"] == "A"]
    a_full = int((A["n_legs"] == 2).sum()); a_half = int((A["n_legs"] == 1).sum())
    print(f"  entered trades: {len(T):,}  (A={len(A)} [full {a_full}/half {a_half}], "
          f"B={int((T['category']=='B').sum())}, C={int((T['category']=='C').sum())}; "
          f"A no-entry={a_noentry})")

    # ── allocation diagnostics (from the day-level sequencing) ──
    xab = pd.Series([v["X_ab"] for v in day_diag.values()])
    cst = pd.Series([v["cstat"] for v in day_diag.values()])
    perc_days = pd.Series([v["per_C"] for v in day_diag.values() if v["n_C"] > 0])
    alloc_diag = {
        "avg_daily_AB_committed_inr": round(float(xab.mean()), 0),
        "max_daily_AB_committed_inr": round(float(xab.max()), 0),
        "n_days_C_full_1L": int((cst == "full_1L").sum()),
        "n_days_C_equal_split_lt_1L": int((cst == "equal_split").sum()),
        "n_days_C_zero_pool_exhausted": int((cst == "zero").sum()),
        "avg_C_allocation_per_trade_inr": round(float(perc_days.mean()), 0) if len(perc_days) else 0.0,
        "avg_cap_A_full_inr": round(float(A.loc[A["n_legs"] == 2, "capital_deployed"].mean()), 0) if (A["n_legs"] == 2).any() else 0.0,
        "avg_cap_A_half_inr": round(float(A.loc[A["n_legs"] == 1, "capital_deployed"].mean()), 0) if (A["n_legs"] == 1).any() else 0.0,
        "avg_cap_B_inr": round(float(T.loc[T["category"] == "B", "capital_deployed"].mean()), 0) if (T["category"] == "B").any() else 0.0,
        "avg_cap_C_inr": round(float(T.loc[T["category"] == "C", "capital_deployed"].mean()), 0) if (T["category"] == "C").any() else 0.0,
    }
    print(f"  alloc: avg A+B committed/day ₹{alloc_diag['avg_daily_AB_committed_inr']:,.0f} "
          f"(max ₹{alloc_diag['max_daily_AB_committed_inr']:,.0f}); "
          f"C days full/split/zero = {alloc_diag['n_days_C_full_1L']}/"
          f"{alloc_diag['n_days_C_equal_split_lt_1L']}/{alloc_diag['n_days_C_zero_pool_exhausted']}; "
          f"avg C alloc ₹{alloc_diag['avg_C_allocation_per_trade_inr']:,.0f}")

    # ── period tables ──
    daily = pd.DataFrame([{"date": k, **pmetrics(g),
                           "gross_is_winning_day": bool(g["gross_pnl"].sum() > 0),
                           "net_A_is_winning_day": bool(g["netA_pnl"].sum() > 0),
                           "net_B_is_winning_day": bool(g["netB_pnl"].sum() > 0)}
                          for k, g in T.groupby("exit_date")]).sort_values("date").reset_index(drop=True)
    monthly = period_table(T, "month")
    quarterly = period_table(T, "quarter", compounded=True)
    halfy = period_table(T, "half_year")
    yearly = period_table(T, "year", order_fn=lambda x: int(x[0]), compounded=True)

    # ── daily P&L by exit day (streaks + winning/losing days) ──
    de = T.groupby("exit_date").agg(gross=("gross_pnl", "sum"), netA=("netA_pnl", "sum"),
                                    netB=("netB_pnl", "sum")).reset_index().sort_values("exit_date")
    ex_dates = de["exit_date"].tolist()
    strk = {s: streaks(ex_dates, de[s].values) for s in ["gross", "netA", "netB"]}

    # ── daily return series for Sharpe (return on deployed capital) ──
    dd = T.groupby("exit_date").agg(g=("gross_pnl", "sum"), a=("netA_pnl", "sum"),
                                    b=("netB_pnl", "sum"), c=("capital_deployed", "sum")).reset_index()
    rf_daily = RF_ANNUAL / 252 * 100
    def sharpe(x):
        return round((x.mean() - rf_daily) / x.std(ddof=1) * np.sqrt(252), 4) if len(x) > 1 else 0.0
    sh = {"gross": sharpe(dd["g"] / dd["c"] * 100), "net_A": sharpe(dd["a"] / dd["c"] * 100),
          "net_B": sharpe(dd["b"] / dd["c"] * 100)}

    # ════════════════════════ SUMMARY sheet ════════════════════════
    n = len(T)
    def S(metric, g, a, b):
        return {"metric": metric, "gross": g, "net_A@0.23%+0.10%": a, "net_B@0.38%+0.10%": b}
    rows = []
    def col(metric, fn):
        return S(metric, fn("gross"), fn("netA"), fn("netB"))

    rows += [
        S("n_trades", n, n, n),
        col("total_return_fixedbase_pct", lambda s: round(T[f"{s}_pnl"].sum() / BASE_POOL * 100, 4)),
        col("total_pnl_inr", lambda s: round(T[f"{s}_pnl"].sum(), 0)),
        col("win_rate_pct", lambda s: round((T[f"{s}_pnl"] > 0).mean() * 100, 2)),
        col("n_winning_trades", lambda s: int((T[f"{s}_pnl"] > 0).sum())),
        col("n_losing_trades", lambda s: int((T[f"{s}_pnl"] <= 0).sum())),
        col("avg_return_per_trade_pct", lambda s: round(T[f"{s}_ret"].mean(), 4)),
        col("avg_pnl_per_trade_inr", lambda s: round(T[f"{s}_pnl"].mean(), 2)),
        col("median_return_per_trade_pct", lambda s: round(T[f"{s}_ret"].median(), 4)),
        col("avg_return_winning_trades_pct", lambda s: round(T.loc[T[f"{s}_pnl"] > 0, f"{s}_ret"].mean(), 4)),
        col("avg_return_losing_trades_pct", lambda s: round(T.loc[T[f"{s}_pnl"] <= 0, f"{s}_ret"].mean(), 4)),
        col("avg_capital_deployed_inr", lambda s: round(T["capital_deployed"].mean(), 0)),
    ]
    # winning/losing days (by exit day) per series
    rows += [
        col("n_winning_days", lambda s: int((de[{"gross":"gross","netA":"netA","netB":"netB"}[s]] > 0).sum())),
        col("n_losing_days", lambda s: int((de[{"gross":"gross","netA":"netA","netB":"netB"}[s]] <= 0).sum())),
    ]
    # period averages  (period-table cols use net_A/net_B; col() feeds s in gross/netA/netB)
    SER = {"gross": "gross", "netA": "net_A", "netB": "net_B"}
    for lbl, dfp in [("month", monthly), ("quarter", quarterly),
                     ("half_year", halfy), ("year", yearly)]:
        rows.append(col(f"avg_return_per_{lbl}_pct",
                        lambda s, d=dfp: round(d[f"{SER[s]}_total_return_fixedbase_pct"].mean(), 4)))
    rows.append(S("sharpe_ratio", sh["gross"], sh["net_A"], sh["net_B"]))
    # streaks
    def per(t):
        return f"{t[0]} -> {t[1]}" if t[0] is not None else "—"
    rows += [
        S("max_consecutive_winning_days", strk["gross"][0], strk["netA"][0], strk["netB"][0]),
        S("max_consecutive_winning_days_pnl_inr", round(strk["gross"][1][2], 0), round(strk["netA"][1][2], 0), round(strk["netB"][1][2], 0)),
        S("max_consecutive_winning_days_period", per(strk["gross"][1]), per(strk["netA"][1]), per(strk["netB"][1])),
        S("max_consecutive_losing_days", strk["gross"][2], strk["netA"][2], strk["netB"][2]),
        S("max_consecutive_losing_days_pnl_inr", round(strk["gross"][3][2], 0), round(strk["netA"][3][2], 0), round(strk["netB"][3][2], 0)),
        S("max_consecutive_losing_days_period", per(strk["gross"][3]), per(strk["netA"][3]), per(strk["netB"][3])),
    ]

    # ── category funnel (counts; same across cost series) ──
    def SC(metric, val):
        return S(metric, val, val, val)
    rows += [
        SC("--- CATEGORY BREAKDOWN ---", ""),
        SC("n_category_A_signals", n_A),
        SC("n_category_A_entered", int((T["category"] == "A").sum())),
        SC("n_category_A_full_fill", a_full),
        SC("n_category_A_half_fill", a_half),
        SC("n_category_A_no_entry", a_noentry),
        SC("n_category_B_entered", int((T["category"] == "B").sum())),
        SC("n_category_C_entered", int((T["category"] == "C").sum())),
    ]
    # ── allocation-sequencing diagnostics (allocation facts; same across cost series) ──
    rows.append(SC("--- ALLOCATION SEQUENCING DIAGNOSTICS ---", ""))
    for k, v in alloc_diag.items():
        rows.append(SC(k, v))
    # per-category P&L contribution (3 series)
    rows.append(SC("--- CATEGORY P&L (combined, fixedbase %) ---", ""))
    for c in ["A", "B", "C"]:
        gc = T[T["category"] == c]
        rows.append(col(f"category_{c}_total_pnl_inr", lambda s, g=gc: round(g[f"{s}_pnl"].sum(), 0)))
        rows.append(col(f"category_{c}_return_fixedbase_pct", lambda s, g=gc: round(g[f"{s}_pnl"].sum() / BASE_POOL * 100, 4)))
        rows.append(col(f"category_{c}_win_rate_pct", lambda s, g=gc: round((g[f"{s}_pnl"] > 0).mean() * 100, 2) if len(g) else 0.0))

    # ── leg decomposition: long vs short (gross), and per-category standalone legs ──
    rows.append(SC("--- LEG DECOMPOSITION (gross) ---", ""))
    rows.append(SC("long_leg_total_pnl_inr", round(T["long_pnl"].sum(), 0)))
    rows.append(SC("long_leg_return_fixedbase_pct", round(T["long_pnl"].sum() / BASE_POOL * 100, 4)))
    rows.append(SC("long_leg_win_rate_pct", round((T["long_pnl"] > 0).mean() * 100, 2)))
    sh_mask = T["short_notional"] > 0
    rows.append(SC("short_leg_total_pnl_inr", round(T["short_pnl"].sum(), 0)))
    rows.append(SC("short_leg_return_fixedbase_pct", round(T["short_pnl"].sum() / BASE_POOL * 100, 4)))
    rows.append(SC("short_leg_win_rate_pct", round((T.loc[sh_mask, "short_pnl"] > 0).mean() * 100, 2)))
    for c in ["A", "B", "C"]:
        g = T[T["category"] == c]
        rows.append(SC(f"long_leg_category_{c}_pnl_inr", round(g["long_pnl"].sum(), 0)))
        rows.append(SC(f"short_leg_category_{c}_pnl_inr", round(g["short_pnl"].sum(), 0)))

    # ── best/worst period extremes (fixed-base; per series) ──
    def add_extremes(df, name, label_col):
        mx = {t: df.loc[df[f"{t}_total_pnl_inr"].idxmax()] for t in ["gross", "net_A", "net_B"]}
        mn = {t: df.loc[df[f"{t}_total_pnl_inr"].idxmin()] for t in ["gross", "net_A", "net_B"]}
        L = {"gross": "gross", "netA": "net_A", "netB": "net_B"}
        rows.append(SC(f"--- {name.upper()} EXTREMES ---", ""))
        rows.append(col(f"highest_profit_{name}_pnl_inr", lambda s: round(float(mx[L[s]][f"{L[s]}_total_pnl_inr"]), 0)))
        rows.append(S(f"highest_profit_{name}", str(mx["gross"][label_col]), str(mx["net_A"][label_col]), str(mx["net_B"][label_col])))
        rows.append(col(f"highest_loss_{name}_pnl_inr", lambda s: round(float(mn[L[s]][f"{L[s]}_total_pnl_inr"]), 0)))
        rows.append(S(f"highest_loss_{name}", str(mn["gross"][label_col]), str(mn["net_A"][label_col]), str(mn["net_B"][label_col])))
    add_extremes(daily.rename(columns={"date": "date"}), "day", "date")
    add_extremes(monthly, "month", "month")
    add_extremes(quarterly, "quarter", "quarter")
    add_extremes(yearly, "year", "year")

    summary = pd.DataFrame(rows)

    # ════════════════════════ all_trades sheet ════════════════════════
    all_trades = T[["entry_date", "symbol", "category", "cat_a_full", "legs_filled", "leg_prices",
                    "n_legs", "avg_entry", "shares", "capital_deployed", "alloc_intended",
                    "exit_date", "exit_type", "exit_time", "exit_price", "cover_1500_open",
                    "long_pnl", "short_pnl", "combined_pnl", "long_cost_023", "long_cost_038",
                    "short_cost", "gross_pnl", "gross_ret", "netA_pnl", "netA_ret",
                    "netB_pnl", "netB_ret", "quarter", "half_year", "year"]].copy()

    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        all_trades.to_excel(w, sheet_name="all_trades", index=False)
        daily.to_excel(w, sheet_name="daily_performance", index=False)
        monthly.to_excel(w, sheet_name="monthly_performance", index=False)
        quarterly.to_excel(w, sheet_name="quarterly_performance", index=False)
        halfy.to_excel(w, sheet_name="half_yearly_performance", index=False)
        yearly.to_excel(w, sheet_name="yearly_performance", index=False)
        for sht in w.sheets.values():
            for c in sht.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sht.column_dimensions[c[0].column_letter].width = min(width + 2, 32)

    # ── console recap ──
    pd.set_option("display.width", 240)
    print("\n=== TOP-LINE (combined long+short, fixed ₹5L base) ===")
    for lbl in ["total_return_fixedbase_pct", "total_pnl_inr", "win_rate_pct",
                "avg_return_per_trade_pct", "sharpe_ratio"]:
        r = summary[summary["metric"] == lbl].iloc[0]
        print(f"  {lbl:<32}gross={r['gross']}   net_A={r['net_A@0.23%+0.10%']}   net_B={r['net_B@0.38%+0.10%']}")
    print("\n=== LEG DECOMPOSITION (gross) ===")
    print(f"  long  sum_pnl={T['long_pnl'].sum():,.0f}  ({T['long_pnl'].sum()/BASE_POOL*100:+.2f}% base)")
    print(f"  short sum_pnl={T['short_pnl'].sum():,.0f}  ({T['short_pnl'].sum()/BASE_POOL*100:+.2f}% base)")
    print(f"  combined sum_pnl={T['combined_pnl'].sum():,.0f}  ({T['combined_pnl'].sum()/BASE_POOL*100:+.2f}% base)")
    print("\n=== CATEGORY FUNNEL ===")
    print(f"  A signals={n_A}  entered={int((T['category']=='A').sum())} (full {a_full}/half {a_half})  no-entry={a_noentry}")
    print(f"  B entered={int((T['category']=='B').sum())}   C entered={int((T['category']=='C').sum())}")
    print("\n=== YEARLY (combined, 3 series) ===")
    print(yearly[["year", "n_trades", "gross_total_return_fixedbase_pct",
                  "net_A_total_return_fixedbase_pct", "net_B_total_return_fixedbase_pct",
                  "gross_compounded_pnl_inr"]].to_string(index=False))
    print(f"\nSaved -> {XLSX}")


if __name__ == "__main__":
    main()
