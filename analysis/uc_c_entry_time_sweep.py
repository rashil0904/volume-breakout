# -*- coding: utf-8 -*-
"""
uc_c_entry_time_sweep.py
========================
Sweeps ONLY Category C's entry candle from 15:15 → 15:25 (11 one-min opens) in the
composite UC + double-down strategy. Category A (UC pullback ladder) and Category B
(UC 3:00-3:15) are UNCHANGED — their timing is set by UC mechanics, not a chosen clock.

Confirmed interpretation (flags):
  (a) The C signal / qualification is UNCHANGED — the stock still had to qualify exactly as
      before (mcap, 6x volume, +5% vs VWAP-close, evaluated as usual). Only the FILL candle
      moves from 15:15 to T. No re-qualification at T.
  (b) Capital sequencing: A/B intraday commitments (Phase 1) and A second-leg top-ups
      (Phase 2) still resolve by 15:15, so X and therefore each day's per-C allocation are
      FIXED across the sweep. Moving C to T changes only the PRICE C pays (and thus shares,
      exit path, pnl) — not how much capital C receives.
  (c) Grid = 15:15..15:25 inclusive, 1-min steps (11 times).
  (d) A and B entries/pnl are computed once and held constant across the sweep.

Method: run the engine once (scan_symbol + size_and_price) to get A/B trades (fixed) and the
per-day per_C allocation; cache each C signal's 15:15..15:25 entry-day opens and its next-day
exit windows; then for each T re-slice C's fill price, re-derive the exit (14% target on 1-min
highs / 09:45 / 12:00), the double-down short (cover 15:00 next day) and costs.
"""
import sys, bisect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import uc_staggered_dd_report as R

OUTDIR = rb.RESULTS / "uc_c_entry_time_sweep"
GRID = list(range(R.HM_1515, R.HM_1515 + 11))       # 915..925 = 15:15..15:25
IST = R.IST


def hm_lbl(h):
    return f"{h // 60:02d}:{h % 60:02d}"


def capture_c_windows(c_recs):
    """Per C signal (symbol, entry_date): entry-day opens at 15:15..15:25 and next-day exit
    windows (o945, o1200, o1500, pre-0945 highs, 0945-1200 highs). One parquet read per symbol."""
    cap = {}
    by_sym = {}
    for r in c_recs:
        by_sym.setdefault(r["symbol"], []).append(r)
    for si, (sym, recs) in enumerate(by_sym.items(), 1):
        pq = rb.MASTER_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
        by_date = {d: g for d, g in raw.groupby("date")}
        for r in recs:
            ed, nd = r["entry_date"], r["next_date"]
            if ed not in by_date or nd is None or nd not in by_date:
                continue
            eg = by_date[ed]
            eo = dict(zip(eg["hm"].values, eg["open"].values.astype(float)))
            ng = by_date[nd]
            nhm = ng["hm"].values; nhi = ng["high"].values.astype(float)
            nop = dict(zip(nhm, ng["open"].values.astype(float)))
            pre = (nhm >= R.HM_915) & (nhm < R.HM_945)
            bet = (nhm > R.HM_945) & (nhm < R.HM_1200)
            cap[(sym, ed)] = {
                "opens": {t: eo.get(t, np.nan) for t in GRID},
                "o945": nop.get(R.HM_945, np.nan), "o1200": nop.get(R.HM_1200, np.nan),
                "o1500": nop.get(R.HM_1500, np.nan),
                "pre_hm": nhm[pre], "pre_hi": nhi[pre],
                "bet_hm": nhm[bet], "bet_hi": nhi[bet],
            }
        if si % 300 == 0:
            print(f"  …cached {si}/{len(by_sym)} C symbols", flush=True)
    return cap


def c_trade_at(price, per_c, w):
    """Compute one Category C trade filled at `price` with allocation per_c. Returns dict of
    gross/netA/netB pnl + ret + win, or None if unfillable/no valid exit."""
    if not (price == price and price > 0):
        return None
    shares = float(np.floor(per_c / price))
    if shares <= 0:
        return None
    cap_dep = shares * price
    tgt = price * (1 + R.TARGET / 100)
    o945, o1200, o1500 = w["o945"], w["o1200"], w["o1500"]
    # exit path (unchanged logic; 14% target scanned on 1-min highs)
    exit_price = None
    hit = w["pre_hi"] >= tgt
    if hit.any():
        exit_price = tgt
    elif o945 == o945 and o945 > price:
        exit_price = o945
    else:
        hit2 = w["bet_hi"] >= tgt
        if hit2.any():
            exit_price = tgt
        elif o1200 == o1200:
            exit_price = o1200
    if exit_price is None:
        return None
    long_pnl = shares * (exit_price - price)
    has_short = o1500 == o1500
    short_pnl = shares * (exit_price - o1500) if has_short else 0.0
    short_notl = shares * exit_price if has_short else 0.0
    combined = long_pnl + short_pnl
    lc023, lc038 = R.LONG_023 * cap_dep, R.LONG_038 * cap_dep
    sc = R.SHORT_RATE * short_notl
    return {
        "gross": combined, "netA": combined - lc023 - sc, "netB": combined - lc038 - sc,
        "cap": cap_dep,
    }


def metrics(pnls, caps, label):
    pnls = np.asarray(pnls); caps = np.asarray(caps)
    rets = pnls / caps * 100
    return {
        f"{label}_n": len(pnls),
        f"{label}_win_rate_pct": round(float((pnls > 0).mean() * 100), 2) if len(pnls) else 0.0,
        f"{label}_avg_return_pct": round(float(rets.mean()), 4) if len(pnls) else 0.0,
        f"{label}_median_return_pct": round(float(np.median(rets)), 4) if len(pnls) else 0.0,
        f"{label}_total_return_fixedbase_pct": round(float(pnls.sum() / R.BASE_POOL * 100), 4),
        f"{label}_total_pnl_inr": round(float(pnls.sum()), 0),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "passes_all_three",
                                "prev_day_vwap_close", "entry_price_315pm"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    Q = diag[diag["passes_all_three"] == True].reset_index(drop=True)
    print(f"qualifying signals: {len(Q):,}")

    print("Scanning (engine) for A/B + per-day C allocation …")
    records = []
    for sym in sorted(Q["symbol"].unique()):
        records += R.scan_symbol(sym, Q[Q["symbol"] == sym])
    T_def, day_diag = R.size_and_price(records)
    per_C_of = {d: v["per_C"] for d, v in day_diag.items()}

    AB = T_def[T_def["category"].isin(["A", "B"])].copy()
    ab = {s: AB[f"{s}_pnl"].values for s in ["gross", "netA", "netB"]}
    ab_cap = AB["capital_deployed"].values
    ab_ret = {s: AB[f"{s}_ret"].values for s in ["gross", "netA", "netB"]}
    print(f"  fixed A/B trades: {len(AB):,}  (A={int((AB['category']=='A').sum())}, "
          f"B={int((AB['category']=='B').sum())})")

    c_recs = [r for r in records if r["category"] == "C" and r["entered"]
              and per_C_of.get(r["entry_date"], 0) > 0]
    print(f"Caching 15:15-15:25 windows for {len(c_recs):,} Category-C signals …")
    cap = capture_c_windows(c_recs)

    rows = []
    curves = {"T": [], "C_avg_netA": [], "overall_total_netA": []}
    for T in GRID:
        c_g, c_a, c_b, c_cap = [], [], [], []
        n_unfill = 0
        for r in c_recs:
            key = (r["symbol"], r["entry_date"])
            w = cap.get(key)
            if w is None:
                continue
            price = w["opens"].get(T, np.nan)
            res = c_trade_at(price, per_C_of[r["entry_date"]], w)
            if res is None:
                n_unfill += 1; continue
            c_g.append(res["gross"]); c_a.append(res["netA"]); c_b.append(res["netB"]); c_cap.append(res["cap"])
        cm = {}
        for s, arr in [("gross", c_g), ("netA", c_a), ("netB", c_b)]:
            cm.update(metrics(arr, c_cap, f"C_{s}"))
        # overall = fixed A/B + C at T
        ov = {}
        for s, cpnl in [("gross", c_g), ("netA", c_a), ("netB", c_b)]:
            allp = np.concatenate([ab[s], np.asarray(cpnl)])
            allc = np.concatenate([ab_cap, np.asarray(c_cap)])
            allret = np.concatenate([ab_ret[s], np.asarray(cpnl) / np.asarray(c_cap) * 100]) if len(cpnl) else ab_ret[s]
            ov[f"overall_{s}_total_return_fixedbase_pct"] = round(float(allp.sum() / R.BASE_POOL * 100), 4)
            ov[f"overall_{s}_win_rate_pct"] = round(float((allp > 0).mean() * 100), 2)
            ov[f"overall_{s}_avg_return_pct"] = round(float(allret.mean()), 4)
            ov[f"overall_{s}_median_return_pct"] = round(float(np.median(allret)), 4)
        row = {"C_entry_time": hm_lbl(T), "n_C_unfillable": n_unfill, **cm, **ov}
        rows.append(row)
        curves["T"].append(hm_lbl(T))
        curves["C_avg_netA"].append(cm["C_netA_avg_return_pct"])
        curves["overall_total_netA"].append(ov["overall_netA_total_return_fixedbase_pct"])
        print(f"  {hm_lbl(T)}: C n={cm['C_gross_n']} C_avg_ret(netA)={cm['C_netA_avg_return_pct']:+.4f}%  "
              f"overall total(netA)={ov['overall_netA_total_return_fixedbase_pct']:.2f}%")

    sweep = pd.DataFrame(rows)
    by_c = sweep.sort_values("C_netA_avg_return_pct", ascending=False).reset_index(drop=True)
    by_ov = sweep.sort_values("overall_netA_total_return_fixedbase_pct", ascending=False).reset_index(drop=True)

    with pd.ExcelWriter(OUTDIR / "uc_c_entry_time_sweep.xlsx", engine="openpyxl") as w:
        sweep.to_excel(w, sheet_name="by_time", index=False)
        by_c.to_excel(w, sheet_name="ranked_by_C_avg_return", index=False)
        by_ov.to_excel(w, sheet_name="ranked_by_overall_return", index=False)
        for sht in w.sheets.values():
            for c in sht.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sht.column_dimensions[c[0].column_letter].width = min(width + 2, 34)

    # ── chart ──
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = range(len(curves["T"]))
    ax1.plot(x, curves["C_avg_netA"], "-o", color="#2E74B5", label="Category C avg return/trade (net_A)")
    ax1.set_xlabel("Category C entry candle"); ax1.set_ylabel("Cat-C avg return per trade % (net_A)", color="#2E74B5")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(curves["T"], rotation=45)
    ax1.axhline(curves["C_avg_netA"][0], color="#2E74B5", ls=":", alpha=0.5)
    ax1.tick_params(axis="y", labelcolor="#2E74B5"); ax1.grid(axis="y", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, curves["overall_total_netA"], "-s", color="#C55A11", label="Overall total return (net_A)")
    ax2.set_ylabel("Overall total_return_fixedbase % (net_A)", color="#C55A11")
    ax2.tick_params(axis="y", labelcolor="#C55A11")
    ax1.set_title("Category C entry-time sweep (15:15 → 15:25) — A/B fixed", fontweight="bold")
    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(OUTDIR / "uc_c_entry_time_sweep.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 240)
    show = ["C_entry_time", "C_gross_n", "C_netA_win_rate_pct", "C_netA_avg_return_pct",
            "C_netA_median_return_pct", "C_netA_total_return_fixedbase_pct",
            "overall_netA_total_return_fixedbase_pct"]
    print("\n" + "=" * 100 + "\nCATEGORY-C ENTRY-TIME SWEEP (net_A shown; full gross/net_A/net_B in Excel)\n" + "=" * 100)
    print(sweep[show].to_string(index=False))
    b1 = by_c.iloc[0]; b2 = by_ov.iloc[0]; base = sweep.iloc[0]
    print(f"\nBaseline 15:15 : C avg(netA)={base['C_netA_avg_return_pct']:+.4f}%  "
          f"overall total(netA)={base['overall_netA_total_return_fixedbase_pct']:.2f}%")
    print(f"BEST for Cat C : {b1['C_entry_time']}  C avg(netA)={b1['C_netA_avg_return_pct']:+.4f}%")
    print(f"BEST overall   : {b2['C_entry_time']}  overall total(netA)={b2['overall_netA_total_return_fixedbase_pct']:.2f}%"
          + ("   (same candle)" if b1["C_entry_time"] == b2["C_entry_time"] else "   (differs from best-for-C)"))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
