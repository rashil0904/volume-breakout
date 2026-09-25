# -*- coding: utf-8 -*-
"""
double_down_short_variant.py
============================
"Double-down" variant: at each LONG exit, open an equal-size SHORT at the long's exit
price and square it off at the 3:00pm candle OPEN on the SAME (exit) day. Base entry &
long exit unchanged (mcap ₹1,500-5,000 Cr, LB 36, VM 6, 3:15pm entry, +5%, 09:45/12:00
split + 14% target, ₹5L/₹1L). Reuses the canonical long trade set (same exit_types/prices);
does not recompute the long side.

Per trade (shares = long's whole-share count):
  long_pnl   = shares × (long_exit_price − entry)
  short_pnl  = shares × (long_exit_price − price_at_3pm_open)     # short profits if price falls
  combined   = long_pnl + short_pnl
  trade_return_pct = combined / capital_deployed × 100            # no extra margin modeled

Costs: 2 round-trips (long buy+sell, short sell+buyback). Net = combined − rate×long_notional
− rate×short_notional, at rate = 0.23% and 0.38%. Long notional = capital_deployed
(shares×entry); short notional = shares×long_exit_price. FLAT-RATE APPROX — real Indian
intraday short STT/charges differ from delivery; this is a first-order estimate only.

Flags: (a) short squared off at the 3:00pm candle OPEN (not close). (b) no margin modeled;
combined P&L over the long's capital_deployed / fixed ₹5L base. (c) short size = long shares.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import profit_target_sweep as pts
import final_performance_report as fpr

OUTDIR = rb.RESULTS / "double_down_short"
BASE_POOL, TARGET = 500_000, 14.0
HM_0945, HM_1200, HM_1500 = pts.HM_0945, pts.HM_1200, 900
R023, R038 = 0.0023, 0.0038


def build_long_with_ohlc():
    """Replicate build_trades' exit assignment but keep the aligned OHLC so the 3:00pm
    open (short square-off) is available. Returns a per-trade DataFrame."""
    base = ets.load_base_positions()
    e = base["entry"].values.astype(float)
    sh = base["shares"].values.astype(float)
    cap = base["cap"].values.astype(float)
    opens, highs, exit_days = fpr.fetch_ohlc_and_exitday(base)
    pct_high = (highs - e[:, None]) / e[:, None] * 100
    t1, t2 = HM_0945, HM_1200
    ot1, ot2 = opens[:, pts.HCOL[t1]], opens[:, pts.HCOL[t2]]
    ret_t1 = (ot1 - e) / e * 100
    o3pm = opens[:, pts.HCOL[HM_1500]]
    pre_hms = [hm for hm in pts.CANDLE_HMS if hm < t1]
    bet_hms = [hm for hm in pts.CANDLE_HMS if t1 < hm < t2]

    N = len(base)
    et = np.empty(N, dtype=object); hm = np.full(N, np.nan); px = np.full(N, np.nan)
    valid = np.zeros(N, dtype=bool)
    for i in range(N):
        ph = pct_high[i]; a = b = c = None
        for h in pre_hms:
            v = ph[pts.HCOL[h]]
            if not np.isnan(v) and v >= TARGET:
                a, b, c = "early_target_pre_t1", h, e[i] * (1 + TARGET / 100); break
        if a is None:
            if not np.isnan(ot1[i]) and ret_t1[i] > 0:
                a, b, c = "positive_at_t1", t1, ot1[i]
            else:
                for h in bet_hms:
                    v = ph[pts.HCOL[h]]
                    if not np.isnan(v) and v >= TARGET:
                        a, b, c = "early_target_between_t1_t2", h, e[i] * (1 + TARGET / 100); break
                if a is None and not np.isnan(ot2[i]):
                    a, b, c = "exit_at_t2_no_target", t2, ot2[i]
        if a is not None:
            et[i], hm[i], px[i], valid[i] = a, b, c, True

    vi = np.where(valid)[0]
    T = pd.DataFrame({
        "symbol": base["symbol"].values[vi],
        "entry_date": [pd.Timestamp(base["date"].values[i]).date() for i in vi],
        "exit_date": [exit_days[i] for i in vi],
        "exit_time": [f"{int(hm[i])//60:02d}:{int(hm[i])%60:02d}" for i in vi],
        "entry_price": e[vi], "shares": sh[vi].astype(int), "capital_deployed": cap[vi],
        "exit_type": et[vi], "exit_hm": hm[vi], "long_exit_price": px[vi],
        "price_3pm_open": o3pm[vi],
    })
    return T


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Building long trades + 3:00pm opens …")
    T = build_long_with_ohlc()
    print(f"  long trades: {len(T):,}")

    # validate long-only matches canonical
    T["long_pnl"] = T["shares"] * (T["long_exit_price"] - T["entry_price"])
    T["long_ret"] = (T["long_exit_price"] - T["entry_price"]) / T["entry_price"] * 100

    # edge cases: exit at/after 3pm (shouldn't happen) or missing 3pm open -> no short
    T["exit_after_3pm"] = T["exit_hm"] >= HM_1500
    T["no_3pm_open"] = T["price_3pm_open"].isna()
    no_short = T["exit_after_3pm"] | T["no_3pm_open"]
    n_flag = int(no_short.sum())

    # short leg: open at long_exit_price, square off at 3:00pm open (profit if price falls)
    T["short_pnl"] = np.where(no_short, 0.0,
                              T["shares"] * (T["long_exit_price"] - T["price_3pm_open"]))
    T["short_ret"] = np.where(no_short, 0.0,
                              (T["long_exit_price"] - T["price_3pm_open"]) / T["entry_price"] * 100)
    T["combined_pnl"] = T["long_pnl"] + T["short_pnl"]
    T["combined_ret"] = T["combined_pnl"] / T["capital_deployed"] * 100

    # costs: 2 round-trips (rate × long notional + rate × short notional)
    long_notl = T["capital_deployed"].values
    short_notl = np.where(no_short, 0.0, (T["shares"] * T["long_exit_price"]).values)
    for r, tag in [(R023, "023"), (R038, "038")]:
        exp = r * (long_notl + short_notl)
        T[f"net{tag}_combined_pnl"] = T["combined_pnl"] - exp
        T[f"net{tag}_combined_ret"] = T[f"net{tag}_combined_pnl"] / T["capital_deployed"] * 100

    # ── 1. comparison LONG-ONLY vs DOUBLE-DOWN ──
    def blk(pnl, ret, label):
        n = len(pnl)
        return {"strategy": label, "n_trades": n,
                "total_return_fixedbase_pct": round(pnl.sum() / BASE_POOL * 100, 4),
                "total_pnl_inr": round(pnl.sum(), 0),
                "win_rate_pct": round((pnl > 0).mean() * 100, 2),
                "avg_return_per_trade_pct": round(ret.mean(), 4),
                "median_return_per_trade_pct": round(ret.median(), 4)}
    # long-only net (1 round-trip) for fair baseline
    long_net023 = T["long_pnl"] - R023 * T["capital_deployed"]
    long_net038 = T["long_pnl"] - R038 * T["capital_deployed"]
    comp = pd.DataFrame([
        {**blk(T["long_pnl"], T["long_ret"], "LONG_ONLY (gross)")},
        {**blk(long_net023, T["long_ret"] - R023*100, "LONG_ONLY (net@0.23% 1x)")},
        {**blk(long_net038, T["long_ret"] - R038*100, "LONG_ONLY (net@0.38% 1x)")},
        {**blk(T["combined_pnl"], T["combined_ret"], "DOUBLE_DOWN (gross)")},
        {**blk(T["net023_combined_pnl"], T["net023_combined_ret"], "DOUBLE_DOWN (net@0.23% 2x)")},
        {**blk(T["net038_combined_pnl"], T["net038_combined_ret"], "DOUBLE_DOWN (net@0.38% 2x)")},
    ])

    # ── 2. leg decomposition ──
    shorted = ~no_short
    sp = T.loc[shorted, "short_pnl"]; sr = T.loc[shorted, "short_ret"]
    leg = pd.DataFrame([
        {"leg": "long", "total_pnl_inr": round(T["long_pnl"].sum(), 0),
         "total_return_fixedbase_pct": round(T["long_pnl"].sum()/BASE_POOL*100, 4),
         "win_rate_pct": round((T["long_pnl"] > 0).mean()*100, 2),
         "avg_return_per_trade_pct": round(T["long_ret"].mean(), 4)},
        {"leg": "short", "total_pnl_inr": round(sp.sum(), 0),
         "total_return_fixedbase_pct": round(sp.sum()/BASE_POOL*100, 4),
         "win_rate_pct": round((sp > 0).mean()*100, 2),
         "avg_return_per_trade_pct": round(sr.mean(), 4)},
        {"leg": "combined", "total_pnl_inr": round(T["combined_pnl"].sum(), 0),
         "total_return_fixedbase_pct": round(T["combined_pnl"].sum()/BASE_POOL*100, 4),
         "win_rate_pct": round((T["combined_pnl"] > 0).mean()*100, 2),
         "avg_return_per_trade_pct": round(T["combined_ret"].mean(), 4)},
    ])

    # ── 3. short-leg behavior by exit type ──
    st_rows = []
    for etype, g in T[shorted].groupby("exit_type"):
        st_rows.append({"exit_type": etype, "n_trades": len(g),
                        "short_win_rate_pct": round((g["short_pnl"] > 0).mean()*100, 2),
                        "short_avg_return_pct": round(g["short_ret"].mean(), 4),
                        "short_median_return_pct": round(g["short_ret"].median(), 4),
                        "short_total_pnl_inr": round(g["short_pnl"].sum(), 0),
                        "avg_move_exit_to_3pm_pct": round(
                            ((g["price_3pm_open"] - g["long_exit_price"]) / g["long_exit_price"] * 100).mean(), 4)})
    short_by_exit = pd.DataFrame(st_rows)

    with pd.ExcelWriter(OUTDIR / "double_down_short_variant.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="comparison", index=False)
        leg.to_excel(w, sheet_name="leg_decomposition", index=False)
        short_by_exit.to_excel(w, sheet_name="short_by_exit_type", index=False)
        T.to_excel(w, sheet_name="trade_level", index=False)

    pd.set_option("display.width", 220)
    print(f"\nTrades with NO short (exit>=3pm or missing 3pm open): {n_flag}")
    print("\n" + "=" * 120 + "\nCOMPARISON — LONG-ONLY vs DOUBLE-DOWN\n" + "=" * 120)
    print(comp.to_string(index=False))
    print("\n--- LEG DECOMPOSITION ---"); print(leg.to_string(index=False))
    print("\n--- SHORT-LEG BEHAVIOR BY EXIT TYPE (avg_move = mean % change exit->3pm) ---")
    print(short_by_exit.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
