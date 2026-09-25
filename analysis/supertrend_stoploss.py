# -*- coding: utf-8 -*-
"""supertrend_stoploss.py — 13% stop-loss overlay on Supertrend(10,5) long-only (VWAP-close) vs pure
Supertrend-flip exit. Entry = bull flip -> next open. Config1: ST-flip exit only. Config2: 13% SL
(entry*0.87, first daily LOW breach, fill at stop, SL precedence same-day) OR ST bear flip (next open),
whichever first. Traces stopped-out trades' would-have-been Config-1 outcome (opportunity cost).
mcap>1500, cost 0.23%.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import supertrend_10_5_final as ST

DAILY = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "supertrend_stoploss"
COST, MCAP_MIN, SL_PCT = 0.23, 1500.0, 0.13
POOL, PER, MAXPOS = 1_000_000, 100_000, 10
IS_YEARS = {2022, 2023, 2024}


def portfolio(T):
    open_pos = []; cash = POOL; realized = 0.0; taken = 0
    for r in T.sort_values("entry_dt").itertuples():
        keep = []
        for x, nt in open_pos:
            if x <= r.entry_dt:
                cash += PER * (1 + nt / 100); realized += PER * nt / 100
            else:
                keep.append((x, nt))
        open_pos = keep
        if len(open_pos) < MAXPOS and cash >= PER:
            cash -= PER; open_pos.append((r.exit_dt, r.net_return_pct)); taken += 1
    for x, nt in open_pos:
        realized += PER * nt / 100
    return round(realized / POOL * 100, 1), taken


def metrics(T, label):
    r = T["net_return_pct"]; rg = T["return_pct"]; win = r > 0
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    eq = T.sort_values("exit_dt")["net_return_pct"].cumsum().values
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    port, _ = portfolio(T)
    return {"config": label, "n_trades": len(T), "win_rate_pct": round(win.mean() * 100, 2),
            "avg_ret_gross_pct": round(rg.mean(), 3), "avg_ret_net_pct": round(r.mean(), 3),
            "median_ret_net_pct": round(r.median(), 3), "avg_holding_days": round(T["holding_days"].mean(), 1),
            "profit_factor_net": round(gp / gl, 3) if gl > 0 else np.inf, "expectancy_net_pct": round(r.mean(), 3),
            "avg_winner_pct": round(r[win].mean(), 3) if win.any() else 0.0,
            "avg_loser_pct": round(r[~win].mean(), 3) if (~win).any() else 0.0,
            "worst_trade_pct": round(r.min(), 2), "total_return_sum_net_pct": round(r.sum(), 1),
            "portfolio_return_on_10L_pct": port, "max_dd_tradeseq_net_pct": round(dd, 1)}


def build():
    df = pd.read_parquet(DAILY); df["date"] = pd.to_datetime(df["date"])
    snap_dates, snap_dicts = ST.load_mcap()
    def mcap_of(sym, d64):
        idx = int(np.searchsorted(snap_dates, d64, "right")) - 1
        return snap_dicts[max(idx, 0)].get(sym, np.nan)
    t1, t2 = [], []; n_sameday = 0; n_gap = 0
    for sym, s in df.groupby("symbol", sort=False):
        if len(s) < 30:
            continue
        s = s.sort_values("date")
        o = s["open"].values.astype(float); h = s["high"].values.astype(float)
        lo = s["low"].values.astype(float); c = s["close"].values.astype(float); dt = s["date"].values
        d, st, atr = ST.supertrend_full(h, lo, c, 10, 5.0)
        n = len(s); i = 11
        while i < n:
            if d[i] == 1 and d[i - 1] == -1:
                if i + 1 >= n:
                    break
                ei = i + 1
                j = ei
                while j < n and not (d[j] == -1 and d[j - 1] == 1):
                    j += 1
                is_open_st = j >= n
                xi_st = min(j + 1, n - 1) if not is_open_st else n - 1
                mc = mcap_of(sym, dt[i])
                if mc == mc and mc > MCAP_MIN and o[ei] > 0:
                    entry = o[ei]; stop = entry * (1 - SL_PCT)
                    base = {"symbol": sym, "entry_date": pd.Timestamp(dt[ei]).strftime("%Y-%m-%d"),
                            "entry_price": round(entry, 2), "year": pd.Timestamp(dt[ei]).year}
                    # CONFIG 1: ST-only
                    ep1 = o[xi_st] if not is_open_st else c[xi_st]
                    r1 = (ep1 - entry) / entry * 100
                    t1.append({**base, "exit_date": pd.Timestamp(dt[xi_st]).strftime("%Y-%m-%d"), "exit_price": round(ep1, 2),
                               "exit_reason": "supertrend_flip" if not is_open_st else "open_end", "holding_days": int(xi_st - ei),
                               "return_pct": round(r1, 3), "net_return_pct": round(r1 - COST, 3)})
                    # CONFIG 2: 13% SL vs ST, whichever first (scan SL from ei to xi_st)
                    k = -1
                    for kk in range(ei, xi_st + 1):
                        if lo[kk] <= stop:
                            k = kk; break
                    if k >= 0:                                  # SL fires (<= xi_st so at/before ST exit)
                        if k == j and not is_open_st:           # SL day == ST bear-flip signal day
                            n_sameday += 1
                        if o[k] < stop:                         # gapped through stop (fill optimistic)
                            n_gap += 1
                        ep2 = stop; reason = "stop_loss_13"; xi2 = k
                        r2 = -SL_PCT * 100                      # exactly -13% gross
                        would_win_c1 = bool(r1 > 0)
                    else:
                        ep2 = ep1; reason = "supertrend_flip" if not is_open_st else "open_end"; xi2 = xi_st
                        r2 = r1; would_win_c1 = False
                    t2.append({**base, "exit_date": pd.Timestamp(dt[xi2]).strftime("%Y-%m-%d"), "exit_price": round(ep2, 2),
                               "exit_reason": reason, "holding_days": int(xi2 - ei), "return_pct": round(r2, 3),
                               "net_return_pct": round(r2 - COST, 3), "c1_return_pct": round(r1, 3),
                               "stopped_would_win_c1": would_win_c1})
                i = j if j < n else n
            else:
                i += 1
    T1 = pd.DataFrame(t1); T2 = pd.DataFrame(t2)
    for T in (T1, T2):
        T["entry_dt"] = pd.to_datetime(T["entry_date"]); T["exit_dt"] = pd.to_datetime(T["exit_date"])
    return T1, T2, n_sameday, n_gap


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T1, T2, n_sameday, n_gap = build()
    print(f"entries: C1={len(T1):,} C2={len(T2):,} | same-day SL+STflip: {n_sameday} | SL gapped-through-stop: {n_gap}")

    comp = pd.DataFrame([metrics(T1, "1_supertrend_only"), metrics(T2, "2_with_13pct_SL")])
    delta = {"config": "delta (2-1)"}
    for c in comp.columns:
        if c != "config":
            delta[c] = round(comp.iloc[1][c] - comp.iloc[0][c], 4)
    comp = pd.concat([comp, pd.DataFrame([delta])], ignore_index=True)

    # exit-reason split C2
    er = []
    for reason, g in T2.groupby("exit_reason"):
        r = g["net_return_pct"]
        er.append({"exit_reason": reason, "n": len(g), "pct": round(len(g) / len(T2) * 100, 1),
                   "avg_net_ret": round(r.mean(), 3), "avg_holding_days": round(g["holding_days"].mean(), 1)})
    ER = pd.DataFrame(er).sort_values("n", ascending=False)

    # opportunity cost: stopped-out that would've won under C1
    stopped = T2[T2["exit_reason"] == "stop_loss_13"]
    would_win = stopped[stopped["stopped_would_win_c1"]]
    opp = {"n_stopped_out": len(stopped),
           "n_stopped_would_have_won_c1": int(stopped["stopped_would_win_c1"].sum()),
           "pct_of_stops_that_wouldve_won": round(stopped["stopped_would_win_c1"].mean() * 100, 1) if len(stopped) else 0.0,
           "avg_c1_return_of_wouldve_won": round(would_win["c1_return_pct"].mean(), 2) if len(would_win) else 0.0,
           "total_c1_return_given_up_by_stops": round((stopped["c1_return_pct"] - stopped["return_pct"]).sum(), 1)}

    # per-year + IS/OOS
    py = []
    for lbl, dd0 in [("no_SL", T1), ("SL13", T2)]:
        for y, g in dd0.groupby("year"):
            r = g["net_return_pct"]; gl = -r[r < 0].sum()
            py.append({"config": lbl, "year": y, "n": len(g), "avg_net": round(r.mean(), 3),
                       "total_net": round(r.sum(), 1), "worst": round(r.min(), 1),
                       "profit_factor": round(r[r > 0].sum() / gl, 3) if gl > 0 else np.inf})
    PY = pd.DataFrame(py)
    def blk(df):
        r = df["net_return_pct"]; gl = -r[r < 0].sum()
        return {"n": len(df), "avg_net": round(r.mean(), 3), "total_net": round(r.sum(), 1),
                "worst": round(r.min(), 1), "profit_factor": round(r[r > 0].sum() / gl, 3) if gl > 0 else np.inf}
    oos = []
    for lbl, dd0 in [("no_SL", T1), ("SL13", T2)]:
        oos.append({"config": lbl, "period": "IS 2022-24", **blk(dd0[dd0["year"].isin(IS_YEARS)])})
        oos.append({"config": lbl, "period": "OOS 2025+", **blk(dd0[~dd0["year"].isin(IS_YEARS)])})
    OOS = pd.DataFrame(oos)

    with pd.ExcelWriter(OUTDIR / "supertrend_stoploss.xlsx", engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="headline_compare", index=False)
        ER.to_excel(w, sheet_name="exit_reason_split_C2", index=False)
        pd.DataFrame([opp]).to_excel(w, sheet_name="opportunity_cost", index=False)
        PY.to_excel(w, sheet_name="per_year", index=False)
        OOS.to_excel(w, sheet_name="IS_OOS", index=False)
        T2.drop(columns=["entry_dt", "exit_dt"]).to_excel(w, sheet_name="SL13_all_trades", index=False)

    # distribution plot
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(np.clip(T1["net_return_pct"], -40, 120), bins=90, alpha=0.55, label="no SL (ST flip only)", color="#C0504D")
    ax.hist(np.clip(T2["net_return_pct"], -40, 120), bins=90, alpha=0.55, label="13% SL", color="#4472C4")
    ax.axvline(-13.23, color="k", ls=":", lw=1, label="SL level (-13.23% net)")
    ax.set_title("Supertrend(10,5): return distribution — 13% SL truncates the left tail"); ax.legend(); ax.set_xlabel("net return %")
    fig.tight_layout(); fig.savefig(OUTDIR / "return_distribution.png", dpi=120); plt.close(fig)

    pd.set_option("display.width", 250)
    print("\n" + "=" * 100 + "\n13% STOP-LOSS vs SUPERTREND-ONLY EXIT (net 0.23%)\n" + "=" * 100)
    show = ["config", "n_trades", "win_rate_pct", "avg_ret_net_pct", "median_ret_net_pct", "avg_holding_days",
            "profit_factor_net", "avg_winner_pct", "avg_loser_pct", "worst_trade_pct", "total_return_sum_net_pct",
            "portfolio_return_on_10L_pct", "max_dd_tradeseq_net_pct"]
    print(comp[show].to_string(index=False))
    print("\n--- CONFIG 2 EXIT-REASON SPLIT ---"); print(ER.to_string(index=False))
    print("\n--- OPPORTUNITY COST (stopped-out that would've won under C1) ---")
    for k, v in opp.items():
        print(f"  {k:38s}: {v}")
    print("\n--- PER YEAR ---")
    print(PY.pivot(index="year", columns="config", values=["avg_net", "total_net", "worst", "profit_factor"]).to_string())
    print("\n--- IS / OOS ---"); print(OOS.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
