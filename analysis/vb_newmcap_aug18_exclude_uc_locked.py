# -*- coding: utf-8 -*-
"""vb_newmcap_aug18_exclude_uc_locked.py — remove UC-LOCKED trades that could not actually have been entered from
the Aug-18+ baseline backtest (new daily mcap source), then re-run the LOCKED engine (baseline_and_cross_final.
run_config("baseline"), untouched) on the remaining signals. New file; read-only vs locked files.

"UC-locked / could not be entered" (strict-locked proxy, generalised from entry_uc_locked_analysis.py, which only
handled the 20% band):
  For the modeled FILL candle (Category C and A-leg2 = the 15:21 candle; Category B = the UC-touch candle):
    C / A-leg2 15:21 fill:  candle FLAT (range <= 0.2% of price, project FLAT_TOL)
                            AND fill price = the day's high so far (pinned at the top)
                            AND price within 1% of a 5% / 10% / 20% circuit level off the prior plain close
                            -> only buyers, no sellers: the modeled fill price was not actually obtainable.
    B touch fill:           touch candle never traded at/below the modeled fill (uc*0.999) i.e. low > fill
  Calibrated against the live broker outcomes in data/trades (not_filled) and checked for false positives on
  every live-FILLED trade.

Primary result = flagged signals removed BEFORE capital sequencing (same treatment the baseline already gives its own
locked_321 exclusions: the pool is re-split among the remaining names). Sensitivity = flagged trades simply dropped
from the original run (other positions keep their original sizes, as live sizing is fixed at ~15:10).
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC
import vb_baseline_newmcap_aug18 as V

TR = rb.BASE / "data" / "trades" / "trades"
OUT = V.OUT
FLAT_TOL = 0.002
BAND_TOL = 0.01
BANDS = (1.05, 1.10, 1.20)


def bars(sym, folder):
    raw = pd.read_parquet(V.DIRS[folder] / f"{sym}.parquet", columns=["timestamp", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(V.IST)
    return raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)


def flag_locked(T, S_folder, pc_map):
    """Return DataFrame of per-trade lock diagnostics + boolean uc_locked."""
    rows = []; cache = {}
    for r in T.itertuples():
        if r.symbol not in cache:
            cache[r.symbol] = bars(r.symbol, S_folder[r.symbol])
        raw = cache[r.symbol]
        days = sorted(raw["date"].unique()); i = days.index(r.entry_date)
        prev_close = raw[raw["date"] == days[i - 1]].sort_values("hm")["close"].iloc[-1]
        b = raw[raw["date"] == r.entry_date].set_index("hm").sort_index()
        locked, why = False, ""
        a_leg_at_1521 = False
        if r.category == "A":                       # only when a leg price really IS the 15:21 open (not a 19%/17% pullback limit fill)
            lps = [float(x) for x in str(r.leg_prices).split("+")]
            a_leg_at_1521 = any(abs(p - b.loc[BC.HM_1521, "open"]) < 1e-6 for p in lps)
        if r.category == "C" or a_leg_at_1521:
            c = b.loc[BC.HM_1521]
            flat = (c["high"] - c["low"]) / max(c["close"], 1e-9) <= FLAT_TOL
            at_high = c["open"] >= b.loc[:BC.HM_1521, "high"].max() - 1e-9
            ratio = c["open"] / prev_close
            near_band = min(abs(ratio - x) for x in BANDS) <= BAND_TOL
            locked = bool(flat and at_high and near_band)
            why = f"15:21 flat={flat} at_day_high={at_high} ratio_vs_prev_close={ratio:.4f} near_band={near_band} vol={int(c['volume'])}"
        elif r.category == "B":
            pc = pc_map[(r.symbol, r.entry_date)]; uc = pc * BC.UC_M; fill = uc * 0.999
            tm = int(b[(b.index >= BC.HM_1500) & (b.index < BC.HM_1521) & (b["high"] >= uc)].index.min())
            c = b.loc[tm]
            locked = bool(c["low"] > fill + 1e-9)
            why = f"B touch {tm//60:02d}:{tm%60:02d} candle low={c['low']} fill={fill:.2f} low>fill={locked} vol={int(c['volume'])}"
        rows.append({"symbol": r.symbol, "entry_date": r.entry_date, "category": r.category, "uc_locked": locked, "lock_detail": why})
    return pd.DataFrame(rows)


def main():
    S = pd.read_parquet(V.SCAN_FN); S["date"] = pd.to_datetime(S["date"]).dt.date
    new_mcap = V.load_new_mcap()
    folder = S.drop_duplicates("symbol").set_index("symbol")["folder"].to_dict()
    pc_map = {(s, d): p for s, d, p in zip(S["symbol"], S["date"], S["pc"])}

    Tn = pd.read_csv(OUT / "trades_NEW_daily_mcap.csv"); Tn["entry_date"] = pd.to_datetime(Tn["entry_date"]).dt.date
    F = flag_locked(Tn, folder, pc_map)
    Tn = Tn.merge(F[["symbol", "entry_date", "uc_locked", "lock_detail"]], on=["symbol", "entry_date"])
    locked = Tn[Tn["uc_locked"]]
    print(f"modeled trades: {len(Tn)} | flagged UC-locked (could not be entered): {len(locked)}", flush=True)
    print(locked[["entry_date", "symbol", "category", "shares", "avg_entry", "gross_pnl", "netA_pnl", "lock_detail"]].to_string(index=False), flush=True)

    # ---- validation vs live broker outcomes ----
    L = pd.concat([pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[11:21]).date()) for f in sorted(glob.glob(str(TR / "trade_list_*.csv")))])
    Bk = pd.concat([pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[13:23]).date()) for f in sorted(glob.glob(str(TR / "dhan_entries_*.csv")))]
                   + [pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[12:22]).date()) for f in sorted(glob.glob(str(TR / "mtf_entries_*.csv")))])
    st = Bk[["date", "symbol", "status"]].drop_duplicates(["date", "symbol"])
    Tn = Tn.merge(st, left_on=["entry_date", "symbol"], right_on=["date", "symbol"], how="left").drop(columns="date")
    Tn["live_status"] = Tn["status"].fillna("no_broker_row").where(Tn["status"].notna(), "no_broker_row")
    print("\n=== validation vs live broker status (all modeled trades) ===")
    print(pd.crosstab(Tn["uc_locked"], Tn["live_status"]).to_string(), flush=True)

    # ---- primary: remove flagged signals BEFORE sequencing, re-run the locked engine ----
    cand = S[(S["date"] >= V.START) & S["passes_vol"] & S["passes_ret"]].copy()
    cand["in_new"] = [(s in set(new_mcap[d]["symbol"])) if d in new_mcap else False for s, d in zip(cand["symbol"], cand["date"])]
    sig = cand[cand["in_new"]][["symbol", "folder", "date", "pc"]].reset_index(drop=True)
    drop_keys = set(zip(locked["symbol"], locked["entry_date"]))
    sig2 = sig[[(s, d) not in drop_keys for s, d in zip(sig["symbol"], sig["date"])]].reset_index(drop=True)
    cache, extra = V.build_cache(sig2)
    T2, _ = BC.run_config("baseline", cache)
    T2["entry_date"] = pd.to_datetime(T2["entry_date"]).dt.date
    T2.to_csv(OUT / "trades_NEW_daily_mcap_ex_UC_locked.csv", index=False)

    M0 = V.metrics(Tn, "ORIGINAL (117 trades)")
    M1 = V.metrics(T2, "EX-UC-LOCKED, re-sequenced")
    M2 = V.metrics(Tn[~Tn["uc_locked"]], "EX-UC-LOCKED, simple drop")
    R = pd.DataFrame([M0, M1, M2]).set_index("run")
    keys = ["n_trades", "gross_total_inr", "gross_return_pct_of_5L", "gross_win_rate_pct", "gross_avg_ret_per_trade_pct", "gross_max_dd_inr", "gross_max_dd_pct_of_5L", "gross_avg_dd_pct_of_5L",
            "netA_total_inr", "netA_return_pct_of_5L", "netA_win_rate_pct", "netA_avg_ret_per_trade_pct", "netA_max_dd_inr", "netA_max_dd_pct_of_5L", "netA_avg_dd_pct_of_5L",
            "netB_total_inr", "netB_return_pct_of_5L", "netB_win_rate_pct", "netB_avg_ret_per_trade_pct", "netB_max_dd_inr", "netB_max_dd_pct_of_5L", "netB_avg_dd_pct_of_5L",
            "catA_n", "catA_gross_inr", "catB_n", "catB_gross_inr", "catC_n", "catC_gross_inr", "catC_netA_inr", "catC_netB_inr", "long_leg_gross_inr", "short_leg_gross_inr"]
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
    print("\n=== METRICS ===")
    print(R[keys].T.to_string(), flush=True)

    with pd.ExcelWriter(OUT / "vb_newmcap_aug18_ex_UC_locked.xlsx", engine="openpyxl") as w:
        R[keys].T.reset_index().to_excel(w, sheet_name="metrics", index=False)
        locked.to_excel(w, sheet_name="eliminated_UC_locked", index=False)
        Tn.to_excel(w, sheet_name="original_trades_flagged", index=False)
        T2.to_excel(w, sheet_name="trades_ex_UC_locked", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {OUT}/vb_newmcap_aug18_ex_UC_locked.xlsx", flush=True)


if __name__ == "__main__":
    main()
