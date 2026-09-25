# -*- coding: utf-8 -*-
"""cas_auction_manipulation_sweep.py — full parameter sweep testing the "CAS auction manipulation
exploit" hypothesis on NIFTY/SENSEX expiry days (CAS era, Aug-2026+, currently pulled data only).
EXPLORATORY, read-only, no changes to any locked strategy file.

REFERENCE PRICE (per expiry day): PREFERRED = synthetic underlying via put-call parity at the 15:19
candle -- candidate ATM from the underlying spot's own 15:19 OPEN (round to nearest strike interval),
fetch that candidate strike's CE/PE 15:19 OPEN premiums, synthetic = candidate_ATM + (CE - PE), final
ATM = round(synthetic / interval) * interval. FALLBACK (if candidate's CE/PE data missing/unusable at
15:19) = underlying spot's 15:15 CLOSE, ATM = round(that / interval) * interval. Method used is recorded
per expiry day.

STRIKES: ATM + ITM1/ITM2/ITM3 on both CE (ATM - k*interval) and PE (ATM + k*interval) sides, k=0,1,2,3
(k=0 = ATM, added per explicit user follow-up after the ATM-inclusion ambiguity was flagged in the first
run of this sweep). itm_level column: 0=ATM, 1/2/3=ITM1/2/3.

ENTRY: touch-based, LOW <= ref_premium*(1-crash_threshold) any time from 15:20 through the swept
entry-time-limit window (15:20-15:21 / -15:22 / -15:23 / -15:24 for limits 1-4 min). Fill = exactly the
trigger price (limit-style). ref_premium = that LEG's OWN 15:19 OPEN (not the synthetic/fallback index
reference, which is only used for ATM/strike selection).

TARGET: exit at entry_price * multiple (3x..10x) the first time HIGH touches it after entry, else hold to
that day's last available candle (settlement proxy), no stop-loss.

MODE A: 3 independent single-ITM-level backtests (ITM1 only / ITM2 only / ITM3 only, CE+PE each).
MODE B: 1 combined backtest pooling all 3 ITM levels' triggered legs together (6 legs/day).
Both modes reuse the SAME underlying triggered-leg computation -- Mode B is a level-pooled aggregation
of Mode A's per-leg results, not a separately re-run backtest.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "cas_auction_manipulation_sweep"; OUTDIR.mkdir(parents=True, exist_ok=True)
REF_HM = 15 * 60 + 19
ENTRY_START_HM = 15 * 60 + 20
ETL_LIST = [1, 2, 3, 4]          # entry-time-limit, minutes after 3:20pm
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TARGETS = list(range(3, 11))     # 3x .. 10x

INDEXES = {
    "NIFTY": {"opt_dir": rb.BASE / "data" / "options_intraday_full" / "NIFTY",
              "spot_csv": rb.BASE / "data" / "nifty_1min_ohlc.csv", "interval": 50,
              "expiries": ["20260804", "20260811", "20260818", "20260825", "20260901", "20260908"]},
    "SENSEX": {"opt_dir": rb.BASE / "data" / "options_intraday_full" / "SENSEX",
               "spot_csv": rb.BASE / "data" / "sensex_1min_ohlc.csv", "interval": 100,
               "expiries": ["20260806", "20260813", "20260820", "20260827", "20260903", "20260910"]},
}


def load_spot(csv_path):
    s = pd.read_csv(csv_path, parse_dates=["timestamp"])
    ts = s["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return s.assign(ts=ts, date=ts.dt.normalize(), mod=ts.dt.hour * 60 + ts.dt.minute)


def find_leg_file(opt_dir, exp_folder, index_name, strike, otype):
    m = glob.glob(str(opt_dir / exp_folder / f"{index_name}_{int(strike)}_{otype}_*.parquet"))
    return m[0] if m else None


def leg_1519_open(opt_dir, exp_folder, index_name, strike, otype, exp_date):
    fn = find_leg_file(opt_dir, exp_folder, index_name, strike, otype)
    if fn is None:
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "open"])
    df = df[df["timestamp"].dt.normalize() == exp_date]
    df = df[(df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) == REF_HM]
    return float(df["open"].iloc[0]) if len(df) else None


def determine_atm(index_name, cfg, exp_folder, exp_date, spot):
    interval = cfg["interval"]
    day_spot = spot[spot["date"] == exp_date]
    ref1519 = day_spot[day_spot["mod"] == REF_HM]
    ref1515 = day_spot[day_spot["mod"] == 15 * 60 + 15]
    if ref1519.empty:
        return None, None, "no_1519_spot_candle"
    spot_1519_open = float(ref1519["open"].iloc[0])
    candidate_atm = round(spot_1519_open / interval) * interval

    ce_open = leg_1519_open(cfg["opt_dir"], exp_folder, index_name, candidate_atm, "CE", exp_date)
    pe_open = leg_1519_open(cfg["opt_dir"], exp_folder, index_name, candidate_atm, "PE", exp_date)
    if ce_open is not None and pe_open is not None:
        synthetic = candidate_atm + (ce_open - pe_open)
        atm = round(synthetic / interval) * interval
        return atm, spot_1519_open, f"synthetic (candidate={int(candidate_atm)}, CE={ce_open}, PE={pe_open}, synth={round(synthetic,2)})"
    if ref1515.empty:
        return None, None, "no_1515_spot_candle_for_fallback"
    spot_1515_close = float(ref1515["close"].iloc[0])
    atm = round(spot_1515_close / interval) * interval
    return atm, spot_1519_open, f"fallback_15:15close (spot={spot_1515_close})"


def process_leg(fn, exp_date):
    """Returns per-threshold trigger info + settlement, using data from 15:19 through end of day."""
    df = pd.read_parquet(fn)
    df = df[df["timestamp"].dt.normalize() == exp_date].sort_values("timestamp").copy()
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    ref_row = df[df["mod"] == REF_HM]
    if ref_row.empty or df.empty:
        return None
    ref_premium = float(ref_row["open"].iloc[0])
    settlement = float(df["close"].iloc[-1]); settlement_time = df["timestamp"].iloc[-1]
    window = df[df["mod"] >= ENTRY_START_HM].reset_index(drop=True)

    per_threshold = {}
    for thr in THRESHOLDS:
        trigger_px = ref_premium * (1 - thr)
        hit = window[window["low"] <= trigger_px]
        if hit.empty:
            per_threshold[thr] = None
            continue
        first_idx = hit.index[0]
        offset_min = int(window.loc[first_idx, "mod"] - ENTRY_START_HM)
        entry_time = window.loc[first_idx, "timestamp"]
        # forward path for target search: everything AFTER entry_time
        fwd = df[df["timestamp"] > entry_time]
        per_threshold[thr] = {"trigger_price": trigger_px, "offset_min": offset_min,
                               "entry_time": entry_time, "fwd_high": fwd["high"].values,
                               "fwd_ts": fwd["timestamp"].values}
    return {"ref_premium": ref_premium, "settlement": settlement, "settlement_time": settlement_time,
            "per_threshold": per_threshold}


def main():
    all_rows = []       # full (leg, threshold, ETL, target) grid
    day_log = []         # per-leg, per-threshold trigger summary
    atm_log = []

    for index_name, cfg in INDEXES.items():
        spot = load_spot(cfg["spot_csv"])
        interval = cfg["interval"]
        for exp_folder in cfg["expiries"]:
            exp_date = pd.Timestamp(exp_folder)
            atm, spot_1519, method = determine_atm(index_name, cfg, exp_folder, exp_date, spot)
            atm_log.append({"index": index_name, "expiry": exp_folder, "atm": atm, "spot_1519_open": spot_1519, "method": method})
            if atm is None:
                print(f"SKIP {index_name} {exp_folder}: {method}", flush=True)
                continue
            print(f"{index_name} {exp_folder}: ATM={int(atm)} | {method}", flush=True)

            legs = [(atm - k * interval, "CE", k) for k in (0, 1, 2, 3)] + [(atm + k * interval, "PE", k) for k in (0, 1, 2, 3)]
            for strike, otype, itm_level in legs:
                fn = find_leg_file(cfg["opt_dir"], exp_folder, index_name, strike, otype)
                if fn is None:
                    day_log.append({"index": index_name, "expiry": exp_folder, "strike": strike, "type": otype,
                                     "itm_level": itm_level, "issue": "missing_contract_file"})
                    continue
                res = process_leg(fn, exp_date)
                if res is None:
                    day_log.append({"index": index_name, "expiry": exp_folder, "strike": strike, "type": otype,
                                     "itm_level": itm_level, "issue": "no_1519_candle_or_empty_day"})
                    continue

                for thr, info in res["per_threshold"].items():
                    if info is None:
                        day_log.append({"index": index_name, "expiry": exp_folder, "strike": strike, "type": otype,
                                         "itm_level": itm_level, "threshold_pct": int(thr * 100),
                                         "ref_premium": res["ref_premium"], "triggered": False})
                        continue
                    day_log.append({"index": index_name, "expiry": exp_folder, "strike": strike, "type": otype,
                                     "itm_level": itm_level, "threshold_pct": int(thr * 100),
                                     "ref_premium": res["ref_premium"], "triggered": True,
                                     "entry_time": info["entry_time"], "entry_price": round(info["trigger_price"], 2),
                                     "offset_min": info["offset_min"], "settlement": res["settlement"],
                                     "settlement_time": res["settlement_time"]})

                    entry_price = info["trigger_price"]; fwd_high = info["fwd_high"]; fwd_ts = info["fwd_ts"]
                    for tgt in TARGETS:
                        target_px = entry_price * tgt
                        hit_idx = np.where(fwd_high >= target_px)[0]
                        if len(hit_idx):
                            exit_price = target_px; exit_time = fwd_ts[hit_idx[0]]; hit_target = True
                        else:
                            exit_price = res["settlement"]; exit_time = res["settlement_time"]; hit_target = False
                        pnl = exit_price - entry_price
                        for etl in ETL_LIST:
                            if info["offset_min"] > etl:
                                continue
                            all_rows.append({"index": index_name, "expiry": exp_folder, "strike": strike, "type": otype,
                                              "itm_level": itm_level, "threshold_pct": int(thr * 100), "etl": etl,
                                              "target_mult": tgt, "entry_price": round(entry_price, 2),
                                              "exit_price": round(exit_price, 2), "hit_target": hit_target, "pnl": round(pnl, 2)})

    GRID = pd.DataFrame(all_rows)
    DAYLOG = pd.DataFrame(day_log)
    ATM = pd.DataFrame(atm_log)
    print(f"\ntotal grid rows: {len(GRID)}", flush=True)

    # ---- MODE A: per itm_level ----
    modeA = GRID.groupby(["index", "itm_level", "etl", "threshold_pct", "target_mult"]).agg(
        n=("pnl", "size"), total_pnl=("pnl", "sum"), avg_pnl=("pnl", "mean"),
        win_rate=("pnl", lambda s: round((s > 0).mean() * 100, 1))).reset_index()
    modeA["total_pnl"] = modeA["total_pnl"].round(2); modeA["avg_pnl"] = modeA["avg_pnl"].round(2)

    # ---- MODE B: pooled across itm_level ----
    modeB = GRID.groupby(["index", "etl", "threshold_pct", "target_mult"]).agg(
        n=("pnl", "size"), total_pnl=("pnl", "sum"), avg_pnl=("pnl", "mean"),
        win_rate=("pnl", lambda s: round((s > 0).mean() * 100, 1))).reset_index()
    modeB["total_pnl"] = modeB["total_pnl"].round(2); modeB["avg_pnl"] = modeB["avg_pnl"].round(2)

    pd.set_option("display.width", 220)
    print("\n=== ATM determination log ===")
    print(ATM.to_string(index=False))

    for idx in INDEXES:
        print(f"\n{'='*100}\n{idx} -- MODE A (per ITM level) -- BEST 10 by total_pnl\n{'='*100}")
        sub = modeA[modeA["index"] == idx].sort_values("total_pnl", ascending=False)
        print(sub.head(10).drop(columns="index").to_string(index=False))
        print(f"\n{idx} -- MODE B (combined) -- BEST 10 by total_pnl")
        subB = modeB[modeB["index"] == idx].sort_values("total_pnl", ascending=False)
        print(subB.head(10).drop(columns="index").to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "cas_auction_manipulation_sweep.xlsx", engine="openpyxl") as w:
        ATM.to_excel(w, sheet_name="ATM_Determination_Log", index=False)
        DAYLOG.to_excel(w, sheet_name="Day_By_Day_Trigger_Log", index=False)
        modeA.to_excel(w, sheet_name="ModeA_Full_Grid", index=False)
        modeB.to_excel(w, sheet_name="ModeB_Full_Grid", index=False)
        GRID.to_excel(w, sheet_name="Raw_Trade_Level_Grid", index=False)
        for idx in INDEXES:
            modeA[modeA["index"] == idx].sort_values("total_pnl", ascending=False).head(20).to_excel(
                w, sheet_name=f"{idx}_ModeA_Top20", index=False)
            modeB[modeB["index"] == idx].sort_values("total_pnl", ascending=False).head(20).to_excel(
                w, sheet_name=f"{idx}_ModeB_Top20", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 26)

    print(f"\nSaved -> {OUTDIR}/cas_auction_manipulation_sweep.xlsx")


if __name__ == "__main__":
    main()
