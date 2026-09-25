# -*- coding: utf-8 -*-
"""expiry_day_premium_decay_buy.py — EXPLORATORY (not a validated strategy) premium-decay-buy test on
NIFTY/SENSEX expiry days within the CAS-era window (from 2026-08-03 onward, within the currently pulled
dataset). On each expiry day: ATM strike from the underlying's 15:19 candle OPEN (nearest standard strike
-- 50 for Nifty, 100 for Sensex, confirmed from actual traded strikes); 8 legs = ATM CE + 3 ITM CE
(ATM-1..3 strikes) + ATM PE + 3 ITM PE (ATM+1..3 strikes); each leg's own 15:19 OPEN = its reference
premium. From the 15:20 candle onward, each leg is watched independently for a LOW touching a trigger
level = reference_premium * (1 - threshold), for threshold in {80%, 85%, 90%, 95%} -- 4 separate,
independent runs. On touch, BUY 1 lot at exactly the trigger price (limit-style fill). No target/SL --
held to the contract's own last available candle CLOSE that day (proxy for settlement/expiry value,
flagged -- NOT the official exchange-computed settlement price, which is a separate 30-min-average
calculation this dataset cannot reproduce). P&L = settlement_proxy - entry_price, points per contract
(no lot-size/rupee conversion). Read-only diagnostic.
"""
import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "expiry_day_premium_decay_buy"; OUTDIR.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
REF_HM = 15 * 60 + 19          # 15:19 reference candle
ENTRY_START_HM = 15 * 60 + 20  # monitor window: 15:20 open through 15:21 close only
ENTRY_END_HM = 15 * 60 + 21

INDEXES = {
    "NIFTY": {
        "opt_dir": rb.BASE / "data" / "options_intraday_full" / "NIFTY",
        "spot_csv": rb.BASE / "data" / "nifty_1min_ohlc.csv",
        "interval": 50,
        "expiries": ["20260804", "20260811", "20260818", "20260825", "20260901", "20260908"],
    },
    "SENSEX": {
        "opt_dir": rb.BASE / "data" / "options_intraday_full" / "SENSEX",
        "spot_csv": rb.BASE / "data" / "sensex_1min_ohlc.csv",
        "interval": 100,
        "expiries": ["20260806", "20260813", "20260820", "20260827", "20260903"],
    },
}


def load_spot(csv_path):
    s = pd.read_csv(csv_path, parse_dates=["timestamp"])
    ts = s["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    s = s.assign(ts=ts, date=ts.dt.normalize(), mod=ts.dt.hour * 60 + ts.dt.minute)
    return s


def confirm_interval(opt_dir, exp_folder, expected):
    files = glob.glob(str(opt_dir / exp_folder / "*.parquet"))
    strikes = sorted(set(int(re.search(r"_(\d+)_(CE|PE)_", Path(f).name).group(1)) for f in files))
    diffs = np.diff(strikes)
    actual = int(np.median(diffs)) if len(diffs) else None
    return actual, strikes


def find_leg_file(opt_dir, exp_folder, index_name, strike, otype):
    pattern = str(opt_dir / exp_folder / f"{index_name}_{strike}_{otype}_*.parquet")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def run_index(index_name, cfg, entry_start_hm, entry_end_hm):
    spot = load_spot(cfg["spot_csv"])
    rows_log = []   # day-by-day per-leg-per-threshold log
    missing = []

    for exp_folder in cfg["expiries"]:
        exp_date = pd.Timestamp(exp_folder)
        actual_int, strikes_avail = confirm_interval(cfg["opt_dir"], exp_folder, cfg["interval"])
        interval = actual_int or cfg["interval"]

        day_spot = spot[spot["date"] == exp_date]
        ref_row = day_spot[day_spot["mod"] == REF_HM]
        if ref_row.empty:
            missing.append((index_name, exp_folder, "no 15:19 spot candle")); continue
        spot_1519_open = float(ref_row["open"].iloc[0])
        atm = round(spot_1519_open / interval) * interval

        ce_strikes = [atm - k * interval for k in range(0, 4)]   # ATM, ITM1, ITM2, ITM3 (calls)
        pe_strikes = [atm + k * interval for k in range(0, 4)]   # ATM, ITM1, ITM2, ITM3 (puts)
        legs = [(s, "CE", "ATM" if k == 0 else f"ITM{k}") for k, s in enumerate(ce_strikes)] + \
               [(s, "PE", "ATM" if k == 0 else f"ITM{k}") for k, s in enumerate(pe_strikes)]

        for strike, otype, tag in legs:
            fn = find_leg_file(cfg["opt_dir"], exp_folder, index_name, strike, otype)
            if fn is None:
                missing.append((index_name, exp_folder, f"missing contract file {strike} {otype}")); continue
            df = pd.read_parquet(fn)
            df = df[df["timestamp"].dt.normalize() == exp_date].sort_values("timestamp")
            if df.empty:
                missing.append((index_name, exp_folder, f"no candles on expiry day {strike} {otype}")); continue
            df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            ref = df[df["mod"] == REF_HM]
            if ref.empty:
                missing.append((index_name, exp_folder, f"no 15:19 candle {strike} {otype}")); continue
            ref_prem = float(ref["open"].iloc[0])
            settlement = float(df["close"].iloc[-1]); settlement_time = df["timestamp"].iloc[-1]

            window = df[(df["mod"] >= entry_start_hm) & (df["mod"] <= entry_end_hm)]
            for thr in THRESHOLDS:
                trigger_px = ref_prem * (1 - thr)
                hit = window[window["low"] <= trigger_px]
                if hit.empty:
                    rows_log.append({"index": index_name, "expiry": exp_folder, "strike": strike, "type": otype,
                                      "tag": tag, "threshold_pct": int(thr * 100), "ref_premium_1519": ref_prem,
                                      "trigger_price": round(trigger_px, 2), "triggered": False,
                                      "entry_time": None, "entry_price": None, "settlement": settlement,
                                      "settlement_time": settlement_time, "pnl": None})
                    continue
                entry_row = hit.iloc[0]
                entry_time = entry_row["timestamp"]; entry_price = trigger_px   # limit-style fill exactly at trigger
                pnl = settlement - entry_price
                rows_log.append({"index": index_name, "expiry": exp_folder, "strike": strike, "type": otype,
                                  "tag": tag, "threshold_pct": int(thr * 100), "ref_premium_1519": ref_prem,
                                  "trigger_price": round(trigger_px, 2), "triggered": True,
                                  "entry_time": entry_time, "entry_price": round(entry_price, 2),
                                  "settlement": settlement, "settlement_time": settlement_time,
                                  "pnl": round(pnl, 2)})

        print(f"{index_name} {exp_folder}: ATM={atm} (spot 15:19 open={spot_1519_open}, interval={interval}) processed", flush=True)

    LOG = pd.DataFrame(rows_log)
    return LOG, missing


def summarize(LOG, index_name):
    trig = LOG[(LOG["index"] == index_name) & (LOG["triggered"] == True)]
    n_expiries = LOG[LOG["index"] == index_name]["expiry"].nunique()
    rows = []
    for thr in [80, 85, 90, 95]:
        g = trig[trig["threshold_pct"] == thr]
        for side, gg in [("ALL", g), ("CE", g[g["type"] == "CE"]), ("PE", g[g["type"] == "PE"])]:
            n = len(gg)
            rows.append({
                "index": index_name, "threshold_pct": thr, "side": side, "n_expiry_days": n_expiries,
                "n_entries": n, "total_pnl": round(gg["pnl"].sum(), 2) if n else 0.0,
                "avg_pnl": round(gg["pnl"].mean(), 2) if n else None,
                "median_pnl": round(gg["pnl"].median(), 2) if n else None,
                "win_rate_pct": round((gg["pnl"] > 0).mean() * 100, 1) if n else None,
                "n_wins": int((gg["pnl"] > 0).sum()) if n else 0,
                "n_losses": int((gg["pnl"] <= 0).sum()) if n else 0,
            })
    return pd.DataFrame(rows)


WINDOW_CONFIGS = {
    "1min_1520only": (15 * 60 + 20, 15 * 60 + 20),
    "2min_1520to1521": (15 * 60 + 20, 15 * 60 + 21),
}


def main():
    all_windows_summ = {}
    all_windows_log = {}
    all_missing = []

    for wname, (start_hm, end_hm) in WINDOW_CONFIGS.items():
        all_logs = []; all_summaries = []
        for index_name, cfg in INDEXES.items():
            LOG, missing = run_index(index_name, cfg, start_hm, end_hm)
            all_logs.append(LOG); all_missing += [(*m, wname) for m in missing]
            all_summaries.append(summarize(LOG, index_name))
        FULL_LOG = pd.concat(all_logs, ignore_index=True); FULL_LOG["window"] = wname
        FULL_SUMM = pd.concat(all_summaries, ignore_index=True); FULL_SUMM["window"] = wname
        all_windows_log[wname] = FULL_LOG
        all_windows_summ[wname] = FULL_SUMM

    LOG_ALL = pd.concat(all_windows_log.values(), ignore_index=True)
    SUMM_ALL = pd.concat(all_windows_summ.values(), ignore_index=True)
    MISSING = pd.DataFrame(all_missing, columns=["index", "expiry", "issue", "window"])

    pd.set_option("display.width", 260)
    print(f"\n{'='*110}\nCOMPARISON: 1-minute (15:20 only) vs 2-minute (15:20-15:21) entry window, ALL side\n{'='*110}")
    cmp_all = SUMM_ALL[SUMM_ALL["side"] == "ALL"].pivot_table(
        index=["index", "threshold_pct"], columns="window",
        values=["n_entries", "total_pnl", "avg_pnl", "win_rate_pct"])
    cmp_all = cmp_all.reindex(columns=["n_entries", "avg_pnl", "total_pnl", "win_rate_pct"], level=0)
    print(cmp_all.to_string())

    for index_name in INDEXES:
        print(f"\n--- {index_name} CE vs PE, both windows ---")
        s2 = SUMM_ALL[(SUMM_ALL["index"] == index_name) & (SUMM_ALL["side"] != "ALL")]
        print(s2[["window", "threshold_pct", "side", "n_entries", "total_pnl", "avg_pnl", "win_rate_pct"]]
              .sort_values(["window", "threshold_pct", "side"]).to_string(index=False))

    if len(MISSING):
        print(f"\nFLAGGED (missing data): {len(MISSING)} rows")
        print(MISSING.to_string(index=False))

    with pd.ExcelWriter(OUTDIR / "expiry_day_premium_decay_buy_window_comparison.xlsx", engine="openpyxl") as w:
        SUMM_ALL.to_excel(w, sheet_name="Summary_by_Window_Threshold", index=False)
        cmp_flat = cmp_all.copy(); cmp_flat.columns = ["_".join(c) for c in cmp_flat.columns]
        cmp_flat.reset_index().to_excel(w, sheet_name="Comparison_Table_ALL", index=False)
        LOG_ALL.to_excel(w, sheet_name="Day_by_Day_Leg_Log", index=False)
        LOG_ALL[LOG_ALL["triggered"] == True].to_excel(w, sheet_name="Triggered_Entries_Only", index=False)
        if len(MISSING):
            MISSING.to_excel(w, sheet_name="Missing_Flagged", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 28)

    print(f"\nSaved -> {OUTDIR}/expiry_day_premium_decay_buy_window_comparison.xlsx")


if __name__ == "__main__":
    main()
