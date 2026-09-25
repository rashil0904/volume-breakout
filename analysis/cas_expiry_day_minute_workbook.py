# -*- coding: utf-8 -*-
"""cas_expiry_day_minute_workbook.py — builds a single Excel workbook, one sheet per NIFTY/SENSEX expiry
day (CAS-era, from 2026-08-03 onward, within the currently pulled datasets), showing 1-min data from
15:14 to 15:30. Each sheet: underlying spot OHLC per minute + the 8 legs used in the premium-decay-buy
CAS study (ATM CE/PE + 3 ITM strikes each side, per that day's own 15:19 spot open, interval 50 for NIFTY
/ 100 for SENSEX), each leg's own OHLC per minute, columns labeled with actual strike+type. Read-only,
extracts from already-pulled data -- no new pull.
"""
import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUT = rb.RESULTS / "cas_expiry_day_minute_data.xlsx"
REF_HM = 15 * 60 + 19
WINDOW_START_HM = 15 * 60 + 14
WINDOW_END_HM = 15 * 60 + 30

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


def find_leg_file(opt_dir, exp_folder, index_name, strike, otype):
    matches = glob.glob(str(opt_dir / exp_folder / f"{index_name}_{strike}_{otype}_*.parquet"))
    return matches[0] if matches else None


def leg_window(fn, exp_date):
    df = pd.read_parquet(fn)
    df = df[df["timestamp"].dt.normalize() == exp_date].copy()
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(df["mod"] >= WINDOW_START_HM) & (df["mod"] <= WINDOW_END_HM)]
    return df.set_index("mod")[["open", "high", "low", "close"]]


def build_sheet(index_name, cfg, exp_folder):
    exp_date = pd.Timestamp(exp_folder)
    spot = load_spot(cfg["spot_csv"])
    interval = cfg["interval"]

    day_spot = spot[spot["date"] == exp_date]
    ref_row = day_spot[day_spot["mod"] == REF_HM]
    if ref_row.empty:
        return None, f"{index_name} {exp_folder}: no 15:19 spot candle -- skipped"
    spot_1519_open = float(ref_row["open"].iloc[0])
    atm = round(spot_1519_open / interval) * interval

    ce_strikes = [atm - k * interval for k in range(0, 4)]   # ATM + ITM1-3
    pe_strikes = [atm + k * interval for k in range(0, 4)]   # ATM + ITM1-3
    legs = [(s, "CE", "ATM" if k == 0 else f"ITM{k}") for k, s in enumerate(ce_strikes)] + \
           [(s, "PE", "ATM" if k == 0 else f"ITM{k}") for k, s in enumerate(pe_strikes)]

    # ---- time grid 15:14 -> 15:30 ----
    grid = list(range(WINDOW_START_HM, WINDOW_END_HM + 1))
    out = pd.DataFrame({"mod": grid})
    out["Time"] = out["mod"].apply(lambda m: f"{m//60:02d}:{m%60:02d}")

    spot_win = day_spot[(day_spot["mod"] >= WINDOW_START_HM) & (day_spot["mod"] <= WINDOW_END_HM)].set_index("mod")[["open", "high", "low", "close"]]
    spot_win = spot_win.reindex(grid)
    out["SPOT_Open"] = spot_win["open"].values
    out["SPOT_High"] = spot_win["high"].values
    out["SPOT_Low"] = spot_win["low"].values
    out["SPOT_Close"] = spot_win["close"].values

    missing_legs = []
    for strike, otype, tag in legs:
        fn = find_leg_file(cfg["opt_dir"], exp_folder, index_name, strike, otype)
        label = f"{int(strike)}_{otype}"
        if fn is None:
            out[f"{label}_Open"] = np.nan; out[f"{label}_High"] = np.nan
            out[f"{label}_Low"] = np.nan; out[f"{label}_Close"] = np.nan
            missing_legs.append(label); continue
        lw = leg_window(fn, exp_date).reindex(grid)
        out[f"{label}_Open"] = lw["open"].values
        out[f"{label}_High"] = lw["high"].values
        out[f"{label}_Low"] = lw["low"].values
        out[f"{label}_Close"] = lw["close"].values

    out = out.drop(columns=["mod"])
    meta = f"ATM={int(atm)} (spot 15:19 open={spot_1519_open}, interval={interval})"
    if missing_legs:
        meta += f" | MISSING contract files: {', '.join(missing_legs)}"
    return out, meta


def main():
    sheets = {}
    issues = []
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        for index_name, cfg in INDEXES.items():
            for exp_folder in cfg["expiries"]:
                sheet_df, meta = build_sheet(index_name, cfg, exp_folder)
                exp_date_str = pd.Timestamp(exp_folder).strftime("%Y-%m-%d")
                sheet_name = f"{index_name}_{exp_date_str}"[:31]
                if sheet_df is None:
                    issues.append(meta)
                    print(f"SKIP {sheet_name}: {meta}")
                    continue
                sheet_df.to_excel(w, sheet_name=sheet_name, index=False, startrow=1)
                ws = w.sheets[sheet_name]
                ws.cell(row=1, column=1, value=meta)
                print(f"{sheet_name}: {meta}")
        for sh in w.sheets.values():
            for c in sh.columns:
                vals = [x.value for x in c if x.value is not None]
                width = max((len(str(v)) for v in vals), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 16)

    print(f"\nSaved -> {OUT}")
    if issues:
        print(f"\n{len(issues)} skipped sheet(s):")
        for i in issues:
            print(" ", i)


if __name__ == "__main__":
    main()
