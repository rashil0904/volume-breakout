# -*- coding: utf-8 -*-
"""audit_fullchain_banknifty_options.py — READ-ONLY completeness/gap audit of the RAW BankNifty options
MONTHLY-ONLY pull (data/options_intraday_full/BANKNIFTY). Mirrors audit_fullchain_sensex.py: (1) structural -
all 22 monthly expiries present, nothing dropped vs API, CE/PE both sides per strike; (2) per-contract - null
fields, OHLC validity, duplicate timestamps, DTE recompute, expiry-day row present, whole missing trading days
within active window; (3) expiry weekday cross-check vs the derived monthly regimes; (4) spot-vs-strike sanity.
No modify/re-pull.
"""
import sys, time, glob, json
from pathlib import Path
from datetime import datetime
from collections import Counter
import requests, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb, data_loading as dl

ROOT = rb.BASE / "data" / "options_intraday_full" / "BANKNIFTY"
MAN = ROOT.parent / "manifest_banknifty_opt.csv"; ISS = ROOT.parent / "issues_banknifty_opt.csv"
OUTDIR = rb.RESULTS / "fullchain_audit_banknifty_opt"; PARTS = OUTDIR / "parts"; PARTS.mkdir(parents=True, exist_ok=True)
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
UND = "NSE_INDEX|Nifty Bank"; FLOOR, CUTOFF = "2024-10-01", "2026-07-31"
H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]; CUT_D = pd.Timestamp(CUTOFF).date()
BASE = "https://api.upstox.com/v2/expired-instruments"; enc = lambda k: k.replace("|", "%7C")
MONTHLY = ["2024-10-30", "2024-11-27", "2024-12-24", "2025-01-30", "2025-02-27", "2025-03-27", "2025-04-24",
           "2025-05-29", "2025-06-26", "2025-07-31", "2025-08-28", "2025-09-30", "2025-10-28", "2025-11-25",
           "2025-12-30", "2026-01-27", "2026-02-24", "2026-03-30", "2026-04-28", "2026-05-26", "2026-06-30", "2026-07-28"]


def get(u):
    for a in range(4):
        try: r = requests.get(u, headers=H, timeout=30)
        except Exception: time.sleep(1.2 * (a + 1)); continue
        if r.status_code == 200:
            d = r.json().get("data", []); return d.get("candles", []) if isinstance(d, dict) else d
        if r.status_code in (429, 500, 502, 503): time.sleep(1.2 * (a + 1)); continue
        return None
    return None


def audit_expiry(exp, files, tdarr):
    X = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ts = X["timestamp"].values.astype("datetime64[m]"); date = ts.astype("datetime64[D]")
    X["date"] = date
    exp_dte = (np.datetime64(exp) - date).astype("timedelta64[D]").astype("int64")
    X["null"] = X[["open", "high", "low", "close", "volume", "OI", "DTE"]].isna().any(axis=1)
    X["ohlc_bad"] = (X.high < X.low) | (X.open < X.low) | (X.open > X.high) | (X.close < X.low) | (X.close > X.high)
    X["dte_mis"] = X["DTE"].values != exp_dte
    g = X.groupby("symbol", sort=False)
    per = g.agg(nnull=("null", "sum"), nbad=("ohlc_bad", "sum"), ndte=("dte_mis", "sum"),
                min_dte=("DTE", "min"), strike=("strike", "first"), typ=("option_type", "first"), nrows=("close", "size"))
    per["ndup"] = X.duplicated(["symbol", "timestamp"]).groupby(X["symbol"], sort=False).sum()
    dates_by = g["date"].apply(lambda s: set(pd.to_datetime(s).dt.date))
    fl = g["date"].agg(["min", "max"]); expd = pd.Timestamp(exp).date(); rows = []
    for sym in per.index:
        p = per.loc[sym]; fd = pd.Timestamp(fl.loc[sym, "min"]).date(); ld = pd.Timestamp(fl.loc[sym, "max"]).date()
        window = [d for d in tdarr if fd <= d <= min(expd, CUT_D) and d <= ld]
        miss = sorted(set(window) - dates_by.loc[sym]); fx = []
        if p.nnull: fx.append(("null_fields", int(p.nnull)))
        if p.nbad: fx.append(("ohlc_violation", int(p.nbad)))
        if p.ndup: fx.append(("duplicate_timestamp", int(p.ndup)))
        if p.ndte: fx.append(("dte_mismatch", int(p.ndte)))
        if p.min_dte != 0: fx.append(("no_expiry_day_row", int(p.min_dte)))
        if miss: fx.append(("missing_trading_days", len(miss)))
        for it, det in fx:
            rows.append({"symbol": sym, "expiry": exp, "strike": p.strike, "type": p.typ, "issue": it, "detail": str(det)})
    ce = set(per[per.typ == "CE"].strike); pe = set(per[per.typ == "PE"].strike); struct = []
    for s in sorted(ce - pe): struct.append({"expiry": exp, "issue": "one_sided_CE_only", "detail": f"strike {s}"})
    for s in sorted(pe - ce): struct.append({"expiry": exp, "issue": "one_sided_PE_only", "detail": f"strike {s}"})
    cnt = {"contracts": int(len(per)), "flagged": int(len(set(r["symbol"] for r in rows))), "rows": int(per.nrows.sum()),
           "zv_total": int((X.volume == 0).sum()), "struct": struct}
    return rows, cnt


def main():
    t0 = time.time()
    d = pd.read_parquet(DAILY, columns=["date"]); tdarr = np.array(sorted(d["date"].dt.date.unique()))
    exp_dirs = sorted([x for x in ROOT.iterdir() if x.is_dir()])
    present = [f"{x.name[:4]}-{x.name[4:6]}-{x.name[6:]}" for x in exp_dirs if any(x.glob("*.parquet"))]
    struct = [{"expiry": e, "issue": "missing_monthly_expiry", "detail": "0 contracts"} for e in MONTHLY if e not in present]
    try:
        emp = pd.read_csv(ISS); emp = emp[emp.issue == "empty_no_candles"].drop_duplicates(["expiry", "symbol"]).groupby("expiry").size().to_dict()
    except Exception: emp = {}
    print(f"structural: {len(MONTHLY)} MONTHLY expiries expected | present {len(present)} | missing {len(struct)}", flush=True)
    for ei, x in enumerate(exp_dirs, 1):
        e = f"{x.name[:4]}-{x.name[4:6]}-{x.name[6:]}"; part = PARTS / f"{x.name}.json"
        if e not in MONTHLY:
            print(f"  !! UNEXPECTED expiry dir not in monthly list: {e}", flush=True)
        if part.exists(): continue
        files = sorted(glob.glob(str(x / "*.parquet")))
        if not files:
            json.dump({"rows": [], "cnt": {"contracts": 0, "flagged": 0, "rows": 0, "zv_total": 0, "struct": []}}, open(part, "w")); continue
        cons = get(f"{BASE}/option/contract?instrument_key={enc(UND)}&expiry_date={e}") or []
        api_n = len(cons); disk_n = len(files); e_n = emp.get(e, 0); extra = []
        if disk_n + e_n < api_n:
            extra.append({"expiry": e, "issue": "pulled_lt_api", "detail": f"disk {disk_n}+empty {e_n} < api {api_n} (missing {api_n-disk_n-e_n})"})
        rows, cnt = audit_expiry(e, files, tdarr); cnt["struct"] += extra
        json.dump({"rows": rows, "cnt": cnt}, open(part, "w"))
        print(f"  [{ei}/{len(exp_dirs)}] {e} ({WD[datetime.strptime(e,'%Y-%m-%d').weekday()]}): {cnt['contracts']} contracts, {len(rows)} flags, {cnt['rows']:,} rows | {time.time()-t0:.0f}s", flush=True)

    allrows = []; C = Counter(); struct_all = list(struct)
    for p in sorted(PARTS.glob("*.json")):
        j = json.load(open(p)); allrows += j["rows"]; c = j["cnt"]
        for k in ("contracts", "rows", "zv_total"): C[k] += c.get(k, 0)
        struct_all += c.get("struct", [])
    I = pd.DataFrame(allrows); S = pd.DataFrame(struct_all)
    by = I["issue"].value_counts().to_dict() if len(I) else {}
    dows = [(e, WD[datetime.strptime(e, "%Y-%m-%d").weekday()]) for e in present]
    nd = get(f"https://api.upstox.com/v2/historical-candle/{enc(UND)}/day/2026-07-31/2024-06-01") or []
    ndf = pd.DataFrame(nd, columns=["ts", "o", "h", "l", "c", "v", "oi"]); ndf["date"] = pd.to_datetime(ndf.ts).dt.tz_localize(None).dt.date
    ncl = dict(zip(ndf.date, ndf.c)); man = pd.read_csv(MAN); spot = []
    if ncl:
        for e, gg in man.groupby("expiry"):
            ed = pd.Timestamp(e).date(); sp = ncl.get(ed) or ncl.get(min(ncl, key=lambda dd: abs((dd - ed).days)))
            smin, smax = gg.strike.min(), gg.strike.max()
            if not (smin <= sp <= smax): spot.append({"expiry": e, "spot": round(sp), "strike_min": smin, "strike_max": smax, "issue": "spot_outside_strike_range"})

    summ = {"total_contracts": C["contracts"], "contracts_flagged": I["symbol"].nunique() if len(I) else 0,
            "contracts_zero_issues": C["contracts"] - (I["symbol"].nunique() if len(I) else 0),
            "total_rows(raw)": C["rows"], "zero_volume_rows_kept": C["zv_total"],
            "monthly_expiries_present": len(present), "monthly_expiries_missing": len(struct),
            "flag_missing_trading_days": by.get("missing_trading_days", 0), "flag_no_expiry_day_row": by.get("no_expiry_day_row", 0),
            "flag_ohlc_violation": by.get("ohlc_violation", 0), "flag_null_fields": by.get("null_fields", 0),
            "flag_dte_mismatch": by.get("dte_mismatch", 0), "flag_duplicate_timestamp": by.get("duplicate_timestamp", 0),
            "one_sided_strikes": int(S.issue.str.startswith("one_sided").sum()) if len(S) else 0,
            "pulled_lt_api_expiries": int((S.issue == "pulled_lt_api").sum()) if len(S) else 0,
            "spot_strike_anomalies": len(spot)}
    with pd.ExcelWriter(OUTDIR / "fullchain_audit_banknifty_opt.xlsx", engine="openpyxl") as w:
        pd.DataFrame(list(summ.items()), columns=["metric", "value"]).to_excel(w, sheet_name="summary", index=False)
        (S if len(S) else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="structural", index=False)
        (I if len(I) else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="contract_issues", index=False)
        pd.DataFrame([{"expiry": e, "weekday": w} for e, w in dows]).to_excel(w, sheet_name="expiry_weekdays", index=False)
        (pd.DataFrame(spot) if spot else pd.DataFrame([{"note": "all strike ranges bracket spot"}])).to_excel(w, sheet_name="spot_vs_strike", index=False)
    if len(I): I.to_csv(OUTDIR / "flagged_contracts_banknifty_opt.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 90 + "\nBANKNIFTY OPTIONS (MONTHLY-ONLY) — COMPLETENESS/GAP AUDIT (raw pull)\n" + "=" * 90)
    for k, v in summ.items(): print(f"  {k:34s}: {v:,}" if isinstance(v, int) else f"  {k:34s}: {v}")
    print("\n--- structural issues ---"); print(S.to_string(index=False) if len(S) else "  NONE")
    print("\n--- contract issue counts ---"); print(I["issue"].value_counts().to_string() if len(I) else "  NONE")
    print(f"\nSaved -> {OUTDIR} | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
