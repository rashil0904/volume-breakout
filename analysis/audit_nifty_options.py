# -*- coding: utf-8 -*-
"""audit_nifty_options.py — READ-ONLY completeness audit of data/options_intraday_full/NIFTY. Processes
per-expiry in vectorized batches (resumable: skips an expiry whose part file exists). Checks: structural
coverage (Get-Expiries vs present; CE/PE both sides per strike); per-contract completeness (missing whole
trading days; INTERNAL intraday gaps = missing minutes between the day's first & last candle — the true
API-gap signal, since zero-volume minutes are otherwise filled; last candle before expiry); field sanity
(nulls, DTE==calendar recompute, OHLC validity, dup timestamps); expiry weekday cross-check. Report only.
"""
import sys, time, glob, json
from pathlib import Path
from datetime import datetime
from collections import Counter
import requests, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb, data_loading as dl

ROOT = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
MAN = ROOT.parent / "manifest_nifty.csv"
OUTDIR = rb.RESULTS / "options_audit"; PARTS = OUTDIR / "parts"; PARTS.mkdir(parents=True, exist_ok=True)
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
NIFTY = "NSE_INDEX|Nifty 50"; FLOOR, CUTOFF = "2024-10-01", "2026-07-31"
H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CUT_D = pd.Timestamp(CUTOFF).date()


def api_expiries():
    try:
        r = requests.get(f"https://api.upstox.com/v2/expired-instruments/expiries?instrument_key={NIFTY.replace('|','%7C')}", headers=H, timeout=30)
        return sorted([e for e in r.json().get("data", []) if FLOOR <= e <= CUTOFF])
    except Exception:
        return None


def audit_expiry(exp, files, tdarr):
    frames = []; corrupt = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            corrupt.append(Path(f).stem.replace("_", " "))
    corrupt_rows = [{"symbol": s, "expiry": exp, "strike": "", "type": "", "issue": "corrupted_parquet_file", "detail": "unreadable (truncated/mid-write)"} for s in corrupt]
    if not frames:
        return corrupt_rows, {"contracts": len(files), "clean": 0, "oi_zero": 0, "nrows": 0}
    X = pd.concat(frames, ignore_index=True)
    ts = X["timestamp"].values.astype("datetime64[m]")
    date = ts.astype("datetime64[D]"); minute = (ts.astype("int64") % 1440)
    exp_dte = (np.datetime64(exp) - date).astype("timedelta64[D]").astype("int64")
    X["date"] = date; X["minute"] = minute
    X["null"] = X[["open", "high", "low", "close", "volume", "OI", "DTE"]].isna().any(axis=1)
    X["ohlc_bad"] = (X.high < X.low) | (X.open < X.low) | (X.open > X.high) | (X.close < X.low) | (X.close > X.high)
    X["dte_mis"] = X["DTE"].values != exp_dte
    g = X.groupby("symbol", sort=False)
    per = g.agg(n=("close", "size"), nnull=("null", "sum"), nbad=("ohlc_bad", "sum"), ndte=("dte_mis", "sum"),
                min_dte=("DTE", "min"), strike=("strike", "first"), typ=("option_type", "first"))
    per["ndup"] = X.duplicated(["symbol", "timestamp"]).groupby(X["symbol"], sort=False).sum()
    gd = X.groupby(["symbol", "date"], sort=False)["minute"].agg(["min", "max", "size"])
    gd["gap"] = (gd["max"] - gd["min"] + 1) - gd["size"]
    gp = gd[gd["gap"] > 0].groupby(level=0).agg(gapdays=("gap", "size"), gapmin=("gap", "sum"))
    dates_by = g["date"].apply(lambda s: set(pd.to_datetime(s).dt.date))
    firstlast = g["date"].agg(["min", "max"])
    expd = pd.Timestamp(exp).date()
    lo_i = int(np.searchsorted(tdarr, np.datetime64(str(firstlast["min"].min())[:10])))
    rows = []
    for sym in per.index:
        p = per.loc[sym]; fd = pd.Timestamp(firstlast.loc[sym, "min"]).date(); ld = pd.Timestamp(firstlast.loc[sym, "max"]).date()
        window = [d for d in tdarr if fd <= d <= min(expd, CUT_D)]           # trading days in active window
        pres = dates_by.loc[sym]
        miss_days = sorted(d for d in window if d <= ld and d not in pres)
        last_td = window[-1] if window else ld
        fl = []
        if p.nnull:
            fl.append(("null_fields", str(int(p.nnull))))
        if p.nbad:
            fl.append(("ohlc_violation", str(int(p.nbad))))
        if p.ndup:
            fl.append(("duplicate_timestamp", str(int(p.ndup))))
        if p.ndte:
            fl.append(("dte_mismatch", str(int(p.ndte))))
        if p.min_dte != 0:
            fl.append(("no_expiry_day_candle", f"min_DTE={int(p.min_dte)}"))
        if ld < last_td:
            fl.append(("stopped_before_expiry", f"last={ld} vs {last_td}"))
        if miss_days:
            fl.append(("missing_trading_days", f"{len(miss_days)} e.g. {miss_days[:3]}"))
        if sym in gp.index:
            fl.append(("intraday_internal_gaps", f"{int(gp.loc[sym,'gapdays'])} days / {int(gp.loc[sym,'gapmin'])} min"))
        for it, det in fl:
            rows.append({"symbol": sym, "expiry": exp, "strike": p.strike, "type": p.typ, "issue": it, "detail": det})
    rows += corrupt_rows
    oi_zero = int((X["OI"] == 0).sum()); nrows = len(X)
    return rows, {"contracts": len(per) + len(corrupt), "clean": int((~per.index.isin([r["symbol"] for r in rows])).sum()),
                  "oi_zero": oi_zero, "nrows": nrows}


def main():
    t0 = time.time()
    tdarr = np.array(sorted(pd.read_parquet(DAILY, columns=["date"])["date"].dt.date.unique()))
    exp_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    man = pd.read_csv(MAN)
    # ---- structural ----
    present_exp = sorted(man["expiry"].unique()); apiE = api_expiries()
    scope = apiE if apiE else present_exp; struct = []
    for e in scope:
        if e not in present_exp:
            struct.append({"issue": "missing_expiry", "expiry": e, "detail": "0 contracts pulled"})
    for e, gg in man.groupby("expiry"):
        ce = set(gg[gg.option_type == "CE"].strike); pe = set(gg[gg.option_type == "PE"].strike)
        for s in sorted(ce - pe):
            struct.append({"issue": "one_sided_strike_CE_only", "expiry": e, "detail": f"strike {s}: CE but no PE"})
        for s in sorted(pe - ce):
            struct.append({"issue": "one_sided_strike_PE_only", "expiry": e, "detail": f"strike {s}: PE but no CE"})
    pd.DataFrame(struct).to_csv(OUTDIR / "structural.csv", index=False)
    print(f"structural: scope {len(scope)} expiries | present {len(present_exp)} | issues {len(struct)}", flush=True)

    # ---- per-expiry (resumable) ----
    for i, d in enumerate(exp_dirs, 1):
        exp = f"{d.name[:4]}-{d.name[4:6]}-{d.name[6:]}"
        part = PARTS / f"{d.name}.json"
        if part.exists():
            continue
        files = sorted(glob.glob(str(d / "*.parquet")))
        if not files:
            json.dump({"rows": [{"symbol": "", "expiry": exp, "strike": "", "type": "", "issue": "empty_expiry_dir", "detail": "0 parquet files pulled for this expiry"}],
                       "cnt": {"contracts": 0, "clean": 0, "oi_zero": 0, "nrows": 0}}, open(part, "w"))
            print(f"  [{i}/{len(exp_dirs)}] {exp}: EMPTY DIR (0 files) -> flagged", flush=True); continue
        rows, cnt = audit_expiry(exp, files, tdarr)
        json.dump({"rows": rows, "cnt": cnt}, open(part, "w"))
        print(f"  [{i}/{len(exp_dirs)}] {exp}: {cnt['contracts']} contracts, {len(rows)} flags | {time.time()-t0:.0f}s", flush=True)

    # ---- finalize ----
    allrows = []; C = Counter(); oi_zero = nrows = ncontracts = 0
    for p in sorted(PARTS.glob("*.json")):
        j = json.load(open(p)); allrows += j["rows"]
        C["contracts"] += j["cnt"]["contracts"]; C["clean"] += j["cnt"]["clean"]
        oi_zero += j["cnt"]["oi_zero"]; nrows += j["cnt"]["nrows"]
    ISSUES = pd.DataFrame(allrows)
    by = ISSUES["issue"].value_counts().to_dict() if len(ISSUES) else {}
    dows = [(e, WD[datetime.strptime(e, "%Y-%m-%d").weekday()]) for e in present_exp]
    anom = [(e, dd) for e, dd in dows if not ((e <= "2025-08-31" and dd in ("Thu", "Wed")) or (e >= "2025-09-01" and dd in ("Tue", "Mon")))]
    summ = {"total_contracts_checked": C["contracts"], "contracts_zero_issues": C["contracts"] - (ISSUES["symbol"].nunique() if len(ISSUES) else 0),
            "contracts_flagged": ISSUES["symbol"].nunique() if len(ISSUES) else 0, "total_rows": nrows,
            "flag_missing_trading_days": by.get("missing_trading_days", 0), "flag_intraday_internal_gaps": by.get("intraday_internal_gaps", 0),
            "flag_stopped_before_expiry": by.get("stopped_before_expiry", 0), "flag_no_expiry_day_candle": by.get("no_expiry_day_candle", 0),
            "flag_ohlc_violation": by.get("ohlc_violation", 0), "flag_null_fields": by.get("null_fields", 0),
            "flag_dte_mismatch": by.get("dte_mismatch", 0), "flag_duplicate_timestamp": by.get("duplicate_timestamp", 0),
            "structural_issues": len(struct), "oi_zero_pct_of_rows": round(oi_zero / max(nrows, 1) * 100, 1),
            "weekday_anomalies": len(anom)}
    with pd.ExcelWriter(OUTDIR / "nifty_options_audit.xlsx", engine="openpyxl") as w:
        pd.DataFrame(list(summ.items()), columns=["metric", "value"]).to_excel(w, sheet_name="summary", index=False)
        (pd.DataFrame(struct) if struct else pd.DataFrame([{"note": "no structural issues"}])).to_excel(w, sheet_name="structural", index=False)
        (ISSUES if len(ISSUES) else pd.DataFrame([{"note": "clean"}])).to_excel(w, sheet_name="contract_issues", index=False)
        pd.DataFrame([{"regime": "Thu<=Aug2025 / Tue>=Sep2025", "counts": str(dict(Counter(x for _, x in dows))), "anomalies": str(anom)}]).to_excel(w, sheet_name="weekday_check", index=False)
    if len(ISSUES):
        ISSUES.to_csv(OUTDIR / "flagged_contracts.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 88 + "\nNIFTY OPTIONS COMPLETENESS AUDIT (read-only)\n" + "=" * 88)
    for k, v in summ.items():
        print(f"  {k:32s}: {v}")
    print("\n--- structural issues ---"); print(pd.DataFrame(struct).to_string(index=False) if struct else "  NONE")
    print("\n--- contract issue counts by type ---"); print(ISSUES["issue"].value_counts().to_string() if len(ISSUES) else "  NONE")
    print(f"\n  OI note: {summ['oi_zero_pct_of_rows']}% of rows have OI=0 (zero-volume minutes; OI not carried forward -> forward-fill downstream).")
    print(f"  weekday: Thu/Tue regimes, anomalies={anom if anom else 'NONE (Wed/Mon are expected holiday shifts)'}")
    print(f"\nSaved -> {OUTDIR} | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
