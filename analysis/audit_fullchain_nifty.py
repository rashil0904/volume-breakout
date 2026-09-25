# -*- coding: utf-8 -*-
"""audit_fullchain_nifty.py — READ-ONLY verification of the densified full-chain NIFTY options dataset
(data/options_intraday_full/NIFTY). Checks: (1) structural (Get-Expiries coverage; CE/PE both sides per
strike; pulled-vs-API contract count); (2) per-contract candle completeness (every active day == 375
rows post-densify; whole missing trading days = genuine gaps; document via is_synthetic that Upstox
OMITTED zero-vol minutes at source); (3) field sanity (nulls; OHLC validity; zero-vol rows must have
O=H=L=C; DTE==calendar recompute reaching 0; dup timestamps); (4) expiry weekday cross-check; (5)
spot-vs-strike sanity. Report only. Resumable per-expiry (parts). No modify/re-pull/auto-fix.
"""
import sys, time, glob, json
from pathlib import Path
from datetime import datetime
from collections import Counter
import requests, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb, data_loading as dl

ROOT = rb.BASE / "data" / "options_intraday_full" / "NIFTY"
MAN = ROOT.parent / "manifest_nifty.csv"; ISS = ROOT.parent / "issues_nifty.csv"
OUTDIR = rb.RESULTS / "fullchain_audit"; PARTS = OUTDIR / "parts"; PARTS.mkdir(parents=True, exist_ok=True)
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"
NIFTY = "NSE_INDEX|Nifty 50"; FLOOR, CUTOFF = "2024-10-01", "2026-07-31"; FULL = 375
H = {"Accept": "application/json", "Authorization": f"Bearer {dl.ACCESS_TOKEN}"}
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]; CUT_D = pd.Timestamp(CUTOFF).date()
BASE = "https://api.upstox.com/v2/expired-instruments"; enc = lambda k: k.replace("|", "%7C")


def get(u):
    for a in range(4):
        r = requests.get(u, headers=H, timeout=30)
        if r.status_code == 200:
            d = r.json().get("data", []); return d.get("candles", []) if isinstance(d, dict) else d
        if r.status_code in (429, 500, 502, 503): time.sleep(1.2 * (a + 1)); continue
        return None
    return None


def audit_expiry(exp, files, tdset, tdarr):
    X = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ts = X["timestamp"].values.astype("datetime64[m]"); date = ts.astype("datetime64[D]"); minute = ts.astype("int64") % 1440
    X["date"] = date; X["minute"] = minute
    exp_dte = (np.datetime64(exp) - date).astype("timedelta64[D]").astype("int64")
    X["null"] = X[["open", "high", "low", "close", "volume", "OI", "DTE"]].isna().any(axis=1)
    X["ohlc_bad"] = (X.high < X.low) | (X.open < X.low) | (X.open > X.high) | (X.close < X.low) | (X.close > X.high)
    X["dte_mis"] = X["DTE"].values != exp_dte
    flat = (X.open == X.high) & (X.high == X.low) & (X.low == X.close)
    X["zv_bad"] = (X.volume == 0) & (~X.get("is_synthetic", pd.Series(False, index=X.index)).astype(bool)) & (~flat)   # real zero-vol row w/ OHLC variation
    g = X.groupby("symbol", sort=False)
    per = g.agg(nnull=("null", "sum"), nbad=("ohlc_bad", "sum"), ndte=("dte_mis", "sum"), nzv=("zv_bad", "sum"),
                min_dte=("DTE", "min"), strike=("strike", "first"), typ=("option_type", "first"),
                nsyn=("is_synthetic", "sum") if "is_synthetic" in X.columns else ("null", "sum"), nrows=("close", "size"))
    per["ndup"] = X.duplicated(["symbol", "timestamp"]).groupby(X["symbol"], sort=False).sum()
    cpd = X.groupby(["symbol", "date"], sort=False).size()                     # candles per contract-day
    badday = cpd[cpd != FULL]
    bd_by_sym = badday.groupby(level=0).size()
    bad_dates = Counter(pd.to_datetime(badday.index.get_level_values(1)).date)  # which dates are short (session-length signal)
    dates_by = g["date"].apply(lambda s: set(pd.to_datetime(s).dt.date))
    fl = g["date"].agg(["min", "max"])
    expd = pd.Timestamp(exp).date(); rows = []
    for sym in per.index:
        p = per.loc[sym]; fd = pd.Timestamp(fl.loc[sym, "min"]).date(); ld = pd.Timestamp(fl.loc[sym, "max"]).date()
        window = [d for d in tdarr if fd <= d <= min(expd, CUT_D) and d <= ld]
        miss = sorted(set(window) - dates_by.loc[sym]); fx = []
        if p.nnull: fx.append(("null_fields", int(p.nnull)))
        if p.nbad: fx.append(("ohlc_violation", int(p.nbad)))
        if p.nzv: fx.append(("zero_vol_ohlc_inconsistent", int(p.nzv)))
        if p.ndup: fx.append(("duplicate_timestamp", int(p.ndup)))
        if p.ndte: fx.append(("dte_mismatch", int(p.ndte)))
        if p.min_dte != 0: fx.append(("no_expiry_day_row", int(p.min_dte)))
        if sym in bd_by_sym.index: fx.append(("day_not_375_rows", int(bd_by_sym.loc[sym])))
        if miss: fx.append(("missing_trading_days", len(miss)))
        for it, det in fx:
            rows.append({"symbol": sym, "expiry": exp, "strike": p.strike, "type": p.typ, "issue": it, "detail": str(det)})
    ce = set(per[per.typ == "CE"].strike); pe = set(per[per.typ == "PE"].strike); struct = []
    for s in sorted(ce - pe): struct.append({"expiry": exp, "issue": "one_sided_CE_only", "detail": f"strike {s}"})
    for s in sorted(pe - ce): struct.append({"expiry": exp, "issue": "one_sided_PE_only", "detail": f"strike {s}"})
    cnt = {"contracts": int(len(per)), "flagged": int(len(set(r["symbol"] for r in rows))), "rows": int(per.nrows.sum()),
           "syn": int(per.nsyn.sum()) if "is_synthetic" in X.columns else 0, "zv_total": int((X.volume == 0).sum()),
           "struct": struct, "short_dates": {str(k): int(v) for k, v in bad_dates.items()}}
    return rows, cnt


def main():
    t0 = time.time()
    d = pd.read_parquet(DAILY, columns=["date"]); tdarr = np.array(sorted(d["date"].dt.date.unique())); tdset = set(tdarr)
    exp_dirs = sorted([x for x in ROOT.iterdir() if x.is_dir()])
    apiE = sorted([e for e in (get(f"{BASE}/expiries?instrument_key={enc(NIFTY)}") or []) if FLOOR <= e <= CUTOFF])
    present = [f"{x.name[:4]}-{x.name[4:6]}-{x.name[6:]}" for x in exp_dirs if any(x.glob("*.parquet"))]
    # structural: coverage + API-count
    struct = []
    for e in apiE:
        if e not in present: struct.append({"expiry": e, "issue": "missing_expiry", "detail": "0 contracts"})
    try:
        emp = pd.read_csv(ISS); emp = emp[emp.issue == "empty_no_candles"].drop_duplicates(["expiry", "symbol"]).groupby("expiry").size().to_dict()
    except Exception:
        emp = {}
    print(f"structural: {len(apiE)} expiries scope | present {len(present)}", flush=True)
    for ei, x in enumerate(exp_dirs, 1):
        e = f"{x.name[:4]}-{x.name[4:6]}-{x.name[6:]}"; part = PARTS / f"{x.name}.json"
        if part.exists(): continue
        files = sorted(glob.glob(str(x / "*.parquet")))
        if not files:
            json.dump({"rows": [], "cnt": {"contracts": 0, "flagged": 0, "rows": 0, "syn": 0, "zv_total": 0, "struct": [], "short_dates": {}, "empty_dir": True}}, open(part, "w")); continue
        cons = get(f"{BASE}/option/contract?instrument_key={enc(NIFTY)}&expiry_date={e}") or []
        api_n = len(cons); disk_n = len(files); e_n = emp.get(e, 0)
        extra_struct = []
        if disk_n + e_n < api_n:
            extra_struct.append({"expiry": e, "issue": "pulled_lt_api", "detail": f"disk {disk_n}+empty {e_n} < api {api_n} (missing {api_n-disk_n-e_n})"})
        rows, cnt = audit_expiry(e, files, tdset, tdarr); cnt["struct"] += extra_struct
        json.dump({"rows": rows, "cnt": cnt}, open(part, "w"))
        print(f"  [{ei}/{len(exp_dirs)}] {e}: {cnt['contracts']} contracts, {len(rows)} flags, {cnt['rows']:,} rows | {time.time()-t0:.0f}s", flush=True)

    # finalize
    allrows = []; C = Counter(); struct_all = list(struct); short = Counter()
    for p in sorted(PARTS.glob("*.json")):
        j = json.load(open(p)); allrows += j["rows"]; c = j["cnt"]
        for k in ("contracts", "rows", "syn", "zv_total"): C[k] += c.get(k, 0)
        struct_all += c.get("struct", [])
        for dt, n in c.get("short_dates", {}).items(): short[dt] += n
    I = pd.DataFrame(allrows); S = pd.DataFrame(struct_all)
    by = I["issue"].value_counts().to_dict() if len(I) else {}
    # weekday
    dows = [(e, WD[datetime.strptime(e, "%Y-%m-%d").weekday()]) for e in present]
    anom = [(e, w) for e, w in dows if not ((e <= "2025-08-31" and w in ("Thu", "Wed")) or (e >= "2025-09-01" and w in ("Tue", "Mon")))]
    # spot-vs-strike
    nd = get(f"https://api.upstox.com/v2/historical-candle/{enc(NIFTY)}/day/2026-07-31/2024-06-01") or []
    ndf = pd.DataFrame(nd, columns=["ts", "o", "h", "l", "c", "v", "oi"]); ndf["date"] = pd.to_datetime(ndf.ts).dt.tz_localize(None).dt.date
    ncl = dict(zip(ndf.date, ndf.c)); man = pd.read_csv(MAN)
    spot = []
    for e, gg in man.groupby("expiry"):
        ed = pd.Timestamp(e).date(); sp = ncl.get(ed) or ncl.get(min(ncl, key=lambda d: abs((d - ed).days)))
        smin, smax, smed = gg.strike.min(), gg.strike.max(), gg.strike.median()
        if not (smin <= sp <= smax) or not (0.3 * sp < smed < 3 * sp):
            spot.append({"expiry": e, "spot": round(sp), "strike_min": smin, "strike_max": smax, "strike_med": smed, "issue": "strike_range_vs_spot_off"})

    n_short_dates = {k: v for k, v in short.items()}
    summ = {"total_contracts": C["contracts"], "contracts_zero_issues": C["contracts"] - (I["symbol"].nunique() if len(I) else 0),
            "contracts_flagged": I["symbol"].nunique() if len(I) else 0, "total_rows": C["rows"],
            "zero_volume_rows": C["zv_total"], "synthetic_rows(api_omitted_minutes)": C["syn"],
            "flag_missing_trading_days": by.get("missing_trading_days", 0), "flag_day_not_375": by.get("day_not_375_rows", 0),
            "flag_no_expiry_day_row": by.get("no_expiry_day_row", 0), "flag_ohlc_violation": by.get("ohlc_violation", 0),
            "flag_null_fields": by.get("null_fields", 0), "flag_zero_vol_ohlc_inconsistent": by.get("zero_vol_ohlc_inconsistent", 0),
            "flag_dte_mismatch": by.get("dte_mismatch", 0), "flag_duplicate_timestamp": by.get("duplicate_timestamp", 0),
            "structural_issues": len(S), "one_sided_strikes": int(S.issue.str.startswith("one_sided").sum()) if len(S) else 0,
            "pulled_lt_api_expiries": int((S.issue == "pulled_lt_api").sum()) if len(S) else 0,
            "weekday_anomalies": len(anom), "spot_strike_anomalies": len(spot)}
    with pd.ExcelWriter(OUTDIR / "fullchain_audit.xlsx", engine="openpyxl") as w:
        pd.DataFrame(list(summ.items()), columns=["metric", "value"]).to_excel(w, sheet_name="summary", index=False)
        (S if len(S) else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="structural", index=False)
        (I if len(I) else pd.DataFrame([{"note": "none"}])).to_excel(w, sheet_name="contract_issues", index=False)
        pd.DataFrame([{"date": k, "contracts_with_short_day": v} for k, v in sorted(short.items())]).to_excel(w, sheet_name="short_session_dates", index=False)
        (pd.DataFrame(spot) if spot else pd.DataFrame([{"note": "all strike ranges bracket spot"}])).to_excel(w, sheet_name="spot_vs_strike", index=False)
    if len(I): I.to_csv(OUTDIR / "flagged_contracts.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 90 + "\nFULL-CHAIN NIFTY OPTIONS — VERIFICATION (read-only, densified dataset)\n" + "=" * 90)
    for k, v in summ.items(): print(f"  {k:38s}: {v:,}" if isinstance(v, int) else f"  {k:38s}: {v}")
    print("\n--- structural issues (first 12) ---"); print(S.head(12).to_string(index=False) if len(S) else "  NONE")
    print("\n--- contract issue counts ---"); print(I["issue"].value_counts().to_string() if len(I) else "  NONE")
    print("\n--- top 'short day' dates (candles != 375; recurring date = special/short session, not per-contract gap) ---")
    for dt, n in short.most_common(8): print(f"    {dt}: {n:,} contracts")
    print(f"\n  API-OMISSION NOTE: {C['syn']:,} rows are is_synthetic=True = minutes Upstox did NOT return (omitted at source),")
    print(f"    reconstructed in densification. Confirms 'no skipping zero-vol' was NOT achievable from raw Upstox (API limitation).")
    print(f"  weekday anomalies: {anom if anom else 'NONE (Wed/Mon = expected holiday shifts)'}")
    print(f"\nSaved -> {OUTDIR} | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
