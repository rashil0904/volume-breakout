# -*- coding: utf-8 -*-
"""
build_futures_oi_price.py
=========================
Add Nifty FUTURES price (OHLC + settlement) to the daily OI series, from the SAME NSE F&O
bhavcopy that supplies the OI, so price and OI reference the identical contract per day.

RESOLUTION OF THE OI/price contract mismatch:
  The existing OI series (nifty_futures_oi_daily.csv) is COMBINED TOTAL across all 3 expiries,
  but futures OHLC is single-contract. For an internally-consistent price_oi_signal, price and
  OI are aligned to the NEAR-MONTH (front) contract: futures OHLC/settlement + near-month OI.
  combined_oi_all_expiries is retained as a separate labelled column.

  Day-over-day changes (close_change_pct, oi_change_pct) are computed SAME-CONTRACT (each
  contract vs its OWN prior day), so monthly rollovers don't create spurious signals; the
  rollover day (near contract switches) is flagged.

Re-fetches each bhavcopy (resumable, results/fo_price_cache/) to extract per-expiry OHLC+
settle+OI (only summary was cached before). Merges spot Nifty close (nifty_15min_ohlc.csv)
-> spot_close + basis (futures_close - spot_close). Saves data/nifty_futures_oi_price_daily.xlsx.

OI units = NSE OPEN_INT/OpnIntrst as reported: UNITS (contracts × lot size), NOT contracts.
"""
import sys, io, json, time, zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fetch_nifty_futures_oi as FO          # reuse session / urls_for / calendar / cutoff

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT_XLSX = DATA / "nifty_futures_oi_price_daily.xlsx"
SPOT_CSV = DATA / "nifty_15min_ohlc.csv"
PCACHE = REPO / "results" / "fo_price_cache"
IST = "Asia/Kolkata"
SESSION_HMS = list(range(555, 916, 15))


def parse_full(content):
    """All NIFTY index-future rows -> DataFrame(expiry, o, h, l, c, settle, oi)."""
    zf = zipfile.ZipFile(io.BytesIO(content))
    df = pd.read_csv(zf.open(zf.namelist()[0]))
    cols = set(df.columns)
    if {"INSTRUMENT", "SYMBOL", "OPEN_INT", "EXPIRY_DT"} <= cols:                 # OLD
        m = (df["INSTRUMENT"].astype(str).str.upper() == "FUTIDX") & \
            (df["SYMBOL"].astype(str).str.upper() == "NIFTY")
        s = df[m]
        return pd.DataFrame({
            "expiry": pd.to_datetime(s["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce").dt.date,
            "o": pd.to_numeric(s["OPEN"], errors="coerce"), "h": pd.to_numeric(s["HIGH"], errors="coerce"),
            "l": pd.to_numeric(s["LOW"], errors="coerce"), "c": pd.to_numeric(s["CLOSE"], errors="coerce"),
            "settle": pd.to_numeric(s["SETTLE_PR"], errors="coerce"),
            "oi": pd.to_numeric(s["OPEN_INT"], errors="coerce")})
    if {"FinInstrmTp", "TckrSymb", "OpnIntrst", "XpryDt"} <= cols:                # UDiFF
        m = (df["FinInstrmTp"].astype(str).str.upper() == "IDF") & \
            (df["TckrSymb"].astype(str).str.upper() == "NIFTY")
        s = df[m]
        return pd.DataFrame({
            "expiry": pd.to_datetime(s["XpryDt"], format="%Y-%m-%d", errors="coerce").dt.date,
            "o": pd.to_numeric(s["OpnPric"], errors="coerce"), "h": pd.to_numeric(s["HghPric"], errors="coerce"),
            "l": pd.to_numeric(s["LwPric"], errors="coerce"), "c": pd.to_numeric(s["ClsPric"], errors="coerce"),
            "settle": pd.to_numeric(s["SttlmPric"], errors="coerce"),
            "oi": pd.to_numeric(s["OpnIntrst"], errors="coerce")})
    raise ValueError(f"unrecognised schema: {sorted(cols)[:10]}")


def fetch_day(sess, d):
    cf = PCACHE / f"{d.strftime('%Y%m%d')}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    primary, fallback = FO.urls_for(d)
    for url in (primary, fallback):
        for attempt in range(4):
            try:
                r = sess.get(url, timeout=30)
            except Exception:
                time.sleep(2 * (attempt + 1)); continue
            if r.status_code == 200 and r.content[:2] == b"PK":
                try:
                    fut = parse_full(r.content).dropna(subset=["expiry", "oi"])
                except Exception as e:
                    print(f"    {d} parse err: {e}"); break
                if fut.empty:
                    break
                rec = {"date": d.isoformat(), "combined_oi": int(fut["oi"].sum()),
                       "contracts": [{"expiry": e.isoformat(), "o": _f(o), "h": _f(h), "l": _f(l),
                                      "c": _f(c), "settle": _f(st), "oi": int(oi)}
                                     for e, o, h, l, c, st, oi in fut[["expiry", "o", "h", "l", "c", "settle", "oi"]].itertuples(index=False)]}
                cf.write_text(json.dumps(rec)); return rec
            if r.status_code in (401, 403, 429):
                time.sleep(1.5 * (attempt + 1))
                if attempt == 1:
                    sess = FO.new_session()
                continue
            break
    return None


def _f(x):
    return None if pd.isna(x) else float(x)


def spot_daily_close():
    df = pd.read_csv(SPOT_CSV)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    df["date"] = ts.dt.date; df["hm"] = ts.dt.hour * 60 + ts.dt.minute
    df = df[df["hm"].isin(SESSION_HMS)]
    last = df.sort_values("hm").drop_duplicates("date", keep="last").set_index("date")["close"]
    return last


def classify(dc, doi):
    if pd.isna(dc) or pd.isna(doi):
        return "n/a"
    if dc > 0 and doi > 0:
        return "Long Buildup"
    if dc < 0 and doi > 0:
        return "Short Buildup"
    if dc > 0 and doi < 0:
        return "Short Covering"
    if dc < 0 and doi < 0:
        return "Long Unwinding"
    return "Neutral"


def main():
    PCACHE.mkdir(parents=True, exist_ok=True)
    print("Deriving calendar + re-fetching bhavcopy price fields (resumable) …")
    cal = FO.trading_calendar()
    sess = FO.new_session()
    recs, missing = [], []
    t0 = time.time()
    for i, d in enumerate(cal, 1):
        r = fetch_day(sess, d)
        (recs if r else missing).append(r if r else d)
        if i % 150 == 0 or i == len(cal):
            el = time.time() - t0
            print(f"    …{i}/{len(cal)} ({len(recs)} ok, {len(missing)} miss, {el:.0f}s, ETA {el/i*(len(cal)-i):.0f}s)")
        if not (PCACHE / f"{d.strftime('%Y%m%d')}.json").exists():
            time.sleep(0.2)

    # ── long panel: one row per (date, contract) ──
    long_rows = []
    for r in recs:
        for c in r["contracts"]:
            long_rows.append({"date": pd.to_datetime(r["date"]).date(), "combined_oi": r["combined_oi"],
                              "expiry": pd.to_datetime(c["expiry"]).date(), **{k: c[k] for k in ("o", "h", "l", "c", "settle", "oi")}})
    L = pd.DataFrame(long_rows).sort_values(["expiry", "date"]).reset_index(drop=True)
    # same-contract prior day (each contract vs its OWN previous session)
    L["prev_c"] = L.groupby("expiry")["c"].shift(1)
    L["prev_oi"] = L.groupby("expiry")["oi"].shift(1)

    # near (front) contract per date = earliest expiry that day
    near_idx = L.groupby("date")["expiry"].idxmin()
    N = L.loc[near_idx].sort_values("date").reset_index(drop=True)
    N["close_change_pct"] = ((N["c"] - N["prev_c"]) / N["prev_c"] * 100).round(4)
    N["oi_change_pct"] = ((N["oi"] - N["prev_oi"]) / N["prev_oi"] * 100).round(4)
    N["rollover_day"] = N["expiry"].ne(N["expiry"].shift(1))          # near contract switched
    N["price_oi_signal"] = [classify(dc, doi) for dc, doi in zip(N["close_change_pct"], N["oi_change_pct"])]

    # spot + basis
    spot = spot_daily_close()
    N["spot_close"] = N["date"].map(spot)
    N["basis"] = (N["c"] - N["spot_close"]).round(2)
    N["settle_equals_close"] = np.isclose(N["c"].fillna(-1), N["settle"].fillna(-2))

    out = N.rename(columns={"expiry": "contract_expiry", "o": "futures_open", "h": "futures_high",
                            "l": "futures_low", "c": "futures_close", "settle": "settlement_price",
                            "oi": "near_month_oi", "combined_oi": "combined_oi_all_expiries"})
    out = out[["date", "contract_expiry", "futures_open", "futures_high", "futures_low", "futures_close",
               "settlement_price", "settle_equals_close", "close_change_pct",
               "near_month_oi", "oi_change_pct", "price_oi_signal", "rollover_day",
               "combined_oi_all_expiries", "spot_close", "basis"]]

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="futures_oi_price_daily", index=False)
        sig_counts = out["price_oi_signal"].value_counts().rename_axis("price_oi_signal").reset_index(name="days")
        sig_counts.to_excel(w, sheet_name="signal_counts", index=False)
        pd.DataFrame({"note": [
            "Price + near-month OI are the NEAR-MONTH (front) contract; combined_oi_all_expiries is the "
            "sum across all 3 expiries (kept for reference).",
            "close_change_pct / oi_change_pct are SAME-CONTRACT day-over-day (each contract vs its own prior "
            "session); rollover_day=True marks days the near contract switched.",
            "OI units: NSE OPEN_INT/OpnIntrst as reported = UNITS (contracts × lot size), NOT contracts; "
            "Nifty lot size changed over the period.",
            "basis = futures_close - spot_close (Nifty index). settle_equals_close flags identical settlement/close.",
        ]}).to_excel(w, sheet_name="notes", index=False)

    pd.set_option("display.width", 240)
    print("\n" + "=" * 78)
    print("Nifty FUTURES OI + PRICE (near-month) daily — saved")
    print("=" * 78)
    print(f"  Saved to        : {OUT_XLSX}")
    print(f"  Rows            : {len(out)}  | range {out['date'].min()} → {out['date'].max()}")
    print(f"  Missing days    : {len(missing)}")
    print(f"  settle==close   : {int(out['settle_equals_close'].sum())}/{len(out)} days")
    print("\n  price_oi_signal category counts (near-month price vs near-month OI):")
    print(out["price_oi_signal"].value_counts().to_string())
    print(f"    (rollover days, signal excluded from interpretation: {int(out['rollover_day'].sum())})")
    b = out["basis"].dropna()
    print(f"\n  BASIS (futures_close - spot_close): avg {b.mean():.2f} | min {b.min():.2f} "
          f"(max discount) | max {b.max():.2f} (max premium) | % days premium {float((b>0).mean())*100:.1f}%")
    print("\n  first 3 rows:\n" + out.head(3).to_string(index=False))
    print("\n  last 3 rows:\n" + out.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
