# -*- coding: utf-8 -*-
"""
fetch_nifty_futures_oi.py
=========================
Daily end-of-day COMBINED TOTAL open interest for Nifty futures (sum of OI across ALL
Nifty index-futures expiries each trading day), over the existing dataset's date range.

Source: NSE F&O bhavcopy on nsearchives.nseindia.com (same host as fetch_delivery_data.py).
Handles BOTH formats across the range:
  - OLD  (pre 2024-07-08): content/historical/DERIVATIVES/<YYYY>/<MON>/fo<DDMONYYYY>bhav.csv.zip
        cols INSTRUMENT,SYMBOL,EXPIRY_DT,...,OPEN_INT,...  -> filter INSTRUMENT=FUTIDX & SYMBOL=NIFTY
  - NEW  (UDiFF, >= 2024-07-08): content/fo/BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv.zip
        cols FinInstrmTp,TckrSymb,XpryDt,OpnIntrst,... -> filter FinInstrmTp=IDF & TckrSymb=NIFTY

Combined total OI = Σ OI over all NIFTY index-future expiry rows that day.
OI value = the bhavcopy's OPEN_INT / OpnIntrst field AS REPORTED BY NSE (labelled; see notes).
Resumable: one JSON per date cached in results/fo_oi_cache/.
"""
import sys, io, json, time, zipfile
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).parent
MASTER = REPO / "master_data"
OUT_DIR = REPO / "data"
OUT_CSV = OUT_DIR / "nifty_futures_oi_daily.csv"
CACHE = REPO / "results" / "fo_oi_cache"
IST = "Asia/Kolkata"
UDIFF_CUTOFF = date(2024, 7, 8)          # NSE F&O bhavcopy switched to UDiFF format
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
           "Referer": "https://www.nseindia.com/"}


def new_session():
    s = requests.Session(); s.headers.update(HEADERS)
    for u in ("https://www.nseindia.com", "https://www.nseindia.com/all-reports"):
        try:
            s.get(u, timeout=15)
        except requests.RequestException:
            pass
    return s


def urls_for(d):
    """Return ordered (primary, fallback) bhavcopy URLs for a date."""
    old = (f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
           f"{d.year}/{MONTHS[d.month-1]}/fo{d.day:02d}{MONTHS[d.month-1]}{d.year}bhav.csv.zip")
    new = (f"https://nsearchives.nseindia.com/content/fo/"
           f"BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip")
    return (new, old) if d >= UDIFF_CUTOFF else (old, new)


def parse_bhav(content):
    """Read a bhavcopy zip -> DataFrame of NIFTY index futures rows: (expiry, oi)."""
    zf = zipfile.ZipFile(io.BytesIO(content))
    name = zf.namelist()[0]
    df = pd.read_csv(zf.open(name))
    cols = set(df.columns)
    if {"INSTRUMENT", "SYMBOL", "OPEN_INT", "EXPIRY_DT"} <= cols:              # OLD
        m = (df["INSTRUMENT"].astype(str).str.upper() == "FUTIDX") & \
            (df["SYMBOL"].astype(str).str.upper() == "NIFTY")
        sub = df[m]
        return pd.DataFrame({"expiry": sub["EXPIRY_DT"].astype(str),
                             "oi": pd.to_numeric(sub["OPEN_INT"], errors="coerce")})
    if {"FinInstrmTp", "TckrSymb", "OpnIntrst", "XpryDt"} <= cols:             # UDiFF
        m = (df["FinInstrmTp"].astype(str).str.upper() == "IDF") & \
            (df["TckrSymb"].astype(str).str.upper() == "NIFTY")
        sub = df[m]
        return pd.DataFrame({"expiry": sub["XpryDt"].astype(str),
                             "oi": pd.to_numeric(sub["OpnIntrst"], errors="coerce")})
    raise ValueError(f"unrecognised bhavcopy schema: {sorted(cols)[:12]}")


def fetch_day(sess, d):
    """Return dict summary for date d, or None if unavailable."""
    cf = CACHE / f"{d.strftime('%Y%m%d')}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    primary, fallback = urls_for(d)
    for url in (primary, fallback):
        for attempt in range(4):
            try:
                r = sess.get(url, timeout=30)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1)); continue
            if r.status_code == 200 and r.content[:2] == b"PK":               # a real zip
                try:
                    fut = parse_bhav(r.content)
                except Exception as e:
                    print(f"    {d} parse error: {e}"); break
                fut = fut.dropna(subset=["oi"])
                if fut.empty:
                    break
                exp = sorted(fut["expiry"].unique(), key=lambda x: pd.to_datetime(x, errors="coerce"))
                rec = {"date": d.isoformat(), "oi_total": int(fut["oi"].sum()),
                       "n_expiries": int(fut["expiry"].nunique()),
                       "near_expiry": exp[0] if exp else "", "expiries": exp}
                cf.write_text(json.dumps(rec)); return rec
            if r.status_code in (401, 403, 429):
                time.sleep(1.5 * (attempt + 1));
                if attempt == 1:
                    sess = new_session()
                continue
            break                                                             # 404 -> try fallback / give up
    return None


def trading_calendar():
    """MIN/MAX + trading days from the existing stock dataset (union of liquid always-listed symbols)."""
    liquid = ["RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "INFY", "TCS"]
    days = set()
    for s in liquid:
        f = MASTER / f"{s}.parquet"
        if not f.exists():
            continue
        ts = pd.to_datetime(pq.read_table(f, columns=["timestamp"]).column("timestamp").to_pandas(),
                            utc=True).dt.tz_convert(IST)
        days.update(ts.dt.date.unique())
    return sorted(days)


def main():
    CACHE.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
    print("STEP 1 — deriving date range from existing stock data …")
    cal = trading_calendar()
    dmin, dmax = cal[0], cal[-1]
    print(f"    stock calendar: {len(cal):,} trading days | {dmin} → {dmax}")

    print("\nSTEP 3 — fetching NSE F&O bhavcopy per trading day (resumable) …")
    sess = new_session()
    recs, missing = [], []
    t0 = time.time()
    for i, d in enumerate(cal, 1):
        rec = fetch_day(sess, d)
        if rec is None:
            missing.append(d)
        else:
            recs.append(rec)
        if i % 100 == 0 or i == len(cal):
            el = time.time() - t0
            print(f"    …{i}/{len(cal)} ({len(recs)} ok, {len(missing)} missing, "
                  f"{el:.0f}s, ETA {el/i*(len(cal)-i):.0f}s)")
        if not (CACHE / f"{d.strftime('%Y%m%d')}.json").exists():
            time.sleep(0.25)                                                   # be gentle only on live pulls

    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("No F&O data fetched — check NSE access / URLs.")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    df["oi_total_change"] = df["oi_total"].diff().astype("Int64")

    # ── expiry rollover flag: days where the near contract expired (near_expiry == trade date) ──
    def _exp_is_today(row):
        try:
            return pd.to_datetime(row["near_expiry"]).date() == row["date"]
        except Exception:
            return False
    df["is_expiry_day"] = df.apply(_exp_is_today, axis=1)

    out = df[["date", "oi_total", "oi_total_change", "n_expiries", "near_expiry", "is_expiry_day"]].copy()
    out.rename(columns={"oi_total": "combined_oi_all_expiries",
                        "oi_total_change": "combined_oi_change_vs_prev"}, inplace=True)
    out.to_csv(OUT_CSV, index=False)

    # ── STEP 6 sanity ──
    got = set(df["date"]); cal_missing = sorted(set(cal) - got)
    exp_days = df[df["is_expiry_day"]]["date"].tolist()
    print("\n" + "=" * 68)
    print("SANITY CHECK — Nifty futures COMBINED total OI (all expiries)")
    print("=" * 68)
    print(f"  Saved to             : {OUT_CSV}")
    print(f"  Columns              : {list(out.columns)}")
    print(f"  Date range covered   : {df['date'].min()} → {df['date'].max()}")
    print(f"  Trading days (OI)    : {len(df):,}   (stock calendar: {len(cal):,})")
    print(f"  Missing days vs stocks: {len(cal_missing)}"
          + ("  -> " + ", ".join(str(x) for x in cal_missing[:15]) + (" …" if len(cal_missing) > 15 else "") if cal_missing else ""))
    print(f"  n_expiries/day range : {int(df['n_expiries'].min())}–{int(df['n_expiries'].max())} "
          f"(Nifty index futures: typically 3 monthly)")
    print(f"  OI units             : NSE bhavcopy OPEN_INT/OpnIntrst field AS REPORTED "
          f"(number of UNITS, i.e. contracts × lot size — NOT contracts). Label accordingly.")
    print(f"  Combined OI range    : {df['oi_total'].min():,} → {df['oi_total'].max():,}")
    print(f"  Sample (first 3)     : " + "; ".join(
        f"{r.date} oi={r.combined_oi_all_expiries:,} nexp={r.n_expiries}" for r in out.head(3).itertuples()))
    print(f"  Expiry (rollover) days flagged: {len(exp_days)}")
    if exp_days:
        print("     " + ", ".join(str(x) for x in exp_days[:18]) + (" …" if len(exp_days) > 18 else ""))
    print("=" * 68)


if __name__ == "__main__":
    main()
