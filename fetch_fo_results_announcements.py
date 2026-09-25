# -*- coding: utf-8 -*-
"""fetch_fo_results_announcements.py — NSE F&O stocks' quarterly RESULTS announcements WITH TIMESTAMP,
classified pre/during/post market + reaction_session. Source: NSE corporate-announcements API (an_dt =
exchange filing datetime). Universe: fo_mktlots.csv (current F&O membership -> POINT-IN-TIME caveat).
Results = desc containing 'Financial Result'. Session cutoffs: pre<09:15, during 09:15-15:30, post>15:30.
reaction_session: pre/during -> same day; post or weekend/holiday -> next trading day. Saves incrementally.
"""
import sys, io, time, re
from pathlib import Path
from datetime import datetime
import requests
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "fo_results_announcements"
OUT_CSV = OUTDIR / "fo_results_announcements.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
YEARS = [2022, 2023, 2024, 2025, 2026]
INDEX_SYMS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI", "NIFTYIT", "NIFTYINFRA"}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
ASOF = datetime.now().strftime("%Y-%m-%d")


def new_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
    s.get("https://www.nseindia.com/", timeout=15)
    s.get("https://www.nseindia.com/companies-listing/corporate-filings-announcements", timeout=15)
    return s


def fo_universe(s):
    r = s.get("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv", timeout=20)
    df = pd.read_csv(io.StringIO(r.text)); df.columns = [c.strip() for c in df.columns]
    col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
    syms = [str(x).strip().upper() for x in df[col] if str(x).strip() and str(x).strip().upper() != "SYMBOL"]
    return [x for x in dict.fromkeys(syms) if x not in INDEX_SYMS]


def fetch_symbol_year(s, sym, y):
    url = (f"https://www.nseindia.com/api/corporate-announcements?index=equities"
           f"&from_date=01-01-{y}&to_date=31-12-{y}&symbol={sym}")
    for attempt in range(3):
        try:
            r = s.get(url, timeout=25)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def classify(dt):
    m = dt.hour * 60 + dt.minute
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        return "time_unknown"
    if m < 555:
        return "pre_market"
    if m <= 930:
        return "during_market"
    return "post_market"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tdays = sorted(pd.read_parquet(DAILY, columns=["date"])["date"].dt.date.unique())
    tset = set(tdays); tarr = np.array(tdays)
    def next_trading_day(d):
        i = int(np.searchsorted(tarr, d, "left"))
        while i < len(tarr) and tarr[i] <= d:
            i += 1
        return tarr[i] if i < len(tarr) else None

    s = new_session()
    syms = fo_universe(s)
    print(f"F&O universe: {len(syms)} stocks (index symbols removed) | as-of {ASOF}", flush=True)

    rows = []; done = 0; t0 = time.time()
    for si, sym in enumerate(syms, 1):
        if si % 40 == 0:
            s = new_session()                                    # refresh cookies
        recs = []
        for y in YEARS:
            d = fetch_symbol_year(s, sym, y)
            if d:
                recs.extend(d)
            time.sleep(0.5)
        seen = set()
        for rec in recs:
            desc = (rec.get("desc") or "").strip()
            txt = (rec.get("attchmntText") or "").lower()
            dl = desc.lower()
            # 2022-24 label = "Financial Result Updates"; 2025+ results moved under "Outcome of Board
            # Meeting" (financials in the attachment) — catch both, precisely.
            is_result = ("financial result" in dl) or \
                ("board meeting" in dl and ("financial result" in txt or "unaudited financial" in txt
                                            or "audited financial" in txt or "financial results" in txt))
            if not is_result:
                continue
            an = (rec.get("an_dt") or "").strip()
            if not an:
                continue
            try:
                dt = datetime.strptime(an, "%d-%b-%Y %H:%M:%S")
            except Exception:
                try:
                    dt = datetime.strptime(an[:11], "%d-%b-%Y")
                except Exception:
                    continue
            key = (sym, dt.date())
            if key in seen:
                continue
            seen.add(key)
            sess = classify(dt)
            adate = dt.date()
            is_trading_day = adate in tset
            if sess in ("post_market", "time_unknown") or not is_trading_day:
                rsess_date = next_trading_day(adate); rsess = "next_trading_day"
            else:
                rsess_date = adate; rsess = "same_day"
            mth = dt.month
            q = "Q4(JFM)" if mth in (4, 5) else "Q1(AMJ)" if mth in (7, 8) else "Q2(JAS)" if mth in (10, 11) else "Q3(OND)" if mth in (1, 2) else f"M{mth}"
            rows.append({"symbol": sym, "quarter_guess": q, "announcement_date": adate.strftime("%Y-%m-%d"),
                         "announcement_time": dt.strftime("%H:%M:%S") if sess != "time_unknown" else "",
                         "announcement_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                         "announcement_session": sess, "is_trading_day": is_trading_day,
                         "reaction_session": rsess,
                         "reaction_session_date": rsess_date.strftime("%Y-%m-%d") if rsess_date is not None else "",
                         "desc": desc, "source": "NSE corporate-announcements (an_dt)", "as_of": ASOF})
        done += 1
        if rows:                                                 # incremental save
            pd.DataFrame(rows).sort_values(["announcement_date", "symbol"]).to_csv(OUT_CSV, index=False)
        if si % 20 == 0 or si == len(syms):
            print(f"  {si}/{len(syms)} stocks | {len(rows):,} results-announcements | {time.time()-t0:.0f}s", flush=True)

    T = pd.DataFrame(rows).sort_values(["announcement_date", "symbol"]).reset_index(drop=True)
    T.to_csv(OUT_CSV, index=False)

    # summary
    sess_dist = T["announcement_session"].value_counts()
    yr = T.assign(y=pd.to_datetime(T["announcement_date"]).dt.year).groupby(["y", "announcement_session"]).size().unstack(fill_value=0)
    n_time_unknown = int((T["announcement_session"] == "time_unknown").sum())
    n_weekend = int((~T["is_trading_day"]).sum())
    with pd.ExcelWriter(OUTDIR / "fo_results_announcements_summary.xlsx", engine="openpyxl") as w:
        T.to_excel(w, sheet_name="announcements", index=False)
        sess_dist.rename("n").reset_index().to_excel(w, sheet_name="session_distribution", index=False)
        yr.reset_index().to_excel(w, sheet_name="session_by_year", index=False)
        T["reaction_session"].value_counts().rename("n").reset_index().to_excel(w, sheet_name="reaction_session", index=False)

    print("\n" + "=" * 90 + "\nF&O RESULTS ANNOUNCEMENTS — SESSION CLASSIFICATION\n" + "=" * 90)
    print(f"total results-announcements: {len(T):,} | distinct stocks: {T['symbol'].nunique()}")
    print("\nsession distribution:"); print(sess_dist.to_string())
    print(f"\npct post_market: {round((T['announcement_session']=='post_market').mean()*100,1)}% | "
          f"during_market: {round((T['announcement_session']=='during_market').mean()*100,1)}% | "
          f"pre_market: {round((T['announcement_session']=='pre_market').mean()*100,1)}%")
    print(f"\nDATA QUALITY: time_unknown (date but no time): {n_time_unknown} | "
          f"weekend/holiday announcements (->next trading day): {n_weekend}")
    print("\nsession by year:"); print(yr.to_string())
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
