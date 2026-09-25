# -*- coding: utf-8 -*-
"""fetch_bse_results_announcements.py — BSE corporate-announcement RESULT-DECLARATION filings (date+time)
for NSE-F&O stocks, session-classified. Source: BSE AnnSubCategoryGetData (strCat=Result) -> DT_TM is the
filing timestamp. Symbol->BSE-scripcode via BSE ListofScripData (scrip_id / ISIN). Full 2022-2026 range in
one call per scrip. Dedup keeps the EARLIEST filing per (scrip,date). Session pre<9:15/during 9:15-15:30/
post>15:30; reaction_session same-day (pre/during trading day) vs next-trading-day (post / weekend).
"""
import sys, io, time, gzip, json
from pathlib import Path
from datetime import datetime
import requests
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_backtest as rb

OUTDIR = rb.RESULTS / "bse_results_announcements"
OUT_CSV = OUTDIR / "bse_results_announcements.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
INDEX_SYMS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI", "NIFTYIT", "NIFTYINFRA"}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
BSE_H = {"User-Agent": UA, "Accept": "application/json", "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"}
ASOF = datetime.now().strftime("%Y-%m-%d")
FROM, TO = "20220101", "20260812"


def fo_universe():
    s = requests.Session(); s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    s.get("https://www.nseindia.com/", timeout=15)
    r = s.get("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv", timeout=20)
    df = pd.read_csv(io.StringIO(r.text)); df.columns = [c.strip() for c in df.columns]
    col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
    syms = [str(x).strip().upper() for x in df[col] if str(x).strip() and str(x).strip().upper() != "SYMBOL"]
    return [x for x in dict.fromkeys(syms) if x not in INDEX_SYMS]


def bse_scrip_master(s):
    r = s.get("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active", timeout=40)
    j = r.json()
    by_sym, by_isin = {}, {}
    for x in j:
        code = str(x.get("SCRIP_CD", "")).strip(); sid = str(x.get("scrip_id", "")).strip().upper(); isin = str(x.get("ISIN_NUMBER", "")).strip()
        if code and sid and sid not in by_sym:
            by_sym[sid] = code
        if code and isin and isin not in by_isin:
            by_isin[isin] = code
    return by_sym, by_isin


def nse_isins():
    r = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", timeout=60)
    raw = json.load(gzip.GzipFile(fileobj=io.BytesIO(r.content)))
    out = {}
    for i in raw:
        if i.get("segment") == "NSE_EQ":
            sym = str(i.get("trading_symbol", "")).strip().upper()
            ik = i.get("instrument_key", "")
            if sym and "|" in ik:
                out[sym] = ik.split("|")[1]
    return out


def fetch_results(s, code):
    out = []
    for pageno in range(1, 8):
        u = (f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno={pageno}"
             f"&strCat=Result&strPrevDate={FROM}&strScrip={code}&strSearch=P&strToDate={TO}&strType=C&subcategory=-1")
        for attempt in range(3):
            try:
                r = s.get(u, timeout=25)
                if r.status_code == 200:
                    j = r.json(); tbl = j.get("Table", []) if isinstance(j, dict) else []
                    out.extend(tbl)
                    rowcnt = (j.get("Table1") or [{}])[0].get("ROWCNT", len(out)) if isinstance(j, dict) else len(out)
                    if len(tbl) == 0 or len(out) >= rowcnt:
                        return out
                    break
            except Exception:
                time.sleep(1.2 * (attempt + 1))
        else:
            break
    return out


def classify(dt):
    m = dt.hour * 60 + dt.minute
    return "pre_market" if m < 555 else "during_market" if m <= 930 else "post_market"


def quarter_label(dt):
    m, y = dt.month, dt.year
    if m in (1, 2, 3):
        return f"Q3FY{y % 100:02d}"
    if m in (4, 5, 6):
        return f"Q4FY{y % 100:02d}"
    if m in (7, 8, 9):
        return f"Q1FY{(y + 1) % 100:02d}"
    return f"Q2FY{(y + 1) % 100:02d}"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tdays = sorted(pd.read_parquet(DAILY, columns=["date"])["date"].dt.date.unique())
    tset = set(tdays); tarr = np.array(tdays)
    def next_td(d):
        i = int(np.searchsorted(tarr, d, "left"))
        while i < len(tarr) and tarr[i] <= d:
            i += 1
        return tarr[i] if i < len(tarr) else None

    syms = fo_universe(); print(f"F&O universe: {len(syms)} stocks", flush=True)
    s = requests.Session(); s.headers.update(BSE_H); s.get("https://www.bseindia.com/", timeout=15)
    by_sym, by_isin = bse_scrip_master(s); print(f"BSE scrip master: {len(by_sym):,} symbols", flush=True)
    try:
        isin_map = nse_isins(); print(f"NSE ISIN map: {len(isin_map):,}", flush=True)
    except Exception as e:
        isin_map = {}; print("ISIN map failed:", str(e)[:80])

    mapping = {}; unmapped = []
    for sym in syms:
        code = by_sym.get(sym) or by_isin.get(isin_map.get(sym, "__none__"))
        if code:
            mapping[sym] = code
        else:
            unmapped.append(sym)
    print(f"mapped {len(mapping)}/{len(syms)} to BSE codes | unmapped {len(unmapped)}: {unmapped[:15]}", flush=True)

    rows = []; t0 = time.time()
    for si, (sym, code) in enumerate(mapping.items(), 1):
        recs = fetch_results(s, code)
        best = {}
        for rec in recs:
            an = (rec.get("DT_TM") or rec.get("NEWS_DT") or "").strip()
            if not an:
                continue
            try:
                dt = datetime.fromisoformat(an)
            except Exception:
                try:
                    dt = datetime.strptime(an[:10], "%Y-%m-%d")
                except Exception:
                    continue
            sub = (rec.get("NEWSSUB") or rec.get("HEADLINE") or "").strip()
            if "intimation" in sub.lower() and "board meeting" in sub.lower() and not any(k in sub.lower() for k in ["outcome", "result", "financial", "unaudited", "audited"]):
                continue                                              # skip pure board-meeting NOTICE
            d0 = dt.date()
            if d0 not in best or dt < best[d0][0]:                    # keep EARLIEST filing per (scrip,date)
                best[d0] = (dt, sub, rec.get("CATEGORYNAME") or "Result")
        for d0, (dt, sub, cat) in best.items():
            time_known = not (dt.hour == 0 and dt.minute == 0 and dt.second == 0)
            sess = classify(dt) if time_known else "time_unknown"
            is_td = d0 in tset
            if (not time_known) or sess == "post_market" or not is_td:
                rdate, rsess = next_td(d0), "next_trading_day"
            else:
                rdate, rsess = d0, "same_day"
            rows.append({"scrip_code": code, "symbol": sym, "company": "", "quarter": quarter_label(dt),
                         "announcement_date": d0.strftime("%Y-%m-%d"),
                         "announcement_time": dt.strftime("%H:%M:%S") if time_known else "",
                         "announcement_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                         "announcement_session": sess, "is_trading_day": is_td, "reaction_session": rsess,
                         "reaction_session_date": rdate.strftime("%Y-%m-%d") if rdate is not None else "",
                         "headline": sub[:80], "category": cat, "source": "BSE", "as_of": ASOF})
        if rows:
            pd.DataFrame(rows).sort_values(["announcement_date", "symbol"]).to_csv(OUT_CSV, index=False)
        time.sleep(0.4)
        if si % 25 == 0 or si == len(mapping):
            print(f"  {si}/{len(mapping)} scrips | {len(rows):,} result-filings | {time.time()-t0:.0f}s", flush=True)

    T = pd.DataFrame(rows).sort_values(["announcement_date", "symbol"]).reset_index(drop=True)
    T.to_csv(OUT_CSV, index=False)
    sess_dist = T["announcement_session"].value_counts()
    yr = T.assign(y=pd.to_datetime(T["announcement_date"]).dt.year).groupby(["y", "announcement_session"]).size().unstack(fill_value=0)
    n_tu = int((T["announcement_session"] == "time_unknown").sum()); n_we = int((~T["is_trading_day"]).sum())
    with pd.ExcelWriter(OUTDIR / "bse_results_announcements_summary.xlsx", engine="openpyxl") as w:
        T.to_excel(w, sheet_name="announcements", index=False)
        sess_dist.rename("n").reset_index().to_excel(w, sheet_name="session_distribution", index=False)
        yr.reset_index().to_excel(w, sheet_name="session_by_year", index=False)
        if unmapped:
            pd.DataFrame({"unmapped_symbol": unmapped}).to_excel(w, sheet_name="unmapped_fo_symbols", index=False)

    print("\n" + "=" * 88 + "\nBSE F&O RESULTS ANNOUNCEMENTS — SESSION CLASSIFICATION\n" + "=" * 88)
    print(f"total result-filings: {len(T):,} | distinct scrips: {T['symbol'].nunique()} | unmapped F&O: {len(unmapped)}")
    print("\nsession distribution:"); print(sess_dist.to_string())
    print(f"\npct post={round((T['announcement_session']=='post_market').mean()*100,1)}% "
          f"during={round((T['announcement_session']=='during_market').mean()*100,1)}% "
          f"pre={round((T['announcement_session']=='pre_market').mean()*100,1)}%")
    print(f"\nDATA QUALITY: time_unknown {n_tu} | weekend/holiday filings {n_we}")
    print("\nsession by year:"); print(yr.to_string())
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
