# -*- coding: utf-8 -*-
"""inspect_raw_filings.py — pull RAW, UNFILTERED BSE 'Result' category filings for the 6 flagged
(symbol, quarter) events and print every filing inside the quarter window so the true results-
declaration filing can be identified by eye. No filtering, no writing to the dataset.
"""
import sys, time
from pathlib import Path
from datetime import datetime
import requests, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
BSE_H = {"User-Agent": UA, "Accept": "application/json", "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"}

# (symbol, scrip_code, quarter, window_from, window_to)
TARGETS = [
    ("POLYCAB", 542652, "Q3FY23", "20230101", "20230215"),
    ("RVNL",    542649, "Q1FY23", "20220701", "20220831"),
    ("YESBANK",  532648, "Q3FY23", "20230101", "20230215"),
    ("GAIL",     532155, "Q4FY24", "20240501", "20240615"),
    ("INDIGO",   539448, "Q4FY24", "20240501", "20240615"),
    ("CONCOR",   531344, "Q2FY23", "20221101", "20221130"),
]


def fetch(s, code, frm, to):
    out = []
    for pageno in range(1, 25):
        u = (f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno={pageno}"
             f"&strCat=Result&strPrevDate={frm}&strScrip={code}&strSearch=P&strToDate={to}&strType=C&subcategory=-1")
        r = s.get(u, timeout=25)
        if r.status_code != 200:
            break
        j = r.json(); tbl = j.get("Table", []) if isinstance(j, dict) else []
        out.extend(tbl)
        rowcnt = (j.get("Table1") or [{}])[0].get("ROWCNT", len(out)) if isinstance(j, dict) else len(out)
        if not tbl or len(out) >= rowcnt:
            break
        time.sleep(0.2)
    return out


def main():
    s = requests.Session(); s.headers.update(BSE_H); s.get("https://www.bseindia.com/", timeout=15)
    for sym, code, q, frm, to in TARGETS:
        recs = fetch(s, code, frm, to)
        rows = []
        for rec in recs:
            an = (rec.get("DT_TM") or rec.get("NEWS_DT") or "").strip()
            try:
                dt = datetime.fromisoformat(an)
            except Exception:
                continue
            rows.append((dt, (rec.get("NEWSSUB") or rec.get("HEADLINE") or "").strip()))
        rows.sort()
        print("\n" + "=" * 100)
        print(f"{sym} {q}  (scrip {code}, window {frm}-{to}) — {len(rows)} raw Result-category filings")
        print("=" * 100)
        for dt, sub in rows:
            print(f"  {dt.strftime('%Y-%m-%d %H:%M:%S')}  |  {sub[:88]}")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
