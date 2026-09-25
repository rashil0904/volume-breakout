# -*- coding: utf-8 -*-
"""verify_bse_announcements.py — VERIFICATION ONLY. Fresh-pull BSE result-declaration filings for the
F&O stocks already in results/bse_results_announcements/bse_results_announcements.csv, and diff against
the stored values per (symbol, quarter). Outputs a discrepancy report ONLY — does NOT overwrite stored
data. Same BSE approach: strCat=Result (declaration, not board-meeting notice), earliest filing per
(symbol,date) then per (symbol,quarter), session by DT_TM time.
"""
import sys, time
from pathlib import Path
from datetime import datetime
import requests, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

STORED = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
OUTDIR = rb.RESULTS / "bse_verify"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
BSE_H = {"User-Agent": UA, "Accept": "application/json", "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"}
FROM, TO = "20220101", "20260812"


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


def fetch_results(s, code):
    out = []
    for pageno in range(1, 25):          # heavy filers (PSUs/banks) exceed 7 pages of Result-category items
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
                        return out, "ok"
                    break
                if r.status_code in (429, 503):
                    time.sleep(1.5 * (attempt + 1)); continue
                return out, f"http_{r.status_code}"
            except Exception as e:
                time.sleep(1.2 * (attempt + 1)); last = str(e)[:60]
        else:
            return out, f"error:{last if 'last' in dir() else '?'}"
    return out, "ok"


RES_KW = ("result", "financial", "audited", "unaudited", "outcome")
# period phrasings that denote a results filing without the word "result"
PERIOD_KW = ("quarter ended", "quarter and half year", "quarter & half year", "half year ended",
             "half-year ended", "year ended", "nine months ended", "annual account")
# known NON-result filings that also live under BSE "Result" category -> hard exclude (checked FIRST)
EXCLUDE_KW = ("related party", "related-party", "23(9)", "regulation 23", "reg 23", "corporate bond",
              "ease of doing business", "large corporate", "letter dated", "meeting update",
              "postal ballot", "analyst", "investor meet", "earnings call", "transcript", "newspaper", "advertisement")


def reg33(low):
    if "33" not in low:
        return False                                        # SEBI LODR Reg 33 = the financial-results regulation
    return any(t in low for t in ("regulation 33", "reg 33", "reg. 33", "and 33", "& 33", "33 of",
                                  "30, 33", "30,33", "33 and", ", 33"))


def strong_result(low):
    """unambiguous results-declaration signal -> overrides the exclude list."""
    return any(k in low for k in RES_KW) or ("ufr" in low) or reg33(low)


def parse_fresh(recs):
    """recs -> dict (quarter)-> earliest (datetime, headline). Precedence: board-meeting NOTICE dropped;
    STRONG results signal (result/financial/UFR/Reg33) kept even if it also carries an exclude word;
    else EXCLUDE non-results (RPT / Reg 23(9) / bond / meeting-updates); else WEAK 'quarter ended' phrasing
    kept; else logged to review. Then earliest-per-day, earliest-per-quarter."""
    by_day = {}; ambiguous = []; excluded = []
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
        low = sub.lower()
        strong = strong_result(low)
        weak = any(k in low for k in PERIOD_KW)
        if "intimation" in low and "board meeting" in low and not (strong or weak):
            continue                                        # pure board-meeting NOTICE -> exclude
        if strong:
            pass                                            # keep: unambiguous results declaration
        elif any(x in low for x in EXCLUDE_KW):             # non-results under Result cat (RPT / Reg 23(9) / etc.)
            excluded.append((dt.strftime("%Y-%m-%d %H:%M:%S"), sub[:70])); continue
        elif weak:
            pass                                            # keep: 'quarter ended' style results headline
        else:                                               # under Result cat but not a recognizable declaration
            ambiguous.append((dt.strftime("%Y-%m-%d %H:%M:%S"), sub[:70]))
            continue                                        # -> log to review, DO NOT pollute the event set
        d0 = dt.date()
        if d0 not in by_day or dt < by_day[d0][0]:
            by_day[d0] = (dt, sub)
    by_q = {}
    for d0, (dt, sub) in by_day.items():
        q = quarter_label(dt)
        if q not in by_q or dt < by_q[q][0]:
            by_q[q] = (dt, sub)
    return by_q, ambiguous, excluded


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(STORED)
    # rebuild stored timestamp at FULL precision: announcement_date is DD-MM-YYYY, announcement_time has seconds
    st["stored_dt"] = pd.to_datetime(st["announcement_date"].astype(str).str.strip() + " " + st["announcement_time"].astype(str).str.strip(),
                                     format="%d-%m-%Y %H:%M:%S", errors="coerce")
    st = st.dropna(subset=["stored_dt"])
    # stored: earliest declaration per (symbol, quarter)
    st_e = st.sort_values("stored_dt").groupby(["symbol", "quarter"]).first().reset_index()
    scrip_of = dict(zip(st["symbol"], st["scrip_code"]))
    syms = sorted(st["symbol"].unique())
    print(f"F&O symbols in stored dataset: {len(syms)} | stored (symbol,quarter) events: {len(st_e)}", flush=True)

    s = requests.Session(); s.headers.update(BSE_H); s.get("https://www.bseindia.com/", timeout=15)
    fresh = {}; review = []
    t0 = time.time()
    for i, sym in enumerate(syms, 1):
        code = scrip_of.get(sym)
        if pd.isna(code):
            review.append({"symbol": sym, "issue": "no_scrip_code_in_stored"}); continue
        recs, status = fetch_results(s, int(code))
        if status != "ok" and not recs:
            review.append({"symbol": sym, "issue": f"fetch_{status}"})
        by_q, amb, excl = parse_fresh(recs)
        for q, (dt, sub) in by_q.items():
            fresh[(sym, q)] = (dt, sub)
        for a in amb:
            review.append({"symbol": sym, "issue": "ambiguous_headline_under_Result", "detail": f"{a[0]} | {a[1]}"})
        for a in excl:
            review.append({"symbol": sym, "issue": "excluded_nonresult", "detail": f"{a[0]} | {a[1]}"})
        time.sleep(0.3)
        if i % 40 == 0 or i == len(syms):
            print(f"  {i}/{len(syms)} scrips fetched | fresh events {len(fresh):,} | {time.time()-t0:.0f}s", flush=True)

    # compare per (symbol, quarter)
    stored_keys = {(r.symbol, r.quarter): r for r in st_e.itertuples()}
    all_keys = set(stored_keys) | set(fresh)
    rows_ok = rows_disc = 0; disc = []; miss = []
    for k in sorted(all_keys):
        sym, q = k
        sold = stored_keys.get(k); new = fresh.get(k)
        if sold is not None and new is not None:
            old_dt = pd.Timestamp(sold.stored_dt)
            new_dt = new[0]
            same_date = old_dt.date() == new_dt.date()
            same_time = old_dt.strftime("%H:%M:%S") == new_dt.strftime("%H:%M:%S")
            if same_date and same_time:
                rows_ok += 1
            else:
                rows_disc += 1
                dtag = "date+time" if (not same_date and not same_time) else ("date" if not same_date else "time")
                disc.append({"symbol": sym, "quarter": q,
                             "old_date": old_dt.strftime("%Y-%m-%d"), "old_time": old_dt.strftime("%H:%M:%S"),
                             "old_session": sold.announcement_session,
                             "new_date": new_dt.strftime("%Y-%m-%d"), "new_time": new_dt.strftime("%H:%M:%S"),
                             "new_session": classify(new_dt),
                             "diff": dtag, "day_delta": (new_dt.date() - old_dt.date()).days,
                             "new_headline": new[1][:60]})
        elif sold is not None:
            od = pd.Timestamp(sold.stored_dt)
            miss.append({"symbol": sym, "quarter": q, "missing_in": "fresh_pull",
                         "stored_date": od.strftime("%Y-%m-%d"), "stored_time": od.strftime("%H:%M:%S"),
                         "stored_headline": str(getattr(sold, "headline", ""))[:60]})
        else:
            miss.append({"symbol": sym, "quarter": q, "missing_in": "stored_data",
                         "fresh_date": new[0].strftime("%Y-%m-%d"), "fresh_time": new[0].strftime("%H:%M:%S"), "fresh_headline": new[1][:60]})

    DISC = pd.DataFrame(disc); MISS = pd.DataFrame(miss); REV = pd.DataFrame(review)
    summ = pd.DataFrame([{"total_events_checked": len(all_keys), "stored_events": len(stored_keys), "fresh_events": len(fresh),
                          "matching_ok": rows_ok, "discrepancies": rows_disc,
                          "missing_in_fresh": int((MISS["missing_in"] == "fresh_pull").sum()) if len(MISS) else 0,
                          "missing_in_stored": int((MISS["missing_in"] == "stored_data").sum()) if len(MISS) else 0,
                          "review_list_items": len(REV), "symbols_checked": len(syms)}])

    with pd.ExcelWriter(OUTDIR / "bse_verification_report.xlsx", engine="openpyxl") as w:
        summ.to_excel(w, sheet_name="summary", index=False)
        (DISC if len(DISC) else pd.DataFrame([{"note": "no date/time discrepancies"}])).to_excel(w, sheet_name="discrepancies", index=False)
        (MISS if len(MISS) else pd.DataFrame([{"note": "none missing"}])).to_excel(w, sheet_name="missing_either_side", index=False)
        (REV if len(REV) else pd.DataFrame([{"note": "no review items"}])).to_excel(w, sheet_name="review_list", index=False)
    if len(DISC):
        DISC.to_csv(OUTDIR / "discrepancies.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 90 + "\nBSE ANNOUNCEMENT VERIFICATION (fresh vs stored; report only, no overwrite)\n" + "=" * 90)
    print(summ.T.to_string(header=False))
    print("\n--- discrepancies (first 15) ---")
    print(DISC.head(15).to_string(index=False) if len(DISC) else "  NONE")
    print("\n--- missing either side (first 10) ---")
    print(MISS.head(10).to_string(index=False) if len(MISS) else "  NONE")
    print(f"\n--- review list: {len(REV)} items ---")
    print(REV.head(8).to_string(index=False) if len(REV) else "  NONE")
    print(f"\nSaved -> {OUTDIR}  (stored dataset NOT modified)")


if __name__ == "__main__":
    main()
