# -*- coding: utf-8 -*-
"""build_nifty500_announcements.py — ADDITIVE BSE results-announcement pull for the Nifty-500 names NOT
already in the reconciled F&O set (~294 non-F&O). Same approach as F&O (strCat=Result, symbol->scrip via
BSE ListofScripData+ISIN, earliest genuine filing per (scrip,date), session by DT_TM) but with the
RECONCILED/improved declaration filter (exclude RPT/Reg-23(9); keep result/UFR/Reg-33/'quarter ended').
Writes: the non-F&O file, a COMBINED file (reconciled F&O + non-F&O) for the backtest, and a review list
(unmapped symbols + ambiguous filings). Does NOT overwrite the reconciled F&O dataset.
"""
import sys, time
from pathlib import Path
from datetime import datetime
import requests, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import fetch_bse_results_announcements as fb

UNIV = rb.BASE / "data" / "nifty500_universe.csv"
FO_CSV = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"   # reconciled F&O baseline
DAILY = rb.BASE / "data" / "daily_ohlcv_vwapclose.parquet"
OUTDIR = rb.RESULTS / "bse_results_announcements"
NONFO_CSV = OUTDIR / "bse_results_announcements_nifty500_nonfo.csv"
COMB_CSV = OUTDIR / "bse_results_announcements_combined.csv"
REVIEW_CSV = OUTDIR / "nifty500_pull_review.csv"
FROM, TO = "20220101", "20260812"

RES_KW = ("result", "financial", "audited", "unaudited", "outcome")
PERIOD_KW = ("quarter ended", "quarter and half year", "quarter & half year", "half year ended",
             "half-year ended", "year ended", "nine months ended", "annual account")
EXCLUDE_KW = ("related party", "related-party", "23(9)", "regulation 23", "reg 23", "corporate bond",
              "ease of doing business", "large corporate", "letter dated", "meeting update",
              "postal ballot", "analyst", "investor meet", "earnings call", "transcript", "newspaper", "advertisement")


def reg33(low):
    return "33" in low and any(t in low for t in ("regulation 33", "reg 33", "reg. 33", "and 33", "& 33", "33 of", "30, 33", "30,33", "33 and", ", 33"))


def strong(low):
    return any(k in low for k in RES_KW) or "ufr" in low or reg33(low)


def fetch(s, code):
    out = []
    for pg in range(1, 16):
        u = (f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno={pg}"
             f"&strCat=Result&strPrevDate={FROM}&strScrip={code}&strSearch=P&strToDate={TO}&strType=C&subcategory=-1")
        for a in range(3):
            try:
                r = s.get(u, timeout=25)
                if r.status_code == 200:
                    j = r.json(); tbl = j.get("Table", []) if isinstance(j, dict) else []
                    out.extend(tbl)
                    rc = (j.get("Table1") or [{}])[0].get("ROWCNT", len(out)) if isinstance(j, dict) else len(out)
                    if not tbl or len(out) >= rc:
                        return out, "ok"
                    break
                if r.status_code in (429, 503):
                    time.sleep(1.5 * (a + 1)); continue
                return out, f"http_{r.status_code}"
            except Exception:
                time.sleep(1.2 * (a + 1))
        else:
            return out, "error"
    return out, "ok"


def main():
    u = pd.read_csv(UNIV)
    need = u[~u["has_bse_announcements"]].copy()                 # non-F&O / not-yet-pulled names
    print(f"Nifty500 names needing announcement pull: {len(need)}", flush=True)

    tdays = sorted(pd.read_parquet(DAILY, columns=["date"])["date"].dt.date.unique())
    tset = set(tdays); tarr = np.array(tdays)
    def next_td(d):
        i = int(np.searchsorted(tarr, d, "right"))
        return tarr[i] if i < len(tarr) else None

    s = requests.Session(); s.headers.update(fb.BSE_H); s.get("https://www.bseindia.com/", timeout=15)
    by_sym, by_isin = fb.bse_scrip_master(s)
    try:
        isin_map = fb.nse_isins()
    except Exception:
        isin_map = {}
    print(f"BSE scrip master {len(by_sym):,} | NSE ISIN {len(isin_map):,}", flush=True)

    rows = []; review = []; mapped = 0; t0 = time.time()
    comp_of = dict(zip(need["symbol"], need["company"]))
    for si, sym in enumerate(need["symbol"], 1):
        code = by_sym.get(sym) or by_isin.get(isin_map.get(sym, "__none__"))
        if not code:
            review.append({"symbol": sym, "issue": "scrip_code_mapping_failed"}); continue
        mapped += 1
        recs, status = fetch(s, code)
        if status != "ok" and not recs:
            review.append({"symbol": sym, "issue": f"fetch_{status}"})
        best = {}
        for rec in recs:
            an = (rec.get("DT_TM") or rec.get("NEWS_DT") or "").strip()
            try:
                dt = datetime.fromisoformat(an)
            except Exception:
                try:
                    dt = datetime.strptime(an[:10], "%Y-%m-%d")
                except Exception:
                    continue
            sub = (rec.get("NEWSSUB") or rec.get("HEADLINE") or "").strip(); low = sub.lower()
            st = strong(low); wk = any(k in low for k in PERIOD_KW)
            if "intimation" in low and "board meeting" in low and not (st or wk):
                continue
            if st:
                pass
            elif any(x in low for x in EXCLUDE_KW):
                continue
            elif wk:
                pass
            else:
                review.append({"symbol": sym, "issue": "ambiguous_headline", "detail": f"{dt:%Y-%m-%d %H:%M} | {sub[:60]}"}); continue
            d0 = dt.date()
            if d0 not in best or dt < best[d0][0]:
                best[d0] = (dt, sub, rec.get("CATEGORYNAME") or "Result")
        for d0, (dt, sub, cat) in best.items():
            sess = fb.classify(dt); is_td = d0 in tset
            if sess == "post_market" or not is_td:
                rdate, rsess = next_td(d0), "next_trading_day"
            else:
                rdate, rsess = d0, "same_day"
            rows.append({"scrip_code": code, "symbol": sym, "company": comp_of.get(sym, ""), "quarter": fb.quarter_label(dt),
                         "announcement_date": dt.strftime("%d-%m-%Y"), "announcement_time": dt.strftime("%H:%M:%S"),
                         "announcement_datetime": dt.strftime("%d-%m-%Y %H:%M"), "announcement_session": sess,
                         "is_trading_day": str(is_td), "reaction_session": rsess,
                         "reaction_session_date": rdate.strftime("%d-%m-%Y") if rdate is not None else "",
                         "headline": sub[:80], "category": cat, "source": "BSE (nifty500 pull 2026-08-16)", "as_of": "2026-08-16"})
        time.sleep(0.35)
        if si % 30 == 0 or si == len(need):
            print(f"  {si}/{len(need)} | mapped {mapped} | rows {len(rows):,} | review {len(review)} | {time.time()-t0:.0f}s", flush=True)

    nonfo = pd.DataFrame(rows).sort_values(["announcement_date", "symbol"]).reset_index(drop=True)
    nonfo.to_csv(NONFO_CSV, index=False)
    fo = pd.read_csv(FO_CSV, dtype=str)
    comb = pd.concat([fo, nonfo.astype(str)], ignore_index=True)
    comb.to_csv(COMB_CSV, index=False)
    REV = pd.DataFrame(review); REV.to_csv(REVIEW_CSV, index=False)

    n_unmapped = sum(1 for r in review if r["issue"] == "scrip_code_mapping_failed")
    cov = nonfo.groupby("symbol")["quarter"].nunique()
    thin = cov[cov <= 4]                                         # <=4 quarters over 4.5yrs = suspiciously thin
    print("\n" + "=" * 80 + "\nNIFTY500 NON-F&O ANNOUNCEMENT PULL (additive)\n" + "=" * 80)
    print(f"names needing pull   : {len(need)}")
    print(f"  scrip-code mapped  : {mapped}  | mapping FAILED: {n_unmapped}")
    print(f"  with >=1 filing    : {nonfo['symbol'].nunique()}")
    print(f"non-F&O result rows  : {len(nonfo):,}")
    print(f"combined dataset     : {len(comb):,} rows ({fo['symbol'].nunique()} F&O + {nonfo['symbol'].nunique()} non-F&O symbols)")
    print(f"review items         : {len(review)} (unmapped {n_unmapped} + ambiguous {len(review)-n_unmapped-sum(1 for r in review if r['issue'].startswith('fetch'))})")
    print(f"\nDATA-QUALITY: {len(thin)} non-F&O names have <=4 quarters of results (thin coverage): {list(thin.index[:20])}")
    print(f"\nSaved -> {NONFO_CSV.name}, {COMB_CSV.name}, {REVIEW_CSV.name}")


if __name__ == "__main__":
    main()
