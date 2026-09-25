# -*- coding: utf-8 -*-
"""newly_listed_first_results_backtest.py — "buy newly listed stocks ahead of their FIRST post-listing
quarterly results" backtest, two exit variants (morning open / close).

UNIVERSE: 801 candidates = 418 stocks within the original 1609 whose OWN data starts after 2022-01-10
(genuine IPOs that happened to already qualify for the curated list) + 383 confirmed new listings from
the separate pull (data starts 2025+). Listing date proxied by each stock's own first available 1-min
candle (Upstox generally has data from day 1 of listing for actively-traded names).

RESULTS SOURCE: BSE corporate-announcements (AnnSubCategoryGetData, strCat=Result), symbol->BSE scrip
code via BSE ListofScripData + NSE ISIN fallback, and the SAME board-meeting-notice-vs-actual-declaration
filter used by fetch_bse_results_announcements.py (the PEAD strategy's source) -- reused directly, not
reimplemented differently.

SESSION / REACTION-DAY RULE (as explicitly specified for THIS task -- note this differs from the PEAD
script's own internal convention, which treats during_market same as pre_market; here, per instruction):
  pre_market (<9:15)        -> reaction day = SAME day
  during_market (9:15-15:30) -> reaction day = NEXT trading day
  post_market (>15:30)      -> reaction day = NEXT trading day

ENTRY: BUY at CLOSE of the trading day immediately before the reaction day (T-1).
EXIT variant 1 (MORNING OPEN): SELL at OPEN of the reaction day.
EXIT variant 2 (CLOSING): SELL at CLOSE of the reaction day.

Only the FIRST qualifying result (first BSE result filing strictly after the stock's own listing/first-
data date) is used per stock -- subsequent quarters are not backtested here.
"""
import sys, time
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import fetch_bse_results_announcements as bse

OUTDIR = rb.RESULTS / "newly_listed_first_results"; OUTDIR.mkdir(parents=True, exist_ok=True)
IST = "Asia/Kolkata"


def fetch_results_chunked(s, code, frm=None, to=None):
    """BSE's AnnSubCategoryGetData now caps queries at <=12 months (returns {"Status":false,"Message":
    "Date range cannot exceed 12 months."} for wider ranges, which fetch_results() silently reads as
    zero records since it only looks for a "Table" key). Chunk into 12-month windows to work around this
    -- this affects the underlying PEAD data source too, flagged, not just this script."""
    frm_dt = datetime.strptime(frm or bse.FROM, "%Y%m%d").date()
    to_dt = datetime.strptime(to or bse.TO, "%Y%m%d").date()
    out = []
    cur = frm_dt
    while cur <= to_dt:
        chunk_to = min(pd.Timestamp(cur) + pd.DateOffset(months=11, days=25), pd.Timestamp(to_dt)).date()
        for pageno in range(1, 8):
            u = (f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno={pageno}"
                 f"&strCat=Result&strPrevDate={cur.strftime('%Y%m%d')}&strScrip={code}&strSearch=P"
                 f"&strToDate={chunk_to.strftime('%Y%m%d')}&strType=C&subcategory=-1")
            got_page = False
            for attempt in range(3):
                try:
                    r = s.get(u, timeout=25)
                except Exception:
                    time.sleep(1.2 * (attempt + 1)); continue
                if r.status_code == 200:
                    j = r.json()
                    if not isinstance(j, dict) or j.get("Status") is False:
                        break   # this chunk rejected/empty -- stop paging this chunk
                    tbl = j.get("Table", [])
                    out.extend(tbl)
                    rowcnt = (j.get("Table1") or [{}])[0].get("ROWCNT", len(out))
                    got_page = True
                    if len(tbl) == 0 or len(out) >= rowcnt:
                        break
                    break
                time.sleep(1.2 * (attempt + 1))
            if not got_page:
                break
        cur = chunk_to + pd.Timedelta(days=1)
    return out


def classify_session(dt):
    m = dt.hour * 60 + dt.minute
    if m < 555:
        return "pre_market"
    if m <= 930:
        return "during_market"
    return "post_market"


def load_stock_days(symbol, source):
    folder = rb.BASE / source
    fn = folder / f"{symbol}.parquet"
    if not fn.exists():
        return None
    df = pd.read_parquet(fn, columns=["timestamp", "open", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    df = df.assign(ts=ts, date=ts.dt.date).sort_values("ts")
    return df


def main():
    universe = pd.read_csv(rb.RESULTS / "newly_listed_universe_combined.csv", parse_dates=["first_date"])
    universe["first_date"] = universe["first_date"].dt.date
    print(f"newly-listed candidate universe: {len(universe)}", flush=True)

    # ---- BSE scrip mapping (reusing the PEAD infra) ----
    s = __import__("requests").Session(); s.headers.update(bse.BSE_H); s.get("https://www.bseindia.com/", timeout=15)
    by_sym, by_isin = bse.bse_scrip_master(s)
    print(f"BSE scrip master: {len(by_sym):,} symbols", flush=True)
    try:
        isin_map = bse.nse_isins()
        print(f"NSE ISIN map: {len(isin_map):,}", flush=True)
    except Exception as e:
        isin_map = {}; print("ISIN map failed:", str(e)[:100])

    mapping, unmapped = {}, []
    for sym in universe["symbol"]:
        code = by_sym.get(sym) or by_isin.get(isin_map.get(sym, "__none__"))
        if code:
            mapping[sym] = code
        else:
            unmapped.append(sym)
    print(f"mapped {len(mapping)}/{len(universe)} to BSE codes | unmapped {len(unmapped)}", flush=True)

    # ---- fetch BSE result filings per stock, find FIRST one after listing ----
    # RESUMABLE: every processed symbol (found or none) is appended to a JSONL cache immediately, so an
    # interrupted run (session end, manual pause) loses nothing -- re-running skips already-cached symbols.
    import json
    CACHE = OUTDIR / "bse_fetch_cache.jsonl"
    first_results = []; no_result_found = []; processed = set(); recheck = set()
    TO_STR = datetime.now().strftime("%Y%m%d")      # BSE window end = today (the reused PEAD module capped it at 2026-08-12)
    RECHECK_FROM = "20260801"                        # earlier windows were already searched up to 2026-08-12
    latest = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line); latest[c["symbol"]] = c          # last line per symbol wins
    for c in latest.values():
        if c["status"] == "none" and c.get("checked_to") != TO_STR:
            recheck.add(c["symbol"]); continue                          # 'none' found under the old 2026-08-12 cap -> re-check 08-01..today
        processed.add(c["symbol"])
        if c["status"] == "found":
            first_results.append({"symbol": c["symbol"], "source": c["source"],
                                  "listing_date": pd.Timestamp(c["listing_date"]).date(), "bse_code": c["bse_code"],
                                  "announcement_datetime": datetime.fromisoformat(c["announcement_datetime"]),
                                  "headline": c["headline"]})
        else:
            no_result_found.append({"symbol": c["symbol"], "listing_date": pd.Timestamp(c["listing_date"]).date(),
                                    "reason": "no_post_listing_result_found"})
    if latest:
        print(f"resuming: {len(processed)} symbols final ({len(first_results)} found, {len(no_result_found)} none) | "
              f"{len(recheck)} earlier-'none' symbols to re-check for 2026-08-01..{TO_STR}", flush=True)

    def cache_write(rec):
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    t0 = time.time()
    for i, row in universe.iterrows():
        sym = row["symbol"]; listing_date = row["first_date"]; source = row["source"]
        code = mapping.get(sym)
        if not code or sym in processed:
            continue
        recs = (fetch_results_chunked(s, code, frm=RECHECK_FROM, to=TO_STR) if sym in recheck
                else fetch_results_chunked(s, code, to=TO_STR))
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
            if "intimation" in sub.lower() and "board meeting" in sub.lower() and not any(
                    k in sub.lower() for k in ["outcome", "result", "financial", "unaudited", "audited"]):
                continue
            d0 = dt.date()
            if d0 <= listing_date:
                continue   # only results AFTER listing qualify
            if d0 not in best or dt < best[d0][0]:
                best[d0] = (dt, sub)
        if not best:
            no_result_found.append({"symbol": sym, "listing_date": listing_date, "reason": "no_post_listing_result_found"})
            cache_write({"symbol": sym, "status": "none", "source": source, "listing_date": str(listing_date), "bse_code": code, "checked_to": TO_STR})
            time.sleep(0.4)
            continue
        first_date = min(best)
        dt, sub = best[first_date]
        first_results.append({"symbol": sym, "source": source, "listing_date": listing_date, "bse_code": code,
                              "announcement_datetime": dt, "headline": sub})
        cache_write({"symbol": sym, "status": "found", "source": source, "listing_date": str(listing_date), "bse_code": code,
                     "announcement_datetime": dt.isoformat(), "headline": sub, "checked_to": TO_STR})
        time.sleep(0.4)
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(universe)} ({time.time()-t0:.0f}s) | first-results found so far: {len(first_results)}", flush=True)
    print(f"BSE scan complete ({time.time()-t0:.0f}s). first-results found: {len(first_results)} | none found: {len(no_result_found)}", flush=True)

    FR = pd.DataFrame(first_results)
    pd.DataFrame(unmapped, columns=["symbol"]).to_csv(OUTDIR / "unmapped_symbols.csv", index=False)
    pd.DataFrame(no_result_found).to_csv(OUTDIR / "no_first_result_found.csv", index=False)
    FR.to_csv(OUTDIR / "first_results_raw.csv", index=False)

    # ---- session classification + reaction-day mapping (per THIS task's explicit rule) ----
    trades_v1, trades_v2, data_issues = [], [], []
    for _, r in FR.iterrows():
        sym, source, dt = r["symbol"], r["source"], r["announcement_datetime"]
        stock_days = load_stock_days(sym, source)
        if stock_days is None or stock_days.empty:
            data_issues.append({"symbol": sym, "reason": "no_price_data"}); continue
        tdays = sorted(stock_days["date"].unique())
        d0 = dt.date()
        sess = classify_session(dt)
        is_td = d0 in set(tdays)

        if sess == "pre_market" and is_td:
            reaction_day = d0
        else:
            # during_market, post_market, non-trading-day announcement, or pre_market on a non-trading day
            # all resolve to the NEXT trading day, per this task's explicit rule
            later = [d for d in tdays if d > d0]
            reaction_day = later[0] if later else None
        if reaction_day is None:
            data_issues.append({"symbol": sym, "reason": "no_trading_day_after_announcement"}); continue

        earlier = [d for d in tdays if d < reaction_day]
        if not earlier:
            data_issues.append({"symbol": sym, "reason": "no_trading_day_before_reaction_day"}); continue
        entry_date = earlier[-1]

        entry_rows = stock_days[stock_days["date"] == entry_date]
        exit_rows = stock_days[stock_days["date"] == reaction_day]
        if entry_rows.empty or exit_rows.empty:
            data_issues.append({"symbol": sym, "reason": "missing_entry_or_exit_day_candles"}); continue
        entry_price = float(entry_rows.sort_values("ts")["close"].iloc[-1])
        exit_open = float(exit_rows.sort_values("ts")["open"].iloc[0])
        exit_close = float(exit_rows.sort_values("ts")["close"].iloc[-1])
        if entry_price <= 0:
            data_issues.append({"symbol": sym, "reason": "invalid_entry_price"}); continue

        base_row = {"symbol": sym, "listing_date": r["listing_date"], "first_result_date": d0,
                    "session": sess, "reaction_day": reaction_day, "entry_date": entry_date,
                    "entry_price": entry_price, "headline": r["headline"]}
        ret_open = (exit_open - entry_price) / entry_price * 100
        ret_close = (exit_close - entry_price) / entry_price * 100
        trades_v1.append({**base_row, "exit_price": exit_open, "return_pct": round(ret_open, 3),
                          "holding_days": (reaction_day - entry_date).days})
        trades_v2.append({**base_row, "exit_price": exit_close, "return_pct": round(ret_close, 3),
                          "holding_days": (reaction_day - entry_date).days})

    V1 = pd.DataFrame(trades_v1); V2 = pd.DataFrame(trades_v2); DI = pd.DataFrame(data_issues)
    print(f"\nTRADES built: variant1(open)={len(V1)} variant2(close)={len(V2)} | data issues excluded={len(DI)}", flush=True)

    def summarize(T, label):
        if T.empty:
            print(f"\n[{label}] NO TRADES"); return {}
        win = T["return_pct"] > 0
        s = {"variant": label, "n_trades": len(T), "win_rate_pct": round(win.mean() * 100, 2),
             "total_return_pct_sum": round(T["return_pct"].sum(), 2), "avg_return_pct": round(T["return_pct"].mean(), 3),
             "median_return_pct": round(T["return_pct"].median(), 3), "std_return_pct": round(T["return_pct"].std(), 3),
             "max_gain_pct": round(T["return_pct"].max(), 2), "max_loss_pct": round(T["return_pct"].min(), 2),
             "p10": round(T["return_pct"].quantile(0.10), 2), "p90": round(T["return_pct"].quantile(0.90), 2)}
        print(f"\n[{label}] n={s['n_trades']} win%={s['win_rate_pct']} avg={s['avg_return_pct']}% "
              f"median={s['median_return_pct']}% max_gain={s['max_gain_pct']}% max_loss={s['max_loss_pct']}%")
        return s

    s1 = summarize(V1, "MORNING_OPEN_EXIT")
    s2 = summarize(V2, "CLOSING_EXIT")

    with pd.ExcelWriter(OUTDIR / "newly_listed_first_results_backtest.xlsx", engine="openpyxl") as w:
        note = pd.DataFrame([
            {"note": "Universe = 418 recent-IPO stocks already in the original 1609 (own data starts after "
                      "2022-01-10) + 383 confirmed new listings from the separate pull (own data starts 2025+). "
                      "Listing date proxied by each stock's own first available 1-min candle."},
            {"note": "Results source: BSE corporate-announcements (same API/method/board-meeting-exclusion "
                      "filter as fetch_bse_results_announcements.py, the PEAD strategy's source). Only the "
                      "FIRST result filing strictly AFTER the stock's listing date qualifies."},
            {"note": "Reaction-day rule for THIS task (explicitly specified, differs from the PEAD script's "
                      "own internal convention): pre_market->same day; during_market->next trading day; "
                      "post_market->next trading day. Entry = T-1 close (day before reaction day)."},
            {"note": f"Excluded/flagged, not guessed: {len(unmapped)} symbols with no BSE scrip mapping, "
                      f"{len(no_result_found)} with no post-listing result found in BSE history, {len(DI)} "
                      "with unresolvable price data (missing entry/exit day candles etc.)."},
        ])
        note.to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        pd.DataFrame([s1, s2]).to_excel(w, sheet_name="Summary_Comparison", index=False)
        V1.to_excel(w, sheet_name="Trades_MorningOpen", index=False)
        V2.to_excel(w, sheet_name="Trades_Closing", index=False)
        if len(DI):
            DI.to_excel(w, sheet_name="Data_Issues_Excluded", index=False)
        pd.DataFrame(no_result_found).to_excel(w, sheet_name="No_Result_Found", index=False)
        pd.DataFrame(unmapped, columns=["symbol"]).to_excel(w, sheet_name="Unmapped_BSE", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)

    print(f"\nSaved -> {OUTDIR}/newly_listed_first_results_backtest.xlsx")


if __name__ == "__main__":
    main()
