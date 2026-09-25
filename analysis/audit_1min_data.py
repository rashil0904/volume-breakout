# -*- coding: utf-8 -*-
"""
audit_1min_data.py
==================
Data-completeness AUDIT for the 1-minute master_data parquets. Verifies whether the
1-min candle data is complete across all expected stocks and all trading days — no
missing stocks, no missing days, no missing/partial/duplicate/out-of-session candles.
It ONLY audits and reports; it never fetches or fills.

WHAT DEFINES "EXPECTED" (and its reliability — flagged in the verdict):
  • Expected stocks  = symbols in "Companies List.csv" (the project's intended universe).
                       Symbols present on disk but not in the list are reported as EXTRA
                       (stale), not gaps. No re-check of Upstox resolvability is done here.
  • Trading calendar = DERIVED from the data: a date is a trading day if ≥ TRADING_DAY_MIN_FRAC
                       of the max daily live-stock count has candles. NO NSE holiday file exists
                       in the project, so weekday-vs-holiday classification is APPROXIMATE.
  • Session / candles/day = DERIVED per date via the fullest observed session that day
                       (handles half-day / muhurat special sessions automatically). A normal
                       NSE full day = 09:15..15:29 inclusive = 375 one-min candles. Dates whose
                       fullest session < 375 are flagged SPECIAL and excluded from "partial".

PER STOCK we use its OBSERVED trading life [first_date, last_date]: trading days OUTSIDE that
window are treated as not-yet-listed / delisted (legitimately absent, NOT a fetch gap). A
trading day INSIDE that window with no/partial data IS a gap (bounded by data on both sides).

Outputs (printed summary + results/data_audit/*.csv):
  missing_stock_days.csv, partial_days.csv, duplicates.csv, out_of_session.csv,
  per_stock_coverage.csv, strategy_critical_gaps.csv, candles_per_day_histogram.csv,
  special_sessions.csv, missing_or_extra_stocks.csv
strategy_critical_gaps.csv is prioritized: stock-days missing a timestamp the strategy needs
(09:15 / entry-vol window / 15:00-15:29 VWAP window / 15:15 entry / 15:29 full-day-vol / exits)
— those break specific trades even when the day is otherwise mostly complete.
"""

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_loading as dl

IST = "Asia/Kolkata"
OUTDIR = dl.BASE / "results" / "data_audit"
MASTER_DIR = dl.MASTER_DIR

# full NSE session (minutes-of-day): 09:15 .. 15:29 inclusive = 375 candles
HM_OPEN, HM_CLOSE = 555, 929
FULL_SESSION = set(range(HM_OPEN, HM_CLOSE + 1))
FULL_COUNT = len(FULL_SESSION)                       # 375
TRADING_DAY_MIN_FRAC = 0.30                          # date is a trading day if live ≥ frac × peak

# strategy-critical timestamps
CRIT_SINGLES = {555: "09:15", 585: "09:45", 660: "11:00", 720: "12:00",
                899: "14:59", 915: "15:15", 929: "15:29"}
VWAP_WIN = set(range(900, 930))                      # 15:00 .. 15:29  (VWAP-close, 30 candles)
ENTRY_WIN = set(range(555, 900))                     # 09:15 .. 14:59  (cumulative entry volume)


def hm_lbl(hm):
    return f"{int(hm)//60:02d}:{int(hm)%60:02d}"


def max_contiguous(missing_sorted):
    """Longest run of consecutive minutes in a sorted list of missing hm ints.
    A large contiguous block mid-session is a FETCH-GAP signature; scattered single
    minutes are almost always legitimate no-trade minutes on illiquid stocks."""
    if not missing_sorted:
        return 0
    best = run = 1
    for a, b in zip(missing_sorted, missing_sorted[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


# ════════════════════════════════════════════════════════════════════════════
# PASS 1 — read each parquet once; per (stock, day) counts + anomaly detail
# ════════════════════════════════════════════════════════════════════════════
def scan_all(symbols_on_disk):
    records = []                      # (symbol, date, distinct_in, raw_in, dup, n_out)
    detail_hms = {}                   # (symbol, date) -> set(hm)  [anomalous days only]
    dup_rows = []                     # (symbol, date, hm, n_copies)
    oos_rows = []                     # (symbol, date, timestamp, hm)
    fullest = {}                      # date -> (max_distinct_in, hms set)  fullest session that day

    t0 = time.time()
    for si, sym in enumerate(symbols_on_disk, 1):
        pq = MASTER_DIR / f"{sym}.parquet"
        try:
            raw = pd.read_parquet(pq, columns=["timestamp"])
        except Exception as e:
            print(f"  !! unreadable {sym}: {e}", flush=True)
            continue
        if len(raw) == 0:
            continue
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        date = ts.dt.date.values
        hm = (ts.dt.hour * 60 + ts.dt.minute).values
        df = pd.DataFrame({"date": date, "hm": hm})

        # out-of-session rows
        oos = df[(df["hm"] < HM_OPEN) | (df["hm"] > HM_CLOSE)]
        oos_by_date = dict(tuple(oos.groupby("date"))) if len(oos) else {}

        ins = df[(df["hm"] >= HM_OPEN) & (df["hm"] <= HM_CLOSE)]
        for d, g in ins.groupby("date"):
            hms_all = g["hm"].values
            hset = set(hms_all.tolist())
            distinct_in = len(hset)
            raw_in = len(hms_all)
            dup = raw_in - distinct_in
            n_out = int(len(oos_by_date.get(d, [])))
            records.append((sym, d, distinct_in, raw_in, dup, n_out))

            # fullest session seen that day (defines the expected session)
            cur = fullest.get(d)
            if cur is None or distinct_in > cur[0]:
                fullest[d] = (distinct_in, hset)

            anomalous = (distinct_in != FULL_COUNT) or (dup > 0) or (n_out > 0)
            if anomalous:
                detail_hms[(sym, d)] = hset
            if dup > 0:
                vc = pd.Series(hms_all).value_counts()
                for hmv, c in vc[vc > 1].items():
                    dup_rows.append((sym, d, hm_lbl(int(hmv)), int(c)))
            if n_out > 0:
                for _, r in oos_by_date[d].iterrows():
                    oos_rows.append((sym, d, hm_lbl(int(r["hm"])), int(r["hm"])))

        if si % 200 == 0:
            print(f"  …{si}/{len(symbols_on_disk)} parquets scanned "
                  f"({time.time()-t0:.0f}s, {len(records):,} stock-days)", flush=True)

    rec = pd.DataFrame(records, columns=["symbol", "date", "distinct_in", "raw_in", "dup", "n_out"])
    return rec, detail_hms, dup_rows, oos_rows, fullest


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("1-MINUTE DATA COMPLETENESS AUDIT  (audit-only; no fetch/fill)")
    print("=" * 78)

    # ── expected universe ──
    try:
        csv_map = dl.load_symbols_and_isins_from_csv(dl.COMPANIES_CSV)
        expected = set(csv_map.keys())
        expected_src = f"Companies List.csv ({len(expected)} symbols)"
        have_expected_list = True
    except Exception as e:
        expected = set(); have_expected_list = False
        expected_src = f"UNAVAILABLE ({e}) — auditing against observed symbols only"
    on_disk = sorted(p.stem for p in MASTER_DIR.glob("*.parquet"))
    disk_set = set(on_disk)
    missing_stocks = sorted(expected - disk_set) if have_expected_list else []
    extra_stocks = sorted(disk_set - expected) if have_expected_list else []
    print(f"\nExpected universe : {expected_src}")
    print(f"On disk           : {len(on_disk)} parquets")
    if have_expected_list:
        print(f"  missing (expected, no parquet): {len(missing_stocks)}")
        print(f"  extra (parquet, not in list)  : {len(extra_stocks)}")

    print(f"\nScanning {len(on_disk)} parquets (one read each) …")
    rec, detail_hms, dup_rows, oos_rows, fullest = scan_all(on_disk)
    if rec.empty:
        print("No data scanned — aborting."); return
    dmin, dmax = rec["date"].min(), rec["date"].max()
    print(f"  scanned: {len(rec):,} stock-days | date range {dmin} → {dmax}")

    # ── derive trading calendar + per-date expected session ──
    live_per_day = rec.groupby("date")["symbol"].nunique()
    peak_live = int(live_per_day.max())
    thresh = TRADING_DAY_MIN_FRAC * peak_live
    trading_days = sorted(live_per_day[live_per_day >= thresh].index)
    td_set = set(trading_days)
    nontrading_dates = sorted(d for d in live_per_day.index if d not in td_set)  # sparse dates

    exp_count = {d: len(fullest[d][1]) for d in fullest}
    exp_hms = {d: fullest[d][1] for d in fullest}
    special_dates = sorted(d for d in trading_days if exp_count.get(d, 0) < FULL_COUNT)

    # weekday-vs-holiday (approximate — no NSE holiday file)
    all_cal = pd.date_range(dmin, dmax, freq="D")
    weekdays = [d.date() for d in all_cal if d.weekday() < 5]
    presumed_holidays = sorted(set(weekdays) - td_set)

    print(f"\nTrading calendar (derived): {len(trading_days)} trading days "
          f"[{trading_days[0]} → {trading_days[-1]}], peak live stocks/day = {peak_live}")
    print(f"  special sessions (fullest < 375 candles): {len(special_dates)}")
    print(f"  presumed weekday holidays (approx, no NSE cal): {len(presumed_holidays)}")
    print(f"  sparse non-trading dates present in data: {len(nontrading_dates)}")

    # ── per-stock trading life + present-day map ──
    rec = rec.sort_values(["symbol", "date"])
    life = rec.groupby("symbol")["date"].agg(first="min", last="max")
    present = {(s, d): (di, dup, no) for s, d, di, dup, no in
               rec[["symbol", "date", "distinct_in", "dup", "n_out"]].itertuples(index=False)}
    present_days = {s: set(g["date"]) for s, g in rec.groupby("symbol")}

    # ── per stock × expected trading day: classify ──
    missing_stock_days = []      # (symbol, date, reason)
    partial_days = []            # (symbol, date, actual, expected, n_missing, missing_ts, is_special)
    crit_gaps = []               # strategy-critical
    per_stock = []               # coverage

    for sym in on_disk:
        if sym not in present_days:
            per_stock.append((sym, 0, 0, 0, 0.0, None, None, "no_data"))
            continue
        f, l = life.at[sym, "first"], life.at[sym, "last"]
        exp_days = [d for d in trading_days if f <= d <= l]
        n_exp = len(exp_days)
        pd_set = present_days[sym]
        n_present = n_miss = n_partial = 0
        for d in exp_days:
            ehms = exp_hms.get(d, FULL_SESSION)
            ecnt = exp_count.get(d, FULL_COUNT)
            is_special = d in set(special_dates)
            if d not in pd_set:
                n_miss += 1
                missing_stock_days.append((sym, d, "gap_within_trading_life"))
                # every critical ts that the session has is missing
                miss_singles = {lbl for hmv, lbl in CRIT_SINGLES.items() if hmv in ehms}
                crit_gaps.append((sym, d, True, is_special,
                                  1 if 555 in ehms else 0, 1 if 915 in ehms else 0,
                                  1 if 929 in ehms else 0,
                                  len(VWAP_WIN & ehms), len(ENTRY_WIN & ehms),
                                  ";".join(sorted(miss_singles))))
                continue
            n_present += 1
            di, dup, no = present[(sym, d)]
            hset = detail_hms.get((sym, d))
            if hset is None:                       # clean full 375 → complete
                continue
            missing = ehms - hset
            if missing:
                n_partial += 1
                ms = sorted(missing)
                mc = max_contiguous(ms)
                partial_days.append((sym, d, di, ecnt, len(missing), mc,
                                     ";".join(hm_lbl(h) for h in ms[:60]), is_special))
                # critical breakdown
                miss_singles = {lbl for hmv, lbl in CRIT_SINGLES.items()
                                if hmv in ehms and hmv not in hset}
                vwap_miss = len((VWAP_WIN & ehms) - hset)
                entry_miss = len((ENTRY_WIN & ehms) - hset)
                if miss_singles or vwap_miss or entry_miss:
                    crit_gaps.append((sym, d, False, is_special,
                                      1 if (555 in ehms and 555 not in hset) else 0,
                                      1 if (915 in ehms and 915 not in hset) else 0,
                                      1 if (929 in ehms and 929 not in hset) else 0,
                                      vwap_miss, entry_miss, ";".join(sorted(miss_singles))))
        cov = (n_present / n_exp * 100) if n_exp else 0.0
        # fully complete = present every expected day AND no partials
        per_stock.append((sym, n_exp, n_present, n_miss + n_partial, round(cov, 3),
                          str(f), str(l), "complete" if (n_miss == 0 and n_partial == 0) else "has_gaps"))

    # ── frames ──
    msd = pd.DataFrame(missing_stock_days, columns=["symbol", "date", "reason"])
    pdd = pd.DataFrame(partial_days, columns=["symbol", "date", "actual_candles",
                       "expected_candles", "n_missing", "max_contiguous_missing",
                       "missing_timestamps", "is_special_session"])
    dupd = pd.DataFrame(dup_rows, columns=["symbol", "date", "timestamp", "n_copies"])
    oosd = pd.DataFrame(oos_rows, columns=["symbol", "date", "timestamp", "hm"])
    psc = pd.DataFrame(per_stock, columns=["symbol", "expected_trading_days", "present_days",
                       "gap_days", "coverage_pct", "first_date", "last_date", "status"])
    cg = pd.DataFrame(crit_gaps, columns=["symbol", "date", "entire_day_missing", "is_special_session",
                      "missing_0915", "missing_1515", "missing_1529",
                      "n_vwap_window_missing", "n_entry_window_missing", "missing_critical_singles"])
    # cross-reference: does a critical gap land on an ACTUAL signal day? (those truly break a backtest trade)
    signal_set = set()
    try:
        sig = pd.read_csv(dl.BASE / "results" / "diagnostic_table.csv",
                          usecols=["symbol", "date", "passes_all_three"], parse_dates=["date"])
        signal_set = set(zip(sig.loc[sig["passes_all_three"] == True, "symbol"],
                             sig.loc[sig["passes_all_three"] == True, "date"].dt.date))
    except Exception as e:
        print(f"  (signal cross-ref skipped: {e})")
    if len(cg):
        cg["on_signal_day"] = [(s, d) in signal_set for s, d in zip(cg["symbol"], cg["date"])]
    else:
        cg["on_signal_day"] = pd.Series(dtype=bool)
    # critical gaps that actually break a trade: NON-special days only
    cg_real = cg[~cg["is_special_session"]].copy()
    cg_on_signal = cg[cg["on_signal_day"] & ~cg["is_special_session"]].copy()

    hist = (rec.groupby("distinct_in")["symbol"].count().rename("n_stock_days")
            .reset_index().rename(columns={"distinct_in": "candles_in_day"})
            .sort_values("candles_in_day"))
    spec = pd.DataFrame([{"date": d, "fullest_session_candles": exp_count[d],
                          "n_live_stocks": int(live_per_day[d])} for d in special_dates])
    mos = pd.DataFrame({"symbol": missing_stocks + extra_stocks,
                        "issue": ["expected_missing_parquet"] * len(missing_stocks)
                                 + ["extra_not_in_list"] * len(extra_stocks)})

    # ── days complete for ALL live stocks ──
    exp_stockday_total = int(psc["expected_trading_days"].sum())
    present_stockday_total = int(psc["present_days"].sum())
    # a trading day is "clean for all" if no stock live that day is missing/partial on it
    bad_dates = set(msd["date"]).union(set(pdd.loc[~pdd["is_special_session"], "date"]))
    clean_all_days = [d for d in trading_days if d not in bad_dates]

    n_complete_stocks = int((psc["status"] == "complete").sum())
    n_gap_stocks = int((psc["status"] == "has_gaps").sum())

    # ── save CSVs ──
    msd.to_csv(OUTDIR / "missing_stock_days.csv", index=False)
    pdd.to_csv(OUTDIR / "partial_days.csv", index=False)
    dupd.to_csv(OUTDIR / "duplicates.csv", index=False)
    oosd.to_csv(OUTDIR / "out_of_session.csv", index=False)
    psc.sort_values("coverage_pct").to_csv(OUTDIR / "per_stock_coverage.csv", index=False)
    cg.to_csv(OUTDIR / "strategy_critical_gaps.csv", index=False)
    cg_on_signal.to_csv(OUTDIR / "strategy_critical_gaps_on_signals.csv", index=False)
    hist.to_csv(OUTDIR / "candles_per_day_histogram.csv", index=False)
    spec.to_csv(OUTDIR / "special_sessions.csv", index=False)
    mos.to_csv(OUTDIR / "missing_or_extra_stocks.csv", index=False)

    # ════════════════════════ VERDICT ════════════════════════
    cov_overall = present_stockday_total / exp_stockday_total * 100 if exp_stockday_total else 0
    n_partial_real = int((~pdd["is_special_session"]).sum())
    n_missing_days = len(msd)
    n_dups = len(dupd)
    n_oos = len(oosd)
    n_crit_real = len(cg_real)
    n_crit_signal = len(cg_on_signal)
    # fetch-gap SUSPECTS: real partial days with a long contiguous missing block (not scattered no-trade)
    FETCH_GAP_MIN_BLOCK = 15
    pdd_real = pdd[~pdd["is_special_session"]]
    n_fetchgap_suspect = int((pdd_real["max_contiguous_missing"] >= FETCH_GAP_MIN_BLOCK).sum())

    print("\n" + "=" * 78)
    print("COVERAGE SUMMARY")
    print("=" * 78)
    print(f"  expected stock-days (within each stock's trading life) : {exp_stockday_total:,}")
    print(f"  present  stock-days                                    : {present_stockday_total:,}")
    print(f"  overall coverage                                       : {cov_overall:.4f}%")
    print(f"  stocks 100% complete / with gaps                       : {n_complete_stocks} / {n_gap_stocks}")
    print(f"  trading days clean for ALL live stocks / total         : {len(clean_all_days)} / {len(trading_days)}")
    print(f"\n  ISSUE COUNTS")
    print(f"    fully-missing stock-days (gap within life) : {n_missing_days:,}")
    print(f"    partial days (real, non-special)           : {n_partial_real:,}  (mostly no-trade minutes on illiquid names)")
    print(f"      └ fetch-gap SUSPECTS (contig ≥ {FETCH_GAP_MIN_BLOCK} min)   : {n_fetchgap_suspect:,}  <-- likely real fetch gaps")
    print(f"    special-session days (legit < 375)         : {len(special_dates)}")
    print(f"    duplicate candles                          : {n_dups:,}")
    print(f"    out-of-session candles                     : {n_oos:,}")
    print(f"    strategy-critical gaps (non-special)       : {n_crit_real:,}")
    print(f"    STRATEGY-CRITICAL gaps ON ACTUAL SIGNALS   : {n_crit_signal:,}  <-- the ones that break a backtest trade")
    if len(missing_stocks):
        print(f"    expected stocks with NO parquet            : {len(missing_stocks)}")

    print("\n  CANDLES-PER-DAY DISTRIBUTION (spike at 375 = healthy):")
    tot_sd = len(rec)
    for _, r in hist.sort_values("n_stock_days", ascending=False).head(8).iterrows():
        c = int(r["candles_in_day"]); n = int(r["n_stock_days"])
        tag = " (FULL)" if c == FULL_COUNT else (" (special?)" if c in set(exp_count.get(d, 0) for d in special_dates) else "")
        print(f"    {c:>3} candles : {n:>8,} stock-days ({n/tot_sd*100:5.2f}%){tag}")

    full_375_cov = int((rec["distinct_in"] == FULL_COUNT).sum()) / len(rec) * 100
    material = (n_missing_days == 0 and n_fetchgap_suspect == 0 and n_crit_signal == 0
                and n_dups == 0 and n_oos == 0)
    verdict = "COMPLETE — no material gaps" if material else "INCOMPLETE — material gaps present"
    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print(f"  stock-day presence coverage : {cov_overall:.4f}%  (present vs expected within trading life)")
    print(f"  full-375-candle days        : {full_375_cov:.2f}% of present stock-days")
    print( "  NOTE: Upstox INCONSISTENTLY pads no-trade minutes — a zero-trade minute is sometimes")
    print( "        emitted as a volume==0 candle (counted PRESENT here) and sometimes omitted")
    print( "        entirely (counted missing). Absent minutes on illiquid stocks are mostly")
    print( "        scattered single interior no-trade minutes, NOT fetch gaps. Raw <375 is NOT a")
    print( "        defect by itself — the MATERIAL signals are: fully-missing days, contiguous")
    print(f"        blocks ≥ {FETCH_GAP_MIN_BLOCK}min, and critical gaps landing on real signal days.")
    if not material:
        print(f"  material issues: {n_missing_days:,} missing days | {n_fetchgap_suspect:,} fetch-gap suspects | "
              f"{n_crit_signal:,} critical-on-signal | {n_dups:,} dups | {n_oos:,} out-of-session")

    print("\n  RELIABILITY OF THIS AUDIT (read before trusting):")
    print(f"    • Expected symbols: {'Companies List.csv (authoritative-ish)' if have_expected_list else 'NONE — observed only'}")
    print( "    • Trading calendar + holidays DERIVED from data (no NSE holiday file) — APPROXIMATE.")
    print( "    • Session length per day = fullest observed session (special sessions auto-flagged).")
    print( "    • Start-truncated history assumed = listing date, not a gap; interior gaps = fetch problems.")
    print( "    • <375 candle days are mostly no-trade minutes, not gaps — see max_contiguous_missing in partial_days.csv.")
    print(f"\nSaved 10 CSVs → {OUTDIR}")


if __name__ == "__main__":
    main()
