# -*- coding: utf-8 -*-
"""
refetch_missing_1min.py
=======================
TARGETED re-fetch of ONLY the 1-min gaps flagged by audit_1min_data.py — never a full
re-download. Fetches from the SAME broker the data came from (UPSTOX v3 historical-candle),
validates each pull, and merges ADDITIVELY into a NEW store (never mutating the original
until you --swap after review).

BROKER = UPSTOX (flag a). Endpoint reused from fetch_1min_history.py:
  GET /v3/historical-candle/{instrument_key}/minutes/1/{to}/{frm}
  • per-request cap  : 1 month of 1-min data  -> we fetch per (symbol, month) that contains a gap
  • rate limits      : 50/s, 500/min (dl.RATE_LIMITS via fetch_1min_history), throttled + backoff
  • 1-min retention  : Upstox serves 1-min for our full 2022→now range (we already fetched it),
                       so "beyond API history" is not expected to bind; the check is kept anyway
                       (--unfetchable-before). Older-than-broker gaps would need a paid vendor
                       (GDFL/TrueData) — flag c.
  • credentials      : dl.ACCESS_TOKEN (flag b) — same token that just completed the full refresh.
  • instrument token : dl.build_instruments resolves symbol -> Upstox instrument_key (ISIN-based,
                       stable across most corporate actions). A symbol that resolves but returns
                       no data for in-life dates is flagged (possible re-listing/token mismatch).

WHAT IT FETCHES (STEP 1 — real gaps only, from the audit):
  Default MATERIAL queue (the gaps that can actually be real fetch problems):
    • fully-missing stock-days            (missing_stock_days.csv)
    • contiguous-block partials           (partial_days.csv, non-special, max_contiguous_missing ≥ --min-contig)
    • strategy-critical gaps ON SIGNALS   (strategy_critical_gaps.csv, on_signal_day & non-special)  ← priority
  EXCLUDED by construction: not-yet-listed/delisted spans (audit uses each stock's trading life),
  half-day/muhurat special sessions (audit-flagged), and — by default — SCATTERED no-trade partials
  (non-special, contig < --min-contig). Those scattered single minutes are confirmed no-trade
  (Upstox omits some zero-trade minutes; re-fetching returns the same and would mean re-downloading
  nearly everything). Use --include-scattered to fetch them too.

STEP 3 VALIDATION: OHLC sanity (high≥low, high≥open/close, low≤open/close, volume≥0, in-session
  09:15-15:29). Bad candles are NOT merged. STEP 4 MERGE: additive only, keyed by timestamp; a
  same-timestamp value conflict is LOGGED (conflicts.csv), never auto-overwritten (flag d). Merged
  series written to master_data_refetched/<SYM>.parquet. STEP 5 RE-AUDIT: per gap-date closure is
  recomputed from the merged data and each remaining gap is classified:
    closed | confirmed_absent_at_source | unfetchable_beyond_history | fetch_failed | unresolved_symbol

Usage:
  python analysis/refetch_missing_1min.py --dry-run          # build + print the queue, fetch nothing
  python analysis/refetch_missing_1min.py                    # fetch material gaps -> master_data_refetched/
  python analysis/refetch_missing_1min.py --include-scattered
  python analysis/refetch_missing_1min.py --swap             # after review: back up originals, swap in
Resumable via results/refetch_missing/refetch_done.txt.
"""
import argparse, sys, time, shutil, threading
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_loading as dl
import fetch_1min_history as f1          # reuse fetch_month(), Upstox endpoint, relaxed rate limits

AUDIT_DIR   = dl.BASE / "results" / "data_audit"
OUTDIR      = dl.BASE / "results" / "refetch_missing"
NEW_STORE   = dl.BASE / "master_data_refetched"
DONE_FILE   = OUTDIR / "refetch_done.txt"
HM_OPEN, HM_CLOSE = 555, 929
FULL_SESSION = set(range(HM_OPEN, HM_CLOSE + 1))
FULL_COUNT_SENTINEL = 999          # 'remaining' placeholder for a still-fully-missing day
IST = "Asia/Kolkata"
_lock = threading.Lock()


def hm_of(s):                       # "HH:MM" -> minutes-of-day
    h, m = s.split(":"); return int(h) * 60 + int(m)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _lock:
        print(line, flush=True)
        with open(OUTDIR / "refetch.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — build the fetch queue from the audit CSVs
# ════════════════════════════════════════════════════════════════════════════
def build_queue(min_contig, include_scattered):
    need = ["missing_stock_days.csv", "partial_days.csv", "strategy_critical_gaps.csv"]
    if not all((AUDIT_DIR / n).exists() for n in need):
        raise SystemExit(f"Audit output missing in {AUDIT_DIR}. Run analysis/audit_1min_data.py first.")
    msd = pd.read_csv(AUDIT_DIR / "missing_stock_days.csv", parse_dates=["date"])
    pdd = pd.read_csv(AUDIT_DIR / "partial_days.csv", parse_dates=["date"])
    cg  = pd.read_csv(AUDIT_DIR / "strategy_critical_gaps.csv", parse_dates=["date"])

    # target hms per (symbol,date): frozenset of missing minutes; None = whole day (fully-missing)
    q = {}   # (sym, date) -> dict(tier, prio, target_hms, max_contig, on_signal)
    def add(sym, d, tier, prio, hms, mc, sig):
        k = (sym, d.date() if hasattr(d, "date") else d)
        cur = q.get(k)
        if cur is None or prio < cur["prio"]:
            q[k] = dict(tier=tier, prio=prio, target_hms=hms, max_contig=mc, on_signal=sig)
        elif hms is not None and cur["target_hms"] is not None:
            cur["target_hms"] = cur["target_hms"] | hms

    # actual missing minutes per (symbol,date) from partial_days — so critical closure is
    # validated against the REAL missing timestamps (not a trivially-empty target).
    pdd_miss = {}
    for r in pdd.itertuples():
        d = r.date.date() if hasattr(r.date, "date") else r.date
        pdd_miss[(r.symbol, d)] = {hm_of(t) for t in str(r.missing_timestamps).split(";") if t}

    # priority 1 — critical on actual signal days (non-special)
    if "on_signal_day" in cg.columns:
        crit = cg[(cg["on_signal_day"] == True) & (cg["is_special_session"] == False)]
    else:
        crit = cg[cg["is_special_session"] == False]
    for _, r in crit.iterrows():
        d = r["date"].date() if hasattr(r["date"], "date") else r["date"]
        if r["entire_day_missing"]:
            hms = None
        else:
            hms = pdd_miss.get((r["symbol"], d), set())
            if not hms and pd.notna(r["missing_critical_singles"]):   # fallback: the named singles
                hms = {hm_of(t) for t in str(r["missing_critical_singles"]).split(";") if t}
        add(r["symbol"], d, "critical_on_signal", 1, hms, 0, True)

    # priority 2 — fully-missing stock-days
    for _, r in msd.iterrows():
        add(r["symbol"], r["date"], "fully_missing", 2, None, 0, False)

    # priority 3 — contiguous-block partials (real fetch-gap signature)
    pdd_ns = pdd[pdd["is_special_session"] == False].copy()
    contig = pdd_ns[pdd_ns["max_contiguous_missing"] >= min_contig]
    for _, r in contig.iterrows():
        hms = {hm_of(t) for t in str(r["missing_timestamps"]).split(";") if t}
        add(r["symbol"], r["date"], "contiguous_suspect", 3, hms, int(r["max_contiguous_missing"]), False)

    # priority 4 — scattered no-trade partials (only if explicitly requested)
    if include_scattered:
        scat = pdd_ns[pdd_ns["max_contiguous_missing"] < min_contig]
        for _, r in scat.iterrows():
            hms = {hm_of(t) for t in str(r["missing_timestamps"]).split(";") if t}
            add(r["symbol"], r["date"], "scattered_notrade", 4, hms, int(r["max_contiguous_missing"]), False)

    rows = [{"symbol": s, "date": d, **v} for (s, d), v in q.items()]
    Q = pd.DataFrame(rows).sort_values(["prio", "symbol", "date"]).reset_index(drop=True)
    return Q


def month_bounds(y, m):
    frm = date(y, m, 1)
    to = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    to = date.fromordinal(to.toordinal() - 1)
    return frm.isoformat(), to.isoformat()


def ohlc_sane(df):
    """Boolean mask of candles that pass OHLC/session sanity."""
    hm = df["hm"]
    return ((df["high"] >= df["low"]) & (df["high"] >= df["open"] - 1e-6) &
            (df["high"] >= df["close"] - 1e-6) & (df["low"] <= df["open"] + 1e-6) &
            (df["low"] <= df["close"] + 1e-6) & (df["volume"] >= 0) &
            (hm >= HM_OPEN) & (hm <= HM_CLOSE))


def prep(raw):
    ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
    raw = raw.copy()
    raw["ts_ist"] = ts
    raw["date"] = ts.dt.date
    raw["hm"] = ts.dt.hour * 60 + ts.dt.minute
    return raw


MAX_WORKERS = 5


def process_symbol(sym, sub, key, unfetch_before, stop_event):
    """Fetch + validate + additive-merge + classify all gaps for ONE symbol.
    Returns (results, conflicts, status) where status ∈ {'ok','AUTH','aborted'}. Thread-safe:
    touches only its own NEW_STORE parquet and returns its own lists (main thread collects)."""
    results, conflicts = [], []
    gap_dates = {d.date() if hasattr(d, "date") else d for d in sub["date"]}
    tgt = {(r["date"].date() if hasattr(r["date"], "date") else r["date"]): r["target_hms"]
           for _, r in sub.iterrows()}
    tier = {(r["date"].date() if hasattr(r["date"], "date") else r["date"]): r["tier"]
            for _, r in sub.iterrows()}

    if key is None:
        for d in gap_dates:
            results.append((sym, d, tier[d], "unresolved_symbol", 0))
        return results, conflicts, "ok"

    months = sorted({(d.year, d.month) for d in gap_dates})
    refetched = []; failed_months = set()
    for (y, m) in months:
        if stop_event.is_set():
            return results, conflicts, "aborted"
        frm, to = month_bounds(y, m)
        candles, err = f1.fetch_month(key, frm, to)
        if err == "AUTH":
            return results, conflicts, "AUTH"
        if candles is None:
            failed_months.add((y, m)); continue
        refetched.extend(candles)

    rdf = dl._candles_to_df(refetched)
    if len(rdf):
        rdf = prep(rdf)
        good = ohlc_sane(rdf)
        n_bad = int((~good).sum())
        if n_bad:
            log(f"  {sym}: dropped {n_bad} refetched candles failing OHLC/session sanity")
        rdf = rdf[good]

    # merged view = existing ∪ refetched-additions (existing never overwritten)
    exist_pq = dl.MASTER_DIR / f"{sym}.parquet"
    if exist_pq.exists():
        ex = prep(pd.read_parquet(exist_pq))
    else:
        ex = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume",
                                   "open_interest", "ts_ist", "date", "hm"])
    ex_ts = set(ex["ts_ist"]) if len(ex) else set()

    additions = rdf[~rdf["ts_ist"].isin(ex_ts)] if len(rdf) else rdf
    # conflict detection on shared timestamps (vectorized; log, do NOT overwrite)
    if len(rdf) and len(ex):
        cols = ["open", "high", "low", "close", "volume"]
        mrg = rdf[["ts_ist", "date"] + cols].merge(
            ex[["ts_ist"] + cols], on="ts_ist", suffixes=("_r", "_e"))
        for fld in cols:
            ev = mrg[f"{fld}_e"].to_numpy(float); rv = mrg[f"{fld}_r"].to_numpy(float)
            tol = 0.01 if fld == "volume" else np.maximum(0.005, np.abs(ev) * 0.001)
            bad = np.abs(ev - rv) > tol
            for i in np.nonzero(bad)[0]:
                conflicts.append((sym, mrg["date"].iat[i], str(mrg["ts_ist"].iat[i]),
                                  fld, float(ev[i]), float(rv[i])))

    if len(additions):
        keepcols = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
        merged = pd.concat([ex[keepcols] if len(ex) else ex, additions[keepcols]], ignore_index=True)
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        merged = merged.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        merged.to_parquet(NEW_STORE / f"{sym}.parquet", index=False, compression="snappy")

    # STEP 5 — per gap-date closure classification (from merged data)
    merged_by_date = {}
    allrows = pd.concat([ex, additions], ignore_index=True) if len(additions) else ex
    if len(allrows):
        for d, g in allrows.groupby("date"):
            merged_by_date[d] = set(g["hm"])
    for d in gap_dates:
        present = merged_by_date.get(d, set())
        want = tgt[d]
        if want is None:                       # fully-missing: closed if any candles now
            closed = len(present) > 0
            remaining = 0 if closed else FULL_COUNT_SENTINEL
        else:
            still = want - present
            closed = len(still) == 0
            remaining = len(still)
        if closed:
            status = "closed"
        elif (d.year, d.month) in failed_months:
            status = "fetch_failed"
        elif unfetch_before and d < unfetch_before:
            status = "unfetchable_beyond_history"
        else:                                  # month reachable but minute(s) still absent -> real gap
            status = "confirmed_absent_at_source"
        results.append((sym, d, tier[d], status, remaining))
    return results, conflicts, "ok"


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-contig", type=int, default=15,
                    help="min contiguous missing minutes to treat a partial as a fetch-gap suspect")
    ap.add_argument("--include-scattered", action="store_true",
                    help="also refetch scattered (<min-contig) no-trade partials (huge; usually pointless)")
    ap.add_argument("--only-critical", action="store_true",
                    help="fetch ONLY the strategy-critical-on-signal gaps (fast, highest value)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N symbols (test run)")
    ap.add_argument("--dry-run", action="store_true", help="build+print the queue, fetch nothing")
    ap.add_argument("--swap", action="store_true",
                    help="after merge, back up originals and swap refetched store in (requires review of conflicts)")
    ap.add_argument("--unfetchable-before", default="", help="YYYY-MM-DD; gaps older than this = beyond broker history")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    unfetch_before = pd.to_datetime(args.unfetchable_before).date() if args.unfetchable_before else None

    Q = build_queue(args.min_contig, args.include_scattered)
    if args.only_critical:
        Q = Q[Q["tier"] == "critical_on_signal"].reset_index(drop=True)
    tier_counts = Q["tier"].value_counts().to_dict()
    syms = list(dict.fromkeys(Q["symbol"]))                       # preserve priority order
    if args.limit:
        syms = syms[:args.limit]
        Q = Q[Q["symbol"].isin(syms)]
    n_symbol_months = Q.assign(ym=Q["date"].map(lambda d: (d.year, d.month))).groupby(
        "symbol")["ym"].nunique().sum() if len(Q) else 0

    print("=" * 78)
    print("TARGETED 1-MIN GAP REFETCH  (broker: UPSTOX v3;  additive merge; no full re-download)")
    print("=" * 78)
    print(f"gap queue: {len(Q):,} (symbol,date) pairs across {len(syms):,} symbols "
          f"(~{int(n_symbol_months):,} symbol-month requests)")
    for t in ["critical_on_signal", "fully_missing", "contiguous_suspect", "scattered_notrade"]:
        if t in tier_counts:
            print(f"   {t:<20}: {tier_counts[t]:,}")
    if not args.include_scattered:
        print("   (scattered no-trade partials EXCLUDED — use --include-scattered to fetch them)")

    if args.dry_run:
        Q.to_csv(OUTDIR / "fetch_queue_dryrun.csv", index=False)
        print(f"\nDRY RUN — queue written to {OUTDIR/'fetch_queue_dryrun.csv'}; nothing fetched.")
        return

    # resolve instrument keys (Upstox, ISIN-based) — needs the NSE instrument master
    print("\nResolving Upstox instrument keys (downloads NSE instrument master) …")
    instruments, unresolved = dl.build_instruments(dl.COMPANIES_CSV)
    unresolved_set = set(unresolved)
    print(f"  resolved {len(instruments):,} | unresolved {len(unresolved):,}")

    NEW_STORE.mkdir(exist_ok=True)
    done = set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    qby = {s: g for s, g in Q.groupby("symbol")}
    todo = [s for s in syms if s not in done]
    log(f"resuming: {len(done)} symbols already done, {len(todo)} to go ({MAX_WORKERS} workers)")

    results = []          # per gap-date classification
    conflicts = []        # (symbol, date, timestamp, field, existing, refetched)
    t0 = time.time(); aborted = False; ndone = 0
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process_symbol, s, qby[s], instruments.get(s), unfetch_before, stop_event): s
                for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                res, conf, status = fut.result()
            except Exception as e:
                log(f"  !! {s} errored: {e}"); continue
            results.extend(res); conflicts.extend(conf); ndone += 1
            if status == "AUTH":
                aborted = True; stop_event.set()
                log("!! 401 AUTH — token expired. Stopping; resume after refreshing dl.ACCESS_TOKEN.")
                continue
            if status == "aborted":
                continue
            with _lock, open(DONE_FILE, "a", encoding="utf-8") as f:
                f.write(s + "\n")
            if ndone % 50 == 0:
                log(f"[{ndone}/{len(todo)}] {s} | {len(results):,} gap-dates classified | "
                    f"{(time.time()-t0)/60:.1f}m")

    # ── report ──
    R = pd.DataFrame(results, columns=["symbol", "date", "tier", "status", "remaining_missing"])
    R.to_csv(OUTDIR / "refetch_results.csv", index=False)
    pd.DataFrame(conflicts, columns=["symbol", "date", "timestamp", "field",
                                     "existing", "refetched"]).to_csv(OUTDIR / "conflicts.csv", index=False)

    print("\n" + "=" * 78)
    print("REFETCH RESULT" + ("  (ABORTED on AUTH — resume after refreshing token)" if aborted else ""))
    print("=" * 78)
    if len(R):
        for st, n in R["status"].value_counts().items():
            print(f"  {st:<28}: {n:,}")
        print(f"\n  by tier × status:")
        print(R.groupby(["tier", "status"]).size().unstack(fill_value=0).to_string())
        n_closed = int((R["status"] == "closed").sum())
        n_absent = int((R["status"] == "confirmed_absent_at_source").sum())
        print(f"\n  gaps CLOSED                     : {n_closed:,}")
        print(f"  confirmed absent at source (real): {n_absent:,}  (illiquid/halted — no data exists anywhere)")
        print(f"  fetch_failed (retry by re-run)   : {int((R['status']=='fetch_failed').sum()):,}")
        print(f"  unfetchable beyond broker history: {int((R['status']=='unfetchable_beyond_history').sum()):,}")
        print(f"  unresolved symbol (no token)     : {int((R['status']=='unresolved_symbol').sum()):,}")
    print(f"\n  merge conflicts logged (NOT overwritten): {len(conflicts):,}  -> {OUTDIR/'conflicts.csv'}")
    print(f"  merged parquets (additions only) written to: {NEW_STORE}")

    # verdict
    if len(R):
        material_open = int((~R["status"].isin(["closed", "confirmed_absent_at_source"])).sum())
        if material_open == 0:
            print("\nVERDICT: every targeted gap is CLOSED or confirmed a real market gap. "
                  "Data is complete for the strategy's purposes.")
        else:
            print(f"\nVERDICT: {material_open:,} gaps still recoverable — "
                  f"{int((R['status']=='fetch_failed').sum()):,} transient (re-run), "
                  f"{int((R['status']=='unfetchable_beyond_history').sum()):,} need a paid vendor "
                  f"(GDFL/TrueData), {int((R['status']=='unresolved_symbol').sum()):,} token-unresolved.")

    # optional swap (STEP 4/5) — back up originals, then copy merged over
    if args.swap and not aborted:
        if len(conflicts):
            print(f"\n--swap REFUSED: {len(conflicts):,} unresolved conflicts in conflicts.csv. "
                  "Review them first; additive merge never auto-resolves conflicts.")
        else:
            bak = dl.BASE / f"master_data_backup_{time.strftime('%Y%m%d_%H%M%S')}"
            print(f"\n--swap: backing up originals of changed symbols -> {bak}")
            bak.mkdir(exist_ok=True)
            for pq in NEW_STORE.glob("*.parquet"):
                src = dl.MASTER_DIR / pq.name
                if src.exists():
                    shutil.copy2(src, bak / pq.name)
                shutil.copy2(pq, dl.MASTER_DIR / pq.name)
            print(f"  swapped {len(list(NEW_STORE.glob('*.parquet')))} symbols in. "
                  "Re-run analysis/audit_1min_data.py to confirm store-wide completeness.")
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
