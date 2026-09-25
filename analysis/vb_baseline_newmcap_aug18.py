# -*- coding: utf-8 -*-
"""vb_baseline_newmcap_aug18.py — run the LOCKED baseline Volume-Breakout BTST engine
(baseline_and_cross_final.run_config("baseline"), untouched) for entry dates 2026-08-18 onward, with the
market-cap band (1,500-5,000 Cr) taken from the NEW daily source data/market_cap_daily/ instead of the old
semi-annual NSE snapshots. New file, read-only vs every locked file.

Signal rebuild (diagnostic_table.csv only reaches 2026-07-31): replicates prepare_data.py's conditions exactly
  vol   : cumulative volume 09:15-14:59 >= 6 x 36-day rolling mean of full-day (09:15-15:29) volume
          (non-zero days only, strict min_periods=36, shift(1), ffill across holidays)
  ret   : 15:00 candle OPEN >= +5% vs previous day's VWAP-close (last 30 candles, typical price (H+L+C)/3)
  mcap  : symbol present in that date's market_cap_daily file (files are ALREADY band-filtered 1,500-5,000 Cr)
The replication is validated against the existing diagnostic_table.csv on July-2026 rows.
Two runs on identical signals except the mcap source: NEW (daily files) and OLD (Dec-2025 semi-annual
snapshot, the one prepare_data.py would have used for 2026), for the sanity comparison.
"""
import sys, time, json, io, gzip
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC
import uc_staggered_dd_report as R

IST = "Asia/Kolkata"
START = pd.Timestamp("2026-08-18").date()
SCAN_FROM = pd.Timestamp("2026-07-01").date()          # extra July rows purely for validation vs diagnostic_table
MCAP_DIR = rb.BASE / "data" / "market_cap_daily" / "market_cap_daily"
DIRS = {"master_data": rb.BASE / "master_data", "master_data_new_listings": rb.BASE / "master_data"}   # new-listings folder merged into master_data
OUT = rb.RESULTS / "vb_baseline_newmcap_aug18"; OUT.mkdir(parents=True, exist_ok=True)
SCAN_FN = OUT / "signals_scan.parquet"
HM_915, HM_1459, HM_1500, HM_1529 = 555, 899, 900, 929
VOL_WINDOW, VOL_MULT, RET_MIN = 36, 6.0, 5.0


# ───────────────────────── step 1: signal scan over the whole 2,562-stock universe ─────────────────────────
def scan_one(fn, sym, folder):
    raw = pd.read_parquet(fn, columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
    raw["date"] = raw["timestamp"].dt.date
    raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
    all_dates = sorted(raw["date"].unique())
    if len(all_dates) < VOL_WINDOW + 1:
        return []
    fd_vol = raw[(raw["hm"] >= HM_915) & (raw["hm"] <= HM_1529)].groupby("date")["volume"].sum()
    c3_vol = raw[(raw["hm"] >= HM_915) & (raw["hm"] <= HM_1459)].groupby("date")["volume"].sum()
    last30 = raw[(raw["hm"] >= HM_1500) & (raw["hm"] <= HM_1529)].copy()
    last30["tp_vol"] = (last30["high"] + last30["low"] + last30["close"]) / 3 * last30["volume"]
    g = last30.groupby("date")[["tp_vol", "volume"]].sum()
    vwap_raw = (g["tp_vol"] / g["volume"]).where(g["volume"] > 0, np.nan)
    pm3_raw = raw[raw["hm"] == HM_1500].groupby("date")["open"].last()

    n = len(all_dates)
    fd_arr = np.array([fd_vol.get(d, 0) for d in all_dates], dtype=float)
    c3_arr = np.array([c3_vol.get(d, 0) for d in all_dates], dtype=float)
    pm3_arr = np.array([pm3_raw.get(d, np.nan) for d in all_dates], dtype=float)
    vwap_arr = np.array([vwap_raw.get(d, np.nan) for d in all_dates], dtype=float)
    nz = pd.Series(fd_arr, index=all_dates)[fd_arr > 0]
    avg_arr = nz.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean().shift(1).reindex(all_dates, method="ffill").values
    prev_vwap = np.empty(n); prev_vwap[0] = np.nan; prev_vwap[1:] = vwap_arr[:-1]
    df = pd.DataFrame({"date": all_dates, "pc": prev_vwap, "pm3_open": pm3_arr, "cum_vol": c3_arr, "avg_vol": avg_arr})
    df = df.iloc[VOL_WINDOW:]
    df = df[df["date"] >= SCAN_FROM]
    if df.empty:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        df["vol_ratio"] = df["cum_vol"] / df["avg_vol"]
        df["ret_pct"] = (df["pm3_open"] - df["pc"]) / df["pc"] * 100
    df["passes_vol"] = np.where(np.isnan(df["vol_ratio"]), False, df["vol_ratio"] >= VOL_MULT)
    df["passes_ret"] = np.where(np.isnan(df["ret_pct"]), False, df["ret_pct"] >= RET_MIN)
    df.insert(0, "symbol", sym); df.insert(1, "folder", folder)
    return df.to_dict("records")


def run_scan():
    if SCAN_FN.exists():
        print(f"scan cache found: {SCAN_FN}", flush=True)
        return pd.read_parquet(SCAN_FN)
    rows = []; t0 = time.time(); n = 0
    files = [(f, f.stem, k) for k, d in DIRS.items() for f in sorted(d.glob("*.parquet"))]
    print(f"scanning {len(files)} stock files...", flush=True)
    for fn, sym, folder in files:
        n += 1
        try:
            rows.extend(scan_one(fn, sym, folder))
        except Exception as e:
            print(f"  SKIP {sym}: {e}", flush=True)
        if n % 250 == 0:
            print(f"  ...{n}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    S = pd.DataFrame(rows)
    S.to_parquet(SCAN_FN, index=False)
    print(f"scan done in {time.time()-t0:.0f}s: {len(S):,} symbol-day rows", flush=True)
    return S


# ───────────────────────── step 2: validate replication vs the locked diagnostic table ─────────────────────────
def validate(S):
    d = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                    usecols=["symbol", "date", "prev_day_vwap_close", "return_pct_vs_prev_close", "volume_ratio",
                             "passes_volume", "passes_return", "passes_all_three"], parse_dates=["date"])
    d["date"] = d["date"].dt.date
    d = d[(d["date"] >= SCAN_FROM) & (d["date"] <= pd.Timestamp("2026-07-31").date())]
    m = d.merge(S, on=["symbol", "date"], how="left")
    ok = m["pc"].notna()
    out = {"diag_rows_jul": len(d), "matched_in_scan": int(ok.sum()),
           "max_abs_pc_diff": float((m.loc[ok, "pc"] - m.loc[ok, "prev_day_vwap_close"]).abs().max()),
           "max_abs_ret_diff": float((m.loc[ok, "ret_pct"] - m.loc[ok, "return_pct_vs_prev_close"]).abs().max()),
           "vol_flag_mismatch": int((m.loc[ok, "passes_vol"] != m.loc[ok, "passes_volume"]).sum()),
           "ret_flag_mismatch": int((m.loc[ok, "passes_ret"] != m.loc[ok, "passes_return"]).sum()),
           "diag_signals_jul": int(d["passes_all_three"].sum()),
           "scan_signals_same_rows": int((m.loc[ok, "passes_vol"] & m.loc[ok, "passes_ret"]).sum())}
    return out


# ───────────────────────── step 3: mcap sources ─────────────────────────
def load_new_mcap():
    files = {}
    for f in sorted(MCAP_DIR.glob("market_cap_*.csv")):
        d = pd.Timestamp(f.stem.replace("market_cap_", "")).date()
        files[d] = pd.read_csv(f)
    return files


def load_old_snapshot():
    import prepare_data as PD
    snap_dates, snap_eligible, snap_labels = PD._load_mcap_snapshots()
    return snap_labels[-1], snap_eligible[-1]      # snapshot in force for all 2026 dates (last = 2025-12-31)


# ───────────────────────── step 4: cache builder (both folders) for the LOCKED engine ─────────────────────────
def build_cache(sig):
    """sig: DataFrame(symbol, folder, date, pc). Same cache format BC.run_config consumes (mirrors BC.build_cache)."""
    cache = []; extra = {}
    for (sym, folder), g in sig.groupby(["symbol", "folder"]):
        raw = pd.read_parquet(DIRS[folder] / f"{sym}.parquet")
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
        bd = {d: x for d, x in raw.groupby("date")}
        dts = sorted(bd)
        for r in g.itertuples():
            ed, pc = r.date, float(r.pc)
            if ed not in bd or not (pc == pc and pc > 0):
                continue
            j = dts.index(ed)
            nd = dts[j + 1] if j < len(dts) - 1 else None
            if nd is None:
                continue
            gg = bd[ed]
            eg = {"hm": gg["hm"].values, "hi": gg["high"].values.astype(float),
                  "lo": gg["low"].values.astype(float), "op": gg["open"].values.astype(float)}
            ng = bd[nd]
            no = dict(zip(ng["hm"].values, ng["open"].values.astype(float)))
            nhm = ng["hm"].values; nhi = ng["high"].values.astype(float); nlo = ng["low"].values.astype(float)
            hm_t = (nhm >= 555) & (nhm <= BC.T2); lm_t = (nhm >= BC.T1) & (nhm <= BC.COVER_HM)
            cache.append({"symbol": sym, "entry_date": ed, "exit_date": nd, "pc": pc, "eg": eg,
                          "o565": no.get(BC.T1, np.nan), "o719": no.get(BC.T2, np.nan), "o879": no.get(BC.COVER_HM, np.nan),
                          "nhm": nhm[hm_t], "nhi": nhi[hm_t], "lhm": nhm[lm_t], "nlo": nlo[lm_t]})
            v = dict(zip(gg["hm"].values, gg["volume"].values))
            extra[(sym, ed)] = {"vol_1521": v.get(BC.HM_1521, np.nan), "folder": folder}
    return cache, extra


# ───────────────────────── step 5: metrics ─────────────────────────
def drawdown(T, col):
    de = T.groupby("exit_date")[col].sum().sort_index()
    cum = de.cumsum()
    peak = np.maximum.accumulate(np.maximum(cum.values, 0.0))
    dd = peak - cum.values
    return float(dd.max()), float(dd.mean())


def metrics(T, label):
    n = len(T)
    if n == 0:
        return {"run": label, "n_trades": 0}
    out = {"run": label, "n_trades": n, "n_days_with_trades": int(T["entry_date"].nunique())}
    for s in ["gross", "netA", "netB"]:
        p = T[f"{s}_pnl"]
        mdd, add = drawdown(T, f"{s}_pnl")
        out.update({f"{s}_total_inr": round(p.sum(), 0), f"{s}_return_pct_of_5L": round(p.sum() / BC.BASE_POOL * 100, 2),
                    f"{s}_win_rate_pct": round((p > 0).mean() * 100, 2), f"{s}_avg_ret_per_trade_pct": round(T[f"{s}_ret"].mean(), 3),
                    f"{s}_max_dd_inr": round(mdd, 0), f"{s}_max_dd_pct_of_5L": round(mdd / BC.BASE_POOL * 100, 2),
                    f"{s}_avg_dd_inr": round(add, 0), f"{s}_avg_dd_pct_of_5L": round(add / BC.BASE_POOL * 100, 2)})
    for c in ["A", "B", "C"]:
        g = T[T["category"] == c]
        out.update({f"cat{c}_n": len(g), f"cat{c}_win_pct_gross": round((g["gross_pnl"] > 0).mean() * 100, 1) if len(g) else np.nan,
                    f"cat{c}_gross_inr": round(g["gross_pnl"].sum(), 0), f"cat{c}_netA_inr": round(g["netA_pnl"].sum(), 0),
                    f"cat{c}_netB_inr": round(g["netB_pnl"].sum(), 0)})
    out["long_leg_gross_inr"] = round(T["long_pnl"].sum(), 0); out["short_leg_gross_inr"] = round(T["short_pnl"].sum(), 0)
    return out


def fno_underlyings():
    try:
        import requests
        r = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", timeout=60)
        raw = json.load(gzip.GzipFile(fileobj=io.BytesIO(r.content)))
        skip = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"}
        return set(i.get("underlying_symbol") or i.get("trading_symbol", "").split(" ")[0] for i in raw
                   if i.get("segment") == "NSE_FO" and i.get("instrument_type") == "FUT" and i.get("name") not in skip)
    except Exception as e:
        print("F&O list fetch failed:", str(e)[:100]); return None


def main():
    S = run_scan()
    S["date"] = pd.to_datetime(S["date"]).dt.date
    print("\n=== REPLICATION CHECK vs locked diagnostic_table.csv (July 2026 rows) ===", flush=True)
    v = validate(S)
    for k, x in v.items():
        print(f"  {k}: {x}", flush=True)

    new_mcap = load_new_mcap()
    old_lbl, old_elig = load_old_snapshot()
    print(f"\nold source: semi-annual snapshot '{old_lbl}' ({len(old_elig)} eligible symbols) | new source: {len(new_mcap)} daily files", flush=True)

    win = S[S["date"] >= START].copy()
    win["vr"] = win["passes_vol"] & win["passes_ret"]
    cand = win[win["vr"]].copy()
    fdates = sorted(new_mcap)
    cand["in_new"] = [s in set(new_mcap[d]["symbol"]) if d in new_mcap else False for s, d in zip(cand["symbol"], cand["date"])]
    cand["in_old"] = cand["symbol"].isin(old_elig.keys())
    print(f"\nvolume+return signals (any mcap) on/after {START}: {len(cand)} symbol-days | in NEW band: {cand.in_new.sum()} | in OLD band: {cand.in_old.sum()}", flush=True)

    # ---- missing/incomplete mcap flags ----
    miss = []
    for r in cand[~cand["in_new"]].itertuples():
        i = fdates.index(r.date) if r.date in fdates else None
        prev_in = next_in = None
        if i is not None:
            prev_in = (r.symbol in set(new_mcap[fdates[i - 1]]["symbol"])) if i > 0 else None
            next_in = (r.symbol in set(new_mcap[fdates[i + 1]]["symbol"])) if i < len(fdates) - 1 else None
        klass = ("date_has_no_mcap_file" if i is None else
                 "LIKELY_DATA_GAP (in adjacent-day files, absent this day)" if (prev_in or next_in) else "likely_outside_band (absent adjacent days too)")
        miss.append({"date": r.date, "symbol": r.symbol, "vol_ratio": round(r.vol_ratio, 2), "ret_pct": round(r.ret_pct, 2),
                     "in_prev_file": prev_in, "in_next_file": next_in, "in_old_snapshot_band": bool(r.in_old),
                     "old_snapshot_mcap_cr": old_elig.get(r.symbol, np.nan), "classification": klass})
    MISS = pd.DataFrame(miss)
    MISS.to_csv(OUT / "signals_not_in_new_mcap_files.csv", index=False)
    if len(MISS):
        print("\nsignals (vol+ret pass) ABSENT from that day's market_cap_daily file, by classification:", flush=True)
        print(MISS["classification"].value_counts().to_string(), flush=True)

    # ---- stale-file flag: 09-15 == 09-14 ----
    stale = new_mcap[pd.Timestamp("2026-09-15").date()].equals(new_mcap[pd.Timestamp("2026-09-14").date()])
    print(f"\n2026-09-15 file identical to 2026-09-14 (stale copy): {stale}", flush=True)

    # ---- build signal sets & run the LOCKED engine ----
    sig_new = cand[cand["in_new"]][["symbol", "folder", "date", "pc"]].reset_index(drop=True)
    sig_old = cand[cand["in_old"]][["symbol", "folder", "date", "pc"]].reset_index(drop=True)
    results = {}
    for label, sig in [("NEW_daily_mcap", sig_new), ("OLD_snapshot_mcap", sig_old)]:
        cache, extra = build_cache(sig)
        print(f"\n[{label}] signals={len(sig)} cache entries (have next-day data)={len(cache)}", flush=True)
        T, diag_days = BC.run_config("baseline", cache)
        T["entry_date"] = pd.to_datetime(T["entry_date"]).dt.date
        T["folder"] = [extra[(s, d)]["folder"] for s, d in zip(T["symbol"], T["entry_date"])]
        T["vol_1521"] = [extra[(s, d)]["vol_1521"] for s, d in zip(T["symbol"], T["entry_date"])]
        results[label] = (T, diag_days)
        T.to_csv(OUT / f"trades_{label}.csv", index=False)

    Tn, dd_n = results["NEW_daily_mcap"]; To, dd_o = results["OLD_snapshot_mcap"]
    Mn = metrics(Tn, "NEW_daily_mcap"); Mo = metrics(To, "OLD_snapshot_mcap")
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print("\n=== METRICS ===")
    print(pd.DataFrame([Mn, Mo]).set_index("run").T.to_string(), flush=True)

    # ---- trade-set comparison ----
    kn = set(zip(Tn["symbol"], Tn["entry_date"])); ko = set(zip(To["symbol"], To["entry_date"]))
    print(f"\ntrade sets: NEW={len(kn)} OLD={len(ko)} | both={len(kn & ko)} | only NEW={len(kn - ko)} | only OLD={len(ko - kn)}", flush=True)
    only_new = Tn[[(s, d) in (kn - ko) for s, d in zip(Tn["symbol"], Tn["entry_date"])]]
    only_old = To[[(s, d) in (ko - kn) for s, d in zip(To["symbol"], To["entry_date"])]]
    both_new = Tn[[(s, d) in (kn & ko) for s, d in zip(Tn["symbol"], Tn["entry_date"])]]
    for nm, g in [("only in NEW source", only_new), ("only in OLD source", only_old), ("in both (NEW run's P&L)", both_new)]:
        print(f"  {nm}: n={len(g)} gross={g['gross_pnl'].sum():,.0f} netA={g['netA_pnl'].sum():,.0f} netB={g['netB_pnl'].sum():,.0f}", flush=True)

    # ---- CAS check: Category C / A-leg2 fills at a frozen (zero-volume) 15:21 candle ----
    fno = fno_underlyings()
    Tn["is_fno"] = Tn["symbol"].isin(fno) if fno is not None else np.nan
    post_cas = Tn[Tn["entry_date"] >= pd.Timestamp("2026-08-03").date()]
    frozen = post_cas[(post_cas["vol_1521"] == 0)]
    fill_1521 = post_cas[post_cas["legs_filled"].astype(str).str.contains("C_1521|3:21|17%/3:21", regex=True)]
    print(f"\nCAS check (entries >= 2026-08-03): trades={len(post_cas)} | zero-volume 15:21 candle={len(frozen)} | "
          f"F&O-segment trades={int(Tn['is_fno'].sum()) if fno is not None else 'n/a'}", flush=True)
    if len(frozen):
        print(frozen[["symbol", "entry_date", "category", "legs_filled", "is_fno", "gross_pnl"]].to_string(index=False), flush=True)

    # ---- write workbook ----
    with pd.ExcelWriter(OUT / "vb_baseline_newmcap_aug18.xlsx", engine="openpyxl") as w:
        pd.DataFrame([Mn, Mo]).set_index("run").T.reset_index().to_excel(w, sheet_name="metrics_new_vs_old", index=False)
        pd.DataFrame([v]).T.reset_index().to_excel(w, sheet_name="replication_check", index=False)
        Tn.to_excel(w, sheet_name="trades_NEW", index=False)
        To.to_excel(w, sheet_name="trades_OLD", index=False)
        MISS.to_excel(w, sheet_name="signals_not_in_new_files", index=False)
        for nm, g in [("only_in_NEW", only_new), ("only_in_OLD", only_old)]:
            g.to_excel(w, sheet_name=nm, index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    print(f"\nSaved -> {OUT}/vb_baseline_newmcap_aug18.xlsx", flush=True)


if __name__ == "__main__":
    main()
