# -*- coding: utf-8 -*-
"""vb_live_vs_two_benchmarks.py — LIVE executed trades (data/trades) vs two backtested benchmarks for the main NSE
Volume-Breakout BTST strategy: (1) BASELINE timing 9:25/11:59/14:39, (2) VWAP-close-move-BUCKET-dependent exit timing
(best combined t1/t2/cover per bucket from the earlier per-bucket full-grid sweep). New file, read-only vs locked files.

IMPORTANT DATA LIMIT: the trades folder holds ENTRY records only (trade_list, dhan_entries, mtf_entries). There are NO live
exit fills. So "live" P&L here is a RECONSTRUCTION: actual broker fill price + actual order quantity, exit assumed at the
scenario's rule (baseline or bucket timing) on real next-day 1-min prices. Deltas vs benchmarks therefore isolate ENTRY
differences (price/size) for the baseline comparison and entry + timing for the bucket comparison.

Exit engine = the grid sweep's exact per-position logic (branch order target_pre_t1 -> positive_at_t1 -> target<=t2 -> exit t2;
short entered at the long exit, covered at open of cover minute or at -5% target if hit first), validated to reproduce the
locked backtest's own per-trade P&L at the baseline times.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC
import vb_baseline_newmcap_aug18 as V
import vb_t1t2_cover_full_grid_sweep as SW

TR = rb.BASE / "data" / "trades" / "trades"
OUT = rb.RESULTS / "vb_live_vs_two_benchmarks"; OUT.mkdir(parents=True, exist_ok=True)
FIRST_BROKER = pd.Timestamp("2026-08-03").date()
LAST_ENTRY = pd.Timestamp("2026-09-17").date()          # 09-18 entries: exit day (09-21) not in the data yet
HM_1521 = BC.HM_1521
BASE = (BC.T1, BC.T2, BC.COVER_HM)                       # 565 / 719 / 879
# best COMBINED (long+short) cell per bucket, from t1t2_cover_grid_by_vwap_bucket_SUMMARY.xlsx (Cross_Bucket_Summary)
BUCKET_BEST = {"5-10pct": (556, 635, 868), "10-15pct": (558, 659, 897), "15-20pct": (584, 708, 841)}
BUCKET_EDGES = {"5-10pct": (5, 10), "10-15pct": (10, 15), "15-20pct": (15, 20)}
hm2s = lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}"


def bucket_of(x):
    if x != x: return None
    for k, (lo, hi) in BUCKET_EDGES.items():
        if lo <= x < hi: return k
    return None


def sim(avg, shares, D, T1, T2, COV):
    """Grid-sweep exit logic for one position. D: next-day arrays hm/hi/lo/op (hm 555..900)."""
    tgt = avg * BC.LONG_TGT
    hm, hi, lo, op = D["hm"], D["hi"], D["lo"], D["op"]
    opn = dict(zip(hm, op))
    hit = (hm <= 720) & (hi >= tgt)
    th = int(hm[hit].min()) if hit.any() else 10 ** 9
    o1, o2 = opn.get(T1, np.nan), opn.get(T2, np.nan)
    if th <= T1: xp, xt, xhm = tgt, "target_pre_t1", th
    elif o1 == o1 and o1 > avg: xp, xt, xhm = o1, "positive_at_t1", T1
    elif th <= T2: xp, xt, xhm = tgt, "target_by_t2", th
    elif o2 == o2: xp, xt, xhm = o2, "exit_t2", T2
    else: return None
    long_pnl = shares * (xp - avg)
    stgt = xp * BC.SHORT_TGT
    sm = (hm > xhm) & (lo <= stgt)
    sht = int(hm[sm].min()) if sm.any() else 10 ** 9
    oc = opn.get(COV, np.nan)
    if sht <= COV: cover, cht = stgt, sht
    elif oc == oc: cover, cht = oc, COV
    else: cover, cht = np.nan, None
    has = cover == cover
    short_pnl = shares * (xp - cover) if has else 0.0
    snotl = shares * xp if has else 0.0
    cap = shares * avg; comb = long_pnl + short_pnl
    return {"exit_px": xp, "exit_hm": xhm, "exit_type": xt, "cover_px": cover, "cover_hm": cht, "long_pnl": long_pnl,
            "short_pnl": short_pnl, "gross": comb, "netA": comb - BC.R023 * cap - BC.SR * snotl, "cap": cap,
            "ret_pct": comb / cap * 100}


def load_live():
    L = pd.concat([pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[11:21]).date()) for f in sorted(glob.glob(str(TR / "trade_list_*.csv")))])
    Bk = pd.concat([pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[13:23]).date(), src="dhan") for f in sorted(glob.glob(str(TR / "dhan_entries_*.csv")))]
                   + [pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[12:22]).date(), src="mtf") for f in sorted(glob.glob(str(TR / "mtf_entries_*.csv")))])
    return L.reset_index(drop=True), Bk.reset_index(drop=True)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    # ── STEP 1 ───────────────────────────────────────────────────────────────────────────────────
    L, Bk = load_live()
    print("=== STEP 1: live folder ===")
    print(f"trade_list files: {L['date'].nunique()} days ({L['date'].min()}..{L['date'].max()}) cols {['symbol','shares','ref_price']} -> PLANNED list, not executions")
    print(f"broker entry files: {Bk['date'].nunique()} days ({Bk['date'].min()}..{Bk['date'].max()}), rows {len(Bk)}")
    print(pd.crosstab([Bk["src"], Bk["product"]], Bk["status"]).to_string())
    dup = Bk.duplicated(["date", "symbol"]).sum(); print(f"duplicate (date,symbol) broker rows: {dup}")
    F = Bk[(Bk["status"] == "filled")].copy()
    print(f"status=='filled': {len(F)} | with non-null fill_price: {int(F['fill_price'].notna().sum())} | fill_price<=0: {int((F['fill_price']<=0).sum())}")
    excl = Bk[Bk["status"] != "filled"].groupby("status").size().to_dict(); print("excluded (not executed):", excl)
    F_in = F[(F["date"] <= LAST_ENTRY)].copy(); F_late = F[F["date"] > LAST_ENTRY]
    print(f"filled entries used (<= {LAST_ENTRY}): {len(F_in)} | filled but no exit day in data yet: {len(F_late)} -> {F_late['symbol'].tolist()}", flush=True)
    F_in = F_in.merge(L[["date", "symbol", "shares", "ref_price"]].rename(columns={"shares": "list_shares", "ref_price": "list_ref"}),
                      on=["date", "symbol"], how="left", suffixes=("", "_list"))

    # ── STEP 2: backtest trade set for the same dates (new daily mcap, locked baseline engine) ───────────────
    S = pd.read_parquet(V.SCAN_FN); S["date"] = pd.to_datetime(S["date"]).dt.date
    new_mcap = V.load_new_mcap()
    W = S[(S["date"] >= FIRST_BROKER) & (S["date"] <= LAST_ENTRY)]
    cand = W[W["passes_vol"] & W["passes_ret"]].copy()
    cand["in_band"] = [(s in set(new_mcap[d]["symbol"])) if d in new_mcap else False for s, d in zip(cand["symbol"], cand["date"])]
    sig = cand[cand["in_band"]][["symbol", "folder", "date", "pc"]].reset_index(drop=True)
    cache, extra = V.build_cache(sig)
    T, _ = BC.run_config("baseline", cache)
    T["entry_date"] = pd.to_datetime(T["entry_date"]).dt.date; T["exit_date"] = pd.to_datetime(T["exit_date"]).dt.date
    print(f"\nbacktest baseline trades {FIRST_BROKER}..{LAST_ENTRY} (new daily mcap): {len(T)} from {len(sig)} in-band signals", flush=True)

    Tk = T.set_index(["symbol", "entry_date"])
    sigset = set(zip(cand["symbol"], cand["date"])); bandset = set(zip(sig["symbol"], sig["date"]))
    scan_idx = S.set_index(["symbol", "date"])
    recs = []
    for r in F_in.itertuples():
        k = (r.symbol, r.date)
        why = ("MATCHED" if k in Tk.index else
               "signal in band but NOT in sequenced backtest trade set (no entry leg / pool)" if k in bandset else
               "passes vol+return but NOT in that day's new-mcap band file" if k in sigset else
               "does NOT pass vol/return signal criteria")
        s = scan_idx.loc[k] if k in scan_idx.index else None
        recs.append({"symbol": r.symbol, "entry_date": r.date, "src": r.src, "product": r.product, "status": r.status,
                     "order_ts": r.timestamp, "qty": r.quantity, "list_shares": r.list_shares, "list_ref": r.list_ref, "order_ref": r.ref_price,
                     "live_fill": r.fill_price, "match": why,
                     "vol_ratio": (s["vol_ratio"] if s is not None else np.nan), "ret_pct": (s["ret_pct"] if s is not None else np.nan)})
    M = pd.DataFrame(recs)
    print("\n=== STEP 2: match of live executed trades to backtest trade set ===")
    print(M["match"].value_counts().to_string(), flush=True)
    matched = M[M["match"] == "MATCHED"].copy()
    unmatched = M[M["match"] != "MATCHED"].copy()

    # ── per-trade data (entry-day 15:21 candle, VWAP bucket metric, next-day minute arrays) ──────────────
    folder = S.drop_duplicates("symbol").set_index("symbol")["folder"].to_dict()
    pc_map = {(s, d): p for s, d, p in zip(S["symbol"], S["date"], S["pc"])}
    D = {}
    for sym, g in matched.groupby("symbol"):
        raw = pd.read_parquet(V.DIRS[folder[sym]] / f"{sym}.parquet")
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(V.IST)
        raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
        bd = {d: x for d, x in raw.groupby("date")}; dts = sorted(bd); vw = SW._vwap_by_date(raw)
        for r in g.itertuples():
            ed = r.entry_date; nd = dts[dts.index(ed) + 1]
            e = bd[ed].set_index("hm"); n = bd[nd]; n = n[(n["hm"] >= 555) & (n["hm"] <= 900)].drop_duplicates("hm")
            c = e.loc[HM_1521]; pc = pc_map[(sym, ed)]
            D[(sym, ed)] = {"hm": n["hm"].values, "hi": n["high"].values.astype(float), "lo": n["low"].values.astype(float), "op": n["open"].values.astype(float),
                            "exit_date": nd, "o1521": float(c["open"]), "h1521": float(c["high"]), "l1521": float(c["low"]), "c1521": float(c["close"]),
                            "vwap_move_pct": (vw.get(ed, np.nan) - pc) / pc * 100}

    # ── validation: engine reproduces the locked backtest at baseline times, model entry + model shares ───────
    ck = []
    for r in matched.itertuples():
        t = Tk.loc[(r.symbol, r.entry_date)]; d = D[(r.symbol, r.entry_date)]
        e = sim(t["avg_entry"], t["shares"], d, *BASE)
        ck.append({"g": t["gross_pnl"], "g2": e["gross"], "a": t["netA_pnl"], "a2": e["netA"]})
    ck = pd.DataFrame(ck)
    print(f"\nengine validation on {len(ck)} matched trades (model entry+size, baseline times): max|gross diff| {np.abs(ck.g-ck.g2).max():.4f} | max|netA diff| {np.abs(ck.a-ck.a2).max():.4f}", flush=True)

    # ── STEPS 3-4: scenarios per trade ──────────────────────────────────────────────────────────────
    rows = []
    for r in matched.itertuples():
        t = Tk.loc[(r.symbol, r.entry_date)]; d = D[(r.symbol, r.entry_date)]; q = float(r.qty); f = float(r.live_fill); m = float(t["avg_entry"])
        b = bucket_of(d["vwap_move_pct"]); bx = BUCKET_BEST.get(b, BASE)
        L0 = sim(f, q, d, *BASE); L2 = sim(f, q, d, *bx); B1 = sim(m, q, d, *BASE); B2 = sim(m, q, d, *bx)
        if None in (L0, L2, B1, B2):
            continue
        rows.append({
            "symbol": r.symbol, "entry_date": r.entry_date, "exit_date": d["exit_date"], "category": t["category"], "product": r.product,
            "live_order_time": str(r.order_ts)[11:19], "live_entry_px": f, "live_qty": q, "model_entry_px": round(m, 3),
            "vwap_move_pct": round(d["vwap_move_pct"], 2), "bucket": b or "outside 5-20% (baseline timing used)", "bucket_exit": f"{hm2s(bx[0])}/{hm2s(bx[1])}/{hm2s(bx[2])}",
            "LIVE_exit_px_ASSUMED_baseline": round(L0["exit_px"], 2), "LIVE_exit_time_ASSUMED_baseline": hm2s(L0["exit_hm"]), "LIVE_exit_type": L0["exit_type"],
            "LIVE_pnl_baseline_exit": round(L0["gross"]), "LIVE_pnl_bucket_exit": round(L2["gross"]),
            "BT_baseline_pnl": round(B1["gross"]), "BT_bucket_pnl": round(B2["gross"]),
            "delta_live_vs_baseline": round(L0["gross"] - B1["gross"]), "delta_live_vs_bucket": round(L0["gross"] - B2["gross"]),
            "LIVE_ret_pct_baseline": round(L0["ret_pct"], 3), "BT_baseline_ret_pct": round(B1["ret_pct"], 3), "BT_bucket_ret_pct": round(B2["ret_pct"], 3),
            "LIVE_long_pnl": round(L0["long_pnl"]), "LIVE_short_pnl": round(L0["short_pnl"]), "BT_baseline_long_pnl": round(B1["long_pnl"]), "BT_bucket_long_pnl": round(B2["long_pnl"]),
            "LIVE_bucket_long_pnl": round(L2["long_pnl"]), "live_cap": round(f * q), "bt_cap": round(m * q),
            "delta_LONG_live_vs_baseline": round(L0["long_pnl"] - B1["long_pnl"]), "delta_LONG_live_vs_bucket": round(L0["long_pnl"] - B2["long_pnl"]),
            "BT_baseline_short_pnl_MODELLED": round(B1["short_pnl"]), "BT_bucket_short_pnl_MODELLED": round(B2["short_pnl"]),
            "BT_bucket_exit_type": B2["exit_type"], "BT_bucket_exit_time": hm2s(B2["exit_hm"]),
            "LIVE_netA_baseline": round(L0["netA"]), "BT_baseline_netA": round(B1["netA"]), "BT_bucket_netA": round(B2["netA"]),
            "slip_vs_1521open_bps": round((f / d["o1521"] - 1) * 1e4, 1), "model_entry_vs_1521open_bps": round((m / d["o1521"] - 1) * 1e4, 1),
            "fill_outside_1521_range": bool(f > d["h1521"] + 1e-9 or f < d["l1521"] - 1e-9), "slip_vs_order_ref_bps": round((f / r.order_ref - 1) * 1e4, 1),
            "qty_vs_list": (q / r.list_shares if r.list_shares == r.list_shares and r.list_shares else np.nan)})
    P = pd.DataFrame(rows)
    print(f"\ntrades in comparison: {len(P)} (matched {len(matched)})", flush=True)

    def agg(df, lab):
        o = {"set": lab, "n": len(df)}
        for nm, col in (("LIVE (fill+qty, baseline exit)", "LIVE_pnl_baseline_exit"), ("LIVE (fill+qty, bucket exit)", "LIVE_pnl_bucket_exit"),
                        ("BT baseline timing", "BT_baseline_pnl"), ("BT bucket timing", "BT_bucket_pnl")):
            o[nm + " | total Rs"] = int(df[col].sum()); o[nm + " | win%"] = round((df[col] > 0).mean() * 100, 1)
        return o
    A = pd.DataFrame([agg(P, "ALL matched")] + [agg(g, k) for k, g in P.groupby("bucket")]).set_index("set").T
    print("\n=== SUMMARY (gross Rs at LIVE quantities) ===\n" + A.to_string(), flush=True)

    R023 = BC.R023
    def agg_long(df, lab):
        o = {"set": lab, "n": len(df)}
        for nm, col, cap in (("LIVE fill, baseline exit", "LIVE_long_pnl", "live_cap"), ("LIVE fill, bucket exit", "LIVE_bucket_long_pnl", "live_cap"),
                             ("BT baseline timing", "BT_baseline_long_pnl", "bt_cap"), ("BT bucket timing", "BT_bucket_long_pnl", "bt_cap")):
            o[nm + " | gross Rs"] = int(df[col].sum()); o[nm + " | net Rs"] = int((df[col] - R023 * df[cap]).sum()); o[nm + " | win%"] = round((df[col] > 0).mean() * 100, 1)
            o[nm + " | avg ret %"] = round((df[col] / df[cap]).mean() * 100, 3)
        return o
    AL = pd.DataFrame([agg_long(P, "ALL matched (long only)")] + [agg_long(g, k) for k, g in P.groupby("bucket")]).set_index("set").T
    print("\n=== LONG-ONLY SUMMARY (no short leg; net = gross - 0.23% of capital) ===\n" + AL.to_string(), flush=True)
    dl1 = P["LIVE_long_pnl"] - P["BT_baseline_long_pnl"]; dl2 = P["LIVE_long_pnl"] - P["BT_bucket_long_pnl"]; tb = P["LIVE_bucket_long_pnl"] - P["LIVE_long_pnl"]
    rng = np.random.default_rng(1); bs = [tb.sample(len(tb), replace=True, random_state=int(rng.integers(1e9))).sum() for _ in range(2000)]
    TL = {"delta live - baseline (long) total Rs": dl1.sum(), "mean (se)": f"{dl1.mean():.0f} ({dl1.std()/np.sqrt(len(dl1)):.0f})",
          "delta live - bucket (long) total Rs": dl2.sum(),
          "bucket - baseline exit on LIVE entries (long) total Rs": tb.sum(), "mean (se)  ": f"{tb.mean():.0f} ({tb.std()/np.sqrt(len(tb)):.0f})",
          "t-stat": tb.mean() / (tb.std() / np.sqrt(len(tb))), "bootstrap 95% CI total": str(np.percentile(bs, [2.5, 97.5]).round(0).tolist()),
          "trades better / worse / same under bucket exit": f"{int((tb>0).sum())} / {int((tb<0).sum())} / {int((tb==0).sum())}",
          "total excl top-3 gains": int(tb.sort_values().iloc[:-3].sum())}
    print("\n=== LONG-ONLY TRACKING ===\n" + "\n".join(f"  {k}: {v if isinstance(v, str) else round(float(v), 3)}" for k, v in TL.items()), flush=True)

    d1, d2 = P["delta_live_vs_baseline"], P["delta_live_vs_bucket"]
    tr = {"mean|delta| Rs live vs baseline": d1.abs().mean(), "mean|delta| Rs live vs bucket": d2.abs().mean(),
          "sum delta live-baseline": d1.sum(), "sum delta live-bucket": d2.sum(),
          "corr(live,baseline) per-trade Rs": np.corrcoef(P.LIVE_pnl_baseline_exit, P.BT_baseline_pnl)[0, 1], "corr(live,bucket) per-trade Rs": np.corrcoef(P.LIVE_pnl_baseline_exit, P.BT_bucket_pnl)[0, 1],
          "mean delta live-baseline Rs (se)": f"{d1.mean():.0f} ({d1.std()/np.sqrt(len(d1)):.0f})", "mean delta live-bucket Rs (se)": f"{d2.mean():.0f} ({d2.std()/np.sqrt(len(d2)):.0f})",
          "win% live | baseline | bucket": f"{(P.LIVE_pnl_baseline_exit>0).mean()*100:.1f} | {(P.BT_baseline_pnl>0).mean()*100:.1f} | {(P.BT_bucket_pnl>0).mean()*100:.1f}",
          "pure TIMING effect on live entries (bucket - baseline exit) Rs": int((P.LIVE_pnl_bucket_exit - P.LIVE_pnl_baseline_exit).sum())}
    print("\n=== TRACKING ===\n" + "\n".join(f"  {k}: {v if isinstance(v, str) else round(float(v), 3)}" for k, v in tr.items()), flush=True)

    s = P["slip_vs_1521open_bps"]
    print(f"\n=== EXECUTION SLIPPAGE (live fill vs the 15:21 candle open, {len(P)} trades) ===\n  mean {s.mean():+.1f} bps | median {s.median():+.1f} | mean|abs| {s.abs().mean():.1f} | p10 {s.quantile(.1):+.1f} | p90 {s.quantile(.9):+.1f}"
          f"\n  |slip|>30 bps: {int((s.abs()>30).sum())} | fills outside the 15:21 candle high/low: {int(P.fill_outside_1521_range.sum())}"
          f"\n  structural: backtest model entry vs 15:21 open, non-C categories: mean {P[P.category!='C'].model_entry_vs_1521open_bps.mean():+.1f} bps (n={int((P.category!='C').sum())}) | category C: {P[P.category=='C'].model_entry_vs_1521open_bps.abs().max():.2f} bps max", flush=True)
    print(f"  slippage cost in Rs (live baseline-exit P&L vs same qty at model entry): {int((P.LIVE_pnl_baseline_exit - P.BT_baseline_pnl).sum())} total")
    big = P[(s.abs() > 30) | P.fill_outside_1521_range][["symbol", "entry_date", "category", "live_entry_px", "model_entry_px", "slip_vs_1521open_bps", "model_entry_vs_1521open_bps", "fill_outside_1521_range", "delta_live_vs_baseline"]]
    print("\nflagged fills:\n" + big.to_string(index=False), flush=True)

    with pd.ExcelWriter(OUT / "vb_live_vs_two_benchmarks.xlsx", engine="openpyxl") as w:
        pd.DataFrame({"note": [
            "NO live exit fills exist in the trades folder (entries only). LIVE columns = actual broker fill + actual order qty, exit ASSUMED by rule on real 1-min prices. Deltas vs baseline = entry price/size effect; vs bucket = entry + timing.",
            "Executed = status=='filled' (dhan_entries / mtf_entries). Excluded: not_filled, rejected, dry_run. No partial-fill field exists (quantity = ordered qty).",
            "Bucket variant uses each bucket's best COMBINED (long+short) t1/t2/cover from the full-history per-bucket sweep: " + str(BUCKET_BEST) + ". The sweep's history ends 2026-07-31, so this Aug-Sep window is unseen by it. The bucket uses the entry-day 15:00-15:29 VWAP, known at 15:30 before any next-day exit, so no look-ahead.",
            "The trades folder has NO short entries. Long-only sheets are the like-for-like live comparison; combined sheets add a MODELLED short leg that is not in the live data. Sample size is small (n shown); differences between variants over this window are not statistically reliable."]}).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        AL.reset_index().to_excel(w, sheet_name="Summary_LONG_ONLY", index=False)
        pd.DataFrame(list(TL.items()), columns=["metric", "value"]).to_excel(w, sheet_name="Tracking_LONG_ONLY", index=False)
        A.reset_index().to_excel(w, sheet_name="Summary_incl_MODELLED_short", index=False)
        P.to_excel(w, sheet_name="Per_trade", index=False)
        pd.DataFrame(list(tr.items()), columns=["metric", "value"]).to_excel(w, sheet_name="Tracking", index=False)
        unmatched.to_excel(w, sheet_name="Live_unmatched_excluded", index=False)
        F_late.to_excel(w, sheet_name="Live_filled_no_exit_yet", index=False)
        Bk[Bk["status"] != "filled"].to_excel(w, sheet_name="Not_executed_excluded", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {OUT}/vb_live_vs_two_benchmarks.xlsx")


if __name__ == "__main__":
    main()
