# -*- coding: utf-8 -*-
"""vb_live_vs_backtest_enterable.py — trade-by-trade comparison of LIVE trades (data/trades: trade_list + Dhan/MTF
orders) against the backtest trades that could actually have been entered (Aug-18+ new-mcap baseline with the
UC-locked trades removed: trades_NEW_daily_mcap_ex_UC_locked.csv).

There are no live EXIT records in the folder, so "live" P&L = the ACTUAL live entry (broker fill price, and the live
list's position size on the Rs 5L basis) run through the SAME baseline exit rules (9:25 positive / 11:59 rest / 17%
target / short covered 14:39 or at -5%). The exit function is a verbatim copy of baseline_and_cross_final.run_config's
exit block and is validated to reproduce the backtest's own per-trade P&L exactly before it is used on live entries.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import baseline_and_cross_final as BC
import vb_baseline_newmcap_aug18 as V

TR = rb.BASE / "data" / "trades" / "trades"
OUT = V.OUT
LAST_ENTRY = pd.Timestamp("2026-09-17").date()      # 09-18 entries have no exit day yet


def exit_pnl(c, avg, shares):
    """verbatim exit logic of BC.run_config for a single-leg position; c = cache entry (next-day arrays)."""
    cap = shares * avg
    tgt = avg * BC.LONG_TGT
    hit = c["nhi"] >= tgt
    th = int(c["nhm"][hit].min()) if hit.any() else 10 ** 9
    o565, o719, o879 = c["o565"], c["o719"], c["o879"]
    if th <= BC.T1:
        xp, xt, xhm = tgt, "target_pre_0925", th
    elif o565 == o565 and o565 > avg:
        xp, xt, xhm = o565, "positive_0925", BC.T1
    elif th <= BC.T2:
        xp, xt, xhm = tgt, "target_0925_1159", th
    elif o719 == o719:
        xp, xt, xhm = o719, "exit_1159", BC.T2
    else:
        return None
    long_pnl = shares * (xp - avg)
    stgt = xp * BC.SHORT_TGT
    sw = (c["lhm"] > xhm) & (c["nlo"] <= stgt)
    if sw.any():
        cover = stgt
    elif o879 == o879:
        cover = o879
    else:
        cover = np.nan
    has_short = cover == cover
    short_pnl = shares * (xp - cover) if has_short else 0.0
    snotl = shares * xp if has_short else 0.0
    comb = long_pnl + short_pnl
    return {"cap": cap, "long_pnl": long_pnl, "short_pnl": short_pnl, "exit_type": xt,
            "gross_pnl": comb, "netA_pnl": comb - BC.R023 * cap - BC.SR * snotl, "netB_pnl": comb - BC.R038 * cap - BC.SR * snotl}


def summ(df, label):
    n = len(df)
    out = {"set": label, "trades": n}
    for s in ["gross", "netA", "netB"]:
        p = df[f"{s}_pnl"]
        de = df.groupby("exit_date")[f"{s}_pnl"].sum().sort_index(); cum = de.cumsum()
        peak = np.maximum.accumulate(np.maximum(cum.values, 0.0)); dd = float((peak - cum.values).max())
        out.update({f"{s}_pnl": round(p.sum()), f"{s}_pct_of_5L": round(p.sum() / BC.BASE_POOL * 100, 2),
                    f"{s}_win_pct": round((p > 0).mean() * 100, 1), f"{s}_avg_ret_pct": round((p / df["cap"]).mean() * 100, 3),
                    f"{s}_max_dd_pct_of_5L": round(dd / BC.BASE_POOL * 100, 2)})
    return out


def main():
    T2 = pd.read_csv(OUT / "trades_NEW_daily_mcap_ex_UC_locked.csv"); T2["entry_date"] = pd.to_datetime(T2["entry_date"]).dt.date
    T2["exit_date"] = pd.to_datetime(T2["exit_date"]).dt.date
    S = pd.read_parquet(V.SCAN_FN); S["date"] = pd.to_datetime(S["date"]).dt.date
    pc_map = {(s, d): p for s, d, p in zip(S["symbol"], S["date"], S["pc"])}
    folder = S.drop_duplicates("symbol").set_index("symbol")["folder"].to_dict()

    L = pd.concat([pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[11:21]).date()) for f in sorted(glob.glob(str(TR / "trade_list_*.csv")))])
    Bk = pd.concat([pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[13:23]).date()) for f in sorted(glob.glob(str(TR / "dhan_entries_*.csv")))]
                   + [pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[12:22]).date()) for f in sorted(glob.glob(str(TR / "mtf_entries_*.csv")))])
    Lw = L[(L["date"] >= V.START) & (L["date"] <= LAST_ENTRY)][["date", "symbol", "shares", "ref_price"]].rename(columns={"shares": "list_shares", "ref_price": "list_ref"})
    Bw = Bk[(Bk["date"] >= V.START) & (Bk["date"] <= LAST_ENTRY)][["date", "symbol", "status", "fill_price", "quantity", "ref_price"]].rename(columns={"ref_price": "order_ref"})
    assert not Bw.duplicated(["date", "symbol"]).any()

    # ── 1. reconcile the 112 enterable backtest trades against live ─────────────────────────────
    R = T2.merge(Lw, left_on=["entry_date", "symbol"], right_on=["date", "symbol"], how="left").drop(columns="date")
    R = R.merge(Bw, left_on=["entry_date", "symbol"], right_on=["date", "symbol"], how="left").drop(columns="date")
    R["on_list"] = R["list_shares"].notna()
    def bucket(r):
        if r["status"] == "filled": return "1_LIVE FILLED"
        if r["status"] == "not_filled": return "2_live order NOT FILLED"
        if r["status"] == "dry_run": return "3_live DRY-RUN (no real order)"
        return "4_on live list, NO order placed" if r["on_list"] else "5_NOT on live list"
    R["bucket"] = R.apply(bucket, axis=1)
    rec = R.groupby("bucket").agg(trades=("symbol", "count"), gross=("gross_pnl", "sum"), netA=("netA_pnl", "sum"), netB=("netB_pnl", "sum"),
                                  avg_gross_ret_pct=("gross_ret", "mean")).round(2)
    rec.loc["TOTAL backtest (enterable)"] = [len(R), R.gross_pnl.sum(), R.netA_pnl.sum(), R.netB_pnl.sum(), R.gross_ret.mean()]
    rec = rec.round(2)
    pd.set_option("display.width", 220)
    print("=== 1. the 112 enterable backtest trades: what happened live ===")
    print(rec.to_string(), flush=True)
    print("\n  bucket 2 by category:", R[R.bucket.str.startswith("2")].category.value_counts().to_dict(),
          "| bucket 4:", R[R.bucket.str.startswith("4")].symbol.tolist(), flush=True)

    # live filled orders NOT among the backtest trades
    F = Bw[Bw["status"] == "filled"]
    only_live = F.merge(T2[["symbol", "entry_date"]], left_on=["date", "symbol"], right_on=["entry_date", "symbol"], how="left")
    only_live = only_live[only_live["entry_date"].isna()][["date", "symbol", "quantity", "fill_price"]]
    print(f"\nlive filled orders (entries <= 09-17) with NO backtest trade: {len(only_live)} -> {only_live.symbol.tolist()}", flush=True)

    # ── 2. live-as-executed P&L: live fill price + live list sizing, baseline exits ─────────────
    keys = pd.concat([R[R.bucket.str.startswith("1")][["symbol", "entry_date"]], only_live.rename(columns={"date": "entry_date"})[["symbol", "entry_date"]]])
    sig = pd.DataFrame({"symbol": keys["symbol"], "folder": [folder[s] for s in keys["symbol"]], "date": keys["entry_date"],
                        "pc": [pc_map[(s, d)] for s, d in zip(keys["symbol"], keys["entry_date"])]}).reset_index(drop=True)
    cache, _ = V.build_cache(sig)
    cmap = {(c["symbol"], c["entry_date"]): c for c in cache}

    # validation: exit function reproduces the backtest's own P&L for the live-filled backtest trades
    chk = []
    for r in R[R.bucket.str.startswith("1")].itertuples():
        c = cmap[(r.symbol, r.entry_date)]; e = exit_pnl(c, r.avg_entry, r.shares)
        chk.append({"sym": r.symbol, "d": r.entry_date, "model_gross": r.gross_pnl, "re_gross": e["gross_pnl"], "model_netA": r.netA_pnl, "re_netA": e["netA_pnl"]})
    CK = pd.DataFrame(chk)
    print(f"\nexit-function validation on {len(CK)} trades: max |gross diff| {np.abs(CK.model_gross-CK.re_gross).max():.4f} | max |netA diff| {np.abs(CK.model_netA-CK.re_netA).max():.4f}", flush=True)

    rows = []
    for r in R[R.bucket.str.startswith("1")].itertuples():
        c = cmap[(r.symbol, r.entry_date)]
        e_px = exit_pnl(c, r.fill_price, r.shares)              # live fill price, MODEL share count (pure entry-price effect)
        e_lv = exit_pnl(c, r.fill_price, r.list_shares)         # live fill price, LIVE list share count (Rs 5L basis)
        rows.append({"symbol": r.symbol, "entry_date": r.entry_date, "exit_date": c["exit_date"], "category": r.category, "model_entry": r.avg_entry, "live_fill": r.fill_price,
                     "slip_bps": (r.fill_price / r.avg_entry - 1) * 1e4, "model_shares": r.shares, "live_list_shares": r.list_shares,
                     "model_gross": r.gross_pnl, "model_netA": r.netA_pnl, "model_netB": r.netB_pnl,
                     **{f"px_{k}": v for k, v in e_px.items()}, **{f"lv_{k}": v for k, v in e_lv.items()}, "cap_model": r.capital_deployed})
    C = pd.DataFrame(rows)
    ol_rows = []
    for r in only_live.itertuples():
        c = cmap[(r.symbol, r.date)]; lst = Lw[(Lw.date == r.date) & (Lw.symbol == r.symbol)].iloc[0]
        e_lv = exit_pnl(c, r.fill_price, lst.list_shares)
        ol_rows.append({"symbol": r.symbol, "entry_date": r.date, "exit_date": c["exit_date"], "live_fill": r.fill_price, "live_list_shares": lst.list_shares, **{f"lv_{k}": v for k, v in e_lv.items()}})
    OL = pd.DataFrame(ol_rows)

    def frame(df, pre):
        d = pd.DataFrame({"exit_date": df["exit_date"], "cap": df[f"{pre}cap"], "gross_pnl": df[f"{pre}gross_pnl"],
                          "netA_pnl": df[f"{pre}netA_pnl"], "netB_pnl": df[f"{pre}netB_pnl"]}); return d
    M112 = T2.assign(cap=T2["capital_deployed"])[["exit_date", "cap", "gross_pnl", "netA_pnl", "netB_pnl"]]
    M84 = pd.DataFrame({"exit_date": C["exit_date"], "cap": C["cap_model"], "gross_pnl": C["model_gross"], "netA_pnl": C["model_netA"], "netB_pnl": C["model_netB"]})
    LPX = frame(C, "px_"); LLV = frame(C, "lv_")
    LLV86 = pd.concat([LLV, frame(OL, "lv_")], ignore_index=True)
    SM = pd.DataFrame([summ(M112, "BACKTEST enterable (all 112)"), summ(M84, "BACKTEST, the 84 live-filled names"),
                       summ(LPX, "LIVE-as-executed, 84 (live fill price, model size)"), summ(LLV, "LIVE-as-executed, 84 (live fill price + live list size)"),
                       summ(LLV86, "LIVE-as-executed, 86 (incl. 2 live-only names)")])
    print("\n=== 2. P&L comparison (Rs; % = of the Rs 5L pool) ===")
    print(SM.set_index("set").T.to_string(), flush=True)

    print(f"\nentry slippage live fill vs model 15:21 open: mean {C.slip_bps.mean():+.1f} bps, median {C.slip_bps.median():+.1f}, mean|abs| {C.slip_bps.abs().mean():.1f}")
    xt = C.merge(T2[["symbol", "entry_date", "long_exit_type"]], on=["symbol", "entry_date"])
    print(f"exit-type differences (live-fill entry vs model entry) among the 84: {int((xt.px_exit_type != xt.long_exit_type).sum())}")
    print("live-only names:"); print(OL[["symbol", "entry_date", "live_fill", "live_list_shares", "lv_gross_pnl", "lv_netA_pnl"]].round(1).to_string(index=False), flush=True)

    with pd.ExcelWriter(OUT / "vb_live_vs_backtest_enterable.xlsx", engine="openpyxl") as w:
        SM.set_index("set").T.reset_index().to_excel(w, sheet_name="pnl_comparison", index=False)
        rec.reset_index().to_excel(w, sheet_name="reconciliation_112", index=False)
        R.drop(columns=["lock_detail"], errors="ignore").to_excel(w, sheet_name="backtest_112_vs_live", index=False)
        C.to_excel(w, sheet_name="matched_84_trade_by_trade", index=False)
        OL.to_excel(w, sheet_name="live_only_fills", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    print(f"\nSaved -> {OUT}/vb_live_vs_backtest_enterable.xlsx")


if __name__ == "__main__":
    main()
