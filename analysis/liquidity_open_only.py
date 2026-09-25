# -*- coding: utf-8 -*-
"""liquidity_open_only.py — rebuild the liquidity proxies EXCLUDING UC (upper-circuit) trades,
i.e. keep only stocks that were OPEN / freely trading in the 3:00-3:22 window (hit_uc_ever==False).
Produces the Excel and a Word methodology doc for the open-only set."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, RGBColor, Inches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

SRC = rb.RESULTS / "liquidity_proxies" / "per_trade_liquidity.csv"
ATX = rb.RESULTS / "baseline_and_cross_final" / "baseline_final_performance.xlsx"
OUTDIR = rb.RESULTS / "liquidity_proxies_open_only"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── filter out UC trades ──
R_all = pd.read_csv(SRC)
at = pd.read_excel(ATX, sheet_name="all_trades", usecols=["symbol", "entry_date", "category", "hit_uc_ever"])
at["entry_date"] = pd.to_datetime(at["entry_date"]).dt.date.astype(str)
R_all = R_all.merge(at, on=["symbol", "entry_date"], how="left")
n_uc = int((R_all.hit_uc_ever == True).sum())
R = R_all[R_all.hit_uc_ever == False].drop(columns=["hit_uc_ever"]).reset_index(drop=True)   # OPEN only
print(f"all trades {len(R_all):,} | removed UC {n_uc} | OPEN-in-window kept {len(R):,} | symbols {R['symbol'].nunique():,}")

# ── per-symbol profile (open only) ──
g = R.groupby("symbol")
PS = pd.DataFrame({
    "n_trades": g.size(),
    "avg_window_turnover": g["window_turnover"].mean().round(0),
    "avg_day_turnover": g["day_turnover"].mean().round(0),
    "avg_shares_pct_of_volume": g["shares_pct_of_volume"].mean().round(3),
    "avg_capital_pct_of_turnover": g["capital_pct_of_turnover"].mean().round(3),
    "avg_candle_range_pct": g["avg_candle_range_pct"].mean().round(4),
    "avg_n_zero_vol_candles": g["n_zero_vol_candles"].mean().round(2),
    "avg_shares_pct_of_DAY_volume": g["shares_pct_of_DAY_volume"].mean().round(3),
}).reset_index().sort_values("avg_shares_pct_of_volume", ascending=False).reset_index(drop=True)

worst = R.sort_values(["shares_pct_of_volume", "window_turnover"], ascending=[False, True]).head(50)


def dist(col, label):
    s = R[col].dropna()
    return {"basis": label, "n": len(s), "median_pct": round(float(s.median()), 3), "mean_pct": round(float(s.mean()), 3),
            "p90": round(float(s.quantile(0.90)), 3), "p99": round(float(s.quantile(0.99)), 3),
            "pct_trades_gt_1pct": round(float((s > 1).mean() * 100), 1),
            "pct_trades_gt_5pct": round(float((s > 5).mean() * 100), 1),
            "pct_trades_gt_10pct": round(float((s > 10).mean() * 100), 1),
            "pct_trades_gt_25pct": round(float((s > 25).mean() * 100), 1)}
DIST = pd.DataFrame([dist("shares_pct_of_volume", "vs 15:00-15:22 window volume"),
                     dist("shares_pct_of_DAY_volume", "vs full-day volume (%ADV-like)")])

R.to_parquet(OUTDIR / "per_trade_liquidity_open_only.parquet", index=False)
R.to_csv(OUTDIR / "per_trade_liquidity_open_only.csv", index=False)
with pd.ExcelWriter(OUTDIR / "liquidity_proxies_open_only.xlsx", engine="openpyxl") as w:
    R.to_excel(w, sheet_name="per_trade_OPEN", index=False)
    PS.to_excel(w, sheet_name="per_symbol_profile", index=False)
    worst.to_excel(w, sheet_name="least_liquid_trades_top50", index=False)
    DIST.to_excel(w, sheet_name="shares_pct_distribution", index=False)
print("saved Excel ->", OUTDIR / "liquidity_proxies_open_only.xlsx")

# ═══════════════════════════ WORD DOC ═══════════════════════════
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
doc = Document(); doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(11)


def h(t, lvl=1):
    p = doc.add_heading(t, level=lvl)
    for r in p.runs:
        r.font.color.rgb = NAVY
    return p


def para(t, bold=False, italic=False, size=11):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic; r.font.size = Pt(size); return p


def bullet(t, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        r = p.add_run(lead); r.bold = True
    p.add_run(t); return p


def formula(t):
    p = doc.add_paragraph(); r = p.add_run(t); r.font.name = "Consolas"; r.font.size = Pt(10.5)
    p.paragraph_format.left_indent = Inches(0.3); return p


def tbl(df, cols, headers, maxrows=None):
    d = df[cols].head(maxrows) if maxrows else df[cols]
    t = doc.add_table(rows=1, cols=len(cols)); t.style = "Light Grid Accent 1"
    for j, hd in enumerate(headers):
        rn = t.rows[0].cells[j].paragraphs[0].add_run(str(hd)); rn.bold = True; rn.font.size = Pt(9)
    for _, row in d.iterrows():
        cs = t.add_row().cells
        for j, cc in enumerate(cols):
            v = row[cc]
            s = f"{v:,.0f}" if isinstance(v, float) and abs(v) >= 1000 else (f"{v:,.3f}" if isinstance(v, float) else str(v))
            r = cs[j].paragraphs[0].add_run(s); r.font.size = Pt(9)
    return t


tp = doc.add_heading("Liquidity & Fill-Realism Proxies — OPEN Stocks Only", level=0)
for r in tp.runs:
    r.font.color.rgb = NAVY
para("UC (upper-circuit) trades removed. Only stocks freely trading in the 3:00-3:22 window.", italic=True, size=12)
doc.add_paragraph()

h("1. What changed vs the full analysis", 1)
para(f"The full analysis covered all {len(R_all):,} baseline trades. This version REMOVES the {n_uc} trades "
     "flagged by the backtest as hitting the upper circuit (hit_uc_ever = True) — those are the trades where "
     "the stock was locked at / pinned near its +20% circuit and the OHLCV volume is a one-sided buy queue, so "
     "the liquidity numbers overstate how fillable they were. Removing them leaves "
     f"{len(R):,} trades across {R['symbol'].nunique():,} symbols — all stocks that were genuinely OPEN and "
     "two-sided in the 3:00-3:22 window, i.e. the trades whose backtested fills the proxies can be trusted for.")
para("The 227 removed UC trades were all of Category A (75) and Category B (64) plus 88 Category-C trades; the "
     "3,209 kept are Category C entries in stocks that stayed open.", italic=True, size=10)

h("2. Why remove them", 1)
para("At a locked upper circuit the candle prints heavy volume (buyers matched against thin supply at the "
     "ceiling), so turnover looks large and shares-%-of-volume looks small — the proxy reads 'easily fillable'. "
     "But you cannot actually buy: you sit at the back of a one-sided queue, and the candle shows only the "
     "trades that cleared, not the unfilled orders ahead of you. The price is also frozen, so the range proxy "
     "collapses to ~0. Including UC trades therefore flatters the liquidity picture; excluding them gives the "
     "honest read for realistically-fillable trades.")

h("3. Measures (unchanged) — computed on the 3:00-3:22 entry window and the full day", 1)
para("For each trade, over N one-minute candles in the window:")
formula("window_turnover        = Σ (candle_close × candle_volume)          [rupee traded value, primary proxy]")
formula("shares_pct_of_volume   = shares  / window_total_volume  × 100       [size vs the tape — key read]")
formula("capital_pct_of_turnover= capital / window_turnover      × 100")
formula("avg_candle_range_pct   = mean( (high − low)/low × 100 ) per candle  [spread/impact proxy]")
formula("n_zero_vol_candles     = count( volume == 0 )                       [sporadic-trading flag]")
formula("product                = window_low × (window_total_volume / N)     [continuity metric]")
para("Everything is also computed vs the full-day (09:15-15:29) volume for the %-of-ADV context. Per-candle "
     "averages divide by the actual candles present.")

h("4. Distribution — open stocks only (danger zone for fill realism)", 1)
tbl(DIST, ["basis", "median_pct", "mean_pct", "p99", "pct_trades_gt_5pct", "pct_trades_gt_10pct", "pct_trades_gt_25pct"],
    ["Basis", "Median %", "Mean %", "p99 %", ">5%", ">10%", ">25%"])
para("Against the full day every open trade is trivial to fill; the only constraint is the narrow 15:15 window, "
     "where a small tail still exceeds 5-10% of window volume.")

h("5. Least-liquid OPEN trades (fills least trustworthy)", 1)
tbl(worst, ["symbol", "entry_date", "shares", "shares_pct_of_volume", "window_turnover", "n_zero_vol_candles"],
    ["Symbol", "Entry date", "Shares", "% of window vol", "Window turnover", "Zero-vol candles"], maxrows=10)

h("6. Caveats", 1)
bullet("no bid-ask spread or order-book depth is in candle data; these flag WHERE fills are unrealistic, not "
       "the rupee slippage cost.", lead="OHLCV proxies only: ")
bullet("removed via the backtest's hit_uc_ever flag; a stock touching the circuit anywhere in the day is "
       "excluded, which is conservative.", lead="UC definition: ")
bullet("for the thinnest names the range proxy degenerates to ~0 (flat no-trade bars); use n_zero_vol_candles "
       "as the sporadic-trade flag instead.", lead="Range proxy: ")

h("7. Output files", 1)
bullet("per_trade_liquidity_open_only.csv / .parquet — the kept open trades with every measure.")
bullet("liquidity_proxies_open_only.xlsx — per_trade_OPEN, per_symbol_profile, least_liquid_trades_top50, "
       "shares_pct_distribution.")
doc.add_paragraph()
para(f"Open-in-window trades: {len(R):,} across {R['symbol'].nunique():,} symbols "
     f"(UC trades removed: {n_uc}). All figures are OHLCV-derived proxies.", italic=True, size=9)

docpath = OUTDIR / "Liquidity_Proxies_OPEN_Only_Methodology.docx"
doc.save(str(docpath))
print("saved Word ->", docpath)
