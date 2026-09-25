# -*- coding: utf-8 -*-
"""make_liquidity_doc.py — generate a detailed Word (.docx) explaining the liquidity-proxy method."""
import sys
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

RES = rb.RESULTS / "liquidity_proxies"
OUT = rb.RESULTS / "liquidity_proxies" / "Liquidity_Proxies_Methodology.docx"

R = pd.read_csv(RES / "per_trade_liquidity.csv")
PS = pd.read_excel(RES / "liquidity_proxies.xlsx", sheet_name="per_symbol_profile")
DIST = pd.read_excel(RES / "liquidity_proxies.xlsx", sheet_name="shares_pct_distribution")
worst = pd.read_excel(RES / "liquidity_proxies.xlsx", sheet_name="least_liquid_trades_top50")

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
doc = Document()
st = doc.styles["Normal"].font
st.name = "Calibri"; st.size = Pt(11)


def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = NAVY
    return p


def para(text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = bold; r.italic = italic; r.font.size = Pt(size)
    return p


def bullet(text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        r = p.add_run(bold_lead); r.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    return p


def formula(text):
    p = doc.add_paragraph()
    r = p.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(10.5); r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    p.paragraph_format.left_indent = Inches(0.3)
    return p


def table_from_df(df, cols=None, headers=None, maxrows=None):
    cols = cols or list(df.columns)
    headers = headers or cols
    d = df[cols].head(maxrows) if maxrows else df[cols]
    t = doc.add_table(rows=1, cols=len(cols)); t.style = "Light Grid Accent 1"
    for j, hd in enumerate(headers):
        c = t.rows[0].cells[j].paragraphs[0].add_run(str(hd)); c.bold = True; c.font.size = Pt(9)
    for _, row in d.iterrows():
        cells = t.add_row().cells
        for j, cc in enumerate(cols):
            v = row[cc]
            s = f"{v:,.0f}" if isinstance(v, float) and abs(v) >= 1000 else (f"{v:,.3f}" if isinstance(v, float) else str(v))
            rn = cells[j].paragraphs[0].add_run(s); rn.font.size = Pt(9)
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
tp = doc.add_heading("Liquidity & Fill-Realism Proxies", level=0)
for r in tp.runs:
    r.font.color.rgb = NAVY
para("OHLCV-based executability analysis of the volume-breakout BTST backtest", italic=True, size=12)
para("Methodology: what was calculated, from which data, and exactly how.", italic=True, size=10)
doc.add_paragraph()

# ── 1. PURPOSE ──
h("1. Purpose", 1)
para("The backtest assumes every signalled trade is filled at the 3:15 p.m. (15:15) one-minute candle. "
     "That assumption is only as good as the stock's liquidity at that moment. This analysis builds a set of "
     "OHLCV-derived proxies that quantify, for every trade, how realistically it could actually have been "
     "filled — i.e. how large the intended position was relative to the volume genuinely trading at the time, "
     "how much money was changing hands, how wide the bars were, and whether the stock was even trading in "
     "each minute. The goal is to flag the trades whose backtested fills are least trustworthy, because those "
     "are the trades where real-world slippage would be worst.")

p = doc.add_paragraph()
r = p.add_run("Important limitation (stated up front): "); r.bold = True; r.font.color.rgb = RGBColor(0xA0, 0x30, 0x00)
p.add_run("Every measure here is derived purely from 1-minute OHLCV candles. True liquidity — the bid-ask "
          "spread and the depth of the order book — is NOT present in candle data and cannot be reconstructed "
          "from it. These proxies correlate with execution difficulty but they do not measure execution cost "
          "in rupees. They tell you WHERE fills are unrealistic, not the exact slippage.")

# ── 2. DATA SOURCES ──
h("2. Data sources", 1)
bullet("  — the finalised baseline run, sheet 'all_trades': 3,436 trades, each with symbol, entry_date, "
       "shares, and capital_deployed.", bold_lead="Trade set: baseline_final_performance.xlsx ")
bullet("  — the project's per-symbol 1-minute OHLCV parquet files (master_data/<SYMBOL>.parquet), the same "
       "raw data the backtest itself is built on. 689 distinct symbols were traded.", bold_lead="1-minute price/volume: ")
bullet("  — the 20 real executed trades (entry date ≤ 2026-07-31) with measured entry slippage, used only for an "
       "informal cross-check at the end.", bold_lead="Live tradebook (cross-check only): ")
para("Every timestamp is converted to India time (Asia/Kolkata) before any minute-of-day filtering. All 3,436 "
     "trades matched to 1-minute data — zero were dropped for missing data.")

# ── 3. THE TWO WINDOWS ──
h("3. The two measurement windows", 1)
para("Each measure is computed over two windows on the entry day, because they answer different questions:")
bullet("(hm 900–922). The 23 one-minute candles immediately around the 3:15 fill. This is the window that "
       "actually matters for executability, because the strategy enters at 15:15 — you can only trade against "
       "the volume present in those minutes.", bold_lead="Entry window — 15:00 to 15:22 inclusive ")
bullet("(hm 555–929). The whole regular session (up to ~375 candles). This gives the classic '% of average "
       "daily volume' context — how big the position is relative to the entire day's trading, i.e. whether it "
       "could be filled at all if worked across the session rather than crammed into the close.",
       bold_lead="Full day — 09:15 to 15:29 inclusive ")
para("Result preview: the constraint turns out to be the narrow entry window, not the full day (Section 8).", italic=True)

# ── 4. THE MEASURES ──
h("4. The measures — definitions and formulas", 1)
para("For a given trade, let the window contain N one-minute candles, each with open/high/low/close and a "
     "share volume. 'shares' and 'capital_deployed' are the trade's own size from the backtest.")

h("4.1  Traded value / turnover  (primary liquidity proxy, measure #1)", 2)
formula("window_turnover      = Σ ( candle_close × candle_volume )   over the N candles")
formula("avg_candle_turnover  = window_turnover / N")
para("The rupee value that changed hands in the window. A large turnover means a deep, active tape that can "
     "absorb an order; a tiny turnover means the opposite. This is the single best OHLCV liquidity proxy.")

h("4.2  Trade-size-to-volume  (the key executability read, measure #2)", 2)
formula("shares_pct_of_volume    = shares  / window_total_volume   × 100")
formula("capital_pct_of_turnover = capital / window_turnover       × 100")
para("How big you are relative to the tape. If you need to buy 400 shares and only 100 traded in the whole "
     "window, shares_pct_of_volume = 400% — your order is larger than everything that traded, so the "
     "backtested fill is impossible at that price. This is computed against BOTH the window volume (strict, "
     "entry-relevant) and the full-day volume (the %ADV-style context).")

h("4.3  Intraday range proxy  (spread / impact proxy, measure #3)", 2)
formula("avg_candle_range_pct = mean over the window of ( (high − low) / low × 100 )")
para("The average high-to-low span of each one-minute bar, in percent. Wide bars imply thin books and large "
     "price impact; tight bars imply a liquid, orderly market. See the caveat in Section 9 — this measure "
     "degenerates for stocks that barely trade.")

h("4.4  Zero / thin-volume candles  (sporadic-trading flag, measure #4)", 2)
formula("n_zero_vol_candles = count of candles in the window with volume == 0")
para("How many of the minutes had no trading at all. A high count means the stock trades only sporadically "
     "near the close — you cannot rely on being filled minute-by-minute. This turns out to be the most "
     "reliable thin-trade flag (Section 9).")

h("4.5  Window-low × avg-volume product  (continuity with the earlier metric, measure #5)", 2)
formula("window_low           = min( candle_low )  over the window")
formula("avg_vol_per_candle   = window_total_volume / N")
formula("product              = window_low × avg_vol_per_candle")
para("Kept unchanged from the earlier request so the two analyses reconcile.")

# ── 5. DIVISOR NOTE ──
h("5. Divisor convention", 1)
para("Averages per candle divide by the ACTUAL number of candles present (N), not a fixed 23 or 375. This "
     "only matters when candles are missing; here all trade-days had the full 23 window candles (a couple of "
     "days had 22 in earlier variants and are handled the same way — divide by what is actually there).")

# ── 6. AGGREGATION ──
h("6. Aggregation — per-symbol liquidity profile", 1)
para("Because the strategy re-trades the same names, each symbol gets a profile: the simple mean of its "
     "per-trade values across all its trades — average window turnover, average shares_pct_of_volume, average "
     "candle range, average zero-volume-candle count. Symbols are then ranked worst-to-best by average "
     "shares_pct_of_volume (biggest-relative-to-tape first). 689 symbols profiled; 608 were traded more than once.")

# ── 7. LEAST-LIQUID FLAGGING ──
h("7. Flagging the least-liquid trades", 1)
para("Trades are sorted by shares_pct_of_volume (descending) then window_turnover (ascending) to surface the "
     "positions whose backtested fills are least realistic. The worst offenders:")
table_from_df(worst, cols=["symbol", "entry_date", "shares", "shares_pct_of_volume", "window_turnover", "n_zero_vol_candles"],
              headers=["Symbol", "Entry date", "Shares", "% of window vol", "Window turnover ₹", "Zero-vol candles"], maxrows=8)
para("The top three ask to buy more shares than traded in the entire 15:00–15:22 window (shares_pct > 100%), "
     "with 17–22 of the 23 minutes showing zero volume — those fills are fiction at 3:15 p.m.", italic=True)

# ── 8. DISTRIBUTION ──
h("8. Distribution — the danger zone for fill realism", 1)
para("What fraction of the 3,436 trades are large relative to the tape:")
table_from_df(DIST, cols=["basis", "median_pct", "mean_pct", "p99", "pct_trades_gt_5pct", "pct_trades_gt_10pct", "pct_trades_gt_25pct"],
              headers=["Basis", "Median %", "Mean %", "p99 %", ">5%", ">10%", ">25%"])
para("Reading: against the full day, every trade is trivial (worst ≈ 0.16% of daily volume) — everything is "
     "fillable if worked across the session. The whole constraint is the narrow 15:15 window: against those 23 "
     "minutes, roughly 1.4% of trades exceed 5%, 0.7% exceed 10%, and 0.3% exceed 25% of window volume. The "
     "backtest's assumption of a clean 15:15 fill is therefore realistic for ~99% of trades and unrealistic for "
     "the small illiquid tail.")

# ── 9. CAVEATS ──
h("9. Caveats and limitations", 1)
bullet("No bid-ask spread or order-book depth exists in candle data. The proxies flag WHERE fills are "
       "unrealistic, not the rupee cost of slippage.", bold_lead="OHLCV only: ")
bullet("For stocks that barely trade, zero-volume minutes are reported as flat bars (high = low), so their "
       "range computes to ~0% — which falsely looks 'tight/liquid'. Do NOT use the range proxy as the thin-trade "
       "flag; use n_zero_vol_candles, which correctly lights up (7–22 zero candles) exactly where the range "
       "proxy fails.", bold_lead="Range proxy degenerates: ")
bullet("shares_pct_of_volume vs the 23-minute window is the strict, decision-useful number; vs the full day it "
       "is the looser %ADV context. They are reported side by side.", bold_lead="Window vs day: ")

# ── 10. SLIPPAGE TIE-IN ──
h("10. Informal cross-check against live slippage", 1)
para("Across the 20 real executed trades, the correlation between shares_pct_of_window_volume and the measured "
     "absolute entry slippage was about −0.03, and between window volume and slippage about +0.13 — i.e. "
     "essentially no relationship. That is expected and reassuring rather than contradictory: the live trades "
     "were all small (56–635 shares, ₹30–37k) and sat firmly in the safe liquidity zone (all well under 5% of "
     "window volume), which is exactly why every one showed under 5 basis points of slippage. The live book "
     "never entered the danger zone, so the cross-check cannot test it — it only confirms that small orders in "
     "liquid names fill essentially at the reference price.")

# ── 11. OUTPUTS ──
h("11. Output files", 1)
bullet("per_trade_liquidity.csv / .parquet — all 3,436 trades with every measure over both windows.")
bullet("liquidity_proxies.xlsx — sheets: per_trade, per_symbol_profile (689 symbols, worst→best), "
       "least_liquid_trades_top50, shares_pct_distribution.")
bullet("live_slippage_vs_liquidity.csv — the 20-trade cross-check.")
doc.add_paragraph()
para(f"Generated from {len(R):,} per-trade rows across {R['symbol'].nunique():,} symbols. "
     "All figures are OHLCV-derived proxies, not order-book measurements.", italic=True, size=9)

doc.save(str(OUT))
print("saved ->", OUT)
