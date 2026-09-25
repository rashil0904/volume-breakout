# -*- coding: utf-8 -*-
"""make_cs_btst_doc.py — detailed Word write-up of the NIFTY Weekly Credit Spread + Daily Close-Direction
BTST strategies (individual + combined), built from their existing backtest outputs."""
import sys
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

OUT = rb.RESULTS / "combined_cs_btst" / "credit_spread_btst_explained.docx"
COMB = pd.read_excel(rb.RESULTS / "combined_cs_btst" / "combined_credit_spread_btst_report.xlsx", sheet_name="Combined_Summary")
BTG = pd.read_excel(rb.RESULTS / "btst_close_direction" / "btst_close_direction.xlsx", sheet_name="By_Group")
BLUE = RGBColor(0x1F, 0x49, 0x7D)


def H(doc, txt, lvl):
    h = doc.add_heading(txt, level=lvl)
    for r in h.runs: r.font.color.rgb = BLUE
    return h


def P(doc, txt, bold=False, size=10.5):
    p = doc.add_paragraph(); r = p.add_run(txt); r.bold = bold; r.font.size = Pt(size); return p


def B(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet"); r = p.add_run(it); r.font.size = Pt(10.5)


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i].paragraphs[0].add_run(str(h)); c.bold = True; c.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(9.5)
    return t


def metric(name):
    row = COMB[COMB.metric == name]
    return (row.iloc[0]["Credit Spread"], row.iloc[0]["BTST"], row.iloc[0]["Combined"]) if len(row) else ("", "", "")


def main():
    d = Document()
    d.styles["Normal"].font.name = "Calibri"; d.styles["Normal"].font.size = Pt(10.5)
    t = d.add_heading("NIFTY Options Income Strategies", 0)
    for r in t.runs: r.font.color.rgb = BLUE
    sub = d.add_paragraph(); rr = sub.add_run("Weekly Credit Spread  +  Daily Close-Direction BTST — Individual & Combined Report")
    rr.italic = True; rr.font.size = Pt(12); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dd = d.add_paragraph(); r2 = dd.add_run("Backtest window: Oct 2024 - Jul 2026  |  Gross option-premium points  |  No capital / INR / position sizing")
    r2.font.size = Pt(9.5); r2.font.color.rgb = RGBColor(0x66, 0x66, 0x66); dd.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── 1. EXECUTIVE SUMMARY ──
    H(d, "1. Executive Summary", 1)
    P(d, "This report documents two independent NIFTY options-income strategies and their combined performance. "
         "Both are backtested on the same period (Oct 2024 - Jul 2026, limited by NIFTY spot 1-minute coverage ending 15-Jul-2026) "
         "and both report purely in GROSS option-premium points - no capital allocation, no INR sizing, and no position-sizing model. "
         "The 'Combined' view is a pure point-additive stacking of the two, i.e. their per-trade premium P&L is simply summed, "
         "assuming unlimited capacity to run both simultaneously (no shared capital or margin is modelled).")
    P(d, "Headline comparison (gross premium points):", bold=True)
    hs = ["Metric", "Credit Spread", "BTST", "Combined"]
    keys = ["Total return (points)", "Total trades", "Win rate %", "Avg return / trade (points)", "Median return / trade (points)",
            "Avg WIN (points)", "Avg LOSS (points)", "Max profit / best trade (points)", "Max loss / worst trade (points)",
            "Avg drawdown (points)", "Max drawdown (points)", "Avg DD period (days, peak->recovery)", "Max DD period (days)", "Return / Max-DD"]
    table(d, hs, [[k, *metric(k)] for k in keys])
    P(d, "Key takeaways:", bold=True)
    B(d, ["BTST carries the bulk of the return (+9,149 of the combined +11,121 points, ~82%), trading far more often (386 vs 92).",
          "The Credit Spread has a much higher win rate (70.7% vs 52.3%) - it sells premium and wins small, often; BTST is a directional overnight bet that wins ~half the time but with larger winners (best trade +983 vs +93).",
          "DIVERSIFICATION: the combined maximum drawdown (614.6) is well below the sum of the two individual max drawdowns (530.6 + 402.6 = 933) - because their drawdowns do not fully coincide, running both together is smoother than either alone would imply additively.",
          "Both are GROSS (no transaction cost / slippage). Each is multi-leg and, for BTST, held overnight - real costs would materially reduce these figures."])

    # ── 2. CREDIT SPREAD ──
    H(d, "2. Strategy A - NIFTY Weekly Credit Spread", 1)
    H(d, "2.1 Concept", 2)
    P(d, "A weekly premium-selling strategy. Each week it sells a 200-point-wide vertical credit spread on NIFTY options in the "
         "direction of a breakout (or an end-of-day fallback), collects the net credit, and aims to buy the spread back cheap once "
         "most of the credit has decayed. It is a high-win-rate, limited-risk income strategy: many small wins, occasional larger losses "
         "when the market runs through the short strike.")
    H(d, "2.2 Weekly cycle & entry day", 2)
    B(d, ["The cycle is tied to ACTUAL NIFTY expiries (derived from the pulled options calendar - it correctly handles the historical Thursday->Tuesday expiry shift and holiday-adjusted expiries; no weekday is hard-coded).",
          "Entry day = the first trading session after the previous week's expiry. DTE (calendar days to the current expiry) therefore varies naturally - 3 to 7 across the sample, averaging ~5.8 - depending on holidays.",
          "Only ONE trade is taken per weekly cycle."])
    H(d, "2.3 Entry logic (direction)", 2)
    B(d, ["Reference levels: the 3-day High and 3-day Low = the highest high / lowest low of the 3 completed trading days before the entry day.",
          "Breakout (checked intraday, touch-basis, throughout the entry day): if NIFTY's low touches/breaks the 3-day Low -> BEARISH -> Call Credit Spread (sell ATM CE, buy CE 200 points higher). If the high breaks the 3-day High -> BULLISH -> Put Credit Spread (sell ATM PE, buy PE 200 points lower). Whichever breaks first triggers the trade.",
          "2:30pm fallback: if neither level is breached by 2:30pm, compare spot to the day's open - RED (below open) -> Call Credit Spread; GREEN (above open) -> Put Credit Spread. (A fallback-time sweep confirmed 2:30pm is already near-optimal; later checks are worse.)",
          "ATM strike = nearest 50 to the spot at the moment of entry. The spread is 200 points wide."])
    H(d, "2.4 Exit logic", 2)
    B(d, ["90% profit target (checked every 1 minute, intraday, every day from entry to expiry): the moment the cost to CLOSE the spread (buy back short + sell long) decays to 10% of the entry net credit (i.e. 90% of max profit captured), the spread is closed at market.",
          "DTE-0 settlement (if the target never hits): squared off at the actual expiry settlement = the spread's intrinsic value at NIFTY's settlement price (average of the last 30 minutes of spot on expiry day).",
          "No stop loss - a losing spread is held to settlement.",
          "P&L = net credit received - exit debit paid. Max profit = net credit (~79 avg); max loss = 200 - net credit (~-121 worst)."])
    H(d, "2.5 Results", 2)
    csm = pd.read_excel(rb.RESULTS / "weekly_credit_spread" / "weekly_credit_spread.xlsx", sheet_name="Summary")
    B(d, [f"92 trades, 70.7% win rate, +1,972 total premium points, +21.4 avg per trade.",
          "About 66% of trades exit via the 90% target (clean, fast wins); ~34% run to expiry settlement - and the settlement group contains the losers (a spread that finishes in-the-money can lose up to ~120 points, wiping out several target-wins).",
          "So the monthly result is essentially driven by how many trades reach the profit target vs how many drift to settlement in-the-money.",
          "Max drawdown 530.6 points; the longest underwater stretch was ~260 days (return/max-DD 3.72). A high win rate but a thin, insurance-like edge.",
          "Robustness caveat: the strategy's edge is heavily in-sample - out-of-sample P&L (2025+) is much thinner than in-sample."])
    P(d, "Workbook: results/weekly_credit_spread/weekly_credit_spread.xlsx (Summary, Monthly_PnL, Drawdowns, Trades, DTE_Distribution).", size=9)

    # ── 3. BTST ──
    H(d, "3. Strategy B - NIFTY Daily Close-Direction BTST", 1)
    H(d, "3.1 Concept", 2)
    P(d, "A Buy-Today-Sell-Tomorrow (BTST) directional overnight strategy. Every trading day near the close it reads the day's "
         "direction (up or down vs the open) and buys a 2:1 directional option ratio betting that direction continues into the next "
         "morning, exiting shortly after the next day's open. It is a momentum/continuation bet on the overnight gap.")
    H(d, "3.2 Entry logic", 2)
    B(d, ["Entry at 3:20pm each trading day (finalised from a 3:00-3:29pm entry-time sweep). Direction = spot at 3:20pm vs that day's opening price.",
          "RED (spot below open): buy 2 lots ATM PE + sell 1 lot (ATM-300) PE. GREEN (spot above open): buy 2 lots ATM CE + sell 1 lot (ATM+300) CE. This is a net-long directional structure (2 long, 1 short) that cheapens the long via the sold wing.",
          "ATM = nearest 50 to the 3:20pm spot. The sold wing is 300 points away (finalised from a sell-offset sweep 100-300, where wider was better gross).",
          "Maximum 1 trade per day."])
    H(d, "3.3 Contract selection (never DTE-0)", 2)
    B(d, ["The position never uses a contract that is at DTE 0 on the entry day. Equivalently, the contract used is the nearest expiry STRICTLY AFTER the entry day.",
          "On a normal day this is the current week's contract; on EXPIRY day itself it automatically becomes the NEXT week's contract (verified: the next-week chain is already listed and liquid at 3:20pm on expiry day)."])
    H(d, "3.4 Exit & India VIX filter", 2)
    B(d, ["Exit at 9:17am the next trading day (finalised from an entry x exit time-grid sweep - exiting early, right after the open, clearly beat the original 9:25am; the overnight edge decays quickly through the morning).",
          "India VIX 17-19 filter: trades entered when India VIX (at 3:20pm) is between 17 and 19 are EXCLUDED - a prior VIX-bucket analysis identified that band as a consistent net-loser (elevated-but-not-extreme volatility = choppy, directionless overnight moves).",
          "No stop loss."])
    H(d, "3.5 Results", 2)
    def g(name): return BTG[BTG.group == name].iloc[0]
    B(d, [f"386 trades (after the VIX filter removed 48), 52.3% win rate, +9,149 total premium points, +23.7 avg per trade.",
          f"The VIX filter lifted total P&L from ~8,182 to 9,149 and win rate 50.9% -> 52.3%, and the earlier 9:17 exit + filter together cut max drawdown ~3.5x (from ~1,413 to 402.6).",
          f"RED days (down-day -> long puts) out-earn GREEN days: RED {int(g('RED').trades)} trades, +{g('RED').total_pnl:.0f} ({g('RED').avg_pnl:.1f}/trade) vs GREEN {int(g('GREEN').trades)} trades, +{g('GREEN').total_pnl:.0f} ({g('GREEN').avg_pnl:.1f}/trade) - consistent with NIFTY's sharper down-moves.",
          f"Expiry-day trades (forced onto the next-week contract) are positive but lower-yield (+{g('expiry-day (next-week)').avg_pnl:.1f}/trade) than normal-day trades (+{g('normal-day (current-week)').avg_pnl:.1f}/trade), because the next-week contract has more time value so a fixed overnight move captures a smaller % of premium.",
          "Max drawdown 402.6 points; return/max-DD 22.73 - a much smoother curve than the credit spread."])
    P(d, "Workbook: results/btst_close_direction/btst_close_direction.xlsx (Summary, By_Group, Monthly_PnL, Drawdowns, Trades).", size=9)

    # ── 4. COMBINED ──
    H(d, "4. Combined (Point-Additive) Analysis", 1)
    P(d, "The combined report stacks the two strategies with no capital constraint: each trade's premium P&L is summed chronologically "
         "into one equity curve. Because there is no shared capital pool, days where both strategies trade need no conflict resolution - "
         "they are simply added independently.")
    B(d, ["Combined: 478 trades, 55.9% win rate, +11,121 total premium points, +23.3 avg per trade.",
          "The two are complementary: the Credit Spread is a high-win-rate, small-magnitude premium seller; BTST is a lower-win-rate, larger-magnitude directional bet. Blending them raises the overall win rate above BTST alone and adds a steady premium-selling stream.",
          "Diversification benefit (the key result): combined max drawdown = 614.6 points, versus 933 if the two worst stretches coincided (530.6 + 402.6). Their drawdowns are only partly correlated, so the combined equity is more robust than either alone. Combined return/max-DD = 18.09, between the two.",
          "CAPACITY FLAG: this assumes unlimited capacity to run both simultaneously. A real book would face capital/margin limits (each is multi-leg; BTST also carries overnight), which this pure point-sum does not model."])
    P(d, "Workbook: results/combined_cs_btst/combined_credit_spread_btst_report.xlsx (Combined_Summary, Combined_TradeLog, Credit_Spread_Trades, BTST_Trades).", size=9)

    # ── 5. ASSUMPTIONS ──
    H(d, "5. Assumptions, Caveats & Data", 1)
    B(d, ["UNIT: gross option-premium points only. No brokerage, no slippage, no taxes, no INR sizing, no capital/margin model. Both strategies are multi-leg (credit spread = 2 legs; BTST = 3 legs), and BTST is held overnight - real-world costs would materially reduce all figures.",
          "PERIOD: Oct 2024 - Jul 2026. The options dataset runs to 31-Jul-2026 but NIFTY spot 1-minute (needed for entry direction/ATM and settlement) ends 15-Jul-2026, so the last ~2 expiry cycles are excluded.",
          "FILLS: the credit spread's 90% target and DTE-0 settlement, and BTST's entries/exits, use option 1-minute prices (close/asof or the settlement intrinsic). No bid/ask spread is available in the data, so mid/last-trade prices are used - a real book would fill worse.",
          "DATA: NIFTY spot 1-min, NIFTY options 1-min full chain (Oct 2024 - Jul 2026), India VIX 1-min (for the BTST filter), and the actual NSE expiry calendar (derived from the options folders).",
          "ROBUSTNESS: sweeps (fallback time, entry/exit grid, sell offset, VIX filter) were run with an in-sample / out-of-sample split; the BTST timing/VIX choices held out-of-sample, while the credit spread's edge is thinner out-of-sample."])

    OUT.parent.mkdir(parents=True, exist_ok=True); d.save(OUT)
    print(f"Saved -> {OUT}  ({len(d.paragraphs)} paragraphs)")


if __name__ == "__main__":
    main()
