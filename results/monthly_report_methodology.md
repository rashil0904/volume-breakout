# Monthly Performance Report — Methodology & Worked Example

---

## 1. What the Report Measures

Each of the 8 sheets in `monthly_performance_report.xlsx` covers one combination of:
- **Parameter set** (lookback window + volume multiplier threshold)
- **Exit strategy** (Standard 3pm next-day OR Split 9:45am + 11am next-day)

Every row in a sheet is one **calendar month**. The 12 columns tell you how that month's trades performed, measured in real rupees against a fixed ₹6,00,000 capital pool.

---

## 2. How a Trade Is Generated

### Step 1 — Signal conditions (evaluated at 3:00 PM each trading day)

Three conditions must ALL be true for a stock to trigger:

| # | Condition | Threshold |
|---|---|---|
| 1 | Market cap (₹ Cr) | ₹1,500–5,000 Cr (from closest preceding NSE semi-annual snapshot) |
| 2 | Volume ratio: cumulative vol 9:15–14:45 ÷ rolling-average full-day vol | ≥ vol_mult (e.g. 6× or 7×) |
| 3 | Open of 15:00 candle ≥ X% above prev-day VWAP close | ≥ 5% |

*Prev-day VWAP close = volume-weighted avg price of the 15:00 and 15:15 candles of the previous trading day.*  
*Rolling average uses the last N full trading days (e.g. N=30 or 36), excluding zero-volume days.*

### Step 2 — Entry

The trade is entered at the **open of the 15:15 candle** (3:15pm) on the signal day.

### Step 3 — Exit

| Exit Strategy | When | How |
|---|---|---|
| Standard | Open of next-day 15:00 candle (3:00 PM) | 100% of shares |
| Split | Open of next-day 09:45 candle (9:45 AM) | 50% of shares |
| | Open of next-day 11:00 candle (11:00 AM) | remaining 50% |

---

## 3. Position Sizing

**Pool:** ₹6,00,000 (fixed for life of backtest — no compounding)

| # signals that day | Target per stock |
|---|---|
| 1 – 6 | ₹1,00,000 |
| 7 or more | ₹6,00,000 ÷ n (split equally) |

**Shares bought** = `floor(target ÷ entry_price)`  
If floor gives 0 shares, that trade is skipped.  
**Actual capital** = `shares × entry_price`

### Example (Feb 15, 2022 — 2 signals)

| Stock | Entry price | Target | Shares (floor) | Actual capital |
|---|---:|---:|---:|---:|
| EXCELINDUS | ₹1,244.10 | ₹1,00,000 | floor(100000 ÷ 1244.10) = **80** | 80 × 1244.10 = ₹99,528 |
| VISHNU | ₹236.25 | ₹1,00,000 | floor(100000 ÷ 236.25) = **423** | 423 × 236.25 = ₹99,934 |

*2 signals ≤ 6, so each gets ₹1L target. Actual capital is slightly less due to floor rounding.*

---

## 4. Worked Example — February 2022

### All trades that month (Baseline LB=30, VM=7, Standard 3pm exit)

| Date | Symbol | n that day | Entry (₹) | Exit 3pm (₹) | Shares | Capital (₹) | P&L (₹) | Return % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Feb 15 | EXCELINDUS | 2 | 1,244.10 | 1,309.00 | 80 | 99,528 | +5,192 | +5.22% |
| Feb 15 | VISHNU | 2 | 236.25 | 248.05 | 423 | 99,934 | +4,991 | +4.99% |
| Feb 16 | TIIL | 1 | 875.50 | 883.95 | 114 | 99,807 | +963 | +0.97% |
| Feb 18 | BBOX | 1 | 174.35 | 163.20 | 573 | 99,903 | −6,389 | −6.40% |
| Feb 25 | ADVENZYMES | 1 | 310.40 | 318.35 | 322 | 99,949 | +2,560 | +2.56% |
| Feb 28 | ALEMBICLTD | 2 | 89.70 | 87.15 | 1,114 | 99,926 | −2,841 | −2.84% |
| Feb 28 | KSB | 2 | 219.40 | 219.20 | 455 | 99,827 | −91 | −0.09% |

---

## 5. Column-by-Column Calculation

Below, every column is explained and computed from the Feb 2022 example above.

---

### Column 1 — Revenue (₹)

**What it is:** Sum of all P&L rupees from trades entered in this month.

**Formula:**
```
Revenue = Σ pnl  for all trades in month
```

**Feb 2022 example:**
```
5,192 + 4,991 + 963 − 6,389 + 2,560 − 2,841 − 91  =  ₹4,386
```

---

### Column 2 — Biggest Single-Day Loss (₹)

**What it is:** The single worst calendar day in the month (most negative daily total P&L). Useful for stress-testing: "what's the most I could lose in one afternoon?"

**Formula:**
```
For each trading day: daily_pnl = Σ pnl of all trades entered that day
Biggest 1-Day Loss = min(daily_pnl values in month)
```

**Feb 2022 daily totals:**

| Date | Stocks | Daily P&L |
|---|---|---:|
| Feb 15 | EXCELINDUS + VISHNU | +10,183 |
| Feb 16 | TIIL | +963 |
| **Feb 18** | **BBOX** | **−6,389** |
| Feb 25 | ADVENZYMES | +2,560 |
| Feb 28 | ALEMBICLTD + KSB | −2,932 |

**Biggest 1-Day Loss = −₹6,389** (Feb 18 when BBOX fell 6.4%)

---

### Column 3 — Max Drawdown (₹ and %)

This is the most nuanced column. It captures **how far the portfolio fell from its previous all-time high**, across a **single unbroken equity curve** that starts on Day 1 of the entire backtest (not just the month).

#### Why a continuous curve matters

If January ends badly and February starts at a new low, the February drawdown should reflect the full damage since the last peak — even if that peak was in December. Resetting each month would hide a losing streak that spans months.

#### Step-by-step construction

**Step 1: Build the equity curve (runs across the entire backtest)**

Start at ₹6,00,000. Each trading day, add that day's net P&L:
```
Equity[day] = Equity[day−1] + daily_pnl[day]
```
(Non-trading days are skipped — equity stays flat.)

**Step 2: Track the running peak**

At every point, record the highest equity seen so far:
```
Running_Peak[day] = max(Equity[Day 1], Equity[Day 2], ..., Equity[day])
```
This peak only ever goes up; it never resets.

**Step 3: Compute drawdown**

```
Drawdown_₹[day]   = Running_Peak[day] − Equity[day]      (always ≥ 0)
Drawdown_%[day]   = Drawdown_₹[day] / Running_Peak[day] × 100
```

**Step 4: For each month, report the worst (maximum) drawdown**

```
Max_DD_₹  for Month = max( Drawdown_₹[day]  for all trading days in that month )
Max_DD_%  for Month = Drawdown_% at the day when Drawdown_₹ was maximum
```

#### Feb 2022 equity curve (assuming ₹6L start, no prior months for simplicity)

| Date | Daily P&L | Equity | Running Peak | DD (₹) | DD (%) |
|---|---:|---:|---:|---:|---:|
| Feb 15 | +10,183 | 6,10,183 | **6,10,183** | 0 | 0.00% |
| Feb 16 | +963 | 6,11,147 | **6,11,147** | 0 | 0.00% |
| Feb 18 | −6,389 | 6,04,758 | 6,11,147 | 6,389 | 1.05% |
| Feb 25 | +2,560 | 6,07,318 | 6,11,147 | 3,829 | 0.63% |
| Feb 28 | −2,932 | 6,04,386 | 6,11,147 | **6,761** | **1.11%** |

**Max Drawdown for Feb 2022 = ₹6,761 (1.11%)**  
It occurred on Feb 28 — the equity never recovered back to the Feb 16 peak before the month ended.

Note: In a real multi-year run, the Running Peak might be from a much earlier month. For example, if the portfolio peaked in October 2024 at ₹18L and dropped to ₹16L in February 2025, the Feb 2025 drawdown would be measured against ₹18L, not just February's own starting value.

---

### Column 4 — Trades Per Month

**What it is:** Count of individual stock-trades executed with an entry in this month.

**Feb 2022:** 7 trades (EXCELINDUS, VISHNU, TIIL, BBOX, ADVENZYMES, ALEMBICLTD, KSB)

---

### Column 5 — Trading Sessions Per Month

**What it is:** Count of distinct calendar days on which at least one trade was entered.

**Feb 2022:** 5 sessions (Feb 15, 16, 18, 25, 28)

*Note: This is never more than the number of trading days in the month, but is often less — most days no signal fires.*

---

### Column 6 — Win % and Loss %

**What it is:** Share of individual trades (not days) that made money vs. lost money.

**Formula:**
```
Win %  = (trades where pnl > 0) / total_trades × 100
Loss % = (trades where pnl ≤ 0) / total_trades × 100
```

**Feb 2022:**
- Winners: EXCELINDUS, VISHNU, TIIL, ADVENZYMES → 4
- Losers:  BBOX, ALEMBICLTD, KSB → 3
- **Win % = 4/7 × 100 = 57.14%,  Loss % = 42.86%**

Note: Win % + Loss % = 100% always. Break-even trades (pnl = 0 exactly) count as losses.

---

### Column 7 — Avg Win Return % and Avg Loss Return %

**What it is:** Among winning trades only, the average percentage return. Separately for losing trades. This shows whether wins are large or small relative to losses.

**Formula:**
```
return_pct = (exit_price − entry_price) / entry_price × 100   [per trade]

Avg Win Return  % = mean( return_pct  for trades where pnl > 0 )
Avg Loss Return % = mean( return_pct  for trades where pnl ≤ 0 )
```

**Feb 2022:**
- Winning returns: +5.22%, +4.99%, +0.97%, +2.56%  →  avg = **+3.43%**
- Losing returns:  −6.40%, −2.84%, −0.09%           →  avg = **−3.11%**

*Interpretation: Wins average +3.43%, losses average −3.11%. The strategy has a slight edge on magnitude of wins vs losses, plus >50% win rate → positive expected value.*

---

### Column 8 — Average Return Per Trade %

**What it is:** Mean return_pct across ALL trades in the month, winners and losers combined. The single best one-number summary of the month's performance per trade.

**Formula:**
```
Avg Return / Trade % = mean( return_pct  for ALL trades in month )
```

**Feb 2022:**
```
(5.22 + 4.99 + 0.97 − 6.40 + 2.56 − 2.84 − 0.09) / 7  =  4.41 / 7  =  +0.63%
```

**Feb 2022 Average Return Per Trade = +0.63%**

*This means on average each trade earned 0.63% of its entry price. Multiplied by ~₹1L per trade = ~₹630 expected value per trade.*

---

## 6. How the Two Exit Strategies Differ in Calculation

### Standard (3pm next-day)
```
pnl        = shares × (exit_3pm_open − entry_price)
return_pct = (exit_3pm_open − entry_price) / entry_price × 100
```

### Split (50% at 9:45am + 50% at 11am next-day)
```
shares_first  = floor(total_shares / 2)
shares_second = total_shares − shares_first    [one extra share if odd]

pnl = shares_first × (exit_945_open − entry_price)
    + shares_second × (exit_1100_open − entry_price)

blended_exit_price = (exit_945_open + exit_1100_open) / 2
return_pct = (blended_exit_price − entry_price) / entry_price × 100
```

The split exit captures some early-morning momentum (9:45am) while leaving half the position to benefit if the rally continues to 11am.

---

## 7. Quick Sanity Check

After reading any month's row, you can roughly verify:

```
Revenue ≈ Avg Return/Trade % × (₹1L per trade) × Trades
```

**Feb 2022:** 0.63% × ₹1,00,000 × 7 = ₹4,410  ✓ (actual ₹4,386, difference due to rounding)

And:
```
Avg Return/Trade % ≈ (Win% × Avg Win%) + (Loss% × Avg Loss%)
                   = (0.5714 × 3.43%) + (0.4286 × −3.11%)
                   = 1.96% − 1.33% = +0.63%  ✓
```

---

*Generated by `monthly_report.py`. All trades use 3:15pm entry. P&L is booked on entry date. Equity curve starts at ₹6,00,000 on the first trading day of the backtest and never resets.*
