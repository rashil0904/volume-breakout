"""
Fill these in from your Upstox Developer App (https://developer.upstox.com/apps).
API_KEY / API_SECRET come from your registered app.
REDIRECT_URI must exactly match the one configured in your app settings.
"""

API_KEY      = "YOUR_API_KEY_HERE"
API_SECRET   = "YOUR_API_SECRET_HERE"
REDIRECT_URI = "https://your-redirect-uri.com"   # must match app config exactly

# ── Sandbox mode ──────────────────────────────────────────────────────────────
# Set True to use sandbox endpoints and a separate token file.
# Set False only when you are ready for live trading (ask before changing).
SANDBOX = True

# Token files — kept separate so a sandbox token can never reach a live endpoint
TOKEN_FILE          = "sandbox_access_token.json" if SANDBOX else "access_token.json"

# Order placement URL
ORDER_URL = (
    "https://sandbox.upstox.com/v3/order/place"
    if SANDBOX else
    "https://api-hft.upstox.com/v3/order/place"
)

# Where open positions get tracked between the buy leg and the sell leg
POSITIONS_FILE = "positions.json"

# "D" = delivery (CNC), "I" = intraday (MIS)
# Holding overnight → use delivery.
PRODUCT_TYPE = "D"
