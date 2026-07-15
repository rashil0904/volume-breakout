"""
Upstox access tokens are valid only until ~3:30 AM the next trading day, so this
needs to be run once per day before the market session (e.g. each morning).

Flow:
  1. Run this script. It prints a login URL.
  2. Open that URL in a browser, log in to Upstox, approve access.
  3. Upstox redirects you to your REDIRECT_URI with `?code=XXXX` in the URL.
  4. Copy that `code` value and paste it back here when prompted.
  5. The script exchanges it for an access_token and caches it to TOKEN_FILE.

This is manual by design — Upstox's login step requires your password + OTP,
which isn't something to automate/store in a script.

Sandbox note:
  When SANDBOX=True, this script uses a Sandbox App's API key/secret.
  The token produced is a sandbox token and only works against sandbox endpoints.
  It is saved to sandbox_access_token.json (never overwrites a live token).
"""

import json
import requests
from urllib.parse import urlencode
import config

BASE_AUTH_URL = "https://api.upstox.com/v2"


def get_login_url():
    params = {
        "client_id":     config.API_KEY,
        "redirect_uri":  config.REDIRECT_URI,
        "response_type": "code",
    }
    return f"{BASE_AUTH_URL}/login/authorization/dialog?" + urlencode(params)


def exchange_code_for_token(auth_code: str) -> dict:
    url = f"{BASE_AUTH_URL}/login/authorization/token"
    headers = {
        "accept":       "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "code":          auth_code,
        "client_id":     config.API_KEY,
        "client_secret": config.API_SECRET,
        "redirect_uri":  config.REDIRECT_URI,
        "grant_type":    "authorization_code",
    }
    resp = requests.post(url, headers=headers, data=data)
    resp.raise_for_status()
    token_data = resp.json()

    with open(config.TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)

    return token_data


def load_access_token() -> str:
    try:
        with open(config.TOKEN_FILE) as f:
            token_data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"Token file '{config.TOKEN_FILE}' not found.\n"
            f"Run:  python upstox_auth.py"
        )
    return token_data["access_token"]


if __name__ == "__main__":
    mode = "SANDBOX" if config.SANDBOX else "LIVE"
    print(f"=== Upstox Auth  [{mode}] ===")
    print(f"Token will be saved to: {config.TOKEN_FILE}\n")
    print("1. Open this URL in your browser and log in:\n")
    print(get_login_url())
    print("\n2. After approving, you'll be redirected to a URL containing '?code=...'")
    code = input("3. Paste the 'code' value here: ").strip()

    token_data = exchange_code_for_token(code)
    print(f"\nAccess token saved to {config.TOKEN_FILE}")
    print("Token valid until roughly 3:30 AM tomorrow.")
