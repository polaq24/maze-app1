"""dxFeed connector — REST API primario, WebSocket fallback.
Il REST pubblico è gratuito e non richiede auth (dati delayed ~15min).
Per real-time servono credenziali valide e endpoint corretto."""

import logging, sys
from datetime import datetime, date
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# REST API pubblica (gratuita, delayed)
REST = "https://tools.dxfeed.com/rest"

# WebSocket endpoints (tentativo live con credenziali Deepchart)
WS_LIVE = "wss://get-prod-dxlink.dxfeed.com/"
WS_DELAYED = "wss://get-prod-dxlink-d15m.dxfeed.com/delayed"
USER = "MFFUMmyGZbweas"
PASS = "4dy4JH3d!UniV=sU?IYC"


def fetch_spot():
    """SPX spot via REST. Gratuito, delayed ~15min."""
    import requests
    try:
        r = requests.get(f"{REST}/event/quote/SPX", timeout=10)
        if r.status_code != 200:
            log.warning(f"REST quote: HTTP {r.status_code}")
            return None
        data = r.json()
        # dxFeed REST quote format: { "Quote": { "SPX": { "bidPrice": X, "askPrice": Y } } }
        quote = data.get("Quote", {}).get("SPX", {})
        for field in ("askPrice", "bidPrice", "lastPrice", "closePrice"):
            v = quote.get(field)
            if v and v > 0:
                return float(v)
        return None
    except Exception as e:
        log.debug(f"REST spot: {e}")
        return None


def _rest_option_chain(expiry_date: date):
    """Fetch option chain for a specific expiry via dxFeed REST API.
    Returns DataFrame or None."""
    import requests
    try:
        url = f"{REST}/event/option/sale/SPX?date={expiry_date.isoformat()}"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        # Format: { "OptionSale": { "SPX": [...] } }
        sales = data.get("OptionSale", {}).get("SPX", [])
        if not sales:
            return None
        rows = []
        for s in sales:
            strike = s.get("strikePrice")
            opt_type = s.get("optionType", "").upper()
            price = s.get("lastPrice") or s.get("bidPrice") or s.get("askPrice")
            if strike and opt_type in ("CALL", "PUT", "C", "P") and price:
                rows.append({
                    "strike": float(strike),
                    "lastPrice": float(price),
                    "type": "call" if opt_type in ("CALL", "C") else "put",
                    "implied_volatility": s.get("volatility"),
                })
        if rows:
            return pd.DataFrame(rows).sort_values("strike")
        return None
    except Exception as e:
        log.debug(f"REST chain {expiry_date}: {e}")
        return None


def fetch_chains(min_dte=3, max_dte=120, n_chains=10):
    """Fetch SPX option chains via dxFeed REST API.
    Returns list of DataFrames or None."""
    # Fast-fail: verifica se REST è raggiungibile
    import requests
    try:
        probe = requests.get(f"{REST}/event/quote/SPX", timeout=5)
        if probe.status_code != 200:
            log.info("dxFeed REST non disponibile (fallback immediato)")
            return None
    except Exception as e:
        log.debug(f"dxFeed REST fail: {e}")
        return None
    # Build expiration list
    today = date.today()
    exps = []
    for d in range(max_dte + 1):
        dt = __import__("datetime").timedelta(days=d) + today
        dte = (dt - today).days
        if dte >= min_dte and dt.weekday() in (0, 2, 4):
            exps.append((dte, dt))
    exps.sort()
    if len(exps) > n_chains:
        indices = np.linspace(0, len(exps) - 1, n_chains).astype(int)
        exps = [exps[i] for i in indices]

    chains = []
    for dte, exp_date in exps:
        log.info(f"REST chain: {exp_date} ({dte} DTE)...")
        df = _rest_option_chain(exp_date)
        if df is not None and not df.empty:
            df["dte"] = dte
            chains.append(df)

    return chains if chains else None


# =====================================================================
# WebSocket tentativo (per real-time, se le credenziali funzionano)
# =====================================================================

def _ws_fetch_spot():
    """Tenta WebSocket live. Richiede websockets package."""
    try:
        import websockets
    except ImportError:
        return None

    async def _run():
        import websockets as ws
        for url in [WS_LIVE, WS_DELAYED]:
            try:
                log.info(f"WS connect: {url}")
                conn = await asyncio.wait_for(ws.connect(url, max_size=2**24, ping_interval=30), timeout=10)
                # Handshake
                for _ in range(10):
                    msg = await asyncio.wait_for(conn.recv(), timeout=5)
                    log.info(f"  WS: {msg[:80]}")
                    if msg == "UNAUTHORIZED":
                        token = base64.b64encode(f"{USER}:{PASS}".encode()).decode()
                        await conn.send(f"auth:{token}")
                        resp = await asyncio.wait_for(conn.recv(), timeout=5)
                        log.info(f"  WS auth: {resp[:80]}")
                        if "auth:" in resp or "protocol:" in resp:
                            break
                    if msg.startswith("protocol:") or msg.startswith("auth:"):
                        break
                # Subscribe SPX
                await conn.send("+Quote:SPX")
                t0 = datetime.now()
                while (datetime.now() - t0).seconds < 15:
                    msg = await asyncio.wait_for(conn.recv(), timeout=3)
                    if "SPX" in msg and "Quote" in msg:
                        parts = msg.split(",")
                        for p in parts[2:5]:
                            try:
                                v = float(p)
                                if v > 0:
                                    await conn.close()
                                    return v
                            except ValueError:
                                pass
                await conn.close()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.info(f"  WS fail: {e}")
                continue
        return None

    import asyncio, base64
    try:
        return asyncio.run(_run())
    except Exception as e:
        log.debug(f"WS spot: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("--- REST API (delayed, gratuito) ---")
    spot = fetch_spot()
    print(f"SPX spot (REST): {spot}")

    if spot:
        chains = fetch_chains()
        if chains:
            print(f"Catene: {len(chains)}")
            for df in chains:
                dte = df["dte"].iloc[0]
                print(f"  DTE {dte}: {len(df)} opzioni")
        else:
            print("Nessuna chain via REST")

    print("\n--- WebSocket (real-time se credenziali valide) ---")
    ws_spot = _ws_fetch_spot()
    print(f"SPX spot (WS): {ws_spot}")

