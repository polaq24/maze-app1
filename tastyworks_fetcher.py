"""Tastyworks (Tastytrade) real-time SPX option chain fetcher.
Richiede: pip install tastytrade
Registrazione: tastytrade.com (deposito $0, OPRA incluso)"""

import logging
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)

# Inserisci qui le tue credenziali Tastyworks
TASTY_USER = ""   # ← inserisci username
TASTY_PASS = ""   # ← inserisci password


def _get_session():
    """Return authenticated Tastytrade session or None."""
    if not TASTY_USER or not TASTY_PASS:
        log.info("Tastyworks credenziali mancanti: imposta TASTY_USER e TASTY_PASS")
        return None
    try:
        from tastytrade import Session
        sess = Session(login=TASTY_USER, password=TASTY_PASS, remember_me=True)
        log.info(f"Tastyworks OK: {sess}")
        return sess
    except Exception as e:
        log.warning(f"Tastyworks login: {e}")
        return None


def fetch_spot() -> Optional[float]:
    """Real-time SPX spot via Tastyworks."""
    sess = _get_session()
    if not sess:
        return None
    try:
        from tastytrade.instruments import InstrumentType
        # Get SPX underlying price
        from tastytrade import instruments
        spx = instruments.get_instrument(sess, "SPX", InstrumentType.EQUITY_INDEX)
        # This approach might use a different API call
        quote = instruments.get_quote(sess, "SPX")
        if quote:
            return float(quote.get("lastPrice") or quote.get("mid") or quote.get("mark"))
    except Exception as e:
        log.debug(f"Tasty spot: {e}")
    return None


def fetch_multiple_chains(min_dte=3, max_dte=120, n_chains=10):
    """Fetch SPX option chains via Tastyworks API.
    Returns list of DataFrames with columns: strike, lastPrice, type, dte, implied_volatility, gamma, open_interest
    """
    sess = _get_session()
    if not sess:
        return None

    try:
        from tastytrade.option_chain import get_option_chain
        from tastytrade.instruments import InstrumentType

        # Get full option chain for SPX
        chain = get_option_chain(sess, "SPX")
        if not chain or not hasattr(chain, "streamer_symbol"):
            log.warning("Tastyworks: no chain data")
            return None

        # Get underlying price from the chain
        S0 = float(chain.underlying_price) if hasattr(chain, "underlying_price") else None

        # Build list of available expirations
        today = date.today()
        expirations = chain.expirations if hasattr(chain, "expirations") else []

        # Filter by DTE range
        exp_dates = []
        for exp in expirations:
            exp_date = exp if isinstance(exp, date) else datetime.strptime(str(exp)[:10], "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                exp_dates.append((dte, exp_date))

        exp_dates.sort()
        if len(exp_dates) > n_chains:
            indices = np.linspace(0, len(exp_dates) - 1, n_chains).astype(int)
            exp_dates = [exp_dates[i] for i in indices]

        chains = []
        for dte, exp_date in exp_dates:
            # Get option chain for this expiration
            try:
                strikes = chain.get_strikes(exp_date)
                if not strikes:
                    continue

                rows = []
                for strike_info in strikes:
                    K = float(strike_info.strike)
                    otype = "C" if strike_info.option_type == "CALL" else "P"

                    # Get quote data
                    try:
                        quote = chain.get_quote(exp_date, K, "CALL" if otype == "C" else "PUT")
                        if quote:
                            last = getattr(quote, "lastPrice", None) or getattr(quote, "close", None)
                            bid = getattr(quote, "bid", None)
                            ask = getattr(quote, "ask", None)
                            price = last or bid or ask
                            iv = getattr(quote, "impliedVolatility", None)
                            gamma = getattr(quote, "gamma", None)
                            oi = getattr(quote, "openInterest", None)

                            if price and price > 0:
                                rows.append({
                                    "strike": K,
                                    "lastPrice": float(price),
                                    "type": otype.lower(),
                                    "dte": dte,
                                    "implied_volatility": float(iv) if iv and iv > 0 else None,
                                    "gamma": float(gamma) if gamma else None,
                                    "open_interest": float(oi) if oi else None,
                                })
                    except Exception:
                        pass

                if rows:
                    df = pd.DataFrame(rows).sort_values("strike")
                    if not df.empty:
                        chains.append(df)
                        log.info(f"  Tasty {exp_date} ({dte} DTE): {len(df)} options")

            except Exception as e:
                log.debug(f"Tasty chain {exp_date}: {e}")

        return chains if chains else None

    except ImportError:
        log.warning("tastytrade non installato: pip install tastytrade")
        return None
    except Exception as e:
        log.warning(f"Tastyworks error: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("=== Tastyworks Test ===")
    print("Prima modifica TASTY_USER e TASTY_PASS nel file!")
    spot = fetch_spot()
    print(f"Spot: {spot}")
    if spot:
        chains = fetch_multiple_chains()
        if chains:
            print(f"Catene: {len(chains)}")
            for df in chains:
                print(f"  DTE {df['dte'].iloc[0]}: {len(df)} options, "
                      f"IV={df['implied_volatility'].mean():.3f}" 
                      if "implied_volatility" in df and df['implied_volatility'].notna().any()
                      else f"  DTE {df['dte'].iloc[0]}: {len(df)} options")
