"""Quant Analytics Engine: GEX, Monte Carlo, Macro, PCA.
Tutti i calcoli istituzionali OPRA-free, basati su OpenBB + yfinance."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from datetime import datetime, date
from typing import Optional
import logging

log = logging.getLogger(__name__)

# =============================================================================
# GAMMA EXPOSURE (GEX)
# =============================================================================

def compute_gex(option_df: pd.DataFrame, spot: float) -> dict:
    """Calcola GEX per strike da option chain.
    
    GEX = gamma × 100 × OI × spot × (±1)
    gamma = per-azione (BS); 100 = moltiplicatore contratto
    
    Args:
        option_df: DataFrame con colonne strike, option_type, gamma, open_interest, lastPrice
        spot: prezzo corrente SPX
        
    Returns:
        dict con:
        - strikes: array di strike
        - gex: array di GEX per strike
        - total_gex: GEX totale
        - zero_gamma: livello zero gamma (interpolato)
        - max_gex_strike: strike con GEX massimo positivo
        - min_gex_strike: strike con GEX massimo negativo
    """
    if option_df is None or option_df.empty:
        return {"strikes": np.array([]), "gex": np.array([]),
                "total_gex": 0, "zero_gamma": spot,
                "max_gex_strike": spot, "min_gex_strike": spot}
    
    has_gamma = "gamma" in option_df.columns
    has_oi = "open_interest" in option_df.columns
    has_type = "option_type" in option_df.columns
    
    if not has_gamma:
        log.debug("GEX: gamma non disponibile")
        return {"strikes": np.array([]), "gex": np.array([]),
                "total_gex": 0, "zero_gamma": spot,
                "max_gex_strike": spot, "min_gex_strike": spot}
    
    df = option_df.copy()
    if has_type:
        df["otype"] = df["option_type"].str.upper().str[0]
    else:
        df["otype"] = "C"
    
    # Determina peso per GEX: preferisce volume se OI è sparso lontano da spot
    def _near_pct(col):
        tot = df[col].fillna(0).sum()
        if tot <= 0: return 0
        near = df[df["strike"].between(spot*0.92, spot*1.08)][col].fillna(0).sum()
        return near / tot
    oi_ok = has_oi and df["open_interest"].fillna(0).sum() > 0
    vol_ok = "volume" in df.columns and df["volume"].fillna(0).sum() > 0
    oi_near = _near_pct("open_interest") if oi_ok else 0
    vol_near = _near_pct("volume") if vol_ok else 0
    if oi_ok and oi_near >= 0.3:
        weight_col = "open_interest"
    elif vol_ok and vol_near >= 0.3:
        weight_col = "volume"
    elif oi_ok:
        weight_col = "open_interest"
    elif vol_ok:
        weight_col = "volume"
    else:
        weight_col = None
    
    if weight_col:
        df["gex"] = df.apply(
            lambda r: float(r["gamma"]) * 100 * float(r[weight_col]) * spot *
                      (1 if r.get("otype", "C") == "C" else -1),
            axis=1
        )
    else:
        # Gamma pura (unweighted) — mostra profilo gamma aggregato per strike
        df["gex"] = df.apply(
            lambda r: float(r["gamma"]) * 100 * spot *
                      (1 if r.get("otype", "C") == "C" else -1),
            axis=1
        )
    
    # Aggrega per strike
    by_strike = df.groupby("strike")["gex"].sum().sort_index()
    
    strikes = by_strike.index.values
    gex = by_strike.values
    
    total_gex = gex.sum()
    
    # Zero gamma level: cumulativo (net gamma totale incrocia zero)
    if len(gex) == 0 or np.all(gex == 0):
        return {"strikes": np.array([]), "gex": np.array([]),
                "total_gex": 0, "zero_gamma": spot,
                "max_gex_strike": spot, "min_gex_strike": spot}

    cumsum = np.cumsum(gex)
    zero_gamma = spot
    # Se tutto il GEX e' dello stesso segno, ZG e' oltre la catena
    if gex.min() >= 0:
        # Net long gamma a tutti gli strike → ZG < strike minimo
        zg_src = "beyond_low"
    elif gex.max() <= 0:
        # Net short gamma a tutti gli strike → ZG > strike massimo
        zg_src = "beyond_high"
    else:
        zg_src = "interp"
        for i in range(len(cumsum) - 1):
            if cumsum[i] == 0:
                continue
            if cumsum[i] * cumsum[i + 1] < 0:
                w = abs(cumsum[i]) / (abs(cumsum[i]) + abs(cumsum[i + 1]))
                zero_gamma = strikes[i] * (1 - w) + strikes[i + 1] * w
                break
        else:
            # Nessun incrocio nonostante segni misti → weighted avg
            zg_src = "weighted"
            w_sum = np.abs(gex).sum()
            if w_sum > 0:
                zero_gamma = np.average(strikes, weights=np.abs(gex))
    
    if zg_src in ("beyond_low", "beyond_high"):
        # ZG non osservabile → weighted average (centro di massa GEX)
        w_sum = np.abs(gex).sum()
        if w_sum > 0:
            zero_gamma = np.average(strikes, weights=np.abs(gex))
        # Satura entro la catena
        zero_gamma = np.clip(zero_gamma, strikes.min(), strikes.max())
    
    max_gex_idx = np.argmax(gex)
    min_gex_idx = np.argmin(gex)
    
    return {
        "strikes": strikes,
        "gex": gex,
        "total_gex": total_gex,
        "zero_gamma": zero_gamma,
        "max_gex_strike": strikes[max_gex_idx],
        "min_gex_strike": strikes[min_gex_idx],
        "max_gex": gex[max_gex_idx],
        "min_gex": gex[min_gex_idx],
    }


# =============================================================================
# EXPOSURE PROFILE (GEX + Delta + Charm per strike con split Call/Put)
# =============================================================================

def compute_exposure_profile(option_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Tabella per-strike con GEX, Delta, Charm, OI (Call / Put / Net).

    Per ogni strike calcola:
      - gex_call, gex_put, gex_net   (gamma × 100 × OI × spot × ±1)
      - dex_call, dex_put, dex_net   (delta  × 100 × OI × ±1)
      - charm_call, charm_put, charm_net
      - oi_call, oi_put, oi_total

    Returns DataFrame sorted per strike, oppure DataFrame vuoto.
    """
    if option_df is None or option_df.empty:
        return pd.DataFrame()
    df = option_df.copy()
    needed = {"strike", "option_type", "gamma", "delta"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    if "open_interest" in df.columns:
        oi_col = "open_interest"
    elif "openInterest" in df.columns:
        oi_col = "openInterest"
    else:
        return pd.DataFrame()

    df["otype"] = df["option_type"].str.lower().str[0]
    w = df[oi_col].fillna(0)
    has_charm = "charm" in df.columns
    extra_greeks = [g for g in ["vanna", "speed", "vomma", "color"] if g in df.columns]

    df["gex"] = df["gamma"].fillna(0) * 100.0 * w * spot * np.where(df["otype"] == "c", 1, -1)
    df["dex"] = df["delta"].fillna(0) * 100.0 * w * np.where(df["otype"] == "c", 1, -1)
    if has_charm:
        df["charm_contrib"] = df["charm"].fillna(0) * 100.0 * w * np.where(df["otype"] == "c", 1, -1)
    df["oi_contrib"] = w
    for g in extra_greeks:
        df[f"{g}_contrib"] = df[g].fillna(0) * 100.0 * w * np.where(df["otype"] == "c", 1, -1)

    agg_dict = dict(gex=("gex", "sum"), dex=("dex", "sum"), oi=("oi_contrib", "sum"))
    if has_charm:
        agg_dict["charm"] = ("charm_contrib", "sum")
    for g in extra_greeks:
        agg_dict[g] = (f"{g}_contrib", "sum")

    calls = df[df["otype"] == "c"].groupby("strike").agg(**agg_dict)
    puts = df[df["otype"] == "p"].groupby("strike").agg(**agg_dict)

    result = pd.DataFrame(index=sorted(set(calls.index) | set(puts.index)))
    result.index.name = "strike"

    for side, grp in [("call", calls), ("put", puts)]:
        for col in ["gex", "dex", "oi"]:
            result[f"{col}_{side}"] = grp[col].reindex(result.index, fill_value=0)
        if has_charm:
            result[f"charm_{side}"] = grp["charm"].reindex(result.index, fill_value=0)
        else:
            result[f"charm_{side}"] = 0.0
        for g in extra_greeks:
            result[f"{g}_{side}"] = grp[g].reindex(result.index, fill_value=0)

    result["gex_net"] = result["gex_call"] + result["gex_put"]
    result["dex_net"] = result["dex_call"] + result["dex_put"]
    result["charm_net"] = result["charm_call"] + result["charm_put"]
    result["oi_total"] = result["oi_call"] + result["oi_put"]
    for g in extra_greeks:
        result[f"{g}_net"] = result[f"{g}_call"] + result[f"{g}_put"]
    result = result.reset_index()

    cols = ["strike", "gex_call", "gex_put", "gex_net", "dex_call", "dex_put", "dex_net",
            "charm_call", "charm_put", "charm_net", "oi_call", "oi_put", "oi_total"]
    for g in extra_greeks:
        cols += [f"{g}_call", f"{g}_put", f"{g}_net"]
    return result[cols]


# =============================================================================
# MONTE CARLO PATH SIMULATION DA PDF
# =============================================================================

def monte_carlo_paths(K, pdf, n_paths=5000, n_steps=50, seed=42):
    """Genera percorsi Monte Carlo dalla distribuzione di probabilita'.
    
    Steps:
    1. Calcola CDF dalla PDF
    2. Genera numeri uniformi U[0,1]
    3. Inversione CDF → prezzi finali
    4. Interpola percorsi come random walk verso il prezzo finale
    
    Returns:
        paths: array (n_paths, n_steps) di prezzi
        final_prices: array di prezzi finali
    """
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(pdf) * (K[1] - K[0])
    cdf = np.clip(cdf, 0, 1)
    
    # Estrai prezzi finali dalla CDF
    U = rng.uniform(0.001, 0.999, n_paths)
    final_prices = np.interp(U, cdf, K)
    
    # Genera percorsi: interpolazione cubica da spot a prezzo finale
    # Usiamo un modello semplice: ponte browniano geometrico
    paths = np.zeros((n_paths, n_steps))
    spot = K[np.argmin(np.abs(cdf - 0.5))]  # mediana come proxy spot
    
    for i in range(n_paths):
        # Random walk con drift verso il prezzo finale
        t = np.linspace(0, 1, n_steps)
        # Ponte browniano: X(t) = spot + (final - spot) * t + rumore browniano
        sigma = np.std(final_prices) * 0.3
        brownian = rng.normal(0, sigma / np.sqrt(n_steps), n_steps).cumsum()
        brownian = brownian - t * brownian[-1]  # bridge: parte e arriva a 0
        paths[i] = spot + (final_prices[i] - spot) * t + brownian
    
    return paths, final_prices


def monte_carlo_confidence(paths, percentiles=[5, 25, 50, 75, 95]):
    """Calcola percentili dei percorsi Monte Carlo."""
    return {p: np.percentile(paths, p, axis=0) for p in percentiles}


# =============================================================================
# MACRO DATA FETCHER (VIX, DXY, Copper)
# =============================================================================

def fetch_macro():
    """Scarica VIX, DXY, Copper via yfinance.
    Returns dict con prezzi correnti o None se non disponibile."""
    try:
        import yfinance as yf
    except ImportError:
        return {"vix": None, "dxy": None, "copper": None}
    
    result = {}
    tickers = {
        "vix": "^VIX",
        "vix1d": "^VIX1D",
        "vix9d": "^VIX9D",
        "vvix": "^VVIX",
        "vxn": "^VXN",
        "dxy": "DX-Y.NYB",
        "copper": "HG=F",
    }
    
    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                result[name] = float(hist["Close"].iloc[-1])
            else:
                result[name] = None
        except Exception as e:
            log.debug(f"Macro {name}: {e}")
            result[name] = None
    
    return result


def vxn_from_chain(option_df: pd.DataFrame) -> Optional[float]:
    """Calcola proxy VXN dalla IV ATM delle opzioni QQQ (Tradier real-time).

    Prende la IV media pesata per volume delle opzioni ATM ±5% dallo strike
    più vicino allo spot. Se available, usa la scadenza più vicina a 30 DTE.
    """
    if option_df is None or option_df.empty:
        return None
    df = option_df.copy()
    if "implied_volatility" not in df.columns or "strike" not in df.columns:
        return None
    spot = df["strike"].median()
    atm_mask = df["strike"].between(spot * 0.95, spot * 1.05)
    atm = df[atm_mask].copy()
    if atm.empty:
        return None
    ivv = pd.to_numeric(atm["implied_volatility"], errors="coerce")
    iv_mask = (ivv >= 0.05) & (ivv <= 2.0)
    ivs = ivv[iv_mask]
    if ivs.empty:
        return None
    if "volume" in atm.columns and atm["volume"].fillna(0).sum() > 0:
        w = pd.to_numeric(atm["volume"], errors="coerce").fillna(0)[iv_mask]
        if w.sum() > 0:
            return float(np.average(ivs, weights=w))
    return float(ivs.median())


def regime_from_vix(vix: float) -> str:
    """Determina il regime di mercato dal VIX."""
    if vix is None:
        return "UNKNOWN"
    if vix < 14:
        return "LOW_VOL (Risk-On)"
    elif vix < 20:
        return "NORMAL"
    elif vix < 30:
        return "HIGH_VOL (Risk-Off)"
    else:
        return "CRISIS"


# =============================================================================
# PCA / RIDGE REGRESSION (per riduzione rumore)
# =============================================================================

def pca_decomposition(data: np.ndarray, n_components: int = 3):
    """PCA per isolare i driver principali del mercato."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    if len(data) < n_components + 1:
        return None
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)
    
    return {
        "components": components,
        "explained_variance": pca.explained_variance_ratio_,
        "loadings": pca.components_,
        "cumulative_var": pca.explained_variance_ratio_.cumsum(),
    }


def ridge_predict(X_train, y_train, X_test, alpha=1.0):
    """Ridge regression per predizioni out-of-sample."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    Xt = scaler.transform(X_test) if len(X_test) > 0 else X_test
    
    model = Ridge(alpha=alpha)
    model.fit(Xs, y_train)
    
    preds = model.predict(Xt) if len(Xt) > 0 else model.predict(Xs)
    return preds, model.coef_, model.intercept_


# =============================================================================
# QUICK INTRA DAY METRICS
# =============================================================================

def expected_move(S0: float, iv: float, dte: float) -> dict:
    """Calcola Expected Move meccanico: S × IV × sqrt(DTE/365)."""
    iv = max(iv, 0.08)
    T = max(dte, 1) / 365.0
    move = S0 * iv * np.sqrt(T)
    return {
        "expected_move": round(move, 2),
        "expected_move_pct": round(move / S0 * 100, 2),
        "upper": round(S0 + move, 2),
        "lower": round(S0 - move, 2),
        "range_pct": round(move * 2 / S0 * 100, 2),
    }


# =============================================================================
# GARCH MONTE CARLO (EWMA volatility + paths)
# =============================================================================

def garch_mc_paths(prices: np.ndarray, n_paths=3000, n_steps=50, lambda_decay=0.94, seed=42):
    """GARCH-style MC usando EWMA volatility (no dipendenza da arch package).
    
    Args:
        prices: array dei prezzi storici
        n_paths: numero di percorsi
        n_steps: step per path
        lambda_decay: decay factor EWMA (0.94 = daily, 0.97 = weekly)
    
    Returns:
        paths: (n_paths, n_steps) simulated prices
        final_prices: (n_paths,) final prices
        vol_term: dict con vol iniziale, finale, media
    """
    rng = np.random.default_rng(seed)
    log_returns = np.diff(np.log(prices))
    
    # EWMA variance
    var = np.var(log_returns[-60:])  # seed con varianza 60gg
    ewma_vars = [var]
    for r in log_returns[-252:]:
        var = lambda_decay * var + (1 - lambda_decay) * r**2
        ewma_vars.append(var)
    ewma_vars = np.array(ewma_vars)
    vol_0 = np.sqrt(ewma_vars[-1])
    vol_mean = np.sqrt(np.mean(ewma_vars))
    vol_terminal = np.sqrt(ewma_vars[-1] * (1 - lambda_decay**n_steps) / (1 - lambda_decay))
    
    drift = np.mean(log_returns[-252:]) if len(log_returns) >= 252 else np.mean(log_returns)
    S0 = prices[-1]
    
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = S0
    
    for i in range(n_paths):
        vol_path = vol_0 * np.sqrt(np.cumsum(lambda_decay**np.arange(n_steps)) / 
                                   np.sum(lambda_decay**np.arange(n_steps)))
        vol_path = np.clip(vol_path, vol_0 * 0.3, vol_0 * 3.0)
        noise = rng.normal(0, 1, n_steps)
        rets = drift - 0.5 * vol_path**2 + vol_path * noise
        paths[i] = S0 * np.exp(np.cumsum(rets))
    
    return paths, paths[:, -1], {
        "vol_initial": float(vol_0),
        "vol_mean": float(vol_mean),
        "vol_terminal": float(vol_terminal),
    }


# =============================================================================
# HIGHER-ORDER GREEKS (Vanna, Charm, Speed, Vomma, Color)
# =============================================================================

def _norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def _norm_cdf(x):
    try:
        from scipy.stats import norm as _scipy_norm
        return _scipy_norm.cdf(x)
    except ImportError:
        x_c = np.clip(x, -10, 10)
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x_c + 0.044715 * x_c**3)))


def compute_all_greeks(S, K, T, r, sigma, typ="call"):
    """Calcola tutti i Greeks fino al 3° ordine.
    Returns dict con: delta, gamma, vega, theta, rho,
                      vanna, charm, speed, vomma, color."""
    if T <= 0 or sigma <= 0:
        return {g: 0.0 for g in ["delta","gamma","vega","theta","rho",
                                  "vanna","charm","speed","vomma","color"]}
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    phi = _norm_pdf(d1)
    N = _norm_cdf if typ == "call" else lambda x: 1 - _norm_cdf(x)
    n_cdf = _norm_cdf
    
    # 1st order
    if typ == "call":
        delta = n_cdf(d1)
        theta = (-S * phi * sigma / (2 * sqrt_T) - r * K * np.exp(-r * T) * n_cdf(d2)) / 365
    else:
        delta = -n_cdf(-d1)
        theta = (-S * phi * sigma / (2 * sqrt_T) + r * K * np.exp(-r * T) * n_cdf(-d2)) / 365
    
    gamma = phi / (S * sigma * sqrt_T)
    vega = S * phi * sqrt_T / 100
    rho = K * T * np.exp(-r * T) * (n_cdf(d2) if typ == "call" else -n_cdf(-d2)) / 100
    
    # 2nd order
    d1_sq = d1**2
    vanna = -vega * d2 / (sigma * S) if sigma > 0 else 0  # dDelta/dVol
    charm = -phi * (2 * (r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T) + 
                    (r - d2 / sqrt_T) / (sigma * sqrt_T)) / 365  # dDelta/dTime
    
    # 3rd order
    speed = -gamma / S * (d1 / (sigma * sqrt_T) + 1)  # dGamma/dS
    vomma = vega * d1 * d2 / sigma  # dVega/dVol = d2V/dVol2
    color = -phi / (2 * S * sigma * sqrt_T * T) * (
        2 * (r * T + 1) / (sigma * sqrt_T) + (2 * r * T - d1 * sigma * sqrt_T) / (sigma * sqrt_T)
    ) / 365  # dGamma/dTime
    
    return {
        "delta": float(delta), "gamma": float(gamma), "vega": float(vega),
        "theta": float(theta), "rho": float(rho),
        "vanna": float(vanna), "charm": float(charm),
        "speed": float(speed), "vomma": float(vomma), "color": float(color),
        "d1": float(d1), "d2": float(d2),
    }


# =============================================================================
# GAMMA WALLS (Call Wall, Put Wall)
# =============================================================================

def gamma_walls(option_df: pd.DataFrame, spot: float) -> dict:
    """Identifica i Gamma Walls (concentrazioni di gamma).
    
    Returns:
        put_wall: strike con max gamma put
        call_wall: strike con max gamma call
        wall_strength: forza relativa delle walls
    """
    required = {"gamma", "option_type", "strike"}
    if option_df is None or option_df.empty or not required.issubset(option_df.columns):
        return {"put_wall": spot, "call_wall": spot, "put_strength": 0, "call_strength": 0}
    
    df = option_df.copy()
    df["otype"] = df["option_type"].str.upper().str[0]
    
    def _near(col):
        tot = df[col].fillna(0).sum()
        if tot <= 0: return 0
        near = df[df["strike"].between(spot*0.92, spot*1.08)][col].fillna(0).sum()
        return near / tot
    oi_ok = "open_interest" in df.columns and df["open_interest"].fillna(0).sum() > 0
    vol_ok = "volume" in df.columns and df["volume"].fillna(0).sum() > 0
    oi_n = _near("open_interest") if oi_ok else 0
    vol_n = _near("volume") if vol_ok else 0
    if oi_ok and oi_n >= 0.3:
        w = df["open_interest"].fillna(0)
    elif vol_ok and vol_n >= 0.3:
        w = df["volume"].fillna(0)
    elif oi_ok:
        w = df["open_interest"].fillna(0)
    elif vol_ok:
        w = df["volume"].fillna(0)
    else:
        w = pd.Series(1.0, index=df.index)
    
    df["gamma_oi"] = df["gamma"].fillna(0) * w * spot * 0.01
    
    puts = df[df["otype"] == "P"].groupby("strike")["gamma_oi"].sum()
    calls = df[df["otype"] == "C"].groupby("strike")["gamma_oi"].sum()
    
    put_strength = float(puts.max()) if not puts.empty else 0
    call_strength = float(calls.max()) if not calls.empty else 0
    put_wall = float(puts.idxmax()) if not puts.empty and put_strength > 0 else spot
    call_wall = float(calls.idxmax()) if not calls.empty and call_strength > 0 else spot
    
    return {
        "put_wall": put_wall,
        "call_wall": call_wall,
        "put_strength": put_strength,
        "call_strength": call_strength,
        "gamma_profile": (puts.index.values, puts.values, 
                          calls.index.values, calls.values),
    }


# =============================================================================
# PREMIUM FLOW (Put vs Call premium per strike)
# =============================================================================

def premium_flow(option_df: pd.DataFrame, spot: float) -> dict:
    """Analisi del flusso premi Put vs Call.
    
    Returns:
        per-strike premium comparison
        total put/call premium
        put_call_ratio
    """
    if option_df is None or option_df.empty:
        return {"ratio": 1.0, "total_put": 0, "total_call": 0, "by_strike": []}
    
    df = option_df.copy()
    if "option_type" in df.columns:
        df["otype"] = df["option_type"].str.upper().str[0]
    elif "type" in df.columns:
        df["otype"] = df["type"].str.upper().str[0]
    else:
        return {"ratio": 1.0, "total_put": 0, "total_call": 0, "by_strike": []}
    
    price_col = "lastPrice" if "lastPrice" in df.columns else ("last_price" if "last_price" in df.columns else "last")
    if price_col not in df.columns:
        return {"ratio": 1.0, "total_put": 0, "total_call": 0, "by_strike": []}
    
    df["premium"] = df[price_col].fillna(0)
    def _near(col):
        tot = df[col].fillna(0).sum()
        if tot <= 0: return 0
        near = df[df["strike"].between(spot*0.92, spot*1.08)][col].fillna(0).sum()
        return near / tot
    oi_ok = "open_interest" in df.columns and df["open_interest"].fillna(0).sum() > 0
    vol_ok = "volume" in df.columns and df["volume"].fillna(0).sum() > 0
    oi_n = _near("open_interest") if oi_ok else 0
    vol_n = _near("volume") if vol_ok else 0
    if oi_ok and oi_n >= 0.3:
        df["premium"] *= df["open_interest"].fillna(0).clip(0, 1e6)
    elif vol_ok and vol_n >= 0.3:
        df["premium"] *= df["volume"].fillna(0).clip(0, 1e6)
    elif oi_ok:
        df["premium"] *= df["open_interest"].fillna(0).clip(0, 1e6)
    elif vol_ok:
        df["premium"] *= df["volume"].fillna(0).clip(0, 1e6)
    
    puts = df[df["otype"] == "P"].groupby("strike")["premium"].sum()
    calls = df[df["otype"] == "C"].groupby("strike")["premium"].sum()
    all_strikes = sorted(set(puts.index) | set(calls.index))
    
    by_strike = []
    for k in all_strikes:
        by_strike.append({
            "strike": float(k),
            "put_premium": float(puts.get(k, 0)),
            "call_premium": float(calls.get(k, 0)),
            "net": float(puts.get(k, 0) - calls.get(k, 0)),
        })
    
    total_put = float(puts.sum()) if not puts.empty else 0
    total_call = float(calls.sum()) if not calls.empty else 0
    ratio = total_put / max(total_call, 1)
    
    return {
        "ratio": ratio,
        "total_put": total_put,
        "total_call": total_call,
        "by_strike": by_strike,
    }


# =============================================================================
# LINEAR REGRESSION CHANNELS (SPY/QQQ)
# =============================================================================

def linreg_channels(prices: np.ndarray, window=89) -> dict:
    """Canali di regressione lineare con bande a deviazioni standard.
    
    Returns:
        dict con slope, intercept, channel lines per +/-1,2,3 sigma
    """
    if len(prices) < window:
        window = len(prices)
    y = prices[-window:]
    x = np.arange(window)
    
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residuals = y - fitted
    std = np.std(residuals)
    
    channels = {}
    for n, color in [(1, "green"), (2, "gold"), (3, "red")]:
        channels[f"+{n}sigma"] = fitted + n * std
        channels[f"-{n}sigma"] = fitted - n * std
    
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "std": float(std),
        "fitted": fitted,
        "x": x,
        "channels": channels,
        "r_squared": float(np.corrcoef(x, y)[0, 1]**2),
    }


# =============================================================================
# INTRADAY ANOMALY DETECTION
# =============================================================================

def intraday_anomalies(prices: pd.Series, volumes: pd.Series = None, 
                       window=20, z_threshold=2.0) -> dict:
    """Rileva anomalie intraday: volumi e prezzi fuori norma.
    
    Returns:
        anomalies: list di dict con tipo, magnitudo, descrizione
        z_scores: dict con z-score attuali
    """
    anomalies = []
    z_scores = {}
    
    if prices is not None and len(prices) > window:
        price_ret = prices.pct_change().dropna()
        recent = price_ret.tail(window)
        z_price = (price_ret.iloc[-1] - recent.mean()) / max(recent.std(), 1e-10)
        z_scores["price_z"] = float(z_price)
        if abs(z_price) > z_threshold:
            anomalies.append({
                "type": "PRICE",
                "magnitude": float(abs(z_price)),
                "description": f"Prezzo devia {z_price:.1f}σ dalla media {window}gg",
            })
    

def volume_profile(prices: np.ndarray = None, option_df: pd.DataFrame = None,
                   spot: float = None, n_bins: int = 20) -> dict:
    """Volume Profile da option chain (vol × price) o da prezzi storici.
    
    Con option_df: raggruppa volume per strike (price del sottostante).
    Con solo prices: simula distribuzione da differenze prezzo giornaliere.
    
    Returns:
        poc, vah, val, value_area_pct, distribution (per strike)
    """
    if option_df is not None and not option_df.empty and "volume" in option_df.columns:
        v = option_df["volume"].fillna(0)
        if v.sum() > 0:
            # Volume profile reale dalla option chain
            by_price = option_df.groupby("strike")["volume"].sum()
            prices_a = by_price.index.values
            vols = by_price.values
        else:
            prices_a, vols = None, None
    else:
        prices_a, vols = None, None
    
    if prices_a is None and prices is not None and len(prices) > 5:
        vol_sim = np.abs(np.diff(prices[-n_bins:], prepend=prices[-n_bins]))
        vol_sim = np.clip(vol_sim, 1, None)
        bins = np.linspace(prices.min(), prices.max(), n_bins)
        idx = np.clip(np.digitize(prices[-len(vol_sim):], bins) - 1, 0, n_bins - 2)
        dist = np.zeros(n_bins - 1)
        bin_c = (bins[:-1] + bins[1:]) / 2
        for i, v in zip(idx, vol_sim):
            dist[i] += v
    elif prices_a is not None:
        bins = np.linspace(prices_a.min(), prices_a.max(), min(n_bins, len(prices_a)))
        bin_c = (bins[:-1] + bins[1:]) / 2
        dist = np.zeros(len(bin_c))
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (prices_a >= lo) & (prices_a < hi)
            dist[i] = vols[mask].sum()
    else:
        return {"poc": spot if spot else 0, "vah": 0, "val": 0,
                "value_area_pct": 0, "distribution": []}
    
    total = dist.sum()
    if total <= 0:
        return {"poc": spot if spot else float(prices[-1] if prices is not None else 0),
                "vah": 0, "val": 0, "value_area_pct": 0, "distribution": []}
    
    poc_idx = np.argmax(dist)
    poc = float(bin_c[poc_idx])
    
    order = np.argsort(-dist)
    va_mask = np.zeros(len(dist), dtype=bool)
    cum = 0.0
    for o in order:
        va_mask[o] = True
        cum += dist[o]
        if cum / total >= 0.7:
            break
    
    va_idx = np.where(va_mask)[0]
    vah = float(bin_c[va_idx.max()])
    val = float(bin_c[va_idx.min()])
    
    distribution = [{"price": float(bin_c[i]), "volume": float(dist[i]),
                     "is_va": bool(va_mask[i])} for i in range(len(bin_c))]
    
    return {"poc": poc, "vah": vah, "val": val,
            "value_area_pct": float(cum / total),
            "distribution": distribution}


def intraday_anomalies(prices: pd.Series, volumes: pd.Series = None, 
                       window=20, z_threshold=2.0) -> dict:
    """Rileva anomalie intraday: volumi e prezzi fuori norma."""
    if prices is None or len(prices) < window + 2:
        return {"anomalies": [], "z_scores": {"price_z": 0, "volume_z": 0}}
    
    recent = prices.tail(window)
    z_price = (prices.iloc[-1] - recent.mean()) / max(recent.std(), 1)
    anomalies = []
    z_scores = {"price_z": float(z_price), "volume_z": 0}
    if abs(z_price) > z_threshold:
        anomalies.append({
            "type": "PRICE",
            "magnitude": float(abs(z_price)),
            "description": f"Prezzo devia {z_price:.1f}σ dalla media {window}gg",
        })
    
    if volumes is not None and len(volumes) > window:
        vol_recent = volumes.tail(window)
        z_vol = (volumes.iloc[-1] - vol_recent.mean()) / max(vol_recent.std(), 1)
        z_scores["volume_z"] = float(z_vol)
        if abs(z_vol) > z_threshold:
            anomalies.append({
                "type": "VOLUME",
                "magnitude": float(abs(z_vol)),
                "description": f"Volume devia {z_vol:.1f}σ dalla media {window}gg",
            })
    
    return {"anomalies": anomalies, "z_scores": z_scores}


# =============================================================================
# DOM BOOKMAP (simulato da option chain)
# =============================================================================

def dom_bookmap(option_df: pd.DataFrame, spot: float, n_levels=30) -> dict:
    """Calcola DOM Bookmap: valore nozionale (premio × volume × 100) per strike.
    
    Usa mid=(bid+ask)/2 quando last_price è 0 (opzioni non trade oggi).
    """
    if option_df is None or option_df.empty:
        return {"levels": np.array([]), "liquidity": np.array([]), "big_trades": []}
    
    df = option_df.copy()
    has_vol = "volume" in df.columns and df["volume"].fillna(0).sum() > 0
    
    if has_vol:
        has_last = "last_price" in df.columns
        has_bidask = "bid" in df.columns and "ask" in df.columns
        if has_last and has_bidask:
            df["premium"] = df.apply(
                lambda r: r["last_price"] if r["last_price"] > 0 else (r["bid"] + r["ask"]) / 2,
                axis=1
            )
        elif has_last:
            df["premium"] = df["last_price"]
        elif has_bidask:
            df["premium"] = (df["bid"] + df["ask"]) / 2
        else:
            df["premium"] = 0.0
        df["premium"] = df["premium"].fillna(0).clip(lower=0)
        df["notional"] = df["premium"] * df["volume"].fillna(0) * 100
        by_strike = df.groupby("strike")["notional"].sum().sort_index()
    elif "open_interest" in df.columns and df["open_interest"].fillna(0).sum() > 0:
        by_strike = df.groupby("strike")["open_interest"].sum().sort_index()
    else:
        by_strike = df.groupby("strike").size().sort_index()
    
    strikes = by_strike.index.values
    liquidity = by_strike.values.astype(float)
    
    # Big trades: cluster > 2 std sopra la media
    mu, sg = liquidity.mean(), liquidity.std()
    big_trades = []
    for i, (k, l) in enumerate(zip(strikes, liquidity)):
        if l > mu + 2 * sg:
            big_trades.append({
                "strike": float(k),
                "size": float(l),
                "distance_pct": float((k / spot - 1) * 100),
            })
    
    return {
        "levels": strikes,
        "liquidity": liquidity,
        "big_trades": sorted(big_trades, key=lambda x: -x["size"]),
    }


# =============================================================================
# DELTA EXPOSURE (DEX)
# =============================================================================

def compute_dex(option_df: pd.DataFrame, spot: float) -> dict:
    """Calcola Delta Exposure per strike.
    
    DEX = delta × OI × 100
    
    Returns:
        strikes, dex (call/put/net), total_dex, wall levels
    """
    if option_df is None or option_df.empty:
        return {"strikes": np.array([]), "dex": np.array([]),
                "call_dex": np.array([]), "put_dex": np.array([]),
                "total_dex": 0, "call_wall": spot, "put_wall": spot}
    
    has_delta = "delta" in option_df.columns
    has_type = "option_type" in option_df.columns
    
    if not has_delta or not has_type:
        return {"strikes": np.array([]), "dex": np.array([]),
                "call_dex": np.array([]), "put_dex": np.array([]),
                "total_dex": 0, "call_wall": spot, "put_wall": spot}
    
    df = option_df.copy()
    df["otype"] = df["option_type"].str.upper().str[0]
    
    def _near(col):
        tot = df[col].fillna(0).sum()
        if tot <= 0: return 0
        near = df[df["strike"].between(spot*0.92, spot*1.08)][col].fillna(0).sum()
        return near / tot
    oi_ok = "open_interest" in df.columns and df["open_interest"].fillna(0).sum() > 0
    vol_ok = "volume" in df.columns and df["volume"].fillna(0).sum() > 0
    oi_n = _near("open_interest") if oi_ok else 0
    vol_n = _near("volume") if vol_ok else 0
    if oi_ok and oi_n >= 0.3:
        w = df["open_interest"].fillna(0)
    elif vol_ok and vol_n >= 0.3:
        w = df["volume"].fillna(0)
    elif oi_ok:
        w = df["open_interest"].fillna(0)
    elif vol_ok:
        w = df["volume"].fillna(0)
    else:
        w = pd.Series(1.0, index=df.index)
    
    df["dex"] = df["delta"].fillna(0) * w * 100
    
    calls = df[df["otype"] == "C"].groupby("strike")["dex"].sum()
    puts = df[df["otype"] == "P"].groupby("strike")["dex"].sum()
    all_strikes = sorted(set(calls.index) | set(puts.index))
    
    strikes = np.array(all_strikes)
    call_a = np.array([calls.get(k, 0) for k in all_strikes])
    put_a = np.array([puts.get(k, 0) for k in all_strikes])
    net = call_a - put_a
    
    call_wall = calls.idxmax() if not calls.empty else spot
    put_wall = puts.idxmax() if not puts.empty else spot
    
    return {
        "strikes": strikes,
        "dex": net,
        "call_dex": call_a,
        "put_dex": put_a,
        "total_dex": float(net.sum()),
        "call_wall": float(call_wall),
        "put_wall": float(put_wall),
    }


# =============================================================================
# OPEN INTEREST PROFILE
# =============================================================================

def open_interest_profile(option_df: pd.DataFrame, spot: float) -> dict:
    """Profilo Open Interest per strike (Put/Call/Net)."""
    if option_df is None or option_df.empty:
        return {"strikes": np.array([]), "oi": np.array([]), "call_oi": np.array([]), "put_oi": np.array([])}
    
    df = option_df.copy()
    if "option_type" in df.columns:
        df["otype"] = df["option_type"].str.upper().str[0]
    elif "type" in df.columns:
        df["otype"] = df["type"].str.upper().str[0]
    else:
        return {"strikes": np.array([]), "oi": np.array([]), "call_oi": np.array([]), "put_oi": np.array([])}
    
    def _near(col):
        tot = df[col].fillna(0).sum()
        if tot <= 0: return 0
        near = df[df["strike"].between(spot*0.92, spot*1.08)][col].fillna(0).sum()
        return near / tot
    oi_col = "open_interest" if "open_interest" in df.columns else "openInterest"
    oi_ok = oi_col in df.columns and df[oi_col].fillna(0).sum() > 0
    vol_ok = "volume" in df.columns and df["volume"].fillna(0).sum() > 0
    oi_n = _near(oi_col) if oi_ok else 0
    vol_n = _near("volume") if vol_ok else 0
    if oi_ok and oi_n >= 0.3:
        pass
    elif vol_ok and vol_n >= 0.3:
        oi_col = "volume"
    elif oi_ok:
        pass
    elif vol_ok:
        oi_col = "volume"
    else:
        return {"strikes": np.array([]), "oi": np.array([]), "call_oi": np.array([]), "put_oi": np.array([])}
    
    calls = df[df["otype"] == "C"].groupby("strike")[oi_col].sum()
    puts = df[df["otype"] == "P"].groupby("strike")[oi_col].sum()
    all_strikes = sorted(set(calls.index) | set(puts.index))
    
    strikes = np.array(all_strikes)
    call_o = np.array([calls.get(k, 0) for k in all_strikes])
    put_o = np.array([puts.get(k, 0) for k in all_strikes])
    
    return {
        "strikes": strikes,
        "oi": call_o + put_o,
        "call_oi": call_o,
        "put_oi": put_o,
        "max_oi_strike": float(strikes[np.argmax(call_o + put_o)]),
    }


# =============================================================================
# GEX CENTROID MAP (evoluzione temporale centroidi)
# =============================================================================

def gex_centroid_map(option_df: pd.DataFrame, spot: float, by_dte: bool = True) -> dict:
    """Evoluzione della zero-gamma level per DTE.
    
    Per ogni DTE, calcola il profilo GEX completo e restituisce
    lo zero_gamma (punto di pareggio net gamma) come centroide.
    
    Returns:
        centroids: dict {dte: zero_gamma_strike}
        net_gex: dict {dte: net_gex}
    """
    if option_df is None or option_df.empty or "dte" not in option_df.columns:
        return {"centroids": {}, "net_gex": {}}
    
    df = option_df.copy()
    if "option_type" in df.columns:
        df["otype"] = df["option_type"].str.upper().str[0]
    elif "type" in df.columns:
        df["otype"] = df["type"].str.upper().str[0]
    else:
        return {"centroids": {}, "net_gex": {}}
    
    centroids = {}
    net_gex = {}
    for key, grp in df.groupby("dte"):
        g = compute_gex(grp, spot)
        has_weight = any(c in grp.columns and grp[c].fillna(0).sum() > 0
                         for c in ("open_interest", "openInterest", "volume"))
        if not has_weight:
            continue
        centroids[float(key)] = float(g.get("zero_gamma", spot))
        net_gex[float(key)] = g.get("total_gex", 0)
    
    return {"centroids": centroids, "net_gex": net_gex}


# =============================================================================
# GREEKS ENRICHMENT (da yfinance → gamma/delta sintetici via BS)
# =============================================================================

def enrich_greeks(option_df: pd.DataFrame, spot: float, rate: float = 0.05) -> pd.DataFrame:
    """Aggiunge colonne gamma, delta, option_type a option chain yfinance.
    
    Normalizza anche nomi colonne (openInterest→open_interest, etc.)
    e calcola Greeks sintetici via Black-Scholes da IV/price.
    """
    if option_df is None or option_df.empty:
        return option_df
    
    df = option_df.copy()
    
    col_map = {
        "openInterest": "open_interest",
        "impliedVolatility": "implied_volatility",
        "lastPrice": "last_price",
        "last": "last_price",
        "option_type": "option_type",
    }
    rename = {k: v for k, v in col_map.items() if k in df.columns and k != v}
    if rename:
        df = df.rename(columns=rename)
    
    type_col = "option_type" if "option_type" in df.columns else "type" if "type" in df.columns else None
    if type_col is None:
        return option_df
    
    df["option_type"] = df[type_col].str.lower().str[0]
    has_iv = "implied_volatility" in df.columns
    has_price = "last_price" in df.columns
    
    if not has_price:
        return df
    
    # Normalizza prezzi (non droppare righe — anche price=0 serve per OI/struttura)
    df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce").fillna(0)
    
    # IV di riferimento: ATM se disponibile, altrimenti 20%
    ref_iv = 0.2
    if has_iv:
        valid_ivs = df[df["implied_volatility"].notna() & (df["implied_volatility"] > 0.05)]["implied_volatility"]
        if not valid_ivs.empty:
            ref_iv = float(valid_ivs.median())
    
    df["gamma"] = 0.0
    df["delta"] = 0.0
    df["implied_volatility"] = df["implied_volatility"].fillna(ref_iv) if has_iv else ref_iv
    
    for idx, row in df.iterrows():
        K = float(row["strike"])
        price = float(row["last_price"])
        dte_raw = max(float(row.get("dte", 30)), 1)
        T = dte_raw / 365.0
        typ = "call" if row["option_type"] == "c" else "put"
        
        sigma = ref_iv
        if has_iv:
            iv_val = float(row["implied_volatility"])
            if 0.05 <= iv_val <= 2.0:
                sigma = iv_val
        if sigma == ref_iv and price > 0.05:
            iv_est = _newton_iv(spot, K, T, price, typ)
            if 0.05 <= iv_est <= 2.0:
                sigma = iv_est
        
        g = compute_all_greeks(spot, K, T, rate, sigma, typ)
        df.at[idx, "gamma"] = g["gamma"]
        df.at[idx, "delta"] = g["delta"]
        df.at[idx, "vega"] = g["vega"]
        df.at[idx, "theta"] = g["theta"]
        df.at[idx, "vanna"] = g["vanna"]
        df.at[idx, "charm"] = g["charm"]
        df.at[idx, "speed"] = g["speed"]
        df.at[idx, "vomma"] = g["vomma"]
        df.at[idx, "color"] = g["color"]
        df.at[idx, "implied_volatility"] = sigma
    
    return df


def _newton_iv(S, K, T, price, typ):
    """Newton-Raphson IV solver."""
    from scipy.stats import norm
    if price <= 0 or T <= 0:
        return 0.2
    intrinsic = max(0, S - K) if typ == "call" else max(0, K - S)
    if price <= intrinsic + 0.05:
        return 0.2
    def bs(sigma):
        d1 = (np.log(S / K) + (0.05 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if typ == "call":
            return S * norm.cdf(d1) - K * np.exp(-0.05 * T) * norm.cdf(d2)
        return K * np.exp(-0.05 * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    iv = 0.2
    for _ in range(30):
        p_est = bs(iv)
        diff = p_est - price
        if abs(diff) < 1e-6:
            break
        d1 = (np.log(S / K) + (0.05 + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1) if iv > 1e-6 else 1.0
        if abs(vega) < 1e-12:
            break
        iv -= diff / vega
        iv = np.clip(iv, 0.005, 5.0)
    return float(np.clip(iv, 0.05, 2.0))


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test macro
    print("--- MACRO ---")
    macro = fetch_macro()
    print(f"VIX: {macro.get('vix')} | DXY: {macro.get('dxy')} | Copper: {macro.get('copper')}")
    print(f"Regime: {regime_from_vix(macro.get('vix'))}")
    
    # Test expected move
    print("\n--- EXPECTED MOVE (SPX=5500, IV=0.15, DTE=30) ---")
    em = expected_move(5500, 0.15, 30)
    print(f"Move: {em['expected_move']} ({em['expected_move_pct']}%)")
    print(f"Range: {em['lower']} - {em['upper']}")
    
    # Test higher Greeks
    print("\n--- GREEKS (SPX=5500, K=5500, T=0.12, r=0.05, sigma=0.15) ---")
    g = compute_all_greeks(5500, 5500, 0.12, 0.05, 0.15, "call")
    print(f"Delta={g['delta']:.4f} Gamma={g['gamma']:.6f} Vega={g['vega']:.4f}")
    print(f"Vanna={g['vanna']:.6f} Vomma={g['vomma']:.6f} Charm={g['charm']:.6f}")
    

def gamma_bands(option_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Calcola gamma bands per ticker da option chain.
    
    Restituisce DataFrame con colonne:
        direzione, zona, pct_low, pct_high, livello_lo, livello_hi, centro, nota
    """
    ZONES = [
        ("Zona di rottura ▲", 1.50, 2.00),
        ("Resistenza primaria ▲", 0.75, 1.50),
        ("Resistenza 2° ▲", 0.35, 0.75),
        ("Attrito ▲", 0.15, 0.35),
        ("── SPOT (Max Pain zone) ──", -0.15, 0.15),
        ("Attrito ▼", -0.35, -0.15),
        ("Supporto 2° ▼", -0.75, -0.35),
        ("Supporto primario ▼", -1.50, -0.75),
        ("Zona di rottura ▼", -2.00, -1.50),
    ]
    
    gex = compute_gex(option_df, spot)
    walls = gamma_walls(option_df, spot)
    oi = open_interest_profile(option_df, spot)
    
    zg = gex.get("zero_gamma", spot)
    cw = walls.get("call_strength", 0)
    pw = walls.get("put_strength", 0)
    cw_k = walls.get("call_wall", spot)
    pw_k = walls.get("put_wall", spot)
    max_oi = oi.get("max_oi_strike", spot)
    total_gex = gex.get("total_gex", 0)
    
    rows = []
    for name, pct_lo, pct_hi in ZONES:
        lo = spot * (1 + pct_lo / 100)
        hi = spot * (1 + pct_hi / 100)
        centro = (lo + hi) / 2
        
        if "rottura" in name.lower():
            is_up = "▲" in name
            if is_up:
                direzione = "▲"
                nota = "GEX short gamma" if total_gex > -50000 else "GEX short gamma estremo" if total_gex < 0 else "Resistenza di fine range"
            else:
                direzione = "▼"
                nota = "GEX short gamma" if total_gex > -50000 else "GEX short gamma estremo" if total_gex < 0 else "Supporto di fine range"
        elif "primaria" in name.lower():
            if "▲" in name:
                direzione = "▲"
                nota = "Call wall — forte pinning" if cw > 0 and abs(cw_k - centro) / centro < 0.01 else "Resistenza chiave"
            else:
                direzione = "▼"
                nota = "Put wall — forte pinning" if pw > 0 and abs(pw_k - centro) / centro < 0.01 else "Supporto chiave"
        elif "2°" in name:
            if "▲" in name:
                direzione = "▲"
                nota = "Strike con alto OI" if max_oi and abs(max_oi - centro) / centro < 0.015 else "Resistenza secondaria"
            else:
                direzione = "▼"
                nota = "Strike con alto OI" if max_oi and abs(max_oi - centro) / centro < 0.015 else "Supporto secondario"
        elif "Attrito" in name:
            if "▲" in name:
                direzione = "▲"
                nota = "Range — bassa direzionalità" if abs(zg - centro) / centro < 0.02 else "Attrito rialzista"
            else:
                direzione = "▼"
                nota = "Range — bassa direzionalità" if abs(zg - centro) / centro < 0.02 else "Attrito ribassista"
        else:
            direzione = "●"
            nota = "Zona magnetica ATM — attrazione massima a fine sessione"
        
        rows.append(dict(
            direzione=direzione, zona=name,
            pct_low=pct_lo, pct_high=pct_hi,
            livello_lo=round(lo, 2), livello_hi=round(hi, 2),
            centro=round(centro, 2), nota=nota,
        ))
    
    return pd.DataFrame(rows)


def gex_regime_heatmap(combined_df, spot, r=0.05, n_prices=40, n_days=16):
    """Proietta regime GEX (POS/NEG) su griglia prezzo × tempo futuro."""
    from scipy.stats import norm
    df = combined_df.copy()
    needed = ["strike", "dte", "implied_volatility", "open_interest", "option_type"]
    if not all(c in df.columns for c in needed):
        return None, None, None
    df = df.dropna(subset=needed)
    df = df[(df["dte"] > 0) & (df["implied_volatility"] > 0.01) & (df["open_interest"] > 0)]
    if len(df) == 0:
        return None, None, None
    K = df["strike"].values.astype(float)
    T = (df["dte"].values.astype(float) / 365).clip(1/365)
    sig = df["implied_volatility"].values.astype(float)
    OI = df["open_interest"].values.astype(float)
    flg = np.where(df["option_type"].str.lower().str[0] == "c", 1, -1)
    prices = np.linspace(spot * 0.95, spot * 1.05, n_prices)
    days = np.arange(0, 31, 30 // (n_days - 1))[:n_days]
    S_g = prices[:, None, None]
    T_g = np.maximum(T[None, None, :] - days[None, :, None] / 365, 1/365)
    d1 = (np.log(S_g / K) + (r + 0.5 * sig**2) * T_g) / (sig * np.sqrt(T_g))
    gamma_g = norm.pdf(d1) / (S_g * sig * np.sqrt(T_g))
    gex_matrix = (gamma_g * OI * S_g * flg).sum(axis=2)
    return prices, days, gex_matrix


def oi_term_heatmap(combined_df, spot, max_dte=45, bin_width=2.5):
    """Heatmap Open Interest per strike × DTE con strike binned.

    Raggruppa strike in bin di `bin_width` per ridurre rumore visivo.
    Returns dict con strikes, dtes, total_oi_matrix, call_pct_matrix, call_oi, put_oi
    oppure None se dati insufficienti.
    """
    if combined_df is None or combined_df.empty:
        return None
    df = combined_df.copy()
    oi_col = "open_interest" if "open_interest" in df.columns else "openInterest"
    if oi_col not in df.columns:
        return None
    if "option_type" in df.columns:
        df["otype"] = df["option_type"].str.lower().str[0]
    elif "type" in df.columns:
        df["otype"] = df["type"].str.lower().str[0]
    else:
        return None
    df = df[(df["dte"].between(0, max_dte)) & (df[oi_col].fillna(0) > 0)].copy()
    if len(df) == 0:
        return None

    # Bin strikes to nearest bin_width
    df["strike_bin"] = (df["strike"] / bin_width).round() * bin_width

    calls = df[df["otype"] == "c"].pivot_table(
        index="strike_bin", columns="dte", values=oi_col, aggfunc="sum", fill_value=0)
    puts = df[df["otype"] == "p"].pivot_table(
        index="strike_bin", columns="dte", values=oi_col, aggfunc="sum", fill_value=0)

    all_strikes = sorted(set(calls.index) | set(puts.index))
    all_dtes = sorted(set(calls.columns) | set(puts.columns))

    call_mat = calls.reindex(index=all_strikes, columns=all_dtes, fill_value=0).values.astype(float)
    put_mat = puts.reindex(index=all_strikes, columns=all_dtes, fill_value=0).values.astype(float)
    total = call_mat + put_mat
    call_pct = np.divide(call_mat, total, out=np.zeros_like(total, dtype=float), where=total > 0)

    return {
        "strikes": np.array(all_strikes),
        "dtes": np.array(all_dtes),
        "total_oi": total,
        "call_oi": call_mat,
        "put_oi": put_mat,
        "call_pct": call_pct,
    }


# =============================================================================
# EOD SNAPSHOT (end-of-day persistenza dati)
# =============================================================================

def save_eod_snapshot(conn, date_str, ticker, spot, gamma_env, zero_gamma,
                      total_gex, vvr, call_wall, put_wall, skew, em_up, em_down,
                      regime, g_score, i_score, shift_prob, vix):
    """Salva snapshot end-of-day nella tabella eod_snapshots.

    Chiamata a ogni refresh del terminale (INSERT OR REPLACE per data).
    L'ultimo snapshot del giorno viene poi letto dal morning briefing.
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eod_snapshots (
            date TEXT PRIMARY KEY,
            ticker TEXT,
            spot REAL,
            gamma_env TEXT,
            zero_gamma REAL,
            total_gex REAL,
            vvr REAL,
            call_wall REAL,
            put_wall REAL,
            skew REAL,
            em_up REAL,
            em_down REAL,
            regime TEXT,
            g_score REAL,
            i_score REAL,
            shift_prob REAL,
            vix REAL
        )
    """)
    cursor.execute("""
        INSERT OR REPLACE INTO eod_snapshots
        (date, ticker, spot, gamma_env, zero_gamma, total_gex, vvr,
         call_wall, put_wall, skew, em_up, em_down,
         regime, g_score, i_score, shift_prob, vix)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (date_str, ticker, spot, gamma_env, zero_gamma, total_gex, vvr,
          call_wall, put_wall, skew, em_up, em_down,
          regime, g_score, i_score, shift_prob, vix))
    conn.commit()


def get_latest_eod_snapshot(db_path, ticker="QQQ"):
    """Recupera l'ultimo snapshot disponibile per il ticker."""
    import sqlite3, os
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT * FROM eod_snapshots WHERE ticker=? ORDER BY date DESC LIMIT 1",
            conn, params=(ticker,))
        conn.close()
        if not df.empty:
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return None


# =============================================================================
# CPCV — COPULA PCA VOLATILITY
# =============================================================================

def cpcv_risk_assessment(combined_df, spot, n_components=3):
    """
    Copula-PCA Volatility Risk Assessment.

    PCA sulla superficie di volatilità implicita per estrarre
    i fattori di rischio sistematici. La copula gaussiana sui
    PC scores quantifica la dipendenza di coda tra i vari
    punti della superficie.

    Returns dict con regime_score [0,1], regime_label, component_ratios.
    """
    result = {"regime_score": 0.5, "regime_label": "NORMAL", "component_ratios": []}
    if combined_df is None or combined_df.empty or spot <= 0:
        return result

    df = combined_df.copy()
    needed = ["strike", "dte", "implied_volatility", "open_interest"]
    if not all(c in df.columns for c in needed):
        return result

    df = df[(df["implied_volatility"] > 0.01) & (df["open_interest"] > 0)].copy()
    if len(df) < 15:
        return result

    df["moneyness"] = (df["strike"] / spot).round(2)
    df = df[df["moneyness"].between(0.85, 1.15)]

    pivot = df.pivot_table(
        index="moneyness", columns="dte",
        values="implied_volatility", aggfunc="mean"
    )
    if pivot.shape[0] < 5 or pivot.shape[1] < 3:
        return result

    pivot = pivot.interpolate(axis=1).bfill(axis=1).ffill(axis=1)
    pivot = pivot.interpolate(axis=0).bfill(axis=0).ffill(axis=0)

    vals = pivot.values
    means = vals.mean(axis=1, keepdims=True)
    stds = vals.std(axis=1, keepdims=True).clip(1e-8)
    X = (vals - means) / stds
    X = np.nan_to_num(X)

    n = min(n_components, min(X.shape) - 1)
    if n < 1:
        return result

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    explained = eigvals / eigvals.sum()

    pc1_ratio = explained[0]
    regime_score = min(1, max(0, (pc1_ratio - 0.35) / 0.35))

    labels = ["LOW_VOL", "NORMAL", "HIGH_VOL", "EXTREME"]
    label = labels[min(3, int(regime_score * 4))]

    return {
        "regime_score": round(regime_score, 4),
        "regime_label": label,
        "component_ratios": [round(float(v), 4) for v in explained[:n]],
        "n_points": int(np.prod(X.shape)),
    }


# =============================================================================
# ERGODICITY — ERGODIC ECONOMICS ADJUSTMENT
# =============================================================================

def ergodic_market_gap(combined_df, spot, r=0.05):
    """
    Ergodic Market Gap.

    Sotto la dinamica ergodica di Peters (2019), il tasso di
    crescita atteso del sottostante è g = μ - σ²/2 (time-average)
    invece di μ (ensemble-average). Per le opzioni, il gap ergodico
    medio è pari a Σ wi × σi² × Ti / 2, pesato per OI.

    Returns dict con ergodic_gap, ergodic_score [0,1], n_options.
    """
    result = {"ergodic_gap": 0, "ergodic_score": 0.5, "n_options": 0}
    if combined_df is None or combined_df.empty or spot <= 0:
        return result

    df = combined_df.copy()
    needed = ["strike", "dte", "implied_volatility", "open_interest", "option_type"]
    if not all(c in df.columns for c in needed):
        return result

    df = df[(df["implied_volatility"] > 0.01) & (df["dte"] > 0) & (df["open_interest"] > 0)].copy()
    if df.empty:
        return result

    K = df["strike"].values.astype(float)
    T = (df["dte"].values.astype(float) / 365).clip(1/365)
    sig = df["implied_volatility"].values.astype(float)
    OI = df["open_interest"].values.astype(float)

    gaps = sig ** 2 * T / 2
    w = OI / OI.sum()
    weighted_gap = float(np.average(gaps, weights=w))

    score = min(1, max(0, weighted_gap * 15))

    return {
        "ergodic_gap": round(weighted_gap, 4),
        "ergodic_score": round(score, 4),
        "n_options": int(len(df)),
        "max_gap": round(float(gaps.max()), 4),
        "min_gap": round(float(gaps.min()), 4),
    }


# =============================================================================
# PROBABILISTIC LEVELS (B&L, Barrier, IV Extremes, ETF Rebalancing)
# =============================================================================

def bl_quantile_levels(chains: list, spot: float, target_dte: int = 30) -> dict:
    """Quantili risk-neutral via Breeden-Litzenberger (Pchip + diff finite + Brent).

    Implementazione model-free con solo numpy/scipy:
      1. PDF lato call:  f(K) = e^{rT} · C''(K)   per K >= spot
      2. PDF lato put:   f(K) = e^{rT} · P''(K)   per K <= spot
      3. Fusione e normalizzazione → CDF cumulativa
      4. Brent per quantili esatti di coda

    PchipInterpolator preserva la monotonicita' dei dati, eliminando
    oscillazioni spurie che CubicSpline produce su prezzi rumorosi.
    Derivata seconda via differenze finite centrali (stability > spl(x,2)).

    Args:
        chains: lista di DataFrames option chain per diverse expiry
        spot: prezzo corrente
        target_dte: DTE target

    Returns:
        dict con quantiles, dte, K, pdf, truncated, o vuoto.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.integrate import simpson
    from scipy.optimize import brentq

    if not chains:
        return {}

    best_chain = None
    best_dte = None
    best_delta = 999
    for df in chains:
        if df is None or df.empty:
            continue
        ds = df["dte"].iloc[0] if "dte" in df.columns else 999
        delta = abs(ds - target_dte)
        if delta < best_delta:
            best_delta = delta
            best_chain = df
            best_dte = ds

    if best_chain is None:
        return {}

    df = best_chain.copy()
    price_col = None
    for col in ["mid", "last_price", "lastPrice", "last", "close", "price"]:
        if col in df.columns:
            price_col = col
            break
    if price_col is None:
        return {}

    df = df.dropna(subset=["strike", price_col])
    df = df[df[price_col] >= 0.01]
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=["strike"])
    if df.empty:
        return {}

    type_col = "option_type" if "option_type" in df.columns else "type" if "type" in df.columns else None
    if type_col is None:
        return {}

    calls = df[df[type_col].astype(str).str.lower().str[0] == "c"].copy()
    puts = df[df[type_col].astype(str).str.lower().str[0] == "p"].copy()

    otm_calls = calls[calls["strike"] >= spot].copy()
    otm_puts = puts[puts["strike"] <= spot].copy()

    if otm_calls.empty or otm_puts.empty:
        return {}

    T = max(best_dte, 1) / 365.0
    r = 0.0

    def _pdf_from_pchip(ks, pr):
        ksort = np.argsort(ks)
        ks, pr = ks[ksort], pr[ksort]
        n = len(ks)
        if n < 3:
            return None, None
        try:
            spl = PchipInterpolator(ks, pr)
        except Exception:
            return None, None
        Kg = np.linspace(ks.min(), ks.max(), max(n * 2, 100))
        h = max(np.diff(Kg).mean(), 1e-4)
        # Differenze finite centrali per derivata seconda
        C2 = (spl(Kg + h) - 2 * spl(Kg) + spl(Kg - h)) / (h * h)
        C2 = np.clip(C2, 0, None)
        if C2.sum() <= 0:
            return None, None
        return Kg, C2

    # ── PDF lato call ──
    ks_c = otm_calls["strike"].values.astype(float)
    pr_c = otm_calls[price_col].values.astype(float)
    res_c = _pdf_from_pchip(ks_c, pr_c)
    if res_c is None or res_c[0] is None:
        return {}

    # ── PDF lato put ──
    ks_p = otm_puts["strike"].values.astype(float)
    pr_p = otm_puts[price_col].values.astype(float)
    res_p = _pdf_from_pchip(ks_p, pr_p)
    if res_p is None or res_p[0] is None:
        return {}

    Kg_c, C2_c = res_c
    Kg_p, C2_p = res_p
    pdf_c = np.exp(r * T) * C2_c
    pdf_p = np.exp(r * T) * C2_p

    # ── Fusione ──
    K_grid = np.unique(np.concatenate([Kg_p, Kg_c]))
    K_grid.sort()
    K_grid = K_grid[(K_grid >= Kg_p.min()) & (K_grid <= Kg_c.max())]

    pdf_grid = np.zeros_like(K_grid)
    put_zone = K_grid <= spot
    pdf_grid[put_zone] = np.interp(K_grid[put_zone], Kg_p, pdf_p, left=0, right=0)
    call_zone = K_grid >= spot
    pdf_grid[call_zone] = np.interp(K_grid[call_zone], Kg_c, pdf_c, left=0, right=0)

    total = simpson(pdf_grid, K_grid)
    if total <= 0:
        return {}
    pdf = pdf_grid / total

    # ── CDF ──
    cdf = np.zeros_like(K_grid)
    for i in range(1, len(K_grid)):
        cdf[i] = simpson(pdf[: i + 1], K_grid[: i + 1])
    cdf = np.clip(cdf, 0, 1)

    # ── Quantili via Brent ──
    def _cdf_val(k):
        return float(np.interp(k, K_grid, cdf))

    def _quantile_brent(p):
        target = p / 100.0
        if target <= cdf[0]:
            return round(float(K_grid[0]), 1)
        if target >= cdf[-1]:
            return round(float(K_grid[-1]), 1)
        idx = np.searchsorted(cdf, target)
        idx = np.clip(idx, 1, len(cdf) - 1)
        lo = K_grid[idx - 1]
        hi = K_grid[min(idx, len(K_grid) - 1)]
        for _ in range(5):
            flo = _cdf_val(lo) - target
            fhi = _cdf_val(hi) - target
            if flo * fhi < 0:
                break
            lo = max(K_grid[0], lo - (hi - lo))
            hi = min(K_grid[-1], hi + (hi - lo))
        else:
            return float(np.interp(target, cdf, K_grid))
        try:
            k = brentq(lambda kk: _cdf_val(kk) - target, lo, hi, xtol=0.01)
            return round(float(k), 1)
        except (ValueError, RuntimeError):
            return float(np.interp(target, cdf, K_grid))

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    levels = {pctl: _quantile_brent(pctl) for pctl in percentiles}

    q90, q95, q99 = levels.get(90, 0), levels.get(95, 0), levels.get(99, 0)
    right_truncated = (q90 >= q95) or (q95 - q90 < 0.5)
    left_truncated = (levels.get(1, 0) >= levels.get(5, 0))

    return {
        "quantiles": levels,
        "dte": best_dte,
        "K": K_grid.tolist(),
        "pdf": pdf.tolist(),
        "cdf": cdf.tolist(),
        "truncated": {"right": right_truncated, "left": left_truncated},
    }


def knockout_barriers(spot: float, iv: float, dte: float,
                      ratios: list = None) -> dict:
    """Calcola livelli Knock-Out barriera in unita' di volatilita'.

    b = (1/σ√T) * ln(B/S₀)

    Ogni barriera e' un livello dove la probabilita' di touch (sotto BS)
    raggiunge una certa soglia. Le barriere sono espresse come:
    - ratio: frazione di spot (es 0.95 = 5% sotto)
    - sigma_distance: deviazioni standard b = ln(ratio) / (σ√T)
    - touch_prob: P(min(S_t) <= B) approssimata 2 * N(b) per barriere down

    0DTE (dte=0) usa T = 1/365: σ√T e' piccolo ma le barriere rimangono
    calcolabili su distanze di volatilita' ragionevoli.

    Args:
        spot: prezzo corrente
        iv: volatilita' implicita annualizzata
        dte: giorni alla scadenza
        ratios: lista di rapporti barriera [down1, down2, up1, up2]

    Returns:
        list di dict con barriera, ratio, sigma_distance, touch_prob
    """
    # 0DTE (dte=0) e' valido: T = 1/365, σ√T piccolo ma barriere calcolabili
    if iv <= 0 or dte < 0:
        return []

    if ratios is None:
        ratios = [0.92, 0.95, 0.975, 0.99, 1.01, 1.025, 1.05, 1.08]

    T = max(dte, 1) / 365.0
    sigma = max(iv, 0.05)
    sigma_sqrt_T = sigma * np.sqrt(T)

    levels = []
    for r in ratios:
        if r <= 0:
            continue
        B = spot * r
        b = np.log(B / spot) / sigma_sqrt_T
        # Touch probability approssimata (barriera down: 2*N(b), up: 2*N(-b))
        if r < 1:
            touch_p = 2.0 * _norm_cdf(b)
        elif r > 1:
            touch_p = 2.0 * _norm_cdf(-b)
        else:
            touch_p = 1.0

        levels.append({
            "strike": round(B, 1),
            "ratio": r,
            "sigma_distance": round(b, 3),
            "touch_prob": round(min(touch_p, 1.0), 4),
            "direction": "down" if r < 1 else "up",
        })

    return levels


def iv_skew_extremes(option_df: pd.DataFrame, spot: float) -> dict:
    """Trova gli estremi dello skew di volatilita' implicita.

    Basic High/Low: IV massima/minima nello strike range ±5% da spot
    Interpolated High/Low: estremi della spline cubica IV su tutto lo strike range

    Args:
        option_df: DataFrame con colonne strike, implied_volatility, option_type
        spot: prezzo corrente

    Returns:
        dict con basic/interpolated extremes
    """
    if option_df is None or option_df.empty:
        return {}

    df = option_df.copy()
    has_iv = "implied_volatility" in df.columns
    has_type = "option_type" in df.columns or "type" in df.columns

    if not has_iv:
        return {}

    type_col = "option_type" if "option_type" in df.columns else "type"
    iv_mask = (df["implied_volatility"].notna()
               & (df["implied_volatility"] >= 0.05)
               & (df["implied_volatility"] <= 2.0))
    df = df[iv_mask].copy()

    if df.empty:
        return {}

    # Basic: range ±5% da spot
    near = df[df["strike"].between(spot * 0.95, spot * 1.05)].copy()
    basic = {}
    if not near.empty:
        basic = {
            "basic_high": float(near.loc[near["implied_volatility"].idxmax(), "strike"]),
            "basic_high_iv": float(near["implied_volatility"].max()),
            "basic_low": float(near.loc[near["implied_volatility"].idxmin(), "strike"]),
            "basic_low_iv": float(near["implied_volatility"].min()),
        }

    # Interpolated: spline su range ±10% da spot (evita code 0DTE insensate)
    # con sanity check: high deve essere sul lato put (strike < spot),
    # low sul lato call (strike > spot). Se invertiti, scambia.
    interpolated = {}
    try:
        from scipy.interpolate import CubicSpline
        lo, hi = spot * 0.90, spot * 1.10
        sub = df[df["strike"].between(lo, hi)].copy()
        if len(sub) >= 6:
            grouped = sub.groupby("strike", as_index=False)["implied_volatility"].mean()
            grouped = grouped.sort_values("strike")
            K_s = grouped["strike"].values
            IV_s = grouped["implied_volatility"].values
            K_min, K_max = float(K_s.min()), float(K_s.max())
            cs = CubicSpline(K_s, IV_s, bc_type="natural")
            K_dense = np.linspace(K_min, K_max, 300)
            IV_dense = cs(K_dense)

            idx_max = int(np.argmax(IV_dense))
            idx_min = int(np.argmin(IV_dense))
            ik_hi = float(K_dense[idx_max])
            ik_lo = float(K_dense[idx_min])

            # Equity skew: put IV > call IV → high su put (< spot), low su call (> spot)
            if ik_hi > spot and ik_lo < spot:
                ik_hi, ik_lo = ik_lo, ik_hi
                idx_max, idx_min = idx_min, idx_max

            # Clamp al range osservato (spline puo' estrapolare oltre i dati)
            if ik_hi < K_min or ik_hi > K_max:
                ik_hi = basic.get("basic_high", float(K_s[idx_max]))
            if ik_lo < K_min or ik_lo > K_max:
                ik_lo = basic.get("basic_low", float(K_s[idx_min]))

            interpolated = {
                "interp_high": ik_hi,
                "interp_high_iv": float(IV_dense[idx_max]),
                "interp_low": ik_lo,
                "interp_low_iv": float(IV_dense[idx_min]),
            }
    except Exception:
        pass

    return {**basic, **interpolated}


def etf_rebalancing_levels(spot: float, spy_return: float = None,
                           spy_atm_dex: float = None) -> dict:
    """Stima i livelli critici legati al rebalancing degli ETF a leva.

    Gli ETF a leva (SSO +200%, QLD +200%, TQQQ +300%, SQQQ -300%)
    devono ribilanciare il loro delta giornaliero quando il sottostante muove.

    Il flusso netto e': Net_Flow = sum_i AUM_i * (leverage_i - 1) * return

    Args:
        spot: prezzo corrente SPX/SPY
        spy_return: rendimento % intraday di SPY (None = assume 0)
        spy_atm_dex: delta exposure ATM per stima impatto (None = skip)

    Returns:
        dict con flussi stimati e livelli critici
    """
    # AUM approssimativi degli ETF leva su SPY (dic 2024-2025)
    LEV_ETFS = {
        "SSO":  {"aum_bn": 4.2, "lev": 2.0},
        "UPRO": {"aum_bn": 3.8, "lev": 3.0},
        "SDS":  {"aum_bn": 1.5, "lev": -2.0},
        "SPXU": {"aum_bn": 0.9, "lev": -3.0},
    }

    result = {
        "total_flow_bn": 0.0,
        "net_direction": "neutral",
        "by_etf": {},
        "critical_levels": [],
    }

    if spy_return is not None and abs(spy_return) > 0.0001:
        total_flow = 0.0
        for name, info in LEV_ETFS.items():
            flow = info["aum_bn"] * (info["lev"] - 1) * spy_return
            total_flow += flow
            result["by_etf"][name] = round(flow, 4)

        result["total_flow_bn"] = round(total_flow, 4)
        result["net_direction"] = "buy" if total_flow > 0 else "sell"

    if spy_atm_dex is not None and abs(spy_atm_dex) > 0:
        # Livello critico: dove il flusso leva e' > 10% del DEX ATM
        flow_abs = abs(result["total_flow_bn"])
        dex_bn = abs(spy_atm_dex) / 1e9
        if dex_bn > 0 and (flow_abs / dex_bn) > 0.10:
            result["critical_levels"].append({
                "type": "leverage_rebalancing",
                "impact": round(flow_abs / dex_bn * 100, 1),
                "note": f"Flusso leva {flow_abs:.2f}B = {flow_abs/dex_bn*100:.0f}% del DEX ATM",
            })

    return result


def compute_probabilistic_levels(chains: list, spot: float, iv: float, dte: int) -> dict:
    """Calcola tutti i livelli probabilistici in un'unica chiamata.

    Unisce B&L quantiles, barrier levels, IV extremes, e volume profile
    in un unico dict per il rendering nel terminale.

    Args:
        chains: lista DataFrames per tutte le expiry
        spot: prezzo corrente
        iv: volatilita' implicita ATM
        dte: DTE target

    Returns:
        dict completo con tutti i livelli
    """
    result = {"spot": spot, "iv": iv, "dte": dte}

    # B&L Risk Moments
    bl = bl_quantile_levels(chains, spot, target_dte=min(dte, 45))
    if bl:
        result["bl_quantiles"] = bl

    # Knock-Out Barriers
    barriers = knockout_barriers(spot, iv, dte)
    if barriers:
        result["barriers"] = barriers

    return result


# =============================================================================
# BKM MODEL-FREE IMPLIED MOMENTS (Bakshi-Kapadia-Madan 2003)
# =============================================================================

def implied_risk_moments(option_df: pd.DataFrame, spot: float,
                         r: float = 0, T: float = None) -> dict:
    """Calcola momenti risk-neutral model-free (BKM 2003).

    Estrae varianza, skewness e kurtosis dalla catena opzioni OTM
    senza assumere un modello parametrico. I momenti risk-neutral
    incorporano i premi per il rischio di coda (λ₂, λ₃, λ₄).

    Formula (BKM):
      V  = ∫ (2(1 - ln(K/S)) / K²) · OTM(K) dK
      W  = ∫ (6 ln(K/S) - 3 ln²(K/S)) / K² · OTM(K) dK   [call side]
      W  = ∫ (6 ln(S/K) + 3 ln²(S/K)) / K² · OTM(K) dK   [put side]
      X  = ∫ (12 ln²(K/S) - 4 ln³(K/S)) / K² · OTM(K) dK [call side]
      X  = ∫ (12 ln²(S/K) + 4 ln³(S/K)) / K² · OTM(K) dK [put side]

    Args:
        option_df: DataFrame con strike, option_type, last_price / mid
        spot: prezzo corrente
        r: tasso risk-free (default 0)
        T: time-to-maturity in anni (se None, usa DTE medio)

    Returns:
        dict con varianza, skewness, kurtosis, e componenti V/W/X
    """
    if option_df is None or option_df.empty:
        return {}

    df = option_df.copy()
    if "strike" not in df.columns:
        return {}
    has_type = "option_type" in df.columns
    has_price = "last_price" in df.columns or "mid" in df.columns
    if not has_type or not has_price:
        return {}

    type_col = "option_type"
    price_col = "last_price" if "last_price" in df.columns else "mid"

    # Filtra prezzi validi
    df = df[df[price_col].notna() & (df[price_col] > 0)].copy()
    if df.empty:
        return {}

    # Determina T dal DTE medio se non fornito
    if T is None and "dte" in df.columns:
        T = float(df["dte"].mean()) / 365.0
    if T is None or T <= 0:
        T = 30.0 / 365.0

    # Separa call e put OTM
    calls = df[(df[type_col].str.lower().str[0] == "c") & (df["strike"] >= spot)].copy()
    puts = df[(df[type_col].str.lower().str[0] == "p") & (df["strike"] <= spot)].copy()

    if calls.empty and puts.empty:
        return {}

    # Funzioni peso BKM
    def _integrate_otm(otm_df, side):
        """Integra OTM(K) · peso(K) / K² (trapezoidale manuale)."""
        if otm_df.empty or len(otm_df) < 3:
            return 0.0, 0.0, 0.0
        K = otm_df["strike"].values.astype(float)
        opt_price = otm_df[price_col].values.astype(float)

        if side == "call":
            m = K / spot
            ln_m = np.log(m)
            w_V = 2.0 * (1.0 - ln_m) / (K ** 2)
            w_W = (6.0 * ln_m - 3.0 * ln_m ** 2) / (K ** 2)
            w_X = (12.0 * ln_m ** 2 - 4.0 * ln_m ** 3) / (K ** 2)
        else:
            m = spot / K
            ln_m = np.log(m)
            w_V = 2.0 * (1.0 + ln_m) / (K ** 2)
            w_W = (6.0 * ln_m + 3.0 * ln_m ** 2) / (K ** 2)
            w_X = (12.0 * ln_m ** 2 + 4.0 * ln_m ** 3) / (K ** 2)

        integrand_V = opt_price * w_V
        integrand_W = opt_price * w_W
        integrand_X = opt_price * w_X
        # Trapezoidale manuale: ΔK · avg(f(K_i), f(K_{i+1}))
        dK = np.diff(K)
        V = float(np.sum(0.5 * dK * (integrand_V[:-1] + integrand_V[1:])))
        W = float(np.sum(0.5 * dK * (integrand_W[:-1] + integrand_W[1:])))
        X = float(np.sum(0.5 * dK * (integrand_X[:-1] + integrand_X[1:])))
        return V, W, X

    V_c, W_c, X_c = _integrate_otm(calls, "call")
    V_p, W_p, X_p = _integrate_otm(puts, "put")

    V = V_c + V_p
    W = W_c - W_p  # segno: call +W, put -W
    X = X_c + X_p

    if V <= 0:
        return {}

    # BKM momenti risk-neutral
    mu = np.exp(r * T) - 1 - np.exp(r * T) * V / 2 - np.exp(r * T) * W / 6 - np.exp(r * T) * X / 24
    sigma2 = np.exp(r * T) * V - mu ** 2
    sigma = np.sqrt(sigma2) if sigma2 > 0 else 0.0

    skew = 0.0
    if sigma > 1e-10:
        skew = (np.exp(r * T) * W - 3 * mu * np.exp(r * T) * V + 2 * mu ** 3) / (sigma ** 3)

    kurt = 0.0
    if sigma > 1e-10:
        kurt = (np.exp(r * T) * X - 4 * mu * np.exp(r * T) * W
                + 6 * mu ** 2 * np.exp(r * T) * V - 3 * mu ** 4) / (sigma ** 4)

    return {
        "variance": round(sigma2, 6),
        "volatility": round(sigma, 6),
        "skewness": round(float(skew), 4),
        "kurtosis": round(float(kurt), 4),
        "V_contract": round(V, 6),
        "W_contract": round(W, 6),
        "X_contract": round(X, 6),
        "n_calls": len(calls),
        "n_puts": len(puts),
        "T_years": round(T, 4),
    }


# =============================================================================
# STATISTICAL LEARNING — Box Plot, Z‑Score, Correlazione, Rumore
# =============================================================================

def box_plot_stats(prices: np.ndarray) -> dict:
    """Calcola statistiche Box Plot da una serie prezzi.

    Q25, Q50, Q75, IQR, whiskers ±1.5·IQR, skewness, kurtosis.
    """
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 4:
        return {}

    q25 = float(np.percentile(arr, 25))
    q50 = float(np.percentile(arr, 50))
    q75 = float(np.percentile(arr, 75))
    iqr = q75 - q25
    lo_whisker = q25 - 1.5 * iqr
    hi_whisker = q75 + 1.5 * iqr

    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr, fisher=True))  # excess kurtosis (normale = 0)

    return {
        "q25": round(q25, 2), "q50": round(q50, 2), "q75": round(q75, 2),
        "iqr": round(iqr, 2),
        "lo_whisker": round(lo_whisker, 2), "hi_whisker": round(hi_whisker, 2),
        "skewness": round(skew, 4), "kurtosis": round(kurt, 4),
    }


def z_score_analysis(current_price: float, prices: np.ndarray,
                     window: int = 89, excess_kurtosis: float = None) -> dict:
    """Calcola Z‑Score, category e stop loss dinamici.

    Se excess_kurtosis e' fornito (>0), applica Cornish‑Fisher expansion
    per aggiustare le soglie di interpretazione alle code grasse.

    Categorie base (normale):
      |Z| < 1.0  → normale (68 %)
      |Z| > 2.0  → esaurimento (95 %)
      |Z| > 3.0  → evento estremo (99.7 %)

    Con code grasse (excess_kurtosis > 0), le soglie si allargano:
      soglia_2σ_cf ≈ 2 + 2·(skew/6 + kurt/24)
    """
    arr = np.asarray(prices[-window:], dtype=float) if len(prices) > window else np.asarray(prices, dtype=float)
    if len(arr) < 4:
        return {}

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma < 1e-10:
        return {}

    z = (current_price - mu) / sigma

    # Cornish-Fisher: soglie corrette per skew/kurtosi
    sk = float(stats.skew(arr))
    ek_hist = float(stats.kurtosis(arr, fisher=True))
    ek = excess_kurtosis if excess_kurtosis is not None else ek_hist
    # Usa skew storico + kurtosi (BKM se fornita, altrimenti storica)
    cf_shift = 2.0 * (sk / 6.0 + ek / 24.0)
    threshold_2s = 2.0 + cf_shift
    threshold_3s = 3.0 + 1.5 * cf_shift  # piu' ampia per 3σ

    # Usa la funzione standalone CF per metriche aggiuntive
    _cf = cornish_fisher_adjustment(z, sk, ek)
    _cat = _cf["category"] if excess_kurtosis is not None else (
        "extreme" if abs(z) > 3.0 else "exhaustion" if abs(z) > 2.0 else "warning" if abs(z) > 1.0 else "normal"
    )

    return {
        "z_score": round(z, 4),
        "z_cf_adjusted": _cf["cf_z_score"],
        "z_cf_p_value": _cf["cf_p_value"],
        "cf_threshold_2s": _cf["threshold_2s"],
        "cf_threshold_3s": _cf["threshold_3s"],
        "category": _cat,        "mean": round(mu, 2),
        "std": round(sigma, 2),
        "stop_1s": round(current_price - 1.0 * sigma, 2),
        "stop_2s": round(current_price - 2.0 * sigma, 2),
        "stop_reverse_1s": round(current_price + 1.0 * sigma, 2),
        "stop_reverse_2s": round(current_price + 2.0 * sigma, 2),
        "excess_kurtosis": round(ek, 4),
        "skewness": round(sk, 4),
    }


def cornish_fisher_adjustment(z_score: float, skew: float, excess_kurtosis: float) -> dict:
    """Corregge Z-score per skew e kurtosis (espansione di Cornish-Fisher).

    Una distribuzione con code grasse (excess_kurtosis > 0) allarga le soglie:
    un movimento di 2σ puo' essere "normale". La correzione CF riporta
    le soglie alla correte probabilita' sotto la distribuzione empirica.

    Formula (primo ordine):
        z_cf = z + (z² - 1)·sk/6 + (z³ - 3z)·ek/24

    Args:
        z_score: Z-score osservato
        skew: skewness campionaria
        excess_kurtosis: kurtosi in eccesso (kurtosis - 3)

    Returns:
        dict: cf_z_score, cf_p_value, thresholds, category
    """
    z = z_score
    sk = skew
    ek = excess_kurtosis

    cf_z = z + (z * z - 1.0) * sk / 6.0 + (z * z * z - 3.0 * z) * ek / 24.0
    cf_p = 2.0 * (1.0 - stats.norm.cdf(abs(cf_z)))

    z2 = 2.0
    z3 = 3.0
    th2 = z2 + (z2 * z2 - 1.0) * sk / 6.0 + (z2 * z2 * z2 - 3.0 * z2) * ek / 24.0
    th3 = z3 + (z3 * z3 - 1.0) * sk / 6.0 + (z3 * z3 * z3 - 3.0 * z3) * ek / 24.0

    if abs(z) > th3:
        cat = "extreme"
    elif abs(z) > th2:
        cat = "exhaustion"
    else:
        cat = "normal"

    return {
        "cf_z_score": round(cf_z, 4),
        "cf_p_value": round(cf_p, 6),
        "threshold_2s": round(th2, 4),
        "threshold_3s": round(th3, 4),
        "category": cat,
    }


def calculate_expected_shortfall(pdf: np.ndarray, cdf: np.ndarray,
                                  K: np.ndarray, alpha: float = 0.05) -> dict:
    """Expected Shortfall (CVaR) dalla densita' risk-neutral B&L.

    ES(α) = E[K | K ≤ VaR(α)] = (1/α) ∫₀^{α} VaR(u) du

    Usando la PDF risk-neutral, calcola il valor medio della coda sinistra
    oltre il VaR(α). Piu' informativo del VaR perche' cattura la magnitudo
    dello scenario peggiore.

    Args:
        pdf: PDF risk-neutral normalizzata (da bl_quantile_levels)
        cdf: CDF cumulativa corrispondente
        K: griglia strike
        alpha: livello di coda (default 0.05 = 5%)

    Returns:
        dict: VaR, ES, alpha, loss_pct (rispetto a spot presunto=max(K)-...)
    """
    from scipy.integrate import simpson

    if len(pdf) != len(cdf) or len(pdf) < 5:
        return {}
    if alpha <= 0 or alpha >= 1:
        return {}

    var_idx = np.searchsorted(cdf, alpha)
    var_idx = np.clip(var_idx, 1, len(K) - 1)
    var = K[var_idx]

    tail_mask = K <= var
    tail_K = K[tail_mask]
    tail_pdf = pdf[tail_mask]
    if tail_pdf.sum() <= 0:
        return {}

    tail_pdf_norm = tail_pdf / tail_pdf.sum()
    es = simpson(tail_K * tail_pdf_norm, tail_K)

    return {
        "VaR": round(float(var), 2),
        "ES": round(float(es), 2),
        "alpha": alpha,
    }


def adapted_var(spot: float, barrier: float, sigma: float,
                dte: int, drift: float = 0.0) -> dict:
    """Probabilita' di toccare una barriera prima della scadenza (Adapted VaR).

    Formula chiusa per il primo passaggio del moto browniano geometrico:
      P(τ_B ≤ T) = Φ(a) + exp(2·ν·ln(B/S₀)/σ²)·Φ(b)

    dove ν = μ - σ²/2, a = (ln(B/S₀) - νT)/(σ√T), b = (ln(B/S₀) + νT)/(σ√T)

    A differenza del VaR tradizionale (solo a scadenza), questa stima la
    probabilita' di essere liquidati in qualsiasi momento della sessione
    (Larcher). Fondamentale per scalping con leva.

    Args:
        spot: prezzo corrente
        barrier: livello barriera (stop loss)
        sigma: volatilita' annualizzata (es. 0.20 per 20%)
        dte: giorni alla scadenza
        drift: drift annualizzato (0 = risk-neutral, storico ≈ 0.05-0.10)

    Returns:
        dict: touch_prob, barrier_type, distance_pct, dte
    """
    if sigma <= 0 or dte <= 0 or spot <= 0 or barrier <= 0:
        return {}

    T = dte / 365.0
    nu = drift - 0.5 * sigma * sigma
    log_ratio = np.log(barrier / spot)

    vol_sqrt_t = sigma * np.sqrt(T)
    if vol_sqrt_t < 1e-12:
        return {}

    a = (log_ratio - nu * T) / vol_sqrt_t
    b = (log_ratio + nu * T) / vol_sqrt_t

    touch_prob = stats.norm.cdf(a) + np.exp(2.0 * nu * log_ratio / (sigma * sigma)) * stats.norm.cdf(b)
    # Satura a 1 per barriere molto vicine
    touch_prob = min(float(touch_prob), 1.0)

    direction = "down" if barrier < spot else "up"
    distance_pct = abs(barrier - spot) / spot * 100

    return {
        "touch_prob": round(touch_prob, 6),
        "barrier_type": direction,
        "distance_pct": round(distance_pct, 2),
        "dte": dte,
        "barrier": round(float(barrier), 2),
        "spot": round(float(spot), 2),
    }


def correlation_signal(price_changes: np.ndarray,
                       flow_changes: np.ndarray) -> dict:
    """Pearson r tra flussi e prezzo, qualita' del segnale.

    r → +1  segnale forte direzionale
    r →  0  nessuna relazione lineare (rumore)
    r → −1  segnale inverso
    """
    pc = np.asarray(price_changes, dtype=float)
    fc = np.asarray(flow_changes, dtype=float)
    if len(pc) < 6 or len(fc) < 6:
        return {}
    if np.std(pc, ddof=1) < 1e-10 or np.std(fc, ddof=1) < 1e-10:
        return {}

    r, p_val = stats.pearsonr(pc, fc)

    if abs(r) > 0.7:
        quality = "strong"
    elif abs(r) > 0.4:
        quality = "moderate"
    elif abs(r) > 0.2:
        quality = "weak"
    else:
        quality = "none"

    return {
        "r": round(r, 4),
        "p_value": round(p_val, 6),
        "quality": quality,
        "direction": "same" if r > 0 else "inverse",
    }


def noise_filter(prices: np.ndarray, window: int = 20,
                 ohlc: np.ndarray = None) -> dict:
    arr = np.asarray(prices, dtype=float)
    if len(arr) < window * 2 + 1:
        return {}

    if ohlc is not None and ohlc.shape[0] >= window * 2:
        oa = np.asarray(ohlc, dtype=float)
        var_recent = _yang_zhang_slice(oa[-window:])
        var_prior = _yang_zhang_slice(oa[-(window * 2):-window])
        sigma_recent = np.sqrt(var_recent)
        method = "yang_zhang"
        rets = np.diff(np.log(arr))
    else:
        returns = np.diff(np.log(arr))
        recent = returns[-window:]
        prior = returns[-(window * 2):-window]
        var_recent = float(np.var(recent, ddof=1))
        var_prior = float(np.var(prior, ddof=1))
        sigma_recent = float(np.std(recent, ddof=1))
        method = "log_returns"
        rets = returns

    ratio = var_recent / var_prior if var_prior > 1e-10 else 1.0
    directionality = abs(np.mean(rets[-window:])) / (sigma_recent + 1e-10)

    if ratio > 2.0:
        verdict = "noise_dominant"
    elif ratio > 1.5:
        verdict = "noise_increasing"
    else:
        verdict = "normal"

    return {
        "var_recent": round(var_recent, 6),
        "var_prior": round(var_prior, 6),
        "sigma_recent": round(sigma_recent, 4),
        "variance_ratio": round(ratio, 4),
        "directionality": round(directionality, 4),
        "verdict": verdict,
        "method": method,
    }


def _yang_zhang_slice(ohlc: np.ndarray) -> float:
    """Stimatore Yang‑Zhang su un batch N×4 (O,H,L,C).

    σ²_YZ = σ²_overnight + k·σ²_RS + (1−k)·σ²_close
    con k=0.5 (default).
    σ²_RS = (h−c)(h−o) + (l−c)(l−o)   (Rogers‑Satchell)
    σ²_overnight = var(ln(oᵢ / cᵢ₋₁))
    σ²_close = var(ln(cᵢ / cᵢ₋₁))
    """
    o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
    n = len(ohlc)
    if n < 2:
        return 0.0
    # Rogers-Satchell: σ²_RS = (h-c)*(h-o) + (l-c)*(l-o)
    rs = ((h - c) * (h - o) + (l - c) * (l - o)).mean()
    # Close-to-close var
    cc_var = np.var(np.diff(np.log(c)), ddof=1) if n > 1 else 0.0
    # Open var (overnight jump)
    oi = np.log(o[1:] / c[:-1])
    oi_var = np.var(oi, ddof=1) if len(oi) > 1 else 0.0
    # Yang-Zhang = σ²_overnight + 0.5·σ²_RS + 0.5·σ²_cc
    yz = oi_var + 0.5 * rs + 0.5 * cc_var
    return max(yz, 1e-12)


def structural_confluence(
    spot: float,
    sigma_bands: dict = None,
    bl_quantiles: dict = None,
    iv_extremes: dict = None,
    ko_barriers: list = None,
    tolerance_bp: float = 5.0,
) -> list:
    """Identifica confluenze tra famiglie di livelli indipendenti.

    Ogni livello e' taggato con una famiglia:
      - 'statistical': σ‑bande da Z‑score
      - 'bl': quantili B&L
      - 'iv_skew': estremi IV skew
      - 'barrier': KO barriers

    Se >= 2 famiglie cadono entro tolerance_bp (basis point di spot),
    viene generato un alert 'Magnete Strutturale'.

    Returns:
        list di dict: [{strike, famiglie, livelli, n_famiglie, strength}, ...]
    """
    tolerance = tolerance_bp / 10000 * spot
    levels = []  # (strike, famiglia, etichetta)

    # 1. Statistical: σ‑bande
    if sigma_bands:
        mu = sigma_bands.get("mean", spot)
        sigma = sigma_bands.get("std", 0)
        if sigma > 0:
            levels.append((mu, "statistical", "μ"))
            for k in [1, 2, 3]:
                levels.append((mu - k * sigma, "statistical", f"-{k}σ"))
                levels.append((mu + k * sigma, "statistical", f"+{k}σ"))

    # 2. B&L quantiles
    if bl_quantiles:
        qs = bl_quantiles.get("quantiles", {})
        for p, v in qs.items():
            label = f"Q{p:02d}"
            if p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
                levels.append((v, "bl", label))

    # 3. IV skew extremes
    if iv_extremes:
        for key, fam_label in [("basic_high", "IVmax@"), ("basic_low", "IVmin@"),
                                ("interp_high", "IVmaxS"), ("interp_low", "IVminS")]:
            v = iv_extremes.get(key)
            if v is not None:
                levels.append((v, "iv_skew", fam_label))

    # 4. KO barriers
    if ko_barriers:
        for b in ko_barriers:
            d = "⬇" if b.get("direction") == "down" else "⬆"
            tp = b.get("touch_prob", None)
            label = f"KO{d}"
            if tp is not None:
                label += f" {tp:.0%}"
            levels.append((b["strike"], "barrier", label))

    if len(levels) < 2:
        return []

    levels.sort(key=lambda x: x[0])

    # Cluster greedy: merge livelli entro tolerance
    clusters = []
    current = [levels[0]]
    for lv in levels[1:]:
        if lv[0] - current[-1][0] <= tolerance:
            current.append(lv)
        else:
            if len(current) > 1:
                clusters.append(current)
            current = [lv]
    if len(current) > 1:
        clusters.append(current)

    result = []
    for cl in clusters:
        strikes = [c[0] for c in cl]
        familias = list(set(c[1] for c in cl))
        n = len(familias)
        strike_avg = float(np.mean(strikes))
        strength = n  # 2 = confluenza, 3+ = magnete forte
        label = "Magnete" if n >= 3 else "Confluenza"
        result.append({
            "strike": round(strike_avg, 1),
            "n_famiglie": n,
            "famiglie": familias,
            "livelli": [
                {"strike": round(c[0], 1), "famiglia": c[1], "label": c[2]}
                for c in cl
            ],
            "label": label,
            "strength": strength,
            "tolerance": round(tolerance, 1),
        })

    result.sort(key=lambda r: r["strength"], reverse=True)
    return result


def decompose_risk_premia(bkm_moments: dict, historical_returns: np.ndarray = None,
                          hist_skew: float = None, hist_kurt: float = None) -> dict:
    """Decomposizione λ dei risk premia (Vázquez / Bergomi‑Guyon).

    Confronta i momenti risk‑neutral (BKM) con i momenti storici (P) per
    isolare i premi per:

      λ₂ = σ²_Q / σ²_P − 1        convessità (convexity risk premium)
      λ₃ = skew_Q − skew_P        skew risk premium
      λ₄ = kurt_Q − kurt_P        coda (tail/kurtosis risk premium)

    Include il bound teorico (Eq. 66, Vázquez):
      kurt_Q ≥ skew_Q² + 1
    Se violato, λ₄ viene saturato per evitare stime instabili.
    """
    if not bkm_moments or bkm_moments.get("variance", 0) <= 0:
        return {}

    var_q = bkm_moments["variance"]
    skew_q = bkm_moments.get("skewness", 0)
    kurt_q = bkm_moments.get("kurtosis", 3)

    # Bound teorico: kurt_Q ≥ skew_Q² + 1 (moment problem)
    kurt_bound = skew_q ** 2 + 1.0
    kurt_saturated = kurt_q < kurt_bound
    if kurt_saturated:
        kurt_q_safe = max(kurt_q, kurt_bound)
    else:
        kurt_q_safe = kurt_q

    # Momenti P (fisici) da rendimenti storici o argomenti
    if historical_returns is not None and len(historical_returns) > 20:
        r = np.asarray(historical_returns, dtype=float)
        var_p = float(np.var(r, ddof=1))
        skew_p = float(stats.skew(r)) if hist_skew is None else hist_skew
        kurt_p = float(stats.kurtosis(r, fisher=False)) if hist_kurt is None else hist_kurt
    elif hist_skew is not None and hist_kurt is not None:
        skew_p = hist_skew
        kurt_p = hist_kurt
        var_p = var_q
    else:
        return {}

    if var_p is None or var_p <= 1e-12:
        var_p = var_q

    lam2 = var_q / var_p - 1.0 if var_p > 1e-12 else 0.0
    lam3 = skew_q - skew_p
    lam4 = kurt_q_safe - kurt_p

    parts = []
    if lam2 > 0.3:
        parts.append(f"λ₂+{lam2:.2f}")
    elif lam2 < -0.1:
        parts.append(f"λ₂{lam2:.2f}")

    if abs(lam3) > 0.1:
        parts.append(f"λ₃{'↗' if lam3>0 else '↘'}{abs(lam3):.2f}")

    if lam4 > 0.5:
        w = "⚠" if kurt_saturated else " "
        parts.append(f"λ₄{w}{lam4:.2f}")

    if not parts:
        summary = "premi neutrali"
    else:
        summary = " + ".join(parts)

    return {
        "lam2": round(lam2, 4),
        "lam3": round(lam3, 4),
        "lam4": round(lam4, 4),
        "var_q": round(var_q, 6),
        "var_p": round(var_p, 6),
        "skew_q": round(skew_q, 4),
        "skew_p": round(skew_p, 4),
        "kurt_q": round(kurt_q_safe, 4),
        "kurt_p": round(kurt_p, 4),
        "kurt_saturated": kurt_saturated,
        "summary": summary,
    }


# =============================================================================
# TEST / MAIN
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf
    
    print("--- TEST GAMMA BANDS ---")
    tk = yf.Ticker("SPY")
    try:
        exps = tk.options[:3]
        chains = []
        today = datetime.now().date()
        for e in exps:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if dte < 0: continue
            c = tk.option_chain(e)
            df = pd.concat([c.calls.assign(type="call"), c.puts.assign(type="put")], ignore_index=True)
            df["dte"] = dte
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            df = df.dropna(subset=["strike"])
            chains.append(enrich_greeks(df, 5500))
        if chains:
            opt = pd.concat(chains, ignore_index=True)
            gb = gamma_bands(opt, 5500)
            print(gb.to_string(index=False))
    except Exception as e:
        print(f"Errore: {e}")
    
    print("\n--- GARCH MC (synthetic) ---")
    synth = 5000 + 100 * np.cumsum(np.random.randn(500))
    paths, finals, vol = garch_mc_paths(synth, n_paths=100, n_steps=20)
    print(f"Paths shape: {paths.shape}, Vol: {vol['vol_initial']:.4f}")
    
    # Test LinReg
    print("\n--- LINREG CHANNELS ---")
    lr = linreg_channels(synth, window=100)
    print(f"Slope={lr['slope']:.4f} R²={lr['r_squared']:.4f} Std={lr['std']:.2f}")
