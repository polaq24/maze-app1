"""
Macro Quant — Multi-Asset Regime Detection & Directional Bias
==============================================================
Integra VIX, DXY, Copper, SPX per determinare il regime macro,
il bias direzionale e prevedere cambi di regime imminenti.

Include:
  - Walk-forward validation per robustezza out-of-sample
  - Correzione di Bonferroni per significativita statistica
  - Report automatico Daily/Weekly Breakdown

Filosofia: nessuna discrezionalita, solo relazioni statistiche
           quantitative tra fattori macroeconomici.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings
import logging
from itertools import combinations

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

@dataclass
class Config:
    tickers: Dict[str, str] = field(default_factory=lambda: {
        "SPX": "^GSPC",
        "VIX": "^VIX",
        "VIX1D": "^VIX1D",
        "VIX9D": "^VIX9D",
        "VVIX": "^VVIX",
        "DXY": "DX-Y.NYB",
        "COPPER": "HG=F",
        "WTI": "CL=F",
        "SKEW": "^SKEW",
    })
    lookback_days: int = 504       # 2 anni di trading
    regime_window: int = 63        # ~3 mesi per regime detection
    shift_window: int = 21         # ~1 mese per prevedere shift
    correlation_window: int = 42   # ~2 mesi per correlazioni
    risk_free_rate: float = 0.05
    bonferroni_alpha: float = 0.05  # alpha complessivo
    walk_forward_splits: int = 6    # fold per walk-forward
    output_dir: str = "macro_charts" # cartella per i grafici HTML
    signal_file: str = "raw/last_signal.txt"  # output per sistema trading
    intraday_interval_min: int = 60  # refresh ogni N min in intraday mode

CONFIG = Config()


# =============================================================================
# ENUM - Regimi di mercato
# =============================================================================

class Regime(Enum):
    EXPANSION = "EXPANSION"
    SLOWDOWN = "SLOWDOWN"
    REFLATION = "REFLATION"
    STAGFLATION = "STAGFLATION"
    TRANSITION = "TRANSIZIONE"

    def encode(self) -> int:
        return {
            Regime.EXPANSION: 1,
            Regime.SLOWDOWN: 2,
            Regime.REFLATION: 3,
            Regime.STAGFLATION: 4,
            Regime.TRANSITION: 5,
        }.get(self, 5)

class Bias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


# =============================================================================
# DATA - Multi-Asset DataFetcher
# =============================================================================

class DataFetcher:
    """Recupera dati multi-asset (VIX, DXY, Copper, SPX) da yfinance."""

    def __init__(self):
        self._yf = self._check_yfinance()

    def _check_yfinance(self):
        try:
            import yfinance as yf
            return yf
        except ImportError:
            log.warning("yfinance non installato. Uso dati sintetici.")
            return None

    def fetch_all(self) -> Dict[str, pd.Series]:
        """Scarica i prezzi di chiusura per tutti i ticker configurati."""
        if not self._yf:
            return self._synthetic_data()

        end = datetime.now()
        # Richiedi 3 anni per assicurarsi sovrapposizione sufficiente
        start = end - timedelta(days=CONFIG.lookback_days * 3)
        result = {}

        for name, ticker in CONFIG.tickers.items():
            try:
                t = self._yf.Ticker(ticker)
                hist = t.history(start=start.strftime("%Y-%m-%d"),
                                 end=end.strftime("%Y-%m-%d"))
                if not hist.empty:
                    result[name] = hist["Close"]
                    log.info(f"{name} ({ticker}): {len(hist)} giorni, "
                             f"date [{hist.index[0].date()} .. {hist.index[-1].date()}]")
                else:
                    log.warning(f"{name}: nessun dato, uso sintetico")
                    result[name] = self._synthetic_single(name)
            except Exception as e:
                log.warning(f"{name}: errore ({e}), uso sintetico")
                result[name] = self._synthetic_single(name)

        return result

    def _synthetic_data(self) -> Dict[str, pd.Series]:
        log.info("Generazione dati sintetici per tutti gli asset...")
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=CONFIG.lookback_days * 2, freq="B")

        spx = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, len(dates))))
        vix = 15 + 10 * np.random.exponential(0.5, len(dates)).cumsum()
        vix = 10 + (vix - vix.min()) / (vix.max() - vix.min()) * 25

        dxy_ret = np.random.normal(0, 0.003, len(dates))
        dxy = 100 * np.exp(np.cumsum(dxy_ret))

        copper = 3.5 + 0.5 * np.random.randn(len(dates)).cumsum() / 20
        copper = np.clip(copper, 2.5, 5.0)

        return {
            "SPX": pd.Series(spx, index=dates, name="SPX"),
            "VIX": pd.Series(vix, index=dates, name="VIX"),
            "DXY": pd.Series(dxy, index=dates, name="DXY"),
            "COPPER": pd.Series(copper, index=dates, name="COPPER"),
        }

    def _synthetic_single(self, name: str) -> pd.Series:
        np.random.seed(hash(name) % (2**31))
        dates = pd.date_range(end=datetime.now(), periods=CONFIG.lookback_days * 2, freq="B")
        vals = np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates))))
        return pd.Series(vals * 100, index=dates, name=name)

    @staticmethod
    def align_series(data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Allinea tutte le serie sui date comuni con tolleranza."""
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # Forward-fill per gestire giorni non comuni (es. futures vs indices)
        df = df.ffill().bfill().dropna()
        if df.empty:
            # Fallback: prendi l'intersezione degli ultimi N giorni
            common = None
            for name, s in data.items():
                dates = set(s.dropna().index[-252:])
                common = dates if common is None else common & dates
            if common:
                common = sorted(common)[-252:]
                df = pd.DataFrame({k: v.reindex(common) for k, v in data.items()}).ffill().dropna()
        return df


# =============================================================================
# CORRELATION ANALYZER - Matrice di correlazione dinamica
# =============================================================================

class CorrelationAnalyzer:
    """Calcola correlazioni mobili tra i fattori macro."""

    def __init__(self, window: int = CONFIG.correlation_window):
        self.window = window

    def rolling_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Matrice di correlazione mobile tra tutti i fattori."""
        returns = df.pct_change().dropna()
        pairs = list(combinations(df.columns, 2))
        results = {}
        for a, b in pairs:
            corr = returns[a].rolling(self.window).corr(returns[b])
            results[f"{a}_{b}"] = corr
        return pd.DataFrame(results)

    def latest_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change().dropna().corr()

    def directional_bias(self, df: pd.DataFrame, target: str = "SPX") -> Bias:
        """
        Determina il bias direzionale basato sulle correlazioni dei fattori.
        - VIX sale -> bearish (correlazione negativa con SPX)
        - DXY sale -> bearish per SPX (storica correlazione negativa)
        - Copper sale -> bullish (crescita economica)
        - SPX momentum -> bias proprio
        """
        returns = df.pct_change().dropna().tail(self.window)
        latest = returns.iloc[-1] if len(returns) > 0 else pd.Series()

        if len(returns) < 20:
            return Bias.NEUTRAL

        corr = returns.corr()
        score = 0.0

        # Regole quantitative di contribuzione al bias
        if target in corr.columns and "VIX" in corr.columns:
            # Se VIX e target sono negativamente correlati e VIX sale -> negativo
            vix_corr = corr.loc[target, "VIX"]
            if "VIX" in latest and latest["VIX"] > 0:
                score -= abs(vix_corr)  # VIX sale -> bearish
            else:
                score += abs(vix_corr) * 0.5

        if target in corr.columns and "DXY" in corr.columns:
            dxy_corr = corr.loc[target, "DXY"]
            if "DXY" in latest and latest["DXY"] > 0:
                score -= abs(dxy_corr) * 0.7
            else:
                score += abs(dxy_corr) * 0.3

        if target in corr.columns and "COPPER" in corr.columns:
            copper_corr = corr.loc[target, "COPPER"]
            if "COPPER" in latest and latest["COPPER"] > 0:
                score += abs(copper_corr)
            else:
                score -= abs(copper_corr) * 0.5

        # Momentum del target stesso
        if target in returns.columns:
            ret_5d = returns[target].tail(5).mean()
            ret_21d = returns[target].tail(21).mean()
            score += (ret_5d + ret_21d) * 50  # scala

        if score > 0.5:
            return Bias.BULLISH
        elif score > 0.1:
            return Bias.BULLISH
        elif score < -0.5:
            return Bias.BEARISH
        elif score < -0.1:
            return Bias.BEARISH
        else:
            return Bias.NEUTRAL


# =============================================================================
# REGIME DETECTOR - Identificazione del regime macro
# =============================================================================

class RegimeDetector:
    """
    Identifica il regime macro GIP (Growth-Inflation-Policy).
    Basato sul framework descritto in wiki/concepts/macro-operativa.md

    Regimi:
      EXPANSION  : G↑ I↓/→   P=→    VIX<14, Copper↑, DXY↓
      SLOWDOWN   : G↓ I→     P=dovish  VIX14-25, Copper↓, WTI↓
      REFLATION  : G↑ I↑     P=hawkish VIX>20, Copper↑↑, WTI>80
      STAGFLATION: G↓ I↑     P=trapped VIX>25, WTI>85, DXY↑
    """

    def __init__(self, window: int = CONFIG.regime_window):
        self.window = window

    def _compute_trends(self, prices: pd.DataFrame) -> Dict[str, float]:
        """Pendenza normalizzata dei prezzi su finestra."""
        trends = {}
        for col in prices.columns:
            y = prices[col].values
            x = np.arange(len(y))
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                normalized = slope / (y.mean() + 1e-10) * 100
                trends[col] = round(normalized, 4)
        return trends

    def _score_growth(self, prices: pd.DataFrame, trends: Dict) -> float:
        """G > 0 = crescita in accelerazione."""
        score = 0.0
        if "SPX" in trends:
            score += np.clip(trends["SPX"] / 0.05, -1, 1)
        if "COPPER" in trends:
            score += np.clip(trends["COPPER"] / 0.08, -1, 1)
        return np.clip(score / 2, -1, 1)

    def _score_inflation(self, prices: pd.DataFrame, trends: Dict) -> float:
        """I > 0 = inflazione in salita."""
        score = 0.0
        if "COPPER" in trends:
            score += np.clip(trends["COPPER"] / 0.06, -1, 1)
        copper_level = prices["COPPER"].iloc[-1] if "COPPER" in prices else 0
        if copper_level > 4.5:
            score += 0.5
        if "WTI" in trends:
            score += np.clip(trends["WTI"] / 0.08, -1, 1) * 0.7
        wti_level = prices["WTI"].iloc[-1] if "WTI" in prices.columns else 0
        if wti_level > 80:
            score += 0.5
        elif wti_level > 90:
            score += 0.8
        if "DXY" in trends:
            dxy = trends["DXY"]
            if dxy > 0.02:
                score += 0.3
            elif dxy < -0.02:
                score -= 0.3
        return np.clip(score / 3, -1, 1)

    def detect(self, df: pd.DataFrame) -> Dict:
        """Identifica il regime GIP corrente."""
        if df.empty or len(df) < self.window:
            return {"regime": Regime.TRANSITION, "confidence": 0.0,
                    "gip": {}, "signals": {}}

        prices = df.tail(self.window)
        trends = self._compute_trends(prices)

        G = self._score_growth(prices, trends)
        I = self._score_inflation(prices, trends)

        vix_level = float(prices["VIX"].iloc[-1]) if "VIX" in prices else 20
        wti_level = float(prices["WTI"].iloc[-1]) if "WTI" in prices.columns else 75
        copper_level = float(prices["COPPER"].iloc[-1]) if "COPPER" in prices else 4.0
        dxy_level = float(prices["DXY"].iloc[-1]) if "DXY" in prices else 100

        # Regime decision
        if G > 0.3 and I < -0.2:
            regime = Regime.EXPANSION
            confidence = abs(G) + abs(I)
        elif G > 0.3 and I > 0.2:
            regime = Regime.REFLATION
            confidence = abs(G) + abs(I)
        elif G < -0.3 and I < -0.2:
            regime = Regime.SLOWDOWN
            confidence = abs(G) + abs(I)
        elif G < -0.3 and I > 0.2:
            regime = Regime.STAGFLATION
            confidence = abs(G) + abs(I)
        else:
            regimes = {
                Regime.EXPANSION: max(0, G),
                Regime.SLOWDOWN: max(0, -G) if I < 0 else 0,
                Regime.REFLATION: I if G > 0 else 0,
                Regime.STAGFLATION: I if G < 0 else 0,
            }
            regime = max(regimes, key=regimes.get)
            confidence = regimes[regime]
            if confidence < 0.25:
                regime = Regime.TRANSITION

        # VIX term structure
        vix_term = "N/D"
        if "VIX9D" in prices.columns and "VIX" in prices.columns:
            v9 = prices["VIX9D"].iloc[-1]
            vix30 = prices["VIX"].iloc[-1]
            if v9 > vix30 * 1.05:
                vix_term = "BACKWARDATION"
            elif v9 < vix30 * 0.95:
                vix_term = "CONTANGO"
            else:
                vix_term = "PIATTO"

        # SKEW
        skew_level = float(prices["SKEW"].iloc[-1]) if "SKEW" in df.columns else 0
        skew_signal = "NEUTRO"
        if skew_level > 130:
            skew_signal = "PUT_UP (institutional fear)"
        elif skew_level < 115:
            skew_signal = "RISK_ON (complacency)"

        signals = {
            "g_score": round(G, 3),
            "i_score": round(I, 3),
            "vix_level": vix_level,
            "vix_term": vix_term,
            "wti_level": wti_level,
            "copper_level": copper_level,
            "dxy_level": dxy_level,
            "skew_level": skew_level,
            "skew_signal": skew_signal,
        }

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "gip": {"G": round(G, 3), "I": round(I, 3)},
            "signals": signals,
        }

    def detect_history(self, df: pd.DataFrame) -> pd.Series:
        """Assegna regime GIP a ogni data storica tramite finestra mobile."""
        if df.empty or len(df) < self.window + 1:
            return pd.Series(dtype=str)
        regimes = []
        dates = df.index[self.window:]
        for i in range(self.window, len(df)):
            window_df = df.iloc[i - self.window : i]
            result = self.detect(window_df)
            regimes.append(result["regime"].value)
        return pd.Series(regimes, index=dates, name="regime")


# =============================================================================
# REGIME SHIFT PREDICTOR - Previsione cambi di regime
# =============================================================================

class RegimeShiftPredictor:
    """
    Prevede cambi di regime imminenti analizzando divergenze
    tra correlazioni a breve e lungo termine, e accelerazioni
    nei trend dei fattori macroeconomici.
    """

    def __init__(self, short_window: int = 10, long_window: int = CONFIG.shift_window):
        self.short_w = short_window
        self.long_w = long_window

    def predict(self, df: pd.DataFrame) -> Dict:
        """Predice se un cambio di regime e imminente."""
        if df.empty or len(df) < self.long_w + 10:
            return {"shift_probability": 0.0, "signals": [], "direction": None}

        returns = df.pct_change().dropna()
        signals = []
        shift_prob = 0.0

        # 1. Divergenza correlazioni short vs long
        for target in ["SPX"]:
            for factor in ["VIX", "DXY", "COPPER"]:
                if factor not in returns.columns or target not in returns.columns:
                    continue
                short_corr = returns[target].tail(self.short_w).corr(returns[factor].tail(self.short_w))
                long_corr = returns[target].tail(self.long_w).corr(returns[factor].tail(self.long_w))
                diff = abs(short_corr - long_corr)

                if diff > 0.3:
                    signals.append(f"Divergenza {target}-{factor}: {diff:.2f}")
                    shift_prob += 0.15

        # 2. Accelerazione VIX (shock di volatilita imminente)
        if "VIX" in returns.columns:
            vix_ret = returns["VIX"].tail(5)
            vix_accel = vix_ret.diff().mean()
            if abs(vix_accel) > 0.02:
                signals.append(f"Accelerazione VIX: {vix_accel:+.4f}")
                shift_prob += 0.12

            # VIX che sale mentre SPX sale -> divergenza pericolosa
            if "SPX" in returns.columns:
                vix_recent = returns["VIX"].tail(3).mean()
                spx_recent = returns["SPX"].tail(3).mean()
                if vix_recent > 0.01 and spx_recent > 0:
                    signals.append(f"Divergenza VIX-SPX: VIX+ SPX+")
                    shift_prob += 0.10

        # 3. Accelerazione trend Copper (ciclo economico)
        if "COPPER" in returns.columns:
            cu_short = returns["COPPER"].tail(5).mean()
            cu_long = returns["COPPER"].tail(self.long_w).mean()
            cu_accel = cu_short - cu_long
            if abs(cu_accel) > 0.005:
                signals.append(f"Ciclo Copper acceleration: {cu_accel:+.4f}")
                shift_prob += 0.10

        # 4. DXY spike
        if "DXY" in returns.columns:
            dxy_vol = returns["DXY"].tail(self.short_w).std()
            dxy_recent = abs(returns["DXY"].tail(3).mean())
            if dxy_recent > 2 * dxy_vol:
                signals.append(f"DXY spike/panic move")
                shift_prob += 0.15

        # 5. Volume di correlazioni che si rompono
        corr_df = df.pct_change().dropna().tail(self.long_w).corr()
        # Se le correlazioni storiche cambiano segno
        if "SPX" in corr_df.index and "VIX" in corr_df.columns:
            if corr_df.loc["SPX", "VIX"] > -0.2:
                signals.append(f"Corr SPX-VIX anomala: {corr_df.loc['SPX', 'VIX']:.2f}")
                shift_prob += 0.08

        shift_prob = min(shift_prob, 1.0)
        direction = None
        if shift_prob > 0.3:
            # Direzione probabile del cambio
            if "SPX" in returns.columns:
                spx_mom = returns["SPX"].tail(3).mean()
                direction = "RIBASSO" if spx_mom < 0 else "RIMBALZO"

        return {
            "shift_probability": round(shift_prob, 3),
            "signals": signals,
            "direction": direction,
        }


# =============================================================================
# STATISTICAL VALIDATOR - Significativita e robustezza
# =============================================================================

class StatisticalValidator:
    """
    Validazione statistica dei fattori macro:
      - Walk-Forward Validation: test su finestre successive
      - Bonferroni Correction: correzione per test multipli
      - P-value computation per ogni fattore
    """

    def __init__(self, n_splits: int = CONFIG.walk_forward_splits):
        self.n_splits = n_splits

    def walk_forward_validation(self, df: pd.DataFrame, target: str = "SPX") -> Dict:
        """
        Walk-forward: suddivide i dati in n_splits finestre consecutive.
        Per ogni finestra calcola la correlazione target-fattori e misura
        la stabilita dei coefficienti.
        """
        if df.empty or len(df) < 100:
            return {"stable_factors": [], "unstable_factors": [], "stability_score": 0.0}

        returns = df.pct_change().dropna()
        split_size = len(returns) // self.n_splits
        factors = [c for c in returns.columns if c != target]

        results = []
        for i in range(self.n_splits):
            start = i * split_size
            end = start + split_size if i < self.n_splits - 1 else len(returns)
            window = returns.iloc[start:end]
            if len(window) < 10:
                continue
            corrs = {}
            for f in factors:
                if f in window.columns:
                    corrs[f] = window[target].corr(window[f])
            results.append(corrs)

        if not results:
            return {"stable_factors": [], "unstable_factors": [], "stability_score": 0.0}

        # Valuta stabilita: bassa std dev tra finestre = fattore stabile
        results_df = pd.DataFrame(results)
        stability = results_df.std()

        stable = []
        unstable = []
        for factor, std_val in stability.items():
            if std_val < 0.15:
                stable.append(factor)
            else:
                unstable.append(factor)

        stability_score = (len(stable) / max(len(factors), 1)) * 100

        return {
            "stable_factors": stable,
            "unstable_factors": unstable,
            "stability_score": round(stability_score, 1),
            "fold_correlations": results_df.to_dict(),
        }

    def bonferroni_correction(self, df: pd.DataFrame, target: str = "SPX") -> Dict:
        """
        Applica la correzione di Bonferroni per test multipli.
        H0: la correlazione tra fattore e target e = 0 (nessuna relazione).
        Se p-value corretto < alpha, il fattore e statisticamente significativo.
        """
        from scipy.stats import pearsonr

        if df.empty or len(df) < 30:
            return {"significant_factors": [], "corrected_alpha": 0.0}

        returns = df.pct_change().dropna().tail(CONFIG.lookback_days)
        factors = [c for c in returns.columns if c != target]
        n_tests = len(factors)

        if n_tests == 0:
            return {"significant_factors": [], "corrected_alpha": 0.0}

        corrected_alpha = CONFIG.bonferroni_alpha / n_tests

        results = []
        for f in factors:
            clean = returns[[target, f]].dropna()
            if len(clean) < 30:
                continue
            corr, p_value = pearsonr(clean[target], clean[f])
            significant = p_value < corrected_alpha
            results.append({
                "factor": f,
                "correlation": round(corr, 4),
                "p_value": round(p_value, 6),
                "corrected_alpha": round(corrected_alpha, 6),
                "significant": significant,
                "interpretation": self._interpret_corr(corr, f),
            })

        significant_factors = [r for r in results if r["significant"]]

        return {
            "n_tests": n_tests,
            "corrected_alpha": round(corrected_alpha, 6),
            "results": results,
            "significant_factors": significant_factors,
        }

    def _interpret_corr(self, corr: float, factor: str) -> str:
        if factor == "VIX":
            if corr < -0.3:
                return "Fortemente anti-correlato (VIX sale = SPX scende)"
            elif corr < -0.1:
                return "Leggermente anti-correlato"
            else:
                return "Correlazione anormale (possibile regime shift)"
        elif factor == "DXY":
            if corr < -0.2:
                return "Dollaro forte pesa su SPX"
            elif corr > 0.1:
                return "Dollaro debole supporta SPX"
            else:
                return "Correlazione neutra"
        elif factor == "COPPER":
            if corr > 0.2:
                return "Crescita economica supporta SPX"
            elif corr < -0.1:
                return "Rame debole = warning economico"
            else:
                return "Correlazione neutra"
        return ""


# =============================================================================
# MACRO BREAKDOWN REPORT - Report Daily/Weekly
# =============================================================================

class MacroBreakdownReport:
    """Genera report strutturato del Macro Breakdown giornaliero/settimanale."""

    BAR = "=" * 78

    def generate(self, df: pd.DataFrame, regime: Dict, bias: Bias,
                 shift: Dict, validation: Dict, bonferroni: Dict,
                 correlations: pd.DataFrame, frequency: str = "DAILY") -> str:
        lines = [
            self.BAR,
            f"  MACRO BREAKDOWN — {frequency}",
            f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.BAR,
        ]

        lines += self._section_prices(df, regime.get("signals", {}))
        lines += self._section_regime(regime)
        lines += self._section_bias(bias, regime)
        lines += self._section_shift(shift)
        lines += self._section_correlations(correlations, df)
        lines += self._section_validation(bonferroni, validation)
        lines += self._section_signal_synthesis(regime, bias, shift)
        lines.append(f"\n{self.BAR}")
        lines.append("  LEGENDA: [S] Significativo  [NS] Non significativo")
        lines.append("  Fattori con significativita Bonferroni -> affidabili")
        lines.append("  Walk-forward stable -> robusti su piu finestre temporali\n")

        return "\n".join(lines)

    def _section_prices(self, df: pd.DataFrame, signals: Dict) -> List[str]:
        prices = {}
        for col in df.columns:
            if col in df.columns:
                try:
                    prices[col] = round(float(df[col].iloc[-1]), 2)
                except (ValueError, IndexError):
                    continue
        lines = [f"\n  {'PREZZI CORRENTI':^76}"]
        lines.append(f"  {'-' * 50}")
        for k, v in prices.items():
            lines.append(f"  {k:<10} {v:>12.2f}")
        return lines

    def _section_regime(self, regime: Dict) -> List[str]:
        r = regime.get("regime", Regime.TRANSITION)
        conf = regime.get("confidence", 0)
        gip = regime.get("gip", {})
        signals = regime.get("signals", {})

        lines = [f"\n  {'REGIME GIP (Growth-Inflation-Policy)':^76}"]
        lines.append(f"  {'-' * 50}")
        lines.append(f"  Regime: {r.value:<20} Confidenza: {conf}")
        lines.append(f"  G-Score (Growth):     {gip.get('G', 0):>+7.3f}")
        lines.append(f"  I-Score (Inflation):  {gip.get('I', 0):>+7.3f}")
        if "vix_level" in signals:
            v = signals["vix_level"]
            v_tag = "ALTO" if v > 20 else "BASSO" if v < 15 else "NORMALE"
            lines.append(f"  VIX: {v:.1f} ({v_tag})")
        if "vix_term" in signals:
            lines.append(f"  VIX term structure: {signals['vix_term']}")
        if "wti_level" in signals:
            wti = signals["wti_level"]
            wti_tag = ""
            if wti > 85: wti_tag = "STAGFLATION"
            elif wti > 80: wti_tag = "REFLATION"
            elif wti > 70: wti_tag = "NEUTRALE"
            else: wti_tag = "DEFLATION"
            lines.append(f"  WTI: ${wti:.1f} ({wti_tag})")
        if "copper_level" in signals:
            lines.append(f"  Copper: ${signals['copper_level']:.2f}")
        if "dxy_level" in signals:
            lines.append(f"  DXY: {signals['dxy_level']:.1f}")
        if "skew_signal" in signals:
            lines.append(f"  SKEW ({signals.get('skew_level', 0):.0f}): {signals['skew_signal']}")

        # Mappa GIP visiva
        lines.append(f"\n  Mappa GIP:")
        lines.append(f"               I↓ (defl)     I→ (neutra)     I↑ (infl)")
        lines.append(f"  G↑ (growth)  [EXPANSION]    [EXPANSION]    [REFLATION]")
        lines.append(f"  G↓ (slow)    [SLOWDOWN]     [SLOWDOWN]     [STAGFLATION]")
        g_val = gip.get("G", 0)
        i_val = gip.get("I", 0)
        g_pos = "G↑" if g_val > 0 else "G↓"
        i_pos = "I↓" if i_val < -0.1 else "I↑" if i_val > 0.1 else "I→"
        lines.append(f"  Sei qui:      {g_pos} {i_pos} → {r.value}")
        return lines

    def _section_bias(self, bias: Bias, regime: Dict) -> List[str]:
        lines = [f"\n  {'BIAS DIREZIONALE':^76}"]
        lines.append(f"  {'-' * 50}")
        r = regime.get("regime", Regime.TRANSITION)
        lines.append(f"  Bias: {bias.value}")
        if r == Regime.EXPANSION:
            lines.append(f"  Implicazione: trend rialzista, pullback comprabili")
        elif r == Regime.REFLATION:
            lines.append(f"  Implicazione: volatile, continuazione su breakout")
        elif r == Regime.SLOWDOWN:
            lines.append(f"  Implicazione: range, reversal a livelli")
        elif r == Regime.STAGFLATION:
            lines.append(f"  Implicazione: trend ribassista, no fade")
        return lines

    def _section_shift(self, shift: Dict) -> List[str]:
        prob = shift.get("shift_probability", 0) * 100
        signals = shift.get("signals", [])
        direction = shift.get("direction")

        lines = [f"\n  {'REGIME SHIFT PREDICTION':^76}"]
        lines.append(f"  {'-' * 50}")
        lines.append(f"  Probabilita shift: {prob:.1f}%")
        if direction:
            lines.append(f"  Direzione prevista: {direction}")
        if signals:
            lines.append(f"  Segnali ({len(signals)}):")
            for s in signals[:5]:
                lines.append(f"    > {s}")
        else:
            lines.append(f"  Nessun segnale di shift imminente.")
        return lines

    def _section_correlations(self, corr_matrix: pd.DataFrame, df: pd.DataFrame) -> List[str]:
        if corr_matrix.empty:
            return [f"\n  {'MATRICE CORRELAZIONI':^76}", f"  (dati insufficienti)"]
        lines = [f"\n  {'MATRICE CORRELAZIONI (rendimenti)':^76}"]
        lines.append(f"  {'-' * 50}")
        # Formatta matrice
        for idx in corr_matrix.index:
            row = []
            for col in corr_matrix.columns:
                val = corr_matrix.loc[idx, col]
                row.append(f"{val:>+7.4f}")
            lines.append(f"  {idx:<8} " + "  ".join(row))

        # Relazioni chiave
        if "SPX" in corr_matrix.index:
            lines.append(f"\n  Relazioni chiave con SPX:")
            for col in corr_matrix.columns:
                if col == "SPX":
                    continue
                v = corr_matrix.loc["SPX", col]
                label = "positiva" if v > 0 else "negativa"
                strength = "forte" if abs(v) > 0.5 else "moderata" if abs(v) > 0.3 else "debole"
                lines.append(f"    SPX-{col:<8}: {v:>+7.4f} ({strength} correlazione {label})")
        return lines

    def _section_validation(self, bonferroni: Dict, validation: Dict) -> List[str]:
        lines = [f"\n  {'VALIDAZIONE STATISTICA':^76}"]
        lines.append(f"  {'-' * 50}")

        # Bonferroni
        lines.append(f"  [Bonferroni Correction]")
        lines.append(f"  Alpha corretto: {bonferroni.get('corrected_alpha', 'N/D')}")
        lines.append(f"  Test effettuati: {bonferroni.get('n_tests', 0)}")
        sig = bonferroni.get("significant_factors", [])
        if sig:
            lines.append(f"  Fattori significativi: {len(sig)}")
            for s in sig:
                lines.append(f"    [S] {s['factor']:<8} corr={s['correlation']:>+7.4f}  "
                             f"p={s['p_value']:.6f}  {s['interpretation']}")
        else:
            lines.append(f"  Nessun fattore significativo dopo correzione.")

        # Walk-forward
        lines.append(f"\n  [Walk-Forward Validation]")
        stable = validation.get("stable_factors", [])
        unstable = validation.get("unstable_factors", [])
        score = validation.get("stability_score", 0)
        lines.append(f"  Stabilita: {score:.1f}%")
        if stable:
            lines.append(f"  Fattori STABILI: {', '.join(stable)}")
        if unstable:
            lines.append(f"  Fattori INSTABILI: {', '.join(unstable)}")

        fold_corrs = validation.get("fold_correlations", {})
        if fold_corrs:
            lines.append(f"  Correlazioni per fold:")
            for factor, corrs in fold_corrs.items():
                vals = [f"{v:>+.3f}" for v in corrs.values()]
                lines.append(f"    {factor:<8}: " + " ".join(vals))

        return lines

    def _section_signal_synthesis(self, regime: Dict, bias: Bias,
                                   shift: Dict) -> List[str]:
        lines = [f"\n  {'SINTESI OPERATIVA':^76}"]
        lines.append(f"  {'-' * 50}")

        r = regime.get("regime", Regime.TRANSITION)
        sp = shift.get("shift_probability", 0)
        gip = regime.get("gip", {})

        # Azione per regime GIP
        regime_actions = {
            Regime.EXPANSION: "LONG BIAS — pullback comprati, range/fade funzionano",
            Regime.REFLATION: "CONTINUATION BIAS — breakout/continuation, no fade",
            Regime.SLOWDOWN: "RANGE BIAS — reversal a livelli, size ridotto",
            Regime.STAGFLATION: "NO FADE — solo trend intraday, size ridotto",
            Regime.TRANSITION: "ATTESA — regime non chiaro, skip o size minima",
        }
        action = regime_actions.get(r, "ATTESA / NEUTRALE")

        lines.append(f"  Azione: {action}")
        lines.append(f"  Regime GIP: {r.value} | Bias: {bias.value} | Shift: {sp*100:.0f}%")

        # Regole di overriding dalla macro-operativa
        signals = regime.get("signals", {})
        vix = signals.get("vix_level", 0)
        wti = signals.get("wti_level", 0)
        skew = signals.get("skew_level", 0)

        if vix > 25:
            lines.append(f"  ⚠  VIX > 25: GEX rotto, non basare decisioni su gamma")
        if wti > 85 and gip.get("G", 0) < 0:
            lines.append(f"  ⚠  WTI > $85 con G negativo: bias short prioritario")
        if skew > 130:
            lines.append(f"  ⚠  SKEW > 130: institutional fear, reversal piu affidabili")
        elif skew < 115:
            lines.append(f"  ⚠  SKEW < 115: risk-on complacency, continuazioni")
        if sp > 0.5:
            lines.append(f"  ⚠  ALTA PROBABILITA' DI REGIME SHIFT — preparare piano di contingenza")
        elif sp > 0.3:
            lines.append(f"  Attenzione: possibili cambi di correlazione nei prossimi giorni")

        return lines


# =============================================================================
# VISUALIZER — Grafici interattivi (Heatmap + 3D Regime Space)
# =============================================================================

class Visualizer:
    """
    Genera grafici interattivi salvati come HTML:
      - Heatmap delle correlazioni macro
      - Scatter 3D del Regime Space (SPX x VIX x DXY)
    """

    REGIME_COLORS = {
        "EXPANSION": "green",
        "SLOWDOWN": "blue",
        "REFLATION": "orange",
        "STAGFLATION": "red",
        "TRANSIZIONE": "gray",
    }

    def __init__(self, output_dir: str = CONFIG.output_dir):
        self.output_dir = output_dir

    def _ensure_dir(self):
        import os
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # -----------------------------------------------------------------
    # 1. HEATMAP interattiva (correlazioni correnti)
    # -----------------------------------------------------------------
    def heatmap(self, corr_matrix: pd.DataFrame, title: str = "Macro Correlation Matrix") -> str:
        """
        Genera una heatmap interattiva della matrice di correlazione.
        Ritorna il path del file HTML salvato.
        """
        import plotly.graph_objects as go

        self._ensure_dir()
        filepath = f"{self.output_dir}/heatmap_{datetime.now():%Y%m%d_%H%M}.html"

        labels = corr_matrix.columns.tolist()
        z = corr_matrix.values

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdYlGn",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{v:+.3f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate="%{x} vs %{y}<br>Correlazione: %{z:+.3f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            width=700,
            height=600,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
        )

        fig.write_html(filepath)
        log.info(f"Heatmap salvata: {filepath}")
        return filepath

    # -----------------------------------------------------------------
    # 2. HEATMAP EVOLUTIVA (correlazioni rolling nel tempo)
    # -----------------------------------------------------------------
    def rolling_heatmap(self, df: pd.DataFrame, window: int = 63) -> str:
        """
        Heatmap dell'evoluzione temporale delle correlazioni SPX-fattori.
        Asse X = date, Asse Y = fattori, Colore = correlazione rolling.
        """
        import plotly.graph_objects as go

        self._ensure_dir()
        filepath = f"{self.output_dir}/rolling_corr_{datetime.now():%Y%m%d_%H%M}.html"

        returns = df.pct_change().dropna()
        factors = [c for c in returns.columns if c != "SPX"]
        if not factors:
            return ""

        rolling_data = {}
        for f in factors:
            rolling_data[f] = returns["SPX"].rolling(window).corr(returns[f])

        roll_df = pd.DataFrame(rolling_data).dropna()

        fig = go.Figure(data=go.Heatmap(
            z=roll_df.T.values,
            x=roll_df.index.tolist(),
            y=factors,
            colorscale="RdYlGn",
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate="Data: %{x}<br>Fattore: %{y}<br>Corr: %{z:+.3f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=f"Rolling Correlation SPX vs Fattori (window={window}gg)", x=0.5),
            width=1000,
            height=450,
            xaxis=dict(title="Data"),
            yaxis=dict(title="Fattore", autorange="reversed"),
        )

        fig.write_html(filepath)
        log.info(f"Rolling heatmap salvata: {filepath}")
        return filepath

    # -----------------------------------------------------------------
    # 3. GRAFICO 3D — Regime Space
    # -----------------------------------------------------------------
    def regime_3d(self, df: pd.DataFrame, regime_series: pd.Series,
                  x_col: str = "SPX", y_col: str = "VIX", z_col: str = "DXY") -> str:
        """
        Scatter plot 3D interattivo: ogni punto = giorno storico,
        colorato per regime macro. Esporta HTML.
        """
        import plotly.graph_objects as go
        import numpy as np

        self._ensure_dir()
        filepath = f"{self.output_dir}/regime_3d_{datetime.now():%Y%m%d_%H%M}.html"

        plot_data = df[[x_col, y_col, z_col]].loc[regime_series.index].copy()
        plot_data["regime"] = regime_series

        if plot_data.empty:
            log.warning("Nessun dato per grafico 3D")
            return ""

        # Colori per regime
        unique_regimes = plot_data["regime"].unique()
        color_map = {r: self.REGIME_COLORS.get(r, "gray") for r in unique_regimes}

        fig = go.Figure()

        for regime_name in unique_regimes:
            mask = plot_data["regime"] == regime_name
            subset = plot_data[mask]

            # Campionamento per performance
            if len(subset) > 500:
                subset = subset.sample(500, random_state=42)

            fig.add_trace(go.Scatter3d(
                x=subset[x_col],
                y=subset[y_col],
                z=subset[z_col],
                mode="markers",
                name=regime_name,
                marker=dict(
                    size=4,
                    color=color_map[regime_name],
                    opacity=0.7,
                ),
                hovertemplate=(
                    f"{x_col}: %{{x:.1f}}<br>"
                    f"{y_col}: %{{y:.1f}}<br>"
                    f"{z_col}: %{{z:.1f}}<br>"
                    f"Regime: {regime_name}<extra></extra>"
                ),
            ))

        # Punto corrente evidenziato
        latest = plot_data.iloc[-1:]
        latest_regime = latest["regime"].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=latest[x_col],
            y=latest[y_col],
            z=latest[z_col],
            mode="markers",
            name=f"CORRENTE ({latest_regime})",
            marker=dict(size=12, color="yellow", symbol="diamond",
                        line=dict(width=2, color="black")),
            hovertemplate=(
                f"<b>CORRENTE</b><br>"
                f"{x_col}: %{{x:.1f}}<br>"
                f"{y_col}: %{{y:.1f}}<br>"
                f"{z_col}: %{{z:.1f}}<br>"
                f"Regime: {latest_regime}<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=dict(text="3D Macro Regime Space", x=0.5, font=dict(size=18)),
            scene=dict(
                xaxis=dict(title=x_col),
                yaxis=dict(title=y_col),
                zaxis=dict(title=z_col),
            ),
            width=900,
            height=700,
            legend=dict(title="Regime", x=0.85, y=0.95),
        )

        fig.write_html(filepath)
        log.info(f"3D Regime plot salvato: {filepath}")
        return filepath

    # -----------------------------------------------------------------
    # 4. Z-Score heatmap storica
    # -----------------------------------------------------------------
    def zscore_heatmap(self, df: pd.DataFrame, window: int = 252) -> str:
        """
        Heatmap degli Z-Score degli asset nel tempo.
        Mostra quando un asset e' 'anomalo' rispetto alla sua media storica.
        """
        import plotly.graph_objects as go

        self._ensure_dir()
        filepath = f"{self.output_dir}/zscore_{datetime.now():%Y%m%d_%H%M}.html"

        zscores = df.apply(lambda x: (x - x.rolling(window).mean()) / x.rolling(window).std())
        zscores = zscores.dropna()

        if zscores.empty:
            return ""

        fig = go.Figure(data=go.Heatmap(
            z=zscores.T.values,
            x=zscores.index.tolist(),
            y=zscores.columns.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            zmin=-3,
            zmax=3,
            hovertemplate="Data: %{x}<br>Asset: %{y}<br>Z-Score: %{z:+.2f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=f"Z-Score Macro Asset (window={window}gg)", x=0.5),
            width=1000,
            height=400,
        )

        fig.write_html(filepath)
        log.info(f"Z-Score heatmap salvata: {filepath}")
        return filepath


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class MacroQuantSystem:
    """Orchestratore principale del sistema Quant-Macro."""

    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.corr_analyzer = CorrelationAnalyzer()
        self.regime_detector = RegimeDetector()
        self.shift_predictor = RegimeShiftPredictor()
        self.validator = StatisticalValidator()
        self.reporter = MacroBreakdownReport()
        self.visualizer = Visualizer()

    def run(self, frequency: str = "DAILY") -> Tuple[str, Dict, Dict, Bias, Dict]:
        log.info(f"=== Macro Quant Breakdown ({frequency}) ===")

        # 1. Fetch data
        raw = self.data_fetcher.fetch_all()
        df = DataFetcher.align_series(raw)
        log.info(f"Dati allineati: {len(df)} giorni, {list(df.columns)}")

        # 2. Correlazioni
        corr_matrix = self.corr_analyzer.latest_correlation_matrix(df)
        rolling_corrs = self.corr_analyzer.rolling_correlations(df)

        # 3. Regime detection
        regime = self.regime_detector.detect(df)
        log.info(f"Regime: {regime['regime'].value} (conf: {regime['confidence']})")

        # 4. Direzional bias
        bias = self.corr_analyzer.directional_bias(df, target="SPX")
        log.info(f"Bias: {bias.value}")

        # 5. Regime shift prediction
        shift = self.shift_predictor.predict(df)
        log.info(f"Shift prob: {shift['shift_probability']:.1%}")

        # 6. Walk-forward validation
        validation = self.validator.walk_forward_validation(df)
        log.info(f"Walk-forward stabilita: {validation['stability_score']:.1f}%")

        # 7. Bonferroni correction
        bonferroni = self.validator.bonferroni_correction(df)
        n_sig = len(bonferroni.get("significant_factors", []))
        log.info(f"Bonferroni: {n_sig} fattori significativi")

        # 8. Report testuale
        report = self.reporter.generate(
            df=df,
            regime=regime,
            bias=bias,
            shift=shift,
            validation=validation,
            bonferroni=bonferroni,
            correlations=corr_matrix,
            frequency=frequency,
        )

        # 9. Grafici interattivi (salvati come HTML)
        charts = {}

        charts["heatmap"] = self.visualizer.heatmap(corr_matrix)

        if len(df) > 100:
            charts["rolling_heatmap"] = self.visualizer.rolling_heatmap(df)

        if len(df) > CONFIG.regime_window + 10:
            regime_hist = self.regime_detector.detect_history(df)
            if not regime_hist.empty:
                charts["regime_3d"] = self.visualizer.regime_3d(df, regime_hist)

        if len(df) > 252:
            charts["zscore"] = self.visualizer.zscore_heatmap(df)

        log.info(f"Grafici generati: {len(charts)}")

        return report, charts, regime, bias, shift

    def write_signal(self, regime: Dict, bias: Bias, shift: Dict,
                     df: pd.DataFrame, filepath: str = None) -> str:
        """Scrive segnale compatto per il sistema trading (last_signal.txt)."""
        fp = filepath or CONFIG.signal_file
        signals = regime.get("signals", {})
        gip = regime.get("gip", {})

        lines = [
            f"MACRO SIGNAL — {datetime.now():%Y-%m-%d %H:%M}",
            f"REGIME: {regime['regime'].value}",
            f"BIAS: {bias.value}",
            f"G_SCORE: {gip.get('G', 0):+.3f}",
            f"I_SCORE: {gip.get('I', 0):+.3f}",
            f"VIX: {signals.get('vix_level', 0):.1f}",
            f"VIX_TERM: {signals.get('vix_term', 'N/D')}",
            f"WTI: ${signals.get('wti_level', 0):.1f}",
            f"COPPER: ${signals.get('copper_level', 0):.2f}",
            f"DXY: {signals.get('dxy_level', 0):.1f}",
            f"SKEW: {signals.get('skew_level', 0):.0f}",
            f"SHIFT_PROB: {shift.get('shift_probability', 0)*100:.0f}%",
            f"CONFIDENCE: {regime.get('confidence', 0):.1f}",
        ]

        # Azione sintetica
        r = regime["regime"]
        sp = shift.get("shift_probability", 0)
        action_map = {
            Regime.EXPANSION: "LONG",
            Regime.REFLATION: "LONG_CAUTIOUS",
            Regime.SLOWDOWN: "NEUTRAL_RANGE",
            Regime.STAGFLATION: "SHORT",
            Regime.TRANSITION: "WAIT",
        }
        lines.append(f"ACTION: {action_map.get(r, 'WAIT')}")

        if sp > 0.5:
            lines.append("ALERT: REGIME_SHIFT_HIGH_PROBABILITY")
        elif sp > 0.3:
            lines.append("ALERT: REGIME_SHIFT_WATCH")

        text = "\n".join(lines) + "\n"

        import os
        os.makedirs(os.path.dirname(fp) if os.path.dirname(fp) else ".", exist_ok=True)
        with open(fp, "w") as f:
            f.write(text)
        log.info(f"Segnale scritto: {fp}")
        return fp


# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys, time

    frequency = "DAILY"
    skip_charts = False
    open_charts = False
    write_signal = False
    intraday = False

    for a in sys.argv[1:]:
        a_upper = a.upper()
        if a_upper in ("DAILY", "WEEKLY"):
            frequency = a_upper
        elif a == "--no-charts":
            skip_charts = True
        elif a == "--open":
            open_charts = True
        elif a == "--signal":
            write_signal = True
        elif a == "--intraday":
            intraday = True

    system = MacroQuantSystem()

    def run_once(signal_path=None):
        report, charts, regime, bias, shift = system.run(frequency=frequency)
        print(report)

        if signal_path or write_signal:
            system.write_signal(regime, bias, shift, None,
                                filepath=signal_path)

        if charts and not skip_charts:
            print(f"\n  Grafici interattivi generati ({len(charts)}):")
            for name, path in charts.items():
                print(f"    {name:<20} {path}")
            if open_charts:
                import subprocess, os
                print(f"\n  Apro {len(charts)} grafici nel browser...")
                for name, path in charts.items():
                    abs_path = os.path.abspath(path)
                    if not os.path.exists(abs_path):
                        continue
                    subprocess.run(f'start "" "{abs_path}"', shell=True,
                                   capture_output=True)
        return regime

    if intraday:
        import os as _os
        log_dir = _os.path.dirname(CONFIG.signal_file) if _os.path.dirname(CONFIG.signal_file) else "."
        _os.makedirs(log_dir, exist_ok=True)
        log_file = _os.path.join(log_dir, "scanner_log.txt")

        print(f"\n  Intraday mode — refresh ogni {CONFIG.intraday_interval_min} min")
        print(f"  Signal: {_os.path.abspath(CONFIG.signal_file)}")
        print(f"  Log:    {_os.path.abspath(log_file)}")
        print("  Ctrl+C per fermare\n")

        while True:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*78}")
            print(f"  Scan @ {ts}")
            print(f"{'='*78}")

            try:
                regime_out = run_once(signal_path=CONFIG.signal_file)
                with open(log_file, "a") as lf:
                    lf.write(f"[{ts}] Regime: {regime_out['regime'].value} | "
                             f"G: {regime_out['gip']['G']:+.3f} I: {regime_out['gip']['I']:+.3f}\n")
            except Exception as e:
                log.error(f"Scan error: {e}")
                with open(log_file, "a") as lf:
                    lf.write(f"[{ts}] ERROR: {e}\n")

            print(f"\n  Prossimo scan tra {CONFIG.intraday_interval_min} min...")
            time.sleep(CONFIG.intraday_interval_min * 60)
    else:
        run_once()


if __name__ == "__main__":
    main()
