# MAZE CAPITAL — Pre-Market Roadmap

## ═══ Setup ═══
- [ ] Ticker: SPY / QQQ / SPX
- [ ] DTE: 0 (oggi) per intraday, 7-14 per struttura
- [ ] Spot attuale: _____
- [ ] VIX: _____
- [ ] Frequenza: DAILY / WEEKLY

---

## ═══ 1. Macroscenario (Banner) ═══
- [ ] Regime: G+/I+ | G-/I+ | G+/I- | G-/I- (crescita/inflazione)
- [ ] Confidence: >0.6 = segnale affidabile
- [ ] Bias: bullish / bearish / neutral
- [ ] VIX < 15 = risk-on, 15-22 = normale, >22 = paura

**Azione**: bias deve allinearsi al setup tecnico. Se conflitto = no trade.

---

## ═══ 2. GEX (Gamma Exposure) ═══
- [ ] Zero Gamma level: _____
- [ ] Spot vs ZG: sopra = resistenza gamma / sotto = supporto
- [ ] Total GEX: positivo = mkt short gamma (range) / negativo = mkt long gamma (trend)
- [ ] Max GEX strike (bull wall): _____
- [ ] Min GEX strike (bear wall): _____

**Regola**: Spot lontano da ZG (>0.5%) = direzionale. Spot su ZG = pinning.

---

## ═══ 3. DEX (Delta Exposure) ═══
- [ ] Call wall: _____
- [ ] Put wall: _____
- [ ] Net DEX: positivo = call heavy / negativo = put heavy

**Regola**: Call wall sopra spot = resistenza. Put wall sotto spot = supporto. Più sono vicini e forti, più il prezzo rimane intrappolato.

---

## ═══ 4. DOM Bookmap ═══
- [ ] Big trade cluster sopra spot: _____
- [ ] Big trade cluster sotto spot: _____
- [ ] Liquidità concentrata ATM? sì/no

**Regola**: Cluster di grandi dimensioni = magnet. Prezzo tende a tornare.

---

## ═══ 5. OI Profile + Premium ═══
- [ ] Max OI strike (call): _____  (put): _____
- [ ] P/C Premium Ratio: >1.5 paura / <0.7 euforia
- [ ] Put premium totale: $_____  Call premium totale: $_____

**Regola**: P/C ratio alto + put wall sotto spot = hedging in corso → bias ribassista.

---

## ═══ 6. Gamma Bands ═══
- [ ] Spot in zona: ▲ rottura / ▲ resistenza / ▲ attrito / ● SPOT / ▼ attrito / ▼ supporto / ▼ rottura
- [ ] Banda più vicina sopra: _____
- [ ] Banda più vicina sotto: _____

**Regola**: In zona rottura (±1.5-2%) = extension trade (mean reversion). In attrito (±0.15-0.35%) = range, meglio non fare.

---

## ═══ 7. Regression + Volume Profile ═══
- [ ] Slope 89d: positiva / negativa / piatta
- [ ] R²: >0.8 = trend forte / <0.5 = range
- [ ] Prezzo vs sigma: +1σ / -1σ / +2σ / -2σ / in range
- [ ] POC (fair value): _____
- [ ] Value Area: _____ (val) – _____ (vah)
- [ ] Spot vs VA: sopra = esteso / dentro = fair / sotto = ipervenduto

**Regola**: +2σ o -2σ = mean reversion. Spot fuori VA con slope forte = trend, non reversal.

---

## ═══ 8. GEX Centroid (Term Structure) ═══
- [ ] Centroidi aumentano con DTE? = gamma rialzista
- [ ] Centroidi diminuiscono con DTE? = gamma ribassista
- [ ] Tutti allineati sopra/sotto spot? = direzionale

**Regola**: Se tutti i centroidi > spot = call gamma domina ovunque → bullish. Se misti = range.

---

## ═══ 9. PDF + Walls ═══
- [ ] Expected Value vs Spot: EV _____ > spot? = bias rialzista implicito
- [ ] Code della distribuzione: simmetrica / skew rialzista / skew ribassista
- [ ] Put wall in corrispondenza di supporto PDF?

---

## ═══ 10. Options Chain ═══
- [ ] IV ATM: _____%
- [ ] Skew: put IV vs call IV a parità di distanza
- [ ] Expected Move: ±$_____ (±____%)

**Regola**: Skew put > call = paura. Se IV ATM > 30% su SPY = spike in corso.

---

## ═══ Score Summary ═══
| Fattore | Bullish | Bearish | Neutro |
|---------|---------|---------|--------|
| GEX     | □       | □       | □      |
| DEX     | □       | □       | □      |
| GEX Centroid | □ | □       | □      |
| Gamma Bands | □ | □      | □      |
| Regression  | □ | □      | □      |
| Premium | □       | □       | □      |
| VIX/Regime | □ | □       | □      |

**Verde**: ____ / **Rosso**: ____ / **Neutro**: ____

---

## ═══ Trade Plan ═══
- [ ] Direzione: LONG / SHORT / HEDGE / NO TRADE
- [ ] Zona d'ingresso: _____
- [ ] Stop: _____ (oltre _____strike/sigma)
- [ ] Target 1: _____ (ZG / wall / POC / VA limite)
- [ ] Target 2: _____ (max GEX / 2σ)
- [ ] Dimensione: _____%
- [ ] DTE opzioni: _____
- [ ] Struttura: call/put / spread / ratio

---

*Generato da MAZE CAPITAL Terminal v4.0*
