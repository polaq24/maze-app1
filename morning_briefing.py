"""
MAZE CAPITAL — Morning Briefing v1.0
=====================================
Scheduled daily PDF report at 15:00 IT (09:00 ET).
Sends institutional-grade market analysis via Gmail.

Dependencies: reportlab, yfinance, numpy, pandas, schedule
Install: pip install reportlab yfinance schedule
"""

import sys, os, json, sqlite3, smtplib, io, re
from pathlib import Path
from datetime import datetime, timedelta, date
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

import numpy as np
import pandas as pd
import yfinance as yf

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER

sys.path.insert(0, str(Path(__file__).parent))
from quant_analytics import gex_regime_heatmap, gamma_bands, expected_move

# ── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    "email_from": "",
    "email_to":   "",
    "gmail_app_password": "",
    "ticker": "QQQ",
    "db_path": str(Path(__file__).parent / "macro_history.db"),
    "vault_path": str(Path(__file__).parent.parent),
    "send_at": "15:00",
    "timezone": "Europe/Rome",
}

CONFIG_PATH = Path(__file__).parent / "config_briefing.json"
if CONFIG_PATH.exists():
    CONFIG.update(json.loads(CONFIG_PATH.read_text()))

# ── COLORI REPORT ────────────────────────────────────────────────────────────
C_BG      = "#0a0e17"
C_CARD    = "#111827"
C_BORDER  = "#1f2937"
C_ACCENT  = "#00d2ff"
C_GREEN   = "#22c55e"
C_RED     = "#ef4444"
C_PURPLE  = "#a855f7"
C_GOLD    = "#fbbf24"
C_TEXT     = "#e2e8f0"
C_TEXT_M  = "#64748b"

# =============================================================================
# DATA COLLECTOR
# =============================================================================

class DataCollector:
    """Raccoglie tutti i dati per il briefing mattutino."""

    def __init__(self, ticker="QQQ"):
        self.ticker = ticker
        self.db = CONFIG["db_path"]
        self.data = {}

    def run(self) -> dict:
        self._macro_db()
        self._overnight_futures()
        self._asia_europe()
        self._macro_spot()
        self._economic_calendar()
        self._vault_notes()
        return self.data

    def _macro_db(self):
        """Legge macro_history.db + eod_snapshots.

        I dati GEX (γ-ENV, ZG, GEX totale, VVR, CW, PW, Expected Move)
        vengono presi da eod_snapshots (ultimo snapshot della sessione).
        I dati macro (regime, g_score, i_score, shift_prob, vix) da macro_history.
        """
        self.data["macro"] = None
        if not os.path.exists(self.db):
            return
        try:
            conn = sqlite3.connect(self.db)

            # 1. macro_history
            df = pd.read_sql("SELECT * FROM macro_history ORDER BY timestamp DESC LIMIT 10", conn)
            if not df.empty:
                self.data["macro"] = df
                last = df.iloc[0].to_dict()
                self.data["macro_last"] = last
                self.data["macro_prev"] = df.iloc[1].to_dict() if len(df) > 1 else None
            else:
                last = {}

            # 2. eod_snapshots (sempre, anche se macro_history è vuoto)
            snap = pd.read_sql(
                "SELECT * FROM eod_snapshots WHERE ticker=? ORDER BY date DESC LIMIT 1",
                conn, params=(self.ticker,))
            if not snap.empty:
                s = snap.iloc[0].to_dict()
                self.data["snapshot_date"] = s.get("date", "")
                # Se macro_history esisteva, sovrascrivi campi GEX
                if last:
                    if s.get("gamma_env"): last["gex_env"] = s["gamma_env"]
                    if s.get("zero_gamma") is not None: last["gex_zg"] = s["zero_gamma"]
                    if s.get("total_gex") is not None: last["gex_total"] = s["total_gex"]
                    if s.get("vvr") is not None: last["vvr"] = s["vvr"]
                    if s.get("skew") is not None: last["skew"] = s["skew"]
                    if s.get("spot") is not None: last["spot"] = s["spot"]
                    if s.get("call_wall") is not None: last["call_wall"] = s["call_wall"]
                    if s.get("put_wall") is not None: last["put_wall"] = s["put_wall"]
                    if s.get("em_up") is not None: last["em_up"] = s["em_up"]
                    if s.get("em_down") is not None: last["em_down"] = s["em_down"]
                    if s.get("regime") and last.get("regime") in (None, "TRANSIZIONE"):
                        for k in ("regime", "g_score", "i_score", "shift_prob", "vix"):
                            if s.get(k) is not None: last[k] = s[k]
                else:
                    # macro_history vuoto → costruiamo macro_last dallo snapshot
                    self.data["macro_last"] = {
                        "gex_env": s.get("gamma_env", "—"),
                        "gex_zg": s.get("zero_gamma", 0),
                        "gex_total": s.get("total_gex", 0),
                        "vvr": s.get("vvr", 0),
                        "skew": s.get("skew", 0),
                        "spot": s.get("spot", 0),
                        "call_wall": s.get("call_wall"),
                        "put_wall": s.get("put_wall"),
                        "em_up": s.get("em_up"),
                        "em_down": s.get("em_down"),
                        "regime": s.get("regime", ""),
                        "g_score": s.get("g_score", 0),
                        "i_score": s.get("i_score", 0),
                        "shift_prob": s.get("shift_prob", 0),
                        "vix": s.get("vix", 0),
                    }

            conn.close()
        except Exception as e:
            self.data["macro_error"] = str(e)

    def _overnight_futures(self):
        """Futures US variazione % overnight via yfinance."""
        futures = {"ES=F": "S&P 500", "NQ=F": "Nasdaq", "YM=F": "Dow", "CL=F": "Oil", "GC=F": "Gold"}
        results = {}
        for sym, name in futures.items():
            try:
                h = yf.Ticker(sym).history(period="5d", interval="1d")
                if len(h) >= 2:
                    close_yest = h["Close"].iloc[-2]
                    close_today = h["Close"].iloc[-1]
                    chg = ((close_today / close_yest) - 1) * 100
                    results[name] = {"price": round(close_today, 2), "chg_pct": round(chg, 2)}
            except Exception:
                pass
        self.data["futures"] = results

    def _asia_europe(self):
        """Indici Asia chiusura + Europa apertura."""
        indices = {
            "^N225": "Nikkei 225", "^HSI": "Hang Seng", "000001.SS": "Shanghai",
            "^GDAXI": "DAX", "^FTSE": "FTSE 100", "^FCHI": "CAC 40",
        }
        results = {}
        for sym, name in indices.items():
            try:
                h = yf.Ticker(sym).history(period="3d", interval="1d")
                if len(h) >= 2:
                    chg = ((h["Close"].iloc[-1] / h["Close"].iloc[-2]) - 1) * 100
                    results[name] = {"price": round(h["Close"].iloc[-1], 2), "chg_pct": round(chg, 2)}
            except Exception:
                pass
        self.data["indices"] = results

    def _macro_spot(self):
        """VIX, DXY, Copper spot."""
        for sym, name in [("^VIX", "VIX"), ("DX-Y.NYB", "DXY"), ("HG=F", "Copper")]:
            try:
                h = yf.Ticker(sym).history(period="2d", interval="1d")
                if not h.empty:
                    self.data[name.lower()] = round(h["Close"].iloc[-1], 2)
            except Exception:
                pass

    def _economic_calendar(self):
        """Calendario economico del giorno."""
        today_str = date.today().isoformat()
        try:
            import requests
            r = requests.get(
                f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today_str}&to={today_str}",
                timeout=5,
            )
            if r.ok:
                events = r.json()
                self.data["calendar"] = [
                    e for e in events
                    if e.get("impact", "").lower() in ("high", "medium")
                ][:8]
            else:
                self.data["calendar"] = []
        except Exception:
            self.data["calendar"] = []

    def _vault_notes(self):
        """Cerca playbook nel vault che matchano il regime corrente.

        Cerca nella cartella wiki/playbooks/ file con sezione
        "Regime adatto" che menzionano il regime attuale. Estrae
        Trigger, Entry, Stop, Target. Include anche checklist
        premarket e, come fallback, note keyword-match.
        """
        vault = Path(CONFIG["vault_path"])
        regime = ""
        if self.data.get("macro_last"):
            regime = str(self.data["macro_last"].get("regime", "")).upper()
        notes = []
        try:
            playbooks_dir = vault / "wiki" / "playbooks"
            if playbooks_dir.exists():
                for pb_file in sorted(playbooks_dir.glob("*.md")):
                    text = pb_file.read_text(encoding="utf-8", errors="ignore")
                    # Salta se non menziona il regime corrente in Regime adatto
                    regime_section = ""
                    if "regime adatto" in text.lower():
                        # Estrai sezione dopo "Regime adatto"
                        parts = text.lower().split("regime adatto")
                        if len(parts) > 1:
                            regime_section = parts[1].split("##")[0].strip()[:300]
                    if regime and regime not in ("TRANSIZIONE", "") and regime.lower() not in regime_section.lower():
                        # Fallback: controlla se il file menziona il regime ovunque
                        if regime.lower() not in text.lower():
                            continue

                    # Estratto Trigger / Entry / Stop / Target
                    title = ""
                    for line in text.split("\n"):
                        if line.startswith("title:"):
                            title = line.split(":", 1)[1].strip().strip('"').strip("'")
                            break
                    if not title:
                        title = pb_file.stem.replace("-", " ").title()

                    def _sec(name):
                        """Estrae testo di una sezione dopo ## Name fino alla prossima ##."""
                        pat = f"## {name}"
                        if pat in text:
                            after = text.split(pat, 1)[1]
                            # Prendi fino al prossimo ## o fine file
                            return after.split("##")[0].strip()[:200]
                        return ""

                    trigger = _sec("Trigger")
                    entry = _sec("Entry")
                    stop = _sec("Stop")
                    target = _sec("Target")
                    regime_ok = _sec("Regime adatto")[:120]

                    notes.append({
                        "title": title,
                        "regime_match": regime_ok,
                        "trigger": trigger,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "source": "playbook",
                    })

            # Premarket checklist
            checklist_path = vault / "wiki" / "checklists" / "premarket-scalping-checklist.md"
            if checklist_path.exists():
                ck_text = checklist_path.read_text(encoding="utf-8", errors="ignore")
                ck_lines = []
                for line in ck_text.split("\n"):
                    stripped = line.strip()
                    if stripped and not stripped.startswith("---") and not stripped.startswith("title:") \
                       and not stripped.startswith("type:") and not stripped.startswith("tags:") \
                       and not stripped.startswith("created:") and not stripped.startswith("updated:") \
                       and not stripped.startswith("sources:"):
                        ck_lines.append(stripped)
                notes.append({
                    "title": "Premarket Scalping Checklist",
                    "regime_match": "",
                    "trigger": "",
                    "entry": "",
                    "stop": "",
                    "target": "",
                    "checklist": "\n".join(ck_lines[:30])[:600],
                    "source": "checklist",
                })

            # Keyword fallback per altre note rilevanti
            keywords = ["regime", "volatility", "gamma", "market", "today", "setup",
                         "scalping", "trend", "range", "vwap", "profile"]
            found_count = 0
            for md_file in vault.rglob("*.md"):
                if ".claude" in str(md_file) or ".codex" in str(md_file):
                    continue
                rel = str(md_file.relative_to(vault))
                if rel.startswith("wiki/playbooks") or rel.startswith("wiki/checklists") \
                   or rel.startswith(".obsidian") or rel.startswith(".codex"):
                    continue
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                found_kw = [kw for kw in keywords if kw.lower() in text.lower()]
                if len(found_kw) >= 2 and len(text) < 5000:
                    snippet_lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("#")]
                    snippet = " ".join(snippet_lines[:5])[:300]
                    title = md_file.stem.replace("-", " ").title()
                    notes.append({
                        "title": title,
                        "regime_match": "",
                        "trigger": "",
                        "entry": "",
                        "stop": "",
                        "target": "",
                        "snippet": snippet,
                        "tags": found_kw,
                        "source": "keyword",
                    })
                    found_count += 1
                    if found_count >= 3:
                        break

            self.data["vault_notes"] = notes[:8]
        except Exception as e:
            self.data["vault_notes"] = []


# =============================================================================
# TYPE OF DAY CLASSIFICATION (framework GURUFLIX)
# =============================================================================

def classify_type_of_day(gex_env, vvr, gex_total, regime, shift_prob, cw, pw, zg):
    """Classifica la giornata secondo framework GURUFLIX."""
    vvr = vvr or 0
    gex_total = gex_total or 0
    gex_pos = gex_total > 0
    gex_neg = gex_total < 0
    vvr_compresso = vvr < 4.5
    vvr_normale = 4.5 <= vvr <= 6.5
    vvr_elevato = vvr > 6.5
    shift_alto = (shift_prob or 0) > 0.5

    if gex_pos and vvr_compresso:
        return {
            "type": "ROTAZIONE",
            "color": "#fbbf24",
            "reason": f"γ-ENV POS + VVR {vvr:.1f} compresso + OI concentrato → range day likelihood alta",
            "setups_seek": "Fake Breakout (fade al livello), Hide Behind Elephant (absorption trade)",
            "setups_avoid": "Momentum Breakout, Trend Extension (falsa rottura probabile)",
            "key_levels": f"CW {cw:,.0f}, ZG {zg:,.0f}" if cw else f"ZG {zg:,.0f}",
        }
    elif gex_pos and vvr_normale:
        return {
            "type": "TREND BILANCIATO",
            "color": "#22c55e",
            "reason": f"γ-ENV POS + VVR {vvr:.1f} normale → dealer assorbono, trend ordinato",
            "setups_seek": "VWAP Fade, Second Chance Breakout Retest, Mean Reversion in VA",
            "setups_avoid": "Breakout Momentum su estensioni (gamma POS assorbe spinta)",
            "key_levels": f"CW {cw:,.0f}, ZG {zg:,.0f}, PW {pw:,.0f}" if cw and pw else f"ZG {zg:,.0f}",
        }
    elif gex_pos and vvr_elevato:
        return {
            "type": "VOLATILE CONTROLLATA",
            "color": "#22c55e",
            "reason": f"γ-ENV POS + VVR {vvr:.1f} elevato → volatilità ma assorbita dai dealer",
            "setups_seek": "CW Rejection, Elephant fade a supporto, Gamma Scalp",
            "setups_avoid": "Breakout Momentum senza assorbimento confermato",
            "key_levels": f"CW {cw:,.0f}" if cw else f"ZG {zg:,.0f}",
        }
    elif gex_neg and vvr_compresso:
        return {
            "type": "COMPRESSIONE NEGATIVA",
            "color": "#a855f7",
            "reason": f"γ-ENV NEG + VVR {vvr:.1f} compresso → accumulo prima di espansione",
            "setups_seek": "Breakout Momentum su rottura ZG, PW Sweep",
            "setups_avoid": "Fade, Reversal trading (gamma NEG amplifica)",
            "key_levels": f"PW {pw:,.0f}, ZG {zg:,.0f}" if pw else f"ZG {zg:,.0f}",
        }
    elif gex_neg and vvr_normale:
        return {
            "type": "ACCELERAZIONE",
            "color": "#ef4444",
            "reason": f"γ-ENV NEG + VVR {vvr:.1f} normale → dealer amplificano, trend acceleration",
            "setups_seek": "Momentum Breakout, PW Defense, Continuation scalps",
            "setups_avoid": "Mean Reversion, Fade (gamma NEG punisce controtendenza)",
            "key_levels": f"PW {pw:,.0f}" if pw else f"ZG {zg:,.0f}",
        }
    elif gex_neg and vvr_elevato:
        return {
            "type": "TAIL RISK",
            "color": "#dc2626",
            "reason": f"γ-ENV NEG + VVR {vvr:.1f} elevato → dealer amplificano, estensioni violente",
            "setups_seek": "Put Wall Defense (absorptions), Collar trades, hedges",
            "setups_avoid": "Tutte le entries controtendenza, sizing aggressivo",
            "key_levels": f"PW {pw:,.0f}" if pw else f"ZG {zg:,.0f}",
        }

    fallback_type = "ROTAZIONE" if gex_pos else "ACCELERAZIONE"
    fallback_color = "#fbbf24" if gex_pos else "#ef4444"
    return {
        "type": fallback_type,
        "color": fallback_color,
        "reason": f"γ-ENV {'POS' if gex_pos else 'NEG'} | VVR {vvr:.1f}",
        "setups_seek": "Osservare price action ai livelli chiave",
        "setups_avoid": "Trading impulsivo senza conferma footprint",
        "key_levels": f"ZG {zg:,.0f}",
    }


# =============================================================================
# PDF GENERATOR
# =============================================================================

class PDFReport:
    """Genera il PDF del briefing con reportlab."""

    def __init__(self, data: dict):
        self.d = data
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        self.s_title = ParagraphStyle("Title", parent=self.styles["Title"],
            fontSize=22, textColor=colors.white, spaceAfter=4*mm,
            fontName="Helvetica-Bold")
        self.s_h1 = ParagraphStyle("H1", parent=self.styles["Heading1"],
            fontSize=14, textColor=colors.HexColor(C_ACCENT), spaceBefore=6*mm, spaceAfter=3*mm,
            fontName="Helvetica-Bold", borderPadding=(0,0,2,0))
        self.s_h2 = ParagraphStyle("H2", parent=self.styles["Heading2"],
            fontSize=11, textColor=colors.white, spaceBefore=4*mm, spaceAfter=2*mm,
            fontName="Helvetica-Bold")
        self.s_body = ParagraphStyle("Body", parent=self.styles["Normal"],
            fontSize=8, textColor=colors.HexColor(C_TEXT), leading=11,
            fontName="Helvetica", spaceAfter=2*mm)
        self.s_small = ParagraphStyle("Small", parent=self.styles["Normal"],
            fontSize=7, textColor=colors.HexColor(C_TEXT_M), leading=9,
            fontName="Helvetica")
        self.s_metric = ParagraphStyle("Metric", parent=self.styles["Normal"],
            fontSize=16, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER)
        self.s_metric_lbl = ParagraphStyle("MetricLbl", parent=self.styles["Normal"],
            fontSize=7, textColor=colors.HexColor(C_TEXT_M), alignment=TA_CENTER)

    def _hr(self):
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor(C_BORDER),
                          spaceBefore=2*mm, spaceAfter=2*mm)

    def _metric_cell(self, label, value, color=C_ACCENT, sub=""):
        val = f'<font color="{color}">{value}</font>'
        sub_html = f'<br/><font size="6" color="{C_TEXT_M}">{sub}</font>' if sub else ""
        return Paragraph(f'{val}<br/><font size="7" color="{C_TEXT_M}">{label}</font>{sub_html}',
                         self.s_metric_lbl)

    def _section_header(self, text):
        return Paragraph(text, self.s_h1)

    def _badge(self, text, color=C_ACCENT):
        bg = color + "22"
        return f'<font backcolor="{bg}" color="{color}">&nbsp;{text}&nbsp;</font>'

    def _card(self, content, bg=C_CARD, border=C_BORDER):
        """Wrapper table che simula una card con background e bordo."""
        t = Table([[content]], colWidths=[168*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor(bg)),
            ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(border)),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        return t

    def generate(self) -> bytes:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=16*mm, rightMargin=16*mm,
                                topMargin=14*mm, bottomMargin=14*mm)
        story = []
        today_str = date.today().strftime("%A %d %B %Y")
        m = self.d.get("macro_last", {})

        def sv(k, default="—"):
            return m.get(k, default)

        # ═══════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════
        snap_date = self.d.get("snapshot_date", "")
        snap_info = f"  |  snapshot: {snap_date}" if snap_date else ""
        ticker_info = self.d.get("ticker", "QQQ")

        header_data = [
            [Paragraph(
                f'<font color="{C_ACCENT}" size="22"><b>MAZE CAPITAL</b></font>'
                f'<font color="{C_TEXT_M}" size="10">  morning briefing</font>',
                self.s_metric_lbl),
             Paragraph(
                f'<font color="{C_TEXT_M}" size="7">{today_str}<br/>{ticker_info}  |  pre-market{snap_info}</font>',
                self.s_metric_lbl)],
        ]
        hdr = Table(header_data, colWidths=[100*mm, 68*mm])
        hdr.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        story.append(hdr)
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(C_ACCENT),
                                spaceBefore=2*mm, spaceAfter=4*mm))

        # ═══════════════════════════════════════════════════════════════════
        # 1. EXECUTIVE SUMMARY
        # ═══════════════════════════════════════════════════════════════════
        story.append(self._section_header("1. Executive Summary"))
        gex_env = sv("gex_env")
        bias = sv("bias", "NEUTRAL")
        vix = sv("vix", 0)
        gex_total = sv("gex_total", 0)
        vvr = sv("vvr", 0)

        bias_color = C_GREEN if bias == "BULLISH" else C_RED if bias == "BEARISH" else C_GOLD
        badge_row = (
            f'{self._badge(f"γ-ENV {gex_env}", C_GREEN if gex_env=="POS" else C_PURPLE)}  '
            f'{self._badge(f"BIAS {bias}", bias_color)}  '
            f'{self._badge(f"VIX {float(vix):.1f}", C_TEXT)}'
        )
        story.append(Paragraph(badge_row, self.s_body))
        story.append(Spacer(1, 3*mm))

        # Metric row card
        gex_c = C_GREEN if float(gex_total) > 0 else C_PURPLE
        vvr_c = C_GREEN if float(vvr) < 4.5 else C_GOLD if float(vvr) < 6.5 else C_RED
        zg = sv("gex_zg", 0)
        conf = sv("confidence", 0)
        conf_c = C_GREEN if float(conf) > 0.6 else C_GOLD
        cw = sv("call_wall")
        pw = sv("put_wall")

        metrics = [
            [self._metric_cell("γ-ENV", gex_env, C_GREEN if gex_env=="POS" else C_PURPLE),
             self._metric_cell("GEX", f"${float(gex_total):+,.0f}", gex_c),
             self._metric_cell("VVR", f"{float(vvr):.1f}", vvr_c),
             self._metric_cell("ZG", f"${float(zg):,.0f}", C_GOLD),
             self._metric_cell("Conf", f"{float(conf)*100:.0f}%", conf_c),
             self._metric_cell("CW/PW",
                               f"${float(cw):,.0f}" if cw and cw != "—" else "—",
                               C_GREEN if gex_env=="POS" else C_PURPLE)],
        ]
        mt = Table(metrics, colWidths=[28*mm]*6)
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor(C_CARD)),
            ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(C_BORDER)),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor(C_BORDER)),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 2),
            ("RIGHTPADDING", (0,0), (-1,-1), 2),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(mt)
        story.append(Spacer(1, 4*mm))

        # ═══════════════════════════════════════════════════════════════════
        # 2. MACRO OVERNIGHT CONTEXT
        # ═══════════════════════════════════════════════════════════════════
        story.append(self._section_header("2. Macro Overnight Context"))

        fut = self.d.get("futures", {})
        if fut:
            fut_data = [[Paragraph("<b>Asset</b>", self.s_body),
                         Paragraph("<b>Price</b>", self.s_body),
                         Paragraph("<b>Chg %</b>", self.s_body)]]
            for name, vals in fut.items():
                chg = vals.get("chg_pct", 0)
                clr = C_GREEN if chg > 0 else C_RED if chg < 0 else C_TEXT
                fut_data.append([
                    Paragraph(name, self.s_body),
                    Paragraph(f"{vals.get('price',''):}", self.s_body),
                    Paragraph(f'<font color="{clr}">{chg:+.2f}%</font>', self.s_body),
                ])
            t = Table(fut_data, colWidths=[50*mm, 30*mm, 30*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor(C_CARD)),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(C_BORDER)),
                ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor(C_BORDER)),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(t)

        idx = self.d.get("indices", {})
        if idx:
            story.append(Spacer(1, 2*mm))
            idx_data = [[Paragraph("<b>Index</b>", self.s_body),
                         Paragraph("<b>Price</b>", self.s_body),
                         Paragraph("<b>Chg %</b>", self.s_body)]]
            for name, vals in idx.items():
                chg = vals.get("chg_pct", 0)
                clr = C_GREEN if chg > 0 else C_RED if chg < 0 else C_TEXT
                idx_data.append([
                    Paragraph(name, self.s_body),
                    Paragraph(f"{vals.get('price',''):}", self.s_body),
                    Paragraph(f'<font color="{clr}">{chg:+.2f}%</font>', self.s_body),
                ])
            t = Table(idx_data, colWidths=[50*mm, 30*mm, 30*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor(C_CARD)),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(C_BORDER)),
                ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor(C_BORDER)),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(t)

        spot_line = []
        for k in ["vix", "dxy", "copper"]:
            v = self.d.get(k)
            if v:
                spot_line.append(f"{k.upper()}: {v}")
        if spot_line:
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(" | ".join(spot_line), self.s_small))

        # ═══════════════════════════════════════════════════════════════════
        # 3. OPTIONS STRUCTURE
        # ═══════════════════════════════════════════════════════════════════
        story.append(self._section_header("3. Options Structure"))

        zg_v = sv("gex_zg", 0)
        tg_v = sv("gex_total", 0)
        vvr_v = sv("vvr", 0)
        cw_v = sv("call_wall")
        pw_v = sv("put_wall")
        skew_v = sv("skew", 0)
        vix_term = sv("vix_term", "")

        opt_lines = [
            f'<b>γ-ENV:</b> {gex_env}  |  <b>ZG:</b> ${float(zg_v):,.0f}  |  '
            f'<b>GEX:</b> ${float(tg_v):+,.0f}  |  <b>VVR:</b> {float(vvr_v):.1f}'
        ]
        if cw_v and cw_v != "—":
            opt_lines.append(f'<b>Call Wall:</b> ${float(cw_v):,.0f}')
        if pw_v and pw_v != "—":
            opt_lines.append(f'<b>Put Wall:</b> ${float(pw_v):,.0f}')
        story.append(Paragraph("<br/>".join(opt_lines), self.s_body))
        story.append(Spacer(1, 1*mm))

        story.append(Paragraph(
            f'<font color="{C_TEXT_M}" size="7">VIX Term: {vix_term}  |  SKEW: {float(skew_v):.1f}</font>',
            self.s_small))
        story.append(Spacer(1, 2*mm))

        # Expected Move
        em_up = m.get("em_up")
        em_down = m.get("em_down")
        if em_up and em_down:
            em_val = (em_up - em_down) / 2
            spot_snap = float(sv("spot", 0))
            em_pct = em_val / spot_snap * 100 if spot_snap else 0
            story.append(Paragraph(
                f'<b>Expected Move</b>  (snapshot {snap_date}):  '
                f'+${em_val:.0f} / −${em_val:.0f}  ({em_pct:.1f}%)  '
                f'<font color="{C_TEXT_M}" size="7">range ${float(em_down):.0f}–${float(em_up):.0f}</font>',
                self.s_body))
        else:
            spot_price = self.d.get("futures", {}).get("S&P 500", {}).get("price", 0)
            vix_val = float(sv("vix", 0))
            if spot_price and vix_val:
                em = expected_move(spot_price, vix_val / 100, 1)
                if em:
                    story.append(Paragraph(
                        f'<b>Expected Move</b>  (estimated):  '
                        f'+${em.get("upper", 0):.0f} / ${em.get("lower", 0):.0f}  '
                        f'({em.get("expected_move_pct", 0):.1f}%)',
                        self.s_body))

        # ═══════════════════════════════════════════════════════════════════
        # 4. TYPE OF DAY (GURUFLIX) + SCENARI APERTURA
        # ═══════════════════════════════════════════════════════════════════
        story.append(self._section_header("4. Type of Day Prediction & Scenari Apertura"))

        gex_env_s   = sv("gex_env")
        vvr_f       = float(sv("vvr", 0))
        gex_total_f = float(sv("gex_total", 0))
        regime_s    = sv("regime", "")
        shift_f     = float(sv("shift_prob", 0))
        cw_f        = sv("call_wall")
        pw_f        = sv("put_wall")
        zg_f        = float(sv("gex_zg", 0))
        spot_f      = float(sv("spot", 0))
        em_up_f     = float(m.get("em_up") or 0)
        em_dn_f     = float(m.get("em_down") or 0)
        em_val      = (em_up_f + em_dn_f) / 2 if (em_up_f and em_dn_f) else 0
        em_pct      = em_val / spot_f * 100 if spot_f else 0
        cw_num      = float(cw_f) if cw_f and cw_f != "—" else None
        pw_num      = float(pw_f) if pw_f and pw_f != "—" else None

        tod = classify_type_of_day(
            gex_env_s, vvr_f, gex_total_f, regime_s, shift_f,
            cw_num, pw_num, zg_f)

        # Card tipo giornata
        tod_bg = tod["color"] + "18"
        tod_t = Table([[Paragraph(
            f'<font color="{tod["color"]}" size="14"><b>{tod["type"]}</b></font>'
            f'<font color="{C_TEXT_M}" size="8">  {tod["reason"][:90]}</font>',
            self.s_body)]], colWidths=[168*mm])
        tod_t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor(tod_bg)),
            ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(tod["color"])),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(tod_t)
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            f'<font color="{C_GREEN}"><b>▸ Cerca:</b></font>  {tod["setups_seek"]}',
            self.s_small))
        story.append(Paragraph(
            f'<font color="{C_RED}"><b>▸ Evita:</b></font>  {tod["setups_avoid"]}',
            self.s_small))
        story.append(Paragraph(
            f'<font color="{C_GOLD}"><b>▸ Livelli EOD:</b></font>  {tod["key_levels"]}',
            self.s_small))
        story.append(Spacer(1, 3*mm))

        # Nota dati EOD
        story.append(Paragraph(
            f'<font color="{C_TEXT_M}" size="7">'
            f'Dati snapshot {snap_date} — il regime intraday si conferma o cambia '
            f'all\'apertura. Controlla γ-ENV sul terminale nei primi 5 minuti.</font>',
            self.s_small))
        story.append(Spacer(1, 3*mm))

        # ── Tabella scenari apertura ──────────────────────────────────────
        story.append(Paragraph(
            f'<font color="{C_GOLD}"><b>SCENARI APERTURA QQQ</b></font>'
            f'<font color="{C_TEXT_M}" size="7">  (basati su EOD {snap_date})</font>',
            self.s_body))
        story.append(Spacer(1, 1*mm))

        em_up_level = round(spot_f + em_val, 2) if spot_f and em_val else 0
        em_dn_level = round(spot_f - em_val, 2) if spot_f and em_val else 0

        def sc_row(scenario, livello, env_atteso, playbook, size, bg):
            return ([
                Paragraph(f'<b>{scenario}</b>', self.s_small),
                Paragraph(livello, self.s_small),
                Paragraph(env_atteso, self.s_small),
                Paragraph(playbook, self.s_small),
                Paragraph(size, self.s_small),
            ], bg)

        cw_str = f"${cw_num:,.0f}" if cw_num else "CW"
        pw_str = f"${pw_num:,.0f}" if pw_num else "PW"
        zg_str = f"${zg_f:,.0f}" if zg_f else "ZG"
        sp_str = f"${spot_f:,.2f}" if spot_f else "Spot"

        scenarios = [
            sc_row(
                "BREAKOUT RIALZISTA",
                f"Apre > {cw_str}",
                f"POS ma walls fragili",
                f"Fake Breakout SHORT da {cw_str} con assorbimento",
                "Size RIDOTTA",
                "#0a1e0a",
            ),
            sc_row(
                f"RANGE DAY (caso base)",
                f"Apre {zg_str}–{cw_str}",
                "POS confermato — dealer smorzano",
                f"Gamma Fade / Dual Flow — Long da {zg_str}, Short da {cw_str}",
                "Size NORMALE",
                "#0a1220",
            ),
            sc_row(
                "GAMMA FLIP",
                f"Apre < {zg_str} e rimane sotto",
                "NEG probabile — dealer amplificano",
                f"Momentum SHORT — no reversal, no mean rev",
                "Size -50%",
                "#1a0a2e",
            ),
            sc_row(
                "OUTSIDE EXPECTED MOVE",
                f"< ${em_dn_level:.1f} o > ${em_up_level:.1f}" if em_val else "Fuori EM",
                "Overextended — snapback o accelerazione",
                "NO nuove entrate — gestisci posizioni aperte",
                "ZERO — stop trading",
                "#1a1400",
            ),
        ]

        sc_header = [
            Paragraph("<b>Scenario</b>", self.s_small),
            Paragraph("<b>Livello QQQ</b>", self.s_small),
            Paragraph("<b>γ-ENV atteso</b>", self.s_small),
            Paragraph("<b>Playbook</b>", self.s_small),
            Paragraph("<b>Size</b>", self.s_small),
        ]
        sc_data = [sc_header] + [row for row, _ in scenarios]
        sc_tbl = Table(sc_data, colWidths=[30*mm, 28*mm, 38*mm, 58*mm, 22*mm])
        sc_style = [
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor(C_CARD)),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor(C_GOLD)),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(C_BORDER)),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor(C_BORDER)),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]
        for i, (_, bg) in enumerate(scenarios):
            sc_style.append(("BACKGROUND", (0, i+1), (-1, i+1), colors.HexColor(bg)))
        sc_tbl.setStyle(TableStyle(sc_style))
        story.append(sc_tbl)
        story.append(Spacer(1, 1*mm))
        story.append(Paragraph(
            f'<font color="{C_TEXT_M}" size="6">'
            f'EM giornaliero: ±${em_val:.2f} ({em_pct:.2f}%)  →  '
            f'range atteso {em_dn_level:.2f} / {em_up_level:.2f}  |  '
            f'Bias EOD: {sv("bias", "NEUTRAL")}</font>',
            self.s_small))

        # Bias (mantenuto per compatibilità)
        bias_str = sv("bias", "NEUTRAL")
        bias_desc = {
            "BULLISH": "Bias rialzista: favorire acquisti su debolezza",
            "BEARISH": "Bias ribassista: favorire vendite su forza",
            "NEUTRAL": "Bias neutrale: aspettare conferma intraday",
        }.get(bias_str, "")

        # ═══════════════════════════════════════════════════════════════════
        # 5. TECHNICAL LEVELS
        # ═══════════════════════════════════════════════════════════════════
        story.append(self._section_header("5. Technical Levels"))
        g_score = float(sv("g_score", 0))
        i_score = float(sv("i_score", 0))
        shift_p = float(sv("shift_prob", 0))
        story.append(Paragraph(
            f'<b>Regime:</b> {regime_s}  |  <b>G-Score:</b> {g_score:.2f}  |  '
            f'<b>I-Score:</b> {i_score:.2f}  |  <b>Shift Prob:</b> {shift_p*100:.0f}%',
            self.s_body))

        # ═══════════════════════════════════════════════════════════════════
        # 6. ECONOMIC CALENDAR
        # ═══════════════════════════════════════════════════════════════════
        cal = self.d.get("calendar", [])
        if cal:
            story.append(self._section_header("6. Economic Calendar"))
            cal_data = [[Paragraph("<b>Time</b>", self.s_body),
                         Paragraph("<b>Event</b>", self.s_body),
                         Paragraph("<b>Impact</b>", self.s_body)]]
            for ev in cal[:6]:
                imp = ev.get("impact", "Low").capitalize()
                imp_color = C_RED if imp == "High" else C_GOLD if imp == "Medium" else C_TEXT_M
                cal_data.append([
                    Paragraph(ev.get("time", "—"), self.s_body),
                    Paragraph(ev.get("event", "—"), self.s_body),
                    Paragraph(f'<font color="{imp_color}">{imp}</font>', self.s_body),
                ])
            t = Table(cal_data, colWidths=[30*mm, 70*mm, 20*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor(C_CARD)),
                ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(C_BORDER)),
                ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor(C_BORDER)),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(t)

        # ═══════════════════════════════════════════════════════════════════
        # 7. RECENT HISTORY
        # ═══════════════════════════════════════════════════════════════════
        macro = self.d.get("macro")
        if macro is not None and len(macro) >= 3:
            story.append(self._section_header("7. Recent History (5 days)"))
            hist = macro.head(5)
            h_cols = ["timestamp", "regime", "gex_env", "gex_total", "vvr", "vix", "bias"]
            h_data = [[Paragraph(f"<b>{c}</b>", self.s_body) for c in h_cols]]
            for _, row in hist.iterrows():
                r = []
                for c in h_cols:
                    v = row.get(c, "—")
                    if c == "gex_total":
                        v = f"${v:+,.0f}" if pd.notna(v) else "—"
                    elif c == "vvr":
                        v = f"{v:.1f}" if pd.notna(v) else "—"
                    elif isinstance(v, float):
                        v = f"{v:.1f}"
                    else:
                        v = str(v) if pd.notna(v) else "—"
                    r.append(Paragraph(v, self.s_body))
                h_data.append(r)
            t = Table(h_data, colWidths=[35*mm, 25*mm, 20*mm, 28*mm, 15*mm, 15*mm, 25*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor(C_CARD)),
                ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(C_BORDER)),
                ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor(C_BORDER)),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING", (0,0), (-1,-1), 3),
                ("RIGHTPADDING", (0,0), (-1,-1), 3),
                ("FONTSIZE", (0,0), (-1,-1), 7),
            ]))
            story.append(t)

        # ═══════════════════════════════════════════════════════════════════
        # 8. VAULT INSIGHTS
        # ═══════════════════════════════════════════════════════════════════
        vault = self.d.get("vault_notes", [])
        if vault:
            story.append(self._section_header("8. Vault Insights"))
            for note in vault:
                src = note.get("source", "")
                if src == "playbook":
                    story.append(Paragraph(
                        f'<font color="{C_ACCENT}"><b>{note["title"]}</b></font>'
                        f'<font color="{C_TEXT_M}">  [playbook]</font>',
                        self.s_body))
                    if note.get("trigger"):
                        story.append(Paragraph(
                            f'<font color="{C_GREEN}">▸ Trigger:</font> {note["trigger"][:120]}',
                            self.s_small))
                    if note.get("entry"):
                        story.append(Paragraph(
                            f'<font color="{C_ACCENT}">▸ Entry:</font> {note["entry"][:120]}',
                            self.s_small))
                    if note.get("stop"):
                        story.append(Paragraph(
                            f'<font color="{C_PURPLE}">▸ Stop:</font> {note["stop"][:80]}',
                            self.s_small))
                    if note.get("target"):
                        story.append(Paragraph(
                            f'<font color="{C_GOLD}">▸ Target:</font> {note["target"][:80]}',
                            self.s_small))
                    if note.get("regime_match"):
                        story.append(Paragraph(
                            f'<font color="{C_TEXT_M}" size="6">Regime: {note["regime_match"]}</font>',
                            self.s_small))
                elif src == "checklist":
                    story.append(Paragraph(
                        f'<font color="{C_GOLD}"><b>{note["title"]}</b></font>'
                        f'<font color="{C_TEXT_M}">  [checklist premarket]</font>',
                        self.s_body))
                    if note.get("checklist"):
                        story.append(Paragraph(note["checklist"][:400], self.s_small))
                else:
                    story.append(Paragraph(
                        f'<b>{note["title"]}</b>'
                        f'<font color="{C_TEXT_M}">  [{" ".join(note.get("tags", []))}]</font>',
                        self.s_body))
                    if note.get("snippet"):
                        story.append(Paragraph(note["snippet"], self.s_small))
                story.append(Spacer(1, 1*mm))

        # ═══════════════════════════════════════════════════════════════════
        # FOOTER
        # ═══════════════════════════════════════════════════════════════════
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(C_BORDER),
                                spaceBefore=4*mm, spaceAfter=2*mm))
        story.append(Paragraph(
            f'<font color="{C_TEXT_M}" size="6">MAZE CAPITAL  ·  Generated {datetime.now():%Y-%m-%d %H:%M}  ·  '
            f'Data: yfinance + GEXBot + Tradier  ·  For institutional use only</font>',
            self.s_small))

        doc.build(story)
        return buf.getvalue()


# =============================================================================
# EMAIL SENDER
# =============================================================================

def send_email(pdf_bytes: bytes, subject: str = None):
    """Invia il PDF via Gmail SMTP con App Password."""
    if not CONFIG.get("gmail_app_password") or not CONFIG.get("email_to"):
        print("⚠ Config email mancante. Salvo PDF su disco.")
        out_path = Path(__file__).parent / f"briefing_{date.today().isoformat()}.pdf"
        out_path.write_bytes(pdf_bytes)
        print(f"✅ PDF salvato: {out_path}")
        return

    msg = MIMEMultipart()
    msg["From"] = CONFIG.get("email_from", CONFIG["email_to"])
    msg["To"] = CONFIG["email_to"]
    msg["Subject"] = subject or f"MAZE CAPITAL Morning Briefing — {date.today():%d %b %Y}"

    body = (
        f"<html><body style='background:#0a0e17;color:#e2e8f0;font-family:system-ui;padding:20px;'>"
        f"<h2 style='color:#00d2ff;'>⬡ MAZE CAPITAL</h2>"
        f"<p>Morning Briefing del {date.today():%d/%m/%Y}.</p>"
        f"<p>Il PDF è allegato a questa email.</p>"
        f"<hr style='border-color:#1f2937;'>"
        f"<p style='color:#64748b;font-size:11px;'>Per uso istituzionale — Dati: yfinance + GEXBot + Tradier</p>"
        f"</body></html>"
    )
    msg.attach(MIMEText(body, "html"))

    part = MIMEBase("application", "octet-stream")
    part.set_payload(pdf_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="briefing_{date.today().isoformat()}.pdf"')
    msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(CONFIG["email_from"], CONFIG["gmail_app_password"])
        server.send_message(msg)
        server.quit()
        print(f"✅ Briefing inviato a {CONFIG['email_to']}")
    except Exception as e:
        print(f"❌ Errore invio email: {e}")
        out_path = Path(__file__).parent / f"briefing_{date.today().isoformat()}.pdf"
        out_path.write_bytes(pdf_bytes)
        print(f"✅ PDF salvato su disco: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def is_us_market_day() -> bool:
    """Controlla se oggi è un giorno di borsa US (lun-ven, non festivo)."""
    today = date.today()
    if today.weekday() >= 5:
        return False
    holidays = [
        date(today.year, 1, 1),   # New Year
        date(today.year, 1, 20),  # MLK Day (3rd Mon Jan)
        date(today.year, 2, 17),  # Presidents Day (3rd Mon Feb)
        date(today.year, 4, 18),  # Good Friday
        date(today.year, 5, 26),  # Memorial Day (last Mon May)
        date(today.year, 6, 19),  # Juneteenth (observed)
        date(today.year, 7, 4),   # Independence Day
        date(today.year, 9, 1),   # Labor Day (1st Mon Sep)
        date(today.year, 11, 27), # Thanksgiving (4th Thu Nov)
        date(today.year, 12, 25), # Christmas
    ]
    # Aggiusta MLK/Presidents/Memorial/Labor/Thanksgiving dinamicamente
    # MLK: 3rd Monday of January
    mlk = _nth_weekday(2026, 1, 3, 0)
    if mlk: holidays.append(mlk)
    pres = _nth_weekday(2026, 2, 3, 0)
    if pres: holidays.append(pres)
    memo = _nth_weekday(2026, 5, 5, 0)  # last Monday
    if memo: holidays.append(memo)
    lab = _nth_weekday(2026, 9, 1, 0)
    if lab: holidays.append(lab)
    tgiv = _nth_weekday(2026, 11, 4, 3)  # 4th Thursday
    if tgiv: holidays.append(tgiv)
    # Juneteenth observed
    jun = date(2026, 6, 19)
    if jun.weekday() == 5: jun = date(2026, 6, 18)
    if jun.weekday() == 6: jun = date(2026, 6, 20)
    holidays.append(jun)
    # Independence Day observed
    jul4 = date(2026, 7, 4)
    if jul4.weekday() == 5: jul4 = date(2026, 7, 3)
    if jul4.weekday() == 6: jul4 = date(2026, 7, 5)
    holidays.append(jul4)

    return today not in holidays


def _nth_weekday(year, month, n, weekday):
    """n-esimo giorno `weekday` (0=lun) del mese."""
    try:
        d = date(year, month, 1)
        count = 0
        while d.month == month:
            if d.weekday() == weekday:
                count += 1
                if count == n:
                    return d
            d += timedelta(days=1)
    except Exception:
        pass
    return None


def generate_and_send(force=False):
    """Esegue l'intero flusso: data → PDF → email."""
    print(f"\n{'='*60}")
    print(f"MAZE CAPITAL — Morning Briefing {date.today()}")
    print(f"{'='*60}")

    if not is_us_market_day() and not force:
        print("⏭ Oggi non è giorno di borsa US. Skippo.")
        return

    print("📡 Raccolta dati...")
    dc = DataCollector(CONFIG.get("ticker", "QQQ"))
    data = dc.run()

    print("📄 Generazione PDF...")
    pdf = PDFReport(data)
    pdf_bytes = pdf.generate()
    print(f"✅ PDF generato ({len(pdf_bytes)} bytes)")

    print("📧 Invio...")
    send_email(pdf_bytes)
    print("✅ Done.\n")


def main():
    """Entry point: esecuzione immediata o scheduler."""
    import argparse
    parser = argparse.ArgumentParser(description="MAZE CAPITAL Morning Briefing")
    parser.add_argument("--now", action="store_true", help="Esegui subito")
    parser.add_argument("--schedule", action="store_true", help="Avvia scheduler giornaliero")
    parser.add_argument("--force", action="store_true", help="Forza esecuzione anche fuori borsa US")
    args = parser.parse_args()

    if args.now or not args.schedule:
        generate_and_send(force=args.force)
    if args.schedule:
        import schedule
        import time
        send_time = CONFIG.get("send_at", "15:00")
        schedule.every().day.at(send_time).do(generate_and_send)
        print(f"⏰ Scheduler avviato. Invio ogni giorno alle {send_time} IT (giorni borsa US).")
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == "__main__":
    main()
