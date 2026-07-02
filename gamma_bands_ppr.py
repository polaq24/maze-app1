"""
Gamma Bands PPR (Price Position Range) — Standalone Viewer
Legge gamma_bands_SPY.csv e mostra le bande in un grafico interattivo.
Run: streamlit run gamma_bands_ppr.py
"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="PPR Gamma Bands", layout="centered")
DARK = "#0a0e17"; CARD = "#111827"; CARD_B = "#1f2937"
GREEN = "#22c55e"; RED = "#ef4444"; GOLD = "#fbbf24"; TEXT_M = "#64748b"

st.markdown(f"""
<style>
    .stApp {{ background:{DARK}; color:#e2e8f0; }}
    h1 {{ color:#e2e8f0!important; font-size:18px!important; }}
    .band {{ background:{CARD}; border:1px solid {CARD_B}; border-radius:6px; padding:6px 10px; margin:3px 0; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<h1>⬡ Gamma Bands — PPR (Price Position Range)</h1>', unsafe_allow_html=True)

gb_path = Path("C:/Users/Gabriel/Downloads/gamma_bands_SPY.csv")
if not gb_path.exists():
    st.error(f"File non trovato: {gb_path}")
    st.stop()

gb = pd.read_csv(gb_path)
spot = float(gb.loc[gb["direzione"] == "●", "centro"].iloc[0])

st.markdown(f'<div style="color:{TEXT_M};font-size:12px;margin-bottom:10px;">'
            f'SPOT: <b style="color:white;">{spot:.1f}</b> | '
            f'Bande: {len(gb)} | '
            f'Range: {gb["livello_lo"].min():.0f} – {gb["livello_hi"].max():.0f}</div>',
            unsafe_allow_html=True)

# ── Grafico principale ──
fig = go.Figure()
colors = {"▲": "rgba(34,197,94,0.25)", "▼": "rgba(239,68,68,0.25)", "●": "rgba(255,255,255,0.08)"}
border = {"▲": "rgba(34,197,94,0.7)", "▼": "rgba(239,68,68,0.7)", "●": "rgba(255,255,255,0.3)"}

for _, r in gb.iterrows():
    fig.add_trace(go.Scatter(
        x=[r["livello_lo"], r["livello_hi"]],
        y=[0, 0],
        mode="lines+markers",
        line=dict(color=border.get(r["direzione"], "#888"), width=6),
        marker=dict(size=10, color=border.get(r["direzione"], "#888"), symbol="diamond"),
        name=r["zona"],
        hovertext=f"<b>{r['zona']}</b><br>"
                  f"{r['livello_lo']:.0f} – {r['livello_hi']:.0f}<br>"
                  f"Centro: {r['centro']:.0f}<br>"
                  f"{r['nota']}",
        hoverinfo="text",
        showlegend=False
    ))
    fig.add_annotation(x=r["centro"], y=0, text=r["zona"].split(" ")[0],
                       font=dict(size=8, color=TEXT_M), showarrow=False, yshift=14)

fig.add_vline(x=spot, line=dict(color="white", width=2),
              annotation_text=f"SPOT {spot:.0f}", annotation_font_size=10, annotation_font_color="white")

fig.update_layout(
    height=200,
    xaxis=dict(range=[gb["livello_lo"].min() - 10, gb["livello_hi"].max() + 10],
               showgrid=True, gridcolor="rgba(255,255,255,0.08)",
               tickfont=dict(size=9, color=TEXT_M), title=None),
    yaxis=dict(visible=False, range=[-0.5, 0.5]),
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    hovermode="closest"
)
st.plotly_chart(fig, use_container_width=True)

# ── Tabella riepilogativa ──
st.markdown("### Dettaglio bande")
for _, r in gb.iterrows():
    clr = {"▲": GREEN, "▼": RED, "●": "#888"}.get(r["direzione"], "#888")
    icon = {"▲": "🟢", "▼": "🔴", "●": "⚪"}.get(r["direzione"], "⚪")
    st.markdown(f'<div class="band" style="border-left:3px solid {clr};">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
                f'<span>{icon} <b style="color:{clr};">{r["zona"]}</b></span>'
                f'<span style="color:white;">{r["livello_lo"]:.1f} – {r["livello_hi"]:.1f}</span>'
                f'<span style="color:{TEXT_M};">{r["pct_low"]} / {r["pct_high"]}</span>'
                f'</div>'
                f'<div style="font-size:10px;color:{TEXT_M};margin-top:2px;">{r["nota"]}</div>'
                f'</div>', unsafe_allow_html=True)
