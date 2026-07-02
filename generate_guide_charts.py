import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import uniform_filter1d
import os

OUT = r'C:\Users\Gabriel\Desktop\2nd-brain\.claude\guide_images'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
DPI = 180

# ════════════════════════════════════════════════
# 1. Gamma Bands Shift Intraday
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor('#fafafa')

t = np.linspace(0, 65, 131)
price = 707 - 0.045 * t - 0.015 * np.maximum(t - 15, 0)**1.3
price += rng.normal(0, 0.06, 131)
price = np.clip(price, 696.5, None)
price[0] = 707
price[10:20] -= 0.15  # a dip

# Support levels
support_open = 704.8 * np.ones_like(t)
support_recalc = 704.8 - 0.12 * np.maximum(t - 8, 0)
support_recalc = np.where(t > 30, 702.0 - 0.07 * np.maximum(t - 30, 0), support_recalc)
support_recalc = np.maximum(support_recalc, 697.5)

ax.plot(t, price, 'k-', lw=1.8, label='Price', zorder=5)
ax.plot(t, support_recalc, '#d13841', ls='--', lw=2.2, label='Updated Support (bands)', zorder=4)
ax.plot(t, support_open, '#e67e22', ls='--', lw=1.5, label='Opening Support', zorder=3, alpha=0.7)

ax.axvline(8, color='#888', ls=':', lw=1, alpha=0.5)
ax.text(8.5, 705.5, 'Refresh #1\nbands adjust', fontsize=7, color='#888', fontstyle='italic')
ax.axvline(30, color='#888', ls=':', lw=1, alpha=0.5)
ax.text(30.5, 702.5, 'Refresh #N\nnew put wall\nforms lower', fontsize=7, color='#888', fontstyle='italic')

# Shaded zone between old & new support
ax.fill_between(t, support_open, support_recalc, alpha=0.08, color='#d13841',
                label='Support degradation')

ax.set_xlabel('Minutes from Open', fontsize=10)
ax.set_ylabel('Price ($)', fontsize=10)
ax.set_title('Gamma Bands — Dynamic Support Adjustment Intraday', fontsize=12, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
ax.set_ylim(696, 708.5)
ax.grid(alpha=0.15)
fig.tight_layout()
fig.savefig(f'{OUT}/01_gamma_bands_shift.png', dpi=DPI)
plt.close(fig)
print('1/12 Gamma Bands shift done')

# ════════════════════════════════════════════════
# 2. B&L PDF with Quantiles
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor('#fafafa')

K = np.linspace(620, 790, 500)
spot = 707
s = np.where(K < spot, 25, 22)
pdf = np.exp(-0.5 * ((K - spot + 3) / s)**2) / (s * np.sqrt(2*np.pi))
pdf += 0.25 * np.exp(-0.5 * ((K - spot + 20) / 35)**2) / (35 * np.sqrt(2*np.pi))
pdf += 0.08 * np.exp(-0.5 * ((K - spot - 35) / 40)**2) / (40 * np.sqrt(2*np.pi))
pdf /= pdf.sum() * (K[1] - K[0])

cdf = np.cumsum(pdf) * (K[1] - K[0])
q25 = K[np.searchsorted(cdf, 0.25)]
q50 = K[np.searchsorted(cdf, 0.50)]
q75 = K[np.searchsorted(cdf, 0.75)]

ax.plot(K, pdf, '#1a3a5c', lw=2.2, label='Risk-Neutral PDF (Breeden-Litzenberger)', zorder=5)
ax.axvline(spot, color='orange', ls=':', lw=1.8, zorder=4, label=f'Spot ${spot}')
ax.axvline(q25, color='#d13841', ls='--', lw=1.5, alpha=0.8, label=f'Q25 ${q25:.0f}')
ax.axvline(q50, color='#2a9d5c', ls='--', lw=2, label=f'Q50 ${q50:.0f}')
ax.axvline(q75, color='#d13841', ls='--', lw=1.5, alpha=0.8, label=f'Q75 ${q75:.0f}')

ax.fill_between(K, pdf, where=(K >= q25) & (K <= q75), alpha=0.15, color='#2a9d5c', label=f'IQR (${q25:.0f}–${q75:.0f})')
ax.fill_between(K, pdf, where=(K <= q25), alpha=0.08, color='#d13841')
ax.fill_between(K, pdf, where=(K >= q75), alpha=0.08, color='#d13841')

ax.annotate(f'Q50 = ${q50:.0f}\n(risk-neutral fair value)',
            xy=(q50, pdf[np.argmin(np.abs(K-q50))]),
            xytext=(q50+20, pdf.max()*0.75),
            fontsize=8, color='#2a9d5c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2a9d5c', lw=1.2),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('Strike Price', fontsize=10)
ax.set_ylabel('Probability Density', fontsize=10)
ax.set_title('Breeden-Litzenberger — Risk-Neutral Quantiles & IQR', fontsize=12, fontweight='bold')
ax.legend(fontsize=7.5, framealpha=0.9)
ax.set_xlim(645, 770)
ax.grid(alpha=0.12)
fig.tight_layout()
fig.savefig(f'{OUT}/02_bl_quantiles.png', dpi=DPI)
plt.close(fig)
print('2/12 B&L quantiles done')

# ════════════════════════════════════════════════
# 3. Confluenza Strutturale
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor('#fafafa')

tools = ['Gamma Bands\n(S/R)', 'B&L Q25/IQR', 'KO Range\n(1σ barrier)', 'ZG Flip', 'Call Wall', 'ETF Flow\n(institutional)']
levels = [708, 706, 709, 707.5, 750, 710]
colors_t = ['#1a3a5c', '#2a9d5c', '#d13841', '#d4a017', '#8e44ad', '#e67e22']
markers = ['s', 'D', '^', 'o', 'v', 'P']

for i, (tool, lvl, clr, mkr) in enumerate(zip(tools, levels, colors_t, markers)):
    y = 5 - i
    ax.scatter(lvl, y, s=400, c=clr, marker=mkr, edgecolors='white', linewidth=1.5, zorder=6)
    ax.text(lvl + 2.5, y, f'{tool} @ ${lvl}', fontsize=8, va='center', color=clr, fontweight='bold')

confluence_zone = (706, 710)
ax.axvspan(confluence_zone[0], confluence_zone[1], alpha=0.1, color='#2a9d5c')
ax.annotate('CONFLUENCE ZONE\n706 – 710',
            xy=(708, 5.6), fontsize=11, ha='center', va='center',
            color='#2a9d5c', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2a9d5c', alpha=0.95))

# Outlier call wall
ax.scatter(750, 5-0, s=400, c='#8e44ad', marker='v', edgecolors='white', linewidth=1.5, zorder=6, alpha=0.5)
ax.text(752, 5, 'Call Wall @ 750\n(outlier — not in confluence)', fontsize=7.5, va='center', color='#8e44ad', fontstyle='italic')

ax.set_xlim(695, 760)
ax.set_ylim(-0.8, 6.5)
ax.set_xlabel('Price Level', fontsize=10)
ax.set_yticks([])
ax.set_title('Structural Confluence — Independent Tools Converging', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.2)
fig.tight_layout()
fig.savefig(f'{OUT}/03_confluenza.png', dpi=DPI)
plt.close(fig)
print('3/12 Confluenza done')

# ════════════════════════════════════════════════
# 4. GEX Profile — ZG & Call Wall
# ════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#fafafa')

strikes = np.arange(650, 770, 2.5)
spot = 708

gex = np.zeros_like(strikes)
for i, k in enumerate(strikes):
    if k < 670:       gex[i] = -rng.uniform(0.3, 0.8)
    elif k < 685:     gex[i] = -rng.uniform(0.8, 2.0)
    elif k < 695:     gex[i] = -rng.uniform(1.5, 3.5)
    elif k < 705:     gex[i] = -rng.uniform(0.5, 2.0)
    elif k < 715:     gex[i] = rng.uniform(0.3, 1.5)
    elif k < 730:     gex[i] = rng.uniform(0.5, 1.0)
    elif k < 745:     gex[i] = rng.uniform(0.2, 0.6)
    else:             gex[i] = -rng.uniform(0.5, 1.0)

wall_idx = np.argmin(np.abs(strikes - 750))
gex[wall_idx] = -12.0
gex[wall_idx-1] = -4.0
gex[wall_idx+1] = -3.0
gex = uniform_filter1d(gex, size=3)

ax1.bar(strikes, gex, width=2.2, color=['#d13841' if g < 0 else '#2a9d5c' for g in gex], alpha=0.8, zorder=3)

cum = np.cumsum(gex)
zg_idx = np.argmin(np.abs(cum))
zg = strikes[zg_idx]

ax2 = ax1.twinx()
ax2.plot(strikes, cum, '#1a3a5c', lw=2.5, zorder=4, label='Cumulative GEX')
ax2.axhline(0, color='#1a3a5c', lw=0.8, alpha=0.3)

ax2.axvline(zg, color='#1a3a5c', ls='--', lw=2, zorder=5)
ax2.annotate(f'Gamma Flip @ ${zg:.0f}',
             xy=(zg, 0), xytext=(zg + 10, cum.max() * 0.55),
             fontsize=10, color='#1a3a5c', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#1a3a5c', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1a3a5c', alpha=0.9))

ax1.axvline(spot, color='orange', ls=':', lw=2, zorder=5)
ax1.text(spot + 0.5, ax1.get_ylim()[1] * 0.92, f'Spot ${spot}', fontsize=9, color='orange', fontweight='bold')

max_pain = strikes[np.argmax(np.abs(gex))]
ax1.axvline(max_pain, color='#8e44ad', ls=':', lw=1.5, zorder=5)
ax1.text(max_pain + 1, ax1.get_ylim()[1] * 0.78, f'Call Wall\n{max_pain:.0f}', fontsize=8.5, color='#8e44ad', fontweight='bold')

ax1.set_xlabel('Strike Price', fontsize=10)
ax1.set_ylabel('GEX ($γ per 1% spot move)', color='#555', fontsize=9)
ax2.set_ylabel('Cumulative GEX', color='#1a3a5c', fontsize=9)
ax1.set_title('Gamma Exposure Profile — Zero Gamma (ZG) & Call Wall', fontsize=12, fontweight='bold', pad=8)

from matplotlib.patches import Patch
legend = [
    Patch(facecolor='#d13841', alpha=0.8, label='Negative γ (amplifying)'),
    Patch(facecolor='#2a9d5c', alpha=0.8, label='Positive γ (stabilising)'),
    plt.Line2D([0], [0], color='#1a3a5c', lw=2.5, label='Cumulative GEX'),
]
ax1.legend(handles=legend, fontsize=8, loc='upper left', framealpha=0.9)
ax1.grid(axis='x', alpha=0.12)
ax1.set_xlim(645, 770)
fig.tight_layout()
fig.savefig(f'{OUT}/04_zg_gex.png', dpi=DPI)
plt.close(fig)
print('4/12 ZG/GEX done')

# ════════════════════════════════════════════════
# 5. Intraday Decision Tree
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#fafafa')

def nbox(x, y, w, h, text, fc='#1a3a5c', fs=7, ec='white'):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                  facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color='white', fontweight='bold')

def nlink(x1, y1, x2, y2, label='', color='#999'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, fontsize=6.5, color=color, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, pad=1, boxstyle='round'))

nbox(3.5, 9, 3, 0.7, 'SESSION OPENS', '#0d2137', 9)
nlink(5, 9, 1, 7.5, 'Step 1')
nlink(5, 9, 5, 7.5, 'Step 2')
nlink(5, 9, 9, 7.5, 'Step 3')

nbox(0.5, 7, 3, 0.7, 'Identify STATIC levels\nCall/Put Wall, Max OI', '#1a3a5c', 7)
nbox(3.5, 7, 3, 0.7, 'Identify SEMI-STATIC\nBands, Q50, ZG', '#1a3a5c', 7)
nbox(6.5, 7, 3, 0.7, 'Identify DYNAMIC\nAdapted VaR, KO Range', '#1a3a5c', 7)
nlink(1.5, 7, 3, 5.5, '')
nlink(5, 7, 5, 5.5, '')
nlink(8.5, 7, 7, 5.5, '')

nbox(2.5, 5.2, 5, 0.7, 'Any confluence?  →  Enter setup\nNo confluence  →  Wait', '#2a5a8c', 8)
nlink(5, 5.2, 5, 3.8, '30s refresh')

nbox(3.5, 3.5, 3, 0.6, 'REFRESH\nSemi-static levels\nmoved?', '#d13841', 8)
nlink(3.5, 3.8, 1.5, 2.5, '< 0.2%\nNOISE')
nlink(5, 3.5, 5, 2.5, '0.2–0.5%\nRIVALUTA')
nlink(6.5, 3.5, 8.5, 2.5, '> 0.5%\nNEW LEVEL')

nbox(0.2, 2.2, 2.5, 0.7, 'IGNORE\nNo action needed', '#2a9d5c', 7)
nbox(3.5, 2.2, 3, 0.7, 'RIVALUTA\nUpdate stops, reassess', '#d4a017', 7)
nbox(7.3, 2.2, 2.5, 0.7, 'NEW LEVEL\nRecalculate everything', '#d13841', 7)
nlink(1.5, 2.2, 4, 1.2, '')
nlink(5, 2.2, 5, 1.2, '')
nlink(8.5, 2.2, 6, 1.2, '')

nbox(3, 0.8, 4, 0.7, 'DURING THE TRADE\nConfluence stronger → hold/add\nConfluence stable → hold\nConfluence weaker → tighten\nConfluence dead → EXIT', '#0d2137', 7)

ax.set_title('INTRADAY DECISION TREE', fontsize=14, fontweight='bold', pad=8, color='#0d2137')
fig.tight_layout()
fig.savefig(f'{OUT}/05_decision_tree.png', dpi=DPI)
plt.close(fig)
print('5/12 Decision tree done')

# ════════════════════════════════════════════════
# 6. Expected Shortfall & VaR
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor('#fafafa')

K = np.linspace(620, 790, 500)
spot = 707
s_left, s_right = 24, 22
pdf = np.exp(-0.5 * ((K - spot + 2) / np.where(K < spot, s_left, s_right))**2) / (np.where(K < spot, s_left, s_right) * np.sqrt(2*np.pi))
pdf += 0.3 * np.exp(-0.5 * ((K - spot + 18) / 32)**2) / (32 * np.sqrt(2*np.pi))
pdf += 0.1 * np.exp(-0.5 * ((K - spot - 30) / 38)**2) / (38 * np.sqrt(2*np.pi))
pdf /= pdf.sum() * (K[1] - K[0])

cdf = np.cumsum(pdf) * (K[1] - K[0])
alpha = 0.05
var_idx = np.searchsorted(cdf, alpha)
var_5 = K[var_idx]
tail = K[:var_idx]
es_5 = np.average(tail, weights=pdf[:var_idx])

ax.plot(K, pdf, '#1a3a5c', lw=2.2, label='Risk-Neutral PDF', zorder=5)
ax.axvline(spot, color='orange', ls=':', lw=1.8, label=f'Spot ${spot}', zorder=4)
ax.axvline(var_5, color='#d4a017', ls='--', lw=2, label=f'VaR₅% = ${var_5:.0f}', zorder=5)
ax.axvline(es_5, color='#d13841', ls='-', lw=2.5, label=f'ES₅% = ${es_5:.0f}', zorder=6)

ax.fill_between(K, pdf, where=(K <= var_5), alpha=0.25, color='#d13841', label='Tail (5%)')
ax.fill_between(K, pdf, where=(K <= es_5), alpha=0.12, color='#8e44ad')

ax.annotate(f'Expected Shortfall (CVaR)\nAvg loss in worst {alpha*100:.0f}% = ${es_5:.0f}',
            xy=(es_5, pdf[var_idx]*0.8),
            xytext=(es_5-35, pdf.max()*0.6),
            fontsize=8.5, color='#d13841', fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='#d13841', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('Price', fontsize=10)
ax.set_ylabel('Probability Density', fontsize=10)
ax.set_title('Value at Risk (VaR) & Expected Shortfall (ES / CVaR)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, framealpha=0.9)
ax.set_xlim(645, 770)
ax.grid(alpha=0.12)
fig.tight_layout()
fig.savefig(f'{OUT}/06_expected_shortfall.png', dpi=DPI)
plt.close(fig)
print('6/12 Expected Shortfall done')

# ════════════════════════════════════════════════
# 7. IV Skew — Volatility Smile
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor('#fafafa')

strikes = np.arange(650, 770, 5)
spot = 708
dte = 7

iv_skew = 18 + 0.4 * (spot - strikes)
iv_skew = np.where(strikes < spot, iv_skew + 5 * np.exp(-0.01 * (strikes - spot)**2), iv_skew)
iv_skew = np.where(strikes > spot, iv_skew + 2 * np.exp(-0.005 * (strikes - spot - 20)**2), iv_skew)
iv_skew += rng.normal(0, 0.15, len(strikes))
iv_skew = uniform_filter1d(iv_skew, size=3)

ax.plot(strikes, iv_skew, '#1a3a5c', lw=2.5, zorder=5)
ax.axvline(spot, color='orange', ls=':', lw=1.8, label=f'Spot ${spot}', zorder=4)
ax.fill_between(strikes, iv_skew, alpha=0.08, color='#1a3a5c')

iv_atm = np.interp(spot, strikes, iv_skew)
ax.scatter(spot, iv_atm, s=80, c='orange', edgecolors='white', linewidth=1.5, zorder=6)
ax.annotate(f'ATM σ = {iv_atm:.1f}%',
            xy=(spot, iv_atm), xytext=(spot+12, iv_atm+0.8),
            fontsize=9, color='orange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='orange', lw=1.2),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.annotate('PUT SKEW\n(steep decline —\nfear premium)',
            xy=(660, np.interp(660, strikes, iv_skew)),
            xytext=(640, 33), fontsize=8, color='#d13841', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d13841', lw=1.2),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.annotate('CALL SKEW\n(mild — possible\ncall wall premium)',
            xy=(750, np.interp(750, strikes, iv_skew)),
            xytext=(745, 27), fontsize=8, color='#8e44ad', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.2),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('Strike Price', fontsize=10)
ax.set_ylabel('Implied Volatility (%)', fontsize=10)
ax.set_title(f'IV Skew — {dte}-Day Options', fontsize=12, fontweight='bold')
ax.grid(alpha=0.12)
ax.set_xlim(645, 770)
fig.tight_layout()
fig.savefig(f'{OUT}/07_iv_skew.png', dpi=DPI)
plt.close(fig)
print('7/12 IV Skew done')

# ════════════════════════════════════════════════
# 8. KO Range
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 3.5))
fig.patch.set_facecolor('#fafafa')

spot = 707
sigma = 18
ko_1s_up = spot * (1 + sigma/100)
ko_1s_dn = spot * (1 - sigma/100)
ko_2s_up = spot * (1 + 2*sigma/100)
ko_2s_dn = spot * (1 - 2*sigma/100)
ko_range = np.sqrt(252/7) * sigma / 100 * spot  # ~1.7% for 7dte

levels = [ko_2s_dn - 15, ko_1s_dn, spot, ko_1s_up, ko_2s_up + 15]
labels = ['', f'1σ KO\n${ko_1s_dn:.0f}', f'SPOT\n${spot}', f'1σ KO\n${ko_1s_up:.0f}', '']
colors = ['#d13841', '#d13841', 'orange', '#2a9d5c', '#2a9d5c']
widths = [0.5, 2, 2, 2, 0.5]

for i, (lv, lb, cl, w) in enumerate(zip(levels, labels, colors, widths)):
    y = 1 if i in [1,3] else 0
    ax.scatter(lv, y, s=200, c=cl, edgecolors='white', linewidth=1.5, zorder=6)
    ax.text(lv, y+0.15, lb, fontsize=8.5, ha='center', color=cl, fontweight='bold')

ax.axvspan(ko_1s_dn, ko_1s_up, alpha=0.08, color='#2a9d5c', label='1σ KO Range')
ax.axvspan(ko_2s_dn, ko_2s_up, alpha=0.05, color='#d13841', label='2σ KO Range')

ax.annotate(f'Expected Move\n({ko_range:.1f})',
            xy=(spot, 1), xytext=(spot, 2.5),
            fontsize=9, ha='center', color='#1a3a5c', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlim(ko_2s_dn - 18, ko_2s_up + 18)
ax.set_ylim(-0.5, 3.5)
ax.set_yticks([])
ax.set_xlabel('Price', fontsize=10)
ax.set_title(f'KO Range — {dte}-Day Knock-Out Barriers (σ = {sigma:.1f}%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.grid(axis='x', alpha=0.15)
fig.tight_layout()
fig.savefig(f'{OUT}/08_ko_range.png', dpi=DPI)
plt.close(fig)
print('8/12 KO Range done')

# ════════════════════════════════════════════════
# 9. Z-score with Cornish-Fisher Correction
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor('#fafafa')

x = np.linspace(-4, 4, 300)
skew, ek = -0.4, 1.8

# CF correction
z_cf = x + skew/6 * (x**2 - 1) + ek/24 * (x**3 - 3*x) - skew**2/36 * (2*x**3 - 5*x)

pdf_normal = np.exp(-0.5 * x**2) / np.sqrt(2*np.pi)
pdf_cf = np.exp(-0.5 * z_cf**2) / np.sqrt(2*np.pi) * np.gradient(z_cf, x)

ax.plot(x, pdf_normal, '#888', lw=1.5, ls='--', label='Normal (Gaussian)', alpha=0.7)
ax.plot(x, pdf_cf, '#1a3a5c', lw=2.5, label=f'Cornish-Fisher\n(skew={skew}, kurtosis={ek})', zorder=5)

# Thresholds
for z_th, lbl, clr in [(2, '2σ (normal)', '#d4a017'), (3, '3σ (normal)', '#d13841')]:
    ax.axvline(z_th, color=clr, ls=':', lw=1.2, alpha=0.6)

cf_threshold = z_cf[np.searchsorted(x, 2)]
ax.axvline(cf_threshold, color='#d13841', ls='--', lw=2, label=f'CF 2σ → {cf_threshold:.1f}σ', zorder=5)
ax.annotate(f'With fat tails,\nthe 2σ threshold\nbecomes {cf_threshold:.1f}σ',
            xy=(cf_threshold, pdf_cf[np.searchsorted(x, cf_threshold)]*0.5),
            xytext=(cf_threshold+0.5, pdf_cf.max()*0.6),
            fontsize=8, color='#d13841', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d13841', lw=1.2),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('Standard Deviations (σ)', fontsize=10)
ax.set_ylabel('Probability Density', fontsize=10)
ax.set_title('Cornish-Fisher Correction — Adjusting Z-score for Fat Tails', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, framealpha=0.9)
ax.set_xlim(-4, 4)
ax.grid(alpha=0.12)
fig.tight_layout()
fig.savefig(f'{OUT}/09_zscore_cf.png', dpi=DPI)
plt.close(fig)
print('9/12 Z-score CF done')

# ════════════════════════════════════════════════
# 10. Risk Premia — λ₂ λ₃ λ₄
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 3.5))
fig.patch.set_facecolor('#fafafa')

lambdas = ['λ₂\n(Variance)', 'λ₃\n(Skew)', 'λ₄\n(Kurtosis)']
vals = [0.25, -0.35, 0.40]
errors = [0.08, 0.10, 0.12]
colors = [('#2a9d5c', '#1a5c3a'), ('#d13841', '#8b2020'), ('#d4a017', '#8a6b0a')]

bars = ax.bar(lambdas, vals, yerr=errors, capsize=6,
              color=[c[0] for c in colors], edgecolor=[c[1] for c in colors],
              linewidth=1.5, alpha=0.85, width=0.5, zorder=3)

ax.axhline(0, color='black', lw=0.8)

for bar, v, (_, ec) in zip(bars, vals, colors):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.03 if v > 0 else -0.08),
            f'{v:+.2f}', ha='center', fontsize=11, fontweight='bold', color=ec)

ax.annotate('Expensive puts:\nmarket fears tail event',
            xy=(1.5, -0.35), xytext=(2.3, -0.55),
            fontsize=8, color='#d13841', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d13841', lw=1.2),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.annotate('Vol risk premium:\nshort vol has edge',
            xy=(0.5, 0.22), xytext=(1, 0.55),
            fontsize=8, color='#2a9d5c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2a9d5c', lw=1.2),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_ylabel('Premium (implied − historical)', fontsize=10)
ax.set_title('Risk Premia Decomposition — λ₂, λ₃, λ₄ (Vázquez, Bergomi-Guyon)', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.15)
fig.tight_layout()
fig.savefig(f'{OUT}/10_risk_premia.png', dpi=DPI)
plt.close(fig)
print('10/12 Risk premia done')

# ════════════════════════════════════════════════
# 11. Macro GIP Matrix
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor('#fafafa')

regimes = [
    ['', 'Inflation +', 'Inflation −'],
    ['Growth +', 'GOLDILOCKS\nRisk-on\nLong equities', 'REFLATION\nCyclicals\nCommodities'],
    ['Growth −', 'STAGFLATION\nRisk-off\nHedge, reduce leverage', 'SLOWDOWN\nDefensive\nBonds, low-beta'],
]

colors_r = {
    'GOLDILOCKS': '#2a9d5c',
    'REFLATION': '#d4a017',
    'STAGFLATION': '#d13841',
    'SLOWDOWN': '#1a3a5c',
}

table_data = [[regimes[i][j] for j in range(3)] for i in range(3)]
col_labels = ['', 'HIGH Inflation', 'LOW Inflation']
row_labels = ['', 'HIGH Growth', 'LOW Growth']

# Draw manually
ax.axis('off')
cell_w, cell_h = 0.33, 0.33

for i in range(3):
    for j in range(3):
        x, y = j * cell_w, (2-i) * cell_h
        text = regimes[i][j]
        if text in colors_r:
            fc = colors_r[text]
            ec = fc
            tc = 'white'
        elif text in ['GOLDILOCKS', 'REFLATION', 'STAGFLATION', 'SLOWDOWN']:
            fc = '#e0e0e0'; ec = '#ccc'; tc = 'black'
        else:
            fc = '#f0f4f8'; ec = '#ccc'; tc = '#333'

        rect = mpatches.FancyBboxPatch((x, y), cell_w, cell_h,
                                         boxstyle="round,pad=0.08",
                                         facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)

        if text in colors_r:
            ax.text(x + cell_w/2, y + cell_h*0.6, text.split('\n')[0],
                    ha='center', va='center', fontsize=10, fontweight='bold', color=tc)
            for li, line in enumerate(text.split('\n')[1:], 1):
                ax.text(x + cell_w/2, y + cell_h*(0.6 - li*0.2),
                        line, ha='center', va='center', fontsize=7.5, color=tc)
        else:
            ax.text(x + cell_w/2, y + cell_h/2, text,
                    ha='center', va='center', fontsize=9, color=tc, fontweight='bold' if i==0 or j==0 else 'normal')

ax.set_xlim(0, 3*cell_w)
ax.set_ylim(0, 3*cell_h)
ax.set_title('GIP Macro Regime Matrix\n(Growth / Inflation / Policy)', fontsize=12, fontweight='bold', pad=10)
fig.tight_layout()
fig.savefig(f'{OUT}/11_macro_gip.png', dpi=DPI)
plt.close(fig)
print('11/12 Macro GIP done')

# ════════════════════════════════════════════════
# 12. Architecture Data Flow
# ════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')
fig.patch.set_facecolor('#fafafa')

def dbox(x, y, w, h, text, fc='#1a3a5c', fs=9, ec='white', tc='white'):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                  facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color=tc, fontweight='bold')

def dlink(x1, y1, x2, y2, label='', color='#999'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if label:
        ax.text((x1+x2)/2 + 0.2, (y1+y2)/2, label, fontsize=7, color=color, fontweight='bold')

# Row 0: Sources
dbox(1, 4.8, 2.5, 0.8, 'TRADIER\nOptions chain real-time', '#0d2137', 8)
dbox(4.5, 4.8, 2.5, 0.8, 'GEXBot\nInstitutional GEX/ZG', '#0d2137', 8)
dbox(8, 4.8, 2.5, 0.8, 'YAHOO FINANCE\nFallback / history', '#0d2137', 8)
dlink(2.25, 4.8, 3.5, 3.8, '')
dlink(5.75, 4.8, 5.5, 3.8, '')
dlink(9.25, 4.8, 7.5, 3.8, '')

# Row 1: Processing
dbox(2, 3.2, 7.5, 1, 'quant_analytics.py\nGEX | B&L quantiles | Z-score | ES | λ premia | Adapted VaR | Confluence\n30s refresh cycle', '#2a5a8c', 8)
dlink(5.75, 3.2, 5.75, 2.2, 'Metrics')

# Row 2: Terminal
dbox(2, 1.5, 7.5, 1, 'TERMINAL (Streamlit — port 8506)\nCol SX: GEX, Bands, Greeks  |  Col Centro: Macro, Flow  |  Col DX: Levels, SL, Confluence', '#1a3a5c', 8)

# Row 3: Persistence
dbox(2, 0.3, 3, 0.7, 'SQLite\nmaze_alerts.db', '#555', 8)
dbox(6, 0.3, 3.5, 0.7, 'HTML Guide\nMaze_Terminal_Guide.html', '#555', 8)
dlink(5.75, 1.5, 3.5, 0.65, 'Alerts')
dlink(5.75, 1.5, 7.75, 0.65, 'Docs')

ax.set_title('MAZE CAPITAL TERMINAL — System Architecture', fontsize=13, fontweight='bold', pad=8, color='#0d2137')
fig.tight_layout()
fig.savefig(f'{OUT}/12_architecture.png', dpi=DPI)
plt.close(fig)
print('12/12 Architecture done')

# ────────────────────────────────────────
print(f'\nAll 12 charts saved to {OUT}')
print(f'Run: python {os.path.basename(__file__)}')
