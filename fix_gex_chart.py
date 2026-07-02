import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = r'C:\Users\Gabriel\Desktop\2nd-brain\.claude\guide_images'

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#fafafa')

strikes = np.arange(650, 770, 2.5)
spot = 708

# Realistic GEX profile
rng = np.random.default_rng(42)

gex = np.zeros_like(strikes)
for i, k in enumerate(strikes):
    if k < 670:
        gex[i] = -rng.uniform(0.3, 0.8)   # deep OTM puts
    elif k < 685:
        gex[i] = -rng.uniform(0.8, 2.0)   # put wall building
    elif k < 695:
        gex[i] = -rng.uniform(1.5, 3.5)   # max put concentration
    elif k < 705:
        gex[i] = -rng.uniform(0.5, 2.0)   # near ATM puts
    elif k < 715:
        gex[i] = rng.uniform(0.3, 1.5)    # ATM calls (positive gamma)
    elif k < 730:
        gex[i] = rng.uniform(0.5, 1.0)    # OTM calls
    elif k < 745:
        gex[i] = rng.uniform(0.2, 0.6)    # further OTM
    else:
        gex[i] = -rng.uniform(0.5, 1.0)   # small call wall tail

# Big call wall at 750
wall_idx = np.argmin(np.abs(strikes - 750))
gex[wall_idx] = -12.0
gex[wall_idx-1] = -4.0
gex[wall_idx+1] = -3.0

# Smooth a bit
from scipy.ndimage import uniform_filter1d
gex = uniform_filter1d(gex, size=3)

# Bars
colors = ['#d13841' if g < 0 else '#2a9d5c' for g in gex]
ax1.bar(strikes, gex, width=2.2, color=colors, alpha=0.85, zorder=3)

# Cumulative
cum = np.cumsum(gex)
zg_idx = np.argmin(np.abs(cum))
zg = strikes[zg_idx]

ax2 = ax1.twinx()
ax2.plot(strikes, cum, '#1a3a5c', lw=2.5, zorder=4, label='Cumulative GEX')
ax2.axhline(0, color='#1a3a5c', lw=0.8, ls='-', alpha=0.4)

# ZG
ax2.axvline(zg, color='#1a3a5c', ls='--', lw=2, zorder=5)
ax2.annotate(f'Gamma Flip @ {zg:.0f}',
             xy=(zg, 0), xytext=(zg + 8, cum.max() * 0.6),
             fontsize=10, color='#1a3a5c', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#1a3a5c', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1a3a5c', alpha=0.9))

# Spot
ax1.axvline(spot, color='orange', ls=':', lw=2, zorder=5)
ax1.text(spot + 0.5, ax1.get_ylim()[1] * 0.92, f'Spot ${spot}', fontsize=9, color='orange', fontweight='bold')

# Max Pain
max_pain = strikes[np.argmax(np.abs(gex))]
ax1.axvline(max_pain, color='#8e44ad', ls=':', lw=1.5, zorder=5)
ax1.text(max_pain + 0.5, ax1.get_ylim()[1] * 0.82, f'Call Wall\n{max_pain:.0f}', fontsize=8, color='#8e44ad', fontweight='bold')

# Labels
ax1.set_xlabel('Strike Price', fontsize=10)
ax1.set_ylabel('GEX ($ γ per 1% move)', color='#555', fontsize=9)
ax2.set_ylabel('Cumulative GEX', color='#1a3a5c', fontsize=9)
ax1.set_title('Gamma Exposure Profile — Zero Gamma (ZG) & Call Wall', fontsize=13, fontweight='bold', pad=10)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d13841', alpha=0.85, label='Negative γ (amplifying)'),
    Patch(facecolor='#2a9d5c', alpha=0.85, label='Positive γ (stabilising)'),
    plt.Line2D([0], [0], color='#1a3a5c', lw=2.5, label='Cumulative GEX'),
]
ax1.legend(handles=legend_elements, fontsize=8, loc='upper left', framealpha=0.9)

ax1.grid(axis='x', alpha=0.15)
ax1.set_xlim(645, 770)
fig.tight_layout()
fig.savefig(f'{OUT}/04_zg_gex.png', dpi=180)
plt.close(fig)
print('ZG/GEX chart fixed')
