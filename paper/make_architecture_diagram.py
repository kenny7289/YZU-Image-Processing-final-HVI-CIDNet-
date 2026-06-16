import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.5)
ax.axis('off')


def box(x, y, w, h, text, fc='#e8f0fe', ec='#1a73e8', fontsize=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                        linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize)
    return (x, y, w, h)


def arrow(p1, p2, **kw):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=14,
                         linewidth=1.2, color='#444444', **kw)
    ax.add_patch(a)


# Input / output
in_box = box(0.2, 2.8, 1.6, 1.0, 'RGB\nlow-light\ninput', fc='#fff3cd', ec='#b08800')
out_box = box(10.2, 2.8, 1.6, 1.0, 'RGB\nenhanced\noutput', fc='#d4edda', ec='#1e7e34')

# HVI transform
hvi_fwd = box(2.1, 2.8, 1.5, 1.0, 'RGB$\\to$HVI\n(HVIT)', fc='#e2d9f3', ec='#6f42c1')
hvi_inv = box(8.4, 2.8, 1.5, 1.0, 'HVI$\\to$RGB\n(PHVIT)', fc='#e2d9f3', ec='#6f42c1')

# I branch (top)
i_enc = box(3.9, 4.4, 2.0, 0.9, 'I-branch encoder\n(Intensity UNet)', fc='#e8f0fe', ec='#1a73e8')
i_dec = box(6.3, 4.4, 2.0, 0.9, 'I-branch decoder\n(Intensity UNet)', fc='#e8f0fe', ec='#1a73e8')

# HV branch (bottom)
hv_enc = box(3.9, 1.1, 2.0, 0.9, 'HV-branch encoder\n(Color UNet)', fc='#fde2e2', ec='#c5221f')
hv_dec = box(6.3, 1.1, 2.0, 0.9, 'HV-branch decoder\n(Color UNet)', fc='#fde2e2', ec='#c5221f')

# LCA cross-attention
lca = box(5.5, 2.85, 1.0, 0.95, 'LCA\ncross-\nattention\n($\\times$6)', fc='#fff0e0', ec='#e8710a', fontsize=8)

# main flow arrows
arrow((1.8, 3.3), (2.1, 3.3))
arrow((3.6, 3.3), (3.9, 3.3 + 0.85))   # to I encoder (slightly up)
arrow((3.6, 3.3), (3.9, 1.55))         # to HV encoder
arrow((5.9, 4.85), (6.3, 4.85))
arrow((5.9, 1.55), (6.3, 1.55))
arrow((8.3, 4.85), (8.9, 3.6))
arrow((8.3, 1.55), (8.9, 3.2))
arrow((9.9, 3.3), (10.2, 3.3))

# LCA bidirectional links to both branches
arrow((5.7, 4.4), (5.8, 3.8), linestyle='dashed')
arrow((5.8, 2.85), (5.7, 2.0), linestyle='dashed')
arrow((5.8, 3.6), (6.0, 4.4), linestyle='dashed')
arrow((6.0, 1.1), (5.8, 2.6), linestyle='dashed')

ax.text(6, 5.7, 'Dual-branch CIDNet (Intensity branch + Color/HV branch with Lighten Cross-Attention)',
        ha='center', fontsize=10, weight='bold')

plt.tight_layout()
out_path = os.path.join(FIG_DIR, 'architecture_diagram.png')
plt.savefig(out_path, dpi=200)
print('Saved:', out_path)
