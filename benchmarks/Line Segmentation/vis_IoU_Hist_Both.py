import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


def load_protocol_ious(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    ious = []
    for img in data.get('per_image', []):
        matched_ious = img.get('per_line_iou', [])
        num_gt = img.get('num_gt', 0)
        num_pred = img.get('num_pred', 0)
        num_matched = len(matched_ious)

        ious.extend(matched_ious)

        num_unmatched = (num_gt - num_matched) + (num_pred - num_matched)
        if num_unmatched > 0:
            ious.extend([0.0] * num_unmatched)
    return np.array(ious)


print("Loading data...")
ious_p1_all = load_protocol_ious('evaluation_protocol1.json')
ious_p2_all = load_protocol_ious('evaluation_protocol2.json')


# --- VERIFICATION PRINTS ---
def print_stats(name, ious_all):
    ious_matched = ious_all[ious_all > 0]
    m_iou_all = np.mean(ious_all)
    m_iou_m = np.mean(ious_matched)
    total_lines = len(ious_all)
    matched_lines = len(ious_matched)
    pct_75 = (ious_all >= 0.75).mean() * 100

    print(f"\n--- {name} ---")
    print(f"Total Lines (GT + Pred) : {total_lines:,}")
    print(f"Matched Lines (IoU > 0) : {matched_lines:,}")
    print(f"mIoU(all) [With Zeros]  : {m_iou_all:.4f}  <-- Must match Table 8")
    print(f"mIoU(m)   [Matched Only]: {m_iou_m:.4f}  <-- Must match Table 8")
    print(f"IoU >= 0.75 (Over All)  : {pct_75:.2f}%")


print_stats("Protocol 1 (All 500 Writers)", ious_p1_all)
print_stats("Protocol 2 (Official Test Set)", ious_p2_all)
print("\n")

# --- PLOTTING ---
ious_p1_plot = ious_p1_all[ious_p1_all > 0]
ious_p2_plot = ious_p2_all[ious_p2_all > 0]

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

protocols = [
    {'data_plot': ious_p1_plot, 'title': '(a) Line-Level IoU: Evaluation Protocol 1', 'color': '#005F73',
     'kde_color': '#0A9396', 'ax': axs[0]},
    {'data_plot': ious_p2_plot, 'title': '(b) Line-Level IoU: Evaluation Protocol 2', 'color': '#2A9D8F',
     'kde_color': '#E76F51', 'ax': axs[1]}
]

MEAN_COLOR = '#CA6702'
THRESHOLD_COLOR = '#AE2012'

for p in protocols:
    ax = p['ax']
    ious = p['data_plot']

    n, bins, patches = ax.hist(ious, bins=35, range=(0.0, 1.0), density=True,
                               color=p['color'], alpha=0.55, edgecolor='white', linewidth=0.8,
                               label='Matched IoU Frequency')

    kde = gaussian_kde(ious)
    x_grid = np.linspace(0.01, 1, 300)
    kde_y = kde(x_grid)
    ax.plot(x_grid, kde_y, color=p['kde_color'], linewidth=2.5, label='KDE Density')

    mean_matched = np.mean(ious)
    ax.axvline(mean_matched, color=MEAN_COLOR, linestyle='--', linewidth=1.8,
               label=f'Mean Matched IoU ({mean_matched:.3f})')
    ax.axvspan(0.75, 1.0, color='#E9D8A6', alpha=0.25, label='High Precision Zone (>=0.75)')

    ax.set_title(p['title'], fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Intersection over Union (IoU)', fontsize=10, fontweight='bold', labelpad=6)
    ax.set_ylabel('Probability Density (Matched Lines)', fontsize=10, fontweight='bold', labelpad=6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, max(n.max(), kde_y.max()) * 1.15)
    ax.grid(True, linestyle='--', alpha=0.4)

    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)

plt.tight_layout()
plt.savefig('protocol_comparison_scientific.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
