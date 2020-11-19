import matplotlib.pyplot as plt
from seaborn import distplot
import numpy as np

##Fig. 3 LayerNumber
layernumer = np.array([3, 6, 8, 12, 24])
x = np.arange(0, 5)
gat_acc = np.array([82.35, 65.00, 57.85, 33.20, 30.60])
gdt_acc = np.array([85.28, 85.60, 85.70, 85.33, 85.60])
diff_gcn_acc = np.array([83.1, 82.4, 46.7, 34.6, 33.4])
f = plt.figure(figsize=(10.5,9.6))
ax = plt.subplot()
bar_width = 0.2
opacity = 0.8

rects1 = ax.bar(x, gdt_acc, bar_width,
alpha=opacity,
label='MAGNA')

rects2 = ax.bar(x + 2*bar_width, gat_acc, bar_width,
alpha=opacity,
label='GAT')

rects3 = ax.bar(x + bar_width, diff_gcn_acc, bar_width,
alpha=opacity,
label='Diff-GCN (PPR)')
ax.set_ylim([20, 95])
ax.set_yticks([20, 40, 60, 80, 90])
ax.set_yticklabels([20, 40, 60, 80, 90])
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_xlabel('Depth of GNN layers', fontsize=30)
ax.set_ylabel('Accuracy (%)', fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels(layernumer)
ax.legend(fontsize=23, ncol=3, frameon=False)

plt.tight_layout()
plt.show()
f.savefig('diff_gcn_depth_cora.pdf', bbox_inches='tight', dpi=100)
