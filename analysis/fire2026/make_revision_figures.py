"""
Figures for the revised manuscript.

Style matches the two existing figures deliberately: Okabe-Ito colours, distinct
markers AND line styles for every series, so the panels remain readable in
greyscale and under colour-vision deficiency (Rule 19).

  fig3_delta_across_loss   architecture contrast against realised validation loss.
                           Zero is drawn heavily because the scientific point is
                           that the contrast crosses it.
  fig4_scorer_and_category two panels: the same contrasts under both scoring
                           formulations, and the per-category spread that the
                           aggregate hides.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/home/Debz/Research/HyperloopBert"
OUT = os.path.join(ROOT, "Submission", "images")
DAT = os.path.join(ROOT, "analysis/fire2026/out")
os.makedirs(OUT, exist_ok=True)

STYLE = {
    "Vanilla vs Looped":    dict(color="#0072B2", marker="s", ls="--"),
    "Vanilla vs ALBERT":    dict(color="#D55E00", marker="^", ls="-."),
    "Vanilla vs Hyperloop": dict(color="#009E73", marker="D", ls=":"),
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# ------------------------------------------------- Fig 3: contrast across loss
ab = pd.read_csv(os.path.join(DAT, "allband_contrasts.csv"))
# collapse bands that resolve to the same pair of snapshots
ab = ab.drop_duplicates(subset=["Contrast", "Loss_Vanilla", "Loss_Other"])

fig, ax = plt.subplots(figsize=(3.34, 2.45))
for name, s in STYLE.items():
    g = ab[ab.Contrast == name].sort_values("Loss_Vanilla", ascending=False)
    if g.empty:
        continue
    x = (g.Loss_Vanilla + g.Loss_Other) / 2      # midpoint of the matched pair
    ax.errorbar(x, g.Delta,
                yerr=[g.Delta - g.CI_Low, g.CI_High - g.Delta],
                color=s["color"], marker=s["marker"], ls=s["ls"],
                lw=1.1, ms=3.4, capsize=2, elinewidth=0.7, label=name, zorder=3)

ax.axhline(0.0, color="black", lw=1.0, zorder=2)          # the point of the figure
ax.invert_xaxis()
ax.set_xlabel("Matched validation loss (nats)")
ax.set_ylabel(r"Contrast $\Delta$ (unshared $-$ shared)")
ax.legend(loc="upper center", bbox_to_anchor=(0.47, -0.30), ncol=1,
          frameon=False, handlelength=2.2)
fig.savefig(os.path.join(OUT, "fig3_delta_across_loss.pdf"))
fig.savefig(os.path.join(OUT, "fig3_delta_across_loss.png"))
plt.close(fig)
print("wrote fig3_delta_across_loss  (points: "
      + ", ".join(f"{k.split()[-1]}={len(ab[ab.Contrast==k])}" for k in STYLE) + ")")

# --------------------------------- Fig 4: scorer sensitivity + category spread
full = pd.read_csv(os.path.join(DAT, "contrasts_band2.2.csv"))
chg = pd.read_csv(os.path.join(DAT, "contrasts_changed_token_scorer.csv"))
cat = pd.read_csv(os.path.join(DAT, "per_category_contrasts.csv"))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))

# (a) same contrasts, two scoring formulations
ax = axes[0]
names = ["Vanilla vs Looped", "Vanilla vs ALBERT", "Vanilla vs Hyperloop"]
y = np.arange(len(names))
f = full.set_index("Contrast").reindex(names)
c = chg.set_index("Contrast").reindex(names)
ax.errorbar(f.Delta, y + 0.13,
            xerr=[f.Delta - f.CI_Low, f.CI_High - f.Delta],
            fmt="o", color="#000000", ms=3.6, capsize=2, elinewidth=0.8,
            label="full-sentence PLL", zorder=3)
ax.errorbar(c.Delta, y - 0.13,
            xerr=[c.Delta - c.CI_Low, c.CI_High - c.Delta],
            fmt="s", color="#CC79A7", ms=3.6, capsize=2, elinewidth=0.8,
            mfc="none", label="changed-token", zorder=3)
ax.axvline(0.0, color="black", lw=1.0, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels([n.replace("Vanilla vs ", "vs ") for n in names])
ax.set_xlabel(r"Contrast $\Delta$ at the matched point")
ax.set_title("(a) scoring formulation", loc="left", fontsize=8)
ax.invert_yaxis()               # read top-to-bottom in the same order as the tables
ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.02, 0.98))
ax.margins(y=0.22)

# (b) per-category contrast, showing sign changes
ax = axes[1]
order = (cat[cat.Contrast == "Vanilla vs Looped"]
         .sort_values("Delta").Category.tolist())
for name, s in STYLE.items():
    g = cat[cat.Contrast == name].set_index("Category").reindex(order)
    ax.plot(g.Delta.values, np.arange(len(order)), color=s["color"],
            marker=s["marker"], ls="none", ms=3.6, label=name, zorder=3)
ax.axvline(0.0, color="black", lw=1.0, zorder=2)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels(order)
ax.set_xlabel(r"Per-category contrast $\Delta$")
ax.set_title("(b) bias category", loc="left", fontsize=8)
ax.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, -0.02))
ax.margins(y=0.06)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_scorer_and_category.pdf"))
fig.savefig(os.path.join(OUT, "fig4_scorer_and_category.png"))
plt.close(fig)
print("wrote fig4_scorer_and_category")
