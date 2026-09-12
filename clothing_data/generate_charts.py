"""
Generate the two management-analytics charts used in the Trueform Clothing
case study, from competitor_positioning.csv.

That CSV is a structured, ordinal read of the competitive analysis in the
original report (Section 4, Competitive Environment & Positioning): each
retailer's price tier and adaptive specialisation is scored 1 to 5 from the
report's own comparative claims, cited in the note column. No comparison
here is invented; each score traces back to a specific sentence in the
report and its reference.

Run with:
    python3 clothing_data/generate_charts.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#16294A"
NAVY_LIGHT = "#33547F"
AMBER = "#E0A030"
AMBER_DARK = "#BD831C"
MUTED = "#8A93A6"
LINE = "#E7DFD0"

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": LINE,
    "axes.labelcolor": "#5C6579",
    "text.color": NAVY,
    "xtick.color": "#5C6579",
    "ytick.color": "#5C6579",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

IMG = "clothing_data/images"
df = pd.read_csv("clothing_data/competitor_positioning.csv")

# ---------------------------------------------------------------------------
# 1. Competitive positioning map: price tier vs adaptive specialisation
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 7))

for _, row in df.iterrows():
    is_trueform = row.retailer == "Trueform Clothing"
    color = AMBER if is_trueform else NAVY_LIGHT
    size = 340 if is_trueform else 220
    ax.scatter(row.price_tier, row.adaptive_specialisation, s=size, color=color,
               zorder=4, edgecolor="white", linewidth=1.5)
    dx, dy = 0.12, 0.12
    ax.annotate(row.retailer, (row.price_tier, row.adaptive_specialisation),
                xytext=(row.price_tier + dx, row.adaptive_specialisation + dy),
                fontsize=11.5, fontweight=("bold" if is_trueform else "normal"),
                color=(AMBER_DARK if is_trueform else NAVY))

ax.axvline(3, color=LINE, linewidth=1, zorder=1)
ax.axhline(3, color=LINE, linewidth=1, zorder=1)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.5, 5.8)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["Budget", "", "Mid-market", "", "Premium"])
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["None", "", "Some", "", "Full specialist"])
ax.set_xlabel("Price positioning")
ax.set_ylabel("Adaptive specialisation")
ax.set_title("The only retailer in the specialist, premium quadrant",
              fontsize=14, fontweight="bold", color=NAVY, loc="left", pad=14)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{IMG}/01_positioning_map.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Pricing ladder
# ---------------------------------------------------------------------------
d = df.sort_values("price_tier")
colors = [AMBER if r == "Trueform Clothing" else NAVY_LIGHT for r in d.retailer]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
y = range(len(d))
ax.barh(y, d.price_tier, color=colors, height=0.58, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(d.retailer)
ax.set_xlim(0, 5.6)
ax.set_xticks([1, 3, 5])
ax.set_xticklabels(["Budget", "Mid-market", "Premium"])
ax.set_title("Price positioning against direct competitors",
             fontsize=14, fontweight="bold", color=NAVY, loc="left", pad=12)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", color=LINE, linewidth=0.8, zorder=0)
fig.tight_layout()
fig.savefig(f"{IMG}/02_pricing_ladder.png", dpi=150, facecolor="white")
plt.close(fig)

print("Saved 2 charts to", IMG)
