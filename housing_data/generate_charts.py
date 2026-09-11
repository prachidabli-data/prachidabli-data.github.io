"""
Generate the Python/Matplotlib charts used in the housing case study, from
the real generated data in this folder. No placeholders: every chart below
is produced directly from fact_tenders.csv and fact_contracts.csv.

Run with:
    python3 housing_data/generate_charts.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

NAVY = "#16294A"
NAVY_LIGHT = "#33547F"
AMBER = "#E0A030"
AMBER_DARK = "#BD831C"
MUTED = "#8A93A6"
LINE = "#E7DFD0"
CREAM = "#FAF6EE"

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

IMG = "housing_data/images"

ft = pd.read_csv("housing_data/fact_tenders.csv")
fc = pd.read_csv("housing_data/fact_contracts.csv")
dr = pd.read_csv("housing_data/dim_region.csv")
dc = pd.read_csv("housing_data/dim_contractor.csv")
ds = pd.read_csv("housing_data/dim_status.csv")

# ---------------------------------------------------------------------------
# 1. Funnel
# ---------------------------------------------------------------------------
stages = ["Identified", "Reached\nBidding", "Reached\nSubmitted", "Won"]
counts = [
    len(ft),
    (ft.reached_stage_id >= 2).sum(),
    (ft.reached_stage_id >= 3).sum(),
    ft.is_won.sum(),
]

fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(stages))
bars = ax.barh(y, counts, color=AMBER, height=0.55, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(stages)
ax.invert_yaxis()
ax.set_xlim(0, max(counts) * 1.18)
for yi, c in zip(y, counts):
    ax.text(c + max(counts) * 0.015, yi, f"{c:,}", va="center", fontsize=13, fontweight="bold", color=NAVY)
ax.set_title("The bid funnel narrows at every stage", fontsize=15, fontweight="bold", color=NAVY, loc="left", pad=14)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.xaxis.set_visible(False)
fig.tight_layout()
fig.savefig(f"{IMG}/01_funnel.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Monthly trend (bid volume + 3-month rolling win rate), small multiples
# ---------------------------------------------------------------------------
ft["id_month"] = ft.identified_date_key.str[:7]
months = sorted(ft.id_month.unique())
count_by_month = ft.groupby("id_month").size().reindex(months)
dec = ft[ft.is_decided == 1]
win_by_month = (dec.groupby("id_month").is_won.sum() / dec.groupby("id_month").is_decided.sum() * 100).reindex(months)
win_3m = win_by_month.rolling(3, min_periods=1).mean()
labels = [pd.Period(m).strftime("%b %y") for m in months]
tick_idx = list(range(0, len(months), 3))

fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)

ax = axes[0]
ax.bar(range(len(months)), count_by_month.values, color=NAVY_LIGHT, width=0.65, zorder=3)
ax.set_title("Bid volume surges into each financial year end, dips over summer",
             fontsize=13.5, fontweight="bold", color=NAVY, loc="left")
ax.set_ylabel("Tenders identified")
ax.grid(axis="y", color=LINE, linewidth=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.plot(range(len(months)), win_3m.values, color=AMBER_DARK, linewidth=2.5, zorder=3)
ax.fill_between(range(len(months)), win_3m.values, color=AMBER, alpha=0.15, zorder=2)
ax.set_title("Win rate (3-month rolling) climbs through late 2025",
             fontsize=13.5, fontweight="bold", color=NAVY, loc="left")
ax.set_ylabel("Win rate, %")
ax.grid(axis="y", color=LINE, linewidth=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())

ax.set_xticks(tick_idx)
ax.set_xticklabels([labels[i] for i in tick_idx], rotation=0)

fig.tight_layout()
fig.savefig(f"{IMG}/02_monthly_trend.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Region: volume vs. average value (small multiples, one hue each)
# ---------------------------------------------------------------------------
ftr = ft.merge(dr, on="region_id")
vol = ftr.groupby("region_short").size().sort_values(ascending=False)
avgval = ftr.groupby("region_short").bid_value.mean().reindex(vol.index) / 1000

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
ax.bar(vol.index, vol.values, color=NAVY_LIGHT, zorder=3)
ax.set_title("Bid volume by region", fontsize=13.5, fontweight="bold", color=NAVY, loc="left")
ax.set_ylabel("Tenders")
ax.grid(axis="y", color=LINE, linewidth=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", rotation=35)

ax = axes[1]
ax.bar(avgval.index, avgval.values, color=AMBER, zorder=3)
ax.set_title("Average bid value by region", fontsize=13.5, fontweight="bold", color=NAVY, loc="left")
ax.set_ylabel("Average bid value, £000s")
ax.grid(axis="y", color=LINE, linewidth=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", rotation=35)

fig.suptitle("Northern and midlands regions win on volume; London and the South East win on value",
             fontsize=13, color="#5C6579", y=1.03)
fig.tight_layout()
fig.savefig(f"{IMG}/03_region.png", dpi=150, facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Contractor league table (on-time %, coloured by tier)
# ---------------------------------------------------------------------------
fcc = fc.merge(dc, on="contractor_id")
league = (fcc.groupby(["contractor_name", "tier"]).is_on_time.mean() * 100).reset_index()
league = league.sort_values("is_on_time", ascending=True)

TIER_COLOR = {"Tier 1": NAVY, "Tier 2": AMBER, "Tier 3": MUTED}
colors = league.tier.map(TIER_COLOR)

fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(league))
ax.barh(y, league.is_on_time, color=colors, height=0.62, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(league.contractor_name)
for yi, v in zip(y, league.is_on_time):
    ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=10.5, color=NAVY)
ax.set_xlim(0, 105)
ax.set_title("Contractor league table, ranked by on-time delivery", fontsize=14.5, fontweight="bold", color=NAVY, loc="left", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=LINE, linewidth=0.8, zorder=0)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER_COLOR.values()]
ax.legend(handles, TIER_COLOR.keys(), loc="lower right", frameon=False)

fig.tight_layout()
fig.savefig(f"{IMG}/04_contractor_league.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Contract status breakdown
# ---------------------------------------------------------------------------
statmap = dict(zip(ds.status_id, ds.status_name))
fc["status_name"] = fc.status_id.map(statmap)
status_counts = fc.status_name.value_counts()
STATUS_COLOR = {"Completed": NAVY, "Active": NAVY_LIGHT, "At-risk": AMBER_DARK, "On-hold": MUTED}
order = ["Completed", "Active", "At-risk", "On-hold"]
status_counts = status_counts.reindex(order).fillna(0)

fig, ax = plt.subplots(figsize=(8, 3.6))
y = np.arange(len(order))
ax.barh(y, status_counts.values, color=[STATUS_COLOR[s] for s in order], height=0.55, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.invert_yaxis()
for yi, v in zip(y, status_counts.values):
    ax.text(v + 3, yi, f"{int(v)}", va="center", fontsize=12, fontweight="bold", color=NAVY)
ax.set_xlim(0, status_counts.max() * 1.2)
ax.set_title("Contract status today (413 contracts)", fontsize=14.5, fontweight="bold", color=NAVY, loc="left", pad=12)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.xaxis.set_visible(False)
fig.tight_layout()
fig.savefig(f"{IMG}/05_status.png", dpi=150, facecolor="white")
plt.close(fig)

print("Saved 5 charts to", IMG)
