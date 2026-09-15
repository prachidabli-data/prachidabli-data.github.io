"""
Generate the Matplotlib charts used in the Infrastructure Commercial &
Financial Analytics case study, from the real generated data in this
folder. No placeholders: every chart below is produced directly from
fact_evm.csv and fact_procurement.csv.

This is the Python reproduction of an analysis originally built in Power
BI on real, confidential client data. The charts and numbers here are
computed from the synthetic dataset only.

Run with:
    python3 infrastructure_data/generate_charts.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

NAVY = "#16294A"
NAVY_LIGHT = "#33547F"
AMBER = "#E0A030"
AMBER_DARK = "#BD831C"
MUTED = "#8A93A6"
LINE = "#E7DFD0"
GOOD = "#3F7D5A"
BAD = "#B23A3A"

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

IMG = "infrastructure_data/images"

fe = pd.read_csv("infrastructure_data/fact_evm.csv")
fp = pd.read_csv("infrastructure_data/fact_procurement.csv")
dp = pd.read_csv("infrastructure_data/dim_project.csv")
dc = pd.read_csv("infrastructure_data/dim_cost_category.csv")

latest = fe.sort_values("date_key").groupby("project_id").tail(1).merge(
    dp[["project_id", "project_name", "budget_at_completion"]], on="project_id")
latest["cpi"] = latest.earned_value / latest.actual_cost
latest["spi"] = latest.earned_value / latest.planned_value
latest["cv"] = latest.earned_value - latest.actual_cost

# ---------------------------------------------------------------------------
# 1. Portfolio EVM S-curve
# ---------------------------------------------------------------------------
months = sorted(fe.date_key.unique())
port = []
for proj_id, g in fe.groupby("project_id"):
    g = g.sort_values("date_key").set_index("date_key")
    g = g.reindex(months).ffill()
    port.append(g[["planned_value", "earned_value", "actual_cost"]].fillna(0))
port_sum = sum(port)
port_sum.index = pd.to_datetime(months)

fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(port_sum.index, port_sum.planned_value / 1e6, color=NAVY, linewidth=2.5, label="Planned Value")
ax.plot(port_sum.index, port_sum.earned_value / 1e6, color=AMBER_DARK, linewidth=2.5, label="Earned Value")
ax.plot(port_sum.index, port_sum.actual_cost / 1e6, color=BAD, linewidth=2.5, linestyle="--", label="Actual Cost")
ax.set_ylabel("Cumulative £m")
ax.set_title("The portfolio is running behind plan and slightly over cost",
              fontsize=14.5, fontweight="bold", color=NAVY, loc="left", pad=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=LINE, linewidth=0.8, zorder=0)
ax.legend(loc="upper left", frameon=False, fontsize=11.5)
fig.tight_layout()
fig.savefig(f"{IMG}/01_evm_scurve.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. CPI by project
# ---------------------------------------------------------------------------
d = latest.sort_values("cpi")
colors = [GOOD if v >= 1 else BAD for v in d.cpi]
fig, ax = plt.subplots(figsize=(10.5, 7.5))
y = np.arange(len(d))
ax.barh(y, d.cpi, color=colors, height=0.62, zorder=3)
ax.axvline(1.0, color=NAVY, linewidth=1.2, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels(d.project_name, fontsize=10.5)
ax.set_xlabel("Cost Performance Index (CPI)")
ax.set_title(f"{(d.cpi<1).sum()} of {len(d)} projects are over budget",
             fontsize=14, fontweight="bold", color=NAVY, loc="left", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=LINE, linewidth=0.8, zorder=0)
fig.tight_layout()
fig.savefig(f"{IMG}/02_cpi_by_project.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. SPI by project
# ---------------------------------------------------------------------------
d = latest.sort_values("spi")
colors = [GOOD if v >= 1 else BAD for v in d.spi]
fig, ax = plt.subplots(figsize=(10.5, 7.5))
y = np.arange(len(d))
ax.barh(y, d.spi, color=colors, height=0.62, zorder=3)
ax.axvline(1.0, color=NAVY, linewidth=1.2, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels(d.project_name, fontsize=10.5)
ax.set_xlabel("Schedule Performance Index (SPI)")
ax.set_title(f"{(d.spi<1).sum()} of {len(d)} projects are behind schedule",
             fontsize=14, fontweight="bold", color=NAVY, loc="left", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=LINE, linewidth=0.8, zorder=0)
fig.tight_layout()
fig.savefig(f"{IMG}/03_spi_by_project.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Cost variance by project
# ---------------------------------------------------------------------------
d = latest.sort_values("cv")
colors = [GOOD if v >= 0 else BAD for v in d.cv]
fig, ax = plt.subplots(figsize=(10.5, 7.5))
y = np.arange(len(d))
ax.barh(y, d.cv / 1e6, color=colors, height=0.62, zorder=3)
ax.axvline(0, color=NAVY, linewidth=1.2, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels(d.project_name, fontsize=10.5)
ax.set_xlabel("Cost Variance, £m (Earned Value minus Actual Cost)")
ax.set_title("Cost variance by project", fontsize=14, fontweight="bold", color=NAVY, loc="left", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=LINE, linewidth=0.8, zorder=0)
fig.tight_layout()
fig.savefig(f"{IMG}/04_cost_variance.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Procurement spend by cost category
# ---------------------------------------------------------------------------
cat_spend = fp.merge(dc, on="category_id").groupby("category_name").actual_spend.sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.bar(cat_spend.index, cat_spend.values / 1e6, color=NAVY_LIGHT, zorder=3)
ax.set_ylabel("Actual spend, £m")
ax.set_title("Procurement spend by cost category",
             fontsize=14, fontweight="bold", color=NAVY, loc="left", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=LINE, linewidth=0.8, zorder=0)
fig.tight_layout()
fig.savefig(f"{IMG}/05_procurement_by_category.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Savings vs baseline, by cost category
# ---------------------------------------------------------------------------
cat_savings = fp.merge(dc, on="category_id").groupby("category_name").agg(
    savings=("savings", "sum"), baseline=("baseline_estimate", "sum")).reset_index()
cat_savings["savings_pct"] = cat_savings.savings / cat_savings.baseline * 100
cat_savings = cat_savings.sort_values("savings_pct")
colors = [GOOD if v >= 0 else BAD for v in cat_savings.savings_pct]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
y = np.arange(len(cat_savings))
ax.barh(y, cat_savings.savings_pct, color=colors, height=0.58, zorder=3)
ax.axvline(0, color=NAVY, linewidth=1.2, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels(cat_savings.category_name)
ax.set_xlabel("Savings against baseline, %")
ax.set_title("Procurement savings by cost category",
             fontsize=14, fontweight="bold", color=NAVY, loc="left", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=LINE, linewidth=0.8, zorder=0)
fig.tight_layout()
fig.savefig(f"{IMG}/06_savings_by_category.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Project health matrix: SPI vs CPI
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 7.5))
troubled_mask = (latest.cpi < 1) & (latest.spi < 1)
colors = np.where(troubled_mask, BAD, np.where((latest.cpi >= 1) & (latest.spi >= 1), GOOD, AMBER_DARK))
ax.scatter(latest.spi, latest.cpi, s=140, color=colors, edgecolor="white", linewidth=1.2, zorder=4)
ax.axvline(1.0, color=LINE, linewidth=1, zorder=1)
ax.axhline(1.0, color=LINE, linewidth=1, zorder=1)
for _, row in latest[troubled_mask].iterrows():
    ax.annotate(row.project_name, (row.spi, row.cpi), xytext=(6, 6), textcoords="offset points",
                fontsize=9.5, color=BAD, fontweight="bold")
ax.set_xlabel("Schedule Performance Index (SPI)")
ax.set_ylabel("Cost Performance Index (CPI)")
ax.set_title(f"{troubled_mask.sum()} projects sit in the behind-and-over-budget quadrant",
             fontsize=14, fontweight="bold", color=NAVY, loc="left", pad=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{IMG}/07_health_matrix.png", dpi=150, facecolor="white")
plt.close(fig)

print("Saved 7 charts to", IMG)
