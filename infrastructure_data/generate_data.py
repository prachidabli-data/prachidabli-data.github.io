"""
Generate the synthetic infrastructure portfolio dataset used in the
Infrastructure Commercial & Financial Analytics case study.

The analysis this project reproduces was originally built in Power BI on
real, confidential client data during an infrastructure commercial role.
None of that data can appear here. Every table below is generated from
scratch with a fixed random seed, built to reproduce the structure and
behaviour of a real EVM and procurement dataset without using or
approximating any real, client, or identifiable data.

Run with:
    python3 infrastructure_data/generate_data.py

Writes 6 CSVs into this folder and prints a validation summary comparing
the generated data against the target ranges it was designed to hit.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

RNG = np.random.default_rng(20230101)

OUT = "infrastructure_data"
START = date(2023, 1, 1)
END = date(2025, 12, 31)

# ---------------------------------------------------------------------------
# dim_date
# ---------------------------------------------------------------------------
days = pd.date_range(START, END, freq="D")
dim_date = pd.DataFrame({"date_key": days.strftime("%Y-%m-%d")})
dim_date["year"] = days.year
dim_date["quarter"] = days.quarter
dim_date["quarter_label"] = dim_date.year.astype(str) + " Q" + dim_date.quarter.astype(str)
dim_date["month_no"] = days.month
dim_date["month_name"] = days.strftime("%B")
dim_date["month_year"] = days.strftime("%b %Y")
dim_date["month_year_sort"] = days.strftime("%Y%m").astype(int)
fin_year_start = np.where(days.month >= 4, days.year, days.year - 1)
dim_date["fin_year"] = [f"FY{str(y)[2:]}/{str(y + 1)[2:]}" for y in fin_year_start]
is_month_end = days.to_series(index=range(len(days))).dt.is_month_end.values
dim_date["is_month_end"] = np.where(is_month_end, "Yes", "No")
dim_date.to_csv(f"{OUT}/dim_date.csv", index=False)

month_ends = dim_date[dim_date.is_month_end == "Yes"].reset_index(drop=True)

# ---------------------------------------------------------------------------
# dim_project
# ---------------------------------------------------------------------------
PROJECT_TYPES = ["Highways", "Rail", "Water", "Energy", "Public Building"]
REGIONS = ["North West", "North East", "Yorkshire and Humber", "Midlands",
           "South East", "South West", "Scotland"]

NAME_STEMS = {
    "Highways": ["A47 Corridor Upgrade", "Weston Bypass", "Colne Valley Interchange",
                 "Ridgeway Junction Improvement"],
    "Rail": ["Meridian Transit Interchange", "Lower Dene Rail Link", "Ashcombe Station Renewal",
             "Trent Valley Electrification"],
    "Water": ["Thameside Flood Defence", "Calder Reservoir Upgrade", "Millbrook Water Treatment Works",
              "Severnside Pumping Station", "Kelder Storm Drainage"],
    "Energy": ["Trent Valley Substation", "Northmoor Wind Connection", "Estuary Grid Reinforcement",
               "Fenwick Solar Array Works", "Blackstone Battery Storage"],
    "Public Building": ["Ashgrove Community Hospital", "Kingsmead Library Redevelopment",
                         "Oldport Civic Centre", "Riverside Leisure Centre", "Fairholme School Extension"],
}

rows = []
pid = 1
for ptype, names in NAME_STEMS.items():
    for name in names:
        rows.append({"project_id": pid, "project_name": name, "project_type": ptype})
        pid += 1

dim_project = pd.DataFrame(rows)
n_proj = len(dim_project)
dim_project["region"] = RNG.choice(REGIONS, size=n_proj)

# Budget at completion: skewed toward smaller projects with a few large ones.
bac = np.round(np.exp(RNG.normal(np.log(6_800_000), 0.82, size=n_proj)) / 1000) * 1000
bac = np.clip(bac, 2_000_000, 60_000_000)
dim_project["budget_at_completion"] = bac.astype(int)

# Start dates spread across the whole window, so the portfolio has a mix of
# projects that finish, are still running, and are not yet complete.
start_offsets = RNG.integers(0, 900, size=n_proj)
starts = [START + timedelta(days=int(o)) for o in start_offsets]
dim_project["start_date"] = [d.isoformat() for d in starts]

# Duration scales loosely with budget: bigger projects run longer.
duration_months = np.clip((8 + (bac / 1_000_000) * 0.35 + RNG.normal(0, 3, size=n_proj)).astype(int), 9, 34)
planned_ends = [s.replace(day=1) + pd.DateOffset(months=int(m)) for s, m in zip(starts, duration_months)]
planned_ends = [pd.Timestamp(pe).date() for pe in planned_ends]
dim_project["planned_end_date"] = [d.isoformat() for d in planned_ends]
dim_project["duration_months"] = duration_months

# ---------------------------------------------------------------------------
# Per-project performance profile (drives fact_evm)
# ---------------------------------------------------------------------------
# spi_factor and cpi_factor centred so ~60% of projects sit at or above 1.0
# on each measure, with a handful clearly troubled on both at once.
spi_factor = np.clip(RNG.normal(1.07, 0.10, size=n_proj), 0.72, 1.20)
cpi_factor = np.clip(RNG.normal(1.035, 0.115, size=n_proj), 0.70, 1.22)
# force a small, clearly troubled cluster (both CPI and SPI below 0.9)
troubled = RNG.choice(n_proj, size=3, replace=False)
spi_factor[troubled] = RNG.uniform(0.74, 0.88, size=3)
cpi_factor[troubled] = RNG.uniform(0.72, 0.87, size=3)
dim_project["_spi_factor"] = spi_factor
dim_project["_cpi_factor"] = cpi_factor

# status: complete if planned end already passed before dataset end,
# in delivery if still running, on hold for two of the troubled projects
on_hold_idx = set(troubled[:2].tolist())
status = []
for i, row in dim_project.iterrows():
    pe = pd.Timestamp(row.planned_end_date).date()
    if i in on_hold_idx:
        status.append("On hold")
    elif pe <= END:
        status.append("Complete")
    else:
        status.append("In delivery")
dim_project["status"] = status

dim_project_out = dim_project.drop(columns=["_spi_factor", "_cpi_factor"])
dim_project_out.to_csv(f"{OUT}/dim_project.csv", index=False)

# ---------------------------------------------------------------------------
# dim_supplier
# ---------------------------------------------------------------------------
SUPPLIER_TYPES = ["Groundworks", "Steelworks", "M&E", "Plant Hire", "Design Consultancy", "Materials Supply"]
SUPPLIER_NAMES = [
    ("Northbridge Aggregates", "Groundworks"), ("Ironclad Steelworks", "Steelworks"),
    ("Merrow Groundworks", "Groundworks"), ("Calder M&E Services", "M&E"),
    ("Ashfield Plant Hire", "Plant Hire"), ("Riverstone Design Partners", "Design Consultancy"),
    ("Coastal Materials Supply", "Materials Supply"), ("Draymoor Civils", "Groundworks"),
    ("Vantage M&E Solutions", "M&E"), ("Northmoor Plant & Tool Hire", "Plant Hire"),
    ("Sterling Steel Fabrication", "Steelworks"), ("Oldport Design Consultancy", "Design Consultancy"),
    ("Fenwick Materials Group", "Materials Supply"), ("Harcourt Groundworks", "Groundworks"),
    ("Blackstone Plant Hire", "Plant Hire"),
]
dim_supplier = pd.DataFrame(SUPPLIER_NAMES, columns=["supplier_name", "supplier_type"])
dim_supplier.insert(0, "supplier_id", range(1, len(dim_supplier) + 1))
dim_supplier.to_csv(f"{OUT}/dim_supplier.csv", index=False)

# ---------------------------------------------------------------------------
# dim_cost_category
# ---------------------------------------------------------------------------
CATEGORIES = ["Labour", "Materials", "Plant", "Subcontract", "Design", "Overheads"]
dim_cost_category = pd.DataFrame({"category_id": range(1, 7), "category_name": CATEGORIES})
dim_cost_category.to_csv(f"{OUT}/dim_cost_category.csv", index=False)

# ---------------------------------------------------------------------------
# fact_evm: monthly cumulative PV / EV / AC per project
# ---------------------------------------------------------------------------
def s_curve(t, duration):
    """Cumulative fraction of work planned by month t (0-indexed), logistic S-curve."""
    x = (t - duration / 2) / (duration / 6)
    return 1 / (1 + np.exp(-x))

evm_rows = []
for _, proj in dim_project.iterrows():
    start = pd.Timestamp(proj.start_date)
    dur = int(proj.duration_months)
    spi_f = proj._spi_factor
    cpi_f = proj._cpi_factor
    bac_val = proj.budget_at_completion

    if proj.status == "Complete":
        active_months = dur
    elif proj.status == "On hold":
        active_months = min(dur, int(dur * 0.55))
    else:
        months_elapsed = (pd.Timestamp(END) - start).days / 30.44
        active_months = int(np.clip(months_elapsed, 1, dur))

    p0 = s_curve(0, dur)
    p_end = s_curve(dur, dur)
    for m in range(1, active_months + 1):
        month_date = (start + pd.DateOffset(months=m)).normalize()
        if month_date.date() > END:
            break
        p_frac = (s_curve(m, dur) - p0) / (p_end - p0)
        p_frac = float(np.clip(p_frac, 0, 1))
        # PV can never exceed BAC by definition (100% planned = BAC); apply
        # noise before clamping so the clamp is the true final word.
        pv_cum = min(bac_val * p_frac * RNG.normal(1.0, 0.01), bac_val)
        ev_cum = min(pv_cum * spi_f * RNG.normal(1.0, 0.012), bac_val)
        ac_cum = (ev_cum / cpi_f) * RNG.normal(1.0, 0.012)
        # find nearest month-end date_key on or before month_date
        candidates = month_ends[pd.to_datetime(month_ends.date_key) <= month_date]
        if candidates.empty:
            continue
        date_key = candidates.date_key.iloc[-1]
        evm_rows.append({
            "project_id": proj.project_id,
            "date_key": date_key,
            "planned_value": round(pv_cum, 2),
            "earned_value": round(ev_cum, 2),
            "actual_cost": round(ac_cum, 2),
        })

fact_evm = pd.DataFrame(evm_rows).drop_duplicates(subset=["project_id", "date_key"])
# enforce monotonic non-decreasing cumulative values per project
fact_evm = fact_evm.sort_values(["project_id", "date_key"])
for col in ["planned_value", "earned_value", "actual_cost"]:
    fact_evm[col] = fact_evm.groupby("project_id")[col].cummax()
fact_evm.to_csv(f"{OUT}/fact_evm.csv", index=False)

# ---------------------------------------------------------------------------
# fact_procurement: project x supplier x category
# ---------------------------------------------------------------------------
proc_rows = []
for _, proj in dim_project.iterrows():
    n_lines = RNG.integers(3, 6)
    cats = RNG.choice(dim_cost_category.category_id, size=n_lines, replace=False)
    supps = RNG.choice(dim_supplier.supplier_id, size=n_lines, replace=False)
    # share of BAC committed through procurement, split across the lines
    shares = RNG.dirichlet(np.ones(n_lines) * 2.2) * RNG.uniform(0.55, 0.85)
    for cat_id, supp_id, share in zip(cats, supps, shares):
        baseline = proj.budget_at_completion * share
        commit_var = RNG.normal(1.0, 0.03)
        committed = baseline * commit_var
        # most lines save a little against baseline; some overspend
        savings_pct = RNG.normal(0.045, 0.07)
        actual = committed * (1 - savings_pct)
        proc_rows.append({
            "project_id": proj.project_id,
            "supplier_id": int(supp_id),
            "category_id": int(cat_id),
            "baseline_estimate": round(baseline, 2),
            "committed_value": round(committed, 2),
            "actual_spend": round(actual, 2),
            "savings": round(baseline - actual, 2),
        })

fact_procurement = pd.DataFrame(proc_rows)
fact_procurement.to_csv(f"{OUT}/fact_procurement.csv", index=False)

# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------
latest = fact_evm.sort_values("date_key").groupby("project_id").tail(1)
latest = latest.merge(dim_project[["project_id", "budget_at_completion"]], on="project_id")
latest["cpi"] = latest.earned_value / latest.actual_cost
latest["spi"] = latest.earned_value / latest.planned_value

portfolio_bac = dim_project.budget_at_completion.sum()
portfolio_cpi = latest.earned_value.sum() / latest.actual_cost.sum()
portfolio_spi = latest.earned_value.sum() / latest.planned_value.sum()
over_budget = (latest.cpi < 1).sum()
behind_schedule = (latest.spi < 1).sum()
total_savings = fact_procurement.savings.sum()
total_baseline = fact_procurement.baseline_estimate.sum()

print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Projects:                 {n_proj}")
print(f"Portfolio BAC:            £{portfolio_bac:,.0f}")
print(f"Portfolio CPI:            {portfolio_cpi:.3f}")
print(f"Portfolio SPI:            {portfolio_spi:.3f}")
print(f"Projects over budget:     {over_budget} of {n_proj} ({over_budget/n_proj:.0%})")
print(f"Projects behind schedule: {behind_schedule} of {n_proj} ({behind_schedule/n_proj:.0%})")
print(f"Troubled (CPI&SPI<0.9):   {((latest.cpi<0.9)&(latest.spi<0.9)).sum()}")
print(f"Procurement baseline:     £{total_baseline:,.0f}")
print(f"Procurement savings:      £{total_savings:,.0f} ({total_savings/total_baseline:.1%})")
print(f"fact_evm rows:            {len(fact_evm):,}")
print(f"fact_procurement rows:    {len(fact_procurement):,}")
