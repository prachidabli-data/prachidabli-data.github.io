"""
Generate a synthetic housing sector tender and contract portfolio.

Reconstructs the structure and behaviour of a real housing provider's
tender-to-contract pipeline (2024-01-01 to 2025-12-31), entirely from
documented rules. No real, client, or identifiable data is used anywhere:
every figure below is generated from a fixed random seed so the whole
dataset, and every number in the case study, is reproducible from this
script alone.

Produces 8 CSVs into this folder:
  dim_date, dim_region, dim_contractor, dim_stage, dim_status,
  dim_measure_selector, fact_tenders, fact_contracts
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

RNG_SEED = 20240101
rng = np.random.default_rng(RNG_SEED)

WINDOW_START = date(2024, 1, 1)
WINDOW_END = date(2025, 12, 31)

# ---------------------------------------------------------------------------
# dim_date
# ---------------------------------------------------------------------------
n_days = (WINDOW_END - WINDOW_START).days + 1
dates = [WINDOW_START + timedelta(days=i) for i in range(n_days)]

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
               "Oct", "Nov", "Dec"]


def fin_year(d):
    fy_start = d.year if d.month >= 4 else d.year - 1
    return f"FY{str(fy_start)[2:]}/{str(fy_start + 1)[2:]}"


def is_month_end(d):
    nxt = d + timedelta(days=1)
    return int(nxt.month != d.month)


dim_date = pd.DataFrame({
    "date_key": [d.isoformat() for d in dates],
    "year": [d.year for d in dates],
    "quarter": [(d.month - 1) // 3 + 1 for d in dates],
    "quarter_label": [f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in dates],
    "month_no": [d.month for d in dates],
    "month_name": [MONTH_NAMES[d.month - 1] for d in dates],
    "month_year": [f"{MONTH_SHORT[d.month - 1]} {d.year}" for d in dates],
    "month_year_sort": [d.year * 100 + d.month for d in dates],
    "fin_year": [fin_year(d) for d in dates],
    "is_month_end": [is_month_end(d) for d in dates],
})
date_keys = dim_date["date_key"].tolist()
date_lookup = {d.isoformat(): d for d in dates}

# ---------------------------------------------------------------------------
# dim_region
# ---------------------------------------------------------------------------
REGIONS = [
    ("R01", "North West", "NW"),
    ("R02", "North East", "NE"),
    ("R03", "Yorkshire & Humber", "Y&H"),
    ("R04", "West Midlands", "WM"),
    ("R05", "East of England", "EoE"),
    ("R06", "London", "LDN"),
    ("R07", "South East", "SE"),
    ("R08", "South West", "SW"),
]
dim_region = pd.DataFrame(REGIONS, columns=["region_id", "region_name", "region_short"])

# volume weight: northern / midlands regions bid more often; London and
# South East carry higher value but lower volume.
REGION_VOLUME_WEIGHT = {
    "R01": 1.35, "R02": 1.15, "R03": 1.25, "R04": 1.20,
    "R05": 0.95, "R06": 0.65, "R07": 0.75, "R08": 0.90,
}
REGION_VALUE_MEAN = {
    "R01": 460_000, "R02": 420_000, "R03": 445_000, "R04": 485_000,
    "R05": 565_000, "R06": 1_050_000, "R07": 945_000, "R08": 605_000,
}
REGION_WIN_BONUS = {
    "R01": 0.05, "R02": -0.05, "R03": 0.0, "R04": 0.05,
    "R05": 0.0, "R06": -0.10, "R07": 0.0, "R08": 0.05,
}

# ---------------------------------------------------------------------------
# dim_contractor
# ---------------------------------------------------------------------------
CONTRACTORS = [
    ("K01", "Northfield Construction", "Tier 1"),
    ("K02", "Bridgeway Partners", "Tier 1"),
    ("K03", "Ashcroft Building Group", "Tier 1"),
    ("K04", "Meridian Property Services", "Tier 2"),
    ("K05", "Oakstone Contractors", "Tier 2"),
    ("K06", "Harborne & Vale", "Tier 2"),
    ("K07", "Riverside Maintenance Co", "Tier 2"),
    ("K08", "Kestrel Build Solutions", "Tier 2"),
    ("K09", "Fenwick Estates Ltd", "Tier 3"),
    ("K10", "Grovepoint Services", "Tier 3"),
    ("K11", "Anchor Point Contracts", "Tier 3"),
    ("K12", "Willowmead Property Group", "Tier 3"),
]
dim_contractor = pd.DataFrame(CONTRACTORS, columns=["contractor_id", "contractor_name", "tier"])

TIER_WIN_BONUS = {"Tier 1": 0.65, "Tier 2": 0.15, "Tier 3": -0.20}
TIER_ONTIME_P = {"Tier 1": 0.93, "Tier 2": 0.86, "Tier 3": 0.79}

# ---------------------------------------------------------------------------
# dim_stage / dim_status / dim_measure_selector
# ---------------------------------------------------------------------------
dim_stage = pd.DataFrame([
    (1, "Identified", 0),
    (2, "Bidding", 0),
    (3, "Submitted", 0),
    (4, "Won", 1),
    (5, "Lost", 1),
], columns=["stage_id", "stage_name", "is_terminal"])

dim_status = pd.DataFrame([
    (1, "Active"),
    (2, "Completed"),
    (3, "At-risk"),
    (4, "On-hold"),
], columns=["status_id", "status_name"])

dim_measure_selector = pd.DataFrame([
    (1, "Value"),
    (2, "Count"),
    (3, "Win Rate"),
], columns=["measure_id", "measure_name"])

# ---------------------------------------------------------------------------
# fact_tenders
# ---------------------------------------------------------------------------
N_TENDERS = 1546

# Monthly seasonality weight: surge into the UK financial year end
# (Jan-Mar), dip over summer (Jun-Aug).
MONTH_WEIGHT = {
    1: 1.55, 2: 1.45, 3: 1.60, 4: 1.05, 5: 0.95, 6: 0.70,
    7: 0.65, 8: 0.70, 9: 1.00, 10: 1.05, 11: 1.10, 12: 0.90,
}
months = pd.period_range("2024-01", "2025-12", freq="M")
month_weights = np.array([MONTH_WEIGHT[m.month] for m in months], dtype=float)
month_weights /= month_weights.sum()

chosen_months = rng.choice(len(months), size=N_TENDERS, p=month_weights)
region_ids = [r[0] for r in REGIONS]
region_p = np.array([REGION_VOLUME_WEIGHT[r] for r in region_ids], dtype=float)
region_p /= region_p.sum()
contractor_ids = [c[0] for c in CONTRACTORS]

identified_dates = []
for mi in chosen_months:
    per = months[mi]
    days_in_month = per.days_in_month
    day = rng.integers(1, days_in_month + 1)
    identified_dates.append(date(per.year, per.month, int(day)))

regions_for_tenders = rng.choice(region_ids, size=N_TENDERS, p=region_p)
contractors_for_tenders = rng.choice(contractor_ids, size=N_TENDERS)
tier_lookup = dict(zip(dim_contractor["contractor_id"], dim_contractor["tier"]))

# bid value: lognormal around the region's mean, with a mild upward drift
# across the two years so 2025 outbids 2024 overall.
bid_values = []
for d, r in zip(identified_dates, regions_for_tenders):
    mean = REGION_VALUE_MEAN[r]
    # Note: the identified-year growth story is carried by the natural lag
    # between identification and contract start (many late-year tenders
    # start the following year), not by a drift applied here; see the
    # start-year validation below.
    mu = np.log(mean)
    bid_values.append(float(rng.lognormal(mean=mu - 0.18, sigma=0.55)))
bid_values = np.round(np.clip(bid_values, 60_000, 4_500_000), 2)

# funnel: reached_stage_id (1=Identified only, 2=reached Bidding, 3=reached
# Submitted), targets roughly 1546 -> 1498 -> 1430
funnel_roll = rng.random(N_TENDERS)
reached_stage_id = np.where(funnel_roll < 48 / N_TENDERS, 1,
                    np.where(funnel_roll < (48 + 68) / N_TENDERS, 2, 3))

# win score for everything that reached Submitted; top scorers win.
win_score = np.full(N_TENDERS, -np.inf)
mask_submitted = reached_stage_id == 3
noise = rng.normal(0, 1, N_TENDERS)
for i in np.where(mask_submitted)[0]:
    tier = tier_lookup[contractors_for_tenders[i]]
    win_score[i] = TIER_WIN_BONUS[tier] + REGION_WIN_BONUS[regions_for_tenders[i]] + noise[i]

N_WON = 413
won_idx = set(np.argsort(-win_score)[:N_WON])

days_from_end = np.array([(WINDOW_END - d).days for d in identified_dates])

outcome = np.empty(N_TENDERS, dtype=object)
stage_id = np.empty(N_TENDERS, dtype=int)
is_won = np.zeros(N_TENDERS, dtype=int)
is_decided = np.zeros(N_TENDERS, dtype=int)
decision_dates = [None] * N_TENDERS

for i in range(N_TENDERS):
    rs = reached_stage_id[i]
    recent = days_from_end[i] < rng.integers(45, 200)
    if i in won_idx:
        outcome[i] = "Won"
        stage_id[i] = 4
        is_won[i] = 1
        is_decided[i] = 1
    elif rs == 3:
        if recent and rng.random() < 0.68:
            outcome[i] = "In progress"
            stage_id[i] = 3
        else:
            outcome[i] = "Lost"
            stage_id[i] = 5
            is_decided[i] = 1
    else:
        if recent and rng.random() < 0.75:
            outcome[i] = "In progress"
            stage_id[i] = rs
        else:
            outcome[i] = "Lost"
            stage_id[i] = 5
            is_decided[i] = 1
    if is_decided[i]:
        lag = int(rng.integers(21, 100))
        dd = identified_dates[i] + timedelta(days=lag)
        if dd > WINDOW_END:
            dd = WINDOW_END
        decision_dates[i] = dd

tender_ids = [f"T{str(i + 1).zfill(4)}" for i in range(N_TENDERS)]
contract_id_for_tender = [None] * N_TENDERS
won_order = sorted(won_idx, key=lambda i: identified_dates[i])
for rank, i in enumerate(won_order):
    contract_id_for_tender[i] = f"C{str(rank + 1).zfill(4)}"

fact_tenders = pd.DataFrame({
    "tender_id": tender_ids,
    "identified_date_key": [d.isoformat() for d in identified_dates],
    "decision_date_key": [d.isoformat() if d else "" for d in decision_dates],
    "region_id": regions_for_tenders,
    "contractor_id": contractors_for_tenders,
    "stage_id": stage_id,
    "reached_stage_id": reached_stage_id,
    "bid_value": bid_values,
    "outcome": outcome,
    "is_won": is_won,
    "is_decided": is_decided,
    "contract_id": [c if c else "" for c in contract_id_for_tender],
})

# ---------------------------------------------------------------------------
# fact_contracts
# ---------------------------------------------------------------------------
contracts_rows = []
for rank, i in enumerate(won_order):
    contract_id = f"C{str(rank + 1).zfill(4)}"
    region = regions_for_tenders[i]
    contractor = contractors_for_tenders[i]
    tier = tier_lookup[contractor]
    decision_d = decision_dates[i]
    start_lag = int(rng.integers(7, 45))
    start_d = decision_d + timedelta(days=start_lag)
    if start_d > WINDOW_END:
        start_d = WINDOW_END

    duration = int(rng.integers(60, 366))
    max_duration = max((WINDOW_END - start_d).days, 14)
    duration = min(duration, max_duration)
    planned_end_d = start_d + timedelta(days=duration)
    if planned_end_d > WINDOW_END:
        planned_end_d = WINDOW_END

    # load dip: contracts starting in the Jan-Mar surge months run a higher
    # risk of slipping, since delivery capacity is stretched.
    load_dip = 0.06 if start_d.month in (1, 2, 3) else 0.0
    on_time_p = TIER_ONTIME_P[tier] - load_dip
    is_on_time = int(rng.random() < on_time_p)

    elapsed = (WINDOW_END - start_d).days
    total_span = max((planned_end_d - start_d).days, 1)
    progress = elapsed / total_span

    if progress >= 1.0:
        status_id = 2  # Completed
        if is_on_time:
            overrun = 0
        else:
            overrun = int(rng.integers(10, 70))
        end_d = planned_end_d + timedelta(days=overrun)
        if end_d > WINDOW_END:
            end_d = WINDOW_END
        pct_complete = 1.0
    else:
        end_d = None
        pct_complete = round(float(np.clip(progress + rng.normal(0, 0.05), 0.02, 0.98)), 3)
        at_risk_p = {"Tier 1": 0.10, "Tier 2": 0.22, "Tier 3": 0.38}[tier]
        if rng.random() < 0.05:
            status_id = 4  # On-hold
        elif (not is_on_time) or rng.random() < at_risk_p:
            status_id = 3  # At-risk
        else:
            status_id = 1  # Active

    contracts_rows.append({
        "contract_id": contract_id,
        "start_date_key": start_d.isoformat(),
        "planned_end_key": planned_end_d.isoformat(),
        "end_date_key": end_d.isoformat() if end_d else "",
        "region_id": region,
        "contractor_id": contractor,
        "status_id": status_id,
        "contract_value": bid_values[i],
        "pct_complete": pct_complete,
        "is_on_time": is_on_time,
    })

fact_contracts = pd.DataFrame(contracts_rows)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
tables = {
    "dim_date": dim_date,
    "dim_region": dim_region,
    "dim_contractor": dim_contractor,
    "dim_stage": dim_stage,
    "dim_status": dim_status,
    "dim_measure_selector": dim_measure_selector,
    "fact_tenders": fact_tenders,
    "fact_contracts": fact_contracts,
}
for name, df in tables.items():
    df.to_csv(f"housing_data/{name}.csv", index=False)

# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(f"Total tenders:        {len(fact_tenders)} (target ~1,546)")
print(f"Won tenders:          {fact_tenders['is_won'].sum()} (target ~413)")
decided = fact_tenders["is_decided"].sum()
win_rate = fact_tenders["is_won"].sum() / decided
print(f"Decided tenders:      {decided}")
print(f"Win rate:             {win_rate:.1%} (target ~30%)")
print(f"Total contracts:      {len(fact_contracts)} (target ~413)")
on_time_pct = fact_contracts["is_on_time"].mean()
print(f"On-time %:            {on_time_pct:.1%} (target ~87%)")

fact_contracts["start_year"] = fact_contracts["start_date_key"].str[:4].astype(int)
value_by_year = fact_contracts.groupby("start_year")["contract_value"].sum()
print("\nPortfolio value by contract start year (target ~£96M 2024 / ~£137M 2025):")
for yr, v in value_by_year.items():
    print(f"  {yr}: £{v / 1e6:.1f}M")
print(f"  Total: £{fact_contracts['contract_value'].sum() / 1e6:.1f}M (target ~£233M)")

n_stage1 = (fact_tenders["reached_stage_id"] >= 1).sum()
n_stage2 = (fact_tenders["reached_stage_id"] >= 2).sum()
n_stage3 = (fact_tenders["reached_stage_id"] >= 3).sum()
n_won = fact_tenders["is_won"].sum()
print(f"\nFunnel: {n_stage1} -> {n_stage2} -> {n_stage3} -> {n_won} "
      f"(target 1546 -> 1498 -> 1430 -> 413)")

status_counts = fact_contracts["status_id"].map(dict(zip(dim_status.status_id, dim_status.status_name))).value_counts()
print(f"\nContract status breakdown:\n{status_counts.to_string()}")

tier_win = fact_tenders.merge(dim_contractor, on="contractor_id")
tier_win = tier_win[tier_win["is_decided"] == 1].groupby("tier")["is_won"].mean()
print(f"\nWin rate by contractor tier:\n{tier_win.to_string()}")

tier_ontime = fact_contracts.merge(dim_contractor, on="contractor_id").groupby("tier")["is_on_time"].mean()
print(f"\nOn-time % by contractor tier:\n{tier_ontime.to_string()}")

print("\nDone. CSVs written to housing_data/")
