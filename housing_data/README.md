# Housing Sector Portfolio Analytics: data, model and DAX

This folder documents the fourth portfolio project: a Power BI and DAX
specialist piece built around a housing provider's tender and contract
pipeline. The full write-up is `housing-case-study.html` at the root of
the repository.

## Why the data is synthetic

The project is based on a real housing sector analyst role, tracking a
live tender and contract book on a UK portfolio management platform. That
underlying work is confidential and cannot appear here in any form. Every
table, date, contractor name and figure in this folder was generated from
scratch by `generate_housing_data.py`, a documented script with a fixed
random seed, built to reproduce the structure and behaviour of that kind
of pipeline without using or approximating any real, client, or
identifiable data.

Run it yourself with:

```bash
python3 generate_housing_data.py
```

It writes the 8 CSVs below into this folder and prints a validation
summary comparing the generated data against the target ranges it was
designed to hit.

## The 8 CSVs

A star schema with two fact grains, a continuous date table, and one
disconnected table for the dashboard's measure switcher.

| Table | Grain | Rows | Description |
|---|---|---|---|
| `dim_date` | one row per day | 731 | 2024-01-01 to 2025-12-31, no gaps |
| `dim_region` | one row per region | 8 | UK regions |
| `dim_contractor` | one row per contractor | 12 | Tier 1 / Tier 2 / Tier 3 |
| `dim_stage` | one row per funnel stage | 5 | Identified, Bidding, Submitted, Won, Lost |
| `dim_status` | one row per contract status | 4 | Active, Completed, At-risk, On-hold |
| `dim_measure_selector` | disconnected | 3 | Value, Count, Win Rate |
| `fact_tenders` | one row per bid | 1,546 | identification through to outcome |
| `fact_contracts` | one row per won contract | 413 | award through to delivery |

### Generation rules

The generator encodes the following rules, so the story in the dashboard
follows from how the data was built rather than being asserted afterwards:

1. Bid volume rises before the UK financial year end (January to March)
   and dips over summer.
2. Overall win rate sits around 30%, varying by region and by contractor
   tier (Tier 1 wins more, and delivers on time more, than Tier 2 or 3).
3. Portfolio value trends upward across the two years, so year on year
   growth and the cumulative running total both read positive.
4. London and the South East carry higher average bid values; northern
   and midlands regions carry higher bid volume at a lower average value.
5. On time delivery sits in the mid to high 80s, dipping under load in
   contracts that start during the January to March surge, so it lands
   just below a 90% target.
6. The bid funnel narrows at each stage: every tender is Identified, most
   reach Bidding, fewer reach Submitted, and a minority are Won.
7. Referential integrity throughout: every Won tender has exactly one
   matching contract, `contract_value` equals the winning `bid_value`,
   and every date falls inside `dim_date`.

### Validation summary (actual output of this seed)

| Metric | Target | Actual |
|---|---|---|
| Total tenders | ~1,546 | 1,546 |
| Won tenders / contracts | ~413 | 413 |
| Win rate (won / decided) | ~30% | 28.8% |
| On time delivery | ~87% | 86.7% |
| Portfolio value, 2024 | ~£96M | £87.2M |
| Portfolio value, 2025 | ~£137M | £144.8M |
| Portfolio value, total | ~£233M | £232.0M |
| Funnel (reached stage 1 / 2 / 3 / Won) | 1,546 / 1,498 / 1,430 / 413 | 1,546 / 1,504 / 1,430 / 413 |

Win rate by contractor tier: Tier 1 49.6%, Tier 2 26.5%, Tier 3 16.6%.
On time % by contractor tier: Tier 1 94.4%, Tier 2 83.3%, Tier 3 79.8%.

## The Power BI model

Star schema, two fact tables on conformed dimensions:

- `dim_date` is marked as the official date table, on `date_key`.
- `dim_measure_selector` stays disconnected: no relationships. It exists
  only to be read by `SELECTEDVALUE` in the measure switcher.
- Active relationships (both many to one, single direction):
  `fact_tenders[identified_date_key]` to `dim_date[date_key]`, and
  `fact_contracts[start_date_key]` to `dim_date[date_key]`.

### Base measures

```dax
Total Bids = COUNTROWS(fact_tenders)
Decided Bids = SUM(fact_tenders[is_decided])
Won Bids = SUM(fact_tenders[is_won])
Win Rate = DIVIDE([Won Bids], [Decided Bids])
Total Contracts = COUNTROWS(fact_contracts)
Portfolio Value = SUM(fact_contracts[contract_value])
On-Time % = DIVIDE(SUM(fact_contracts[is_on_time]), COUNTROWS(fact_contracts))
```

### Time intelligence

```dax
Portfolio Value PY = CALCULATE([Portfolio Value], DATEADD(dim_date[date_key], -1, YEAR))
Portfolio Value YoY % = DIVIDE([Portfolio Value] - [Portfolio Value PY], [Portfolio Value PY])
Cumulative Portfolio Value = CALCULATE([Portfolio Value], FILTER(ALL(dim_date[date_key]), dim_date[date_key] <= MAX(dim_date[date_key])))
Win Rate 3M Rolling = CALCULATE([Win Rate], DATESINPERIOD(dim_date[date_key], MAX(dim_date[date_key]), -3, MONTH))
```

### League tables (RANKX)

```dax
Contractor Rank = RANKX(ALL(dim_contractor[contractor_name]), [On-Time %], , DESC, DENSE)
Region Rank by Win Rate = RANKX(ALL(dim_region[region_name]), [Win Rate], , DESC, DENSE)
```

### Dynamic measure switcher

```dax
Selected Metric =
VAR m = SELECTEDVALUE(dim_measure_selector[measure_name], "Value")
RETURN SWITCH(m, "Value", [Portfolio Value], "Count", [Total Bids], "Win Rate", [Win Rate])
```

### Funnel (stage conversion)

```dax
Reached Bidding = CALCULATE([Total Bids], fact_tenders[reached_stage_id] >= 2)
Reached Submitted = CALCULATE([Total Bids], fact_tenders[reached_stage_id] >= 3)
```

Funnel reads as: Total Bids, then Reached Bidding, then Reached Submitted,
then Won Bids.

## Dashboard layout

One dense, stretched canvas in the brand palette, deliberately avoiding
gauges and donuts everywhere except the two places they are genuinely the
right chart:

- Title bar with three slicers: Year, Region, Stage.
- A row of 6 KPI cards with year on year indicators: Pipeline Value, Win
  Rate, Active Contracts, Portfolio Value, On-Time %, Avg Contract Value.
  A gauge at the end of the row shows On-Time % against a 90% target.
- Pipeline half (left): bid conversion funnel, win rate trend (3 month
  rolling), pipeline value by region (via the region RANKX measure).
- Contracts half (right): portfolio value over time as a running total
  (with the dynamic measure switcher toggle), a donut of portfolio value
  by status (re-slices with the switcher), the contractor league table
  (via RANKX), and delivery status by region.
- A full width contract ledger detail matrix along the bottom.

## What is not in this repository

The Power BI file (`.pbix`) and its exported screenshot
(`housing_dashboard.png`, referenced by `housing-case-study.html`) are
built separately in Power BI Desktop, since this environment cannot run
Power BI. They are added to the repository afterwards, alongside this
generated data.
