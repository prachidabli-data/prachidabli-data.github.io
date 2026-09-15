# Infrastructure Commercial & Financial Analytics: data and model

This folder documents the sixth portfolio project: a commercial and
operational analysis across a 23 project UK infrastructure portfolio,
covering Earned Value Management (EVM) and procurement savings. The full
write-up is `infrastructure-case-study.html` at the root of the repository.

## Why the data is synthetic

This analysis was originally built and delivered in Power BI, on real,
confidential client data, during an infrastructure commercial analytics
role. That underlying work cannot appear here in any form. Every table,
project, supplier and figure in this folder was generated from scratch by
`generate_data.py`, a documented script with a fixed random seed, built to
reproduce the structure and behaviour of a real EVM and procurement
dataset without using or approximating any real, client, or identifiable
data.

Run it yourself with:

```bash
python3 infrastructure_data/generate_data.py
```

It writes the 6 CSVs below into this folder and prints a validation
summary comparing the generated data against the target ranges it was
designed to hit.

## The 6 CSVs

A star schema with two fact grains: monthly EVM snapshots per project, and
procurement lines per project, supplier and cost category.

| Table | Grain | Rows | Description |
|---|---|---|---|
| `dim_date` | one row per day | 1,096 | 2023-01-01 to 2025-12-31, no gaps |
| `dim_project` | one row per project | 23 | Highways, Rail, Water, Energy, Public Building |
| `dim_supplier` | one row per supplier | 15 | Groundworks, Steelworks, M&E, Plant Hire, Design, Materials |
| `dim_cost_category` | one row per category | 6 | Labour, Materials, Plant, Subcontract, Design, Overheads |
| `fact_evm` | project x month | 240 | cumulative Planned Value, Earned Value, Actual Cost |
| `fact_procurement` | project x supplier x category | 92 | baseline, committed, actual spend, savings |

### Generation rules

The generator encodes the following rules, so the story in the analysis
follows from how the data was built rather than being asserted afterwards:

1. Most projects track an S-curve: Planned Value, Earned Value and Actual
   Cost all rise over the project life, with Earned Value and Actual Cost
   following a schedule and cost performance factor sampled per project.
2. A realistic spread of health: roughly 60% of projects on or under
   budget (CPI >= 1) and on or ahead of schedule (SPI >= 1), with the
   remainder behind on one or both measures. A small cluster of clearly
   troubled projects sits below 0.9 on both CPI and SPI at once.
3. Procurement savings are positive overall (actual spend below baseline
   on average), but individual supplier and category lines can and do
   overspend against their baseline.
4. Realistic magnitudes: project budgets from roughly £2.7M to £31.9M,
   a portfolio total in the low hundreds of millions.
5. Referential integrity throughout: Planned Value is capped at each
   project's Budget at Completion (100% planned progress cannot exceed the
   total budget by definition), Earned Value is capped the same way, every
   fact row references a valid project, supplier, category or date, and
   every date falls inside `dim_date`.

### Validation summary (actual output of this seed)

| Metric | Actual |
|---|---|
| Projects | 23 |
| Portfolio Budget at Completion | £260,961,000 |
| Portfolio CPI | 1.021 |
| Portfolio SPI | 0.983 |
| Projects over budget (CPI < 1) | 9 of 23 (39%) |
| Projects behind schedule (SPI < 1) | 10 of 23 (43%) |
| Projects below 0.9 on both CPI and SPI | 3 |
| Procurement baseline | £176,459,389 |
| Procurement savings | £7,625,375 (4.3%) |

## The Power BI model and DAX

The original build modelled the same two fact tables against `dim_date`,
`dim_project`, `dim_supplier` and `dim_cost_category`, with `dim_date`
marked as the official date table.

### Base measures

```dax
Planned Value = SUM(fact_evm[planned_value])
Earned Value  = SUM(fact_evm[earned_value])
Actual Cost   = SUM(fact_evm[actual_cost])

CPI = DIVIDE([Earned Value], [Actual Cost])
SPI = DIVIDE([Earned Value], [Planned Value])

Cost Variance     = [Earned Value] - [Actual Cost]
Schedule Variance = [Earned Value] - [Planned Value]
```

### Forecasting

```dax
Budget at Completion = SUM(dim_project[budget_at_completion])
Estimate at Completion (EAC) = DIVIDE([Budget at Completion], [CPI])
Variance at Completion       = [Budget at Completion] - [Estimate at Completion (EAC)]

Cumulative Earned Value =
CALCULATE([Earned Value],
    FILTER(ALL(dim_date[date_key]), dim_date[date_key] <= MAX(dim_date[date_key])))
```

### Procurement

```dax
Procurement Savings = SUM(fact_procurement[baseline_estimate]) - SUM(fact_procurement[actual_spend])
Savings % = DIVIDE([Procurement Savings], SUM(fact_procurement[baseline_estimate]))
```

### Interactive setup

Slicers by project, region and project type on the summary page, with
drill-through from any project to a detail page showing its CPI, SPI,
cost variance and EAC as KPI cards, conditionally formatted red below the
1.0 (or, for EAC, above budget) threshold.

## The Python and Matplotlib charts

`generate_charts.py` reads the CSVs with pandas and renders 7 charts with
Matplotlib, saved to `images/`. This is the portfolio-safe reproduction of
the same analysis described above. Run it yourself with:

```bash
python3 infrastructure_data/generate_charts.py
```

| File | Chart |
|---|---|
| `01_evm_scurve.png` | Portfolio cumulative Planned Value, Earned Value and Actual Cost |
| `02_cpi_by_project.png` | Cost Performance Index by project, sorted, over/under budget in colour |
| `03_spi_by_project.png` | Schedule Performance Index by project, sorted, behind/ahead schedule in colour |
| `04_cost_variance.png` | Cost Variance by project, in £m |
| `05_procurement_by_category.png` | Actual procurement spend by cost category |
| `06_savings_by_category.png` | Procurement savings against baseline, by cost category |
| `07_health_matrix.png` | SPI against CPI for every project, the behind-and-over-budget quadrant |
