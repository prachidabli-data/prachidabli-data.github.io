# Housing Sector Portfolio Analytics: data, model and queries

This folder documents the fourth portfolio project: an Excel, MySQL and
Python piece built around a housing provider's tender and contract
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
small lookup table that drives the metric toggle on the Excel pivot
chart and the browser widget.

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

## The MySQL model

The 8 CSVs load straight into a MySQL 8.0+ schema: two fact tables,
`fact_tenders` and `fact_contracts`, joined to `dim_date`, `dim_region`,
`dim_contractor`, `dim_stage` and `dim_status` on their surrogate keys.
`dim_measure_selector` is not joined to anything: it is just a 3 row
lookup list read by the Excel pivot chart's field toggle and by the
browser widget.

`housing_data/validate_sql.py` loads the same 8 CSVs into a local SQLite
database (used only as a stand in query engine, since this environment
has no MySQL server installed) and runs the three queries below, using
only ANSI standard window function syntax that is unchanged on MySQL
8.0+. Run it yourself with:

```bash
python3 housing_data/validate_sql.py
```

### Rolling 3 month win rate

```sql
SELECT
  d.month_year,
  ROUND(100.0 * SUM(t.is_won) / SUM(t.is_decided), 1) AS win_rate_pct,
  ROUND(
    100.0 * SUM(SUM(t.is_won)) OVER (ORDER BY d.month_year_sort ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    / SUM(SUM(t.is_decided)) OVER (ORDER BY d.month_year_sort ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
  1) AS win_rate_3m_rolling_pct
FROM fact_tenders t
JOIN dim_date d ON d.date_key = t.identified_date_key
GROUP BY d.month_year, d.month_year_sort
ORDER BY d.month_year_sort;
```

Real output: the 3 month rolling win rate climbs from 23.8% in June 2025
to 45.1% by December 2025.

### Contractor league table

```sql
SELECT
  c.contractor_name,
  c.tier,
  COUNT(*) AS contracts,
  ROUND(100.0 * SUM(f.is_on_time) / COUNT(*), 1) AS on_time_pct,
  RANK() OVER (ORDER BY SUM(f.is_on_time) * 1.0 / COUNT(*) DESC) AS on_time_rank
FROM fact_contracts f
JOIN dim_contractor c ON c.contractor_id = f.contractor_id
GROUP BY c.contractor_name, c.tier
ORDER BY on_time_rank;
```

Real output: Northfield Construction (Tier 1) ranks first at 96.6% on
time, and Anchor Point Contracts, a Tier 3 contractor, ranks third at
91.3%, ahead of several Tier 2 contractors. A simple average by tier
would have hidden that individual contractor.

### Portfolio value by year, year on year

```sql
SELECT
  SUBSTR(f.start_date_key, 1, 4) AS start_year,
  ROUND(SUM(f.contract_value) / 1000000.0, 1) AS portfolio_value_gbp_m,
  ROUND(
    100.0 * (SUM(f.contract_value) - LAG(SUM(f.contract_value)) OVER (ORDER BY SUBSTR(f.start_date_key, 1, 4)))
    / LAG(SUM(f.contract_value)) OVER (ORDER BY SUBSTR(f.start_date_key, 1, 4)),
  1) AS yoy_pct
FROM fact_contracts f
GROUP BY start_year
ORDER BY start_year;
```

Real output: £87.2M in 2024, £144.8M in 2025, a year on year change of
+66.0%.

## The Excel dashboard

One workbook, built on PivotTables against the same 8 CSVs, laid out as
a single dashboard sheet in the brand palette:

- Slicers across the top: Year, Region, Stage.
- A row of KPI cells (Pipeline Value, Win Rate, Active Contracts,
  Portfolio Value, On-Time %, Avg Contract Value), each a PivotTable
  value with conditional formatting (data bars and a traffic light icon
  set) standing in for a gauge.
- Pipeline half (left): a PivotChart funnel of bid conversion, a PivotChart
  line of the 3 month rolling win rate, and a PivotChart bar of pipeline
  value by region.
- Contracts half (right): a PivotChart of portfolio value over time as a
  running total, with a field toggle driven by `dim_measure_selector`
  (Value, Count, Win Rate) so one chart answers three questions, a
  PivotChart of portfolio value by status, the contractor league table
  as a sorted PivotTable, and delivery status by region.
- A full width contract ledger PivotTable detail view along the bottom.

## The Python and Matplotlib charts

`housing_data/generate_charts.py` reads the same CSVs with pandas and
renders 5 charts with Matplotlib, saved to `housing_data/images/`. Run
it yourself with:

```bash
python3 housing_data/generate_charts.py
```

| File | Chart |
|---|---|
| `01_funnel.png` | The bid funnel narrowing from Identified through to Won |
| `02_monthly_trend.png` | Monthly bid volume and the 3 month rolling win rate, as small multiples |
| `03_region.png` | Bid volume by region against average bid value by region |
| `04_contractor_league.png` | The contractor league table, on time % coloured by tier |
| `05_status.png` | Contract status breakdown across the current portfolio |

## What is not in this repository

The Excel workbook (`.xlsx`) and its exported screenshot
(`housing_excel_dashboard.png`, referenced by `housing-case-study.html`)
are built separately in Excel, since this environment cannot run desktop
Excel. They are added to the repository afterwards, alongside this
generated data.
