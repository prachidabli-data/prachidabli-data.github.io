"""
Load the housing CSVs into a local SQLite database and run the three
queries shown in the case study, to confirm they are correct and to
capture real output numbers.

SQLite is used here only as a stand-in query engine (no MySQL server is
available in this environment); all three queries stick to ANSI-standard
window-function syntax that runs unchanged on MySQL 8.0+.
"""

import sqlite3
import pandas as pd

conn = sqlite3.connect(":memory:")

for name in ["dim_date", "dim_region", "dim_contractor", "dim_stage",
             "dim_status", "fact_tenders", "fact_contracts"]:
    df = pd.read_csv(f"housing_data/{name}.csv")
    df.to_sql(name, conn, index=False)

print("=" * 70)
print("QUERY 1: Rolling 3-month win rate, by month identified")
print("=" * 70)
q1 = """
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
"""
df1 = pd.read_sql(q1, conn)
print(df1.tail(8).to_string(index=False))

print("\n" + "=" * 70)
print("QUERY 2: Contractor league table, ranked by on-time %")
print("=" * 70)
q2 = """
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
"""
df2 = pd.read_sql(q2, conn)
print(df2.to_string(index=False))

print("\n" + "=" * 70)
print("QUERY 3: Portfolio value by year, with year-over-year change")
print("=" * 70)
q3 = """
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
"""
df3 = pd.read_sql(q3, conn)
print(df3.to_string(index=False))

conn.close()
