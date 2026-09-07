"""
Generate a synthetic undergraduate exam-performance dataset.

Mirrors an undergraduate BSc Mathematics "Statistical Modelling" coursework
brief: build an illustrative dataset with realistic relationships baked in,
then use it to demonstrate distributions, correlation and regression.

The data is entirely synthetic (no real students), generated with a fixed
random seed so the case study numbers are reproducible.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_STUDENTS = 320

rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Independent (explanatory) variables
# ---------------------------------------------------------------------------

# Weekly study hours: roughly normal, clipped to a sensible range
study_hours = rng.normal(loc=12, scale=4.2, size=N_STUDENTS)
study_hours = np.clip(study_hours, 0.5, 28)

# Attendance rate (%): skewed towards high attendance, some low outliers
attendance_rate = rng.beta(a=6, b=1.6, size=N_STUDENTS) * 100
attendance_rate = np.clip(attendance_rate, 20, 100)

# Sleep hours per night: roughly normal around 6.8 hours
sleep_hours = rng.normal(loc=6.8, scale=1.15, size=N_STUDENTS)
sleep_hours = np.clip(sleep_hours, 3.5, 10.5)

# Prior GPA (0-4 scale): roughly normal
prior_gpa = rng.normal(loc=3.0, scale=0.42, size=N_STUDENTS)
prior_gpa = np.clip(prior_gpa, 1.6, 4.0)

# Extracurricular hours per week: Poisson-distributed count
extracurricular_hours = rng.poisson(lam=4.0, size=N_STUDENTS).astype(float)

# Study method: categorical, randomly assigned with unequal group sizes
study_method = rng.choice(
    ["Self-study", "Group study", "Tutoring"],
    size=N_STUDENTS,
    p=[0.5, 0.32, 0.18],
)

# Small, deliberate effect of study method baked into the score below
method_effect = pd.Series(study_method).map(
    {"Self-study": 0.0, "Group study": 2.4, "Tutoring": 4.1}
).to_numpy()

# ---------------------------------------------------------------------------
# 2. Dependent variable: exam score (0-100)
# ---------------------------------------------------------------------------
# A linear combination of the predictors above, plus a mild penalty for
# short sleep (below ~6 hours) and irreducible noise, then clipped to 0-100.

sleep_penalty = np.where(sleep_hours < 6.0, (6.0 - sleep_hours) * 1.8, 0.0)

noise = rng.normal(loc=0, scale=7.5, size=N_STUDENTS)

exam_score = (
    12.0
    + 1.15 * study_hours
    + 0.27 * attendance_rate
    + 8.5 * prior_gpa
    - sleep_penalty
    + method_effect
    - 0.30 * extracurricular_hours
    + noise
)
exam_score = np.clip(exam_score, 0, 100)

# ---------------------------------------------------------------------------
# 3. Assemble and save
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "student_id": [f"S{str(i+1).zfill(4)}" for i in range(N_STUDENTS)],
    "study_hours_per_week": study_hours.round(2),
    "attendance_rate": attendance_rate.round(1),
    "sleep_hours": sleep_hours.round(2),
    "prior_gpa": prior_gpa.round(2),
    "extracurricular_hours": extracurricular_hours.astype(int),
    "study_method": study_method,
    "exam_score": exam_score.round(1),
})

out_path = "statistical-modelling-project-1/data/student_exam_performance.csv"
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} rows to {out_path}")
print(df.select_dtypes("number").describe().round(2))
