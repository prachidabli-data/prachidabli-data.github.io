"""
Generate a synthetic undergraduate exam-performance dataset.

Mirrors an undergraduate BSc Mathematics "Statistical Modelling" coursework
brief: build an illustrative dataset with realistic relationships baked in,
then use it to demonstrate distributions, correlation and regression.

The five numeric predictors are NOT generated independently. Real study
habits, attendance, sleep, prior attainment and extracurricular load tend
to move together (a "conscientiousness" effect), so a target correlation
structure is imposed via a Gaussian copula before each variable is given
its own marginal distribution. This means the predictors carry realistic,
moderate multicollinearity rather than the near-zero correlation an
independent draw would produce.

The data is entirely synthetic (no real students), generated with a fixed
random seed so the case study numbers are reproducible.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, beta as beta_dist, poisson

RNG_SEED = 42
N_STUDENTS = 320

rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Correlated latent variables (Gaussian copula)
# ---------------------------------------------------------------------------
# Order: study_hours, attendance, sleep, prior_gpa, extracurricular
# Target correlations encode a plausible "conscientiousness" factor: students
# who study more also tend to attend more and have a higher prior GPA;
# heavy extracurricular load trades off slightly against study time and sleep.
target_corr = np.array([
    [1.00,  0.30, -0.15,  0.35, -0.10],
    [0.30,  1.00,  0.05,  0.25, -0.05],
    [-0.15, 0.05,  1.00,  0.10, -0.10],
    [0.35,  0.25,  0.10,  1.00, -0.05],
    [-0.10, -0.05, -0.10, -0.05, 1.00],
])

L = np.linalg.cholesky(target_corr)
z_indep = rng.standard_normal(size=(N_STUDENTS, 5))
z_corr = z_indep @ L.T  # correlated standard normals, columns as above

z_study, z_att, z_sleep, z_gpa, z_extra = z_corr.T

# ---------------------------------------------------------------------------
# 2. Map each latent column to its target marginal distribution
# ---------------------------------------------------------------------------

# Study hours: approx normal, so an affine transform of the latent normal
# preserves the copula's linear correlation exactly.
study_hours = 12.0 + 4.2 * z_study
study_hours = np.clip(study_hours, 0.5, 28)

# Sleep hours: approx normal.
sleep_hours = 6.8 + 1.15 * z_sleep
sleep_hours = np.clip(sleep_hours, 3.5, 10.5)

# Prior GPA: approx normal.
prior_gpa = 3.0 + 0.42 * z_gpa
prior_gpa = np.clip(prior_gpa, 1.6, 4.0)

# Attendance rate (%): skewed Beta marginal, applied via the copula
# (uniform-transform then inverse-CDF), which distorts the exact linear
# correlation slightly but keeps the intended direction and rough magnitude.
u_att = norm.cdf(z_att)
attendance_rate = beta_dist.ppf(u_att, a=6, b=1.6) * 100
attendance_rate = np.clip(attendance_rate, 20, 100)

# Extracurricular hours: Poisson marginal via the same copula transform.
u_extra = norm.cdf(z_extra)
extracurricular_hours = poisson.ppf(u_extra, mu=4.0).astype(float)

# ---------------------------------------------------------------------------
# 3. Study method: categorical, assigned independently of the above
# ---------------------------------------------------------------------------
study_method = rng.choice(
    ["Self-study", "Group study", "Tutoring"],
    size=N_STUDENTS,
    p=[0.5, 0.32, 0.18],
)

# Small, deliberate effect of study method baked into the score below.
# Reference category is "Group study" (dropped by patsy/pandas as the first
# level alphabetically), so these effects are relative to Group study.
method_effect = pd.Series(study_method).map(
    {"Self-study": 0.0, "Group study": 2.4, "Tutoring": 4.1}
).to_numpy()

# ---------------------------------------------------------------------------
# 4. Dependent variable: exam score (0-100)
# ---------------------------------------------------------------------------
# A linear combination of the predictors above, plus a mild penalty for
# short sleep (below ~6 hours) and irreducible noise, then clipped to 0-100.
# These are the TRUE generating coefficients — the regression later in the
# analysis recovers estimates of these from noisy, correlated data.

TRUE_COEFS = {
    "intercept": 12.0,
    "study_hours_per_week": 1.15,
    "attendance_rate": 0.27,
    "prior_gpa": 8.5,
    "extracurricular_hours": -0.30,
    "sleep_penalty_per_hour_below_6": -1.8,
    "study_method_self_study_vs_group": 0.0 - 2.4,
    "study_method_tutoring_vs_group": 4.1 - 2.4,
    "noise_sd": 7.5,
}

sleep_penalty = np.where(sleep_hours < 6.0, (6.0 - sleep_hours) * 1.8, 0.0)
noise = rng.normal(loc=0, scale=TRUE_COEFS["noise_sd"], size=N_STUDENTS)

exam_score = (
    TRUE_COEFS["intercept"]
    + TRUE_COEFS["study_hours_per_week"] * study_hours
    + TRUE_COEFS["attendance_rate"] * attendance_rate
    + TRUE_COEFS["prior_gpa"] * prior_gpa
    - sleep_penalty
    + method_effect
    + TRUE_COEFS["extracurricular_hours"] * extracurricular_hours
    + noise
)
exam_score = np.clip(exam_score, 0, 100)

# ---------------------------------------------------------------------------
# 5. Assemble and save
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
print("\nRealised predictor correlation matrix (Pearson, on generated data):")
predictor_cols = ["study_hours_per_week", "attendance_rate", "sleep_hours",
                   "prior_gpa", "extracurricular_hours"]
print(df[predictor_cols].corr().round(3))
print("\nTrue generating coefficients (for later comparison to fitted model):")
for k, v in TRUE_COEFS.items():
    print(f"  {k}: {v}")
