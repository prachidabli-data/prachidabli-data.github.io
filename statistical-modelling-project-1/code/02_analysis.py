"""
Statistical Modelling Project I — analysis.

Demonstrates: descriptive statistics, distribution checks, correlation
analysis, simple & multiple linear regression, and hypothesis testing
(two-sample t-test, one-way ANOVA), on the synthetic student exam-
performance dataset.

Outputs:
  - Console summary of every test (also written to results_summary.txt)
  - Charts saved to ../images/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

sns.set_theme(style="whitegrid", font_scale=1.02)
PALETTE = {"navy": "#16294A", "amber": "#E0A030", "navy_light": "#33547F"}

DATA_PATH = "statistical-modelling-project-1/data/student_exam_performance.csv"
IMG_DIR = "statistical-modelling-project-1/images"
RESULTS_PATH = "statistical-modelling-project-1/results_summary.txt"

df = pd.read_csv(DATA_PATH)

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("1. DESCRIPTIVE STATISTICS")
log("=" * 70)
desc = df.select_dtypes("number").describe().round(2)
log(desc)

# ---------------------------------------------------------------------------
# 2. DISTRIBUTION OF EXAM SCORE
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("2. DISTRIBUTION OF EXAM SCORE")
log("=" * 70)

shapiro_stat, shapiro_p = stats.shapiro(df["exam_score"])
log(f"Shapiro-Wilk normality test: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
log("-> " + ("Fails to reject normality (p > .05)" if shapiro_p > 0.05
             else "Rejects normality (p <= .05)"))
log(f"Skewness: {stats.skew(df['exam_score']):.3f}, "
    f"Kurtosis (excess): {stats.kurtosis(df['exam_score']):.3f}")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
sns.histplot(df["exam_score"], bins=22, kde=True, color=PALETTE["navy_light"],
             edgecolor="white", ax=ax)
mu, sigma = df["exam_score"].mean(), df["exam_score"].std()
ax.axvline(mu, color=PALETTE["amber"], linestyle="--", linewidth=2,
           label=f"Mean = {mu:.1f}")
ax.set_title("Distribution of Exam Scores (n = 320)", fontsize=13, weight="bold")
ax.set_xlabel("Exam score")
ax.set_ylabel("Number of students")
ax.legend()
fig.tight_layout()
fig.savefig(f"{IMG_DIR}/01_distribution_exam_score.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. CORRELATION ANALYSIS
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("3. CORRELATION ANALYSIS (Pearson)")
log("=" * 70)

num_cols = ["study_hours_per_week", "attendance_rate", "sleep_hours",
            "prior_gpa", "extracurricular_hours", "exam_score"]
corr = df[num_cols].corr(method="pearson").round(3)
log(corr)

for col in num_cols[:-1]:
    r, p = stats.pearsonr(df[col], df["exam_score"])
    log(f"  exam_score vs {col}: r = {r:.3f}, p = {p:.4g}")

fig, ax = plt.subplots(figsize=(7, 5.6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
            square=True, linewidths=0.6, linecolor="white",
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(f"{IMG_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. SIMPLE LINEAR REGRESSION: exam_score ~ study_hours_per_week
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("4. SIMPLE LINEAR REGRESSION — exam_score ~ study_hours_per_week")
log("=" * 70)

simple_model = smf.ols("exam_score ~ study_hours_per_week", data=df).fit()
log(simple_model.summary())

slope = simple_model.params["study_hours_per_week"]
intercept = simple_model.params["Intercept"]
r2_simple = simple_model.rsquared

fig, ax = plt.subplots(figsize=(7.5, 5.2))
sns.regplot(x="study_hours_per_week", y="exam_score", data=df,
            scatter_kws={"alpha": 0.45, "color": PALETTE["navy_light"], "s": 28},
            line_kws={"color": PALETTE["amber"], "linewidth": 2.5}, ax=ax)
ax.set_title(f"Exam Score vs Weekly Study Hours\n"
             f"score = {intercept:.1f} + {slope:.2f} x hours   "
             f"(R² = {r2_simple:.3f})", fontsize=12.5, weight="bold")
ax.set_xlabel("Study hours per week")
ax.set_ylabel("Exam score")
fig.tight_layout()
fig.savefig(f"{IMG_DIR}/03_regression_study_hours.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. MULTIPLE LINEAR REGRESSION
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("5. MULTIPLE LINEAR REGRESSION")
log("=" * 70)

multi_model = smf.ols(
    "exam_score ~ study_hours_per_week + attendance_rate + sleep_hours "
    "+ prior_gpa + extracurricular_hours + C(study_method)",
    data=df,
).fit()
log(multi_model.summary())

# Variance Inflation Factors (multicollinearity check)
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[["study_hours_per_week", "attendance_rate", "sleep_hours",
        "prior_gpa", "extracurricular_hours"]].copy()
X = sm.add_constant(X)
vif = pd.DataFrame({
    "variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
}).round(2)
log("\nVariance Inflation Factors:")
log(vif.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. HYPOTHESIS TEST — two-sample t-test on attendance split
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("6. TWO-SAMPLE T-TEST — attendance >= 85% vs < 85%")
log("=" * 70)

high_att = df.loc[df["attendance_rate"] >= 85, "exam_score"]
low_att = df.loc[df["attendance_rate"] < 85, "exam_score"]

t_stat, t_p = stats.ttest_ind(high_att, low_att, equal_var=False)
log(f"High-attendance group: n = {len(high_att)}, mean = {high_att.mean():.2f}, "
    f"sd = {high_att.std():.2f}")
log(f"Low-attendance group:  n = {len(low_att)}, mean = {low_att.mean():.2f}, "
    f"sd = {low_att.std():.2f}")
log(f"Welch's t-test: t = {t_stat:.3f}, p = {t_p:.4g}")
log("-> " + ("Statistically significant difference (p < .05)" if t_p < 0.05
             else "No statistically significant difference (p >= .05)"))

# 95% CI for the mean difference
mean_diff = high_att.mean() - low_att.mean()
se_diff = np.sqrt(high_att.var(ddof=1) / len(high_att) + low_att.var(ddof=1) / len(low_att))
ci_low, ci_high = mean_diff - 1.96 * se_diff, mean_diff + 1.96 * se_diff
log(f"Mean difference = {mean_diff:.2f} points, 95% CI = [{ci_low:.2f}, {ci_high:.2f}]")

# ---------------------------------------------------------------------------
# 7. ONE-WAY ANOVA — exam_score by study_method
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("7. ONE-WAY ANOVA — exam_score by study_method")
log("=" * 70)

groups = [g["exam_score"].values for _, g in df.groupby("study_method")]
f_stat, anova_p = stats.f_oneway(*groups)
log(f"F-statistic = {f_stat:.3f}, p = {anova_p:.4g}")
log("-> " + ("At least one group mean differs significantly (p < .05)" if anova_p < 0.05
             else "No statistically significant difference between groups (p >= .05)"))

group_means = df.groupby("study_method")["exam_score"].agg(["count", "mean", "std"]).round(2)
log(group_means)

fig, ax = plt.subplots(figsize=(7.5, 5))
order = ["Self-study", "Group study", "Tutoring"]
sns.boxplot(x="study_method", y="exam_score", data=df, order=order,
            palette=[PALETTE["navy_light"], PALETTE["amber"], PALETTE["navy"]], ax=ax)
sns.stripplot(x="study_method", y="exam_score", data=df, order=order,
              color="black", alpha=0.25, size=3, jitter=0.2, ax=ax)
ax.set_title(f"Exam Score by Study Method\nOne-way ANOVA: F = {f_stat:.2f}, p = {anova_p:.4g}",
             fontsize=12.5, weight="bold")
ax.set_xlabel("")
ax.set_ylabel("Exam score")
fig.tight_layout()
fig.savefig(f"{IMG_DIR}/04_boxplot_study_method.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Save log
# ---------------------------------------------------------------------------
with open(RESULTS_PATH, "w") as f:
    f.write("\n".join(log_lines))

log(f"\nSaved full results to {RESULTS_PATH}")
log(f"Saved charts to {IMG_DIR}/")
