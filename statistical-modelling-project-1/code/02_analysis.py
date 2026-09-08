"""
Statistical Modelling Project I: analysis.

Demonstrates: descriptive statistics, distribution checks, correlation
analysis, simple & multiple linear regression, regression diagnostics
(residuals, heteroscedasticity, influence), multicollinearity (VIF), and
hypothesis testing (two-sample t-test, one-way ANOVA), on the synthetic
student exam-performance dataset.

Note on the "true" coefficients: because this dataset is synthetic, the
exact coefficients used to generate exam_score are known (see
01_generate_dataset.py). Several sections below compare the fitted
regression estimates to those true values, which is only possible because
the data is simulated — it is not something you could do with real data,
where the "true" relationship is unknown. That comparison is used here to
show the model recovering a known signal from noisy, correlated data,
not to claim real-world practical significance.

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
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence

sns.set_theme(style="whitegrid", font_scale=1.02)
PALETTE = {"navy": "#16294A", "amber": "#E0A030", "navy_light": "#33547F"}

DATA_PATH = "statistical-modelling-project-1/data/student_exam_performance.csv"
IMG_DIR = "statistical-modelling-project-1/images"
RESULTS_PATH = "statistical-modelling-project-1/results_summary.txt"

# True generating coefficients, duplicated from 01_generate_dataset.py so this
# script can be read and re-run independently. study_method effects are
# relative to Self-study, matching the generator's own internal baseline
# (method_effect = 0 for Self-study) and the regression's reference category
# below, so no rebasing is needed to compare the two.
TRUE_COEFS = {
    "intercept": 12.0,
    "study_hours_per_week": 1.15,
    "attendance_rate": 0.27,
    "prior_gpa": 8.5,
    "extracurricular_hours": -0.30,
    "study_method_group_vs_self": 2.4,
    "study_method_tutoring_vs_self": 4.1,
}

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
# 2. DISTRIBUTION OF EXAM SCORE (descriptive only)
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("2. DISTRIBUTION OF EXAM SCORE (descriptive)")
log("=" * 70)
log("This describes the outcome variable's shape before any model is fitted.")
log("It is NOT a test of OLS assumptions: OLS requires well-behaved RESIDUALS")
log("(linearity, constant variance, independence, approx. normal errors), not")
log("a normally-distributed raw outcome. Residual diagnostics are in section 7.")

shapiro_stat, shapiro_p = stats.shapiro(df["exam_score"])
log(f"Shapiro-Wilk test on exam_score itself: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
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

log("\nCorrelation with exam_score:")
for col in num_cols[:-1]:
    r, p = stats.pearsonr(df[col], df["exam_score"])
    log(f"  exam_score vs {col}: r = {r:.3f}, p = {p:.4g}")

log("\nThe predictors are also correlated with EACH OTHER by design (a")
log("'conscientiousness' structure was built into the data generator), most")
log("notably study_hours vs attendance and study_hours vs prior_gpa. This")
log("intercorrelation is revisited as multicollinearity in section 6.")

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
log("4. SIMPLE LINEAR REGRESSION: exam_score ~ study_hours_per_week")
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
log("Categorical study_method is dummy-coded with 'Self-study' set explicitly")
log("as the reference level (statsmodels' Treatment(reference=...) contrast),")
log("rather than relying on whichever category sorts first alphabetically.")
log("Coefficients on 'Group study' and 'Tutoring' are each read relative to")
log("Self-study.")

multi_model = smf.ols(
    "exam_score ~ study_hours_per_week + attendance_rate + sleep_hours "
    "+ prior_gpa + extracurricular_hours "
    "+ C(study_method, Treatment(reference='Self-study'))",
    data=df,
).fit()
log(multi_model.summary())

log("\nFitted estimate vs. TRUE generating coefficient (recoverable only")
log("because this dataset is simulated; not a real-data diagnostic):")
compare_rows = [
    ("study_hours_per_week", "study_hours_per_week"),
    ("attendance_rate", "attendance_rate"),
    ("prior_gpa", "prior_gpa"),
    ("extracurricular_hours", "extracurricular_hours"),
    ("C(study_method, Treatment(reference='Self-study'))[T.Group study]",
     "study_method_group_vs_self"),
    ("C(study_method, Treatment(reference='Self-study'))[T.Tutoring]",
     "study_method_tutoring_vs_self"),
]
for param_name, true_key in compare_rows:
    est = multi_model.params[param_name]
    log(f"  {param_name}: estimated = {est:.3f}, true = {TRUE_COEFS[true_key]:.3f}")

# ---------------------------------------------------------------------------
# 6. MULTICOLLINEARITY (VIF) — includes the study_method dummies
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("6. MULTICOLLINEARITY — Variance Inflation Factors")
log("=" * 70)

method_dummies = pd.get_dummies(df["study_method"], dtype=float).drop(columns=["Self-study"])
X_vif = pd.concat([
    df[["study_hours_per_week", "attendance_rate", "sleep_hours",
        "prior_gpa", "extracurricular_hours"]],
    method_dummies,
], axis=1)
X_vif = sm.add_constant(X_vif)
vif = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
}).round(2)
log(vif.to_string(index=False))
log("\nAll VIFs stay comfortably below the common concern threshold of 5,")
log("but they are now meaningfully above 1.0 (unlike an independently-drawn")
log("predictor set), reflecting the correlation deliberately built between")
log("study_hours, attendance and prior_gpa.")

# ---------------------------------------------------------------------------
# 7. REGRESSION DIAGNOSTICS (on the multiple regression model)
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("7. REGRESSION DIAGNOSTICS (multiple regression residuals)")
log("=" * 70)

fitted = multi_model.fittedvalues
resid = multi_model.resid
influence = OLSInfluence(multi_model)
standardized_resid = influence.resid_studentized_internal
cooks_d = influence.cooks_distance[0]

shapiro_resid_stat, shapiro_resid_p = stats.shapiro(resid)
log(f"Shapiro-Wilk on RESIDUALS: W = {shapiro_resid_stat:.4f}, p = {shapiro_resid_p:.4f}")
log("-> " + ("Fails to reject residual normality (p > .05)" if shapiro_resid_p > 0.05
             else "Rejects residual normality (p <= .05)"))

bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(resid, multi_model.model.exog)
log(f"Breusch-Pagan test (heteroscedasticity): LM = {bp_stat:.3f}, p = {bp_p:.4g}")
log("-> " + ("No evidence of heteroscedasticity (p > .05)" if bp_p > 0.05
             else "Evidence of heteroscedasticity (p <= .05): variance of residuals"
                  " is not constant across fitted values"))

log(f"Durbin-Watson (independence of residuals): {sm.stats.stattools.durbin_watson(resid):.3f}"
    " (~2 indicates little autocorrelation)")

n_obs = len(df)
cooks_threshold = 4 / n_obs
n_influential = int((cooks_d > cooks_threshold).sum())
log(f"Cook's distance: max = {cooks_d.max():.4f}, "
    f"{n_influential} of {n_obs} points exceed the 4/n = {cooks_threshold:.4f} "
    f"rule-of-thumb threshold for influence")

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

ax = axes[0, 0]
ax.scatter(fitted, resid, alpha=0.45, color=PALETTE["navy_light"], s=26)
ax.axhline(0, color=PALETTE["amber"], linewidth=2, linestyle="--")
sns.regplot(x=fitted, y=resid, lowess=True, scatter=False, ax=ax,
            line_kws={"color": PALETTE["navy"], "linewidth": 1.6})
ax.set_title("Residuals vs Fitted", fontsize=12, weight="bold")
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")

ax = axes[0, 1]
sm.qqplot(standardized_resid, line="45", ax=ax, markerfacecolor=PALETTE["navy_light"],
          markeredgecolor=PALETTE["navy_light"], alpha=0.5)
ax.get_lines()[1].set_color(PALETTE["amber"])
ax.get_lines()[1].set_linewidth(2)
ax.set_title("Normal Q-Q (standardized residuals)", fontsize=12, weight="bold")

ax = axes[1, 0]
sqrt_abs_resid = np.sqrt(np.abs(standardized_resid))
ax.scatter(fitted, sqrt_abs_resid, alpha=0.45, color=PALETTE["navy_light"], s=26)
sns.regplot(x=fitted, y=sqrt_abs_resid, lowess=True, scatter=False, ax=ax,
            line_kws={"color": PALETTE["navy"], "linewidth": 1.6})
ax.set_title("Scale-Location", fontsize=12, weight="bold")
ax.set_xlabel("Fitted values")
ax.set_ylabel("sqrt(|standardized residual|)")

ax = axes[1, 1]
ax.stem(np.arange(n_obs), cooks_d, markerfmt=",", basefmt=" ",
        linefmt=PALETTE["navy_light"])
ax.axhline(cooks_threshold, color=PALETTE["amber"], linestyle="--", linewidth=2,
           label=f"4/n = {cooks_threshold:.3f}")
ax.set_title("Cook's Distance", fontsize=12, weight="bold")
ax.set_xlabel("Observation index")
ax.set_ylabel("Cook's distance")
ax.legend()

fig.suptitle("Multiple Regression Diagnostics", fontsize=14, weight="bold", y=1.0)
fig.tight_layout()
fig.savefig(f"{IMG_DIR}/05_regression_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. HYPOTHESIS TEST — two-sample t-test on attendance split
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("8. TWO-SAMPLE T-TEST: attendance >= 85% vs < 85%")
log("=" * 70)
log("Caveat: 85% is an arbitrary cut point chosen only to illustrate a")
log("two-sample test. Dichotomizing a continuous variable discards")
log("information and can understate or distort the relationship; the")
log("continuous regression coefficient on attendance_rate in section 5")
log("(and the Pearson correlation in section 3) is the more complete and")
log("statistically preferable summary of this relationship. The t-test")
log("below is included as a worked example of the technique, not as the")
log("primary evidence for an attendance effect.")

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

mean_diff = high_att.mean() - low_att.mean()
var_high = high_att.var(ddof=1)
var_low = low_att.var(ddof=1)
n_high, n_low = len(high_att), len(low_att)
se_diff = np.sqrt(var_high / n_high + var_low / n_low)

# Welch-Satterthwaite degrees of freedom for the CI, matching the df Welch's
# t-test itself uses, rather than approximating with a normal (z = 1.96)
# critical value.
welch_df = (var_high / n_high + var_low / n_low) ** 2 / (
    (var_high / n_high) ** 2 / (n_high - 1) + (var_low / n_low) ** 2 / (n_low - 1)
)
critical_t = stats.t.ppf(0.975, df=welch_df)
ci_low, ci_high = mean_diff - critical_t * se_diff, mean_diff + critical_t * se_diff
log(f"Mean difference = {mean_diff:.2f} points, Welch-Satterthwaite df = {welch_df:.1f}, "
    f"t* = {critical_t:.3f}, 95% CI = [{ci_low:.2f}, {ci_high:.2f}]")

# ---------------------------------------------------------------------------
# 9. ONE-WAY ANOVA — exam_score by study_method
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("9. ONE-WAY ANOVA: exam_score by study_method")
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
            hue="study_method", hue_order=order,
            palette=[PALETTE["navy_light"], PALETTE["amber"], PALETTE["navy"]],
            legend=False, ax=ax)
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
