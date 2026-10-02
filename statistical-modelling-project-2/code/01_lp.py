"""
Stage 1: Linear programming baseline for Statistical Modelling Project II.

Question: if a student's exam score depended on their hours in a purely
linear, deterministic way, with no diminishing returns and no uncertainty,
how should they split a 40 hour weekly budget across self-study, group
study, tutoring, lecture attendance and extracurricular activity?

This deliberately ignores everything that makes the real problem hard
(uncertainty, fatigue, forgetting, sequential decisions across a term).
That is the point: Stage 1 is the naive baseline that Stage 2's MDP is
built to improve on.

Objective (maximise, so we minimise its negative for scipy's linprog):
    score = intercept
          + self_study_hours   * study_hours_coef
          + group_study_hours  * (study_hours_coef + group_study_premium_per_hour)
          + tutoring_hours     * (study_hours_coef + tutoring_premium_per_hour)
          + lecture_hours      * attendance_coef_per_lecture_hour
          + extracurricular_hours * extracurricular_coef

Prior GPA and sleep are not decisions in this scenario (GPA is fixed, and
sleep sits outside the weekly hours budget entirely, see the note in
build_lp), so neither term appears in the objective. The "predicted score"
below should be read as the decision-controlled part of the score plus
Project I's intercept, not a full forecast for a specific student.

Constraints:
    total hours                <= LP_TOTAL_HOURS_BUDGET (default 40)
    tutoring hours              <= LP_TUTORING_MAX_HOURS (limited paid slots)
    group study hours           <= LP_GROUP_STUDY_MAX_HOURS (depends on others)
    lecture hours                <= LP_LECTURE_MAX_HOURS (100 percent attendance)
    extracurricular hours       >= LP_EXTRACURRICULAR_MIN_HOURS (existing commitment)
    all hours                   >= 0
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
import common

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

VARS = ["self_study", "group_study", "tutoring", "lecture", "extracurricular"]
VAR_LABELS = {
    "self_study": "Self-study",
    "group_study": "Group study",
    "tutoring": "Tutoring",
    "lecture": "Lecture",
    "extracurricular": "Extracurricular",
}

CONSTRAINT_NAMES = [
    "Weekly hours budget",
    "Tutoring hours cap",
    "Group study hours cap",
    "Lecture hours cap",
    "Extracurricular minimum",
]


def per_hour_coefficients() -> dict:
    """
    Points per hour for each decision variable, built from Project I's
    fitted coefficients plus the Stage 1 premiums derived in common.py.

    Sleep is deliberately absent here. Project I's regression estimated a
    sleep effect that was not significant, but the true generating model
    (known because Project I's data was synthetic) had a real penalty for
    sleeping under 6 hours a night. Since the LP only sees what Project I's
    regression estimated, it cannot see sleep's true effect either, which
    is exactly why sleep is protected as a fixed constraint outside the
    weekly hours budget rather than something the LP could trade away.
    """
    lecture_coef_per_hour = common.PROJECT_I_ATTENDANCE_COEF * (100.0 / common.LP_SCHEDULED_LECTURE_HOURS)
    return {
        "self_study": common.PROJECT_I_STUDY_HOURS_COEF,
        "group_study": common.PROJECT_I_STUDY_HOURS_COEF + common.LP_GROUP_STUDY_PREMIUM_PER_HOUR,
        "tutoring": common.PROJECT_I_STUDY_HOURS_COEF + common.LP_TUTORING_PREMIUM_PER_HOUR,
        "lecture": lecture_coef_per_hour,
        "extracurricular": common.PROJECT_I_EXTRACURRICULAR_COEF,
    }


def build_lp(total_hours_budget: float = None):
    """Build and solve the Stage 1 LP. Returns (result, coefs, A_ub, b_ub)."""
    if total_hours_budget is None:
        total_hours_budget = common.LP_TOTAL_HOURS_BUDGET

    coefs = per_hour_coefficients()
    c = [-coefs[v] for v in VARS]

    A_ub = [
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, -1],
    ]
    b_ub = [
        total_hours_budget,
        common.LP_TUTORING_MAX_HOURS,
        common.LP_GROUP_STUDY_MAX_HOURS,
        common.LP_LECTURE_MAX_HOURS,
        -common.LP_EXTRACURRICULAR_MIN_HOURS,
    ]
    bounds = [(0, None)] * len(VARS)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return result, coefs, A_ub, b_ub


def predicted_score(allocation: dict) -> float:
    coefs = per_hour_coefficients()
    return common.PROJECT_I_INTERCEPT + sum(coefs[v] * allocation[v] for v in VARS)


def shadow_prices(result) -> dict:
    """
    Dual values (shadow prices) of each A_ub constraint, converted back
    into "extra points per unit relaxation" for the original maximisation
    problem (linprog solved the negated, minimisation form, so signs flip
    back here).
    """
    marginals = result.ineqlin.marginals
    return {name: -m for name, m in zip(CONSTRAINT_NAMES, marginals)}


def brute_force_grid_search(total_hours_budget: float = None, extracurricular_max_check: int = 10) -> dict:
    """
    Coarse grid search over feasible allocations, used only to validate the
    LP result independently. Tutoring, group study and lecture hours are
    swept over their full integer range; extracurricular hours are swept
    from the minimum up to a small cap (since its coefficient is negative,
    the optimum should always sit at the minimum, and this check confirms
    that rather than assuming it); self-study soaks up whatever hours
    remain in the budget.
    """
    if total_hours_budget is None:
        total_hours_budget = common.LP_TOTAL_HOURS_BUDGET

    coefs = per_hour_coefficients()
    best_score = -np.inf
    best_alloc = None

    for tutoring in range(0, int(common.LP_TUTORING_MAX_HOURS) + 1):
        for group_study in range(0, int(common.LP_GROUP_STUDY_MAX_HOURS) + 1):
            for lecture in range(0, int(common.LP_LECTURE_MAX_HOURS) + 1):
                for extracurricular in range(int(common.LP_EXTRACURRICULAR_MIN_HOURS), extracurricular_max_check + 1):
                    used = tutoring + group_study + lecture + extracurricular
                    self_study = total_hours_budget - used
                    if self_study < 0:
                        continue
                    alloc = {
                        "self_study": self_study,
                        "group_study": group_study,
                        "tutoring": tutoring,
                        "lecture": lecture,
                        "extracurricular": extracurricular,
                    }
                    score = sum(coefs[v] * alloc[v] for v in VARS)
                    if score > best_score:
                        best_score = score
                        best_alloc = alloc

    return {"allocation": best_alloc, "decision_score": best_score}


def sensitivity_sweep(budgets=range(20, 51)) -> "pd.DataFrame":
    """Optimal predicted score for a range of weekly hour budgets."""
    import pandas as pd

    rows = []
    for budget in budgets:
        result, _, _, _ = build_lp(budget)
        allocation = dict(zip(VARS, result.x))
        rows.append({"budget": budget, "predicted_score": predicted_score(allocation)})
    return pd.DataFrame(rows)


def plot_allocation(allocation: dict, path):
    p = common.PALETTE
    labels = [VAR_LABELS[v] for v in VARS]
    values = [allocation[v] for v in VARS]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, values, color=p["amber"], zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.0f}h",
                ha="center", fontsize=11, fontweight="bold", color=p["navy"])
    ax.set_ylabel("Hours per week")
    ax.set_title("The LP baseline: a corner solution, not a spread",
                  fontsize=14, fontweight="bold", color=p["navy"], loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=p["line"], linewidth=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_sensitivity(sweep_df, path):
    p = common.PALETTE
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(sweep_df.budget, sweep_df.predicted_score, color=p["navy"], linewidth=2.5, zorder=3)
    ax.axvline(common.LP_TOTAL_HOURS_BUDGET, color=p["amber_dark"], linestyle="--", linewidth=1.5, zorder=2)
    ax.set_xlabel("Weekly hours budget")
    ax.set_ylabel("Optimal predicted score")
    ax.set_title("Score rises fastest before the capped activities are maxed out",
                  fontsize=14, fontweight="bold", color=p["navy"], loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=p["line"], linewidth=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import pandas as pd

    result, coefs, A_ub, b_ub = build_lp()
    allocation = dict(zip(VARS, result.x))
    print("Optimal allocation (hours per week):")
    for v in VARS:
        print(f"  {v}: {allocation[v]:.4f}")
    print(f"Predicted score: {predicted_score(allocation):.4f}")
    print("Shadow prices:")
    for name, price in shadow_prices(result).items():
        print(f"  {name}: {price:.4f}")

    grid = brute_force_grid_search()
    lp_decision_score = predicted_score(allocation) - common.PROJECT_I_INTERCEPT
    print("\nBrute-force grid search best decision-only score:", round(grid["decision_score"], 4))
    print("LP decision-only score:", round(lp_decision_score, 4))
    print("Grid search matches LP:", abs(grid["decision_score"] - lp_decision_score) < 1e-6)

    result41, _, _, _ = build_lp(41)
    allocation41 = dict(zip(VARS, result41.x))
    score_gain = predicted_score(allocation41) - predicted_score(allocation)
    print(f"\nScore at 41h budget: {predicted_score(allocation41):.4f} "
          f"(gain of {score_gain:.4f}, shadow price predicted {shadow_prices(result)['Weekly hours budget']:.4f})")

    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    common.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    alloc_df = pd.DataFrame([{"activity": VAR_LABELS[v], "hours_per_week": allocation[v]} for v in VARS])
    alloc_df.to_csv(common.DATA_DIR / "lp_allocation.csv", index=False)

    shadow_df = pd.DataFrame(
        [{"constraint": name, "shadow_price": price} for name, price in shadow_prices(result).items()]
    )
    shadow_df.to_csv(common.DATA_DIR / "lp_shadow_prices.csv", index=False)

    sweep = sensitivity_sweep()
    sweep.to_csv(common.DATA_DIR / "lp_sensitivity.csv", index=False)

    plot_allocation(allocation, common.IMAGES_DIR / "01_lp_allocation.png")
    plot_sensitivity(sweep, common.IMAGES_DIR / "02_lp_sensitivity.png")
    print("\nSaved lp_allocation.csv, lp_shadow_prices.csv, lp_sensitivity.csv to data/")
    print("Saved 01_lp_allocation.png, 02_lp_sensitivity.png to images/")
