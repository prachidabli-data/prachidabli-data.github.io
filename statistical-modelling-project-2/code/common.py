"""
Shared parameters and scenario definitions for Statistical Modelling Project II.

Every constant below is labelled as one of:
    FROM PROJECT I      : a real, fitted value taken from Project I's regression
    ASSUMPTION          : a choice made for this project, not measured by Project I

This file is the single source of truth for scenario parameters. Every stage
(LP, MDP, MCTS, risk extension) imports from here rather than re-declaring
its own numbers, so the notebook and the scripts always agree.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# ---------------------------------------------------------------------------
# Chart palette, matching the rest of the portfolio (presentational only,
# not part of the scenario itself)
# ---------------------------------------------------------------------------
PALETTE = {
    "navy": "#16294A",
    "navy_light": "#33547F",
    "amber": "#E0A030",
    "amber_dark": "#BD831C",
    "muted": "#8A93A6",
    "line": "#E7DFD0",
    "good": "#3F7D5A",
    "bad": "#B23A3A",
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
STUDENT_CSV = DATA_DIR / "student_exam_performance.csv"


def load_student_data() -> pd.DataFrame:
    """Load Project I's synthetic dataset of 320 students."""
    return pd.read_csv(STUDENT_CSV)


def mean_weekly_study_hours() -> float:
    """
    Mean of study_hours_per_week across all 320 students in Project I's
    dataset. FROM PROJECT I (a real descriptive statistic of the dataset,
    not a fitted coefficient).

    Used to convert Project I's categorical study-method bonus (a fixed
    points-per-week premium for Group study or Tutoring relative to
    Self-study) into a per-hour premium, since Stage 1's linear programme
    allocates hours, not a choice of single method for the whole week.
    """
    return load_student_data().study_hours_per_week.mean()


MEAN_WEEKLY_STUDY_HOURS = round(float(mean_weekly_study_hours()), 4)

# ---------------------------------------------------------------------------
# FROM PROJECT I: fitted regression coefficients (multiple OLS)
# ---------------------------------------------------------------------------
PROJECT_I_INTERCEPT = 10.96
PROJECT_I_STUDY_HOURS_COEF = 1.13          # points per hour of study per week
PROJECT_I_ATTENDANCE_COEF = 0.28           # points per percentage point of attendance
PROJECT_I_PRIOR_GPA_COEF = 7.73            # points per GPA point (GPA is fixed, not a decision)
PROJECT_I_EXTRACURRICULAR_COEF = -0.52     # points per hour of extracurricular activity
PROJECT_I_GROUP_STUDY_DUMMY = 1.83         # points per week, Group study vs Self-study
PROJECT_I_TUTORING_DUMMY = 5.56            # points per week, Tutoring vs Self-study
PROJECT_I_SLEEP_COEF_ESTIMATED = 0.40      # estimated by Project I, not significant (p = 0.295)
PROJECT_I_R_SQUARED = 0.637

# FROM PROJECT I: the true, generating values (known because Project I's
# dataset was itself synthetic, built from a programmed model). These are
# what the regression was trying, and partly failing, to recover.
TRUE_STUDY_HOURS_COEF = 1.15
TRUE_ATTENDANCE_COEF = 0.27
TRUE_PRIOR_GPA_COEF = 8.50
TRUE_EXTRACURRICULAR_COEF = -0.30
TRUE_GROUP_STUDY_DUMMY = 2.40
TRUE_TUTORING_DUMMY = 4.10
TRUE_SLEEP_PENALTY_PER_HOUR_BELOW_6 = -1.80   # missed entirely by the regression
TRUE_NOISE_SD = 7.50                          # exam score noise, used in Stage 4

# ---------------------------------------------------------------------------
# ASSUMPTION: Stage 1, linear programming baseline
# ---------------------------------------------------------------------------
# Project I's study-method dummies are a fixed bonus for using that method at
# all that week, not a per-hour rate. Converting them into per-hour premiums
# so they fit into an hours-allocation LP is a modelling choice for this
# project, not something Project I estimated.
LP_GROUP_STUDY_PREMIUM_PER_HOUR = PROJECT_I_GROUP_STUDY_DUMMY / MEAN_WEEKLY_STUDY_HOURS
LP_TUTORING_PREMIUM_PER_HOUR = PROJECT_I_TUTORING_DUMMY / MEAN_WEEKLY_STUDY_HOURS

LP_SCHEDULED_LECTURE_HOURS = 15   # ASSUMPTION: hours corresponding to 100% attendance

LP_TOTAL_HOURS_BUDGET = 40        # ASSUMPTION: discretionary hours per week
LP_TUTORING_MAX_HOURS = 3         # ASSUMPTION: limited paid slots
LP_GROUP_STUDY_MAX_HOURS = 6      # ASSUMPTION: depends on others being available
LP_LECTURE_MAX_HOURS = LP_SCHEDULED_LECTURE_HOURS  # cannot exceed 100% attendance
LP_EXTRACURRICULAR_MIN_HOURS = 2  # ASSUMPTION: existing commitment

# ---------------------------------------------------------------------------
# ASSUMPTION: Stage 2, Markov decision process
# ---------------------------------------------------------------------------
MDP_TERM_WEEKS = 12
MDP_KNOWLEDGE_MAX = 10
MDP_FATIGUE_LEVELS = 3            # 0 fresh, 1 tired, 2 exhausted
MDP_TUTORING_SESSIONS_MAX = 4

MDP_START_STATE = {"t": 0, "k": 2, "f": 0, "s": 4}   # ASSUMPTION: given starting point

# Effective per-hour points implied by Project I's coefficients plus the
# Stage 1 per-hour premiums above, used only to set the *ratio* between the
# three study actions' knowledge-gain probabilities. The absolute
# probabilities are an assumption; the ratio between them is anchored to
# Project I's numbers.
_EFFECTIVE_SELF_STUDY = PROJECT_I_STUDY_HOURS_COEF
_EFFECTIVE_GROUP_STUDY = PROJECT_I_STUDY_HOURS_COEF + LP_GROUP_STUDY_PREMIUM_PER_HOUR
_EFFECTIVE_TUTORING = PROJECT_I_STUDY_HOURS_COEF + LP_TUTORING_PREMIUM_PER_HOUR

# ASSUMPTION: base success probability for one week of self-study, then
# scaled by the effective per-hour ratios above for group study and
# tutoring. Tutoring's target expected gain is split into a P(+1) and a
# small P(+2), rather than a single P(+1), per the brief.
MDP_P_SELF_STUDY = 0.50
MDP_P_GROUP_STUDY = round(MDP_P_SELF_STUDY * (_EFFECTIVE_GROUP_STUDY / _EFFECTIVE_SELF_STUDY), 2)
_TUTORING_TARGET_EXPECTED_GAIN = MDP_P_SELF_STUDY * (_EFFECTIVE_TUTORING / _EFFECTIVE_SELF_STUDY)
MDP_P_TUTORING_PLUS2 = 0.10        # ASSUMPTION: small chance of a double gain
MDP_P_TUTORING_PLUS1 = round(_TUTORING_TARGET_EXPECTED_GAIN - 2 * MDP_P_TUTORING_PLUS2, 2)

# ASSUMPTION: fatigue multiplies study success probabilities down.
MDP_FATIGUE_FACTOR = {0: 1.00, 1: 0.75, 2: 0.40}

# ASSUMPTION: studying raises fatigue by 1 with this probability (capped at
# the maximum level); resting lowers fatigue by 1 with no knowledge gain.
MDP_P_FATIGUE_INCREASE = 0.50

# ASSUMPTION: terminal reward, calibrated so it spans roughly Project I's
# observed exam score range (k = 0 gives about 45, k = 10 gives about 95).
MDP_TERMINAL_A = 45.0
MDP_TERMINAL_B = 5.0


def terminal_expected_score(k: int) -> float:
    """E[score | final knowledge k], clipped to the 0 to 100 range."""
    return float(np.clip(MDP_TERMINAL_A + MDP_TERMINAL_B * k, 0.0, 100.0))


# ---------------------------------------------------------------------------
# ASSUMPTION: Stage 4, risk-sensitive extension
# ---------------------------------------------------------------------------
RISK_TARGET_GRADE = 70.0   # ASSUMPTION: example target used in the brief (for example, a scholarship threshold)


if __name__ == "__main__":
    print("Mean weekly study hours (from Project I data):", MEAN_WEEKLY_STUDY_HOURS)
    print("LP group study premium per hour (assumption):", round(LP_GROUP_STUDY_PREMIUM_PER_HOUR, 4))
    print("LP tutoring premium per hour (assumption):", round(LP_TUTORING_PREMIUM_PER_HOUR, 4))
    print("MDP P(self-study):", MDP_P_SELF_STUDY)
    print("MDP P(group study):", MDP_P_GROUP_STUDY)
    print("MDP P(tutoring +1):", MDP_P_TUTORING_PLUS1, "P(tutoring +2):", MDP_P_TUTORING_PLUS2)
