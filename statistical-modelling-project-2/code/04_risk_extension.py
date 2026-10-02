"""
Stage 4: risk-sensitive extension for Statistical Modelling Project II.

Stages 2 and 3 both maximise the expected final score. This stage asks a
different question: what if the student cares about clearing a specific
grade (for example 70, to keep a scholarship), rather than the average
outcome? That is solved exactly the same way, backward induction on the
same state space, but with a different terminal reward: the probability
of scoring at least the target grade, using the exam noise already
defined for this project (common.TRUE_NOISE_SD), rather than the expected
score itself.
"""

from pathlib import Path
import importlib
import sys

sys.path.append(str(Path(__file__).resolve().parent))
import common

mdp = importlib.import_module("02_mdp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import norm


def target_grade_terminal(k: int, target: float = None) -> float:
    """P(final exam score >= target | knowledge k), from the normal CDF."""
    if target is None:
        target = common.RISK_TARGET_GRADE
    mean_score = common.terminal_expected_score(k)
    return float(norm.cdf((mean_score - target) / common.TRUE_NOISE_SD))


def solve_risk_sensitive(target: float = None):
    if target is None:
        target = common.RISK_TARGET_GRADE
    return mdp.solve(terminal_fn=lambda k: target_grade_terminal(k, target))


def count_disagreements(policy_a: dict, policy_b: dict) -> int:
    return sum(
        1
        for t in policy_a
        for state, action in policy_a[t].items()
        if policy_b[t][state] != action
    )


def threshold_sweep(expected_value_policy: dict, targets) -> pd.DataFrame:
    """How many states change their optimal action, for a range of target grades."""
    rows = []
    for target in targets:
        _, risk_policy, _ = solve_risk_sensitive(target)
        n_diff = count_disagreements(expected_value_policy, risk_policy)
        rows.append({"target": target, "states_differing": n_diff})
    return pd.DataFrame(rows)


def simulate_final_knowledge(policy_fn, n_sims: int, rng: np.random.Generator, start=None) -> np.ndarray:
    """Same trajectory simulation as mdp.simulate_policy, but returns final
    knowledge rather than the expected score, so Stage 4 can apply exam
    noise itself and compute percentile-based risk metrics."""
    if start is None:
        start = (common.MDP_START_STATE["k"], common.MDP_START_STATE["f"], common.MDP_START_STATE["s"])
    finals = np.empty(n_sims, dtype=int)
    for i in range(n_sims):
        k, f, s = start
        for t in range(mdp.T_MAX):
            action = policy_fn(t, k, f, s)
            outcomes = mdp.transitions((k, f, s), action)
            next_states, probs = zip(*outcomes)
            idx = rng.choice(len(next_states), p=probs)
            k, f, s = next_states[idx]
        finals[i] = k
    return finals


def simulate_noisy_scores(policy_fn, n_sims: int, rng: np.random.Generator, start=None) -> np.ndarray:
    """Realised exam scores (with noise), not the expected score used as
    the MDP's terminal reward, since the risk metrics below (percentiles,
    probability of reaching a target) only make sense on realised scores."""
    ks = simulate_final_knowledge(policy_fn, n_sims, rng, start)
    means = np.array([common.terminal_expected_score(k) for k in ks])
    scores = rng.normal(loc=means, scale=common.TRUE_NOISE_SD)
    return np.clip(scores, 0.0, 100.0)


def summarise(scores: np.ndarray, target: float = None) -> dict:
    if target is None:
        target = common.RISK_TARGET_GRADE
    worst_decile_cutoff = np.percentile(scores, 10)
    worst_decile_mean = scores[scores <= worst_decile_cutoff].mean()
    return {
        "mean": scores.mean(),
        "p_reach_target": (scores >= target).mean(),
        "p5": np.percentile(scores, 5),
        "worst_decile_mean": worst_decile_mean,
    }


ACTION_COLOURS = {
    "rest": common.PALETTE["muted"],
    "self_study": common.PALETTE["navy_light"],
    "group_study": common.PALETTE["amber"],
    "tutoring": common.PALETTE["amber_dark"],
}
ACTION_ORDER = ["rest", "self_study", "group_study", "tutoring"]


def plot_score_distributions(expected_scores: np.ndarray, risk_scores: np.ndarray, target: float, path):
    p = common.PALETTE
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bins = np.linspace(20, 100, 41)
    ax.hist(expected_scores, bins=bins, alpha=0.75, color=p["navy_light"], label="Expected-score policy", zorder=3)
    ax.hist(risk_scores, bins=bins, alpha=0.75, color=p["amber"], label="Target-grade policy", zorder=3)
    ax.axvline(target, color=p["bad"], linestyle="--", linewidth=1.5, zorder=4, label=f"Target grade ({target:.0f})")
    ax.set_xlabel("Final exam score")
    ax.set_ylabel("Simulated terms")
    ax.set_title("Same student, two goals: matching the average score or clearing a grade",
                  fontsize=12.5, fontweight="bold", color=p["navy"], loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_policy_diff_heatmap(expected_value_policy: dict, risk_policy: dict, f: int, s: int, path, title: str):
    grid_diff = np.zeros((mdp.K_MAX + 1, mdp.T_MAX))
    for t in range(mdp.T_MAX):
        for k in range(mdp.K_MAX + 1):
            grid_diff[k, t] = 1.0 if expected_value_policy[t][(k, f, s)] != risk_policy[t][(k, f, s)] else 0.0

    p = common.PALETTE
    cmap = ListedColormap(["#FFFFFF", p["bad"]])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.imshow(grid_diff, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_xlabel("Week")
    ax.set_ylabel("Knowledge")
    ax.set_xticks(range(mdp.T_MAX))
    ax.set_title(title, fontsize=13, fontweight="bold", color=p["navy"], loc="left", pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#FFFFFF", ec=p["line"]),
               plt.Rectangle((0, 0), 1, 1, color=p["bad"])]
    ax.legend(handles, ["Same action", "Different action"], loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    common.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    start = (common.MDP_START_STATE["k"], common.MDP_START_STATE["f"], common.MDP_START_STATE["s"])
    V_ev, policy_ev, _ = mdp.solve()
    V_risk, policy_risk, _ = solve_risk_sensitive()

    n_diff = count_disagreements(policy_ev, policy_risk)
    total_states = len(policy_ev) * len(list(mdp.all_states()))
    print(f"States where the two policies choose a different action at target 70: "
          f"{n_diff} of {total_states} ({n_diff / total_states:.1%})")

    print(f"\nAction at the start state: expected-score policy chooses "
          f"{policy_ev[0][start]}, target-grade policy chooses {policy_risk[0][start]}")

    rng_ev = np.random.default_rng(common.SEED)
    rng_risk = np.random.default_rng(common.SEED + 1)
    n_sims = 10_000

    scores_ev = simulate_noisy_scores(mdp.optimal_policy_fn(policy_ev), n_sims, rng_ev, start)
    scores_risk = simulate_noisy_scores(mdp.optimal_policy_fn(policy_risk), n_sims, rng_risk, start)

    summary_ev = summarise(scores_ev)
    summary_risk = summarise(scores_risk)

    print("\nExpected-score policy:", {k: round(v, 4) for k, v in summary_ev.items()})
    print("Target-grade policy:  ", {k: round(v, 4) for k, v in summary_risk.items()})

    summary_df = pd.DataFrame([
        {"policy": "Expected-score", **summary_ev},
        {"policy": "Target-grade", **summary_risk},
    ])
    summary_df.to_csv(common.DATA_DIR / "risk_comparison.csv", index=False)

    print("\nThreshold sweep: how many states change action, by target grade")
    sweep_df = threshold_sweep(policy_ev, targets=[50, 55, 60, 65, 70, 75, 80, 85, 90])
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv(common.DATA_DIR / "risk_threshold_sweep.csv", index=False)

    plot_score_distributions(scores_ev, scores_risk, common.RISK_TARGET_GRADE,
                              common.IMAGES_DIR / "09_risk_score_distributions.png")
    plot_policy_diff_heatmap(policy_ev, policy_risk, f=0, s=4,
                              path=common.IMAGES_DIR / "10_risk_policy_diff_fresh_full_sessions.png",
                              title="Where the two policies disagree (fresh, 4 tutoring sessions left)")
    plot_policy_diff_heatmap(policy_ev, policy_risk, f=1, s=2,
                              path=common.IMAGES_DIR / "11_risk_policy_diff_tired_some_sessions.png",
                              title="Where the two policies disagree (tired, 2 tutoring sessions left)")

    print("\nSaved risk_comparison.csv, risk_threshold_sweep.csv to data/")
    print("Saved 3 charts to images/")
