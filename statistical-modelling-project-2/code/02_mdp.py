"""
Stage 2: exact Markov decision process for Statistical Modelling Project II.

State: (t, k, f, s)
    t   week index, 0 to 12 (t = 12 is the terminal, end-of-term state)
    k   knowledge, 0 to 10, capped
    f   fatigue, 0 (fresh), 1 (tired), 2 (exhausted)
    s   tutoring sessions remaining, 0 to 4

Actions: self_study, group_study, tutoring (only if s > 0), rest.

This module solves the MDP exactly by backward induction (dynamic
programming), which is feasible here only because the state space is
small (165 states per week, 13 weeks). Stage 3 checks whether Monte Carlo
tree search, which never sees this table, can recover the same answer.
"""

from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent))
import common

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

K_MAX = common.MDP_KNOWLEDGE_MAX
F_MAX = common.MDP_FATIGUE_LEVELS - 1
S_MAX = common.MDP_TUTORING_SESSIONS_MAX
T_MAX = common.MDP_TERM_WEEKS

ACTIONS_IN_TIE_BREAK_ORDER = ["rest", "self_study", "group_study", "tutoring"]


def available_actions(s: int) -> list:
    acts = ["self_study", "group_study", "rest"]
    if s > 0:
        acts.append("tutoring")
    return acts


def _fatigue_dist(f: int) -> list:
    """
    Outcomes of studying on fatigue: +1 with probability
    MDP_P_FATIGUE_INCREASE, capped at F_MAX, else unchanged. Modelled as
    independent of the knowledge-gain outcome (an assumption: fatigue and
    learning are two separate coin flips each week, not correlated).
    """
    f_up = min(f + 1, F_MAX)
    p_up = common.MDP_P_FATIGUE_INCREASE
    if f_up == f:
        return [(f, 1.0)]
    return [(f_up, p_up), (f, 1.0 - p_up)]


def _study_knowledge_dist(base_p: float, f: int) -> list:
    """Self-study or group-study knowledge outcome, fatigue-scaled."""
    p_eff = base_p * common.MDP_FATIGUE_FACTOR[f]
    return [(1, p_eff), (0, 1.0 - p_eff)]


def _tutoring_knowledge_dist(f: int) -> list:
    """
    Tutoring knowledge outcome, fatigue-scaled. Both the P(+1) and P(+2)
    success probabilities are scaled by the same fatigue factor (an
    assumption: fatigue reduces both equally, rather than only the larger
    gain), so the ratio between them is preserved under fatigue.
    """
    factor = common.MDP_FATIGUE_FACTOR[f]
    p1 = common.MDP_P_TUTORING_PLUS1 * factor
    p2 = common.MDP_P_TUTORING_PLUS2 * factor
    return [(0, 1.0 - p1 - p2), (1, p1), (2, p2)]


def transitions(state: tuple, action: str, k_max: int = None) -> list:
    """
    All (next_state, probability) pairs for a state and action. next_state
    is (k, f, s); the caller advances t separately.

    k_max overrides the knowledge cap (used only by Stage 3's scale-up
    experiment, which solves a larger problem than Stage 2's; it defaults
    to Stage 2's own K_MAX everywhere else).
    """
    if k_max is None:
        k_max = K_MAX
    k, f, s = state
    if action == "rest":
        f2 = max(f - 1, 0)
        return [((k, f2, s), 1.0)]

    if action == "tutoring":
        knowledge_dist = _tutoring_knowledge_dist(f)
        s2 = s - 1
    elif action == "self_study":
        knowledge_dist = _study_knowledge_dist(common.MDP_P_SELF_STUDY, f)
        s2 = s
    elif action == "group_study":
        knowledge_dist = _study_knowledge_dist(common.MDP_P_GROUP_STUDY, f)
        s2 = s
    else:
        raise ValueError(f"Unknown action: {action}")

    fatigue_dist = _fatigue_dist(f)
    outcomes = {}
    for dk, p_k in knowledge_dist:
        k2 = min(k + dk, k_max)
        for f2, p_f in fatigue_dist:
            key = (k2, f2, s2)
            outcomes[key] = outcomes.get(key, 0.0) + p_k * p_f
    return list(outcomes.items())


def all_states(k_max: int = None):
    if k_max is None:
        k_max = K_MAX
    for k in range(k_max + 1):
        for f in range(F_MAX + 1):
            for s in range(S_MAX + 1):
                yield (k, f, s)


def solve(t_max: int = None, k_max: int = None, terminal_fn=None):
    """
    Backward induction. Returns (V, policy, solve_seconds), where V[t] and
    policy[t] are dicts keyed by (k, f, s).

    t_max and k_max override the term length and knowledge cap (used only
    by Stage 3's scale-up experiment); both default to Stage 2's own
    problem size everywhere else, so this is unchanged from Phase 3.

    terminal_fn(k) overrides the terminal reward (used only by Stage 4's
    risk-sensitive extension, which maximises a probability rather than
    an expected score); it defaults to common.terminal_expected_score,
    so this is also unchanged from Phase 3.
    """
    if t_max is None:
        t_max = T_MAX
    if k_max is None:
        k_max = K_MAX
    if terminal_fn is None:
        terminal_fn = common.terminal_expected_score

    start_time = time.perf_counter()
    states = list(all_states(k_max))

    V = {t_max: {state: terminal_fn(state[0]) for state in states}}
    policy = {}

    for t in range(t_max - 1, -1, -1):
        V[t] = {}
        policy[t] = {}
        for state in states:
            k, f, s = state
            best_action = None
            best_value = -np.inf
            for action in ACTIONS_IN_TIE_BREAK_ORDER:
                if action == "tutoring" and s == 0:
                    continue
                value = sum(
                    prob * V[t + 1][next_state]
                    for next_state, prob in transitions((k, f, s), action, k_max)
                )
                if value > best_value:
                    best_value = value
                    best_action = action
            V[t][state] = best_value
            policy[t][state] = best_action

    solve_seconds = time.perf_counter() - start_time
    return V, policy, solve_seconds


def simulate_policy(policy_fn, n_sims: int, rng: np.random.Generator, start_state=None):
    """
    Simulate n_sims independent terms under a policy function
    policy_fn(t, k, f, s) -> action. Returns an array of final terminal
    scores (common.terminal_expected_score applied to final knowledge).
    """
    if start_state is None:
        start_state = (common.MDP_START_STATE["k"], common.MDP_START_STATE["f"], common.MDP_START_STATE["s"])

    final_scores = np.empty(n_sims)
    for i in range(n_sims):
        k, f, s = start_state
        for t in range(T_MAX):
            action = policy_fn(t, k, f, s)
            outcomes = transitions((k, f, s), action)
            next_states, probs = zip(*outcomes)
            idx = rng.choice(len(next_states), p=probs)
            k, f, s = next_states[idx]
        final_scores[i] = common.terminal_expected_score(k)
    return final_scores


def optimal_policy_fn(policy):
    return lambda t, k, f, s: policy[t][(k, f, s)]


def always_self_study_fn():
    return lambda t, k, f, s: "self_study"


def always_tutoring_fn():
    return lambda t, k, f, s: "tutoring" if s > 0 else "self_study"


def alternate_study_rest_fn():
    return lambda t, k, f, s: "self_study" if t % 2 == 0 else "rest"


def policy_to_dataframe(policy, V):
    rows = []
    for t in sorted(policy.keys()):
        for (k, f, s), action in policy[t].items():
            rows.append({"t": t, "k": k, "f": f, "s": s, "action": action, "value": V[t][(k, f, s)]})
    return pd.DataFrame(rows)


ACTION_COLOURS = {
    "rest": common.PALETTE["muted"],
    "self_study": common.PALETTE["navy_light"],
    "group_study": common.PALETTE["amber"],
    "tutoring": common.PALETTE["amber_dark"],
}
ACTION_ORDER = ["rest", "self_study", "group_study", "tutoring"]


def plot_policy_heatmap(policy, f: int, s: int, path, title: str):
    grid = np.zeros((K_MAX + 1, T_MAX))
    for t in range(T_MAX):
        for k in range(K_MAX + 1):
            action = policy[t][(k, f, s)]
            grid[k, t] = ACTION_ORDER.index(action)

    cmap = ListedColormap([ACTION_COLOURS[a] for a in ACTION_ORDER])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower", vmin=-0.5, vmax=len(ACTION_ORDER) - 0.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("Knowledge")
    ax.set_xticks(range(T_MAX))
    ax.set_title(title, fontsize=13, fontweight="bold", color=common.PALETTE["navy"], loc="left", pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACTION_COLOURS[a]) for a in ACTION_ORDER]
    labels = [a.replace("_", " ") for a in ACTION_ORDER]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_policy_comparison(results: dict, path):
    p = common.PALETTE
    names = list(results.keys())
    means = [np.mean(results[n]) for n in names]
    colours = [p["amber"] if n == "Optimal (MDP)" else p["navy_light"] for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(names, means, color=colours, zorder=3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.5, f"{m:.1f}", ha="center",
                fontsize=11, fontweight="bold", color=p["navy"])
    ax.set_ylabel("Mean simulated final score")
    ax.set_title("The optimal policy beats every simple fixed policy",
                  fontsize=14, fontweight="bold", color=p["navy"], loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=p["line"], linewidth=0.8, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    V, policy, solve_seconds = solve()
    n_states = len(list(all_states())) * (T_MAX + 1)
    print(f"State-space size: {n_states} state-values ({len(list(all_states()))} states per week, {T_MAX + 1} weeks)")
    print(f"Solve time: {solve_seconds:.4f} seconds")

    start = (common.MDP_START_STATE["k"], common.MDP_START_STATE["f"], common.MDP_START_STATE["s"])
    v_start = V[0][start]
    print(f"V*(start) = {v_start:.4f}")

    # Check (a): V at t = T_MAX equals the terminal reward
    check_a = all(
        abs(V[T_MAX][state] - common.terminal_expected_score(state[0])) < 1e-9 for state in all_states()
    )
    print(f"Check (a), V at t=12 equals terminal reward: {check_a}")

    rng = np.random.default_rng(common.SEED)
    n_sims = 10_000

    scores_optimal = simulate_policy(optimal_policy_fn(policy), n_sims, rng, start)
    mean_optimal = scores_optimal.mean()
    se_optimal = scores_optimal.std(ddof=1) / np.sqrt(n_sims)
    ci_low, ci_high = mean_optimal - 1.96 * se_optimal, mean_optimal + 1.96 * se_optimal
    print(f"\nCheck (b): simulated mean score under optimal policy: {mean_optimal:.4f}")
    print(f"95 percent CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"V*(start) {v_start:.4f} inside CI: {ci_low <= v_start <= ci_high}")

    fixed_policies = {
        "Always self-study": always_self_study_fn(),
        "Always tutoring": always_tutoring_fn(),
        "Alternate study/rest": alternate_study_rest_fn(),
    }
    results = {"Optimal (MDP)": scores_optimal}
    for name, fn in fixed_policies.items():
        results[name] = simulate_policy(fn, n_sims, rng, start)

    print("\nCheck (c): optimal policy versus fixed policies")
    for name, scores in results.items():
        print(f"  {name}: mean {scores.mean():.4f}")
    beats_all = all(scores_optimal.mean() >= results[name].mean() for name in fixed_policies)
    print(f"Optimal beats all fixed policies: {beats_all}")

    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    common.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    policy_df = policy_to_dataframe(policy, V)
    policy_df.to_csv(common.DATA_DIR / "mdp_policy.csv", index=False)

    plot_policy_heatmap(policy, f=0, s=4, path=common.IMAGES_DIR / "03_mdp_policy_fresh_full_sessions.png",
                         title="Optimal action by week and knowledge (fresh, 4 tutoring sessions left)")
    plot_policy_heatmap(policy, f=1, s=2, path=common.IMAGES_DIR / "04_mdp_policy_tired_some_sessions.png",
                         title="Optimal action by week and knowledge (tired, 2 tutoring sessions left)")
    plot_policy_heatmap(policy, f=2, s=0, path=common.IMAGES_DIR / "05_mdp_policy_exhausted_no_sessions.png",
                         title="Optimal action by week and knowledge (exhausted, 0 tutoring sessions left)")
    plot_policy_comparison(results, common.IMAGES_DIR / "06_mdp_policy_comparison.png")

    print("\nSaved mdp_policy.csv to data/")
    print("Saved 3 policy heatmaps and 1 comparison chart to images/")
