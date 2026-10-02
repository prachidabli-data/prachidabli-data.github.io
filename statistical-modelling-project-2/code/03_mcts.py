"""
Stage 3: Monte Carlo tree search (MCTS, using UCT/UCB1 selection) for
Statistical Modelling Project II.

MCTS is only ever given the MDP's transition function (mdp.transitions,
mdp.available_actions) as a simulator. It never reads the exact value
table or policy solved in code/02_mdp.py; those are held back purely so
this module can be graded against a known correct answer.

Stochastic transitions are handled by keying each action's children on the
specific next state sampled, rather than a single deterministic child per
action. Repeated simulations that sample the same action from the same
node naturally populate different children in proportion to how often the
environment actually produces each outcome, which is what folds the
environment's "chance" step into an otherwise standard tree, without a
separate chance-node type.
"""

from pathlib import Path
import importlib
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent))
import common

mdp = importlib.import_module("02_mdp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Exploration constant for UCB1. ASSUMPTION: classic UCB1 is derived for
# rewards in [0, 1] with c = sqrt(2); this problem's terminal rewards span
# roughly 45 to 95, so c cannot be reused at its textbook value of about
# 1.4, which would barely explore at all against Q-values fifty times
# that size. Rather than rescale c linearly by the reward range (which
# would suggest something like c = 70 and, tested directly, over-explores
# badly at this branching factor), a small empirical sweep at the maximum
# budget tested (20,000) compared root-value estimates across c = 1, 2, 5,
# 10, 20 and found c = 5 gave the best (highest) root value, so that is
# what is used throughout. See DEBUG_LOG.md for the sweep.
UCB_C = 5.0


class Node:
    __slots__ = ("t", "state", "N", "W", "children", "action_N", "action_W")

    def __init__(self, t: int, state: tuple):
        self.t = t
        self.state = state
        self.N = 0
        self.W = 0.0
        self.children = {}   # action -> {next_state: Node}
        self.action_N = {}   # action -> visit count
        self.action_W = {}   # action -> total backpropagated return


def terminal_reward(k: int, k_max: int) -> float:
    """
    Terminal score for a given knowledge level and knowledge cap. For
    Stage 2's own problem size (k_max = mdp.K_MAX), this is exactly
    common.terminal_expected_score. For Stage 3's scale-up experiment
    (a larger k_max), a and b are recalibrated so the score still spans
    45 (k = 0) to 95 (k = k_max), matching the brief's calibration rule
    at whatever knowledge scale is being tested.
    """
    if k_max == mdp.K_MAX:
        return common.terminal_expected_score(k)
    a, b = 45.0, 50.0 / k_max
    return float(np.clip(a + b * k, 0.0, 100.0))


def sample_next_state(state: tuple, action: str, rng: np.random.Generator, k_max: int) -> tuple:
    outcomes = mdp.transitions(state, action, k_max)
    next_states, probs = zip(*outcomes)
    idx = rng.choice(len(next_states), p=probs)
    return next_states[idx]


def rollout(node: "Node", rng: np.random.Generator, t_max: int, k_max: int) -> float:
    """Uniformly random valid actions from node to the end of term."""
    t = node.t
    k, f, s = node.state
    while t < t_max:
        actions = mdp.available_actions(s)
        action = actions[rng.integers(len(actions))]
        k, f, s = sample_next_state((k, f, s), action, rng, k_max)
        t += 1
    return terminal_reward(k, k_max)


def ucb1(node: "Node", action: str, c: float) -> float:
    q = node.action_W[action] / node.action_N[action]
    explore = c * np.sqrt(np.log(node.N) / node.action_N[action])
    return q + explore


def run_simulation(root: "Node", rng: np.random.Generator, c: float, t_max: int, k_max: int) -> None:
    node = root
    path = []
    while True:
        if node.t >= t_max:
            return_value = terminal_reward(node.state[0], k_max)
            break
        s = node.state[2]
        actions = mdp.available_actions(s)
        untried = [a for a in actions if node.action_N.get(a, 0) == 0]
        action = untried[rng.integers(len(untried))] if untried else max(
            actions, key=lambda a: ucb1(node, a, c)
        )
        next_state = sample_next_state(node.state, action, rng, k_max)
        path.append((node, action))
        bucket = node.children.setdefault(action, {})
        if next_state in bucket:
            node = bucket[next_state]
            continue
        new_node = Node(node.t + 1, next_state)
        bucket[next_state] = new_node
        return_value = rollout(new_node, rng, t_max, k_max)
        node = new_node
        break

    for n, a in path:
        n.N += 1
        n.W += return_value
        n.action_N[a] = n.action_N.get(a, 0) + 1
        n.action_W[a] = n.action_W.get(a, 0.0) + return_value
    node.N += 1
    node.W += return_value


def run_mcts(t: int, state: tuple, budget: int, rng: np.random.Generator,
             c: float = UCB_C, t_max: int = None, k_max: int = None) -> "Node":
    if t_max is None:
        t_max = mdp.T_MAX
    if k_max is None:
        k_max = mdp.K_MAX
    root = Node(t, state)
    for _ in range(budget):
        run_simulation(root, rng, c, t_max, k_max)
    return root


def root_value(root: "Node") -> float:
    return root.W / root.N if root.N > 0 else float("nan")


def action_q_values(root: "Node") -> dict:
    return {a: root.action_W[a] / n for a, n in root.action_N.items() if n > 0}


def recommended_action(root: "Node") -> str:
    q = action_q_values(root)
    return max(q, key=q.get)


def exact_q(V: dict, t: int, state: tuple, action: str, k_max: int = None) -> float:
    """One-step lookahead Q-value from the exact solution, used only to
    grade MCTS afterwards. Never used inside MCTS itself."""
    return sum(prob * V[t + 1][next_state] for next_state, prob in mdp.transitions(state, action, k_max))


# ---------------------------------------------------------------------------
# Experiment 1: convergence of the root value estimate with search budget
# ---------------------------------------------------------------------------
def convergence_experiment(start: tuple, budgets=(100, 500, 1000, 5000, 20000),
                            n_seeds: int = 20, c: float = UCB_C) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 * budget + seed)
            root = run_mcts(0, start, budget, rng, c=c)
            rows.append({"budget": budget, "seed": seed, "root_value": root_value(root)})
    return pd.DataFrame(rows)


def plot_convergence(df: pd.DataFrame, v_star: float, path):
    p = common.PALETTE
    summary = df.groupby("budget").root_value.agg(["mean", "sem"]).reset_index()
    summary["ci"] = 1.96 * summary["sem"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.errorbar(summary.budget, summary["mean"], yerr=summary["ci"], marker="o",
                color=p["amber_dark"], ecolor=p["amber"], capsize=4, linewidth=2, markersize=6, zorder=3)
    ax.axhline(v_star, color=p["navy"], linestyle="--", linewidth=1.5, zorder=2, label="Exact V*(start)")
    ax.set_xscale("log")
    ax.set_xlabel("Simulation budget (log scale)")
    ax.set_ylabel("Root value estimate")
    ax.set_title("MCTS improves with budget, but has not closed the gap by 20,000",
                  fontsize=13, fontweight="bold", color=p["navy"], loc="left", pad=12)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=p["line"], linewidth=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: policy agreement against the exact solution
# ---------------------------------------------------------------------------
def policy_agreement_experiment(V: dict, policy: dict, n_states: int = 200,
                                 budget: int = 2000, seed: int = 0, c: float = UCB_C) -> pd.DataFrame:
    master_rng = np.random.default_rng(seed)
    decision_states = [(t, k, f, s) for t in range(mdp.T_MAX) for (k, f, s) in mdp.all_states()]
    idx = master_rng.choice(len(decision_states), size=n_states, replace=False)
    sampled = [decision_states[i] for i in idx]

    rows = []
    for i, (t, k, f, s) in enumerate(sampled):
        rng = np.random.default_rng(seed * 1_000_003 + i)
        root = run_mcts(t, (k, f, s), budget, rng, c=c)
        mcts_action = recommended_action(root)
        exact_action = policy[t][(k, f, s)]
        agree = mcts_action == exact_action
        value_lost = 0.0 if agree else (
            exact_q(V, t, (k, f, s), exact_action) - exact_q(V, t, (k, f, s), mcts_action)
        )
        rows.append({"t": t, "k": k, "f": f, "s": s, "exact_action": exact_action,
                      "mcts_action": mcts_action, "agree": agree, "value_lost": value_lost})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 3: online performance, replanning with MCTS every week
# ---------------------------------------------------------------------------
def simulate_mcts_online(start: tuple, n_trajectories: int, budget: int,
                          seed: int = 0, c: float = UCB_C) -> np.ndarray:
    master_rng = np.random.default_rng(seed)
    final_scores = np.empty(n_trajectories)
    for i in range(n_trajectories):
        k, f, s = start
        for t in range(mdp.T_MAX):
            plan_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
            root = run_mcts(t, (k, f, s), budget, plan_rng, c=c)
            action = recommended_action(root)
            outcomes = mdp.transitions((k, f, s), action)
            next_states, probs = zip(*outcomes)
            step_idx = master_rng.choice(len(next_states), p=probs)
            k, f, s = next_states[step_idx]
        final_scores[i] = common.terminal_expected_score(k)
    return final_scores


# ---------------------------------------------------------------------------
# Experiment 4: scale-up, exact DP versus MCTS at a larger problem size
# ---------------------------------------------------------------------------
def scale_up_experiment(scaled_t_max: int = 24, scaled_k_max: int = 30,
                         mcts_budget: int = 5000, seed: int = 0) -> dict:
    original_start = (common.MDP_START_STATE["k"], common.MDP_START_STATE["f"], common.MDP_START_STATE["s"])
    scaled_start = (round(common.MDP_START_STATE["k"] * scaled_k_max / mdp.K_MAX),
                    common.MDP_START_STATE["f"], common.MDP_START_STATE["s"])

    _, _, original_dp_time = mdp.solve()
    _, _, scaled_dp_time = mdp.solve(t_max=scaled_t_max, k_max=scaled_k_max)

    original_states = (mdp.T_MAX + 1) * len(list(mdp.all_states()))
    scaled_states = (scaled_t_max + 1) * len(list(mdp.all_states(scaled_k_max)))

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    run_mcts(0, original_start, mcts_budget, rng)
    original_mcts_time = time.perf_counter() - t0

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    run_mcts(0, scaled_start, mcts_budget, rng, t_max=scaled_t_max, k_max=scaled_k_max)
    scaled_mcts_time = time.perf_counter() - t0

    return {
        "original_states": original_states,
        "scaled_states": scaled_states,
        "original_dp_time": original_dp_time,
        "scaled_dp_time": scaled_dp_time,
        "original_mcts_time": original_mcts_time,
        "scaled_mcts_time": scaled_mcts_time,
        "mcts_budget": mcts_budget,
    }


def plot_scale_up(result: dict, path):
    p = common.PALETTE
    labels = [f"Original\n({result['original_states']:,} state-values)",
              f"Scaled up\n({result['scaled_states']:,} state-values)"]
    dp_times = [result["original_dp_time"], result["scaled_dp_time"]]
    mcts_times = [result["original_mcts_time"], result["scaled_mcts_time"]]

    x = np.arange(2)
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, dp_times, width, label="Exact DP", color=p["navy"], zorder=3)
    ax.bar(x + width / 2, mcts_times, width, label=f"MCTS (budget {result['mcts_budget']:,})",
           color=p["amber"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Solve time, seconds")
    ax.set_title("Exact DP stays fast at this scale; MCTS cost tracks budget, not state count",
                  fontsize=12.5, fontweight="bold", color=p["navy"], loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=p["line"], linewidth=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    common.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    start = (common.MDP_START_STATE["k"], common.MDP_START_STATE["f"], common.MDP_START_STATE["s"])
    V, policy, _ = mdp.solve()
    v_star = V[0][start]

    print("=== Convergence experiment ===")
    conv_df = convergence_experiment(start)
    conv_df.to_csv(common.DATA_DIR / "mcts_convergence.csv", index=False)
    plot_convergence(conv_df, v_star, common.IMAGES_DIR / "07_mcts_convergence.png")
    print(conv_df.groupby("budget").root_value.mean())
    print(f"V*(start) = {v_star:.4f}")

    print("\n=== Policy agreement experiment ===")
    agree_df = policy_agreement_experiment(V, policy)
    agree_df.to_csv(common.DATA_DIR / "mcts_policy_agreement.csv", index=False)
    agreement_rate = agree_df.agree.mean()
    print(f"Agreement rate: {agreement_rate:.1%} ({agree_df.agree.sum()} of {len(agree_df)})")
    disagreements = agree_df[~agree_df.agree]
    if len(disagreements):
        print(f"Mean value lost on disagreement: {disagreements.value_lost.mean():.4f}")
        print(f"Max value lost on disagreement: {disagreements.value_lost.max():.4f}")

    print("\n=== Online performance experiment ===")
    n_traj = 100
    online_budget = 1000
    mcts_scores = simulate_mcts_online(start, n_traj, online_budget)
    optimal_scores = mdp.simulate_policy(mdp.optimal_policy_fn(policy), n_traj,
                                          np.random.default_rng(0), start)
    print(f"MCTS online replanning (budget {online_budget}, {n_traj} terms): mean {mcts_scores.mean():.4f}")
    print(f"Exact optimal policy ({n_traj} terms):                          mean {optimal_scores.mean():.4f}")

    print("\n=== Scale-up experiment ===")
    scale_result = scale_up_experiment()
    for key, value in scale_result.items():
        print(f"  {key}: {value}")
    plot_scale_up(scale_result, common.IMAGES_DIR / "08_mcts_scale_up.png")

    print("\nSaved mcts_convergence.csv, mcts_policy_agreement.csv to data/")
    print("Saved 07_mcts_convergence.png, 08_mcts_scale_up.png to images/")
