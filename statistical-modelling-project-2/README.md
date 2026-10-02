# Statistical Modelling Project II: Planning a Term Under Uncertainty

A follow-on from Statistical Modelling Project I. Project I asked what predicts a
student's exam score. This project asks a different question: given that model,
how should a student actually spend a limited term to get the best result?

Four stages, in increasing order of realism:

1. **Linear programming baseline** (`code/01_lp.py`): a deterministic weekly hour
   allocation, no uncertainty, solved with scipy's HiGHS solver.
2. **Exact Markov decision process** (`code/02_mdp.py`): the same decision,
   spread across a 12 week term with uncertainty in outcomes, solved exactly by
   backward induction (dynamic programming).
3. **Monte Carlo tree search** (`code/03_mcts.py`): an approximate, simulation
   based planner (UCT/UCB1), validated against Stage 2's exact answer rather
   than assumed to be correct.
4. **Risk sensitive extension** (`code/04_risk_extension.py`): the same exact
   method as Stage 2, but maximising the probability of reaching a target grade
   instead of the average score.

## How to run

From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m ipykernel install --user --name statistical-modelling-project-2
jupyter notebook notebook/decision_modelling.ipynb
```

Inside the notebook, use Kernel then Restart and Run All. Every number, table
and chart in the notebook and in the case study page is generated this way,
from a fresh kernel, with a fixed random seed (42) throughout, so a full rerun
should reproduce the same results exactly.

The four stages can also be run as standalone scripts in order, from the same
virtual environment:

```bash
python code/01_lp.py
python code/02_mdp.py
python code/03_mcts.py
python code/04_risk_extension.py
```

Each writes its own CSVs to `data/` and charts to `images/`.

## Folder guide

```
code/common.py              Single source of truth for every scenario parameter
code/01_lp.py                Stage 1: linear programming baseline
code/02_mdp.py                Stage 2: exact Markov decision process
code/03_mcts.py               Stage 3: Monte Carlo tree search
code/04_risk_extension.py     Stage 4: risk sensitive extension
notebook/decision_modelling.ipynb   All four stages, narrated, with charts
data/                         CSV outputs (allocations, policies, comparisons)
images/                       PNG charts, also used on the case study page
DEBUG_LOG.md                  Every genuine bug or finding, logged as it happened
```

## Parameters: from Project I versus assumed for this project

Every constant in `code/common.py` is tagged as one of the two categories
below. This project only ever uses real numbers where Project I actually
measured something; everywhere else, a value had to be assumed to turn a
regression into a planning scenario, and that is stated plainly rather than
left implicit.

| Parameter | Value | Source |
|---|---|---|
| Mean weekly study hours | 11.9094 | FROM PROJECT I (dataset mean) |
| Study hours coefficient | 1.13 pts/hr | FROM PROJECT I (fitted) |
| Attendance coefficient | 0.28 pts/pp | FROM PROJECT I (fitted) |
| Group study dummy | 1.83 pts/wk | FROM PROJECT I (fitted) |
| Tutoring dummy | 5.56 pts/wk | FROM PROJECT I (fitted) |
| Exam score noise (SD) | 7.50 | FROM PROJECT I (true generating value) |
| True sleep penalty | -1.80 pts/hr below 6 | FROM PROJECT I (true, missed by the regression) |
| Weekly hours budget | 40 | ASSUMPTION |
| Tutoring hours cap (LP) | 3 | ASSUMPTION |
| Group study hours cap (LP) | 6 | ASSUMPTION |
| Term length (MDP) | 12 weeks | ASSUMPTION |
| Knowledge levels (MDP) | 0 to 10 | ASSUMPTION |
| Tutoring sessions available (MDP) | 4 | ASSUMPTION |
| P(self-study succeeds) | 0.50 | ASSUMPTION |
| P(group study succeeds) | 0.57 | ASSUMPTION, ratio anchored to Project I's coefficients |
| Fatigue success multiplier | 1.00 / 0.75 / 0.40 | ASSUMPTION |
| Terminal reward (knowledge to score) | 45 + 5k | ASSUMPTION, calibrated to Project I's score range |
| MCTS exploration constant (c) | 5 | Tuned empirically against the exact answer |
| Risk sensitive target grade | 70 | ASSUMPTION (example: a scholarship threshold) |

## Honesty discipline

This project follows the same rule as Project I: results are reported as they
came out, not adjusted to look more expected. Three findings illustrate this.

Self-study never appears as the optimal action anywhere in the 2,145 state
MDP policy, because group study has no weekly capacity cap at this stage
(unlike the LP's 6 hour cap) and a higher success probability, so it
dominates outright. MCTS disagrees with the exact optimal policy roughly
23.5% of the time, including at the very first decision of the term. And the
risk-sensitive and expected-value policies genuinely differ at every target
grade tested, from 50 to 90. None of these were tuned away; all three are
documented in `DEBUG_LOG.md` and explained on the case study page.

## Talking points

Three things worth raising in an interview about this build:

1. **A real bug, caught by comparing two numbers that should have agreed.**
   MCTS's root value printed as exactly 0.0000 while its own action Q-values
   looked sensible (73 to 76). The backpropagation loop updated each node's
   visit counts and per-action totals, but never the node's own cumulative
   return. The fix was a single added line, but the bug would have been easy
   to miss without checking a number against the components it was built
   from.
2. **Choosing to validate rather than assume MCTS is correct.** Rather than
   trusting an approximate planner on faith, this project solved the same
   problem exactly first (Stage 2), then measured MCTS against that exact
   answer: convergence as the simulation budget grows, agreement rate with
   the optimal action, and the real cost in value when they disagree. That
   is a more defensible way to present an approximate method than citing its
   own internal numbers alone.
3. **Reporting a finding instead of hiding it.** Self-study never being
   optimal looked, at first glance, like a bug. Tracing it back to a real
   modelling asymmetry between Stage 1 and Stage 2 (a capacity cap present
   in one and absent in the other) turned it from a suspicious result into an
   honest, explained property of the model, which is a more useful outcome
   than quietly patching it to look more balanced.

## Limitations

Every number in this project is either drawn from Project I's synthetic,
reconstructed dataset or from assumptions made to turn that dataset into a
planning scenario; none of it describes a real student. Every stage except
Stage 4 optimises the average outcome, not consistency or downside risk.
Project I's own regression missed the true sleep effect entirely, and that
miss is inherited here rather than corrected. Natural next steps: a capacity
limit on group study in the MDP, a heuristic (rather than uniformly random)
rollout policy for MCTS, and a more general risk measure such as CVaR.
