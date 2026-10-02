# Debug log

A running log of genuine bugs, wrong results, and validation failures found
during this build, and how each was fixed. Entries are added only when
something real goes wrong, in the order it happens.

## Phase 1

No issues found. The notebook ran cleanly from a fresh kernel on the first
attempt.

## Phase 2

Minor: the sensitivity chart's title was cut off at the right edge of the
figure on first render, because the title string was too long for the
8.5 inch figure width at the chosen font size. Caught by looking at the
rendered PNG rather than trusting the code. Fixed by shortening the title
text.

Everything else validated on the first attempt: the shadow price of the
weekly hours budget matched exactly when re-solving at 41 hours, and the
brute-force grid search matched the LP's decision-only score to within
floating-point precision (both 55.2725).

## Phase 3

Not a bug, but worth recording as a genuine, checked finding rather than
something quietly patched: self-study is never the optimal action anywhere
in the 2,145-state policy (confirmed by counting actions in the exported
policy table). The cause is a real modelling asymmetry between stages:
Stage 1's linear programme capped group study at 6 hours a week, but
Stage 2's weekly MDP action has no such capacity limit on group study, so
its higher success probability (0.57 against self-study's 0.50) makes it
weakly dominate self-study in every state with no offsetting cost. Left as
is and reported honestly in the notebook rather than adding a capacity
constraint after the fact to make self-study reappear.

All three validation checks passed on the first attempt: V at t=12 matched
the terminal reward exactly for all 2,145 states, the 10,000-run simulated
mean (82.01) fell inside the 95 percent CI around V*(start) (81.98), and
the optimal policy beat all three fixed policies (82.0 against 71.7, 75.1
and 70.1).

## Phase 4

Bug: the root's own value estimate printed as exactly 0.0000 on the first
run, while the four action Q-values it was built from all looked
sensible (73 to 76). Caught by comparing the two numbers, since a root
value of zero was obviously wrong given its own children's Q-values.
Diagnosis: the backpropagation loop updated `N`, `action_N` and
`action_W` for every node on the simulated path, but never updated that
node's own `W`, so every node's cumulative return stayed at zero except
the newly expanded leaf at the very end of the path (which was updated
separately, one line below the loop). Fixed by adding `n.W +=
return_value` inside the same loop.

Tuning, not a bug: classic UCB1 uses an exploration constant of
`sqrt(2)`, derived for rewards scaled to [0, 1]. This problem's terminal
rewards span roughly 45 to 95, so that constant explores far too little
in practice. A linear rescaling by the reward's range suggested
something like c = 70, but tested directly this over-explored badly
(the root value estimate at budget 20,000 was worse than smaller values
of c). A small sweep at budget 20,000 over c in {1, 2, 5, 10, 20} found
c = 5 gave the highest root value estimate, so that is what the rest of
this stage uses.

Minor bug: the scale-up experiment's reported state-space size for the
*original* problem was 1,980, not 2,145 as reported in Phase 3, because
it multiplied states-per-week by `T_MAX` (12) instead of `T_MAX + 1`
(13), missing the terminal week. Caught by comparing the two numbers
across phases rather than assuming they would already agree. Fixed by
adding the missing `+ 1`; the scaled problem's count was already correct.

Genuine, checked finding, not a bug: even at the very first decision
(the start state), MCTS's recommended action (tutoring) disagrees with
the exact optimal action (group study). Across 200 sampled states the
two agree 76.5 percent of the time, and the value lost on the 47
disagreements is small (mean 0.15 points, maximum 0.90), meaning MCTS's
mistakes are typically near-ties rather than clearly wrong choices.
Reported as found rather than treated as something to fix, since MCTS is
expected to be approximate.

## Phase 5

No bugs found. The generalised `solve()` (already parameterised for
Stage 3's scale-up test) accepted a custom terminal reward function with
no further changes, and re-running Phase 3's own solve afterward gave an
identical V*(start) (81.9763), confirming the new `terminal_fn` parameter
did not disturb the default path.

The threshold sweep and the two policies' simulated outcomes both
produced sensible, checkable numbers on the first run: the target-grade
policy trades a negligible amount of mean score (81.78 against 81.82)
for a better 5th percentile score (63.72 against 63.10) and a better
worst-decile mean (62.38 against 62.05), which is exactly the shape a
risk-sensitive objective is supposed to produce, so no further
adjustment was made.

## Phase 6

No bugs. This phase only read values already computed and validated in
Sections 2 to 5 (no new simulation or solving), so there was nothing new
to break. The one thing worth checking was whether the mid-term example
state (week 7, knowledge 3, fatigue exhausted, 1 tutoring session left)
would show a materially different action from the general "recover
first, cram at the end" pattern already seen in Section 3's heatmaps.
It did not: the policy rests at every week from 0 to 10 for this exact
state and only spends its last tutoring session at week 11, consistent
with, not contradicting, what was already found.
