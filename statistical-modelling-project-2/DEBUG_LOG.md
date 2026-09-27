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
