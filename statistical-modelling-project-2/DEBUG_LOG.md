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
