#!/usr/bin/env bash
# Scope gate (26Q3-REPO guardrail 2): non-test, non-vendored Python source
# must stay within LOC_BUDGET. vendored/ is exempt — the tamper gate already
# pins it byte-for-byte. Raising the budget means editing LOC_BUDGET in the
# same diff: visible and reviewable, never silent.
set -euo pipefail
cd "$(dirname "$0")/.."
budget=$(tr -d '[:space:]' < LOC_BUDGET)
actual=$(find flaime_serving -name '*.py' -not -path '*/vendored/*' -print0 | xargs -0 --no-run-if-empty cat | wc -l)
if [ "$actual" -gt "$budget" ]; then
  echo "LoC budget exceeded: $actual / $budget non-test, non-vendored source lines (see LOC_BUDGET)"
  exit 1
fi
echo "LoC budget OK: $actual / $budget"
