# flaime-serving

Standalone audio-in → text-out ASR inference layer for FLAIME community demos
(XEUS + wav2vec2). Public-PyPI dependencies only — no FLAIME or PhoNet repo access
needed to install or run. Model code is vendored as frozen, hash-pinned copies from
FLAIME.

Private under IndigiGenius, consistent with community data governance. Epic:
[FLAIME#608](https://github.com/IndigiGenius/FLAIME/issues/608); architecture,
task map, and full guardrails in FLAIME's
`docs/planning/tasks/26Q3-REPO/26Q3-REPO-00-overview.md`.

## Setup

```
uv sync
uv run pytest
```

Checkpoints stay external: `CHECKPOINTS_DIR` + `DEMO_CHECKPOINT_FILE` (or
`DEMO_LANGUAGES_CONFIG` routing YAML) env vars, checkpoint dir mounted read-only,
weights never committed.

## Frozen public API (guardrail)

`flaime_serving/__init__.py` exports exactly eight names — they land incrementally
during the extraction (REPO-03/04/21) and then the API is closed:

`ASRInferenceEngine`, `ASRModelFactory`, `EnginePool`, `LanguageNotSupportedError`,
`LanguageRouter`, `RouteResult`, `TranscriptionResult`, `beam_search_ctc_decode`

A new export is a scope change requiring maintainer sign-off.

## Vendoring rules (binding)

1. **Frozen, never forked.** Files under `flaime_serving/vendored/` are copies of
   FLAIME files at a recorded commit. Never edited after landing — no fixes, no
   improvements. Only permitted delta: import-line rewrites (`flaime.*` →
   `flaime_serving.vendored.*`).
2. **Provenance manifest.** `VENDORED_FROM.json` records per file: upstream path,
   FLAIME commit SHA, sha256, and status `VERBATIM` or `ADAPTED` (recorded deltas).
3. **Tamper gate.** `scripts/check_vendored.sh` (CI + pre-commit) recomputes every
   sha256 and fails on any mismatch or unlisted vendored file. Editing a vendored
   file therefore requires touching the manifest in the same diff.
4. **Re-vendoring is a deliberate task**, only when the demo adopts a new checkpoint
   era: re-copy, re-record SHAs, run the full suite. Loader code and checkpoints
   move together or not at all.

## Scope guardrails

- **Move-only / copy-only.** Extraction PRs contain file moves or vendored copies
  plus the minimum import/path rewrites to make them work. Zero behavior changes;
  "while I'm here" improvements are rejected in review and filed as issues.
- **LoC budget.** `LOC_BUDGET` (1,500) caps non-test, non-vendored Python source;
  `scripts/check_loc_budget.sh` enforces it in CI + pre-commit. `vendored/` is
  exempt — the tamper gate already pins it byte-for-byte.
- **Named non-goals.** No FastAPI/server mode, no streaming ASR, no vendoring of
  model families the demo doesn't route, no config framework, no HF Hub checkpoint
  downloads, no speculative abstraction layers.
