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

## CLI

```
uv run flaime-serve transcribe audio.wav --checkpoint /path/to/ckpt --model-type xeus
uv run flaime-serve ui --checkpoint /path/to/ckpt --model-type xeus
```

`ui` shells out to `apps/demo/app.py`, which ships from `flaime-demo` (26Q3-REPO-08+),
not this repo — it errors with a clear message if that path isn't present.

## Router smoke test

End-to-end routing + real transcription against `facebook/wav2vec2-base-960h`
(~380 MB download, network required):

```
uv run python scripts/smoke_router.py
uv run python scripts/smoke_router.py --device cuda   # if GPU available
```

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
  model families the demo doesn't route, no config framework, no speculative
  abstraction layers.
- **Offline by default.** `ASRInferenceEngine.load()` refuses a checkpoint value
  that would resolve from the HuggingFace Hub — relative, absent from disk,
  exactly one `/` — and raises `RemoteCheckpointRefused`. Hub IDs remain
  supported for callers that want them (`allow_remote=True`, `EnginePool(
  allow_remote=True)`, or `--allow-remote` on the CLI); they are simply never
  reached by accident. This list previously claimed "no HF Hub checkpoint
  downloads" while the loader implemented one silently.

## Licensing

The **code** in this repository is Apache-2.0 — see `LICENSE` and `NOTICE`.

**No model weights are hosted, bundled, or distributed here.** This repository
ships software that runs a model *you* supply: you train it or obtain it
yourself, and how you use it is between you and whoever licensed it to you.
Nothing here grants or restricts rights to any checkpoint.

As a courtesy, the foundation models this code can load carry their own terms —
verified against the upstream model cards on 2026-07-31:

| Foundation model | License |
|---|---|
| `facebook/wav2vec2-base`, `-base-960h` | Apache-2.0 |
| `espnet/xeus` | CC-BY-NC-SA-4.0 |
| `facebook/mms-1b-all` | CC-BY-NC-4.0 |

Two of the three are non-commercial, and XEUS's ShareAlike term extends to
checkpoints fine-tuned from it. Apache-2.0 on this code does not alter those
terms in either direction: it grants you the software, not the weights.

The upstream model card governs; this table is a pointer, not legal advice.
