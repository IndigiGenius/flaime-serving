"""flaime-serving: standalone audio-in to text-out ASR inference layer.

The public API is frozen at exactly eight names (epic guardrail 3, see README).
This is the final freeze, audited in REPO-04: no new exports may be added to
this module without maintainer sign-off.
"""

from flaime_serving.ctc_beam_search import beam_search_ctc_decode
from flaime_serving.inference import ASRInferenceEngine, TranscriptionResult
from flaime_serving.router import (
    EnginePool,
    LanguageNotSupportedError,
    LanguageRouter,
    RouteResult,
)
from flaime_serving.vendored.model_factory import ASRModelFactory

__all__: list[str] = [
    "ASRInferenceEngine",
    "ASRModelFactory",
    "EnginePool",
    "LanguageNotSupportedError",
    "LanguageRouter",
    "RouteResult",
    "TranscriptionResult",
    "beam_search_ctc_decode",
]
