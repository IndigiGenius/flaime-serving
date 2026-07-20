"""flaime-serving: standalone audio-in to text-out ASR inference layer.

The public API is frozen at exactly eight names (epic guardrail 3, see README);
they land incrementally in 26Q3-REPO-03/04/21 and the final freeze is audited
in REPO-04.
"""

from flaime_serving.vendored.model_factory import ASRModelFactory

__all__: list[str] = ["ASRModelFactory"]
