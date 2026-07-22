"""Vendored from FLAIME: sample-packing configuration (extracted stub).

Upstream: flaime/infrastructure/datasets/dynamic_batching.py — PackingConfig
only; the upstream module also carries the packing logic, which inference does
not need. Status ADAPTED (extraction), recorded in VENDORED_FROM.json.
Frozen — do not edit (see vendoring rules in README).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PackingConfig:
    """Configuration for sample packing.

    Attributes:
        max_duration_seconds: Maximum duration per packed sample (default: 30.0)
        max_samples_per_pack: Safety limit on samples per pack to bound memory
        sample_rate: Audio sample rate in Hz
        gap_seconds: Small gap between concatenated utterances for clarity
        enabled: Whether packing is enabled (allows easy toggling)
    """

    max_duration_seconds: float = 30.0
    max_samples_per_pack: int = 10
    sample_rate: int = 16000
    gap_seconds: float = 0.1
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_duration_seconds <= 0:
            raise ValueError(
                f"max_duration_seconds must be positive, got {self.max_duration_seconds}"
            )
        if self.max_samples_per_pack < 1:
            raise ValueError(
                f"max_samples_per_pack must be >= 1, got {self.max_samples_per_pack}"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.gap_seconds < 0:
            raise ValueError(f"gap_seconds must be >= 0, got {self.gap_seconds}")

    @property
    def max_samples(self) -> int:
        """Maximum number of audio samples (at sample_rate)."""
        return int(self.max_duration_seconds * self.sample_rate)

    @property
    def gap_samples(self) -> int:
        """Number of silence samples for inter-utterance gap."""
        return int(self.gap_seconds * self.sample_rate)
