"""Vendored from FLAIME: main-process detection (extracted stub).

Upstream: flaime/utils/distributed.py — get_rank_and_world_size and
is_main_process only; the upstream module's DDP setup/teardown (and its torch
imports) are training-only and not needed for inference. Status ADAPTED
(extraction), recorded in VENDORED_FROM.json.
Frozen — do not edit (see vendoring rules in README).
"""

import os


def get_rank_and_world_size() -> tuple[int, int]:
    """Auto-detect rank and world size from environment variables.

    Checks for torchrun environment variables (LOCAL_RANK, WORLD_SIZE) first,
    then falls back to SLURM variables (SLURM_LOCALID, SLURM_NTASKS).

    Returns:
        Tuple of (rank, world_size). Returns (0, 1) if not in distributed mode.
    """
    # Try torchrun environment variables first
    rank = os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")

    if rank is not None and world_size is not None:
        return int(rank), int(world_size)

    # Fall back to SLURM environment variables
    rank = os.environ.get("SLURM_LOCALID")
    world_size = os.environ.get("SLURM_NTASKS")

    if rank is not None and world_size is not None:
        return int(rank), int(world_size)

    # Default: not distributed
    return 0, 1


def is_main_process(rank: int | None = None) -> bool:
    """Check if current process is the main process (rank 0).

    Args:
        rank: The rank to check. If None, auto-detects from environment.

    Returns:
        True if rank is 0, False otherwise.
    """
    if rank is None:
        rank, _ = get_rank_and_world_size()
    return rank == 0
