"""Prefix CTC beam search decoding (pure acoustic, no LM).

Reference algorithm: Hannun 2014, "First-Pass Large Vocabulary Continuous Speech
Recognition using Bi-Directional Recurrent DNNs" (§3, prefix beam search).

Entirely log-space with torch.Tensor inputs; Python-level inner loop (small beam
widths, inference-only usage). LM/lexicon fusion is out of scope here.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

_NEG_INF = -float("inf")


def _logsumexp2(a: float, b: float) -> float:
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def beam_search_ctc_decode(
    log_probs: torch.Tensor,
    beam_width: int = 5,
    blank: int = 0,
) -> list[list[int]]:
    """Prefix CTC beam search decoder.

    Args:
        log_probs: Log-probabilities with shape ``(batch, time, vocab)`` or
            ``(time, vocab)`` (single-utterance convenience; still returns a
            batched list-of-lists of length 1).
        beam_width: Number of beams kept at each timestep (>= 1). ``beam_width=1``
            is short-circuited to frame-wise greedy argmax + blank/duplicate
            collapse, which is the documented equivalence.
        blank: Index of the CTC blank token in ``vocab``.

    Returns:
        One decoded token-id list per batch element. Blanks are removed and CTC
        duplicate-collapse has been applied (i.e. pure prefix semantics, not raw
        per-frame argmax).

    Raises:
        ValueError: On invalid ``beam_width``, empty time dimension, out-of-range
            ``blank``, or unsupported tensor rank.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    if log_probs.ndim == 2:
        log_probs = log_probs.unsqueeze(0)
    elif log_probs.ndim != 3:
        raise ValueError(
            "log_probs must have shape (batch, time, vocab) or (time, vocab); "
            f"got {tuple(log_probs.shape)}"
        )

    _, time_steps, vocab = log_probs.shape
    if time_steps == 0:
        raise ValueError(
            f"log_probs has empty time dimension: shape={tuple(log_probs.shape)}"
        )
    if not 0 <= blank < vocab:
        raise ValueError(f"blank index {blank} out of range [0, {vocab})")

    log_probs_cpu = log_probs.detach().cpu()

    if beam_width == 1:
        return _greedy_collapse(log_probs_cpu, blank)

    return [
        _prefix_beam_search(log_probs_cpu[i], beam_width, blank, vocab)
        for i in range(log_probs_cpu.shape[0])
    ]


def _greedy_collapse(log_probs: torch.Tensor, blank: int) -> list[list[int]]:
    predicted = torch.argmax(log_probs, dim=-1)
    results: list[list[int]] = []
    for b in range(predicted.shape[0]):
        seq: list[int] = []
        prev: int | None = None
        for tok in predicted[b].tolist():
            tok = int(tok)
            if tok == blank:
                prev = None
                continue
            if tok != prev:
                seq.append(tok)
                prev = tok
        results.append(seq)
    return results


def _prefix_beam_search(
    log_probs: torch.Tensor, beam_width: int, blank: int, vocab: int
) -> list[int]:
    time_steps = log_probs.shape[0]

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, _NEG_INF)}

    for t in range(time_steps):
        lp_t = log_probs[t].tolist()
        next_beams: dict[tuple[int, ...], list[float]] = defaultdict(
            lambda: [_NEG_INF, _NEG_INF]
        )

        for prefix, (pb, pnb) in beams.items():
            p_total = _logsumexp2(pb, pnb)
            last = prefix[-1] if prefix else None

            # 1. Extend with blank — prefix unchanged, flows into pb.
            entry = next_beams[prefix]
            entry[0] = _logsumexp2(entry[0], p_total + lp_t[blank])

            # 2. Extend with each non-blank token.
            for c in range(vocab):
                if c == blank:
                    continue
                lp_c = lp_t[c]

                if c == last:
                    # 2a. Same char, no blank separator — prefix unchanged,
                    #     contribution only from pnb (can't stack on pb path
                    #     without emitting a duplicate).
                    same_entry = next_beams[prefix]
                    same_entry[1] = _logsumexp2(same_entry[1], pnb + lp_c)
                    # 2b. Same char via blank separator — appends duplicate.
                    extended = prefix + (c,)
                    new_entry = next_beams[extended]
                    new_entry[1] = _logsumexp2(new_entry[1], pb + lp_c)
                else:
                    # 2c. Different char (or empty prefix) — appends c.
                    extended = prefix + (c,)
                    new_entry = next_beams[extended]
                    new_entry[1] = _logsumexp2(new_entry[1], p_total + lp_c)

        scored = sorted(
            (
                (prefix, _logsumexp2(probs[0], probs[1]))
                for prefix, probs in next_beams.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:beam_width]
        beams = {
            prefix: (next_beams[prefix][0], next_beams[prefix][1])
            for prefix, _score in scored
        }

    best_prefix = max(beams.items(), key=lambda kv: _logsumexp2(kv[1][0], kv[1][1]))[0]
    return list(best_prefix)
