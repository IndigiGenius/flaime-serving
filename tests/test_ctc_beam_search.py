"""Tests for prefix CTC beam search (moved from FLAIME 26Q3-REPO-02).

Covers these acceptance criteria:
  - Module/function exist with the documented signature.
  - Accepts (batch, time, vocab) and (time, vocab) shapes.
  - beam_width=1 is equivalent to greedy + blank/duplicate collapse.
  - Peaked one-hot input decodes to the hidden sequence exactly.
  - Beam search beats greedy on a constructed disagreement case.
  - Input validation: bad beam_width / empty logits / bad blank index.
"""

from __future__ import annotations

import math

import pytest
import torch

from flaime_serving.ctc_beam_search import _logsumexp2, beam_search_ctc_decode

NEG_INF = float("-inf")


def _greedy_decode_and_collapse(
    logits: torch.Tensor, blank: int = 0
) -> list[list[int]]:
    """Local greedy-decode + collapse reference, used only as a test baseline.

    Avoids depending on flaime.infrastructure.models.ctc_utils, which also holds
    training-only code (compute_ctc_loss) out of scope for flaime-serving.
    """
    predicted = torch.argmax(logits, dim=-1)
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


def _one_hot_log_probs(
    token_ids: list[int], vocab: int, peak: float = 10.0
) -> torch.Tensor:
    """Build (time, vocab) log-probs sharply peaked on `token_ids`."""
    time = len(token_ids)
    logits = torch.full((time, vocab), -peak)
    for t, tid in enumerate(token_ids):
        logits[t, tid] = peak
    return logits.log_softmax(dim=-1)


def test_batched_shape_returns_one_list_per_batch() -> None:
    log_probs = torch.randn(3, 20, 8).log_softmax(-1)
    out = beam_search_ctc_decode(log_probs, beam_width=3, blank=0)
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(seq, list) for seq in out)
    assert all(all(isinstance(tid, int) for tid in seq) for seq in out)


def test_unbatched_shape_is_accepted() -> None:
    log_probs = torch.randn(20, 8).log_softmax(-1)
    out = beam_search_ctc_decode(log_probs, beam_width=3, blank=0)
    assert isinstance(out, list)
    assert len(out) == 1  # single-utterance convenience: still returns list-of-lists


def test_beam1_matches_greedy_collapse() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 40, 10)
    log_probs = logits.log_softmax(-1)

    beam = beam_search_ctc_decode(log_probs, beam_width=1, blank=0)
    greedy_collapsed = _greedy_decode_and_collapse(logits, blank=0)
    assert beam == greedy_collapsed


def test_peaked_input_decodes_hidden_sequence_exactly() -> None:
    # Hidden tokens: 3, 5, 3 (non-adjacent duplicates allowed once blanks separate).
    # We interleave blanks so the collapsed sequence is literally [3, 5, 3].
    hidden = [3, 0, 5, 0, 3]
    vocab = 10
    log_probs = _one_hot_log_probs(hidden, vocab=vocab)  # (time=5, vocab=10)
    out = beam_search_ctc_decode(log_probs, beam_width=4, blank=0)
    assert out == [[3, 5, 3]]


def test_beam_beats_greedy_on_ambiguous_input() -> None:
    # Construct a 2-timestep, 3-token case (blank=0, token A=1, token B=2) where
    # greedy picks B at each step (→ collapsed "B") but the joint-prob-maximizing
    # CTC path is <blank, A> (→ collapsed "A"). The trick: at t=0 B has highest
    # single-step prob but the alignment <blank,A> accumulates higher joint prob
    # once we sum over both emit-A paths.
    #
    # We just assert beam search returns a DIFFERENT sequence than greedy here,
    # AND that its total log-prob under the model is >= greedy's.
    log_probs = torch.tensor(
        [
            [
                math.log(0.45),
                math.log(0.10),
                math.log(0.45),
            ],  # t=0: blank & B tie, A small
            [
                math.log(0.05),
                math.log(0.55),
                math.log(0.40),
            ],  # t=1: A wins but not by much
        ]
    )

    greedy_collapsed = _greedy_decode_and_collapse(log_probs.unsqueeze(0), blank=0)[0]
    beam_out = beam_search_ctc_decode(log_probs, beam_width=3, blank=0)[0]

    # Behavioral: beam disagrees with greedy on this constructed case.
    assert beam_out != greedy_collapsed, (
        f"Expected beam≠greedy on ambiguous input; got beam={beam_out}, greedy={greedy_collapsed}"
    )

    # Score: for this tiny case, brute-force all CTC paths and sum probs per collapsed sequence.
    vocab = log_probs.shape[1]
    seq_logp: dict[tuple[int, ...], float] = {}
    for c0 in range(vocab):
        for c1 in range(vocab):
            lp = float(log_probs[0, c0] + log_probs[1, c1])
            prev: int | None = None
            collapsed: list[int] = []
            for tok in (c0, c1):
                if tok == 0:
                    prev = None
                    continue
                if tok != prev:
                    collapsed.append(tok)
                    prev = tok
            key = tuple(collapsed)
            seq_logp[key] = _logsumexp2(seq_logp.get(key, NEG_INF), lp)

    assert seq_logp.get(tuple(beam_out), NEG_INF) >= seq_logp.get(tuple(greedy_collapsed), NEG_INF) - 1e-12

@pytest.mark.parametrize("bad_width", [0, -1, -5])
def test_invalid_beam_width_raises(bad_width: int) -> None:
    log_probs = torch.randn(1, 5, 4).log_softmax(-1)
    with pytest.raises(ValueError, match="beam_width"):
        beam_search_ctc_decode(log_probs, beam_width=bad_width, blank=0)


def test_empty_log_probs_raises() -> None:
    log_probs = torch.empty(1, 0, 4)
    with pytest.raises(ValueError, match="empty|time"):
        beam_search_ctc_decode(log_probs, beam_width=3, blank=0)


@pytest.mark.parametrize("bad_blank", [-1, 4, 99])
def test_invalid_blank_index_raises(bad_blank: int) -> None:
    log_probs = torch.randn(1, 5, 4).log_softmax(-1)  # vocab=4 → valid blanks are 0..3
    with pytest.raises(ValueError, match="blank"):
        beam_search_ctc_decode(log_probs, beam_width=3, blank=bad_blank)


# --- _logsumexp2 -----------------------------------------------------------
#
# log-space pairwise addition used in the prefix-beam inner loop.
# Must be numerically stable (no overflow/underflow at extreme magnitudes)
# and correctly handle the -inf "log 0" sentinel.


def test_logsumexp2_matches_analytic_on_small_values() -> None:
    # log(exp(a) + exp(b)) computed directly is safe for small magnitudes.
    a, b = -1.0, -2.0
    expected = math.log(math.exp(a) + math.exp(b))
    assert _logsumexp2(a, b) == pytest.approx(expected, rel=1e-12)


def test_logsumexp2_equals_log_of_sum_for_known_probs() -> None:
    # f(log p, log q) == log(p + q)
    p, q = 0.3, 0.4
    assert _logsumexp2(math.log(p), math.log(q)) == pytest.approx(
        math.log(p + q), rel=1e-12
    )


def test_logsumexp2_is_commutative() -> None:
    for a, b in [(-1.0, -2.0), (0.0, -10.0), (-1e-9, -1e-9), (5.5, 5.5)]:
        assert _logsumexp2(a, b) == pytest.approx(_logsumexp2(b, a), rel=0, abs=0)


def test_logsumexp2_equal_inputs_give_x_plus_log2() -> None:
    for x in [-10.0, -1.0, 0.0, 1.0, 1000.0]:
        assert _logsumexp2(x, x) == pytest.approx(x + math.log(2.0), rel=1e-12)


def test_logsumexp2_neg_inf_is_identity() -> None:
    # -inf is the log-space zero; f(-inf, x) == x and f(x, -inf) == x.
    for x in [-100.0, -1.0, 0.0, 3.5]:
        assert _logsumexp2(NEG_INF, x) == x
        assert _logsumexp2(x, NEG_INF) == x


def test_logsumexp2_both_neg_inf_stays_neg_inf() -> None:
    assert _logsumexp2(NEG_INF, NEG_INF) == NEG_INF


def test_logsumexp2_no_overflow_at_large_positive() -> None:
    # exp(1000) overflows float64; a stable impl returns 1000 + log(2).
    result = _logsumexp2(1000.0, 1000.0)
    assert math.isfinite(result)
    assert result == pytest.approx(1000.0 + math.log(2.0), rel=1e-12)


def test_logsumexp2_no_underflow_at_large_negative() -> None:
    # exp(-1000) underflows to 0; naive log(exp(-1000)+exp(-1001)) → log(0) = -inf.
    # Stable impl keeps full precision.
    result = _logsumexp2(-1000.0, -1001.0)
    assert math.isfinite(result)
    assert result == pytest.approx(-1000.0 + math.log1p(math.exp(-1.0)), rel=1e-12)


def test_logsumexp2_dominates_max() -> None:
    # log(exp a + exp b) >= max(a, b), always.
    for a, b in [(-5.0, -5.1), (0.0, -50.0), (100.0, 99.999), (NEG_INF, -3.0)]:
        assert _logsumexp2(a, b) >= max(a, b) - 1e-15


def test_logsumexp2_precision_near_equal_inputs() -> None:
    # When a ≈ b, naive exp(a-m)+exp(b-m) loses precision; log1p form preserves it.
    x = -7.0
    eps = 1e-12
    result = _logsumexp2(x, x + eps)
    # Ground truth: x + log(1 + exp(eps)) ≈ x + log(2) + eps/2 for small eps.
    expected = x + math.log(2.0) + eps / 2.0
    assert result == pytest.approx(expected, abs=1e-14)
