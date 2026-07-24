"""Tests for standalone XEUS model components (no ESPnet dependency).

Focused unit tests for the frontend, preencoder, and encoder blocks
implemented in flaime/infrastructure/models/xeus_standalone.py.
"""

import torch

from flaime_serving.vendored.xeus_standalone import (
    _EBranchformerBlock,
    _Encoder,
    _Frontend,
    _MultiHeadSelfAttention,
)


class TestFrontendNormalization:
    """Audio normalization must be per-utterance, not batch-wide.

    Regression test for 26Q1-XEUS-BTM-02 bug #2: `F.layer_norm(x, x.shape)`
    normalized over the entire (B, T) tensor, so padding zeros contaminated
    the normalization statistics of real utterances. ESPnet XEUS pretraining
    uses per-utterance zero-mean unit-variance normalization, so the bug
    feeds the CNN off-distribution features on every forward pass with
    variable-length batches.
    """

    def test_frontend_output_independent_of_batch_neighbors(self) -> None:
        torch.manual_seed(42)
        frontend = _Frontend()
        frontend.eval()

        # Real signal (1 second of speech-like random noise)
        x_signal = torch.randn(1, 16000)
        # Zero "padding" sibling — present in many multilingual batches
        x_zeros = torch.zeros(1, 16000)

        with torch.no_grad():
            # Run sample alone
            out_alone, _ = frontend(x_signal)

            # Run sample batched with a zero-padded sibling
            x_batch = torch.cat([x_signal, x_zeros], dim=0)
            out_batch, _ = frontend(x_batch)
            out_from_batch = out_batch[:1]

        assert torch.allclose(out_alone, out_from_batch, atol=1e-5), (
            "Frontend output for a given sample depends on its batch "
            "neighbors. Normalization is batch-wide rather than "
            "per-utterance, contaminating real utterances with "
            "padding-zero statistics."
        )

    def test_frontend_output_independent_of_batch_size(self) -> None:
        """Same signal at batch sizes 1, 2, 4 must produce the same features."""
        torch.manual_seed(42)
        frontend = _Frontend()
        frontend.eval()

        x_signal = torch.randn(1, 16000)

        with torch.no_grad():
            out_b1, _ = frontend(x_signal)
            out_b2, _ = frontend(x_signal.repeat(2, 1))
            out_b4, _ = frontend(x_signal.repeat(4, 1))

        assert torch.allclose(out_b1, out_b2[:1], atol=1e-5)
        assert torch.allclose(out_b1, out_b4[:1], atol=1e-5)

    def test_frontend_norm_excludes_padding(self) -> None:
        """When lengths are provided, padding zeros must not affect normalization."""
        torch.manual_seed(42)
        frontend = _Frontend()
        frontend.eval()

        real_len = 8000
        pad_len = 16000

        x_short = torch.randn(1, real_len)
        # Pad to full length with zeros
        x_padded = torch.nn.functional.pad(x_short, (0, pad_len - real_len))
        lengths = torch.tensor([real_len])

        with torch.no_grad():
            # Run unpadded (no lengths needed — all samples same length)
            out_unpadded, _ = frontend(x_short)
            # Run padded with correct lengths
            out_padded, _ = frontend(x_padded, lengths=lengths)

        # Output frames for real_len through the CNN frontend
        out_frames_short = out_unpadded.shape[1]
        out_frames_padded = out_padded.shape[1]
        # Padded version has more output frames, but the real portion
        # (up to out_frames_short) should match the unpadded output.
        assert out_frames_padded >= out_frames_short
        assert torch.allclose(
            out_unpadded, out_padded[:, :out_frames_short, :], atol=1e-4
        ), (
            "Frontend normalization includes padding zeros in statistics. "
            "Short utterances padded to batch max get distorted."
        )


class TestAttentionPaddingMask:
    """Self-attention must ignore padded positions via a key-padding mask.

    Regression test for 26Q1-XEUS-BTM-02 bug #3: `_MultiHeadSelfAttention`
    called `F.scaled_dot_product_attention` without `attn_mask`. Padded
    positions contaminated attention outputs for real tokens — a major
    issue in variable-length multilingual batches.
    """

    def test_attention_ignores_padded_positions(self) -> None:
        torch.manual_seed(42)
        attn = _MultiHeadSelfAttention(dropout_rate=0.0)
        attn.eval()

        B, T, D = 2, 20, 1024
        valid_len = 10

        # padding_mask: (B, T) bool, True = valid, False = padding
        padding_mask = torch.zeros(B, T, dtype=torch.bool)
        padding_mask[:, :valid_len] = True

        x1 = torch.randn(B, T, D)
        x2 = x1.clone()
        # Scramble values at padded positions
        x2[:, valid_len:, :] = torch.randn(B, T - valid_len, D) * 100.0

        with torch.no_grad():
            out1 = attn(x1, padding_mask)
            out2 = attn(x2, padding_mask)

        # Valid positions must produce identical outputs regardless of
        # what sits at padded positions — that's what masking enforces.
        assert torch.allclose(
            out1[:, :valid_len, :], out2[:, :valid_len, :], atol=1e-4
        ), (
            "Self-attention leaks padded-position values into valid-position "
            "outputs. Key-padding mask is not being applied to SDPA."
        )

    def test_attention_without_mask_still_works(self) -> None:
        """Calling forward without a padding_mask must still work (no regression)."""
        torch.manual_seed(42)
        attn = _MultiHeadSelfAttention(dropout_rate=0.0)
        attn.eval()

        x = torch.randn(2, 20, 1024)
        with torch.no_grad():
            out = attn(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_encoder_threads_padding_mask_from_lengths(
        self, monkeypatch: object
    ) -> None:
        """_Encoder must build a padding mask from lengths and pass it to
        every block's self-attention.

        Numeric comparison at the encoder output isn't reliable here — the
        CGMLP/fusion convs (kernel=31) smear padding values across all
        valid positions over 19 blocks regardless of attention masking.
        So we spy on _MultiHeadSelfAttention.forward to verify the mask is
        plumbed through.
        """
        import flaime_serving.vendored.xeus_standalone as xs

        # Shrink the encoder to 1 block for a fast CPU test — the wiring
        # we're verifying is identical across blocks, and building 19
        # full E-Branchformer blocks with 4096-dim FFNs is prohibitively
        # slow on CPU just for an init-time memory allocation.
        monkeypatch.setattr(xs, "_NUM_BLOCKS", 1)  # type: ignore[attr-defined]

        torch.manual_seed(42)
        encoder = _Encoder(dropout_rate=0.0)
        encoder.eval()

        captured_masks: list[torch.Tensor | None] = []
        original_forward = _MultiHeadSelfAttention.forward

        def spy_forward(
            inner_self: _MultiHeadSelfAttention,
            x: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            captured_masks.append(padding_mask)
            return original_forward(inner_self, x, padding_mask)

        monkeypatch.setattr(  # type: ignore[attr-defined]
            _MultiHeadSelfAttention, "forward", spy_forward
        )

        B, T, D = 2, 30, 1024
        valid_len = 15
        lengths = torch.full((B,), valid_len, dtype=torch.long)
        x = torch.randn(B, T, D)

        with torch.no_grad():
            encoder(x, lengths, use_final_output=True)

        # Should have been called once per E-Branchformer block.
        assert len(captured_masks) > 0, "attention was not called at all"
        for mask in captured_masks:
            assert mask is not None, (
                "_Encoder did not thread a padding mask through attention."
            )
            assert mask.shape == (B, T)
            assert mask[:, :valid_len].all(), "valid positions marked as padding"
            assert not mask[:, valid_len:].any(), "padded positions marked as valid"


def _espnet_reference_block_forward(
    block: _EBranchformerBlock, x: torch.Tensor
) -> torch.Tensor:
    """ESPnet-equivalent forward for a single _EBranchformerBlock.

    Mirrors EBranchformerEncoderLayer.forward() in espnet2/asr/encoder/
    e_branchformer_encoder.py. Kept in the test module so any drift
    between our block forward and the reference is caught by direct
    numeric comparison, without needing pretrained weights to expose
    the collapse behavior.
    """
    # Macaron FFN (half-step residual)
    residual = x
    x = block.norm_ff_macaron(x)
    x = residual + 0.5 * block.dropout(block.feed_forward_macaron(x))

    # Parallel branches
    x_att = block.dropout(block.attn(block.norm_mha(x), None))
    x_mlp = block.dropout(block.cgmlp(block.norm_mlp(x)))

    # Merge: depthwise conv over concat, then x_concat + x_tmp residual
    x_concat = torch.cat([x_att, x_mlp], dim=-1)
    x_tmp = x_concat.transpose(1, 2)
    x_tmp = block.depthwise_conv_fusion(x_tmp)
    x_tmp = x_tmp.transpose(1, 2)
    x = x + block.dropout(block.merge_proj(x_concat + x_tmp))

    # FFN BEFORE norm_final
    residual = x
    x = block.norm_ff(x)
    x = residual + 0.5 * block.dropout(block.feed_forward(x))

    return block.norm_final(x)


class TestEBranchformerMatchesEspnet:
    """Block forward must stay numerically identical to the ESPnet reference.

    Regression test for 26Q1-XEUS-BTM-02: our _EBranchformerBlock had two
    deviations from ESPnet's EBranchformerEncoderLayer that together
    collapsed temporal variation to adj_cos=0.999 by block 4 on raw XEUS
    features:

      1. Merge used ``merge_proj(x_tmp)`` instead of the reference
         ``merge_proj(x_concat + x_tmp)``, dropping the un-smoothed
         residual. Each block accumulated depthwise-conv smoothing.
      2. ``norm_final`` was applied before the post-merge FFN instead of
         after, so pretrained γ/β normalized the wrong tensor.

    The collapse only manifests with pretrained weights, so a feature-
    variance test with random init can't catch regressions. Instead we
    diff the output against an in-test reference implementation of the
    correct forward — any future drift will fail the allclose check.
    """

    def test_block_forward_matches_espnet_reference(self, monkeypatch: object) -> None:
        import flaime_serving.vendored.xeus_standalone as xs

        # Shrink for a fast CPU test — we're checking structural
        # equivalence, not scaling behavior.
        monkeypatch.setattr(xs, "_HIDDEN_SIZE", 128)  # type: ignore[attr-defined]
        monkeypatch.setattr(xs, "_NUM_HEADS", 4)  # type: ignore[attr-defined]
        monkeypatch.setattr(xs, "_FFN_DIM", 512)  # type: ignore[attr-defined]
        monkeypatch.setattr(xs, "_CGMLP_DIM", 512)  # type: ignore[attr-defined]

        torch.manual_seed(0)
        block = _EBranchformerBlock(dropout_rate=0.0)
        block.eval()

        x = torch.randn(1, 20, 128)

        with torch.no_grad():
            our_out = block(x)
            ref_out = _espnet_reference_block_forward(block, x)

        assert torch.allclose(our_out, ref_out, atol=1e-6), (
            "Our _EBranchformerBlock.forward diverges from the ESPnet "
            "EBranchformerEncoderLayer.forward reference. Verify that: "
            "(a) merge uses merge_proj(x_concat + x_tmp), not "
            "merge_proj(x_tmp); (b) norm_final is applied AFTER the "
            "post-merge FFN, not before."
        )
