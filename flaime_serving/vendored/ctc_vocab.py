"""CTC vocabulary expansion for multilingual training.

Scans dataset transcriptions to build a character-level vocabulary,
then builds a new CTC tokenizer and resizes the model prediction head
to cover all characters in the training data.

The default wav2vec2-base-960h tokenizer has 32 UPPERCASE English tokens
with do_lower_case=False. Common Voice text is mixed case, so most
characters map to <unk>, corrupting CTC labels entirely.

Task: 26Q1-HPO-05.6 - CTC multilingual vocab expansion
"""

import json
import os
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from torch import nn
from transformers import Wav2Vec2CTCTokenizer

from flaime_serving.vendored.distributed import is_main_process


def build_ctc_tokenizer(characters: set[str]) -> Wav2Vec2CTCTokenizer:
    """Build a CTC tokenizer from a set of characters.

    Creates a new Wav2Vec2CTCTokenizer with do_lower_case=True that covers
    all characters in the training data. The vocabulary uses UPPERCASE
    characters because Wav2Vec2CTCTokenizer._tokenize uppercases input
    when do_lower_case=True (confusing but standard HuggingFace behavior).

    Vocabulary layout:
        0: <pad> (CTC blank token)
        1: <s>
        2: </s>
        3: <unk>
        4: | (word delimiter, represents space)
        5+: sorted UPPERCASE characters from the dataset

    Args:
        characters: Set of all unique characters found in training data.

    Returns:
        A Wav2Vec2CTCTokenizer with the complete character vocabulary.
    """
    # NFC normalization before uppercasing to prevent decomposed duplicates
    normalized = {unicodedata.normalize("NFC", c) for c in characters}
    # Uppercase all characters for the vocab. Wav2Vec2CTCTokenizer's _tokenize
    # uppercases input when do_lower_case=True, so vocab must be UPPERCASE.
    upper_chars = {c.upper() for c in normalized if c.strip()}

    # Remove space (handled by word delimiter "|") and special token chars
    upper_chars.discard(" ")
    upper_chars.discard("|")

    # Build vocab with special tokens first
    vocab: dict[str, int] = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,
    }

    # Add all characters in sorted order (deterministic across ranks for DDP)
    for idx, char in enumerate(sorted(upper_chars), start=5):
        if char not in vocab:
            vocab[char] = idx

    # Write vocab to temp JSON file (required by Wav2Vec2CTCTokenizer)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(vocab, f)
        vocab_file = f.name

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=True,
    )

    return tokenizer


def save_ctc_vocab(tokenizer: Wav2Vec2CTCTokenizer, path: str | Path) -> None:
    """Save CTC tokenizer vocabulary to a JSON file.

    Saves the full vocab dict so it can be loaded deterministically
    during evaluation without needing to re-scan training data.

    Args:
        tokenizer: The CTC tokenizer whose vocab to save.
        path: Output file path (e.g. ``output_dir/ctc_vocab.json``).
    """
    vocab = tokenizer.get_vocab()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_ctc_tokenizer(path: str | Path) -> Wav2Vec2CTCTokenizer:
    """Load a CTC tokenizer from a saved vocabulary JSON file.

    Args:
        path: Path to the vocab JSON file saved by ``save_ctc_vocab``.

    Returns:
        A Wav2Vec2CTCTokenizer with the exact vocabulary from training.

    Raises:
        FileNotFoundError: If the vocab file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CTC vocab file not found: {path}")

    # Wav2Vec2CTCTokenizer requires a file path
    tokenizer = Wav2Vec2CTCTokenizer(
        str(path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=True,
    )
    return tokenizer


def _read_phonet_transcripts_fast(train_ds: Any) -> list[str] | None:
    """Read transcriptions from PhoNet wrappers without decoding audio.

    PhoNet's ``__getitem__`` chain decodes the full audio waveform AND
    converts it to a Python list (``audio_array.tolist()``) on every call —
    millions of float ops per sample. Walking 800 samples just to read
    ``transcription`` strings hangs for tens of minutes.

    Drill through the wrapper layers to reach the underlying
    ``BaseDatasetImplementation`` instance and call ``_get_transcript_entry``
    directly, which returns text without touching audio. Returns ``None`` if
    the dataset isn't a recognizable PhoNet stack — caller falls back to the
    Arrow / row-iteration paths.

    Wrapper layers handled (outer → inner):
    - ``torch.utils.data.ConcatDataset`` → ``.datasets`` (list)
    - ``PhoNetSplitView`` → ``._ds`` + ``._n`` (truncation)
    - ``PhoNetHuggingFaceDataset`` → ``.dataset`` + ``.target_sample_rate``
    - ``BaseDatasetImplementation`` subclass → ``_get_transcript_entry``,
      ``__len__``, ``ipa`` flag

    Surfaced by 26Q2-PHONET-01 P4 cluster smoke (job 1456 stuck for 40min
    in this codepath; py-spy showed 4 ranks all in ``__getitem__`` →
    ``audio_array.tolist()``).
    """
    from torch.utils.data import ConcatDataset

    def _walk(ds: Any, limit: int | None = None) -> list[str]:
        if isinstance(ds, ConcatDataset):
            out: list[str] = []
            for child in ds.datasets:
                # Cap the limit across children TOTAL, not per child, so a
                # truncation pushed down onto a ConcatDataset yields `limit`
                # rows overall (not limit × num_children).
                remaining = None if limit is None else limit - len(out)
                if remaining is not None and remaining <= 0:
                    break
                out.extend(_walk(child, remaining))
            return out

        # FLAIME PhoNetSplitView: ._ds is the next layer, ._n is truncation.
        # Push the truncation DOWN as a row limit so the underlying loop only
        # processes _n rows — otherwise IPA mode runs G2P over the entire
        # underlying dataset (tens of thousands of rows) before slicing to _n,
        # re-introducing the multi-minute hang the fast path exists to avoid.
        inner = getattr(ds, "_ds", None)
        truncate = getattr(ds, "_n", None)
        if inner is not None:
            if isinstance(truncate, int):
                limit = truncate if limit is None else min(truncate, limit)
            return _walk(inner, limit)

        # PhoNetHuggingFaceDataset: .dataset is the underlying PhoNet class
        underlying = getattr(ds, "dataset", None)
        if underlying is not None and hasattr(underlying, "_get_transcript_entry"):
            ipa = bool(getattr(underlying, "ipa", False))
            n = len(underlying)
            if limit is not None:
                n = min(n, limit)

            # The CommonVoice mixin's _get_transcript_entry override runs G2P
            # on every call (commonvoice.py:309). G2P (g2p library) has a
            # latent crash — assert len(c) <= 1 in tokenizer.is_word_character
            # fires on certain inputs (surfaced by 26Q2-PHONET-01 P4 cluster
            # smoke).
            #
            # In NATIVE mode we only need ``entry.transcript``, which the base
            # implementation populates, so we bypass the mixin (and its G2P
            # crash) by resolving BaseDatasetImplementation._get_transcript_entry
            # directly via MRO.
            #
            # In IPA mode we need ``entry.phonetic``, which ONLY the mixin's G2P
            # populates — the base entry leaves it empty. Bypassing the mixin
            # there yields all-empty transcripts and a fail-fast "no non-empty
            # samples" degenerate vocab (26Q1-IPA-INT-02 cluster run), so we
            # must call the mixin and skip individual rows that trip the g2p
            # assert (mirroring PhoNetSplitView._safe_get).
            base_get = None
            if not ipa:
                try:
                    from phonet.datasets.base_implementation import (
                        BaseDatasetImplementation,
                    )

                    if isinstance(underlying, BaseDatasetImplementation):
                        base_get = BaseDatasetImplementation._get_transcript_entry
                except ImportError:
                    pass

            # Per-row failure modes to skip — shared with the DataLoader retry
            # (PhoNetSplitView._safe_get) via one source of truth so the scan
            # and the loader never disagree on what is skippable
            # (IndigiGenius/PhoNet#351).
            from flaime_serving.vendored.phonet_integration import (
                phonet_bad_row_exceptions,
            )

            bad_row_exceptions = phonet_bad_row_exceptions()

            texts: list[str] = []
            skipped = 0  # rows that raised a per-row data error
            empty = 0  # rows whose transcript/phonetic was empty — e.g. g2p
            # returned "" without raising (PhoNet caches an empty phonetic),
            # which a plain exception counter would never surface.
            for idx in range(n):
                try:
                    if base_get is not None:
                        entry = base_get(underlying, idx)
                    else:
                        entry = underlying._get_transcript_entry(idx)
                except bad_row_exceptions:
                    skipped += 1
                    continue
                text = entry.phonetic if ipa else entry.transcript
                if not text or not text.strip():
                    empty += 1
                    continue
                texts.append(text)

            # Report partial loss (rank 0 only) but do NOT fail here: a
            # per-language hard-stop would let one degraded language veto an
            # otherwise-healthy multilingual build. A language that fully fails
            # is caught by extract_characters' per-language guard.
            if (skipped or empty) and is_main_process():
                print(
                    f"[ctc_vocab] transcript scan skipped {skipped} bad + "
                    f"{empty} empty of {n} rows"
                )
            return texts

        # Not a PhoNet stack — bail so caller can try Arrow / row paths.
        raise _NotPhoNetStack

    try:
        return _walk(train_ds)
    except _NotPhoNetStack:
        return None
    except (AttributeError, IndexError):
        return None


class _NotPhoNetStack(Exception):
    """Sentinel raised when the dataset doesn't look like a PhoNet wrapper stack."""


def _read_text_column_bypass_transform(train_ds: Any, text_field: str) -> list[str]:
    """Read a single text column without triggering set_transform.

    FLAIME's BTM pipeline applies ``set_transform(decode_audio_batch)`` to
    HuggingFace datasets, which decodes MP3 audio on every row access. If
    we iterate ``for sample in train_ds`` just to read ``transcription``,
    all audio is silently MP3-decoded — turning a ~1s column read into a
    20-30 minute scan.

    This helper reads the text column directly from the underlying Arrow
    table (or falls back to the dataset's internal storage), bypassing
    the transform entirely. Falls back to row iteration if the dataset
    does not expose an Arrow backend (e.g. streaming datasets or mocks).
    """
    # Fast path for PhoNet wrappers (no Arrow table; ``__getitem__`` decodes
    # audio and tolist()s it — even worse than HF's set_transform path).
    phonet_texts = _read_phonet_transcripts_fast(train_ds)
    if phonet_texts is not None:
        return phonet_texts

    # Preferred: HuggingFace Dataset exposes the raw Arrow table via .data
    try:
        arrow_table = train_ds.data
        # Some versions wrap in a ConcatenationTable with a .table attribute
        if hasattr(arrow_table, "table"):
            arrow_table = arrow_table.table
        column_names = getattr(arrow_table, "column_names", None)
        if column_names is None:
            raise AttributeError
        field = text_field if text_field in column_names else None
        if field is None and "text" in column_names:
            field = "text"
        if field is not None:
            return arrow_table.column(field).to_pylist()
    except (AttributeError, KeyError, TypeError):
        pass

    # Fallback: row iteration (will trigger set_transform)
    texts: list[str] = []
    for sample in train_ds:
        text = sample.get(text_field, sample.get("text", ""))
        if text:
            texts.append(text)
    return texts


def extract_characters(
    datasets: dict[str, dict[str, Any]],
    text_field: str = "transcription",
    min_char_count: int = 1,
) -> set[str]:
    """Scan all training transcriptions and collect unique characters.

    Iterates over the "train" split of each language dataset and collects
    every unique character found in the transcription text. Does NOT
    lowercase — that's the tokenizer's responsibility.

    Reads the text column directly from the Arrow backend to avoid
    triggering set_transform's MP3 audio decode (which turns a 1s column
    read into a 20-30 minute scan).

    When ``min_char_count > 1``, characters appearing fewer than that many
    times across the whole corpus are dropped. This evicts cross-language
    contamination and typo-junk (e.g. a handful of stray Hangul syllables
    leaking into an all-Latin preset) before they waste vocab slots and
    corrupt CTC targets. Counting happens AFTER NFC normalization so
    composed/decomposed forms of the same character are tallied together.
    Use a small floor (2-3) for full runs; leave it at 1 for tiny smoke
    runs where legitimate characters may also be rare.

    Args:
        datasets: Dict mapping language codes to dict with train/val/test
            datasets. Each train sample should have a text field.
        text_field: Primary transcription field name (falls back to "text").
        min_char_count: Minimum corpus-wide occurrences for a character to
            be kept. Default 1 keeps every character (no filtering).

    Returns:
        Set of all unique characters found across all training data.

    Raises:
        RuntimeError: If no non-empty samples are found, or if the frequency
            floor evicts every character (degenerate vocab).
    """
    char_counts: Counter[str] = Counter()
    empty_languages: list[str] = []
    for lang, lang_data in datasets.items():
        train_ds = lang_data.get("train")
        if train_ds is None:
            continue
        lang_contributed = False
        for text in _read_text_column_bypass_transform(train_ds, text_field):
            if text and text.strip():
                # NFC normalization: merge decomposed characters (e + ◌́ → é)
                # to prevent duplicate vocab entries for languages with diacritics
                text = unicodedata.normalize("NFC", text)
                char_counts.update(text)
                lang_contributed = True
        if not lang_contributed:
            empty_languages.append(lang)

    if not char_counts:
        raise RuntimeError(
            f"extract_characters found no non-empty {text_field!r} samples "
            "across all training datasets. Likely causes: g2p failure for "
            "every language in IPA mode, broken data loader, or the wrong "
            "field name. Refusing to build a degenerate vocab."
        )
    # Per-language guard: a language present in the build but contributing zero
    # usable transcripts (g2p fully failed, empty split, wrong field) would be
    # silently omitted from the vocab — its text then trains/evals as all-<unk>.
    # Fail loud naming it rather than ship a model that can't represent it.
    if empty_languages:
        raise RuntimeError(
            f"extract_characters: language(s) {empty_languages} yielded no "
            f"usable {text_field!r} transcripts (every row empty or skipped). "
            "Likely g2p failure for those languages (IPA mode), a broken loader, "
            "or an empty split. Refusing to silently drop a language from the "
            "CTC vocab."
        )

    # Apply the floor case-insensitively. build_ctc_tokenizer uppercases the
    # surviving set, so a letter whose occurrences are split across rare upper-
    # and lower-case forms would be wrongly evicted if each form were counted
    # alone — tally case variants together against the floor.
    if min_char_count > 1:
        upper_counts: Counter[str] = Counter()
        for char, count in char_counts.items():
            upper_counts[char.upper()] += count
        characters = {
            char for char in char_counts if upper_counts[char.upper()] >= min_char_count
        }
    else:
        characters = set(char_counts)
    if not characters:
        raise RuntimeError(
            f"extract_characters: min_char_count={min_char_count} evicted "
            f"every character (corpus had {len(char_counts)} distinct chars, "
            "all below the floor). Lower min_char_count or use more data. "
            "Refusing to build a degenerate vocab."
        )
    return characters


def resize_ctc_head(model: Any, new_vocab_size: int) -> None:
    """Replace the CTC prediction head with a new one sized for the vocabulary.

    Wav2Vec2ForCTC.resize_token_embeddings() throws NotImplementedError,
    so we manually replace lm_head with a fresh nn.Linear. Old weights are
    NOT copied since the old tokenizer had the wrong character set.

    No-ops when the head already has the correct size (e.g. after merging,
    where the merged weights must be preserved).

    Args:
        model: HuggingFace CTC model (e.g. Wav2Vec2ForCTC) with .lm_head
            and .config.hidden_size attributes.
        new_vocab_size: Number of tokens in the new vocabulary.
    """
    current_head = getattr(model, "lm_head", None)
    if current_head is not None and current_head.weight.shape[0] == new_vocab_size:
        return
    hidden_size: int = model.config.hidden_size
    model.lm_head = nn.Linear(hidden_size, new_vocab_size)
    model.config.vocab_size = new_vocab_size


# Model types that use a generic English-only CTC tokenizer and need
# vocabulary expansion for multilingual training.
_NEEDS_VOCAB_EXPANSION = {"wav2vec2", "conformer", "xeus"}


def needs_vocab_expansion(model_type: str) -> bool:
    """Check if a model type needs CTC vocabulary expansion.

    wav2vec2 and conformer ship with English-only UPPERCASE tokenizers
    that drop all non-English and lowercase characters. MMS has its own
    multilingual tokenizer. Whisper uses encoder-decoder, not CTC.

    Args:
        model_type: Model type string (e.g. "wav2vec2", "mms", "whisper").

    Returns:
        True if the model needs vocabulary expansion.
    """
    return model_type in _NEEDS_VOCAB_EXPANSION


def maybe_resize_ctc_head_from_state_dict(
    model: Any, state_dict: dict[str, Any]
) -> bool:
    """Resize model's CTC head to match a checkpoint's vocabulary size.

    Handles both CTC head conventions used in FLAIME:

    * wav2vec2 / conformer: inner HuggingFace model at ``wrapper.model`` with
      a ``lm_head`` attribute. State dict key is ``model.lm_head.weight``.
    * XEUS: ``ctc_projection`` attribute directly on the wrapper. State dict
      key is ``ctc_projection.weight`` (no prefix).

    Safe to call on any model — no-ops when the key is absent or sizes
    already match.

    Args:
        model: FLAIME model wrapper (e.g. Wav2Vec2ASRModel, XEUSASRModel).
        state_dict: Checkpoint state dict.

    Returns:
        True if the head was resized, False otherwise.
    """
    # wav2vec2 / conformer: ``wrapper.model.lm_head``
    ckpt_weight = state_dict.get("model.lm_head.weight")
    if ckpt_weight is not None:
        ckpt_vocab_size: int = ckpt_weight.shape[0]
        inner_model = getattr(model, "model", None)
        if inner_model is None:
            return False
        current_head = getattr(inner_model, "lm_head", None)
        if current_head is None:
            return False
        if current_head.weight.shape[0] != ckpt_vocab_size:
            resize_ctc_head(inner_model, ckpt_vocab_size)
            return True
        return False

    # XEUS: ``wrapper.ctc_projection`` directly on the wrapper
    ckpt_weight = state_dict.get("ctc_projection.weight")
    if ckpt_weight is not None:
        ckpt_vocab_size = ckpt_weight.shape[0]
        current_head = getattr(model, "ctc_projection", None)
        if current_head is None:
            return False
        if current_head.weight.shape[0] != ckpt_vocab_size:
            resize_xeus_ctc_head(model, ckpt_vocab_size)
            return True
        return False

    return False


def resize_xeus_ctc_head(
    model: Any, new_vocab_size: int, blank_bias_init: float | None = None
) -> None:
    """Resize the CTC projection head on an XEUS model.

    XEUS uses ``ctc_projection`` (not ``lm_head``) and optionally has
    ``intermediate_decoders`` with per-layer projections that must also
    be resized to keep vocabulary dimensions consistent.

    No-ops when the head already has the correct size (e.g. after merging,
    where the merged weights must be preserved).

    Args:
        model: XEUS model (``XEUSASRModel`` or bare ``nn.Module`` with
            a ``ctc_projection`` attribute).
        new_vocab_size: Number of tokens in the new vocabulary.
        blank_bias_init: If set, initialize bias[0] (blank token) to this
            value after resizing to counter CTC blank-collapse attractor.
    """
    ctc_proj = getattr(model, "ctc_projection", None)
    if ctc_proj is None:
        return
    if ctc_proj.weight.shape[0] == new_vocab_size:
        return
    old_vocab_size: int = ctc_proj.weight.shape[0]
    hidden_size: int = ctc_proj.in_features
    model.ctc_projection = nn.Linear(hidden_size, new_vocab_size)
    if blank_bias_init is not None:
        import torch

        with torch.no_grad():
            model.ctc_projection.bias.data[0] = blank_bias_init
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            f"Resized CTC head: {hidden_size} -> {old_vocab_size} "
            f"=> {hidden_size} -> {new_vocab_size}"
        )
        if blank_bias_init is not None:
            print(f"Set blank bias init: ctc_projection.bias[0] = {blank_bias_init}")

    # Resize intermediate decoders if present (multi-loss XEUS)
    intermediate = getattr(model, "intermediate_decoders", None)
    if intermediate is not None:
        projections = getattr(intermediate, "projections", None)
        if projections is not None:
            for key in list(projections.keys()):
                projections[key] = nn.Linear(hidden_size, new_vocab_size)


def expand_xeus_ctc_vocabulary(
    model: Any,
    processor: Any,
    datasets: dict[str, dict[str, Any]],
    blank_bias_init: float | None = None,
    min_char_count: int = 1,
) -> "Wav2Vec2CTCTokenizer":
    """Full CTC vocabulary expansion pipeline for XEUS models.

    Extracts characters from training data, builds a new tokenizer,
    resizes the model's ``ctc_projection`` head (and intermediate
    decoders), and sets the tokenizer on the processor.

    Args:
        model: XEUS model wrapper (``XEUSASRModel``).
        processor: ``XEUSProcessor`` whose ``tokenizer`` will be set.
        datasets: Dict mapping language codes to train/val/test datasets.
        blank_bias_init: If set, initialize bias[0] (blank token) after resize.
        min_char_count: Drop characters appearing fewer than this many times
            corpus-wide (default 1 = keep all). See ``extract_characters``.

    Returns:
        The new Wav2Vec2CTCTokenizer.
    """
    characters = extract_characters(datasets, min_char_count=min_char_count)
    tokenizer = build_ctc_tokenizer(characters)
    resize_xeus_ctc_head(model, len(tokenizer), blank_bias_init=blank_bias_init)
    processor.tokenizer = tokenizer
    return tokenizer


def expand_ctc_vocabulary(
    model: Any,
    processor: Any,
    datasets: dict[str, dict[str, Any]],
    min_char_count: int = 1,
) -> Wav2Vec2CTCTokenizer:
    """Full CTC vocabulary expansion pipeline.

    Extracts characters from training data, builds a new tokenizer,
    resizes the model's CTC prediction head (skipped if already correct
    size), and replaces the processor's tokenizer.

    Args:
        model: HuggingFace CTC model (e.g. Wav2Vec2ForCTC).
        processor: Wav2Vec2Processor whose tokenizer will be replaced.
        datasets: Dict mapping language codes to train/val/test datasets.
        min_char_count: Drop characters appearing fewer than this many times
            corpus-wide (default 1 = keep all). See ``extract_characters``.

    Returns:
        The new Wav2Vec2CTCTokenizer.
    """
    characters = extract_characters(datasets, min_char_count=min_char_count)
    tokenizer = build_ctc_tokenizer(characters)
    resize_ctc_head(model, len(tokenizer))
    processor.tokenizer = tokenizer
    return tokenizer
