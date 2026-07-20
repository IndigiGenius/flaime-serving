"""Package-level tests for the flaime_serving scaffold (26Q3-REPO-01)."""

import importlib


def test_import_flaime_serving() -> None:
    module = importlib.import_module("flaime_serving")
    assert module is not None


def test_public_api_is_frozen_subset() -> None:
    """Only landed frozen-API names are exported (final freeze audited in REPO-04)."""
    import flaime_serving

    assert list(getattr(flaime_serving, "__all__", [])) == ["ASRModelFactory"]
