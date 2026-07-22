"""Package-level tests for the flaime_serving scaffold (26Q3-REPO-01)."""

import importlib


def test_import_flaime_serving() -> None:
    module = importlib.import_module("flaime_serving")
    assert module is not None


def test_no_public_api_yet() -> None:
    """The frozen 8-name API lands in REPO-03/04/21; the scaffold exports nothing."""
    import flaime_serving

    assert list(getattr(flaime_serving, "__all__", [])) == []
