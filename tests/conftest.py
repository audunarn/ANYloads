"""Never touch the real user's trust list from a test."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    # A unique, gitignored repository-local basetemp: concurrent runs never
    # fight over one Windows temp directory.
    if getattr(config.option, "basetemp", None) is None:
        config.option.basetemp = str(_ROOT / f".pytest_tmp_{uuid4().hex}")


@pytest.fixture(autouse=True)
def isolated_trust_store(tmp_path, monkeypatch):
    path = tmp_path / "trusted.json"
    monkeypatch.setenv("ANYLOADS_TRUSTED_CODE_FILE", str(path))
    return path
