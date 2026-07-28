import sys
from pathlib import Path

from scripts.queue_rald_guided_query import resolve_python


def test_resolve_python_uses_current_interpreter_without_path_lookup(monkeypatch) -> None:
    monkeypatch.delenv("PYTHON", raising=False)

    assert resolve_python() == Path(sys.executable).resolve()


def test_resolve_python_rejects_missing_configured_executable(monkeypatch) -> None:
    monkeypatch.setenv("PYTHON", "definitely-not-a-python-executable")

    try:
        resolve_python()
    except FileNotFoundError as error:
        assert "Configured Python executable not found" in str(error)
    else:
        raise AssertionError("Missing configured interpreter was accepted")
