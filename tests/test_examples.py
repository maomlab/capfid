"""Smoke test: the quickstart example runs end to end."""
import pathlib
import runpy

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")


def test_quickstart_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = pathlib.Path(__file__).parents[1] / "examples" / "quickstart.py"
    runpy.run_path(str(script), run_name="__main__")
    assert (tmp_path / "quickstart_capacity_fidelity.png").exists()
