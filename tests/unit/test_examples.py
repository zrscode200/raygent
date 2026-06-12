"""Smoke tests for documented runnable examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("example", "expected"),
    (
        ("examples/minimal_query.py", "result[success]"),
        ("examples/project_profile.py", "Project profile is ready"),
        ("examples/reusable_factory.py", "factory session: user-alice"),
        ("examples/sdk_callbacks.py", "kernel events: query.turn.started"),
    ),
)
def test_documented_examples_run(example: str, expected: str) -> None:
    completed = subprocess.run(
        [sys.executable, example],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected in completed.stdout
