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


@pytest.mark.parametrize(
    ("module", "expected"),
    (
        ("recipes.create_raygent.minimal", "minimal preset ready"),
        ("recipes.create_raygent.chat", "chat transcript:"),
        ("recipes.create_raygent.project_reader", "project_reader tools: Read, Glob, Grep"),
        ("recipes.create_raygent.repo_maintainer", "repo_maintainer output_dir:"),
        ("recipes.create_raygent.memory_agent", "memory_agent prompt_provider: True"),
        ("recipes.create_raygent.long_running_task", "long_running_task output_dir:"),
        ("recipes.create_raygent.full_developer", "full_developer bash_enabled: True"),
    ),
)
def test_create_raygent_recipes_run(module: str, expected: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected in completed.stdout
