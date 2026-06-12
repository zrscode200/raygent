from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from raygent_harness.adapters.model_protocols import PreparedModelRequest


def test_prepared_model_request_freezes_provider_payloads() -> None:
    request = PreparedModelRequest(
        protocol_id="example",
        model="model-1",
        body={"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-beta": "enabled"},
        options={"stream": True},
    )

    body = cast(Mapping[str, object], request.body)
    messages = cast(tuple[object, ...], body["messages"])
    first_message = cast(Mapping[str, object], messages[0])

    assert first_message["role"] == "user"
    with pytest.raises(TypeError):
        cast(Any, body)["messages"] = ()


def test_core_does_not_import_adapter_layer() -> None:
    project_root = Path(__file__).resolve().parents[4]
    core_root = project_root / "src" / "raygent_harness" / "core"

    offenders = [
        path.relative_to(project_root)
        for path in core_root.rglob("*.py")
        if "raygent_harness.adapters" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
