from __future__ import annotations

from typing import Any, cast

import pytest

from raygent_harness.core.config import QueryConfig


def test_query_config_experiments_are_read_only_snapshot() -> None:
    source = {"streaming_tool_execution": True}
    config = QueryConfig(model="model-1", experiments=source)

    source["streaming_tool_execution"] = False
    source["fork_subagent"] = True

    assert config.experiments == {"streaming_tool_execution": True}

    with pytest.raises(TypeError):
        cast("Any", config.experiments)["streaming_tool_execution"] = False

    with pytest.raises(TypeError):
        cast("Any", config.experiments)["new_flag"] = True
