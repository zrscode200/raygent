from __future__ import annotations

import json
from pathlib import Path

import pytest

from raygent_harness.core.task import AgentRouteRecord
from raygent_harness.services.agent_routes import (
    JsonAgentRouteRecordStore,
    agent_route_record_to_dict,
    normalize_agent_route_record_for_resume,
)


def _route_record(
    *,
    task_id: str = "agent-1",
    parent_session_id: str = "session-1",
) -> AgentRouteRecord:
    return AgentRouteRecord(
        agent_id=task_id,
        task_id=task_id,
        task_type="local_agent",
        name="researcher",
        parent_agent_id="parent",
        parent_session_id=parent_session_id,
        runtime_session_id="runtime-old",
        agent_type="worker",
        description="research worker",
        model="model-1",
        system_prompt="worker system",
        tool_names=("Read", "Write"),
        permission_mode="default",
        cwd="/repo",
        worktree_path="/repo/.raygent/worktrees/agent-1",
        worktree_branch="raygent/agent-1",
        worktree_slug="agent-1",
        worktree_created_at=1.0,
        worktree_touched_at=2.0,
        worktree_cleanup_policy="remove_if_clean",
        transcript_path="/repo/.raygent/transcripts/session-1/subagents/agent-1.jsonl",
        is_sidechain=True,
        content_replacement_replay=True,
        route_registered_at=10.0,
    )


@pytest.mark.asyncio
async def test_json_agent_route_record_store_round_trips_full_record(
    tmp_path: Path,
) -> None:
    store = JsonAgentRouteRecordStore(tmp_path)
    record = _route_record()

    await store.save(record)
    result = await store.list_records("session-1")

    assert result.warnings == ()
    assert result.records == (record,)
    assert store.path_for_record("session-1", "agent-1").is_file()


def test_normalize_agent_route_record_for_resume_rebinds_parent_identity() -> None:
    record = _route_record(parent_session_id="old-session")

    normalized = normalize_agent_route_record_for_resume(
        record,
        parent_session_id="resumed-session",
        runtime_session_id="resumed-runtime",
    )

    assert normalized.parent_session_id == "resumed-session"
    assert normalized.runtime_session_id == "resumed-runtime"
    assert normalized.agent_id == record.agent_id
    assert normalized.tool_names == record.tool_names
    assert normalized.worktree_path == record.worktree_path
    assert normalized.route_registered_at == 10.0


@pytest.mark.asyncio
async def test_json_agent_route_record_store_skips_bad_records_fail_soft(
    tmp_path: Path,
) -> None:
    store = JsonAgentRouteRecordStore(tmp_path)
    good = _route_record(task_id="good")
    await store.save(good)

    record_dir = store.record_dir("session-1")
    (record_dir / "bad-json.json").write_text("{not-json", encoding="utf-8")
    mismatched = _route_record(task_id="mismatch", parent_session_id="other-session")
    (record_dir / "mismatch.json").write_text(
        json.dumps(agent_route_record_to_dict(mismatched)),
        encoding="utf-8",
    )
    (record_dir / "wrong-type.json").write_text(
        json.dumps(
            {
                **agent_route_record_to_dict(good),
                "task_id": "wrong-type",
                "agent_id": "wrong-type",
                "task_type": "remote_agent",
            }
        ),
        encoding="utf-8",
    )

    result = await store.list_records("session-1")

    assert result.records == (good,)
    assert len(result.warnings) == 3
    assert all("skipped route record" in warning for warning in result.warnings)
