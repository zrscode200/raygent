from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from raygent_harness.agents.models import AgentContextPolicy, AgentDefinition
from raygent_harness.context_providers import (
    GitCommandResult,
    GitStatusContextProvider,
    ProjectInstructionConfig,
    ProjectInstructionsContextProvider,
    ReadAdjacentProjectInstructionsContextProvider,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import ContextFragment
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import message_param_from_api_message
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.model_types import ModelRequest
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.core.tool_execution import ToolExecutionResult, run_tool_use
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    TranscriptMessageEntry,
    TranscriptScope,
)
from raygent_harness.tools.agent_tool import AGENT_TOOL_NAME, build_agent_tool
from raygent_harness.tools.file_read_tool import FILE_READ_TOOL_NAME
from raygent_harness.tools.file_tools import (
    create_file_tooling_runtime,
    create_file_tools_catalog_provider,
)
from tests.fakes import FakeModelProvider


def _ctx(cwd: Path, *, agent_id: str | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _request_messages(request: ModelRequest) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], message_param_from_api_message(message))
        for message in request.messages
    ]


def _request_text(request: ModelRequest) -> str:
    return "\n".join(
        str(message.get("content", "")) for message in _request_messages(request)
    )


async def _git_runner(
    args: tuple[str, ...],
    _cwd: Path,
    _timeout_s: float,
    _max_output_bytes: int,
) -> GitCommandResult:
    if args == ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"):
        return GitCommandResult(returncode=0, stdout="true\n")
    if args == ("--no-optional-locks", "rev-parse", "--abbrev-ref", "HEAD"):
        return GitCommandResult(returncode=0, stdout="main\n")
    if args == (
        "--no-optional-locks",
        "symbolic-ref",
        "--short",
        "refs/remotes/origin/HEAD",
    ):
        return GitCommandResult(returncode=0, stdout="origin/main\n")
    if args == ("config", "user.name"):
        return GitCommandResult(returncode=0, stdout="Test User\n")
    if args == ("--no-optional-locks", "status", "--short"):
        return GitCommandResult(returncode=0, stdout=" M pkg/feature/src/main.py\n")
    if args == ("--no-optional-locks", "log", "--oneline", "-n", "5"):
        return GitCommandResult(returncode=0, stdout="abc123 init\n")
    return GitCommandResult(returncode=1, stderr="unexpected command")


@pytest.mark.asyncio
async def test_main_loop_model_input_context_is_ordered_deduped_and_transient(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    target = root / "pkg" / "feature" / "src" / "main.py"
    _write(target, "print('hi')\n")
    _write(root / "AGENTS.md", "ROOT-TURN-ENTRY-POLICY\n")
    _write(root / "pkg" / "feature" / "AGENTS.md", "FEATURE-READ-ADJACENT-POLICY\n")
    _write(
        root / "pkg" / "feature" / ".claude" / "rules" / "python.md",
        "---\npaths: src/*.py\n---\nFEATURE-CONDITIONAL-RULE\n",
    )
    _write(
        root / ".claude" / "rules" / "root-python.md",
        "---\npaths: pkg/feature/src/*.py\n---\nROOT-CONDITIONAL-RULE\n",
    )

    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_read_1",
                        "name": FILE_READ_TOOL_NAME,
                        "input": {"file_path": str(target)},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_read_2",
                        "name": FILE_READ_TOOL_NAME,
                        "input": {"file_path": str(target)},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        )
    )
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    instruction_config = ProjectInstructionConfig(
        cwd=root,
        workspace_root=root,
        project_filenames=("AGENTS.md",),
        local_filenames=(),
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        transcript_store=transcript_store,
        tool_catalog_provider=create_file_tools_catalog_provider(
            runtime=create_file_tooling_runtime(),
        ),
        context_providers=(
            GitStatusContextProvider(
                cwd=root,
                command_runner=_git_runner,
            ),
            ProjectInstructionsContextProvider(instruction_config),
        ),
        post_tool_context_providers=(
            ReadAdjacentProjectInstructionsContextProvider(instruction_config),
        ),
    )
    engine = QueryEngine(
        QueryConfig(model="m", session_id="s", system_prompt="base"),
        deps,
        _ctx(root),
    )

    events = [event async for event in engine.submit_message("read twice")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert len(provider.requests) == 3

    first_messages = _request_messages(provider.requests[0])
    first_text = _request_text(provider.requests[0])
    assert provider.requests[0].system_prompt.startswith("base\n\n")
    assert "<git_status>" in provider.requests[0].system_prompt
    assert "Current branch: main" in provider.requests[0].system_prompt
    assert "Main branch (you will usually use this for PRs): main" in (
        provider.requests[0].system_prompt
    )
    assert "Git user: Test User" in provider.requests[0].system_prompt
    assert "Status:\nM pkg/feature/src/main.py" in provider.requests[0].system_prompt
    assert "Recent commits:\nabc123 init" in provider.requests[0].system_prompt
    assert "ROOT-TURN-ENTRY-POLICY" in str(first_messages[0]["content"])
    assert "FEATURE-READ-ADJACENT-POLICY" not in first_text
    assert "FEATURE-CONDITIONAL-RULE" not in first_text
    assert "ROOT-CONDITIONAL-RULE" not in first_text
    assert first_messages[-1] == {"role": "user", "content": "read twice"}

    second_messages = _request_messages(provider.requests[1])
    second_text = _request_text(provider.requests[1])
    assert "Git user: Test User" in provider.requests[1].system_prompt
    assert "ROOT-TURN-ENTRY-POLICY" in str(second_messages[0]["content"])
    assert "FEATURE-READ-ADJACENT-POLICY" in str(second_messages[1]["content"])
    assert "FEATURE-CONDITIONAL-RULE" in str(second_messages[1]["content"])
    assert "ROOT-CONDITIONAL-RULE" in str(second_messages[1]["content"])
    assert second_messages[2] == {"role": "user", "content": "read twice"}
    assert second_text.count("FEATURE-READ-ADJACENT-POLICY") == 1
    assert second_text.count("FEATURE-CONDITIONAL-RULE") == 1
    assert second_text.count("ROOT-CONDITIONAL-RULE") == 1

    third_text = _request_text(provider.requests[2])
    assert "Git user: Test User" in provider.requests[2].system_prompt
    assert third_text.count("FEATURE-READ-ADJACENT-POLICY") == 1
    assert third_text.count("FEATURE-CONDITIONAL-RULE") == 1
    assert third_text.count("ROOT-CONDITIONAL-RULE") == 1

    persisted_engine_text = "\n".join(
        str(message.get("content", ""))
        for message in engine._messages  # pyright: ignore[reportPrivateUsage]
    )
    assert "ROOT-TURN-ENTRY-POLICY" not in persisted_engine_text
    assert "FEATURE-READ-ADJACENT-POLICY" not in persisted_engine_text
    assert "FEATURE-CONDITIONAL-RULE" not in persisted_engine_text
    assert "ROOT-CONDITIONAL-RULE" not in persisted_engine_text

    transcript_entries = await transcript_store.read_entries(TranscriptScope(session_id="s"))
    transcript_text = "\n".join(
        str(entry.message.get("content", ""))
        for entry in transcript_entries
        if isinstance(entry, TranscriptMessageEntry)
    )
    assert "ROOT-TURN-ENTRY-POLICY" not in transcript_text
    assert "FEATURE-READ-ADJACENT-POLICY" not in transcript_text
    assert "FEATURE-CONDITIONAL-RULE" not in transcript_text
    assert "ROOT-CONDITIONAL-RULE" not in transcript_text


class CustomContextProvider:
    async def __call__(
        self,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        return (
            ContextFragment(
                id="custom",
                content="CUSTOM-CHILD-CONTEXT",
                channel="user_context",
                source="custom",
            ),
        )


async def _run_agent_tool(
    *,
    deps: QueryDeps,
    ctx: ToolUseContext,
    input_: dict[str, Any],
) -> list[ToolExecutionResult]:
    events = [
        event
        async for event in run_tool_use(
            tool_use=ToolUseBlock(
                id="toolu_agent",
                name=AGENT_TOOL_NAME,
                input=input_,
                index=0,
            ),
            assistant_message={"role": "assistant", "content": []},
            tools=ctx.tools,
            deps=deps,
            ctx=ctx,
        )
    ]
    return [event for event in events if isinstance(event, ToolExecutionResult)]


@pytest.mark.asyncio
async def test_agent_policy_omits_project_and_git_at_child_model_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "PROJECT-CONTEXT-SHOULD-BE-OMITTED\n")
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "child result"},)
    )
    instruction_config = ProjectInstructionConfig(
        cwd=root,
        workspace_root=root,
        project_filenames=("AGENTS.md",),
        local_filenames=(),
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        context_providers=(
            GitStatusContextProvider(
                cwd=root,
                command_runner=_git_runner,
                agent_scope="all",
            ),
            ProjectInstructionsContextProvider(instruction_config),
            CustomContextProvider(),
        ),
    )
    agent_tool = build_agent_tool(
        parent_config=QueryConfig(model="parent-model"),
        parent_deps=deps,
        agent_definitions=(
            AgentDefinition(
                agent_type="explorer",
                description="Explore without heavy inherited context.",
                system_prompt="explorer system",
                tools=(),
                context_policy=AgentContextPolicy.minimal(),
            ),
        ),
    )
    ctx = replace(_ctx(root), tools=(agent_tool,))

    results = await _run_agent_tool(
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "inspect project",
            "subagent_type": "explorer",
            "run_in_background": False,
        },
    )

    assert results
    assert len(provider.requests) == 1
    request = provider.requests[0]
    request_text = _request_text(request)
    assert "CUSTOM-CHILD-CONTEXT" in request_text
    assert "PROJECT-CONTEXT-SHOULD-BE-OMITTED" not in request_text
    assert "<git_status>" not in request.system_prompt
    assert "inspect project" in str(_request_messages(request)[-1]["content"])
