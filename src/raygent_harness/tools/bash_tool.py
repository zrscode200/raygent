"""Concrete model-callable Bash tool over the local_bash task lifecycle.


Raygent keeps this headless and provider-neutral: the tool delegates all process
lifecycle, output persistence, timeouts, output caps, stall notifications, and
process-tree cleanup to `core.tasks.local_bash`. The wrapper only owns the
model-callable contract, restricted command validation, permission decision, and
foreground/background result formatting.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from raygent_harness.core.permissions import (
    OtherPermissionDecisionReason,
    PermissionAskDecision,
    PermissionDenyDecision,
    PermissionResult,
    ToolPermissionContext,
)
from raygent_harness.core.stall_watchdog import LastOutputObserver, StallWatchdog
from raygent_harness.core.task import TERMINAL_STATUSES, AppStateStore
from raygent_harness.core.tasks.local_bash import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_S,
    LocalBashState,
    LocalBashTask,
    read_local_bash_output,
    spawn_local_bash,
)
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    ValidationResult,
    build_tool,
)
from raygent_harness.services.task_output import (
    DEFAULT_MAX_READ_BYTES,
    TaskOutputReadResult,
    TaskOutputStore,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import QueryDeps, ToolCatalogProvider
    from raygent_harness.skills.models import SkillDefinition


BASH_TOOL_NAME = "Bash"
BASH_TOOL_ALIAS = "Shell"
BASH_MAX_RESULT_SIZE_CHARS = 30_000
BASH_DEFAULT_OUTPUT_READ_BYTES = 20_000
BASH_RESTRICTED_PROFILE_NAME = "restricted Bash profile"
BASH_PROMPT = """Run a restricted local Bash command.

Use Bash for short, bounded shell commands when a dedicated tool is not a better
fit. Prefer Glob/Grep/Read for file discovery and reading. Commands run through
Raygent's restricted Bash profile and must be approved by the permission layer
before process start.

Rules:
- Set run_in_background=true for long-running commands.
- Foreground commands block until they complete or hit timeout_s.
- Background commands return a task_id and output_file; use Read on output_file
  or TaskStop with task_id later.
- Do not use Bash for writes, redirects, command substitution, shell scripts, or
  unsafe pipelines; use dedicated Write/Edit tools for file mutation.
"""

_ALLOWED_COMMANDS = frozenset(
    {
        "cat",
        "cut",
        "du",
        "echo",
        "file",
        "find",
        "git",
        "grep",
        "head",
        "ls",
        "printf",
        "pwd",
        "rg",
        "sort",
        "strings",
        "tail",
        "tr",
        "uniq",
        "wc",
        "which",
    }
)
_ALLOWED_GIT_SUBCOMMANDS = frozenset(
    {
        "describe",
        "diff",
        "grep",
        "log",
        "ls-files",
        "remote",
        "rev-parse",
        "show",
        "status",
    }
)
_FORBIDDEN_OPERATOR_TOKENS = frozenset(
    {
        "&",
        "&&",
        ";",
        "(",
        ")",
        "<",
        "<<",
        "<<<",
        "<>",
        ">",
        ">>",
        ">|",
        ">&",
        "&>",
        "|&",
        "||",
    }
)
_FORBIDDEN_SUBSTRINGS = (
    "$",
    "$(",
    "`",
    "<(",
    ">(",
    "${",
    "\n",
    "\r",
)
_FORBIDDEN_FIND_FLAGS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"})
_FORBIDDEN_SORT_FLAGS = frozenset({"-o", "--output"})
_FORBIDDEN_RG_FLAGS = frozenset({"--pre", "--pre-glob"})
_FORBIDDEN_GIT_FLAGS = frozenset({"--ext-diff", "--paginate", "-p"})
_FORBIDDEN_GIT_FLAGS_WITH_VALUE = frozenset(
    {"-c", "--config-env", "--exec-path", "--output", "--open-files-in-pager"}
)
_FORBIDDEN_REMOTE_SUBCOMMANDS = frozenset(
    {
        "add",
        "remove",
        "rm",
        "rename",
        "set-branches",
        "set-head",
        "set-url",
        "update",
        "prune",
    }
)


class BashInput(BaseModel):
    command: str = Field(description="The restricted shell command to execute.")
    description: str | None = Field(
        default=None,
        description="Short active-voice description of what the command does.",
    )
    timeout_s: float = Field(
        default=DEFAULT_TIMEOUT_S,
        gt=0,
        le=600,
        description="Hard timeout in seconds before the process tree is killed.",
    )
    run_in_background: bool = Field(
        default=False,
        description="Set true to return immediately with a background task id.",
    )
    max_output_bytes: int = Field(
        default=DEFAULT_MAX_OUTPUT_BYTES,
        gt=0,
        le=64 * 1024 * 1024,
        description="Maximum captured output bytes before the process is killed.",
    )
    output_read_bytes: int = Field(
        default=BASH_DEFAULT_OUTPUT_READ_BYTES,
        ge=0,
        le=DEFAULT_MAX_READ_BYTES,
        description="Foreground result tail bytes to include in the tool result.",
    )


class RestrictedBashValidation(BaseModel):
    allowed: bool
    message: str | None = None


def build_bash_tool(
    *,
    deps: QueryDeps,
    output_store: TaskOutputStore | None = None,
    watchdog: StallWatchdog | None = None,
) -> Tool:
    """Build the concrete `Bash` tool over `deps.task_store`."""

    async def validate_input(
        input_: BaseModel,
        _ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        result = validate_restricted_bash_command(parsed.command)
        if not result.allowed:
            return ValidationError(message=result.message or "Command is not allowed")
        return ValidationOk()

    async def check_permissions(
        input_: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        parsed = _coerce_input(input_)
        result = validate_restricted_bash_command(parsed.command)
        if not result.allowed:
            return PermissionDenyDecision(
                message=result.message or "Command is not allowed",
                decision_reason=OtherPermissionDecisionReason(
                    reason="restricted_bash_profile"
                ),
            )
        description = _description(parsed)
        return PermissionAskDecision(
            message=f"Run {BASH_RESTRICTED_PROFILE_NAME} command: {description}",
            updated_input=parsed.model_dump(exclude_none=True),
            decision_reason=OtherPermissionDecisionReason(
                reason="restricted_bash_profile"
            ),
        )

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        validation = validate_restricted_bash_command(parsed.command)
        if not validation.allowed:
            yield ToolResult(
                content=validation.message or "Command is not allowed",
                is_error=True,
            )
            return

        task_watchdog = watchdog if parsed.run_in_background else _NoopStallWatchdog()
        state = await spawn_local_bash(
            parsed.command,
            deps.task_store,
            description=_description(parsed),
            cwd=ctx.cwd,
            timeout_s=parsed.timeout_s,
            max_output_bytes=parsed.max_output_bytes,
            tool_use_id=ctx.tool_use_id,
            agent_id=ctx.agent_id,
            watchdog=task_watchdog,
            output_store=output_store,
        )

        if parsed.run_in_background:
            yield ToolResult(content=_background_content(state))
            return

        # Foreground Bash already returns its terminal output in this tool
        # result. Pre-mark the task so the shared local_bash driver does not
        # enqueue a duplicate terminal notification for the next model input.
        deps.task_store.update_task(state.id, lambda task: replace(task, notified=True))

        try:
            final = await _wait_until_done_or_abort(
                state.id,
                deps.task_store,
                ctx.abort_event,
            )
        except asyncio.CancelledError:
            await LocalBashTask().kill(state.id, deps.task_store)
            raise

        output = await read_local_bash_output(
            state.id,
            deps.task_store,
            output_store=output_store,
            max_bytes=parsed.output_read_bytes,
            tail=True,
        )
        yield ToolResult(
            content=_foreground_content(final, output),
            is_error=final.status != "completed",
        )

    return build_tool(
        ToolSpec(
            name=BASH_TOOL_NAME,
            aliases=(BASH_TOOL_ALIAS,),
            description="Run a restricted local Bash command.",
            search_hint="run a restricted shell command",
            input_model=BashInput,
            call=call,
            prompt=BASH_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=False,
            is_read_only=False,
            is_destructive=True,
            is_open_world=True,
            interrupt_behavior="cancel",
            max_result_size_chars=BASH_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Running {_description(_coerce_input(input_))}"
            ),
        )
    )


def create_bash_catalog_provider(
    *,
    parent_deps: QueryDeps,
    output_store: TaskOutputStore | None = None,
    watchdog: StallWatchdog | None = None,
    enabled: bool = True,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Create a catalog provider that appends Bash when enabled."""

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        bash_tool = build_bash_tool(
            deps=parent_deps,
            output_store=output_store,
            watchdog=watchdog,
        )
        without_existing = _without_colliding_tools(tuple(tools), (bash_tool,))
        if not enabled:
            return without_existing
        return (*without_existing, bash_tool)

    return provider


def validate_restricted_bash_command(command: str) -> RestrictedBashValidation:
    """Fail-closed validator for the v1 restricted Bash profile.

    This is deliberately narrower than the reference classifier stack. It only
    allows simple read/search/status commands and safe pipelines between them.
    """

    stripped = command.strip()
    if not stripped:
        return _blocked("command is required for Bash")
    for marker in _FORBIDDEN_SUBSTRINGS:
        if marker in command:
            return _blocked(
                f"Bash command is outside the {BASH_RESTRICTED_PROFILE_NAME}: "
                "command substitution, multiline commands, and parameter "
                "expansion are not allowed."
            )

    try:
        lexer = shlex.shlex(stripped, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = tuple(lexer)
    except ValueError as exc:
        return _blocked(f"Invalid Bash command syntax: {exc}")

    if not tokens:
        return _blocked("command is required for Bash")
    if any(token in _FORBIDDEN_OPERATOR_TOKENS for token in tokens):
        return _blocked(
            f"Bash command is outside the {BASH_RESTRICTED_PROFILE_NAME}: "
            "redirects, compound operators, backgrounding, and subshells are "
            "not allowed."
        )

    segments = _split_pipeline(tokens)
    if segments is None:
        return _blocked(
            f"Bash command is outside the {BASH_RESTRICTED_PROFILE_NAME}: "
            "empty or malformed pipelines are not allowed."
        )

    for segment in segments:
        result = _validate_command_segment(segment)
        if not result.allowed:
            return result
    return RestrictedBashValidation(allowed=True)


def _split_pipeline(tokens: tuple[str, ...]) -> tuple[tuple[str, ...], ...] | None:
    segments: list[tuple[str, ...]] = []
    current: list[str] = []
    for token in tokens:
        if token == "|":
            if not current:
                return None
            segments.append(tuple(current))
            current = []
            continue
        current.append(token)
    if not current:
        return None
    segments.append(tuple(current))
    return tuple(segments)


def _validate_command_segment(tokens: tuple[str, ...]) -> RestrictedBashValidation:
    if not tokens:
        return _blocked("empty command segment is not allowed")
    command = tokens[0]
    if _looks_like_env_assignment(command):
        return _blocked(
            f"Bash command is outside the {BASH_RESTRICTED_PROFILE_NAME}: "
            "environment assignment prefixes are not allowed."
        )
    if "/" in command:
        return _blocked(
            f"Bash command is outside the {BASH_RESTRICTED_PROFILE_NAME}: "
            "commands must use allowlisted command names, not executable paths."
        )
    if command not in _ALLOWED_COMMANDS:
        return _blocked(
            f"Bash command '{command}' is outside the "
            f"{BASH_RESTRICTED_PROFILE_NAME}."
        )
    if command == "git":
        return _validate_git_segment(tokens)
    if command == "find":
        return _validate_find_segment(tokens)
    if command == "rg":
        return _validate_rg_segment(tokens)
    if command == "sort":
        return _validate_sort_segment(tokens)
    return RestrictedBashValidation(allowed=True)


def _validate_git_segment(tokens: tuple[str, ...]) -> RestrictedBashValidation:
    for token in tokens[1:]:
        if _is_forbidden_git_token(token):
            return _blocked(
                f"Bash git flag '{token}' is outside the "
                f"{BASH_RESTRICTED_PROFILE_NAME}."
            )
    subcommand_index = _git_subcommand_index(tokens)
    if subcommand_index is None:
        return _blocked("Bash git commands must include a safe read-only subcommand.")
    subcommand = tokens[subcommand_index]
    if subcommand not in _ALLOWED_GIT_SUBCOMMANDS:
        return _blocked(
            f"Bash git subcommand '{subcommand}' is outside the "
            f"{BASH_RESTRICTED_PROFILE_NAME}."
        )
    if subcommand == "remote":
        remainder = tokens[subcommand_index + 1 :]
        for token in remainder:
            if token.startswith("-"):
                continue
            if token in _FORBIDDEN_REMOTE_SUBCOMMANDS:
                return _blocked(
                    f"Bash git remote subcommand '{token}' is outside the "
                    f"{BASH_RESTRICTED_PROFILE_NAME}."
                )
    return RestrictedBashValidation(allowed=True)


def _git_subcommand_index(tokens: tuple[str, ...]) -> int | None:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-C", "--git-dir", "--work-tree"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return index
    return None


def _validate_find_segment(tokens: tuple[str, ...]) -> RestrictedBashValidation:
    for token in tokens[1:]:
        if token in _FORBIDDEN_FIND_FLAGS or _is_forbidden_find_output_flag(token):
            return _blocked(
                f"Bash find flag '{token}' is outside the "
                f"{BASH_RESTRICTED_PROFILE_NAME}."
            )
    return RestrictedBashValidation(allowed=True)


def _validate_rg_segment(tokens: tuple[str, ...]) -> RestrictedBashValidation:
    for token in tokens[1:]:
        if token in _FORBIDDEN_RG_FLAGS or any(
            token.startswith(f"{flag}=") for flag in _FORBIDDEN_RG_FLAGS
        ):
            return _blocked(
                f"Bash rg flag '{token}' is outside the "
                f"{BASH_RESTRICTED_PROFILE_NAME}."
            )
    return RestrictedBashValidation(allowed=True)


def _validate_sort_segment(tokens: tuple[str, ...]) -> RestrictedBashValidation:
    for token in tokens[1:]:
        if token in _FORBIDDEN_SORT_FLAGS or token.startswith("--output="):
            return _blocked(
                f"Bash sort output flag '{token}' is outside the "
                f"{BASH_RESTRICTED_PROFILE_NAME}."
            )
    return RestrictedBashValidation(allowed=True)


def _is_forbidden_find_output_flag(token: str) -> bool:
    return token == "-fls" or token.startswith(("-fprint", "-fprintf"))


def _is_forbidden_git_token(token: str) -> bool:
    if token in _FORBIDDEN_GIT_FLAGS:
        return True
    for flag in _FORBIDDEN_GIT_FLAGS_WITH_VALUE:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def _background_content(state: LocalBashState) -> list[dict[str, object]]:
    text = (
        f"Command running in background with ID: {state.id}. "
        f"Output is being written to: {state.output_file or '(unavailable)'}. "
        "Use Read on output_file to inspect logs, or TaskStop with task_id to stop it."
    )
    return [
        {"type": "text", "text": text},
        {
            "type": "bash_task",
            "task_id": state.id,
            "status": state.status,
            "command": state.command,
            "description": state.description,
            "output_file": state.output_file,
            "run_in_background": True,
        },
    ]


def _foreground_content(
    final: LocalBashState,
    output: TaskOutputReadResult,
) -> list[dict[str, object]]:
    status_line = _status_line(final)
    output_text = _decode_output(output.content).rstrip()
    if not output_text:
        output_text = "(No output)"
    truncation = _truncation_note(output)
    text = (
        f"{status_line}\n"
        f"Output file: {final.output_file or '(unavailable)'}\n"
        f"Bytes: {output.bytes_total}\n\n"
        f"{output_text}"
        f"{truncation}"
    )
    return [
        {"type": "text", "text": _cap_text(text, BASH_MAX_RESULT_SIZE_CHARS)},
        {
            "type": "bash_result",
            "task_id": final.id,
            "status": final.status,
            "exit_code": final.exit_code,
            "output_file": final.output_file,
            "bytes_total": output.bytes_total,
            "start_offset": output.start_offset,
            "next_offset": output.next_offset,
            "truncated_before": output.truncated_before,
            "truncated_after": output.truncated_after,
            "killed_by_timeout": final.killed_by_timeout,
            "killed_by_size": final.killed_by_size,
        },
    ]


def _status_line(final: LocalBashState) -> str:
    if final.status == "completed":
        return f"Command completed (exit {final.exit_code})."
    if final.killed_by_timeout:
        return f"Command killed after timeout_s={final.timeout_s:.3g}."
    if final.killed_by_size:
        return f"Command killed after output exceeded {final.max_output_bytes} bytes."
    if final.status == "killed":
        return f"Command killed (exit {final.exit_code})."
    return f"Command failed (exit {final.exit_code})."


def _truncation_note(output: TaskOutputReadResult) -> str:
    notes: list[str] = []
    if output.truncated_before:
        notes.append("earlier output omitted")
    if output.truncated_after:
        notes.append("later output omitted")
    if not notes:
        return ""
    return "\n\n[Output truncated: " + ", ".join(notes) + ".]"


def _decode_output(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


async def _wait_until_done_or_abort(
    task_id: str,
    store: AppStateStore,
    abort_event: asyncio.Event,
) -> LocalBashState:
    while True:
        task = store.tasks.get(task_id)
        if isinstance(task, LocalBashState) and task.status in TERMINAL_STATUSES:
            return task
        if abort_event.is_set():
            await LocalBashTask().kill(task_id, store)
            final = store.tasks.get(task_id)
            if isinstance(final, LocalBashState):
                return final
        await asyncio.sleep(0.05)


class _NoopStallWatchdog:
    """Foreground Bash has no model-facing background-stall signal."""

    def start(
        self,
        task_id: str,
        description: str,
        observer: LastOutputObserver,
        store: AppStateStore,
        *,
        tool_use_id: str | None = None,
        agent_id: str | None = None,
    ) -> asyncio.Task[None]:
        del description, observer, store, tool_use_id, agent_id
        return asyncio.create_task(
            _wait_forever(),
            name=f"stall-watchdog-disabled:{task_id}",
        )


async def _wait_forever() -> None:
    event = asyncio.Event()
    await event.wait()


def _description(input_: BashInput) -> str:
    if input_.description is not None and input_.description.strip():
        return input_.description.strip()
    return input_.command.strip()


def _blocked(message: str) -> RestrictedBashValidation:
    return RestrictedBashValidation(allowed=False, message=message)


def _looks_like_env_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    name, _sep, _value = token.partition("=")
    return bool(name) and name.replace("_", "A").isalnum() and not name[0].isdigit()


def _coerce_input(input_: BaseModel) -> BashInput:
    if isinstance(input_, BashInput):
        return input_
    return BashInput.model_validate(input_.model_dump())


def _cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Output truncated to fit tool result bounds.]"


def _without_colliding_tools(
    tools: tuple[Tool, ...],
    runtime_tools: tuple[Tool, ...],
) -> tuple[Tool, ...]:
    reserved_names: set[str] = set()
    for tool in runtime_tools:
        reserved_names.update({tool.name, *tool.aliases})
    return tuple(
        tool for tool in tools if {tool.name, *tool.aliases}.isdisjoint(reserved_names)
    )


__all__ = [
    "BASH_DEFAULT_OUTPUT_READ_BYTES",
    "BASH_MAX_RESULT_SIZE_CHARS",
    "BASH_PROMPT",
    "BASH_RESTRICTED_PROFILE_NAME",
    "BASH_TOOL_ALIAS",
    "BASH_TOOL_NAME",
    "BashInput",
    "RestrictedBashValidation",
    "build_bash_tool",
    "create_bash_catalog_provider",
    "validate_restricted_bash_command",
]
