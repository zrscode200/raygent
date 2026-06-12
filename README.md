# Raygent

Raygent is a headless, model/provider/platform-agnostic agent harness kernel for
long-running agent sessions.

It provides the runtime substrate for agent loops, tool orchestration,
background tasks, memory, compaction, transcript replay, observability, and
multi-agent coordination. It intentionally does not ship a CLI, UI, hosted
backend, product telemetry dashboard, or live provider SDK client.

## Current Status

Raygent is pre-1.0. The kernel is broad enough to embed in a controlled
application or test harness, but public API boundaries are documented rather
than frozen.

Use the package as a library by wiring your own model provider and choosing the
optional tools, context providers, memory services, transcript stores, and event
sinks you want to install. The recommended ergonomic entrypoints are
`raygent_harness.sdk.create_raygent(...)` for one-shot sessions and
`raygent_harness.sdk.RaygentFactory` for reusable product-owned session
builders. Both assemble the low-level kernel primitives into
conversation-scoped `RaygentSession` objects.

For common embedding shapes, `create_raygent(...)` also accepts documented
presets such as `minimal`, `chat`, `project_reader`, `repo_maintainer`,
`memory_agent`, `long_running_task`, and `full_developer`. Presets are
inspectable compositions over the same factory options; broad developer
capabilities require explicit safety and permission choices.

## What The Kernel Provides

Core runtime:

- `QueryEngine` conversation container and `submit_message(...)` turn API.
- Frozen per-turn `QueryConfig` and injectable `QueryDeps` dependency container.
- Provider-neutral model request, response, streaming, token, media, and error
  types.
- Stop hooks, recovery ladder, compaction layers, and budget handling.
- Transcript/session persistence, compact-boundary replay, sidechain transcripts,
  and content replacement records.

Tools and tasks:

- Tool contract, permission context, permission engine, tool hooks, ToolSearch,
  concrete file `Read`/`Write`/`Edit`/`NotebookEdit`, local `Glob`/`Grep`,
  restricted `Bash`, SkillTool, AgentTool, TeamCreate, SendMessage, and
  TaskStop.
- Streaming tool execution and provider-neutral progress facts.
- Local bash tasks, local agents, in-process teammates, remote-agent protocol
  seam, task notifications, stop-task behavior, and file-backed task output.

Memory and context:

- Memory directory primitives, team memory sync, query-time memory recall, and
  restricted child-agent memory extraction runner.
- Opt-in context providers for environment facts, git status, and project
  instruction files.
- Kernel observability/eval event bus with redacted default payloads.

Advanced runtime:

- Forked skills, foreground/fork/background agent paths, worktree isolation,
  named local-agent routing, coordinator runtime, behavioral expansion seams,
  and handoff classification.

## What Raygent Does Not Ship Yet

- Live Anthropic/OpenAI/other provider SDK clients.
- A CLI or product UI.
- SDK/progress schemas for a specific product surface.
- Hosted remote-agent backend, auth, archive policy, or transport.
- Live MCP transport/client process supervision beyond the current provider-
  neutral MCP identity/client seam.
- Structured media/PDF/notebook read-adjacent instruction attachment.

Raygent does include transport-free protocol translators for Anthropic
Messages-shaped and OpenAI Responses-shaped payloads. They prove request/stream
translation shape, but they do not perform network calls.

## Install For Development

```bash
uv sync
```

Run validation:

```bash
uv run pytest
uv run ruff check src tests
uv run pyright src tests
```

Run examples:

```bash
uv run python -m recipes.create_raygent.project_reader
```

```bash
uv run python examples/minimal_query.py
uv run python examples/project_profile.py
uv run python examples/reusable_factory.py
uv run python examples/sdk_callbacks.py
uv run python examples/with_tools_and_context.py
```

## Minimal Embedding Shape

For most embedders, use the SDK factory:

```python
from raygent_harness.sdk import create_raygent

session = create_raygent(
    provider=my_model_provider,
    model="demo-model",
    cwd=".",
    tools="none",
    context="none",
)

result = await session.run_until_result("Hello")
print(result.result)
```

The factory keeps the kernel headless and provider-neutral. It does not create a
vendor SDK client, read credentials, install a product UI, or enable filesystem
mutation unless a profile or explicit tools request it.

Advanced embedders can still wire the kernel directly. A Raygent turn needs four
pieces:

1. `QueryConfig`: model id, system prompt, tools, budgets, session id.
2. `QueryDeps`: model provider, task store, permissions, optional services.
3. `ToolUseContext`: session identity, cancellation event, cwd, prompt context.
4. `QueryEngine`: conversation container that persists messages across turns.

```python
import asyncio

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_provider import ModelProvider
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext

async def main(provider: ModelProvider) -> None:
    config = QueryConfig(model="demo-model", session_id="session-1")
    deps = QueryDeps(task_store=AppStateStore(), model_provider=provider)
    ctx = ToolUseContext(
        session_id="session-1",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="session-1", depth=0),
    )
    engine = QueryEngine(config, deps, ctx)

    async for event in engine.submit_message("Hello"):
        if isinstance(event, SDKResult):
            print(event.result)
```

See `examples/minimal_query.py` for a complete runnable one-shot factory
example, `examples/reusable_factory.py` for a reusable factory/product-wrapper
pattern, and `examples/provider_runtime_bridge.py` for a transport-free provider
adapter example.

## Provider Support

Core consumes the `ModelProvider` protocol. Any provider can be used if it
implements:

- `complete(ModelRequest) -> ModelResponse`
- `stream(ModelRequest) -> AsyncIterator[ModelStreamEvent]`
- `count_tokens(TokenCountRequest) -> int`
- `resolve_model(...) -> str`
- `model_info(...) -> ModelInfo`
- `classify_error(...) -> ProviderError`

Built-in protocol translators live under
`raygent_harness.adapters.model_protocols`:

- `AnthropicMessagesAdapter`
- `OpenAIResponsesAdapter`

They are transport-free adapters, not live clients.

## Documentation Map

- `docs/kernel_integration_guide.md`: embedding guide and kernel/adapter
  boundaries.
- `docs/public_api_surface.md`: recommended module imports and API stability
  tiers.

## Design Constraints

- Keep core headless.
- Keep model/provider SDKs outside core.
- Prefer explicit dependency injection over globals.
- Treat product UI, hosted backends, and product-specific telemetry as adapter
  work unless they change model-visible kernel semantics.
- Keep raw prompt/code/file/model content out of default observability events.
