# Kernel Integration Guide

Status: pre-1.0 stabilization documentation.

Raygent is designed to be embedded. The kernel owns agent-loop semantics; the
embedding application owns concrete provider transport, credentials, UI, product
settings, and deployment policy.

## Factory-First Embedding

Most embedders should start with the SDK factory:

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

`create_raygent(...)` builds the same kernel objects described below and returns
a conversation-scoped `RaygentSession`. The factory is intentionally headless:
it does not create live provider SDK clients, read credentials, install a CLI/UI,
enable product telemetry, or mutate global process state.

For applications that create many sessions, use the reusable factory and an
explicit config object:

```python
from raygent_harness.sdk import (
    RaygentFactory,
    RaygentFactoryConfig,
    RaygentModelOptions,
    RaygentSessionOptions,
)

factory = RaygentFactory()

session = factory.create_session(
    RaygentFactoryConfig(
        model_options=RaygentModelOptions(
            provider=my_model_provider,
            model="demo-model",
            system_prompt="You are running inside my application.",
        ),
        session_options=RaygentSessionOptions(
            cwd=".",
            session_id="user-123",
        ),
    )
)
```

Product layers that want their own construction front should depend on the
narrow structural Protocol rather than subclassing Raygent internals:

```python
from dataclasses import dataclass

from raygent_harness.core.model_provider import ModelProvider
from raygent_harness.sdk import (
    RaygentFactoryConfig,
    RaygentModelOptions,
    RaygentSession,
    RaygentSessionFactory,
    RaygentSessionOptions,
)

@dataclass(frozen=True)
class ProductSessionBuilder:
    factory: RaygentSessionFactory
    provider: ModelProvider

    def create_user_session(self, user_id: str) -> RaygentSession:
        return self.factory.create_session(
            RaygentFactoryConfig(
                model_options=RaygentModelOptions(
                    provider=self.provider,
                    model="demo-model",
                ),
                session_options=RaygentSessionOptions(
                    session_id=f"user-{user_id}",
                ),
            )
        )
```

`Raygent*Options` are explicit wiring groups, not presets. They expose where an
embedding application can pass already-constructed model, tool, context,
permission, memory, persistence, agent/coordinator, and observability services.
Raygent does not silently enable advanced capabilities or choose product policy.

For common product construction paths, use a preset:

```python
from raygent_harness.sdk import create_raygent

session = create_raygent(
    provider=my_model_provider,
    model="demo-model",
    cwd=".",
    preset="project_reader",
)
```

Presets are documented compositions over the same factory options. Inspect them
before use when policy matters:

```python
from raygent_harness.sdk import describe_raygent_preset, resolve_raygent_preset

description = describe_raygent_preset("repo_maintainer")
resolved = resolve_raygent_preset("repo_maintainer")
```

Supported presets are:

- `minimal`
- `chat`
- `embedded_app`
- `project_reader`
- `code_review`
- `repo_maintainer`
- `research_agent`
- `memory_agent`
- `long_running_task`
- `full_developer`

Use `RaygentPresetOptions` for local storage roots and explicit safety
acknowledgements. `memory_agent` requires caller-provided `RaygentMemoryOptions`.
`repo_maintainer` installs project file tools, so it requires filesystem
mutation acknowledgement plus an explicit permission surface. `full_developer`
requires explicit permission options and acknowledgements for broad filesystem,
shell, agent, MCP, and worktree authority.

Low-level factory profiles are explicit:

- `tools="none"` installs no SDK-owned tools.
- `tools="file"` installs file `Read`/`Write`/`Edit`/`NotebookEdit`.
- `tools="project"` adds local discovery, `TaskStop`, and `ToolSearch`; Bash is
  still opt-in through `RaygentToolProfileOptions(enable_bash=True)`.
- `context="environment"` adds bounded environment facts.
- `context="project"` adds environment, git status, and project instructions.

`RaygentSession.run(...)` yields the existing `QueryEngine` SDK message stream.
`run_until_result(...)` returns the terminal `SDKResult`. `RaygentRunCallbacks`
and `add_kernel_event_callback(...)` provide adapter hooks without replacing the
SDK stream or adding a product event schema.

See:

- `examples/minimal_query.py`: minimal factory session.
- `examples/project_profile.py`: conservative project profile.
- `examples/reusable_factory.py`: reusable factory and product-wrapper pattern.
- `examples/sdk_callbacks.py`: callback and kernel-event handling.
- `recipes/create_raygent/`: copyable preset construction recipes.

## Low-Level Runtime Pieces

Advanced embedders can bypass the factory and assemble the kernel directly. A
turn runs through these objects:

- `QueryConfig`: immutable per-turn configuration.
- `QueryDeps`: injectable dependencies and optional services.
- `ToolUseContext`: session/runtime context passed to tools and child loops.
- `QueryEngine`: conversation container and public turn API.
- `ModelProvider`: provider-neutral model backend protocol.

The low-level construction flow is:

1. Build a `ModelProvider` implementation.
2. Build optional tool catalog, context providers, memory services, transcript
   store, observability bus, and permission context.
3. Create `QueryConfig`, `QueryDeps`, and `ToolUseContext`.
4. Create one `QueryEngine` per conversation.
5. Iterate `engine.submit_message(...)` and consume SDK-shaped events until the
   final `SDKResult`.

## Core Imports

Recommended imports for factory-first embedding:

```python
from raygent_harness.sdk import create_raygent
```

Recommended imports for a direct low-level embedding:

```python
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_provider import ModelProvider
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
```

Raygent intentionally keeps top-level exports small. Import from the module that
owns the concept rather than assuming everything is available from
`raygent_harness`.

## Model Provider Boundary

The kernel does not import provider SDKs. A concrete provider adapts Raygent
model types to the provider's transport and back.

Implement `ModelProvider` with:

- `complete(request)` for non-streaming turns;
- `stream(request)` for streaming turns;
- `count_tokens(request)` when exact counts are available;
- `resolve_model(requested, context)` for aliases or model routing;
- `model_info(model)` for context windows, output limits, and capabilities;
- `classify_error(error)` for recovery categories.

Protocol translators under `raygent_harness.adapters.model_protocols` can help
build a provider without putting transport in core:

- `AnthropicMessagesAdapter`
- `OpenAIResponsesAdapter`

These adapters prepare provider-shaped request bodies and parse provider-shaped
stream/error payloads. A live integration still has to send the request and feed
provider events back through the adapter.

For embedders that want to reuse those translators directly,
`raygent_harness.adapters.model_runtime` provides the runtime bridge:

- `ProtocolModelProvider`: wraps a protocol adapter plus an injected transport
  and implements `ModelProvider`;
- `ProviderTransport`: the narrow async interface for complete, stream, and
  token-count calls;
- `ProviderModelCatalog` / `ProviderModelEntry`: optional provider/deployment
  metadata converted into `ModelInfo` rows;
- `ProviderRetryPolicy`: explicit opt-in retry/backoff and pre-yield
  stream-to-complete fallback policy.

The split is intentional:

- protocol adapters own provider wire-shape translation;
- transports own SDK/HTTP/SSE clients, credentials, endpoint selection, and
  account/deployment policy;
- `ProtocolModelProvider` owns provider-neutral runtime mechanics such as
  model lookup, retry classification, abort checks, and stream fallback events;
- the query loop continues to depend only on `ModelProvider`.

See `examples/provider_runtime_bridge.py` for a no-network fake transport wired
through `ProtocolModelProvider` into both direct provider calls and
`QueryEngine`.

## Tools

Tools are `Tool` protocol objects. Prefer `build_tool(ToolSpec(...))` so the
fail-closed defaults are applied consistently.

For text-file tools, use:

```python
from raygent_harness.tools import create_file_tooling_runtime

file_runtime = create_file_tooling_runtime()
tools = file_runtime.tools
pre_hooks = file_runtime.pre_tool_use_hooks
post_hooks = file_runtime.post_tool_use_hooks
```

The full tool catalog for a turn can be supplied directly in `QueryConfig.tools`
or through a `QueryDeps.tool_catalog_provider`.

## Permissions

Permissions are part of `QueryDeps`, not global process state.

Useful modules:

- `raygent_harness.core.permissions`
- `raygent_harness.core.permission_engine`
- `raygent_harness.core.tool_hooks`

For non-interactive examples, use a tool with explicit `check_permissions` that
returns `PermissionAllowDecision`, or configure an appropriate permission
context and engine policy. Do not rely on missing permission axes being safe;
Raygent defaults tools to fail closed.

## Context Providers

Context providers add non-persistent model-visible context at turn entry. They
are opt-in.

```python
from raygent_harness.context_providers.defaults import build_default_context_providers

context_providers = build_default_context_providers(
    cwd=".",
    include_environment=True,
    include_git_status=True,
    include_project_instructions=True,
)
```

Use this when the embedding application wants environment facts, git status, or
project instructions to reach the model without becoming ordinary conversation
history.

## Memory

Memory is optional. The core integration points are:

- `QueryDeps.memory_prompt_provider`
- `QueryDeps.memory_recall_provider`
- `QueryDeps.memory_extractor`

Useful packages:

- `raygent_harness.memdir`
- `raygent_harness.services.extract_memories`
- `raygent_harness.services.team_memory_sync`

For a simple embedding, start without memory. Add memory only after the model
provider and tool permission story are stable.

## Transcripts And Replay

Use `raygent_harness.services.transcript` for JSONL transcript storage, replay,
sidechain transcript loading, and content replacement records.

`QueryEngine` can be constructed normally for a new conversation or with
`QueryEngine.from_replay(...)` for replayed state.

## Observability

Raygent has a provider-neutral event bus under
`raygent_harness.core.observability`.

Use it for debugging, evals, and adapter integration. Default event payloads are
metadata-oriented and should not be treated as raw prompt/code/file capture.
If an embedding needs raw trace capture, design that as an explicit retention
policy outside the default kernel path.

## Kernel vs Adapter Boundary

Keep these in kernel-level code:

- model-visible transcript/state continuity;
- tool execution and permission semantics;
- task lifecycle, cancellation, cleanup, and notifications;
- compaction, recovery, memory, transcript replay, and observability facts;
- provider-neutral request/response/error shapes.

Keep these in adapters or applications:

- provider SDK clients and credentials;
- product-specific SDK/progress schemas;
- UI panes, terminal panes, and rendering;
- hosted remote backend/auth/archive policy;
- product telemetry dashboards;
- concrete deployment configuration.

## Suggested First Embedding

1. Implement a small `ModelProvider` around your chosen model API.
2. Run `examples/minimal_query.py` and mirror its factory construction pattern.
3. Run `examples/reusable_factory.py` if your application needs a reusable
   factory or product-owned construction wrapper.
4. Use `tools="project"` / `context="project"` only after safe minimal turns are
   working.
5. If your provider uses an Anthropic Messages-shaped or OpenAI
   Responses-shaped API, run `examples/provider_runtime_bridge.py` and replace
   only the example transport with your SDK/HTTP/SSE client.
6. Add custom tools or catalog providers.
7. Add transcript storage.
8. Add observability sinks or SDK callbacks.
9. Add memory only after the basic turn/tool flow is stable.
