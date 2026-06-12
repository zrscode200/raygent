# Public API Surface

Status: pre-1.0 stabilization note.

Raygent's public API is not frozen yet. This document describes the intended
module-level import surface so embedders can use the harness
without depending on incidental internals.

## Stability Tiers

- `recommended`: use these modules for normal embedding work.
- `advanced`: valid extension points, but expect to read the module docs and
  tests before using them.
- `experimental/internal`: available in source, but not a stable embedding
  contract yet.

## Recommended Kernel Surface

Embedding factory:

- `raygent_harness.sdk`
  - `create_raygent(...)`
  - `RaygentFactory`
  - `RaygentSession`
  - `RaygentSessionFactory`
  - `RaygentFactoryConfig`
  - `RaygentModelOptions`
  - `RaygentSessionOptions`
  - `RaygentToolOptions`
  - `RaygentContextOptions`
  - `RaygentPermissionOptions`
  - `RaygentMemoryOptions`
  - `RaygentPersistenceOptions`
  - `RaygentAgentOptions`
  - `RaygentObservabilityOptions`
  - `RaygentPreset`
  - `RaygentOverlay`
  - `RaygentPresetOptions`
  - `RaygentPresetDescription`
  - `RaygentPresetResolution`
  - `RaygentPresetCompatibilityError`
  - `RaygentCallbackHandle`
  - `RaygentKernelEventCallback`
  - `RaygentKernelEventCallbackSink`
  - `RaygentRuntimeHandles`
  - `RaygentRunCallbacks`
  - `RaygentSDKError`
  - `RaygentSDKMessageCallback`
  - `RaygentSDKProtocolError`
  - `RaygentSDKResultCallback`
  - `RaygentSessionBusyError`
  - `RaygentSessionClosedError`
  - `RaygentToolProfile`
  - `RaygentToolProfileOptions`
  - `RaygentToolSelection`
  - `RaygentContextProfile`
  - `RaygentContextProfileOptions`
  - `RaygentContextSelection`
  - `list_raygent_presets(...)`
  - `describe_raygent_preset(...)`
  - `resolve_raygent_preset(...)`

`raygent_harness.sdk` is the recommended ergonomic entrypoint for headless
embedding. It assembles the same kernel primitives listed below and returns a
conversation-scoped `RaygentSession`. The package-level `raygent_harness`
namespace remains intentionally minimal; import the factory from
`raygent_harness.sdk`.

`create_raygent(...)` is the one-shot compatibility wrapper. `RaygentFactory`
is the reusable concrete assembly object for applications that create multiple
sessions from explicit `RaygentFactoryConfig` values. `RaygentSessionFactory`
is a narrow structural Protocol for product/application code that wants to
depend on a factory seam without subclassing Raygent internals.

`Raygent*Options` groups are explicit capability wiring surfaces. Product
presets are a convenience layer over those groups, not a replacement for them.
They expose existing kernel seams for model/provider configuration, session
identity, tools/hooks/media services, context providers, permissions, memory,
persistence, agents/coordinator/remotes, and observability. Products decide
which concrete services to inject; the SDK does not silently enable advanced
behaviors.

`create_raygent(..., preset=..., overlays=...)` is available for common
construction paths. Presets resolve to ordinary factory options and can be
inspected before use:

```python
from raygent_harness.sdk import describe_raygent_preset, resolve_raygent_preset

description = describe_raygent_preset("project_reader")
resolved = resolve_raygent_preset("project_reader")
```

Preset construction API:

- `preset`: one named composition from `RaygentPreset`. It may be supplied to
  `create_raygent(...)` or stored on `RaygentFactoryConfig`.
- `overlays`: optional additive composition flags from `RaygentOverlay`.
  Overlays are only valid when `preset` is also set.
- `preset_options`: a `RaygentPresetOptions` value for local roots, services,
  and explicit safety acknowledgements.
- `resolve_raygent_preset(...)`: returns `RaygentPresetResolution`, including
  ordinary `factory_kwargs`, `enabled_capabilities`, `required_options`,
  `safety_notes`, `tool_names`, `context_profile`, `enable_bash`, and whether
  explicit permission options are required.
- `describe_raygent_preset(...)`: returns `RaygentPresetDescription` metadata
  for documentation, UI display, or preflight checks.
- `list_raygent_presets(...)`: returns the supported preset names in
  documentation order.

Preset defaults fill only unset factory surfaces. Flat `create_raygent(...)`
arguments still take precedence over option groups, and explicit option groups
remain escape hatches over preset defaults:

- `RaygentToolOptions.tools` overrides preset tool defaults.
- `RaygentContextOptions.context` overrides preset context defaults.
- `RaygentPersistenceOptions.transcript_store` overrides preset transcript
  store defaults.
- `RaygentPersistenceOptions.task_output_dir` overrides preset task-output
  defaults.

When a preset creates local transcript storage and no `preset_options.project_root`
or `preset_options.transcript_dir` is supplied, the SDK derives the root from
the session cwd: flat `cwd` first, then `RaygentSessionOptions.cwd`, then `"."`.
Callers that need product-owned storage policy should pass an explicit
`RaygentPersistenceOptions.transcript_store` or `RaygentPresetOptions`.

Required preset names:

- `minimal`: model/provider only; no SDK-owned tools, context, memory, or
  persistence.
- `chat`: conversation session with local JSONL transcript support.
- `embedded_app`: application embedding profile with transcripts and
  observability, but no filesystem tools.
- `project_reader`: project context with `Read`, `Glob`, `Grep`, and
  `ToolSearch` only.
- `code_review`: read/search-oriented repository review profile with mutation
  disabled by default.
- `repo_maintainer`: project workflow profile with persistence and recovery-
  oriented state. It installs project file tools, so it requires
  `RaygentPresetOptions(allow_filesystem_mutation=True)` plus an explicit
  permission surface.
- `research_agent`: search/MCP-oriented profile with low filesystem authority;
  concrete MCP/web/search services remain caller-provided.
- `memory_agent`: memory-ready profile that requires caller-provided
  `RaygentMemoryOptions`.
- `long_running_task`: durable long-session profile with transcripts, task
  output, compaction, and recovery orientation.
- `full_developer`: broad developer bundle that fails fast unless explicit
  safety acknowledgements and permission options are supplied.

Supported overlays are `transcripts`, `observability`, `memory`, `goals`,
`compaction`, `recovery`, `task_output`, `readonly_tools`, `file_tools`,
`bash`, `agents`, `coordinator`, `mcp`, and `worktree`. Mutating/open-world
overlays require explicit acknowledgement through `RaygentPresetOptions` and an
explicit permission surface; they are not enabled silently.

`RaygentSession` exposes existing kernel streams and handles rather than a
separate runtime loop:

- `run(...)` / `submit_message(...)`: yield existing `QueryEngine` SDK messages.
  Both accept optional `RaygentRunCallbacks` for adapter-side `on_message`,
  `on_result`, and run-scoped `on_kernel_event` hooks.
- `run_until_result(...)`: consume one turn and return the terminal `SDKResult`.
- `add_kernel_event_callback(...)`: attach a persistent callback to the
  session's existing `KernelEventBus`, returning a detachable
  `RaygentCallbackHandle`.
- `abort()`: cooperatively signal the session abort event.
- `close()`: idempotently signal abort, drain scheduled memory extraction tasks
  when possible, flush optional transcript persistence, mark the session closed,
  and reject future turns. If a turn is active, it signals abort and raises
  `RaygentSessionBusyError`; the caller must finish or `aclose()` the active
  run stream before close is a persistence boundary. It does not kill unrelated
  tasks in the shared task store.
- `handles`: a `RaygentRuntimeHandles` object exposing session id, cwd, task
  store, session-scoped task-output store/path, optional transcript store,
  transcript scope/path when available, observability bus, and abort event.

Callback helpers are adapter conveniences, not a product event schema:

- SDK messages remain the primary run stream.
- `RaygentRunCallbacks.on_message` sees every yielded `SDKMessage`;
  `on_result` sees only terminal `SDKResult` messages.
- `on_message` and `on_result` exceptions intentionally propagate to the caller
  and close the wrapped kernel stream before the session becomes available for
  another turn.
- `on_kernel_event` uses the existing session event bus for the duration of one
  run and is detached when the run generator closes.
- `RaygentKernelEventCallbackSink` is synchronous because `KernelEventBus`
  sinks are synchronous. Sink failures are captured by the bus and never mutate
  query/tool/task semantics.
- There is no global event bus, late-attachment backlog, product telemetry
  dashboard, raw-content telemetry default, or product-specific mutation hook.

Low-level SDK tool/context profiles remain explicit and conservative:

- `tools="none"`: no SDK-owned tools.
- `tools="file"`: `Read`, `Write`, `Edit`, and `NotebookEdit`, with file-tool
  hooks kept attached to `QueryDeps`.
- `tools="project"`: file tools plus `Glob`, `Grep`, `TaskStop`, and
  `ToolSearch`. Restricted `Bash` is added only with
  `RaygentToolProfileOptions(enable_bash=True)`.
- `context="environment"`: bounded environment facts only.
- `context="project"`: environment, git status, and project-instruction
  providers. Instruction filenames/rule dirs are configurable through
  `RaygentContextProfileOptions`.

Advanced Agent, Skill, MCP, remote, team, SendMessage, coordinator, memory, and
shell behaviors remain explicit opt-ins through lower-level providers/options;
the SDK factory does not install them silently.

Copyable construction examples live outside the runtime package under
`recipes/create_raygent/`. Runtime library behavior remains under
`src/raygent_harness/`.

Conversation and loop:

- `raygent_harness.core.query_engine`
  - `QueryEngine`
  - `SDKSystemInit`
  - `SDKAssistantMessage`
  - `SDKUserMessage`
  - `SDKCompactBoundary`
  - `SDKResult`
- `raygent_harness.core.config`
  - `QueryConfig`
  - `SamplingParams`
  - `TurnBudget`
- `raygent_harness.core.deps`
  - `QueryDeps`
- `raygent_harness.core.tool`
  - `Tool`
  - `ToolSpec`
  - `ToolUseContext`
  - `ToolResult`
  - `ToolProgress`
  - `ToolCallError`
  - `QueryTracking`
  - `build_tool`
- `raygent_harness.core.task`
  - `AppStateStore`
  - task notification and task state primitives

Model/provider boundary:

- `raygent_harness.core.model_provider`
  - `ModelProvider`
  - `UnavailableModelProvider`
  - `classify_exception_by_name`
- `raygent_harness.core.model_types`
  - provider-neutral model request, response, content, stream, usage, token, and
    error dataclasses
  - `TokenCountRequest`
  - `TokenCountResult`
- `raygent_harness.core.messages`
  - `MessageParam`
  - `user_message(...)`
  - `assistant_message(...)`
  - conversion helpers between transcript-shaped messages and model responses
- `raygent_harness.core.context_providers`
  - `ContextFragment`
  - `ContextKind`
  - `ContextProvider`
  - `PostToolContextProvider`
  - `context_provider_kind(...)`
  - `filter_context_providers_by_kind(...)`
  - `render_system_context(...)`
  - `render_user_context_messages(...)`

Permissions and hooks:

- `raygent_harness.core.permissions`
- `raygent_harness.core.permission_engine`
- `raygent_harness.core.tool_hooks`
- `raygent_harness.core.stop_hooks`

Observability:

- `raygent_harness.core.observability`

## Recommended Optional Packages

Tools:

- `raygent_harness.tools`
  - file tools, local discovery tools (`Glob`/`Grep`), restricted local
    `Bash`, MCP tool adapters, ToolSearch, SkillTool, AgentTool, TeamCreate,
    SendMessage, TaskStop, backend protocols, and catalog-provider builders
  - `NOTEBOOK_EDIT_TOOL_NAME`
  - `NOTEBOOK_EDIT_PROMPT`
  - `NOTEBOOK_EDIT_MAX_RESULT_SIZE_CHARS`
  - `NotebookEditInput`
  - `NotebookEditResult`
  - `NotebookEditToolError`
  - `build_notebook_edit_tool(...)`
  - `create_notebook_edit_catalog_provider(...)`
  - `parse_notebook_cell_id(...)`

Context providers:

- `raygent_harness.context_providers.defaults`
  - `build_default_context_providers(...)`
- `raygent_harness.context_providers.environment`
  - `EnvironmentContextProvider`
  - `GitStatusContextProvider`
  - `GitCommandResult`
  - `GitCommandRunner`
  - `default_git_command_runner(...)`
- `raygent_harness.context_providers.project_instructions`
  - `ProjectInstructionConfig`
  - `ProjectInstructionFile`
  - `ProjectInstructionsContextProvider`
  - `ReadAdjacentProjectInstructionsContextProvider`
  - `ConditionalInstructionRule`
  - `discover_project_instruction_files(...)`
  - `discover_conditional_instruction_rules(...)`
  - `instruction_rule_matches_path(...)`
  - `resolve_project_instructions_for_target_path(...)`
- `raygent_harness.context_providers.transcript_search`
  - `TranscriptSearchContextProvider`
  - `TranscriptSearchQueryResolver`

Memory:

- `raygent_harness.memdir`
- `raygent_harness.services.extract_memories`
  - scheduler, restricted child extraction runner, and default concrete
    extraction tool-catalog helper
- `raygent_harness.services.team_memory_sync`

Transcript and output storage:

- `raygent_harness.services.transcript`
  - transcript entry, JSONL store, replay, and bounded search primitives
  - `TranscriptSearchCompactMode`
  - `TranscriptSearchMatch`
  - `TranscriptSearchOrder`
  - `TranscriptSearchRequest`
  - `TranscriptSearchResult`
  - `TranscriptSearchScope`
  - `TranscriptSearchService`
- `raygent_harness.services.task_output`

Bounded improvement proposals and gates:

- `raygent_harness.improvement`
  - proposal-only, non-mutating improvement records and service primitives
  - gate/evaluation records are supplied-result policy checks, not execution
  - optional model-backed proposal generator uses a tool-free
    `ModelProvider.complete(...)` request with `ModelRequest.tools == ()`
  - `DEFAULT_IMPROVEMENT_MODEL_GENERATOR_SYSTEM_PROMPT`
  - `ImprovementTarget`
  - `ImprovementTargetKind`
  - `ImprovementEvidence`
  - `ImprovementEvidenceSource`
  - `ImprovementEvidenceBounds`
  - `ImprovementEvidenceValidationError`
  - `BoundedImprovementEvidence`
  - `ImprovementGateKind`
  - `ImprovementGateStatus`
  - `ImprovementGateDecision`
  - `ImprovementGateResult`
  - `ImprovementGateEvaluation`
  - `ImprovementGatePolicy`
  - `ImprovementGateError`
  - `ImprovementGateValidationError`
  - `ImprovementModelGenerator`
  - `ImprovementModelGeneratorError`
  - `ImprovementPatchCandidateStatus`
  - `ImprovementPatchCandidatePlan`
  - `ImprovementPatchCandidatePlanner`
  - `ImprovementPatchCandidateError`
  - `ImprovementPatchCandidateValidationError`
  - `ImprovementDiagnosis`
  - `ImprovementEvaluationCheck`
  - `ImprovementEvaluationPlan`
  - `ImprovementProposal`
  - `ImprovementRequiredPermission`
  - `ImprovementProposalRequest`
  - `ImprovementProposalGenerator`
  - `ImprovementService`
  - `ImprovementServiceError`
  - `ImprovementValidationError`
  - `ImprovementRun`
  - `ImprovementRunStatus`
  - `validate_bounded_improvement_evidence(...)`
  - `improvement_evidence_text_chars(...)`
  - `improvement_target_to_dict(...)`
  - `improvement_target_from_dict(...)`
  - `improvement_evidence_to_dict(...)`
  - `improvement_evidence_from_dict(...)`
  - `improvement_gate_result_to_dict(...)`
  - `improvement_gate_result_from_dict(...)`
  - `improvement_gate_evaluation_to_dict(...)`
  - `improvement_gate_evaluation_from_dict(...)`
  - `improvement_patch_candidate_plan_to_dict(...)`
  - `improvement_patch_candidate_plan_from_dict(...)`
  - `improvement_diagnosis_to_dict(...)`
  - `improvement_diagnosis_from_dict(...)`
  - `improvement_evaluation_check_to_dict(...)`
  - `improvement_evaluation_check_from_dict(...)`
  - `improvement_evaluation_plan_to_dict(...)`
  - `improvement_evaluation_plan_from_dict(...)`
  - `improvement_proposal_to_dict(...)`
  - `improvement_proposal_from_dict(...)`
  - `improvement_run_to_dict(...)`
  - `improvement_run_from_dict(...)`

The improvement package is an RSI-001/RSI-002A/RSI-002B/RSI-003A contract surface. It
produces structured proposal records from bounded evidence, can derive
reviewable gate decisions from caller-supplied gate results, and can optionally
ask an injected model provider for proposal JSON through `ImprovementModelGenerator`.
The records and gate layer does not mutate files, create worktrees,
execute shell commands, call models, request permissions, commit, promote candidates,
train models, or parse product `/goal` commands. `ImprovementService` validates
bounded proposal data and may invoke an injected generator; with the optional
model generator, that invocation is model-call only: it uses one non-streaming
`ModelProvider.complete(...)` request with `ModelRequest.tools == ()`, then
returns data for `ImprovementService` validation. It also includes data-only
patch candidate plans for reviewed proposals. `ImprovementPatchCandidatePlan`
stores data-only patch candidate plans. The candidate status is `planned` only.
Candidate records are not authorization grants, and they do not allocate
worktrees or materialize patches. Later patching, archive, and product
orchestration layers should compose around these records rather than weakening
this proposal-only, model-call-only, supplied-result gate, and data-only
candidate boundary.

Worktrees and remote-agent seam:

- `raygent_harness.services.worktree`
- `raygent_harness.services.remote_agent`
- `raygent_harness.services.agent_routes`
  - `AgentRouteRecordStore`
  - `AgentRouteRecordLoadResult`
  - `JsonAgentRouteRecordStore`
  - `normalize_agent_route_record_for_resume(...)`
- `raygent_harness.services.mcp`
  - provider-neutral MCP identity, tool-name, server-state, client request /
    call-context / result seam, and in-memory test-client models

Provider protocol translators:

- `raygent_harness.adapters.model_protocols`
  - `AnthropicMessagesAdapter`
  - `OpenAIResponsesAdapter`

These translators are transport-free. They are useful when implementing a live
provider adapter, but they are not live clients.

Provider runtime bridge:

- `raygent_harness.adapters.model_runtime`
  - `ProviderModelCatalog`
  - `ProviderModelEntry`
  - `ProtocolModelProvider`
  - `ProviderPayloadError`
  - `ProviderRetryDecision`
  - `ProviderRetryPolicy`
  - `ProviderTransport`
  - `ProviderTransportRequest`
  - `RetryOperation`
  - `capabilities_from_modalities`
  - `classify_retry_decision`
  - `merge_model_info`
  - `merge_model_infos`
  - `registry_from_catalogs`
  - `should_fallback_stream_to_complete`

This bridge turns a protocol translator plus an injected transport into a
concrete `ModelProvider`. It remains no-SDK and no-network by default; real
clients, credentials, auth, and endpoint policy belong to the embedding
application or an external adapter package. Catalog helpers convert
provider/deployment metadata into Raygent-owned `ModelInfo` rows consumed by the
existing `ModelRegistry`. Retry policy is explicit and opt-in:
`ProviderRetryPolicy()` preserves no-retry behavior, while embedders can enable
bounded retries, rate-limit retry-after handling, stream transport retries, or
pre-yield stream-to-complete fallback at the runtime bridge.

Transport and provider result metadata fields named `safe_metadata` are a
contract with adapter authors: Raygent freezes and forwards them, but does not
scrub provider-specific secrets or prompt content from those fields. Live
adapters must only populate metadata that is safe for logs, replay diagnostics,
and observability.

## Advanced Surface

Use these when building nontrivial agent runtimes:

- `raygent_harness.core.child_query`
- `raygent_harness.core.tool_execution`
- `raygent_harness.core.tool_orchestration`
- `raygent_harness.core.streaming_tool_executor`
- `raygent_harness.coordinator`
  - `CoordinatorRuntime`
  - `CoordinatorRuntimeConfig`
  - `CoordinatorRuntimeSnapshot`
  - `CoordinatorRuntimeSnapshotStore`
  - `CoordinatorRuntimeSnapshotLoadResult`
  - `JsonCoordinatorRuntimeSnapshotStore`
  - `coordinator_runtime_snapshot_to_dict(...)`
  - `coordinator_runtime_snapshot_from_dict(...)`
- `raygent_harness.agents`
  - `AgentContextPolicy`
  - `AgentDefinition`
  - `deps_for_agent_context_policy(...)`
- `raygent_harness.skills`
- `raygent_harness.services.compact`
- `raygent_harness.services.handoff`
- `raygent_harness.services.runtime_recovery`
  - `RuntimeRecoveryRequest`
  - `RuntimeRecoveryResult`
  - `RuntimeRecoveryService`
  - `RuntimeRecoveryTaskOutputStatus`
  - `RuntimeRecoveryWarning`
  - `RuntimeRecoveryWarningSource`
  - `RuntimeRecoveryWorktreeStatus`
  - `resume_runtime_session(...)`
- `raygent_harness.services.file_media`
  - `PDF_MAX_EXTRACT_SIZE_BYTES`
  - `PDF_PAGE_RENDER_DPI`
  - `CommandBackedPdfDocumentService`
  - `CommandResult`
  - `CommandRunner`
  - `FileMediaClassification`
  - `NOTEBOOK_LARGE_OUTPUT_THRESHOLD_CHARS`
  - `NOTEBOOK_MAX_PROCESSED_BYTES`
  - `NOTEBOOK_MAX_RAW_BYTES`
  - `NOTEBOOK_OUTPUT_TEXT_MAX_CHARS`
  - `NotebookCellOutput`
  - `NotebookCellSource`
  - `NotebookOutputImage`
  - `NotebookParseResult`
  - `NotebookServiceError`
  - `NotebookServiceErrorReason`
  - `PdfDocumentService`
  - `PdfExtractedPage`
  - `PdfPageCountResult`
  - `PdfPageExtractionRequest`
  - `PdfPageExtractionResult`
  - `PdfServiceError`
  - `PdfServiceErrorReason`
  - `SubprocessCommandRunner`
  - `classify_file_extension(...)`
  - `classify_file_media(...)`
  - `classify_magic_bytes(...)`
  - `detect_supported_image_media_type(...)`
  - `extension_for_path(...)`
  - `is_blocked_device_path(...)`
  - `notebook_cells_to_content(...)`
  - `notebook_cells_to_json(...)`
  - `parse_notebook_content(...)`
- `raygent_harness.services.task_notification_replay`
  - `TaskNotificationReplayCoordinator`
  - `TaskNotificationReplayRecord`
  - `TaskNotificationReplayResult`
  - `TaskNotificationReplaySource`
  - `remote_agent_restore_replay_record(...)`
  - `remote_agent_terminal_dedupe_key(...)`
  - `replay_task_notifications(...)`

## Experimental Or Internal Surface

Avoid depending on these directly unless you are extending Raygent itself:

- private helpers prefixed with `_`;
- concrete task driver registries;
- implementation-specific dataclass fields used only for tests or replay
  internals;
- adapter fixture harnesses under `tests/`.

## Top-Level Package

`raygent_harness.__init__` currently exports only package metadata. This is
intentional for pre-1.0: top-level re-exports would create an accidental API
freeze before the embedding story has been dogfooded.

Use module-level imports for now. If import ergonomics become painful, add a
small curated top-level export set in a dedicated API-hardening pass.
