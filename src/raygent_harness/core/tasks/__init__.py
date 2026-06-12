"""Task registry — eager-loads each task type so `register_task_impl`
side-effects fire on package import.

sufficient to populate the registry.

Adding a new task type requires one line here.
"""

from __future__ import annotations

# Side-effect imports: each module's `register_task_impl(...)` call at
# module-init populates `core.task._REGISTRY` so `get_task_by_type(...)`
# returns the impl for that type. Don't remove unless you also remove
# the corresponding task type from `TaskType`.
from raygent_harness.core.tasks import (
    in_process_teammate as _in_process_teammate,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from raygent_harness.core.tasks import (
    local_agent as _local_agent,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from raygent_harness.core.tasks import (
    local_bash as _local_bash,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from raygent_harness.core.tasks import (
    remote_agent as _remote_agent,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

__all__: list[str] = []
