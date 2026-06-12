"""services — non-core subsystems that plug into the agent loop.

Each subpackage owns policy + data structures for one concern (compaction,
memory, etc.). Layers are wired into the loop via `QueryDeps`; data shapes
referenced by `core` types live in `core/state.py` to avoid cycles.
"""
