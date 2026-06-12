from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from raygent_harness.goals import (
    GoalCheckpoint,
    GoalEventEmitter,
    GoalSpec,
    GoalStateConflictError,
    InMemoryGoalEventSink,
    InMemoryGoalStore,
    JsonGoalStore,
    create_goal_state,
    safe_goal_component,
)


def test_in_memory_goal_store_enforces_single_active_goal_per_session() -> None:
    store = InMemoryGoalStore()
    first = create_goal_state(
        goal_id="g_1",
        session_id="s",
        spec=GoalSpec(objective="first"),
        now=1.0,
    )
    second = create_goal_state(
        goal_id="g_2",
        session_id="s",
        spec=GoalSpec(objective="second"),
        now=2.0,
    )

    store.create(first)

    with pytest.raises(GoalStateConflictError, match="active goal"):
        store.create(second)


def test_in_memory_goal_store_records_redacted_transition_events() -> None:
    sink = InMemoryGoalEventSink()
    emitter = GoalEventEmitter(sinks=(sink,), clock=lambda: 123.0)
    store = InMemoryGoalStore(event_emitter=emitter)
    state = create_goal_state(
        goal_id="g_1",
        session_id="s",
        spec=GoalSpec(objective="secret objective text"),
        now=1.0,
    )

    store.create(state)
    updated = store.update(
        "g_1",
        lambda current: current.with_status(
            "complete",
            reason="verified by tests",
            now=2.0,
        ),
    )

    events = store.list_events("g_1")
    assert updated.status == "complete"
    assert [event.type for event in events] == ["goal_created", "goal_completed"]
    assert [event.type for event in sink.events] == ["goal_created", "goal_completed"]
    assert events[0].data["objective_char_count"] == len("secret objective text")
    assert "secret objective text" not in str(events[0].data)


def test_json_goal_store_round_trips_state_and_events(tmp_path: Path) -> None:
    sink = InMemoryGoalEventSink()
    emitter = GoalEventEmitter(sinks=(sink,), clock=lambda: 123.0)
    store = JsonGoalStore(tmp_path, event_emitter=emitter)
    state = replace(
        create_goal_state(
            goal_id="g_1",
            session_id="s",
            spec=GoalSpec(objective="durable goal"),
            now=1.0,
        ),
        summary="persistent summary",
        checkpoints=(
            GoalCheckpoint(
                checkpoint_id="cp_1",
                summary="checkpoint summary",
                created_at=2.0,
            ),
        ),
    )

    store.create(state)
    store.update(
        "g_1",
        lambda current: current.with_accounting(
            turn_delta=2,
            token_delta=30,
            time_delta_s=4.0,
            now=5.0,
        ),
    )

    reloaded = JsonGoalStore(tmp_path)
    restored = reloaded.get("g_1")

    assert restored is not None
    assert restored.summary == "persistent summary"
    assert restored.checkpoints[0].checkpoint_id == "cp_1"
    assert restored.turn_count == 2
    assert reloaded.get_active_for_session("s") == restored
    assert reloaded.list_for_session("s") == (restored,)
    assert [event.type for event in reloaded.list_events("g_1")] == [
        "goal_created",
        "goal_updated",
    ]
    assert [event.type for event in sink.events] == ["goal_created", "goal_updated"]


def test_json_goal_store_skips_corrupt_event_records(tmp_path: Path) -> None:
    sink = InMemoryGoalEventSink()
    emitter = GoalEventEmitter(sinks=(sink,), clock=lambda: 123.0)
    store = JsonGoalStore(tmp_path, event_emitter=emitter)
    store.create(
        create_goal_state(
            goal_id="g_1",
            session_id="s",
            spec=GoalSpec(objective="durable goal"),
            now=1.0,
        )
    )
    with store.events_path("s").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "bad_type",
                    "type": "not_real",
                    "sequence": 2,
                    "goal_id": "g_1",
                    "session_id": "s",
                    "status": "active",
                    "created_at": 124.0,
                    "data": {},
                    "version": 1,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "id": "bad_status",
                    "type": "goal_updated",
                    "sequence": 3,
                    "goal_id": "g_1",
                    "session_id": "s",
                    "status": "not_real",
                    "created_at": 125.0,
                    "data": {},
                    "version": 1,
                }
            )
            + "\n"
        )
        handle.write("{\"type\":\"not_real\"}\n")
        handle.write("not-json\n")

    events = JsonGoalStore(tmp_path).list_events("g_1")

    assert [event.type for event in events] == ["goal_created"]


def test_json_goal_store_enforces_single_active_goal_after_reload(
    tmp_path: Path,
) -> None:
    store = JsonGoalStore(tmp_path)
    store.create(
        create_goal_state(
            goal_id="g_1",
            session_id="s",
            spec=GoalSpec(objective="first"),
            now=1.0,
        )
    )

    reloaded = JsonGoalStore(tmp_path)

    with pytest.raises(GoalStateConflictError, match="active goal"):
        reloaded.create(
            create_goal_state(
                goal_id="g_2",
                session_id="s",
                spec=GoalSpec(objective="second"),
                now=2.0,
            )
        )


def test_json_goal_store_moves_state_sidecar_after_session_change(
    tmp_path: Path,
) -> None:
    store = JsonGoalStore(tmp_path)
    state = create_goal_state(
        goal_id="g_1",
        session_id="s_1",
        spec=GoalSpec(objective="move session"),
        now=1.0,
    )
    store.create(state)
    old_path = store.state_path("s_1", "g_1")

    updated = store.update(
        "g_1",
        lambda current: replace(current, session_id="s_2", updated_at=2.0),
    )

    assert updated.session_id == "s_2"
    assert not old_path.exists()
    assert JsonGoalStore(tmp_path).get("g_1") == updated
    assert JsonGoalStore(tmp_path).list_for_session("s_1") == ()
    assert JsonGoalStore(tmp_path).list_for_session("s_2") == (updated,)


def test_json_goal_store_skips_corrupt_state_sidecars(tmp_path: Path) -> None:
    store = JsonGoalStore(tmp_path)
    state = create_goal_state(
        goal_id="g_good",
        session_id="s",
        spec=GoalSpec(objective="valid"),
        now=1.0,
    )
    store.create(state)
    bad_path = store.state_dir("s") / "bad.json"
    bad_path.write_text("{not-json", encoding="utf-8")
    bad_status_path = store.state_dir("s") / "bad-status.json"
    raw_state = json.loads(store.state_path("s", "g_good").read_text(encoding="utf-8"))
    raw_state["goal_id"] = "g_bad_status"
    raw_state["status"] = "not_real"
    bad_status_path.write_text(json.dumps(raw_state), encoding="utf-8")
    mismatch_path = store.state_dir("s") / "mismatch.json"
    mismatch = replace(state, goal_id="g_mismatch", session_id="other")
    mismatch_path.write_text(
        json.dumps({"goal_id": mismatch.goal_id, "session_id": mismatch.session_id}),
        encoding="utf-8",
    )

    reloaded = JsonGoalStore(tmp_path)

    assert reloaded.list_for_session("s") == (state,)


def test_safe_goal_component_blocks_path_escape() -> None:
    assert "/" not in safe_goal_component("../session/id")
    assert safe_goal_component("...") != ""
    assert safe_goal_component("a/b") != safe_goal_component("a_b")
    assert len(safe_goal_component("x" * 5000)) < 128
