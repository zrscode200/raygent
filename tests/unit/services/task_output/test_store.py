from __future__ import annotations

import os
from pathlib import Path

import pytest

from raygent_harness.services.task_output import FileTaskOutputStore
from raygent_harness.services.task_output.store import read_task_output_file_tail


@pytest.mark.asyncio
async def test_file_task_output_store_reads_tail_and_ranges(tmp_path: Path) -> None:
    store = FileTaskOutputStore(tmp_path, session_id="s")
    ref = await store.init_task_output("b1")

    await store.append_task_output("b1", b"abcdef")
    await store.append_task_output("b1", b"ghij")
    await store.flush_task_output("b1")

    assert ref.path is not None
    assert Path(ref.path).read_bytes() == b"abcdefghij"

    tail = await store.read_tail("b1", max_bytes=4)
    assert tail.content == b"ghij"
    assert tail.start_offset == 6
    assert tail.bytes_total == 10
    assert tail.truncated_before is True
    assert tail.truncated_after is False

    middle = await store.read_range("b1", offset=2, max_bytes=3)
    assert middle.content == b"cde"
    assert middle.start_offset == 2
    assert middle.next_offset == 5
    assert middle.truncated_before is True
    assert middle.truncated_after is True


@pytest.mark.asyncio
async def test_file_task_output_store_rejects_preexisting_path(tmp_path: Path) -> None:
    store = FileTaskOutputStore(tmp_path, session_id="s")
    path = store.path_for_task("b1")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"do not overwrite")

    with pytest.raises(FileExistsError):
        await store.init_task_output("b1")

    assert path.read_bytes() == b"do not overwrite"


@pytest.mark.asyncio
async def test_file_task_output_store_does_not_follow_symlink(tmp_path: Path) -> None:
    store = FileTaskOutputStore(tmp_path, session_id="s")
    path = store.path_for_task("b1")
    path.parent.mkdir(parents=True)
    target = tmp_path / "target.txt"
    target.write_bytes(b"secret")
    try:
        os.symlink(target, path)
    except OSError as exc:  # pragma: no cover - platform privilege guard
        pytest.skip(f"symlink unavailable: {exc}")

    with pytest.raises(OSError):
        await store.init_task_output("b1")

    assert target.read_bytes() == b"secret"


@pytest.mark.asyncio
async def test_direct_file_read_does_not_follow_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"secret")
    link = tmp_path / "link.output"
    try:
        os.symlink(target, link)
    except OSError as exc:  # pragma: no cover - platform privilege guard
        pytest.skip(f"symlink unavailable: {exc}")

    with pytest.raises(OSError):
        await read_task_output_file_tail("b1", link)


@pytest.mark.asyncio
async def test_file_task_output_store_sanitizes_task_id(tmp_path: Path) -> None:
    store = FileTaskOutputStore(tmp_path, session_id="s")
    ref = await store.init_task_output("../b/1")

    assert ref.path is not None
    output_path = Path(ref.path)
    output_path.relative_to(store.task_dir)
    assert output_path.name.endswith(".output")
