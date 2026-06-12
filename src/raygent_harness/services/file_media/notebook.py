"""Provider-neutral Jupyter notebook parsing and rendering helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

NOTEBOOK_MAX_RAW_BYTES = 25 * 1024 * 1024
NOTEBOOK_MAX_PROCESSED_BYTES = 256 * 1024
NOTEBOOK_LARGE_OUTPUT_THRESHOLD_CHARS = 10_000
NOTEBOOK_OUTPUT_TEXT_MAX_CHARS = 10_000

NotebookServiceErrorReason = Literal["invalid_json", "invalid_schema", "too_large"]


class NotebookServiceError(Exception):
    """Model-visible notebook parse/render failure."""

    def __init__(self, reason: NotebookServiceErrorReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass(frozen=True)
class NotebookOutputImage:
    """Base64 notebook output image."""

    image_data: str
    media_type: Literal["image/png", "image/jpeg"]


@dataclass(frozen=True)
class NotebookCellOutput:
    """Processed notebook output for model-visible content."""

    output_type: str
    text: str | None = None
    image: NotebookOutputImage | None = None


@dataclass(frozen=True)
class NotebookCellSource:
    """Processed notebook cell source and bounded outputs."""

    cell_type: str
    source: str
    cell_id: str
    execution_count: int | None = None
    language: str | None = None
    outputs: tuple[NotebookCellOutput, ...] = ()


@dataclass(frozen=True)
class NotebookParseResult:
    """Parsed notebook cells plus bounded-size accounting."""

    raw_content: str
    cells: tuple[NotebookCellSource, ...]
    language: str
    processed_size_bytes: int

    @property
    def cell_count(self) -> int:
        return len(self.cells)


def parse_notebook_content(
    raw_content: str,
    *,
    file_path: str = "<notebook>",
    max_processed_bytes: int = NOTEBOOK_MAX_PROCESSED_BYTES,
) -> NotebookParseResult:
    """Parse notebook JSON into bounded, provider-neutral cell records."""

    try:
        raw_loaded: object = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise NotebookServiceError(
            "invalid_json",
            f"Notebook is not valid JSON: {file_path} ({exc.msg}).",
        ) from exc

    if not isinstance(raw_loaded, Mapping):
        raise NotebookServiceError(
            "invalid_schema",
            f"Notebook root must be a JSON object: {file_path}.",
        )
    raw_notebook = cast(Mapping[object, object], raw_loaded)
    raw_cells_obj = raw_notebook.get("cells")
    if not isinstance(raw_cells_obj, list):
        raise NotebookServiceError(
            "invalid_schema",
            f"Notebook is missing a cells list: {file_path}.",
        )
    raw_cells = cast(list[object], raw_cells_obj)

    language = _notebook_language(raw_notebook)
    cells = tuple(
        _process_cell(
            raw_cell,
            index=index,
            code_language=language,
            file_path=file_path,
        )
        for index, raw_cell in enumerate(raw_cells)
    )
    processed_size = len(_processed_cells_json(cells).encode())
    if processed_size > max_processed_bytes:
        raise NotebookServiceError(
            "too_large",
            _notebook_too_large_message(
                file_path=file_path,
                size=processed_size,
                max_size=max_processed_bytes,
            ),
        )
    return NotebookParseResult(
        raw_content=raw_content,
        cells=cells,
        language=language,
        processed_size_bytes=processed_size,
    )


def notebook_cells_to_content(cells: tuple[NotebookCellSource, ...]) -> list[dict[str, Any]]:
    """Map processed notebook cells to Raygent tool-result content blocks."""

    content: list[dict[str, Any]] = []
    for cell in cells:
        _append_text_block(content, _cell_content_text(cell))
        for output in cell.outputs:
            if output.text:
                _append_text_block(content, f"\n{output.text}")
            if output.image is not None:
                content.append(
                    {
                        "type": "image",
                        "media_type": output.image.media_type,
                        "source": {
                            "type": "base64",
                            "media_type": output.image.media_type,
                            "data": output.image.image_data,
                        },
                        "metadata": {
                            "cell_id": cell.cell_id,
                            "output_type": output.output_type,
                        },
                    }
                )
    return content


def notebook_cells_to_json(cells: tuple[NotebookCellSource, ...]) -> str:
    """Return deterministic JSON for tests, diagnostics, and size accounting."""

    return json.dumps(
        [_cell_to_dict(cell) for cell in cells],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _process_cell(
    raw_cell: object,
    *,
    index: int,
    code_language: str,
    file_path: str,
) -> NotebookCellSource:
    if not isinstance(raw_cell, Mapping):
        raise NotebookServiceError(
            "invalid_schema",
            f"Notebook cell {index} must be a JSON object: {file_path}.",
        )
    cell = cast(Mapping[object, object], raw_cell)
    raw_cell_type = cell.get("cell_type")
    if not isinstance(raw_cell_type, str):
        raise NotebookServiceError(
            "invalid_schema",
            f"Notebook cell {index} is missing a string cell_type: {file_path}.",
        )
    cell_type = raw_cell_type
    raw_id = cell.get("id")
    cell_id = raw_id if isinstance(raw_id, str) and raw_id else f"cell-{index}"
    execution_count = _execution_count(cell.get("execution_count"))
    language = code_language if cell_type == "code" else None
    outputs: tuple[NotebookCellOutput, ...] = ()
    if cell_type == "code":
        outputs = _process_outputs(
            cell.get("outputs"),
            cell_index=index,
            file_path=file_path,
        )
    return NotebookCellSource(
        cell_type=cell_type,
        source=_source_text(cell.get("source")),
        cell_id=cell_id,
        execution_count=execution_count if cell_type == "code" else None,
        language=language,
        outputs=outputs,
    )


def _process_outputs(
    raw_outputs: object,
    *,
    cell_index: int,
    file_path: str,
) -> tuple[NotebookCellOutput, ...]:
    if raw_outputs is None:
        return ()
    if not isinstance(raw_outputs, list):
        return ()
    output_items = cast(list[object], raw_outputs)
    outputs = tuple(
        output
        for raw_output in output_items
        if (output := _process_output(raw_output)) is not None
    )
    if _outputs_are_large(outputs):
        return (
            NotebookCellOutput(
                output_type="stream",
                text=(
                    "Outputs are too large to include. Use Bash with: "
                    f"cat {json.dumps(file_path)} | jq '.cells[{cell_index}].outputs'"
                ),
            ),
        )
    return outputs


def _process_output(raw_output: object) -> NotebookCellOutput | None:
    if not isinstance(raw_output, Mapping):
        return None
    output = cast(Mapping[object, object], raw_output)
    raw_output_type = output.get("output_type")
    if not isinstance(raw_output_type, str):
        return None
    if raw_output_type == "stream":
        return NotebookCellOutput(
            output_type=raw_output_type,
            text=_process_output_text(output.get("text")),
        )
    if raw_output_type in {"execute_result", "display_data"}:
        data = output.get("data")
        data_mapping: Mapping[object, object] = (
            cast(Mapping[object, object], data) if isinstance(data, Mapping) else {}
        )
        return NotebookCellOutput(
            output_type=raw_output_type,
            text=_process_output_text(data_mapping.get("text/plain")),
            image=_extract_image(data_mapping),
        )
    if raw_output_type == "error":
        ename = str(output.get("ename") or "")
        evalue = str(output.get("evalue") or "")
        traceback = _source_text(output.get("traceback"))
        text = f"{ename}: {evalue}"
        if traceback:
            text += f"\n{traceback}"
        return NotebookCellOutput(
            output_type=raw_output_type,
            text=_process_output_text(text),
        )
    return None


def _extract_image(data: Mapping[object, object]) -> NotebookOutputImage | None:
    png = _image_data(data.get("image/png"))
    if png is not None:
        return NotebookOutputImage(image_data=png, media_type="image/png")
    jpeg = _image_data(data.get("image/jpeg"))
    if jpeg is not None:
        return NotebookOutputImage(image_data=jpeg, media_type="image/jpeg")
    return None


def _image_data(value: object) -> str | None:
    text = _optional_source_text(value)
    if text is None:
        return None
    stripped = "".join(text.split())
    return stripped or None


def _process_output_text(value: object) -> str:
    text = _source_text(value)
    if len(text) <= NOTEBOOK_OUTPUT_TEXT_MAX_CHARS:
        return text
    return (
        text[:NOTEBOOK_OUTPUT_TEXT_MAX_CHARS]
        + "\n... [notebook output truncated]"
    )


def _outputs_are_large(outputs: tuple[NotebookCellOutput, ...]) -> bool:
    total = 0
    for output in outputs:
        total += len(output.text or "")
        if output.image is not None:
            total += len(output.image.image_data)
        if total > NOTEBOOK_LARGE_OUTPUT_THRESHOLD_CHARS:
            return True
    return False


def _cell_content_text(cell: NotebookCellSource) -> str:
    metadata: list[str] = []
    if cell.cell_type != "code":
        metadata.append(f"<cell_type>{cell.cell_type}</cell_type>")
    if cell.cell_type == "code" and cell.language and cell.language != "python":
        metadata.append(f"<language>{cell.language}</language>")
    return (
        f'<cell id="{cell.cell_id}">'
        f"{''.join(metadata)}{cell.source}"
        f'</cell id="{cell.cell_id}">'
    )


def _append_text_block(content: list[dict[str, Any]], text: str) -> None:
    if content and content[-1].get("type") == "text":
        content[-1]["text"] = f"{content[-1]['text']}\n{text}"
        return
    content.append({"type": "text", "text": text})


def _processed_cells_json(cells: tuple[NotebookCellSource, ...]) -> str:
    return notebook_cells_to_json(cells)


def _cell_to_dict(cell: NotebookCellSource) -> dict[str, Any]:
    data: dict[str, Any] = {
        "cellType": cell.cell_type,
        "source": cell.source,
        "cell_id": cell.cell_id,
    }
    if cell.execution_count is not None:
        data["execution_count"] = cell.execution_count
    if cell.language is not None:
        data["language"] = cell.language
    if cell.outputs:
        data["outputs"] = [_output_to_dict(output) for output in cell.outputs]
    return data


def _output_to_dict(output: NotebookCellOutput) -> dict[str, Any]:
    data: dict[str, Any] = {"output_type": output.output_type}
    if output.text is not None:
        data["text"] = output.text
    if output.image is not None:
        data["image"] = {
            "image_data": output.image.image_data,
            "media_type": output.image.media_type,
        }
    return data


def _source_text(value: object) -> str:
    text = _optional_source_text(value)
    return "" if text is None else text


def _optional_source_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        items = cast(list[object], value)
        return "".join(str(item) for item in items)
    return str(value)


def _execution_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _notebook_language(notebook: Mapping[object, object]) -> str:
    metadata = notebook.get("metadata")
    if not isinstance(metadata, Mapping):
        return "python"
    language_info = cast(Mapping[object, object], metadata).get("language_info")
    if not isinstance(language_info, Mapping):
        return "python"
    name = cast(Mapping[object, object], language_info).get("name")
    return name if isinstance(name, str) and name else "python"


def _notebook_too_large_message(
    *,
    file_path: str,
    size: int,
    max_size: int,
) -> str:
    return (
        f"Notebook content ({_format_file_size(size)}) exceeds maximum allowed "
        f"processed size ({_format_file_size(max_size)}). Use Bash with jq to "
        "read specific portions, for example:\n"
        f"  cat {json.dumps(file_path)} | jq '.cells[:20]'\n"
        f"  cat {json.dumps(file_path)} | jq '.cells | length'\n"
        f"  cat {json.dumps(file_path)} | jq '.cells[] | select(.cell_type==\"code\") | .source'"
    )


def _format_file_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"
