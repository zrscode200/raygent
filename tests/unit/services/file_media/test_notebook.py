from __future__ import annotations

import json

import pytest

from raygent_harness.services.file_media import (
    NOTEBOOK_LARGE_OUTPUT_THRESHOLD_CHARS,
    NotebookServiceError,
    notebook_cells_to_content,
    notebook_cells_to_json,
    parse_notebook_content,
)


def test_parse_notebook_normalizes_cells_sources_language_and_outputs() -> None:
    raw = json.dumps(
        {
            "metadata": {"language_info": {"name": "r"}},
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Title\n", "Body"],
                },
                {
                    "id": "code-id",
                    "cell_type": "code",
                    "source": "print('hi')",
                    "execution_count": 7,
                    "outputs": [
                        {"output_type": "stream", "text": ["hello", "\n"]},
                        {
                            "output_type": "execute_result",
                            "data": {
                                "text/plain": "42",
                                "image/png": "iVBORw0K Ggo=",
                            },
                        },
                        {
                            "output_type": "error",
                            "ename": "ValueError",
                            "evalue": "bad",
                            "traceback": ["line 1", "line 2"],
                        },
                    ],
                },
            ],
        }
    )

    result = parse_notebook_content(raw, file_path="/tmp/analysis.ipynb")

    assert result.language == "r"
    assert result.cell_count == 2
    markdown, code = result.cells
    assert markdown.cell_id == "cell-0"
    assert markdown.cell_type == "markdown"
    assert markdown.source == "# Title\nBody"
    assert code.cell_id == "code-id"
    assert code.language == "r"
    assert code.execution_count == 7
    assert [output.output_type for output in code.outputs] == [
        "stream",
        "execute_result",
        "error",
    ]
    assert code.outputs[1].image is not None
    assert code.outputs[1].image.media_type == "image/png"
    assert code.outputs[1].image.image_data == "iVBORw0KGgo="
    assert "ValueError: bad" in str(code.outputs[2].text)
    assert result.processed_size_bytes == len(notebook_cells_to_json(result.cells).encode())


def test_notebook_cells_render_to_merged_text_and_image_blocks() -> None:
    result = parse_notebook_content(
        json.dumps(
            {
                "metadata": {"language_info": {"name": "python"}},
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "display(img)",
                        "outputs": [
                            {
                                "output_type": "display_data",
                                "data": {
                                    "text/plain": "image",
                                    "image/jpeg": "/9j/anBlZw==",
                                },
                            }
                        ],
                    },
                    {"cell_type": "markdown", "source": "notes"},
                ],
            }
        )
    )

    content = notebook_cells_to_content(result.cells)

    assert content[0]["type"] == "text"
    assert '<cell id="cell-0">display(img)</cell id="cell-0">' in content[0]["text"]
    assert "\nimage" in content[0]["text"]
    assert content[1] == {
        "type": "image",
        "media_type": "image/jpeg",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "/9j/anBlZw==",
        },
        "metadata": {
            "cell_id": "cell-0",
            "output_type": "display_data",
        },
    }
    assert content[2]["type"] == "text"
    assert "<cell_type>markdown</cell_type>notes" in content[2]["text"]


def test_parse_notebook_replaces_large_outputs_with_guidance() -> None:
    result = parse_notebook_content(
        json.dumps(
            {
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "print('big')",
                        "outputs": [
                            {
                                "output_type": "stream",
                                "text": "x" * (NOTEBOOK_LARGE_OUTPUT_THRESHOLD_CHARS + 1),
                            }
                        ],
                    }
                ],
            }
        ),
        file_path="/tmp/big.ipynb",
    )

    assert len(result.cells[0].outputs) == 1
    output = result.cells[0].outputs[0]
    assert output.output_type == "stream"
    assert output.text is not None
    assert "Outputs are too large to include" in output.text
    assert "jq '.cells[0].outputs'" in output.text


def test_parse_notebook_enforces_aggregate_processed_size() -> None:
    raw = json.dumps(
        {
            "metadata": {},
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": "x" * 200,
                }
            ],
        }
    )

    with pytest.raises(NotebookServiceError) as exc_info:
        parse_notebook_content(raw, file_path="/tmp/large.ipynb", max_processed_bytes=50)

    assert exc_info.value.reason == "too_large"
    assert "Notebook content" in exc_info.value.message
    assert "jq '.cells[:20]'" in exc_info.value.message


def test_parse_notebook_rejects_invalid_json_and_schema() -> None:
    with pytest.raises(NotebookServiceError) as json_exc:
        parse_notebook_content("{", file_path="/tmp/bad.ipynb")
    with pytest.raises(NotebookServiceError) as root_exc:
        parse_notebook_content("[]", file_path="/tmp/bad.ipynb")
    with pytest.raises(NotebookServiceError) as cells_exc:
        parse_notebook_content("{}", file_path="/tmp/bad.ipynb")
    with pytest.raises(NotebookServiceError) as cell_exc:
        parse_notebook_content(
            json.dumps({"cells": [{"source": ""}]}),
            file_path="/tmp/bad.ipynb",
        )

    assert json_exc.value.reason == "invalid_json"
    assert root_exc.value.reason == "invalid_schema"
    assert cells_exc.value.reason == "invalid_schema"
    assert cell_exc.value.reason == "invalid_schema"


def test_parse_notebook_accepts_empty_notebooks() -> None:
    result = parse_notebook_content('{"metadata": {}, "cells": []}')

    assert result.cells == ()
    assert result.cell_count == 0
