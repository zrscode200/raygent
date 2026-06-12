from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from raygent_harness.core.media_budget import EXCESS_MEDIA_REMOVED_PLACEHOLDER
from raygent_harness.core.messages import thaw_json
from raygent_harness.core.model_request_normalization import (
    SYNTHETIC_TOOL_RESULT_PLACEHOLDER,
    TOOL_REFERENCES_REMOVED_PLACEHOLDER,
    UNAVAILABLE_TOOL_REFERENCES_REMOVED_PLACEHOLDER,
    UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER,
    UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER,
    normalize_model_request_for_provider,
)
from raygent_harness.core.model_types import (
    ApiMessage,
    FrozenJson,
    MediaContentBlock,
    ModelCapabilities,
    ModelInfo,
    ModelMessage,
    ModelRequest,
    ModelToolSpec,
    TextContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
)


def _request(
    *messages: ApiMessage,
    supports_tool_references: bool = False,
    supports_images: bool = False,
    supports_documents: bool = False,
    max_media_items_per_request: int | None = None,
    tool_names: tuple[str, ...] = (),
) -> tuple[ModelRequest, ModelInfo]:
    info = ModelInfo(
        model="model-1",
        max_media_items_per_request=max_media_items_per_request,
        capabilities=ModelCapabilities(
            supports_tool_references=supports_tool_references,
            supports_images=supports_images,
            supports_documents=supports_documents,
        ),
    )
    tools = tuple(
        ModelToolSpec(
            name=name,
            description=f"{name} tool",
            input_schema={"type": "object"},
        )
        for name in tool_names
    )
    return ModelRequest(model="model-1", messages=messages, tools=tools), info


def _assistant_tool_use(
    tool_use_id: str,
    *,
    caller: object | None = None,
) -> ApiMessage:
    provider_metadata = (
        cast(FrozenJson, {"caller": caller}) if caller is not None else None
    )
    return ApiMessage(
        message=ModelMessage(
            role="assistant",
            content=(
                ToolUseContentBlock(
                    id=tool_use_id,
                    name="ToolSearch",
                    input={"query": "Read"},
                    provider_metadata=provider_metadata,
                ),
            ),
        )
    )


def _user_tool_result(tool_use_id: str, content: FrozenJson) -> ApiMessage:
    return ApiMessage(
        message=ModelMessage(
            role="user",
            content=(
                ToolResultContentBlock(
                    tool_use_id=tool_use_id,
                    content=content,
                ),
            ),
        )
    )


def _user_image(image_id: str) -> ApiMessage:
    return ApiMessage(
        message=ModelMessage(
            role="user",
            content=(
                MediaContentBlock(
                    media_kind="image",
                    media_type="image/png",
                    data={
                        "type": "image",
                        "media_type": "image/png",
                        "id": image_id,
                    },
                ),
            ),
        )
    )


def test_unsupported_target_strips_tool_reference_and_caller_api_bound_only() -> None:
    tool_reference = {"type": "tool_reference", "tool_name": "Read"}
    request, info = _request(
        _assistant_tool_use("toolu_1", caller={"type": "tool_search"}),
        _user_tool_result("toolu_1", cast(FrozenJson, [tool_reference])),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    original_tool_use = cast(
        ToolUseContentBlock,
        request.messages[0].message.content[0],
    )
    assert original_tool_use.provider_metadata is not None
    assert "caller" in cast(Mapping[str, object], original_tool_use.provider_metadata)
    original_result = cast(ToolResultContentBlock, request.messages[1].message.content[0])
    assert thaw_json(original_result.content) == [tool_reference]

    normalized_tool_use = cast(
        ToolUseContentBlock,
        normalized.messages[0].message.content[0],
    )
    assert normalized_tool_use.provider_metadata is None
    normalized_result = cast(
        ToolResultContentBlock,
        normalized.messages[1].message.content[0],
    )
    assert thaw_json(normalized_result.content) == [
        {"type": "text", "text": TOOL_REFERENCES_REMOVED_PLACEHOLDER}
    ]
    provider_payload = normalized.messages[1].provider_payload
    assert provider_payload is not None
    assert thaw_json(provider_payload) == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [
                    {"type": "text", "text": TOOL_REFERENCES_REMOVED_PLACEHOLDER}
                ],
            }
        ],
    }


def test_supported_target_preserves_tool_reference_and_caller() -> None:
    request, info = _request(
        _assistant_tool_use("toolu_1", caller={"type": "tool_search"}),
        _user_tool_result(
            "toolu_1",
            cast(
                FrozenJson,
                [
                    {"type": "text", "text": "selected"},
                    {"type": "tool_reference", "tool_name": "Read"},
                ],
            ),
        ),
        supports_tool_references=True,
        tool_names=("Read",),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    assert normalized is request


def test_supported_target_strips_only_unavailable_tool_references() -> None:
    request, info = _request(
        _assistant_tool_use("toolu_1", caller={"type": "tool_search"}),
        _user_tool_result(
            "toolu_1",
            cast(
                FrozenJson,
                [
                    {"type": "tool_reference", "tool_name": "Read"},
                    {"type": "tool_reference", "tool_name": "Missing"},
                ],
            ),
        ),
        supports_tool_references=True,
        tool_names=("Read",),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    normalized_tool_use = cast(
        ToolUseContentBlock,
        normalized.messages[0].message.content[0],
    )
    assert normalized_tool_use.provider_metadata is not None
    normalized_result = cast(
        ToolResultContentBlock,
        normalized.messages[1].message.content[0],
    )
    assert thaw_json(normalized_result.content) == [
        {"type": "tool_reference", "tool_name": "Read"}
    ]


def test_supported_target_replaces_all_unavailable_tool_references() -> None:
    request, info = _request(
        _assistant_tool_use("toolu_1", caller={"type": "tool_search"}),
        _user_tool_result(
            "toolu_1",
            cast(
                FrozenJson,
                [{"type": "tool_reference", "tool_name": "Missing"}],
            ),
        ),
        supports_tool_references=True,
        tool_names=("Read",),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    normalized_result = cast(
        ToolResultContentBlock,
        normalized.messages[1].message.content[0],
    )
    assert thaw_json(normalized_result.content) == [
        {"type": "text", "text": UNAVAILABLE_TOOL_REFERENCES_REMOVED_PLACEHOLDER}
    ]


def test_media_budget_strips_oldest_top_level_media_and_preserves_recent() -> None:
    request, info = _request(
        _user_image("old"),
        _user_image("recent"),
        supports_images=True,
        max_media_items_per_request=1,
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    assert normalized.media_budget is not None
    assert normalized.media_budget.original_media_items == 2
    assert normalized.media_budget.retained_media_items == 1
    assert normalized.media_budget.stripped_media_items == 1

    first_content = normalized.messages[0].message.content
    assert isinstance(first_content[0], TextContentBlock)
    assert EXCESS_MEDIA_REMOVED_PLACEHOLDER in first_content[0].text
    second_content = normalized.messages[1].message.content
    assert isinstance(second_content[0], MediaContentBlock)
    second_data = thaw_json(second_content[0].data)
    assert isinstance(second_data, Mapping)
    assert second_data["id"] == "recent"

    original_first = request.messages[0].message.content
    assert isinstance(original_first[0], MediaContentBlock)


def test_media_budget_counts_and_strips_nested_tool_result_media() -> None:
    request, info = _request(
        _assistant_tool_use("toolu_1"),
        ApiMessage(
            message=ModelMessage(
                role="user",
                content=(
                    ToolResultContentBlock(
                        tool_use_id="toolu_1",
                        content=cast(
                            FrozenJson,
                            [
                                {
                                    "type": "image",
                                    "media_type": "image/png",
                                    "id": "old-nested",
                                },
                                {"type": "text", "text": "metadata"},
                            ],
                        ),
                    ),
                    MediaContentBlock(
                        media_kind="image",
                        media_type="image/png",
                        data={
                            "type": "image",
                            "media_type": "image/png",
                            "id": "recent-top-level",
                        },
                    ),
                ),
            )
        ),
        supports_images=True,
        max_media_items_per_request=1,
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    assert normalized.media_budget is not None
    assert normalized.media_budget.nested_media_items == 1
    assert normalized.media_budget.top_level_media_items == 1
    assert normalized.media_budget.stripped_media_items == 1

    user_content = normalized.messages[1].message.content
    tool_result = cast(ToolResultContentBlock, user_content[0])
    assert thaw_json(tool_result.content) == [
        {
            "type": "text",
            "text": f"{EXCESS_MEDIA_REMOVED_PLACEHOLDER} (image)",
        },
        {"type": "text", "text": "metadata"},
    ]
    assert isinstance(user_content[1], MediaContentBlock)


def test_media_budget_strips_nested_before_top_level_within_same_message() -> None:
    request, info = _request(
        _assistant_tool_use("toolu_1"),
        ApiMessage(
            message=ModelMessage(
                role="user",
                content=(
                    MediaContentBlock(
                        media_kind="image",
                        media_type="image/png",
                        data={
                            "type": "image",
                            "media_type": "image/png",
                            "id": "top-level-before-tool-result",
                        },
                    ),
                    ToolResultContentBlock(
                        tool_use_id="toolu_1",
                        content=cast(
                            FrozenJson,
                            [
                                {
                                    "type": "image",
                                    "media_type": "image/png",
                                    "id": "nested-in-tool-result",
                                }
                            ],
                        ),
                    ),
                ),
            )
        ),
        supports_images=True,
        max_media_items_per_request=1,
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    user_content = normalized.messages[1].message.content
    assert isinstance(user_content[0], MediaContentBlock)
    tool_result = cast(ToolResultContentBlock, user_content[1])
    assert thaw_json(tool_result.content) == [
        {
            "type": "text",
            "text": f"{EXCESS_MEDIA_REMOVED_PLACEHOLDER} (image)",
        }
    ]


def test_pairing_repair_inserts_missing_result_and_strips_orphaned_result() -> None:
    request, info = _request(
        _assistant_tool_use("toolu_missing"),
        ApiMessage(
            message=ModelMessage(
                role="user",
                content=(
                    ToolResultContentBlock(
                        tool_use_id="toolu_orphan",
                        content="stale",
                    ),
                    TextContentBlock(text="keep this user content"),
                ),
            )
        ),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    user_content = normalized.messages[1].message.content
    assert len(user_content) == 2
    synthetic = cast(ToolResultContentBlock, user_content[0])
    assert synthetic.tool_use_id == "toolu_missing"
    assert synthetic.content == SYNTHETIC_TOOL_RESULT_PLACEHOLDER
    assert synthetic.is_error is True
    assert cast(TextContentBlock, user_content[1]).text == "keep this user content"


def test_pairing_repair_strips_tool_result_after_non_assistant_message() -> None:
    request, info = _request(
        ApiMessage(
            message=ModelMessage(
                role="user",
                content=(TextContentBlock(text="plain user"),),
            )
        ),
        ApiMessage(
            message=ModelMessage(
                role="user",
                content=(
                    ToolResultContentBlock(tool_use_id="toolu_orphan", content="stale"),
                    TextContentBlock(text="keep text"),
                ),
            )
        ),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    assert normalized.messages[0] is request.messages[0]
    second_content = normalized.messages[1].message.content
    assert second_content == (TextContentBlock(text="keep text"),)


def test_pairing_repair_dedupes_duplicate_tool_uses_and_results() -> None:
    request, info = _request(
        ApiMessage(
            message=ModelMessage(
                role="assistant",
                content=(
                    ToolUseContentBlock(id="toolu_1", name="Read", input={}),
                    ToolUseContentBlock(id="toolu_1", name="Read", input={}),
                ),
            )
        ),
        ApiMessage(
            message=ModelMessage(
                role="user",
                content=(
                    ToolResultContentBlock(tool_use_id="toolu_1", content="first"),
                    ToolResultContentBlock(tool_use_id="toolu_1", content="duplicate"),
                ),
            )
        ),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    assistant_tool_uses = [
        block
        for block in normalized.messages[0].message.content
        if isinstance(block, ToolUseContentBlock)
    ]
    user_tool_results = [
        block
        for block in normalized.messages[1].message.content
        if isinstance(block, ToolResultContentBlock)
    ]
    assert len(assistant_tool_uses) == 1
    assert len(user_tool_results) == 1
    assert user_tool_results[0].content == "first"


def test_unsupported_target_strips_top_level_media_api_bound_only() -> None:
    request, info = _request(
        ApiMessage(
            message=ModelMessage(
                role="user",
                content=(
                    TextContentBlock(text="inspect"),
                    MediaContentBlock(
                        media_kind="image",
                        media_type="image/png",
                        data={"source": {"type": "base64", "data": "abc"}},
                    ),
                    MediaContentBlock(
                        media_kind="document",
                        media_type="application/pdf",
                        data={"source": {"type": "base64", "data": "pdf"}},
                    ),
                ),
            )
        ),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    assert isinstance(request.messages[0].message.content[1], MediaContentBlock)
    assert isinstance(request.messages[0].message.content[2], MediaContentBlock)
    normalized_content = normalized.messages[0].message.content
    assert normalized_content == (
        TextContentBlock(text="inspect"),
        TextContentBlock(text=UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER),
        TextContentBlock(text=UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER),
    )
    assert normalized.messages[0].provider_payload is not None


def test_unsupported_target_strips_nested_tool_result_media_api_bound_only() -> None:
    nested_media = cast(
        FrozenJson,
        [
            {"type": "text", "text": "read result"},
            {
                "type": "image",
                "media_type": "image/png",
                "source": {"type": "base64", "data": "abc"},
            },
            {
                "type": "document",
                "media_type": "application/pdf",
                "source": {"type": "base64", "data": "pdf"},
            },
        ],
    )
    request, info = _request(
        _assistant_tool_use("toolu_read"),
        _user_tool_result("toolu_read", nested_media),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    original_result = cast(ToolResultContentBlock, request.messages[1].message.content[0])
    assert thaw_json(original_result.content) == thaw_json(nested_media)
    normalized_result = cast(
        ToolResultContentBlock,
        normalized.messages[1].message.content[0],
    )
    assert thaw_json(normalized_result.content) == [
        {"type": "text", "text": "read result"},
        {"type": "text", "text": UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER},
        {"type": "text", "text": UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER},
    ]


def test_unsupported_target_strips_notebook_output_images_api_bound_only() -> None:
    notebook_content = cast(
        FrozenJson,
        [
            {
                "type": "text",
                "text": '<cell id="plot-cell">display(plot)</cell id="plot-cell">',
            },
            {
                "type": "image",
                "media_type": "image/png",
                "source": {"type": "base64", "data": "iVBORw0KGgo="},
                "metadata": {"cell_id": "plot-cell", "output_type": "display_data"},
            },
        ],
    )
    request, info = _request(
        _assistant_tool_use("toolu_read"),
        _user_tool_result("toolu_read", notebook_content),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    original_result = cast(ToolResultContentBlock, request.messages[1].message.content[0])
    assert thaw_json(original_result.content) == thaw_json(notebook_content)
    normalized_result = cast(
        ToolResultContentBlock,
        normalized.messages[1].message.content[0],
    )
    assert thaw_json(normalized_result.content) == [
        {
            "type": "text",
            "text": '<cell id="plot-cell">display(plot)</cell id="plot-cell">',
        },
        {"type": "text", "text": UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER},
    ]
    provider_payload = normalized.messages[1].provider_payload
    assert provider_payload is not None
    provider_payload_text = str(thaw_json(provider_payload))
    assert UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER in provider_payload_text
    assert "iVBORw0KGgo=" not in provider_payload_text
    assert "image/png" not in provider_payload_text
    assert "source" not in provider_payload_text


def test_unsupported_target_strips_pdf_page_images_api_bound_only() -> None:
    pdf_page_content = cast(
        FrozenJson,
        [
            {"type": "text", "text": "PDF page extraction: paper.pdf pages 1-2"},
            {
                "type": "image",
                "media_type": "image/jpeg",
                "source": {"type": "base64", "data": "/9j/page1"},
                "metadata": {
                    "file_path": "/tmp/paper.pdf",
                    "page_number": 1,
                    "original_size": 100,
                },
            },
            {
                "type": "image",
                "media_type": "image/jpeg",
                "source": {"type": "base64", "data": "/9j/page2"},
                "metadata": {
                    "file_path": "/tmp/paper.pdf",
                    "page_number": 2,
                    "original_size": 100,
                },
            },
        ],
    )
    request, info = _request(
        _assistant_tool_use("toolu_pdf"),
        _user_tool_result("toolu_pdf", pdf_page_content),
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    original_result = cast(ToolResultContentBlock, request.messages[1].message.content[0])
    assert thaw_json(original_result.content) == thaw_json(pdf_page_content)
    normalized_result = cast(
        ToolResultContentBlock,
        normalized.messages[1].message.content[0],
    )
    assert thaw_json(normalized_result.content) == [
        {"type": "text", "text": "PDF page extraction: paper.pdf pages 1-2"},
        {"type": "text", "text": UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER},
        {"type": "text", "text": UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER},
    ]
    provider_payload = normalized.messages[1].provider_payload
    assert provider_payload is not None
    provider_payload_text = str(thaw_json(provider_payload))
    assert provider_payload_text.count(UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER) == 2
    assert "/9j/page" not in provider_payload_text
    assert "image/jpeg" not in provider_payload_text
    assert "source" not in provider_payload_text
    assert "page_number" not in provider_payload_text


def test_media_budget_strips_oldest_pdf_page_before_notebook_output_image() -> None:
    request, info = _request(
        _assistant_tool_use("toolu_pdf"),
        _user_tool_result(
            "toolu_pdf",
            cast(
                FrozenJson,
                [
                    {"type": "text", "text": "PDF page extraction: paper.pdf pages 1"},
                    {
                        "type": "image",
                        "media_type": "image/jpeg",
                        "source": {"type": "base64", "data": "/9j/page1"},
                        "metadata": {"page_number": 1, "file_path": "/tmp/paper.pdf"},
                    },
                ],
            ),
        ),
        _assistant_tool_use("toolu_notebook"),
        _user_tool_result(
            "toolu_notebook",
            cast(
                FrozenJson,
                [
                    {
                        "type": "text",
                        "text": '<cell id="plot-cell">display(plot)</cell id="plot-cell">',
                    },
                    {
                        "type": "image",
                        "media_type": "image/png",
                        "source": {"type": "base64", "data": "iVBORw0KGgo="},
                        "metadata": {
                            "cell_id": "plot-cell",
                            "output_type": "display_data",
                        },
                    },
                ],
            ),
        ),
        supports_images=True,
        max_media_items_per_request=1,
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    assert normalized.media_budget is not None
    assert normalized.media_budget.original_media_items == 2
    assert normalized.media_budget.nested_media_items == 2
    assert normalized.media_budget.retained_media_items == 1
    assert normalized.media_budget.stripped_media_items == 1
    pdf_result = cast(ToolResultContentBlock, normalized.messages[1].message.content[0])
    notebook_result = cast(
        ToolResultContentBlock,
        normalized.messages[3].message.content[0],
    )
    assert thaw_json(pdf_result.content) == [
        {"type": "text", "text": "PDF page extraction: paper.pdf pages 1"},
        {"type": "text", "text": f"{EXCESS_MEDIA_REMOVED_PLACEHOLDER} (image)"},
    ]
    notebook_content = thaw_json(notebook_result.content)
    assert isinstance(notebook_content, list)
    retained_image = cast(list[object], notebook_content)[1]
    assert isinstance(retained_image, Mapping)
    assert retained_image["type"] == "image"
    assert cast(Mapping[str, object], retained_image["metadata"])["cell_id"] == "plot-cell"


def test_supported_target_preserves_matching_media_capabilities() -> None:
    nested_media = cast(
        FrozenJson,
        [
            {"type": "image", "media_type": "image/png", "source": {"data": "abc"}},
            {
                "type": "document",
                "media_type": "application/pdf",
                "source": {"data": "pdf"},
            },
        ],
    )
    request, info = _request(
        _assistant_tool_use("toolu_read"),
        _user_tool_result("toolu_read", nested_media),
        supports_images=True,
        supports_documents=True,
    )

    normalized = normalize_model_request_for_provider(request, model_info=info)

    assert normalized.messages == request.messages
    assert normalized.media_budget is not None
    assert normalized.media_budget.original_media_items == 2
    assert normalized.media_budget.stripped_media_items == 0
