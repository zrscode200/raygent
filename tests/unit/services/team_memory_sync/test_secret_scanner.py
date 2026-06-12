from __future__ import annotations

from raygent_harness.services.team_memory_sync.secret_scanner import (
    get_secret_label,
    redact_secrets,
    scan_for_secrets,
)


def test_scan_for_secrets_reports_rule_and_label_without_secret_value() -> None:
    secret = "ghp_" + "a" * 36

    matches = scan_for_secrets(f"token={secret}")

    assert [(match.rule_id, match.label) for match in matches] == [
        ("github-pat", "GitHub PAT")
    ]
    assert secret not in repr(matches)


def test_scan_for_secrets_deduplicates_matches_by_rule() -> None:
    first = "AKIA" + "A" * 16
    second = "ASIA" + "B" * 16

    matches = scan_for_secrets(f"{first}\n{second}")

    assert [(match.rule_id, match.label) for match in matches] == [
        ("aws-access-token", "AWS Access Token")
    ]


def test_scan_for_secrets_detects_private_key_blocks() -> None:
    body = "A" * 64
    content = f"-----BEGIN PRIVATE KEY-----\n{body}\n-----END PRIVATE KEY-----"

    assert [(match.rule_id, match.label) for match in scan_for_secrets(content)] == [
        ("private-key", "Private Key")
    ]


def test_scan_uses_reference_ascii_word_boundary_semantics() -> None:
    secret = "AKIA" + "A" * 16

    assert [(match.rule_id, match.label) for match in scan_for_secrets(f"é{secret}")] == [
        ("aws-access-token", "AWS Access Token")
    ]


def test_scan_uses_reference_ascii_word_character_semantics() -> None:
    unicode_non_secret = "AIza" + ("é" * 35)

    assert scan_for_secrets(f"{unicode_non_secret} ") == ()


def test_get_secret_label_applies_reference_special_cases() -> None:
    assert get_secret_label("openai-api-key") == "OpenAI API Key"
    assert get_secret_label("pypi-upload-token") == "PyPI Upload Token"
    assert get_secret_label("custom-secret-kind") == "Custom Secret Kind"


def test_redact_secrets_preserves_boundary_characters() -> None:
    secret = "AIza" + "a" * 35

    assert redact_secrets(f'key="{secret}";') == 'key="[REDACTED]";'
