"""Client-side high-confidence secret scanning for team memory.

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from re import Pattern


@dataclass(frozen=True)
class SecretMatch:
    """A detected secret rule without exposing the secret value."""

    rule_id: str
    label: str


@dataclass(frozen=True)
class _SecretRule:
    id: str
    source: str
    flags: int = 0


_ANT_KEY_PFX = "-".join(("sk", "ant", "api"))

_SECRET_RULES: tuple[_SecretRule, ...] = (
    _SecretRule(
        "aws-access-token",
        r"\b((?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b",
    ),
    _SecretRule("gcp-api-key", r"\b(AIza[\w-]{35})(?:[\x60'\"\s;]|\\[nr]|$)"),
    _SecretRule(
        "azure-ad-client-secret",
        r"(?:^|[\\'\"\x60\s>=:(,)])"
        r"([a-zA-Z0-9_~.]{3}\dQ~[a-zA-Z0-9_~.-]{31,34})"
        r"(?:$|[\\'\"\x60\s<),])",
    ),
    _SecretRule(
        "digitalocean-pat",
        r"\b(dop_v1_[a-f0-9]{64})(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule(
        "digitalocean-access-token",
        r"\b(doo_v1_[a-f0-9]{64})(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule(
        "anthropic-api-key",
        rf"\b({_ANT_KEY_PFX}03-[a-zA-Z0-9_\-]{{93}}AA)(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule(
        "anthropic-admin-api-key",
        r"\b(sk-ant-admin01-[a-zA-Z0-9_\-]{93}AA)(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule(
        "openai-api-key",
        r"\b(sk-(?:proj|svcacct|admin)-"
        r"(?:[A-Za-z0-9_-]{74}|[A-Za-z0-9_-]{58})T3BlbkFJ"
        r"(?:[A-Za-z0-9_-]{74}|[A-Za-z0-9_-]{58})\b"
        r"|sk-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20})"
        r"(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule(
        "huggingface-access-token",
        r"\b(hf_[a-zA-Z]{34})(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule("github-pat", r"ghp_[0-9a-zA-Z]{36}"),
    _SecretRule("github-fine-grained-pat", r"github_pat_\w{82}"),
    _SecretRule("github-app-token", r"(?:ghu|ghs)_[0-9a-zA-Z]{36}"),
    _SecretRule("github-oauth", r"gho_[0-9a-zA-Z]{36}"),
    _SecretRule("github-refresh-token", r"ghr_[0-9a-zA-Z]{36}"),
    _SecretRule("gitlab-pat", r"glpat-[\w-]{20}"),
    _SecretRule("gitlab-deploy-token", r"gldt-[0-9a-zA-Z_\-]{20}"),
    _SecretRule("slack-bot-token", r"xoxb-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*"),
    _SecretRule("slack-user-token", r"xox[pe](?:-[0-9]{10,13}){3}-[a-zA-Z0-9-]{28,34}"),
    _SecretRule("slack-app-token", r"xapp-\d-[A-Z0-9]+-\d+-[a-z0-9]+", re.IGNORECASE),
    _SecretRule("twilio-api-key", r"SK[0-9a-fA-F]{32}"),
    _SecretRule(
        "sendgrid-api-token",
        r"\b(SG\.[a-zA-Z0-9=_\-.]{66})(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule("npm-access-token", r"\b(npm_[a-zA-Z0-9]{36})(?:[\x60'\"\s;]|\\[nr]|$)"),
    _SecretRule("pypi-upload-token", r"pypi-AgEIcHlwaS5vcmc[\w-]{50,1000}"),
    _SecretRule(
        "databricks-api-token",
        r"\b(dapi[a-f0-9]{32}(?:-\d)?)(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule("hashicorp-tf-api-token", r"[a-zA-Z0-9]{14}\.atlasv1\.[a-zA-Z0-9\-_=]{60,70}"),
    _SecretRule("pulumi-api-token", r"\b(pul-[a-f0-9]{40})(?:[\x60'\"\s;]|\\[nr]|$)"),
    _SecretRule(
        "postman-api-token",
        r"\b(PMAK-[a-fA-F0-9]{24}-[a-fA-F0-9]{34})(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule(
        "grafana-api-key",
        r"\b(eyJrIjoi[A-Za-z0-9+/]{70,400}={0,3})(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule(
        "grafana-cloud-api-token",
        r"\b(glc_[A-Za-z0-9+/]{32,400}={0,3})(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule(
        "grafana-service-account-token",
        r"\b(glsa_[A-Za-z0-9]{32}_[A-Fa-f0-9]{8})(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule("sentry-user-token", r"\b(sntryu_[a-f0-9]{64})(?:[\x60'\"\s;]|\\[nr]|$)"),
    _SecretRule(
        "sentry-org-token",
        r"\bsntrys_eyJpYXQiO[a-zA-Z0-9+/]{10,200}"
        r"(?:LCJyZWdpb25fdXJs|InJlZ2lvbl91cmwi|cmVnaW9uX3VybCI6)"
        r"[a-zA-Z0-9+/]{10,200}={0,2}_[a-zA-Z0-9+/]{43}",
    ),
    _SecretRule(
        "stripe-access-token",
        r"\b((?:sk|rk)_(?:test|live|prod)_[a-zA-Z0-9]{10,99})"
        r"(?:[\x60'\"\s;]|\\[nr]|$)",
    ),
    _SecretRule("shopify-access-token", r"shpat_[a-fA-F0-9]{32}"),
    _SecretRule("shopify-shared-secret", r"shpss_[a-fA-F0-9]{32}"),
    _SecretRule(
        "private-key",
        r"-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----"
        r"[\s\S-]{64,}?"
        r"-----END[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----",
        re.IGNORECASE,
    ),
)

_SPECIAL_LABEL_PARTS = {
    "aws": "AWS",
    "gcp": "GCP",
    "api": "API",
    "pat": "PAT",
    "ad": "AD",
    "tf": "TF",
    "oauth": "OAuth",
    "npm": "NPM",
    "pypi": "PyPI",
    "jwt": "JWT",
    "github": "GitHub",
    "gitlab": "GitLab",
    "openai": "OpenAI",
    "digitalocean": "DigitalOcean",
    "huggingface": "HuggingFace",
    "hashicorp": "HashiCorp",
    "sendgrid": "SendGrid",
}

_JS_WORD_CHARS = "A-Za-z0-9_"
_JS_WORD_BOUNDARY = (
    rf"(?:(?<![{_JS_WORD_CHARS}])(?=[{_JS_WORD_CHARS}])|"
    rf"(?<=[{_JS_WORD_CHARS}])(?![{_JS_WORD_CHARS}]))"
)


def _capitalize(value: str) -> str:
    return value[:1].upper() + value[1:]


def get_secret_label(rule_id: str) -> str:
    """Return the reference-style human label for a gitleaks rule id."""
    return " ".join(
        _SPECIAL_LABEL_PARTS.get(part, _capitalize(part)) for part in rule_id.split("-")
    )


def _translate_js_regex_source(source: str) -> str:
    """Translate JS ASCII word tokens to explicit Python regex equivalents."""
    output: list[str] = []
    in_class = False
    index = 0

    while index < len(source):
        char = source[index]
        if char == "\\" and index + 1 < len(source):
            escaped = source[index + 1]
            if escaped == "b":
                output.append(r"\b" if in_class else _JS_WORD_BOUNDARY)
            elif escaped == "w":
                output.append(_JS_WORD_CHARS if in_class else f"[{_JS_WORD_CHARS}]")
            elif escaped == "W":
                output.append(f"^{_JS_WORD_CHARS}" if in_class else f"[^{_JS_WORD_CHARS}]")
            elif escaped == "d":
                output.append("0-9" if in_class else "[0-9]")
            elif escaped == "D":
                output.append("^0-9" if in_class else "[^0-9]")
            else:
                output.append(char)
                output.append(escaped)
            index += 2
            continue

        if char == "[" and not in_class:
            in_class = True
        elif char == "]" and in_class:
            in_class = False
        output.append(char)
        index += 1

    return "".join(output)


@cache
def _compiled_rules() -> tuple[tuple[str, Pattern[str]], ...]:
    return tuple(
        (rule.id, re.compile(_translate_js_regex_source(rule.source), rule.flags))
        for rule in _SECRET_RULES
    )


@cache
def _redact_rules() -> tuple[Pattern[str], ...]:
    return tuple(
        re.compile(_translate_js_regex_source(rule.source), rule.flags)
        for rule in _SECRET_RULES
    )


def scan_for_secrets(content: str) -> tuple[SecretMatch, ...]:
    """Scan content and return one match per rule id, never the secret value."""
    matches: list[SecretMatch] = []
    seen: set[str] = set()

    for rule_id, regex in _compiled_rules():
        if rule_id in seen:
            continue
        if regex.search(content) is not None:
            seen.add(rule_id)
            matches.append(SecretMatch(rule_id=rule_id, label=get_secret_label(rule_id)))

    return tuple(matches)


def _redact_match(match: re.Match[str]) -> str:
    if match.lastindex:
        try:
            start = match.start(1)
            end = match.end(1)
        except IndexError:
            start = end = -1
        if start >= 0 and end >= 0:
            start_offset = start - match.start(0)
            end_offset = end - match.start(0)
            return f"{match.group(0)[:start_offset]}[REDACTED]{match.group(0)[end_offset:]}"
    return "[REDACTED]"


def redact_secrets(content: str) -> str:
    """Replace detected secret values with `[REDACTED]`, preserving boundaries."""
    for regex in _redact_rules():
        content = regex.sub(_redact_match, content)
    return content


__all__ = [
    "SecretMatch",
    "get_secret_label",
    "redact_secrets",
    "scan_for_secrets",
]
