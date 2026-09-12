# Copyright 2026 perplexity-to-openai contributors

"""Perplexity private-API transport (no browser).

Ground truth for this protocol was captured live from the web app (2026-08):
POST https://www.perplexity.ai/rest/sse/perplexity_ask streams SSE JSON events.
Multi-turn continuity is server-side: follow-ups carry last_backend_uuid +
read_write_token (both returned in the first event) and the thread's URL slug
as Referer. Works with curl_cffi Chrome impersonation + session cookies.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from curl_cffi.requests import AsyncSession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from curl_cffi.requests.models import Response

log = logging.getLogger("pplx")

BASE_URL = "https://www.perplexity.ai"
ASK_URL = f"{BASE_URL}/rest/sse/perplexity_ask"
SESSION_URL = f"{BASE_URL}/api/auth/session"
RATE_LIMIT_URL = f"{BASE_URL}/rest/rate-limit/status"
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

HTTP_OK = 200
HTTP_SERVER_ERROR_MIN = 500

# Free-plan-verified modes. "internet" mode fails with GENERIC_FAILED_RESPONSE
# on free accounts but works on Pro; users can opt into it via model suffix.
MODES = {"concise", "copilot", "internet", "pro", "academic", "writing", "math"}

# Requested-model aliases -> Perplexity model_preference. Anything else is
# passed through as-is (Perplexity slugs like gpt5, claude40opusthinking...).
MODEL_ALIASES = {
    "gpt-4o": "turbo",
    "gpt-4o-mini": "turbo",
    "gpt-4.1": "turbo",
    "gpt-4.1-mini": "turbo",
    "gpt-4.1-nano": "turbo",
    "gpt-4": "turbo",
    "gpt-3.5-turbo": "turbo",
    "sonar": "turbo",
    "sonar-small": "turbo",
    "sonar-medium": "turbo",
    "sonar-pro": "pplx_pro",
    "sonar-reasoning": "pplx_reasoning",
    "pplx": "pplx_pro",
}
KNOWN_MODELS = [
    "turbo",
    "pplx_pro",
    "pplx_pro_upgraded",
    "pplx_alpha",
    "pplx_beta",
    "pplx_reasoning",
    "pplx_research",
    "pplx_research_upgraded",
    "gpt41",
    "gpt5",
    "gpt5_thinking",
    "o3",
    "o3pro",
    "o4mini",
    "o1",
    "claude2",
    "claude37sonnetthinking",
    "claude40opus",
    "claude40opusthinking",
    "claude41opusthinking",
    "claude45sonnet",
    "claude45sonnetthinking",
    "experimental",
    "grok",
    "grok4",
    "gemini2flash",
    "gemini",
    "mistral",
    "llama_x_large",
    "r1",
]

CHUNK_RE = re.compile(r"^/chunks/(\d+)$")
TEXT_PAYLOAD_CHUNK_RE = re.compile(r"/text_payload/chunks/(\d+)$")


def _iter_text_payloads(val: object) -> Iterator[tuple[str | None, list[Any] | None]]:
    """Find text payloads nested anywhere in a patch value.

    Instant answers arrive as one nested workflow tree.

    Args:
        val: Patch value to search recursively.

    Yields:
        Tuples of decoded text (when present) and chunk list (when present).

    """
    if isinstance(val, dict):
        tp = val.get("text_payload")
        if isinstance(tp, dict):
            text = tp.get("text")
            text_str = text if isinstance(text, str) else None
            chunks = tp.get("chunks")
            chunks_list: list[Any] | None = chunks if isinstance(chunks, list) else None
            yield text_str, chunks_list
            return
        for v in val.values():
            yield from _iter_text_payloads(v)
    elif isinstance(val, list):
        for v in val:
            yield from _iter_text_payloads(v)


@dataclass
class ThreadState:
    """Server-side Perplexity conversation handle plus cached metadata."""

    backend_uuid: str | None = None
    read_write_token: str | None = None
    slug: str | None = None
    title: str | None = None
    model: str | None = None
    mode: str = "copilot"
    account_id: int | None = None
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Refresh the last-used timestamp."""
        self.last_used = time.time()


@dataclass
class AskEvent:
    """One streamed ask outcome: meta/text/sources/related/error/done."""

    kind: str  # "meta" | "text" | "sources" | "related" | "error" | "done"
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AskError(Exception):
    """Failed ask with a machine-readable code and retry hint."""

    code: str
    message: str
    retryable: bool = False  # True => safe to retry as a brand-new thread


def resolve_model(requested: str | None) -> str:
    """Map a requested model name to a Perplexity model preference.

    Args:
        requested: OpenAI-style model name, alias, or Perplexity slug.

    Returns:
        Perplexity model preference slug, defaulting to ``"turbo"``.

    """
    if not requested:
        return "turbo"
    name = requested.strip()
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    if name in KNOWN_MODELS:
        return name
    if name.startswith("pplx-"):
        return name[5:]
    # Unknown names default to the free-tier model rather than erroring.
    return "turbo"


def resolve_mode(requested: str | None, _model: str) -> str:
    """Derive the search mode from model hints.

    Args:
        requested: Requested model name, possibly suffixed with a mode hint.
        _model: Resolved model preference, reserved for future
            model-dependent mode defaults; currently unused.

    Returns:
        Mode slug, defaulting to ``"copilot"``.

    """
    if requested:
        low = requested.lower()
        for m in MODES:
            if low.endswith(f"-{m}") or low == m:
                return m
    return "copilot"


_IMAGE_PART_TYPES = {"input_image", "image"}
_FILE_PART_TYPES = {"input_file", "file"}


def _url_from_value(value: dict[str, Any] | str | None) -> str | None:
    """Extract a URL string from a plain or ``{"url": ...}`` value.

    Args:
        value: Raw payload value.

    Returns:
        Non-empty URL string, or None when absent.

    """
    if isinstance(value, dict):
        nested = value.get("url")
        if isinstance(nested, str):
            return nested or None
        return None
    if isinstance(value, str):
        return value or None
    return None


def _data_url_mime(url: str, default: str) -> str:
    """Return the MIME type declared by a data URL.

    Args:
        url: Attachment URL, possibly a data: URL.
        default: MIME type used when the URL declares none.

    Returns:
        Declared MIME type for data: URLs, else ``default``.

    """
    if url.startswith("data:"):
        return url.split(";", maxsplit=1)[0][5:] or default
    return default


def build_image_attachment(url: str) -> dict[str, Any]:
    """Build a Perplexity image attachment for a URL.

    Args:
        url: Remote or data: image URL.

    Returns:
        Perplexity image attachment object.

    """
    mime = _data_url_mime(url, "image/png") if url.startswith("data:") else "image/jpeg"
    return {
        "type": "image",
        "content_type": mime,
        "name": "image",
        "url": url,
        "size": len(url),
    }


def build_file_attachment(url: str, name: str) -> dict[str, Any]:
    """Build a Perplexity file attachment for a URL.

    Args:
        url: Remote or data: file URL.
        name: Attachment filename.

    Returns:
        Perplexity file attachment object.

    """
    return {
        "type": "file",
        "content_type": _data_url_mime(url, "application/octet-stream"),
        "name": name,
        "url": url,
        "size": len(url),
    }


def _optional_image_attachment(
    value: dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    """Build an image attachment from a raw URL value.

    Args:
        value: Plain URL string or ``{"url": ...}`` mapping.

    Returns:
        Perplexity image attachment, or None when no URL is present.

    """
    url = _url_from_value(value)
    if url is None:
        return None
    return build_image_attachment(url)


def _file_attachment_from_part(part: dict[str, Any]) -> dict[str, Any] | None:
    """Build a file attachment from an OpenAI file content part.

    Args:
        part: OpenAI ``input_file``/``file`` content part.

    Returns:
        Perplexity file attachment, or None when no file URL is present.

    """
    raw_name = part.get("filename") or part.get("name") or "file"
    name = raw_name if isinstance(raw_name, str) and raw_name else "file"
    url_value = part.get("file_url") or part.get("url")
    data = part.get("file_data")  # base64
    if isinstance(data, str) and data:
        raw_mime = part.get("content_type")
        mime_value = (
            raw_mime
            if isinstance(raw_mime, str) and raw_mime
            else "application/octet-stream"
        )
        url_value = f"data:{mime_value};base64,{data}"
    url = _url_from_value(url_value)
    if url is None:
        return None
    return build_file_attachment(url, name)


def build_attachment(part: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize an OpenAI content part into a Perplexity attachment.

    Args:
        part: OpenAI chat content part (image or file).

    Returns:
        Perplexity attachment object, or None for unsupported parts.

    """
    raw_type = part.get("type", "")
    ptype = raw_type if isinstance(raw_type, str) else ""
    if ptype == "image_url":
        return _optional_image_attachment(part.get("image_url"))
    if ptype in _IMAGE_PART_TYPES:
        return _optional_image_attachment(part.get("image_url") or part.get("url"))
    if ptype in _FILE_PART_TYPES:
        return _file_attachment_from_part(part)
    return None


_TEXT_MIMES = {
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-python",
    "application/x-sh",
    "application/x-yaml",
    "application/sql",
    "application/csv",
}
_TEXT_EXT = {
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".json",
    ".md",
    ".markdown",
    ".csv",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".html",
    ".htm",
    ".css",
    ".log",
    ".tex",
    ".r",
    ".rb",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".php",
    ".lua",
    ".pl",
    ".kt",
    ".kts",
    ".swift",
    ".scala",
    ".dockerfile",
    ".env",
    ".gitignore",
    ".lock",
    ".diff",
    ".patch",
    ".ipynb",
    ".gradle",
    ".properties",
    ".proto",
    ".vue",
    ".svelte",
    ".astro",
    ".rss",
    ".atom",
}

TEXT_APPEND_CAP = 64 * 1024  # per file
TEXT_APPEND_TOTAL_CAP = 32 * 1024  # total injected text into the query


def is_text_like(att: dict[str, Any]) -> bool:
    """Check whether an attachment looks like decodable text.

    Args:
        att: Perplexity attachment object.

    Returns:
        True when the MIME type or filename extension is textual.

    """
    raw_mime = att.get("content_type", "")
    mime = raw_mime if isinstance(raw_mime, str) else ""
    raw_name = att.get("name", "")
    name = raw_name if isinstance(raw_name, str) else ""
    if any(mime.startswith(p) for p in _TEXT_MIMES):
        return True
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return ext in _TEXT_EXT


def extract_attachment_text(att: dict[str, Any]) -> str | None:
    """Decode bounded text content from a data-URL attachment.

    Args:
        att: Perplexity attachment object.

    Returns:
        Decoded UTF-8 text, or None when the attachment is not a data:
        URL or its payload is not decodable.

    """
    raw_url = att.get("url", "")
    if not isinstance(raw_url, str):
        return None
    if not raw_url.startswith("data:"):
        return None
    head, _, payload = raw_url.partition(",")
    # `payload` is `str` at runtime (partition of `str`); str() pins that
    # for the checker so the decoder below stays precisely typed.
    encoded = str(payload[:TEXT_APPEND_CAP])
    try:
        raw = base64.b64decode(encoded) if ";base64" in head else encoded.encode()
        decoded = raw.decode("utf-8", "replace")
    except (ValueError, binascii.Error):
        log.debug("Skipping undecodable attachment payload", exc_info=True)
        return None
    else:
        return decoded


@dataclass
class AskSpec:
    """Inputs for one Perplexity ask payload."""

    query: str
    frontend_uuid: str
    frontend_context_uuid: str
    model: str
    mode: str
    attachments: list[dict[str, Any]] | None = None
    thread: ThreadState | None = None
    system: str | None = None
    timezone: str = "UTC"


def build_payload(spec: AskSpec) -> dict[str, Any]:
    """Build the JSON body for POST /rest/sse/perplexity_ask.

    Args:
        spec: Ask parameters (query, model, mode, thread, and options).

    Returns:
        Mapping with ``params`` and ``query_str`` keys for the request.

    """
    query = f"{spec.system}\n\n{spec.query}" if spec.system else spec.query
    params: dict[str, Any] = {
        "attachments": spec.attachments or [],
        "language": "en-US",
        "timezone": spec.timezone,
        "search_focus": "internet",
        "sources": ["web"],
        "search_recency_filter": None,
        "frontend_uuid": spec.frontend_uuid,
        "mode": spec.mode,
        "model_preference": spec.model,
        "is_related_query": False,
        "is_sponsored": False,
        "frontend_context_uuid": spec.frontend_context_uuid,
        "prompt_source": "user",
        "query_source": "followup" if spec.thread else "home",
        "is_incognito": False,
        "time_from_first_type": 0,
        "local_search_enabled": False,
        "use_schematized_api": True,
        "send_back_text_in_streaming_api": False,
        "supported_block_use_cases": [
            "answer_modes",
            "media_items",
            "knowledge_cards",
            "inline_entity_cards",
            "place_widgets",
            "finance_widgets",
            "sports_widgets",
            "news_widgets",
            "shopping_widgets",
            "jobs_widgets",
            "search_result_widgets",
            "inline_images",
            "inline_assets",
            "placeholder_cards",
            "diff_blocks",
            "inline_knowledge_cards",
            "entity_group_v2",
            "refinement_filters",
            "canvas_mode",
            "maps_preview",
            "answer_tabs",
            "price_comparison_widgets",
            "preserve_latex",
            "generic_onboarding_widgets",
            "in_context_suggestions",
            "pending_followups",
            "inline_claims",
            "unified_assets",
            "workflow_steps",
            "workflow_widgets",
            "navigation_results",
            "background_agents",
        ],
        "client_coordinates": None,
        "mentions": [],
        "dsl_query": query,
        "skip_search_enabled": True,
        "is_nav_suggestions_disabled": False,
        "source": "default",
        "always_search_override": False,
        "override_no_search": False,
        "should_ask_for_mcp_tool_confirmation": True,
        "supports_tool_approval_modal": True,
        "browser_agent_allow_once_from_toggle": False,
        "force_enable_browser_agent": False,
        "supported_features": ["browser_agent_permission_banner_v1.1"],
        "extended_context": False,
        "version": "2.18",
    }
    if spec.thread is not None and spec.thread.backend_uuid:
        params.update(
            {
                "last_backend_uuid": spec.thread.backend_uuid,
                "read_write_token": spec.thread.read_write_token,
                "followup_source": "link",
            },
        )
    return {"params": params, "query_str": query}


class _TextAccumulator:
    """Reassemble a streamed answer from segment patches.

    Segments arrive as per-index patches while an authoritative full text
    may arrive separately; emission waits for a contiguous run from index
    0 so deltas always concatenate to the final answer.
    """

    def __init__(self) -> None:
        """Initialize empty segment state."""
        self._segments: dict[int, str] = {}
        self._next_index = 0
        self._pending: list[str] = []
        self._full_text: str | None = None
        self._emitted = ""

    def take(self, index: int, value: str) -> None:
        """Buffer segment ``index`` and release newly contiguous runs."""
        if not value:
            return
        self._segments[index] = value
        while self._next_index in self._segments:
            segment = self._segments.pop(self._next_index)
            self._next_index += 1
            self._pending.append(segment)
            self._emitted += segment

    def note_full_text(self, text: str) -> None:
        """Record authoritative full text for later reconciliation."""
        self._full_text = text

    def drain(self) -> str:
        """Return newly contiguous text reconciled with the full text.

        Returns:
            Text delta to emit, possibly empty.

        """
        delta = "".join(self._pending)
        self._pending.clear()
        full_text = self._full_text
        if full_text is None:
            return delta
        if full_text.startswith(self._emitted):
            tail = full_text[len(self._emitted) :]
            if tail and not delta.endswith(tail):
                delta += tail
        elif not full_text.startswith(delta):
            delta += full_text
        self._emitted = full_text
        self._full_text = None
        return delta


@dataclass
class AskOptions:
    """Optional knobs for one :meth:`PerplexityClient.ask` call."""

    attachments: list[dict[str, Any]] | None = None
    thread: ThreadState | None = None
    system: str | None = None
    timezone: str = "UTC"


def _http_error_event(status_code: int) -> AskEvent:
    """Build the error event for a non-200 ask response.

    Args:
        status_code: HTTP status returned by Perplexity.

    Returns:
        Error event, retryable for 5xx statuses.

    """
    return AskEvent(
        "error",
        data={
            "error_code": f"HTTP_{status_code}",
            "message": f"Perplexity returned HTTP {status_code}",
            "retryable": status_code >= HTTP_SERVER_ERROR_MIN,
        },
    )


def _sse_payload_text(raw: bytes | str) -> str | None:
    """Extract the payload of one SSE data line.

    Args:
        raw: Raw stream line, bytes or text.

    Returns:
        Stripped payload, or None for keepalives and empty lines.

    """
    line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    return payload or None


def _decode_sse_payload(payload: str) -> dict[str, Any] | None:
    """Decode one SSE JSON payload.

    Args:
        payload: Stripped SSE data payload.

    Returns:
        Event mapping, or None when undecodable or not an object.

    """
    try:
        decoded = json.loads(payload)
    except ValueError:
        log.debug("Skipping undecodable SSE payload", exc_info=True)
        return None
    else:
        if not isinstance(decoded, dict):
            return None
        return decoded


def _is_ask_error(event_data: dict[str, Any]) -> bool:
    """Check whether a decoded SSE payload carries an upstream error.

    Args:
        event_data: Decoded SSE payload.

    Returns:
        True when ``error_code`` is present.

    """
    return bool(event_data.get("error_code"))


def _ask_error_event(event_data: dict[str, Any]) -> AskEvent:
    """Build the error event for an upstream failure payload.

    Args:
        event_data: Decoded SSE payload with ``error_code``.

    Returns:
        Error event for the upstream failure.

    """
    code = event_data["error_code"]
    message = event_data.get("text") or "Perplexity upstream error"
    return AskEvent(
        "error",
        data={
            "error_code": code,
            "message": message,
            "retryable": code == "GENERIC_FAILED_RESPONSE",
        },
    )


def _adopt_initial_thread_metadata(
    thread: ThreadState,
    event_data: dict[str, Any],
) -> None:
    """Adopt stream metadata fields that are still unset on the thread."""
    if event_data.get("backend_uuid") and not thread.backend_uuid:
        thread.backend_uuid = event_data["backend_uuid"]
    if event_data.get("read_write_token") and not thread.read_write_token:
        thread.read_write_token = event_data["read_write_token"]
    if event_data.get("thread_url_slug") and not thread.slug:
        thread.slug = event_data["thread_url_slug"]
    if event_data.get("thread_title") and not thread.title:
        thread.title = event_data["thread_title"]
    # NB: the stream key is `display_model` while the handle field is
    # `model`, so the original lookup never matched and the value was
    # always adopted; preserve that exactly.
    if event_data.get("display_model"):
        thread.model = event_data["display_model"]
    if event_data.get("mode") and not thread.mode:
        thread.mode = event_data["mode"]


def _refresh_thread_metadata(thread: ThreadState, event_data: dict[str, Any]) -> None:
    """Overwrite thread handle fields with authoritative stream values."""
    if event_data.get("backend_uuid"):
        thread.backend_uuid = event_data["backend_uuid"]
    if event_data.get("read_write_token"):
        thread.read_write_token = event_data["read_write_token"]
    if event_data.get("thread_url_slug"):
        thread.slug = event_data["thread_url_slug"]
    if event_data.get("thread_title"):
        thread.title = event_data["thread_title"]


def _adopt_thread_metadata(thread: ThreadState, event_data: dict[str, Any]) -> None:
    """Copy stream metadata onto the thread handle in place."""
    _adopt_initial_thread_metadata(thread, event_data)
    _refresh_thread_metadata(thread, event_data)


def _sources_event(block: dict[str, Any]) -> AskEvent | None:
    """Build a sources event from a sources answer-mode block.

    Args:
        block: Stream block mapping.

    Returns:
        Sources event, or None for other blocks or empty results.

    """
    if block.get("intended_usage") != "sources_answer_mode":
        return None
    mode_block = block.get("sources_mode_block") or {}
    if not isinstance(mode_block, dict):
        return None
    results = mode_block.get("web_results") or []
    if not results:
        return None
    return AskEvent("sources", data={"results": results})


def _accumulate_markdown_value(
    value: dict[str, Any],
    accumulator: _TextAccumulator,
) -> None:
    """Accumulate text from a markdown diff-patch value mapping."""
    chunks = value.get("chunks")
    if isinstance(chunks, list):
        for index, chunk in enumerate(chunks):
            accumulator.take(index, chunk if isinstance(chunk, str) else "")
    elif isinstance(value.get("answer"), str):
        accumulator.take(0, value["answer"])


def _accumulate_markdown_patch(
    patch: dict[str, Any],
    accumulator: _TextAccumulator,
) -> None:
    """Accumulate text from one markdown diff patch."""
    if not isinstance(patch, dict):
        return
    path = patch.get("path")
    if path == "/progress":
        return
    value = patch.get("value")
    if isinstance(value, dict):
        _accumulate_markdown_value(value, accumulator)
    elif isinstance(value, str):
        match = CHUNK_RE.fullmatch(path or "")
        if match:
            accumulator.take(int(match.group(1)), value)


def _accumulate_text_payloads(
    value: dict[str, Any],
    accumulator: _TextAccumulator,
) -> None:
    """Accumulate text from nested workflow text payloads."""
    for full_text, chunks in _iter_text_payloads(value):
        if isinstance(full_text, str) and full_text:
            accumulator.note_full_text(full_text)
        if chunks:
            for index, chunk in enumerate(chunks):
                accumulator.take(index, chunk if isinstance(chunk, str) else "")


def _accumulate_workflow_patch(
    patch: dict[str, Any],
    accumulator: _TextAccumulator,
) -> None:
    """Accumulate text from one workflow diff patch."""
    if not isinstance(patch, dict):
        return
    path = patch.get("path") or ""
    value = patch.get("value")
    if isinstance(value, dict):
        _accumulate_text_payloads(value, accumulator)
    elif isinstance(value, str):
        if path.endswith("/text_payload/text"):
            accumulator.note_full_text(value)
        else:
            match = TEXT_PAYLOAD_CHUNK_RE.search(path)
            if match:
                accumulator.take(int(match.group(1)), value)


def _accumulate_block_text(
    block: dict[str, Any],
    accumulator: _TextAccumulator,
) -> None:
    """Accumulate answer text from one stream block's diff patches."""
    diff_block = block.get("diff_block") or {}
    if not isinstance(diff_block, dict):
        return
    field = diff_block.get("field")
    patches = diff_block.get("patches")
    if not isinstance(patches, list):
        return
    if field == "markdown_block":
        for patch in patches:
            _accumulate_markdown_patch(patch, accumulator)
    elif field == "workflow_block":
        for patch in patches:
            _accumulate_workflow_patch(patch, accumulator)


def _collect_text_events(
    event_data: dict[str, Any],
    accumulator: _TextAccumulator,
) -> list[AskEvent]:
    """Collect sources and text events from a block-carrying payload.

    Args:
        event_data: Decoded SSE payload containing ``blocks``.
        accumulator: Segment buffer reassembling streamed text.

    Returns:
        Sources events plus at most one text delta event.

    """
    events: list[AskEvent] = []
    blocks = event_data.get("blocks", [])
    if not isinstance(blocks, list):
        blocks = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        sources = _sources_event(block)
        if sources is not None:
            events.append(sources)
            continue
        _accumulate_block_text(block, accumulator)
    delta = accumulator.drain()
    if delta:
        events.append(AskEvent("text", text=delta))
    return events


def _related_event(event_data: dict[str, Any]) -> AskEvent | None:
    """Build a related-queries event from a payload.

    Args:
        event_data: Decoded SSE payload.

    Returns:
        Related event, or None when no related items are present.

    """
    raw_items = event_data.get("related_query_items")
    if not isinstance(raw_items, list):
        return None
    items = [
        item.get("text", "")
        for item in raw_items
        if isinstance(item, dict) and item.get("text")
    ]
    if not items:
        return None
    return AskEvent("related", data={"items": items})


def _handle_ask_event(
    event_data: dict[str, Any],
    thread: ThreadState,
    accumulator: _TextAccumulator,
) -> list[AskEvent]:
    """Process one decoded SSE payload into events.

    Args:
        event_data: Decoded SSE payload.
        thread: Conversation handle updated in place.
        accumulator: Segment buffer reassembling streamed text.

    Returns:
        Events to emit for this payload.

    """
    events: list[AskEvent] = []
    if event_data.get("final_sse_message") is False or "blocks" in event_data:
        _adopt_thread_metadata(thread, event_data)
        events.extend(_collect_text_events(event_data, accumulator))
    related = _related_event(event_data)
    if related is not None:
        events.append(related)
    return events


async def _drain_ask_response(
    response: Response,
    thread: ThreadState,
    accumulator: _TextAccumulator,
) -> AsyncIterator[AskEvent]:
    """Consume the SSE stream and yield ask events.

    Args:
        response: Open Perplexity SSE response.
        thread: Conversation handle updated in place.
        accumulator: Segment buffer reassembling streamed text.

    Yields:
        AskEvent: Text, sources, related, error, or final done events.

    """
    if response.status_code != HTTP_OK:
        yield _http_error_event(response.status_code)
        return
    async for raw in response.aiter_lines():
        payload = _sse_payload_text(raw)
        if payload is None:
            continue
        event_data = _decode_sse_payload(payload)
        if event_data is None:
            continue
        if _is_ask_error(event_data):
            yield _ask_error_event(event_data)
            return
        for event in _handle_ask_event(event_data, thread, accumulator):
            yield event
        if event_data.get("final_sse_message") is True:
            break
    thread.touch()
    yield AskEvent("done", data={"thread": thread})


class PerplexityClient:
    """Perplexity transport bound to one account's cookies."""

    def __init__(
        self,
        cookies: dict[str, str],
        *,
        impersonate: str = "chrome",
        max_concurrent: int = 2,
        timeout: float = 180.0,
    ) -> None:
        """Cache account cookies and concurrency budget.

        Args:
            cookies: Session cookies for perplexity.ai.
            impersonate: curl_cffi browser impersonation profile.
            max_concurrent: Maximum concurrent asks on this client.
            timeout: Default request timeout in seconds.

        """
        self._cookies = dict(cookies)
        self._impersonate = impersonate
        self._timeout = timeout
        self._session: AsyncSession[Response] | None = None
        self._slots = asyncio.Semaphore(max(max_concurrent, 1))

    async def session(self) -> AsyncSession[Response]:
        """Return the cached session, creating it on first use.

        Returns:
            Shared async session for this client's cookies.

        """
        if self._session is None:
            self._session = AsyncSession(
                impersonate=self._impersonate,
                headers={
                    "User-Agent": UA,
                    "Origin": BASE_URL,
                    "Accept": "text/event-stream",
                    "x-perplexity-request-reason": "perplexity-query-state-provider",
                },
                cookies=self._cookies,
                timeout=self._timeout,
            )
        return self._session

    async def close(self) -> None:
        """Drop the cached session, ignoring teardown errors."""
        if self._session is not None:
            try:
                await self._session.close()
            except (OSError, RuntimeError):
                log.debug("Ignoring Perplexity session close error", exc_info=True)
            self._session = None

    async def _fetch_session_user(self) -> str | None:
        """Fetch the session user id without swallowing transport errors.

        Returns:
            Authenticated user id, or None when the session is invalid.

        """
        session = await self.session()
        response = await session.get(SESSION_URL, timeout=15)
        if response.status_code != HTTP_OK:
            return None
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        user = payload.get("user")
        if not isinstance(user, dict):
            return None
        user_id = user.get("id")
        return user_id if isinstance(user_id, str) else None

    async def check_session(self) -> str | None:
        """Return the user id when the cookie session is valid.

        Returns:
            Authenticated user id, or None when the session is
            invalid or unreachable.

        """
        try:
            user_id = await self._fetch_session_user()
        except (OSError, ValueError):
            log.debug("Perplexity session check failed", exc_info=True)
            return None
        else:
            return user_id

    async def _fetch_quota(self) -> bool | None:
        """Fetch free-query availability without swallowing transport errors.

        Returns:
            True/False when known, None when the payload is unexpected.

        """
        session = await self.session()
        response = await session.get(RATE_LIMIT_URL, timeout=15)
        if response.status_code != HTTP_OK:
            return None
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        free_queries = payload.get("free_queries") or {}
        if not isinstance(free_queries, dict):
            return None
        available = free_queries.get("available")
        return bool(available)

    async def quota_available(self) -> bool | None:
        """Return free-query availability when the status endpoint answers.

        Returns:
            True/False when known, None when the endpoint is unreachable.

        """
        try:
            available = await self._fetch_quota()
        except (OSError, ValueError):
            log.debug("Perplexity quota check failed", exc_info=True)
            return None
        else:
            return available

    async def ask(
        self,
        query: str,
        *,
        model: str,
        mode: str,
        options: AskOptions | None = None,
    ) -> AsyncIterator[AskEvent]:
        """Stream one Perplexity ask.

        The SSE response is consumed with an explicitly managed stream
        (rather than ``async with``) because yielding inside the context
        manager would skip its cleanup when the caller stops early.

        Args:
            query: User query text.
            model: Resolved Perplexity model preference.
            mode: Resolved search mode.
            options: Optional attachments, thread, system prompt, timezone.

        Yields:
            AskEvent: Text, sources, related, error, or final done events.

        """
        opts = options if options is not None else AskOptions()
        if opts.thread is not None:
            thread = opts.thread
        else:
            thread = ThreadState(model=model, mode=mode)
        body = build_payload(
            AskSpec(
                query=query,
                frontend_uuid=str(uuid.uuid4()),
                frontend_context_uuid=str(uuid.uuid4()),
                model=model,
                mode=mode,
                attachments=opts.attachments,
                thread=thread if thread.backend_uuid else None,
                system=opts.system,
                timezone=opts.timezone,
            ),
        )
        referer = f"{BASE_URL}/search/{thread.slug}" if thread.slug else f"{BASE_URL}/"
        headers = {"Referer": referer, "x-request-id": str(uuid.uuid4())}
        accumulator = _TextAccumulator()
        await self._slots.acquire()
        try:
            session = await self.session()
            # Same as `async with session.stream(...)` (request(stream=True) +
            # `aclose()` in `finally`), but yielding inside `async with` would
            # skip the context manager's cleanup when the caller stops early.
            response = await session.request(
                "POST",
                ASK_URL,
                json=body,
                headers=headers,
                stream=True,
            )
            try:
                async for event in _drain_ask_response(
                    response,
                    thread,
                    accumulator,
                ):
                    yield event
            finally:
                await response.aclose()
        finally:
            self._slots.release()
