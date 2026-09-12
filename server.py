# Copyright 2026 perplexity-to-openai contributors

"""OpenAI-compatible API over Perplexity (Chat Completions + Responses).

FastAPI app on port 64130. No browser: transport is curl_cffi + session
cookies. Multi-turn: Perplexity keeps thread state server-side; the proxy
maps conversation identity (message-history hash or previous_response_id)
to Perplexity's backend_uuid/read_write_token handle.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from curl_cffi.requests import AsyncSession
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from account_pool import Account, AccountPool
from db_store import Store
from pplx_transport import (
    TEXT_APPEND_TOTAL_CAP,
    AskError,
    AskEvent,
    AskOptions,
    ThreadState,
    build_attachment,
    extract_attachment_text,
    is_text_like,
    resolve_mode,
    resolve_model,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("pplx.server")

ACCOUNTS_FILE = os.environ.get("PPLX_ACCOUNTS", "accounts.txt")
API_KEY = os.environ.get("PPLX_API_KEY")  # optional bearer enforcement
MAX_CONCURRENT = int(os.environ.get("PPLX_MAX_CONCURRENT", "2"))
THREAD_LRU = 1024
RESPONSE_LRU = 512
FETCH_SIZE_CAP = 20 * 1024 * 1024
HTTP_TOO_MANY_REQUESTS = 429
TZ = os.environ.get("PPLX_TIMEZONE", "UTC")
INCLUDE_SOURCES = os.environ.get("PPLX_INCLUDE_SOURCES", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pool = AccountPool(ACCOUNTS_FILE, max_concurrent=MAX_CONCURRENT)
store = Store()


@dataclass(frozen=True)
class AskSpec:
    """Parameters for one Perplexity ask.

    Attributes:
        model: Resolved Perplexity model slug.
        mode: Search mode (copilot, concise, ...).
        attachments: Prepared Perplexity attachments, if any.
        thread: Existing thread handle for follow-ups, if any.
        system: System prompt, if any.

    """

    model: str
    mode: str
    attachments: list[dict[str, Any]] | None
    thread: ThreadState | None
    system: str | None


@dataclass(frozen=True)
class ChatStreamSpec:
    """Canvas for streaming one chat completion.

    Attributes:
        completion_id: OpenAI completion id echoed in every chunk.
        model: Model name echoed in every chunk.
        created: Creation epoch echoed in every chunk.
        prompt_tokens: Estimated prompt tokens for the usage trailer.
        include_usage: Whether to emit the trailing usage chunk.
        final_text: Renders the appended tail text (sources appendix).

    """

    completion_id: str
    model: str
    created: int
    prompt_tokens: int
    include_usage: bool
    final_text: Callable[[], str]


@dataclass(frozen=True)
class ResponsesStreamSpec:
    """Canvas for streaming one Responses API result.

    Attributes:
        model: Model name echoed in lifecycle events.
        request: Validated request payload echoed in lifecycle events.
        prompt_tokens: Estimated prompt tokens for the usage block.
        resp_id: Response id, generated when None.
        msg_id: Message item id, generated when None.
        previous_id: Previous response id being continued, if any.
        final_text: Renders appended tail text from final sources.

    """

    model: str
    request: dict[str, Any]
    prompt_tokens: int
    resp_id: str | None
    msg_id: str | None
    previous_id: str | None
    final_text: Callable[[list[dict[str, Any]]], str]


@dataclass(frozen=True)
class ResponseSpec:
    """Ingredients for one non-streaming Responses object.

    Attributes:
        model: Model name for the response.
        text: Assistant text for the output item.
        sources: Upstream sources for URL citations.
        request: Validated request payload echoed back.
        prompt_tokens: Estimated prompt tokens.
        completion_tokens: Estimated completion tokens.
        previous_id: Previous response id being continued, if any.
        resp_id: Response id, generated when None.

    """

    model: str
    text: str
    sources: list[dict[str, Any]]
    request: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    previous_id: str | None = None
    resp_id: str | None = None


@dataclass
class _StreamCapture:
    """Mutable capture of thread, text, and sources seen while streaming."""

    thread: ThreadState | None = None
    text_parts: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _MessageAccumulator:
    """Collects query text, history items, attachments, and injections."""

    query_parts: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    injections: list[str] = field(default_factory=list)

    def add_part(self, part: object, *, last: bool, role: str) -> None:
        """Fold one content part into query text, history, or attachments.

        Text-like file attachments for the newest message are injected
        into the query instead of being uploaded.

        Args:
            part: One message content part.
            last: Whether the part belongs to the newest message.
            role: Role used for history items from older messages.

        """
        text = _text_of_part(part)
        if text:
            if last:
                self.query_parts.append(text)
            else:
                self.history.append({"role": role, "content": text})
        if not isinstance(part, dict):
            return
        # Re-key through str keys: build_attachment only performs
        # str-literal lookups, so dropping non-str keys preserves
        # behavior while satisfying dict[str, Any] invariance.
        str_part: dict[str, Any] = {k: v for k, v in part.items() if isinstance(k, str)}
        att = build_attachment(str_part)
        if att is None or not last:
            return
        injection = _attachment_injection(att)
        if injection is not None:
            self.injections.append(injection)
            return
        self.attachments.append(att)


@dataclass
class _AttemptOutcome:
    """Mutable result of one ask attempt consumed by the retry loop."""

    thread: ThreadState | None
    attempt: int = 0
    got_text: bool = False
    retry_requested: bool = False


async def _register_thread(key: str, thread_state: ThreadState) -> ThreadState:
    """Persist a thread handle under a conversation key.

    Args:
        key: Conversation-key hash for the thread.
        thread_state: Thread handle to store.

    Returns:
        The stored thread handle.

    """
    await asyncio.to_thread(store.save_thread, key, thread_state, THREAD_LRU)
    return thread_state


async def _register_answer(answer: str, thread_state: ThreadState) -> None:
    """Associate a thread with its answer text.

    The client's echo of that answer on the next turn then identifies
    the same Perplexity conversation.

    Args:
        answer: Assistant text to anchor the thread to.
        thread_state: Thread handle to store under the anchor.

    """
    if answer:
        anchor = hashlib.sha256(answer.encode()).hexdigest()
        await asyncio.to_thread(
            store.save_thread_by_answer,
            anchor,
            thread_state,
            THREAD_LRU,
        )


def _last_user_index(messages: list[Any]) -> int | None:
    """Return the index of the last user/system/developer message.

    Args:
        messages: Raw OpenAI message dicts, oldest first.

    Returns:
        The index, or None when no user message is present.

    """
    last: int | None = None
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") in {
            "user",
            "system",
            "developer",
        }:
            last = index
    return last


async def _lookup_thread(
    messages: list[Any],
) -> tuple[ThreadState | None, str | None]:
    """Find the Perplexity thread for a messages list.

    Strategy: the OpenAI SDK resends the full history; the message right
    before the newest user message is our previous assistant answer (echoed
    verbatim). Match on its text hash first; fall back to a hash of the
    prefix for clients that do not echo assistant messages.

    Args:
        messages: Raw OpenAI message dicts, oldest first.

    Returns:
        A (thread, answer_anchor) pair; either entry may be None.

    """
    last_user = _last_user_index(messages)
    if last_user is None:
        return None, None
    prev: Any = messages[last_user - 1] if last_user > 0 else None
    if not isinstance(prev, dict) or prev.get("role") != "assistant":
        return None, None
    content: Any = prev.get("content")
    if isinstance(content, list):
        content = "".join(_text_of_part(p) for p in content)
    if isinstance(content, str) and content:
        anchor = hashlib.sha256(content.encode()).hexdigest()
        found_by_answer = await asyncio.to_thread(
            store.get_thread_by_answer,
            anchor,
        )
        if isinstance(found_by_answer, ThreadState):
            return found_by_answer, anchor
    history_key = _conversation_key(messages[:last_user])
    found_by_key = await asyncio.to_thread(store.get_thread, history_key)
    if isinstance(found_by_key, ThreadState):
        return found_by_key, None
    return None, None


async def _resolve_chat_thread(
    messages: list[Any],
    history: list[dict[str, Any]],
) -> ThreadState | None:
    """Find the stored Perplexity thread for a message history.

    Tries the answer-anchor lookup first, then the conversation key.

    Args:
        messages: Raw OpenAI message dicts, oldest first.
        history: Normalized history items for the key lookup.

    Returns:
        The stored thread handle, or None for a new conversation.

    """
    thread, _anchor = await _lookup_thread(messages)
    if thread is not None:
        return thread
    stored = await asyncio.to_thread(store.get_thread, _conversation_key(history))
    return stored if isinstance(stored, ThreadState) else None


def _stored_thread(prev: dict[str, Any]) -> ThreadState | None:
    """Extract the stored Perplexity thread from a saved response.

    Args:
        prev: Saved response payload including a thread handle.

    Returns:
        The thread handle, or None when absent or malformed.

    """
    raw = prev.get("_thread")
    return raw if isinstance(raw, ThreadState) else None


def _conversation_key(history: list[Any]) -> str:
    """Hash everything except the last user message to a thread id.

    Args:
        history: Normalized history items excluding the newest message.

    Returns:
        The stable hex conversation key.

    """
    h = hashlib.sha256()
    for item in history:
        h.update(json.dumps(item, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _now() -> int:
    """Return the current epoch seconds.

    Returns:
        Integer seconds since the epoch.

    """
    return int(time.time())


def _est_tokens(text: str) -> int:
    """Roughly estimate token count as one token per four characters.

    Args:
        text: Text to estimate.

    Returns:
        At least one token.

    """
    return max(1, len(text) // 4)


def _prompt_tokens(query: str, history: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens for a query plus its history text.

    Args:
        query: Final user query text.
        history: Normalized history items.

    Returns:
        Estimated prompt token count.

    """
    return _est_tokens(query) + sum(_est_tokens(_history_text(h)) for h in history)


def _history_text(h: dict[str, Any]) -> str:
    """Extract countable text from a history item.

    The item content may be a string or a list of parts.

    Args:
        h: Normalized history item.

    Returns:
        The countable text, or an empty string.

    """
    content: Any = h.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # str() pins the element type for the checker; the isinstance
        # filter guarantees every element already is a str, so this
        # is the identity at runtime.
        return " ".join(str(p) for p in content if isinstance(p, str))
    return ""


# --------------------------------------------------------------------------
# Request normalization
# --------------------------------------------------------------------------


def _text_of_part(part: object) -> str:
    """Return the text carried by one message content part.

    Args:
        part: A string part or a typed content-part dict.

    Returns:
        The part text, or an empty string for non-text parts.

    """
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    raw_type = part.get("type", "")
    t = raw_type if isinstance(raw_type, str) else ""
    if t in {"text", "input_text"}:
        raw_text = part.get("text", "")
        return raw_text if isinstance(raw_text, str) else ""
    return ""


def _iter_parts(content: object) -> list[Any]:
    """Normalize message content to a part list.

    Args:
        content: A string, a part list, or anything else.

    Returns:
        A single-item list for strings, the list itself for lists,
        else an empty list.

    """
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return content
    return []


def _attachment_injection(att: dict[str, Any]) -> str | None:
    """Render the query injection for a text-like file attachment.

    Args:
        att: Normalized Perplexity attachment.

    Returns:
        Markdown injection text, or None when the attachment should
        be uploaded instead.

    """
    if att.get("type") != "file" or not is_text_like(att):
        return None
    content_text = extract_attachment_text(att)
    if not content_text or len(content_text) > TEXT_APPEND_TOTAL_CAP:
        return None
    raw_name = att.get("name", "file")
    name = raw_name if isinstance(raw_name, str) else "file"
    return f"[Attached file: {name}]\n```\n{content_text}\n```"


def _assemble_query(query_parts: list[str], injections: list[str]) -> str:
    """Join query parts and file-text injections into the final query.

    Args:
        query_parts: Text parts from the newest message.
        injections: Rendered attachment injections.

    Returns:
        The stripped query with injections appended.

    """
    query = "\n".join(filter(None, query_parts)).strip()
    if not injections:
        return query
    joined = "\n\n".join(injections)
    return f"{query}\n\n{joined}" if query else joined


def _item_role(item: dict[str, Any]) -> str:
    """Return the role string of a Responses input item.

    Args:
        item: One Responses input item.

    Returns:
        The role, or an empty string when missing or not a string.

    """
    role = item.get("role")
    return role if isinstance(role, str) else ""


def _responses_system(data: dict[str, Any]) -> str | None:
    """Return the Responses instructions value when it is a string.

    Args:
        data: Validated Responses request payload.

    Returns:
        The instructions string, or None when absent or not a string.

    """
    raw: Any = data.get("instructions")
    return raw if isinstance(raw, str) else None


def _responses_user_items(items: list[Any]) -> list[Any]:
    """Select the items carrying conversation content.

    Prefers user/system/developer items, falling back to all items
    when none carry a role (e.g. raw content-part lists).

    Args:
        items: Raw Responses input items.

    Returns:
        The role-bearing items, or all items when none qualify.

    """
    user_items = [
        item
        for item in items
        if isinstance(item, dict)
        and item.get("role") in {"user", "system", "developer"}
    ]
    return user_items or items


def _fold_responses_system(
    items: list[Any],
    system: str | None,
) -> str | None:
    """Fold system-role message content into the system prompt.

    Args:
        items: Raw Responses input items.
        system: Instructions value, kept when message content is empty.

    Returns:
        The merged system prompt, if any.

    """
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            continue
        if item.get("role") == "system" and system is None:
            system = (
                " ".join(_text_of_part(p) for p in _iter_parts(item.get("content")))
                or system
            )
    return system


def _normalize_history(
    messages: list[Any],
) -> tuple[list[dict[str, Any]], str | None, str, list[dict[str, Any]], None]:
    """Split OpenAI messages into history, system, query, and attachments.

    Only the last user message is sent to Perplexity; everything
    earlier is represented by the thread handle.

    Args:
        messages: Raw OpenAI message dicts, oldest first.

    Returns:
        A (history, system, query, attachments, None) tuple: history key
        items, the system prompt, the last-message query text, prepared
        attachments, and a reserved None slot.

    """
    history: list[dict[str, Any]] = []
    system: str | None = None
    last_index = _last_user_index(messages)
    if last_index is None:
        return [], None, "", [], None

    for msg in messages[:last_index]:
        if not isinstance(msg, dict):
            continue
        history.append(
            {
                "role": msg.get("role"),
                "content": [_text_of_part(p) for p in _iter_parts(msg.get("content"))],
            },
        )
        if msg.get("role") in {"system", "developer"} and system is None:
            sys_text = " ".join(
                filter(
                    None,
                    (_text_of_part(p) for p in _iter_parts(msg.get("content"))),
                ),
            )
            if sys_text:
                system = sys_text

    acc = _MessageAccumulator()
    last = messages[last_index]
    if isinstance(last, dict):
        for part in _iter_parts(last.get("content")):
            acc.add_part(part, last=True, role="user")
    return (
        history,
        system,
        _assemble_query(acc.query_parts, acc.injections),
        acc.attachments,
        None,
    )


def _normalize_responses_input(
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None, str, list[dict[str, Any]]]:
    """Convert Responses API input into history, system, and query.

    Args:
        data: Validated Responses request payload.

    Returns:
        A (history, system, query, attachments) tuple: history key
        items, the system prompt, the last-message query text with
        file-text injections, and prepared attachments.

    """
    raw_input: Any = data.get("input", "")
    system = _responses_system(data)
    if isinstance(raw_input, str):
        return [], system, raw_input, []

    items: list[Any] = raw_input if isinstance(raw_input, list) else []
    system = _fold_responses_system(items, system)
    user_items = _responses_user_items(items)
    if not user_items:
        return [], system, "", []

    acc = _MessageAccumulator()
    last_item = user_items[-1]
    for item in items:
        if not isinstance(item, dict):
            continue
        role = _item_role(item)
        if role not in {"user", "system", "developer"}:
            continue
        for part in _iter_parts(item.get("content")):
            acc.add_part(part, last=item is last_item, role=role)
    return (
        acc.history,
        system,
        _assemble_query(acc.query_parts, acc.injections),
        acc.attachments,
    )


def _chat_completion_response(
    model: str,
    text: str,
    sources: list[dict[str, Any]],
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Build an OpenAI chat.completion payload for a finished ask.

    Args:
        model: Model name echoed in the payload.
        text: Assistant text for the choice message.
        sources: Upstream sources for URL citations.
        prompt_tokens: Estimated prompt tokens.
        completion_tokens: Estimated completion tokens.

    Returns:
        A chat.completion response dict.

    """
    annotations = _url_citations(text, sources)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text or None,
        "refusal": None,
    }
    if annotations:
        message["annotations"] = annotations
    return {
        "id": f"chatcmpl-{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "system_fingerprint": None,
        "service_tier": "default",
    }


def _url_citations(
    text: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cite sources whose title or URL appears in the answer text.

    Args:
        text: Assistant text to search for citations.
        sources: Upstream sources with title and URL.

    Returns:
        Up to eight url_citation annotation dicts.

    """
    out: list[dict[str, Any]] = []
    if not text or not sources:
        return out
    for src in sources[:8]:
        if not isinstance(src, dict):
            continue
        raw_url = src.get("url")
        if not isinstance(raw_url, str) or not raw_url:
            continue
        url = raw_url
        raw_title = src.get("title")
        title = raw_title if isinstance(raw_title, str) and raw_title else url
        idx = text.find(title[:60])
        if idx < 0:
            idx = text.find(url)
        if idx < 0:
            continue
        out.append(
            {
                "type": "url_citation",
                "start_index": idx,
                "end_index": idx + len(title[:60]),
                "url": url,
                "title": title,
            },
        )
    return out


SOURCE_APPENDIX_MAX = 50


def _source_appendix(sources: list[dict[str, Any]], query: str) -> str:
    r"""Bridge source appendix for llmcord-go's "Show Sources" button.

    Format mirrors grok-to-openai's source-attribution.js, which llmcord-go
    parses for any provider (FinalizeXAIResponseAnswer strips it from the
    visible answer and feeds it to Show Sources):
        \n\nSources
        1. [Title](url) (domain) via `query`

        Search Queries
        1. `query`

    Args:
        sources: Upstream sources with title and URL.
        query: Original user query echoed in the appendix.

    Returns:
        The appendix markdown, or an empty string without sources.

    """
    entries: list[str] = []
    clean_query = query.replace("`", "'") if query else ""
    for src in sources[:SOURCE_APPENDIX_MAX]:
        if not isinstance(src, dict):
            continue
        raw_url = src.get("url")
        url_text = raw_url if isinstance(raw_url, str) else ""
        url = url_text.strip().replace("\n", "").replace("\r", "")
        # llmcord-go's markdown-link regex stops at ')' and whitespace.
        url = url.replace(")", "%29").replace(" ", "%20")
        if not url:
            continue
        raw_title = src.get("title")
        title_text = raw_title if isinstance(raw_title, str) else ""
        title = (title_text.strip().replace("[", "").replace("]", "")) or url
        entry = f"[{title}]({url})"
        host = _host_of(url)
        if title != url and host:
            entry += f" ({host})"
        if clean_query:
            entry += f" via `{clean_query}`"
        entries.append(entry)
    if not entries:
        return ""
    lines = ["Sources"]
    lines.extend(f"{i}. {entry}" for i, entry in enumerate(entries, start=1))
    if clean_query:
        lines.extend(["", "Search Queries", f"1. `{clean_query}`"])
    return "\n\n" + "\n".join(lines)


def _host_of(url: str) -> str:
    """Return the lowercased host of a URL.

    Args:
        url: URL to parse.

    Returns:
        The host, or an empty string when unparseable.

    """
    try:
        netloc = urlparse(url).netloc or ""
    except ValueError:
        return ""
    return netloc.lower()


def _include_sources(*, flag: bool | None) -> bool:
    """Resolve whether to append the sources appendix.

    Args:
        flag: Per-request override, if any.

    Returns:
        The override, or the server default when None.

    """
    return INCLUDE_SOURCES if flag is None else flag


def _apply_chat_sources(
    resp: dict[str, Any],
    *,
    text: str,
    sources: list[dict[str, Any]],
    query: str,
    include_sources: bool | None,
) -> None:
    """Append the sources appendix to a chat completion when enabled.

    Args:
        resp: Chat completion payload mutated in place.
        text: Assistant text the appendix extends.
        sources: Upstream sources for the appendix.
        query: Original user query echoed in the appendix.
        include_sources: Per-request override of the server default.

    """
    if not _include_sources(flag=include_sources):
        return
    appendix = _source_appendix(sources, query)
    if appendix:
        resp["choices"][0]["message"]["content"] = (text or "") + appendix


def _apply_response_sources(
    resp: dict[str, Any],
    *,
    text: str,
    sources: list[dict[str, Any]],
    query: str,
    include_sources: bool | None,
) -> None:
    """Append the sources appendix to a Responses object when enabled.

    Args:
        resp: Responses payload mutated in place.
        text: Assistant text the appendix extends.
        sources: Upstream sources for the appendix.
        query: Original user query echoed in the appendix.
        include_sources: Per-request override of the server default.

    """
    if not _include_sources(flag=include_sources):
        return
    appendix = _source_appendix(sources, query)
    if appendix:
        resp["output"][0]["content"][0]["text"] = (text or "") + appendix


def _response_object(spec: ResponseSpec) -> dict[str, Any]:
    """Build an OpenAI Responses object for a finished ask.

    Args:
        spec: Model, text, sources, request echo, and token counts.

    Returns:
        A completed response object dict.

    """
    resp_id = spec.resp_id or f"resp_{secrets.token_hex(12)}"
    annotations = _url_citations(spec.text, spec.sources)
    content: dict[str, Any] = {
        "type": "output_text",
        "text": spec.text or "",
        "annotations": annotations,
    }
    return {
        "id": resp_id,
        "object": "response",
        "created_at": _now(),
        "status": "completed",
        "completed_at": _now(),
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": spec.request.get("instructions"),
        "max_output_tokens": spec.request.get("max_output_tokens"),
        "max_tool_calls": None,
        "model": spec.model,
        "output": [
            {
                "id": f"msg_{secrets.token_hex(12)}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [content],
            },
        ],
        "parallel_tool_calls": True,
        "previous_response_id": spec.previous_id,
        "reasoning": {"effort": None, "summary": None},
        "service_tier": "default",
        "store": bool(spec.request.get("store", True)),
        "temperature": spec.request.get("temperature"),
        "text": {"format": spec.request.get("text") or {"type": "text"}},
        "tool_choice": spec.request.get("tool_choice", "auto"),
        "tools": spec.request.get("tools", []),
        "top_logprobs": 0,
        "top_p": spec.request.get("top_p"),
        "truncation": spec.request.get("truncation", "disabled"),
        "usage": {
            "input_tokens": spec.prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": spec.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": spec.prompt_tokens + spec.completion_tokens,
        },
        "user": spec.request.get("user"),
        "metadata": spec.request.get("metadata") or {},
    }


async def _store_response(
    resp: dict[str, Any],
    thread: ThreadState,
) -> dict[str, Any]:
    """Attach the thread handle, persist the response, return public copy.

    Args:
        resp: Completed response payload mutated in place.
        thread: Thread handle stored alongside the response.

    Returns:
        A copy of the payload without the internal thread handle.

    """
    resp["_thread"] = thread
    await asyncio.to_thread(store.save_response, resp["id"], resp, cap=RESPONSE_LRU)
    out = dict(resp)
    out.pop("_thread", None)
    return out


# --------------------------------------------------------------------------
# Perplexity ask orchestration
# --------------------------------------------------------------------------


async def _fetch_remote_to_data_url(url: str) -> str:
    """Download a remote image/file so attachments stay data URLs.

    Args:
        url: Remote http(s) URL or an existing data URL.

    Returns:
        The content as a base64 data URL, or the input when it
        already is one.

    Raises:
        ValueError: If the download exceeds the size cap.

    """
    if url.startswith("data:"):
        return url
    async with AsyncSession(impersonate="chrome", timeout=30) as s:
        r = await s.get(url)
        r.raise_for_status()
        body = r.content
        if len(body) > FETCH_SIZE_CAP:
            msg = "attachment too large"
            raise ValueError(msg)
        mime = r.headers.get("content-type", "application/octet-stream").split(";")[0]
        return f"data:{mime};base64," + base64.b64encode(body).decode()


async def _prepare_attachments(
    attachments: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Resolve remote attachment URLs to data URLs, dropping failures.

    Args:
        attachments: Normalized Perplexity attachments, if any.

    Returns:
        Attachments with remote URLs inlined as data URLs.

    """
    out: list[dict[str, Any]] = []
    for raw_att in attachments or []:
        if not isinstance(raw_att, dict):
            continue
        raw_url: Any = raw_att.get("url", "")
        url = raw_url if isinstance(raw_url, str) else ""
        att = raw_att
        if url.startswith(("http://", "https://")):
            try:
                att = dict(raw_att, url=await _fetch_remote_to_data_url(url))
            # Curl failures surface as OSError subclasses (verified:
            # RequestsError inherits OSError); shaped errors re-raise
            # as ValueError/KeyError from parsing helpers.
            except (OSError, ValueError, KeyError) as e:
                raw_name: Any = raw_att.get("name")
                log.warning(
                    "attachment fetch failed (%s), dropping: %s",
                    raw_name if isinstance(raw_name, str) else "?",
                    e,
                )
                continue
        out.append(att)
    return out


class NoHealthyAccountError(Exception):
    """No healthy Perplexity account is available for an ask."""


async def _pick_ask_account(
    thread: ThreadState | None,
) -> tuple[Account, asyncio.Semaphore] | None:
    """Select the account that must serve an ask.

    Perplexity threads are account-bound: a follow-up must reuse the
    cookies of the account that owns the thread, otherwise the backend
    starts a brand-new conversation.

    Args:
        thread: Existing thread handle, if this ask continues one.

    Returns:
        An (account, semaphore) pair, or None when none is healthy.

    """
    if thread is not None and thread.account_id is not None:
        picked = pool.get(thread.account_id)
        if picked is not None:
            return picked
    return await pool.pick()


def _event_code(ev: AskEvent) -> str:
    """Return the upstream error code for an error event.

    Args:
        ev: The ask event to inspect.

    Returns:
        The error code, or "UNKNOWN" when missing or malformed.

    """
    raw: Any = ev.data.get("error_code", "UNKNOWN")
    return raw if isinstance(raw, str) and raw else "UNKNOWN"


def _event_message(ev: AskEvent) -> str:
    """Return the upstream error message for an error event.

    Args:
        ev: The ask event to inspect.

    Returns:
        The message, or a default when missing or malformed.

    """
    raw: Any = ev.data.get("message", "Perplexity upstream error")
    return raw if isinstance(raw, str) and raw else "Perplexity upstream error"


def _event_sources(ev: AskEvent) -> list[dict[str, Any]]:
    """Return the source dicts carried by a sources event.

    Args:
        ev: The ask event to inspect.

    Returns:
        Source dicts, or an empty list for other shapes.

    """
    raw: Any = ev.data.get("results", [])
    if not isinstance(raw, list):
        return []
    return [r for r in raw if isinstance(r, dict)]


def _event_thread(ev: AskEvent) -> ThreadState | None:
    """Return the thread handle carried by a done event.

    Args:
        ev: The ask event to inspect.

    Returns:
        The thread handle, or None for other events or shapes.

    """
    if ev.kind != "done":
        return None
    raw: Any = ev.data.get("thread")
    return raw if isinstance(raw, ThreadState) else None


def _handle_error_event(
    ev: AskEvent,
    acct: Account,
    outcome: _AttemptOutcome,
) -> None:
    """Translate one upstream error event into a raise or a retry mark.

    Quota exhaustion records a quota failure and raises; a first-attempt
    GENERIC_FAILED_RESPONSE marks the outcome for retry so the caller
    restarts with a fresh thread; anything else raises.

    Args:
        ev: The upstream error event.
        acct: Account to charge quota failures against.
        outcome: Mutable attempt state marked for retry when warranted.

    Raises:
        AskError: For quota exhaustion and fatal upstream errors.

    """
    code = _event_code(ev)
    if code == "FREE_TIER_RATE_LIMITED":
        pool.record_failure(acct, code, quota=True)
        msg = "Perplexity free-tier daily query limit reached"
        raise AskError(code, msg, retryable=False)
    if code == "GENERIC_FAILED_RESPONSE" and outcome.attempt == 1:
        # Stale thread handle or transient upstream failure:
        # retry exactly once as a brand-new thread.
        outcome.retry_requested = True
        return
    msg = _event_message(ev)
    raise AskError(code, msg, retryable=False)


async def _consume_attempt(
    query: str,
    spec: AskSpec,
    attachments: list[dict[str, Any]],
    acct: Account,
    outcome: _AttemptOutcome,
) -> AsyncIterator[AskEvent]:
    """Stream one attempt's events, translating error events via helper.

    Error events are delegated to _handle_error_event, which raises for
    quota exhaustion and fatal upstream errors; a retryable first-attempt
    GENERIC_FAILED_RESPONSE marks the outcome for retry and ends the
    stream instead of raising.

    Args:
        query: The user query to send.
        spec: Model, mode, thread, and system context.
        attachments: Prepared Perplexity attachments.
        acct: Account executing this attempt.
        outcome: Mutable attempt state updated while streaming.

    Yields:
        AskEvent: Text, sources, related, and done events from upstream.

    """
    thread = outcome.thread
    if thread is None or thread.backend_uuid is None:
        thread = ThreadState(
            model=spec.model,
            mode=spec.mode,
            account_id=acct.index,
        )
        outcome.thread = thread
    options = AskOptions(
        attachments=attachments,
        thread=thread,
        system=spec.system,
        timezone=TZ,
    )
    async for ev in acct.client.ask(
        query,
        model=spec.model,
        mode=spec.mode,
        options=options,
    ):
        if ev.kind == "error":
            _handle_error_event(ev, acct, outcome)
            if outcome.retry_requested:
                return
            continue
        if ev.kind == "text":
            outcome.got_text = True
        if ev.kind == "done":
            done_thread = _event_thread(ev)
            if done_thread is not None:
                outcome.thread = done_thread
        yield ev


async def _forward_attempt_events(
    gen: AsyncIterator[AskEvent],
    outcome: _AttemptOutcome,
) -> AsyncIterator[AskEvent]:
    """Forward attempt events while tracking the terminal thread.

    Args:
        gen: Upstream attempt events.
        outcome: Mutable attempt state receiving the done thread.

    Yields:
        AskEvent: Each upstream event unchanged.

    """
    async for ev in gen:
        if ev.kind == "done":
            done_thread = _event_thread(ev)
            if done_thread is not None:
                outcome.thread = done_thread
        yield ev


def _retry_thread(
    spec: AskSpec,
    acct: Account,
    outcome: _AttemptOutcome,
) -> ThreadState | None:
    """Return a fresh thread when the attempt requested a retry.

    Args:
        spec: Model and mode for the replacement thread.
        acct: Account owning the replacement thread.
        outcome: Finished attempt state.

    Returns:
        A brand-new thread handle, or None when no retry was requested.

    """
    if not outcome.retry_requested:
        return None
    pool.record_failure(acct, "GENERIC_FAILED_RESPONSE")
    return ThreadState(model=spec.model, mode=spec.mode, account_id=acct.index)


def _check_attempt_thread(outcome: _AttemptOutcome) -> ThreadState:
    """Return the attempt thread, raising when the backend gave none.

    Args:
        outcome: Finished attempt state.

    Returns:
        The attempt's thread handle.

    Raises:
        AskError: When no thread handle arrived and no text either.

    """
    thread = outcome.thread
    if thread is None or (thread.backend_uuid is None and not outcome.got_text):
        code = "NO_BACKEND_UUID"
        msg = "Perplexity did not return a thread handle"
        raise AskError(code, msg, retryable=False)
    return thread


async def _ask_with_account(
    query: str,
    spec: AskSpec,
    attachments: list[dict[str, Any]],
    acct: Account,
) -> AsyncIterator[AskEvent]:
    """Run attempts for one account, retrying once as a new thread.

    Args:
        query: The user query to send.
        spec: Model, mode, thread, and system context.
        attachments: Prepared Perplexity attachments.
        acct: Account executing the attempts.

    Yields:
        AskEvent: Text, sources, related, and done events from upstream.

    Raises:
        AskError: If the request fails or yields no thread handle.

    """
    thread = spec.thread
    attempt = 0
    while True:
        attempt += 1
        outcome = _AttemptOutcome(thread=thread, attempt=attempt)
        events = _consume_attempt(query, spec, attachments, acct, outcome)
        try:
            async for ev in _forward_attempt_events(events, outcome):
                yield ev
        except AskError as e:
            if e.retryable and attempt == 1:
                continue
            raise
        except (OSError, ValueError, KeyError) as e:
            # Curl failures surface as OSError subclasses (verified:
            # RequestsError inherits OSError).
            pool.record_failure(acct, f"{type(e).__name__}: {e}")
            code = "UPSTREAM_ERROR"
            msg = f"Perplexity request failed: {e}"
            raise AskError(code, msg, retryable=False) from e
        fresh = _retry_thread(spec, acct, outcome)
        if fresh is not None:
            thread = fresh
            continue
        thread = _check_attempt_thread(outcome)
        pool.record_success(acct)
        return


async def _stream_ask(
    query: str,
    spec: AskSpec,
) -> AsyncIterator[AskEvent]:
    """Run one ask, yielding AskEvents as they arrive.

    Handles account selection, per-account concurrency, quota marking,
    and exactly one retry-as-new-thread on GENERIC_FAILED_RESPONSE.

    Args:
        query: The user query to send to Perplexity.
        spec: Model, mode, attachments, thread, and system context.

    Yields:
        AskEvent: Text, sources, related, and done events from upstream.

    Raises:
        NoHealthyAccountError: If no healthy account is available.

    """
    attachments = await _prepare_attachments(spec.attachments)
    picked = await _pick_ask_account(spec.thread)
    if picked is None:
        raise NoHealthyAccountError
    acct, sem = picked
    acct.active += 1
    acquired = False
    try:
        await sem.acquire()
        acquired = True
        async for ev in _ask_with_account(query, spec, attachments, acct):
            yield ev
    finally:
        if acquired:
            sem.release()
        acct.active -= 1


async def _stream_with_accounts(
    query: str,
    spec: AskSpec,
) -> AsyncIterator[AskEvent]:
    """Yield ask events, trying the next account on quota exhaustion.

    Args:
        query: The user query to send.
        spec: Model, mode, attachments, thread, and system context.

    Yields:
        AskEvent: Events from the first account that serves the ask.

    Raises:
        AskError: If no healthy account remains or upstream fails.

    """
    tried = 0
    while True:
        try:
            async for ev in _stream_ask(query, spec):
                yield ev
        except NoHealthyAccountError as err:
            code = "NO_HEALTHY_ACCOUNT"
            msg = "No healthy Perplexity account available (quota or cooldown)"
            raise AskError(code, msg, retryable=False) from err
        except AskError as e:
            if e.code == "FREE_TIER_RATE_LIMITED" and tried < pool.size - 1:
                tried += 1
                continue
            raise
        else:
            return


async def _collect_ask(
    query: str,
    spec: AskSpec,
) -> tuple[ThreadState, str, list[dict[str, Any]]]:
    """Accumulate ask events into a final non-streaming result.

    Args:
        query: The user query to send.
        spec: Model, mode, attachments, thread, and system context.

    Returns:
        A (thread, text, sources) triple for the completed ask.

    Raises:
        AskError: If the request fails or yields no thread handle.

    """
    text_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    thread = spec.thread
    async for ev in _stream_with_accounts(query, spec):
        if ev.kind == "text":
            text_parts.append(ev.text)
        elif ev.kind == "sources":
            sources = _event_sources(ev)
        elif ev.kind == "done":
            done_thread = _event_thread(ev)
            if done_thread is not None:
                thread = done_thread
    if thread is None or thread.backend_uuid is None:
        code = "NO_BACKEND_UUID"
        msg = "Perplexity did not return a thread handle"
        raise AskError(code, msg, retryable=False)
    return thread, "".join(text_parts), sources


async def _collect_and_bind(
    query: str,
    spec: AskSpec,
    history: list[dict[str, Any]],
) -> tuple[ThreadState, str, list[dict[str, Any]]]:
    """Run a non-streaming ask and persist the thread/answer binding.

    Args:
        query: The user query to send.
        spec: Model, mode, attachments, thread, and system context.
        history: History items used to derive the conversation key.

    Returns:
        A (thread, text, sources) triple for the completed ask.

    """
    thread_out, text, sources = await _collect_ask(query, spec)
    await _register_thread(_conversation_key(history), thread_out)
    await _register_answer(text, thread_out)
    return thread_out, text, sources


async def _answer_chat_once(
    query: str,
    spec: AskSpec,
    history: list[dict[str, Any]],
    prompt_tokens: int,
    *,
    include_sources: bool | None,
) -> dict[str, Any]:
    """Run one non-streaming chat ask and build the completion payload.

    Args:
        query: The user query to send.
        spec: Model, mode, attachments, thread, and system context.
        history: History items used to derive the conversation key.
        prompt_tokens: Estimated prompt tokens for usage reporting.
        include_sources: Append the sources appendix when True.

    Returns:
        An OpenAI chat.completion payload.

    Raises:
        _http_from_ask_error: The mapped 429/502 HTTP error.

    """
    try:
        _thread_out, text, sources = await _collect_and_bind(query, spec, history)
    except AskError as e:
        raise _http_from_ask_error(e) from e
    resp = _chat_completion_response(
        spec.model,
        text,
        sources,
        prompt_tokens=prompt_tokens,
        completion_tokens=_est_tokens(text),
    )
    _apply_chat_sources(
        resp,
        text=text,
        sources=sources,
        query=query,
        include_sources=include_sources,
    )
    return resp


async def _answer_response_once(
    req: ResponsesRequest,
    query: str,
    spec: AskSpec,
    history: list[dict[str, Any]],
    prompt_tokens: int,
) -> dict[str, Any]:
    """Run one non-streaming Responses ask and build the response object.

    Args:
        req: Validated Responses request for echoes and overrides.
        query: The user query to send.
        spec: Model, mode, attachments, thread, and system context.
        history: History items used to derive the conversation key.
        prompt_tokens: Estimated prompt tokens for usage reporting.

    Returns:
        A public response object dict (no internal thread handle).

    Raises:
        _http_from_ask_error: The mapped 429/502 HTTP error.

    """
    try:
        thread_out, text, sources = await _collect_and_bind(query, spec, history)
    except AskError as e:
        raise _http_from_ask_error(e) from e
    resp = _response_object(
        ResponseSpec(
            model=spec.model,
            text=text,
            sources=sources,
            request=req.model_dump(),
            prompt_tokens=prompt_tokens,
            completion_tokens=_est_tokens(text),
            previous_id=req.previous_response_id,
        ),
    )
    _apply_response_sources(
        resp,
        text=text,
        sources=sources,
        query=query,
        include_sources=req.include_sources,
    )
    return await _store_response(resp, thread_out)


async def _response_thread_and_error(
    previous_id: str | None,
    raw_input: object,
    history: list[dict[str, Any]],
) -> tuple[ThreadState | None, JSONResponse | None]:
    """Resolve the stored thread for a Responses request.

    Follows previous_response_id when given, else falls back to the
    conversation-key lookup used for chat completions.

    Args:
        previous_id: Client-supplied previous response id, if any.
        raw_input: Raw Responses input for the history fallback.
        history: Normalized history items for the key lookup.

    Returns:
        A (thread, error) pair; error is a 400 JSONResponse when the
        previous id is unknown, with thread None.

    """
    if previous_id is not None:
        prev = await asyncio.to_thread(store.get_response, previous_id)
        if prev is None:
            return None, _unknown_previous_response()
        return _stored_thread(prev), None
    items = raw_input if isinstance(raw_input, list) else []
    return await _resolve_chat_thread(items, history), None


def _missing_query_response() -> JSONResponse:
    """Build the 400 response for requests without user content.

    Returns:
        An invalid_request JSONResponse.

    """
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "No user message with content found",
                "type": "invalid_request_error",
                "code": "invalid_request",
            },
        },
    )


def _unknown_previous_response() -> JSONResponse:
    """Build the 400 response for an unknown previous_response_id.

    Returns:
        An invalid_previous_response_id JSONResponse.

    """
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Unknown previous_response_id",
                "type": "invalid_request_error",
                "code": "invalid_previous_response_id",
            },
        },
    )


# --------------------------------------------------------------------------
# OpenAI SSE emitters
# --------------------------------------------------------------------------


def _sse(data: dict[str, Any]) -> str:
    """Format one SSE frame for an event payload.

    Args:
        data: Event payload, optionally with a "type" for named events.

    Returns:
        The SSE-formatted frame string.

    """
    if "type" in data:
        return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"
    return f"data: {json.dumps(data)}\n\n"


def _sse_response(body: AsyncIterator[str]) -> StreamingResponse:
    """Wrap an SSE chunk stream in a streaming response.

    Args:
        body: Async iterator of SSE-formatted chunks.

    Returns:
        A text/event-stream response with no-cache headers.

    """
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _forward_chat_events(
    gen: AsyncIterator[AskEvent],
    capture: _StreamCapture,
) -> AsyncIterator[AskEvent]:
    """Forward ask events while capturing text, sources, and thread.

    Args:
        gen: Upstream ask events.
        capture: Mutable capture for persistence after the stream.

    Yields:
        AskEvent: Each upstream event unchanged.

    """
    async for ev in gen:
        if ev.kind == "text":
            capture.text_parts.append(ev.text)
        elif ev.kind == "sources":
            capture.sources = _event_sources(ev)
        elif ev.kind == "done":
            done_thread = _event_thread(ev)
            if done_thread is not None:
                capture.thread = done_thread
        yield ev


async def _forward_responses_events(
    gen: AsyncIterator[AskEvent],
    capture: _StreamCapture,
) -> AsyncIterator[AskEvent]:
    """Forward ask events while capturing the terminal thread handle.

    Args:
        gen: Upstream ask events.
        capture: Mutable capture for persistence after the stream.

    Yields:
        AskEvent: Each upstream event unchanged.

    """
    async for ev in gen:
        if ev.kind == "done":
            done_thread = _event_thread(ev)
            if done_thread is not None:
                capture.thread = done_thread
        yield ev


async def _persist_chat_thread(key: str, capture: _StreamCapture) -> None:
    """Persist the streamed thread handle and answer anchor.

    Args:
        key: Conversation-key hash for the thread.
        capture: Capture filled while streaming.

    """
    if capture.thread is None:
        return
    await _register_thread(key, capture.thread)
    await _register_answer("".join(capture.text_parts), capture.thread)


def _output_text(completed: dict[str, Any]) -> str:
    """Extract the assistant text from a completed Responses object.

    Args:
        completed: The completed response payload.

    Returns:
        The first output text, or an empty string when the shape
        is unexpected.

    """
    output = completed.get("output", [])
    if not isinstance(output, list) or not output:
        return ""
    first = output[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content", [])
    if not isinstance(content, list) or not content:
        return ""
    first_part = content[0]
    if not isinstance(first_part, dict):
        return ""
    text = first_part.get("text", "")
    return text if isinstance(text, str) else ""


async def _persist_response_stream(
    history: list[dict[str, Any]],
    resp_id: str,
    capture: _StreamCapture,
    completed: dict[str, Any] | None,
) -> None:
    """Persist a streamed Responses result and its thread binding.

    Args:
        history: History items used to derive the conversation key.
        resp_id: Response id the completed payload was stored under.
        capture: Capture filled while streaming.
        completed: Final completed payload, if the stream finished.

    """
    if completed is None:
        return
    final_stored = dict(completed)
    final_stored["_thread"] = capture.thread
    await asyncio.to_thread(
        store.save_response,
        resp_id,
        final_stored,
        cap=RESPONSE_LRU,
    )
    if capture.thread is None:
        return
    await _register_thread(_conversation_key(history), capture.thread)
    await _register_answer(_output_text(completed), capture.thread)


async def _stream_chat_completions(
    gen: AsyncIterator[AskEvent],
    spec: ChatStreamSpec,
) -> AsyncIterator[str]:
    """Stream OpenAI chat-completion SSE chunks for upstream events.

    Args:
        gen: Upstream ask events.
        spec: Completion id, model, token counts, and tail renderer.

    Yields:
        SSE-formatted chat.completion.chunk payloads, ending with a
        literal [DONE] sentinel.

    """
    base = {
        "id": spec.completion_id,
        "object": "chat.completion.chunk",
        "created": spec.created,
        "model": spec.model,
        "system_fingerprint": None,
    }
    first = dict(
        base,
        choices=[
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "logprobs": None,
                "finish_reason": None,
            },
        ],
    )
    yield _sse(first)
    completion_tokens = 0
    async for ev in gen:
        if ev.kind == "text":
            completion_tokens += _est_tokens(ev.text)
            chunk = dict(
                base,
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": ev.text},
                        "logprobs": None,
                        "finish_reason": None,
                    },
                ],
            )
            yield _sse(chunk)
    tail = spec.final_text()
    if tail:
        completion_tokens += _est_tokens(tail)
        chunk = dict(
            base,
            choices=[
                {
                    "index": 0,
                    "delta": {"content": tail},
                    "logprobs": None,
                    "finish_reason": None,
                },
            ],
        )
        yield _sse(chunk)
    yield _sse(
        dict(
            base,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": "stop",
                },
            ],
        ),
    )
    if spec.include_usage:
        usage = {
            "prompt_tokens": spec.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": spec.prompt_tokens + completion_tokens,
        }
        yield _sse(dict(base, choices=[], usage=usage))
    # OpenAI chat-completions streams terminate with a literal [DONE] sentinel.
    # Clients (e.g. llmcord-go) treat EOF before [DONE] as a dropped stream.
    yield "data: [DONE]\n\n"


async def _stream_responses(
    gen: AsyncIterator[AskEvent],
    spec: ResponsesStreamSpec,
) -> AsyncIterator[tuple[dict[str, Any], str]]:
    """Stream real-time Responses API events as AskEvents arrive upstream.

    Args:
        gen: Upstream ask events.
        spec: Model, request echo, token counts, ids, and tail renderer.

    Yields:
        Tuples of the in-progress response snapshot and the SSE chunk.

    """
    resp_id = spec.resp_id or f"resp_{secrets.token_hex(12)}"
    msg_id = spec.msg_id or f"msg_{secrets.token_hex(12)}"
    created = _now()

    resp_in_prog = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "completed_at": None,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": spec.request.get("instructions"),
        "max_output_tokens": spec.request.get("max_output_tokens"),
        "max_tool_calls": None,
        "model": spec.model,
        "output": [],
        "parallel_tool_calls": True,
        "previous_response_id": spec.previous_id,
        "reasoning": {"effort": None, "summary": None},
        "service_tier": "default",
        "store": bool(spec.request.get("store", True)),
        "temperature": spec.request.get("temperature"),
        "text": {"format": spec.request.get("text") or {"type": "text"}},
        "tool_choice": spec.request.get("tool_choice", "auto"),
        "tools": spec.request.get("tools", []),
        "top_logprobs": 0,
        "top_p": spec.request.get("top_p"),
        "truncation": spec.request.get("truncation", "disabled"),
        "usage": None,
        "user": spec.request.get("user"),
        "metadata": spec.request.get("metadata") or {},
    }

    yield resp_in_prog, _sse({"type": "response.created", "response": resp_in_prog})
    yield (
        resp_in_prog,
        _sse(
            {"type": "response.in_progress", "response": resp_in_prog},
        ),
    )

    output_item_in_prog = {
        "id": msg_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    yield (
        resp_in_prog,
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": output_item_in_prog,
            },
        ),
    )
    yield (
        resp_in_prog,
        _sse(
            {
                "type": "response.content_part.added",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ),
    )

    accumulated_text: list[str] = []
    sources: list[dict[str, Any]] = []

    async for ev in gen:
        if ev.kind == "sources":
            sources = _event_sources(ev)
        elif ev.kind == "text" and ev.text:
            accumulated_text.append(ev.text)
            yield (
                resp_in_prog,
                _sse(
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": ev.text,
                    },
                ),
            )

    appendix = spec.final_text(sources)
    if appendix:
        accumulated_text.append(appendix)
        yield (
            resp_in_prog,
            _sse(
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": appendix,
                },
            ),
        )

    full_text = "".join(accumulated_text)
    annotations = _url_citations(full_text, sources)
    content_part = {
        "type": "output_text",
        "text": full_text,
        "annotations": annotations,
    }
    output_item_done = {
        "id": msg_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [content_part],
    }

    yield (
        resp_in_prog,
        _sse(
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "text": full_text,
                "annotations": annotations,
            },
        ),
    )
    yield (
        resp_in_prog,
        _sse(
            {
                "type": "response.content_part.done",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": content_part,
            },
        ),
    )
    yield (
        resp_in_prog,
        _sse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": output_item_done,
            },
        ),
    )

    completion_tokens = _est_tokens(full_text)
    resp_completed = dict(
        resp_in_prog,
        status="completed",
        completed_at=_now(),
        output=[output_item_done],
        usage={
            "input_tokens": spec.prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": spec.prompt_tokens + completion_tokens,
        },
    )

    yield (
        resp_completed,
        _sse({"type": "response.completed", "response": resp_completed}),
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start the account pool for the application's lifetime.

    Args:
        _app: FastAPI application (unused, required by the protocol).

    Yields:
        None: Control to the running application.

    """
    await pool.start()
    log.info("accounts: %d", pool.size)
    try:
        yield
    finally:
        await pool.close()


app = FastAPI(title="Perplexity OpenAI-compatible API", lifespan=lifespan)


@app.middleware("http")
async def api_key_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Enforce the optional shared API key on /v1/ routes.

    Args:
        request: Incoming HTTP request.
        call_next: Next middleware or route handler.

    Returns:
        A 401 JSONResponse for bad keys, else the downstream response.

    """
    if API_KEY and request.url.path.startswith("/v1/"):
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {API_KEY}":
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    },
                },
            )
    return await call_next(request)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Report server and account-pool health.

    Returns:
        Status, account, thread, and response counts.

    """
    return {
        "status": "ok",
        "accounts": pool.size,
        "accounts_detail": pool.status(),
        "threads": await asyncio.to_thread(store.count_threads),
        "responses": await asyncio.to_thread(store.count_responses),
    }


MODEL_LIST = [
    {"id": "turbo", "object": "model", "owned_by": "perplexity"},
    {"id": "pplx_pro", "object": "model", "owned_by": "perplexity"},
    {"id": "pplx_pro_upgraded", "object": "model", "owned_by": "perplexity"},
    {"id": "pplx_reasoning", "object": "model", "owned_by": "perplexity"},
    {"id": "gpt-4o", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "gpt-4o-mini", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "sonar", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "sonar-pro", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "sonar-reasoning", "object": "model", "owned_by": "perplexity-alias"},
    {"id": "pplx-copilot", "object": "model", "owned_by": "perplexity-mode"},
    {"id": "pplx-concise", "object": "model", "owned_by": "perplexity-mode"},
]


class ChatRequest(BaseModel):
    """Chat completion request body."""

    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    messages: list[Any] = []
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    max_tokens: int | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None
    include_sources: bool | None = None


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    req: ChatRequest,
) -> StreamingResponse | JSONResponse | dict[str, Any]:
    """Serve an OpenAI Chat Completions request via Perplexity.

    Resolves the model, replays the conversation onto a stored
    Perplexity thread, and returns either a completion payload or
    an SSE stream.

    Args:
        req: Validated chat completion request.

    Returns:
        A chat.completion payload, a 400 JSONResponse, or an SSE stream.

    """
    model = resolve_model(req.model)
    mode = resolve_mode(req.model, model)
    history, system, query, attachments, _ = _normalize_history(req.messages)
    if not query:
        return _missing_query_response()
    key = _conversation_key(history)
    thread = await _resolve_chat_thread(req.messages, history)
    prompt_tokens = _prompt_tokens(query, history)
    spec = AskSpec(
        model=model,
        mode=mode,
        attachments=attachments,
        thread=thread,
        system=system,
    )
    if not req.stream:
        return await _answer_chat_once(
            query,
            spec,
            history,
            prompt_tokens,
            include_sources=req.include_sources,
        )

    include_usage = bool((req.stream_options or {}).get("include_usage"))
    completion_id = f"chatcmpl-{secrets.token_hex(12)}"
    created = _now()

    async def sse_gen() -> AsyncIterator[str]:
        """Stream chat SSE chunks, then persist the thread binding.

        Yields:
            SSE-formatted chat completion chunks.

        """
        gen = _stream_with_accounts(query, spec)
        capture = _StreamCapture()

        def appendix() -> str:
            """Render the sources appendix when enabled.

            Returns:
                The appendix markdown, or an empty string when disabled.

            """
            if _include_sources(flag=req.include_sources):
                return _source_appendix(capture.sources, query)
            return ""

        stream = _stream_chat_completions(
            _forward_chat_events(gen, capture),
            ChatStreamSpec(
                completion_id=completion_id,
                model=model,
                created=created,
                prompt_tokens=prompt_tokens,
                include_usage=include_usage,
                final_text=appendix,
            ),
        )
        try:
            async for chunk in stream:
                yield chunk
            await _persist_chat_thread(key, capture)
        except AskError as e:
            yield _sse_error(e)

    return _sse_response(sse_gen())


def _sse_error(e: AskError) -> str:
    """Format an ask failure as an SSE error frame.

    Args:
        e: The ask failure to report.

    Returns:
        An SSE error event frame.

    """
    body = {
        "error": {
            "message": e.message,
            "type": "upstream_error",
            "code": e.code.lower(),
        },
    }
    return _sse({"type": "error", "error": body["error"]})


class ResponsesRequest(BaseModel):
    """Responses request body."""

    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    input: Any = ""
    instructions: str | None = None
    stream: bool = False
    previous_response_id: str | None = None
    max_output_tokens: int | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None
    store: bool | None = True
    temperature: float | None = None
    top_p: float | None = None
    tools: list[Any] = []
    tool_choice: Any = "auto"
    text: Any = None
    reasoning: Any = None
    truncation: Any = "disabled"
    include_sources: bool | None = None


def _http_from_ask_error(e: AskError) -> HTTPException:
    """Map an ask failure to an HTTP error response.

    Args:
        e: The ask failure to translate.

    Returns:
        A 429 HTTPException for quota exhaustion, else a 502.

    """
    if e.code in {"FREE_TIER_RATE_LIMITED", "NO_HEALTHY_ACCOUNT"}:
        status = HTTP_TOO_MANY_REQUESTS
    elif e.code in {"NO_BACKEND_UUID", "UPSTREAM_ERROR", "GENERIC_FAILED_RESPONSE"}:
        status = 502
    else:
        status = 502
    quota = status == HTTP_TOO_MANY_REQUESTS
    return HTTPException(
        status_code=status,
        detail={
            "message": e.message,
            "type": "insufficient_quota" if quota else "upstream_error",
            "code": "rate_limit_exceeded" if quota else e.code.lower(),
        },
    )


@app.post("/v1/responses", response_model=None)
async def create_response(
    req: ResponsesRequest,
) -> StreamingResponse | JSONResponse | dict[str, Any]:
    """Serve an OpenAI Responses request via Perplexity.

    Resolves the model, replays the conversation onto a stored
    Perplexity thread (via previous_response_id or history), and
    returns either a response object or an SSE stream.

    Args:
        req: Validated Responses request.

    Returns:
        A response object, a 400 JSONResponse, or an SSE stream.

    """
    model = resolve_model(req.model)
    mode = resolve_mode(req.model, model)
    history, system, query, attachments = _normalize_responses_input(
        req.model_dump(),
    )
    if req.instructions:
        system = (
            req.instructions if system is None else f"{req.instructions}\n\n{system}"
        )
    if not query:
        return _missing_query_response()
    thread, err = await _response_thread_and_error(
        req.previous_response_id,
        req.input,
        history,
    )
    if err is not None:
        return err
    prompt_tokens = _prompt_tokens(query, history)
    spec = AskSpec(
        model=model,
        mode=mode,
        attachments=attachments,
        thread=thread,
        system=system,
    )
    if not req.stream:
        return await _answer_response_once(req, query, spec, history, prompt_tokens)

    resp_id = f"resp_{secrets.token_hex(12)}"
    msg_id = f"msg_{secrets.token_hex(12)}"

    async def sse_gen() -> AsyncIterator[str]:
        """Stream Responses SSE events, then persist the result.

        Yields:
            SSE-formatted Responses event chunks.

        """
        gen = _stream_with_accounts(query, spec)
        capture = _StreamCapture()

        def appendix(srcs: list[dict[str, Any]]) -> str:
            """Render the sources appendix when enabled.

            Returns:
                The appendix markdown, or an empty string when disabled.

            """
            if _include_sources(flag=req.include_sources):
                return _source_appendix(srcs, query)
            return ""

        stream = _stream_responses(
            _forward_responses_events(gen, capture),
            ResponsesStreamSpec(
                model=model,
                request=req.model_dump(),
                prompt_tokens=prompt_tokens,
                resp_id=resp_id,
                msg_id=msg_id,
                previous_id=req.previous_response_id,
                final_text=appendix,
            ),
        )
        completed: dict[str, Any] | None = None
        try:
            async for resp_obj, chunk in stream:
                completed = resp_obj
                yield chunk
            await _persist_response_stream(history, resp_id, capture, completed)
        except AskError as e:
            yield _sse_error(e)

    return _sse_response(sse_gen())


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str) -> dict[str, Any]:
    """Fetch a previously stored Responses object.

    Args:
        response_id: Id of the stored response.

    Returns:
        The stored response without its internal thread handle.

    Raises:
        HTTPException: 404 when the id is unknown.

    """
    resp = await asyncio.to_thread(store.get_response, response_id)
    if resp is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"No response found with id {response_id}",
                "type": "invalid_request_error",
                "code": "response_not_found",
            },
        )
    out = dict(resp)
    out.pop("_thread", None)
    return out


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List the models served by this proxy.

    Returns:
        An OpenAI model-list payload.

    """
    return {"object": "list", "data": MODEL_LIST}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> dict[str, Any]:
    """Fetch one model entry by id.

    Args:
        model_id: Model id to look up.

    Returns:
        The model entry.

    Raises:
        HTTPException: 404 when the id is unknown.

    """
    for m in MODEL_LIST:
        if m["id"] == model_id:
            return m
    raise HTTPException(
        status_code=404,
        detail={
            "message": f"The model '{model_id}' does not exist",
            "type": "invalid_request_error",
            "code": "model_not_found",
        },
    )
