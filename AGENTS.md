# Repository Guidelines

## Project Overview

FastAPI proxy on `:64130` exposing OpenAI Chat Completions + Responses APIs backed by Perplexity private SSE. No browser at runtime — `curl_cffi` Chrome impersonation + Netscape session cookies in `accounts.txt`.

## Architecture & Data Flow

Flat 4-module service, no packages:

`server.py` (facade/orchestration) → `account_pool.py` (LB) → `pplx_transport.py` (`PerplexityClient`) → `perplexity.ai`; `db_store.py` (SQLite) is restart-proof memory on the side.

- Chat non-stream: `POST /v1/chat/completions` → `_normalize_history` (query/history/system split) → `_lookup_thread`/`_resolve_chat_thread` → `resolve_model/mode` → `AskSpec` → `_prepare_attachments` → `_stream_with_accounts` → `_collect_ask` → `_register_thread`+`_register_answer` → `_chat_completion_response`.
- Chat stream: same setup, `sse_gen()` + `_stream_chat_completions` yields `chat.completion.chunk` SSE; mid-stream `AskError` becomes `_sse_error` frame, not HTTP error.
- Responses: `POST /v1/responses` → `_normalize_responses_input` → `_response_thread_and_error` (`previous_response_id` → `store.get_response`, unknown ID → 400) → same ask pipeline → `_response_object` + `_store_response` (embeds `_thread` for follow-ups). `GET /v1/responses/{id}` replays payload minus `_thread`.
- Upstream attempt: `pool.pick()/get()` (sticky to `thread.account_id`) → `PerplexityClient.ask()` (fresh `frontend_uuids`, `Referer=/search/{slug}`, `x-request-id`) → `POST /rest/sse/perplexity_ask` → `_drain_ask_response` → `AskEvent(text|sources|related|error|done)`. Only new query + thread handle sent; history never re-sent.
- Thread resolution: `_conversation_key` (`sha256` history minus last turn) + answer-echo `sha256`; stored in SQLite via `asyncio.to_thread`.
- Failover: `FREE_TIER_RATE_LIMITED` → next account; first-attempt `GENERIC_FAILED_RESPONSE` → retry once as new thread. See `server.py:_stream_with_accounts`, `_ask_with_account`, `_handle_error_event`.

## Key Directories

No `src/`, `tests/`, `scripts/`, `docs/`, `tools/`. All at repo root:

```text
server.py          # app + routes + orchestration (~2536 lines)
pplx_transport.py  # SSE transport, payload builder, attachments (~1207 lines)
account_pool.py    # Account/AccountPool (~334 lines)
db_store.py        # Store SQLite WAL (~524 lines)
restart.sh         # ops script
README.md          # only real doc
```

## Development Commands

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt  # pip path
uv sync                           # uv path (includes ty dev group)
uvicorn server:app --host 0.0.0.0 --port 64130
./restart.sh                      # kill uvicorn.*server:app, nohup restart, poll /healthz, tail -f server.log
curl -fsS http://127.0.0.1:64130/healthz
ruff check . && ruff format --check .  # config-only, external binary
ty check                            # strict, warnings are errors
```

Manual smoke probes (from `README.md`):

```bash
curl http://127.0.0.1:64130/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Say hi"}]}'
curl -N http://127.0.0.1:64130/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"Explain entropy in one sentence"}]}'
curl http://127.0.0.1:64130/v1/responses -H 'Content-Type: application/json' -d '{"model":"gpt-4o","input":"Say hi"}'
```

Env: `PPLX_ACCOUNTS=accounts.txt`, `PPLX_API_KEY` (optional Bearer gate), `PPLX_MAX_CONCURRENT=2`, `PPLX_TIMEZONE=UTC`, `PPLX_INCLUDE_SOURCES=0`, `PPLX_DB_PATH=state.db`.

## Code Conventions & Common Patterns

- Naming: `snake_case` funcs/vars, `_leading_underscore` private helpers, `UPPER_SNAKE` constants (`ASK_URL`, `QUOTA_TTL=120s`, `COOLDOWN=300s`).
- Specs vs. accumulators: frozen `@dataclass` specs (`AskSpec`, `ChatStreamSpec`, `ResponsesStreamSpec` in `server.py:74-164`) vs. mutable `_`-prefixed helpers (`_StreamCapture`, `_MessageAccumulator`, `_TextAccumulator`).
- Typing/style: `from __future__ import annotations`, `X | None` unions, `TYPE_CHECKING`-gated imports, PEP 695 generics (`_safe[T, **P]` in `db_store.py`), Google-style docstrings (`Args/Returns/Raises`) — enforced by `ruff select=["ALL"]`.
- Async: everything streams as `AsyncIterator[AskEvent]`/`AsyncIterator[str]`; streaming formatters are pure translators over the same event stream collectors consume. All `Store` I/O via `asyncio.to_thread`; per-op SQLite open/commit/close.
- Concurrency: two-level guard — `PerplexityClient._slots` semaphore + `AccountPool._semaphores[i]` + `Account.active` least-in-flight pick (`pool.pick()` sorts by `(active, round-robin)`).
- Errors: `AskError(code,message,retryable)` → `_http_from_ask_error` (quota → `429 insufficient_quota`, else `502 upstream_error`); `db_store._safe` degrades `sqlite3.Error` to miss/no-op; unknown models default to `turbo`; attachment fetch drops failures; `RequestsError` caught as `OSError` subclass.
- IDs: public IDs `secrets.token_hex`, upstream IDs `uuid4`.
- DI: module singletons `pool = AccountPool(...)` + `store = Store()` at `server.py` import time; `lifespan()` calls `pool.start()`/`pool.close()`.
- Gotchas: `accounts.txt` hot-reloads on mtime (no restart); `include_sources` defaults OFF (`PPLX_INCLUDE_SOURCES=0`); `internet` mode fails on free accounts — default `copilot`/`concise`; never delete `state.db` to reset unless dropping conversations.

## Important Files

- Entry: `server.py` (`server:app`; routes `/v1/chat/completions`, `/v1/responses`, `/v1/responses/{id}`, `/v1/models`, `/healthz`).
- Key modules: `pplx_transport.py` (`PerplexityClient`, `build_payload`, `resolve_model`/`resolve_mode`), `account_pool.py` (`parse_accounts`, `pick`/`get`/`record_success`/`record_failure`), `db_store.py` (`get/save_thread`, `get/save_thread_by_answer`, `get/save_response`, caps 1024/512).
- Config: `pyproject.toml` (only config — metadata + `uv` + `ruff`/`ty`), `requirements.txt`, `uv.lock`, `.gitignore`.
- Ops/docs: `restart.sh`, `README.md`.
- Runtime data (gitignored, never commit): `accounts.txt` (`account N:` + Netscape cookie lines incl. `__Secure-next-auth.session-token`, `cf_clearance`), `state.db` (WAL), `server.log`.

## Runtime/Tooling Preferences

- Runtime: CPython `>=3.12` only (`pyproject.toml:requires-python`, `ruff target py312`). No Node/Bun/TS, Dockerfile, Compose, Makefile, CI.
- Package manager: `uv` primary (`uv.lock` v1 rev 3, `[tool.uv] package=false`), `pip install -r requirements.txt` documented fallback.
- Deps: `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `curl_cffi>=0.9`, `pydantic>=2.0`; dev-only `ty>=0.0.78`. Ruff binary external (not in lockfile).
- Lint/type: `ruff preview=true`, `lint.select=["ALL"]` (ignore only `incorrect-blank-line-before-class`, `multi-line-summary-second-line`); `ty rules.all="error"`, `strict-equality` + `strict-generic-narrowing`, `error-on-warning=true`.
- Never commit: `accounts*.txt`, `*.cookies`, `*.har`, `.venv/`, `*.log`, `*.db*`, `.env` (see `.gitignore`).
- Always use codebase-memory-mcp.

## Testing & QA

- No tests: no `tests/`, `conftest.py`, `*_test.py`, `pytest`/`coverage`/`tox` config or deps (verified by glob + `uv.lock` grep). Do not invent `pytest` commands.
- QA = static checks + live smoke: `ruff check`, `ty check`, `./restart.sh` + `curl /healthz`, then README `curl` probes above.
- Always use https://docs.astral.sh/ruff/ with everything enabled and https://docs.astral.sh/ty/ with everything enabled, then fix all of the issues. Make sure to actually fix all of the issues instead of suppressing them.
- Verification residue (`server.log`, `state.db`) is gitignored, not a test artifact.
