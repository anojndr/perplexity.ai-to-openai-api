# Copyright 2026 perplexity-to-openai contributors

"""SQLite storage for persisting threads and responses across restarts.

Connections are opened per operation and always closed; runtime sqlite
failures degrade to cache-miss / no-op so a storage problem never fails a
request whose upstream answer already succeeded.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final, overload

from pplx_transport import ThreadState

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

log = logging.getLogger("pplx.store")

DB_PATH = os.environ.get("PPLX_DB_PATH", "state.db")

_ALLOWED_EVICT_TABLES: Final[dict[str, tuple[str, str]]] = {
    "threads": ("key", "last_used"),
    "threads_by_answer": ("answer_hash", "last_used"),
    "responses": ("id", "created_at"),
}
_EVICT_STATEMENTS: Final[dict[str, str]] = {
    "threads": """DELETE FROM threads WHERE key IN (
        SELECT key FROM threads ORDER BY last_used DESC LIMIT -1 OFFSET ?
    )""",
    "threads_by_answer": """DELETE FROM threads_by_answer
        WHERE answer_hash IN (
            SELECT answer_hash FROM threads_by_answer
            ORDER BY last_used DESC LIMIT -1 OFFSET ?
        )""",
    "responses": """DELETE FROM responses WHERE id IN (
        SELECT id FROM responses ORDER BY created_at DESC LIMIT -1 OFFSET ?
    )""",
}


@overload
def _safe[T, **P](
    default: None = None,
) -> Callable[[Callable[P, T]], Callable[P, T | None]]: ...
@overload
def _safe[D, T, **P](
    default: D,
) -> Callable[[Callable[P, T]], Callable[P, T | D]]: ...
def _safe[D, T, **P](
    default: D | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T | D | None]]:
    """Return a decorator degrading sqlite3 failures to ``default``.

    The wrapped callable keeps its signature; only its return type is
    widened with the type of ``default``. Persistence problems must
    never break requests, so sqlite3.Error is logged and swallowed.

    Args:
        default: Value returned when the wrapped call fails.

    Returns:
        A decorator converting sqlite3.Error into ``default``.

    """

    def deco(fn: Callable[P, T]) -> Callable[P, T | D | None]:
        """Wrap one store callable with sqlite3 error handling.

        Args:
            fn: Callable performing a single SQLite operation.

        Returns:
            The wrapped callable returning ``default`` on failure.

        """

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | D | None:
            """Invoke the callable, degrading storage errors to default.

            Args:
                *args: Positional arguments forwarded to ``fn``.
                **kwargs: Keyword arguments forwarded to ``fn``.

            Returns:
                The wrapped result, or ``default`` on sqlite3.Error.

            """
            try:
                return fn(*args, **kwargs)
            except sqlite3.Error:
                log.exception(
                    "store.%s failed",
                    getattr(fn, "__name__", type(fn).__name__),
                )
                return default

        return wrapper

    return deco


class Store:
    """Persist threads and responses in SQLite across restarts.

    Connections are opened per operation and always closed, and every
    public accessor degrades storage failures to cache-miss / no-op.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        """Initialize the store, creating tables when missing.

        Args:
            db_path: Filesystem path of the SQLite database file.

        """
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Open a per-operation SQLite connection.

        Yields:
            Open connection with Row factory; commits on clean exit
            and always closes.

        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and indexes when missing."""
        # journal_mode=WAL persists in the database file; set once here.
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS threads (
                    key TEXT PRIMARY KEY,
                    backend_uuid TEXT,
                    read_write_token TEXT,
                    slug TEXT,
                    title TEXT,
                    model TEXT,
                    mode TEXT,
                    account_id INTEGER,
                    last_used REAL
                );

                CREATE TABLE IF NOT EXISTS threads_by_answer (
                    answer_hash TEXT PRIMARY KEY,
                    backend_uuid TEXT,
                    read_write_token TEXT,
                    slug TEXT,
                    title TEXT,
                    model TEXT,
                    mode TEXT,
                    account_id INTEGER,
                    last_used REAL
                );

                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    backend_uuid TEXT,
                    read_write_token TEXT,
                    slug TEXT,
                    title TEXT,
                    model TEXT,
                    mode TEXT,
                    account_id INTEGER,
                    last_used REAL,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_threads_last_used
                    ON threads(last_used);
                CREATE INDEX IF NOT EXISTS idx_threads_by_answer_last_used
                    ON threads_by_answer(last_used);
                CREATE INDEX IF NOT EXISTS idx_responses_created_at
                    ON responses(created_at);
            """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_thread(row: sqlite3.Row | None) -> ThreadState | None:
        """Convert a threads-like result row into a ThreadState.

        Args:
            row: SQLite row from a threads-like table, if any.

        Returns:
            The decoded thread, or None on cache miss.

        """
        if row is None:
            return None
        return ThreadState(
            backend_uuid=row["backend_uuid"],
            read_write_token=row["read_write_token"],
            slug=row["slug"],
            title=row["title"],
            model=row["model"],
            mode=row["mode"] or "copilot",
            account_id=row["account_id"],
            last_used=row["last_used"] or time.time(),
        )

    # -- threads (conversation-key -> handle) --------------------------------

    @_safe()
    def get_thread(self, key: str) -> ThreadState | None:
        """Return the thread stored for a conversation key.

        Args:
            key: Conversation key identifying the stored thread.

        Returns:
            The stored thread, or None on cache miss or storage error.

        """
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM threads WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE threads SET last_used = ? WHERE key = ?",
                (time.time(), key),
            )
            return self._row_to_thread(row)

    @_safe()
    def save_thread(self, key: str, thread: ThreadState, cap: int = 1024) -> None:
        """Persist a thread under a conversation key.

        The least-recently-used rows beyond ``cap`` are evicted.

        Args:
            key: Conversation key identifying the stored thread.
            thread: Thread handle to persist; ``last_used`` is refreshed.
            cap: Maximum rows kept in the threads table.

        """
        now = time.time()
        thread.last_used = now
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO threads (
                    key, backend_uuid, read_write_token, slug, title,
                    model, mode, account_id, last_used
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    backend_uuid = excluded.backend_uuid,
                    read_write_token = excluded.read_write_token,
                    slug = excluded.slug,
                    title = excluded.title,
                    model = excluded.model,
                    mode = excluded.mode,
                    account_id = excluded.account_id,
                    last_used = excluded.last_used
            """,
                (
                    key,
                    thread.backend_uuid,
                    thread.read_write_token,
                    thread.slug,
                    thread.title,
                    thread.model,
                    thread.mode,
                    thread.account_id,
                    now,
                ),
            )
            self._evict(conn, "threads", "key", "last_used", cap)

    # -- threads_by_answer (echoed-answer hash -> handle) --------------------

    @_safe()
    def get_thread_by_answer(self, answer_hash: str) -> ThreadState | None:
        """Return the thread stored for an echoed-answer hash.

        Args:
            answer_hash: Hash identifying the stored thread.

        Returns:
            The stored thread, or None on cache miss or storage error.

        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM threads_by_answer WHERE answer_hash = ?",
                (answer_hash,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE threads_by_answer SET last_used = ? WHERE answer_hash = ?",
                (time.time(), answer_hash),
            )
            return self._row_to_thread(row)

    @_safe()
    def save_thread_by_answer(
        self,
        answer_hash: str,
        thread: ThreadState,
        cap: int = 1024,
    ) -> None:
        """Persist a thread under an echoed-answer hash.

        The least-recently-used rows beyond ``cap`` are evicted.

        Args:
            answer_hash: Hash identifying the stored thread.
            thread: Thread handle to persist; ``last_used`` is refreshed.
            cap: Maximum rows kept in the threads_by_answer table.

        """
        now = time.time()
        thread.last_used = now
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO threads_by_answer (
                    answer_hash, backend_uuid, read_write_token, slug,
                    title, model, mode, account_id, last_used
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(answer_hash) DO UPDATE SET
                    backend_uuid = excluded.backend_uuid,
                    read_write_token = excluded.read_write_token,
                    slug = excluded.slug,
                    title = excluded.title,
                    model = excluded.model,
                    mode = excluded.mode,
                    account_id = excluded.account_id,
                    last_used = excluded.last_used
            """,
                (
                    answer_hash,
                    thread.backend_uuid,
                    thread.read_write_token,
                    thread.slug,
                    thread.title,
                    thread.model,
                    thread.mode,
                    thread.account_id,
                    now,
                ),
            )
            self._evict(conn, "threads_by_answer", "answer_hash", "last_used", cap)

    # -- responses -----------------------------------------------------------

    @_safe()
    def get_response(self, resp_id: str) -> dict[str, Any] | None:
        """Return the stored response payload for an id.

        Args:
            resp_id: Response id identifying the stored payload.

        Returns:
            The stored payload, or None on cache miss or storage error.

        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM responses WHERE id = ?",
                (resp_id,),
            ).fetchone()
            if row is None:
                return None
            loaded: Any = json.loads(row["data_json"])
            if not isinstance(loaded, dict):
                return None
            data: dict[str, Any] = loaded
            thread = self._row_to_thread(row)
            if thread is not None:
                data["_thread"] = thread
            return data

    @_safe()
    def save_response(self, resp_id: str, data: dict[str, Any], cap: int = 512) -> None:
        """Persist a response payload under an id.

        The least-recently-used rows beyond ``cap`` are evicted.

        Args:
            resp_id: Response id identifying the stored payload.
            data: Payload to persist; a ``_thread`` entry is stored in
                columns instead of the JSON blob.
            cap: Maximum rows kept in the responses table.

        """
        candidate = data.get("_thread")
        thread: ThreadState | None = (
            candidate if isinstance(candidate, ThreadState) else None
        )
        created_raw = data.get("created_at")
        created_at: float = (
            created_raw if isinstance(created_raw, (int, float)) else time.time()
        )
        clean = {k: v for k, v in data.items() if k != "_thread"}
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO responses (
                    id, data_json, backend_uuid, read_write_token,
                    slug, title, model, mode, account_id, last_used,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    data_json = excluded.data_json,
                    backend_uuid = excluded.backend_uuid,
                    read_write_token = excluded.read_write_token,
                    slug = excluded.slug,
                    title = excluded.title,
                    model = excluded.model,
                    mode = excluded.mode,
                    account_id = excluded.account_id,
                    last_used = excluded.last_used,
                    created_at = excluded.created_at
            """,
                (
                    resp_id,
                    json.dumps(clean),
                    thread.backend_uuid if thread else None,
                    thread.read_write_token if thread else None,
                    thread.slug if thread else None,
                    thread.title if thread else None,
                    thread.model if thread else None,
                    thread.mode if thread else "copilot",
                    thread.account_id if thread else None,
                    thread.last_used if thread else time.time(),
                    created_at,
                ),
            )
            self._evict(conn, "responses", "id", "created_at", cap)

    # -- misc ------------------------------------------------------------------

    @staticmethod
    def _evict(
        conn: sqlite3.Connection,
        table: str,
        pk: str,
        order_col: str,
        cap: int,
    ) -> None:
        """Delete rows beyond ``cap``, keeping the newest ones.

        Only allowlisted ``(table, pk, order_col)`` triples run, and the
        statement is a precomputed constant, so no SQL is ever built
        from call arguments.

        Args:
            conn: Open SQLite connection running the eviction.
            table: Table to evict from; must be allowlisted.
            pk: Primary-key column; must match the allowlisted column.
            order_col: Recency column; must match the allowlisted one.
            cap: Number of newest rows to keep; non-positive keeps all.

        Raises:
            ValueError: If the identifier triple is not allowlisted.

        """
        if cap <= 0:
            return
        expected = _ALLOWED_EVICT_TABLES.get(table)
        if expected is None or expected != (pk, order_col):
            msg = f"unsupported eviction target: {table}.{pk}"
            raise ValueError(msg)
        conn.execute(_EVICT_STATEMENTS[table], (cap,))

    @_safe(default=0)
    def count_threads(self) -> int:
        """Return the number of stored conversation threads.

        Returns:
            The thread count, or 0 on storage error.

        """
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM threads").fetchone()
            if row is None:
                return 0
            value = row["c"]
            return value if isinstance(value, int) else 0

    @_safe(default=0)
    def count_responses(self) -> int:
        """Return the number of stored responses.

        Returns:
            The response count, or 0 on storage error.

        """
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM responses").fetchone()
            if row is None:
                return 0
            value = row["c"]
            return value if isinstance(value, int) else 0
