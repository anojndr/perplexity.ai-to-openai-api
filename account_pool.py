# Copyright 2026 perplexity-to-openai contributors

"""Load-balanced pool of Perplexity accounts from a Netscape cookie file.

accounts.txt holds any number of account blocks:
    account 1:
    <netscape cookie lines>
    account 2:
    ...

The file is re-read when its mtime changes, so adding accounts takes effect
without restarting the server. Accounts are skipped while quota-exhausted,
failing, or cooling down; selection prefers the least-loaded healthy account.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from pplx_transport import PerplexityClient

log = logging.getLogger("pplx.accounts")

QUOTA_TTL = 120.0  # rate-limit/status cache
COOLDOWN = 300.0  # after consecutive failures
FAIL_THRESHOLD = 3  # consecutive failures -> cooldown
NETSCAPE_MIN_COLUMNS = 7  # domain, flag, path, secure, expiry, name, value


class AccountSpec(TypedDict):
    """Cookies and label for a single account block.

    Attributes:
        label: Display label parsed from the account header line.
        cookies: Cookie name to value mapping for the account.

    """

    label: str
    cookies: dict[str, str]


@dataclass
class Account:
    """One pooled Perplexity account and its health state.

    Attributes:
        index: Position of the account within the pool.
        label: Display label parsed from the account header line.
        cookies: Cookie name to value mapping for the session.
        client: Transport client bound to this account's cookies.
        active: Number of requests currently using the account.
        consecutive_failures: Failure count since the last success.
        cooldown_until: Epoch time until which the account is skipped.
        quota_known: Last known quota flag, or None when unknown.
        quota_checked_at: Epoch time of the last quota refresh.
        last_error: Human-readable reason for the last failure, if any.

    """

    index: int
    label: str
    cookies: dict[str, str]
    client: PerplexityClient
    active: int = 0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    quota_known: bool | None = None
    quota_checked_at: float = 0.0
    last_error: str | None = None

    def healthy(self, now: float) -> bool:
        """Check whether the account may receive new work.

        Args:
            now: Current time in seconds since the epoch.

        Returns:
            True when the account is neither cooling down nor quota-exhausted.

        """
        return now >= self.cooldown_until and self.quota_known is not False


def parse_accounts(path: str) -> list[AccountSpec]:
    """Parse the Netscape-cookie account file into cookie dicts.

    Args:
        path: Filesystem path of the accounts file to read.

    Returns:
        Specs for every account block that yielded at least one cookie.

    """
    accounts: list[AccountSpec] = []
    current: AccountSpec | None = None
    with Path(path).open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("account"):
                current = {"label": line, "cookies": {}}
                accounts.append(current)
                continue
            if current is None or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < NETSCAPE_MIN_COLUMNS:
                continue
            current["cookies"][parts[5]] = parts[6]
    return [a for a in accounts if a["cookies"]]


class AccountPool:
    """Round-robin pool of Perplexity accounts backed by a cookie file.

    The pool reloads the file when its mtime changes and steers new work to
    the least-loaded healthy account.

    Attributes:
        path: Filesystem path of the Netscape-cookie accounts file.
        max_concurrent: Maximum concurrent sessions per account.

    """

    def __init__(self, path: str, *, max_concurrent: int = 2) -> None:
        """Initialize the pool.

        Args:
            path: Filesystem path of the Netscape-cookie accounts file.
            max_concurrent: Maximum concurrent sessions per account.

        """
        self.path = path
        self.max_concurrent = max_concurrent
        self._accounts: list[Account] = []
        self._rr = 0
        self._mtime: float | None = None
        self._semaphores: dict[int, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Load accounts from disk into the pool."""
        await self.reload()

    async def reload(self) -> None:
        """Reload the accounts file when its mtime changed.

        Existing clients are kept for unchanged accounts so sessions are
        reused; replaced accounts have their old client closed.
        """
        try:
            st = await asyncio.to_thread(Path(self.path).stat)
            if self._mtime == st.st_mtime and self._accounts:
                return
            raw = parse_accounts(self.path)
        except FileNotFoundError:
            log.exception("accounts file missing: %s", self.path)
            return
        except (OSError, ValueError, RuntimeError):
            log.exception("failed to parse accounts: %s", self.path)
            return
        async with self._lock:
            if (
                self._mtime is not None
                and self._mtime == st.st_mtime
                and self._accounts
            ):
                return
            # Keep existing clients for unchanged accounts (session reuse).
            existing = {a.index: a for a in self._accounts}
            new_list = []
            for i, spec in enumerate(raw):
                old = existing.get(i)
                if old is not None and old.cookies == spec["cookies"]:
                    new_list.append(old)
                else:
                    if old is not None:
                        await old.client.close()
                    new_list.append(
                        Account(
                            index=i,
                            label=spec["label"],
                            cookies=spec["cookies"],
                            client=PerplexityClient(
                                spec["cookies"],
                                max_concurrent=self.max_concurrent,
                            ),
                        ),
                    )
            self._accounts = new_list
            self._semaphores = {
                a.index: asyncio.Semaphore(self.max_concurrent) for a in new_list
            }
            self._mtime = st.st_mtime
            log.info("accounts loaded: %d", len(new_list))

    @property
    def size(self) -> int:
        """Number of accounts currently in the pool.

        Returns:
            Count of loaded accounts.

        """
        return len(self._accounts)

    @staticmethod
    async def _refresh_quota(acct: Account) -> None:
        """Refresh the cached quota flag for one account.

        Skips the check while the cached value is still fresh and records
        failures at debug level so one slow status probe never breaks picks.

        Args:
            acct: Account whose quota flag should be refreshed.

        """
        now = time.time()
        if now - acct.quota_checked_at < QUOTA_TTL:
            return
        try:
            acct.quota_known = await acct.client.quota_available()
            acct.quota_checked_at = now
        except (OSError, RuntimeError, ValueError) as exc:
            log.debug("quota refresh failed: %s", exc)

    async def pick(self) -> tuple[Account, asyncio.Semaphore] | None:
        """Pick the healthiest account for new work.

        The least-loaded healthy account wins with a round-robin tie-break.

        Returns:
            Account and semaphore pair, or None when all accounts are unhealthy.

        """
        await self.reload()
        await asyncio.sleep(0)  # let pending health updates land
        now = time.time()
        async with self._lock:
            candidates = [a for a in self._accounts if a.healthy(now)]
            if not candidates:
                return None
            for a in candidates:
                await AccountPool._refresh_quota(a)
            candidates = [a for a in candidates if a.healthy(time.time())]
            if not candidates:
                return None
            candidates.sort(
                key=lambda a: (a.active, (a.index - self._rr) % len(self._accounts)),
            )
            self._rr = (self._rr + 1) % max(len(self._accounts), 1)
            acct = candidates[0]
            return acct, self._semaphores[acct.index]

    def get(self, index: int) -> tuple[Account, asyncio.Semaphore] | None:
        """Return a specific account by index when it is healthy.

        Args:
            index: Position of the account in the pool.

        Returns:
            Account and semaphore pair, or None when missing or unhealthy.

        """
        now = time.time()
        acct = self._accounts[index] if 0 <= index < len(self._accounts) else None
        if acct is None or not acct.healthy(now):
            return None
        return acct, self._semaphores[acct.index]

    @staticmethod
    def record_success(acct: Account) -> None:
        """Mark an account request as successful.

        Args:
            acct: Account that completed a request successfully.

        """
        acct.consecutive_failures = 0
        acct.last_error = None
        acct.quota_known = None  # re-check on next pick

    @staticmethod
    def record_failure(acct: Account, error: str, *, quota: bool = False) -> None:
        """Record a failed request against an account.

        Args:
            acct: Account that failed a request.
            error: Human-readable failure reason for status reporting.
            quota: Whether the failure signals quota exhaustion.

        """
        acct.consecutive_failures += 1
        acct.last_error = error
        if quota:
            acct.quota_known = False
            acct.quota_checked_at = time.time()
        if acct.consecutive_failures >= FAIL_THRESHOLD:
            acct.cooldown_until = time.time() + COOLDOWN
            acct.consecutive_failures = 0
            log.warning("account %d cooling down: %s", acct.index, error)

    def status(self) -> list[dict[str, Any]]:
        """Summarize health state for every account.

        Returns:
            Per-account status dicts with load, health, quota, and error fields.

        """
        now = time.time()
        return [
            {
                "index": a.index,
                "label": a.label,
                "active": a.active,
                "healthy": a.healthy(now),
                "quota_available": a.quota_known,
                "last_error": a.last_error,
                "cooldown_until": a.cooldown_until,
            }
            for a in self._accounts
        ]

    async def close(self) -> None:
        """Close every account client session."""
        for a in self._accounts:
            await a.client.close()
