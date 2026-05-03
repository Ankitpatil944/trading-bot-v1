"""
Kite Connect authentication: login URL, request_token exchange, token persistence.

Run once interactively to obtain access_token, then reuse persisted token until expiry.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _retry(
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    label: str,
) -> T:
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — network/API surface is broad
            last = exc
            delay = base_delay * (2**i)
            logger.warning(
                "retry_failed",
                extra={"event": "retry", "label": label, "attempt": i + 1, "error": str(exc)},
            )
            time.sleep(delay)
    assert last is not None
    raise last


@dataclass
class TokenStore:
    """File-backed access token persistence."""

    path: Path

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("token_load_failed", extra={"event": "auth", "error": str(exc)})
            return None

    def save(self, api_key: str, access_token: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"api_key": api_key, "access_token": access_token}
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("token_saved", extra={"event": "auth", "path": str(self.path)})


class KiteAuth:
    """Builds a logged-in `KiteConnect` client from env, file, or interactive token."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        token_file: str,
        *,
        max_retries: int = 5,
        retry_base_seconds: float = 0.5,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.store = TokenStore(Path(token_file))
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def login_url(self) -> str:
        kite = KiteConnect(api_key=self.api_key)
        return kite.login_url()

    def create_session(self, request_token: str) -> str:
        kite = KiteConnect(api_key=self.api_key)
        data = _retry(
            lambda: kite.generate_session(request_token, api_secret=self.api_secret),
            attempts=self.max_retries,
            base_delay=self.retry_base_seconds,
            label="generate_session",
        )
        access_token = data["access_token"]
        self.store.save(self.api_key, access_token)
        logger.info(
            "session_created",
            extra={"event": "auth", "user_id": data.get("user_id"), "public_token": data.get("public_token")},
        )
        return access_token

    def resolve_access_token(self, env_token: Optional[str]) -> str:
        if env_token:
            return env_token
        blob = self.store.load()
        if blob and blob.get("access_token"):
            return str(blob["access_token"])
        raise RuntimeError(
            "No access token: set KITE_ACCESS_TOKEN or run interactive login "
            "(open login_url, complete login, pass request_token to create_session)."
        )

    def connect(self, access_token: Optional[str] = None) -> KiteConnect:
        token = self.resolve_access_token(access_token or os.environ.get("KITE_ACCESS_TOKEN"))
        kite = KiteConnect(api_key=self.api_key, access_token=token)
        # Validate
        prof = _retry(
            kite.profile,
            attempts=self.max_retries,
            base_delay=self.retry_base_seconds,
            label="profile",
        )
        logger.info(
            "kite_connected",
            extra={"event": "auth", "user_name": prof.get("user_name"), "user_id": prof.get("user_id")},
        )
        return kite
