"""HTTP client for the OpenAI-compatible chat completions API.

Network errors, HTTP 429 and HTTP 5xx are retried up to three times, waiting
2, 4 and 8 seconds, or longer when the server sends a Retry-After header (at
most 60 seconds). Any other error status is returned at once. The API key is
sent in the Authorization header and never logged. Requests name this tool
in the User-Agent header; the API rejects the default urllib value.
"""

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

RETRY_WAITS = (2.0, 4.0, 8.0)
MAX_RETRY_AFTER = 60.0
USER_AGENT = "prompt-build/0.1.0"


@dataclass(frozen=True)
class HttpResult:
    status: int | None
    body: dict | None
    error: str | None
    attempts: int


@dataclass(frozen=True)
class _Attempt:
    status: int | None
    body: dict | None
    error: str | None
    retry_after: float | None = None


def _is_retryable(status: int | None) -> bool:
    return status is None or status == 429 or status >= 500


def _retry_after(value: str | None) -> float | None:
    try:
        return min(float(value), MAX_RETRY_AFTER) if value else None
    except ValueError:
        return None


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        team_id: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        opener: Callable[..., object] = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        if team_id:
            self._headers["X-Prime-Team-ID"] = team_id
        self._timeout = timeout
        self._sleep = sleep
        self._open = opener

    def complete(self, payload: dict) -> HttpResult:
        data = json.dumps(payload).encode("utf-8")
        attempts = 0
        while True:
            attempts += 1
            attempt = self._post(data)
            done = attempt.error is None or not _is_retryable(attempt.status)
            if done or attempts > len(RETRY_WAITS):
                return HttpResult(attempt.status, attempt.body, attempt.error, attempts)
            self._sleep(max(RETRY_WAITS[attempts - 1], attempt.retry_after or 0.0))

    def _post(self, data: bytes) -> _Attempt:
        request = urllib.request.Request(self._url, data=data, headers=self._headers, method="POST")
        try:
            with self._open(request, timeout=self._timeout) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            retry_after = _retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            return _Attempt(exc.code, None, f"HTTP {exc.code}: {detail}", retry_after)
        except (OSError, http.client.HTTPException) as exc:
            return _Attempt(None, None, f"network error: {type(exc).__name__}: {exc}")
        try:
            return _Attempt(status, json.loads(raw), None)
        except json.JSONDecodeError:
            return _Attempt(status, None, "response body is not JSON")
