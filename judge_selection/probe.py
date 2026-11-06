"""Capability probe for the judge shortlist.

Each candidate gets one request with four small PNG images, solid squares
of known colours, and is asked to name the colours in order. The request
asks for a JSON schema. If the API refuses the schema, the request is sent
once more with the format given in words only. A model passes if it names
all four colours in the right order; it supports the schema if the first
request is accepted and its answer matches the schema.

Every request and reply is written to runs/probe/results.jsonl.

Usage, from the repository root:
    python judge_selection/probe.py
The API key is read from PRIME_API_KEY, which may be set in .env at the
repository root. That file is ignored by git.
"""

import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://api.pinference.ai/api/v1/chat/completions"
USER_AGENT = "judge-selection/0.1.0"
TIMEOUT = 120
MAX_TOKENS = 1024

CANDIDATES = (
    "google/gemma-3-27b-it",
    "google/gemini-2.5-flash-lite",
    "meta-llama/llama-4-maverick",
    "google/gemini-3.8-flash",
    "google/gemini-4-argon",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.7",
    "meta/muse-spark-open",
)

COLOURS = (("red", (220, 30, 30)), ("green", (30, 160, 60)), ("blue", (30, 70, 220)), ("yellow", (240, 210, 20)))

QUESTION = (
    "You are shown four images. Each shows one solid coloured square on a white background. "
    "Name the colour of the square in each image, in the order the images are given. "
    'Use one plain word per colour. Answer with one JSON object: {"colours": ["...", "...", "...", "..."]}'
)

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "colours",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"colours": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4}},
            "required": ["colours"],
            "additionalProperties": False,
        },
    },
}

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def png(rgb: tuple[int, int, int], size: int = 64, border: int = 12) -> bytes:
    """A size x size PNG: a solid square of `rgb` inside a white border."""
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            inside = border <= x < size - border and border <= y < size - border
            row += bytes(rgb if inside else (255, 255, 255))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def payload(model: str, with_schema: bool) -> dict:
    content = [{"type": "text", "text": QUESTION}]
    for _, rgb in COLOURS:
        url = "data:image/png;base64," + base64.b64encode(png(rgb)).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": url}})
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": MAX_TOKENS}
    if with_schema:
        body["response_format"] = SCHEMA
    return body


def post(body: dict, api_key: str) -> tuple[int | None, dict | None, str | None, float]:
    request = urllib.request.Request(
        BASE_URL, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, json.loads(response.read()), None, time.monotonic() - start
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return exc.code, None, f"HTTP {exc.code}: {detail}", time.monotonic() - start
    except (OSError, json.JSONDecodeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}", time.monotonic() - start


def answer(body: dict | None) -> list[str] | None:
    """The colours named in the reply, or None if the reply is not the asked-for JSON."""
    try:
        content = body["choices"][0]["message"]["content"] or ""
        start, end = content.index("{"), content.rindex("}") + 1
        colours = json.loads(content[start:end])["colours"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not isinstance(colours, list) or not all(isinstance(c, str) for c in colours):
        return None
    return [c.strip().lower() for c in colours]


def load_key() -> str:
    env = REPO / ".env"
    if "PRIME_API_KEY" not in os.environ and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == "PRIME_API_KEY":
                os.environ["PRIME_API_KEY"] = value.strip().strip("'\"")
    key = os.environ.get("PRIME_API_KEY", "")
    if not key:
        sys.exit("PRIME_API_KEY is not set")
    return key


def main() -> int:
    api_key = load_key()
    out = ROOT / "runs" / "probe"
    out.mkdir(parents=True, exist_ok=True)
    expected = [name for name, _ in COLOURS]
    rows = []
    with (out / "results.jsonl").open("a", encoding="utf-8", newline="\n") as log:
        for model in CANDIDATES:
            result = {"model": model, "schema_accepted": False, "schema_ok": False, "images_ok": False}
            for with_schema in (True, False):
                body = payload(model, with_schema)
                status, reply, error, seconds = post(body, api_key)
                colours = answer(reply)
                log.write(json.dumps({
                    "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "model": model, "with_schema": with_schema, "status": status, "error": error,
                    "seconds": round(seconds, 2), "response": reply, "colours": colours,
                }, ensure_ascii=False) + "\n")
                log.flush()
                if with_schema and error is None:
                    result["schema_accepted"] = True
                    result["schema_ok"] = colours is not None and len(colours) == 4
                if error is None:
                    result["images_ok"] = colours == expected
                    result["answer"] = colours
                    result["seconds"] = round(seconds, 2)
                    result["cost"] = ((reply or {}).get("usage") or {}).get("cost")
                    result["served_model"] = (reply or {}).get("model")
                    break
                result["schema_error" if with_schema else "error"] = error[:160]
                if status == 402:
                    break
            rows.append(result)
            print(json.dumps(result))
            if "HTTP 402" in result.get("schema_error", "") + result.get("error", ""):
                print("The account has no balance left. Stopping.")
                break
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
