"""Render an SVG in its own Prime sandbox (README, "Scoring").

Each call starts one sandbox from the render image with all outbound traffic
denied, writes the SVG, runs the render script that the image holds, reads
the measures and frames back, and deletes the sandbox. The policy's SVG
therefore never runs anywhere with network access or with any credential.

A render failure that the SVG causes comes back as {"error": ...} and is
scored 0. A failure of the sandbox itself raises.
"""

import asyncio
import json
import logging
import shlex

from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

RENDER_SCRIPT = "/opt/render/render_frames.py"
WORKDIR = "/tmp/render"
DENY_ALL = ["0.0.0.0/0"]
COMMAND_MARGIN = 60

logger = logging.getLogger(__name__)

# Process-wide cap on sandboxes in flight, so a burst of rollouts cannot exceed the account's
# creation rate. Created lazily so that it binds to the running event loop.
_slots: asyncio.Semaphore | None = None


def _semaphore(limit: int) -> asyncio.Semaphore:
    global _slots
    if _slots is None:
        _slots = asyncio.Semaphore(limit)
    return _slots


async def _read_text(client: AsyncSandboxClient, sandbox_id: str, path: str) -> str:
    return (await client.read_file(sandbox_id, path)).content


async def render(svg: str, times: list[float], tolerance: int, image: str, *,
                 max_sandboxes: int, time_limit: int) -> dict:
    """Measures for one SVG, with the frames as base64 PNG under "frames"."""
    async with _semaphore(max_sandboxes), AsyncSandboxClient() as client:
        sandbox = await client.create(CreateSandboxRequest(
            name="isometric-render", docker_image=image, cpu_cores=1, memory_gb=2,
            timeout_minutes=10, idle_timeout_minutes=5, network_denylist=DENY_ALL,
        ))
        try:
            await client.wait_for_creation(sandbox.id)
            await client.execute_command(sandbox.id, f"mkdir -p {WORKDIR}", timeout=60)
            await client.upload_bytes(sandbox.id, f"{WORKDIR}/scene.svg", svg.encode("utf-8"), "scene.svg")
            argv = ["python3", RENDER_SCRIPT, "scene.svg", WORKDIR, str(tolerance), str(time_limit),
                    *map(str, times)]
            result = await client.execute_command(
                sandbox.id, " ".join(shlex.quote(a) for a in argv), working_dir=WORKDIR,
                timeout=time_limit + COMMAND_MARGIN)
            if result.exit_code != 0:
                raise RuntimeError(f"render script failed: {result.stderr[-500:]}")
            measures = json.loads(await _read_text(client, sandbox.id, f"{WORKDIR}/result.json"))
            measures["frames"] = [await _read_text(client, sandbox.id, f"{WORKDIR}/frame{i}.b64")
                                  for i in range(measures.get("frame_count", 0))]
            return measures
        finally:
            try:
                await client.delete(sandbox.id)
            except Exception:
                logger.exception("could not delete render sandbox %s", sandbox.id)
