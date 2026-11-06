"""Check the renderer in one Prime sandbox against the fixture SVGs.

Starts one sandbox from the render image (render_image/Dockerfile) with all
outbound traffic blocked, copies in tests/fixtures/*.svg,
renders each fixture, and compares the not-blank and motion decisions with
tests/fixtures/expected.json. It reports the sandbox start-up time and the
time per render, saves the frames under runs/renderer_check/, and always
deletes the sandbox at the end.

Usage, from this environment's folder:
    RENDER_IMAGE=<image reference> python scripts/check_renderer.py
RENDER_IMAGE is the reference `prime images push` printed for the render
image. The API key is read from PRIME_API_KEY (or the Prime CLI login).
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

from prime_sandboxes import APIClient, CreateSandboxRequest, SandboxClient

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from svg_scoring.motion import CHANGE_TOLERANCE, frame_times, judge_frames  # noqa: E402

TIME_LIMIT = 60

RENDER_SCRIPT = "/opt/render/render_frames.py"
WORKDIR = "/tmp/render"
RENDER_TIMEOUT = 120


def main() -> int:
    image = os.environ["RENDER_IMAGE"]
    fixtures = HERE / "tests" / "fixtures"
    expected = json.loads((fixtures / "expected.json").read_text(encoding="utf-8"))
    out = HERE / "runs" / "renderer_check"
    out.mkdir(parents=True, exist_ok=True)
    client = SandboxClient(APIClient())
    started = time.monotonic()
    sandbox = client.create(CreateSandboxRequest(
        name="renderer-check", docker_image=image, cpu_cores=1, memory_gb=2, disk_size_gb=10,
        timeout_minutes=20, idle_timeout_minutes=10, network_denylist=["0.0.0.0/0"],
    ))
    report = {"results": {}}
    try:
        client.wait_for_creation(sandbox.id)
        report["startup_seconds"] = round(time.monotonic() - started, 1)
        print(f"sandbox up in {report['startup_seconds']} s")
        client.execute_command(sandbox.id, f"mkdir -p {WORKDIR}")
        mismatches = 0
        for name, want in sorted(expected.items()):
            text = (fixtures / name).read_text(encoding="utf-8")
            client.upload_bytes(sandbox.id, f"{WORKDIR}/{name}", text.encode("utf-8"), name)
            times = " ".join(str(t) for t in frame_times(text))
            began = time.monotonic()
            result = client.execute_command(
                sandbox.id,
                f"mkdir -p out_{name} && python3 {RENDER_SCRIPT} {name} out_{name} {CHANGE_TOLERANCE} {TIME_LIMIT} {times}",
                working_dir=WORKDIR, timeout=RENDER_TIMEOUT,
            )
            seconds = round(time.monotonic() - began, 2)
            if result.exit_code != 0:
                print(f"{name}: render failed\n{result.stderr[-1500:]}")
                report["results"][name] = {"error": result.stderr[-1500:]}
                mismatches += 1
                continue
            measures = json.loads(client.read_file(sandbox.id, f"{WORKDIR}/out_{name}/result.json").content)
            measures["frames"] = [client.read_file(sandbox.id, f"{WORKDIR}/out_{name}/frame{i}.b64").content
                                  for i in range(measures.get("frame_count", 0))]
            sizes = [len(frame) for frame in measures["frames"]]
            if "error" in measures:
                print(f"{name}: {measures['error']}")
                mismatches += 1
                continue
            checks = judge_frames(measures)
            got = {"not_blank": checks.not_blank, "moves": checks.moves}
            ok = got == want
            mismatches += not ok
            for i, frame in enumerate(measures["frames"]):
                (out / f"{Path(name).stem}_frame{i}.png").write_bytes(base64.b64decode(frame))
            report["browser"] = measures.get("browser")
            report["results"][name] = {
                "ok": ok, "got": got, "want": want, "seconds": seconds, "times": measures["times"],
                "frame_b64_bytes": sizes, "colour_shares": measures["colour_shares"][:6], "largest_change": round(checks.largest_change, 4),
                "largest_brightness_change": round(checks.largest_brightness_change, 2),
            }
            print(f"{name}: {'ok' if ok else 'MISMATCH'} {got} in {seconds} s "
                  f"(change {checks.largest_change:.4f}, brightness {checks.largest_brightness_change:.1f})")
    finally:
        client.delete(sandbox.id)
        report["total_seconds"] = round(time.monotonic() - started, 1)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"sandbox deleted; total {report['total_seconds']} s; report: {out / 'report.json'}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
