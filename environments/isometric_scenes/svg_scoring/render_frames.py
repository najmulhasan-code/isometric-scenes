"""Render one SVG to frames and measure them. Runs inside the render sandbox.

Usage: python render_frames.py <svg file> <out dir> <tolerance> <time limit> <time> [<time> ...]

The SVG is parsed as XML and placed in a blank 512x512 page. After fonts are
ready, every SMIL, CSS and Web animation is paused and set to each frame
time, and the page is captured as PNG. The frames are then measured in the
page itself: the share of the first frame covered by each colour (colours
quantised to 4 bits per channel; shares under 0.1% are left out), each
frame's mean brightness, and for every pair of frames the share of pixels
whose colour changed by more than `tolerance` in some channel.

Each frame is written to <out dir>/frame<i>.b64 as base64 PNG, and the
measures to <out dir>/result.json. All network requests are refused.

A failure the SVG causes (the browser cannot parse it, crashes, or takes
longer than the time limit in seconds) is reported as {"error": ...} in
result.json, with exit status 0, so it is scored as the policy's failure.
Any other exit status means the sandbox itself failed.
"""

import base64
import json
import os
import signal
import sys

# Sandboxes do not keep the image's environment variables, so the browser folder is set here.
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/ms-playwright")

from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

SIZE = 512

PAGE = f"""<!doctype html><html><head><style>
html, body {{ margin: 0; width: {SIZE}px; height: {SIZE}px; overflow: hidden; background: #fff; }}
#stage > svg {{ display: block; width: {SIZE}px; height: {SIZE}px; }}
</style></head><body><div id="stage"></div></body></html>"""

INSERT = """(text) => {
  const doc = new DOMParser().parseFromString(text, "image/svg+xml");
  if (doc.querySelector("parsererror")) return false;
  const svg = document.importNode(doc.documentElement, true);
  document.getElementById("stage").appendChild(svg);
  return true;
}"""

SEEK = """(t) => {
  const svg = document.querySelector("#stage > svg");
  if (svg.pauseAnimations) { svg.pauseAnimations(); svg.setCurrentTime(t); }
  for (const a of document.getAnimations()) { a.pause(); a.currentTime = t * 1000; }
  return new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
}"""

MEASURE = """async ([frames, tolerance]) => {
  const images = [];
  for (const b64 of frames) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d");
    context.drawImage(bitmap, 0, 0);
    images.push(context.getImageData(0, 0, bitmap.width, bitmap.height).data);
  }
  const pixels = images[0].length / 4;
  const bins = new Map();
  for (let i = 0; i < images[0].length; i += 4) {
    const key = ((images[0][i] >> 4) << 8) | ((images[0][i + 1] >> 4) << 4) | (images[0][i + 2] >> 4);
    bins.set(key, (bins.get(key) || 0) + 1);
  }
  const colour_shares = [...bins.values()].map((n) => n / pixels).filter((s) => s >= 0.001)
    .sort((a, b) => b - a);
  const brightness = images.map((d) => {
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    return sum / pixels;
  });
  const changed = [];
  for (let a = 0; a < images.length; a++) {
    for (let b = a + 1; b < images.length; b++) {
      let count = 0;
      const x = images[a], y = images[b];
      for (let i = 0; i < x.length; i += 4) {
        if (Math.abs(x[i] - y[i]) > tolerance || Math.abs(x[i + 1] - y[i + 1]) > tolerance
            || Math.abs(x[i + 2] - y[i + 2]) > tolerance) count++;
      }
      changed.push([a, b, count / pixels]);
    }
  }
  return { colour_shares, brightness, changed };
}"""


class OutOfTime(Exception):
    pass


def _alarm(signum, frame):
    raise OutOfTime()


def render(svg_text: str, tolerance: int, limit: int, times: list[float]) -> tuple[dict, list[str]]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--disable-gpu", "--font-render-hinting=none"])
        try:
            page = browser.new_page(viewport={"width": SIZE, "height": SIZE})
            page.set_default_timeout(limit * 1000)
            page.route("**/*", lambda route: route.abort())
            page.set_content(PAGE)
            if not page.evaluate(INSERT, svg_text):
                return {"error": "the browser could not parse the SVG"}, []
            page.evaluate("document.fonts.ready.then(() => true)")
            frames = []
            for t in times:
                page.evaluate(SEEK, t)
                png = page.screenshot(clip={"x": 0, "y": 0, "width": SIZE, "height": SIZE}, type="png")
                frames.append(base64.b64encode(png).decode("ascii"))
            measures = page.evaluate(MEASURE, [frames, tolerance])
            return {"times": times, **measures, "browser": browser.version}, frames
        finally:
            browser.close()


def main() -> int:
    svg_path, out_dir, tolerance, limit, *times = sys.argv[1:]
    with open(svg_path, encoding="utf-8") as handle:
        svg_text = handle.read()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(limit))
    try:
        result, frames = render(svg_text, int(tolerance), int(limit), [float(t) for t in times])
    except OutOfTime:
        result, frames = {"error": f"rendering took longer than {limit} s"}, []
    except PlaywrightError as error:
        result, frames = {"error": f"the browser failed: {str(error)[:300]}"}, []
    finally:
        signal.alarm(0)
    for i, frame in enumerate(frames):
        with open(os.path.join(out_dir, f"frame{i}.b64"), "w", encoding="ascii") as handle:
            handle.write(frame)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump({**result, "frame_count": len(frames)}, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
