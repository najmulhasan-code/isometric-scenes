"""Render the random sample's SVGs to PNG tiles for the example figure.

Each SVG from OUT_DIR/random_sample/ has its on* event attributes removed and is placed
in a blank 512x512 page in a headless Chromium browser, offline and with every request
blocked. Its animations are paused at TIME seconds (the same seeking the reward's
renderer uses), and the page is captured.
Clears and writes OUT_DIR/random_sample/tiles/<file>.png.

Needs Playwright with a Chromium build: `pip install playwright` and then
`playwright install chromium`, or set PLAYWRIGHT_CHANNEL=msedge to use an installed Edge.
"""

import json
import os
import shutil
import sys

from playwright.sync_api import sync_playwright

from common import MODELS, OUT_DIR

SIZE, TIME = 512, 1.0

PAGE = f"""<!doctype html><html><head><style>
html, body {{ margin: 0; width: {SIZE}px; height: {SIZE}px; overflow: hidden; background: #fff; }}
#stage > svg {{ display: block; width: {SIZE}px; height: {SIZE}px; }}
</style></head><body><div id="stage"></div></body></html>"""

INSERT = """(text) => {
  const stage = document.getElementById("stage");
  stage.replaceChildren();
  const doc = new DOMParser().parseFromString(text, "image/svg+xml");
  if (doc.querySelector("parsererror")) return false;
  for (const el of doc.querySelectorAll("*"))
    for (const a of [...el.attributes]) if (a.name.toLowerCase().startsWith("on")) el.removeAttribute(a.name);
  stage.appendChild(document.importNode(doc.documentElement, true));
  return true;
}"""

SEEK = """(t) => {
  const svg = document.querySelector("#stage > svg");
  if (svg.pauseAnimations) { svg.pauseAnimations(); svg.setCurrentTime(t); }
  for (const a of document.getAnimations()) { a.pause(); a.currentTime = t * 1000; }
  return new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
}"""


def main() -> int:
    sample = OUT_DIR / "random_sample"
    index = json.loads((sample / "index.json").read_text(encoding="utf-8"))
    tiles = sample / "tiles"
    shutil.rmtree(tiles, ignore_errors=True)
    tiles.mkdir()
    files = [item[m] for item in index["items"] for m in MODELS if item[m]]
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("PLAYWRIGHT_CHANNEL") or None, headless=True)
        context = browser.new_context(offline=True, viewport={"width": SIZE, "height": SIZE})
        page = context.new_page()
        page.route("**/*", lambda route: route.abort())
        page.set_content(PAGE)
        for name in files:
            out = tiles / name.replace(".svg", ".png")
            if not page.evaluate(INSERT, (sample / name).read_text(encoding="utf-8")):
                continue  # does not parse: the figure shows it as broken
            page.evaluate(SEEK, TIME)
            page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": SIZE, "height": SIZE})
        browser.close()
    print(f"rendered {len(list(tiles.glob('*.png')))} of {len(files)} SVGs to random_sample/tiles/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
