"""Code gates on the policy's answer (README, "Scoring").

A rollout passes only if its answer holds exactly one SVG document that is
well-formed, small, self-contained and safe to render. Visible text must
come from a quoted span of the prompt, so the policy cannot write the
prompt into the picture to persuade the judge.

The checks use only the standard library. A DOCTYPE or ENTITY declaration
is refused before parsing, so the parser never expands entities.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

SVG_NS = "http://www.w3.org/2000/svg"
MAX_BYTES = 64 * 1024
MAX_QUOTE_USES = 2

FORBIDDEN_ELEMENTS = frozenset({"script", "foreignObject", "iframe", "image", "feImage"})
ANIMATION_ELEMENTS = frozenset({"animate", "animateTransform", "animateMotion", "animateColor", "set"})
TEXT_ELEMENTS = frozenset({"text"})

CLOCK = r"[+-]?\s*(?:\d+(?:\.\d+)?|\.\d+)\s*(?:h|min|s|ms)?|[+-]?\s*\d+:\d{2}(?::\d{2})?(?:\.\d+)?"
TIME_VALUE = re.compile(rf"^(?:{CLOCK})$")
SYNCBASE = re.compile(rf"^[A-Za-z_][\w.\-]*\.(?:begin|end)(?:\s*(?:{CLOCK}))?$")
URL_REFERENCE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.I | re.S)
QUOTED = re.compile(r'"([^"]+)"')


@dataclass
class GateResult:
    svg: str | None
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.svg is not None and not self.failures


SVG_TAG = re.compile(r"<svg\b[^>]*?(/?)>|</svg\s*>", re.I)


def extract_svg(answer: str) -> str | None:
    """The answer's single top-level SVG document, or None if there is not exactly one."""
    documents, depth, start = [], 0, 0
    for tag in SVG_TAG.finditer(answer):
        opening = tag.group(0)[1] != "/"
        if opening and depth == 0:
            start = tag.start()
        if opening and not tag.group(1):
            depth += 1
        elif not opening:
            depth -= 1
            if depth < 0:
                return None
        if depth == 0:
            documents.append(answer[start : tag.end()])
    if depth != 0 or len(documents) != 1:
        return None
    return documents[0]


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def _external_urls(text: str) -> list[str]:
    return [target for _, target in URL_REFERENCE.findall(text) if not target.strip().startswith("#")]


def _begin_ok(value: str) -> bool:
    for part in value.split(";"):
        part = part.strip()
        if not part or not (TIME_VALUE.match(part) or SYNCBASE.match(part)):
            return False
    return True


def check(answer: str, prompt: str) -> GateResult:
    """Every gate failure for one answer; an empty list means the answer passed."""
    svg = extract_svg(answer)
    if svg is None:
        return GateResult(None, ["not exactly one <svg> document"])
    result = GateResult(svg)
    failures = result.failures
    if len(svg.encode("utf-8")) > MAX_BYTES:
        failures.append(f"larger than {MAX_BYTES} bytes")
        return result
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)", svg, re.I):
        failures.append("has a DOCTYPE or ENTITY declaration")
        return result
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as error:
        failures.append(f"not well-formed XML: {error}")
        return result
    if root.tag != f"{{{SVG_NS}}}svg":
        failures.append("root is not an svg element in the SVG namespace")

    quotes = [_normalise(q) for q in QUOTED.findall(prompt)]
    quote_uses: dict[str, int] = {}
    for element in root.iter():
        name = _local(element.tag)
        if name in FORBIDDEN_ELEMENTS:
            failures.append(f"forbidden element <{name}>")
        for attribute, value in element.attrib.items():
            attribute_name = _local(attribute)
            if attribute_name.lower().startswith("on"):
                failures.append(f"event handler attribute {attribute_name}")
            if attribute_name == "href" and not value.strip().startswith("#"):
                failures.append(f"external reference {value[:60]!r}")
            if _external_urls(value):
                failures.append(f"external url() in {attribute_name}")
        if name == "style" and element.text:
            if "@import" in element.text.lower():
                failures.append("@import in a style element")
            if _external_urls(element.text):
                failures.append("external url() in a style element")
        if name in ANIMATION_ELEMENTS:
            if element.get("attributeName", "").rsplit(":", 1)[-1] == "href":
                failures.append("animation targets an href")
            for timing in ("begin", "end"):
                value = element.get(timing)
                if value is not None and not _begin_ok(value):
                    failures.append(f"animation {timing} {value[:40]!r} is not a time value")
        if name in TEXT_ELEMENTS:
            text = _normalise("".join(element.itertext()))
            if not text:
                continue
            if text not in quotes:
                failures.append(f"text {text[:40]!r} is not a quoted span of the prompt")
                continue
            quote_uses[text] = quote_uses.get(text, 0) + 1
            if quote_uses[text] > MAX_QUOTE_USES:
                failures.append(f"text {text[:40]!r} used more than {MAX_QUOTE_USES} times")
    result.failures = list(dict.fromkeys(failures))
    return result
