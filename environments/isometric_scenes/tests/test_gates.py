import unittest
from pathlib import Path

from svg_scoring.gates import MAX_BYTES, check, extract_svg

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HEAD = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'


def svg(body: str) -> str:
    return f"{HEAD}{body}</svg>"


class ExtractTest(unittest.TestCase):
    def test_one_document_is_taken_from_surrounding_text(self):
        self.assertEqual(extract_svg(f"Here it is:\n{svg('')}\nDone."), svg(""))

    def test_two_documents_or_none_are_refused(self):
        self.assertIsNone(extract_svg(svg("") + svg("")))
        self.assertIsNone(extract_svg("no drawing here"))

    def test_a_nested_svg_is_part_of_one_document(self):
        nested = svg('<svg x="1"><rect width="1" height="1"/></svg>')
        self.assertEqual(extract_svg(nested), nested)


class GateTest(unittest.TestCase):
    def failures(self, body: str, prompt: str = "a room") -> list[str]:
        return check(svg(body), prompt).failures

    def test_fixtures_pass(self):
        for path in FIXTURES.glob("*.svg"):
            with self.subTest(path.name):
                self.assertTrue(check(path.read_text(encoding="utf-8"), "a room").passed)

    def test_root_must_be_in_the_svg_namespace(self):
        self.assertTrue(check('<svg viewBox="0 0 1 1"></svg>', "a room").failures)

    def test_malformed_xml_fails(self):
        self.assertTrue(check(f"{HEAD}<rect></svg>", "a room").failures)

    def test_doctype_and_entities_are_refused_before_parsing(self):
        answer = svg('<!DOCTYPE svg [<!ENTITY a "aaaa">]><text>&a;</text>')
        self.assertIn("has a DOCTYPE or ENTITY declaration", check(answer, "a room").failures)

    def test_size_limit(self):
        big = svg("<desc>" + "x" * MAX_BYTES + "</desc>")
        self.assertTrue(check(big, "a room").failures)

    def test_forbidden_elements(self):
        for body in ("<script>1</script>", "<image href='#a'/>", "<foreignObject/>",
                     "<filter><feImage href='#a'/></filter>"):
            with self.subTest(body):
                self.assertTrue(self.failures(body))

    def test_event_handlers(self):
        self.assertTrue(self.failures('<rect onclick="x()"/>'))

    def test_external_references(self):
        for body in ('<use href="http://example.com/a.svg#b"/>',
                     '<use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="data:image/png;base64,AA"/>',
                     '<rect style="fill: url(http://example.com/p.png)"/>',
                     "<style>rect { fill: url('data:image/png;base64,AA') }</style>",
                     "<style>@import 'x.css';</style>"):
            with self.subTest(body):
                self.assertTrue(self.failures(body))

    def test_internal_references_are_allowed(self):
        body = '<defs><linearGradient id="g"/></defs><rect fill="url(#g)"/><use href="#g"/>'
        self.assertEqual(self.failures(body), [])

    def test_animation_must_start_on_a_time_value(self):
        self.assertEqual(self.failures('<rect><animate attributeName="x" begin="0.5s; a.end+1s" dur="1s"/></rect>'), [])
        self.assertTrue(self.failures('<rect><animate attributeName="x" begin="click" dur="1s"/></rect>'))
        self.assertTrue(self.failures('<rect><set attributeName="x" begin="indefinite" to="1"/></rect>'))

    def test_animation_may_not_target_href(self):
        self.assertTrue(self.failures('<use href="#a"><set attributeName="href" to="#b"/></use>'))
        self.assertTrue(self.failures('<use href="#a"><set attributeName="xlink:href" to="#b"/></use>'))

    def test_animation_must_end_on_a_time_value(self):
        self.assertTrue(self.failures('<rect><animate attributeName="x" end="click" dur="1s"/></rect>'))

    def test_a_doctype_outside_the_svg_is_ignored(self):
        answer = '<?xml version="1.0"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "x.dtd">\n' + svg("")
        self.assertEqual(check(answer, "a room").failures, [])

    def test_text_must_be_a_quoted_span_of_the_prompt(self):
        prompt = 'a shop with a sign reading "open"'
        self.assertEqual(check(svg("<text>OPEN</text>"), prompt).failures, [])
        self.assertTrue(check(svg("<text>closed</text>"), prompt).failures)
        self.assertTrue(check(svg("<text>isometric shop</text>"), "a shop").failures)

    def test_a_quoted_span_may_appear_at_most_twice(self):
        prompt = 'a sign reading "open"'
        self.assertEqual(check(svg("<text>open</text>" * 2), prompt).failures, [])
        self.assertTrue(check(svg("<text>open</text>" * 3), prompt).failures)


if __name__ == "__main__":
    unittest.main()
