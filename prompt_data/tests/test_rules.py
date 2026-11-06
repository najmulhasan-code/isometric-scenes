import unittest
from pathlib import Path

from prompt_build.lexicon import load_lexicon
from prompt_build.rules import (
    check_distractor,
    format_failures,
    in_scope_failures,
    inserted_spans,
)

LEXICON = load_lexicon(Path(__file__).resolve().parent.parent / "terms")
VALID = "a small kitchen with a copper kettle on the stove"


class FormatRulesTest(unittest.TestCase):
    def test_valid_prompt_passes(self):
        self.assertEqual(in_scope_failures(VALID, LEXICON), [])

    def test_opening(self):
        for prompt in (
            "A small kitchen with a kettle",
            "the small kitchen with a kettle",
            "a small kitchen with a kettle.",
            "a small kitchen with a kettle ",
            "a small kitchen\nwith a kettle",
            "a small kitchen\rwith a kettle",
            "a small kitchen\u2028with a kettle",
            "",
        ):
            with self.subTest(prompt=prompt):
                self.assertIn("R1", format_failures(prompt, LEXICON))
        self.assertNotIn("R1", format_failures("an attic with a round window", LEXICON))

    def test_length(self):
        self.assertIn("R2", format_failures("a red kettle", LEXICON))
        self.assertNotIn("R2", format_failures("a red kettle steaming", LEXICON))
        self.assertNotIn("R2", format_failures("a " + " ".join(["lamp"] * 24), LEXICON))
        self.assertIn("R2", format_failures("a " + " ".join(["lamp"] * 25), LEXICON))

    def test_style_terms(self):
        self.assertIn("R5", format_failures("an isometric bakery with bread", LEXICON))
        self.assertNotIn("R5", format_failures("a desk with a 3d printer and a lamp", LEXICON))

    def test_capitals_outside_quotes(self):
        for prompt in (
            "a cafe in Paris with two tables",
            "a cafe (Paris) with two tables",
            "a desk with an iPhone and a lamp",
            "a desk with a TV and a lamp",
        ):
            with self.subTest(prompt=prompt):
                self.assertIn("R6", format_failures(prompt, LEXICON))
        self.assertNotIn("R6", format_failures('a shop door with an "OPEN" sign', LEXICON))

    def test_curly_quotes_are_checked_as_straight(self):
        self.assertEqual(format_failures("a shop door with an \u201cOPEN\u201d sign", LEXICON), [])
        self.assertIn(
            "R7", format_failures("a shop door with a \u201cPARIS CAFE MENU\u201d sign", LEXICON)
        )

    def test_quoted_text(self):
        self.assertNotIn("R7", format_failures('a shop door with a "NO ENTRY" sign', LEXICON))
        self.assertIn("R7", format_failures('a shop door with a "NO ENTRY HERE" sign', LEXICON))
        self.assertIn("R7", format_failures('a shop door with a "OPEN sign', LEXICON))
        self.assertIn("R7", format_failures('a "BAR" sign and a "CAFE" sign on a wall', LEXICON))

    def test_overused_words(self):
        self.assertIn("R9", format_failures("a quaint cottage kitchen with a kettle", LEXICON))

    def test_out_of_scope_terms(self):
        self.assertIn("R8", in_scope_failures("a busy cafe counter with two cups", LEXICON))
        self.assertIn("R8", in_scope_failures("a kitchen with a cat by the stove", LEXICON))
        self.assertNotIn("R8", in_scope_failures("a stall at a farmers market", LEXICON))


class InsertedSpansTest(unittest.TestCase):
    def test_middle_insertion(self):
        base = "a bakery counter with bread".split()
        prompt = "a bakery counter with a cat asleep and bread".split()
        self.assertEqual(inserted_spans(prompt, base), [(4, 8)])

    def test_end_insertion(self):
        base = "a quiet pier at night".split()
        prompt = "a quiet pier at night with an angler".split()
        self.assertEqual(inserted_spans(prompt, base), [(5, 8)])

    def test_changed_word_has_no_span(self):
        base = "a quiet pier at night".split()
        prompt = "a calm pier at night with an angler".split()
        self.assertEqual(inserted_spans(prompt, base), [])

    def test_repeated_word_gives_several_spans(self):
        base = "a lamp and a desk".split()
        prompt = "a lamp and a cat and a desk".split()
        self.assertEqual(inserted_spans(prompt, base), [(2, 5), (3, 6), (4, 7)])

    def test_not_longer(self):
        self.assertEqual(inserted_spans(["a", "b"], ["a", "b"]), [])


class DistractorTest(unittest.TestCase):
    BASE = "a bakery counter with bread on wooden shelves"

    def test_valid_pair(self):
        prompt = "a bakery counter with a cat asleep by the till and bread on wooden shelves"
        result = check_distractor(prompt, self.BASE, "animal", LEXICON)
        self.assertEqual(result.failures, ())
        self.assertEqual(result.excluded_element, "cat")

    def test_wrong_kind(self):
        prompt = "a bakery counter with a cat asleep and bread on wooden shelves"
        result = check_distractor(prompt, self.BASE, "human", LEXICON)
        self.assertIn("D2", result.failures)

    def test_two_elements(self):
        prompt = "a bakery counter with a cat and a dog and bread on wooden shelves"
        result = check_distractor(prompt, self.BASE, "animal", LEXICON)
        self.assertIn("D2", result.failures)

    def test_non_ascii_words_keep_token_offsets(self):
        base = "a small caf\u00e9 with two tables"
        prompt = "a small caf\u00e9 with a dog asleep and two tables"
        result = check_distractor(prompt, base, "animal", LEXICON)
        self.assertEqual(result.failures, ())
        self.assertEqual(result.excluded_element, "dog")

    def test_allowed_phrase_across_span_edge(self):
        result = check_distractor("a hot dog and plate on a table", "a hot plate on a table", "animal", LEXICON)
        self.assertIn("D2", result.failures)

    def test_added_punctuation_breaks_the_span(self):
        prompt = "a bakery counter, with a cat asleep, and bread on wooden shelves"
        result = check_distractor(prompt, self.BASE, "animal", LEXICON)
        self.assertIn("D1", result.failures)

    def test_base_failures_are_prefixed(self):
        base = "a busy bakery counter with bread"
        prompt = "a busy bakery counter with a cat and bread"
        result = check_distractor(prompt, base, "animal", LEXICON)
        self.assertIn("base_R8", result.failures)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            check_distractor(self.BASE, self.BASE, "animals", LEXICON)


if __name__ == "__main__":
    unittest.main()
