import unittest
from pathlib import Path

from prompt_build.lexicon import PhraseMask, Term, TermMatcher, load_lexicon
from prompt_build.text import matching_tokens, normalise_punctuation

TERMS_DIR = Path(__file__).resolve().parent.parent / "terms"
LEXICON = load_lexicon(TERMS_DIR)


def found(text: str) -> list[tuple[str, str]]:
    return [(m.term.lemma, m.term.kind) for m in LEXICON.matcher.find(text)]


class TextTest(unittest.TestCase):
    def test_punctuation_becomes_separator(self):
        self.assertEqual(matching_tokens("A Jack-o'-Lantern!"), ["a", "jack", "o", "lantern"])

    def test_nfkc_folds_compatibility_characters(self):
        self.assertEqual(matching_tokens("\uff43\uff41\uff54"), ["cat"])

    def test_curly_quotes_are_straightened(self):
        self.assertEqual(normalise_punctuation("\u201cOPEN\u201d \u2018sign\u2019"), "\"OPEN\" 'sign'")
        self.assertEqual(normalise_punctuation("\u201eOPEN\u201c \u00abBAR\u00bb"), '"OPEN" "BAR"')


    def test_unicode_hyphens_become_ascii(self):
        self.assertEqual(normalise_punctuation("a non" + chr(0x2011) + "stick pan"), "a non-stick pan")


class TermMatcherTest(unittest.TestCase):
    def test_scenery_silhouettes_are_allowed(self):
        self.assertEqual(found("pine silhouettes against a dusk sky"), [])
        self.assertEqual(found("a silhouette in a lit window"), [("silhouette", "human")])

    def test_whole_tokens_only(self):
        self.assertEqual(found("a manhole cover next to a catalog"), [])

    def test_plural_maps_to_lemma(self):
        self.assertEqual(found("two cats on a wall"), [("cat", "animal")])

    def test_multi_word_term_counts_once(self):
        self.assertEqual(found("a teddy bear on a bed"), [("teddy bear", "face_figure")])
        self.assertEqual(found("a gingerbread man on a plate"), [("gingerbread man", "face_figure")])

    def test_allowed_phrases_are_masked(self):
        for text in (
            "a computer mouse beside a hot dog",
            "a chef's knife on a board",
            "a record player under fairy lights",
            "a clock with its hour hand at three",
            "a hand saw and a skeleton key on a bench",
            "a hand towel beside a soap dish",
            "a desk with a wax seal and a seal stamp",
            "a garage lit by a cage light",
            "a bowl of prawn crackers",
            "a welding mask on a hook",
            "a fairy ring of mushrooms",
            "a plate of dragon fruit",
            "a monster truck in a lot",
            "a snake plant on a shelf",
            "a cassette player and a tape player",
            "a fire guard in front of a stove",
            "a nest of tables by a sofa",
            "a ping pong bat on a table",
            "a turkey sandwich and a lobster roll",
            "a crater field with alien plants",
        ):
            with self.subTest(text=text):
                self.assertEqual(found(text), [])

    def test_plural_aliens_are_figures(self):
        self.assertEqual(found("a moon base with two aliens"), [("alien", "face_figure")])

    def test_allowed_phrase_does_not_hide_other_terms(self):
        self.assertEqual(found("a hot dog stand with a dog"), [("dog", "animal")])

    def test_animal_shaped_food_is_not_allowed(self):
        self.assertEqual(found("a bowl of gummy bears"), [("bear", "animal")])

    def test_apostrophe_term(self):
        self.assertEqual(found("a jack-o'-lantern on a step"), [("jack o lantern", "face_figure")])

    def test_conflicting_entries_raise(self):
        terms = [Term(("cat",), "animal", "cat"), Term(("cat",), "human", "cat")]
        with self.assertRaises(ValueError):
            TermMatcher(terms, PhraseMask([]))


class LexiconTest(unittest.TestCase):
    def test_mentions_motion(self):
        self.assertTrue(LEXICON.mentions_motion("a lantern swaying in the wind"))
        self.assertFalse(LEXICON.mentions_motion("a lantern on a hook"))
        self.assertTrue(LEXICON.mentions_motion("a desk fan spinning beside a lamp"))

    def test_sounds_are_not_motion(self):
        self.assertFalse(LEXICON.mentions_motion("a fridge humming in a dark kitchen"))
        self.assertFalse(LEXICON.mentions_motion("a wind chime tinkling on a porch"))

    def test_motion_allow_list(self):
        self.assertFalse(LEXICON.mentions_motion("a rocking chair on rolling hills"))
        self.assertFalse(LEXICON.mentions_motion("a floating island with running shoes"))
        self.assertTrue(LEXICON.mentions_motion("a rocking chair rocking on a porch"))

    def test_spec_examples_ignore_case_and_punctuation(self):
        self.assertTrue(LEXICON.is_spec_example("A rooftop with a water tower at sunset."))


if __name__ == "__main__":
    unittest.main()
