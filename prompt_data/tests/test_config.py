import tempfile
import unittest
from pathlib import Path

from prompt_build.__main__ import check_example_lists, check_templates
from prompt_build.config import load_config
from prompt_build.lexicon import load_lexicon

ROOT = Path(__file__).resolve().parent.parent

CATEGORY = """
[[categories]]
slug = "space"
name = "space"
definition = "any scene off Earth or in orbit"
focus = ["moon bases"]
"""


def write_config(directory: Path, body: str) -> Path:
    path = directory / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


class ConfigTest(unittest.TestCase):
    def test_project_config_loads(self):
        config = load_config(ROOT / "config.toml")
        self.assertEqual(config.categories[0].slug, config.held_out_category)
        self.assertEqual(len(config.categories), 9)

    def test_rejects_bad_values(self):
        cases = {
            "unknown held-out category": ('held_out_category = "ocean"', "slot_table = 10000"),
            "offset out of range": ('held_out_category = "space"', "slot_table = 5000"),
            "duplicate offsets": ('held_out_category = "space"', "a = 10000\nb = 10000"),
        }
        for name, (held_out, offsets) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                body = (
                    f'spec_version = "1.4"\nbase_seed = 1\n{held_out}\n'
                    f"[seed_offsets]\n{offsets}\n{CATEGORY}"
                )
                with self.assertRaises(ValueError):
                    load_config(write_config(Path(directory), body))


class ProjectFilesTest(unittest.TestCase):
    def test_example_lists_and_templates_are_consistent(self):
        lexicon = load_lexicon(ROOT / "terms")
        self.assertEqual(check_example_lists(ROOT / "lists", lexicon), [])
        self.assertEqual(check_templates(ROOT / "templates"), [])

    def test_template_check_reports_missing_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("in_scope.txt", "distractor.txt", "prescreen.txt"):
                (Path(directory) / name).write_text("Write $n scenes.", encoding="utf-8")
            self.assertEqual(len(check_templates(Path(directory))), 3)


if __name__ == "__main__":
    unittest.main()
