import unittest

from comfyfleet.naming import (
    MAX_NAME_LENGTH,
    instance_name_from_workflow,
    resolve_instance_name,
    sanitize_stem,
)


class NamingTests(unittest.TestCase):
    def test_stem_is_lowercased(self):
        self.assertEqual(instance_name_from_workflow("My_Flow.json"), "my_flow")

    def test_illegal_characters_become_underscores(self):
        self.assertEqual(instance_name_from_workflow("My Flow.json"), "my_flow")
        self.assertEqual(sanitize_stem("Café.v2"), "caf_v2")

    def test_long_name_is_stable_and_capped(self):
        stem = "A" * 80
        first = sanitize_stem(stem)
        second = sanitize_stem(stem)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), MAX_NAME_LENGTH)
        self.assertRegex(first, r"^[a-z0-9][a-z0-9_-]*$")
        self.assertIn("-", first)

    def test_sixty_three_chars_do_not_gain_a_hash(self):
        stem = "b" * 63
        self.assertEqual(sanitize_stem(stem), stem)

    def test_punctuation_only_gets_a_fallback_name(self):
        self.assertEqual(sanitize_stem("---"), "wf")

    def test_blank_request_uses_the_workflow_filename(self):
        self.assertEqual(resolve_instance_name("Portrait.json", None), "portrait")
        self.assertEqual(resolve_instance_name("My Flow.json", "  "), "my_flow")

    def test_typed_request_wins_and_is_sanitized(self):
        self.assertEqual(resolve_instance_name("Portrait.json", "studio"), "studio")
        self.assertEqual(resolve_instance_name("Portrait.json", "My Studio"), "my_studio")


if __name__ == "__main__":
    unittest.main()
