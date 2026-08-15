import unittest

from train_tab import class_index_for_shortcut


class LabelShortcutTests(unittest.TestCase):
    def test_number_selects_matching_class(self):
        self.assertEqual(class_index_for_shortcut("1", 4), 0)
        self.assertEqual(class_index_for_shortcut("4", 4), 3)

    def test_out_of_range_or_non_number_is_ignored(self):
        self.assertIsNone(class_index_for_shortcut("5", 4))
        self.assertIsNone(class_index_for_shortcut("0", 4))
        self.assertIsNone(class_index_for_shortcut("x", 4))


if __name__ == "__main__":
    unittest.main()
