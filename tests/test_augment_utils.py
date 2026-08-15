import json
import os
import tempfile
import unittest
from unittest.mock import patch

import augment_utils as au


class CompetitionAugmentDefaultsTests(unittest.TestCase):
    def test_defaults_use_balanced_competition_augmentation(self):
        online = au.DEFAULT_CONFIG["online"]
        self.assertEqual(online["degrees"], 180.0)
        self.assertEqual(online["fliplr"], 0.5)
        self.assertEqual(online["flipud"], 0.5)
        self.assertEqual(online["hsv_h"], 0.2)
        self.assertEqual(online["hsv_s"], 0.5)
        self.assertEqual(online["hsv_v"], 0.3)
        self.assertEqual(online["scale"], 0.3)
        self.assertEqual(online["translate"], 0.1)
        self.assertEqual(online["mosaic"], 0.8)
        self.assertFalse(au.DEFAULT_CONFIG["offline_gray"])
        self.assertFalse(au.DEFAULT_CONFIG["offline_blur"])

    def test_new_classes_are_not_mirrored_and_stale_classes_are_removed(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, "augment.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"flip_allowed": {"old_class": True, "circle": True}}, f)
            with patch.object(au.du, "AUG_CONFIG", config_path):
                with patch.object(au.du, "load_classes", return_value=["circle", "arrow_left"]):
                    cfg = au.load_config()

            self.assertEqual(cfg["flip_allowed"], {"circle": True, "arrow_left": False})

    def test_default_online_params_are_sent_unchanged(self):
        with patch.object(au.du, "load_classes", return_value=[]):
            cfg = au.load_config()
        self.assertEqual(au.online_params(cfg), au.DEFAULT_CONFIG["online"])


if __name__ == "__main__":
    unittest.main()
