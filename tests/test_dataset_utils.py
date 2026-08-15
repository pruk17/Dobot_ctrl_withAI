import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

import dataset_utils as du


@contextmanager
def temporary_dataset():
    with tempfile.TemporaryDirectory() as root:
        base = os.path.join(root, "dataset")
        values = {
            "BASE_DIR": base,
            "IMAGES_TRAIN": os.path.join(base, "images", "train"),
            "IMAGES_VAL": os.path.join(base, "images", "val"),
            "LABELS_TRAIN": os.path.join(base, "labels", "train"),
            "LABELS_VAL": os.path.join(base, "labels", "val"),
            "CLASSES_FILE": os.path.join(base, "classes.txt"),
            "DATA_YAML": os.path.join(base, "data.yaml"),
            "MODEL_DIR": os.path.join(base, "model"),
            "MODEL_PATH": os.path.join(base, "model", "best.pt"),
            "AUG_CONFIG": os.path.join(base, "augment.json"),
        }
        with patch.multiple(du, **values):
            du.ensure_dirs()
            yield root


def add_labeled_image(folder, name, class_id=0):
    path = os.path.join(folder, name)
    du.save_image(path, np.zeros((20, 20, 3), dtype=np.uint8))
    with open(du.label_path_for_image(path), "w", encoding="utf-8") as f:
        f.write(f"{class_id} 0.500000 0.500000 0.500000 0.500000\n")


class DatasetValidationTests(unittest.TestCase):
    def test_valid_dataset_reports_counts(self):
        with temporary_dataset():
            du.save_classes(["circle"])
            for index in range(4):
                add_labeled_image(du.IMAGES_TRAIN, f"train_{index}.jpg")
            add_labeled_image(du.IMAGES_VAL, "val.jpg")

            errors, warnings, stats = du.validate_dataset()

            self.assertEqual(errors, [])
            self.assertEqual(len(warnings), 1)
            self.assertEqual(stats, {"classes": 1, "train": 4, "val": 1, "boxes": 5})

    def test_unlabeled_image_blocks_training(self):
        with temporary_dataset():
            du.save_classes(["circle"])
            for index in range(4):
                add_labeled_image(du.IMAGES_TRAIN, f"train_{index}.jpg")
            path = os.path.join(du.IMAGES_VAL, "missing.jpg")
            du.save_image(path, np.zeros((20, 20, 3), dtype=np.uint8))

            errors, _, _ = du.validate_dataset()

            self.assertTrue(any("ยังไม่มี Label" in error for error in errors))

    def test_box_outside_image_blocks_training(self):
        with temporary_dataset():
            du.save_classes(["circle"])
            for index in range(4):
                add_labeled_image(du.IMAGES_TRAIN, f"train_{index}.jpg")
            add_labeled_image(du.IMAGES_VAL, "outside.jpg")
            label = os.path.join(du.LABELS_VAL, "outside.txt")
            with open(label, "w", encoding="utf-8") as f:
                f.write("0 0.950000 0.500000 0.200000 0.200000\n")

            errors, _, _ = du.validate_dataset()

            self.assertTrue(any("พิกัดกรอบไม่ถูกต้อง" in error for error in errors))

    def test_box_on_edge_allows_six_decimal_rounding(self):
        with temporary_dataset():
            du.save_classes(["circle"])
            for index in range(4):
                add_labeled_image(du.IMAGES_TRAIN, f"train_{index}.jpg")
            add_labeled_image(du.IMAGES_VAL, "edge.jpg")
            label = os.path.join(du.LABELS_VAL, "edge.txt")
            with open(label, "w", encoding="utf-8") as f:
                f.write("0 0.918941 0.572231 0.162119 0.271268\n")

            errors, _, _ = du.validate_dataset()

            self.assertFalse(any("พิกัดกรอบไม่ถูกต้อง" in error for error in errors))

    def test_import_remaps_class_ids_and_supports_nested_splits(self):
        with temporary_dataset() as root:
            du.save_classes(["square"])
            incoming = os.path.join(root, "incoming")
            os.makedirs(os.path.join(incoming, "images", "train"))
            os.makedirs(os.path.join(incoming, "labels", "train"))
            with open(os.path.join(incoming, "classes.txt"), "w", encoding="utf-8") as f:
                f.write("circle\nsquare\n")
            source_image = os.path.join(incoming, "images", "train", "sample.jpg")
            du.save_image(source_image, np.zeros((20, 20, 3), dtype=np.uint8))
            with open(os.path.join(incoming, "labels", "train", "sample.txt"), "w", encoding="utf-8") as f:
                f.write("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.3 0.3\n")

            count, classes = du.import_labeled_yolo_dataset(incoming)

            self.assertEqual(count, 1)
            self.assertEqual(classes, ["square", "circle"])
            imported = du.list_images(du.IMAGES_TRAIN)
            label = du.label_path_for_image(os.path.join(du.IMAGES_TRAIN, imported[0]))
            with open(label, "r", encoding="utf-8") as f:
                ids = [int(line.split()[0]) for line in f]
            self.assertEqual(ids, [1, 0])

    def test_reset_moves_old_dataset_to_backup(self):
        with temporary_dataset():
            du.save_classes(["circle"])
            add_labeled_image(du.IMAGES_TRAIN, "old.jpg")

            backup = du.backup_and_reset_dataset()

            self.assertTrue(os.path.exists(os.path.join(backup, "images", "train", "old.jpg")))
            self.assertEqual(du.load_classes(), [])
            self.assertEqual(du.all_images(), [])

    def test_delete_image_removes_label_and_only_its_augmentations(self):
        with temporary_dataset():
            du.save_classes(["circle"])
            add_labeled_image(du.IMAGES_TRAIN, "bad.jpg")
            add_labeled_image(du.IMAGES_TRAIN, "keep.jpg")
            bad_aug = os.path.join(du.IMAGES_TRAIN, "bad__aug_blur.jpg")
            du.save_image(bad_aug, np.zeros((20, 20, 3), dtype=np.uint8))
            with open(os.path.join(du.LABELS_TRAIN, "bad__aug_blur.txt"), "w", encoding="utf-8") as f:
                f.write("0 0.5 0.5 0.5 0.5\n")

            removed = du.delete_dataset_image(os.path.join(du.IMAGES_TRAIN, "bad.jpg"))

            self.assertEqual(removed, 4)
            self.assertFalse(os.path.exists(os.path.join(du.IMAGES_TRAIN, "bad.jpg")))
            self.assertFalse(os.path.exists(bad_aug))
            self.assertTrue(os.path.exists(os.path.join(du.IMAGES_TRAIN, "keep.jpg")))


class ClassNameTests(unittest.TestCase):
    def test_protocol_safe_names(self):
        self.assertTrue(du.is_valid_class_name("arrow_left2"))
        self.assertFalse(du.is_valid_class_name("arrow,left"))
        self.assertFalse(du.is_valid_class_name("วงกลม"))


if __name__ == "__main__":
    unittest.main()
