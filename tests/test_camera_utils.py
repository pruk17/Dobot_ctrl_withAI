import unittest
from unittest.mock import patch

import camera_utils


class CameraCandidateTests(unittest.TestCase):
    def test_linux_device_indexes_do_not_need_to_be_contiguous(self):
        devices = ["/dev/video0", "/dev/video1", "/dev/video6", "/dev/video7", "/dev/video-x"]
        with patch.object(camera_utils.sys, "platform", "linux"):
            with patch.object(camera_utils.glob, "glob", return_value=devices):
                self.assertEqual(camera_utils.candidate_camera_indices(), [0, 1, 6, 7])

    def test_other_platforms_use_bounded_scan(self):
        with patch.object(camera_utils.sys, "platform", "win32"):
            self.assertEqual(camera_utils.candidate_camera_indices(4), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
