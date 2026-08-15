# -*- coding: utf-8 -*-
"""
camera_utils.py
---------------
Helper for finding out which webcam indices actually work on this machine, so the
operator can pick the external USB camera instead of the laptop's built-in one.

Index 0 is almost always the built-in camera. A USB camera usually shows up at index
1 or higher, so try them one by one.

Note: a camera can only be opened by one process at a time. If the main application
has a preview running, this scan finds nothing - close the preview first.
"""

import glob
import os
import re
import sys

import cv2


def candidate_camera_indices(max_index=10):
    """Return camera indexes worth probing on the current operating system.

    Linux camera indexes are not guaranteed to be contiguous. Reading /dev/video*
    avoids missing an external camera at (for example) index 6 while also avoiding
    slow attempts to open device numbers that do not exist. Other platforms fall
    back to a bounded scan.
    """
    if sys.platform.startswith("linux"):
        indexes = []
        for path in glob.glob("/dev/video*"):
            match = re.fullmatch(r"video(\d+)", os.path.basename(path))
            if match:
                indexes.append(int(match.group(1)))
        if indexes:
            return sorted(set(indexes))
    return list(range(max_index))


def list_available_cameras(max_index=10):
    """Open candidate indexes and return only devices that deliver an image frame."""
    available = []
    for i in candidate_camera_indices(max_index):
        cap = cv2.VideoCapture(i)
        try:
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    available.append(i)
        finally:
            cap.release()
    return available
