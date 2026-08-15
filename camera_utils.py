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

import cv2


def list_available_cameras(max_index=6):
    """Try to open every index from 0 to max_index-1 and return those that deliver a frame."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        try:
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    available.append(i)
        finally:
            cap.release()
    return available
