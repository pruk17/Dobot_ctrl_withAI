# -*- coding: utf-8 -*-
"""
camera_utils.py
-----------------
ฟังก์ชันช่วยตรวจสอบว่ามีกล้อง (webcam) ต่ออยู่กี่ตัว และ index เท่าไรบ้าง
เพื่อให้ผู้ใช้เลือกกล้องภายนอก (USB Webcam) แทนกล้อง Built-in ของโน้ตบุ๊กได้

หมายเหตุ: โดยทั่วไป index 0 มักจะเป็นกล้อง Built-in ของโน้ตบุ๊กเสมอ
ถ้าต่อกล้อง USB เพิ่ม มักจะไปโผล่ที่ index 1, 2, ... ให้ลองเลือกดูทีละตัว
"""

import cv2


def list_available_cameras(max_index=6):
    """ลองเปิดกล้องตั้งแต่ index 0 ถึง max_index-1 คืนค่าเป็น list ของ index ที่เปิดได้จริง"""
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
