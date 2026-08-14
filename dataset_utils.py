# -*- coding: utf-8 -*-
"""
dataset_utils.py
-----------------
ฟังก์ชันช่วยเหลือสำหรับจัดการชุดข้อมูล (dataset) รูปแบบ YOLO
- โครงสร้างโฟลเดอร์:
    dataset/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt   (YOLO format: class_id cx cy w h  -- normalized 0-1)
        labels/val/*.txt
        classes.txt          (รายชื่อคลาส บรรทัดละ 1 คลาส เรียงตาม index)
        data.yaml             (ไฟล์ config สำหรับ ultralytics YOLO)
        model/best.pt          (โมเดลที่ฝึกสอนเสร็จแล้ว ถูกคัดลอกมาไว้ที่นี่เพื่อให้ใช้งานง่าย)
"""

import os
import random
import shutil
import yaml

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
IMAGES_TRAIN = os.path.join(BASE_DIR, "images", "train")
IMAGES_VAL = os.path.join(BASE_DIR, "images", "val")
LABELS_TRAIN = os.path.join(BASE_DIR, "labels", "train")
LABELS_VAL = os.path.join(BASE_DIR, "labels", "val")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.txt")
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")


def ensure_dirs():
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "w", encoding="utf-8") as f:
            f.write("")


def load_classes():
    ensure_dirs()
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def save_classes(classes):
    ensure_dirs()
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")


def add_class(name):
    classes = load_classes()
    if name not in classes:
        classes.append(name)
        save_classes(classes)
    return classes


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


def label_path_for_image(image_path):
    """แปลง path ของไฟล์รูปภาพ -> path ของไฟล์ label (.txt) ที่ตำแหน่งคู่กัน (train หรือ val)"""
    folder = os.path.dirname(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    if folder == IMAGES_TRAIN:
        return os.path.join(LABELS_TRAIN, base + ".txt")
    elif folder == IMAGES_VAL:
        return os.path.join(LABELS_VAL, base + ".txt")
    else:
        # fallback: เก็บไว้ข้าง ๆ ไฟล์รูป
        return os.path.join(folder, base + ".txt")


def unlabeled_images(folder=None):
    """คืนรายการรูปภาพที่ยังไม่มีไฟล์ label หรือไฟล์ label ว่างเปล่า"""
    folders = [folder] if folder else [IMAGES_TRAIN, IMAGES_VAL]
    result = []
    for fd in folders:
        for img in list_images(fd):
            img_path = os.path.join(fd, img)
            lbl_path = label_path_for_image(img_path)
            if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
                result.append(img_path)
    return result


def all_images():
    result = []
    for fd in [IMAGES_TRAIN, IMAGES_VAL]:
        for img in list_images(fd):
            result.append(os.path.join(fd, img))
    return result


def rebalance_train_val(val_ratio=0.2, seed=42):
    """
    รวมรูปภาพทั้งหมด (train+val) กลับมาสุ่มแบ่งใหม่ตามสัดส่วนที่กำหนด
    ใช้ก่อนเริ่ม Train เพื่อให้แน่ใจว่ามีชุด validation
    """
    ensure_dirs()
    imgs = all_images()
    if len(imgs) == 0:
        return 0, 0
    rnd = random.Random(seed)
    rnd.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio)) if len(imgs) >= 5 else 0
    val_set = set(imgs[:n_val])

    for img_path in imgs:
        lbl_path = label_path_for_image(img_path)
        base = os.path.basename(img_path)
        lbl_base = os.path.basename(lbl_path) if os.path.exists(lbl_path) else None
        target_img_dir = IMAGES_VAL if img_path in val_set else IMAGES_TRAIN
        target_lbl_dir = LABELS_VAL if img_path in val_set else LABELS_TRAIN

        new_img_path = os.path.join(target_img_dir, base)
        if os.path.abspath(new_img_path) != os.path.abspath(img_path):
            shutil.move(img_path, new_img_path)

        if lbl_base and os.path.exists(lbl_path):
            new_lbl_path = os.path.join(target_lbl_dir, lbl_base)
            if os.path.abspath(new_lbl_path) != os.path.abspath(lbl_path):
                shutil.move(lbl_path, new_lbl_path)

    n_train = len(imgs) - len(val_set)
    return n_train, len(val_set)


def write_data_yaml():
    ensure_dirs()
    classes = load_classes()
    data = {
        "path": BASE_DIR,
        "train": "images/train",
        "val": "images/val" if len(list_images(IMAGES_VAL)) > 0 else "images/train",
        "names": {i: c for i, c in enumerate(classes)},
    }
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return DATA_YAML
