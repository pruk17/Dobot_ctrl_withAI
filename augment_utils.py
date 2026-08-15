# -*- coding: utf-8 -*-
"""
augment_utils.py
----------------
Data augmentation settings, and the offline half of the augmentation pipeline.

There are two kinds of augmentation in this project.

ONLINE, done by ultralytics during training. Every epoch the data loader applies
random rotation, scaling, colour shifts and flips in memory. Nothing is written to
disk and the randomness differs on every epoch, so the model never sees exactly the
same picture twice. This is the stronger option and is used for everything that can
be applied to all images equally.

OFFLINE, done here. A transformed copy of an image is written next to the original
with the marker "__aug" in its name, together with a matching label file. This is
what Roboflow does. It is only used for the two cases online augmentation cannot
cover:

  1. Horizontal flip that must apply to some classes but not others. YOLO's fliplr
     works per image, and one image may contain several classes, so per class control
     is impossible there. Here we can look at the label file first and only flip
     images whose classes all allow it.
  2. Grayscale and blur, which ultralytics only offers through the optional
     albumentations package, and even then hard-coded at a 1 percent probability.
     Writing them ourselves avoids the extra dependency, which matters because no
     internet is available on competition day.

Generated files are deleted and rebuilt before every training run, so they never pile
up, and they are hidden from the labelling UI by dataset_utils.list_images().
"""

import copy
import json
import os

import cv2

import dataset_utils as du

# Balanced defaults for a small competition-day dataset.  They add substantially
# more variation than the conservative preset, without pushing colour/scale as hard
# as the original aggressive preset.  Horizontal mirroring is still enabled only
# for classes explicitly marked as safe, because arrows and text can change meaning.
DEFAULT_CONFIG = {
    "flip_allowed": {},        # class name -> bool. Missing/new classes default to False.
    "offline_gray": False,     # also train on a grayscale copy of every image
    "offline_blur": False,     # also train on a blurred copy of every image
    "blur_strength": 3,        # Gaussian kernel size, odd, 3 to 7
    "online": {
        "degrees": 180.0,      # random rotation, plus/minus this many degrees
        "fliplr": 0.5,         # applied only when every class in an image allows mirroring
        "flipud": 0.5,         # useful for symmetric shape classes and varied placement
        "hsv_h": 0.2,          # moderate colour variation
        "hsv_s": 0.5,          # moderate saturation variation
        "hsv_v": 0.3,          # room-light variation
        "scale": 0.3,          # moderate camera-distance variation
        "translate": 0.1,      # random shift
        "mosaic": 0.8,         # strong enough to help a small dataset, not on every sample
    },
}

# Label, unit hint and explanation for every online parameter, used by the dialog.
ONLINE_FIELDS = [
    ("degrees", "หมุนภาพสุ่ม", "องศา", 0.0, 180.0, 5.0,
     "แผ่นสัญลักษณ์ถูกวางมุมไหนก็ได้บนถาด ค่า 180 ครอบคลุมทุกมุม"),
    ("fliplr", "พลิกซ้ายขวา", "โอกาส 0-1", 0.0, 1.0, 0.1,
     "ค่ากลาง 0.5 ใช้เฉพาะคลาสที่ติ๊กว่าอนุญาต ห้ามติ๊กกับลูกศร/ตัวอักษรที่กลับด้านแล้วเปลี่ยนความหมาย"),
    ("flipud", "พลิกบนล่าง", "โอกาส 0-1", 0.0, 1.0, 0.1,
     "ค่ากลาง 0.5 เพิ่มความหลากหลาย เหมาะกับรูปทรงสมมาตร; ลดเป็น 0 หากการกลับด้านเปลี่ยนความหมาย"),
    ("hsv_h", "เปลี่ยนเฉดสี", "0-1", 0.0, 1.0, 0.05,
     "ค่ากลาง 0.2 เพิ่มความทนต่อสีและสมดุลแสง; ลดลงหากสีใช้แยกคลาส"),
    ("hsv_s", "เปลี่ยนความอิ่มสี", "0-1", 0.0, 1.0, 0.05,
     "ค่ากลาง 0.5 จำลองความต่างของกล้องและสภาพแสง"),
    ("hsv_v", "เปลี่ยนความสว่าง", "0-1", 0.0, 1.0, 0.05,
     "รับมือกับแสงในห้องแข่งที่ต่างจากห้องซ้อม"),
    ("scale", "ย่อขยายภาพ", "0-1", 0.0, 1.0, 0.1,
     "ค่ากลาง 0.3 รับมือระยะกล้องและขนาดวัตถุที่เปลี่ยนไป"),
    ("translate", "เลื่อนตำแหน่ง", "0-1", 0.0, 1.0, 0.05,
     "รับมือกับชิ้นงานที่วางไม่ตรงกลางเฟรม"),
    ("mosaic", "ต่อ 4 ภาพเป็นภาพเดียว", "โอกาส 0-1", 0.0, 1.0, 0.1,
     "ค่า 0.8 ช่วยชุดข้อมูลหน้างานที่มีน้อย แต่ยังเหลือภาพปกติให้โมเดลเรียนรู้"),
]


# ------------------------------------------------------------------- config ---
def load_config():
    """Read the saved settings, filling in any key that is missing."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(du.AUG_CONFIG):
        try:
            with open(du.AUG_CONFIG, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except (OSError, ValueError):
            saved = {}
        for key, value in saved.items():
            if key == "online" and isinstance(value, dict):
                cfg["online"].update(value)
            elif key in cfg:
                cfg[key] = value
    # Keep only current classes. Unknown/new classes default to "do not mirror";
    # the operator can explicitly allow safe symmetric classes in the dialog.
    saved_flip = cfg.get("flip_allowed", {})
    cfg["flip_allowed"] = {
        name: bool(saved_flip.get(name, False)) for name in du.load_classes()
    }
    return cfg


def save_config(cfg):
    du.ensure_dirs()
    with open(du.AUG_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------- flip logic ---
def classes_blocking_flip(cfg):
    """Class names the operator marked as unsafe to mirror."""
    return [name for name, allowed in cfg.get("flip_allowed", {}).items() if not allowed]


def flip_mode(cfg):
    """Decide how horizontal flip is applied.

    Returns 'online' when YOLO can flip every image itself, 'offline' when only some
    classes may be flipped and the copies have to be made here, or 'none' when no
    class allows it.
    """
    allowed = cfg.get("flip_allowed", {})
    if not allowed:
        return "online"
    blocked = classes_blocking_flip(cfg)
    if not blocked:
        return "online"
    if len(blocked) == len(allowed):
        return "none"
    return "offline"


# --------------------------------------------------------------- label file ---
def _read_label(path):
    """Return the label file as a list of (class_id, cx, cy, w, h)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                rows.append((int(parts[0]), *[float(p) for p in parts[1:]]))
            except ValueError:
                continue
    return rows


def _write_label(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for cid, cx, cy, bw, bh in rows:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _aug_paths(img_path, suffix):
    """Build the image and label paths for a generated variant."""
    folder = os.path.dirname(img_path)
    base, ext = os.path.splitext(os.path.basename(img_path))
    new_name = f"{base}{du.AUG_MARK}_{suffix}{ext}"
    new_img = os.path.join(folder, new_name)
    new_lbl = du.label_path_for_image(new_img)
    return new_img, new_lbl


# ------------------------------------------------------------------ cleanup ---
def clear_augmented():
    """Delete every generated file. Called before a re-split and before training, so
    old variants can never be counted twice or leak into the validation set."""
    removed = 0
    for folder in (du.IMAGES_TRAIN, du.IMAGES_VAL, du.LABELS_TRAIN, du.LABELS_VAL):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if du.is_augmented(name):
                try:
                    os.remove(os.path.join(folder, name))
                    removed += 1
                except OSError:
                    pass
    return removed


# --------------------------------------------------------------- generation ---
def build_offline_augmentations(cfg, log=None):
    """Create the extra training images described by cfg.

    Only the training split is touched. Augmenting the validation split would make the
    reported accuracy meaningless, because the model would be graded on pictures
    derived from the ones it was taught with.

    Returns a dict with the number of files created per kind.
    """
    def _log(msg):
        if log:
            log(msg)

    du.ensure_dirs()
    counts = {"flip": 0, "gray": 0, "blur": 0}

    classes = du.load_classes()
    blocked_ids = {
        classes.index(name) for name in classes_blocking_flip(cfg) if name in classes
    }
    do_flip = flip_mode(cfg) == "offline" and cfg["online"].get("fliplr", 0) > 0
    do_gray = bool(cfg.get("offline_gray"))
    do_blur = bool(cfg.get("offline_blur"))
    if not (do_flip or do_gray or do_blur):
        return counts

    ksize = int(cfg.get("blur_strength", 3))
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, min(9, ksize))

    for name in du.list_images(du.IMAGES_TRAIN):
        img_path = os.path.join(du.IMAGES_TRAIN, name)
        lbl_path = du.label_path_for_image(img_path)
        rows = _read_label(lbl_path)
        if not rows:
            continue  # unlabelled images teach nothing, so no point copying them
        image = du.read_image(img_path)
        if image is None:
            continue

        if do_flip and not any(cid in blocked_ids for cid, *_ in rows):
            new_img, new_lbl = _aug_paths(img_path, "flip")
            if du.save_image(new_img, cv2.flip(image, 1)):
                # Mirroring the picture mirrors the box centre as well: cx -> 1 - cx.
                _write_label(new_lbl, [(cid, 1.0 - cx, cy, bw, bh) for cid, cx, cy, bw, bh in rows])
                counts["flip"] += 1

        if do_gray:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Back to three channels: YOLO expects colour input, and a plain grayscale
            # file would be loaded with a different channel count.
            new_img, new_lbl = _aug_paths(img_path, "gray")
            if du.save_image(new_img, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)):
                _write_label(new_lbl, rows)
                counts["gray"] += 1

        if do_blur:
            new_img, new_lbl = _aug_paths(img_path, "blur")
            if du.save_image(new_img, cv2.GaussianBlur(image, (ksize, ksize), 0)):
                _write_label(new_lbl, rows)
                counts["blur"] += 1

    made = [f"{v} images ({k})" for k, v in counts.items() if v]
    if made:
        _log("[AUG] Offline augmentation generated: " + ", ".join(made))
    return counts


# ------------------------------------------------------------ online params ---
def online_params(cfg):
    """Build the augmentation keyword arguments for model.train().

    fliplr is forced to 0 when any class may not be mirrored, because YOLO would
    otherwise flip those images too and teach the model a label that contradicts
    what the picture now shows.
    """
    params = dict(cfg["online"])
    mode = flip_mode(cfg)
    if mode in ("offline", "none"):
        params["fliplr"] = 0.0
    return params


def describe(cfg):
    """One line summary written to the training log, so the run is reproducible."""
    mode = flip_mode(cfg)
    mode_text = {
        "online": "horizontal flip by YOLO on each epoch",
        "offline": "offline horizontal flip for safe classes only",
        "none": "horizontal flip disabled",
    }[mode]
    online = ", ".join(f"{k}={v}" for k, v in sorted(cfg["online"].items()))
    extra = []
    if cfg.get("offline_gray"):
        extra.append("grayscale")
    if cfg.get("offline_blur"):
        extra.append(f"blur k={cfg.get('blur_strength', 3)}")
    extra_text = (" | offline copies: " + ", ".join(extra)) if extra else ""
    return f"{mode_text}{extra_text} | {online}"
