# -*- coding: utf-8 -*-
"""
train_tab.py
------------
Mode 1: dataset creation and model training.

  - capture images from a USB webcam to build a dataset
  - import images, or a whole dataset that was labelled elsewhere (YOLO format)
  - draw bounding boxes and save them as YOLO label files
  - split the data into train/val and train a YOLOv8 model

Layout note: every panel is packed so that the controls keep their space and the
image areas absorb whatever is left over. The camera preview starts as an empty
label and grows the moment a frame is displayed, so if it were packed normally it
would push the class controls off the bottom of the screen. Two things prevent that:
the preview lives in a container with pack_propagate(False), so the image can never
enlarge its parent, and the controls below it are packed with side="bottom", so pack
gives them their space first.
"""

import os
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
from PIL import Image, ImageTk

import augment_utils as au
import camera_utils
import dataset_utils as du
import ui_theme
from augment_dialog import AugmentDialog

LEFT_COLUMN_WIDTH = 430


def class_index_for_shortcut(key, class_count):
    """Map keyboard keys 1-9 to a zero-based class index, or None."""
    if len(key) == 1 and key in "123456789":
        index = int(key) - 1
        if index < class_count:
            return index
    return None


class BoxClassDialog(tk.Toplevel):
    """Small modal picker shown after a new bounding box has been drawn."""

    def __init__(self, master, classes, initial_class="", x=None, y=None):
        super().__init__(master)
        self.result = None
        self.classes = classes
        self.initial_class = initial_class if initial_class in classes else classes[0]
        self.title("เลือกคลาสของกรอบ")
        self.configure(bg=ui_theme.BG)
        self.resizable(False, False)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        ttk.Label(self, text="กรอบนี้คือคลาสอะไร?", style="Header.TLabel").pack(
            anchor="w", padx=14, pady=(12, 4)
        )
        ttk.Label(
            self, text="กดเลข 1–9 เพื่อเลือก · Enter = คลาสล่าสุด · Esc = ยกเลิก",
            style="Sub.TLabel",
        ).pack(anchor="w", padx=14, pady=(0, 8))

        grid = ttk.Frame(self)
        grid.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        for index, name in enumerate(classes):
            shortcut = f"{index + 1}. " if index < 9 else ""
            style = "Accent.TButton" if name == self.initial_class else "TButton"
            ttk.Button(
                grid, text=shortcut + name, style=style,
                command=lambda selected=name: self._choose(selected),
            ).grid(row=index // 3, column=index % 3, sticky="ew", padx=3, pady=3)
        for column in range(min(3, len(classes))):
            grid.columnconfigure(column, weight=1)

        self.bind("<KeyPress>", self._on_key)
        self.grab_set()
        self.update_idletasks()
        px = self.winfo_pointerx() if x is None else x
        py = self.winfo_pointery() if y is None else y
        px = min(max(0, px + 8), max(0, self.winfo_screenwidth() - self.winfo_reqwidth()))
        py = min(max(0, py + 8), max(0, self.winfo_screenheight() - self.winfo_reqheight()))
        self.geometry(f"+{px}+{py}")
        self.focus_force()

    def _on_key(self, event):
        index = class_index_for_shortcut(event.char, len(self.classes))
        if index is not None:
            self._choose(self.classes[index])
        elif event.keysym in ("Return", "KP_Enter"):
            self._choose(self.initial_class)
        elif event.keysym == "Escape":
            self._cancel()

    def _choose(self, class_name):
        self.result = class_name
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class TrainTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        du.ensure_dirs()

        self.cap = None
        self.preview_running = False
        self._last_frame = None
        self.current_label_image_path = None
        self.current_label_cv_img = None
        self.current_boxes = []  # (class_id, x1, y1, x2, y2) in original image pixels
        self.labels_dirty = False
        self.draw_start = None
        self.canvas_scale = 1.0
        self.canvas_offset = (0, 0)
        self._canvas_size = (0, 0)

        self._build_ui()
        self._refresh_class_list()
        self._refresh_unlabeled_list()

    # ----------------------------------------------------------------- UI ---
    def _build_ui(self):
        # The training panel is packed first, against the bottom edge, so it always
        # keeps its height no matter how tall the panels above want to be.
        bottom = ttk.LabelFrame(self, text="ฝึกสอนโมเดล (Training)")
        bottom.pack(side="bottom", fill="x", padx=12, pady=(0, 12))
        self._build_training_panel(bottom)

        top = ttk.Frame(self)
        top.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        left = ttk.LabelFrame(top, text="กล้อง Webcam / จัดการคลาส", width=LEFT_COLUMN_WIDTH)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        self._build_camera_panel(left)

        right = ttk.LabelFrame(top, text="ติดป้ายกำกับข้อมูล (Data Labeling)")
        right.pack(side="left", fill="both", expand=True)
        self._build_labeling_panel(right)

    def _build_camera_panel(self, left):
        # --- top anchored -------------------------------------------------
        cam_select_row = ttk.Frame(left)
        cam_select_row.pack(side="top", fill="x", padx=10, pady=(10, 4))
        ttk.Label(cam_select_row, text="เลือกกล้อง:").pack(side="left")
        self.camera_index_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(
            cam_select_row, textvariable=self.camera_index_var, state="readonly", width=6, values=["0"]
        )
        self.camera_combo.pack(side="left", padx=6)
        ttk.Button(cam_select_row, text="ค้นหากล้อง", command=self.refresh_cameras).pack(side="left", padx=4)

        ttk.Label(
            left, text="Built-in มักเป็น 0 · USB ภายนอกมักเป็น 1 ขึ้นไป", style="Sub.TLabel"
        ).pack(side="top", anchor="w", padx=10, pady=(0, 6))

        # --- bottom anchored, packed in reverse visual order ---------------
        ttk.Button(
            left, text="นำเข้าชุดข้อมูลที่ติดป้ายแล้ว (YOLO)", command=self.import_labeled_dataset
        ).pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        ttk.Button(
            left, text="เริ่ม Dataset ใหม่ (สำรองชุดเดิม)", style="Danger.TButton",
            command=self.reset_dataset,
        ).pack(side="bottom", fill="x", padx=10, pady=(0, 4))
        ttk.Button(left, text="นำเข้ารูปภาพจากไฟล์", command=self.import_images).pack(
            side="bottom", fill="x", padx=10, pady=(2, 4)
        )

        add_class_row = ttk.Frame(left)
        add_class_row.pack(side="bottom", fill="x", padx=10, pady=(0, 8))
        self.new_class_var = tk.StringVar()
        entry = ttk.Entry(add_class_row, textvariable=self.new_class_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.add_class())
        ttk.Button(add_class_row, text="เพิ่มคลาส", command=self.add_class).pack(side="left", padx=(6, 0))

        self.class_listbox = tk.Listbox(
            left, height=3, bg=ui_theme.CARD_BG, fg=ui_theme.TEXT, relief="flat",
            highlightthickness=1, highlightbackground=ui_theme.BORDER, selectbackground=ui_theme.ACCENT,
            selectforeground="white", font=ui_theme.FONT_NORMAL,
        )
        self.class_listbox.pack(side="bottom", fill="x", padx=10, pady=(0, 6))
        ttk.Label(left, text="รายชื่อคลาส (class):").pack(side="bottom", anchor="w", padx=10, pady=(6, 2))

        ttk.Separator(left, orient="horizontal").pack(side="bottom", fill="x", padx=10, pady=4)

        ttk.Button(
            left, text="ถ่ายภาพ (บันทึกเข้าชุดข้อมูล)", style="Accent.TButton", command=self.capture_image
        ).pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        ttk.Label(
            left, text="ถ่ายภาพก่อน แล้วกำหนดหลายคลาสได้ตอนวาดกรอบ", style="Sub.TLabel"
        ).pack(side="bottom", anchor="w", padx=10, pady=(0, 6))

        cam_btns = ttk.Frame(left)
        cam_btns.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        ttk.Button(cam_btns, text="เปิดกล้อง", style="Accent.TButton", command=self.start_preview).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ttk.Button(cam_btns, text="ปิดกล้อง", command=self.stop_preview).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )

        # --- fills whatever space is left ---------------------------------
        # The requested height is the preview's preferred size: pack hands it that much
        # first and any spare space on top, and because this is the last widget packed
        # it is also the one that shrinks when the window gets too small - never the
        # controls above and below it.
        self.preview_frame = tk.Frame(left, bg=ui_theme.BORDER, height=240)
        self.preview_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 8))
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_frame, bg="#0c0f16")
        self.preview_label.pack(fill="both", expand=True)

    def _build_labeling_panel(self, right):
        control_row = ttk.Frame(right)
        control_row.pack(side="top", fill="x", padx=10, pady=10)
        ttk.Label(control_row, text="คลาสล่าสุด (Enter ใน popup):").pack(side="left")
        self.label_class_var = tk.StringVar()
        self.label_class_combo = ttk.Combobox(
            control_row, textvariable=self.label_class_var, state="readonly", width=15
        )
        self.label_class_combo.pack(side="left", padx=6)

        ttk.Button(
            control_row, text="ลบกรอบที่เลือก", style="Danger.TButton", command=self.delete_selected_box
        ).pack(side="left", padx=3)
        ttk.Button(
            control_row, text="บันทึก Label", style="Accent.TButton", command=self.save_current_labels
        ).pack(side="left", padx=3)

        ttk.Label(
            right,
            text="ลากวาดกรอบก่อน แล้วเลือกคลาสจาก popup ด้วยเมาส์หรือปุ่มเลข 1–9",
            style="Sub.TLabel",
        ).pack(side="top", anchor="w", padx=10, pady=(0, 4))

        nav_row = ttk.Frame(right)
        nav_row.pack(side="top", fill="x", padx=10, pady=(0, 6))
        ttk.Button(nav_row, text="ก่อนหน้า", command=self.prev_unlabeled).pack(side="left", padx=(0, 4))
        self.image_selector_var = tk.StringVar()
        self.image_selector = ttk.Combobox(
            nav_row, textvariable=self.image_selector_var, state="readonly", width=42
        )
        self.image_selector.pack(side="left", fill="x", expand=True, padx=4)
        self.image_selector.bind("<<ComboboxSelected>>", self._on_image_selected)
        ttk.Button(nav_row, text="ถัดไป", command=self.next_unlabeled).pack(side="left", padx=4)
        ttk.Button(
            nav_row, text="ลบภาพนี้", style="Danger.TButton", command=self.delete_current_image
        ).pack(side="left", padx=(4, 0))

        self.label_status_var = tk.StringVar(value="ยังไม่มีรูปภาพให้ติดป้ายกำกับ")
        ttk.Label(right, textvariable=self.label_status_var, style="Sub.TLabel").pack(
            side="top", anchor="w", padx=10
        )

        body = ttk.Frame(right)
        body.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.box_listbox = tk.Listbox(
            body, width=24, bg=ui_theme.CARD_BG, fg=ui_theme.TEXT, relief="flat",
            highlightthickness=1, highlightbackground=ui_theme.BORDER, selectbackground=ui_theme.ACCENT,
            selectforeground="white", font=ui_theme.FONT_NORMAL,
        )
        self.box_listbox.pack(side="right", fill="y", padx=(10, 0))

        canvas_card = tk.Frame(body, bg=ui_theme.BORDER, padx=1, pady=1)
        canvas_card.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_card, bg="#171b24", width=320, height=240, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _build_training_panel(self, bottom):
        row1 = ttk.Frame(bottom)
        row1.pack(fill="x", padx=10, pady=8)
        ttk.Label(row1, text="โมเดล YOLO:").pack(side="left")
        self.model_size_var = tk.StringVar(value="yolov8n.pt")
        ttk.Combobox(
            row1, textvariable=self.model_size_var, state="readonly", width=14,
            values=["yolov8n.pt", "yolov8s.pt", "yolov8n.yaml (ฝึกใหม่ทั้งหมด ไม่ใช้ pretrained)"]
        ).pack(side="left", padx=6)

        ttk.Label(row1, text="Epochs:").pack(side="left", padx=(14, 0))
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(row1, from_=1, to=1000, textvariable=self.epochs_var, width=6).pack(side="left", padx=6)

        ttk.Label(row1, text="Image size:").pack(side="left", padx=(14, 0))
        self.imgsz_var = tk.IntVar(value=640)
        ttk.Combobox(
            row1, textvariable=self.imgsz_var, state="readonly", width=6, values=[320, 416, 640]
        ).pack(side="left", padx=6)

        ttk.Button(row1, text="ตั้งค่าการเพิ่มข้อมูล", command=self.open_augment_dialog).pack(
            side="left", padx=(18, 6)
        )
        ttk.Button(row1, text="1) แบ่งชุด Train/Val (80/20)", command=self.split_dataset).pack(
            side="left", padx=6
        )
        self.train_btn = ttk.Button(
            row1, text="2) เริ่มฝึกสอนโมเดล (Train)", style="Accent.TButton", command=self.start_training
        )
        self.train_btn.pack(side="left", padx=6)

        self.log_text = tk.Text(bottom, height=4, wrap="word")
        ui_theme.style_log_text(self.log_text)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    # ------------------------------------------------------------- camera ---
    def refresh_cameras(self):
        self._log("[SYS] Scanning connected cameras... (may take 10-30 seconds)")
        self.update_idletasks()
        found = camera_utils.list_available_cameras()
        if not found:
            messagebox.showwarning(
                "แจ้งเตือน",
                "ไม่พบกล้องที่เชื่อมต่ออยู่\n\n"
                "ถ้ามีโปรแกรมอื่นเปิดกล้องค้างไว้ (รวมถึงหน้าต่างนี้เอง) จะสแกนไม่เจอ "
                "ให้กดปิดกล้องก่อนแล้วลองใหม่",
            )
            return
        values = [str(i) for i in found]
        self.camera_combo["values"] = values
        # The highest index is picked automatically because it is usually the external
        # USB camera; the competition rules forbid the built-in one, which is index 0.
        self.camera_index_var.set(values[-1])
        self._log(f"[SYS] Camera indexes found: {values} (auto-selected {values[-1]})")

    def start_preview(self):
        if self.preview_running:
            return
        idx = int(self.camera_index_var.get())
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            messagebox.showerror(
                "ผิดพลาด", f"ไม่สามารถเปิดกล้อง (index {idx}) ได้ กรุณากด 'ค้นหากล้อง' แล้วเลือก index ใหม่"
            )
            return
        self.preview_running = True
        self._update_preview()

    def stop_preview(self):
        self.preview_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.preview_label.configure(image="")
        self.preview_label.imgtk = None

    def _update_preview(self):
        if not self.preview_running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self._last_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            size = ui_theme.fit_size(
                w, h, self.preview_frame.winfo_width(), self.preview_frame.winfo_height()
            )
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb).resize(size))
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
        self.after(30, self._update_preview)

    def capture_image(self):
        if self.cap is None or not self.preview_running:
            messagebox.showwarning("แจ้งเตือน", "กรุณาเปิดกล้องก่อนถ่ายภาพ")
            return
        if self._last_frame is None:
            messagebox.showwarning("แจ้งเตือน", "ยังไม่มีภาพจากกล้อง")
            return
        du.ensure_dirs()
        path = du.unique_image_path("capture", ".jpg")
        fname = os.path.basename(path)
        if not du.save_image(path, self._last_frame):
            messagebox.showerror("ผิดพลาด", "บันทึกภาพไม่สำเร็จ")
            return
        self._log(f"[DATA] Image saved: {fname} -- not labeled yet")
        self._refresh_unlabeled_list()

    # ------------------------------------------------------------ classes ---
    def _refresh_class_list(self):
        classes = du.load_classes()
        self.class_listbox.delete(0, tk.END)
        for c in classes:
            self.class_listbox.insert(tk.END, c)
        self.label_class_combo["values"] = classes
        if classes and not self.label_class_var.get():
            self.label_class_var.set(classes[0])
        if not classes:
            self.label_class_var.set("")

    def add_class(self):
        name = self.new_class_var.get().strip()
        if not name:
            return
        if not du.is_valid_class_name(name):
            messagebox.showwarning(
                "ชื่อคลาสไม่ถูกต้อง",
                "ใช้เฉพาะตัวอักษรอังกฤษ A-Z, a-z ตัวเลข และ _ เท่านั้น\nเช่น circle หรือ arrow_left",
            )
            return
        du.add_class(name)
        self.new_class_var.set("")
        self._refresh_class_list()
        self._log(f"[DATA] Class added: {name}")

    # ------------------------------------------------------------- import ---
    def import_images(self):
        files = filedialog.askopenfilenames(
            title="เลือกไฟล์รูปภาพ", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not files:
            return
        du.ensure_dirs()
        for f in files:
            ext = os.path.splitext(f)[1]
            dest = du.unique_image_path("imported", ext)
            shutil.copy(f, dest)
        self._log(f"[DATA] Imported {len(files)} images -- assign classes while drawing boxes")
        self._refresh_unlabeled_list()

    def import_labeled_dataset(self):
        """Import an external folder that contains images/ and labels/ in YOLO format."""
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ dataset (ต้องมีโฟลเดอร์ images และ labels)")
        if not folder:
            return
        try:
            count, _ = du.import_labeled_yolo_dataset(folder)
        except (OSError, ValueError) as exc:
            messagebox.showerror("นำเข้าไม่สำเร็จ", str(exc))
            return
        self._log(f"[DATA] Imported labeled dataset: {count} images")
        self._refresh_class_list()
        self._refresh_unlabeled_list()

    # ----------------------------------------------------------- labeling ---
    def _refresh_unlabeled_list(self):
        self.unlabeled_paths = du.unlabeled_images()
        self.all_paths = du.all_images()
        self.image_path_by_name = {
            self._image_display_name(path): path for path in self.all_paths
        }
        self.image_selector["values"] = list(self.image_path_by_name)
        if self.current_label_image_path not in self.all_paths:
            self.current_label_image_path = None
        if self.all_paths and self.current_label_image_path is None:
            self._load_image_for_labeling(self.all_paths[0])
        elif not self.all_paths:
            self.current_label_cv_img = None
            self.current_boxes = []
            self.labels_dirty = False
            self.image_selector_var.set("")
            self.canvas.delete("all")
        self._update_label_status()

    def _image_display_name(self, path):
        split = os.path.basename(os.path.dirname(path))
        return f"{split}/{os.path.basename(path)}"

    def _update_label_status(self):
        current = ""
        if self.current_label_image_path in self.all_paths:
            index = self.all_paths.index(self.current_label_image_path) + 1
            current = f" | ภาพที่ {index}/{len(self.all_paths)}"
            self.image_selector_var.set(self._image_display_name(self.current_label_image_path))
        self.label_status_var.set(
            f"รูปภาพทั้งหมด {len(self.all_paths)} ภาพ | ยังไม่ติดป้าย {len(self.unlabeled_paths)} ภาพ{current}"
        )

    def _on_image_selected(self, _event=None):
        path = self.image_path_by_name.get(self.image_selector_var.get())
        if not path or path == self.current_label_image_path:
            return
        if self.labels_dirty:
            self.save_current_labels(silent=True)
            self._log("[DATA] Labels auto-saved before changing image")
        self._load_image_for_labeling(path)

    def _load_image_for_labeling(self, path):
        self.current_label_image_path = path
        self.current_label_cv_img = du.read_image(path)
        self.current_boxes = []
        self.labels_dirty = False
        lbl_path = du.label_path_for_image(path)
        classes = du.load_classes()
        if self.current_label_cv_img is not None and os.path.exists(lbl_path) and classes:
            h, w = self.current_label_cv_img.shape[:2]
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    self.current_boxes.append((cid, x1, y1, x2, y2))
        self._render_canvas()
        self._update_label_status()

    def _on_canvas_resize(self, event):
        # Re-render only when the size really changed, otherwise every drag event
        # would trigger a redraw loop.
        if (event.width, event.height) != self._canvas_size:
            self._canvas_size = (event.width, event.height)
            self._render_canvas()

    def _render_canvas(self):
        self.canvas.delete("all")
        if self.current_label_cv_img is None:
            return
        h, w = self.current_label_cv_img.shape[:2]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 640, 480
        scale = min(cw / w, ch / h)
        self.canvas_scale = scale
        disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
        self.canvas_offset = ((cw - disp_w) // 2, (ch - disp_h) // 2)

        rgb = cv2.cvtColor(self.current_label_cv_img, cv2.COLOR_BGR2RGB)
        self._canvas_imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb).resize((disp_w, disp_h)))
        ox, oy = self.canvas_offset
        self.canvas.create_image(ox, oy, anchor="nw", image=self._canvas_imgtk)

        classes = du.load_classes()
        self.box_listbox.delete(0, tk.END)
        for i, (cid, x1, y1, x2, y2) in enumerate(self.current_boxes):
            cname = classes[cid] if 0 <= cid < len(classes) else f"class#{cid}"
            dx1, dy1 = x1 * scale + ox, y1 * scale + oy
            dx2, dy2 = x2 * scale + ox, y2 * scale + oy
            self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="#00ff66", width=2)
            self.canvas.create_text(dx1 + 4, dy1 + 10, text=cname, fill="#00ff66", anchor="w")
            self.box_listbox.insert(tk.END, f"{i}: {cname}")

    def _on_canvas_press(self, event):
        self.draw_start = (event.x, event.y)

    def _on_canvas_drag(self, event):
        if not self.draw_start:
            return
        self._render_canvas()
        x0, y0 = self.draw_start
        self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="yellow", width=1, dash=(4, 2))

    def _on_canvas_release(self, event):
        if not self.draw_start or self.current_label_cv_img is None:
            self.draw_start = None
            return
        classes = du.load_classes()
        if not classes:
            messagebox.showwarning("ยังไม่มีคลาส", "กรุณาเพิ่มคลาสอย่างน้อย 1 คลาสก่อนติดป้ายกำกับ")
            self.draw_start = None
            self._render_canvas()
            return

        ox, oy = self.canvas_offset
        scale = self.canvas_scale or 1.0
        x0, y0 = self.draw_start
        x1c, y1c = event.x, event.y
        # Canvas coordinates back to original image pixels.
        ix0 = (min(x0, x1c) - ox) / scale
        iy0 = (min(y0, y1c) - oy) / scale
        ix1 = (max(x0, x1c) - ox) / scale
        iy1 = (max(y0, y1c) - oy) / scale

        h, w = self.current_label_cv_img.shape[:2]
        ix0, ix1 = max(0, min(ix0, w)), max(0, min(ix1, w))
        iy0, iy1 = max(0, min(iy0, h)), max(0, min(iy1, h))

        if abs(ix1 - ix0) > 4 and abs(iy1 - iy0) > 4:
            dialog = BoxClassDialog(
                self.winfo_toplevel(), classes, self.label_class_var.get(), event.x_root, event.y_root
            )
            self.wait_window(dialog)
            cls_name = dialog.result
            if cls_name is not None:
                cid = classes.index(cls_name)
                self.label_class_var.set(cls_name)
                self.current_boxes.append((cid, ix0, iy0, ix1, iy1))
                self.labels_dirty = True
        self.draw_start = None
        self._render_canvas()

    def delete_selected_box(self):
        sel = self.box_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.current_boxes):
            del self.current_boxes[idx]
            self.labels_dirty = True
        self._render_canvas()

    def save_current_labels(self, silent=False):
        if not self.current_label_image_path or self.current_label_cv_img is None:
            return
        h, w = self.current_label_cv_img.shape[:2]
        lbl_path = du.label_path_for_image(self.current_label_image_path)
        os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cid, x1, y1, x2, y2 in self.current_boxes:
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        self.labels_dirty = False
        if not silent:
            self._log(f"[DATA] Labels saved: {os.path.basename(lbl_path)} ({len(self.current_boxes)} boxes)")
        self._refresh_unlabeled_list()

    def _step_image(self, step):
        if not self.all_paths:
            return
        if self.labels_dirty:
            self.save_current_labels(silent=True)
            self._log("[DATA] Labels auto-saved before changing image")
        try:
            idx = self.all_paths.index(self.current_label_image_path)
        except ValueError:
            idx = 0
        idx = (idx + step) % len(self.all_paths)
        self._load_image_for_labeling(self.all_paths[idx])

    def next_unlabeled(self):
        self._step_image(1)

    def prev_unlabeled(self):
        self._step_image(-1)

    def delete_current_image(self):
        path = self.current_label_image_path
        if not path:
            return
        if str(self.train_btn["state"]) == "disabled":
            messagebox.showwarning("กำลังฝึกสอน", "ไม่สามารถลบภาพระหว่างฝึกสอนได้")
            return
        if not messagebox.askyesno(
            "ลบภาพออกจาก Dataset",
            f"ต้องการลบ {os.path.basename(path)} หรือไม่?\n"
            "Label และภาพ augmentation ที่สร้างจากภาพนี้จะถูกลบด้วย",
        ):
            return
        old_index = self.all_paths.index(path) if path in self.all_paths else 0
        try:
            removed = du.delete_dataset_image(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("ลบภาพไม่สำเร็จ", str(exc))
            return
        self.current_label_image_path = None
        self.current_label_cv_img = None
        self.current_boxes = []
        self.labels_dirty = False
        self._refresh_unlabeled_list()
        if self.all_paths:
            self._load_image_for_labeling(self.all_paths[min(old_index, len(self.all_paths) - 1)])
        self._log(f"[DATA] Deleted image and {removed} related files: {os.path.basename(path)}")

    # ----------------------------------------------------------- training ---
    def open_augment_dialog(self):
        AugmentDialog(self.winfo_toplevel(), on_saved=self._on_augment_saved)

    def _on_augment_saved(self, cfg):
        self._log("[AUG] Augmentation settings saved: " + au.describe(cfg))

    def split_dataset(self):
        if str(self.train_btn["state"]) == "disabled":
            messagebox.showwarning("กำลังฝึกสอน", "รอให้การฝึกสอนเสร็จก่อนแบ่ง Dataset ใหม่")
            return
        if self.labels_dirty:
            self.save_current_labels(silent=True)
        # Generated files are dropped first, otherwise the re-split would scatter them
        # into the validation set and the reported accuracy would be measured on
        # pictures derived from the training images.
        removed = au.clear_augmented()
        n_train, n_val = du.rebalance_train_val(val_ratio=0.2)
        if removed:
            self._log(f"[AUG] Removed {removed} previously generated files")
        self._log(f"[DATA] Dataset split completed: train={n_train} images, val={n_val} images")
        self.current_label_image_path = None
        self._refresh_unlabeled_list()

    def start_training(self):
        if self.labels_dirty:
            self.save_current_labels(silent=True)
        errors, warnings, stats = du.validate_dataset(min_images=5, require_validation=True)
        if errors:
            shown = errors[:12]
            more = f"\n...และอีก {len(errors) - len(shown)} รายการ" if len(errors) > len(shown) else ""
            messagebox.showerror("Dataset ยังไม่พร้อมฝึกสอน", "\n".join(shown) + more)
            return
        if warnings and not messagebox.askyesno(
            "คำเตือน Dataset", "\n".join(warnings) + "\n\nต้องการฝึกสอนต่อหรือไม่?"
        ):
            return
        du.write_data_yaml()
        options = (self.model_size_var.get(), self.epochs_var.get(), self.imgsz_var.get())
        self._log(
            f"[DATA] Dataset validation passed: train={stats['train']}, val={stats['val']}, "
            f"classes={stats['classes']}, boxes={stats['boxes']}"
        )
        self.train_btn.configure(state="disabled")
        threading.Thread(target=self._train_worker, args=(options,), daemon=True).start()

    def _train_worker(self, options):
        """Runs in a worker thread. Tk is not thread safe, so every UI call goes
        through self.after(0, ...) to run on the main loop instead."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self._log_threadsafe("[ERR] Ultralytics is not installed. Run: pip install ultralytics")
            self.after(0, lambda: self.train_btn.configure(state="normal"))
            return

        model_choice, epochs, imgsz = options
        base_model = model_choice.split(" ")[0]  # drop the Thai description after the filename

        # Rebuild the offline variants from scratch so a changed setting takes effect
        # and old copies from a previous run cannot pile up.
        cfg = au.load_config()
        au.clear_augmented()
        au.build_offline_augmentations(cfg, log=self._log_threadsafe)
        aug_params = au.online_params(cfg)

        self._log_threadsafe(f"[TRAIN] Starting training: base={base_model}, epochs={epochs}, imgsz={imgsz}")
        self._log_threadsafe("[AUG] Augmentation: " + au.describe(cfg))
        self._log_threadsafe("[TRAIN] Detailed progress is shown in the application terminal")
        try:
            model = YOLO(base_model)
            model.train(
                data=du.DATA_YAML, epochs=epochs, imgsz=imgsz,
                project=du.BASE_DIR, name="train_run", exist_ok=True,
                **aug_params,
            )
            best_path = os.path.join(du.BASE_DIR, "train_run", "weights", "best.pt")
            if os.path.exists(best_path):
                du.ensure_dirs()
                shutil.copy(best_path, du.MODEL_PATH)
                self._log_threadsafe(f"[TRAIN] Training completed. Best model saved to: {du.MODEL_PATH}")
            else:
                self._log_threadsafe("[ERR] Training finished but best.pt was not found. Check the terminal log.")
        except Exception as e:
            self._log_threadsafe(f"[ERR] Training failed: {e}")
        finally:
            self.after(0, lambda: self.train_btn.configure(state="normal"))

    # ---------------------------------------------------------------- log ---
    def _log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _log_threadsafe(self, msg):
        self.after(0, lambda: self._log(msg))

    def reset_dataset(self):
        if str(self.train_btn["state"]) == "disabled":
            messagebox.showwarning("กำลังฝึกสอน", "ไม่สามารถเปลี่ยน Dataset ระหว่างฝึกสอนได้")
            return
        if not messagebox.askyesno(
            "เริ่ม Dataset ใหม่",
            "โปรแกรมจะย้าย Dataset และโมเดลชุดปัจจุบันไปเก็บใน dataset_backups\n"
            "แล้วสร้าง Dataset ว่างสำหรับการแข่งขัน ต้องการดำเนินการหรือไม่?",
        ):
            return
        if self.labels_dirty:
            self.save_current_labels(silent=True)
        self.stop_preview()
        backup_path = du.backup_and_reset_dataset()
        self.current_label_image_path = None
        self.current_label_cv_img = None
        self.current_boxes = []
        self.labels_dirty = False
        self._canvas_imgtk = None
        self.canvas.delete("all")
        self._refresh_class_list()
        self.label_class_var.set("")
        self._refresh_unlabeled_list()
        self._log(f"[DATA] New dataset started. Previous dataset backed up to: {backup_path}")
