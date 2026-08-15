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

        self.capture_class_var = tk.StringVar()
        self.capture_class_combo = ttk.Combobox(left, textvariable=self.capture_class_var, state="readonly")
        self.capture_class_combo.pack(side="bottom", fill="x", padx=10, pady=(2, 8))
        ttk.Label(left, text="คลาสที่จะบันทึกเมื่อถ่ายภาพ:").pack(side="bottom", anchor="w", padx=10)

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
        ttk.Label(control_row, text="คลาสสำหรับวาดกรอบ:").pack(side="left")
        self.label_class_var = tk.StringVar()
        self.label_class_combo = ttk.Combobox(
            control_row, textvariable=self.label_class_var, state="readonly", width=15
        )
        self.label_class_combo.pack(side="left", padx=6)

        ttk.Button(control_row, text="ก่อนหน้า", command=self.prev_unlabeled).pack(side="left", padx=3)
        ttk.Button(control_row, text="ถัดไป", command=self.next_unlabeled).pack(side="left", padx=3)
        ttk.Button(
            control_row, text="ลบกรอบที่เลือก", style="Danger.TButton", command=self.delete_selected_box
        ).pack(side="left", padx=3)
        ttk.Button(
            control_row, text="บันทึก Label", style="Accent.TButton", command=self.save_current_labels
        ).pack(side="left", padx=3)

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
        self._log("[SYS] กำลังค้นหากล้องที่เชื่อมต่อ ... (ใช้เวลาราว 10-30 วินาที)")
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
        self._log(f"[SYS] พบกล้อง index: {values} (เลือก {values[-1]} ให้อัตโนมัติ)")

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
        cls = self.capture_class_var.get()
        if not cls:
            messagebox.showwarning(
                "แจ้งเตือน",
                "ยังไม่ได้เลือกคลาสที่จะบันทึก\n\n"
                "ถ้ายังไม่มีคลาสเลย ให้พิมพ์ชื่อคลาสในช่องด้านล่าง (เช่น circle) "
                "แล้วกด 'เพิ่มคลาส' ก่อน",
            )
            return
        du.ensure_dirs()
        idx = len(du.list_images(du.IMAGES_TRAIN)) + len(du.list_images(du.IMAGES_VAL))
        fname = f"{cls}_{idx:05d}.jpg"
        path = os.path.join(du.IMAGES_TRAIN, fname)
        du.save_image(path, self._last_frame)
        self._log(f"บันทึกภาพ: {fname} (คลาส: {cls}) -- ยังไม่ได้ติดป้ายกำกับ")
        self._refresh_unlabeled_list()

    # ------------------------------------------------------------ classes ---
    def _refresh_class_list(self):
        classes = du.load_classes()
        self.class_listbox.delete(0, tk.END)
        for c in classes:
            self.class_listbox.insert(tk.END, c)
        self.capture_class_combo["values"] = classes
        self.label_class_combo["values"] = classes
        if classes and not self.capture_class_var.get():
            self.capture_class_var.set(classes[0])
        if classes and not self.label_class_var.get():
            self.label_class_var.set(classes[0])

    def add_class(self):
        name = self.new_class_var.get().strip()
        if not name:
            return
        du.add_class(name)
        self.new_class_var.set("")
        self._refresh_class_list()
        self._log(f"เพิ่มคลาสใหม่: {name}")

    # ------------------------------------------------------------- import ---
    def import_images(self):
        cls = self.capture_class_var.get()
        if not cls:
            messagebox.showwarning("แจ้งเตือน", "กรุณาเลือก/เพิ่มคลาสก่อนนำเข้ารูปภาพ")
            return
        files = filedialog.askopenfilenames(
            title="เลือกไฟล์รูปภาพ", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not files:
            return
        du.ensure_dirs()
        for f in files:
            ext = os.path.splitext(f)[1]
            idx = len(du.list_images(du.IMAGES_TRAIN)) + len(du.list_images(du.IMAGES_VAL))
            dest = os.path.join(du.IMAGES_TRAIN, f"{cls}_{idx:05d}{ext}")
            shutil.copy(f, dest)
        self._log(f"นำเข้ารูปภาพ {len(files)} ไฟล์ (คลาส: {cls})")
        self._refresh_unlabeled_list()

    def import_labeled_dataset(self):
        """Import an external folder that contains images/ and labels/ in YOLO format."""
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ dataset (ต้องมีโฟลเดอร์ images และ labels)")
        if not folder:
            return
        images_src = os.path.join(folder, "images")
        labels_src = os.path.join(folder, "labels")
        classes_src = os.path.join(folder, "classes.txt")
        if not os.path.isdir(images_src) or not os.path.isdir(labels_src):
            messagebox.showerror("ผิดพลาด", "โฟลเดอร์ที่เลือกต้องมี images/ และ labels/ อยู่ภายใน")
            return
        du.ensure_dirs()
        if os.path.exists(classes_src):
            with open(classes_src, "r", encoding="utf-8") as f:
                incoming = [line.strip() for line in f if line.strip()]
            existing = du.load_classes()
            for c in incoming:
                if c not in existing:
                    existing.append(c)
            du.save_classes(existing)

        count = 0
        for img_name in os.listdir(images_src):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            base = os.path.splitext(img_name)[0]
            lbl_name = base + ".txt"
            src_img = os.path.join(images_src, img_name)
            src_lbl = os.path.join(labels_src, lbl_name)
            shutil.copy(src_img, os.path.join(du.IMAGES_TRAIN, img_name))
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(du.LABELS_TRAIN, lbl_name))
            count += 1
        self._log(f"นำเข้าชุดข้อมูลที่ติดป้ายแล้วจำนวน {count} ภาพ")
        self._refresh_class_list()
        self._refresh_unlabeled_list()

    # ----------------------------------------------------------- labeling ---
    def _refresh_unlabeled_list(self):
        self.unlabeled_paths = du.unlabeled_images()
        self.all_paths = du.all_images()
        self.label_status_var.set(
            f"รูปภาพทั้งหมด {len(self.all_paths)} ภาพ | ยังไม่ติดป้าย {len(self.unlabeled_paths)} ภาพ"
        )
        if self.all_paths and self.current_label_image_path is None:
            self._load_image_for_labeling(self.all_paths[0])

    def _load_image_for_labeling(self, path):
        self.current_label_image_path = path
        self.current_label_cv_img = du.read_image(path)
        self.current_boxes = []
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
        cls_name = self.label_class_var.get()
        if not cls_name:
            messagebox.showwarning("แจ้งเตือน", "กรุณาเลือกคลาสสำหรับวาดกรอบก่อน")
            self.draw_start = None
            return
        cid = classes.index(cls_name)

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
            self.current_boxes.append((cid, ix0, iy0, ix1, iy1))
        self.draw_start = None
        self._render_canvas()

    def delete_selected_box(self):
        sel = self.box_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.current_boxes):
            del self.current_boxes[idx]
        self._render_canvas()

    def save_current_labels(self):
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
        self._log(f"บันทึก label: {os.path.basename(lbl_path)} ({len(self.current_boxes)} กรอบ)")
        self._refresh_unlabeled_list()

    def _step_image(self, step):
        if not self.all_paths:
            return
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

    # ----------------------------------------------------------- training ---
    def open_augment_dialog(self):
        AugmentDialog(self.winfo_toplevel(), on_saved=self._on_augment_saved)

    def _on_augment_saved(self, cfg):
        self._log("บันทึกการตั้งค่าเพิ่มข้อมูลแล้ว: " + au.describe(cfg))

    def split_dataset(self):
        # Generated files are dropped first, otherwise the re-split would scatter them
        # into the validation set and the reported accuracy would be measured on
        # pictures derived from the training images.
        removed = au.clear_augmented()
        n_train, n_val = du.rebalance_train_val(val_ratio=0.2)
        if removed:
            self._log(f"ลบภาพที่สร้างเพิ่มไว้เดิม {removed} ไฟล์")
        self._log(f"แบ่งชุดข้อมูลใหม่: train={n_train} ภาพ, val={n_val} ภาพ")
        self.current_label_image_path = None
        self._refresh_unlabeled_list()

    def start_training(self):
        classes = du.load_classes()
        if not classes:
            messagebox.showwarning("แจ้งเตือน", "กรุณาเพิ่มคลาสอย่างน้อย 1 คลาสก่อนฝึกสอน")
            return
        if len(du.all_images()) < 5:
            messagebox.showwarning("แจ้งเตือน", "ควรมีรูปภาพที่ติดป้ายกำกับแล้วอย่างน้อยหลายภาพก่อนฝึกสอน")
        du.write_data_yaml()
        self.train_btn.configure(state="disabled")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        """Runs in a worker thread. Tk is not thread safe, so every UI call goes
        through self.after(0, ...) to run on the main loop instead."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self._log_threadsafe("ERROR: ไม่พบไลบรารี ultralytics กรุณาติดตั้งด้วย: pip install ultralytics")
            self.after(0, lambda: self.train_btn.configure(state="normal"))
            return

        model_choice = self.model_size_var.get()
        base_model = model_choice.split(" ")[0]  # drop the Thai description after the filename
        epochs = self.epochs_var.get()
        imgsz = self.imgsz_var.get()

        # Rebuild the offline variants from scratch so a changed setting takes effect
        # and old copies from a previous run cannot pile up.
        cfg = au.load_config()
        au.clear_augmented()
        au.build_offline_augmentations(cfg, log=self._log_threadsafe)
        aug_params = au.online_params(cfg)

        self._log_threadsafe(f"เริ่มฝึกสอนโมเดล: base={base_model}, epochs={epochs}, imgsz={imgsz}")
        self._log_threadsafe("การเพิ่มข้อมูล: " + au.describe(cfg))
        self._log_threadsafe("ความคืบหน้าแบบละเอียดจะแสดงในหน้าต่าง Terminal ที่รันโปรแกรม")
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
                self._log_threadsafe(f"ฝึกสอนเสร็จสิ้น! บันทึกโมเดลไว้ที่: {du.MODEL_PATH}")
            else:
                self._log_threadsafe("ฝึกสอนเสร็จสิ้น แต่ไม่พบไฟล์ best.pt (โปรดตรวจสอบ log ใน Terminal)")
        except Exception as e:
            self._log_threadsafe(f"เกิดข้อผิดพลาดระหว่างฝึกสอน: {e}")
        finally:
            self.after(0, lambda: self.train_btn.configure(state="normal"))

    # ---------------------------------------------------------------- log ---
    def _log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _log_threadsafe(self, msg):
        self.after(0, lambda: self._log(msg))
