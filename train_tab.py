# -*- coding: utf-8 -*-
"""
train_tab.py
-------------
โหมดที่ 1: การสร้างและฝึกสอนโมเดล (Model Training)
คุณสมบัติ:
  - นำเข้าชุดข้อมูลรูปภาพจากกล้อง Webcam เพื่อสร้างชุดข้อมูลได้
  - นำเข้าชุดข้อมูลที่ผ่านการติดป้ายกำกับ (Data Labeling) แล้ว หรือทำ Data Labeling ภายในโปรแกรม
  - เลือกใช้งานโมเดล AI สำหรับตรวจจับวัตถุ (YOLOv8 เป็นค่าเริ่มต้น เลือกขนาดโมเดลได้)
  - ฝึกสอนโมเดล (Training) และนำโมเดลไปใช้งาน (Inference) ได้
"""

import os
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
from PIL import Image, ImageTk

import dataset_utils as du
import camera_utils
import ui_theme


class TrainTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        du.ensure_dirs()

        self.cap = None
        self.preview_running = False
        self.current_label_image_path = None
        self.current_label_cv_img = None
        self.current_boxes = []  # list of (class_id, x1, y1, x2, y2) in ORIGINAL image pixel coords
        self.draw_start = None
        self.canvas_scale = 1.0
        self.canvas_offset = (0, 0)

        self._build_ui()
        self._refresh_class_list()
        self._refresh_unlabeled_list()

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self):
        # แบ่งซ้าย (กล้อง+คลาส) / ขวา (การติดป้ายกำกับ) / ล่าง (การฝึกสอน)
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.LabelFrame(top, text="📷  กล้อง Webcam  /  จัดการคลาส")
        left.pack(side="left", fill="y", padx=(0, 10))

        cam_select_row = ttk.Frame(left)
        cam_select_row.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(cam_select_row, text="เลือกกล้อง:").pack(side="left")
        self.camera_index_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(
            cam_select_row, textvariable=self.camera_index_var, state="readonly", width=6, values=["0"]
        )
        self.camera_combo.pack(side="left", padx=6)
        ttk.Button(cam_select_row, text="🔍 ค้นหากล้อง", command=self.refresh_cameras).pack(side="left", padx=4)

        ttk.Label(
            left, text="Built-in มักเป็น 0 · USB ภายนอกมักเป็น 1 ขึ้นไป", style="Sub.TLabel"
        ).pack(anchor="w", padx=10, pady=(0, 6))

        preview_card = tk.Frame(left, bg=ui_theme.BORDER, padx=1, pady=1)
        preview_card.pack(padx=10, pady=(0, 8))
        self.preview_label = tk.Label(preview_card, bg="#0c0f16")
        self.preview_label.pack()

        cam_btns = ttk.Frame(left)
        cam_btns.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(cam_btns, text="▶  เปิดกล้อง", style="Accent.TButton", command=self.start_preview).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ttk.Button(cam_btns, text="⏹  ปิดกล้อง", command=self.stop_preview).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )

        ttk.Label(left, text="คลาสที่จะบันทึกเมื่อถ่ายภาพ:").pack(anchor="w", padx=10)
        self.capture_class_var = tk.StringVar()
        self.capture_class_combo = ttk.Combobox(left, textvariable=self.capture_class_var, state="readonly")
        self.capture_class_combo.pack(fill="x", padx=10, pady=(2, 8))

        ttk.Button(
            left, text="📸  ถ่ายภาพ (บันทึกเข้าชุดข้อมูล)", style="Accent.TButton", command=self.capture_image
        ).pack(fill="x", padx=10, pady=(0, 12))

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=10, pady=4)

        ttk.Label(left, text="รายชื่อคลาส (class):").pack(anchor="w", padx=10, pady=(6, 2))
        self.class_listbox = tk.Listbox(
            left, height=6, bg=ui_theme.CARD_BG, fg=ui_theme.TEXT, relief="flat",
            highlightthickness=1, highlightbackground=ui_theme.BORDER, selectbackground=ui_theme.ACCENT,
            selectforeground="white", font=ui_theme.FONT_NORMAL,
        )
        self.class_listbox.pack(fill="x", padx=10, pady=(0, 6))

        add_class_row = ttk.Frame(left)
        add_class_row.pack(fill="x", padx=10, pady=(0, 8))
        self.new_class_var = tk.StringVar()
        ttk.Entry(add_class_row, textvariable=self.new_class_var).pack(side="left", fill="x", expand=True)
        ttk.Button(add_class_row, text="➕ เพิ่มคลาส", command=self.add_class).pack(side="left", padx=(6, 0))

        ttk.Button(left, text="📁  นำเข้ารูปภาพจากไฟล์", command=self.import_images).pack(
            fill="x", padx=10, pady=(2, 4)
        )
        ttk.Button(
            left, text="📦  นำเข้าชุดข้อมูลที่ติดป้ายแล้ว (YOLO)", command=self.import_labeled_dataset
        ).pack(fill="x", padx=10, pady=(0, 10))

        # ------------------------------------------------------------ right: labeling
        right = ttk.LabelFrame(top, text="🏷️  ติดป้ายกำกับข้อมูล (Data Labeling)")
        right.pack(side="left", fill="both", expand=True)

        control_row = ttk.Frame(right)
        control_row.pack(fill="x", padx=10, pady=10)
        ttk.Label(control_row, text="คลาสสำหรับวาดกรอบ:").pack(side="left")
        self.label_class_var = tk.StringVar()
        self.label_class_combo = ttk.Combobox(control_row, textvariable=self.label_class_var, state="readonly", width=15)
        self.label_class_combo.pack(side="left", padx=6)

        ttk.Button(control_row, text="⬅ ก่อนหน้า", command=self.prev_unlabeled).pack(side="left", padx=3)
        ttk.Button(control_row, text="ถัดไป ➡", command=self.next_unlabeled).pack(side="left", padx=3)
        ttk.Button(control_row, text="🗑 ลบกรอบที่เลือก", style="Danger.TButton", command=self.delete_selected_box).pack(
            side="left", padx=3
        )
        ttk.Button(control_row, text="💾 บันทึก Label", style="Accent.TButton", command=self.save_current_labels).pack(
            side="left", padx=3
        )

        self.label_status_var = tk.StringVar(value="ยังไม่มีรูปภาพให้ติดป้ายกำกับ")
        ttk.Label(right, textvariable=self.label_status_var, style="Sub.TLabel").pack(anchor="w", padx=10)

        body = ttk.Frame(right)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        canvas_card = tk.Frame(body, bg=ui_theme.BORDER, padx=1, pady=1)
        canvas_card.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_card, bg="#171b24", width=860, height=620, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.box_listbox = tk.Listbox(
            body, width=28, bg=ui_theme.CARD_BG, fg=ui_theme.TEXT, relief="flat",
            highlightthickness=1, highlightbackground=ui_theme.BORDER, selectbackground=ui_theme.ACCENT,
            selectforeground="white", font=ui_theme.FONT_NORMAL,
        )
        self.box_listbox.pack(side="left", fill="y", padx=(10, 0))

        # ------------------------------------------------------------ bottom: training
        bottom = ttk.LabelFrame(self, text="🚀  ฝึกสอนโมเดล (Training)")
        bottom.pack(fill="x", padx=14, pady=(0, 14))

        row1 = ttk.Frame(bottom)
        row1.pack(fill="x", padx=10, pady=10)
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
        ttk.Combobox(row1, textvariable=self.imgsz_var, state="readonly", width=6,
                     values=[320, 416, 640]).pack(side="left", padx=6)

        ttk.Button(row1, text="🔀 1) แบ่งชุด Train/Val (80/20)", command=self.split_dataset).pack(
            side="left", padx=(18, 6)
        )
        self.train_btn = ttk.Button(
            row1, text="🚀 2) เริ่มฝึกสอนโมเดล (Train)", style="Accent.TButton", command=self.start_training
        )
        self.train_btn.pack(side="left", padx=6)

        self.log_text = tk.Text(bottom, height=8, wrap="word")
        ui_theme.style_log_text(self.log_text)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    # ------------------------------------------------------------- camera ---
    def refresh_cameras(self):
        self._log("[SYS] กำลังค้นหากล้องที่เชื่อมต่อ ...")
        found = camera_utils.list_available_cameras()
        if not found:
            messagebox.showwarning("แจ้งเตือน", "ไม่พบกล้องที่เชื่อมต่ออยู่")
            return
        values = [str(i) for i in found]
        self.camera_combo["values"] = values
        # เลือก index สูงสุดที่พบให้อัตโนมัติ เพราะมักจะเป็นกล้อง USB ภายนอก (ไม่ใช่ built-in index 0)
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

    def _update_preview(self):
        if not self.preview_running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self._last_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((560, 420))
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
        self.after(30, self._update_preview)

    def capture_image(self):
        if self.cap is None or not self.preview_running:
            messagebox.showwarning("แจ้งเตือน", "กรุณาเปิดกล้องก่อนถ่ายภาพ")
            return
        if not hasattr(self, "_last_frame"):
            messagebox.showwarning("แจ้งเตือน", "ยังไม่มีภาพจากกล้อง")
            return
        cls = self.capture_class_var.get()
        if not cls:
            messagebox.showwarning("แจ้งเตือน", "กรุณาเลือกคลาสที่จะบันทึกก่อนถ่ายภาพ")
            return
        du.ensure_dirs()
        idx = len(du.list_images(du.IMAGES_TRAIN)) + len(du.list_images(du.IMAGES_VAL))
        fname = f"{cls}_{idx:05d}.jpg"
        path = os.path.join(du.IMAGES_TRAIN, fname)
        cv2.imwrite(path, self._last_frame)
        self._log(f"บันทึกภาพ: {fname} (คลาส: {cls}) -- ยังไม่ได้ติดป้ายกำกับ")
        self._refresh_unlabeled_list()

    # -------------------------------------------------------------- classes ---
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
        for i, f in enumerate(files):
            ext = os.path.splitext(f)[1]
            idx = len(du.list_images(du.IMAGES_TRAIN)) + len(du.list_images(du.IMAGES_VAL))
            dest = os.path.join(du.IMAGES_TRAIN, f"{cls}_{idx:05d}{ext}")
            shutil.copy(f, dest)
        self._log(f"นำเข้ารูปภาพ {len(files)} ไฟล์ (คลาส: {cls})")
        self._refresh_unlabeled_list()

    def import_labeled_dataset(self):
        """นำเข้าโฟลเดอร์ที่มีโครงสร้าง images/ และ labels/ (YOLO format) ที่ติดป้ายมาแล้วจากภายนอก"""
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
                incoming = [l.strip() for l in f if l.strip()]
            existing = du.load_classes()
            for c in incoming:
                if c not in existing:
                    existing.append(c)
            du.save_classes(existing)

        count = 0
        for img_name in du._collect_images_recursive(images_src) if hasattr(du, "_collect_images_recursive") else os.listdir(images_src):
            pass
        # ทำแบบเรียบง่าย: คัดลอกทุกไฟล์ภาพ+label คู่กันเข้ามาที่ images/train, labels/train
        for img_name in os.listdir(images_src):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            base = os.path.splitext(img_name)[0]
            lbl_name = base + ".txt"
            src_img = os.path.join(images_src, img_name)
            src_lbl = os.path.join(labels_src, lbl_name)
            dst_img = os.path.join(du.IMAGES_TRAIN, img_name)
            shutil.copy(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(du.LABELS_TRAIN, lbl_name))
            count += 1
        self._log(f"นำเข้าชุดข้อมูลที่ติดป้ายแล้วจำนวน {count} ภาพ")
        self._refresh_class_list()
        self._refresh_unlabeled_list()

    # ------------------------------------------------------------ labeling ---
    def _refresh_unlabeled_list(self):
        self.unlabeled_paths = du.unlabeled_images()
        # รวมภาพที่ label แล้วด้วย เพื่อให้แก้ไขย้อนหลังได้ -- ใช้ all_images() แทน
        self.all_paths = du.all_images()
        self.label_status_var.set(f"รูปภาพทั้งหมด {len(self.all_paths)} ภาพ | ยังไม่ติดป้าย {len(self.unlabeled_paths)} ภาพ")
        if self.all_paths and self.current_label_image_path is None:
            self._load_image_for_labeling(self.all_paths[0])

    def _load_image_for_labeling(self, path):
        self.current_label_image_path = path
        self.current_label_cv_img = cv2.imread(path)
        self.current_boxes = []
        # โหลด label เดิมถ้ามี
        lbl_path = du.label_path_for_image(path)
        classes = du.load_classes()
        if os.path.exists(lbl_path) and classes:
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

    def _render_canvas(self):
        self.canvas.delete("all")
        if self.current_label_cv_img is None:
            return
        h, w = self.current_label_cv_img.shape[:2]
        cw = self.canvas.winfo_width() or 640
        ch = self.canvas.winfo_height() or 480
        scale = min(cw / w, ch / h)
        self.canvas_scale = scale
        disp_w, disp_h = int(w * scale), int(h * scale)
        self.canvas_offset = ((cw - disp_w) // 2, (ch - disp_h) // 2)

        rgb = cv2.cvtColor(self.current_label_cv_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((disp_w, disp_h))
        self._canvas_imgtk = ImageTk.PhotoImage(image=img)
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
        # แปลงพิกัด canvas -> พิกัดภาพจริง
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

    # ------------------------------------------------------------- training ---
    def split_dataset(self):
        n_train, n_val = du.rebalance_train_val(val_ratio=0.2)
        self._log(f"แบ่งชุดข้อมูลใหม่: train={n_train} ภาพ, val={n_val} ภาพ")
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
        try:
            from ultralytics import YOLO
        except ImportError:
            self._log("ERROR: ไม่พบไลบรารี ultralytics กรุณาติดตั้งด้วย: pip install ultralytics")
            self.train_btn.configure(state="normal")
            return

        model_choice = self.model_size_var.get()
        base_model = model_choice.split(" ")[0]  # ตัดคำอธิบายออก เหลือแค่ชื่อไฟล์
        epochs = self.epochs_var.get()
        imgsz = self.imgsz_var.get()

        self._log(f"เริ่มฝึกสอนโมเดล: base={base_model}, epochs={epochs}, imgsz={imgsz}")
        try:
            model = YOLO(base_model)
            results = model.train(
                data=du.DATA_YAML, epochs=epochs, imgsz=imgsz, project=du.BASE_DIR, name="train_run", exist_ok=True
            )
            # หา best.pt ที่ ultralytics สร้างไว้ แล้วคัดลอกมาไว้ในตำแหน่งมาตรฐาน
            best_path = os.path.join(du.BASE_DIR, "train_run", "weights", "best.pt")
            if os.path.exists(best_path):
                du.ensure_dirs()
                shutil.copy(best_path, du.MODEL_PATH)
                self._log(f"ฝึกสอนเสร็จสิ้น! บันทึกโมเดลไว้ที่: {du.MODEL_PATH}")
            else:
                self._log("ฝึกสอนเสร็จสิ้น แต่ไม่พบไฟล์ best.pt (โปรดตรวจสอบ log ด้านบน)")
        except Exception as e:
            self._log(f"เกิดข้อผิดพลาดระหว่างฝึกสอน: {e}")
        finally:
            self.train_btn.configure(state="normal")

    # ------------------------------------------------------------------ log ---
    def _log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
