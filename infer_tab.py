# -*- coding: utf-8 -*-
"""
infer_tab.py
-------------
โหมดที่ 2: การใช้งานโมเดล (Model Deployment & Inference) + การสื่อสารกับแขนหุ่นยนต์ผ่าน TCP/IP

คุณสมบัติตามข้อกำหนด:
  - โหลดโมเดลที่ผ่านการฝึกสอน (Trained Model) มาใช้งานร่วมกับกล้อง Webcam
  - แสดงภาพจากกล้อง Webcam แบบ Real-time
  - ปุ่มเริ่มการตรวจจับ (Detect: S) -> จับภาพ, ประมวลผล (Inference), นับจำนวนวัตถุแต่ละคลาส
  - ส่งผลตรวจจับไปยังหุ่นยนต์ผ่าน TCP/IP รูปแบบ "circle,3,square,2" หรือ "none"
  - หลังส่งผลแล้วเข้าสถานะรอ (Wait State) รอรับ "Done" จากหุ่นยนต์ก่อนพร้อมตรวจจับรอบถัดไป
  - เป็น TCP Client เชื่อมต่อ Dobot MG400 (TCP Server), กำหนด IP/Port ได้
  - ตอนเชื่อมต่อสำเร็จ ส่ง "Ready" และรอ "ACK" (ครั้งเดียวตอนเริ่มต้น)
  - แสดงประวัติการรับ-ส่งข้อมูล (Communication Log)
"""

import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
from PIL import Image, ImageTk

import dataset_utils as du
import camera_utils
import ui_theme
from tcp_client import RobotClient


class InferTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.cap = None
        self.preview_running = False
        self.model = None
        self.model_path = None
        self.robot = RobotClient(log_callback=self._log_threadsafe, status_callback=self._on_status_change)
        self.state = "idle"  # idle | connected | ready | waiting_done
        self._last_frame = None

        self._build_ui()
        self.bind_all("s", self._on_key_s)  # ปุ่มลัด "S" สำหรับเริ่มการตรวจจับ (Start Detection)

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.LabelFrame(top, text="🧠  โมเดล  &  กล้อง Webcam (Real-time)")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        model_row = ttk.Frame(left)
        model_row.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Button(model_row, text="📂  โหลดโมเดล (.pt)", style="Accent.TButton", command=self.load_model).pack(
            side="left"
        )
        self.model_label_var = tk.StringVar(value="ยังไม่ได้โหลดโมเดล")
        ttk.Label(model_row, textvariable=self.model_label_var, style="Sub.TLabel").pack(side="left", padx=10)

        cam_select_row = ttk.Frame(left)
        cam_select_row.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(cam_select_row, text="เลือกกล้อง:").pack(side="left")
        self.camera_index_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(
            cam_select_row, textvariable=self.camera_index_var, state="readonly", width=6, values=["0"]
        )
        self.camera_combo.pack(side="left", padx=6)
        ttk.Button(cam_select_row, text="🔍 ค้นหากล้อง", command=self.refresh_cameras).pack(side="left", padx=4)

        preview_card = tk.Frame(left, bg=ui_theme.BORDER, padx=1, pady=1)
        preview_card.pack(padx=10, pady=(0, 8))
        self.preview_label = tk.Label(preview_card, bg="#0c0f16")
        self.preview_label.pack()

        cam_btns = ttk.Frame(left)
        cam_btns.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(cam_btns, text="▶  เปิดกล้อง", style="Accent.TButton", command=self.start_preview).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ttk.Button(cam_btns, text="⏹  ปิดกล้อง", command=self.stop_preview).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )

        conf_row = ttk.Frame(left)
        conf_row.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(conf_row, text="Confidence threshold:").pack(side="left")
        self.conf_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(conf_row, from_=0.05, to=0.95, increment=0.05, textvariable=self.conf_var, width=6).pack(
            side="left", padx=6
        )

        self.detect_btn = ttk.Button(
            left, text="🎯  เริ่มการตรวจจับ  (Detect: S)", style="Accent.TButton",
            command=self.on_detect_pressed, state="disabled"
        )
        self.detect_btn.pack(fill="x", padx=10, pady=(4, 8))

        self.result_var = tk.StringVar(value="ผลการตรวจจับล่าสุด: -")
        result_card = tk.Frame(left, bg=ui_theme.CARD_BG, highlightthickness=1, highlightbackground=ui_theme.BORDER)
        result_card.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(
            result_card, textvariable=self.result_var, font=ui_theme.FONT_STATUS,
            background=ui_theme.CARD_BG, foreground=ui_theme.ACCENT_DARK
        ).pack(anchor="w", padx=10, pady=8)

        # ------------------------------------------------------------ right: TCP/IP
        right = ttk.LabelFrame(top, text="🤖  การสื่อสารกับแขนหุ่นยนต์  (TCP/IP Socket)")
        right.pack(side="left", fill="both", expand=True)

        conn_row = ttk.Frame(right)
        conn_row.pack(fill="x", padx=10, pady=10)
        ttk.Label(conn_row, text="IP Address:").grid(row=0, column=0, sticky="w")
        self.ip_var = tk.StringVar(value="192.168.1.6")
        ttk.Entry(conn_row, textvariable=self.ip_var, width=16).grid(row=0, column=1, padx=6)

        ttk.Label(conn_row, text="Port:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.port_var = tk.StringVar(value="29999")
        ttk.Entry(conn_row, textvariable=self.port_var, width=8).grid(row=0, column=3, padx=6)

        self.connect_btn = ttk.Button(
            conn_row, text="🔌  เชื่อมต่อ", style="Accent.TButton", command=self.connect_robot
        )
        self.connect_btn.grid(row=0, column=4, padx=(12, 4))
        ttk.Button(conn_row, text="❌  ตัดการเชื่อมต่อ", style="Danger.TButton", command=self.disconnect_robot).grid(
            row=0, column=5, padx=4
        )

        status_card = tk.Frame(right, bg=ui_theme.CARD_BG, highlightthickness=1, highlightbackground=ui_theme.BORDER)
        status_card.pack(fill="x", padx=10, pady=(0, 10))
        self.status_dot = tk.Canvas(status_card, width=14, height=14, bg=ui_theme.CARD_BG, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(10, 4), pady=8)
        self._status_dot_id = self.status_dot.create_oval(2, 2, 12, 12, fill=ui_theme.NEUTRAL, outline="")
        self.status_var = tk.StringVar(value="สถานะ: ยังไม่ได้เชื่อมต่อ")
        self.status_label = tk.Label(
            status_card, textvariable=self.status_var, font=ui_theme.FONT_STATUS,
            bg=ui_theme.CARD_BG, fg=ui_theme.NEUTRAL
        )
        self.status_label.pack(side="left", padx=(0, 10), pady=8)

        ttk.Label(right, text="ประวัติการรับ-ส่งข้อมูล (Communication Log):").pack(anchor="w", padx=10)
        self.log_text = tk.Text(right, height=26, wrap="word")
        ui_theme.style_log_text(self.log_text)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        hint = ttk.Label(
            self,
            text=(
                "💡 หากไม่มีหุ่นยนต์จริงสำหรับทดสอบ สามารถใช้โปรแกรมจำลอง TCP Server เช่น Hercules หรือ "
                "mock_robot_server.py ที่แนบมาให้ เพื่อจำลองการรับ-ส่งข้อมูล "
                "(ต้องตอบ 'ACK' หลังรับ 'Ready' และตอบ 'Done' หลังรับผลตรวจจับ)"
            ),
            style="Sub.TLabel", wraplength=1500,
        )
        hint.pack(anchor="w", padx=18, pady=(0, 10))

    # ------------------------------------------------------------- model ---
    def load_model(self):
        path = filedialog.askopenfilename(
            title="เลือกไฟล์โมเดลที่ฝึกสอนแล้ว (.pt)",
            initialdir=du.MODEL_DIR if os.path.isdir(du.MODEL_DIR) else ".",
            filetypes=[("PyTorch model", "*.pt")],
        )
        if not path:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            messagebox.showerror("ผิดพลาด", "ไม่พบไลบรารี ultralytics กรุณาติดตั้งด้วย: pip install ultralytics")
            return
        try:
            self.model = YOLO(path)
            self.model_path = path
            self.model_label_var.set(f"โมเดล: {os.path.basename(path)}")
            self._log(f"[SYS] โหลดโมเดลสำเร็จ: {path}")
            self._update_detect_btn_state()
        except Exception as e:
            messagebox.showerror("ผิดพลาด", f"โหลดโมเดลไม่สำเร็จ: {e}")

    # ------------------------------------------------------------ camera ---
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
            img = Image.fromarray(rgb).resize((880, 660))
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
        self.after(30, self._update_preview)

    # ---------------------------------------------------------------- TCP ---
    def connect_robot(self):
        ip = self.ip_var.get().strip()
        port = self.port_var.get().strip()
        if not ip or not port:
            messagebox.showwarning("แจ้งเตือน", "กรุณากรอก IP Address และ Port")
            return

        def worker():
            ok, msg = self.robot.connect(ip, port)
            self.after(0, lambda: self._after_connect(ok, msg))

        import threading

        threading.Thread(target=worker, daemon=True).start()

    def _after_connect(self, ok, msg):
        if not ok:
            messagebox.showerror("เชื่อมต่อไม่สำเร็จ", msg)
        self._update_detect_btn_state()

    def disconnect_robot(self):
        self.robot.close()
        self._update_detect_btn_state()

    def _on_status_change(self, state):
        self.state = state
        text_map = {
            "connected": "🟠 เชื่อมต่อแล้ว (รอ ACK)",
            "ready": "🟢 พร้อมทำงาน (Ready)",
            "waiting_done": "🟠 รอหุ่นยนต์ทำงานเสร็จ (Wait State - รอ 'Done')",
            "disconnected": "🔴 ยังไม่ได้เชื่อมต่อ",
        }
        color = ui_theme.STATE_COLORS.get(state, ui_theme.NEUTRAL)

        def _apply():
            self.status_var.set(text_map.get(state, f"สถานะ: {state}"))
            self.status_label.configure(fg=color)
            self.status_dot.itemconfig(self._status_dot_id, fill=color)

        self.after(0, _apply)
        self.after(0, self._update_detect_btn_state)

    def _update_detect_btn_state(self):
        can_detect = (
            self.model is not None
            and self.robot.connected
            and self.state == "ready"
        )
        self.detect_btn.configure(state="normal" if can_detect else "disabled")

    # ------------------------------------------------------------ detect ---
    def _on_key_s(self, event):
        # กรณีไม่มีการพัฒนา UI เฉพาะ ให้ปุ่ม "S" บนแป้นพิมพ์เป็นปุ่มเริ่มตรวจจับ (Start Detection)
        if str(self.detect_btn["state"]) == "normal":
            self.on_detect_pressed()

    def on_detect_pressed(self):
        if self.model is None:
            messagebox.showwarning("แจ้งเตือน", "กรุณาโหลดโมเดลก่อน")
            return
        if self._last_frame is None:
            messagebox.showwarning("แจ้งเตือน", "กรุณาเปิดกล้องก่อน")
            return
        if not self.robot.connected or self.state != "ready":
            messagebox.showwarning("แจ้งเตือน", "ยังไม่พร้อมสื่อสารกับหุ่นยนต์ (ต้องเชื่อมต่อและได้รับ ACK ก่อน)")
            return

        frame = self._last_frame.copy()
        self.detect_btn.configure(state="disabled")

        conf = float(self.conf_var.get())
        results = self.model.predict(source=frame, conf=conf, verbose=False)
        names = results[0].names
        counts = {}
        if results[0].boxes is not None:
            for cls_id in results[0].boxes.cls.tolist():
                cname = names[int(cls_id)]
                counts[cname] = counts.get(cname, 0) + 1

        if counts:
            parts = []
            for cname, cnt in counts.items():
                parts.append(cname)
                parts.append(str(cnt))
            result_string = ",".join(parts)
        else:
            result_string = "none"

        self.result_var.set(f"ผลการตรวจจับล่าสุด: {result_string}")
        self._log(f"[SYS] ผลการตรวจจับ (Inference): {result_string}")

        self.robot.send_detection_result(result_string, wait_for_done_callback=self._after_done)

    def _after_done(self, success, message):
        self.after(0, lambda: self._log(f"[SYS] {message}"))
        self.after(0, self._update_detect_btn_state)

    # ------------------------------------------------------------------ log ---
    def _log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _log_threadsafe(self, msg):
        self.after(0, lambda: self._log(msg))
