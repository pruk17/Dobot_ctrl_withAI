# -*- coding: utf-8 -*-
"""
infer_tab.py
------------
Mode 2: run a trained model against the webcam and talk to the robot over TCP/IP.

  - load a trained model (.pt) and show the webcam live
  - "S" (or the button) grabs the current frame, runs inference and counts the
    objects of every class
  - the result is sent to the robot as "circle,3,square,2", or "none"
  - afterwards the program stays in a wait state until the robot answers "Done"

The wait state can time out. When that happens the program deliberately stays in the
wait state instead of unlocking itself, because a late "Done" would otherwise be read
as the answer to the next detection. The operator clears it with the cancel button.
"""

import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
from PIL import Image, ImageTk

import dataset_utils as du
import camera_utils
import ui_theme
from tcp_client import RobotClient

RIGHT_COLUMN_WIDTH = 560

# Widget classes where a keystroke means "the user is typing", not "start detection".
TEXT_INPUT_WIDGETS = (tk.Entry, tk.Text, tk.Spinbox, tk.Listbox, ttk.Entry, ttk.Combobox, ttk.Spinbox)

# A frame older than this is not trusted for a detection. Without this guard the
# camera can silently stop delivering frames while the last picture stays in memory,
# and a detection would then report what the camera saw minutes ago.
CAMERA_STALE_SECONDS = 2.0

# Consecutive failed reads (~1 s at 30 ms per frame) before the camera is reopened.
CAMERA_RETRY_AFTER = 30


def format_detection_result(names, class_ids):
    """Return deterministic protocol/readable strings ordered by model class id."""
    counts = {}
    for raw_id in class_ids:
        cid = int(raw_id)
        counts[cid] = counts.get(cid, 0) + 1
    if not counts:
        return "none", "ไม่พบวัตถุ"

    parts = []
    readable = []
    for cid in sorted(counts):
        cname = names[cid] if not isinstance(names, dict) else names.get(cid, str(cid))
        cname = str(cname)
        if not du.is_valid_class_name(cname):
            raise ValueError(
                f"ชื่อคลาส '{cname}' ใช้ส่ง TCP ไม่ได้ ต้องใช้เฉพาะ A-Z, a-z, 0-9 และ _"
            )
        parts.extend((cname, str(counts[cid])))
        readable.append(f"{cname} = {counts[cid]}")
    return ",".join(parts), "  ".join(readable)


class InferTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.cap = None
        self.preview_running = False
        self.model = None
        self.model_path = None
        self.state = "disconnected"  # disconnected | connected | ready | waiting_done | wait_timeout
        self._last_frame = None
        # Annotated snapshot from the most recent detection.  The camera continues
        # to be read in the background so the next cycle still uses a fresh frame,
        # while this image remains visible until the operator starts that cycle.
        self._detection_frame = None
        self.inference_running = False
        # Camera health. _last_frame_time is when _last_frame was actually captured,
        # which is what makes a stale picture detectable at all.
        self._camera_index = None
        self._last_frame_time = 0.0
        self._read_failures = 0
        self._camera_reopened = False
        self._frame_was_fresh = False

        self.robot = RobotClient(
            log_callback=self._log_threadsafe,
            status_callback=self._on_status_change,
            ack_timeout=5.0,
            done_timeout=180.0,
            ready_retries=5,
        )

        self._build_ui()

        # Both cases of the shortcut key are bound: with Caps Lock on, or with Shift
        # held, Tk reports "S" and a lowercase-only binding would never fire.
        self.bind_all("<KeyPress-s>", self._on_detect_key)
        self.bind_all("<KeyPress-S>", self._on_detect_key)

    # ----------------------------------------------------------------- UI ---
    def _build_ui(self):
        hint = ttk.Label(
            self,
            text=(
                "หากไม่มีหุ่นยนต์จริงสำหรับทดสอบ ใช้โปรแกรมจำลอง TCP Server เช่น Hercules หรือ "
                "mock_robot_server.py ที่แนบมาให้ (ต้องตอบ 'ACK' หลังรับ 'Ready' และตอบ 'Done' หลังรับผลตรวจจับ)"
            ),
            style="Sub.TLabel", wraplength=1400,
        )
        hint.pack(side="bottom", anchor="w", padx=18, pady=(0, 8))

        top = ttk.Frame(self)
        top.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        left = ttk.LabelFrame(top, text="โมเดล และกล้อง Webcam (Real-time)")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self._build_vision_panel(left)

        right = ttk.LabelFrame(top, text="การสื่อสารกับแขนหุ่นยนต์ (TCP/IP Socket)", width=RIGHT_COLUMN_WIDTH)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)
        self._build_tcp_panel(right)

    def _build_vision_panel(self, left):
        # --- top anchored -------------------------------------------------
        model_row = ttk.Frame(left)
        model_row.pack(side="top", fill="x", padx=10, pady=(10, 6))
        ttk.Button(model_row, text="โหลดโมเดล (.pt)", style="Accent.TButton", command=self.load_model).pack(
            side="left"
        )
        self.model_label_var = tk.StringVar(value="ยังไม่ได้โหลดโมเดล")
        ttk.Label(model_row, textvariable=self.model_label_var, style="Sub.TLabel").pack(side="left", padx=10)

        cam_select_row = ttk.Frame(left)
        cam_select_row.pack(side="top", fill="x", padx=10, pady=(0, 6))
        ttk.Label(cam_select_row, text="เลือกกล้อง:").pack(side="left")
        self.camera_index_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(
            cam_select_row, textvariable=self.camera_index_var, state="readonly", width=6, values=["0"]
        )
        self.camera_combo.pack(side="left", padx=6)
        ttk.Button(cam_select_row, text="ค้นหากล้อง", command=self.refresh_cameras).pack(side="left", padx=4)

        # --- bottom anchored, packed in reverse visual order ---------------
        self.result_var = tk.StringVar(value="ผลการตรวจจับล่าสุด: -")
        result_card = tk.Frame(
            left, bg=ui_theme.CARD_BG, highlightthickness=1, highlightbackground=ui_theme.BORDER
        )
        result_card.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        tk.Label(
            result_card, textvariable=self.result_var, font=ui_theme.FONT_STATUS,
            bg=ui_theme.CARD_BG, fg=ui_theme.ACCENT_DARK, anchor="w", justify="left",
        ).pack(anchor="w", padx=10, pady=8)

        # Packed before the detect button so it ends up directly underneath it.
        self.reset_view_btn = ttk.Button(
            left, text="รีเซ็ตภาพ (กลับไปแสดงภาพสด)", command=self.reset_view, state="disabled"
        )
        self.reset_view_btn.pack(side="bottom", fill="x", padx=10, pady=(0, 8))

        self.detect_btn = ttk.Button(
            left, text="เริ่มการตรวจจับ (Detect: S)", style="Accent.TButton",
            command=self.on_detect_pressed, state="disabled"
        )
        self.detect_btn.pack(side="bottom", fill="x", padx=10, pady=(4, 4))

        conf_row = ttk.Frame(left)
        conf_row.pack(side="bottom", fill="x", padx=10, pady=(0, 4))
        ttk.Label(conf_row, text="Confidence threshold:").pack(side="left")
        self.conf_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(
            conf_row, from_=0.05, to=0.95, increment=0.05, textvariable=self.conf_var, width=6
        ).pack(side="left", padx=6)

        cam_btns = ttk.Frame(left)
        cam_btns.pack(side="bottom", fill="x", padx=10, pady=(0, 8))
        ttk.Button(cam_btns, text="เปิดกล้อง", style="Accent.TButton", command=self.start_preview).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ttk.Button(cam_btns, text="ปิดกล้อง", command=self.stop_preview).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )

        # --- fills whatever space is left ---------------------------------
        # Requested height is the preferred size; pack gives it that plus any spare
        # space, and shrinks it first when the window is too small, so the detect
        # button and the result card can never be pushed off screen.
        self.preview_frame = tk.Frame(left, bg=ui_theme.BORDER, height=320)
        self.preview_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 8))
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_frame, bg="#0c0f16")
        self.preview_label.pack(fill="both", expand=True)

    def _build_tcp_panel(self, right):
        conn_row = ttk.Frame(right)
        conn_row.pack(side="top", fill="x", padx=10, pady=(10, 4))
        ttk.Label(conn_row, text="IP Address:").grid(row=0, column=0, sticky="w")
        self.ip_var = tk.StringVar(value="192.168.1.6")
        ttk.Entry(conn_row, textvariable=self.ip_var, width=16).grid(row=0, column=1, padx=6)
        ttk.Label(conn_row, text="Port:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        # 9000 is the default port for the TCP server written in the robot program.
        self.port_var = tk.StringVar(value="9000")
        ttk.Entry(conn_row, textvariable=self.port_var, width=8).grid(row=0, column=3, padx=6)

        btn_row = ttk.Frame(right)
        btn_row.pack(side="top", fill="x", padx=10, pady=(0, 6))
        self.connect_btn = ttk.Button(
            btn_row, text="เชื่อมต่อ", style="Accent.TButton", command=self.connect_robot
        )
        self.connect_btn.pack(side="left")
        ttk.Button(btn_row, text="ตัดการเชื่อมต่อ", style="Danger.TButton", command=self.disconnect_robot).pack(
            side="left", padx=6
        )

        timeout_row = ttk.Frame(right)
        timeout_row.pack(side="top", fill="x", padx=10, pady=(0, 8))
        ttk.Label(timeout_row, text="รอ Done ไม่เกิน (วินาที):").pack(side="left")
        self.done_timeout_var = tk.IntVar(value=180)
        ttk.Spinbox(
            timeout_row, from_=10, to=900, increment=10, textvariable=self.done_timeout_var, width=6,
            command=self._apply_done_timeout,
        ).pack(side="left", padx=6)
        ttk.Label(
            timeout_row, text="ตั้งให้นานกว่าเวลาที่หุ่นจัดวางจริง", style="Sub.TLabel"
        ).pack(side="left", padx=4)

        status_card = tk.Frame(
            right, bg=ui_theme.CARD_BG, highlightthickness=1, highlightbackground=ui_theme.BORDER
        )
        status_card.pack(side="top", fill="x", padx=10, pady=(0, 6))
        self.status_dot = tk.Canvas(status_card, width=14, height=14, bg=ui_theme.CARD_BG, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(10, 6), pady=8)
        self._status_dot_id = self.status_dot.create_oval(2, 2, 12, 12, fill=ui_theme.NEUTRAL, outline="")
        self.status_var = tk.StringVar(value="สถานะ: ยังไม่ได้เชื่อมต่อ")
        self.status_label = tk.Label(
            status_card, textvariable=self.status_var, font=ui_theme.FONT_STATUS,
            bg=ui_theme.CARD_BG, fg=ui_theme.NEUTRAL, anchor="w", justify="left", wraplength=460,
        )
        self.status_label.pack(side="left", padx=(0, 10), pady=8)

        self.cancel_wait_btn = ttk.Button(
            right, text="ยกเลิกการรอ Done และตัดการเชื่อมต่อ", style="Warn.TButton",
            command=self.cancel_wait, state="disabled"
        )
        self.cancel_wait_btn.pack(side="top", fill="x", padx=10, pady=(0, 8))

        ttk.Label(right, text="ประวัติการรับ-ส่งข้อมูล (Communication Log):").pack(
            side="top", anchor="w", padx=10
        )
        self.log_text = tk.Text(right, height=10, wrap="word")
        ui_theme.style_log_text(self.log_text)
        self.log_text.pack(side="top", fill="both", expand=True, padx=10, pady=(4, 10))

    # -------------------------------------------------------------- model ---
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
            self._log(f"[SYS] Model loaded successfully: {path}")
            self._update_button_states()
        except Exception as e:
            messagebox.showerror("ผิดพลาด", f"โหลดโมเดลไม่สำเร็จ: {e}")

    # ------------------------------------------------------------- camera ---
    def refresh_cameras(self):
        self._log("[SYS] Scanning connected cameras... (may take 10-30 seconds)")
        self.update_idletasks()
        found = camera_utils.list_available_cameras()
        if not found:
            messagebox.showwarning(
                "แจ้งเตือน",
                "ไม่พบกล้องที่เชื่อมต่ออยู่\n\n"
                "ถ้ามีโปรแกรมอื่นเปิดกล้องค้างไว้ (รวมถึงแท็บ Training) จะสแกนไม่เจอ "
                "ให้กดปิดกล้องก่อนแล้วลองใหม่",
            )
            return
        values = [str(i) for i in found]
        self.camera_combo["values"] = values
        # Highest index first: that is usually the external USB camera, which the rules require.
        self.camera_index_var.set(values[-1])
        self._log(f"[SYS] Camera indexes found: {values} (auto-selected {values[-1]})")

    def _open_capture(self, idx):
        """Open a camera, preferring DirectShow.

        OpenCV picks the MSMF backend by default on Windows. On USB webcams it
        regularly stops delivering frames after a while and only reports
        "can't grab frame", which leaves the preview frozen. DirectShow is markedly
        more stable for long sessions, so it is tried first and MSMF is the fallback.
        """
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
        return cv2.VideoCapture(idx)

    def start_preview(self):
        if self.preview_running:
            return
        idx = int(self.camera_index_var.get())
        self.cap = self._open_capture(idx)
        if not self.cap.isOpened():
            messagebox.showerror(
                "ผิดพลาด", f"ไม่สามารถเปิดกล้อง (index {idx}) ได้ กรุณากด 'ค้นหากล้อง' แล้วเลือก index ใหม่"
            )
            return
        self._camera_index = idx
        self._detection_frame = None
        self._read_failures = 0
        self._camera_reopened = False
        self._last_frame_time = time.monotonic()
        self._frame_was_fresh = False
        self.preview_running = True
        self._update_preview()

    def stop_preview(self):
        self.preview_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.preview_label.configure(image="")
        self.preview_label.imgtk = None
        self._last_frame = None
        self._detection_frame = None
        self._read_failures = 0
        self._camera_reopened = False
        self._frame_was_fresh = False
        self._update_button_states()

    def _render_preview_frame(self, frame):
        """Render one OpenCV BGR frame inside the fixed preview area."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        size = ui_theme.fit_size(
            w, h, self.preview_frame.winfo_width(), self.preview_frame.winfo_height()
        )
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb).resize(size))
        self.preview_label.imgtk = imgtk
        self.preview_label.configure(image=imgtk)

    def _frame_is_fresh(self):
        """True when the newest captured frame is recent enough to act on."""
        return (
            self._last_frame is not None
            and (time.monotonic() - self._last_frame_time) <= CAMERA_STALE_SECONDS
        )

    def _handle_camera_stall(self):
        """The camera stopped delivering frames. Reopen it once, then give up loudly."""
        if self._camera_reopened:
            self._log("[ERR] Camera is still silent after reopening. Preview stopped.")
            self.stop_preview()
            messagebox.showerror(
                "กล้องหยุดส่งภาพ",
                "กล้องหยุดส่งภาพ และเปิดใหม่อัตโนมัติแล้วยังไม่กลับมา\n\n"
                "ลองถอดสาย USB แล้วเสียบใหม่ จากนั้นกด 'เปิดกล้อง' อีกครั้ง",
            )
            return

        self._camera_reopened = True
        self._read_failures = 0
        self._log("[SYS] Camera stopped delivering frames. Reopening it once...")
        if self.cap:
            self.cap.release()
        self.cap = self._open_capture(self._camera_index)
        if not self.cap.isOpened():
            self._log("[ERR] Reopening the camera failed. Preview stopped.")
            self.stop_preview()
            messagebox.showerror(
                "กล้องหยุดส่งภาพ",
                "เปิดกล้องใหม่ไม่สำเร็จ ลองถอดสาย USB แล้วเสียบใหม่ จากนั้นกด 'เปิดกล้อง' อีกครั้ง",
            )

    def _update_preview(self):
        if not self.preview_running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if ok:
            if self._read_failures:
                self._log(f"[SYS] Camera recovered after {self._read_failures} failed reads.")
            self._read_failures = 0
            self._camera_reopened = False
            self._last_frame = frame
            self._last_frame_time = time.monotonic()
            shown_frame = self._detection_frame if self._detection_frame is not None else frame
            self._render_preview_frame(shown_frame)
        else:
            self._read_failures += 1
            if self._read_failures >= CAMERA_RETRY_AFTER:
                self._handle_camera_stall()
                if not self.preview_running:
                    return

        # Only touch the buttons when the picture changes between fresh and stale,
        # so this is not re-evaluated 33 times a second for nothing.
        fresh = self._frame_is_fresh()
        if fresh != self._frame_was_fresh:
            self._frame_was_fresh = fresh
            if not fresh:
                self._log(
                    "[ERR] ภาพจากกล้องไม่สด -- ปิดปุ่มตรวจจับไว้ชั่วคราว "
                    "เพื่อไม่ให้ส่งผลจากภาพเก่าไปยังหุ่นยนต์"
                )
            self._update_button_states()

        self.after(30, self._update_preview)

    # ----------------------------------------------------------------- TCP ---
    def _apply_done_timeout(self):
        try:
            self.robot.done_timeout = float(self.done_timeout_var.get())
        except (ValueError, tk.TclError):
            pass

    def connect_robot(self):
        ip = self.ip_var.get().strip()
        port = self.port_var.get().strip()
        if not ip or not port:
            messagebox.showwarning("แจ้งเตือน", "กรุณากรอก IP Address และ Port")
            return

        self._apply_done_timeout()
        self.connect_btn.configure(state="disabled")

        def worker():
            ok, msg = self.robot.connect(ip, port)
            self.after(0, lambda: self._after_connect(ok, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _after_connect(self, ok, msg):
        self.connect_btn.configure(state="normal")
        if not ok:
            messagebox.showerror("เชื่อมต่อไม่สำเร็จ", msg)
        self._update_button_states()

    def disconnect_robot(self):
        self.robot.close()
        self._update_button_states()

    def cancel_wait(self):
        """Drop the connection so a late Done can never leak into the next round."""
        self.cancel_wait_btn.configure(state="disabled")
        self.robot.close()
        self._log("[SYS] Wait cancelled. Reconnect and complete Ready/ACK before the next cycle.")
        self._update_button_states()

    def _on_status_change(self, state):
        self.state = state
        text_map = {
            "connected": "เชื่อมต่อแล้ว (ยังไม่ได้รับ ACK)",
            "ready": "พร้อมทำงาน (Ready)",
            "waiting_done": "รอหุ่นยนต์ทำงานเสร็จ (Wait State - รอ 'Done')",
            "wait_timeout": "หมดเวลารอ 'Done' - ยังค้างสถานะรอ กดยกเลิกเพื่อเริ่มรอบใหม่",
            "disconnected": "ยังไม่ได้เชื่อมต่อ",
        }
        color = ui_theme.STATE_COLORS.get(state, ui_theme.NEUTRAL)

        def _apply():
            self.status_var.set("สถานะ: " + text_map.get(state, state))
            self.status_label.configure(fg=color)
            self.status_dot.itemconfig(self._status_dot_id, fill=color)
            self._update_button_states()

        self.after(0, _apply)

    def _update_button_states(self):
        can_detect = (
            self.model is not None and self._frame_is_fresh() and not self.inference_running
            and self.robot.connected and self.state == "ready"
        )
        self.detect_btn.configure(state="normal" if can_detect else "disabled")
        can_cancel = self.state in ("waiting_done", "wait_timeout")
        self.cancel_wait_btn.configure(state="normal" if can_cancel else "disabled")
        # Available whenever a picture is on screen, whether it is frozen or not.
        self.reset_view_btn.configure(state="normal" if self.preview_running else "disabled")

    def reset_view(self):
        """Manual reset: live preview back, and detection allowed again.

        This is a different route out of a cycle than receiving "Done". Done means the
        robot confirmed it finished; a reset means the operator decided to carry on
        without that confirmation. Both end up in the same ready state, so the log
        says explicitly which one happened.

        Anything still queued on the socket is discarded on the way, so a "Done" the
        robot already sent for the abandoned round cannot answer the next one.
        """
        self._detection_frame = None
        self._log("[RESET] Manual reset by operator (not a Done from the robot).")
        self._log("[RESET] Preview returned to the live camera feed.")

        if self.state in ("waiting_done", "wait_timeout"):
            self._log("[RESET] Abandoning the wait for Done and clearing the socket buffer...")
            self.robot.resume_ready(done_callback=lambda: self.after(0, self._after_reset))
        else:
            self._update_button_states()
            self._log("[RESET] Ready for the next detection.")

    def _after_reset(self):
        self._update_button_states()
        if str(self.detect_btn["state"]) == "normal":
            self._log("[RESET] Ready for the next detection.")
        else:
            self._log(
                "[RESET] Preview is live, but detection is still locked -- check the camera feed and the robot connection."
            )

    # -------------------------------------------------------------- detect ---
    def _on_detect_key(self, event):
        """Keyboard shortcut. bind_all sees every keystroke in the application, so
        typing an "s" into the IP field or the class name box must not start a
        detection - hence the check on which widget currently has focus."""
        if isinstance(event.widget, TEXT_INPUT_WIDGETS):
            return
        if str(self.detect_btn["state"]) == "normal":
            self.on_detect_pressed()

    def on_detect_pressed(self):
        if self.model is None:
            messagebox.showwarning("แจ้งเตือน", "กรุณาโหลดโมเดลก่อน")
            return
        if self._last_frame is None:
            messagebox.showwarning("แจ้งเตือน", "กรุณาเปิดกล้องก่อน")
            return
        # Last line of defence: never send a result taken from a picture the camera
        # stopped refreshing, or the robot would act on what was in front of it before.
        if not self._frame_is_fresh():
            messagebox.showwarning(
                "แจ้งเตือน",
                "ภาพจากกล้องไม่สด กล้องอาจหยุดส่งภาพ\n\nกรุณากด 'ปิดกล้อง' แล้ว 'เปิดกล้อง' ใหม่ก่อนตรวจจับ",
            )
            return
        if not self.robot.connected or self.state != "ready":
            messagebox.showwarning("แจ้งเตือน", "ยังไม่พร้อมสื่อสารกับหุ่นยนต์ (ต้องเชื่อมต่อและได้รับ ACK ก่อน)")
            return

        frame = self._last_frame.copy()
        conf = float(self.conf_var.get())
        # Return to the live feed while processing.  The annotated snapshot replaces
        # it as soon as inference finishes.
        self._detection_frame = None
        self._render_preview_frame(frame)
        self.inference_running = True
        self._update_button_states()
        self._log("[SYS] Running inference...")
        threading.Thread(target=self._infer_worker, args=(frame, conf), daemon=True).start()

    def _infer_worker(self, frame, conf):
        try:
            results = self.model.predict(source=frame, conf=conf, verbose=False)
            names = results[0].names
            class_ids = []
            if results[0].boxes is not None:
                class_ids = results[0].boxes.cls.tolist()
            result_string, readable = format_detection_result(names, class_ids)
            # Ultralytics plot() draws boxes, class names and confidence scores on
            # the exact snapshot used for inference and returns an OpenCV BGR image.
            annotated_frame = results[0].plot(labels=True, conf=True)
            self.after(
                0,
                lambda: self._finish_inference(
                    result_string, readable, annotated_frame, None
                ),
            )
        except Exception as exc:
            self.after(0, lambda exc=exc: self._finish_inference(None, None, None, str(exc)))

    def _finish_inference(self, result_string, readable, annotated_frame, error):
        self.inference_running = False
        if error:
            self._log(f"[ERR] Inference failed: {error}")
            messagebox.showerror("Inference ไม่สำเร็จ", error)
            self._update_button_states()
            return

        self._detection_frame = annotated_frame
        self._render_preview_frame(annotated_frame)
        self.result_var.set(f"ผลการตรวจจับล่าสุด: {readable}   |   ส่ง: {result_string}")
        self._log(f"[SYS] Detection result: {result_string}")
        if not self.robot.connected or self.state != "ready":
            self._log("[ERR] Connection state changed during inference; result was not sent.")
            messagebox.showwarning("ไม่ได้ส่งผล", "การเชื่อมต่อไม่พร้อม กรุณาเชื่อมต่อใหม่แล้วตรวจจับอีกครั้ง")
            self._update_button_states()
            return
        self.robot.send_detection_result(result_string, wait_for_done_callback=self._after_done)

    def _after_done(self, success, message):
        def _apply():
            self._log(f"[SYS] {message}")
            if success:
                # Robot finished the round: drop the frozen annotated snapshot so the
                # preview returns to the live feed, ready for the next detection.
                self._detection_frame = None
            self._update_button_states()

        self.after(0, _apply)

    # ----------------------------------------------------------------- log ---
    def _log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _log_threadsafe(self, msg):
        self.after(0, lambda: self._log(msg))
