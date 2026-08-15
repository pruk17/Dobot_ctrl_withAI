# -*- coding: utf-8 -*-
"""
augment_dialog.py
-----------------
Settings window for data augmentation, opened from the training tab.

It is a separate Toplevel rather than another panel in the main window because the
explanations next to every field need room, and the main window is already tight on
a 1080 pixel high screen.
"""

import tkinter as tk
from tkinter import ttk

import augment_utils as au
import dataset_utils as du
import ui_theme


class AugmentDialog(tk.Toplevel):
    def __init__(self, master, on_saved=None):
        super().__init__(master)
        self.title("ตั้งค่าการเพิ่มข้อมูล (Data Augmentation)")
        self.configure(bg=ui_theme.BG)
        self.on_saved = on_saved
        self.cfg = au.load_config()

        self.flip_vars = {}
        self.online_vars = {}
        self.gray_var = tk.BooleanVar(value=self.cfg.get("offline_gray", False))
        self.blur_var = tk.BooleanVar(value=self.cfg.get("offline_blur", False))
        self.blur_strength_var = tk.IntVar(value=self.cfg.get("blur_strength", 3))

        self._build_ui()

        self.transient(master)
        self.grab_set()
        self.resizable(True, True)
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), min(self.winfo_reqheight(), self.winfo_screenheight() - 80)
        x = master.winfo_rootx() + 60
        y = max(0, master.winfo_rooty() + 20)
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ----------------------------------------------------------------- UI ---
    def _build_ui(self):
        intro = ttk.Label(
            self,
            text=(
                "การเพิ่มข้อมูล (augmentation) คือการสร้างภาพหลากหลายรูปแบบจากภาพที่ถ่ายไว้ "
                "เพื่อให้โมเดลทนต่อการวางมุมต่างกัน แสงต่างกัน และสีต่างกัน โดยไม่ต้องถ่ายภาพเพิ่มเอง"
            ),
            style="Sub.TLabel", wraplength=740,
        )
        intro.pack(anchor="w", padx=16, pady=(14, 8))

        self._build_flip_section()
        self._build_offline_section()
        self._build_online_section()

        btn_row = ttk.Frame(self)
        btn_row.pack(side="bottom", fill="x", padx=16, pady=(6, 14))
        ttk.Button(
            btn_row, text="คืนค่าที่แนะนำ", command=self.reset_defaults
        ).pack(side="left")
        ttk.Button(btn_row, text="บันทึก", style="Accent.TButton", command=self.save).pack(side="right")
        ttk.Button(btn_row, text="ยกเลิก", command=self.destroy).pack(side="right", padx=6)

    def _build_flip_section(self):
        box = ttk.LabelFrame(self, text="1. คลาสไหนพลิกซ้ายขวาได้บ้าง")
        box.pack(fill="x", padx=16, pady=(0, 10))

        ttk.Label(
            box,
            text=(
                "ติ๊กไว้ = อนุญาตให้พลิกภาพของคลาสนั้นเป็นภาพกระจก "
                "ค่าเริ่มต้นจะไม่ติ๊กเพื่อความปลอดภัย ให้ติ๊กเฉพาะเมื่อเห็นสัญลักษณ์จริงแล้วมั่นใจ "
                "ว่าภาพสะท้อนไม่เปลี่ยนความหมาย เช่น อย่าติ๊กเมื่อมีคลาสที่เป็นลูกศรหรือตัวอักษร "
                "เช่น ถ้ามีทั้ง arrow_left และ arrow_right หรือมีทั้งตัว b และตัว d "
                "ส่วนรูปทรงสมมาตรอย่างวงกลม สี่เหลี่ยม ดาว ติ๊กไว้ได้ตามปกติ"
            ),
            style="Sub.TLabel", wraplength=720,
        ).pack(anchor="w", padx=12, pady=(6, 6))

        classes = du.load_classes()
        if not classes:
            ttk.Label(box, text="ยังไม่มีคลาสในชุดข้อมูล", style="Sub.TLabel").pack(
                anchor="w", padx=12, pady=(0, 10)
            )
            return

        grid = ttk.Frame(box)
        grid.pack(fill="x", padx=12, pady=(0, 10))
        for i, name in enumerate(classes):
            var = tk.BooleanVar(value=self.cfg["flip_allowed"].get(name, False))
            self.flip_vars[name] = var
            cb = ttk.Checkbutton(grid, text=name, variable=var, command=self._update_flip_hint)
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 24), pady=2)

        self.flip_hint_var = tk.StringVar()
        ttk.Label(box, textvariable=self.flip_hint_var, style="Sub.TLabel", wraplength=720).pack(
            anchor="w", padx=12, pady=(0, 10)
        )
        self._update_flip_hint()

    def _update_flip_hint(self):
        blocked = [name for name, var in self.flip_vars.items() if not var.get()]
        if not blocked:
            text = (
                "ทุกคลาสอนุญาต จะใช้การพลิกของ YOLO ซึ่งสุ่มใหม่ทุก epoch "
                "ไม่สร้างไฟล์เพิ่มและได้ความหลากหลายมากกว่า"
            )
        elif len(blocked) == len(self.flip_vars):
            text = "ไม่มีคลาสใดอนุญาต จะไม่พลิกภาพเลย"
        else:
            text = (
                "มีคลาสที่ห้ามพลิก (" + ", ".join(blocked) + ") "
                "จะปิดการพลิกของ YOLO แล้วสร้างไฟล์ภาพพลิกไว้ล่วงหน้าเฉพาะภาพที่ไม่มีคลาสเหล่านี้"
            )
        self.flip_hint_var.set(text)

    def _build_offline_section(self):
        box = ttk.LabelFrame(self, text="2. สร้างสำเนาภาพเพิ่มลงดิสก์")
        box.pack(fill="x", padx=16, pady=(0, 10))

        ttk.Label(
            box,
            text=(
                "สองอย่างนี้ YOLO ไม่มีให้ใช้ในตัว จึงต้องสร้างเป็นไฟล์ภาพเพิ่มไว้ก่อนเทรน "
                "ไฟล์ที่สร้างจะมีคำว่า __aug อยู่ในชื่อ ถูกลบและสร้างใหม่ทุกครั้งที่กดเทรน "
                "และไม่แสดงในหน้าติดป้ายกำกับ"
            ),
            style="Sub.TLabel", wraplength=720,
        ).pack(anchor="w", padx=12, pady=(6, 6))

        ttk.Checkbutton(
            box, text="สร้างสำเนาภาพขาวดำ", variable=self.gray_var
        ).pack(anchor="w", padx=12)
        ttk.Label(
            box,
            text=(
                "บังคับให้โมเดลตัดสินจากรูปทรงแทนสี ใช้เมื่อสัญลักษณ์อาจมาในหลายสี "
                "ห้ามเปิดถ้าคลาสต่างกันแยกด้วยสีเป็นหลัก เพราะจะลบข้อมูลที่ใช้แยกคลาสทิ้ง"
            ),
            style="Sub.TLabel", wraplength=700,
        ).pack(anchor="w", padx=34, pady=(0, 8))

        blur_row = ttk.Frame(box)
        blur_row.pack(fill="x", padx=12)
        ttk.Checkbutton(blur_row, text="สร้างสำเนาภาพเบลอ", variable=self.blur_var).pack(side="left")
        ttk.Label(blur_row, text="ระดับความเบลอ:").pack(side="left", padx=(20, 4))
        ttk.Combobox(
            blur_row, textvariable=self.blur_strength_var, state="readonly", width=4,
            values=[3, 5, 7, 9],
        ).pack(side="left")
        ttk.Label(blur_row, text="(3 = เบาสุด)", style="Sub.TLabel").pack(side="left", padx=6)

        ttk.Label(
            box,
            text=(
                "จำลองกล้อง USB ที่ไล่โฟกัสไม่ทัน หรือมือสั่นตอนวางชิ้นงาน "
                "แนะนำ 3 ถึง 5 ถ้าเบลอแรงเกินไปโมเดลจะเห็นเป็นก้อนเบลอ ๆ แล้วตรวจจับผิดพลาดง่ายขึ้น"
            ),
            style="Sub.TLabel", wraplength=700,
        ).pack(anchor="w", padx=34, pady=(2, 10))

    def _build_online_section(self):
        box = ttk.LabelFrame(self, text="3. การสุ่มแปลงภาพระหว่างเทรน (YOLO ทำให้ทุก epoch)")
        box.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        ttk.Label(
            box,
            text="ค่าเหล่านี้ไม่สร้างไฟล์เพิ่ม แต่สุ่มใหม่ทุกรอบการเทรน ตั้ง 0 เพื่อปิดการแปลงนั้น",
            style="Sub.TLabel", wraplength=720,
        ).pack(anchor="w", padx=12, pady=(6, 6))

        grid = ttk.Frame(box)
        grid.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        grid.columnconfigure(3, weight=1)

        for row, (key, label, unit, lo, hi, step, hint) in enumerate(au.ONLINE_FIELDS):
            var = tk.DoubleVar(value=float(self.cfg["online"].get(key, 0.0)))
            self.online_vars[key] = var
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Spinbox(
                grid, from_=lo, to=hi, increment=step, textvariable=var, width=7
            ).grid(row=row, column=1, sticky="w", padx=8)
            ttk.Label(grid, text=unit, style="Sub.TLabel").grid(row=row, column=2, sticky="w")
            ttk.Label(grid, text=hint, style="Sub.TLabel", wraplength=430).grid(
                row=row, column=3, sticky="w", padx=(12, 0)
            )

    # -------------------------------------------------------------- actions ---
    def reset_defaults(self):
        for key, var in self.online_vars.items():
            var.set(au.DEFAULT_CONFIG["online"][key])
        for var in self.flip_vars.values():
            var.set(False)
        self.gray_var.set(au.DEFAULT_CONFIG["offline_gray"])
        self.blur_var.set(au.DEFAULT_CONFIG["offline_blur"])
        self.blur_strength_var.set(au.DEFAULT_CONFIG["blur_strength"])
        if self.flip_vars:
            self._update_flip_hint()

    def save(self):
        cfg = au.load_config()
        for name, var in self.flip_vars.items():
            cfg["flip_allowed"][name] = bool(var.get())
        cfg["offline_gray"] = bool(self.gray_var.get())
        cfg["offline_blur"] = bool(self.blur_var.get())
        cfg["blur_strength"] = int(self.blur_strength_var.get())
        for key, var in self.online_vars.items():
            try:
                cfg["online"][key] = round(float(var.get()), 4)
            except (ValueError, tk.TclError):
                pass
        au.save_config(cfg)
        if self.on_saved:
            self.on_saved(cfg)
        self.destroy()
