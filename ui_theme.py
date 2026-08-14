# -*- coding: utf-8 -*-
"""
ui_theme.py
------------
ธีมสี/ฟอนต์กลางของโปรแกรม ใช้ ttk.Style เพื่อให้หน้าตาโปรแกรมดูทันสมัย สะอาดตา
เรียก apply_theme(root) หนึ่งครั้งตอนสร้างหน้าต่างหลัก แล้วทุก widget ใน ttk จะใช้ธีมนี้โดยอัตโนมัติ
"""

import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------- palette ---
BG = "#f4f6fb"          # พื้นหลังหลัก
CARD_BG = "#ffffff"      # พื้นหลังการ์ด/กรอบ
TEXT = "#1f2430"         # สีตัวอักษรหลัก
SUBTEXT = "#6b7280"       # สีตัวอักษรรอง
BORDER = "#e2e5ec"        # สีเส้นขอบ

ACCENT = "#3d6bff"        # สีหลัก (น้ำเงิน) สำหรับปุ่ม action หลัก
ACCENT_DARK = "#2c53d6"
ACCENT_TEXT = "#ffffff"

SUCCESS = "#1ba672"       # เขียว: พร้อมทำงาน / Ready
WARNING = "#e08e0b"       # ส้ม: กำลังรอ / เชื่อมต่ออยู่
DANGER = "#e0483d"        # แดง: ตัดการเชื่อมต่อ / error
NEUTRAL = "#8a8f9c"        # เทา: ยังไม่เริ่ม/idle

LOG_BG = "#12161f"
LOG_FG = "#4ade80"

FONT_FAMILY = "Segoe UI"
FONT_NORMAL = (FONT_FAMILY, 10)
FONT_BOLD = (FONT_FAMILY, 10, "bold")
FONT_HEADER = (FONT_FAMILY, 11, "bold")
FONT_STATUS = (FONT_FAMILY, 11, "bold")
FONT_MONO = ("Consolas", 10)

STATE_COLORS = {
    "idle": NEUTRAL,
    "disconnected": DANGER,
    "connected": WARNING,
    "waiting_done": WARNING,
    "ready": SUCCESS,
}


def apply_theme(root):
    root.configure(bg=BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=BG, foreground=TEXT, font=FONT_NORMAL)

    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD_BG)

    style.configure("TLabel", background=BG, foreground=TEXT, font=FONT_NORMAL)
    style.configure("Sub.TLabel", background=BG, foreground=SUBTEXT, font=(FONT_FAMILY, 9))
    style.configure("Header.TLabel", background=BG, foreground=TEXT, font=(FONT_FAMILY, 13, "bold"))

    style.configure(
        "TLabelframe", background=BG, foreground=TEXT, bordercolor=BORDER,
        relief="solid", borderwidth=1
    )
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT_DARK, font=FONT_HEADER)

    style.configure("TSeparator", background=BORDER)

    # ------------------------------------------------------------- buttons ---
    style.configure(
        "TButton", font=FONT_NORMAL, padding=(10, 7), background="#e9edf7",
        foreground=TEXT, borderwidth=0, focuscolor=BG
    )
    style.map(
        "TButton",
        background=[("active", "#dbe1f2"), ("disabled", "#eef0f5")],
        foreground=[("disabled", SUBTEXT)],
    )

    style.configure(
        "Accent.TButton", font=FONT_BOLD, padding=(12, 8), background=ACCENT,
        foreground=ACCENT_TEXT, borderwidth=0, focuscolor=BG
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_DARK), ("disabled", "#a9b8ef")],
        foreground=[("disabled", "#f0f2fb")],
    )

    style.configure(
        "Danger.TButton", font=FONT_NORMAL, padding=(10, 7), background="#fbe4e1",
        foreground=DANGER, borderwidth=0, focuscolor=BG
    )
    style.map("Danger.TButton", background=[("active", "#f6cfc9")])

    # ------------------------------------------------------------- inputs ---
    style.configure(
        "TEntry", fieldbackground=CARD_BG, foreground=TEXT, bordercolor=BORDER,
        lightcolor=BORDER, darkcolor=BORDER, padding=6
    )
    style.configure(
        "TCombobox", fieldbackground=CARD_BG, foreground=TEXT, bordercolor=BORDER,
        arrowsize=14, padding=6
    )
    style.map("TCombobox", fieldbackground=[("readonly", CARD_BG)])
    style.configure("TSpinbox", fieldbackground=CARD_BG, foreground=TEXT, arrowsize=12, padding=4)

    # ------------------------------------------------------------ notebook ---
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(8, 8, 8, 0))
    style.configure(
        "TNotebook.Tab", font=(FONT_FAMILY, 11, "bold"), padding=(18, 10),
        background="#e9edf7", foreground=SUBTEXT, borderwidth=0
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", ACCENT)],
        foreground=[("selected", ACCENT_TEXT)],
        expand=[("selected", (0, 0, 0, 0))],
    )

    return style


def style_log_text(text_widget):
    """จัดสไตล์ให้กล่อง Text ที่ใช้แสดง log ให้ดูทันสมัยขึ้น"""
    text_widget.configure(
        bg=LOG_BG, fg=LOG_FG, insertbackground=LOG_FG, font=FONT_MONO,
        relief="flat", padx=10, pady=8, borderwidth=0, highlightthickness=1,
        highlightbackground=BORDER, highlightcolor=ACCENT,
    )
