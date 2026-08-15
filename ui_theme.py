# -*- coding: utf-8 -*-
"""
ui_theme.py
-----------
Central colour palette, fonts and ttk styling for the whole application.
Call apply_theme(root) once, right after the main window is created, and every
ttk widget picks up the theme automatically.

Font sizes are given in PIXELS (negative numbers), not points. That matters
because the application enables Windows DPI awareness: with point sizes Tk would
scale every font by the monitor DPI factor and cancel out the extra screen space
that DPI awareness buys us. Pixel sizes render identically on every machine.
"""

import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------- palette ---
BG = "#f4f6fb"           # window background
CARD_BG = "#ffffff"      # card / panel background
TEXT = "#1f2430"         # primary text
SUBTEXT = "#6b7280"      # secondary text
BORDER = "#e2e5ec"       # hairline borders

ACCENT = "#3d6bff"       # primary action colour
ACCENT_DARK = "#2c53d6"
ACCENT_TEXT = "#ffffff"

SUCCESS = "#1ba672"      # green: ready
WARNING = "#e08e0b"      # amber: connected / waiting
DANGER = "#e0483d"       # red: disconnected / timeout / error
NEUTRAL = "#8a8f9c"      # grey: idle

LOG_BG = "#12161f"
LOG_FG = "#4ade80"

FONT_FAMILY = "Segoe UI"
FONT_NORMAL = (FONT_FAMILY, -13)
FONT_BOLD = (FONT_FAMILY, -13, "bold")
FONT_SMALL = (FONT_FAMILY, -12)
FONT_HEADER = (FONT_FAMILY, -14, "bold")
FONT_TITLE = (FONT_FAMILY, -17, "bold")
FONT_STATUS = (FONT_FAMILY, -14, "bold")
FONT_TAB = (FONT_FAMILY, -14, "bold")
FONT_MONO = ("Consolas", -13)

# Connection states reported by RobotClient, mapped to the status indicator colour.
STATE_COLORS = {
    "idle": NEUTRAL,
    "disconnected": DANGER,
    "connected": WARNING,
    "waiting_done": WARNING,
    "wait_timeout": DANGER,
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
    style.configure("Sub.TLabel", background=BG, foreground=SUBTEXT, font=FONT_SMALL)
    style.configure("Header.TLabel", background=BG, foreground=TEXT, font=FONT_TITLE)

    style.configure(
        "TLabelframe", background=BG, foreground=TEXT, bordercolor=BORDER,
        relief="solid", borderwidth=1
    )
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT_DARK, font=FONT_HEADER)

    style.configure("TSeparator", background=BORDER)

    # ------------------------------------------------------------- buttons ---
    style.configure(
        "TButton", font=FONT_NORMAL, padding=(10, 6), background="#e9edf7",
        foreground=TEXT, borderwidth=0, focuscolor=BG
    )
    style.map(
        "TButton",
        background=[("active", "#dbe1f2"), ("disabled", "#eef0f5")],
        foreground=[("disabled", SUBTEXT)],
    )

    style.configure(
        "Accent.TButton", font=FONT_BOLD, padding=(12, 7), background=ACCENT,
        foreground=ACCENT_TEXT, borderwidth=0, focuscolor=BG
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_DARK), ("disabled", "#a9b8ef")],
        foreground=[("disabled", "#f0f2fb")],
    )

    style.configure(
        "Danger.TButton", font=FONT_NORMAL, padding=(10, 6), background="#fbe4e1",
        foreground=DANGER, borderwidth=0, focuscolor=BG
    )
    style.map("Danger.TButton", background=[("active", "#f6cfc9")])

    style.configure(
        "Warn.TButton", font=FONT_BOLD, padding=(10, 6), background="#fbeed6",
        foreground="#8a5a06", borderwidth=0, focuscolor=BG
    )
    style.map("Warn.TButton", background=[("active", "#f5e2bb"), ("disabled", "#f2f0ec")])

    # -------------------------------------------------------------- inputs ---
    style.configure(
        "TEntry", fieldbackground=CARD_BG, foreground=TEXT, bordercolor=BORDER,
        lightcolor=BORDER, darkcolor=BORDER, padding=5
    )
    style.configure(
        "TCombobox", fieldbackground=CARD_BG, foreground=TEXT, bordercolor=BORDER,
        arrowsize=14, padding=5
    )
    # A readonly combobox draws its text with the selection colours once it has focus.
    # Without these two lines the clam theme paints white on white and the value looks
    # blank the moment the widget is clicked.
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", CARD_BG)],
        selectbackground=[("readonly", CARD_BG)],
        selectforeground=[("readonly", TEXT)],
    )
    style.configure("TSpinbox", fieldbackground=CARD_BG, foreground=TEXT, arrowsize=12, padding=4)

    # ------------------------------------------------------------ notebook ---
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(8, 6, 8, 0))
    style.configure(
        "TNotebook.Tab", font=FONT_TAB, padding=(16, 8),
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
    """Give a plain tk.Text widget the dark console look used for the log panes."""
    text_widget.configure(
        bg=LOG_BG, fg=LOG_FG, insertbackground=LOG_FG, font=FONT_MONO,
        relief="flat", padx=10, pady=8, borderwidth=0, highlightthickness=1,
        highlightbackground=BORDER, highlightcolor=ACCENT,
    )


def fit_size(src_w, src_h, box_w, box_h, fallback=(480, 360)):
    """Scale (src_w, src_h) to fit inside (box_w, box_h) while keeping the aspect ratio.

    Tk reports a width/height of 1 for widgets that have not been laid out yet, so
    anything that small falls back to a sane default instead of producing a 1px image.
    """
    if box_w <= 1 or box_h <= 1:
        box_w, box_h = fallback
    if src_w <= 0 or src_h <= 0:
        return fallback
    scale = min(box_w / src_w, box_h / src_h)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))
