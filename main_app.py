# -*- coding: utf-8 -*-
"""
main_app.py
-----------
Entry point. Builds the main window and the two mode tabs.

Run with:  python main_app.py
"""

import ctypes
import sys
import tkinter as tk
from tkinter import ttk

import ui_theme
from train_tab import TrainTab
from infer_tab import InferTab


def enable_dpi_awareness():
    """Tell Windows this process draws at the real pixel resolution.

    Without this, a display scaled to 125% makes Tk believe a 1920x1080 screen is only
    1536x864. The layout is then drawn into that smaller area and stretched back up by
    Windows, which costs a quarter of the vertical space and blurs the text. Must be
    called before the first Tk window exists.
    """
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system DPI aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class MainApp(tk.Tk):
    def __init__(self):
        enable_dpi_awareness()
        super().__init__()
        self.title("Dobot MG400 · AI Vision Program — BRICS Skills Competition Thailand 2026")

        # Size the window to the screen instead of hard-coding 1440x900, so the layout
        # still fits on a smaller display or a projector.
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(1500, screen_w - 80)
        win_h = min(1040, screen_h - 70)
        self.geometry(f"{win_w}x{win_h}+{max(0, (screen_w - win_w) // 2)}+20")
        self.minsize(1000, 640)

        ui_theme.apply_theme(self)

        header = ttk.Frame(self)
        header.pack(fill="x", padx=14, pady=(10, 2))
        ttk.Label(
            header, text="Dobot MG400 · AI Vision Program", style="Header.TLabel"
        ).pack(side="left")
        ttk.Label(
            header, text="BRICS Skills Competition Thailand 2026", style="Sub.TLabel"
        ).pack(side="left", padx=(10, 0), pady=(3, 0))

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        self.train_tab = TrainTab(notebook)
        self.infer_tab = InferTab(notebook)

        notebook.add(self.train_tab, text="  โหมดที่ 1 · Training  ")
        notebook.add(self.infer_tab, text="  โหมดที่ 2 · Deployment & Inference  ")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        for closer in (
            lambda: self.train_tab.stop_preview(),
            lambda: self.infer_tab.stop_preview(),
            lambda: self.infer_tab.robot.close(),
        ):
            try:
                closer()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
