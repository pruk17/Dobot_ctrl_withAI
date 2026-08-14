import tkinter as tk
from tkinter import ttk

import ui_theme
from train_tab import TrainTab
from infer_tab import InferTab


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dobot MG400 · AI Vision Program — BRICS Skills Competition Thailand 2026")
        self.geometry("1440x900")
        self.minsize(1200, 760)

        ui_theme.apply_theme(self)

        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(14, 4))
        ttk.Label(
            header, text="🤖  Dobot MG400 · AI Vision Program", style="Header.TLabel"
        ).pack(side="left")
        ttk.Label(
            header, text="BRICS Skills Competition Thailand 2026", style="Sub.TLabel"
        ).pack(side="left", padx=(10, 0), pady=(4, 0))

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=14, pady=(8, 14))

        self.train_tab = TrainTab(notebook)
        self.infer_tab = InferTab(notebook)

        notebook.add(self.train_tab, text="  🏋  โหมดที่ 1 · Training  ")
        notebook.add(self.infer_tab, text="  🎯  โหมดที่ 2 · Deployment & Inference  ")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        try:
            self.train_tab.stop_preview()
        except Exception:
            pass
        try:
            self.infer_tab.stop_preview()
            self.infer_tab.robot.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
