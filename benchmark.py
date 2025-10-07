import os
import sys
import json
import threading
import subprocess
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Optional
import pandas as pd

APP_TITLE = "Holovibes Benchmark GUI"
BENCH_DIR = "benchmark"
HOLO_EXE = os.path.join("build", "bin", "Holovibes.exe")
SETTINGS_FILE = "settings.json"  # for custom nsight path


class HolovibesGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x660")
        self.minsize(900, 560)

        # State
        self.input_file = tk.StringVar(value="")
        self.nrows_preview = tk.IntVar(value=10)
        self.csv_dir = tk.StringVar(value=BENCH_DIR)
        self.is_running = False

        # nsys path
        self.nsys_path: Optional[str] = None

        # load parameters from settings file
        self.settings = self._load_settings()

        self.create_widgets()
        self.refresh_checks()
        self.refresh_csv_list()

    # ----------------------- UI -----------------------
    def create_widgets(self):
        # Top frame: Input selection + actions
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        # Input file chooser
        ttk.Label(top, text="Input file (.holo) :").grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(top, textvariable=self.input_file, width=60)
        self.input_entry.grid(row=0, column=1, padx=6, sticky="we")
        btn_browse = ttk.Button(top, text="Browse…", command=self.browse_input)
        btn_browse.grid(row=0, column=2, padx=4)

        # Run benchmark
        self.btn_run = ttk.Button(
            top, text="Run Benchmark", command=self.on_run_benchmark
        )
        self.btn_run.grid(row=0, column=3, padx=8)

        # Clean
        self.btn_clean = ttk.Button(top, text="Clean Reports", command=self.on_clean)
        self.btn_clean.grid(row=0, column=4, padx=4)

        # Checks row
        checks = ttk.Frame(top)
        checks.grid(row=1, column=0, columnspan=5, pady=(8, 0), sticky="we")
        self.lbl_exe = ttk.Label(checks, text="")
        self.lbl_exe.pack(side="left", padx=(0, 16))

        nsys_box = ttk.Frame(checks)
        nsys_box.pack(side="left")
        self.lbl_nsys = ttk.Label(nsys_box, text="")
        self.lbl_nsys.pack(side="left")
        self.btn_set_nsys = ttk.Button(
            nsys_box, text="Define nsys…", command=self.choose_nsys_path
        )
        self.btn_set_nsys.pack(side="left", padx=(8, 0))

        # Middle: Notebook with Preview & Logs
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # Preview tab
        tab_prev = ttk.Frame(nb)
        nb.add(tab_prev, text="Reports (CSV)")

        # CSV directory label + refresh
        dir_frame = ttk.Frame(tab_prev, padding=(10, 10, 10, 0))
        dir_frame.pack(fill="x")
        ttk.Label(dir_frame, text="Reports folder:").pack(side="left")
        self.entry_dir = ttk.Entry(dir_frame, textvariable=self.csv_dir, width=40)
        self.entry_dir.pack(side="left", padx=6)
        ttk.Button(dir_frame, text="Change…", command=self.change_csv_dir).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(dir_frame, text="Refresh", command=self.refresh_csv_list).pack(
            side="left"
        )

        # Split: list of CSV left, preview right
        split = ttk.Panedwindow(tab_prev, orient="horizontal")
        split.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: CSV list
        left = ttk.Frame(split)
        split.add(left, weight=1)

        self.csv_list = tk.Listbox(left, height=18)
        self.csv_list.pack(fill="both", expand=True, side="left")
        self.csv_list.bind("<<ListboxSelect>>", self.on_csv_select)
        scroll_left = ttk.Scrollbar(
            left, orient="vertical", command=self.csv_list.yview
        )
        self.csv_list.config(yscrollcommand=scroll_left.set)
        scroll_left.pack(side="right", fill="y")

        # Right: preview + options
        right = ttk.Frame(split)
        split.add(right, weight=3)

        opt_frame = ttk.Frame(right)
        opt_frame.pack(fill="x")
        ttk.Label(opt_frame, text="Lines to preview:").pack(side="left")
        spin = ttk.Spinbox(
            opt_frame,
            from_=1,
            to=1000,
            textvariable=self.nrows_preview,
            width=6,
            command=self.preview_selected_csv,
        )
        spin.pack(side="left", padx=6)
        ttk.Button(
            opt_frame, text="Open in Excel", command=self.open_csv_external
        ).pack(side="left", padx=(8, 0))

        self.preview_text = ScrolledText(right, height=20, wrap="none")
        self.preview_text.pack(fill="both", expand=True, pady=(6, 0))

        # Logs tab
        tab_logs = ttk.Frame(nb)
        nb.add(tab_logs, text="Console / Logs")
        self.log_text = ScrolledText(tab_logs, height=20, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Bottom: help tip
        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        tip = (
            "Tip: Click 'Run Benchmark' to execute nsys + Holovibes, "
            "then check the generated CSV files in the Reports tab."
        )
        ttk.Label(bottom, text=tip).pack(side="left")

        # Configure grid weights
        top.grid_columnconfigure(1, weight=1)

    # ----------------------- Callbacks -----------------------
    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Choose a .holo file",
            filetypes=[("HOLO Files", "*.holo"), ("All Files", "*.*")],
        )
        if path:
            self.input_file.set(path)
            self.refresh_checks()

    def change_csv_dir(self):
        path = filedialog.askdirectory(title="Choose CSV Reports Folder")
        if path:
            self.csv_dir.set(path)
            self.refresh_csv_list()

    def on_run_benchmark(self):
        if self.is_running:
            return

        # Re-evaluate checks right before running
        self.refresh_checks()

        # Basic validations
        if not self._has_holovibes():
            messagebox.showerror(
                "error",
                "Holovibes.exe not found. Please compile it first in build/bin/.",
            )
            return
        if not self._has_nsys():
            # Propose to set nsys now
            if messagebox.askyesno(
                "Nsight Systems not found",
                "NVIDIA Nsight Systems (nsys) is not found in the PATH.\n"
                "Do you want to manually select the executable now?",
            ):
                if not self.choose_nsys_path():
                    return  # User canceled or invalid path
            else:
                return

        in_file = self.input_file.get().strip()
        if in_file and not (os.path.exists(in_file) and in_file.endswith(".holo")):
            messagebox.showerror(
                "error", "Input file must exist and have .holo extension."
            )
            return

        # Disable buttons while running
        self.set_running(True)
        self.log("=== Starting benchmark ===")
        self.log(f"Input: {in_file if in_file else '(none)'}")
        self.log(f"PATH={os.environ.get('PATH', '')}")
        self.log(f"nsys resolved : {self.nsys_path}")

        t = threading.Thread(
            target=self._run_benchmark_thread, args=(in_file,), daemon=True
        )
        t.start()

    def on_clean(self):
        try:
            if os.path.exists(BENCH_DIR):
                shutil.rmtree(BENCH_DIR)
                self.log("Benchmark folder deleted.")
            else:
                self.log("No benchmark folder to delete.")
            self.refresh_csv_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clean up: {e}")

    def on_csv_select(self, event=None):
        self.preview_selected_csv()

    def open_csv_external(self):
        sel = self._selected_csv_path()
        if not sel:
            messagebox.showinfo("Info", "Please select a CSV file first.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(sel)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", sel])
            else:
                subprocess.Popen(["xdg-open", sel])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")

    # ----------------------- Backend -----------------------
    def _run_benchmark_thread(self, input_file: str):
        try:
            os.makedirs(BENCH_DIR, exist_ok=True)

            # Use absolute nsys path
            nsys = self.nsys_path or "nsys"

            # Build command
            cmd = [
                nsys,
                "profile",
                "--stats=true",
                "--output=" + os.path.join(BENCH_DIR, "benchmark_report"),
                HOLO_EXE,
            ]
            if input_file:
                cmd.extend(["--input", input_file])

            self._run_and_stream(cmd)

            # Export stats to CSV
            rep = os.path.join(BENCH_DIR, "benchmark_report.nsys-rep")
            cmd_stats = [
                nsys,
                "stats",
                rep,
                "--format=csv",
                "--output=" + os.path.join(BENCH_DIR, "benchmark_report"),
                "--force-overwrite=true",
                "--force-export=true",
            ]
            self._run_and_stream(cmd_stats)

            # Remove heavy files
            for extra in (".nsys-rep", ".sqlite"):
                path = os.path.join(BENCH_DIR, "benchmark_report" + extra)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        self.log(f"removed : {path}")
                    except Exception as e:
                        self.log(f"Failed to remove {path} : {e}")

            self.log("=== Benchmark finished ===")
            self.refresh_csv_list()

        except FileNotFoundError as e:
            self.log(f"❌ Command not found : {e}")
            messagebox.showerror("Error", f"Command not found : {e}")
        except Exception as e:
            self.log(f"❌ Error : {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.set_running(False)

    def _run_and_stream(self, cmd):
        self.log(f"$ {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        for line in iter(process.stdout.readline, ""):
            self.log(line.rstrip())
        process.stdout.close()
        rc = process.wait()
        self.log(f"[exit code: {rc}]")
        if rc != 0:
            raise RuntimeError(f"Command failed (code {rc}) : {' '.join(cmd)}")

    def refresh_csv_list(self):
        self.csv_list.delete(0, tk.END)
        directory = self.csv_dir.get().strip() or BENCH_DIR
        if not os.path.isdir(directory):
            return
        files = sorted([f for f in os.listdir(directory) if f.lower().endswith(".csv")])
        for f in files:
            size = os.path.getsize(os.path.join(directory, f))
            self.csv_list.insert(tk.END, f"{f}  ({size} bytes)")

    def _selected_csv_path(self):
        sel = self.csv_list.curselection()
        if not sel:
            return None
        filename_label = self.csv_list.get(sel[0])
        # Extract the real filename before the two spaces we inserted
        filename = filename_label.split("  (", 1)[0]
        return os.path.join(self.csv_dir.get().strip() or BENCH_DIR, filename)

    def preview_selected_csv(self):
        path = self._selected_csv_path()
        self.preview_text.delete(1.0, tk.END)
        if not path:
            return
        try:
            df = pd.read_csv(path)
            rows, cols = df.shape
            head = df.head(int(self.nrows_preview.get()))
            # Build a simple textual preview
            self.preview_text.insert(tk.END, f"📄 File : {path}\n")
            self.preview_text.insert(tk.END, f"Lines : {rows} | Columns : {cols}\n")
            self.preview_text.insert(tk.END, f"Column names : {list(df.columns)}\n\n")
            if df.empty:
                self.preview_text.insert(tk.END, "⚠️ File is empty.\n")
            else:
                self.preview_text.insert(tk.END, head.to_string(index=False))
        except Exception as e:
            self.preview_text.insert(tk.END, f"❌ Error reading file: {e}")

    # ----------------------- Helpers -----------------------
    def set_running(self, running: bool):
        self.is_running = running
        state = "disabled" if running else "normal"
        self.btn_run.config(state=state)
        self.btn_clean.config(state=state)
        self.btn_set_nsys.config(state=("disabled" if running else "normal"))

    def log(self, text: str):
        self.log_text.insert(tk.END, text + ("\n" if not text.endswith("\n") else ""))
        self.log_text.see(tk.END)
        self.update_idletasks()

    def _has_holovibes(self) -> bool:
        return os.path.exists(HOLO_EXE)

    def _resolve_nsys(self) -> Optional[str]:
        """
        Resolves nsys via:
        1) custom path (if it exists and is executable),
        2) shutil.which (PATH + PATHEXT).
        """
        # 1) Custom path
        custom = (self.settings or {}).get("nsys_path")
        if custom:
            p = self._validate_nsys_path(custom)
            if p:
                return p

        # 2) System PATH
        if sys.platform.startswith("win"):
            for c in ("nsys.exe", "nsys"):
                p = shutil.which(c)
                if p:
                    return p
            return None
        else:
            return shutil.which("nsys")

    def _validate_nsys_path(self, path: str) -> Optional[str]:
        """Validates a user-provided path."""
        if not path:
            return None
        # Remove any quotes
        path = path.strip().strip('"')
        if not os.path.isfile(path):
            return None
        if not os.access(path, os.X_OK):
            return None
        # Ensure the name looks like nsys(.exe)
        base = os.path.basename(path).lower()
        if sys.platform.startswith("win"):
            if not (base == "nsys.exe" or base == "nsys"):
                return None
        else:
            if base != "nsys":
                return None
        return os.path.abspath(path)

    def _has_nsys(self) -> bool:
        # Re-resolve (PATH may change)
        self.nsys_path = self._resolve_nsys()
        return self.nsys_path is not None

    def _ok_color(self, ok: bool) -> str:
        return "#1b873f" if ok else "#b00020"

    def _check_holovibes_text(self) -> str:
        return (
            "Holovibes.exe : OK"
            if self._has_holovibes()
            else "Holovibes.exe : not found (build/bin/Holovibes.exe)"
        )

    def _check_nsys_text(self) -> str:
        return (
            f"nsys (Nsight Systems) : OK ({self.nsys_path})"
            if self._has_nsys()
            else "nsys (Nsight Systems) : not found — set the executable"
        )

    def refresh_checks(self):
        hv_ok = self._has_holovibes()
        ns_ok = self._has_nsys()
        self.lbl_exe.config(
            text=self._check_holovibes_text(), foreground=self._ok_color(hv_ok)
        )
        self.lbl_nsys.config(
            text=self._check_nsys_text(), foreground=self._ok_color(ns_ok)
        )
        self.btn_set_nsys.config(
            state=("normal" if not self.is_running else "disabled")
        )

    # ----------- parameters -----------
    def _load_settings(self) -> dict:
        try:
            if os.path.isfile(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_settings(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"⚠️ Impossible to save {SETTINGS_FILE} : {e}")

    def choose_nsys_path(self) -> bool:
        """Opens a file dialog to set the nsys path. Returns True if successful."""
        if sys.platform.startswith("win"):
            filetypes = [
                ("nsys.exe", "nsys.exe"),
                ("All executables", "*.exe"),
                ("All files", "*.*"),
            ]
        else:
            filetypes = [("nsys", "nsys"), ("All files", "*.*")]
        path = filedialog.askopenfilename(
            title="Select nsys executable",
            filetypes=filetypes,
        )
        if not path:
            return False

        valid = self._validate_nsys_path(path)
        if not valid:
            messagebox.showerror(
                "Invalid path",
                "The selected file does not appear to be a valid nsys executable.",
            )
            return False

        self.nsys_path = valid
        # Remember in settings
        self.settings["nsys_path"] = self.nsys_path
        self._save_settings()

        self.refresh_checks()
        self.log(f"nsys manually set: {self.nsys_path}")
        return True


def main():
    app = HolovibesGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
