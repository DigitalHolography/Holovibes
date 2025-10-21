#!/usr/bin/env python3
"""
Small GUI to inspect local jitter from a timestamps CSV produced by recording.

CSV is expected to have a header with at least one of:
  - frame_index
  - synced_ts_us
  - camera_ts_us
  - offset_us

The tool computes inter-frame intervals (dT) from the selected timestamp column
and plots a rolling (trailing) standard deviation of dT as the local jitter.

Usage:
  - Double-click / run without args: opens a Tk GUI
  - CLI (time-series jitter):
      python jitter.py path/to/file_timestamps.csv --mode series [--column synced_ts_us] [--window 50]
  - CLI (distribution vs ideal line t0..tn):
      python jitter.py path/to/file_timestamps.csv --mode distribution [--column synced_ts_us] [--bins auto] [--fit]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")
except Exception as e:  # pragma: no cover - environment dependent
    # GUI/Matplotlib might be missing in some environments; CLI still works
    tk = None
    plt = None


EXPECTED_COLUMNS = [
    "frame_index",
    "synced_ts_us",
    "camera_ts_us",
    "offset_us",
]


@dataclass
class TimestampData:
    frame_index: np.ndarray
    ts_us: np.ndarray
    column_name: str


def read_timestamps_csv(
    path: str, preferred_column: str | None = None
) -> TimestampData:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in reader.fieldnames or []]

        # Choose column: preferred if present, else first matching known ts column.
        ts_candidates = [c for c in (preferred_column,) if c] + [
            c for c in ("synced_ts_us", "camera_ts_us") if c in header
        ]
        ts_col = None
        for c in ts_candidates:
            if c in header:
                ts_col = c
                break
        if ts_col is None:
            raise ValueError(
                f"No valid timestamp column found. Header has: {header}. Expected one of: synced_ts_us,camera_ts_us."
            )

        # frame index optional; if missing, infer as 0..N-1
        has_frame_index = "frame_index" in header
        frame_idx: List[int] = []
        ts_us: List[float] = []
        for row in reader:
            try:
                ts_val = float(row[ts_col])
            except Exception:
                # Skip malformed rows
                continue
            ts_us.append(ts_val)
            if has_frame_index:
                try:
                    frame_idx.append(int(row["frame_index"]))
                except Exception:
                    frame_idx.append(len(frame_idx))

        if not ts_us:
            raise ValueError("No timestamp data found in file.")

        if not has_frame_index:
            frame_idx = list(range(len(ts_us)))

        return TimestampData(
            frame_index=np.asarray(frame_idx, dtype=np.int64),
            ts_us=np.asarray(ts_us, dtype=np.float64),
            column_name=ts_col,
        )


def compute_intervals_us(ts_us: np.ndarray) -> np.ndarray:
    # Intervals between consecutive timestamps (microseconds)
    if ts_us.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(ts_us)


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling standard deviation using a numerically stable approach.

    For i < window-1, result is NaN.
    """
    n = arr.size
    if window <= 1 or n == 0:
        return np.full(n, np.nan, dtype=np.float64)

    csum = np.cumsum(arr, dtype=np.float64)
    csum2 = np.cumsum(arr * arr, dtype=np.float64)

    res = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        s = csum[i] - (csum[i - window] if i >= window else 0.0)
        s2 = csum2[i] - (csum2[i - window] if i >= window else 0.0)
        mean = s / window
        var = max(s2 / window - mean * mean, 0.0)
        res[i] = np.sqrt(var)
    return res


def compute_local_jitter(
    ts_us: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: (frame_idx_for_dt, dt_ms, jitter_ms)
    - dt_ms is inter-frame interval in ms
    - jitter_ms is trailing rolling std(dt_ms) over the window
    Indexing uses the trailing sample's frame index for dt.
    """
    dt_us = compute_intervals_us(ts_us)
    dt_ms = dt_us / 1000.0
    jitter_ms = rolling_std(dt_ms, window)
    # dt aligns between frames k-1 and k; associate with frame index k
    frame_idx_for_dt = np.arange(1, 1 + dt_ms.size)
    return frame_idx_for_dt, dt_ms, jitter_ms


def _ideal_line_endpoints(ts_us: np.ndarray) -> Tuple[float, float]:
    n = ts_us.size
    if n < 2:
        return float("nan"), float("nan")
    period_us = (ts_us[-1] - ts_us[0]) / (n - 1)
    return float(ts_us[0]), float(period_us)


def _ideal_line_fit(ts_us: np.ndarray) -> Tuple[float, float]:
    n = ts_us.size
    if n < 2:
        return float("nan"), float("nan")
    k = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(k, ts_us.astype(np.float64), 1)
    return float(intercept), float(slope)


def compute_distribution_residuals(ts_us: np.ndarray, fit: bool = False) -> Tuple[np.ndarray, float]:
    """Residuals to ideal line; returns (residual_ms, period_ms).
    If fit=True, use least-squares to estimate period; else use endpoints.
    """
    n = ts_us.size
    if n < 2:
        return np.array([], dtype=np.float64), float("nan")
    if fit:
        a_us, period_us = _ideal_line_fit(ts_us)
    else:
        a_us, period_us = _ideal_line_endpoints(ts_us)
    k = np.arange(n, dtype=np.float64)
    ideal = a_us + k * period_us
    residual_ms = (ts_us - ideal) / 1000.0
    return residual_ms, period_us / 1000.0


def plot_jitter(
    ax,
    frame_idx_for_dt: np.ndarray,
    dt_ms: np.ndarray,
    jitter_ms: np.ndarray,
    title: str,
):
    ax.clear()
    ax.plot(
        frame_idx_for_dt,
        dt_ms,
        color="#4C78A8",
        alpha=0.4,
        linewidth=1.0,
        label="dT (ms)",
    )
    ax.plot(
        frame_idx_for_dt,
        jitter_ms,
        color="#F58518",
        linewidth=1.5,
        label="Local jitter (std, ms)",
    )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")


def plot_distribution(ax, residual_ms: np.ndarray, bins: int | str, title: str):
    ax.clear()
    # Histogram
    ax.hist(residual_ms, bins=bins, color="#4C78A8", alpha=0.75, edgecolor="black")
    ax.axvline(
        0.0, color="#F58518", linestyle="--", linewidth=1.5, label="Ideal (0 ms error)"
    )
    mu = float(np.nanmean(residual_ms)) if residual_ms.size else float("nan")
    sigma = float(np.nanstd(residual_ms)) if residual_ms.size else float("nan")
    ax.set_xlabel("Timing error vs ideal (ms)")
    ax.set_ylabel("Count")
    ax.set_title(
        title + (f"\nmean={mu:.3f} ms, std={sigma:.3f} ms" if residual_ms.size else "")
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")


def run_cli(args):  # pragma: no cover - simple convenience
    data = read_timestamps_csv(args.csv, preferred_column=args.column)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    if args.mode == "distribution":
        residual_ms, ideal_period_ms = compute_distribution_residuals(data.ts_us)
        bins = args.bins if args.bins != "auto" else "auto"
        title = (
            f"Distribution of timing error vs ideal line (column={data.column_name})\n"
            f"{os.path.basename(args.csv)} | ideal period ≈ {ideal_period_ms:.3f} ms"
        )
        plot_distribution(ax, residual_ms, bins, title)
    else:
        fi, dt_ms, jitter_ms = compute_local_jitter(data.ts_us, args.window)
        title = f"Local jitter (window={args.window}, column={data.column_name})\n{os.path.basename(args.csv)}"
        plot_jitter(ax, fi, dt_ms, jitter_ms, title)
    plt.tight_layout()
    plt.show()


def run_cli2(args):  # pragma: no cover - new CLI with fit capability
    data = read_timestamps_csv(args.csv, preferred_column=args.column)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    if args.mode == "distribution":
        residual_ms, period_ms = compute_distribution_residuals(
            data.ts_us, fit=getattr(args, "fit", False)
        )
        bins = args.bins if args.bins != "auto" else "auto"
        fps = (1000.0 / period_ms) if (period_ms and np.isfinite(period_ms) and period_ms > 0) else float("nan")
        title = (
            f"Distribution of timing error vs {'fitted' if getattr(args, 'fit', False) else 'endpoint'} ideal line (column={data.column_name})\n"
            f"{os.path.basename(args.csv)} | period ≈ {period_ms:.3f} ms | FPS ≈ {fps:.3f}"
        )
        plot_distribution(ax, residual_ms, bins, title)
        print(f"Estimated period: {period_ms:.6f} ms, FPS: {fps:.6f}")
    else:
        fi, dt_ms, jitter_ms = compute_local_jitter(data.ts_us, args.window)
        title = f"Local jitter (window={args.window}, column={data.column_name})\n{os.path.basename(args.csv)}"
        plot_jitter(ax, fi, dt_ms, jitter_ms, title)
    plt.tight_layout()
    plt.show()


class JitterApp:
    def __init__(self, root: tk.Tk):  # type: ignore[valid-type]
        self.root = root
        self.root.title("Jitter Inspector")

        self.csv_path_var = tk.StringVar()
        self.column_var = tk.StringVar(value="synced_ts_us")
        self.window_var = tk.StringVar(value="50")
        self.mode_var = tk.StringVar(value="series")  # series | distribution
        self.bins_var = tk.StringVar(value="auto")
        self.fit_var = tk.BooleanVar(value=False)

        self._build_ui()

        self.fig, self.ax = plt.subplots(figsize=(10, 5)) if plt else (None, None)
        self.canvas = None
        if plt:
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.data: TimestampData | None = None

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # File row
        frow = ttk.Frame(main)
        frow.pack(fill=tk.X)
        ttk.Label(frow, text="CSV file:").pack(side=tk.LEFT)
        ttk.Entry(frow, textvariable=self.csv_path_var, width=60).pack(
            side=tk.LEFT, padx=6, fill=tk.X, expand=True
        )
        ttk.Button(frow, text="Browse", command=self._browse).pack(side=tk.LEFT)

        # Options row
        orow = ttk.Frame(main)
        orow.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(orow, text="Column:").pack(side=tk.LEFT)
        ttk.Combobox(
            orow,
            textvariable=self.column_var,
            values=["synced_ts_us", "camera_ts_us"],
            width=15,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(orow, text="Mode:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Combobox(
            orow,
            textvariable=self.mode_var,
            values=["series", "distribution"],
            width=14,
        ).pack(side=tk.LEFT, padx=6)
        # Options for series
        ttk.Label(orow, text="Window:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(orow, textvariable=self.window_var, width=6).pack(
            side=tk.LEFT, padx=6
        )
        # Options for distribution
        ttk.Label(orow, text="Bins:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(orow, textvariable=self.bins_var, width=6).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(orow, text="Fit period", variable=self.fit_var).pack(side=tk.LEFT, padx=(12, 0))

        # Buttons row
        brow = ttk.Frame(main)
        brow.pack(fill=tk.X, pady=(6, 6))
        ttk.Button(brow, text="Compute & Plot", command=self._compute_and_plot).pack(
            side=tk.LEFT
        )
        ttk.Button(brow, text="Save Plot", command=self._save_plot).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(brow, text="Quit", command=self.root.destroy).pack(side=tk.RIGHT)

        # Plot area
        self.plot_frame = ttk.Frame(main)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select timestamps CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)

    def _compute_and_plot(self):
        if not plt:
            messagebox.showerror(
                "Unavailable", "Matplotlib/Tkinter not available in this environment."
            )
            return
        path = self.csv_path_var.get().strip()
        if not path:
            messagebox.showwarning("Missing file", "Please select a CSV file.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"File not found:\n{path}")
            return
        try:
            self.data = read_timestamps_csv(
                path, preferred_column=self.column_var.get()
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        mode = self.mode_var.get()
        if mode == "distribution":
            # Bins can be 'auto' or integer
            bins_str = self.bins_var.get().strip()
            bins: int | str
            if bins_str.lower() == "auto":
                bins = "auto"
            else:
                try:
                    bins = int(bins_str)
                except Exception:
                    messagebox.showerror(
                        "Invalid bins", "Bins must be an integer or 'auto'"
                    )
                    return
            residual_ms, ideal_period_ms = compute_distribution_residuals(
                self.data.ts_us
            )
            title = (
                f"Distribution of timing error vs ideal line (column={self.data.column_name})\n"
                f"{os.path.basename(path)} | ideal period ≈ {ideal_period_ms:.3f} ms"
            )
            plot_distribution(self.ax, residual_ms, bins, title)
        else:
            try:
                window = int(self.window_var.get())
                if window < 2:
                    raise ValueError
            except Exception:
                messagebox.showerror("Invalid window", "Window must be an integer >= 2")
                return
            fi, dt_ms, jitter_ms = compute_local_jitter(self.data.ts_us, window)
            title = f"Local jitter (window={window}, column={self.data.column_name})\n{os.path.basename(path)}"
            plot_jitter(self.ax, fi, dt_ms, jitter_ms, title)
        # Ensure distribution plot reflects current 'fit' option and shows period/FPS
        if mode == "distribution":
            bins_str = self.bins_var.get().strip()
            if bins_str.lower() == "auto":
                bins_val = "auto"
            else:
                try:
                    bins_val = int(bins_str)
                except Exception:
                    bins_val = "auto"
            residual_ms, period_ms = compute_distribution_residuals(
                self.data.ts_us, fit=self.fit_var.get()
            )
            fps = (
                1000.0 / period_ms if (period_ms and np.isfinite(period_ms) and period_ms > 0) else float("nan")
            )
            title = (
                f"Distribution of timing error vs {'fitted' if self.fit_var.get() else 'endpoint'} ideal line (column={self.data.column_name})\n"
                f"{os.path.basename(path)} | period ≈ {period_ms:.3f} ms | FPS ≈ {fps:.3f}"
            )
            plot_distribution(self.ax, residual_ms, bins_val, title)
        self.fig.tight_layout()
        self.canvas.draw()

    def _save_plot(self):
        if not (plt and self.fig):
            return
        path = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.fig.savefig(path, dpi=150)


def main(argv: List[str] | None = None):  # pragma: no cover - entrypoint
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description="Plot local jitter from timestamps CSV"
    )
    parser.add_argument("csv", nargs="?", help="Path to timestamps CSV")
    parser.add_argument(
        "--column", choices=["synced_ts_us", "camera_ts_us"], default="synced_ts_us"
    )
    parser.add_argument(
        "--mode", choices=["series", "distribution"], default="series", help="Plot type"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling window size (frames) for series mode",
    )
    parser.add_argument(
        "--bins",
        default="auto",
        help="Histogram bins for distribution mode (int or 'auto')",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit the ideal line via least squares to estimate real period/FPS",
    )
    args = parser.parse_args(argv)

    if args.csv:
        # Use enhanced CLI that can fit period
        run_cli2(args)
    else:
        if tk is None or plt is None:
            print("Tkinter/Matplotlib not available; run with a CSV path for CLI mode.")
            parser.print_help()
            return 2
        root = tk.Tk()
        app = JitterApp(root)
        root.geometry("1000x600")
        root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
