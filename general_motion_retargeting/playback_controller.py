"""GUI playback controller for robot motion visualization.

Provides a Tkinter-based GUI with pause/resume, frame seeking, and
real-time status display. Thread-safe for use with MuJoCo viewer.
"""

import queue
import threading
from dataclasses import dataclass


@dataclass
class PlaybackState:
    """Represents the current playback state."""
    frame_idx: int
    paused: bool


class PlaybackController:
    """Thread-safe playback controller with Tkinter GUI.
    
    Provides frame seeking, pause/resume, and status display.
    Falls back gracefully if Tkinter is unavailable.
    
    Args:
        num_frames: Total number of frames in the motion.
        title: Window title for the GUI.
    """
    def __init__(self, num_frames: int, title: str = "Playback Control"):
        self.num_frames = max(1, int(num_frames))
        self._state = PlaybackState(frame_idx=0, paused=False)
        self._lock = threading.Lock()
        self._commands: "queue.Queue[tuple[str, int | bool | None]]" = queue.Queue()
        self._closed = False
        self._thread = None
        self._title = title

        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception:
            self._tk = None
            self._ttk = None
            return

        self._tk = tk
        self._ttk = ttk
        self._thread = threading.Thread(target=self._run_ui, daemon=True)
        self._thread.start()

    def _run_ui(self):
        tk = self._tk
        ttk = self._ttk

        root = tk.Tk()
        root.title(self._title)
        root.geometry("420x150")
        root.resizable(False, False)

        paused_var = tk.BooleanVar(value=self._state.paused)
        frame_var = tk.IntVar(value=self._state.frame_idx)
        entry_var = tk.StringVar(value="0")
        status_var = tk.StringVar(value=self._status_text(self._state.frame_idx, self._state.paused))
        slider_updating = {"active": False}

        def clamp_frame(value):
            return max(0, min(self.num_frames - 1, int(value)))

        def on_close():
            self._closed = True
            root.destroy()

        def on_toggle():
            self._commands.put(("set_paused", paused_var.get()))

        def on_slider(value):
            if slider_updating["active"]:
                return
            self._commands.put(("set_frame", clamp_frame(float(value))))

        def on_jump(*_args):
            try:
                target = clamp_frame(int(entry_var.get()))
            except ValueError:
                return
            self._commands.put(("set_frame", target))

        ttk.Label(root, text="Frame").pack(anchor="w", padx=12, pady=(10, 0))
        slider = ttk.Scale(root, from_=0, to=self.num_frames - 1, orient="horizontal", command=on_slider)
        slider.pack(fill="x", padx=12, pady=(4, 8))

        row = ttk.Frame(root)
        row.pack(fill="x", padx=12)

        pause_btn = ttk.Checkbutton(row, text="Pause", variable=paused_var, command=on_toggle)
        pause_btn.pack(side="left")

        ttk.Label(row, text="Jump to frame:").pack(side="left", padx=(18, 6))
        entry = ttk.Entry(row, textvariable=entry_var, width=10)
        entry.pack(side="left")
        entry.bind("<Return>", on_jump)

        jump_btn = ttk.Button(row, text="Go", command=on_jump)
        jump_btn.pack(side="left", padx=(6, 0))

        ttk.Label(root, textvariable=status_var).pack(anchor="w", padx=12, pady=(12, 0))

        def poll_state():
            if self._closed:
                return
            with self._lock:
                state = PlaybackState(self._state.frame_idx, self._state.paused)
            slider_updating["active"] = True
            slider.set(state.frame_idx)
            slider_updating["active"] = False
            paused_var.set(state.paused)
            frame_var.set(state.frame_idx)
            status_var.set(self._status_text(state.frame_idx, state.paused))
            root.after(50, poll_state)

        root.protocol("WM_DELETE_WINDOW", on_close)
        poll_state()
        root.mainloop()

    def _status_text(self, frame_idx: int, paused: bool) -> str:
        mode = "paused" if paused else "playing"
        return f"Frame: {frame_idx} / {self.num_frames - 1}   Mode: {mode}"

    def is_available(self) -> bool:
        return self._tk is not None

    def is_closed(self) -> bool:
        return self._closed

    def apply_pending_commands(self):
        while True:
            try:
                cmd, value = self._commands.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                if cmd == "set_paused":
                    self._state.paused = bool(value)
                elif cmd == "set_frame":
                    self._state.frame_idx = max(0, min(self.num_frames - 1, int(value)))

    def set_state(self, frame_idx: int, paused: bool | None = None):
        with self._lock:
            self._state.frame_idx = max(0, min(self.num_frames - 1, int(frame_idx)))
            if paused is not None:
                self._state.paused = bool(paused)

    def get_state(self) -> PlaybackState:
        with self._lock:
            return PlaybackState(self._state.frame_idx, self._state.paused)

    def close(self):
        self._closed = True
