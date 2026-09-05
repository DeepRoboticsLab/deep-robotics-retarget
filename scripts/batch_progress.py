"""Two-line terminal display used only by the batch retargeting scripts."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import os
import sys
import time

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text


class BatchProgress:
    def __init__(self, total, target_folder):
        os.makedirs(target_folder, exist_ok=True)
        self.log_path = os.path.join(target_folder, "batch_retarget.log")
        self.console = Console(file=sys.stderr)
        self.progress = Progress(
            TextColumn("Processed {task.completed:.0f}/{task.total:.0f}"),
            BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("Elapsed"), TimeElapsedColumn(),
            TextColumn("ETA"), TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            console=self.console,
        )
        self.task = self.progress.add_task("", total=total, status="")
        self.latest = Text("Latest: waiting for a file", no_wrap=True, overflow="ellipsis")
        self.latest_time = -1
        self.failed = 0
        self.live = Live(Group(self.progress, self.latest), console=self.console,
                         refresh_per_second=4, redirect_stdout=False, redirect_stderr=False)

    def __enter__(self):
        self.stack = ExitStack()
        log = self.stack.enter_context(open(self.log_path, "a", buffering=1))
        self.stack.enter_context(redirect_stdout(log))
        self.stack.enter_context(redirect_stderr(log))
        self.live.start()
        return self

    def __exit__(self, *exc):
        self.stack.close()
        self.live.stop()

    def started(self, name, frames, started_at=None):
        started_at = time.monotonic() if started_at is None else started_at
        # Workers may deliver events out of order.
        if started_at >= self.latest_time:
            self.latest_time = started_at
            self.latest.plain = f"Latest: {frames} frames | {name}"

    def advance(self, success=True):
        self.failed += not success
        status = f"Failed {self.failed} (see batch_retarget.log)" if self.failed else ""
        self.progress.update(self.task, advance=1, status=status)

    def files(self, paths):
        for path in paths:
            yield path
            self.advance()

    def error(self, message):
        print(message, file=sys.stderr)
        self.failed += 1
        self.progress.update(self.task, status=f"Failed {self.failed} (see batch_retarget.log)")

    def confirm_overwrite(self, path):
        # Preserve the BVH script's existing interactive overwrite choice.
        self.live.stop()
        try:
            return self.console.input(f"  文件已存在: {path}\n  是否覆盖? [y/N]: ", markup=False).strip().lower() in ("y", "yes")
        finally:
            self.live.start()
