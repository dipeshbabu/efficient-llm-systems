"""Bounded subprocess execution for runtime, benchmark, and diagnostic clients.

This is a cooperative process runner, not a sandbox. POSIX descendants must
remain in the new process group; Windows descendants stay in a Job Object.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

OutputStream = Literal["stdout", "stderr"]
OutputCallback = Callable[[OutputStream, str], None]
_CHUNK_BYTES = 8192
_POLL_SECONDS = 0.025
_CLEANUP_SECONDS = 2.0


@dataclass(frozen=True)
class ProcessResult:
    """Execution evidence; arguments and environment values are never retained.

    Output is UTF-8 with replacement for invalid bytes. Each stream retains its
    first ``max_output_bytes`` bytes. Child output is not automatically redacted.
    ``elapsed_s`` includes launch and cleanup, so it may exceed ``timeout_s``.
    """

    command_fingerprint: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_s: float
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class ProcessError(RuntimeError):
    """Process launch or cleanup failed; messages omit arguments and paths."""


def _read_pipe(
    pipe: BinaryIO,
    stream: OutputStream,
    events: queue.Queue[tuple[OutputStream, bytes | OSError | None]],
    stop: threading.Event,
) -> None:
    def send(data: bytes | OSError | None) -> None:
        while not stop.is_set():
            try:
                events.put((stream, data), timeout=_POLL_SECONDS)
                return
            except queue.Full:
                continue

    try:
        while not stop.is_set():
            data = pipe.read(_CHUNK_BYTES)
            if not data:
                break
            send(data)
    except OSError as exc:
        send(exc)
    finally:
        pipe.close()
        send(None)


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        # The helper owns the job handle; killing it closes that handle and
        # terminates the entire job. It never starts user code outside the job.
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_process(
    command: Sequence[str],
    *,
    timeout_s: float,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    merge_stderr: bool = False,
    max_output_bytes: int = 4 * 1024 * 1024,
    on_output: OutputCallback | None = None,
) -> ProcessResult:
    """Run an argument vector with a deadline and guaranteed cleanup attempts.

    No shell is used and stdin is closed. ``env`` replaces the inherited
    environment when supplied. ``on_output`` receives decoded chunks, including
    output beyond the retention limit, and must return promptly. Its exceptions
    (including KeyboardInterrupt) propagate after child cleanup.

    The deadline starts before launch and is checked independently of output.
    OS process creation itself cannot be interrupted by Python. Cleanup has a
    separate two-second bound. Timeout returns partial evidence with
    ``timed_out=True``; nonzero exits are returned without raising.
    """
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a nonempty sequence of strings")
    args = list(command)
    if any(not isinstance(arg, str) or "\0" in arg for arg in args) or not args[0]:
        raise ValueError("command must contain strings without NUL bytes")
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes < 0
    ):
        raise ValueError("max_output_bytes must be a nonnegative integer")
    fingerprint = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(args, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )
    creationflags = 0
    if sys.platform == "win32":
        # Python/Windows may route batch files through cmd.exe even with
        # shell=False. Require an executable to preserve argument boundaries.
        if args[0].rstrip(" .").lower().endswith((".bat", ".cmd")):
            raise ValueError("Windows batch files are not supported; use an executable")
        creationflags = subprocess.CREATE_NO_WINDOW
        args = [
            sys.executable,
            "-I",
            str(Path(__file__).with_name("_process_windows.py")),
            *args,
        ]
    started = time.monotonic()
    deadline = started + timeout_s
    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            bufsize=0,
            cwd=cwd,
            env=env,
            start_new_session=sys.platform != "win32",
            creationflags=creationflags,
        )
    except OSError as exc:
        raise ProcessError(
            f"Process launch failed ({type(exc).__name__}; {fingerprint})"
        ) from None

    events: queue.Queue[tuple[OutputStream, bytes | OSError | None]] = queue.Queue(
        maxsize=64
    )
    stop = threading.Event()
    readers: list[threading.Thread] = []
    retained = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}
    decoders = {
        stream: codecs.getincrementaldecoder("utf-8")("replace") for stream in retained
    }
    timed_out = False
    ended = False
    cleanup_deadline: float | None = None
    pending = 0
    try:
        for stream, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            if pipe is not None:
                reader = threading.Thread(
                    target=_read_pipe,
                    args=(pipe, stream, events, stop),
                    name=f"metria-process-{process.pid}-{stream}",
                    daemon=True,
                )
                reader.start()
                readers.append(reader)
                pending += 1
        while pending or not ended:
            now = time.monotonic()
            returncode = process.poll()
            if not ended and (returncode is not None or now >= deadline):
                timed_out = returncode is None and now >= deadline
                cleanup_deadline = now + _CLEANUP_SECONDS
                _kill_tree(process)
                ended = True
                deadline = cleanup_deadline
            if ended and now >= deadline:
                raise ProcessError(
                    f"Process output cleanup exceeded its deadline ({fingerprint})"
                )
            try:
                stream, data = events.get(
                    timeout=min(_POLL_SECONDS, max(0.001, deadline - now))
                )
            except queue.Empty:
                continue
            if isinstance(data, OSError):
                raise ProcessError(
                    f"Process output read failed ({fingerprint})"
                ) from None
            if data is None:
                pending -= 1
            else:
                remaining = max_output_bytes - len(retained[stream])
                retained[stream].extend(data[:remaining])
                truncated[stream] |= len(data) > remaining
            if on_output is not None:
                chunk = decoders[stream].decode(data or b"", final=data is None)
                if chunk:
                    on_output(stream, chunk)
    finally:
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + _CLEANUP_SECONDS
        try:
            if not ended:
                _kill_tree(process)
            try:
                process.wait(timeout=max(0, cleanup_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise ProcessError(
                    f"Process cleanup exceeded its deadline ({fingerprint})"
                ) from None
        finally:
            stop.set()
            for reader in readers:
                reader.join(timeout=max(0, cleanup_deadline - time.monotonic()))
            # A thread closes its own pipe to avoid blocking on a read lock.
            # Pipes without a started reader still belong to this thread.
            if not readers and process.stdout is not None:
                process.stdout.close()
            if len(readers) < 2 and process.stderr is not None:
                process.stderr.close()
            if any(reader.is_alive() for reader in readers):
                raise ProcessError(f"Process pipe reader did not stop ({fingerprint})")

    return ProcessResult(
        command_fingerprint=fingerprint,
        returncode=process.returncode,
        stdout=retained["stdout"].decode("utf-8", errors="replace"),
        stderr=retained["stderr"].decode("utf-8", errors="replace"),
        timed_out=timed_out,
        elapsed_s=time.monotonic() - started,
        stdout_truncated=truncated["stdout"],
        stderr_truncated=truncated["stderr"],
    )
