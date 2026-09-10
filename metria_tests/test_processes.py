"""Real-process lifecycle regressions, run on Linux, macOS, and Windows."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import pytest

from metria import ProcessError, run_process


def _python(code: str) -> list[str]:
    return [sys.executable, "-u", "-c", code]


def _alive(pid: int) -> bool:
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
        if not handle:
            if ctypes.get_last_error() != 87:  # ERROR_INVALID_PARAMETER: PID gone
                raise ctypes.WinError(ctypes.get_last_error())
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
        finally:
            kernel.CloseHandle(handle)
    # A killed orphan may remain a zombie until the host init process reaps it.
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=2
    ).stdout.strip()
    return bool(status) and not status.startswith("Z")


def test_silent_child_has_wall_clock_timeout():
    started = time.monotonic()
    result = run_process(_python("import time; time.sleep(30)"), timeout_s=0.4)
    assert result.timed_out
    assert result.returncode != 0
    assert result.stdout == result.stderr == ""
    assert time.monotonic() - started < 4


def test_timeout_retains_output_without_a_newline():
    result = run_process(
        _python(
            "import os,time; os.write(1,b'partial stdout'); os.write(2,b'partial stderr'); time.sleep(30)"
        ),
        timeout_s=1,
    )
    assert result.timed_out
    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"


def test_diagnostic_source_entry_point_needs_no_package_install():
    script = (
        Path(__file__).resolve().parents[1] / "tools/diagnostics/turbo_hardware_diag.py"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--platform-support" in result.stdout


def test_noisy_child_cannot_defeat_deadline_or_retention_limit():
    result = run_process(
        _python(
            "import os\nwhile True:\n os.write(1,b'x'*8192)\n os.write(2,b'y'*8192)"
        ),
        timeout_s=0.7,
        max_output_bytes=1000,
    )
    assert result.timed_out
    assert result.stdout == "x" * 1000
    assert result.stderr == "y" * 1000
    assert result.stdout_truncated and result.stderr_truncated
    assert result.elapsed_s < 4


def test_streamed_output_unblocks_child_and_preserves_utf8(tmp_path):
    acknowledgement = tmp_path / "ack"
    chunks = []

    def on_output(stream, chunk):
        chunks.append((stream, chunk))
        if "ready" in "".join(text for _, text in chunks):
            acknowledgement.touch()

    result = run_process(
        _python(
            "import os,sys,time; from pathlib import Path\n"
            "os.write(1,b'ready \\xe2'); time.sleep(0.05); os.write(1,b'\\x82\\xac')\n"
            "while not Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
            "os.write(2,b'done'); sys.exit(7)"
        )
        + [str(acknowledgement)],
        timeout_s=5,
        on_output=on_output,
    )
    assert not result.timed_out
    assert result.returncode == 7
    assert result.stdout == "ready €"
    assert result.stderr == "done"
    assert "".join(text for stream, text in chunks if stream == "stdout") == "ready €"


def test_environment_directory_and_literal_arguments(tmp_path):
    result = run_process(
        _python(
            "import os,sys; print(os.getcwd()); print(os.environ['METRIA_TEST']); print(sys.argv[1]); print(sys.stdin.read())"
        )
        + ["$(echo secret); & echo other"],
        timeout_s=5,
        cwd=tmp_path,
        env={**os.environ, "METRIA_TEST": "environment value"},
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        str(tmp_path),
        "environment value",
        "$(echo secret); & echo other",
        "",
    ]
    assert "secret" not in result.command_fingerprint
    assert "environment value" not in result.command_fingerprint


def test_merge_stderr_and_zero_retention():
    chunks = []
    result = run_process(
        _python("import os; os.write(1,b'first'); os.write(2,b'second')"),
        timeout_s=5,
        merge_stderr=True,
        max_output_bytes=0,
        on_output=lambda stream, chunk: chunks.append((stream, chunk)),
    )
    assert result.stdout == result.stderr == ""
    assert result.stdout_truncated and not result.stderr_truncated
    assert all(stream == "stdout" for stream, _ in chunks)
    assert "".join(chunk for _, chunk in chunks) == "firstsecond"


@pytest.fixture
def process_tree(tmp_path):
    script = tmp_path / "tree.py"
    pids = tmp_path / "pids"
    script.write_text(
        "import os,sys,subprocess,time\n"
        "from pathlib import Path\n"
        "depth=int(sys.argv[1]); pids=Path(sys.argv[2]); mode=sys.argv[3]\n"
        "with pids.open('a') as out: out.write(str(os.getpid())+'\\n')\n"
        "if depth:\n"
        " subprocess.Popen([sys.executable,'-u',__file__,str(depth-1),str(pids),mode])\n"
        "else:\n"
        " print('ready',flush=True)\n"
        "if depth==2 and mode=='success':\n"
        " while len(pids.read_text().splitlines())<3: time.sleep(0.01)\n"
        " sys.exit(0)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    yield script, pids
    # Ensure even a regression in the runner cannot leave test children behind.
    for pid in map(int, pids.read_text().splitlines() if pids.exists() else []):
        if _alive(pid):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=3
                )
            else:
                import signal

                os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("mode", ["timeout", "success", "exception", "interrupt"])
def test_entire_process_tree_is_cleaned_up(process_tree, mode):
    script, pid_file = process_tree
    command = [sys.executable, "-u", str(script), "2", str(pid_file), mode]

    def on_output(stream, chunk):
        if "ready" in chunk:
            if mode == "exception":
                raise RuntimeError("callback failed")
            if mode == "interrupt":
                raise KeyboardInterrupt

    if mode in {"exception", "interrupt"}:
        with pytest.raises(RuntimeError if mode == "exception" else KeyboardInterrupt):
            run_process(command, timeout_s=5, on_output=on_output)
    else:
        result = run_process(command, timeout_s=1.5, on_output=on_output)
        assert result.timed_out == (mode == "timeout")
        assert result.returncode == 0 if mode == "success" else result.returncode != 0
    pids = list(map(int, pid_file.read_text().splitlines()))
    assert len(pids) == 3
    deadline = time.monotonic() + 2
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.025)
    assert not any(_alive(pid) for pid in pids)


def test_missing_executable_does_not_retain_arguments_or_paths(tmp_path):
    command = [str(tmp_path / "secret-model-path-missing"), "--api-key=secret-token"]
    try:
        result = run_process(command, timeout_s=5)
    except ProcessError as exc:
        evidence = str(exc)
    else:
        assert result.returncode != 0
        evidence = repr(result)
    assert "secret-model-path" not in evidence
    assert "secret-token" not in evidence
    assert str(tmp_path) not in evidence


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout_s"):
        run_process(_python("pass"), timeout_s=timeout)


@pytest.mark.skipif(os.name != "nt", reason="Windows batch files use an implicit shell")
@pytest.mark.parametrize("name", ["script.bat", "script.CMD", "script.cmd. "])
def test_windows_batch_files_are_rejected(name):
    with pytest.raises(ValueError, match="batch files"):
        run_process([name, "& echo secret"], timeout_s=1)


@pytest.mark.parametrize("command", [[], "echo hi", [""], ["echo", "a\0b"], [1]])
def test_invalid_command(command):
    with pytest.raises(ValueError, match="command"):
        run_process(command, timeout_s=1)


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_retention_limit(limit):
    with pytest.raises(ValueError, match="max_output_bytes"):
        run_process(_python("pass"), timeout_s=1, max_output_bytes=limit)
