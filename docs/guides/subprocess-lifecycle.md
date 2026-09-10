# Bounded subprocess execution

Use `metria.processes.run_process` for new runtime, benchmark, and diagnostic
commands. It provides one lifecycle for quiet processes and streamed output:

```python
import sys

from metria import run_process

result = run_process(
    [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"],
    timeout_s=1,
    max_output_bytes=1024,
)
assert result.timed_out
assert "started" in result.stdout
```

The timeout starts before launch and uses a monotonic clock. Output reads run
independently, so a missing newline, silent child, or continuously writing child
cannot postpone the deadline. As with Python's subprocess APIs, operating-system
process creation cannot itself be interrupted. Cleanup has a separate two-second
budget; `elapsed_s` includes both launch and cleanup.

The runner closes stdin, takes an argument vector without a shell, and returns
the exit code instead of raising for nonzero exits. `cwd` selects the working
directory; `env`, when supplied, replaces the inherited environment. Use
`merge_stderr=True` to preserve stdout/stderr interleaving in `stdout`.
Windows batch files are rejected because Windows can run them through an
implicit shell; pass an executable instead.

Each output stream retains at most its first `max_output_bytes` bytes (4 MiB by
default). Truncation is explicit in `stdout_truncated` and `stderr_truncated`.
Streams decode as UTF-8, replacing invalid bytes. An optional
`on_output(stream, chunk)` callback receives incremental decoded output,
including output beyond the retention limit. Chunks need not end at line breaks.
Callbacks must return promptly; callback errors and `KeyboardInterrupt` propagate
after cleanup. Keep slow output sinks outside the callback.

Results retain a SHA-256 fingerprint of the argument vector, without the command
arguments, environment values, or working directory. Launch and cleanup errors
also omit arguments and paths. Fingerprints are deterministic identifiers, not
encryption. Child stdout/stderr is retained verbatim; callers must redact it
before publishing if their command emits credentials or private data.

## Process containment

- Linux and macOS: the command starts in a new session. Cleanup sends `SIGKILL`
  to its process group, including descendants that inherited the group.
- Windows: an isolated Python helper joins a Job Object before launching the
  requested command. The job disallows breakaway and kills members when its only
  handle closes. Terminating the helper therefore terminates its descendants.
  Failure to establish containment prevents command execution and returns 126;
  failure to launch the requested command returns 127, with a generic diagnostic.

Cleanup also runs after normal exit, so a successful parent cannot leave its
background workers running. This API manages cooperative command trees, not
hostile programs: deliberately detached POSIX daemons that create another session
or process group are unsupported. A descendant that keeps a pipe open beyond
cleanup causes `ProcessError`, rather than an indefinite wait. Kernel-level
uninterruptible process states cannot be resolved by a Python timeout.

The hardware diagnostic consumes this API for benchmark runs and `_run_cmd`
probes. It preserves streamed logs and partial output after a timeout. The full
source checkout supplies the shared library for the Bash launcher and direct
Python entry point; no additional runtime dependency or network install is
required. Existing adapters can migrate incrementally to this lifecycle.

See Python's [subprocess documentation](https://docs.python.org/3/library/subprocess.html)
and Microsoft's [Job Objects documentation](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
for the platform mechanisms and their limits.
