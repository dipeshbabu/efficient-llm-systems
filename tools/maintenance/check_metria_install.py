"""Smoke-test an installed root distribution with an isolated Python interpreter.

Run with the target environment's Python and -I, from outside the source tree.
Only the standard library and installed Metria distribution are required.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from importlib.metadata import distribution
from pathlib import Path

import metria


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    dist = distribution("metria")
    assert dist.metadata["Name"] == "metria"
    assert dist.version == metria.__version__ == args.version
    assert dist.metadata["License-Expression"] == "Apache-2.0"
    assert dist.metadata["Requires-Python"] == "<3.15,>=3.10"
    assert not dist.requires, "root package must not install an inference stack"
    files = {str(path).replace("\\", "/") for path in (dist.files or ())}
    assert any(p.endswith(".dist-info/licenses/LICENSE") for p in files)
    assert any(p.endswith(".dist-info/licenses/NOTICE") for p in files)
    assert Path(metria.__file__).parent == Path(str(dist.locate_file("metria")))
    assert callable(metria.verify_recipe)
    assert callable(metria.execute_study)
    cli = Path(sys.executable).with_name(
        "metria.exe" if sys.platform == "win32" else "metria"
    )
    with tempfile.TemporaryDirectory(prefix="metria-install-") as directory:
        for arguments, expected in [
            (["--version"], f"metria {args.version}"),
            (["--help"], "verify"),
            (["verify", "--help"], "--output"),
            (["recipe", "--help"], "validate"),
            (["compare", "--help"], "compare"),
        ]:
            result = subprocess.run(
                [str(cli), *arguments],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
            assert expected in result.stdout, result.stdout
    print(f"Installed Metria {args.version}: metadata, SDK, and CLI passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
