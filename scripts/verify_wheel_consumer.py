"""Run the canonical consumer contract against an isolated installed wheel.

Usage: python scripts/verify_wheel_consumer.py path/to/unified_llm.whl
Only fixtures and the reference consumer are copied, never SDK source.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path


def verify(wheel: Path) -> None:
    """Install one built artifact and exercise the actual consumer tests."""
    wheel = wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        raise ValueError("expected a built .whl artifact")
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="unified-llm-consumer-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-I", "-m", "pip", "install", str(wheel), "pytest>=8,<10", "pytest-asyncio>=0.24,<2"],
            cwd=root,
            check=True,
        )
        # Isolated mode ignores PYTHONPATH/user-site and excludes the current
        # directory, so a source checkout cannot satisfy this package import.
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import pathlib, sys, unified_llm; "
                "p = pathlib.Path(unified_llm.__file__).resolve(); "
                "assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()), p; "
                "assert p.with_name('py.typed').is_file(); "
                "print('Verified installed SDK:', p)",
            ],
            cwd=root,
            check=True,
        )
        for name in ("examples", "tests"):
            (root / name).mkdir()
            shutil.copyfile(repository / name / "__init__.py", root / name / "__init__.py")
        shutil.copyfile(repository / "examples/support_triage.py", root / "examples/support_triage.py")
        shutil.copyfile(repository / "tests/test_consumer_contract.py", root / "tests/test_consumer_contract.py")
        # No repository conftest, editable install, or project pytest config is
        # present. Pytest adds only the copied consumer/fixture packages.
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pytest",
                "tests/test_consumer_contract.py",
                "-q",
                "--strict-config",
                "--strict-markers",
                "-o",
                "asyncio_mode=auto",
            ],
            cwd=root,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    verify(args.wheel)


if __name__ == "__main__":
    main()
