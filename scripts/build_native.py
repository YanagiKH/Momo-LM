from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and test every Momo-LM native backend")
    parser.add_argument("--release", action="store_true", help="Build optimized native libraries")
    parser.add_argument("--skip-python", action="store_true", help="Skip the CPython extension")
    arguments = parser.parse_args()
    if not arguments.skip_python:
        run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"])
    cargo = shutil.which("cargo")
    if cargo is None:
        raise SystemExit("cargo is required to build the Rust backend")
    cargo_command = [cargo, "test", "--manifest-path", "native/rust/Cargo.toml", "--locked"]
    if arguments.release:
        cargo_command.append("--release")
    run(cargo_command)
    if shutil.which("cmake"):
        configuration = "Release" if arguments.release else "Debug"
        run(["cmake", "-S", "native", "-B", "build/native", f"-DCMAKE_BUILD_TYPE={configuration}"])
        run(["cmake", "--build", "build/native", "--config", configuration])
        run(["ctest", "--test-dir", "build/native", "-C", configuration, "--output-on-failure"])
    print("native toolchain validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
