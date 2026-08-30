from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and test every Momo-LM native backend")
    parser.add_argument("--release", action="store_true", help="Build optimized native libraries")
    parser.add_argument("--skip-python", action="store_true", help="Skip the CPython extension")
    parser.add_argument(
        "--sanitizers",
        action="store_true",
        help="Enable address and undefined-behavior sanitizers for C/C++ tests",
    )
    arguments = parser.parse_args()
    cargo = shutil.which("cargo")
    cmake = shutil.which("cmake")
    if cargo is None:
        raise SystemExit("cargo is required to validate the Rust ABI v2 backend")
    if cmake is None:
        raise SystemExit("cmake is required to validate the C/C++ ABI v2 backend")
    if not arguments.skip_python:
        native_environment = dict(os.environ)
        native_environment["MOMO_REQUIRE_NATIVE"] = "1"
        run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"],
            environment=native_environment,
        )
    run([cargo, "fmt", "--manifest-path", "native/rust/Cargo.toml", "--", "--check"])
    cargo_command = [cargo, "test", "--manifest-path", "native/rust/Cargo.toml", "--locked"]
    if arguments.release:
        cargo_command.append("--release")
    run(cargo_command)
    run(
        [
            cargo,
            "clippy",
            "--manifest-path",
            "native/rust/Cargo.toml",
            "--locked",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]
    )
    configuration = "Release" if arguments.release else "Debug"
    configure = [
        cmake,
        "-S",
        "native",
        "-B",
        "build/native",
        f"-DCMAKE_BUILD_TYPE={configuration}",
    ]
    if arguments.sanitizers:
        configure.append("-DMOMO_ENABLE_SANITIZERS=ON")
    run(configure)
    run([cmake, "--build", "build/native", "--config", configuration])
    run(["ctest", "--test-dir", "build/native", "-C", configuration, "--output-on-failure"])
    print("native toolchain validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
