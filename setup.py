from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent


class NativeBuild(build_ext):
    def _required(self) -> bool:
        return os.environ.get("MOMO_REQUIRE_NATIVE", "").lower() in {"1", "true", "yes"}

    def build_extension(self, extension: Extension) -> None:
        try:
            super().build_extension(extension)
        except Exception as exc:
            if self._required():
                raise
            self.warn(f"optional Momo-LM C/C++ backend was skipped: {exc}")

    def run(self) -> None:
        try:
            super().run()
        finally:
            self._build_rust_backend()

    def _build_rust_backend(self) -> None:
        if os.environ.get("MOMO_BUILD_RUST", "1").lower() in {"0", "false", "no"}:
            return
        cargo = shutil.which("cargo")
        if cargo is None:
            if self._required():
                raise RuntimeError("cargo is required when MOMO_REQUIRE_NATIVE=1")
            self.warn("optional Momo-LM Rust backend was skipped because cargo was not found")
            return
        manifest = ROOT / "native" / "rust" / "Cargo.toml"
        target = Path(self.build_temp) / "momo-rust-target"
        command = [cargo, "build", "--manifest-path", str(manifest), "--release", "--locked", "--target-dir", str(target)]
        try:
            subprocess.run(command, check=True)
            names = {
                "win32": "momo_core_rust.dll",
                "cygwin": "momo_core_rust.dll",
                "darwin": "libmomo_core_rust.dylib",
            }
            library = target / "release" / names.get(sys.platform, "libmomo_core_rust.so")
            destination = Path(self.get_ext_fullpath("momo_lm._native")).parent / "_native_libs"
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, destination / library.name)
        except (OSError, subprocess.SubprocessError) as exc:
            if self._required():
                raise
            self.warn(f"optional Momo-LM Rust backend was skipped: {exc}")


compile_args = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17", "-fvisibility=hidden"]

setup(
    ext_modules=[
        Extension(
            "momo_lm._native",
            sources=[
                "native/python/module.cpp",
                "native/src/runtime.cpp",
                "native/src/tensor.c",
            ],
            include_dirs=["native/include"],
            define_macros=[("MOMO_BUILDING_DLL", "1")],
            extra_compile_args=compile_args,
            language="c++",
        )
    ],
    cmdclass={"build_ext": NativeBuild},
)
