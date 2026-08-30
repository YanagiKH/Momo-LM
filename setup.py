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
    _library_names = (
        "momo_core_rust.dll",
        "libmomo_core_rust.dylib",
        "libmomo_core_rust.so",
    )

    def _required(self) -> bool:
        return os.environ.get("MOMO_REQUIRE_NATIVE", "").lower() in {"1", "true", "yes"}

    @staticmethod
    def _remove(path: Path) -> None:
        path.unlink(missing_ok=True)

    def _clean_cpp_destinations(self, extension: Extension) -> None:
        exact = Path(self.get_ext_fullpath(extension.name))
        directories = {exact.parent, ROOT / "momo_lm"}
        self._remove(exact)
        for directory in directories:
            for suffix in ("*.so", "*.pyd", "*.dylib", "*.dll"):
                for candidate in directory.glob(f"_native{suffix}"):
                    self._remove(candidate)

    def _rust_destination(self) -> Path:
        return Path(self.get_ext_fullpath("momo_lm._native")).parent / "_native_libs"

    def _clean_rust_destinations(self) -> Path:
        destination = self._rust_destination()
        for name in self._library_names:
            self._remove(destination / name)
        source_destination = ROOT / "momo_lm" / "_native_libs"
        for name in self._library_names:
            self._remove(source_destination / name)
        return destination

    def build_extension(self, extension: Extension) -> None:
        self._clean_cpp_destinations(extension)
        try:
            super().build_extension(extension)
        except Exception as exc:
            self._clean_cpp_destinations(extension)
            if self._required():
                raise
            self.warn(f"optional Momo-LM C/C++ backend was skipped: {exc}")
            return
        output = Path(self.get_ext_fullpath(extension.name))
        if not output.is_file():
            self._clean_cpp_destinations(extension)
            message = f"C/C++ compiler did not produce the expected extension: {output}"
            if self._required():
                raise RuntimeError(message)
            self.warn(f"optional Momo-LM C/C++ backend was skipped: {message}")

    def run(self) -> None:
        for extension in self.extensions:
            extension.optional = not self._required()
        try:
            super().run()
            self._build_rust_backend()
        except BaseException:
            if self._required():
                for extension in self.extensions:
                    self._clean_cpp_destinations(extension)
                self._clean_rust_destinations()
            raise

    def _build_rust_backend(self) -> None:
        destination = self._clean_rust_destinations()
        if os.environ.get("MOMO_BUILD_RUST", "1").lower() in {"0", "false", "no"}:
            if self._required():
                raise RuntimeError("MOMO_BUILD_RUST=0 cannot be used with MOMO_REQUIRE_NATIVE=1")
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
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, destination / library.name)
        except (OSError, subprocess.SubprocessError) as exc:
            self._clean_rust_destinations()
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
                "native/src/attention.c",
                "native/src/runtime.cpp",
                "native/src/tensor.c",
            ],
            include_dirs=["native/include"],
            define_macros=[("MOMO_BUILDING_DLL", "1")],
            extra_compile_args=compile_args,
            language="c++",
            optional=True,
        )
    ],
    cmdclass={"build_ext": NativeBuild},
)
