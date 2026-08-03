from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

root = Path(SPECPATH).parent

datas = [
    (str(root / "momo_lm" / "web"), "momo_lm/web"),
    (str(root / "momo_lm" / "assets"), "momo_lm/assets"),
]
binaries = collect_dynamic_libs("momo_lm")

a = Analysis(
    [str(root / "packaging" / "entrypoint.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["momo_lm", "momo_lm._native"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Momo-LM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Momo-LM",
)
