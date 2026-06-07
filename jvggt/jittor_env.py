# Configure MSVC for Jittor on Windows before ``import jittor``.
# See: jittor_utils sets cc_path empty -> downloads bundled msvc (often breaks on compile).

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from typing import Optional


def _prepend_path(directory: str) -> None:
    directory = os.path.normpath(directory)
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if directory not in parts:
        os.environ["PATH"] = directory + (os.pathsep + path if path else "")


def _find_cl_in_path() -> Optional[str]:
    cl = shutil.which("cl")
    if cl and os.path.isfile(cl):
        return os.path.normpath(cl)
    return None


def _find_cl_via_vswhere() -> Optional[str]:
    pf = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return None
    try:
        install = subprocess.check_output(
            [
                vswhere,
                "-latest",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if not install or not os.path.isdir(install):
        return None

    patterns = [
        os.path.join(install, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe"),
        os.path.join(install, "VC", "Tools", "MSVC", "*", "bin", "HostX64", "x64", "cl.exe"),
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    candidates.sort()
    return os.path.normpath(candidates[-1])


def _apply_vcvars64(cl_path: str) -> bool:
    """Run vcvars64.bat and merge INCLUDE/LIB/PATH into the current process."""
    # .../VC/Tools/MSVC/<ver>/bin/Hostx64/x64/cl.exe -> .../VC/Auxiliary/Build/vcvars64.bat
    bin_dir = os.path.dirname(cl_path)
    vc_root = bin_dir
    for _ in range(5):
        vc_root = os.path.dirname(vc_root)
    vcvars = os.path.join(vc_root, "Auxiliary", "Build", "vcvars64.bat")
    if not os.path.isfile(vcvars):
        return False
    try:
        out = subprocess.check_output(
            f'cmd /u /c ""{vcvars}" >nul 2>&1 && set"',
            shell=True,
        )
    except subprocess.CalledProcessError:
        return False
    text = out.decode("utf-16-le", errors="replace")
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.upper() in ("PATH", "INCLUDE", "LIB", "LIBPATH"):
            os.environ[key] = value
    return True


def configure_jittor_compiler(verbose: bool = True) -> bool:
    """
  Must be called before ``import jittor``.

  Sets ``cc_path`` to the system ``cl.exe`` so Jittor skips the bundled msvc.zip.
  """
    if os.name != "nt":
        return True

    if os.environ.get("cc_path"):
        cl = os.environ["cc_path"]
        _prepend_path(os.path.dirname(cl))
        if verbose:
            print(f"[jvggt] Using cc_path from environment: {cl}")
        return True

    cl = _find_cl_in_path() or _find_cl_via_vswhere()
    if not cl:
        if verbose:
            print(
                "[jvggt] WARNING: Visual Studio cl.exe not found.\n"
                "  Install 'Visual Studio 2022 Build Tools' -> C++ desktop development,\n"
                "  then run from 'x64 Native Tools Command Prompt for VS 2022',\n"
                "  or: .\\run_demo_jvggt.ps1"
            )
        return False

    os.environ["cc_path"] = cl
    _prepend_path(os.path.dirname(cl))
    _apply_vcvars64(cl)

    if verbose:
        print(f"[jvggt] Using system MSVC: {cl}")

    return True
