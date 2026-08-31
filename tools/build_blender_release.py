#!/usr/bin/env python3
"""Build and validate the official Blender 4.2 GPUCloth extension archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = ROOT / "GPUCloth"
DEFAULT_BLENDER = Path(
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
)
REQUIRED_CORE_NATIVE = (
    "GPUCloth.dll",
    "optix_obstacle_collision.ptx",
    "optix_self_collision.ptx",
)
REQUIRED_PACKAGED_LIBRARIES = (
    "dstorage.dll",
    "dstoragecore.dll",
    "pthread.dll",
)
REQUIRED_CUDA_RUNTIME = (
    "cublas64_13.dll",
    "cusparse64_12.dll",
    "cusolver64_12.dll",
)
FORBIDDEN_PACKAGE_NAMES = {
    "GPUClothInternal.dll",
    "ClothGeometryTests.exe",
    "gtest.dll",
    "gtest_main.dll",
    "pybindings.py",
    "pycloth.py",
    "tests.py",
}
WINDOWS_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, env: dict[str, str]) -> str:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{subprocess.list2cmdline(command)}"
        )
    return result.stdout


def isolated_blender_environment(profile_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    paths = {
        "BLENDER_USER_CONFIG": profile_root / "config",
        "BLENDER_USER_SCRIPTS": profile_root / "scripts",
        "BLENDER_USER_EXTENSIONS": profile_root / "extensions",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def stage_package(stage: Path, core_build_dir: Path) -> None:
    if not PACKAGE_SOURCE.is_dir():
        raise FileNotFoundError(f"package source not found: {PACKAGE_SOURCE}")
    if not core_build_dir.is_dir():
        raise FileNotFoundError(f"core build directory not found: {core_build_dir}")

    shutil.copytree(
        PACKAGE_SOURCE,
        stage,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")

    lib_dir = stage / "lib"
    for name in REQUIRED_PACKAGED_LIBRARIES:
        path = lib_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"packaged dependency not found: {path}")
    for name in REQUIRED_CORE_NATIVE:
        source = core_build_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"core release payload not found: {source}")
        shutil.copy2(source, lib_dir / name)

    forbidden = sorted(
        str(path.relative_to(stage))
        for path in stage.rglob("*")
        if path.is_file() and path.name in FORBIDDEN_PACKAGE_NAMES
    )
    if forbidden:
        raise RuntimeError(f"forbidden release files: {forbidden}")


def validate_archive_member_name(name: str) -> None:
    path = PurePosixPath(name)
    parts = name.split("/")
    if (
        not name
        or path.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RuntimeError(f"unsafe archive path: {name}")
    invalid_characters = '<>:"|?*'
    for part in parts:
        if (
            part.endswith((" ", "."))
            or any(character in invalid_characters for character in part)
            or any(ord(character) < 32 for character in part)
            or part.split(".", 1)[0].upper() in WINDOWS_DEVICE_NAMES
        ):
            raise RuntimeError(f"unsafe Windows archive path: {name}")


def validate_archive_inventory(archive: Path, stage: Path) -> list[dict[str, object]]:
    expected = sorted(
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file()
    )
    with zipfile.ZipFile(archive) as package:
        entries = [entry for entry in package.infolist() if not entry.is_dir()]
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise RuntimeError("release archive contains duplicate paths")
        for name in names:
            validate_archive_member_name(name)
        if sorted(names) != expected:
            missing = sorted(set(expected) - set(names))
            unexpected = sorted(set(names) - set(expected))
            raise RuntimeError(
                f"release inventory mismatch; missing={missing}; "
                f"unexpected={unexpected}"
            )
        return [
            {
                "path": entry.filename,
                "size": entry.file_size,
                "sha256": hashlib.sha256(package.read(entry)).hexdigest(),
            }
            for entry in sorted(entries, key=lambda item: item.filename)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-build-dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blender = args.blender.resolve()
    core_build_dir = args.core_build_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not blender.is_file():
        raise FileNotFoundError(f"Blender executable not found: {blender}")

    manifest = tomllib.loads(
        (PACKAGE_SOURCE / "blender_manifest.toml").read_text(encoding="utf-8")
    )
    archive = output_dir / (
        f"{manifest['id']}-{manifest['version']}-windows-x64.zip"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()

    with tempfile.TemporaryDirectory(prefix="gpucloth-release-") as temp_name:
        temp_root = Path(temp_name)
        stage = temp_root / "package"
        stage_package(stage, core_build_dir)
        env = isolated_blender_environment(temp_root / "blender-profile")

        run(
            [
                str(blender),
                "--factory-startup",
                "--command",
                "extension",
                "validate",
                str(stage),
            ],
            env=env,
        )
        run(
            [
                str(blender),
                "--factory-startup",
                "--command",
                "extension",
                "build",
                "--source-dir",
                str(stage),
                "--output-filepath",
                str(archive),
            ],
            env=env,
        )
        run(
            [
                str(blender),
                "--factory-startup",
                "--command",
                "extension",
                "validate",
                str(archive),
            ],
            env=env,
        )
        files = validate_archive_inventory(archive, stage)

    package_hash = sha256(archive)
    inventory = {
        "package": archive.name,
        "package_size": archive.stat().st_size,
        "package_sha256": package_hash,
        "external_runtime": {
            "packaged": False,
            "required_libraries": list(REQUIRED_CUDA_RUNTIME),
            "search": ["CUDA_PATH/bin/x64", "CUDA_PATH/bin", "PATH"],
        },
        "files": files,
    }
    inventory_path = archive.with_suffix(".inventory.json")
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive.with_suffix(".zip.sha256").write_text(
        f"{package_hash}  {archive.name}\n",
        encoding="ascii",
    )
    print(f"PACKAGE {archive}")
    print(f"SHA256 {package_hash}")
    print(f"INVENTORY {inventory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
