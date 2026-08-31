#!/usr/bin/env python3
"""Install a release ZIP into an isolated Blender profile and test it twice."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from build_blender_release import REQUIRED_CUDA_RUNTIME


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = Path(
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
)
DEFAULT_SMOKE = ROOT / "tests" / "blender_extension_smoke.py"
REPO_ID = "gpucloth_test"
PACKAGE_ID = "gpucloth"
MODULE_ID = f"bl_ext.{REPO_ID}.{PACKAGE_ID}"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--smoke-script", type=Path, default=DEFAULT_SMOKE)
    return parser.parse_args()


def profile_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    paths = {
        "BLENDER_USER_CONFIG": root / "config",
        "BLENDER_USER_SCRIPTS": root / "scripts",
        "BLENDER_USER_EXTENSIONS": root / "extensions",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def resolve_cuda_runtime(env: dict[str, str]) -> Path:
    candidates = []
    cuda_path = env.get("CUDA_PATH")
    if cuda_path:
        candidates.extend(
            (Path(cuda_path) / "bin" / "x64", Path(cuda_path) / "bin")
        )
    candidates.extend(
        Path(entry)
        for entry in env.get("PATH", "").split(os.pathsep)
        if entry
    )
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        if all((resolved / name).is_file() for name in REQUIRED_CUDA_RUNTIME):
            print(
                "CUDA_RUNTIME "
                + json.dumps(
                    {
                        "directory": str(resolved),
                        "libraries": list(REQUIRED_CUDA_RUNTIME),
                    },
                    sort_keys=True,
                )
            )
            return resolved
    raise RuntimeError(
        "compatible CUDA runtime directory not found; expected together: "
        + ", ".join(REQUIRED_CUDA_RUNTIME)
    )


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    blender = args.blender.resolve()
    smoke = args.smoke_script.resolve()
    for kind, path in (
        ("release package", package),
        ("Blender executable", blender),
        ("smoke script", smoke),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{kind} not found: {path}")
    resolve_cuda_runtime(os.environ.copy())

    with tempfile.TemporaryDirectory(prefix="gpucloth-blender-test-") as temp_name:
        temp_root = Path(temp_name)
        repo_dir = temp_root / "repository"
        repo_dir.mkdir()
        env = profile_environment(temp_root / "profile")

        run(
            [
                str(blender),
                "--factory-startup",
                "--command",
                "extension",
                "validate",
                str(package),
            ],
            env=env,
        )
        run(
            [
                str(blender),
                "--command",
                "extension",
                "repo-add",
                REPO_ID,
                "--name",
                "GPUCloth Test",
                "--directory",
                str(repo_dir),
                "--clear-all",
            ],
            env=env,
        )
        run(
            [
                str(blender),
                "--command",
                "extension",
                "install-file",
                "--repo",
                REPO_ID,
                "--enable",
                str(package),
            ],
            env=env,
        )

        smoke_command = [
            str(blender),
            "--background",
            "--python-exit-code",
            "91",
            "--python",
            str(smoke),
            "--",
            "--module",
            MODULE_ID,
            "--repo-dir",
            str(repo_dir),
            "--phase",
            "active",
        ]
        first = run(smoke_command, env=env)
        second = run(smoke_command, env=env)
        marker = "GPUCLOTH_RELEASE_SMOKE"
        if marker not in first or marker not in second:
            raise RuntimeError("Blender smoke receipt is missing")

        run(
            [
                str(blender),
                "--command",
                "extension",
                "remove",
                PACKAGE_ID,
            ],
            env=env,
        )
        absent = run(
            [
                str(blender),
                "--background",
                "--python-exit-code",
                "91",
                "--python",
                str(smoke),
                "--",
                "--module",
                MODULE_ID,
                "--repo-dir",
                str(repo_dir),
                "--phase",
                "absent",
            ],
            env=env,
        )
        if marker not in absent:
            raise RuntimeError("Blender uninstall receipt is missing")
        if (repo_dir / PACKAGE_ID).exists():
            raise RuntimeError("extension files remain after removal")

    print("BLENDER_RELEASE_GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
