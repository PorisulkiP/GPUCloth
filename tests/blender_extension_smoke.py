"""Blender-side install, registration, and native ABI smoke gate."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import addon_utils
import bpy


REQUIRED_NATIVE = (
    "GPUCloth.dll",
    "optix_obstacle_collision.ptx",
    "optix_self_collision.ptx",
    "dstorage.dll",
    "dstoragecore.dll",
    "pthread.dll",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: object, message: str) -> None:
    if not condition:
        fail(message)


def parse_args() -> argparse.Namespace:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("active", "absent"), required=True)
    return parser.parse_args(args)


def check_absent(module_name: str) -> None:
    enabled, loaded = addon_utils.check(module_name)
    require(not enabled and not loaded, f"extension remains enabled: {(enabled, loaded)}")
    require(module_name not in sys.modules, "extension module remains imported")
    require(not hasattr(bpy.types.Object, "GPUCloth"), "Object.GPUCloth remains registered")
    require(
        not hasattr(bpy.types.Scene, "gpu_cloth_helper"),
        "Scene.gpu_cloth_helper remains registered",
    )
    print("GPUCLOTH_RELEASE_SMOKE " + json.dumps({"phase": "absent"}))


def check_async_auto_prepare(operators) -> None:
    mesh = bpy.data.meshes.new("GPUClothAsyncSmokeMesh")
    cloth = bpy.data.objects.new("GPUClothAsyncSmoke", mesh)
    bpy.context.scene.collection.objects.link(cloth)
    bpy.context.view_layer.objects.active = cloth
    cloth.select_set(True)
    original_steps = operators.GPUCloth_PrepareSimulation._prepare_steps

    def fake_steps(_runner, _context):
        yield operators._prepare_progress(25, "Capturing Blender inputs")
        yield operators._prepare_native(
            time.sleep, (0.15,), 60, "Building native cloth")
        yield operators._prepare_progress(90, "Finalizing Blender ownership")
        return {'FINISHED'}

    try:
        operators.GPUCloth_PrepareSimulation._prepare_steps = fake_steps
        cloth.GPUCloth.auto_prepare = True
        started = time.perf_counter()
        cloth.GPUCloth.execution_backend = "GPU"
        switch_seconds = time.perf_counter() - started
        require(switch_seconds < 1.0, f"GPU switch blocked for {switch_seconds:.3f}s")
        require(
            operators.auto_prepare_pending(cloth),
            "GPU switch did not queue automatic preparation",
        )

        operators._run_auto_prepare()
        if bpy.app.timers.is_registered(operators._run_auto_prepare):
            bpy.app.timers.unregister(operators._run_auto_prepare)
        require(operators.prepare_task_active(cloth), "async task did not start")

        operators._advance_prepare_task()
        require(
            bpy.context.scene.gpu_cloth_helper.prepare_progress == 25,
            "capture progress was not published",
        )
        operators._advance_prepare_task()
        require(
            bpy.context.scene.gpu_cloth_helper.prepare_state == "BUILDING",
            "native worker phase was not published",
        )
        cloth["gpucloth_async_ui_probe"] = True

        deadline = time.monotonic() + 5.0
        while operators.prepare_task_active() and time.monotonic() < deadline:
            delay = operators._advance_prepare_task()
            if delay:
                time.sleep(min(float(delay), 0.05))
        require(not operators.prepare_task_active(), "async task did not finish")
        helper = bpy.context.scene.gpu_cloth_helper
        require(helper.prepare_state == "READY", f"unexpected state: {helper.prepare_state}")
        require(helper.prepare_progress == 100, "async task did not reach 100 percent")
        require(bool(cloth["gpucloth_async_ui_probe"]), "Blender data probe was lost")
        cloth.GPUCloth.execution_backend = "CPU"
    finally:
        operators.GPUCloth_PrepareSimulation._prepare_steps = original_steps
        operators.cancel_auto_prepare()
        deadline = time.monotonic() + 5.0
        while operators.prepare_task_active() and time.monotonic() < deadline:
            delay = operators._advance_prepare_task()
            if delay:
                time.sleep(min(float(delay), 0.05))
        if bpy.app.timers.is_registered(operators._run_auto_prepare):
            bpy.app.timers.unregister(operators._run_auto_prepare)
        if bpy.app.timers.is_registered(operators._advance_prepare_task):
            bpy.app.timers.unregister(operators._advance_prepare_task)
        bpy.data.objects.remove(cloth, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def check_active(module_name: str, repo_dir: Path) -> None:
    require(bpy.app.version[:2] == (4, 2), f"unexpected Blender: {bpy.app.version}")
    enabled, loaded = addon_utils.check(module_name)
    require(enabled and loaded, f"extension is not active: {(enabled, loaded)}")

    package = importlib.import_module(module_name)
    package_root = Path(package.__file__).resolve().parent
    require(
        package_root.is_relative_to(repo_dir.resolve()),
        f"extension loaded outside isolated repository: {package_root}",
    )
    metadata = addon_utils.module_bl_info(package)
    require(metadata["version"] == (0, 1, 2), "unexpected add-on version")
    require(metadata["blender"] == (4, 2, 0), "unexpected Blender minimum")
    for name in REQUIRED_NATIVE:
        require((package_root / "lib" / name).is_file(), f"missing native payload: {name}")

    require(hasattr(bpy.types.Object, "GPUCloth"), "Object.GPUCloth is not registered")
    require(
        hasattr(bpy.types.Scene, "gpu_cloth_helper"),
        "Scene.gpu_cloth_helper is not registered",
    )
    require(hasattr(bpy.ops.gpucloth, "load_dll"), "load operator is not registered")

    mesh = bpy.data.meshes.new("GPUClothReleaseSmokeMesh")
    cloth = bpy.data.objects.new("GPUClothReleaseSmoke", mesh)
    bpy.context.scene.collection.objects.link(cloth)
    try:
        require(cloth.GPUCloth is not None, "object settings are unavailable")
    finally:
        bpy.data.objects.remove(cloth, do_unlink=True)
        bpy.data.meshes.remove(mesh)

    result = bpy.ops.gpucloth.load_dll()
    require(result == {"FINISHED"}, f"DLL load failed: {result}")
    operators = importlib.import_module(
        module_name + ".Cpp_Compatibility.operators"
    )
    native = operators.g_dll
    require(native is not None, "DLL owner is missing after load")
    exports = tuple(operators._GPUCLOTH_V3_EXPORT_SIGNATURES)
    require(len(exports) == 57, f"unexpected export count: {len(exports)}")
    require(len(set(exports)) == len(exports), "duplicate exported ABI names")
    require(all(name.startswith("GPUCloth_v3_") for name in exports), "invalid ABI name")

    mesh = bpy.data.meshes.new("GPUClothLifecycleMesh")
    cloth = bpy.data.objects.new("GPUClothLifecycle", mesh)
    bpy.context.scene.collection.objects.link(cloth)
    bpy.context.view_layer.objects.active = cloth
    cloth.select_set(True)
    try:
        require(
            not tuple(item for item in cloth.modifiers if item.type == "CLOTH"),
            "smoke object unexpectedly has a Cloth modifier",
        )
        cloth.GPUCloth.auto_prepare = False
        cloth.GPUCloth.execution_backend = "GPU"
        modifiers = tuple(
            item for item in cloth.modifiers if item.type == "CLOTH"
        )
        require(len(modifiers) == 1, "GPU activation did not create one Cloth modifier")
        modifier = modifiers[0]
        require(modifier.name == "Cloth", "created Cloth modifier has unexpected name")
        require(cloth.GPUCloth.is_active, "CPU-to-GPU switch did not commit")
        require(not modifier.show_viewport, "CPU Cloth viewport owner remains active")
        cloth.GPUCloth.execution_backend = "CPU"
        require(not cloth.GPUCloth.is_active, "GPU-to-CPU switch did not commit")
        require(
            cloth.modifiers.get(modifier.name) is not None,
            "created Cloth modifier was removed",
        )
        require(modifier.show_viewport, "CPU Cloth viewport owner was not restored")
    finally:
        bpy.data.objects.remove(cloth, do_unlink=True)
        bpy.data.meshes.remove(mesh)

    check_async_auto_prepare(operators)

    abi = operators._validate_product_abi(native)
    layout = operators._validate_descriptor_layout(native)
    report = {
        "phase": "active",
        "module": module_name,
        "blender": ".".join(str(value) for value in bpy.app.version[:3]),
        "abi": f"{abi.abi_major}.{abi.abi_minor}.{abi.abi_patch}",
        "feature_schema": int(abi.feature_schema_version),
        "feature_count": int(abi.feature_count),
        "descriptor_schema": int(layout.schema_version),
        "exports": len(exports),
        "package_root": str(package_root),
    }
    del native
    result = bpy.ops.gpucloth.unload_dll()
    require(result == {"FINISHED"}, f"DLL unload failed: {result}")
    require(operators.g_dll is None, "DLL owner remains after unload")
    print("GPUCLOTH_RELEASE_SMOKE " + json.dumps(report, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.phase == "absent":
        check_absent(args.module)
    else:
        check_active(args.module, args.repo_dir)


if __name__ == "__main__":
    main()
