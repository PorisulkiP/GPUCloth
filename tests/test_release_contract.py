import importlib.util
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "GPUCloth" / "blender_manifest.toml"
BUILDER = ROOT / "tools" / "build_blender_release.py"
SMOKE = ROOT / "tests" / "blender_extension_smoke.py"


class ReleaseContractTest(unittest.TestCase):
    def test_blender_42_extension_release_contract(self):
        self.assertTrue(MANIFEST.is_file(), "Blender extension manifest is missing")
        manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest,
            {
                "schema_version": "1.0.0",
                "id": "gpucloth",
                "version": "0.1.3",
                "name": "GPUCloth",
                "tagline": "GPU cloth simulation for Blender",
                "maintainer": "PorisulkiP",
                "type": "add-on",
                "website": "https://github.com/PorisulkiP/GPUCloth",
                "blender_version_min": "4.2.0",
                "license": ["SPDX:AGPL-3.0-only"],
                "platforms": ["windows-x64"],
            },
        )

        self.assertTrue(BUILDER.is_file(), "Release builder is missing")
        spec = importlib.util.spec_from_file_location("build_blender_release", BUILDER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.REQUIRED_CORE_NATIVE,
            (
                "GPUCloth.dll",
                "optix_obstacle_collision.ptx",
                "optix_self_collision.ptx",
            ),
        )
        self.assertEqual(
            module.REQUIRED_PACKAGED_LIBRARIES,
            ("dstorage.dll", "dstoragecore.dll", "pthread.dll"),
        )
        self.assertEqual(
            module.REQUIRED_CUDA_RUNTIME,
            ("cublas64_13.dll", "cusparse64_12.dll", "cusolver64_12.dll"),
        )

        smoke_source = SMOKE.read_text(encoding="utf-8")
        self.assertIn("addon_utils.check", smoke_source)
        self.assertIn("bpy.ops.gpucloth.load_dll", smoke_source)
        self.assertIn("_validate_product_abi", smoke_source)
        self.assertIn("_validate_descriptor_layout", smoke_source)
        self.assertIn("prepare_task_active", smoke_source)
        self.assertIn("prepare_progress", smoke_source)
        self.assertIn("gpucloth_async_ui_probe", smoke_source)
        self.assertNotIn("GPUCloth_v3_runtime_create", smoke_source)
        self.assertNotIn("prepare_simulation", smoke_source)


if __name__ == "__main__":
    unittest.main()
