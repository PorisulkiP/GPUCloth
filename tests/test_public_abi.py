import ast
import os
import re
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "abi" / "GPUClothV3.exports.allowlist"
HEADER = ROOT / "include" / "GPUCloth" / "gpucloth.h"
OPERATORS = ROOT / "GPUCloth" / "Cpp_Compatibility" / "operators.py"
VERSION_UTILS = ROOT / "GPUCloth" / "utils" / "version_compatibility_utils.py"
SIGNATURE_TABLE = "_GPUCLOTH_V3_EXPORT_SIGNATURES"


def read_allowlist():
    rows = []
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ordinal, symbol = line.split()
        rows.append((int(ordinal), symbol))
    return rows


def read_loader_symbols():
    tree = ast.parse(OPERATORS.read_text(encoding="utf-8"), OPERATORS.name)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == SIGNATURE_TABLE
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        return [
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
    raise AssertionError(f"{SIGNATURE_TABLE} not found")


def load_loader_class_without_blender(class_name):
    """Compile one loader class with tiny bpy/global fakes."""
    tree = ast.parse(OPERATORS.read_text(encoding="utf-8"), OPERATORS.name)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    namespace = {
        "bpy": SimpleNamespace(types=SimpleNamespace(Operator=object)),
        "ctypes": SimpleNamespace(),
        "sys": SimpleNamespace(platform="darwin"),
        "subprocess": subprocess,
    }
    module = ast.Module(body=[class_node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), OPERATORS.name, "exec"),
         namespace)
    return namespace[class_name], namespace


class PublicABIContractTest(unittest.TestCase):
    def test_python_sources_parse_without_blender(self):
        sources = sorted((ROOT / "GPUCloth").rglob("*.py"))
        self.assertTrue(sources)
        for source in sources:
            compile(source.read_bytes(), str(source), "exec")

    def test_legacy_loaders_are_not_shipped(self):
        legacy = {
            ROOT / "GPUCloth" / "Cpp_Compatibility" / "pycloth.py",
            ROOT / "GPUCloth" / "Cpp_Compatibility" / "pybindings.py",
            ROOT / "GPUCloth" / "Cpp_Compatibility" / "tests.py",
        }
        self.assertFalse({path for path in legacy if path.exists()})

    def test_allowlist_is_frozen_and_contiguous(self):
        rows = read_allowlist()
        self.assertEqual(len(rows), 57)
        self.assertEqual([ordinal for ordinal, _ in rows], list(range(1, 58)))
        symbols = [symbol for _, symbol in rows]
        self.assertEqual(len(symbols), len(set(symbols)))
        self.assertTrue(all(symbol.startswith("GPUCloth_v3_") for symbol in symbols))

    def test_header_and_loader_match_allowlist_order(self):
        expected = [symbol for _, symbol in read_allowlist()]
        header_text = HEADER.read_text(encoding="utf-8")
        header_symbols = re.findall(
            r"\b(GPUCloth_v3_[A-Za-z0-9_]+)\s*\(", header_text
        )
        self.assertEqual(len(header_symbols), len(expected))
        self.assertEqual(set(header_symbols), set(expected))
        self.assertEqual(read_loader_symbols(), expected)

    def test_loader_uses_only_packaged_native_library(self):
        source = VERSION_UTILS.read_text(encoding="utf-8")
        self.assertIn("os.path.join(lib_dir, lib_filename)", source)
        self.assertNotIn('"..", "build"', source)
        self.assertNotIn('"src", "build"', source)

    def test_loader_has_no_unconditional_windll_import(self):
        tree = ast.parse(OPERATORS.read_text(encoding="utf-8"), OPERATORS.name)
        ctypes_imports = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "ctypes"
            for alias in node.names
        ]
        self.assertNotIn("windll", ctypes_imports)

    def test_mocked_darwin_loader_skips_cuda_and_reuses_handle(self):
        loader_class, namespace = load_loader_class_without_blender(
            "GPUCloth_LoadDLL")
        namespace["sys"].platform = "darwin"
        fake_os = SimpleNamespace(
            path=os.path,
            pathsep=os.pathsep,
            environ=dict(os.environ),
        )
        native_path = "/package/GPUCloth.dylib"
        namespace["os"] = fake_os
        namespace["vcu"] = SimpleNamespace(
            get_dll_path=lambda _name: native_path,
            get_lib_directory=lambda: "/package",
        )
        native = object()
        load = mock.Mock(return_value=native)
        namespace["cdll"] = SimpleNamespace(LoadLibrary=load)
        namespace["_bind_gpucloth_v3_exports"] = mock.Mock()
        namespace["_validate_product_abi"] = mock.Mock()
        namespace["_validate_descriptor_layout"] = mock.Mock()
        namespace["g_dll"] = None
        namespace["_dll_directory_handles"] = []
        operator = loader_class()
        operator.report = mock.Mock()

        with mock.patch.object(subprocess, "run") as nvidia_probe:
            self.assertTrue(operator._platform_gate())
            self.assertTrue(operator.load_dll())
            self.assertTrue(operator.load_dll())
        nvidia_probe.assert_not_called()
        load.assert_called_once_with(native_path)
        self.assertIs(namespace["g_dll"], native)
        self.assertEqual(fake_os.environ, dict(os.environ))

    def test_mocked_windows_loader_uses_cuda_gate_and_search_path(self):
        loader_class, namespace = load_loader_class_without_blender(
            "GPUCloth_LoadDLL")
        namespace["sys"].platform = "win32"
        fake_env = dict(os.environ)
        fake_os = SimpleNamespace(
            path=os.path,
            pathsep=os.pathsep,
            environ=fake_env,
        )
        native_path = "/package/GPUCloth.dll"
        namespace["os"] = fake_os
        namespace["vcu"] = SimpleNamespace(
            get_dll_path=lambda _name: native_path,
            get_lib_directory=lambda: "/package",
        )
        namespace["subprocess"] = SimpleNamespace(
            PIPE=subprocess.PIPE,
            run=mock.Mock(return_value=SimpleNamespace(
                stdout="CUDA Version: 12.0", stderr="")),
        )
        native = object()
        load = mock.Mock(return_value=native)
        namespace["cdll"] = SimpleNamespace(LoadLibrary=load)
        namespace["_bind_gpucloth_v3_exports"] = mock.Mock()
        namespace["_validate_product_abi"] = mock.Mock()
        namespace["_validate_descriptor_layout"] = mock.Mock()
        namespace["g_dll"] = None
        namespace["_dll_directory_handles"] = []
        operator = loader_class()
        operator.report = mock.Mock()

        self.assertTrue(operator.load_dll())
        namespace["subprocess"].run.assert_called_once()
        self.assertTrue(fake_env["PATH"].startswith(
            os.path.dirname(native_path) + os.pathsep))
        load.assert_called_once_with(native_path)

    def test_mocked_posix_unload_keeps_persistent_handle(self):
        loader_class, namespace = load_loader_class_without_blender(
            "GPUCloth_UnloadDLL")
        native = object()
        namespace["g_dll"] = native
        namespace["_dll_directory_handles"] = []
        namespace["free_gpu_memory"] = mock.Mock(return_value=True)
        operator = loader_class()
        operator.report = mock.Mock()

        self.assertEqual(operator.execute(None), {"FINISHED"})
        self.assertIs(namespace["g_dll"], native)
        namespace["free_gpu_memory"].assert_called_once_with(
            None, shutdown_runtime=True)

    def test_mocked_windows_unload_releases_handle(self):
        loader_class, namespace = load_loader_class_without_blender(
            "GPUCloth_UnloadDLL")
        namespace["sys"].platform = "win32"
        native = SimpleNamespace(_handle=123)
        free_library = mock.Mock(return_value=1)
        win_dll = mock.Mock(return_value=SimpleNamespace(
            FreeLibrary=free_library))
        namespace["g_dll"] = native
        namespace["_dll_directory_handles"] = []
        namespace["free_gpu_memory"] = mock.Mock(return_value=True)
        namespace["c_void_p"] = lambda value: value
        namespace["ctypes"] = SimpleNamespace(
            WinDLL=win_dll,
            WinError=RuntimeError,
        )
        namespace["_close_dll_directories"] = mock.Mock()
        operator = loader_class()
        operator.report = mock.Mock()

        self.assertEqual(operator.execute(None), {"FINISHED"})
        win_dll.assert_called_once_with("kernel32")
        free_library.assert_called_once_with(123)
        self.assertIsNone(namespace["g_dll"])


if __name__ == "__main__":
    unittest.main()
