import ast
import re
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
