"""Regression contract for live timeline rewind without a native cache."""

import ast
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ROOT / "GPUCloth" / "Cpp_Compatibility" / "operators.py"


class _Vertices:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def __len__(self):
        return self.values.size // 3

    def foreach_get(self, _name, output):
        output[:] = self.values

    def foreach_set(self, _name, values):
        self.values = np.asarray(values, dtype=np.float32).copy()


class _Mesh:
    def __init__(self, values):
        self.vertices = _Vertices(values)

    def update(self):
        pass

    def update_tag(self):
        pass


class _Object:
    def __init__(self, values):
        self.data = _Mesh(values)


class _Depsgraph:
    def update(self):
        pass


def _timeline_helpers():
    tree = ast.parse(OPERATORS.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_store_simulation_frame", "_load_simulation_frame"}
    }
    namespace = {
        "np": np,
        "g_clothOBJs": [],
        "_simulation_frame_state": {"positions": {}},
    }
    module = ast.Module(body=list(functions.values()), type_ignores=[])
    exec(compile(module, str(OPERATORS), "exec"), namespace)
    return namespace


class TimelineRewindTest(unittest.TestCase):
    def test_live_frame_snapshot_restores_requested_geometry(self):
        namespace = _timeline_helpers()
        cloth = _Object([0, 0, 0, 1, 0, 0])
        namespace["g_clothOBJs"].append(cloth)

        namespace["_store_simulation_frame"](7)
        cloth.data.vertices.values += 10

        restored = namespace["_load_simulation_frame"](7, _Depsgraph())

        self.assertTrue(restored)
        np.testing.assert_array_equal(
            cloth.data.vertices.values, [0, 0, 0, 1, 0, 0]
        )


if __name__ == "__main__":
    unittest.main()
