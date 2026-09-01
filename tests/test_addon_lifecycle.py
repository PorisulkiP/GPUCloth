"""Observable lifecycle contracts for the Blender add-on boundary."""

import ast
from ctypes import c_float, c_uint, sizeof
from pathlib import Path
import unittest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ROOT / "GPUCloth" / "Cpp_Compatibility" / "operators.py"
BRIDGE = ROOT / "GPUCloth" / "Cpp_Compatibility" / "cloth_settings_bridge.py"
PROPERTIES = ROOT / "GPUCloth" / "Cpp_Compatibility" / "properties.py"
UI = ROOT / "GPUCloth" / "Cpp_Compatibility" / "ui.py"


class AddonLifecycleContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.operators = OPERATORS.read_text(encoding="utf-8")
        cls.bridge = BRIDGE.read_text(encoding="utf-8")
        cls.properties = PROPERTIES.read_text(encoding="utf-8")
        cls.ui = UI.read_text(encoding="utf-8")

    def test_create_reuse_and_reject_duplicate_modifier_contract(self):
        namespace = self._functions(
            self.bridge,
            {"find_cpu_cloth_modifier", "find_cpu_cloth_modifiers",
             "select_backend"},
            {"SCHEMA_VERSION": 1,
             "sync_cpu_to_gpu": lambda _obj, _scene: {
                 "errors": [], "unsupported_non_default": []},
             "_store_report": lambda _obj, _report: None,
             "apply_modifier_ownership": lambda _obj, _backend: None},
        )
        class Modifier:
            def __init__(self, kind, name=None):
                self.type = kind
                self.name = name or kind
        class Modifiers(list):
            def new(self, name, type):
                modifier = Modifier(type, name)
                self.append(modifier)
                return modifier
        class Object:
            def __init__(self, modifiers):
                self.name = "Cube"
                self.modifiers = modifiers
                self.GPUCloth = SimpleNamespace(is_active=False)
        zero = Object(Modifiers((Modifier("SUBSURF"),)))
        self.assertEqual(
            namespace["select_backend"](zero, "GPU"),
            {"errors": [], "unsupported_non_default": []})
        created = namespace["find_cpu_cloth_modifiers"](zero)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].name, "Cloth")
        self.assertTrue(zero.GPUCloth.is_active)

        one = Object(Modifiers((Modifier("CLOTH"), Modifier("SUBSURF"))))
        self.assertEqual(
            namespace["select_backend"](one, "GPU"),
            {"errors": [], "unsupported_non_default": []})
        self.assertTrue(one.GPUCloth.is_active)
        self.assertEqual(len(namespace["find_cpu_cloth_modifiers"](one)), 1)

        two = Object(Modifiers((Modifier("CLOTH"), Modifier("CLOTH"))))
        rejected = namespace["select_backend"](two, "GPU")
        self.assertIn("exactly one Cloth modifier", rejected["errors"][0])
        self.assertFalse(two.GPUCloth.is_active)

    def test_deferred_auto_prepare_default_on_off_cancel_contract(self):
        class Timers:
            def __init__(self): self.registered = set()
            def register(self, callback, first_interval): self.registered.add(callback)
            def unregister(self, callback): self.registered.remove(callback)
            def is_registered(self, callback): return callback in self.registered
        timers = Timers()
        bpy = SimpleNamespace(app=SimpleNamespace(timers=timers))
        namespace = self._functions(
            self.operators,
            {"schedule_auto_prepare", "cancel_auto_prepare"},
            {"bpy": bpy, "_run_auto_prepare": object(),
             "_pending_auto_prepare": None,
             "_auto_prepare_timer_registered": False,
             "_stop_requested": True,
             "cancel_prepare_task": lambda: False},
        )
        obj = SimpleNamespace(name_full="Cube", name="Cube")
        scene = SimpleNamespace(name_full="Scene", name="Scene")
        self.assertTrue(namespace["schedule_auto_prepare"](obj, scene))
        self.assertFalse(namespace["_stop_requested"])
        self.assertTrue(namespace["cancel_auto_prepare"]() is None)
        self.assertFalse(timers.registered)

    def test_deferred_timer_reports_owner_and_restores_selection(self):
        class FakeObject:
            def __init__(self, name, selected):
                self.name = name
                self.name_full = name
                self._selected = selected
                self.GPUCloth = SimpleNamespace(
                    execution_backend="GPU", is_active=True,
                    auto_prepare=True)
            def select_get(self): return self._selected
            def select_set(self, selected): self._selected = selected

        class ViewLayer:
            def __init__(self, objects):
                self.objects = type("Objects", (list,), {})(objects)
                self.objects.active = objects[1]

        class Context:
            window = object()
            @staticmethod
            def temp_override(**_kwargs):
                from contextlib import nullcontext
                return nullcontext()

        target = FakeObject("Cube", False)
        other = FakeObject("Other", True)
        objects = [target, other]
        view_layer = ViewLayer(objects)
        helper = SimpleNamespace(memory_preflight_status="")
        scene = SimpleNamespace(
            name="Scene", name_full="Scene", view_layers=[view_layer],
            gpu_cloth_helper=helper)
        calls = []
        bpy = SimpleNamespace(
            data=SimpleNamespace(
                scenes={"Scene": scene}, objects={"Cube": target}),
            context=Context(),
            ops=SimpleNamespace(gpucloth=SimpleNamespace(
                prepare_simulation=lambda: calls.append("prepare"))),
        )
        namespace = self._functions(
            self.operators, {"_run_auto_prepare"},
            {"bpy": bpy, "_pending_auto_prepare": {
                "scene_name": "Scene", "object_name": "Cube"},
             "_auto_prepare_timer_registered": True, "_stop_requested": False,
             "prepare_task_active": lambda: False,
             "_start_prepare_task": lambda _context, automatic=False: (
                 calls.append(("prepare", automatic)) or True)},
        )
        namespace["_run_auto_prepare"]()
        self.assertEqual(calls, [("prepare", True)])
        self.assertIs(view_layer.objects.active, other)
        self.assertFalse(target.select_get())
        self.assertTrue(other.select_get())

        scene.view_layers = []
        namespace = self._functions(
            self.operators, {"_run_auto_prepare"},
            {"bpy": bpy, "_pending_auto_prepare": {
                "scene_name": "Scene", "object_name": "Cube"},
             "_auto_prepare_timer_registered": True, "_stop_requested": False,
             "prepare_task_active": lambda: False,
             "_start_prepare_task": lambda _context, automatic=False: True},
        )
        namespace["_run_auto_prepare"]()
        self.assertIn("ERROR:", helper.memory_preflight_status)

    def test_prepare_click_is_async_with_owned_progress(self):
        execute = self._method(
            self.operators, "GPUCloth_PrepareSimulation", "execute")
        scheduled = {
            node.func.id
            for node in ast.walk(execute)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        native_calls = {
            node.func.attr
            for node in ast.walk(execute)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("_start_prepare_task", scheduled)
        self.assertNotIn("GPUCloth_v3_cloth_build", native_calls)

        calls = []
        namespace = {
            "_start_prepare_task": lambda context, automatic=False: (
                calls.append((context, automatic)) or True),
        }
        exec(compile(ast.Module(body=[execute], type_ignores=[]),
                     "<contract>", "exec"), namespace)
        reports = []
        operator = SimpleNamespace(
            report=lambda level, message: reports.append((level, message)))
        context = object()
        self.assertEqual(
            namespace["execute"](operator, context), {'FINISHED'})
        self.assertEqual(calls, [(context, False)])

        worker = self._function(self.operators, "_run_prepare_native")
        self.assertNotIn(
            "bpy",
            {node.id for node in ast.walk(worker) if isinstance(node, ast.Name)},
        )
        worker_namespace = {}
        exec(compile(ast.Module(body=[worker], type_ignores=[]),
                     "<contract>", "exec"), worker_namespace)
        outcomes = []
        worker_namespace["_run_prepare_native"](
            SimpleNamespace(put=outcomes.append), 7,
            lambda left, right: left + right, (2, 3))
        self.assertEqual(outcomes, [(7, True, 5)])
        self.assertIn("prepare_progress: IntProperty", self.properties)
        self.assertIn("prepare_status: StringProperty", self.properties)
        self.assertIn('helper, "prepare_progress"', self.ui)

    def test_memory_preflight_allow_reject_contract(self):
        namespace = self._functions(
            self.operators,
            {"_estimate_gpu_memory_bytes", "_require_gpu_memory_preflight"},
            {"sizeof": sizeof, "c_float": c_float, "c_uint": c_uint},
        )
        mesh = SimpleNamespace(vertices=[0] * 8, edges=[0] * 4,
                                  polygons=[0] * 2, loops=[0] * 6)
        obj = SimpleNamespace(data=mesh)
        namespace["_query_gpu_memory_bytes"] = lambda: (1024 * 1024, 2048 * 1024)
        self.assertIsNotNone(
            namespace["_require_gpu_memory_preflight"]([obj], ["PD"]))
        namespace["_query_gpu_memory_bytes"] = lambda: (1, 2048 * 1024)
        with self.assertRaisesRegex(RuntimeError, "preflight rejected"):
            namespace["_require_gpu_memory_preflight"]([obj], ["PD"])

    def test_stop_teardown_and_scrub_block_contract(self):
        calls = []
        bpy = SimpleNamespace(ops=SimpleNamespace(screen=SimpleNamespace(
            animation_cancel=lambda: calls.append("animation_cancel"))))
        stop = self._method(self.operators, "GPUCloth_FreeVRAM", "execute")
        namespace = {
            "bpy": bpy,
            "cancel_auto_prepare": lambda: calls.append("cancel_timer"),
            "prepare_task_active": lambda: False,
            "free_gpu_memory": lambda context, shutdown_runtime=False: (
                calls.append(("free", shutdown_runtime)) or True),
            "_stop_requested": False,
        }
        reports = []
        operator = SimpleNamespace(
            report=lambda level, message: reports.append((level, message)))
        exec(compile(ast.Module(body=[stop], type_ignores=[]),
                     "<contract>", "exec"), namespace)
        self.assertEqual(
            namespace["execute"](operator, SimpleNamespace()), {'FINISHED'})
        self.assertTrue(namespace["_stop_requested"])
        self.assertEqual(calls, ["cancel_timer", "animation_cancel",
                                 ("free", True)])

        handler = self._function(self.operators, "_frame_change_handler")
        namespace = {
            "prepare_task_active": lambda: False,
            "_teardown_failure": False, "_stop_requested": True,
        }
        exec(compile(ast.Module(body=[handler], type_ignores=[]),
                     "<contract>", "exec"), namespace)
        self.assertIsNone(namespace["_frame_change_handler"](
            SimpleNamespace(), SimpleNamespace()))

        update = self._method(
            self.operators, "GPUCloth_UpdateSimulation", "execute")
        namespace = {
            "prepare_task_active": lambda: False,
            "_teardown_failure": False, "_stop_requested": True,
        }
        exec(compile(ast.Module(body=[update], type_ignores=[]),
                     "<contract>", "exec"), namespace)
        self.assertEqual(
            namespace["execute"](operator, SimpleNamespace()), {'CANCELLED'})

    @staticmethod
    def _functions(source, names, initial=None):
        tree = ast.parse(source)
        nodes = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        namespace = dict(initial or {})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<contract>", "exec"), namespace)
        return namespace

    @staticmethod
    def _method(source, class_name, method_name):
        tree = ast.parse(source)
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == class_name)
        return next(node for node in cls.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == method_name)

    @staticmethod
    def _function(source, name):
        tree = ast.parse(source)
        return next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)


if __name__ == "__main__":
    unittest.main()
