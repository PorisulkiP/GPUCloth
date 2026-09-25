"""Observable lifecycle contracts for the Blender add-on boundary."""

import ast
from ctypes import c_float, c_uint, sizeof
import os
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "src" / "python"
OPERATORS = ADDON / "Cpp_Compatibility" / "operators.py"
BRIDGE = ADDON / "Cpp_Compatibility" / "cloth_settings_bridge.py"
PROPERTIES = ADDON / "Cpp_Compatibility" / "properties.py"
UI = ADDON / "Cpp_Compatibility" / "ui.py"


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
             "cancel_prepare_task": lambda: False,
             # The pending request is named through `_live_name`, which answers a
             # Blender ID's name or the fallback for one already freed.  Only the
             # live case is reachable here - both IDs are the test's own
             # namespaces - so the stub carries that half of the contract.
             "_live_name": lambda value, fallback=None: (
                 getattr(value, "name_full", None)
                 or getattr(value, "name", None) or fallback)},
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
                 calls.append(("prepare", automatic)) or True),
             # No live drape sandbox: the deferred prepare is held while one is
             # open, because a rebuild takes the sandbox's native owner with it.
             # This contract is about the timer, so the sandbox question is
             # answered "none" the same way `prepare_task_active` is.
             "live_drape_sandbox_owner": lambda: None},
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
             "_start_prepare_task": lambda _context, automatic=False: True,
             "live_drape_sandbox_owner": lambda: None},
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

    def test_blender_default_collider_surface_contract_is_normalized(self):
        one_sided = 1
        two_sided = 2
        namespace = self._functions(
            self.operators,
            {"_resolve_collider_surface_contract"},
            {"CType": SimpleNamespace(
                GPUCLOTH_COLLIDER_ONE_SIDED_NORMAL=one_sided,
                GPUCLOTH_COLLIDER_TWO_SIDED=two_sided)},
        )
        resolve = namespace["_resolve_collider_surface_contract"]
        self.assertEqual(
            resolve(SimpleNamespace(use_culling=True, use_normal=False)),
            one_sided,
        )
        self.assertEqual(
            resolve(SimpleNamespace(use_culling=True, use_normal=True)),
            one_sided,
        )
        self.assertEqual(
            resolve(SimpleNamespace(use_culling=False, use_normal=False)),
            two_sided,
        )
        self.assertEqual(
            resolve(SimpleNamespace(use_culling=False, use_normal=True)),
            two_sided,
        )

    def test_stop_teardown_and_scrub_block_contract(self):
        stop = self._method(self.operators, "GPUCloth_FreeVRAM", "execute")

        def run_stop(release_ok, playing=False, preparing=False):
            """Stop's effects, with the release either succeeding or failing.

            ``playing`` and ``preparing`` are the two states a drive can still be
            in when Stop runs, and each is the reason not to declare the session
            stopped yet.
            """
            calls, sandbox_aborts, reports = [], [], []
            # The shipped build asks ``bpy.ops.screen.animation_cancel()``
            # directly here; the namespace carries it so the baseline arm of the
            # A/B runs its own path and fails on the difference, not on a name.
            bpy = SimpleNamespace(ops=SimpleNamespace(screen=SimpleNamespace(
                animation_cancel=lambda: calls.append("animation_cancel"))))
            namespace = {
                "bpy": bpy,
                "cancel_auto_prepare": lambda: calls.append("cancel_timer"),
                "prepare_task_active": lambda: preparing,
                "free_gpu_memory": lambda context, shutdown_runtime=False: (
                    calls.append(("free", shutdown_runtime)) or release_ok),
                "_stop_requested": False,
                # Stop asks the drive down through the add-on's own helper - the
                # one ask there is - and the helper reports whether the drive was
                # still up to ask.
                "_stop_animation_playback": (
                    lambda: calls.append("ask_drive") or playing),
                "_animation_is_playing": lambda context: playing,
                # Stop closes a live drape sandbox before freeing the owners: the
                # sandbox owns the Begin snapshot that undoes its preview pose,
                # and the native owner integrates the mesh in place, so that pose
                # can only be restored while the owner holding it still exists.
                "abort_live_drape_sandbox": (
                    lambda context: sandbox_aborts.append(context)),
            }
            operator = SimpleNamespace(
                report=lambda level, message: reports.append((level, message)))
            exec(compile(ast.Module(body=[stop], type_ignores=[]),
                         "<contract>", "exec"), namespace)
            context = SimpleNamespace()
            result = namespace["execute"](operator, context)
            return namespace, result, calls, sandbox_aborts, reports, context

        namespace, result, calls, sandbox_aborts, reports, context = run_stop(True)
        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(sandbox_aborts, [context])
        self.assertEqual(calls, ["cancel_timer", "ask_drive", ("free", True)])
        # The dead-man switch is cleared only once the release has succeeded and
        # the drive has answered.  Leaving it set past a successful Stop is what
        # made "cannot start the simulation even after Stop" permanent, so this
        # half is asserted; the failed-release half below asserts that it stays
        # set, and the drive-still-up half asserts the same for a stop that is
        # still owed.
        self.assertFalse(namespace["_stop_requested"])
        self.assertEqual([level for level, _ in reports], [{'INFO'}])

        namespace, result, calls, _, reports, _ = run_stop(False)
        self.assertEqual(result, {'CANCELLED'})
        self.assertTrue(namespace["_stop_requested"])
        self.assertEqual([level for level, _ in reports], [{'ERROR'}])

        # Defect 3, first leg: the drive has been asked but has not answered -
        # the cancel is answered only after the steps already dispatched have run
        # (measured: 1.054 s after a scrub).  Those steps are frame changes, and
        # the switch is what keeps the frame path from solving them once the
        # release has put no owner in the way.  Clearing it here is what made
        # Stop non-terminal.
        namespace, result, calls, _, reports, _ = run_stop(True, playing=True)
        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(calls, ["cancel_timer", "ask_drive", ("free", True)])
        self.assertTrue(namespace["_stop_requested"])

        # Defect 3, second leg: a preparation in flight.  Stop's cancel of a
        # prepare is cooperative and cannot finish a native dispatch, so the
        # release is attempted and the session is not declared stopped - the
        # prepare clears the switch itself when it lands, which it no longer
        # does, and the owner's next Prepare is what starts the simulation again.
        namespace, result, calls, _, reports, _ = run_stop(
            True, playing=True, preparing=True)
        self.assertEqual(result, {'FINISHED'})
        self.assertEqual(calls, ["cancel_timer", "ask_drive"])
        self.assertTrue(namespace["_stop_requested"])

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
        stopped = []
        self.assertEqual(
            namespace["execute"](
                SimpleNamespace(
                    report=lambda level, message: stopped.append((level, message))),
                SimpleNamespace()),
            {'CANCELLED'})

    def test_prepared_rest_pose_is_not_a_retained_native_owner(self):
        """The prepared rest pose outlives the owners, so it cannot count as one.

        `_reset_owner_python_state` deliberately keeps `_initial_positions` - the
        pose a successful prepare validated and the pose a refused rebuild is
        recovered from - and both teardown paths retire their owners through it.
        While the retained-owner predicate named that pose, the state the gate's
        teardown actually reaches - no native owner left, rest pose still held -
        read as "native owners retained".

        The contract has both halves and each is asserted: the pose alone is not
        a retained owner, and every owner the native teardown really retires
        still is one.
        """
        owners = {
            "g_cloth_handles": [object()],
            "_cloth_input_owners": [object()],
            "_readback_owners": [object()],
            "g_clothOBJs": [object()],
            "g_simulationOBJs": [object()],
            "g_clothCollisionOBJs": [object()],
            "g_proxy_handles": [object()],
            "_collision_keepalive": {"owner": object()},
            "_solver_diagnostics": {"owner": object()},
            "_pin_snapshot_states": [object()],
            "_dynamic_mesh_states": [object()],
            "_collection_snapshots": [object()],
            "_effector_weight_states": [object()],
            "_collider_history": {"owner": object()},
        }
        empty = {name: type(value)() for name, value in owners.items()}

        def predicate(initial_positions, runtime=0, cache=0, retained=None):
            namespace = dict(empty)
            namespace.update({
                "_runtime_handle_value": lambda: runtime,
                "_cache_handle_value": lambda: cache,
                "_initial_positions": list(initial_positions),
            })
            namespace.update(retained or {})
            return self._functions(
                self.operators, {"_runtime_owners_retained"}, namespace,
            )["_runtime_owners_retained"]

        self.assertFalse(predicate(())(), "an empty session retains nothing")
        self.assertFalse(
            predicate(((0.0, 0.0, 0.0),))(),
            "the prepared rest pose holds no native resource and no native "
            "call releases it, so it is not a retained native owner")
        self.assertTrue(predicate((), runtime=0x1001)())
        self.assertTrue(predicate((), cache=0x1002)())
        for name, value in owners.items():
            self.assertTrue(
                predicate((), retained={name: value})(),
                f"{name} must still count as a retained native owner")

    def test_unregister_asks_for_a_free_only_while_it_can_be_performed(self):
        """Teardown refuses on retained owners, never on state it cannot release.

        The gate tears down in the order free -> `gpucloth.unload_dll` ->
        `addon.unregister()`, so unregister runs with no DLL left.  Asking
        `free_gpu_memory` for a release there is a request that can never be
        satisfied, and the refusal it prints names owners that do not exist
        (`RuntimeError: GPUCloth unregister blocked by retained native owners`,
        scenario `bending_sdb_differential`).  Asking is still correct - and the
        answer is still a refusal - while a native owner really is retained with
        no DLL to free it.
        """
        def teardown(runtime=0, initial_positions=(), freed=True):
            calls = []
            namespace = {
                "prepare_task_active": lambda: False,
                "_teardown_failure": False,
                "_runtime_handle_value": lambda: runtime,
                "_cache_handle_value": lambda: 0,
                "_initial_positions": list(initial_positions),
                "free_gpu_memory": lambda shutdown_runtime=False: (
                    calls.append(shutdown_runtime) or freed),
            }
            for name, value in (
                    ("g_cloth_handles", []), ("_cloth_input_owners", []),
                    ("_readback_owners", []), ("g_clothOBJs", []),
                    ("g_simulationOBJs", []), ("g_clothCollisionOBJs", []),
                    ("g_proxy_handles", []), ("_collision_keepalive", {}),
                    ("_solver_diagnostics", {}), ("_pin_snapshot_states", []),
                    ("_dynamic_mesh_states", []), ("_collection_snapshots", []),
                    ("_effector_weight_states", []), ("_collider_history", {})):
                namespace[name] = value
            functions = self._functions(
                self.operators,
                {"_runtime_owners_retained", "ensure_native_teardown"},
                namespace,
            )
            namespace.update(functions)
            return namespace["ensure_native_teardown"], calls

        # The gate's order, after the DLL is gone: nothing native is retained, so
        # the fast path answers without asking for a free the DLL cannot perform.
        ensure, calls = teardown(initial_positions=((0.0, 0.0, 0.0),))
        self.assertTrue(ensure(shutdown_runtime=True))
        self.assertEqual(calls, [])

        # A native owner really is retained with no DLL: the free is attempted,
        # and its refusal is still unregister's refusal.
        ensure, calls = teardown(runtime=0x1001, freed=False)
        self.assertFalse(ensure(shutdown_runtime=True))
        self.assertEqual(calls, [True])

        # The same owner, with a free that can be performed.
        ensure, calls = teardown(runtime=0x1001)
        self.assertTrue(ensure(shutdown_runtime=True))
        self.assertEqual(calls, [True])

    def test_clear_cache_reaches_the_store_the_session_does_not_report(self):
        """Defect 7: a store exists, the session reports nothing, nothing clears it.

        The owner's report: "Старый кэш нельзя удалить без пересоздания ткани,
        так как кнопки для его удаления и менеджмента нет во вкладке Cache".
        Measured on the shipped build: after a live walk over frames 2..6 the
        add-on held a reached state per frame while ``is_baked``,
        ``is_outdated``, ``is_frame_skip`` and ``cached_frame_count`` all stayed
        at their empty values, and a store an earlier run left sat in the cache
        directory.  Neither gate saw that state - the panel drew the button only
        inside ``if scene_s.is_baked:`` and ``poll()`` asked the native flags
        only - so there was no way to delete it at all.

        The observable contract: with the store's files on disk and nothing in
        the session reporting them, the operator is offered, and running it
        removes exactly the store's files while dropping the in-memory past the
        same session reached.  A file that is not the store's is left alone.
        """
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            frames = []
            for frame in (2, 3, 4):
                path = cache / f"frame_{frame:06d}.bin"
                path.write_bytes(struct.pack("<3f", 1.0, 2.0, 3.0))
                frames.append(path)
            status = cache / "gpucloth_cache_status.bin"
            status.write_bytes(b"status")
            unrelated = cache / "notes.txt"
            unrelated.write_text("not the store's file", encoding="utf-8")

            helper = SimpleNamespace(
                cache_dir=str(cache) + os.sep, cache_index=0,
                cache_name="GPUCloth", use_external_cache=False,
                external_cache_dir="", cached_frame_count=0, is_baked=False,
                is_outdated=False, is_frame_skip=False, is_baking=False,
                bake_progress=17, playback_mode=True)
            scene = SimpleNamespace(gpu_cloth_helper=helper)
            retained = {2: ("a",), 3: ("b",)}
            store = self._optional_functions(
                self.operators,
                {"_cache_store_files", "cache_store_summary",
                 "_delete_cache_store"},
                {"os": os,
                 "_active_cache_path": lambda _scene: str(cache),
                 "_simulation_frame_state": {"positions": retained},
                 # The store's own file names, as the native clear owns them
                 # (cache.cu:582-627): the frame payloads and the status file,
                 # with the temporary its atomic write leaves behind.
                 "_CACHE_FRAME_PREFIX": "frame_",
                 "_CACHE_FRAME_SUFFIX": ".bin",
                 "_CACHE_STATUS_FILE": "gpucloth_cache_status.bin",
                 "_CACHE_STATUS_TEMPORARY": "gpucloth_cache_status.bin.tmp"})
            if "cache_store_summary" in store:
                summary = store["cache_store_summary"](scene)
                self.assertEqual(summary["disk_files"], 4)
                self.assertEqual(summary["path"], str(cache))
                self.assertEqual(summary["retained_states"], 2)
                self.assertTrue(summary["has_store"])

            poll = self._method(self.operators, "GPUCloth_FreeCache", "poll")
            context = SimpleNamespace(
                scene=scene, object=SimpleNamespace(type='MESH'))
            namespace = dict(store)
            # The baseline's poll asks the native status only, so it needs a DLL
            # to be present; the fixed one asks the store and never reads it.
            namespace.setdefault("g_dll", object())
            exec(compile(ast.Module(body=[poll], type_ignores=[]),
                         "<contract>", "exec"), namespace)
            # ``poll`` is a classmethod, so ask it the way Blender does: bound
            # to the operator class, with the context.
            self.assertTrue(
                namespace["poll"].__get__(None, object)(context),
                "a store exists but Clear Cache is not offered")

            calls = []

            def clear_retained():
                calls.append("retained")
                retained.clear()

            def native_clear(_runtime_handle, _cache_handle):
                calls.append("native")
                return 0

            execute = self._method(
                self.operators, "GPUCloth_FreeCache", "execute")
            namespace = dict(store)
            namespace.update({
                "CType": SimpleNamespace(GPUCLOTH_ABI_OK=0),
                "g_dll": SimpleNamespace(GPUCloth_v3_cache_clear=native_clear),
                "g_runtime_handle": 0,
                "_cache_handle_value": lambda: 9,
                "_cache_handle_owner": lambda: 9,
                "_simulation_frame_state": {"positions": retained},
                "_clear_retained_frames": clear_retained,
                "_sync_cache_status": lambda _scene: None,
            })
            exec(compile(ast.Module(body=[execute], type_ignores=[]),
                         "<contract>", "exec"), namespace)
            reports = []
            operator = SimpleNamespace(
                report=lambda level, message: reports.append((level, message)))
            result = namespace["execute"](operator, context)

            self.assertEqual(calls, ["native", "retained"])
            for path in frames + [status]:
                self.assertFalse(
                    path.exists(), f"{path.name} survived Clear Cache")
            self.assertTrue(unrelated.exists())
            self.assertEqual(helper.bake_progress, 0)
            self.assertFalse(helper.playback_mode)
            self.assertEqual(result, {'FINISHED'})
            self.assertEqual([level for level, _ in reports], [{'INFO'}])

            # The same store again, this time with no live cache owner at all -
            # the state the owner's own stale cache is in.  The disk delete is
            # then the whole answer, and the status read that follows cannot
            # succeed.  Corrected contract (integrator, on the integrated tree):
            # the operator is judged by whether anything is left, so a refresh
            # that cannot read an owner which no longer exists is the *finished*
            # clear it is - the panel offered the row through the reached states,
            # the past is dropped, the files are gone, and a runtime owner that
            # died took its frames with it.  Measured before the correction: two
            # acceptance arms turned exactly this state into an error dialog,
            # `RuntimeError: Error: Cache status refresh failed: v3 cache owner is
            # not live` (build/r23-acceptance/scenario-acc-live.json,
            # -acc-pause.json), over a store that had just been cleared.
            for path in frames + [status]:
                path.write_bytes(b"stale")
            calls.clear()
            namespace["g_dll"] = None
            namespace["_cache_handle_value"] = lambda: 0
            namespace["_sync_cache_status"] = lambda _scene: (
                _ for _ in ()).throw(RuntimeError("v3 cache owner is not live"))
            reports.clear()
            helper.is_baked = True
            helper.cached_frame_count = 3
            result = namespace["execute"](operator, context)

            self.assertEqual(calls, ["retained"])
            for path in frames + [status]:
                self.assertFalse(path.exists())
            self.assertTrue(unrelated.exists())
            self.assertFalse(helper.is_baked)
            self.assertEqual(helper.cached_frame_count, 0)
            self.assertEqual(result, {'FINISHED'})
            self.assertEqual([level for level, _ in reports], [{'INFO'}])
            if "cache_store_summary" in store:
                self.assertFalse(store["cache_store_summary"](scene)["has_store"])

            # And the failure that is still a failure: the same dead owner, but
            # this time the disk delete could not remove what the row named, so
            # the store is still there after the attempt.  The operator must not
            # claim a clear it did not perform.
            namespace["_delete_cache_store"] = lambda _scene: (
                0, 0, ["frame_000002.bin"])
            for path in frames + [status]:
                path.write_bytes(b"stale")
            reports.clear()
            helper.is_baked = True
            helper.cached_frame_count = 3
            result = namespace["execute"](operator, context)

            self.assertEqual(result, {'CANCELLED'})
            self.assertIn("Cache status refresh failed", reports[0][1])
            if "cache_store_summary" in store:
                self.assertTrue(store["cache_store_summary"](scene)["has_store"])

    def test_clear_cache_asks_the_native_owner_and_reports_what_it_deleted(self):
        """The other half of defect 7: the native store, and the report.

        A live cache owner holds frames this add-on cannot free by itself, so it
        is asked first and its refusal is still a refusal.  When it answers OK
        the disk store goes with it and the report states how much was removed -
        the panel offers the same numbers, so the button is never a bare button
        over an unnamed amount of data.
        """
        asked = []

        def clear(runtime_handle, cache_handle):
            asked.append((runtime_handle, cache_handle))
            return 0

        deleted = []
        helper = SimpleNamespace(
            cache_dir="", cache_index=0, cache_name="GPUCloth",
            use_external_cache=False, cached_frame_count=4, is_baked=False,
            is_outdated=True, is_frame_skip=False, is_baking=False,
            bake_progress=50, playback_mode=True)
        scene = SimpleNamespace(gpu_cloth_helper=helper)
        reports = []
        operator = SimpleNamespace(
            report=lambda level, message: reports.append((level, message)))
        context = SimpleNamespace(
            scene=scene, object=SimpleNamespace(type='MESH'))
        execute = self._method(self.operators, "GPUCloth_FreeCache", "execute")
        namespace = {
            "CType": SimpleNamespace(GPUCLOTH_ABI_OK=0),
            "g_dll": SimpleNamespace(GPUCloth_v3_cache_clear=clear),
            "g_runtime_handle": 7,
            "_cache_handle_value": lambda: 9,
            "_cache_handle_owner": lambda: 9,
            "cache_store_summary": lambda _scene: {
                "path": None, "disk_files": 0, "disk_bytes": 0,
                "native_frames": 4, "flags": ("is_outdated",),
                "retained_states": 0, "has_store": True},
            "_cache_store_files": lambda _scene: (None, []),
            "_delete_cache_store": lambda scene: (
                deleted.append(scene) or (0, 0, [])),
            "_clear_retained_frames": lambda: None,
            "_sync_cache_status": lambda _scene: None,
        }
        exec(compile(ast.Module(body=[execute], type_ignores=[]),
                     "<contract>", "exec"), namespace)
        self.assertEqual(namespace["execute"](operator, context), {'FINISHED'})
        self.assertEqual(asked, [(7, 9)])
        self.assertEqual(deleted, [scene])
        self.assertEqual(helper.bake_progress, 0)
        self.assertFalse(helper.playback_mode)
        self.assertEqual([level for level, _ in reports], [{'INFO'}])
        self.assertIn("0.00", reports[0][1])

        namespace["g_dll"] = SimpleNamespace(
            GPUCloth_v3_cache_clear=lambda runtime, handle: 2)
        reports.clear()
        self.assertEqual(namespace["execute"](operator, context), {'CANCELLED'})
        self.assertEqual([level for level, _ in reports], [{'ERROR'}])

    def test_cache_panel_offers_delete_and_names_what_it_deletes(self):
        """The panel owns the row's visibility and its wording.

        Drawn from the state the old row could not reach - unbaked, with a store
        - it must emit the Clear Cache operator *and* a line naming what will be
        deleted.  With no store at all it must emit neither, because a button
        offered over nothing is the same defect mirrored.
        """
        # The function body resolves ``_t`` from its own globals, so the two
        # language arms are two instances of the same extracted source.
        english = {"_t": lambda text, russian=None: text}
        russian = {"_t": lambda text, russian=None: russian}
        detail = self._optional_function(self.ui, "_cache_store_detail", english)
        store = {"path": r"D:\cache\gpucloth_cache", "disk_files": 6,
                 "disk_bytes": 3 * 1024 * 1024, "native_frames": 0,
                 "flags": (), "retained_states": 4, "has_store": True}
        panel_class = self._class(self.ui, "GPUCLOTH_PT_cache")
        operators_module = SimpleNamespace(
            cache_store_summary=lambda _scene: dict(store))
        panel_namespace = {
            "bpy": SimpleNamespace(types=SimpleNamespace(Panel=object)),
            "operators": operators_module,
            "_t": lambda english_text, russian_text=None: english_text,
            "_icon": lambda name: name,
        }
        if detail is not None:
            panel_namespace["_cache_store_detail"] = detail
        exec(compile(ast.Module(body=[panel_class], type_ignores=[]),
                     "<contract>", "exec"), panel_namespace)
        panel = panel_namespace["GPUCLOTH_PT_cache"]
        scene_s = SimpleNamespace(
            cache_dir="", bake_start=1, bake_end=250, is_baked=False,
            bake_progress=0, playback_mode=False, is_baking=False)
        context = SimpleNamespace(
            scene=SimpleNamespace(gpu_cloth_helper=scene_s),
            object=SimpleNamespace(type='MESH'))
        layout = self._recording_layout()
        panel.draw(SimpleNamespace(layout=layout), context)
        drawn = [row["idname"] for row in self._drawn_rows(layout)]
        self.assertIn("gpucloth.free_cache", drawn)
        self.assertIn("gpucloth.bake_simulation", drawn)
        detail_labels = [text for text in layout.labels
                         if "MiB" in text and "6" in text]
        self.assertTrue(detail_labels, layout.labels)
        self.assertIn("D:\\cache\\gpucloth_cache", detail_labels[0])
        self.assertIn("4", detail_labels[0])

        # The Russian half of the same line, so the panel stays bilingual.
        detail_ru = self._optional_function(self.ui, "_cache_store_detail", russian)
        if detail_ru is not None:
            self.assertIn("Очистить", detail_ru(store))

        # Nothing to delete: no row, and no line describing one.
        operators_module.cache_store_summary = lambda _scene: {
            "path": None, "disk_files": 0, "disk_bytes": 0, "native_frames": 0,
            "flags": (), "retained_states": 0, "has_store": False}
        layout = self._recording_layout()
        panel.draw(SimpleNamespace(layout=layout), context)
        self.assertNotIn("gpucloth.free_cache",
                         [row["idname"] for row in self._drawn_rows(layout)])

        # A bake transaction owns the store: the row is offered but disabled.
        operators_module.cache_store_summary = lambda _scene: dict(store)
        scene_s.is_baking = True
        layout = self._recording_layout()
        panel.draw(SimpleNamespace(layout=layout), context)
        clear_rows = [row for row in self._drawn_rows(layout)
                      if row["idname"] == "gpucloth.free_cache"]
        self.assertEqual(len(clear_rows), 1)
        self.assertFalse(clear_rows[0]["enabled"])

    def _drawn_rows(self, layout):
        """Every operator row a panel created, through its sub-layouts."""
        rows = list(layout.operators)
        for child in layout.rows:
            rows.extend(self._drawn_rows(child))
        return rows

    @staticmethod
    def _recording_layout():
        """A UILayout that records what a panel asked for.

        ``label``/``operator``/``prop`` return the same object so a chained row
        works, and ``enabled`` is recorded on the row that is created after it
        was set - the one property a panel sets outside the call that creates
        the element.
        """
        class Layout:
            def __init__(self):
                self.labels = []
                self.operators = []
                self.props = []
                self.rows = []
                self.enabled = True

            def label(self, **kwargs):
                self.labels.append(kwargs.get("text", ""))
                return self

            def operator(self, idname, **kwargs):
                self.operators.append(
                    {"idname": idname, "text": kwargs.get("text", ""),
                     "enabled": self.enabled})
                return self

            def prop(self, _owner, name, **_kwargs):
                self.props.append(name)
                return self

            def separator(self):
                return self

            def box(self):
                return self

            def row(self, **_kwargs):
                row = Layout()
                row.enabled = self.enabled
                self.rows.append(row)
                return row

            def column(self, **_kwargs):
                return self
        return Layout()

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
    def _optional_functions(source, names, initial=None):
        """``_functions`` for a contract that names a symbol a tree may not have.

        The defect-7 contracts below are behavioural: they drive the operator and
        read the filesystem, and the helpers they inject are only what the
        *tested tree* happens to call.  On a tree without them - the shipped
        build, for the baseline arm of the A/B - the test still has to run and
        fail on the behaviour, not on the import.
        """
        names = {node.name for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)} & set(names)
        return AddonLifecycleContractTest._functions(source, names, initial)

    @staticmethod
    def _optional_function(source, name, initial=None):
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                namespace = dict(initial or {})
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             "<contract>", "exec"), namespace)
                return namespace[name]
        return None

    @staticmethod
    def _class(source, name):
        tree = ast.parse(source)
        return next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == name)

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
