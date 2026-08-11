# Blender GPUCloth Add-on
# Copyright (C) 2023 Bubnov Aleksey
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import bpy
import hashlib
import json
import math
import os
import struct
import sys
import subprocess
import time

import numpy as np
from ctypes import (
    addressof, cdll, windll, POINTER, pointer, cast,
    c_bool, c_float, c_short, c_int, c_uint, c_uint64, c_void_p, c_size_t,
    c_char_p,
    create_string_buffer, sizeof,
)

from . import cpp_types as CType
from .proxy_binding import ProxyBindingError, validate_proxy_binding
from .vertex_channels import (
    VertexChannelError, apply_float_channel, binary_exclusion_mask,
    capture_evaluated_pin_snapshot, capture_material_coordinates,
    prepare_pin_snapshot,
    publish_pin_snapshot,
    vertex_group_weights,
)
from ..utils import version_compatibility_utils as vcu

debug = False
if sys.gettrace() is not None:
    debug = True

# ===========================================================================
#  Глобальное состояние симуляции
# ===========================================================================

g_dll                = None   # Загруженная DLL / .so
g_runtime_initialized = False
g_scene              = None   # Указатель на CType.Scene
g_obj                = []     # list[POINTER(CType.Object)]  — объекты ткани
g_clmd               = []     # list[POINTER(CType.ClothModifierData)]
g_mesh               = []     # list[POINTER(CType.Mesh)]
g_clothOBJs          = []     # list[bpy.types.Object]  — Blender-объекты ткани
g_simulationOBJs     = []     # render owner -> mesh actually sent to solver
g_clothCollisionOBJs = []     # list[POINTER(CType.Object)] — объекты столкновения
_dll_directory_handles = []

# Proxy-res: один handle на объект ткани (None если proxy не активен)
g_proxy_handles      = []     # list[c_void_p | None]
_collision_keepalive  = []     # prevent GC of collision ctypes data
_solver_diagnostics = []
_pin_snapshot_states = []
_dynamic_mesh_states = []
_collection_snapshots = []
_effector_weight_states = []
_collider_history = {}
_drape_status_by_uid = {}
_teardown_failure = False
_MODIFIER_VISIBILITY = (
    "show_viewport", "show_render", "show_in_editmode", "show_on_cage")


def _runtime_owners_retained():
    return bool(
        g_scene is not None or g_obj or g_mesh or g_clmd or
        g_clothOBJs or g_simulationOBJs or g_clothCollisionOBJs or
        g_proxy_handles or _collision_keepalive or _solver_diagnostics or
        _pin_snapshot_states or _dynamic_mesh_states or
        _collection_snapshots or _effector_weight_states or
        _collider_history or
        _initial_positions or _live_arrays)


def _guard_prepare_teardown(execute):
    def wrapped(operator, context):
        active_object = getattr(context, "active_object", None)
        original_mode = getattr(active_object, "mode", None)
        original_springs_built = bool(getattr(
            context.scene, "gpu_cloth_springs_built", False))
        modifier_state = []
        for obj in tuple(getattr(context.scene, "objects", ())):
            for modifier in tuple(getattr(obj, "modifiers", ())):
                if getattr(modifier, "type", None) == 'CLOTH':
                    modifier_state.append((
                        modifier,
                        tuple(bool(getattr(modifier, attribute, False))
                              for attribute in _MODIFIER_VISIBILITY),
                    ))
        if _teardown_failure:
            if not free_gpu_memory(context):
                operator.report(
                    {'ERROR'},
                    "Native teardown retry failed; retained owners block "
                    "prepare")
                return {'CANCELLED'}

        failure = None
        try:
            result = execute(operator, context)
        except Exception as exc:
            failure = exc
            result = {'CANCELLED'}
            operator.report({'ERROR'}, f"Prepare transaction failed: {exc}")

        if original_mode is not None:
            active_object = getattr(context, "active_object", None)
            try:
                if (active_object is not None and
                        active_object.mode != original_mode):
                    bpy.ops.object.mode_set(mode=original_mode)
            except (AttributeError, RuntimeError):
                if failure is None:
                    failure = RuntimeError(
                        f"cannot restore Blender mode {original_mode}")
                    operator.report({'ERROR'}, str(failure))

        native_mutated = bool(getattr(
            operator, "_native_prepare_mutated", False))
        succeeded = (
            result == {'FINISHED'} and failure is None and
            not _teardown_failure)
        if not succeeded:
            if native_mutated and _runtime_owners_retained():
                free_gpu_memory(context)
            if not native_mutated:
                for modifier, visibility in modifier_state:
                    try:
                        for attribute, value in zip(
                                _MODIFIER_VISIBILITY, visibility):
                            setattr(modifier, attribute, value)
                    except (AttributeError, ReferenceError, RuntimeError):
                        pass
            if hasattr(context.scene, "gpu_cloth_springs_built"):
                context.scene.gpu_cloth_springs_built = (
                    False if native_mutated else original_springs_built)

        if _teardown_failure:
            operator.report(
                {'ERROR'},
                "Native teardown failed; retained owners block prepare")
            return {'CANCELLED'}
        if failure is not None:
            return {'CANCELLED'}
        return result
    return wrapped


def ensure_native_teardown(shutdown_runtime=False):
    """Resolve retained native owners before package registration mutation."""
    if not (
            _teardown_failure or _runtime_owners_retained() or
            (shutdown_runtime and g_runtime_initialized)):
        return True
    return free_gpu_memory(shutdown_runtime=shutdown_runtime)


def _blender_session_uid(value, label):
    try:
        session_uid = int(value.session_uid)
    except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        raise RuntimeError(f"{label} has no Blender session UID") from exc
    if session_uid <= 0:
        raise RuntimeError(f"{label} has no Blender session UID")
    return session_uid


def _original_object(obj):
    original = getattr(obj, "original", None)
    return original if original is not None else obj


def _modifier_visibility(modifier):
    return tuple(
        bool(getattr(modifier, attribute, False))
        for attribute in _MODIFIER_VISIBILITY)


def _restore_modifier_visibility(states):
    for modifier, visibility in reversed(states):
        try:
            for attribute, value in zip(
                    _MODIFIER_VISIBILITY, visibility):
                setattr(modifier, attribute, value)
        except (AttributeError, ReferenceError, RuntimeError):
            pass


def _disable_modifier_input_stack(plans):
    states = []
    seen = set()
    try:
        for plan in plans:
            for modifier in (
                    plan["cloth_modifier"], *plan["downstream"]):
                identity = id(modifier)
                if identity in seen:
                    continue
                seen.add(identity)
                states.append((modifier, _modifier_visibility(modifier)))
                for attribute in _MODIFIER_VISIBILITY:
                    setattr(modifier, attribute, False)
    except (AttributeError, ReferenceError, RuntimeError) as exc:
        _restore_modifier_visibility(states)
        raise RuntimeError(
            "cannot isolate a Cloth modifier input stack") from exc
    return states


def _reject_active_shape_keys(obj):
    shape_keys = getattr(getattr(obj, "data", None), "shape_keys", None)
    if shape_keys is None:
        return
    if getattr(shape_keys, "animation_data", None) is not None:
        raise RuntimeError(
            f"animated shape keys before Cloth on {obj.name_full!r} are "
            "not representable")
    key_blocks = tuple(getattr(shape_keys, "key_blocks", ()))
    for key in key_blocks[1:]:
        try:
            value = float(key.value)
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            value = 0.0
        if value != 0.0:
            raise RuntimeError(
                f"active shape key {key.name!r} before Cloth on "
                f"{obj.name_full!r} is not representable")


def _plan_modifier_evaluation(bindings):
    plans = []
    for binding_index, binding in enumerate(bindings):
        cloth_obj = binding["render_object"]
        simulation_obj = binding["simulation_object"]
        modifiers = tuple(getattr(cloth_obj, "modifiers", ()))
        cloth_indices = [
            index for index, modifier in enumerate(modifiers)
            if getattr(modifier, "type", None) == 'CLOTH']
        if len(cloth_indices) != 1:
            raise RuntimeError(
                f"{cloth_obj.name_full!r} requires exactly one Cloth "
                "modifier")
        cloth_index = cloth_indices[0]
        cloth_modifier = modifiers[cloth_index]
        if bool(getattr(cloth_modifier, "use_pin_to_last", False)):
            raise RuntimeError(
                f"{cloth_obj.name_full!r} Cloth use_pin_to_last is not "
                "representable")
        active_upstream = [
            modifier.name for modifier in modifiers[:cloth_index]
            if any(_modifier_visibility(modifier))]
        if active_upstream:
            raise RuntimeError(
                f"{cloth_obj.name_full!r} has active modifiers before "
                f"Cloth: {', '.join(active_upstream)}")

        if binding["uses_proxy"]:
            active_proxy = [
                modifier.name
                for modifier in tuple(
                    getattr(simulation_obj, "modifiers", ()))
                if any(_modifier_visibility(modifier))]
            if active_proxy:
                raise RuntimeError(
                    f"proxy {simulation_obj.name_full!r} has active "
                    f"modifiers: {', '.join(active_proxy)}")
        # Shape keys are evaluated before Cloth.  They are representable when
        # the dynamic-mesh owner snapshots the complete evaluated base mesh,
        # or when the pin snapshot consumes evaluated positions only as
        # goal targets.  With neither owner the static-rest path must reject.
        product_settings = getattr(cloth_obj, "GPUCloth", None)
        retained_pin_target_owner = (
            binding_index < len(_pin_snapshot_states) and
            bool(_pin_snapshot_states[binding_index].get(
                "allows_evaluated_targets", False)))
        evaluated_shape_owner = (
            bool(getattr(product_settings, "use_dynamic_mesh", False)) or
            bool(getattr(product_settings, "vgroup_mass", "")) or
            retained_pin_target_owner)
        if not evaluated_shape_owner:
            _reject_active_shape_keys(simulation_obj)
        plans.append({
            "cloth_object": cloth_obj,
            "simulation_object": simulation_obj,
            "cloth_modifier": cloth_modifier,
            "cloth_modifier_index": cloth_index,
            "downstream": modifiers[cloth_index + 1:],
        })
    return plans


def _mesh_topology(mesh):
    return {
        "vertex_count": len(mesh.vertices),
        "edges": tuple(
            (int(edge.vertices[0]), int(edge.vertices[1]))
            for edge in mesh.edges),
        "polygons": tuple(
            (int(poly.loop_start), int(poly.loop_total))
            for poly in mesh.polygons),
        "loops": tuple(
            (int(loop.vertex_index), int(loop.edge_index))
            for loop in mesh.loops),
    }


def _capture_modifier_input_mesh(obj, depsgraph):
    try:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(
            preserve_all_data_layers=True, depsgraph=depsgraph)
    except (
            AttributeError, ReferenceError, RuntimeError,
            TypeError) as exc:
        raise RuntimeError(
            f"cannot evaluate Cloth input mesh for {obj.name_full!r}") from exc
    try:
        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError(
                f"Cloth input mesh for {obj.name_full!r} is empty")
        topology = _mesh_topology(mesh)
        if topology != _mesh_topology(obj.data):
            raise RuntimeError(
                f"modifier input topology for {obj.name_full!r} differs "
                "from the writable simulation mesh")
        positions = [None] * topology["vertex_count"]
        for vertex in mesh.vertices:
            index = int(vertex.index)
            if (
                    index < 0 or index >= topology["vertex_count"] or
                    positions[index] is not None):
                raise RuntimeError(
                    f"Cloth input mesh for {obj.name_full!r} has an "
                    f"invalid vertex index {index}")
            positions[index] = _finite_float32_tuple(
                (vertex.co[0], vertex.co[1], vertex.co[2]),
                f"Cloth input vertex {index}")
        if any(position is None for position in positions):
            raise RuntimeError(
                f"Cloth input mesh for {obj.name_full!r} has missing "
                "vertices")
    finally:
        evaluated.to_mesh_clear()
    return {
        **topology,
        "positions": tuple(positions),
        "capture_space": "CLOTH_INPUT_LOCAL",
    }


def _collection_selection(collection, label):
    if collection is None:
        return None
    try:
        collection_uid = _blender_session_uid(
            collection, f"{label} collection")
    except RuntimeError as exc:
        if isinstance(exc.__cause__, ReferenceError):
            raise RuntimeError(
                f"{label} collection is stale or unavailable") from exc
        raise
    ordered_objects = []
    object_ids = set()
    visited_collection_ids = set()

    def visit(current):
        current_id = _blender_session_uid(
            current, f"{label} nested collection")
        if current_id in visited_collection_ids:
            return
        visited_collection_ids.add(current_id)
        try:
            direct_objects = tuple(current.objects)
            children = tuple(current.children)
        except (
                AttributeError, ReferenceError, RuntimeError,
                TypeError) as exc:
            raise RuntimeError(
                f"{label} collection is stale or unavailable") from exc
        for obj in direct_objects:
            original = _original_object(obj)
            object_id = _blender_session_uid(
                original, f"{label} collection object")
            if object_id in object_ids:
                continue
            object_ids.add(object_id)
            ordered_objects.append(original)
        for child in children:
            visit(child)

    visit(collection)
    return {
        "collection": collection,
        "collection_id": collection_uid,
        "objects": tuple(ordered_objects),
        "object_ids": frozenset(object_ids),
    }


def _collision_collection_selection(settings):
    return _collection_selection(
        getattr(settings, "collision_collection", None), "collision")


def _occurrence_persistent_id(occurrence):
    try:
        values = tuple(int(value) for value in occurrence.persistent_id)
    except (AttributeError, TypeError, ValueError):
        return ()
    # Blender pads the tuple with INT_MAX. It is not part of the identity.
    return tuple(value for value in values if value != 0x7fffffff)


def _occurrence_instance_id(object_id, parent_id, persistent_id):
    identity = bytearray()
    identity.extend(int(object_id).to_bytes(8, "little", signed=False))
    for value in persistent_id:
        identity.extend(int(value).to_bytes(8, "little", signed=True))
    identity.extend(int(parent_id).to_bytes(8, "little", signed=False))
    return _stable_cache_id(bytes(identity))


def _collection_record_flags(source_object, is_instance, evaluated):
    flags = 0
    if evaluated:
        flags |= CType.GPUCLOTH_COLLECTION_RECORD_EVALUATED
    if is_instance:
        flags |= CType.GPUCLOTH_COLLECTION_RECORD_INSTANCE
    if not bool(getattr(source_object, "hide_viewport", False)):
        flags |= CType.GPUCLOTH_COLLECTION_RECORD_VIEWPORT_ENABLED
    if not bool(getattr(source_object, "hide_render", False)):
        flags |= CType.GPUCLOTH_COLLECTION_RECORD_RENDER_ENABLED
    return flags


def _copy_matrix_world(value, label):
    if value is None:
        raise RuntimeError(f"{label} occurrence has no evaluated matrix")
    try:
        return value.copy()
    except AttributeError:
        return value


def _depsgraph_occurrences(depsgraph, selection, label):
    """Return collection-order members, then raw depsgraph-order instances."""
    try:
        source_occurrences = depsgraph.object_instances
    except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        raise RuntimeError(
            f"{label} dependency graph has no object instances") from exc

    selected_ids = (
        selection["object_ids"] if selection is not None else None)
    result = []

    # A Blender 4.2 collision/effector collection need not be linked into the
    # active view layer. Traverse the selected collection cache directly so
    # unlinked recursive members remain part of the snapshot. Evaluation is
    # best-effort because evaluated_get legitimately fails for unlinked IDs.
    if selection is not None:
        for source_object in selection["objects"]:
            evaluated_object = source_object
            evaluated = False
            try:
                candidate = source_object.evaluated_get(depsgraph)
            except (
                    AttributeError, ReferenceError, RuntimeError,
                    TypeError, ValueError):
                candidate = None
            if candidate is not None:
                evaluated_object = candidate
                evaluated = True
            object_id = _blender_session_uid(
                source_object, f"{label} object")
            matrix_world = _copy_matrix_world(
                getattr(
                    evaluated_object, "matrix_world",
                    getattr(source_object, "matrix_world", None)),
                label)
            result.append({
                "source_object": source_object,
                "evaluated_object": evaluated_object,
                "matrix_world": matrix_world,
                "object_id": object_id,
                "instance_id": 0,
                "parent_id": 0,
                "persistent_id": (),
                "record_flags": _collection_record_flags(
                    source_object, False, evaluated),
            })

    for occurrence in source_occurrences:
        evaluated_object = getattr(occurrence, "object", None)
        if evaluated_object is None:
            continue
        is_instance = bool(getattr(occurrence, "is_instance", False))
        if selection is not None and not is_instance:
            continue
        instance_object = getattr(occurrence, "instance_object", None)
        source_object = _original_object(
            instance_object
            if is_instance and instance_object is not None
            else evaluated_object)
        object_id = _blender_session_uid(source_object, f"{label} object")

        parent = getattr(occurrence, "parent", None)
        parent_original = _original_object(parent) if parent is not None else None
        parent_id = (
            _blender_session_uid(parent_original, f"{label} instance parent")
            if parent_original is not None else 0)
        membership_ids = {object_id}
        if parent_id:
            membership_ids.add(parent_id)
        if selected_ids is not None and not (membership_ids & selected_ids):
            continue

        matrix_world = _copy_matrix_world(
            getattr(
                occurrence, "matrix_world",
                getattr(evaluated_object, "matrix_world", None)),
            label)
        persistent_id = _occurrence_persistent_id(occurrence)
        instance_id = (
            _occurrence_instance_id(object_id, parent_id, persistent_id)
            if is_instance else 0)

        result.append({
            "source_object": source_object,
            "evaluated_object": evaluated_object,
            "matrix_world": matrix_world,
            "object_id": object_id,
            "instance_id": instance_id,
            "parent_id": parent_id,
            "persistent_id": persistent_id,
            "record_flags": _collection_record_flags(
                source_object, is_instance, True),
        })

    return tuple(result)


def native_frame_timescale(scene, speed_multiplier):
    """Convert Blender's dimensionless speed to native seconds per frame."""
    fps = float(scene.render.fps)
    if fps <= 0.0:
        return 0.0
    return float(speed_multiplier) * float(scene.render.fps_base) / fps


def _stable_cache_id(identity_bytes):
    value = 1469598103934665603
    for byte in identity_bytes:
        value ^= byte
        value = (value * 1099511628211) & 0xffffffffffffffff
    return value or 1


def _active_cache_path(scene):
    helper = scene.gpu_cloth_helper
    root = bpy.path.abspath(
        helper.external_cache_dir
        if helper.use_external_cache else helper.cache_dir)
    cache_index = int(helper.cache_index)
    cache_name = str(helper.cache_name)
    if not root or '\0' in root:
        raise RuntimeError("cache path is empty or contains NUL")
    if cache_index < 0:
        raise RuntimeError("cache index must be non-negative")
    if ('\0' in cache_name or not cache_name or
            len(cache_name.encode('utf-8')) >= 255):
        raise RuntimeError("cache name must contain 1..254 UTF-8 bytes")
    if cache_index == 0 and cache_name == "GPUCloth":
        return root
    identity = (
        cache_index.to_bytes(4, "little") +
        cache_name.encode('utf-8'))
    suffix = hashlib.blake2b(
        identity, digest_size=8, person=b"GPUCache").hexdigest()
    return os.path.join(root, f"cache_{cache_index:08x}_{suffix}")


def _configure_cache_features(dll, scene):
    helper = scene.gpu_cloth_helper
    path_bytes = _active_cache_path(scene).encode('utf-8')
    if not path_bytes:
        raise RuntimeError("cache path is empty")
    cache_index = int(helper.cache_index)
    cache_name = str(helper.cache_name)
    name_bytes = cache_name.encode('utf-8')
    path_buffer = create_string_buffer(path_bytes)
    name_buffer = create_string_buffer(name_bytes)
    config = CType.GPUClothCacheConfig()
    config.header.struct_size = sizeof(config)
    config.header.config_version = 1
    if helper.use_external_cache:
        config.storage_mode = CType.GPUCLOTH_CACHE_STORAGE_EXTERNAL
    else:
        config.storage_mode = (
            CType.GPUCLOTH_CACHE_STORAGE_DISK
            if helper.use_disk_cache
            else CType.GPUCLOTH_CACHE_STORAGE_MEMORY)
    compression_modes = {
        'NO': CType.GPUCLOTH_CACHE_COMPRESSION_NONE,
        'LIGHT': CType.GPUCLOTH_CACHE_COMPRESSION_LIGHT,
        'HEAVY': CType.GPUCLOTH_CACHE_COMPRESSION_HEAVY,
    }
    try:
        config.compression_mode = (
            CType.GPUCLOTH_CACHE_COMPRESSION_NONE
            if helper.use_external_cache or not helper.use_disk_cache
            else compression_modes[helper.cache_compression])
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported cache compression {helper.cache_compression}") from exc
    config.frame_start = int(helper.bake_start)
    config.frame_end = int(helper.bake_end)
    config.frame_step = 1
    config.cache_index = cache_index
    config.cache_flags = (
        CType.GPUCLOTH_CACHE_FLAG_EXTERNAL_READ_ONLY |
        (CType.GPUCLOTH_CACHE_FLAG_LIBRARY_PATH
         if helper.use_library_path else 0)
        if helper.use_external_cache else 0)
    config.cache_id = _stable_cache_id(
        path_bytes + b'\0' + cache_index.to_bytes(4, 'little') +
        name_bytes)
    config.path_utf8_address = addressof(path_buffer)
    config.name_utf8_address = addressof(name_buffer)
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    if helper.use_external_cache:
        storage_feature = CType.GPUCLOTH_FEATURE_CACHE_EXTERNAL
    else:
        storage_feature = (
            CType.GPUCLOTH_FEATURE_CACHE_DISK
            if helper.use_disk_cache
            else CType.GPUCLOTH_FEATURE_CACHE_MEMORY)
    for feature in (
            storage_feature,
            *(
                (CType.GPUCLOTH_FEATURE_CACHE_COMPRESSION,)
                if config.storage_mode == CType.GPUCLOTH_CACHE_STORAGE_DISK
                else ()),
            CType.GPUCLOTH_FEATURE_BAKE_RANGE,
            CType.GPUCLOTH_FEATURE_CALCULATE_TO_FRAME,
            CType.GPUCLOTH_FEATURE_CACHE_STATUS,
            CType.GPUCLOTH_FEATURE_CACHE_MULTIPLE):
        config.header.feature_id = feature
        result = int(dll.SIM_configure_feature(header))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"typed cache feature {feature} rejected with {result}")
    return CType.GPUCLOTH_ABI_OK


def _configure_simulation_features(dll, clmd, scene, settings):
    solver_mask = {
        'XPBD': CType.GPUCLOTH_SOLVER_XPBD,
        'PD': CType.GPUCLOTH_SOLVER_PD,
        'Mil2': CType.GPUCLOTH_SOLVER_MIL2,
    }.get(settings.solver_type)
    if solver_mask is None:
        raise RuntimeError(
            f"typed simulation config does not own solver "
            f"{settings.solver_type}")

    config = CType.GPUClothSimulationConfig()
    config.header.struct_size = sizeof(config)
    config.header.config_version = 1
    config.solver_mask = solver_mask
    config.quality_steps = settings.quality_step
    config.time_scale = native_frame_timescale(
        scene, settings.speed_multiplier)
    config.vertex_mass = settings.vertex_mass
    config.gravity[:] = (
        scene.gpu_cloth_helper.gravity_x,
        scene.gpu_cloth_helper.gravity_y,
        scene.gpu_cloth_helper.gravity_z,
    )
    config.air_damping = settings.air_viscosity
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    for feature in (
            CType.GPUCLOTH_FEATURE_TIMESTEP_SPEED,
            CType.GPUCLOTH_FEATURE_MATERIAL_MASS,
            CType.GPUCLOTH_FEATURE_GRAVITY_VECTOR,
            CType.GPUCLOTH_FEATURE_SIMULATION_QUALITY,
            CType.GPUCLOTH_FEATURE_AIR_DAMPING):
        config.header.feature_id = feature
        result = int(dll.SIM_configure_cloth_feature(clmd, header))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"typed simulation feature {feature} rejected with {result}")
    return CType.GPUCLOTH_ABI_OK


def _configure_solver_diagnostics(dll, clmd, event_capacity=16):
    if event_capacity < 1 or event_capacity > 64:
        raise RuntimeError("diagnostic event capacity must be in [1, 64]")
    event_type = CType.GPUClothDiagnosticsEvent * event_capacity
    events = event_type()
    generation = (
        (int(addressof(clmd.contents)) ^
         0x475055434C4F5448) & 0xffffffffffffffff) or 1
    config = CType.GPUClothDiagnosticsConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_SOLVER_DIAGNOSTICS
    config.header.config_version = 1
    config.diagnostics_flags = (
        CType.GPUCLOTH_DIAGNOSTICS_STATUS |
        CType.GPUCLOTH_DIAGNOSTICS_EVENTS)
    config.event_capacity = event_capacity
    config.minimum_severity = CType.GPUCLOTH_DIAGNOSTICS_INFO
    config.event_buffer.struct_size = sizeof(CType.GPUClothBufferView)
    config.event_buffer.element_type = (
        CType.GPUCLOTH_ELEMENT_DIAGNOSTIC_EVENT)
    config.event_buffer.element_count = event_capacity
    config.event_buffer.stride_bytes = sizeof(
        CType.GPUClothDiagnosticsEvent)
    config.event_buffer.data_address = addressof(events)
    config.event_buffer.generation = generation
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    result = int(dll.SIM_configure_cloth_feature(clmd, header))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed solver diagnostics rejected with {result}")
    _solver_diagnostics.append({
        "events_buffer": events,
        "generation": generation,
        "snapshot": None,
        "last_event_sequence": 0,
    })
    return result


def _validate_descriptor_layout(dll):
    layout = CType.GPUClothDescriptorLayout()
    layout.struct_size = sizeof(layout)
    if not dll.SIM_get_descriptor_layout(pointer(layout)):
        raise RuntimeError("native descriptor layout query failed")
    expected = {
        "material_config_size": sizeof(CType.GPUClothMaterialConfig),
        "collision_config_size": sizeof(CType.GPUClothCollisionConfig),
        "collider_config_size": sizeof(CType.GPUClothColliderConfig),
        "mesh_state_config_size": sizeof(CType.GPUClothMeshStateConfig),
        "diagnostics_event_size": sizeof(CType.GPUClothDiagnosticsEvent),
        "diagnostics_status_size": sizeof(CType.GPUClothDiagnosticsStatus),
        "pin_snapshot_config_size": sizeof(
            CType.GPUClothPinSnapshotConfig),
        "collection_record_size": sizeof(
            CType.GPUClothCollectionRecord),
        "collection_snapshot_config_size": sizeof(
            CType.GPUClothCollectionSnapshotConfig),
        "collection_transaction_config_size": sizeof(
            CType.GPUClothCollectionTransactionConfig),
        "collection_query_size": sizeof(CType.GPUClothCollectionQuery),
        "collection_status_size": sizeof(CType.GPUClothCollectionStatus),
        "invariant_witness_size": sizeof(
            CType.GPUClothInvariantWitness),
        "preparation_config_size": sizeof(
            CType.GPUClothPreparationConfig),
        "preparation_status_size": sizeof(
            CType.GPUClothPreparationStatus),
        "drape_config_size": sizeof(CType.GPUClothDrapeConfig),
        "drape_status_size": sizeof(CType.GPUClothDrapeStatus),
    }
    mismatches = [
        f"{name}={int(getattr(layout, name))}, expected={expected_size}"
        for name, expected_size in expected.items()
        if int(getattr(layout, name)) != expected_size]
    if (int(layout.struct_size) != sizeof(layout) or
            int(layout.schema_version) != 4 or
            int(layout.legacy_reserved0) != 0 or mismatches):
        detail = "; ".join(mismatches) if mismatches else "header mismatch"
        raise RuntimeError(f"native descriptor ABI mismatch: {detail}")
    return layout


def _validate_product_abi(dll):
    version = CType.GPUClothABIVersion()
    version.struct_size = sizeof(version)
    if not dll.SIM_get_product_abi_version(pointer(version)):
        raise RuntimeError("native product ABI query failed")
    actual = (
        int(version.abi_major), int(version.abi_minor),
        int(version.abi_patch), int(version.feature_schema_version))
    expected = (2, 0, 0, 4)
    if actual != expected:
        raise RuntimeError(
            f"native product ABI {actual} does not match required "
            f"{expected}")
    return version


def _validate_native_preparation(
        dll, clmd, topology_generation, requested_generation):
    config = CType.GPUClothPreparationConfig()
    config.struct_size = sizeof(config)
    config.config_version = 1
    config.preparation_flags = CType.GPUCLOTH_PREPARATION_CONFIGURED
    config.topology_generation = int(topology_generation)
    config.requested_generation = int(requested_generation)
    status = CType.GPUClothPreparationStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    result = int(dll.SIM_validate_cloth_initial_state(
        clmd, pointer(config), pointer(status)))
    if (result != CType.GPUCLOTH_ABI_OK or
            int(status.result) != CType.GPUCLOTH_PREPARATION_RESULT_READY or
            not (int(status.status_flags) &
                 CType.GPUCLOTH_PREPARATION_STATUS_RUNNABLE)):
        witness = CType.GPUClothInvariantWitness()
        witness.struct_size = sizeof(witness)
        witness.witness_version = 1
        witness_result = int(dll.SIM_get_cloth_invariant_status(
            clmd, pointer(witness)))
        detail = {
            "abi_result": result,
            "preparation_result": int(status.result),
            "last_error": int(status.last_error),
            "witness_result": witness_result,
            "invariant": int(witness.invariant),
            "triangle_i": int(witness.triangle_i),
            "triangle_j": int(witness.triangle_j),
            "edge_i": int(witness.edge_i),
            "vertex_i": int(witness.vertex_i),
            "other_object_id": int(witness.other_object_id),
        }
        raise RuntimeError(
            f"native hard preflight rejected cloth: {detail}")
    return status


_INVARIANT_NAMES = {
    CType.GPUCLOTH_INVARIANT_NONE: "NONE",
    CType.GPUCLOTH_INVARIANT_INVALID_INDEX: "INVALID_INDEX",
    CType.GPUCLOTH_INVARIANT_NONFINITE_STATE: "NONFINITE_STATE",
    CType.GPUCLOTH_INVARIANT_DEGENERATE_TRIANGLE: "DEGENERATE_TRIANGLE",
    CType.GPUCLOTH_INVARIANT_INCONSISTENT_WINDING: "INCONSISTENT_WINDING",
    CType.GPUCLOTH_INVARIANT_INVALID_SEAM: "INVALID_SEAM",
    CType.GPUCLOTH_INVARIANT_SELF_INTERSECTION: "SELF_INTERSECTION",
    CType.GPUCLOTH_INVARIANT_EXTERNAL_INTERSECTION: "EXTERNAL_INTERSECTION",
    CType.GPUCLOTH_INVARIANT_EXTERNAL_CLEARANCE: "EXTERNAL_CLEARANCE",
    CType.GPUCLOTH_INVARIANT_PRESSURE_OPEN_SHELL: "PRESSURE_OPEN_SHELL",
    CType.GPUCLOTH_INVARIANT_PRESSURE_VOLUME: "PRESSURE_VOLUME",
    CType.GPUCLOTH_INVARIANT_CONTACT_OVERFLOW: "CONTACT_OVERFLOW",
    CType.GPUCLOTH_INVARIANT_STALE_GENERATION: "STALE_GENERATION",
    CType.GPUCLOTH_INVARIANT_CUDA_ERROR: "CUDA_ERROR",
    CType.GPUCLOTH_INVARIANT_GRAPH_ERROR: "GRAPH_ERROR",
    CType.GPUCLOTH_INVARIANT_NOT_CONVERGED: "NOT_CONVERGED",
}


def _prepared_cloth_index(cloth_obj):
    target_uid = _blender_session_uid(cloth_obj, "drape cloth")
    for index, candidate in enumerate(g_clothOBJs):
        if _blender_session_uid(candidate, "prepared cloth") == target_uid:
            return index
    return -1


def _new_drape_status():
    status = CType.GPUClothDrapeStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    return status


def _drape_status_data(status):
    return {
        "status_flags": int(status.status_flags),
        "result": int(status.result),
        "last_error": int(status.last_error),
        "step_count": int(status.step_count),
        "consecutive_converged_steps": int(
            status.consecutive_converged_steps),
        "maximum_position_delta": float(status.maximum_position_delta),
        "convergence_tolerance": float(status.convergence_tolerance),
        "begin_generation": int(status.begin_generation),
        "current_generation": int(status.current_generation),
    }


def _remember_drape_status(cloth_obj, status):
    uid = _blender_session_uid(cloth_obj, "drape cloth")
    _drape_status_by_uid[uid] = _drape_status_data(status)
    return _drape_status_by_uid[uid]


def get_drape_ui_status(cloth_obj):
    try:
        uid = _blender_session_uid(cloth_obj, "drape cloth")
    except RuntimeError:
        return None
    return _drape_status_by_uid.get(uid)


def _invariant_witness_data(clmd):
    witness = CType.GPUClothInvariantWitness()
    witness.struct_size = sizeof(witness)
    witness.witness_version = 1
    result = int(g_dll.SIM_get_cloth_invariant_status(
        clmd, pointer(witness)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"invariant status rejected with {result}")
    invariant = int(witness.invariant)
    return {
        "stage": int(witness.stage),
        "result": int(witness.result),
        "invariant": invariant,
        "invariant_name": _INVARIANT_NAMES.get(
            invariant, f"UNKNOWN_{invariant}"),
        "frame": int(witness.frame),
        "substep": int(witness.substep),
        "solver_mask": int(witness.solver_mask),
        "generations": {
            "candidate": int(witness.candidate_generation),
            "detection": int(witness.detection_generation),
            "contact": int(witness.contact_generation),
            "apply": int(witness.apply_generation),
        },
        "cloth_id": int(witness.cloth_id),
        "other_object_id": int(witness.other_object_id),
        "layers": [int(witness.cloth_layer), int(witness.other_layer)],
        "primitives": {
            "triangle_i": int(witness.triangle_i),
            "triangle_j": int(witness.triangle_j),
            "edge_i": int(witness.edge_i),
            "edge_j": int(witness.edge_j),
            "vertex_i": int(witness.vertex_i),
            "vertex_j": int(witness.vertex_j),
            "intersection_type": int(witness.intersection_type),
        },
        "cardinalities": {
            "vf": int(witness.vf_count),
            "ee": int(witness.ee_count),
            "ef": int(witness.ef_count),
            "accepted_owner": int(witness.accepted_owner_count),
            "overflow": int(witness.overflow_count),
        },
        "cuda_status": int(witness.cuda_status),
        "graph_status": int(witness.graph_status),
        "minimum_clearance": float(witness.minimum_clearance),
        "maximum_penetration": float(witness.maximum_penetration),
        "aabb_min": [float(value) for value in witness.aabb_min],
        "aabb_max": [float(value) for value in witness.aabb_max],
        "maximum_velocity": float(witness.maximum_velocity),
        "frame_wall_ns": int(witness.frame_wall_ns),
    }


def get_invariant_ui_status(cloth_obj):
    index = _prepared_cloth_index(cloth_obj)
    if g_dll is None or index < 0:
        return None
    try:
        return _invariant_witness_data(g_clmd[index])
    except (OSError, RuntimeError):
        return None


def get_preparation_ui_status(cloth_obj):
    index = _prepared_cloth_index(cloth_obj)
    if g_dll is None or index < 0:
        return None
    status = CType.GPUClothPreparationStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    try:
        result = int(g_dll.SIM_get_cloth_preparation_status(
            g_clmd[index], pointer(status)))
    except OSError:
        return None
    if result != CType.GPUCLOTH_ABI_OK:
        return None
    return {
        "status_flags": int(status.status_flags),
        "result": int(status.result),
        "last_error": int(status.last_error),
        "preparation_generation": int(status.preparation_generation),
        "topology_generation": int(status.topology_generation),
        "accepted_generation": int(status.accepted_generation),
    }


def _ordered_diagnostic_events(owner, status):
    """Read oldest-to-newest using the native ring cursor, never slot order."""
    events_buffer = owner["events_buffer"]
    capacity = int(status.event_capacity)
    event_count = int(status.event_count)
    write_index = int(status.event_write_index)
    if capacity < 0 or capacity > len(events_buffer):
        raise RuntimeError(
            f"diagnostic capacity {capacity} exceeds caller buffer "
            f"{len(events_buffer)}")
    if event_count < 0 or event_count > capacity:
        raise RuntimeError(
            f"diagnostic count {event_count} exceeds capacity {capacity}")
    if not capacity:
        return []
    if write_index < 0 or write_index >= capacity:
        raise RuntimeError(
            f"diagnostic write index {write_index} exceeds capacity "
            f"{capacity}")

    first_index = (write_index - event_count) % capacity
    ordered = []
    previous_sequence = 0
    for offset in range(event_count):
        event = events_buffer[(first_index + offset) % capacity]
        sequence = int(event.sequence)
        if int(event.struct_size) != sizeof(CType.GPUClothDiagnosticsEvent):
            raise RuntimeError(
                "diagnostic ring contains an invalid event layout")
        if sequence <= previous_sequence:
            raise RuntimeError(
                "diagnostic ring sequence is not strictly ordered")
        previous_sequence = sequence
        ordered.append({
            "struct_size": int(event.struct_size),
            "event_type": int(event.event_type),
            "severity": int(event.severity),
            "result": int(event.result),
            "error_code": int(event.error_code),
            "solver_mask": int(event.solver_mask),
            "sequence": sequence,
        })
    owner["last_event_sequence"] = (
        ordered[-1]["sequence"] if ordered else 0)
    return ordered


def _expose_solver_result(index, status):
    if index < 0 or index >= len(g_clmd):
        return
    result_ptr = g_clmd[index].contents.solver_result
    if not result_ptr:
        return
    result_ptr.contents.status = int(status.solver_result_status)
    result_ptr.contents.max_iterations = int(status.max_iterations)
    result_ptr.contents.min_iterations = int(status.min_iterations)
    result_ptr.contents.avg_iterations = float(status.avg_iterations)
    result_ptr.contents.max_error = float(status.max_error_value)
    result_ptr.contents.min_error = float(status.min_error_value)
    result_ptr.contents.avg_error = float(status.avg_error_value)


def _solver_diagnostic_snapshot(index):
    if index < 0 or index >= len(_solver_diagnostics):
        return None
    owner = _solver_diagnostics[index]
    status = CType.GPUClothDiagnosticsStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    result = int(g_dll.SIM_get_cloth_solver_diagnostics(
        g_clmd[index], pointer(status)))
    status_data = {
        "status_version": int(status.status_version),
        "status_flags": int(status.status_flags),
        "last_result": int(status.last_result),
        "last_error": int(status.last_error),
        "requested_solver_mask": int(status.requested_solver_mask),
        "backend_solver_mask": int(status.backend_solver_mask),
        "solver_result_status": int(status.solver_result_status),
        "convergence_metric": int(status.convergence_metric),
        "event_capacity": int(status.event_capacity),
        "event_count": int(status.event_count),
        "dropped_event_count": int(status.dropped_event_count),
        "event_write_index": int(status.event_write_index),
        "substep_count": int(status.substep_count),
        "min_iterations": int(status.min_iterations),
        "max_iterations": int(status.max_iterations),
        "avg_iterations": float(status.avg_iterations),
        "min_error_value": float(status.min_error_value),
        "max_error_value": float(status.max_error_value),
        "avg_error_value": float(status.avg_error_value),
        "last_error_value": float(status.last_error_value),
        "convergence_tolerance": float(status.convergence_tolerance),
        "total_iterations": int(status.total_iterations),
        "execution_time_ns": int(status.execution_time_ns),
        "execution_time_ms": int(status.execution_time_ns) / 1_000_000.0,
        "sequence": int(status.sequence),
        "solve_count": int(status.solve_count),
        "failure_count": int(status.failure_count),
        "event_generation": int(status.event_generation),
    }
    events = []
    if result == CType.GPUCLOTH_ABI_OK:
        if status_data["event_generation"] != int(owner["generation"]):
            raise RuntimeError(
                "diagnostic event generation changed behind caller buffer")
        events = _ordered_diagnostic_events(owner, status)
        _expose_solver_result(index, status)
    snapshot = {
        "query_result": result,
        "status": status_data,
        "events": events,
    }
    owner["snapshot"] = snapshot
    return {
        "query_result": result,
        "status": dict(status_data),
        "events": [dict(event) for event in events],
    }


def get_solver_diagnostics(cloth=None):
    """Public read-only diagnostics, including real result and native timing."""
    if cloth is None:
        return tuple(
            _solver_diagnostic_snapshot(index)
            for index in range(len(_solver_diagnostics)))
    if isinstance(cloth, int):
        index = cloth
    else:
        cloth_uid = _blender_session_uid(cloth, "diagnostic cloth")
        index = next((
            item_index
            for item_index, item in enumerate(g_clothOBJs)
            if _blender_session_uid(item, "diagnostic cloth") == cloth_uid
        ), -1)
    if index < 0 or index >= len(_solver_diagnostics):
        raise IndexError("cloth diagnostics owner does not exist")
    return _solver_diagnostic_snapshot(index)


def _bounded_float32(value, label, lower, upper):
    try:
        source = float(value)
        converted = float(c_float(source).value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not a float32 value") from exc
    if not math.isfinite(source) or not math.isfinite(converted):
        raise RuntimeError(f"{label} is not finite")
    if source < lower or source > upper:
        raise RuntimeError(f"{label} is outside [{lower:g}, {upper:g}]")
    return converted


def _effective_bending_model(settings):
    bending_model = str(settings.bending_model)
    solver_type = str(settings.solver_type)
    if solver_type == 'Mil2':
        return 'ANGULAR'
    if solver_type == 'PD' and bending_model in {'', 'LINEAR'}:
        # Migrate old saved PD settings: linear bending is no longer exposed.
        return 'ANGULAR'
    if solver_type != 'PD' and bending_model == 'SDB':
        return 'ANGULAR'
    if bending_model not in {'LINEAR', 'ANGULAR', 'SDB'}:
        raise RuntimeError(f"unknown bending model {bending_model!r}")
    return bending_model


def _capture_material_features(settings, material_coordinates):
    config = CType.GPUClothMaterialConfig()
    config.header.struct_size = sizeof(config)
    bending_model = _effective_bending_model(settings)
    config.bending_model = (
        1 if bending_model in {'ANGULAR', 'SDB'} else 0)

    stiffness = tuple(
        _bounded_float32(value, label, 0.0, 10000.0)
        for value, label in zip((
            settings.tension,
            settings.compression,
            settings.shear,
            settings.bending_stiffness,
        ), (
            "tension stiffness",
            "compression stiffness",
            "shear stiffness",
            "bending stiffness",
        ))
    )
    stiffness_max = tuple(
        _bounded_float32(value, label, 0.0, 10000.0)
        for value, label in zip((
            settings.max_tension,
            settings.max_compression,
            settings.max_shear,
            settings.max_bend,
        ), (
            "maximum tension stiffness",
            "maximum compression stiffness",
            "maximum shear stiffness",
            "maximum bending stiffness",
        ))
    )
    for value, maximum, label in zip(
            stiffness, stiffness_max,
            ("tension", "compression", "shear", "bending")):
        if maximum < value:
            raise RuntimeError(
                f"maximum {label} stiffness is below its base value")
    config.stiffness[:] = stiffness
    config.stiffness_max[:] = stiffness_max
    config.damping[:] = tuple(
        _bounded_float32(value, label, lower, upper)
        for value, label, lower, upper in (
            (settings.tension_damp, "tension damping", 0.0, 50.0),
            (settings.compression_damp, "compression damping", 0.0, 50.0),
            (settings.shear_damp, "shear damping", 0.0, 50.0),
            (settings.bending_damping, "bending damping", 0.0, 1000.0),
        ))

    coordinate_payload = None
    if bool(settings.use_anisotropy):
        if material_coordinates is None:
            raise RuntimeError(
                "anisotropy material coordinates were not captured")
        directional = tuple(
            _bounded_float32(value, label, 0.0, 10000.0)
            for value, label in zip((
                settings.tension_u,
                settings.tension_v,
                settings.compression_u,
                settings.compression_v,
                settings.bending_u,
                settings.bending_v,
            ), (
                "tension U stiffness",
                "tension V stiffness",
                "compression U stiffness",
                "compression V stiffness",
                "bending U stiffness",
                "bending V stiffness",
            ))
        )
        directional_max = tuple(
            _bounded_float32(value, label, 0.0, 10000.0)
            for value, label in zip((
                settings.max_tension_u,
                settings.max_tension_v,
                settings.max_compression_u,
                settings.max_compression_v,
                settings.max_bend_u,
                settings.max_bend_v,
            ), (
                "maximum tension U stiffness",
                "maximum tension V stiffness",
                "maximum compression U stiffness",
                "maximum compression V stiffness",
                "maximum bending U stiffness",
                "maximum bending V stiffness",
            ))
        )
        for value, maximum, label in zip(
                directional, directional_max, (
                    "tension U", "tension V",
                    "compression U", "compression V",
                    "bending U", "bending V")):
            if maximum < value:
                raise RuntimeError(
                    f"maximum {label} stiffness is below its base value")
        config.directional_stiffness[:] = directional
        config.directional_stiffness_max[:] = directional_max
        config.material_flags = (
            CType.GPUCLOTH_MATERIAL_ANISOTROPY_ENABLED)

        coordinate_payload = (
            c_float * len(material_coordinates.coordinates))(
                *material_coordinates.coordinates)
        _set_buffer_view(
            config.material_coordinates,
            CType.GPUCLOTH_ELEMENT_FLOAT2,
            material_coordinates.vertex_count,
            sizeof(c_float) * 2,
            addressof(coordinate_payload),
            material_coordinates.topology_generation)

    return {
        "config": config,
        "coordinates": coordinate_payload,
        "bending_model": bending_model,
        "anisotropy": bool(settings.use_anisotropy),
    }


def _publish_material_features(
        dll, clmd, owner, publish_anisotropy=False):
    source = owner["config"]
    if publish_anisotropy:
        if not owner["anisotropy"]:
            return CType.GPUCLOTH_ABI_OK
        features = (CType.GPUCLOTH_FEATURE_ANISOTROPY,)
    else:
        features = [
            CType.GPUCLOTH_FEATURE_STRETCH,
            CType.GPUCLOTH_FEATURE_COMPRESSION,
            CType.GPUCLOTH_FEATURE_SHEAR,
            CType.GPUCLOTH_FEATURE_MATERIAL_DAMPING,
        ]
        if owner["bending_model"] != 'SDB':
            features.insert(
                3,
                (CType.GPUCLOTH_FEATURE_BENDING_ANGULAR
                 if owner["bending_model"] == 'ANGULAR'
                 else CType.GPUCLOTH_FEATURE_BENDING_LINEAR))

    for feature in features:
        config = CType.GPUClothMaterialConfig.from_buffer_copy(bytes(source))
        config.header.feature_id = feature
        if feature == CType.GPUCLOTH_FEATURE_ANISOTROPY:
            config.header.config_version = 2
            config.material_flags = (
                CType.GPUCLOTH_MATERIAL_ANISOTROPY_ENABLED)
        else:
            config.header.config_version = 1
            config.material_flags = 0
        header = cast(
            pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
        result = int(dll.SIM_configure_cloth_feature(clmd, header))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"typed material feature {feature} rejected with {result}")
    return CType.GPUCLOTH_ABI_OK


def _capture_internal_springs_config(settings):
    enabled = bool(settings.use_internal_springs)
    normal_check = enabled and bool(settings.use_internal_springs_normal)

    max_length = _bounded_float32(
        settings.internal_spring_max_length,
        "internal spring maximum length", 0.0, 1000.0) if enabled else 0.0
    max_diversion = _bounded_float32(
        settings.internal_spring_max_diversion,
        "internal spring maximum diversion", 0.0, math.pi / 4.0
    ) if enabled else 0.0
    tension = _bounded_float32(
        settings.internal_tension,
        "internal tension stiffness", 0.0, 10000.0) if enabled else 0.0
    tension_max = _bounded_float32(
        settings.max_internal_tension,
        "maximum internal tension stiffness", 0.0, 10000.0
    ) if enabled else 0.0
    compression = _bounded_float32(
        settings.internal_compression,
        "internal compression stiffness", 0.0, 10000.0
    ) if enabled else 0.0
    compression_max = _bounded_float32(
        settings.max_internal_compression,
        "maximum internal compression stiffness", 0.0, 10000.0
    ) if enabled else 0.0
    if tension_max < tension:
        raise RuntimeError(
            "maximum internal tension stiffness is below its base value")
    if compression_max < compression:
        raise RuntimeError(
            "maximum internal compression stiffness is below its base value")

    config = CType.GPUClothConstraintConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_INTERNAL_SPRINGS
    config.header.config_version = 1
    if enabled:
        config.constraint_flags |= (
            CType.GPUCLOTH_CONSTRAINT_INTERNAL_SPRINGS)
    if normal_check:
        config.constraint_flags |= (
            CType.GPUCLOTH_CONSTRAINT_INTERNAL_NORMAL_CHECK)
    config.sewing_force_max = _bounded_float32(
        settings.max_sewing, "maximum sewing force", 0.0, 10000.0)
    config.internal_spring_max_length = max_length
    config.internal_spring_max_diversion = max_diversion
    config.internal_tension_stiffness = tension
    config.internal_tension_stiffness_max = tension_max
    config.internal_compression_stiffness = compression
    config.internal_compression_stiffness_max = compression_max
    return config


def _publish_internal_springs_config(dll, clmd, prepared_config):
    config = CType.GPUClothConstraintConfig.from_buffer_copy(
        bytes(prepared_config))
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    result = int(dll.SIM_configure_cloth_feature(clmd, header))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed internal-springs config rejected with {result}")
    return result


def _capture_pressure_features(settings):
    if not settings.use_pressure:
        return None
    uniform_pressure = _bounded_float32(
        settings.uniform_pressure_force,
        "uniform pressure force", -100.0, 100.0)
    target_volume = _bounded_float32(
        settings.target_volume, "target volume", 0.0, 1000.0)
    pressure_factor = _bounded_float32(
        settings.pressure_factor, "pressure factor", 0.0, 100.0)
    fluid_density = _bounded_float32(
        settings.fluid_density, "fluid density", -10.0, 10.0)
    if settings.use_pressure_volume and target_volume <= 0.0:
        raise RuntimeError(
            "custom pressure volume requires a positive target volume")

    config = CType.GPUClothPressureConfig()
    config.header.struct_size = sizeof(config)
    config.header.config_version = 1
    config.pressure_flags = CType.GPUCLOTH_PRESSURE_ENABLED
    config.uniform_pressure_force = uniform_pressure
    config.target_volume = (
        target_volume if settings.use_pressure_volume else 0.0)
    config.pressure_factor = pressure_factor
    config.fluid_density = fluid_density
    features = [
        CType.GPUCLOTH_FEATURE_PRESSURE_UNIFORM,
        CType.GPUCLOTH_FEATURE_FLUID_DENSITY,
    ]
    if settings.use_pressure_volume:
        features.append(CType.GPUCLOTH_FEATURE_PRESSURE_VOLUME)
    return config, tuple(features)


def _publish_pressure_features(dll, clmd, prepared):
    if prepared is None:
        return CType.GPUCLOTH_ABI_OK
    prepared_config, features = prepared
    config = CType.GPUClothPressureConfig.from_buffer_copy(
        bytes(prepared_config))
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    for feature in features:
        config.header.feature_id = feature
        result = int(dll.SIM_configure_cloth_feature(clmd, header))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"typed pressure feature {feature} rejected with {result}")
    return CType.GPUCLOTH_ABI_OK


def _rest_shape_key_positions(settings_owner, simulation_obj):
    settings = settings_owner.GPUCloth
    selector = settings.shapekey_rest
    if selector is None or selector == "":
        return None
    if settings.use_dynamic_mesh:
        raise VertexChannelError(
            "rest shape key and dynamic mesh are mutually exclusive")

    shape_keys = getattr(simulation_obj.data, "shape_keys", None)
    key_blocks = (
        getattr(shape_keys, "key_blocks", None)
        if shape_keys is not None else None)
    if key_blocks is None:
        raise VertexChannelError(
            f"rest shape key {selector!r} requires mesh shape keys")

    key_block = None
    if isinstance(selector, str):
        key_block = key_blocks.get(selector)
        if key_block is None:
            try:
                index = int(selector, 10)
            except ValueError:
                index = None
            if index is not None and 0 <= index < len(key_blocks):
                key_block = key_blocks[index]
    elif isinstance(selector, int) and 0 <= selector < len(key_blocks):
        key_block = key_blocks[selector]
    if key_block is None:
        raise VertexChannelError(
            f"rest shape key {selector!r} does not exist")

    vertex_count = len(simulation_obj.data.vertices)
    key_count = len(key_block.data)
    if key_count != vertex_count:
        raise VertexChannelError(
            f"rest shape key {key_block.name!r} has {key_count} points; "
            f"simulation mesh has {vertex_count} vertices")

    positions = np.empty(vertex_count * 3, dtype=np.float32)
    key_block.data.foreach_get("co", positions)
    positions = np.ascontiguousarray(
        positions.reshape((vertex_count, 3)), dtype=np.float32)
    if not np.isfinite(positions).all():
        raise VertexChannelError(
            f"rest shape key {key_block.name!r} contains non-finite positions")
    return key_block.name, positions


def _upload_rest_shape_key(dll, clmd, prepared_rest_shape):
    if prepared_rest_shape is None:
        return CType.GPUCLOTH_ABI_OK

    key_name, positions = prepared_rest_shape
    generation_hash = hashlib.blake2b(
        digest_size=8, person=b"GPURest")
    generation_hash.update(key_name.encode("utf-8"))
    generation_hash.update(len(positions).to_bytes(8, "little"))
    generation_hash.update(positions.tobytes(order="C"))
    generation = int.from_bytes(
        generation_hash.digest(), "little") or 1

    config = CType.GPUClothMeshStateConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_REST_SHAPE_KEY
    config.header.config_version = 1
    config.rest_generation = generation
    config.rest_positions.struct_size = sizeof(CType.GPUClothBufferView)
    config.rest_positions.element_type = CType.GPUCLOTH_ELEMENT_FLOAT3
    config.rest_positions.element_count = len(positions)
    config.rest_positions.stride_bytes = sizeof(c_float) * 3
    config.rest_positions.data_address = positions.ctypes.data
    config.rest_positions.generation = generation
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    result = int(dll.SIM_configure_cloth_feature(clmd, header))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed rest shape key config rejected with {result}")
    return result


def _blender_sewing_edges(settings_owner, simulation_obj):
    if not bool(settings_owner.GPUCloth.use_sewing_springs):
        return []

    loose_edges = [
        edge for edge in simulation_obj.data.edges if edge.is_loose]
    if not loose_edges:
        raise VertexChannelError(
            "Blender sewing is enabled but the evaluated mesh has no "
            "loose seam edges")
    return loose_edges


def _upload_sewing(dll, clmd, settings_owner, simulation_obj):
    loose_edges = _blender_sewing_edges(settings_owner, simulation_obj)
    if not loose_edges:
        return CType.GPUCLOTH_ABI_OK
    record_type = CType.GPUClothSewingRecord * len(loose_edges)
    records = record_type()
    for index, edge in enumerate(loose_edges):
        records[index].seam_id = int(edge.index) + 1
        records[index].vertex_a = int(edge.vertices[0])
        records[index].vertex_b = int(edge.vertices[1])
        records[index].stiffness = 1.0
        records[index].rest_length = 0.0
        records[index].activation = 1.0
        records[index].flags = 0

    config = CType.GPUClothSewingConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_SEWING
    config.header.config_version = 1
    config.records.struct_size = sizeof(CType.GPUClothBufferView)
    config.records.element_type = CType.GPUCLOTH_ELEMENT_SEWING_RECORD
    config.records.element_count = len(records)
    config.records.stride_bytes = sizeof(CType.GPUClothSewingRecord)
    config.records.data_address = addressof(records)
    config.records.generation = 1
    config.phase_count = 1
    config.sewing_flags = 0
    config.activation_speed = 0.0
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    result = int(dll.SIM_configure_cloth_feature(clmd, header))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"typed sewing config rejected with {result}")
    return result


def _cloth_topology_generation(simulation_obj):
    mesh = simulation_obj.data
    triangles = tuple(
        tuple(int(index) for index in triangle.vertices)
        for triangle in vcu.calc_mesh_loop_triangles(mesh))
    return _topology_generation(
        _blender_session_uid(simulation_obj, "pin simulation object"),
        len(mesh.vertices), triangles)


def _capture_pin_snapshot(
        settings_owner, simulation_obj, depsgraph, topology_generation,
        frame_generation):
    return capture_evaluated_pin_snapshot(
        simulation_obj, depsgraph, settings_owner.GPUCloth,
        topology_generation, frame_generation,
        expected_vertex_count=len(simulation_obj.data.vertices),
        identity_obj=settings_owner)


def _publish_prepared_pin_snapshot(dll, clmd, snapshot):
    return publish_pin_snapshot(dll, CType, clmd, snapshot)


def _capture_stiffness_channels(settings_owner, simulation_obj):
    settings = settings_owner.GPUCloth
    channels = []
    for group_name, channel_name, channel in (
            (settings.vgroup_struct, "structural stiffness",
             CType.GPUCLOTH_VERTEX_STRUCTURAL_STIFFNESS),
            (settings.vgroup_shear, "shear stiffness",
             CType.GPUCLOTH_VERTEX_SHEAR_STIFFNESS),
            (settings.vgroup_bend, "bending stiffness",
             CType.GPUCLOTH_VERTEX_BENDING_STIFFNESS),
            (settings.vgroup_intern, "internal stiffness",
             CType.GPUCLOTH_VERTEX_INTERNAL_STIFFNESS)):
        weights = vertex_group_weights(
            simulation_obj, group_name, channel_name)
        if weights is not None:
            channels.append((channel, tuple(weights)))
    return tuple(channels)


def _upload_stiffness_channels(dll, clmd, prepared_channels):
    for channel, weights in prepared_channels:
        apply_float_channel(
            dll, CType, clmd,
            CType.GPUCLOTH_FEATURE_STIFFNESS_VERTEX_GROUPS,
            channel, 1, weights)
    return CType.GPUCLOTH_ABI_OK


def _upload_pressure_weights(
        dll, clmd, settings_owner, simulation_obj):
    if not settings_owner.GPUCloth.use_pressure:
        return CType.GPUCLOTH_ABI_OK
    weights = vertex_group_weights(
        simulation_obj, settings_owner.GPUCloth.vgroup_pressure, "pressure")
    if weights is None:
        return CType.GPUCLOTH_ABI_OK
    return apply_float_channel(
        dll, CType, clmd, CType.GPUCLOTH_FEATURE_PRESSURE_VERTEX_GROUP,
        CType.GPUCLOTH_VERTEX_PRESSURE_WEIGHT, 1, weights)


def _upload_shrink_weights(
        dll, clmd, settings_owner, simulation_obj):
    weights = vertex_group_weights(
        simulation_obj, settings_owner.GPUCloth.vgroup_shrink, "shrink")
    if weights is None:
        return CType.GPUCLOTH_ABI_OK
    return apply_float_channel(
        dll, CType, clmd, CType.GPUCLOTH_FEATURE_SHRINK,
        CType.GPUCLOTH_VERTEX_SHRINK_WEIGHT, 1, weights)


def _upload_object_collision_mask(
        dll, clmd, settings_owner, simulation_obj):
    mask = binary_exclusion_mask(
        simulation_obj, settings_owner.GPUCloth.vgroup_objcol,
        "object collision")
    return apply_float_channel(
        dll, CType, clmd, CType.GPUCLOTH_FEATURE_COLLISION_VERTEX_GROUP,
        CType.GPUCLOTH_VERTEX_OBJECT_COLLISION_MASK, 1, mask)


def _capture_self_collision_mask(settings_owner, simulation_obj):
    group_name = settings_owner.GPUCloth.vgroup_selfcol
    weights = vertex_group_weights(
        simulation_obj, group_name, "self collision")
    return None if weights is None else tuple(weights)


def _upload_self_collision_mask(dll, clmd, prepared_mask):
    if prepared_mask is None:
        return CType.GPUCLOTH_ABI_OK
    return apply_float_channel(
        dll, CType, clmd,
        CType.GPUCLOTH_FEATURE_SELF_COLLISION_VERTEX_GROUP,
        CType.GPUCLOTH_VERTEX_SELF_COLLISION_MASK, 1, prepared_mask)

# Защита от GC для ctypes-массивов, переданных в Cache_write_frame_async
# C++ пишет в фоне — массив должен жить до завершения записи
_live_arrays         = []     # list[c_float array]

def _close_dll_directories():
    for directory_handle in _dll_directory_handles:
        directory_handle.close()
    _dll_directory_handles.clear()

# ─── Effector helpers ──────────────────────────────────────────────────────

def _make_effector_weights(ew):
    """Return Blender's exact EffectorWeights/PFIELD order plus gravity."""
    values = (
        ew.weight_all,             # PFIELD_NULL / global multiplier
        ew.weight_force,           # PFIELD_FORCE
        ew.weight_vortex,          # PFIELD_VORTEX
        ew.weight_magnetic,        # PFIELD_MAGNET
        ew.weight_wind,            # PFIELD_WIND
        ew.weight_curve_guide,     # PFIELD_GUIDE
        ew.weight_texture,         # PFIELD_TEXTURE
        ew.weight_harmonic,        # PFIELD_HARMONIC
        ew.weight_charge,          # PFIELD_CHARGE
        ew.weight_lennard_jones,   # PFIELD_LENNARDJ
        ew.weight_boid,            # PFIELD_BOID
        ew.weight_turbulence,      # PFIELD_TURBULENCE
        ew.weight_drag,            # PFIELD_DRAG
        ew.weight_smoke_flow,      # PFIELD_FLUIDFLOW
        ew.global_gravity,
    )
    return tuple(
        _bounded_float32(value, f"effector weight[{index}]", -200.0, 200.0)
        for index, value in enumerate(values))


_FIELD_TYPE_MAP = {
    'FORCE': 1,
    'VORTEX': 2,
    'MAGNET': 3,
    'WIND': 4,
    'GUIDE': 5,
    'TEXTURE': 6,
    'HARMONIC': 7,
    'CHARGE': 8,
    'LENNARDJ': 9,
    'BOID': 10,
    'TURBULENCE': 11,
    'DRAG': 12,
    'FLUID_FLOW': 13,
}
_SUPPORTED_FIELD_TYPES = {
    'FORCE', 'VORTEX', 'MAGNET', 'WIND', 'HARMONIC', 'DRAG',
}
_FIELD_SHAPE_MAP = {
    'POINT': 0,
    'PLANE': 1,
    'SURFACE': 2,
    'POINTS': 3,
    'LINE': 4,
}
_FIELD_FALLOFF_MAP = {'SPHERE': 0, 'TUBE': 1, 'CONE': 2}


def _collection_object_type(obj):
    return {
        'MESH': CType.GPUCLOTH_COLLECTION_OBJECT_MESH,
        'CURVE': CType.GPUCLOTH_COLLECTION_OBJECT_CURVE,
        'EMPTY': CType.GPUCLOTH_COLLECTION_OBJECT_EMPTY,
    }.get(
        str(getattr(obj, "type", "")),
        CType.GPUCLOTH_COLLECTION_OBJECT_OTHER)


def _set_buffer_view(
        view, element_type, element_count, stride_bytes, data_address,
        generation):
    view.struct_size = sizeof(CType.GPUClothBufferView)
    view.element_type = int(element_type)
    view.element_count = int(element_count)
    view.stride_bytes = int(stride_bytes)
    view.data_address = int(data_address)
    view.generation = int(generation)


def _topology_generation(object_id, vertex_count, triangles):
    hasher = hashlib.blake2b(digest_size=8, person=b"GPUTopology")
    hasher.update(int(object_id).to_bytes(8, "little", signed=False))
    hasher.update(int(vertex_count).to_bytes(8, "little", signed=False))
    for triangle in triangles:
        for index in triangle:
            hasher.update(int(index).to_bytes(4, "little", signed=False))
    return int.from_bytes(hasher.digest(), "little") or 1


def _collision_modifier(occurrence, depsgraph):
    source = occurrence["source_object"]
    evaluated = occurrence["evaluated_object"]
    is_render = str(getattr(depsgraph, "mode", "VIEWPORT")) == 'RENDER'
    source_modifiers = tuple(getattr(source, "modifiers", ()))
    evaluated_modifiers = tuple(getattr(evaluated, "modifiers", ()))
    for modifier_index, source_modifier in enumerate(source_modifiers):
        if source_modifier.type != 'COLLISION':
            continue
        modifier = source_modifier
        if modifier_index < len(evaluated_modifiers):
            candidate = evaluated_modifiers[modifier_index]
            if getattr(candidate, "type", None) == 'COLLISION':
                modifier = candidate
        enabled = (
            bool(getattr(modifier, "show_render", True))
            if is_render
            else bool(getattr(modifier, "show_viewport", True)))
        if enabled:
            return modifier_index
    return None


def _finite_float32_tuple(values, label):
    result = []
    for value in values:
        try:
            source = float(value)
            converted = float(c_float(source).value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{label} contains an invalid float") from exc
        if not math.isfinite(source) or not math.isfinite(converted):
            raise RuntimeError(f"{label} contains a non-finite float")
        result.append(converted)
    return tuple(result)


def _capture_dynamic_mesh_snapshot(
        settings_owner, simulation_obj, depsgraph, topology_generation,
        geometry_generation):
    if not bool(settings_owner.GPUCloth.use_dynamic_mesh):
        return None

    topology_generation = int(topology_generation)
    geometry_generation = int(geometry_generation)
    if topology_generation <= 0 or geometry_generation <= 0:
        raise RuntimeError(
            "dynamic mesh topology/geometry generations must be positive")

    object_id = _blender_session_uid(
        settings_owner, "dynamic mesh cloth owner")
    simulation_id = _blender_session_uid(
        simulation_obj, "dynamic mesh simulation object")
    expected_vertex_count = len(simulation_obj.data.vertices)
    try:
        evaluated = simulation_obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(
            preserve_all_data_layers=False, depsgraph=depsgraph)
    except (
            AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        raise RuntimeError(
            f"cannot evaluate dynamic mesh for "
            f"{simulation_obj.name!r}") from exc

    try:
        if mesh is None:
            raise RuntimeError(
                f"evaluated dynamic mesh for "
                f"{simulation_obj.name!r} is unavailable")
        vertex_count = len(mesh.vertices)
        if vertex_count <= 0:
            raise RuntimeError(
                f"evaluated dynamic mesh for "
                f"{simulation_obj.name!r} is empty")
        if vertex_count != expected_vertex_count:
            raise RuntimeError(
                f"dynamic mesh topology changed for "
                f"{simulation_obj.name!r}: "
                f"{vertex_count} != {expected_vertex_count}")

        triangles = tuple(
            tuple(int(index) for index in triangle.vertices)
            for triangle in vcu.calc_mesh_loop_triangles(mesh))
        if any(
                len(triangle) != 3 or
                any(index < 0 or index >= vertex_count
                    for index in triangle)
                for triangle in triangles):
            raise RuntimeError(
                f"dynamic mesh topology for "
                f"{simulation_obj.name!r} is invalid")
        evaluated_topology = _topology_generation(
            simulation_id, vertex_count, triangles)
        if evaluated_topology != topology_generation:
            raise RuntimeError(
                f"dynamic mesh topology changed for "
                f"{simulation_obj.name!r}")

        positions = [0.0] * (vertex_count * 3)
        seen = [False] * vertex_count
        for vertex in mesh.vertices:
            index = int(vertex.index)
            if index < 0 or index >= vertex_count or seen[index]:
                raise RuntimeError(
                    f"dynamic mesh vertex index {index} is invalid for "
                    f"{simulation_obj.name!r}")
            seen[index] = True
            offset = index * 3
            positions[offset:offset + 3] = _finite_float32_tuple(
                (vertex.co[0], vertex.co[1], vertex.co[2]),
                f"dynamic mesh vertex {index}")
        if not all(seen):
            raise RuntimeError(
                f"dynamic mesh for {simulation_obj.name!r} has missing "
                "vertex indices")
    finally:
        evaluated.to_mesh_clear()

    return {
        "object_id": object_id,
        "topology_generation": topology_generation,
        "geometry_generation": geometry_generation,
        "vertex_count": vertex_count,
        "positions": tuple(positions),
    }


def _prepare_dynamic_mesh_state(snapshot, accepted_snapshot):
    if snapshot is None:
        return None

    vertex_count = int(snapshot["vertex_count"])
    scalar_count = vertex_count * 3
    current = tuple(snapshot["positions"])
    if vertex_count <= 0 or len(current) != scalar_count:
        raise RuntimeError("dynamic mesh snapshot has an invalid count")

    if accepted_snapshot is None:
        previous = current
    else:
        if (
                int(accepted_snapshot["object_id"]) !=
                    int(snapshot["object_id"]) or
                int(accepted_snapshot["topology_generation"]) !=
                    int(snapshot["topology_generation"]) or
                int(accepted_snapshot["vertex_count"]) != vertex_count):
            raise RuntimeError(
                "dynamic mesh accepted history has a different owner or "
                "topology")
        if (
                int(accepted_snapshot["geometry_generation"]) >=
                    int(snapshot["geometry_generation"])):
            raise RuntimeError(
                "dynamic mesh geometry generation did not advance")
        previous = tuple(accepted_snapshot["positions"])
        if len(previous) != scalar_count:
            raise RuntimeError(
                "dynamic mesh accepted history has an invalid count")

    previous_payload = (c_float * scalar_count)(*previous)
    current_payload = (c_float * scalar_count)(*current)
    if addressof(previous_payload) == addressof(current_payload):
        raise RuntimeError(
            "dynamic mesh previous/current buffers must be distinct")

    config = CType.GPUClothMeshStateConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_DYNAMIC_MESH
    config.header.config_version = 1
    config.header.flags = 0
    config.object_id = int(snapshot["object_id"])
    config.topology_generation = int(snapshot["topology_generation"])
    config.geometry_generation = int(snapshot["geometry_generation"])
    config.rest_generation = 0
    config.mesh_flags = CType.GPUCLOTH_MESH_DYNAMIC_BASE
    _set_buffer_view(
        config.positions_previous, CType.GPUCLOTH_ELEMENT_FLOAT3,
        vertex_count, sizeof(c_float) * 3, addressof(previous_payload),
        config.geometry_generation)
    _set_buffer_view(
        config.positions_current, CType.GPUCLOTH_ELEMENT_FLOAT3,
        vertex_count, sizeof(c_float) * 3, addressof(current_payload),
        config.geometry_generation)
    return {
        "config": config,
        "previous": previous_payload,
        "current": current_payload,
        "snapshot": snapshot,
    }


def _capture_cloth_collision_config(settings):
    try:
        collision_quality = int(settings.collision_quality)
        source_quality = settings.collision_quality
        values = _finite_float32_tuple((
            settings.epsilon,
            settings.collision_friction,
            settings.collision_damping,
            settings.clamp,
            settings.selfepsilon,
            settings.self_collision_friction,
            settings.self_clamp,
        ), "cloth collision settings")
        use_object_collision = bool(settings.use_object_collision)
        use_self_collision = bool(settings.use_self_collision)
    except (
            AttributeError, ReferenceError, RuntimeError, TypeError,
            ValueError) as exc:
        raise RuntimeError("cloth collision settings are incomplete") from exc

    if collision_quality != source_quality:
        raise RuntimeError("collision_quality is not an exact integer")
    if collision_quality < 1 or collision_quality > 0x7fff:
        raise RuntimeError("collision_quality is outside [1, 32767]")

    (distance_min, friction, damping, impulse_clamp,
     self_distance_min, self_friction, self_impulse_clamp) = values
    if not 0.001 <= distance_min <= 1.0:
        raise RuntimeError("collision distance is outside [0.001, 1]")
    if not 0.0 <= friction <= 80.0:
        raise RuntimeError("collision friction is outside [0, 80]")
    if not 0.0 <= damping <= 1.0:
        raise RuntimeError("collision damping is outside [0, 1]")
    if not 0.0 <= impulse_clamp <= 100.0:
        raise RuntimeError("collision impulse clamp is outside [0, 100]")
    if not 0.001 <= self_distance_min <= 0.1:
        raise RuntimeError("self-collision distance is outside [0.001, 0.1]")
    if not 0.0 <= self_friction <= 80.0:
        raise RuntimeError("self-collision friction is outside [0, 80]")
    if not 0.0 <= self_impulse_clamp <= 100.0:
        raise RuntimeError(
            "self-collision impulse clamp is outside [0, 100]")

    config = CType.GPUClothCollisionConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_STATIC_OBJECT_COLLISION
    config.header.config_version = 1
    if use_object_collision:
        config.collision_flags |= CType.GPUCLOTH_COLLISION_OBJECT_ENABLED
    if use_self_collision:
        config.collision_flags |= CType.GPUCLOTH_COLLISION_SELF_ENABLED
    config.collision_quality = collision_quality
    config.distance_min = distance_min
    config.friction = friction
    config.damping = damping
    config.impulse_clamp = impulse_clamp
    config.self_distance_min = self_distance_min
    config.self_friction = self_friction
    config.self_impulse_clamp = self_impulse_clamp
    config.self_response = CType.GPUCLOTH_SELF_RESPONSE_OGC
    return config


def _publish_cloth_collision_config(dll, clmd, prepared_config):
    for feature_id in (
            CType.GPUCLOTH_FEATURE_STATIC_OBJECT_COLLISION,
            CType.GPUCLOTH_FEATURE_COLLISION_FRICTION_DAMPING,
            CType.GPUCLOTH_FEATURE_COLLISION_QUALITY_CLAMP,
            CType.GPUCLOTH_FEATURE_SELF_COLLISION,
            CType.GPUCLOTH_FEATURE_SELF_COLLISION_FRICTION):
        config = CType.GPUClothCollisionConfig.from_buffer_copy(
            bytes(prepared_config))
        config.header.feature_id = feature_id
        result = int(dll.SIM_configure_cloth_feature(
            clmd,
            cast(
                pointer(config),
                POINTER(CType.GPUClothFeatureConfigHeader))))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"typed cloth collision config for feature {feature_id} "
                f"rejected with {result}")


def _matrix_signature(matrix, label):
    try:
        values = tuple(
            matrix[row][column]
            for row in range(4)
            for column in range(4))
    except (
            AttributeError, IndexError, ReferenceError, RuntimeError,
            TypeError) as exc:
        raise RuntimeError(f"{label} is not a finite 4x4 matrix") from exc
    return _finite_float32_tuple(values, label)


def _matrix_inverse(matrix, label):
    _matrix_signature(matrix, label)
    try:
        inverse = matrix.inverted()
    except (
            AttributeError, ReferenceError, RuntimeError, TypeError,
            ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"{label} is singular") from exc
    _matrix_signature(inverse, f"{label} inverse")
    return inverse


def _cloth_local_matrix(cloth_inverse, object_matrix, label):
    try:
        relative = cloth_inverse @ object_matrix
    except (
            AttributeError, ReferenceError, RuntimeError, TypeError,
            ValueError) as exc:
        raise RuntimeError(
            f"cannot transform {label} into cloth-local space") from exc
    _matrix_signature(relative, f"{label} cloth-local transform")
    return relative


def _collider_motion_classification(
        history, topology_generation, local_positions, matrix_signature,
        snapshot_generation):
    if history is None:
        return (
            CType.GPUCLOTH_COLLIDER_STATIC,
            CType.GPUCLOTH_FEATURE_STATIC_OBJECT_COLLISION)
    try:
        history_generation = int(history["geometry_generation"])
        history_topology = int(history["topology_generation"])
        history_local = tuple(history["local_positions"])
        history_matrix = tuple(history["matrix_signature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("collider history is invalid") from exc
    if history_generation >= int(snapshot_generation):
        raise RuntimeError("collider history generation is not monotonic")
    if (
            history_topology != int(topology_generation) or
            history_local != tuple(local_positions)):
        return (
            CType.GPUCLOTH_COLLIDER_DEFORMING,
            CType.GPUCLOTH_FEATURE_DEFORMING_OBJECT_COLLISION)
    if history_matrix != tuple(matrix_signature):
        return (
            CType.GPUCLOTH_COLLIDER_MOVING,
            CType.GPUCLOTH_FEATURE_MOVING_OBJECT_COLLISION)
    return (
        CType.GPUCLOTH_COLLIDER_STATIC,
        CType.GPUCLOTH_FEATURE_STATIC_OBJECT_COLLISION)


def _capture_collider_payload(
    occurrence, modifier_index, depsgraph, cloth_owner_id, collection_id,
        snapshot_generation, collider_history, cloth_inverse):
    evaluated = occurrence["evaluated_object"]
    mesh = None
    mesh_owner = None
    candidates = [evaluated]
    source = occurrence["source_object"]
    try:
        source_evaluated = source.evaluated_get(depsgraph)
    except (
            AttributeError, ReferenceError, RuntimeError,
            TypeError, ValueError):
        source_evaluated = None
    for candidate in (source_evaluated, source):
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    last_error = None
    for candidate in candidates:
        try:
            mesh = candidate.to_mesh(
                preserve_all_data_layers=True, depsgraph=depsgraph)
        except (
                AttributeError, ReferenceError, RuntimeError,
                TypeError) as exc:
            last_error = exc
            continue
        if mesh is not None:
            mesh_owner = candidate
            break
    if mesh is None:
        raise RuntimeError(
            f"cannot evaluate collider "
            f"{occurrence['source_object'].name_full!r}") from last_error
    try:
        if mesh is None:
            raise RuntimeError(
                f"collider {occurrence['source_object'].name_full!r} "
                "has no evaluated mesh")
        loop_triangles = tuple(vcu.calc_mesh_loop_triangles(mesh))
        triangles = tuple(
            tuple(int(index) for index in triangle.vertices)
            for triangle in loop_triangles)
        vertex_count = len(mesh.vertices)
        if vertex_count == 0 or not triangles:
            raise RuntimeError(
                f"collider {occurrence['source_object'].name_full!r} "
                "has no evaluated collision surface")
        if any(
                len(triangle) != 3 or
                any(index < 0 or index >= vertex_count for index in triangle)
                for triangle in triangles):
            raise RuntimeError(
                f"collider {occurrence['source_object'].name_full!r} "
                "has invalid evaluated triangles")
        local_positions = [None] * (vertex_count * 3)
        cloth_local_positions = [None] * (vertex_count * 3)
        relative_matrix = _cloth_local_matrix(
            cloth_inverse, occurrence["matrix_world"],
            f"collider {occurrence['source_object'].name_full!r}")
        for vertex in mesh.vertices:
            index = int(vertex.index)
            if index < 0 or index >= vertex_count:
                raise RuntimeError(
                    f"collider {occurrence['source_object'].name_full!r} "
                    f"has invalid evaluated vertex index {index}")
            offset = index * 3
            local = _finite_float32_tuple(
                (vertex.co[0], vertex.co[1], vertex.co[2]),
                f"collider {occurrence['source_object'].name_full!r} "
                f"local vertex {index}")
            cloth_local = vcu.element_multiply(
                relative_matrix, vertex.co)
            cloth_local = _finite_float32_tuple(
                (cloth_local[0], cloth_local[1], cloth_local[2]),
                f"collider {occurrence['source_object'].name_full!r} "
                f"cloth-local vertex {index}")
            local_positions[offset:offset + 3] = local
            cloth_local_positions[offset:offset + 3] = cloth_local
        if any(
                value is None
                for value in local_positions + cloth_local_positions):
            raise RuntimeError(
                f"collider {occurrence['source_object'].name_full!r} "
                "has incomplete evaluated vertices")
    finally:
        mesh_owner.to_mesh_clear()

    local_positions = tuple(local_positions)
    cloth_local_positions = tuple(cloth_local_positions)
    triangle_values = [
        index for triangle in triangles for index in triangle]
    triangle_array = (c_uint * len(triangle_values))(*triangle_values)
    topology_generation = _topology_generation(
        occurrence["object_id"], vertex_count, triangles)
    matrix_signature = _matrix_signature(
        relative_matrix,
        f"collider {occurrence['source_object'].name_full!r} "
        "cloth-local transform")
    history_key = (
        int(cloth_owner_id),
        int(occurrence["object_id"]),
        int(occurrence["instance_id"]),
        int(modifier_index),
    )
    history = collider_history.get(history_key)
    collider_class, feature_id = _collider_motion_classification(
        history, topology_generation, local_positions, matrix_signature,
        snapshot_generation)

    try:
        history_compatible = (
            history is not None and
            int(history["topology_generation"]) == topology_generation and
            len(history["previous_positions"]) ==
                len(cloth_local_positions) and
            len(history["current_positions"]) ==
                len(cloth_local_positions))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("collider history time levels are invalid") from exc
    uses_history = (
        history_compatible and
        collider_class != CType.GPUCLOTH_COLLIDER_STATIC)
    if uses_history:
        previous_positions = _finite_float32_tuple(
            history["previous_positions"],
            "collider previous cloth-local history")
        current_positions = _finite_float32_tuple(
            history["current_positions"],
            "collider current cloth-local history")
    else:
        previous_positions = tuple(
            value for value in cloth_local_positions)
        current_positions = tuple(
            value for value in cloth_local_positions)
    next_positions = tuple(value for value in cloth_local_positions)

    position_type = c_float * len(cloth_local_positions)
    previous_array = position_type(*previous_positions)
    current_array = position_type(*current_positions)
    next_array = position_type(*next_positions)
    if len(cloth_local_positions) and len({
            addressof(previous_array),
            addressof(current_array),
            addressof(next_array)}) != 3:
        raise RuntimeError("collider time-level buffers alias")

    config = CType.GPUClothColliderConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = feature_id
    config.header.config_version = 1
    config.object_id = occurrence["object_id"]
    config.collection_id = collection_id
    config.topology_generation = topology_generation
    config.geometry_generation = snapshot_generation
    config.collider_flags = collider_class
    config.vertex_count = vertex_count
    config.triangle_count = len(triangles)
    triangle_address = addressof(triangle_array) if triangle_values else 0
    _set_buffer_view(
        config.positions_previous, CType.GPUCLOTH_ELEMENT_FLOAT3,
        vertex_count, sizeof(c_float) * 3, addressof(previous_array),
        snapshot_generation)
    _set_buffer_view(
        config.positions_current, CType.GPUCLOTH_ELEMENT_FLOAT3,
        vertex_count, sizeof(c_float) * 3, addressof(current_array),
        snapshot_generation)
    _set_buffer_view(
        config.positions_next, CType.GPUCLOTH_ELEMENT_FLOAT3,
        vertex_count, sizeof(c_float) * 3, addressof(next_array),
        snapshot_generation)
    _set_buffer_view(
        config.triangles, CType.GPUCLOTH_ELEMENT_UINT3, len(triangles),
        sizeof(c_uint) * 3, triangle_address, topology_generation)

    source_object = occurrence["source_object"]
    settings = getattr(source_object, "collision", None)
    if settings is None:
        raise RuntimeError(
            f"collider {source_object.name_full!r} has no Blender "
            "CollisionSettings")
    try:
        config.thickness_outer = float(settings.thickness_outer)
        config.friction = float(settings.cloth_friction)
        config.damping = float(settings.damping)
        config.effector_absorption = float(settings.absorption)
        use_culling = bool(settings.use_culling)
        use_normal = bool(settings.use_normal)
    except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        raise RuntimeError(
            f"collider {source_object.name_full!r} has incomplete Blender "
            "CollisionSettings") from exc
    surface_values = (
        config.thickness_outer, config.friction, config.damping,
        config.effector_absorption)
    if not all(math.isfinite(value) for value in surface_values):
        raise RuntimeError(
            f"collider {source_object.name_full!r} has "
            "non-finite surface settings")
    if not 0.001 <= config.thickness_outer <= 1.0:
        raise RuntimeError(
            f"collider {source_object.name_full!r} thickness_outer is "
            "outside [0.001, 1]")
    if not 0.0 <= config.friction <= 80.0:
        raise RuntimeError(
            f"collider {source_object.name_full!r} cloth_friction is "
            "outside [0, 80]")
    if not 0.0 <= config.damping <= 1.0:
        raise RuntimeError(
            f"collider {source_object.name_full!r} damping is outside [0, 1]")
    if not 0.0 <= config.effector_absorption <= 1.0:
        raise RuntimeError(
            f"collider {source_object.name_full!r} absorption is "
            "outside [0, 1]")
    if use_culling != use_normal:
        raise RuntimeError(
            f"collider {source_object.name_full!r} must select an explicit "
            "surface contract: culling+normal for one-sided, or neither "
            "for two-sided")
    config.sidedness = (
        CType.GPUCLOTH_COLLIDER_ONE_SIDED_NORMAL
        if use_culling else CType.GPUCLOTH_COLLIDER_TWO_SIDED)

    record = CType.GPUClothCollectionRecord()
    record.struct_size = sizeof(record)
    record.record_version = 1
    record.object_type = _collection_object_type(
        occurrence["source_object"])
    record.record_flags = occurrence["record_flags"]
    record.object_id = occurrence["object_id"]
    record.instance_id = occurrence["instance_id"]
    record.source_collection_id = collection_id
    record.topology_generation = topology_generation
    record.geometry_generation = snapshot_generation
    record.modifier_index = modifier_index
    record.payload_kind = CType.GPUCLOTH_COLLECTION_COLLISION
    record.payload_address = addressof(config)
    return {
        "record": record,
        "config": config,
        "positions_previous": previous_array,
        "positions_current": current_array,
        "positions_next": next_array,
        "triangles": triangle_array,
        "history_key": history_key,
        "history_next": {
            "topology_generation": topology_generation,
            "geometry_generation": int(snapshot_generation),
            "local_positions": tuple(
                value for value in local_positions),
            "matrix_signature": tuple(
                value for value in matrix_signature),
            "previous_positions": tuple(
                value for value in current_positions),
            "current_positions": tuple(
                value for value in cloth_local_positions),
        },
    }


def _named_value(name, value_type, value):
    named = CType.GPUClothNamedValue()
    encoded = str(name).encode("ascii")
    if not encoded or len(encoded) >= 48:
        raise RuntimeError(f"invalid effector named value {name!r}")
    named.name = encoded
    named.value_type = int(value_type)
    if value_type == CType.GPUCLOTH_VALUE_FLOAT32:
        named.value_bits = struct.unpack(
            "<I", struct.pack("<f", float(value)))[0]
    elif value_type == CType.GPUCLOTH_VALUE_FLOAT64:
        named.value_bits = struct.unpack(
            "<Q", struct.pack("<d", float(value)))[0]
    elif value_type == CType.GPUCLOTH_VALUE_INT32:
        named.value_bits = int(value) & 0xffffffff
    else:
        named.value_bits = int(value) & 0xffffffffffffffff
    return named


def _validate_effector_field(field, source):
    field_type = str(getattr(field, "type", "NONE"))
    label = f"effector {source.name_full!r}"
    if field_type not in _FIELD_TYPE_MAP:
        raise RuntimeError(
            f"{label} has unknown active field type {field_type!r}")
    if field_type not in _SUPPORTED_FIELD_TYPES:
        raise RuntimeError(
            f"{label} field type {field_type!r} is not representable")

    shape = str(getattr(field, "shape", "POINT"))
    if shape != 'POINT':
        raise RuntimeError(
            f"{label} shape {shape!r} requires unsupported field geometry")
    falloff = str(getattr(field, "falloff_type", "SPHERE"))
    if falloff != 'SPHERE':
        raise RuntimeError(
            f"{label} falloff {falloff!r} is not representable")
    z_direction = str(getattr(field, "z_direction", "BOTH"))
    if z_direction not in {'BOTH', 'POSITIVE', 'NEGATIVE'}:
        raise RuntimeError(
            f"{label} has unknown Z direction {z_direction!r}")

    unsupported_true = (
        "use_radial_min", "use_radial_max", "use_object_coords",
        "use_global_coords", "use_2d_force", "use_root_coords",
        "use_multiple_springs", "use_smoke_density",
        "use_gravity_falloff",
    )
    for name in unsupported_true:
        if bool(getattr(field, name, False)):
            raise RuntimeError(
                f"{label} setting {name!r} is not representable")
    if not bool(getattr(field, "apply_to_location", True)):
        raise RuntimeError(
            f"{label} with location application disabled is not "
            "representable")

    radial_values = (
        ("radial_min", 0.0),
        ("radial_max", 0.0),
        ("radial_falloff", 0.0),
    )
    for name, expected in radial_values:
        value = _bounded_float32(
            getattr(field, name, expected), f"{label} {name}",
            0.0, 1000.0 if name != "radial_falloff" else 10.0)
        if value != expected:
            raise RuntimeError(
                f"{label} non-default setting {name!r} is not "
                "representable")

    wind_factor = _bounded_float32(
        getattr(field, "wind_factor", 0.0), f"{label} wind factor",
        0.0, 1.0)
    expected_wind_factor = 1.0 if field_type == 'WIND' else 0.0
    if wind_factor != expected_wind_factor:
        raise RuntimeError(
            f"{label} non-default wind factor is not representable")

    strength = _bounded_float32(
        getattr(field, "strength", 0.0), f"{label} strength",
        -3.402823466e38, 3.402823466e38)
    flow = _bounded_float32(
        getattr(field, "flow", 0.0), f"{label} flow",
        -3.402823466e38, 3.402823466e38)
    noise = _bounded_float32(
        getattr(field, "noise", 0.0), f"{label} noise", 0.0, 10.0)
    if noise != 0.0:
        raise RuntimeError(
            f"{label} noise requires Blender RNG state that is not "
            "representable")
    size = _bounded_float32(
        getattr(field, "size", 0.0), f"{label} size",
        0.0, 3.402823466e38)
    damping = _bounded_float32(
        getattr(field, "harmonic_damping", 0.0), f"{label} damping",
        -3.402823466e38, 3.402823466e38)
    falloff_power = _bounded_float32(
        getattr(field, "falloff_power", 0.0),
        f"{label} falloff power", 0.0, 10.0)
    distance_min = _bounded_float32(
        getattr(field, "distance_min", 0.0),
        f"{label} minimum distance", 0.0, 1000.0)
    distance_max = _bounded_float32(
        getattr(field, "distance_max", 0.0),
        f"{label} maximum distance", 0.0, 3.402823466e38)
    use_min_distance = bool(
        getattr(field, "use_min_distance", False))
    use_max_distance = bool(
        getattr(field, "use_max_distance", False))
    use_absorption = bool(getattr(field, "use_absorption", False))
    if (
            use_min_distance and use_max_distance and
            distance_max < distance_min):
        raise RuntimeError(
            f"{label} maximum distance is below minimum distance")
    try:
        seed_source = getattr(field, "seed", 1)
        seed = int(seed_source)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} seed is not an integer") from exc
    if seed != seed_source or seed < 1 or seed > 128:
        raise RuntimeError(f"{label} seed is outside [1, 128]")

    return {
        "field_type": field_type,
        "shape": shape,
        "falloff": falloff,
        "z_direction": z_direction,
        "strength": strength,
        "flow": flow,
        "noise": noise,
        "size": size,
        "damping": damping,
        "seed": seed,
        "falloff_power": falloff_power,
        "use_min_distance": use_min_distance,
        "distance_min": distance_min,
        "use_max_distance": use_max_distance,
        "distance_max": distance_max,
        "use_absorption": use_absorption,
    }


def _effector_named_values(settings_values):
    values = [
        _named_value(
            "strength", CType.GPUCLOTH_VALUE_FLOAT32,
            settings_values["strength"]),
        _named_value(
            "flow", CType.GPUCLOTH_VALUE_FLOAT32,
            settings_values["flow"]),
        _named_value(
            "noise", CType.GPUCLOTH_VALUE_FLOAT32,
            settings_values["noise"]),
        _named_value(
            "size", CType.GPUCLOTH_VALUE_FLOAT32,
            settings_values["size"]),
        _named_value(
            "damping", CType.GPUCLOTH_VALUE_FLOAT32,
            settings_values["damping"]),
        _named_value(
            "seed", CType.GPUCLOTH_VALUE_INT32,
            settings_values["seed"]),
        _named_value(
            "falloff_power", CType.GPUCLOTH_VALUE_FLOAT32,
            settings_values["falloff_power"]),
        _named_value(
            "use_min_distance", CType.GPUCLOTH_VALUE_BOOL,
            settings_values["use_min_distance"]),
        _named_value(
            "distance_min", CType.GPUCLOTH_VALUE_FLOAT32,
            settings_values["distance_min"]),
        _named_value(
            "use_max_distance", CType.GPUCLOTH_VALUE_BOOL,
            settings_values["use_max_distance"]),
        _named_value(
            "distance_max", CType.GPUCLOTH_VALUE_FLOAT32,
            settings_values["distance_max"]),
        _named_value(
            "z_direction", CType.GPUCLOTH_VALUE_INT32,
            {'BOTH': 0, 'POSITIVE': 1, 'NEGATIVE': 2}.get(
                settings_values["z_direction"])),
    ]
    return values


def _matrix_payload(matrix):
    values = []
    for column in range(4):
        for row in range(4):
            value = float(matrix[row][column])
            if not math.isfinite(value):
                raise RuntimeError("effector matrix contains non-finite value")
            values.append(value)
    return (c_float * 16)(*values)


def _capture_effector_payload(
        occurrence, field, collection_id, snapshot_generation,
        cloth_inverse):
    source = occurrence["source_object"]
    settings_values = _validate_effector_field(field, source)
    matrix = _cloth_local_matrix(
        cloth_inverse, occurrence["matrix_world"],
        f"effector {source.name_full!r}")
    inverse = _matrix_inverse(
        matrix, f"effector {source.name_full!r} cloth-local transform")
    matrix_array = _matrix_payload(matrix)
    inverse_array = _matrix_payload(inverse)
    named_values_list = _effector_named_values(settings_values)
    named_values_type = (
        CType.GPUClothNamedValue * len(named_values_list))
    named_values = named_values_type(*named_values_list)

    config = CType.GPUClothEffectorConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_EFFECTORS
    config.header.config_version = 1
    config.object_id = occurrence["object_id"]
    config.collection_id = collection_id
    config.generation = snapshot_generation
    if settings_values["use_absorption"]:
        config.effector_flags |= (
            CType.GPUCLOTH_EFFECTOR_USE_ABSORPTION)
    config.field_type = _FIELD_TYPE_MAP[settings_values["field_type"]]
    config.shape_type = _FIELD_SHAPE_MAP[settings_values["shape"]]
    config.falloff_type = _FIELD_FALLOFF_MAP[
        settings_values["falloff"]]
    config.named_value_count = len(named_values)
    config.named_values_address = (
        addressof(named_values) if named_values else 0)
    _set_buffer_view(
        config.object_matrix, CType.GPUCLOTH_ELEMENT_FLOAT, 16,
        sizeof(c_float), addressof(matrix_array), snapshot_generation)
    _set_buffer_view(
        config.inverse_matrix, CType.GPUCLOTH_ELEMENT_FLOAT, 16,
        sizeof(c_float), addressof(inverse_array), snapshot_generation)

    record = CType.GPUClothCollectionRecord()
    record.struct_size = sizeof(record)
    record.record_version = 1
    record.object_type = _collection_object_type(
        occurrence["source_object"])
    record.record_flags = occurrence["record_flags"]
    record.object_id = occurrence["object_id"]
    record.instance_id = occurrence["instance_id"]
    record.source_collection_id = collection_id
    record.geometry_generation = snapshot_generation
    record.payload_kind = CType.GPUCLOTH_COLLECTION_EFFECTOR
    record.payload_address = addressof(config)
    return {
        "record": record,
        "config": config,
        "matrix": matrix_array,
        "inverse": inverse_array,
        "named_values": named_values,
    }


def _collection_snapshot_owner(
        cloth_obj, selection, collection_kind, snapshot_generation, payloads):
    records_type = CType.GPUClothCollectionRecord * len(payloads)
    records = records_type(
        *(payload["record"] for payload in payloads))
    config = CType.GPUClothCollectionSnapshotConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = (
        CType.GPUCLOTH_FEATURE_COLLISION_COLLECTION
        if collection_kind == CType.GPUCLOTH_COLLECTION_COLLISION
        else CType.GPUCLOTH_FEATURE_EFFECTOR_COLLECTION)
    config.header.config_version = 1
    config.cloth_id = _blender_session_uid(cloth_obj, "cloth object")
    config.collection_id = (
        selection["collection_id"] if selection is not None else 0)
    config.snapshot_generation = snapshot_generation
    config.collection_kind = collection_kind
    config.record_count = len(records)
    if records:
        _set_buffer_view(
            config.records, CType.GPUCLOTH_ELEMENT_COLLECTION_RECORD,
            len(records), sizeof(CType.GPUClothCollectionRecord),
            addressof(records), snapshot_generation)
    return {
        "config": config,
        "records": records,
        "payloads": payloads,
        "collection_kind": collection_kind,
        "collection_id": int(config.collection_id),
        "snapshot_generation": snapshot_generation,
    }


def _prepare_collection_snapshots(
        context, depsgraph, cloth_objects, snapshot_generation,
        collider_history):
    cloth_ids = {
        _blender_session_uid(_original_object(obj), "cloth object")
        for obj in cloth_objects}
    prepared = []
    for cloth_obj in cloth_objects:
        cloth_owner_id = _blender_session_uid(cloth_obj, "cloth object")
        settings = cloth_obj.GPUCloth
        try:
            evaluated_cloth = cloth_obj.evaluated_get(depsgraph)
            cloth_world = evaluated_cloth.matrix_world.copy()
        except (
                AttributeError, ReferenceError, RuntimeError,
                TypeError) as exc:
            raise RuntimeError(
                f"cannot evaluate cloth transform for "
                f"{cloth_obj.name_full!r}") from exc
        cloth_inverse = _matrix_inverse(
            cloth_world,
            f"cloth {cloth_obj.name_full!r} evaluated world transform")
        collision_selection = _collision_collection_selection(settings)
        effector_selection = _collection_selection(
            settings.effector_weights.collection, "effector")

        collision_payloads = []
        for occurrence in _depsgraph_occurrences(
                depsgraph, collision_selection, "collision"):
            if occurrence["object_id"] in cloth_ids:
                continue
            if getattr(occurrence["source_object"], "type", None) != 'MESH':
                continue
            modifier_info = _collision_modifier(occurrence, depsgraph)
            if modifier_info is None:
                continue
            modifier_index = modifier_info
            collision_payloads.append(_capture_collider_payload(
                occurrence, modifier_index, depsgraph, cloth_owner_id,
                collision_selection["collection_id"]
                if collision_selection is not None else 0,
                snapshot_generation, collider_history, cloth_inverse))

        effector_payloads = []
        for occurrence in _depsgraph_occurrences(
                depsgraph, effector_selection, "effector"):
            source = occurrence["source_object"]
            field = getattr(
                occurrence["evaluated_object"], "field",
                getattr(source, "field", None))
            field_type = getattr(field, "type", "NONE")
            if field_type == 'NONE':
                continue
            effector_payloads.append(_capture_effector_payload(
                occurrence, field,
                effector_selection["collection_id"]
                if effector_selection is not None else 0,
                snapshot_generation, cloth_inverse))

        effector_collection_id = (
            effector_selection["collection_id"]
            if effector_selection is not None else 0)

        prepared.append({
            "collision": _collection_snapshot_owner(
                cloth_obj, collision_selection,
                CType.GPUCLOTH_COLLECTION_COLLISION,
                snapshot_generation, collision_payloads),
            "effector": _collection_snapshot_owner(
                cloth_obj, effector_selection,
                CType.GPUCLOTH_COLLECTION_EFFECTOR,
                snapshot_generation, effector_payloads),
            "effector_weights": _prepare_effector_weights(
                settings, effector_collection_id),
        })
    return prepared


def _prepare_effector_weights(settings, collection_id):
    config = CType.GPUClothEffectorWeightsConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_EFFECTOR_WEIGHTS
    config.header.config_version = 1
    config.collection_id = int(collection_id)
    config.weight_count = 15
    weights = _make_effector_weights(settings.effector_weights)
    config.weights[:] = weights
    return {
        "config": config,
        "weights": weights,
        "collection_id": int(collection_id),
    }


def _configure_effector_weights(dll, clmd, owner):
    config = owner["config"]
    result = int(dll.SIM_configure_cloth_feature(
        clmd,
        cast(pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed effector weights rejected with {result}")
    return result


def _collection_record_identity(record):
    return (
        int(record.object_type),
        int(record.record_flags),
        int(record.object_id),
        int(record.instance_id),
        int(record.source_collection_id),
        int(record.topology_generation),
        int(record.geometry_generation),
        int(record.modifier_index),
        int(record.payload_kind),
    )


def _query_collection_snapshot(dll, clmd, owner, transaction_id):
    expected_records = owner["records"]
    query = CType.GPUClothCollectionQuery()
    query.struct_size = sizeof(query)
    query.query_version = 1
    query.collection_kind = owner["collection_kind"]
    probe_result = int(dll.SIM_query_cloth_collection(
        clmd, pointer(query)))
    expected_probe_result = (
        CType.GPUCLOTH_ABI_COUNT_MISMATCH
        if expected_records else CType.GPUCLOTH_ABI_OK)
    if (probe_result != expected_probe_result or
            int(query.record_count) != 0 or
            int(query.required_count) != len(expected_records)):
        raise RuntimeError(
            "collection count probe returned a partial or wrong result")

    queried_type = CType.GPUClothCollectionRecord * len(expected_records)
    queried = queried_type()
    if expected_records:
        query.record_capacity = len(queried)
        query.records_address = addressof(queried)
        result = int(dll.SIM_query_cloth_collection(
            clmd, pointer(query)))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"collection query rejected with {result}")
    if (int(query.record_count) != len(expected_records) or
            int(query.required_count) != len(expected_records)):
        raise RuntimeError(
            "collection query count differs from committed snapshot")
    if int(query.transaction_id) != int(transaction_id):
        raise RuntimeError(
            "collection query returned a different transaction")
    if int(query.snapshot_generation) != int(owner["snapshot_generation"]):
        raise RuntimeError(
            "collection query returned a different generation")
    if (int(query.cloth_id) != int(owner["config"].cloth_id) or
            int(query.collection_id) != int(owner["collection_id"])):
        raise RuntimeError(
            "collection query returned a different Blender owner")
    expected = tuple(
        _collection_record_identity(record)
        for record in expected_records)
    actual = tuple(
        _collection_record_identity(record)
        for record in queried)
    if actual != expected:
        raise RuntimeError(
            "collection query order/identity differs from staged snapshot")
    return {
        "transaction_id": int(query.transaction_id),
        "snapshot_generation": int(query.snapshot_generation),
        "collection_id": int(query.collection_id),
        "record_count": int(query.record_count),
        "records": actual,
    }


def _next_collider_history(prepared_collections):
    history = {}
    for owners in prepared_collections:
        for payload in owners["collision"]["payloads"]:
            key = tuple(payload["history_key"])
            if key in history:
                raise RuntimeError(
                    "collider history contains a duplicate occurrence key")
            candidate = payload["history_next"]
            history[key] = {
                "topology_generation": int(
                    candidate["topology_generation"]),
                "geometry_generation": int(
                    candidate["geometry_generation"]),
                "local_positions": tuple(
                    candidate["local_positions"]),
                "matrix_signature": tuple(
                    candidate["matrix_signature"]),
                "previous_positions": tuple(
                    candidate["previous_positions"]),
                "current_positions": tuple(
                    candidate["current_positions"]),
            }
    return history


def _commit_frame_inputs(
        dll, clmds, prepared_collections, prepared_pins,
        prepared_dynamic_meshes, source_generation):
    global _collider_history
    if not (
            len(clmds) == len(prepared_collections) ==
            len(prepared_pins) == len(prepared_dynamic_meshes)):
        raise RuntimeError("staged frame input owners are not aligned")
    next_collider_history = _next_collider_history(
        prepared_collections)
    transaction = CType.GPUClothCollectionTransactionConfig()
    transaction.struct_size = sizeof(transaction)
    transaction.transaction_version = 1
    transaction.source_generation = int(source_generation)
    transaction_id = c_uint64()
    result = int(dll.SIM_begin_collection_transaction(
        pointer(transaction), pointer(transaction_id)))
    if result != CType.GPUCLOTH_ABI_OK or not transaction_id.value:
        raise RuntimeError(
            f"collection transaction begin rejected with {result}")

    committed = False
    try:
        for clmd, owners, pin_owner, dynamic_owner in zip(
                clmds, prepared_collections, prepared_pins,
                prepared_dynamic_meshes):
            for key in ("collision", "effector"):
                owner = owners[key]
                result = int(dll.SIM_stage_cloth_collection(
                    transaction_id.value, clmd, pointer(owner["config"])))
                if result != CType.GPUCLOTH_ABI_OK:
                    raise RuntimeError(
                        f"{key} collection stage rejected with {result}")
            result = int(dll.SIM_stage_cloth_pin_snapshot(
                transaction_id.value, clmd,
                pointer(pin_owner["config"])))
            if result != CType.GPUCLOTH_ABI_OK:
                raise RuntimeError(
                    f"pin snapshot stage rejected with {result}")
            if dynamic_owner is not None:
                result = int(dll.SIM_stage_cloth_mesh_state(
                    transaction_id.value, clmd,
                    pointer(dynamic_owner["config"])))
                if result != CType.GPUCLOTH_ABI_OK:
                    raise RuntimeError(
                        f"dynamic mesh stage rejected with {result}")
        result = int(dll.SIM_commit_collection_transaction(
            transaction_id.value))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"collection transaction commit rejected with {result}")
        committed = True
        # Native commit publishes both history and its source generation.
        _collider_history = next_collider_history
        _input_generation["value"] = int(source_generation)
    finally:
        if not committed:
            abort_result = int(dll.SIM_abort_collection_transaction(
                transaction_id.value))
            if abort_result != CType.GPUCLOTH_ABI_OK:
                raise RuntimeError(
                    f"collection transaction abort rejected with "
                    f"{abort_result}")

    snapshots = []
    for clmd, owners in zip(clmds, prepared_collections):
        status = CType.GPUClothCollectionStatus()
        status.struct_size = sizeof(status)
        status.status_version = 1
        result = int(dll.SIM_get_cloth_collection_status(
            clmd, pointer(status)))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"collection status rejected with {result}")
        collision = _query_collection_snapshot(
            dll, clmd, owners["collision"], transaction_id.value)
        effector = _query_collection_snapshot(
            dll, clmd, owners["effector"], transaction_id.value)
        status_flags = int(status.status_flags)
        if ((status_flags &
                CType.GPUCLOTH_COLLECTION_STATUS_CONFIGURED) == 0 or
                (status_flags &
                 CType.GPUCLOTH_COLLECTION_STATUS_STAGED) != 0 or
                int(status.last_error) != CType.GPUCLOTH_ABI_OK or
                int(status.transaction_id) != int(transaction_id.value) or
                int(status.snapshot_generation) != int(source_generation) or
                int(status.collection_id) != 0 or
                int(status.collision_record_count) !=
                len(owners["collision"]["records"]) or
                int(status.effector_record_count) !=
                len(owners["effector"]["records"]) or
                int(status.record_count) != (
                    len(owners["collision"]["records"]) +
                    len(owners["effector"]["records"])) or
                int(status.commit_generation) == 0):
            raise RuntimeError(
                "collection status differs from committed transaction")
        snapshots.append({
            "status": {
                "status_flags": int(status.status_flags),
                "last_error": int(status.last_error),
                "transaction_id": int(status.transaction_id),
                "snapshot_generation": int(status.snapshot_generation),
                "collection_id": int(status.collection_id),
                "collision_record_count": int(
                    status.collision_record_count),
                "effector_record_count": int(status.effector_record_count),
                "record_count": int(status.record_count),
                "commit_generation": int(status.commit_generation),
            },
            "collision": collision,
            "effector": effector,
        })
    return snapshots


def _publish_frame_inputs(context, depsgraph):
    if not (
            len(g_clothOBJs) == len(g_simulationOBJs) ==
            len(g_clmd) == len(_pin_snapshot_states) ==
            len(_dynamic_mesh_states)):
        raise RuntimeError("frame input owners are not aligned")
    generation = int(_input_generation["value"]) + 1

    # Capture every Blender input first. No native owner sees a partial
    # dependency-graph evaluation if later validation fails.
    bindings = [{
        "render_object": cloth_obj,
        "simulation_object": simulation_obj,
        "uses_proxy": simulation_obj is not cloth_obj,
    } for cloth_obj, simulation_obj in zip(
        g_clothOBJs, g_simulationOBJs)]
    plans = _plan_modifier_evaluation(bindings)
    for plan in plans:
        if any(_modifier_visibility(plan["cloth_modifier"])):
            raise RuntimeError(
                f"CPU Cloth ownership changed on "
                f"{plan['cloth_object'].name_full!r}; reprepare is "
                "required")
    modifier_states = _disable_modifier_input_stack(plans)
    try:
        context.view_layer.update()
        capture_depsgraph = context.evaluated_depsgraph_get()
        prepared_collections = _prepare_collection_snapshots(
            context, capture_depsgraph, g_clothOBJs, generation,
            _collider_history)
        prepared_pins = []
        prepared_dynamic_meshes = []
        for index, (cloth_obj, simulation_obj) in enumerate(zip(
                g_clothOBJs, g_simulationOBJs)):
            prepared_pins.append(_capture_pin_snapshot(
                cloth_obj, simulation_obj, capture_depsgraph,
                _pin_snapshot_states[index]["topology_generation"],
                generation))
            dynamic_state = _dynamic_mesh_states[index]
            enabled = bool(cloth_obj.GPUCloth.use_dynamic_mesh)
            if enabled != bool(dynamic_state["enabled"]):
                raise RuntimeError(
                    "dynamic mesh cannot be enabled or disabled after "
                    "prepare")
            prepared_dynamic_meshes.append(
                _capture_dynamic_mesh_snapshot(
                    cloth_obj, simulation_obj, capture_depsgraph,
                    dynamic_state["topology_generation"], generation))
    finally:
        _restore_modifier_visibility(modifier_states)
        try:
            context.view_layer.update()
        except (AttributeError, ReferenceError, RuntimeError):
            pass
    prepared_pin_owners = [
        prepare_pin_snapshot(CType, snapshot)
        for snapshot in prepared_pins]
    prepared_dynamic_owners = [
        _prepare_dynamic_mesh_state(
            snapshot, _dynamic_mesh_states[index]["accepted"])
        for index, snapshot in enumerate(prepared_dynamic_meshes)]

    if len(_effector_weight_states) != len(prepared_collections):
        raise RuntimeError("effector weight owners are not aligned")
    for state, owners in zip(
            _effector_weight_states, prepared_collections):
        current = owners["effector_weights"]
        if (
                int(state["collection_id"]) !=
                    int(current["collection_id"]) or
                tuple(state["weights"]) != tuple(current["weights"])):
            raise RuntimeError(
                "effector weights or collection changed; reprepare is "
                "required")
    committed_collections = _commit_frame_inputs(
        g_dll, g_clmd, prepared_collections, prepared_pin_owners,
        prepared_dynamic_owners, generation)
    _collection_snapshots.clear()
    _collection_snapshots.extend(committed_collections)

    for index, snapshot in enumerate(prepared_pins):
        _pin_snapshot_states[index]["frame_generation"] = generation
        _pin_snapshot_states[index]["snapshot"] = snapshot
    for index, snapshot in enumerate(prepared_dynamic_meshes):
        _dynamic_mesh_states[index]["pending"] = snapshot


def _accept_dynamic_mesh_snapshot(index):
    state = _dynamic_mesh_states[index]
    pending = state["pending"]
    if pending is None:
        return
    state["accepted"] = pending
    state["pending"] = None

_cache_playback_guard  = {'active': False}
_initial_positions     = []     # list[np.ndarray] — rest positions per cloth object
_bake_range            = {'start': 1, 'end': 250}
_simulation_frame_state = {'last_solved': None}
_cache_source_state = {'generation': 0}
_input_generation = {'value': 0}


def _cache_hash_value(hasher, label, value):
    hasher.update(label.encode('utf-8'))
    hasher.update(b'\0')
    if hasattr(value, "to_tuple"):
        value = value.to_tuple()
    try:
        encoded = repr(tuple(value)).encode('utf-8')
    except TypeError:
        encoded = repr(value).encode('utf-8')
    hasher.update(encoded)
    hasher.update(b'\0')


def _cache_hash_rna_scalars(hasher, label, owner, excluded=()):
    excluded = set(excluded)
    for prop in sorted(
            owner.bl_rna.properties, key=lambda item: item.identifier):
        identifier = prop.identifier
        if identifier == "rna_type" or identifier in excluded:
            continue
        if prop.type not in {'BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING'}:
            continue
        try:
            value = getattr(owner, identifier)
        except (AttributeError, TypeError):
            continue
        _cache_hash_value(hasher, f"{label}.{identifier}", value)


def _cache_source_generation(scene):
    """Stable fingerprint of cache-affecting addon inputs, never solved output."""
    hasher = hashlib.blake2b(digest_size=8, person=b"GPUCloth")
    helper = scene.gpu_cloth_helper
    _cache_hash_rna_scalars(
        hasher, "scene", helper,
        excluded={
            "cache_dir", "cache_index", "cache_name", "use_disk_cache",
            "use_external_cache", "external_cache_dir",
            "use_library_path", "cache_compression",
            "bake_start", "bake_end", "bake_progress",
            "is_baked", "is_baking", "is_outdated", "is_frame_skip",
            "cache_info", "cached_frame_count", "playback_mode",
        })
    _cache_hash_value(hasher, "render.fps", scene.render.fps)
    _cache_hash_value(hasher, "render.fps_base", scene.render.fps_base)

    cloth_objects = sorted(
        (obj for obj in scene.objects
         if obj.type == 'MESH' and hasattr(obj, "GPUCloth")
         and obj.GPUCloth.is_active),
        key=lambda obj: obj.name_full)
    for index, obj in enumerate(cloth_objects):
        _cache_hash_value(hasher, f"cloth[{index}].name", obj.name_full)
        _cache_hash_value(
            hasher, f"cloth[{index}].matrix_world",
            tuple(tuple(row) for row in obj.matrix_world))
        _cache_hash_rna_scalars(
            hasher, f"cloth[{index}].settings", obj.GPUCloth,
            excluded={
                "cpu_sync_copied", "cpu_sync_unsupported",
                "cpu_sync_blockers", "cpu_sync_errors",
                "cpu_sync_report",
            })
        mesh = obj.data
        if index < len(_initial_positions):
            rest = np.asarray(
                _initial_positions[index], dtype=np.float32).reshape(-1)
        else:
            rest = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", rest)
        hasher.update(f"cloth[{index}].rest".encode('utf-8'))
        hasher.update(rest.tobytes())
        _cache_hash_value(
            hasher, f"cloth[{index}].edges",
            tuple(tuple(edge.vertices) for edge in mesh.edges))
        _cache_hash_value(
            hasher, f"cloth[{index}].polygons",
            tuple(tuple(poly.vertices) for poly in mesh.polygons))
        for group_index, group in enumerate(obj.vertex_groups):
            weights = []
            for vertex in mesh.vertices:
                try:
                    weights.append(float(group.weight(vertex.index)))
                except RuntimeError:
                    weights.append(0.0)
            _cache_hash_value(
                hasher, f"cloth[{index}].vgroup[{group_index}]",
                (group.name, tuple(weights)))

    collision_objects = sorted(
        (obj for obj in scene.objects
         if obj.type == 'MESH' and any(
             modifier.type == 'COLLISION' for modifier in obj.modifiers)),
        key=lambda obj: obj.name_full)
    for index, obj in enumerate(collision_objects):
        mesh = obj.data
        _cache_hash_value(
            hasher, f"collider[{index}].matrix_world",
            tuple(tuple(row) for row in obj.matrix_world))
        coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        hasher.update(f"collider[{index}].coords".encode('utf-8'))
        hasher.update(coords.tobytes())
        _cache_hash_value(
            hasher, f"collider[{index}].polygons",
            tuple(tuple(poly.vertices) for poly in mesh.polygons))
        for modifier in obj.modifiers:
            if modifier.type == 'COLLISION':
                _cache_hash_rna_scalars(
                    hasher, f"collider[{index}].modifier", modifier)

    value = int.from_bytes(hasher.digest(), "little")
    return value or 1


def _cache_status_update(operation, scene, error_code=0, frame=-1):
    generation = _cache_source_generation(scene)
    if (operation == CType.GPUCLOTH_CACHE_STATUS_SOURCE_CHANGED and
            frame < 0):
        frame = int(scene.gpu_cloth_helper.bake_start)
    update = CType.GPUClothCacheStatusUpdate()
    update.header.struct_size = sizeof(update)
    update.header.feature_id = CType.GPUCLOTH_FEATURE_CACHE_STATUS
    update.header.config_version = 1
    update.operation = operation
    update.frame = int(frame)
    update.error_code = int(error_code)
    update.source_generation = generation
    result = int(g_dll.SIM_update_cache_status(pointer(update)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed cache status update {operation} rejected with {result}")
    _cache_source_state['generation'] = generation
    return generation


def _query_cache_status():
    status = CType.GPUClothCacheStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    result = int(g_dll.SIM_get_cache_status(pointer(status)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"typed cache status query rejected with {result}")
    return status


def _sync_cache_status(scene):
    status = _query_cache_status()
    helper = scene.gpu_cloth_helper
    helper.is_baking = bool(
        status.flags & CType.GPUCLOTH_CACHE_STATUS_BAKING)
    helper.is_baked = bool(
        status.flags & CType.GPUCLOTH_CACHE_STATUS_BAKED)
    helper.is_outdated = bool(
        status.flags & CType.GPUCLOTH_CACHE_STATUS_OUTDATED)
    helper.is_frame_skip = bool(
        status.flags & CType.GPUCLOTH_CACHE_STATUS_FRAME_SKIP)
    helper.cached_frame_count = int(status.cached_frame_count)
    helper.cache_info = bytes(status.info).split(b'\0', 1)[0].decode(
        'utf-8', errors='replace')
    if helper.is_outdated or helper.is_frame_skip:
        helper.playback_mode = False
    return status


def _bind_optional_runtime_hooks(dll):
    """Bind runtime lifecycle exports when present in newer DLL builds."""
    try:
        dll.SIM_initialize_runtime.argtypes = []
        dll.SIM_initialize_runtime.restype = c_bool
    except AttributeError:
        pass

    try:
        dll.SIM_shutdown_runtime.argtypes = []
        dll.SIM_shutdown_runtime.restype = c_bool
    except AttributeError:
        pass

    try:
        dll.SIM_get_cache_feature_config.argtypes = [
            POINTER(CType.GPUClothCacheConfig)]
        dll.SIM_get_cache_feature_config.restype = c_uint
    except AttributeError:
        pass


def _initialize_runtime_if_available():
    global g_runtime_initialized

    g_runtime_initialized = False
    if g_dll is None:
        return True
    try:
        if not g_dll.SIM_initialize_runtime():
            return False
        g_runtime_initialized = True
    except AttributeError:
        pass
    return True


def _store_initial_positions():
    global _initial_positions
    _initial_positions.clear()
    for cloth_obj in g_clothOBJs:
        nV = len(cloth_obj.data.vertices)
        pos = np.empty(nV * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", pos)
        _initial_positions.append(pos.copy())


def _restore_initial_positions():
    for i, cloth_obj in enumerate(g_clothOBJs):
        if i < len(_initial_positions):
            cloth_obj.data.vertices.foreach_set("co", _initial_positions[i])
            cloth_obj.data.update()
            cloth_obj.data.update_tag()


def _load_cached_frame(scene, depsgraph, frame):
    cache_dir = _active_cache_path(scene).encode('utf-8')
    if not g_dll.Cache_has_frame(frame, cache_dir):
        return False
    updated = False
    for i, cloth_obj in enumerate(g_clothOBJs):
        if i >= len(g_clmd):
            break
        nV = len(cloth_obj.data.vertices)
        pos = (c_float * (nV * 3))()
        loaded = g_dll.Cache_load_frame_gpu(
            frame, g_clmd[i], c_size_t(nV), cache_dir)
        if not loaded:
            loaded = g_dll.Cache_prefetch_frame(
                frame, c_size_t(nV), cache_dir)
        if loaded and g_dll.Cache_get_frame_positions(
                frame, pos, c_size_t(nV)):
            flat = np.frombuffer(pos, dtype=np.float32)
            cloth_obj.data.vertices.foreach_set("co", flat)
            cloth_obj.data.update()
            cloth_obj.data.update_tag()
            updated = True
    if updated:
        depsgraph.update()
    return updated


def _cache_input_change_handler(scene, depsgraph):
    if (_cache_playback_guard['active'] or g_dll is None or
            not g_clothOBJs or _cache_source_state['generation'] == 0):
        return
    helper = scene.gpu_cloth_helper
    if helper.cached_frame_count == 0:
        return
    relevant = False
    for update in depsgraph.updates:
        updated_id = update.id
        if isinstance(updated_id, (bpy.types.Scene, bpy.types.Object,
                                   bpy.types.Mesh)):
            relevant = True
            break
    if not relevant:
        return
    generation = _cache_source_generation(scene)
    if generation == _cache_source_state['generation']:
        return
    try:
        _cache_status_update(
            CType.GPUCLOTH_CACHE_STATUS_SOURCE_CHANGED, scene)
        _sync_cache_status(scene)
    except (OSError, RuntimeError):
        helper.playback_mode = False


def _frame_change_handler(scene, depsgraph):
    if _teardown_failure:
        return
    if _cache_playback_guard['active']:
        return
    scene_s = scene.gpu_cloth_helper
    if g_dll is None or not g_clothOBJs:
        return
    _cache_playback_guard['active'] = True
    try:
        frame = scene.frame_current

        if frame < _bake_range['start']:
            if _initial_positions:
                _restore_initial_positions()
                depsgraph.update()
            return

        if scene_s.is_baked and scene_s.playback_mode:
            if not _load_cached_frame(scene, depsgraph, frame):
                if _initial_positions:
                    _restore_initial_positions()
                    depsgraph.update()
        elif not scene_s.is_baked:
            last_solved = _simulation_frame_state['last_solved']
            if last_solved is not None and frame <= last_solved:
                _load_cached_frame(scene, depsgraph, frame)
                return
            if frame > scene_s.bake_end:
                return
            first = max(
                2, int(scene_s.bake_start),
                (last_solved + 1) if last_solved is not None else 2)
            for solve_frame in range(first, frame + 1):
                if scene.frame_current != solve_frame:
                    scene.frame_set(solve_frame)
                try:
                    result = bpy.ops.gpucloth.update_simulation()
                except RuntimeError:
                    return
                if 'FINISHED' not in result:
                    return
    finally:
        _cache_playback_guard['active'] = False


# ===========================================================================
#  Вспомогательные функции
# ===========================================================================

def free_gpu_memory(context=None, shutdown_runtime=False):
    """Release owners only after native solver teardown is confirmed."""
    global g_dll, g_scene, g_obj, g_mesh, g_clmd
    global g_clothOBJs, g_simulationOBJs, g_clothCollisionOBJs, g_proxy_handles
    global _teardown_failure, _collider_history
    global g_runtime_initialized

    if (g_dll is None and (
            _runtime_owners_retained() or
            (shutdown_runtime and g_runtime_initialized))):
        print(
            "free_gpu_memory: native DLL unavailable while runtime or "
            "owners are retained")
        _teardown_failure = True
        return False

    if g_dll is not None:
        proxy_failed = False
        for index, handle in enumerate(g_proxy_handles):
            if handle is None:
                continue
            try:
                g_dll.ProxySim_free(handle)
                g_proxy_handles[index] = None
            except Exception as exc:
                print(f"free_gpu_memory: ProxySim_free() failed: {exc}")
                proxy_failed = True
        if proxy_failed:
            print(
                "free_gpu_memory: proxy teardown incomplete; "
                "backing owners retained")
            _teardown_failure = True
            return False

        try:
            if shutdown_runtime and g_runtime_initialized:
                try:
                    solver_freed = bool(g_dll.SIM_shutdown_runtime())
                except AttributeError:
                    solver_freed = bool(g_dll.FreeSolverData())
                if solver_freed:
                    g_runtime_initialized = False
            else:
                solver_freed = bool(g_dll.FreeSolverData())
        except Exception as e:
            print(f"free_gpu_memory: native teardown failed: {e}")
            _teardown_failure = True
            return False
        if not solver_freed:
            print(
                "free_gpu_memory: native teardown returned false; "
                "owners retained")
            _teardown_failure = True
            return False

    try:
        from . import cloth_settings_bridge
        for cloth_obj in tuple(g_clothOBJs):
            cloth_settings_bridge.apply_modifier_ownership(
                cloth_obj, "CPU")
    except (
            AttributeError, ReferenceError, RuntimeError,
            TypeError) as exc:
        print(
            f"free_gpu_memory: CPU Cloth visibility restore failed: {exc}")
        _teardown_failure = True
        return False

    g_scene              = None
    g_obj                = []
    g_clmd               = []
    g_mesh               = []
    g_clothOBJs          = []
    g_simulationOBJs     = []
    g_clothCollisionOBJs = []
    g_proxy_handles      = []
    _collision_keepalive.clear()
    _solver_diagnostics.clear()
    _pin_snapshot_states.clear()
    _dynamic_mesh_states.clear()
    _collection_snapshots.clear()
    _effector_weight_states.clear()
    _collider_history = {}
    _drape_status_by_uid.clear()
    _initial_positions.clear()
    _live_arrays.clear()
    _simulation_frame_state['last_solved'] = None
    _cache_source_state['generation'] = 0
    _input_generation['value'] = 0

    if context is not None and hasattr(context.scene, 'gpu_cloth_springs_built'):
        context.scene.gpu_cloth_springs_built = False

    _teardown_failure = False
    return True


# ===========================================================================
#  Оператор: освобождение VRAM
# ===========================================================================

class GPUCloth_FreeVRAM(bpy.types.Operator):
    """Освободить память GPU от данных симуляции"""
    bl_idname = "gpucloth.destroy_simulation_data"
    bl_label  = "Free GPU Memory"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        if not free_gpu_memory(context):
            self.report({'ERROR'}, "Не удалось освободить GPU память.")
            return {'CANCELLED'}
        self.report({'INFO'}, "GPU память освобождена.")
        return {'FINISHED'}


# ===========================================================================
#  Оператор: загрузка DLL
# ===========================================================================

class GPUCloth_LoadDLL(bpy.types.Operator):
    """Загрузить нативную библиотеку GPUCloth (DLL / .so)"""
    bl_idname = "gpucloth.load_dll"
    bl_label  = "Load GPUCloth DLL"

    # ── Проверка CUDA ────────────────────────────────────────────────────────

    def check_cuda_support(self):
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True,
            )
            if ("CUDA Version" in result.stdout or
                    "CUDA UMD Version" in result.stdout):
                return True
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"nvidia-smi завершился с ошибкой: {e}")
        except FileNotFoundError:
            self.report({'ERROR'},
                "nvidia-smi не найден. Убедитесь что установлены драйверы NVIDIA.")
        return False

    # ── Загрузка библиотеки и привязка функций ───────────────────────────────

    def load_dll(self):
        global g_dll, _dll_directory_handles
        if g_dll is not None:
            return True  # уже загружена

        if not self.check_cuda_support():
            return False

        # Ищем DLL через vcu (относительно директории аддона, без хардкода)
        filename = vcu.get_dll_path("GPUCloth.dll")
        if filename is None:
            lib_dir   = vcu.get_lib_directory()
            addon_dir = vcu.get_addon_directory()
            self.report({'ERROR'},
                f"GPUCloth.dll не найдена. "
                f"Искали в: {lib_dir} , {addon_dir} , {addon_dir}\\build\\ . "
                f"Скопируйте GPUCloth.dll в {lib_dir}")
            return False

        dll_dir = os.path.dirname(filename)
        if dll_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, 'add_dll_directory'):
            candidates = [dll_dir]
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                candidates.extend((
                    os.path.join(cuda_path, "bin", "x64"),
                    os.path.join(cuda_path, "bin"),
                ))
            for path_entry in os.environ.get("PATH", "").split(os.pathsep):
                if (path_entry and
                        (os.path.isfile(os.path.join(path_entry, "cublas64_12.dll")) or
                         os.path.isfile(os.path.join(path_entry, "cusparse64_12.dll")))):
                    candidates.append(path_entry)
            for directory in dict.fromkeys(os.path.abspath(path) for path in candidates):
                if os.path.isdir(directory):
                    _dll_directory_handles.append(os.add_dll_directory(directory))

        try:
            g_dll = cdll.LoadLibrary(filename)
            self.report({'INFO'}, f"DLL загружена: {filename}")

            # ── Существующие функции ─────────────────────────────────────────

            g_dll.FillSolverData.argtypes = [POINTER(CType.Scene)]
            g_dll.FillSolverData.restype  = c_bool

            g_dll.FreeSolverData.argtypes = []
            g_dll.FreeSolverData.restype  = c_bool

            g_dll.BuildClothSprings.argtypes = [
                POINTER(CType.ClothModifierData), POINTER(CType.Mesh)]
            g_dll.BuildClothSprings.restype = c_bool

            g_dll.SIM_solver_cloth.argtypes = [
                POINTER(CType.ClothModifierData)]
            g_dll.SIM_solver_cloth.restype = c_bool

            g_dll.AddCloth.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.Mesh),
                POINTER(CType.Object),
            ]
            g_dll.AddCloth.restype = c_bool

            g_dll.RemoveCloth.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.Mesh),
                POINTER(CType.Object),
            ]
            g_dll.RemoveCloth.restype = c_bool

            g_dll.UpdateScene.argtypes = [POINTER(CType.Scene)]
            g_dll.UpdateScene.restype  = c_bool

            # ── Readback позиций вершин ──────────────────────────────────────
            #
            #   void SIM_get_cloth_verts(const ClothModifierData* clmd,
            #                            ClothVertex* out_verts, size_t count)
            #   Вместо прямого чтения mesh_ptr.contents.mvert — единственно
            #   корректный способ получить симулированные позиции.

            g_dll.SIM_get_cloth_verts.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.ClothVertex),
                c_size_t,
            ]
            g_dll.SIM_get_cloth_verts.restype = None

            g_dll.SIM_get_cloth_sewing_count.argtypes = [
                POINTER(CType.ClothModifierData)]
            g_dll.SIM_get_cloth_sewing_count.restype = c_size_t

            g_dll.SIM_sizeof_cloth_vertex.argtypes = []
            g_dll.SIM_sizeof_cloth_vertex.restype = c_size_t

            g_dll.SIM_offsetof_cloth_vertex_x.argtypes = []
            g_dll.SIM_offsetof_cloth_vertex_x.restype = c_size_t

            g_dll.SIM_get_product_abi_version.argtypes = [
                POINTER(CType.GPUClothABIVersion)]
            g_dll.SIM_get_product_abi_version.restype = c_bool

            g_dll.SIM_get_host_layout.argtypes = [
                POINTER(CType.GPUClothHostLayout)]
            g_dll.SIM_get_host_layout.restype = c_bool

            g_dll.SIM_get_descriptor_layout.argtypes = [
                POINTER(CType.GPUClothDescriptorLayout)]
            g_dll.SIM_get_descriptor_layout.restype = c_bool

            g_dll.SIM_get_feature_count.argtypes = []
            g_dll.SIM_get_feature_count.restype = c_size_t

            g_dll.SIM_get_feature_info.argtypes = [
                c_size_t, POINTER(CType.GPUClothFeatureInfo)]
            g_dll.SIM_get_feature_info.restype = c_bool

            g_dll.SIM_query_feature.argtypes = [
                c_uint, POINTER(CType.GPUClothFeatureInfo)]
            g_dll.SIM_query_feature.restype = c_bool

            g_dll.SIM_configure_feature.argtypes = [
                POINTER(CType.GPUClothFeatureConfigHeader)]
            g_dll.SIM_configure_feature.restype = c_uint

            g_dll.SIM_get_cache_feature_config.argtypes = [
                POINTER(CType.GPUClothCacheConfig)]
            g_dll.SIM_get_cache_feature_config.restype = c_uint

            g_dll.SIM_update_cache_status.argtypes = [
                POINTER(CType.GPUClothCacheStatusUpdate)]
            g_dll.SIM_update_cache_status.restype = c_uint

            g_dll.SIM_get_cache_status.argtypes = [
                POINTER(CType.GPUClothCacheStatus)]
            g_dll.SIM_get_cache_status.restype = c_uint

            g_dll.SIM_configure_cloth_feature.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothFeatureConfigHeader),
            ]
            g_dll.SIM_configure_cloth_feature.restype = c_uint

            g_dll.SIM_get_cloth_solver_diagnostics.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothDiagnosticsStatus),
            ]
            g_dll.SIM_get_cloth_solver_diagnostics.restype = c_uint

            g_dll.SIM_set_cloth_vertex_channel.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothVertexChannelConfig),
            ]
            g_dll.SIM_set_cloth_vertex_channel.restype = c_uint

            g_dll.SIM_set_cloth_pin_snapshot.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothPinSnapshotConfig),
            ]
            g_dll.SIM_set_cloth_pin_snapshot.restype = c_uint

            g_dll.SIM_stage_cloth_pin_snapshot.argtypes = [
                c_uint64,
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothPinSnapshotConfig),
            ]
            g_dll.SIM_stage_cloth_pin_snapshot.restype = c_uint

            g_dll.SIM_stage_cloth_mesh_state.argtypes = [
                c_uint64,
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothMeshStateConfig),
            ]
            g_dll.SIM_stage_cloth_mesh_state.restype = c_uint

            g_dll.SIM_begin_collection_transaction.argtypes = [
                POINTER(CType.GPUClothCollectionTransactionConfig),
                POINTER(c_uint64),
            ]
            g_dll.SIM_begin_collection_transaction.restype = c_uint

            g_dll.SIM_stage_cloth_collection.argtypes = [
                c_uint64,
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothCollectionSnapshotConfig),
            ]
            g_dll.SIM_stage_cloth_collection.restype = c_uint

            g_dll.SIM_commit_collection_transaction.argtypes = [c_uint64]
            g_dll.SIM_commit_collection_transaction.restype = c_uint

            g_dll.SIM_abort_collection_transaction.argtypes = [c_uint64]
            g_dll.SIM_abort_collection_transaction.restype = c_uint

            g_dll.SIM_query_cloth_collection.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothCollectionQuery),
            ]
            g_dll.SIM_query_cloth_collection.restype = c_uint

            g_dll.SIM_get_cloth_collection_status.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothCollectionStatus),
            ]
            g_dll.SIM_get_cloth_collection_status.restype = c_uint
            g_dll.SIM_validate_cloth_initial_state.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothPreparationConfig),
                POINTER(CType.GPUClothPreparationStatus),
            ]
            g_dll.SIM_validate_cloth_initial_state.restype = c_uint
            g_dll.SIM_get_cloth_preparation_status.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothPreparationStatus),
            ]
            g_dll.SIM_get_cloth_preparation_status.restype = c_uint
            g_dll.SIM_get_cloth_invariant_status.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothInvariantWitness),
            ]
            g_dll.SIM_get_cloth_invariant_status.restype = c_uint
            g_dll.SIM_begin_cloth_drape.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothDrapeConfig),
                POINTER(CType.GPUClothDrapeStatus),
            ]
            g_dll.SIM_begin_cloth_drape.restype = c_uint
            g_dll.SIM_step_cloth_drape.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothDrapeStatus),
            ]
            g_dll.SIM_step_cloth_drape.restype = c_uint
            g_dll.SIM_apply_cloth_drape.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothDrapeStatus),
            ]
            g_dll.SIM_apply_cloth_drape.restype = c_uint
            g_dll.SIM_cancel_cloth_drape.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothDrapeStatus),
            ]
            g_dll.SIM_cancel_cloth_drape.restype = c_uint
            _validate_product_abi(g_dll)
            _validate_descriptor_layout(g_dll)

            # ── ProxySim API ─────────────────────────────────────────────────
            #
            #   Симуляция грубого proxy-меша + апсэмплинг до hi-res.
            #   GPU path: SIM_solver_cloth() -> ClothVertex.x (proxy)
            #             → ProxySim_apply → hi-res позиции → foreach_set

            g_dll.ProxySim_create.argtypes = [
                c_int, c_int,           # hi_NX,    hi_NY
                c_int, c_int,           # proxy_NX, proxy_NY
                c_int, c_int,           # num_sheets, scene_type
                POINTER(c_float),       # proxy_rest_pos  [nProxy*3]
                POINTER(c_float),       # hi_rest_pos     [nHi*3]
                c_int, c_int,           # nProxy, nHi
            ]
            g_dll.ProxySim_create.restype = c_void_p  # ProxySimHandle*

            g_dll.ProxySim_apply.argtypes = [
                c_void_p,           # ProxySimHandle*
                POINTER(c_float),   # proxy_pos  [nProxy*3]  — вход
                POINTER(c_float),   # hi_out     [nHi*3]     — выход
            ]
            g_dll.ProxySim_apply.restype = None

            g_dll.ProxySim_hi_count.argtypes    = [c_void_p]
            g_dll.ProxySim_hi_count.restype     = c_int

            g_dll.ProxySim_proxy_count.argtypes = [c_void_p]
            g_dll.ProxySim_proxy_count.restype  = c_int

            g_dll.ProxySim_free.argtypes = [c_void_p]
            g_dll.ProxySim_free.restype  = None

            # ── Cache API ────────────────────────────────────────────────────
            #
            #   ЗАПИСЬ (Phase 1, CPU-destination):
            #     SIM_get_cloth_verts → float[] → Cache_write_frame_async
            #     C++ пишет в фоне: pinned RAM → DMA → NVMe
            #
            #   ЧТЕНИЕ (Phase 2, GPU-destination, zero-copy):
            #     Cache_load_frame_gpu: NVMe → DMA → D3D12 resource (VRAM)
            #       → CUDA external memory (view, не копирование)
            #       → scatter_gpu_kernel → ClothVertex.x
            #       → cudaMemcpyAsync D2H → h_flat → foreach_set (viewport)

            g_dll.Cache_write_frame_async.argtypes = [
                c_int,              # frame
                POINTER(c_float),   # positions [nVerts*3]
                c_size_t,           # nVerts
                c_char_p,           # cache_dir (UTF-8)
            ]
            g_dll.Cache_write_frame_async.restype = c_bool

            g_dll.Cache_load_frame_gpu.argtypes = [
                c_int,                              # frame
                POINTER(CType.ClothModifierData),   # clmd (для scatter в ClothVertex.x)
                c_size_t,                           # nVerts
                c_char_p,                           # cache_dir
            ]
            g_dll.Cache_load_frame_gpu.restype = c_bool

            g_dll.Cache_prefetch_frame.argtypes = [
                c_int,      # frame
                c_size_t,   # nVerts
                c_char_p,   # cache_dir
            ]
            g_dll.Cache_prefetch_frame.restype = c_bool

            g_dll.Cache_is_frame_ready.argtypes = [c_int]
            g_dll.Cache_is_frame_ready.restype  = c_bool

            # После Cache_load_frame_gpu (GPU-direct): D2H для foreach_set
            g_dll.Cache_get_frame_positions.argtypes = [
                c_int,              # frame
                POINTER(c_float),   # out_positions [nVerts*3]
                c_size_t,           # nVerts
            ]
            g_dll.Cache_get_frame_positions.restype = c_bool

            g_dll.Cache_free_frame.argtypes = [c_int]
            g_dll.Cache_free_frame.restype  = c_bool

            g_dll.Cache_clear_all.argtypes = [c_char_p]
            g_dll.Cache_clear_all.restype  = c_bool

            g_dll.Cache_has_frame.argtypes = [c_int, c_char_p]
            g_dll.Cache_has_frame.restype  = c_bool

            _bind_optional_runtime_hooks(g_dll)
            if not _initialize_runtime_if_available():
                self.report({'ERROR'}, "SIM_initialize_runtime() failed.")
                g_dll = None
                _close_dll_directories()

        except OSError as e:
            self.report({'ERROR'}, f"Не удалось загрузить DLL: {e}")
            g_dll = None
            _close_dll_directories()
        except (AttributeError, RuntimeError) as e:
            self.report({'ERROR'}, f"DLL ABI несовместим: {e}")
            g_dll = None
            _close_dll_directories()

        return g_dll is not None

    def execute(self, context):
        if not self.load_dll():
            self.report({'ERROR'}, "Не удалось загрузить DLL")
            return {'CANCELLED'}
        return {'FINISHED'}


# ===========================================================================
#  Оператор: выгрузка DLL
# ===========================================================================

class GPUCloth_UnloadDLL(bpy.types.Operator):
    """Выгрузить нативную библиотеку GPUCloth из Blender"""
    bl_idname = "gpucloth.unload_dll"
    bl_label  = "Unload GPUCloth DLL"

    @classmethod
    def poll(cls, context):
        return g_dll is not None

    def execute(self, context):
        global g_dll, _dll_directory_handles
        if g_dll is None:
            self.report({'WARNING'}, "DLL не загружена.")
            return {'CANCELLED'}
        if not free_gpu_memory(context, shutdown_runtime=True):
            self.report(
                {'ERROR'},
                "DLL retained because native solver teardown failed")
            return {'CANCELLED'}
        try:
            # Windows: FreeLibrary через kernel32
            handle = c_void_p(g_dll._handle)
            result = windll.kernel32.FreeLibrary(handle)
            if result == 0:
                import ctypes
                raise ctypes.WinError()
            self.report({'INFO'}, "DLL успешно выгружена.")
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка при выгрузке DLL: {e}")
            return {'CANCELLED'}
        g_dll = None
        _close_dll_directories()
        return {'FINISHED'}


# ===========================================================================
#  Оператор: подготовка симуляции
# ===========================================================================

class GPUCloth_PrepareSimulation(bpy.types.Operator):
    """Подготовить данные и загрузить ткань на GPU"""
    bl_idname = "gpucloth.prepare_simulation"
    bl_label  = "Prepare GPUCloth Simulation"

    @classmethod
    def poll(cls, context):
        return True

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def fill_MVertTri_from_Object(self, obj: bpy.types.Object):
        """Извлекает треугольную топологию из меша объекта.
        Совместимо с Blender 4.1+ через vcu.calc_mesh_loop_triangles()."""
        if obj.type != 'MESH':
            return None
        mesh = obj.data
        # В Blender 4.1+ нужен явный вызов calc_loop_triangles()
        loop_tris = vcu.calc_mesh_loop_triangles(mesh)
        mvert_tris = (CType.MVertTri * len(loop_tris))()
        for i, tri in enumerate(loop_tris):
            mvert_tris[i].tri[0] = tri.vertices[0]
            mvert_tris[i].tri[1] = tri.vertices[1]
            mvert_tris[i].tri[2] = tri.vertices[2]
        return mvert_tris

    def fill_Scene(self, context):
        global g_scene
        scene = context.scene
        g_scene = pointer(CType.Scene())
        g_scene.contents.flag = (
            bool(scene.rigidbody_world.enabled)
            if scene.rigidbody_world is not None else False)
        g_scene.contents.r = CType.RenderData(
            cfra=int(scene.frame_current),
            subframe=float(scene.frame_subframe),
            framelen=float(scene.render.frame_map_old),
            frs_sec=c_short(scene.render.fps),
        )
        g_scene.contents.physics_settings = CType.PhysicsSettings(
            gravity=(c_float * 3)(*scene.gravity),
            flag=CType.PHYS_GLOBAL_GRAVITY,
        )
        self.report({'INFO'}, "Scene заполнен")

    def fill_Object(self, OBJ: bpy.types.Object) -> POINTER(CType.Object):
        import numpy as np
        new_object = CType.Object()
        obmat = np.array(OBJ.matrix_world, dtype=np.float32)
        imat  = np.array(OBJ.matrix_world.inverted(), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                new_object.obmat[i][j] = obmat[i][j]
                new_object.imat[i][j]  = imat[i][j]
        new_object.pd = None

        for modif in OBJ.modifiers:
            if modif.type == 'COLLISION':
                tmp_collision = CType.CollisionModifierData()
                mvertType = CType.MVert * len(OBJ.data.vertices)
                mvert     = mvertType()
                for i, mv in enumerate(OBJ.data.vertices):
                    v = vcu.element_multiply(OBJ.matrix_world, mv.co)
                    mvert[i].co   = (c_float * 3)(*v)
                    mvert[i].flag = 0

                mvert_tri = self.fill_MVertTri_from_Object(OBJ)

                tmp_collision.x               = cast(mvert, POINTER(CType.MVert))
                tmp_collision.xnew            = cast(mvert, POINTER(CType.MVert))
                tmp_collision.xold            = cast(mvert, POINTER(CType.MVert))
                tmp_collision.current_xnew    = cast(mvert, POINTER(CType.MVert))
                tmp_collision.current_x       = cast(mvert, POINTER(CType.MVert))
                tmp_collision.current_v       = cast(mvert, POINTER(CType.MVert))
                tmp_collision.tri             = cast(mvert_tri, POINTER(CType.MVertTri))
                tmp_collision.mvert_num       = len(OBJ.data.vertices)
                loop_tris = vcu.calc_mesh_loop_triangles(OBJ.data)
                tmp_collision.tri_num         = len(loop_tris)
                tmp_collision.time_x          = -1000
                tmp_collision.time_xnew       = -1000
                tmp_collision.is_static       = True
                tmp_collision.bvhtree         = None
                new_object.modifiers          = pointer(tmp_collision)

                _collision_keepalive.extend(
                    [new_object, tmp_collision, mvert, mvert_tri])

        return pointer(new_object)

    def setMesh(self, context, snapshot) -> POINTER(CType.Mesh):
        if not snapshot:
            raise ValueError("Cloth input snapshot must not be None")
        try:
            mesh      = CType.Mesh()
            mesh.totedge = len(snapshot["edges"])
            mesh.totvert = snapshot["vertex_count"]
            mesh.totpoly = len(snapshot["polygons"])
            mesh.totloop = len(snapshot["loops"])

            # Вершины
            mvertType = CType.MVert * snapshot["vertex_count"]
            mvert     = mvertType()
            for i, position in enumerate(snapshot["positions"]):
                mvert[i].co   = (c_float * 3)(*position)
                mvert[i].flag = 0
            mesh.mvert = cast(mvert, POINTER(CType.MVert))

            # Рёбра (crease/bweight убраны в Blender 4.0 как прямые поля,
            # здесь используются для C++ структуры — всегда 0)
            medgeType = CType.MEdge * len(snapshot["edges"])
            medge     = medgeType()
            for index, edge in enumerate(snapshot["edges"]):
                medge[index].v1      = edge[0]
                medge[index].v2      = edge[1]
                medge[index].crease  = 0
                medge[index].bweight = 0
                medge[index].flag    = 35
            mesh.medge = cast(medge, POINTER(CType.MEdge))

            # Полигоны
            mpolyType = CType.MPoly * len(snapshot["polygons"])
            mpoly     = mpolyType()
            for index, polygon in enumerate(snapshot["polygons"]):
                mpoly[index].loopstart = polygon[0]
                mpoly[index].totloop   = polygon[1]
            mesh.mpoly = cast(mpoly, POINTER(CType.MPoly))

            # Loops
            mloopType = CType.MLoop * len(snapshot["loops"])
            mloop     = mloopType()
            for index, loop in enumerate(snapshot["loops"]):
                mloop[index].v = loop[0]
                mloop[index].e = loop[1]
            mesh.mloop = cast(mloop, POINTER(CType.MLoop))

            return pointer(mesh)
        except Exception as e:
            raise RuntimeError(f"setMesh: {e}") from e

    def setClothModifierData(self, context, OBJ) -> POINTER(CType.ClothModifierData):
        clmd     = CType.ClothModifierData()
        sim_parms = CType.ClothSimSettings()
        gs       = OBJ.GPUCloth
        gs_scene = context.scene.gpu_cloth_helper

        # ── Базовые параметры ─────────────────────────────────────────────
        sim_parms.mingoal        = gs.mingoal
        sim_parms.Cvi            = gs.air_viscosity
        sim_parms.Cdis           = 0.0
        sim_parms.gravity[0]     = gs_scene.gravity_x
        sim_parms.gravity[1]     = gs_scene.gravity_y
        sim_parms.gravity[2]     = gs_scene.gravity_z
        sim_parms.mass           = gs.vertex_mass
        sim_parms.structural     = gs.structural
        sim_parms.shear          = gs.shear
        sim_parms.bending        = gs.bending_stiffness
        sim_parms.vgroup_mass    = 0  # set via vertex group data injection
        sim_parms.stepsPerFrame  = gs.quality_step
        sim_parms.maxgoal        = gs.maxgoal
        sim_parms.velocity_smooth= 0.0
        sim_parms.collider_friction = 0.0
        sim_parms.shrink_min     = gs.shrink_min
        sim_parms.shrink_max     = gs.shrink_max
        sim_parms.vgroup_bend    = 0
        sim_parms.vgroup_struct  = 0
        sim_parms.vgroup_shear   = 0
        sim_parms.vgroup_shrink  = 0
        sim_parms.bending_damping= gs.bending_damping
        sim_parms.voxel_cell_size= 0.1
        sim_parms.tension        = gs.tension
        sim_parms.compression    = gs.compression
        sim_parms.tension_damp   = gs.tension_damp
        sim_parms.compression_damp = gs.compression_damp
        sim_parms.shear_damp     = gs.shear_damp
        sim_parms.max_tension    = gs.max_tension
        sim_parms.max_compression = gs.max_compression
        sim_parms.max_shear      = gs.max_shear
        sim_parms.max_bend       = gs.max_bend
        sim_parms.max_struct     = gs.max_struct
        sim_parms.max_sewing     = gs.max_sewing
        sim_parms.vel_damping    = gs.vel_damping

        # ── Internal Springs ───────────────────────────────────────────────
        sim_parms.internal_spring_max_length     = gs.internal_spring_max_length
        sim_parms.internal_spring_max_diversion  = gs.internal_spring_max_diversion
        sim_parms.vgroup_intern  = 0
        sim_parms.internal_tension     = gs.internal_tension
        sim_parms.internal_compression = gs.internal_compression
        sim_parms.max_internal_tension     = gs.max_internal_tension
        sim_parms.max_internal_compression = gs.max_internal_compression

        # ── Anisotropic stiffness ────────────────────────────────────────
        sim_parms.tension_u       = gs.tension_u
        sim_parms.tension_v       = gs.tension_v
        sim_parms.compression_u   = gs.compression_u
        sim_parms.compression_v   = gs.compression_v
        sim_parms.bending_u       = gs.bending_u
        sim_parms.bending_v       = gs.bending_v
        sim_parms.max_tension_u   = gs.max_tension_u
        sim_parms.max_tension_v   = gs.max_tension_v
        sim_parms.max_compression_u = gs.max_compression_u
        sim_parms.max_compression_v = gs.max_compression_v
        sim_parms.max_bend_u      = gs.max_bend_u
        sim_parms.max_bend_v      = gs.max_bend_v

        # ── Effector forces ────────────────────────────────────────────────
        sim_parms.eff_force_scale  = gs.eff_force_scale
        sim_parms.eff_wind_scale   = gs.eff_wind_scale
        sim_parms.effector_weights = None  # allocated separately if needed
        sim_parms.reset            = 0
        sim_parms.presets          = 2
        sim_parms.shapekey_rest    = 0
        sim_parms.defgoal          = gs.defgoal

        # ── Pressure ──────────────────────────────────────────────────────
        sim_parms.fluid_density    = gs.fluid_density
        sim_parms.pressure_factor  = gs.pressure_factor
        sim_parms.target_volume    = (
            gs.target_volume if gs.use_pressure_volume else 0.0)
        sim_parms.uniform_pressure_force = gs.uniform_pressure_force

        # ── Timing ────────────────────────────────────────────────────────
        sim_parms.time_scale       = gs.speed_multiplier
        sim_parms.timescale        = native_frame_timescale(
            context.scene, gs.speed_multiplier)
        sim_parms.dt               = 1
        sim_parms.avg_spring_len   = 0.0
        sim_parms.goalfrict        = gs.goalfrict
        sim_parms.goalspring       = gs.goalspring

        # ── Flags ─────────────────────────────────────────────────────────
        sim_parms.flags = 0
        if gs.use_internal_springs:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS
        if gs.use_internal_springs_normal:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS_NORMAL
        if gs.use_pressure:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_PRESSURE
        if gs.use_pressure_volume:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_PRESSURE_VOL
        if gs.use_dynamic_mesh:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_DYNAMIC_MESH

        # Модель изгиба
        bending_model = _effective_bending_model(gs)
        sim_parms.bending_model = (
            CType.CLOTH_BENDING_ANGULAR
            if bending_model in {'ANGULAR', 'SDB'}
            else CType.CLOTH_BENDING_LINEAR
        )

        # ─�� Тип солвера ───────────────────────────────────────────────────
        _SOLVER_MAP = {
            'XPBD':  CType.SOLVER_XPBD,
            'PD':    CType.SOLVER_PD,
            'MGPBD': CType.SOLVER_MGPBD,
            'Mil2':  CType.SOLVER_Mil2,
            'OGC':   CType.SOLVER_OGC,
        }
        sim_parms.solver_type = _SOLVER_MAP.get(
            gs.solver_type, CType.SOLVER_XPBD
        )

        # ── Tuneable solver config ───────────────────────────────────────
        sim_parms.solver_substeps     = 0
        sim_parms.solver_iterations   = gs.solver_iterations
        sim_parms.solver_omega        = gs.solver_omega
        sim_parms.solver_small_steps  = 1 if gs.use_small_steps else 0
        sim_parms.solver_adaptive     = 1 if gs.use_adaptive else 0
        sim_parms.solver_max_iterations = gs.solver_max_iterations
        sim_parms.solver_convergence_tol = gs.solver_convergence_tol
        sim_parms.solver_ptb_stretch  = gs.ptb_stretch
        sim_parms.solver_ptb_bending  = gs.ptb_bending
        sim_parms.solver_ptb_shear    = gs.ptb_shear
        sim_parms.solver_ptb_seam     = gs.ptb_seam
        sim_parms.solver_use_pt_budget = 1 if gs.use_per_type_budget else 0

        # ── Anisotropy ────────────────────────────────────────────────────
        sim_parms.use_anisotropy  = 1 if gs.use_anisotropy else 0
        sim_parms.tension_u       = gs.tension_u
        sim_parms.tension_v       = gs.tension_v
        sim_parms.compression_u   = gs.compression_u
        sim_parms.compression_v   = gs.compression_v
        sim_parms.bending_u       = gs.bending_u
        sim_parms.bending_v       = gs.bending_v
        sim_parms.max_tension_u   = gs.max_tension_u
        sim_parms.max_tension_v   = gs.max_tension_v
        sim_parms.max_compression_u = gs.max_compression_u
        sim_parms.max_compression_v = gs.max_compression_v
        sim_parms.max_bend_u      = gs.max_bend_u
        sim_parms.max_bend_v      = gs.max_bend_v

        clmd.sim_parms    = pointer(sim_parms)
        clmd.clothObject  = None

        # ── Параметры столкновений ────────────────────────────────────────
        coll_parms = pointer(CType.ClothCollSettings())
        coll_parms.contents.epsilon       = gs.epsilon
        coll_parms.contents.self_friction = gs.self_collision_friction
        coll_parms.contents.friction      = gs.collision_friction
        coll_parms.contents.damping       = gs.collision_damping
        coll_parms.contents.selfepsilon   = gs.selfepsilon
        coll_parms.contents.loop_count    = gs.collision_quality
        # Collection identity/records use the typed product snapshot owner;
        # the legacy DNA pointer is deliberately never a Python authority.
        coll_parms.contents.group         = None
        coll_parms.contents.vgroup_selfcol = 0
        object_collision_group = (
            gs.id_data.vertex_groups.get(gs.vgroup_objcol)
            if gs.vgroup_objcol else None)
        coll_parms.contents.vgroup_objcol = (
            object_collision_group.index + 1
            if object_collision_group is not None else 0)
        coll_parms.contents.clamp          = gs.clamp
        coll_parms.contents.self_clamp     = gs.self_clamp
        coll_parms.contents.flags = 0
        if gs.use_object_collision:
            coll_parms.contents.flags |= (
                CType.CLOTH_COLLSETTINGS_FLAG_ENABLED)
        if gs.use_self_collision:
            coll_parms.contents.flags |= CType.CLOTH_COLLSETTINGS_FLAG_SELF
        clmd.coll_parms = coll_parms

        # ── Результат солвера ─────────────────────────────────────────────
        solver_result = CType.ClothSolverResult()
        solver_result.status         = 0
        solver_result.max_iterations = 0
        solver_result.avg_iterations = 0
        solver_result.max_error      = 0.0
        solver_result.min_error      = 0.0
        solver_result.avg_error      = 0.0
        clmd.solver_result = pointer(solver_result)

        return pointer(clmd)

    # ── Execute ──────────────────────────────────────────────────────────────

    @_guard_prepare_teardown
    def execute(self, context):
        global g_dll, g_scene, g_obj, g_mesh, g_clmd
        global g_clothOBJs, g_simulationOBJs
        global g_clothCollisionOBJs, g_proxy_handles

        self._native_prepare_mutated = False

        # 1. Загружаем DLL если нужно
        if g_dll is None:
            bpy.ops.gpucloth.load_dll()
            if g_dll is None:
                self.report({'ERROR'}, "Не удалось загрузить DLL")
                return {'CANCELLED'}

        # 2. Сохраняем файл (DLL нужен путь к blend для ряда операций)
        if not bpy.data.is_saved:
            bpy.ops.wm.save_as_mainfile(
                filepath=bpy.app.tempdir + 'GPU_Cloth.blend',
                check_existing=False)
        elif bpy.data.is_dirty:
            bpy.ops.wm.save_as_mainfile(
                filepath=bpy.data.filepath, check_existing=False)

        # 3. Переходим в Object mode для корректного считывания данных.
        mode = bpy.context.active_object.mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # 4. Preflight every Blender-owned input before native mutation.
        view_layer_objects = tuple(
            getattr(context.view_layer, "objects", context.scene.objects))
        active_cloth = [
            item for item in view_layer_objects
            if item is not None
            and hasattr(item, 'GPUCloth')
            and item.GPUCloth.is_active
        ]
        if not active_cloth:
            self._native_prepare_mutated = True
            if not free_gpu_memory(context):
                self.report({'ERROR'}, "Не удалось освободить GPU память")
                return {'CANCELLED'}
            bpy.ops.object.mode_set(mode=mode)
            return {'FINISHED'}

        bindings = []
        try:
            for item in active_cloth:
                bindings.append(validate_proxy_binding(item, item.GPUCloth))
        except ProxyBindingError as exc:
            self.report({'ERROR'}, f"Proxy configuration rejected: {exc}")
            return {'CANCELLED'}

        simulation_objects = [
            binding["simulation_object"] for binding in bindings]
        try:
            modifier_plans = _plan_modifier_evaluation(bindings)
        except RuntimeError as exc:
            self.report(
                {'ERROR'}, f"Modifier evaluation rejected: {exc}")
            return {'CANCELLED'}

        modifier_states = []
        try:
            modifier_states = _disable_modifier_input_stack(
                modifier_plans)
            context.view_layer.update()
            depsgraph = context.evaluated_depsgraph_get()
            prepared_input_meshes = []
            prepared_rest_shapes = []
            prepared_pin_snapshots = []
            prepared_collision_configs = []
            prepared_material_features = []
            prepared_internal_configs = []
            prepared_pressure_features = []
            prepared_stiffness_channels = []
            prepared_self_collision_masks = []
            prepared_topology_generations = []
            prepared_dynamic_meshes = []
            for cloth_obj, simulation_obj in zip(
                    active_cloth, simulation_objects):
                prepared_input_meshes.append(
                    _capture_modifier_input_mesh(
                        simulation_obj, depsgraph))
                binary_exclusion_mask(
                    simulation_obj, cloth_obj.GPUCloth.vgroup_objcol,
                    "object collision")
                vertex_group_weights(
                    simulation_obj, cloth_obj.GPUCloth.vgroup_shrink,
                    "shrink")
                _blender_sewing_edges(cloth_obj, simulation_obj)
                prepared_rest_shapes.append(
                    _rest_shape_key_positions(cloth_obj, simulation_obj))
                topology_generation = _cloth_topology_generation(
                    simulation_obj)
                prepared_topology_generations.append(topology_generation)
                material_coordinates = capture_material_coordinates(
                    simulation_obj, cloth_obj.GPUCloth,
                    topology_generation)
                prepared_material_features.append(
                    _capture_material_features(
                        cloth_obj.GPUCloth, material_coordinates))
                prepared_internal_configs.append(
                    _capture_internal_springs_config(cloth_obj.GPUCloth))
                prepared_pressure_features.append(
                    _capture_pressure_features(cloth_obj.GPUCloth))
                prepared_stiffness_channels.append(
                    _capture_stiffness_channels(
                        cloth_obj, simulation_obj))
                prepared_self_collision_masks.append(
                    _capture_self_collision_mask(
                        cloth_obj, simulation_obj))
                prepared_pin_snapshots.append(_capture_pin_snapshot(
                    cloth_obj, simulation_obj, depsgraph,
                    topology_generation, 1))
                prepared_dynamic_meshes.append(
                    _capture_dynamic_mesh_snapshot(
                        cloth_obj, simulation_obj, depsgraph,
                        topology_generation, 1))
                prepared_collision_configs.append(
                    _capture_cloth_collision_config(cloth_obj.GPUCloth))
            prepared_pin_owners = [
                prepare_pin_snapshot(CType, snapshot)
                for snapshot in prepared_pin_snapshots]
            prepared_dynamic_owners = [
                _prepare_dynamic_mesh_state(snapshot, None)
                for snapshot in prepared_dynamic_meshes]
            prepared_collections = _prepare_collection_snapshots(
                context, depsgraph, active_cloth, 1, {})
        except (RuntimeError, VertexChannelError) as exc:
            self.report({'ERROR'}, f"Blender input preflight failed: {exc}")
            return {'CANCELLED'}
        finally:
            _restore_modifier_visibility(modifier_states)
            try:
                context.view_layer.update()
            except (AttributeError, ReferenceError, RuntimeError):
                pass

        # 5. Native mutation starts only after complete preflight.
        self._native_prepare_mutated = True
        if (_runtime_owners_retained() or
                context.scene.gpu_cloth_springs_built):
            if not free_gpu_memory(context):
                self.report({'ERROR'}, "Не удалось освободить GPU память")
                return {'CANCELLED'}

        g_clothOBJs.extend(active_cloth)
        g_simulationOBJs.extend(simulation_objects)
        try:
            from . import cloth_settings_bridge
            for cloth_obj in g_clothOBJs:
                cloth_settings_bridge.apply_modifier_ownership(
                    cloth_obj, "GPU")
        except (
                AttributeError, ReferenceError, RuntimeError,
                TypeError) as exc:
            self.report(
                {'ERROR'}, f"CPU Cloth ownership failed: {exc}")
            free_gpu_memory(context)
            return {'CANCELLED'}

        # 6. Заполняем Mesh + ClothModifierData + Object для каждой ткани
        for index, (cloth_obj, simulation_obj) in enumerate(zip(
                g_clothOBJs, g_simulationOBJs)):
            if (cloth_obj is None or simulation_obj is None
                    or not hasattr(simulation_obj, 'data')):
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

            data_ptr = self.setMesh(
                context, prepared_input_meshes[index])
            if not data_ptr:
                self.report({'ERROR'}, "NULL от setMesh")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}
            g_mesh.append(data_ptr)

            data_ptr = self.setClothModifierData(context, cloth_obj)
            if not data_ptr:
                self.report({'ERROR'}, "NULL от setClothModifierData")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}
            g_clmd.append(data_ptr)

            data_ptr = self.fill_Object(simulation_obj)
            if not data_ptr:
                self.report({'ERROR'}, "NULL от fill_Object (cloth)")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}
            g_obj.append(data_ptr)

        if not (len(g_clothOBJs) == len(g_simulationOBJs) == len(g_clmd)):
            self.report({'ERROR'},
                f"Несоответствие объектов: clothOBJs={len(g_clothOBJs)}, "
                f"simulationOBJs={len(g_simulationOBJs)}, clmd={len(g_clmd)}")
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        # 7. Загружаем сцену на GPU.
        self.fill_Scene(context)

        try:
            requested_cache_playback = bool(
                context.scene.gpu_cloth_helper.playback_mode)
            _configure_cache_features(g_dll, context.scene)
            _query_cache_status()
            _cache_status_update(
                CType.GPUCLOTH_CACHE_STATUS_SOURCE_CHANGED,
                context.scene)
            _sync_cache_status(context.scene)
            if (requested_cache_playback and
                    context.scene.gpu_cloth_helper.is_baked):
                context.scene.gpu_cloth_helper.playback_mode = True
        except (OSError, RuntimeError) as exc:
            self.report({'ERROR'}, f"Cache config failed: {exc}")
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        if not g_dll.FillSolverData(g_scene):
            self.report({'ERROR'}, "FillSolverData вернул ошибку")
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        for i in range(len(g_clothOBJs)):
            try:
                _configure_simulation_features(
                    g_dll, g_clmd[i], context.scene,
                    g_clothOBJs[i].GPUCloth)
                _publish_material_features(
                    g_dll, g_clmd[i], prepared_material_features[i])
                _publish_internal_springs_config(
                    g_dll, g_clmd[i], prepared_internal_configs[i])
                _publish_pressure_features(
                    g_dll, g_clmd[i], prepared_pressure_features[i])
                _publish_cloth_collision_config(
                    g_dll, g_clmd[i], prepared_collision_configs[i])
                _configure_solver_diagnostics(g_dll, g_clmd[i])
            except (OSError, RuntimeError) as exc:
                self.report({'ERROR'}, f"Simulation config failed: {exc}")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}
            try:
                if not g_clmd[i].contents.clothObject:
                    if not g_dll.BuildClothSprings(g_clmd[i], g_mesh[i]):
                        self.report({'ERROR'}, "BuildClothSprings вернул ошибку")
                        free_gpu_memory(context)
                        bpy.ops.object.mode_set(mode=mode)
                        return {'CANCELLED'}
            except OSError as exc:
                self.report(
                    {'ERROR'}, f"OSError в BuildClothSprings: {exc}")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

            try:
                if g_clmd[i].contents.clothObject is None:
                    self.report({'ERROR'}, "clothObject is NULL после BuildClothSprings")
                    bpy.ops.screen.animation_cancel()
                    free_gpu_memory(context)
                    bpy.ops.object.mode_set(mode=mode)
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError: clothObject is NULL")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

            try:
                depsgraph = context.evaluated_depsgraph_get()
                _publish_material_features(
                    g_dll, g_clmd[i], prepared_material_features[i],
                    publish_anisotropy=True)
                _upload_stiffness_channels(
                    g_dll, g_clmd[i], prepared_stiffness_channels[i])
                _upload_rest_shape_key(
                    g_dll, g_clmd[i], prepared_rest_shapes[i])
                _upload_sewing(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i])
                _upload_pressure_weights(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i])
                _upload_shrink_weights(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i])
                _upload_object_collision_mask(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i])
                _upload_self_collision_mask(
                    g_dll, g_clmd[i],
                    prepared_self_collision_masks[i])
            except (OSError, RuntimeError, VertexChannelError) as exc:
                self.report({'ERROR'}, f"Cloth data upload failed: {exc}")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

            try:
                if not g_dll.AddCloth(g_clmd[i], g_mesh[i], g_obj[i]):
                    self.report({'ERROR'}, "AddCloth вернул ошибку")
                    free_gpu_memory(context)
                    bpy.ops.object.mode_set(mode=mode)
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError в AddCloth")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

        try:
            for clmd, owners in zip(g_clmd, prepared_collections):
                _configure_effector_weights(
                    g_dll, clmd, owners["effector_weights"])
            _collection_snapshots.extend(_commit_frame_inputs(
                g_dll, g_clmd, prepared_collections,
                prepared_pin_owners, prepared_dynamic_owners, 1))
            for index, clmd in enumerate(g_clmd):
                _validate_native_preparation(
                    g_dll, clmd,
                    prepared_topology_generations[index], 1)
            _effector_weight_states.extend({
                "collection_id": int(
                    owners["effector_weights"]["collection_id"]),
                "weights": tuple(
                    owners["effector_weights"]["weights"]),
            } for owners in prepared_collections)
            _pin_snapshot_states.extend({
                "topology_generation": int(snapshot.topology_generation),
                "frame_generation": int(snapshot.frame_generation),
                "snapshot": snapshot,
                # The evaluated-target owner is established at prepare and
                # remains valid across a later group-to-empty unpin frame.
                "allows_evaluated_targets": bool(snapshot.group_present),
            } for snapshot in prepared_pin_snapshots)
            _dynamic_mesh_states.extend({
                "enabled": bool(
                    g_clothOBJs[index].GPUCloth.use_dynamic_mesh),
                "topology_generation": int(
                    prepared_topology_generations[index]),
                "accepted": None,
                "pending": snapshot,
            } for index, snapshot in enumerate(prepared_dynamic_meshes))
        except (OSError, RuntimeError) as exc:
            self.report({'ERROR'}, f"Collection transaction failed: {exc}")
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        # 8. Create one upsample owner per render/simulation binding.
        g_proxy_handles.clear()
        for cloth_obj in g_clothOBJs:
            s = cloth_obj.GPUCloth
            if s.use_proxy and s.proxy_object is not None:
                nProxy = len(s.proxy_object.data.vertices)
                nHi    = len(cloth_obj.data.vertices)

                proxy_rest = (c_float * (nProxy * 3))(
                    *[c for v in s.proxy_object.data.vertices for c in v.co])
                hi_rest = (c_float * (nHi * 3))(
                    *[c for v in cloth_obj.data.vertices for c in v.co])

                handle = g_dll.ProxySim_create(
                    s.hi_nx,   s.hi_ny,
                    s.proxy_nx, s.proxy_ny,
                    s.num_sheets, s.proxy_scene_type,
                    proxy_rest, hi_rest,
                    nProxy, nHi,
                )
                g_proxy_handles.append(handle)
                if handle is None:
                    self.report({'ERROR'},
                        f"ProxySim_create вернул NULL для {cloth_obj.name}")
                    free_gpu_memory(context)
                    bpy.ops.object.mode_set(mode=mode)
                    return {'CANCELLED'}
            else:
                g_proxy_handles.append(None)

        # 9. Возвращаем режим редактирования
        bpy.ops.object.mode_set(mode=mode)
        context.scene.gpu_cloth_springs_built = True
        _store_initial_positions()
        helper = context.scene.gpu_cloth_helper
        _bake_range['start'] = int(helper.bake_start)
        _bake_range['end'] = int(helper.bake_end)
        _simulation_frame_state['last_solved'] = max(
            1, int(helper.bake_start) - 1)
        try:
            _cache_status_update(
                CType.GPUCLOTH_CACHE_STATUS_SOURCE_CHANGED,
                context.scene)
            _sync_cache_status(context.scene)
        except (OSError, RuntimeError) as exc:
            self.report({'ERROR'}, f"Cache status setup failed: {exc}")
            free_gpu_memory(context)
            return {'CANCELLED'}
        return {'FINISHED'}


# ===========================================================================
#  Оператор: обновление симуляции (один кадр)
# ===========================================================================

class GPUCloth_UpdateSimulation(bpy.types.Operator):
    """Просчитать один кадр симуляции и обновить меш в Blender"""
    bl_idname = "gpucloth.update_simulation"
    bl_label  = "Update GPUCloth Simulation"

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def _get_positions(self, clmd_ptr, nVerts):
        """
        Читает позиции вершин из GPU через SIM_get_cloth_verts.
        Возвращает плоский ctypes массив float[nVerts*3] (x0,y0,z0,x1,y1,z1,...)
        или None при ошибке.
        """
        buf = (CType.ClothVertex * nVerts)()
        g_dll.SIM_get_cloth_verts(clmd_ptr, buf, c_size_t(nVerts))
        pos = (c_float * (nVerts * 3))()
        for i in range(nVerts):
            pos[i * 3]     = buf[i].x[0]
            pos[i * 3 + 1] = buf[i].x[1]
            pos[i * 3 + 2] = buf[i].x[2]
        return pos

    def _apply_positions(self, blender_obj, pos, nVerts):
        """
        Применяет плоский float3 массив к мешу Blender через foreach_set.
        foreach_set — единственный корректный и быстрый способ в Blender 4.x.
        Прямое присваивание vertices[i].co = ... работает медленно и
        требует mesh.update() для отображения.
        """
        flat = np.frombuffer(pos, dtype=np.float32)
        blender_obj.data.vertices.foreach_set("co", flat)
        blender_obj.data.update()

    def validate_objects(self):
        for obj in [*g_clothOBJs, *g_simulationOBJs]:
            if obj is None or obj.name not in bpy.data.objects:
                self.report({'ERROR'}, f"Объект {obj} не существует в сцене.")
                return False
            if obj.type != 'MESH':
                self.report({'ERROR'}, f"Объект {obj.name} не является MESH.")
                return False
        return True

    # ── Execute ──────────────────────────────────────────────────────────────

    def execute(self, context):
        global g_dll, g_obj, g_mesh, g_clmd
        global g_clothOBJs, g_simulationOBJs

        if _teardown_failure:
            self.report(
                {'ERROR'},
                "Native teardown recovery is required before update")
            return {'CANCELLED'}

        if g_dll is None or context.scene.frame_current < 2:
            return {'FINISHED'}
        if not self.validate_objects():
            return {'FINISHED'}

        scene_s   = context.scene.gpu_cloth_helper
        cache_dir = _active_cache_path(context.scene).encode('utf-8')
        frame     = context.scene.frame_current

        # ── РЕЖИМ ВОСПРОИЗВЕДЕНИЯ из кэша ───────────────────────────────────
        #
        #   Phase 2 (GPU-destination, zero-copy):
        #     Cache_load_frame_gpu: NVMe → D3D12 VRAM → CUDA external memory
        #     → scatter_gpu_kernel → ClothVertex.x
        #     Cache_get_frame_positions: D2H → h_flat → foreach_set
        #
        if scene_s.playback_mode and scene_s.is_baked:
            for i, cloth_obj in enumerate(g_clothOBJs):
                nV  = len(cloth_obj.data.vertices)
                pos = (c_float * (nV * 3))()
                # GPU-direct load для данного кадра
                if g_dll.Cache_load_frame_gpu(
                        frame, g_clmd[i], c_size_t(nV), cache_dir):
                    # D2H для foreach_set (viewport)
                    if g_dll.Cache_get_frame_positions(frame, pos, c_size_t(nV)):
                        self._apply_positions(cloth_obj, pos, nV)
            # Prefetch следующего кадра пока пользователь смотрит текущий
            for cloth_obj in g_clothOBJs:
                g_dll.Cache_prefetch_frame(
                    frame + 1,
                    c_size_t(len(cloth_obj.data.vertices)),
                    cache_dir,
                )
            return {'FINISHED'}

        last_solved = _simulation_frame_state['last_solved']
        if last_solved is not None and frame <= last_solved:
            _load_cached_frame(
                context.scene, context.evaluated_depsgraph_get(), frame)
            return {'FINISHED'}

        # ── РЕЖИМ ЖИВОЙ СИМУЛЯЦИИ ────────────────────────────────────────────
        #
        #   SIM_solver_cloth(clmd) [GPU ~N ms]
        #   SIM_get_cloth_verts() [memcpy D2H ~<1мс]
        #   foreach_set() [Blender ~<1мс]
        #   Cache_write_frame_async() ← возвращает немедленно, пишет в фоне
        #

        if not (len(g_clothOBJs) == len(g_simulationOBJs)
                == len(g_clmd) == len(g_obj) == len(g_mesh)):
            self.report({'ERROR'},
                f"Несоответствие размеров: clothOBJs={len(g_clothOBJs)}, "
                f"simulationOBJs={len(g_simulationOBJs)}, "
                f"clmd={len(g_clmd)}, obj={len(g_obj)}, mesh={len(g_mesh)}")
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}

        total_t0 = time.perf_counter_ns()

        try:
            depsgraph = context.evaluated_depsgraph_get()
            _publish_frame_inputs(context, depsgraph)
            for i in range(len(g_clothOBJs)):
                cloth_obj = g_clothOBJs[i]
                simulation_obj = g_simulationOBJs[i]
                n_sim = len(simulation_obj.data.vertices)

                # 1. GPU симуляция одного кадра
                if not g_dll.SIM_solver_cloth(g_clmd[i]):
                    _solver_diagnostic_snapshot(i)
                    self.report(
                        {'ERROR'},
                        f"SIM_solver_cloth rejected {cloth_obj.name_full}")
                    bpy.ops.screen.animation_cancel()
                    return {'CANCELLED'}
                _accept_dynamic_mesh_snapshot(i)
                diagnostic = _solver_diagnostic_snapshot(i)
                if diagnostic is not None:
                    native_ms = diagnostic["status"]["execution_time_ms"]
                    print(
                        f"[GPUCloth] {cloth_obj.name_full}: "
                        f"native {native_ms:.3f} ms")

                # 2. Readback позиций (D2H через SIM_get_cloth_verts)
                simulation_pos = self._get_positions(g_clmd[i], n_sim)
                if simulation_pos is None:
                    self.report({'ERROR'}, "SIM_get_cloth_verts вернул ошибку")
                    bpy.ops.screen.animation_cancel()
                    return {'CANCELLED'}

                # 3. Proxy upsampling (если активен)
                #    proxy_pos заполняется из результатов симуляции proxy-меша
                handle = (g_proxy_handles[i]
                          if i < len(g_proxy_handles) else None)
                if handle is not None:
                    nP = g_dll.ProxySim_proxy_count(handle)
                    if nP != n_sim:
                        self.report({'ERROR'},
                            f"ProxySim count changed: handle={nP}, mesh={n_sim}")
                        bpy.ops.screen.animation_cancel()
                        return {'CANCELLED'}
                    self._apply_positions(simulation_obj, simulation_pos, nP)
                    nV = g_dll.ProxySim_hi_count(handle)
                    out_hi = (c_float * (nV * 3))()
                    g_dll.ProxySim_apply(handle, simulation_pos, out_hi)
                    pos = out_hi
                else:
                    pos = simulation_pos
                    nV = n_sim

                # 4. Обновляем меш в Blender (foreach_set, Blender 4.x safe)
                self._apply_positions(cloth_obj, pos, nV)

                # 5. Асинхронная запись кэша
                #    Phase 1 (CPU-destination): C++ пишет в фоне
                #    pinned RAM → DMA → NVMe
                #    _live_arrays защищает pos от GC пока C++ работает
                if not scene_s.is_baked:
                    _live_arrays.append(pos)
                    if not g_dll.Cache_write_frame_async(
                            frame, pos, c_size_t(nV), cache_dir):
                        self.report(
                            {'ERROR'},
                            f"Cache write rejected at frame {frame}")
                        return {'CANCELLED'}

        except (OSError, RuntimeError, VertexChannelError) as err:
            print(f"GPUCloth_UpdateSimulation failed: {err}")
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}

        _simulation_frame_state['last_solved'] = frame
        elapsed_ms = (time.perf_counter_ns() - total_t0) / 1_000_000
        print(f"[GPUCloth] кадр {frame}: {elapsed_ms:.2f} мс")
        return {'FINISHED'}


# ===========================================================================
#  Оператор: запекание (bake) симуляции
# ===========================================================================

def _capture_drape_triangle_layers(simulation_obj):
    mesh = simulation_obj.data
    attribute = mesh.attributes.get("gpucloth_drape_layer")
    if attribute is None:
        return None
    if attribute.domain != 'FACE' or attribute.data_type != 'INT':
        raise RuntimeError(
            "gpucloth_drape_layer must be a FACE/INT mesh attribute")
    triangle_count = len(mesh.polygons)
    if len(attribute.data) != triangle_count:
        raise RuntimeError("gpucloth_drape_layer face count mismatch")
    values = np.empty(triangle_count, dtype=np.int32)
    attribute.data.foreach_get("value", values)
    if np.any(values < 0):
        raise RuntimeError("gpucloth_drape_layer cannot contain negatives")
    layer_type = c_uint * triangle_count
    layers = layer_type(*(int(value) for value in values))
    digest = hashlib.blake2b(
        values.tobytes(), digest_size=8, person=b"GPUDrape").digest()
    generation = int.from_bytes(digest, "little") or 1
    return layers, generation


def _readback_drape_preview(index):
    simulation_obj = g_simulationOBJs[index]
    cloth_obj = g_clothOBJs[index]
    n_sim = len(simulation_obj.data.vertices)
    vertices = (CType.ClothVertex * n_sim)()
    g_dll.SIM_get_cloth_verts(
        g_clmd[index], vertices, c_size_t(n_sim))
    positions = (c_float * (n_sim * 3))()
    for vertex in range(n_sim):
        positions[vertex * 3 + 0] = vertices[vertex].x[0]
        positions[vertex * 3 + 1] = vertices[vertex].x[1]
        positions[vertex * 3 + 2] = vertices[vertex].x[2]
    flat = np.frombuffer(positions, dtype=np.float32)
    simulation_obj.data.vertices.foreach_set("co", flat)
    simulation_obj.data.update()

    handle = g_proxy_handles[index] if index < len(g_proxy_handles) else None
    if handle is None:
        if cloth_obj is not simulation_obj:
            cloth_obj.data.vertices.foreach_set("co", flat)
            cloth_obj.data.update()
        return
    if int(g_dll.ProxySim_proxy_count(handle)) != n_sim:
        raise RuntimeError("Drape proxy vertex count changed")
    n_hi = int(g_dll.ProxySim_hi_count(handle))
    high_positions = (c_float * (n_hi * 3))()
    g_dll.ProxySim_apply(handle, positions, high_positions)
    cloth_obj.data.vertices.foreach_set(
        "co", np.frombuffer(high_positions, dtype=np.float32))
    cloth_obj.data.update()


def _selected_drape_owner(context):
    index = _prepared_cloth_index(context.object)
    if g_dll is None or index < 0 or index >= len(g_clmd):
        raise RuntimeError("selected cloth is not prepared")
    return index, context.object, g_clmd[index]


class GPUCloth_BeginDrape(bpy.types.Operator):
    bl_idname = "gpucloth.begin_drape"
    bl_label = "Begin Drape"
    bl_description = "Snapshot state and begin preparation-only draping"

    @classmethod
    def poll(cls, context):
        return bool(
            context.object is not None and g_dll is not None and
            _prepared_cloth_index(context.object) >= 0)

    def execute(self, context):
        try:
            index, cloth_obj, clmd = _selected_drape_owner(context)
            _publish_frame_inputs(
                context, context.evaluated_depsgraph_get())
            config = CType.GPUClothDrapeConfig()
            config.struct_size = sizeof(config)
            config.config_version = 1
            config.max_steps = 240
            config.convergence_window = 8
            config.convergence_tolerance = float(
                clmd.contents.sim_parms.contents.solver_convergence_tol)
            layer_owner = _capture_drape_triangle_layers(
                g_simulationOBJs[index])
            if layer_owner is not None:
                layers, generation = layer_owner
                config.drape_flags = CType.GPUCLOTH_DRAPE_USE_TRIANGLE_LAYERS
                _set_buffer_view(
                    config.triangle_layers, CType.GPUCLOTH_ELEMENT_UINT32,
                    len(layers), sizeof(c_uint), addressof(layers), generation)
            status = _new_drape_status()
            result = int(g_dll.SIM_begin_cloth_drape(
                clmd, pointer(config), pointer(status)))
            _remember_drape_status(cloth_obj, status)
            if result != CType.GPUCLOTH_ABI_OK:
                witness = _invariant_witness_data(clmd)
                self.report(
                    {'ERROR'},
                    f"Drape Begin rejected: {witness['invariant_name']}")
                return {'CANCELLED'}
            self.report({'INFO'}, "Drape sandbox started")
            return {'FINISHED'}
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, f"Drape Begin failed: {exc}")
            return {'CANCELLED'}


class GPUCloth_StepDrape(bpy.types.Operator):
    bl_idname = "gpucloth.step_drape"
    bl_label = "Step Drape"
    bl_description = "Advance Drape without moving timeline or writing cache"

    until_settled: bpy.props.BoolProperty(
        name="Until settled", default=False)

    @classmethod
    def poll(cls, context):
        status = (get_drape_ui_status(context.object)
                  if context.object is not None else None)
        if not status:
            return False
        flags = status["status_flags"]
        return bool(
            flags & CType.GPUCLOTH_DRAPE_STATUS_ACTIVE and
            not flags & (CType.GPUCLOTH_DRAPE_STATUS_CONVERGED |
                         CType.GPUCLOTH_DRAPE_STATUS_FAILED))

    def execute(self, context):
        try:
            index, cloth_obj, clmd = _selected_drape_owner(context)
            status = _new_drape_status()
            limit = 240 if self.until_settled else 1
            for _ in range(limit):
                result = int(g_dll.SIM_step_cloth_drape(
                    clmd, pointer(status)))
                state = _remember_drape_status(cloth_obj, status)
                if result != CType.GPUCLOTH_ABI_OK:
                    witness = _invariant_witness_data(clmd)
                    self.report(
                        {'ERROR'},
                        f"Drape frame rejected: {witness['invariant_name']}")
                    return {'CANCELLED'}
                if (state["status_flags"] &
                        (CType.GPUCLOTH_DRAPE_STATUS_CONVERGED |
                         CType.GPUCLOTH_DRAPE_STATUS_FAILED)):
                    break
            _readback_drape_preview(index)
            state = get_drape_ui_status(cloth_obj)
            if state["status_flags"] & CType.GPUCLOTH_DRAPE_STATUS_FAILED:
                self.report({'ERROR'}, "Drape did not converge in 240 steps")
                return {'CANCELLED'}
            message = ("Drape converged" if state["status_flags"] &
                       CType.GPUCLOTH_DRAPE_STATUS_CONVERGED else
                       f"Drape step {state['step_count']}")
            self.report({'INFO'}, message)
            return {'FINISHED'}
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, f"Drape Step failed: {exc}")
            return {'CANCELLED'}


class GPUCloth_ApplyDrape(bpy.types.Operator):
    bl_idname = "gpucloth.apply_drape"
    bl_label = "Apply Drape"
    bl_description = "Publish converged Drape as frame-0 accepted state"

    @classmethod
    def poll(cls, context):
        status = (get_drape_ui_status(context.object)
                  if context.object is not None else None)
        return bool(status and (
            status["status_flags"] & CType.GPUCLOTH_DRAPE_STATUS_CONVERGED))

    def execute(self, context):
        try:
            index, cloth_obj, clmd = _selected_drape_owner(context)
            status = _new_drape_status()
            result = int(g_dll.SIM_apply_cloth_drape(
                clmd, pointer(status)))
            _remember_drape_status(cloth_obj, status)
            if result != CType.GPUCLOTH_ABI_OK:
                raise RuntimeError(f"native Apply rejected with {result}")
            _readback_drape_preview(index)
            _store_initial_positions()
            _simulation_frame_state['last_solved'] = max(
                1, int(context.scene.gpu_cloth_helper.bake_start) - 1)
            _cache_status_update(
                CType.GPUCLOTH_CACHE_STATUS_SOURCE_CHANGED, context.scene)
            _sync_cache_status(context.scene)
            self.report({'INFO'}, "Drape applied as frame-0 state")
            return {'FINISHED'}
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, f"Drape Apply failed: {exc}")
            return {'CANCELLED'}


class GPUCloth_CancelDrape(bpy.types.Operator):
    bl_idname = "gpucloth.cancel_drape"
    bl_label = "Cancel Drape"
    bl_description = "Restore the exact Begin snapshot"

    @classmethod
    def poll(cls, context):
        status = (get_drape_ui_status(context.object)
                  if context.object is not None else None)
        return bool(status and (
            status["status_flags"] &
            (CType.GPUCLOTH_DRAPE_STATUS_ACTIVE |
             CType.GPUCLOTH_DRAPE_STATUS_FAILED)))

    def execute(self, context):
        try:
            index, cloth_obj, clmd = _selected_drape_owner(context)
            status = _new_drape_status()
            result = int(g_dll.SIM_cancel_cloth_drape(
                clmd, pointer(status)))
            _remember_drape_status(cloth_obj, status)
            if result != CType.GPUCLOTH_ABI_OK:
                raise RuntimeError(f"native Cancel rejected with {result}")
            _readback_drape_preview(index)
            self.report({'INFO'}, "Drape snapshot restored")
            return {'FINISHED'}
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, f"Drape Cancel failed: {exc}")
            return {'CANCELLED'}


def _selected_invariant_json(context):
    _, cloth_obj, clmd = _selected_drape_owner(context)
    payload = {
        "schema": "GPUClothInvariantWitness/1",
        "cloth": cloth_obj.name_full,
        "witness": _invariant_witness_data(clmd),
        "drape": get_drape_ui_status(cloth_obj),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


class GPUCloth_CopyInvariantDiagnostics(bpy.types.Operator):
    bl_idname = "gpucloth.copy_invariant_diagnostics"
    bl_label = "Copy Diagnostics JSON"

    def execute(self, context):
        try:
            context.window_manager.clipboard = _selected_invariant_json(context)
            self.report({'INFO'}, "Invariant diagnostics copied")
            return {'FINISHED'}
        except (OSError, RuntimeError) as exc:
            self.report({'ERROR'}, f"Diagnostics copy failed: {exc}")
            return {'CANCELLED'}


class GPUCloth_SaveInvariantDiagnostics(bpy.types.Operator):
    bl_idname = "gpucloth.save_invariant_diagnostics"
    bl_label = "Save Diagnostics JSON"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "//gpucloth_invariant.json"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        try:
            with open(
                    bpy.path.abspath(self.filepath), "w",
                    encoding="utf-8", newline="\n") as output:
                output.write(_selected_invariant_json(context))
                output.write("\n")
            self.report({'INFO'}, "Invariant diagnostics saved")
            return {'FINISHED'}
        except (OSError, RuntimeError) as exc:
            self.report({'ERROR'}, f"Diagnostics save failed: {exc}")
            return {'CANCELLED'}


class GPUCloth_BakeSimulation(bpy.types.Operator):
    """
    Просчитать симуляцию для всего диапазона кадров и записать кэш на диск.
    Паттерн: FLIP Fluids BakeFluidSimulation (modal с таймером).

    На каждом кадре:
      SIM_solver_cloth() → SIM_get_cloth_verts() → foreach_set() → cache
    """
    bl_idname = "gpucloth.bake_simulation"
    bl_label  = "Запечь симуляцию GPU Cloth"
    _timer    = None

    @classmethod
    def poll(cls, context):
        return (
            g_dll is not None
            and context.scene.gpu_cloth_springs_built
            and not context.scene.gpu_cloth_helper.is_baked
            and not context.scene.gpu_cloth_helper.use_external_cache
        )

    def modal(self, context, event):
        s = context.scene.gpu_cloth_helper

        if event.type == 'TIMER':
            if self._frame > s.bake_end:
                return (
                    {'FINISHED'} if self._finish(context, success=True)
                    else {'CANCELLED'})

            # Просчёт кадра (UpdateSimulation пишет кэш async внутри)
            if not self._step_frame(context, self._frame):
                self._finish(context, success=False)
                self.report(
                    {'ERROR'},
                    f"Cache write failed at frame {self._frame}")
                return {'CANCELLED'}

            # Прогресс
            total = max(s.bake_end - s.bake_start + 1, 1)
            s.bake_progress = int(
                100 * (self._frame - s.bake_start + 1) / total)
            self._frame += 1

            if context.screen:
                for area in context.screen.areas:
                    area.tag_redraw()

        if event.type == 'ESC':
            self._finish(context, success=False)
            self.report({'INFO'}, "Запекание отменено (ESC)")
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _begin(self, context):
        s = context.scene.gpu_cloth_helper
        if s.bake_end < s.bake_start:
            self.report({'ERROR'}, "Bake end precedes bake start")
            return False
        try:
            _configure_cache_features(g_dll, context.scene)
        except (OSError, RuntimeError) as exc:
            self.report({'ERROR'}, f"Cache config failed: {exc}")
            return False
        cache_dir = _active_cache_path(context.scene).encode('utf-8')
        if not g_dll.Cache_clear_all(cache_dir):
            self.report({'ERROR'}, "Cannot initialize cache transaction")
            return False
        try:
            _cache_status_update(
                CType.GPUCLOTH_CACHE_STATUS_BAKE_BEGIN,
                context.scene, frame=int(s.bake_start))
            _sync_cache_status(context.scene)
        except (OSError, RuntimeError) as exc:
            self.report({'ERROR'}, f"Cache status begin failed: {exc}")
            return False
        s.is_baked = False
        s.playback_mode = False
        s.bake_progress = 0
        _bake_range['start'] = s.bake_start
        _bake_range['end'] = s.bake_end
        _simulation_frame_state['last_solved'] = max(
            1, int(s.bake_start) - 1)
        _live_arrays.clear()
        self._frame = s.bake_start
        return True

    def _step_frame(self, context, frame):
        cache_dir = _active_cache_path(context.scene).encode('utf-8')
        _cache_playback_guard['active'] = True
        try:
            context.scene.frame_set(frame)
        finally:
            _cache_playback_guard['active'] = False
        result = bpy.ops.gpucloth.update_simulation()
        return (
            'FINISHED' in result and
            bool(g_dll.Cache_has_frame(frame, cache_dir))
        )

    def _finish(self, context, success: bool):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        s = context.scene.gpu_cloth_helper
        if success:
            try:
                _cache_status_update(
                    CType.GPUCLOTH_CACHE_STATUS_BAKE_COMPLETE,
                    context.scene, frame=int(s.bake_end))
            except (OSError, RuntimeError) as exc:
                self.report({'ERROR'}, f"Cache completion failed: {exc}")
                success = False
        s.bake_progress = 100 if success else 0
        if success:
            s.playback_mode = True
            _bake_range['start'] = s.bake_start
            _bake_range['end']   = s.bake_end
        else:
            s.playback_mode = False
            cache_dir = _active_cache_path(context.scene).encode('utf-8')
            g_dll.Cache_clear_all(cache_dir)
            try:
                _cache_status_update(
                    CType.GPUCLOTH_CACHE_STATUS_BAKE_CANCEL,
                    context.scene, error_code=1, frame=int(self._frame))
            except (OSError, RuntimeError):
                pass
        try:
            _sync_cache_status(context.scene)
        except (OSError, RuntimeError):
            s.is_baked = False
        _live_arrays.clear()
        if success:
            self.report({'INFO'}, "Запекание завершено.")
        return success

    def execute(self, context):
        if not self._begin(context):
            return {'CANCELLED'}
        s = context.scene.gpu_cloth_helper
        total = s.bake_end - s.bake_start + 1
        for completed, frame in enumerate(
                range(s.bake_start, s.bake_end + 1), start=1):
            if not self._step_frame(context, frame):
                self._finish(context, success=False)
                self.report({'ERROR'}, f"Cache write failed at frame {frame}")
                return {'CANCELLED'}
            s.bake_progress = int(100 * completed / total)
        return (
            {'FINISHED'} if self._finish(context, success=True)
            else {'CANCELLED'})

    def invoke(self, context, event):
        if not self._begin(context):
            return {'CANCELLED'}
        self._timer = context.window_manager.event_timer_add(
            0.001, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


# ===========================================================================
#  Оператор: очистка кэша
# ===========================================================================

class GPUCloth_FreeCache(bpy.types.Operator):
    """Удалить все файлы кэша симуляции с диска"""
    bl_idname = "gpucloth.free_cache"
    bl_label  = "Очистить кэш"

    @classmethod
    def poll(cls, context):
        return (
            g_dll is not None
            and not context.scene.gpu_cloth_helper.use_external_cache
            and (
                context.scene.gpu_cloth_helper.is_baked
                or context.scene.gpu_cloth_helper.is_outdated
                or context.scene.gpu_cloth_helper.is_frame_skip
                or context.scene.gpu_cloth_helper.cached_frame_count > 0
            )
        )

    def execute(self, context):
        s         = context.scene.gpu_cloth_helper
        cache_dir = _active_cache_path(context.scene).encode('utf-8')

        if not g_dll.Cache_clear_all(cache_dir):
            self.report({'ERROR'}, "Не удалось очистить кэш")
            return {'CANCELLED'}

        s.bake_progress = 0
        s.playback_mode = False
        try:
            _sync_cache_status(context.scene)
        except (OSError, RuntimeError) as exc:
            self.report({'ERROR'}, f"Cache status refresh failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, "Кэш очищен.")
        return {'FINISHED'}


# ===========================================================================
#  Хелпер: frame_change_post обработчик для экспорта из кэша
# ===========================================================================
#
#   При экспорте Alembic/USD Blender внутренне вызывает scene.frame_set()
#   для каждого кадра.  frame_change_post обработчик загружает позиции
#   из кэша и записывает их в меш, после чего вызывает depsgraph.update()
#   чтобы экспортёр увидел актуальную геометрию.

def _make_cache_handler(cache_dir_bytes):
    """Создаёт frame_change_post обработчик для загрузки кэша при экспорте."""
    _guard = {'active': False}

    def _handler(scene, depsgraph):
        if _guard['active']:
            return
        _guard['active'] = True
        try:
            frame = scene.frame_current
            updated = False
            for i, cloth_obj in enumerate(g_clothOBJs):
                if i >= len(g_clmd):
                    break
                nV  = len(cloth_obj.data.vertices)
                pos = (c_float * (nV * 3))()
                if g_dll.Cache_load_frame_gpu(
                        frame, g_clmd[i], c_size_t(nV), cache_dir_bytes):
                    if g_dll.Cache_get_frame_positions(frame, pos, c_size_t(nV)):
                        flat = np.frombuffer(pos, dtype=np.float32)
                        cloth_obj.data.vertices.foreach_set("co", flat)
                        cloth_obj.data.update()
                        updated = True
            if updated:
                for cloth_obj in g_clothOBJs:
                    cloth_obj.data.update_tag()
                depsgraph.update()
        finally:
            _guard['active'] = False

    return _handler


# ===========================================================================
#  Оператор: экспорт Alembic (.abc)
# ===========================================================================

class GPUCloth_ExportAlembic(bpy.types.Operator):
    """Экспорт запечённой симуляции в Alembic (.abc)"""
    bl_idname  = "gpucloth.export_alembic"
    bl_label   = "Экспорт Alembic"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(
        name="Путь",
        subtype='FILE_PATH',
    )
    filename_ext = ".abc"
    filter_glob: bpy.props.StringProperty(
        default="*.abc",
        options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return (
            g_dll is not None
            and context.scene.gpu_cloth_springs_built
            and context.scene.gpu_cloth_helper.is_baked
            and len(g_clothOBJs) > 0
        )

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "//gpucloth_export.abc"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene_s   = context.scene.gpu_cloth_helper
        cache_dir = _active_cache_path(context.scene).encode('utf-8')

        # Выделяем только объекты ткани для экспорта
        prev_selection = [o for o in context.scene.objects if o.select_get()]
        bpy.ops.object.select_all(action='DESELECT')
        for obj in g_clothOBJs:
            obj.select_set(True)

        handler = _make_cache_handler(cache_dir)
        bpy.app.handlers.frame_change_post.append(handler)
        try:
            filepath = bpy.path.abspath(self.filepath)
            if not filepath.lower().endswith('.abc'):
                filepath += '.abc'
            bpy.ops.wm.alembic_export(
                'EXEC_DEFAULT',
                filepath=filepath,
                start=scene_s.bake_start,
                end=scene_s.bake_end,
                selected=True,
                visible_objects_only=False,
                export_hair=False,
                export_particles=False,
                as_background_job=False,
            )
            self.report({'INFO'}, f"Alembic экспортирован: {filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка экспорта Alembic: {e}")
            return {'CANCELLED'}
        finally:
            if handler in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.remove(handler)
            # Восстанавливаем выделение
            bpy.ops.object.select_all(action='DESELECT')
            for obj in prev_selection:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)

        return {'FINISHED'}


# ===========================================================================
#  Оператор: экспорт USD (.usd / .usdc / .usda)
# ===========================================================================

class GPUCloth_ExportUSD(bpy.types.Operator):
    """Экспорт запечённой симуляции в Universal Scene Description (.usd)"""
    bl_idname  = "gpucloth.export_usd"
    bl_label   = "Экспорт USD"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(
        name="Путь",
        subtype='FILE_PATH',
    )
    filename_ext = ".usdc"
    filter_glob: bpy.props.StringProperty(
        default="*.usd;*.usdc;*.usda",
        options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return (
            g_dll is not None
            and context.scene.gpu_cloth_springs_built
            and context.scene.gpu_cloth_helper.is_baked
            and len(g_clothOBJs) > 0
        )

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "//gpucloth_export.usdc"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene_s   = context.scene.gpu_cloth_helper
        cache_dir = _active_cache_path(context.scene).encode('utf-8')

        prev_selection = [o for o in context.scene.objects if o.select_get()]
        bpy.ops.object.select_all(action='DESELECT')
        for obj in g_clothOBJs:
            obj.select_set(True)

        handler = _make_cache_handler(cache_dir)
        bpy.app.handlers.frame_change_post.append(handler)
        try:
            filepath = bpy.path.abspath(self.filepath)
            valid_ext = ('.usd', '.usdc', '.usda')
            if not any(filepath.lower().endswith(ext) for ext in valid_ext):
                filepath += '.usdc'
            bpy.ops.wm.usd_export(
                'EXEC_DEFAULT',
                filepath=filepath,
                selected_objects_only=True,
                visible_objects_only=False,
                export_animation=True,
                export_hair=False,
            )
            self.report({'INFO'}, f"USD экспортирован: {filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка экспорта USD: {e}")
            return {'CANCELLED'}
        finally:
            if handler in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.remove(handler)
            bpy.ops.object.select_all(action='DESELECT')
            for obj in prev_selection:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)

        return {'FINISHED'}


# ===========================================================================
#  Test scene operators
# ===========================================================================

import bmesh


def _make_grid_mesh(name, nx, ny, half_size, height, pin_corners=False):
    """Create a subdivided grid mesh and return (obj, mesh_data)."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    sx = nx + 1
    sy = ny + 1
    verts = []
    for row in range(sy):
        for col in range(sx):
            x = -half_size + 2.0 * half_size * col / nx
            y = -half_size + 2.0 * half_size * row / ny
            verts.append((x, y, height))

    faces = []
    for row in range(ny):
        for col in range(nx):
            i = row * sx + col
            faces.append((i, i + 1, i + sx + 1, i + sx))

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    if pin_corners:
        for v in obj.data.vertices:
            pinned = False
            if (abs(v.co.x - (-half_size)) < 0.01 and abs(v.co.y - half_size) < 0.01):
                pinned = True
            if (abs(v.co.x - half_size) < 0.01 and abs(v.co.y - half_size) < 0.01):
                pinned = True
            if pinned:
                v.co.z += 0.0
        mesh_data.update()

    return obj, mesh_data


def _make_uv_sphere(name, radius, cx, cy, cz, rings=10, sectors=12):
    """Create a UV sphere collision object and return it."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    segs_loop = sectors
    segs_ring = rings
    import math
    for i in range(segs_ring + 1):
        phi = math.pi * i / segs_ring
        for j in range(segs_loop + 1):
            theta = 2.0 * math.pi * j / segs_loop
            x = cx + radius * math.sin(phi) * math.cos(theta)
            y = cy + radius * math.sin(phi) * math.sin(theta)
            z = cz + radius * math.cos(phi)
            bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()
    w = segs_loop + 1
    for i in range(segs_ring):
        for j in range(segs_loop):
            a = i * w + j
            b = i * w + j + 1
            c = (i + 1) * w + j + 1
            d = (i + 1) * w + j
            bm.faces.new([bm.verts[a], bm.verts[b], bm.verts[c], bm.verts[d]])
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()
    return obj


def _make_cylinder_floor(name, radius, half_len, cx, cy, cz, floor_z, floor_half):
    """Create a cylinder + floor collision object."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    verts = []
    rings = 20
    stacks = 10
    for s in range(stacks + 1):
        y = cy - half_len + 2.0 * half_len * s / stacks
        for r in range(rings + 1):
            theta = 2.0 * math.pi * r / rings
            verts.append((cx + radius * math.cos(theta), y, cz + radius * math.sin(theta)))
    fh = floor_half
    verts.append((-fh, -fh, floor_z))
    verts.append((fh, -fh, floor_z))
    verts.append((fh, fh, floor_z))
    verts.append((-fh, fh, floor_z))

    faces = []
    w = rings + 1
    for s in range(stacks):
        for r in range(rings):
            a = s * w + r
            b = s * w + r + 1
            c = (s + 1) * w + r + 1
            d = (s + 1) * w + r
            faces.append((a, c, b))
            faces.append((b, c, d))
    body_verts = len(verts) - 4
    f0, f1, f2, f3 = body_verts, body_verts + 1, body_verts + 2, body_verts + 3
    faces.append((f0, f1, f2))
    faces.append((f0, f2, f3))

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    return obj


def _make_cushion_mesh(name, nx, ny, half_size, init_z, sep, dome_height):
    """Create a two-sheet cushion mesh."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    sx = nx + 1
    sy = ny + 1
    verts = []
    for sh in range(2):
        for row in range(sy):
            for col in range(sx):
                x = -half_size + 2.0 * half_size * col / nx
                y = -half_size + 2.0 * half_size * row / ny
                pu = col / nx
                pv = row / ny
                dome = math.sin(math.pi * pu) * math.sin(math.pi * pv) * dome_height
                z = (init_z - sep * 0.5 - dome) if sh == 0 else (init_z + sep * 0.5 + dome)
                verts.append((x, y, z))

    faces = []
    for sh in range(2):
        base = sh * sx * sy
        for row in range(ny):
            for col in range(nx):
                i = base + row * sx + col
                faces.append((i, i + 1, i + sx + 1, i + sx))

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    return obj


def _setup_cloth(obj, solver='XPBD', material='COTTON'):
    """Enable GPUCloth on object with given solver and material preset."""
    obj.GPUCloth.is_active = True
    obj.GPUCloth.solver_type = solver
    obj.GPUCloth.material_preset = material


class GPUCloth_TestDrapeOnSphere(bpy.types.Operator):
    """Create DrapeOnSphere test scene: cloth pinned at top, draped over sphere"""
    bl_idname = "gpucloth.test_drape_on_sphere"
    bl_label = "Drape On Sphere"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        cloth_obj, _ = _make_grid_mesh("DrapeCloth", 64, 64, 3.0, 4.0)
        sphere_obj = _make_uv_sphere("CollisionSphere", 1.8, 0.0, 0.0, 0.5)

        col_mod = sphere_obj.modifiers.new(name="Collision", type='COLLISION')

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_cloth(cloth_obj, solver='XPBD', material='COTTON')

        context.scene.gpu_cloth_helper.gravity_z = -9.81

        self.report({'INFO'}, "DrapeOnSphere test scene created")
        return {'FINISHED'}


class GPUCloth_TestTwist(bpy.types.Operator):
    """Create TwistTest scene: cloth pinned at top corners"""
    bl_idname = "gpucloth.test_twist"
    bl_label = "Twist Test"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        cloth_obj, _ = _make_grid_mesh("TwistCloth", 64, 64, 3.0, 0.0)

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_cloth(cloth_obj, solver='XPBD', material='COTTON')

        context.scene.gpu_cloth_helper.gravity_x = 0.0
        context.scene.gpu_cloth_helper.gravity_y = 0.0
        context.scene.gpu_cloth_helper.gravity_z = 0.0

        self.report({'INFO'}, "TwistTest scene created")
        return {'FINISHED'}


class GPUCloth_TestMultiLayerDrop(bpy.types.Operator):
    """Create MultiLayerDrop scene: multiple cloth layers falling on cylinder"""
    bl_idname = "gpucloth.test_multi_layer_drop"
    bl_label = "Multi Layer Drop"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        cloth_obj, _ = _make_grid_mesh("MultiLayerCloth", 64, 64, 3.0, 4.0)
        collision_obj = _make_cylinder_floor(
            "CollisionCylinder",
            1.5, 4.5, 0.0, 0.0, 0.3, -2.0, 12.0)

        col_mod = collision_obj.modifiers.new(name="Collision", type='COLLISION')

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_cloth(cloth_obj, solver='OGC', material='COTTON')
        cloth_obj.GPUCloth.use_self_collision = True
        cloth_obj.GPUCloth.ogc_radius = 150.0
        cloth_obj.GPUCloth.ogc_friction = 0.3

        context.scene.gpu_cloth_helper.gravity_z = -9.81

        self.report({'INFO'}, "MultiLayerDrop test scene created")
        return {'FINISHED'}


class GPUCloth_TestCushionDrop(bpy.types.Operator):
    """Create CushionDrop scene: two-layer cushion falling on floor"""
    bl_idname = "gpucloth.test_cushion_drop"
    bl_label = "Cushion Drop"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        init_z = -2.0 + 3.0 * 0.25 + 0.02 * 0.5 + 2.0
        cushion_obj = _make_cushion_mesh(
            "CushionCloth", 32, 32, 3.0, init_z, 0.02, 3.0 * 0.25)

        floor_mesh = bpy.data.meshes.new("Floor_mesh")
        floor_obj = bpy.data.objects.new("Floor", floor_mesh)
        bpy.context.collection.objects.link(floor_obj)
        fh = 12.0
        fz = -2.0
        floor_verts = [(-fh, -fh, fz), (fh, -fh, fz), (fh, fh, fz), (-fh, fh, fz)]
        floor_faces = [(0, 1, 2, 3)]
        floor_mesh.from_pydata(floor_verts, [], floor_faces)
        floor_mesh.update()
        col_mod = floor_obj.modifiers.new(name="Collision", type='COLLISION')

        bpy.context.view_layer.objects.active = cushion_obj
        cushion_obj.select_set(True)

        _setup_cloth(cushion_obj, solver='XPBD', material='COTTON')

        context.scene.gpu_cloth_helper.gravity_z = -9.81

        self.report({'INFO'}, "CushionDrop test scene created")
        return {'FINISHED'}


class GPUCloth_TestOGCBounds(bpy.types.Operator):
    """Create OGC Bounds test scene: two cloth sheets with self-collision and contact-bounds visualisation"""
    bl_idname = "gpucloth.test_ogc_bounds"
    bl_label  = "OGC Bounds Viz"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        # Two horizontal cloth sheets stacked 60 mm apart so they collide
        upper, _ = _make_grid_mesh("OGCCloth_Upper", 32, 32, 1.5,  0.06)
        lower, _ = _make_grid_mesh("OGCCloth_Lower", 32, 32, 1.5, -0.06)

        # Activate OGC on the upper sheet, show bounds immediately
        bpy.context.view_layer.objects.active = upper
        upper.select_set(True)
        _setup_cloth(upper, solver='OGC', material='COTTON')
        upper.GPUCloth.use_self_collision = True
        upper.GPUCloth.ogc_radius         = 150.0
        upper.GPUCloth.ogc_friction       = 0.3
        upper.GPUCloth.show_ogc_bounds    = True

        # Activate OGC on the lower sheet as well
        lower.select_set(True)
        bpy.context.view_layer.objects.active = lower
        _setup_cloth(lower, solver='OGC', material='COTTON')
        lower.GPUCloth.use_self_collision = True
        lower.GPUCloth.ogc_radius         = 150.0
        lower.GPUCloth.ogc_friction       = 0.3
        lower.GPUCloth.show_ogc_bounds    = True

        context.scene.gpu_cloth_helper.gravity_z = -9.81

        bpy.context.view_layer.objects.active = upper
        self.report({'INFO'}, "OGC Bounds Viz scene created")
        return {'FINISHED'}


# ===========================================================================
#  OGC contact-bounds visualiser (SpaceView3D draw callback)
# ===========================================================================

_ogc_draw_handle = None


def _ogc_bounds_draw():
    """
    Draw two axis-aligned circles (XY, XZ) of radius=ogc_radius around every
    cloth vertex to visualise the OGC contact-offset sphere.

    Called by Blender for every viewport redraw; skips silently when no cloth
    object has show_ogc_bounds=True or self-collision is disabled.
    """
    try:
        import gpu
        from gpu_extras.batch import batch_for_shader
    except ImportError:
        return

    for cloth_obj in g_clothOBJs:
        if cloth_obj is None:
            continue
        s = getattr(cloth_obj, 'GPUCloth', None)
        if s is None or not s.show_ogc_bounds or not s.use_self_collision:
            continue

        radius = s.ogc_radius * 0.001  # mm → m
        mesh   = cloth_obj.data
        nv     = len(mesh.vertices)
        if nv == 0:
            continue

        # Fast bulk position readback (avoids per-vertex Python overhead)
        co = np.empty(nv * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', co)
        co = co.reshape(nv, 3)  # (nv, 3)

        SEGS = 16
        a    = np.linspace(0.0, 2.0 * math.pi, SEGS, endpoint=False, dtype=np.float32)
        ca   = np.cos(a)  # (SEGS,)
        sa   = np.sin(a)

        i0 = np.arange(SEGS)
        i1 = (i0 + 1) % SEGS

        # ── XY circle  (cx + r·cos, cy + r·sin, cz) ──────────────────────
        xy_x = co[:, 0:1]  # (nv,1)
        xy_y = co[:, 1:2]
        xy_z = np.repeat(co[:, 2:3], SEGS, axis=1)  # (nv, SEGS)

        p0_xy = np.stack([xy_x + radius * ca[i0],
                           xy_y + radius * sa[i0],
                           xy_z], axis=-1)  # (nv, SEGS, 3)
        p1_xy = np.stack([xy_x + radius * ca[i1],
                           xy_y + radius * sa[i1],
                           xy_z], axis=-1)

        # ── XZ circle  (cx + r·cos, cy, cz + r·sin) ──────────────────────
        xz_x = co[:, 0:1]
        xz_y = np.repeat(co[:, 1:2], SEGS, axis=1)  # (nv, SEGS)
        xz_z = co[:, 2:3]

        p0_xz = np.stack([xz_x + radius * ca[i0],
                           xz_y,
                           xz_z + radius * sa[i0]], axis=-1)
        p1_xz = np.stack([xz_x + radius * ca[i1],
                           xz_y,
                           xz_z + radius * sa[i1]], axis=-1)

        # Interleave p0/p1 into line-segment pairs: (nv, SEGS, 2, 3)→(N, 3)
        pts_xy = np.stack([p0_xy, p1_xy], axis=2).reshape(-1, 3)
        pts_xz = np.stack([p0_xz, p1_xz], axis=2).reshape(-1, 3)
        pts    = np.concatenate([pts_xy, pts_xz], axis=0)  # (nv*SEGS*4, 3)

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch  = batch_for_shader(shader, 'LINES', {"pos": pts})

        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(1.0)
        shader.bind()
        shader.uniform_float("color", (0.15, 0.90, 0.35, 0.40))
        batch.draw(shader)
        gpu.state.blend_set('NONE')


# ===========================================================================
#  Регистрация
# ===========================================================================

class GPUCloth_SyncCPUSettings(bpy.types.Operator):
    bl_idname = "gpucloth.sync_cpu_settings"
    bl_label = "Import CPU Cloth Settings"
    bl_description = "Import settings already supported by GPUCloth"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and hasattr(obj, 'GPUCloth')

    def execute(self, context):
        from . import cloth_settings_bridge
        result = cloth_settings_bridge.sync_cpu_to_gpu(
            context.object, context.scene)
        if result["errors"] or result["unsupported_non_default"]:
            message = (
                result["errors"][0] if result["errors"]
                else "Unsupported non-default setting: "
                + result["unsupported_non_default"][0])
            self.report({'ERROR'}, message)
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Imported {len(result['copied'])}; unsupported {len(result['unsupported'])}")
        return {'FINISHED'}


class GPUCloth_ShowCPUSyncReport(bpy.types.Operator):
    bl_idname = "gpucloth.show_cpu_sync_report"
    bl_label = "CPU Cloth Import Report"
    bl_description = "Show imported and unsupported CPU Cloth settings"

    @classmethod
    def poll(cls, context):
        obj = context.object
        return bool(
            obj is not None and hasattr(obj, 'GPUCloth')
            and obj.GPUCloth.cpu_sync_report)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=620)

    def draw(self, context):
        from . import cloth_settings_bridge
        report = cloth_settings_bridge.load_report(context.object)
        layout = self.layout
        layout.label(text=f"Imported: {len(report['copied'])}", icon='CHECKMARK')
        layout.label(text=f"Unsupported: {len(report['unsupported'])}", icon='QUESTION')
        if report["unsupported_non_default"]:
            layout.label(
                text=f"Blocking: {len(report['unsupported_non_default'])}", icon='ERROR')
            for name in report["unsupported_non_default"]:
                layout.label(text=name, icon='ERROR')
        blocking = set(report["unsupported_non_default"])
        for name in report["unsupported"]:
            if name not in blocking:
                layout.label(text=name)
        for error in report["errors"]:
            layout.label(text=error, icon='ERROR')

    def execute(self, context):
        return {'FINISHED'}


_OPERATOR_CLASSES = [
    GPUCloth_SyncCPUSettings,
    GPUCloth_ShowCPUSyncReport,
    GPUCloth_FreeVRAM,
    GPUCloth_LoadDLL,
    GPUCloth_UnloadDLL,
    GPUCloth_PrepareSimulation,
    GPUCloth_UpdateSimulation,
    GPUCloth_BeginDrape,
    GPUCloth_StepDrape,
    GPUCloth_ApplyDrape,
    GPUCloth_CancelDrape,
    GPUCloth_CopyInvariantDiagnostics,
    GPUCloth_SaveInvariantDiagnostics,
    GPUCloth_BakeSimulation,
    GPUCloth_FreeCache,
    GPUCloth_ExportAlembic,
    GPUCloth_ExportUSD,
    GPUCloth_TestDrapeOnSphere,
    GPUCloth_TestTwist,
    GPUCloth_TestMultiLayerDrop,
    GPUCloth_TestCushionDrop,
    GPUCloth_TestOGCBounds,
]


def _register_operator_surfaces():
    global _ogc_draw_handle
    registered_classes = []
    frame_handler_added = False
    cache_handler_added = False
    draw_handler_added = False
    try:
        for cls in _OPERATOR_CLASSES:
            bpy.utils.register_class(cls)
            registered_classes.append(cls)
        if _frame_change_handler not in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.append(_frame_change_handler)
            frame_handler_added = True
        if (_cache_input_change_handler not in
                bpy.app.handlers.depsgraph_update_post):
            bpy.app.handlers.depsgraph_update_post.append(
                _cache_input_change_handler)
            cache_handler_added = True
        if _ogc_draw_handle is None:
            _ogc_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                _ogc_bounds_draw, (), 'WINDOW', 'POST_VIEW')
            draw_handler_added = True
    except Exception:
        if draw_handler_added and _ogc_draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(
                _ogc_draw_handle, 'WINDOW')
            _ogc_draw_handle = None
        if (cache_handler_added and _cache_input_change_handler in
                bpy.app.handlers.depsgraph_update_post):
            bpy.app.handlers.depsgraph_update_post.remove(
                _cache_input_change_handler)
        if (frame_handler_added and _frame_change_handler in
                bpy.app.handlers.frame_change_post):
            bpy.app.handlers.frame_change_post.remove(
                _frame_change_handler)
        for cls in reversed(registered_classes):
            bpy.utils.unregister_class(cls)
        raise


def register():
    if not ensure_native_teardown():
        raise RuntimeError(
            "GPUCloth register blocked by retained native owners")
    _close_dll_directories()
    _register_operator_surfaces()

    # Сбрасываем глобальное состояние при регистрации
    # OGC contact-bounds visualiser — register once, draw callback checks flag
def unregister():
    global g_dll, g_runtime_initialized
    if not ensure_native_teardown(shutdown_runtime=True):
        print(
            "GPUCloth unregister retained handlers, classes, DLL, and "
            "owners because native teardown failed")
        return False

    global _ogc_draw_handle
    if _ogc_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_ogc_draw_handle, 'WINDOW')
        _ogc_draw_handle = None

    if _frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_frame_change_handler)
    if _cache_input_change_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(
            _cache_input_change_handler)

    for cls in reversed(_OPERATOR_CLASSES):
        bpy.utils.unregister_class(cls)

    # Очищаем состояние
    g_dll                = None
    g_runtime_initialized = False
    _close_dll_directories()
    return True
