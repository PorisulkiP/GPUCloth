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
from fractions import Fraction

import numpy as np
from ctypes import (
    addressof, cdll, windll, POINTER, pointer, cast,
    c_bool, c_float, c_int, c_uint, c_uint64, c_void_p, c_size_t,
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
g_runtime_handle     = CType.GPUClothV3RuntimeHandle(0)
g_cache_handle       = CType.GPUClothV3CacheHandle(0)
g_cache_owner        = None   # copied v3 cache identity/config; no hot-path query
_runtime_frame_generation = 0
_V3_RUNTIME_DEVICE_ORDINAL = 0
_V3_RUNTIME_APPLICATION_ID = 0x475055434C4F5448
# Blender's floating fps_base RNA value is IEEE-754 binary32-valued. Four
# unit-scale binary32 ulps is the explicit finite conversion tolerance. The
# 2^12 input-fraction bound exceeds 1/sqrt(4*2^-23); the final bounds mirror
# the v3 uint32 denominator and native INT16_MAX numerator guard.
_V3_FPS_NUMERATOR_MAX = 0x7FFF
_V3_FPS_DENOMINATOR_MAX = 0xFFFFFFFF
_V3_FPS_INPUT_DENOMINATOR_MAX = min(_V3_FPS_DENOMINATOR_MAX, 1 << 12)
_V3_FPS_RATE_TOLERANCE = 4.0 * (2.0 ** -23)
g_cloth_handles      = []     # list[GPUClothV3ClothHandle]
_cloth_input_owners  = []     # persistent v3 create/config payloads
_readback_owners     = []     # persistent v3 positions/velocities payloads
g_clothOBJs          = []     # list[bpy.types.Object]  — Blender-объекты ткани
g_simulationOBJs     = []     # render owner -> mesh actually sent to solver
g_clothCollisionOBJs = []     # list[POINTER(CType.Object)] — объекты столкновения
_dll_directory_handles = []

# Proxy v3: one persistent owner record per render/simulation binding.
g_proxy_handles      = []     # list[dict | None]
_collision_keepalive  = []     # prevent GC of collision ctypes data
_solver_diagnostics = []
_pin_snapshot_states = []
_dynamic_mesh_states = []
_collection_snapshots = []
_effector_weight_states = []
_effector_publication_state = {
    "payload_build_count": 0,
    "snapshot_build_count": 0,
    "snapshot_reuse_count": 0,
    "weight_owner_build_count": 0,
    "verification_round_trip_count": 0,
}
_collider_history = {}
_drape_status_by_uid = {}
_teardown_failure = False
_MODIFIER_VISIBILITY = (
    "show_viewport", "show_render", "show_in_editmode", "show_on_cage")


def _effector_publication_metrics():
    return dict(_effector_publication_state)


def _reset_effector_publication_metrics():
    for name in _effector_publication_state:
        _effector_publication_state[name] = 0


def _runtime_owners_retained():
    return bool(
        _runtime_handle_value() or _cache_handle_value() or g_cloth_handles or
        _cloth_input_owners or _readback_owners or
        g_clothOBJs or g_simulationOBJs or g_clothCollisionOBJs or
        g_proxy_handles or _collision_keepalive or _solver_diagnostics or
        _pin_snapshot_states or _dynamic_mesh_states or
        _collection_snapshots or _effector_weight_states or
        _collider_history or
        _initial_positions)


def _reject_unsupported_v3_owners(scene, cloth_objects):
    """Reject owners without a v3 ABI surface before native mutation."""
    unsupported = []
    helper = scene.gpu_cloth_helper
    for cloth_obj in cloth_objects:
        settings = cloth_obj.GPUCloth
        if str(getattr(settings, "solver_type", "")) not in ("PD", "Mil2"):
            unsupported.append(
                f"solver:{cloth_obj.name_full}:{settings.solver_type}")
        if (bool(getattr(settings, "use_dynamic_mesh", False)) and
                str(getattr(settings, "shapekey_rest", ""))):
            unsupported.append(
                f"rest_shape_key_dynamic_mesh:{cloth_obj.name_full}")
        if str(getattr(settings, "bending_model", "")) == "SDB":
            solver_type = str(getattr(settings, "solver_type", ""))
            if solver_type != "PD":
                unsupported.append(
                    f"SDB_bending_solver:{cloth_obj.name_full}:{solver_type}")
            if bool(getattr(settings, "use_anisotropy", False)):
                unsupported.append(
                    f"SDB_bending_anisotropy:{cloth_obj.name_full}")
        if bool(getattr(settings, "use_constraint_network", False)):
            if bool(getattr(settings, "use_dynamic_mesh", False)):
                unsupported.append(
                    f"constraint_network_dynamic_mesh:{cloth_obj.name_full}")
            if bool(getattr(settings, "use_sewing_springs", False)):
                unsupported.append(
                    f"constraint_network_sewing:{cloth_obj.name_full}")
            solver_mask = {
                "PD": CType.GPUCLOTH_SOLVER_PD,
                "Mil2": CType.GPUCLOTH_SOLVER_MIL2,
            }.get(str(getattr(settings, "solver_type", "")))
            if solver_mask is None:
                unsupported.append(
                    f"constraint_network_solver:{cloth_obj.name_full}")
            else:
                from . import cloth_settings_bridge
                try:
                    cloth_settings_bridge.capture_v3_constraint_network(
                        settings, solver_mask)
                except (AttributeError, TypeError, ValueError) as exc:
                    unsupported.append(
                        f"constraint_network_settings:{cloth_obj.name_full}:"
                        f"{exc}")
    if unsupported:
        raise RuntimeError(
            "NOT_CONFIGURABLE: ABI v3 has no owner for " +
            ", ".join(unsupported))


def _runtime_handle_value():
    value = getattr(g_runtime_handle, "value", g_runtime_handle)
    return int(value or 0)


def _cache_handle_value():
    owner = g_cache_owner
    handle = owner.get("handle") if owner is not None else g_cache_handle
    value = getattr(handle, "value", handle)
    return int(value or 0)


def _cache_handle_owner():
    """Return configure-time owned handle; global fallback only for cleanup."""
    owner = g_cache_owner
    return owner["handle"] if owner is not None else g_cache_handle


def _opaque_handle_value(handle):
    """Read a ctypes v3 handle without relying on ctypes.__int__."""
    value = getattr(handle, "value", handle)
    return int(value or 0)


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
    if not (_teardown_failure or _runtime_owners_retained()):
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


def _create_v3_cloth_owner(
        dll, cloth_obj, simulation_obj, mesh_snapshot, topology_generation,
        backend, geometry_generation=1):
    """Create one native v3 cloth; retain every caller-owned input buffer."""
    vertex_count = int(mesh_snapshot["vertex_count"])
    edges = tuple(mesh_snapshot["edges"])
    polygons = tuple(mesh_snapshot["polygons"])
    corners = tuple(mesh_snapshot["loops"])
    if not edges or not polygons or not corners:
        raise RuntimeError(
            "v3 cloth create requires non-empty vertices, edges, faces, "
            "and corners")
    object_id = _blender_session_uid(
        simulation_obj, "v3 cloth simulation object")
    topology_generation = int(topology_generation)
    geometry_generation = int(geometry_generation)
    if topology_generation <= 0 or geometry_generation <= 0:
        raise RuntimeError("v3 cloth generations must be positive")

    positions = (c_float * (vertex_count * 3))(
        *(component for position in mesh_snapshot["positions"]
          for component in position))
    edge_payload = (CType.GPUClothV3MeshEdge * len(edges))()
    source_edges = tuple(getattr(simulation_obj.data, "edges", ()))
    for index, (vertex_a, vertex_b) in enumerate(edges):
        edge_payload[index].vertex_a = int(vertex_a)
        edge_payload[index].vertex_b = int(vertex_b)
        edge_payload[index].edge_flags = (
            CType.GPUCLOTH_V3_MESH_EDGE_LOOSE
            if index < len(source_edges) and
            bool(getattr(source_edges[index], "is_loose", False))
            else 0)
        edge_payload[index].reserved = 0
    face_payload = (CType.GPUClothV3MeshFace * len(polygons))()
    for index, (first_corner, corner_count) in enumerate(polygons):
        face_payload[index].first_corner = int(first_corner)
        face_payload[index].corner_count = int(corner_count)
        face_payload[index].face_flags = 0
        face_payload[index].reserved = 0
    corner_payload = (CType.GPUClothV3MeshCorner * len(corners))()
    for index, (vertex_index, edge_index) in enumerate(corners):
        corner_payload[index].vertex_index = int(vertex_index)
        corner_payload[index].edge_index = int(edge_index)

    object_matrix = _matrix_signature(
        simulation_obj.matrix_world,
        f"{simulation_obj.name_full!r} world transform")
    inverse_matrix = _matrix_signature(
        _matrix_inverse(
            simulation_obj.matrix_world,
            f"{simulation_obj.name_full!r} world transform"),
        f"{simulation_obj.name_full!r} inverse transform")
    config = CType.GPUClothV3ClothCreateConfig()
    config.struct_size = sizeof(config)
    config.config_version = 1
    config.cloth_flags = CType.GPUCLOTH_V3_CLOTH_NONE
    config.backend = int(backend)
    config.object_id = object_id
    config.topology_generation = topology_generation
    config.geometry_generation = geometry_generation
    config.vertex_count = vertex_count
    config.edge_count = len(edges)
    config.face_count = len(polygons)
    config.corner_count = len(corners)
    _set_buffer_view(
        config.positions, CType.GPUCLOTH_ELEMENT_FLOAT3, vertex_count,
        sizeof(c_float) * 3, addressof(positions), geometry_generation)
    _set_buffer_view(
        config.edges, CType.GPUCLOTH_ELEMENT_MESH_EDGE, len(edges),
        sizeof(CType.GPUClothV3MeshEdge), addressof(edge_payload),
        topology_generation)
    _set_buffer_view(
        config.faces, CType.GPUCLOTH_ELEMENT_MESH_FACE, len(polygons),
        sizeof(CType.GPUClothV3MeshFace), addressof(face_payload),
        topology_generation)
    _set_buffer_view(
        config.corners, CType.GPUCLOTH_ELEMENT_MESH_CORNER, len(corners),
        sizeof(CType.GPUClothV3MeshCorner), addressof(corner_payload),
        topology_generation)
    config.object_to_world[:] = object_matrix
    config.world_to_object[:] = inverse_matrix
    config.reserved[:] = (0, 0, 0)

    out_handle = CType.GPUClothV3ClothHandle(0)
    result = int(dll.GPUCloth_v3_cloth_create(
        g_runtime_handle, pointer(config), pointer(out_handle)))
    if result != CType.GPUCLOTH_ABI_OK or not out_handle.value:
        raise RuntimeError(f"v3 cloth create rejected with {result}")

    readback_positions = (c_float * (vertex_count * 3))()
    readback_velocities = (c_float * (vertex_count * 3))()
    readback = CType.GPUClothV3ReadbackConfig()
    readback.struct_size = sizeof(readback)
    readback.config_version = 1
    readback.readback_flags = (
        CType.GPUCLOTH_V3_READBACK_POSITIONS |
        CType.GPUCLOTH_V3_READBACK_VELOCITIES)
    readback.reserved0 = 0
    readback.frame_generation = 0
    _set_buffer_view(
        readback.positions, CType.GPUCLOTH_ELEMENT_FLOAT3, vertex_count,
        sizeof(c_float) * 3, addressof(readback_positions), 0)
    _set_buffer_view(
        readback.velocities, CType.GPUCLOTH_ELEMENT_FLOAT3, vertex_count,
        sizeof(c_float) * 3, addressof(readback_velocities), 0)
    readback.reserved[:] = (0, 0)
    return {
        "handle": out_handle,
        "config": config,
        "positions": positions,
        "edges": edge_payload,
        "faces": face_payload,
        "corners": corner_payload,
        "readback": readback,
        "readback_positions": readback_positions,
        "readback_velocities": readback_velocities,
        "object_id": object_id,
        "topology_generation": topology_generation,
        "geometry_generation": geometry_generation,
        "vertex_count": vertex_count,
    }


def _capture_constraint_network(
        cloth_obj, simulation_obj, topology_generation, geometry_generation,
        solver_mask):
    """Capture loose-edge CN input; native owns a deep copy at configure."""
    from . import cloth_settings_bridge
    captured = cloth_settings_bridge.capture_v3_constraint_network(
        cloth_obj.GPUCloth, solver_mask)
    if captured is None:
        return None
    if bool(getattr(cloth_obj.GPUCloth, "use_sewing_springs", False)):
        raise RuntimeError(
            "constraint network and sewing springs are mutually exclusive")
    loose_edges = tuple(
        edge for edge in getattr(simulation_obj.data, "edges", ())
        if bool(getattr(edge, "is_loose", False)))
    if not loose_edges:
        raise RuntimeError(
            "constraint network requires at least one loose mesh edge")
    object_id = _blender_session_uid(
        simulation_obj, "v3 constraint-network object")
    phase_count = int(captured["phase_count"])
    if phase_count <= 0 or len(loose_edges) < phase_count:
        raise RuntimeError(
            "constraint network requires at least one loose edge per phase")
    records = (CType.GPUClothConstraintNetworkRecord * len(loose_edges))()
    for index, edge in enumerate(loose_edges):
        vertices = tuple(int(value) for value in edge.vertices)
        if len(vertices) != 2 or vertices[0] == vertices[1]:
            raise RuntimeError("constraint network has an invalid loose edge")
        records[index].seam_id = int(edge.index) + 1
        records[index].vertex_a = vertices[0]
        records[index].vertex_b = vertices[1]
        # Deterministic contiguous ownership. Native receives explicit phase
        # per record; no modulo/inference is performed at build time.
        records[index].phase = (
            index * phase_count // len(loose_edges)) + 1
        records[index].reserved = 0
    config = CType.GPUClothConstraintNetworkConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_CONSTRAINT_NETWORK
    config.header.config_version = 2
    config.header.flags = 0
    config.object_id = int(object_id)
    config.topology_generation = int(topology_generation)
    config.geometry_generation = int(geometry_generation)
    config.solver_mask = int(solver_mask)
    config.network_flags = CType.GPUCLOTH_CONSTRAINT_NETWORK_ENABLED
    config.phase_count = phase_count
    config.sewing_speed = float(captured["sewing_speed"])
    config.seam_stiffness = float(captured["seam_stiffness"])
    _set_buffer_view(
        config.constraints, CType.GPUCLOTH_ELEMENT_CONSTRAINT_NETWORK_RECORD,
        len(records), sizeof(CType.GPUClothConstraintNetworkRecord),
        addressof(records), int(topology_generation))
    config.reserved[:] = (0, 0, 0)
    return {
        "config": config,
        "records": records,
        "object_id": object_id,
        "topology_generation": int(topology_generation),
        "geometry_generation": int(geometry_generation),
        "solver_mask": int(solver_mask),
        "record_count": len(records),
    }


def _create_v3_proxy_owner(
        dll, cloth_owner, render_obj, simulation_obj, settings,
        readback_owner):
    """Create one persistent typed proxy owner."""
    if simulation_obj is render_obj:
        raise RuntimeError("v3 proxy requires a distinct simulation object")
    scene_type = int(getattr(settings, "proxy_scene_type", -1))
    if scene_type in (0, 1):
        proxy_flags = CType.GPUCLOTH_PROXY_LOCAL_FRAME
    elif scene_type == 3:
        proxy_flags = CType.GPUCLOTH_PROXY_DIRECT_BARYCENTRIC
    else:
        raise RuntimeError(
            "v3 proxy scene type is not representable without legacy semantics")
    render_x = int(getattr(settings, "hi_nx", 0))
    render_y = int(getattr(settings, "hi_ny", 0))
    proxy_x = int(getattr(settings, "proxy_nx", 0))
    proxy_y = int(getattr(settings, "proxy_ny", 0))
    sheets = int(getattr(settings, "num_sheets", 0))
    if min(render_x, render_y, proxy_x, proxy_y, sheets) <= 0:
        raise RuntimeError("v3 proxy grid settings must be positive")
    render_count = len(render_obj.data.vertices)
    proxy_count = len(simulation_obj.data.vertices)
    expected_render = (render_x + 1) * (render_y + 1) * sheets
    expected_proxy = (proxy_x + 1) * (proxy_y + 1) * sheets
    if render_count != expected_render or proxy_count != expected_proxy:
        raise RuntimeError(
            "v3 proxy topology does not match grid dimensions and sheet count")
    render_object_id = _blender_session_uid(
        render_obj, "v3 proxy render object")
    proxy_object_id = int(cloth_owner["object_id"])
    topology_generation = int(cloth_owner["topology_generation"])
    render_rest = (c_float * (render_count * 3))(
        *(component for vertex in render_obj.data.vertices for component in vertex.co))
    proxy_rest = (c_float * (proxy_count * 3))(
        *(component for vertex in simulation_obj.data.vertices for component in vertex.co))
    config = CType.GPUClothProxyConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_PROXY
    config.header.config_version = 1
    config.header.flags = 0
    config.render_object_id = render_object_id
    config.proxy_object_id = proxy_object_id
    config.topology_generation = topology_generation
    config.render_x_count = render_x
    config.render_y_count = render_y
    config.proxy_x_count = proxy_x
    config.proxy_y_count = proxy_y
    config.render_vertex_count = render_count
    config.proxy_vertex_count = proxy_count
    config.proxy_flags = proxy_flags
    config.reserved0 = 0
    _set_buffer_view(
        config.render_rest_positions, CType.GPUCLOTH_ELEMENT_FLOAT3,
        render_count, sizeof(c_float) * 3, addressof(render_rest),
        topology_generation)
    _set_buffer_view(
        config.proxy_rest_positions, CType.GPUCLOTH_ELEMENT_FLOAT3,
        proxy_count, sizeof(c_float) * 3, addressof(proxy_rest),
        topology_generation)
    config.reserved[:] = (0,)
    out_handle = CType.GPUClothV3ProxyHandle(0)
    result = int(dll.GPUCloth_v3_proxy_create(
        g_runtime_handle, cloth_owner["handle"], pointer(config),
        pointer(out_handle)))
    if result != CType.GPUCLOTH_ABI_OK or not out_handle.value:
        raise RuntimeError(f"v3 proxy create rejected with {result}")

    input_view = CType.GPUClothBufferView()
    output_view = CType.GPUClothBufferView()
    output_positions = (c_float * (render_count * 3))()
    _set_buffer_view(
        input_view, CType.GPUCLOTH_ELEMENT_FLOAT3, proxy_count,
        sizeof(c_float) * 3, addressof(readback_owner["readback_positions"]),
        0)
    _set_buffer_view(
        output_view, CType.GPUCLOTH_ELEMENT_FLOAT3, render_count,
        sizeof(c_float) * 3, addressof(output_positions), 0)
    status = CType.GPUClothV3ProxyStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    result = int(dll.GPUCloth_v3_proxy_get_status(
        out_handle, pointer(status)))
    if result != CType.GPUCLOTH_ABI_OK:
        int(dll.GPUCloth_v3_proxy_destroy(out_handle))
        raise RuntimeError(f"v3 proxy status rejected with {result}")
    if (int(status.render_vertex_count) != render_count or
            int(status.proxy_vertex_count) != proxy_count or
            int(status.topology_generation) != topology_generation or
            int(status.proxy_flags) != proxy_flags):
        int(dll.GPUCloth_v3_proxy_destroy(out_handle))
        raise RuntimeError("v3 proxy status does not match prepared identity")
    return {
        "handle": out_handle,
        "config": config,
        "render_rest": render_rest,
        "proxy_rest": proxy_rest,
        "input_view": input_view,
        "output_view": output_view,
        "output_positions": output_positions,
        "status": status,
        "readback_owner": readback_owner,
        "render_object_id": render_object_id,
        "proxy_object_id": proxy_object_id,
        "topology_generation": topology_generation,
        "render_vertex_count": render_count,
        "proxy_vertex_count": proxy_count,
        "allocation_count": 3,
    }


def _apply_v3_proxy(owner, frame_generation):
    generation = int(frame_generation)
    if generation <= 0:
        raise RuntimeError("v3 proxy frame generation must be positive")
    owner["input_view"].generation = generation
    owner["output_view"].generation = generation
    result = int(g_dll.GPUCloth_v3_proxy_apply(
        owner["handle"], pointer(owner["input_view"]),
        pointer(owner["output_view"])))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"v3 proxy apply rejected with {result}")
    status = owner["status"]
    status.struct_size = sizeof(status)
    status.status_version = 1
    result = int(g_dll.GPUCloth_v3_proxy_get_status(
        owner["handle"], pointer(status)))
    if (result != CType.GPUCLOTH_ABI_OK or
            int(status.last_generation) != generation or
            int(status.apply_count) <= int(owner.get("apply_count", 0))):
        raise RuntimeError("v3 proxy status did not accept applied frame")
    owner["apply_count"] = int(status.apply_count)
    return owner["output_positions"]


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


def _validate_external_cache_playback_source(scene):
    """Reject an incomplete external source before any native owner exists."""
    helper = scene.gpu_cloth_helper
    root = _active_cache_path(scene)
    start = int(helper.bake_start)
    end = int(helper.bake_end)
    if start < 0 or end < start:
        raise RuntimeError("external cache frame range is invalid")
    missing = [
        os.path.join(root, f"frame_{frame:06d}.bin")
        for frame in range(start, end + 1)
        if not os.path.isfile(os.path.join(root, f"frame_{frame:06d}.bin"))]
    if missing:
        raise RuntimeError(
            "external cache source is incomplete; missing frame " +
            os.path.basename(missing[0]))


def _configure_cache_features(dll, scene):
    global g_cache_handle, g_cache_owner
    if _runtime_handle_value() == 0:
        raise RuntimeError("v3 cache requires a live runtime owner")
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
    if helper.use_external_cache:
        storage_feature = CType.GPUCLOTH_FEATURE_CACHE_EXTERNAL
    else:
        storage_feature = (
            CType.GPUCLOTH_FEATURE_CACHE_DISK
            if helper.use_disk_cache
            else CType.GPUCLOTH_FEATURE_CACHE_MEMORY)
    config.header.feature_id = storage_feature
    out_cache = CType.GPUClothV3CacheHandle(0)
    result = int(dll.GPUCloth_v3_cache_configure(
        g_runtime_handle, pointer(config), pointer(out_cache)))
    if result != CType.GPUCLOTH_ABI_OK or not out_cache.value:
        raise RuntimeError(f"v3 cache configure rejected with {result}")
    # Native copied every pointer-bearing field before returning.  Keep a
    # pointer-free config identity locally so frame operations never query the
    # ABI and never retain dead ctypes string addresses.
    owned_config = CType.GPUClothCacheConfig.from_buffer_copy(bytes(config))
    owned_config.path_utf8_address = 0
    owned_config.name_utf8_address = 0
    g_cache_handle = out_cache
    g_cache_owner = {
        "handle": out_cache,
        "cache_id": int(config.cache_id),
        "config": owned_config,
        "path": bytes(path_bytes),
        "name": bytes(name_bytes),
    }
    return CType.GPUCLOTH_ABI_OK


def _cache_requested_for_prepare(scene):
    """Cache owner is created only for an active cache session.

    `use_disk_cache` is a PointCache storage preference, not a request to
    write every live viewport frame.  Bake/playback, external ownership, or
    an existing cached session explicitly request the v3 owner.
    """
    helper = scene.gpu_cloth_helper
    return bool(
        getattr(helper, "use_external_cache", False) or
        getattr(helper, "playback_mode", False) or
        getattr(helper, "is_baked", False) or
        getattr(helper, "is_baking", False) or
        int(getattr(helper, "cached_frame_count", 0)) > 0)


def _v3_configure_feature(dll, cloth_handle, config):
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"typed v3 cloth feature rejected with {result}")
    return result


def _configure_simulation_features(
        dll, cloth_handle, scene, settings,
        object_id=None, topology_generation=None, geometry_generation=None):
    solver_mask = {
        'PD': CType.GPUCLOTH_SOLVER_PD,
        'Mil2': CType.GPUCLOTH_SOLVER_MIL2,
    }.get(settings.solver_type)
    if solver_mask is None:
        raise RuntimeError(
            f"typed simulation config owns only PD/Mil2; got "
            f"{settings.solver_type}")
    if (object_id is None or topology_generation is None or
            geometry_generation is None or int(object_id) == 0 or
            int(topology_generation) == 0 or int(geometry_generation) == 0):
        raise RuntimeError(
            "typed effector scales require nonzero cloth identity/generations")
    from . import cloth_settings_bridge
    force_scale, wind_scale = (
        cloth_settings_bridge.capture_v3_effector_scales(settings))

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
    config.velocity_damping = (
        cloth_settings_bridge.capture_v3_velocity_damping(settings))
    config.simulation_flags = 0
    config.reserved[:] = (0, 0)
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    for feature in (
            CType.GPUCLOTH_FEATURE_TIMESTEP_SPEED,
            CType.GPUCLOTH_FEATURE_MATERIAL_MASS,
            CType.GPUCLOTH_FEATURE_GRAVITY_VECTOR,
            CType.GPUCLOTH_FEATURE_SIMULATION_QUALITY,
            CType.GPUCLOTH_FEATURE_AIR_DAMPING,
            CType.GPUCLOTH_FEATURE_VELOCITY_DAMPING):
        config.header.feature_id = feature
        config.header.config_version = (
            2 if feature == CType.GPUCLOTH_FEATURE_VELOCITY_DAMPING else 1)
        result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"typed simulation feature {feature} rejected with {result}")

    effector_scales = CType.GPUClothEffectorScaleConfig()
    effector_scales.header.struct_size = sizeof(effector_scales)
    effector_scales.header.feature_id = (
        CType.GPUCLOTH_FEATURE_EFFECTOR_SCALES)
    effector_scales.header.config_version = 1
    effector_scales.header.flags = 0
    effector_scales.solver_mask = solver_mask
    effector_scales.reserved0 = 0
    effector_scales.object_id = int(object_id)
    effector_scales.topology_generation = int(topology_generation)
    effector_scales.geometry_generation = int(geometry_generation)
    effector_scales.force_scale = force_scale
    effector_scales.wind_scale = wind_scale
    effector_scales.reserved[:] = (0, 0)
    result = int(dll.GPUCloth_v3_cloth_configure(
        cloth_handle,
        cast(pointer(effector_scales),
             POINTER(CType.GPUClothFeatureConfigHeader))))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed effector scales rejected with {result}")
    return CType.GPUCLOTH_ABI_OK


def _configure_solver_diagnostics(dll, cloth_handle, event_capacity=16):
    if event_capacity < 1 or event_capacity > 64:
        raise RuntimeError("diagnostic event capacity must be in [1, 64]")
    event_type = CType.GPUClothDiagnosticsEvent * event_capacity
    events = event_type()
    generation = ((_opaque_handle_value(cloth_handle) ^ 0x475055434C4C4F54) &
                  0xffffffffffffffff) or 1
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
    result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
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


# Keep dict insertion order aligned with GPUClothV3.exports.allowlist.
_GPUCLOTH_V3_EXPORT_SIGNATURES = {
    "GPUCloth_v3_cache_clear": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle],
    "GPUCloth_v3_cache_configure": [
        CType.GPUClothV3RuntimeHandle,
        POINTER(CType.GPUClothCacheConfig),
        POINTER(CType.GPUClothV3CacheHandle)],
    "GPUCloth_v3_cache_destroy": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle],
    "GPUCloth_v3_cache_flush": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle],
    "GPUCloth_v3_cache_free_frame": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle,
        POINTER(CType.GPUClothV3CacheFrameConfig)],
    "GPUCloth_v3_cache_get_status": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle,
        POINTER(CType.GPUClothCacheStatus)],
    "GPUCloth_v3_cache_has_frame": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle,
        POINTER(CType.GPUClothV3CacheFrameConfig), POINTER(c_uint)],
    "GPUCloth_v3_cache_is_frame_ready": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle,
        POINTER(CType.GPUClothV3CacheFrameConfig), POINTER(c_uint)],
    "GPUCloth_v3_cache_prefetch_frame": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle,
        POINTER(CType.GPUClothV3CacheFrameConfig)],
    "GPUCloth_v3_cache_query": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle,
        POINTER(CType.GPUClothCacheConfig)],
    "GPUCloth_v3_cache_read_frame": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle,
        POINTER(CType.GPUClothV3CacheFrameConfig)],
    "GPUCloth_v3_cache_update_status": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle,
        POINTER(CType.GPUClothCacheStatusUpdate)],
    "GPUCloth_v3_cache_write_frame_async": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3CacheHandle,
        POINTER(CType.GPUClothV3CacheFrameConfig)],
    "GPUCloth_v3_cloth_apply_drape": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothDrapeStatus)],
    "GPUCloth_v3_cloth_begin_drape": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothDrapeConfig),
        POINTER(CType.GPUClothDrapeStatus)],
    "GPUCloth_v3_cloth_build": [CType.GPUClothV3ClothHandle],
    "GPUCloth_v3_cloth_cancel_drape": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothDrapeStatus)],
    "GPUCloth_v3_cloth_configure": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothFeatureConfigHeader)],
    "GPUCloth_v3_cloth_create": [
        CType.GPUClothV3RuntimeHandle,
        POINTER(CType.GPUClothV3ClothCreateConfig),
        POINTER(CType.GPUClothV3ClothHandle)],
    "GPUCloth_v3_cloth_destroy": [CType.GPUClothV3ClothHandle],
    "GPUCloth_v3_cloth_get_collection_status": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothCollectionStatus)],
    "GPUCloth_v3_cloth_get_constraint_network_status": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothV3ConstraintNetworkStatus)],
    "GPUCloth_v3_cloth_get_diagnostics": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothDiagnosticsStatus)],
    "GPUCloth_v3_cloth_get_effector_scales_status": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothV3EffectorScaleStatus)],
    "GPUCloth_v3_cloth_get_invariant_status": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothInvariantWitness)],
    "GPUCloth_v3_cloth_get_preparation_status": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothPreparationStatus)],
    "GPUCloth_v3_cloth_get_sdb_status": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothV3SDBStatus)],
    "GPUCloth_v3_cloth_get_shrink_status": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothV3ShrinkStatus)],
    "GPUCloth_v3_cloth_get_status": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothV3ClothStatus)],
    "GPUCloth_v3_cloth_get_velocity_damping_status": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothV3VelocityDampingStatus)],
    "GPUCloth_v3_cloth_query_collection": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothCollectionQuery)],
    "GPUCloth_v3_cloth_query_material_state": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothV3MaterialStateQuery)],
    "GPUCloth_v3_cloth_readback": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothV3ReadbackConfig)],
    "GPUCloth_v3_cloth_set_pin_snapshot": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothPinSnapshotConfig)],
    "GPUCloth_v3_cloth_set_shrink_config": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothShrinkConfig)],
    "GPUCloth_v3_cloth_set_vertex_channel": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothVertexChannelConfig)],
    "GPUCloth_v3_cloth_step": [CType.GPUClothV3ClothHandle],
    "GPUCloth_v3_cloth_step_drape": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothDrapeStatus)],
    "GPUCloth_v3_cloth_validate_initial_state": [
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothPreparationConfig),
        POINTER(CType.GPUClothPreparationStatus)],
    "GPUCloth_v3_collection_stage_mesh_state": [
        CType.GPUClothV3TransactionHandle,
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothMeshStateConfig)],
    "GPUCloth_v3_collection_stage_pin_snapshot": [
        CType.GPUClothV3TransactionHandle,
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothPinSnapshotConfig)],
    "GPUCloth_v3_collection_stage_snapshot": [
        CType.GPUClothV3TransactionHandle,
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothCollectionSnapshotConfig)],
    "GPUCloth_v3_collection_transaction_abort": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3TransactionHandle],
    "GPUCloth_v3_collection_transaction_begin": [
        CType.GPUClothV3RuntimeHandle,
        POINTER(CType.GPUClothCollectionTransactionConfig),
        POINTER(CType.GPUClothV3TransactionHandle)],
    "GPUCloth_v3_collection_transaction_commit": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3TransactionHandle],
    "GPUCloth_v3_get_abi_info": [
        POINTER(CType.GPUClothV3ABIInfo)],
    "GPUCloth_v3_get_descriptor_layout": [
        POINTER(CType.GPUClothDescriptorLayout)],
    "GPUCloth_v3_get_feature_count": [POINTER(c_uint64)],
    "GPUCloth_v3_get_feature_info": [
        c_uint64, POINTER(CType.GPUClothFeatureInfo)],
    "GPUCloth_v3_proxy_apply": [
        CType.GPUClothV3ProxyHandle,
        POINTER(CType.GPUClothBufferView),
        POINTER(CType.GPUClothBufferView)],
    "GPUCloth_v3_proxy_create": [
        CType.GPUClothV3RuntimeHandle,
        CType.GPUClothV3ClothHandle,
        POINTER(CType.GPUClothProxyConfig),
        POINTER(CType.GPUClothV3ProxyHandle)],
    "GPUCloth_v3_proxy_destroy": [CType.GPUClothV3ProxyHandle],
    "GPUCloth_v3_proxy_get_status": [
        CType.GPUClothV3ProxyHandle,
        POINTER(CType.GPUClothV3ProxyStatus)],
    "GPUCloth_v3_query_feature": [
        c_uint, POINTER(CType.GPUClothFeatureInfo)],
    "GPUCloth_v3_runtime_create": [
        POINTER(CType.GPUClothV3RuntimeConfig),
        POINTER(CType.GPUClothV3RuntimeHandle)],
    "GPUCloth_v3_runtime_destroy": [CType.GPUClothV3RuntimeHandle],
    "GPUCloth_v3_runtime_update": [
        CType.GPUClothV3RuntimeHandle,
        POINTER(CType.GPUClothV3FrameConfig)],
}


def _bind_gpucloth_v3_exports(dll):
    for name, argtypes in _GPUCLOTH_V3_EXPORT_SIGNATURES.items():
        # No default: missing v3 export raises and aborts DLL load.
        function = getattr(dll, name)
        function.argtypes = argtypes
        function.restype = c_uint


def _validate_descriptor_layout(dll):
    layout = CType.GPUClothDescriptorLayout()
    layout.struct_size = sizeof(layout)
    result = int(dll.GPUCloth_v3_get_descriptor_layout(pointer(layout)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"native v3 descriptor layout query failed: result={result}")
    expected = {
        "feature_config_header_size": sizeof(
            CType.GPUClothFeatureConfigHeader),
        "buffer_view_size": sizeof(CType.GPUClothBufferView),
        "named_value_size": sizeof(CType.GPUClothNamedValue),
        "simulation_config_size": sizeof(CType.GPUClothSimulationConfig),
        "material_config_size": sizeof(CType.GPUClothMaterialConfig),
        "pin_config_size": sizeof(CType.GPUClothPinConfig),
        "constraint_config_size": sizeof(CType.GPUClothConstraintConfig),
        "pressure_config_size": sizeof(CType.GPUClothPressureConfig),
        "collision_config_size": sizeof(CType.GPUClothCollisionConfig),
        "collider_config_size": sizeof(CType.GPUClothColliderConfig),
        "mesh_state_config_size": sizeof(CType.GPUClothMeshStateConfig),
        "effector_config_size": sizeof(CType.GPUClothEffectorConfig),
        "effector_weights_config_size": sizeof(
            CType.GPUClothEffectorWeightsConfig),
        "cache_config_size": sizeof(CType.GPUClothCacheConfig),
        "cache_status_update_size": sizeof(
            CType.GPUClothCacheStatusUpdate),
        "cache_status_size": sizeof(CType.GPUClothCacheStatus),
        "sewing_record_size": sizeof(CType.GPUClothSewingRecord),
        "sewing_config_size": sizeof(CType.GPUClothSewingConfig),
        "vertex_channel_config_size": sizeof(
            CType.GPUClothVertexChannelConfig),
        "collision_filter_config_size": sizeof(
            CType.GPUClothCollisionFilterConfig),
        "proxy_config_size": sizeof(CType.GPUClothProxyConfig),
        "diagnostics_config_size": sizeof(CType.GPUClothDiagnosticsConfig),
        "material_state_query_size": sizeof(
            CType.GPUClothV3MaterialStateQuery),
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
        "sdb_status_size": sizeof(CType.GPUClothV3SDBStatus),
        "proxy_status_size": sizeof(CType.GPUClothV3ProxyStatus),
        "velocity_damping_status_size": sizeof(
            CType.GPUClothV3VelocityDampingStatus),
        "effector_scales_config_size": sizeof(
            CType.GPUClothEffectorScaleConfig),
        "effector_scales_status_size": sizeof(
            CType.GPUClothV3EffectorScaleStatus),
        "constraint_network_config_size": sizeof(
            CType.GPUClothConstraintNetworkConfig),
        "constraint_network_status_size": sizeof(
            CType.GPUClothV3ConstraintNetworkStatus),
        "constraint_network_record_size": sizeof(
            CType.GPUClothConstraintNetworkRecord),
        "shrink_config_size": sizeof(CType.GPUClothShrinkConfig),
        "shrink_status_size": sizeof(CType.GPUClothV3ShrinkStatus),
    }
    mismatches = [
        f"{name}={int(getattr(layout, name))}, expected={expected_size}"
        for name, expected_size in expected.items()
        if int(getattr(layout, name)) != expected_size]
    if (int(layout.struct_size) != sizeof(layout) or
            int(layout.schema_version) != 12 or
            mismatches):
        detail = "; ".join(mismatches) if mismatches else "header mismatch"
        raise RuntimeError(f"native v3 descriptor ABI mismatch: {detail}")
    return layout


def _validate_product_abi(dll):
    info = CType.GPUClothV3ABIInfo()
    info.struct_size = sizeof(info)
    result = int(dll.GPUCloth_v3_get_abi_info(pointer(info)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"native v3 ABI query failed: result={result}")

    expected = {
        "struct_size": sizeof(info),
        "struct_version": 1,
        "abi_major": 3,
        "abi_minor": 0,
        "abi_patch": 0,
        "feature_schema_version": 11,
        "backend_mask": CType.GPUCLOTH_V3_BACKEND_MASK_ALL,
        "pointer_width_bits": 64,
        "little_endian": 1,
        "export_manifest_version": 1,
        "reserved0": 0,
    }
    mismatches = [
        f"{name}={int(getattr(info, name))}, expected={expected_value}"
        for name, expected_value in expected.items()
        if int(getattr(info, name)) != expected_value]
    if sys.byteorder != "little":
        mismatches.append(f"host_byteorder={sys.byteorder}, expected=little")
    if any(int(value) != 0 for value in info.reserved):
        mismatches.append("reserved fields are non-zero")
    if mismatches:
        raise RuntimeError("native v3 ABI mismatch: " + "; ".join(mismatches))

    count = c_uint64()
    result = int(dll.GPUCloth_v3_get_feature_count(pointer(count)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"native v3 feature-count query failed: result={result}")
    native_count = int(count.value)
    declared_count = int(info.feature_count)
    if (native_count == 0 or native_count > 0xffffffff or
            declared_count != native_count):
        raise RuntimeError(
            "native v3 feature count mismatch: "
            f"abi={declared_count}, native={native_count}")
    feature_ids = set()
    allowed_feature_flags = (
        CType.GPUCLOTH_FEATURE_BLENDER_CORE |
        CType.GPUCLOTH_FEATURE_RELEASE_REQUIRED |
        CType.GPUCLOTH_FEATURE_EXTENSION)
    known_config_mask = (
        CType.GPUCLOTH_CONFIG_SIMULATION |
        CType.GPUCLOTH_CONFIG_MATERIAL |
        CType.GPUCLOTH_CONFIG_PIN |
        CType.GPUCLOTH_CONFIG_CONSTRAINT |
        CType.GPUCLOTH_CONFIG_PRESSURE |
        CType.GPUCLOTH_CONFIG_COLLISION |
        CType.GPUCLOTH_CONFIG_COLLIDER |
        CType.GPUCLOTH_CONFIG_MESH_STATE |
        CType.GPUCLOTH_CONFIG_EFFECTOR |
        CType.GPUCLOTH_CONFIG_EFFECTOR_WEIGHTS |
        CType.GPUCLOTH_CONFIG_CACHE |
        CType.GPUCLOTH_CONFIG_SEWING |
        CType.GPUCLOTH_CONFIG_VERTEX_CHANNEL |
        CType.GPUCLOTH_CONFIG_COLLISION_FILTER |
        CType.GPUCLOTH_CONFIG_PROXY |
        CType.GPUCLOTH_CONFIG_DIAGNOSTICS |
        CType.GPUCLOTH_CONFIG_COLLECTION)
    configless_features = {"lifecycle", "readback"}
    for index in range(native_count):
        indexed = CType.GPUClothFeatureInfo()
        indexed.struct_size = sizeof(indexed)
        result = int(dll.GPUCloth_v3_get_feature_info(
            c_uint64(index), pointer(indexed)))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"native v3 feature-info query failed: index={index}, "
                f"result={result}")
        if int(indexed.struct_size) != sizeof(indexed):
            raise RuntimeError(
                f"native v3 feature-info size mismatch: index={index}, "
                f"native={int(indexed.struct_size)}, "
                f"expected={sizeof(indexed)}")
        name = bytes(indexed.name).split(b"\0", 1)[0].decode(
            "ascii", errors="strict")
        supported_solver_mask = int(indexed.supported_solver_mask)
        proven_solver_mask = int(indexed.proven_solver_mask)
        expected_status = (
            CType.GPUCLOTH_FEATURE_PROVEN
            if proven_solver_mask == supported_solver_mask and
            proven_solver_mask != 0 else
            CType.GPUCLOTH_FEATURE_PARTIAL
            if supported_solver_mask != 0 else
            CType.GPUCLOTH_FEATURE_MISSING)
        if int(indexed.status) != expected_status:
            raise RuntimeError(
                f"native v3 feature status does not match solver coverage: "
                f"{name}, status={int(indexed.status)}, "
                f"expected={expected_status}, supported={supported_solver_mask}, "
                f"proven={proven_solver_mask}")
        if int(indexed.flags) & ~allowed_feature_flags:
            raise RuntimeError(
                f"native v3 feature exposes unknown flags: {name}, "
                f"flags={int(indexed.flags)}")
        if supported_solver_mask & ~CType.GPUCLOTH_SOLVER_ALL:
            raise RuntimeError(
                f"native v3 feature exposes non-product solver mask: {name}")
        if proven_solver_mask & ~supported_solver_mask:
            raise RuntimeError(
                f"native v3 feature proves unsupported solver: {name}")
        if int(indexed.config_kind_mask) & ~known_config_mask:
            raise RuntimeError(
                f"native v3 feature exposes unknown config kind: {name}, "
                f"mask={int(indexed.config_kind_mask)}")
        expected_config_version = 2 if name in (
            "anisotropy", "velocity_damping", "constraint_network") else 1
        if int(indexed.config_version) != expected_config_version:
            raise RuntimeError(
                f"native v3 feature config version mismatch: {name}, "
                f"native={int(indexed.config_version)}, "
                f"expected={expected_config_version}")
        if (int(indexed.config_kind_mask) == CType.GPUCLOTH_CONFIG_NONE and
                name not in configless_features):
            raise RuntimeError(
                f"native v3 configurable feature has no typed owner: {name}")
        feature_id = int(indexed.feature_id)
        if feature_id in feature_ids:
            raise RuntimeError(
                f"native v3 feature manifest duplicate id: {feature_id}")
        feature_ids.add(feature_id)
        queried = CType.GPUClothFeatureInfo()
        queried.struct_size = sizeof(queried)
        result = int(dll.GPUCloth_v3_query_feature(
            c_uint(feature_id), pointer(queried)))
        if result != CType.GPUCLOTH_ABI_OK or bytes(queried) != bytes(indexed):
            raise RuntimeError(
                f"native v3 feature query mismatch: index={index}, "
                f"feature_id={feature_id}, result={result}")
    return info


def _validate_native_preparation(
        dll, cloth_handle, topology_generation, requested_generation):
    config = CType.GPUClothPreparationConfig()
    config.struct_size = sizeof(config)
    config.config_version = 1
    config.preparation_flags = CType.GPUCLOTH_PREPARATION_CONFIGURED
    config.topology_generation = int(topology_generation)
    config.requested_generation = int(requested_generation)
    status = CType.GPUClothPreparationStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    result = int(dll.GPUCloth_v3_cloth_validate_initial_state(
        cloth_handle, pointer(config), pointer(status)))
    if (result != CType.GPUCLOTH_ABI_OK or
            int(status.result) != CType.GPUCLOTH_PREPARATION_RESULT_READY or
            not (int(status.status_flags) &
                 CType.GPUCLOTH_PREPARATION_STATUS_RUNNABLE)):
        witness = CType.GPUClothInvariantWitness()
        witness.struct_size = sizeof(witness)
        witness.witness_version = 1
        witness_result = int(dll.GPUCloth_v3_cloth_get_invariant_status(
            cloth_handle, pointer(witness)))
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


def _invariant_witness_data(cloth_handle):
    witness = CType.GPUClothInvariantWitness()
    witness.struct_size = sizeof(witness)
    witness.witness_version = 1
    result = int(g_dll.GPUCloth_v3_cloth_get_invariant_status(
        cloth_handle, pointer(witness)))
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
        return _invariant_witness_data(g_cloth_handles[index])
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
        result = int(g_dll.GPUCloth_v3_cloth_get_preparation_status(
            g_cloth_handles[index], pointer(status)))
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


def _solver_diagnostic_snapshot(index):
    if index < 0 or index >= len(_solver_diagnostics):
        return None
    owner = _solver_diagnostics[index]
    status = CType.GPUClothDiagnosticsStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    if index >= len(g_cloth_handles):
        return None
    result = int(g_dll.GPUCloth_v3_cloth_get_diagnostics(
        g_cloth_handles[index], pointer(status)))
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
    solver_type = str(getattr(settings, "solver_type", ""))
    bending_model = str(getattr(settings, "bending_model", ""))
    if solver_type not in {'PD', 'Mil2'}:
        raise RuntimeError(f"unknown solver type {solver_type!r}")
    if bending_model not in {'LINEAR', 'ANGULAR', 'SDB'}:
        raise RuntimeError(f"unknown bending model {bending_model!r}")
    if solver_type != 'PD' and bending_model == 'SDB':
        raise RuntimeError(
            "NOT_CONFIGURABLE: SDB bending is only supported by PD")
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
        dll, cloth_handle, owner, publish_anisotropy=False):
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
        features.insert(
            3,
            (CType.GPUCLOTH_FEATURE_BENDING_SDB
             if owner["bending_model"] == 'SDB' else
             CType.GPUCLOTH_FEATURE_BENDING_ANGULAR
             if owner["bending_model"] == 'ANGULAR' else
             CType.GPUCLOTH_FEATURE_BENDING_LINEAR))

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
        result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
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


def _publish_internal_springs_config(dll, cloth_handle, prepared_config):
    config = CType.GPUClothConstraintConfig.from_buffer_copy(
        bytes(prepared_config))
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed internal-springs config rejected with {result}")
    return result


def _publish_constraint_network(dll, cloth_handle, prepared):
    if prepared is None:
        return CType.GPUCLOTH_ABI_OK
    config = CType.GPUClothConstraintNetworkConfig.from_buffer_copy(
        bytes(prepared["config"]))
    # Keep the copied caller owner live through the native call. Native then
    # owns records independently of this ctypes allocation.
    config.constraints.data_address = addressof(prepared["records"])
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed constraint-network config rejected with {result}")
    return result


def _validate_constraint_network_status(dll, cloth_handle, prepared):
    if prepared is None:
        return None
    status = CType.GPUClothV3ConstraintNetworkStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    result = int(dll.GPUCloth_v3_cloth_get_constraint_network_status(
        cloth_handle, pointer(status)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed constraint-network status rejected with {result}")
    expected = prepared["record_count"]
    expected_phase_mask = (1 << int(prepared["config"].phase_count)) - 1
    expected_policy = CType.constraint_network_stiffness_policy(
        prepared["solver_mask"])
    if (int(status.state) != CType.GPUCLOTH_V3_CONSTRAINT_NETWORK_APPLIED or
            int(status.configured) != 1 or int(status.applied) != 1 or
            int(status.enabled) != 1 or
            int(status.solver_mask) != int(prepared["solver_mask"]) or
            int(CType.constraint_network_stiffness_policy(
                status.solver_mask)) != int(expected_policy) or
            int(status.record_count) != expected or
            int(status.spring_count) != expected or
            int(status.phase_count) != int(prepared["config"].phase_count) or
            int(status.phase_coverage_mask) != expected_phase_mask or
            abs(float(status.sewing_speed) -
                float(prepared["config"].sewing_speed)) > 1e-6 or
            abs(float(status.seam_stiffness) -
                float(prepared["config"].seam_stiffness)) > 1e-6 or
            int(status.object_id) != int(prepared["object_id"]) or
            int(status.topology_generation) !=
                int(prepared["topology_generation"]) or
            int(status.geometry_generation) !=
                int(prepared["geometry_generation"])):
        raise RuntimeError(
            "typed constraint-network status does not match accepted owner")
    return status


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


def _publish_pressure_features(dll, cloth_handle, prepared):
    if prepared is None:
        return CType.GPUCLOTH_ABI_OK
    prepared_config, features = prepared
    config = CType.GPUClothPressureConfig.from_buffer_copy(
        bytes(prepared_config))
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    for feature in features:
        config.header.feature_id = feature
        result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
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


def _upload_rest_shape_key(
        dll, cloth_handle, prepared_rest_shape,
        object_id, topology_generation, geometry_generation):
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
    config.object_id = int(object_id)
    config.topology_generation = int(topology_generation)
    config.geometry_generation = int(geometry_generation)
    config.rest_generation = generation
    config.rest_positions.struct_size = sizeof(CType.GPUClothBufferView)
    config.rest_positions.element_type = CType.GPUCLOTH_ELEMENT_FLOAT3
    config.rest_positions.element_count = len(positions)
    config.rest_positions.stride_bytes = sizeof(c_float) * 3
    config.rest_positions.data_address = positions.ctypes.data
    config.rest_positions.generation = generation
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
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


def _upload_sewing(dll, cloth_handle, settings_owner, simulation_obj):
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
    result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
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
        identity_obj=simulation_obj)


def _publish_prepared_pin_snapshot(dll, cloth_handle, snapshot):
    return publish_pin_snapshot(dll, CType, cloth_handle, snapshot)


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


def _upload_stiffness_channels(dll, cloth_handle, prepared_channels):
    for channel, weights in prepared_channels:
        apply_float_channel(
            dll, CType, cloth_handle,
            CType.GPUCLOTH_FEATURE_STIFFNESS_VERTEX_GROUPS,
            channel, 1, weights)
    return CType.GPUCLOTH_ABI_OK


def _upload_pressure_weights(
        dll, cloth_handle, settings_owner, simulation_obj):
    if not settings_owner.GPUCloth.use_pressure:
        return CType.GPUCLOTH_ABI_OK
    weights = vertex_group_weights(
        simulation_obj, settings_owner.GPUCloth.vgroup_pressure, "pressure")
    if weights is None:
        return CType.GPUCLOTH_ABI_OK
    return apply_float_channel(
        dll, CType, cloth_handle, CType.GPUCLOTH_FEATURE_PRESSURE_VERTEX_GROUP,
        CType.GPUCLOTH_VERTEX_PRESSURE_WEIGHT, 1, weights)


def _upload_shrink_weights(
        dll, cloth_handle, settings_owner, simulation_obj):
    weights = vertex_group_weights(
        simulation_obj, settings_owner.GPUCloth.vgroup_shrink, "shrink")
    if weights is None:
        return CType.GPUCLOTH_ABI_OK
    return apply_float_channel(
        dll, CType, cloth_handle, CType.GPUCLOTH_FEATURE_SHRINK,
        CType.GPUCLOTH_VERTEX_SHRINK_WEIGHT, 1, weights)


def _capture_shrink_bounds(settings_owner):
    from . import cloth_settings_bridge
    try:
        shrink_min, shrink_max = (
            cloth_settings_bridge.capture_v3_shrink_bounds(
                settings_owner.GPUCloth))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"shrink bounds capture failed: {exc}") from exc
    return (
        _bounded_float32(shrink_min, "shrink minimum", -1.0, 1.0),
        _bounded_float32(shrink_max, "shrink maximum", -1.0, 1.0),
    )


def _publish_shrink_bounds(
        dll, cloth_handle, prepared_bounds, solver_mask,
        object_id, topology_generation, geometry_generation):
    if prepared_bounds is None:
        raise RuntimeError("shrink bounds owner is missing")
    shrink_min, shrink_max = prepared_bounds
    config = CType.GPUClothShrinkConfig()
    config.header.struct_size = sizeof(config)
    config.header.feature_id = CType.GPUCLOTH_FEATURE_SHRINK
    config.header.config_version = 1
    config.header.flags = 0
    config.solver_mask = int(solver_mask)
    config.reserved0 = 0
    config.object_id = int(object_id)
    config.topology_generation = int(topology_generation)
    config.geometry_generation = int(geometry_generation)
    config.shrink_min = float(shrink_min)
    config.shrink_max = float(shrink_max)
    config.reserved[:] = (0,)
    result = int(dll.GPUCloth_v3_cloth_set_shrink_config(
        cloth_handle, pointer(config)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed shrink bounds rejected with {result}")
    return result


def _validate_shrink_status(
        dll, cloth_handle, prepared_bounds,
        object_id, topology_generation, geometry_generation):
    status = CType.GPUClothV3ShrinkStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    result = int(dll.GPUCloth_v3_cloth_get_shrink_status(
        cloth_handle, pointer(status)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed shrink status rejected with {result}")
    shrink_min, shrink_max = prepared_bounds
    if (int(status.configured) != 1 or int(status.applied) != 1 or
            int(status.last_result) != CType.GPUCLOTH_ABI_OK or
            int(status.object_id) != int(object_id) or
            int(status.topology_generation) != int(topology_generation) or
            int(status.geometry_generation) != int(geometry_generation) or
            abs(float(status.shrink_min) - float(shrink_min)) > 1e-6 or
            abs(float(status.shrink_max) - float(shrink_max)) > 1e-6 or
            int(status.reserved0) != 0 or
            any(int(value) != 0 for value in status.reserved)):
        raise RuntimeError(
            "typed shrink status does not match accepted owner")
    return status


def _upload_object_collision_mask(
        dll, cloth_handle, settings_owner, simulation_obj):
    mask = binary_exclusion_mask(
        simulation_obj, settings_owner.GPUCloth.vgroup_objcol,
        "object collision")
    return apply_float_channel(
        dll, CType, cloth_handle, CType.GPUCLOTH_FEATURE_COLLISION_VERTEX_GROUP,
        CType.GPUCLOTH_VERTEX_OBJECT_COLLISION_MASK, 1, mask)


def _capture_self_collision_mask(settings_owner, simulation_obj):
    group_name = settings_owner.GPUCloth.vgroup_selfcol
    weights = vertex_group_weights(
        simulation_obj, group_name, "self collision")
    return None if weights is None else tuple(weights)


def _upload_self_collision_mask(dll, cloth_handle, prepared_mask):
    if prepared_mask is None:
        return CType.GPUCLOTH_ABI_OK
    return apply_float_channel(
        dll, CType, cloth_handle,
        CType.GPUCLOTH_FEATURE_SELF_COLLISION_VERTEX_GROUP,
        CType.GPUCLOTH_VERTEX_SELF_COLLISION_MASK, 1, prepared_mask)

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


_COLLIDER_MOTION_GROUP_MAX = 64


def _collider_canonical_motion_topology(
        source, vertex_count, evaluated_triangles, base_generation):
    """Return a persistent patch topology, or None for exact-only geometry.

    Bone weights are used only to choose coherent patches.  No Blender
    skinning formula is assumed: every frame fits a coarse rigid transform to
    the evaluated mesh and bounds the remaining deformation explicitly.
    """
    if getattr(source, "type", None) != 'MESH':
        return None
    try:
        mesh = source.data
        mesh.calc_loop_triangles()
        canonical_triangles = tuple(
            tuple(int(index) for index in triangle.vertices)
            for triangle in mesh.loop_triangles)
        if (len(mesh.vertices) != int(vertex_count) or
                canonical_triangles != tuple(evaluated_triangles)):
            return None
        canonical = _finite_float32_tuple(
            (coordinate for vertex in mesh.vertices for coordinate in vertex.co),
            f"collider {source.name_full!r} canonical vertices")

        armatures = []
        for modifier in getattr(source, "modifiers", ()):
            if (getattr(modifier, "type", None) == 'ARMATURE' and
                    getattr(modifier, "object", None) is not None):
                armatures.append(modifier.object)
        parent = getattr(source, "parent", None)
        if parent is not None and getattr(parent, "type", None) == 'ARMATURE':
            armatures.append(parent)
        unique_armatures = {
            int(armature.as_pointer()): armature for armature in armatures}
        bone_names = set()
        if len(unique_armatures) == 1:
            armature = next(iter(unique_armatures.values()))
            bone_names = {
                str(bone.name) for bone in armature.data.bones}
        group_names = {
            int(group.index): str(group.name)
            for group in getattr(source, "vertex_groups", ())}
        bone_group_indices = {
            index for index, name in group_names.items()
            if name in bone_names}

        triangle_keys = []
        for triangle in mesh.loop_triangles:
            score = {}
            if bone_group_indices:
                for vertex_index in triangle.vertices:
                    for entry in mesh.vertices[int(vertex_index)].groups:
                        group = int(entry.group)
                        weight = float(entry.weight)
                        if group in bone_group_indices and weight > 1.0e-8:
                            score[group] = score.get(group, 0.0) + weight
            if score:
                key = max(
                    score,
                    key=lambda group: (score[group], group_names[group]))
            else:
                key = -1
            triangle_keys.append(key)

        counts = {}
        for key in triangle_keys:
            counts[key] = counts.get(key, 0) + 1
        ordered_keys = sorted(
            counts, key=lambda key: (-counts[key], int(key)))
        if len(ordered_keys) > _COLLIDER_MOTION_GROUP_MAX:
            retained = set(ordered_keys[:_COLLIDER_MOTION_GROUP_MAX - 1])
            triangle_keys = [
                key if key in retained else -2 for key in triangle_keys]
        unique_keys = sorted(set(triangle_keys))
        key_to_group = {
            key: group for group, key in enumerate(unique_keys)}
        triangle_groups = tuple(
            key_to_group[key] for key in triangle_keys)

        canonical_np = np.asarray(canonical, dtype='<f4')
        groups_np = np.asarray(triangle_groups, dtype='<u4')
        hasher = hashlib.blake2b(digest_size=8, person=b"GPUCollCert")
        hasher.update(int(base_generation).to_bytes(8, "little", signed=False))
        hasher.update(canonical_np.tobytes(order='C'))
        hasher.update(groups_np.tobytes(order='C'))
        topology_generation = int.from_bytes(
            hasher.digest(), "little") or 1
        return {
            "canonical": tuple(float(value) for value in canonical_np),
            "triangle_groups": triangle_groups,
            "group_count": len(unique_keys),
            "topology_generation": topology_generation,
        }
    except (
            AttributeError, ReferenceError, RuntimeError, TypeError,
            ValueError, OverflowError):
        return None


def _fit_collider_motion_certificate(
        canonical_topology, triangles, current_positions, next_positions):
    """Fit fixed patch references and bound the complete linear interval."""
    if canonical_topology is None:
        return None
    try:
        canonical = np.asarray(
            canonical_topology["canonical"], dtype=np.float64).reshape(-1, 3)
        current = np.asarray(
            current_positions, dtype=np.float64).reshape(-1, 3)
        next_values = np.asarray(
            next_positions, dtype=np.float64).reshape(-1, 3)
        triangle_array = np.asarray(triangles, dtype=np.int64)
        groups = np.asarray(
            canonical_topology["triangle_groups"], dtype=np.int64)
        group_count = int(canonical_topology["group_count"])
        if (canonical.shape != current.shape or current.shape != next_values.shape or
                triangle_array.shape != (len(groups), 3) or
                group_count <= 0 or group_count > _COLLIDER_MOTION_GROUP_MAX):
            return None

        canonical_to_world = np.empty(
            (group_count, 2, 3, 4), dtype=np.float32)
        residuals = np.empty((group_count, 2), dtype=np.float32)
        fp32_epsilon = float(np.finfo(np.float32).eps)
        for group in range(group_count):
            triangle_mask = groups == group
            vertices = np.unique(triangle_array[triangle_mask].reshape(-1))
            if len(vertices) < 3:
                return None
            source_points = canonical[vertices]
            source_center = source_points.mean(axis=0)
            for endpoint, evaluated in enumerate((current, next_values)):
                target_points = evaluated[vertices]
                target_center = target_points.mean(axis=0)
                covariance = (
                    (source_points - source_center).T @
                    (target_points - target_center))
                left, _, right_t = np.linalg.svd(
                    covariance, full_matrices=True)
                rotation = right_t.T @ left.T
                if np.linalg.det(rotation) < 0.0:
                    right_t[-1, :] *= -1.0
                    rotation = right_t.T @ left.T
                translation = target_center - rotation @ source_center
                emitted_forward = np.concatenate(
                    (rotation, translation[:, None]), axis=1
                ).astype(np.float32)
                if not np.isfinite(emitted_forward).all():
                    return None
                # The device interpolates these exact FP32 endpoint proxies.
                # Bound each exact evaluated endpoint against the emitted
                # surface; convex interpolation then bounds the full interval.
                emitted64 = emitted_forward.astype(np.float64)
                predicted = (
                    source_points @ emitted64[:, :3].T + emitted64[:, 3])
                endpoint_error = float(np.linalg.norm(
                    target_points - predicted, axis=1).max())
                magnitude = max(
                    1.0,
                    float(np.abs(source_points).max()),
                    float(np.abs(target_points).max()),
                    float(np.abs(predicted).max()),
                )
                residual = endpoint_error + 128.0 * fp32_epsilon * (
                    magnitude + endpoint_error + 1.0)
                residual32 = np.nextafter(
                    np.float32(residual), np.float32(np.inf))
                if not np.isfinite(residual32) or residual32 < 0.0:
                    return None
                canonical_to_world[group, endpoint] = emitted_forward
                residuals[group, endpoint] = residual32
        return {
            "canonical_to_world": tuple(
                float(value) for value in canonical_to_world.reshape(-1)),
            "endpoint_residual": tuple(
                float(value) for value in residuals.reshape(-1)),
        }
    except (
            FloatingPointError, IndexError, KeyError, OverflowError,
            TypeError, ValueError, np.linalg.LinAlgError):
        return None


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


def _publish_cloth_collision_config(dll, cloth_handle, prepared_config):
    for feature_id in (
            CType.GPUCLOTH_FEATURE_STATIC_OBJECT_COLLISION,
            CType.GPUCLOTH_FEATURE_COLLISION_FRICTION_DAMPING,
            CType.GPUCLOTH_FEATURE_COLLISION_QUALITY_CLAMP,
            CType.GPUCLOTH_FEATURE_SELF_COLLISION,
            CType.GPUCLOTH_FEATURE_SELF_COLLISION_FRICTION):
        config = CType.GPUClothCollisionConfig.from_buffer_copy(
            bytes(prepared_config))
        config.header.feature_id = feature_id
        result = int(dll.GPUCloth_v3_cloth_configure(
            cloth_handle,
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
    exact_topology_generation = _topology_generation(
        occurrence["object_id"], vertex_count, triangles)
    canonical_topology = _collider_canonical_motion_topology(
        source, vertex_count, triangles, exact_topology_generation)
    topology_generation = (
        int(canonical_topology["topology_generation"])
        if canonical_topology is not None
        else exact_topology_generation)
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

    motion_certificate = _fit_collider_motion_certificate(
        canonical_topology, triangles, current_positions, next_positions)
    canonical_array = None
    triangle_group_array = None
    canonical_to_world_array = None
    endpoint_residual_array = None
    if motion_certificate is not None:
        canonical_values = canonical_topology["canonical"]
        triangle_group_values = canonical_topology["triangle_groups"]
        canonical_to_world_values = motion_certificate["canonical_to_world"]
        endpoint_residual_values = motion_certificate["endpoint_residual"]
        canonical_array = (c_float * len(canonical_values))(*canonical_values)
        triangle_group_array = (
            c_uint * len(triangle_group_values))(*triangle_group_values)
        canonical_to_world_array = (
            c_float * len(canonical_to_world_values))(
                *canonical_to_world_values)
        endpoint_residual_array = (
            c_float * len(endpoint_residual_values))(
                *endpoint_residual_values)

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
    if motion_certificate is not None:
        group_count = int(canonical_topology["group_count"])
        config.motion_group_count = group_count
        config.motion_certificate_flags = (
            CType.GPUCLOTH_COLLIDER_MOTION_CERTIFICATE_PRESENT)
        _set_buffer_view(
            config.canonical_positions,
            CType.GPUCLOTH_ELEMENT_FLOAT3,
            vertex_count,
            sizeof(c_float) * 3,
            addressof(canonical_array),
            topology_generation)
        _set_buffer_view(
            config.triangle_motion_groups,
            CType.GPUCLOTH_ELEMENT_UINT32,
            len(triangles),
            sizeof(c_uint),
            addressof(triangle_group_array),
            topology_generation)
        _set_buffer_view(
            config.motion_group_canonical_to_world,
            CType.GPUCLOTH_ELEMENT_FLOAT,
            group_count * 24,
            sizeof(c_float),
            addressof(canonical_to_world_array),
            snapshot_generation)
        _set_buffer_view(
            config.motion_group_endpoint_residual,
            CType.GPUCLOTH_ELEMENT_FLOAT,
            group_count * 2,
            sizeof(c_float),
            addressof(endpoint_residual_array),
            snapshot_generation)

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
        "canonical_positions": canonical_array,
        "triangle_motion_groups": triangle_group_array,
        "motion_group_canonical_to_world": canonical_to_world_array,
        "motion_group_endpoint_residual": endpoint_residual_array,
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


_EFFECTOR_SIGNATURE_FIELDS = (
    "field_type", "shape", "falloff", "z_direction", "strength",
    "flow", "noise", "size", "damping", "seed", "falloff_power",
    "use_min_distance", "distance_min", "use_max_distance",
    "distance_max", "use_absorption",
)


def _effector_payload_signature(
        occurrence, field, collection_id, cloth_inverse):
    source = occurrence["source_object"]
    settings_values = _validate_effector_field(field, source)
    matrix = _cloth_local_matrix(
        cloth_inverse, occurrence["matrix_world"],
        f"effector {source.name_full!r}")
    inverse = _matrix_inverse(
        matrix, f"effector {source.name_full!r} cloth-local transform")
    return (
        int(_collection_object_type(source)),
        int(occurrence["record_flags"]),
        int(occurrence["object_id"]),
        int(occurrence["instance_id"]),
        int(collection_id),
        tuple(settings_values[name] for name in _EFFECTOR_SIGNATURE_FIELDS),
        _matrix_signature(matrix, "effector cloth-local matrix"),
        _matrix_signature(inverse, "effector cloth-local inverse"),
    )


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
    _effector_publication_state["payload_build_count"] += 1
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


def _retarget_effector_snapshot_owner(owner, snapshot_generation):
    if (owner["collection_kind"] !=
            CType.GPUCLOTH_COLLECTION_EFFECTOR):
        raise RuntimeError("cached effector owner has the wrong kind")
    generation = int(snapshot_generation)
    config = owner["config"]
    config.snapshot_generation = generation
    if owner["records"]:
        config.records.generation = generation
    for index, payload in enumerate(owner["payloads"]):
        payload["config"].generation = generation
        payload["config"].object_matrix.generation = generation
        payload["config"].inverse_matrix.generation = generation
        payload["record"].geometry_generation = generation
        owner["records"][index].geometry_generation = generation
    owner["snapshot_generation"] = generation
    _effector_publication_state["snapshot_reuse_count"] += 1
    return owner


def _collection_snapshot_owner(
        cloth_obj, owner_obj, selection, collection_kind,
        snapshot_generation, payloads):
    if collection_kind == CType.GPUCLOTH_COLLECTION_EFFECTOR:
        _effector_publication_state["snapshot_build_count"] += 1
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
    config.cloth_id = _blender_session_uid(owner_obj, "cloth simulation object")
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
        collider_history, effector_states=None, owner_objects=None):
    if (effector_states is not None and
            len(effector_states) != len(cloth_objects)):
        raise RuntimeError("cached effector owners are not aligned")
    if owner_objects is None:
        owner_objects = cloth_objects
    if len(owner_objects) != len(cloth_objects):
        raise RuntimeError("cloth and simulation owners are not aligned")
    cloth_ids = {
        _blender_session_uid(_original_object(obj), "cloth object")
        for obj in tuple(cloth_objects) + tuple(owner_objects)}
    prepared = []
    for cloth_index, (cloth_obj, owner_obj) in enumerate(
            zip(cloth_objects, owner_objects)):
        cloth_owner_id = _blender_session_uid(
            owner_obj, "cloth simulation object")
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
        effector_sources = []
        effector_signatures = []
        for occurrence in _depsgraph_occurrences(
                depsgraph, effector_selection, "effector"):
            source = occurrence["source_object"]
            field = getattr(
                occurrence["evaluated_object"], "field",
                getattr(source, "field", None))
            field_type = getattr(field, "type", "NONE")
            if field_type == 'NONE':
                continue
            collection_id = (
                effector_selection["collection_id"]
                if effector_selection is not None else 0)
            effector_sources.append((occurrence, field))
            effector_signatures.append(_effector_payload_signature(
                occurrence, field, collection_id, cloth_inverse))

        semantic_signature = tuple(effector_signatures)
        cached_state = (
            effector_states[cloth_index]
            if effector_states is not None else None)
        cached_owner = (
            cached_state.get("snapshot_owner")
            if cached_state is not None else None)
        if (
                cached_owner is not None and
                tuple(cached_state["snapshot_signature"]) ==
                    semantic_signature):
            effector_owner = _retarget_effector_snapshot_owner(
                cached_owner, snapshot_generation)
        else:
            effector_payloads = [
                _capture_effector_payload(
                    occurrence, field,
                    effector_selection["collection_id"]
                    if effector_selection is not None else 0,
                    snapshot_generation, cloth_inverse)
                for occurrence, field in effector_sources]
            effector_owner = _collection_snapshot_owner(
                cloth_obj, owner_obj, effector_selection,
                CType.GPUCLOTH_COLLECTION_EFFECTOR,
                snapshot_generation, effector_payloads)
        effector_owner["semantic_signature"] = semantic_signature

        effector_collection_id = (
            effector_selection["collection_id"]
            if effector_selection is not None else 0)
        if cached_state is not None:
            current_weights = _make_effector_weights(
                settings.effector_weights)
            if (
                    int(cached_state["collection_id"]) !=
                        int(effector_collection_id) or
                    tuple(cached_state["weights"]) !=
                        tuple(current_weights)):
                raise RuntimeError(
                    "effector weights or collection changed; reprepare is "
                    "required")
            effector_weights = cached_state["weights_owner"]
        else:
            effector_weights = _prepare_effector_weights(
                settings, effector_collection_id)

        prepared.append({
            "collision": _collection_snapshot_owner(
                cloth_obj, owner_obj, collision_selection,
                CType.GPUCLOTH_COLLECTION_COLLISION,
                snapshot_generation, collision_payloads),
            "effector": effector_owner,
            "effector_weights": effector_weights,
        })
    return prepared


def _prepare_effector_weights(settings, collection_id):
    _effector_publication_state["weight_owner_build_count"] += 1
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


def _configure_effector_weights(dll, cloth_handle, owner):
    config = owner["config"]
    result = int(dll.GPUCloth_v3_cloth_configure(
        cloth_handle,
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


def _query_collection_snapshot(dll, cloth_handle, owner, transaction_id):
    expected_records = owner["records"]
    query = CType.GPUClothCollectionQuery()
    query.struct_size = sizeof(query)
    query.query_version = 1
    query.collection_kind = owner["collection_kind"]
    probe_result = int(dll.GPUCloth_v3_cloth_query_collection(
        cloth_handle, pointer(query)))
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
        result = int(dll.GPUCloth_v3_cloth_query_collection(
            cloth_handle, pointer(query)))
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
        dll, cloth_handles, prepared_collections, prepared_pins,
        prepared_dynamic_meshes, source_generation,
        verify_committed_state=False):
    global _collider_history
    if not (
            len(cloth_handles) == len(prepared_collections) ==
            len(prepared_pins) == len(prepared_dynamic_meshes)):
        raise RuntimeError("staged frame input owners are not aligned")
    next_collider_history = _next_collider_history(
        prepared_collections)
    transaction = CType.GPUClothCollectionTransactionConfig()
    transaction.struct_size = sizeof(transaction)
    transaction.transaction_version = 1
    transaction.source_generation = int(source_generation)
    transaction_id = c_uint64()
    result = int(dll.GPUCloth_v3_collection_transaction_begin(
        g_runtime_handle, pointer(transaction), pointer(transaction_id)))
    if result != CType.GPUCLOTH_ABI_OK or not transaction_id.value:
        raise RuntimeError(
            f"collection transaction begin rejected with {result}")

    committed = False
    try:
        for cloth_handle, owners, pin_owner, dynamic_owner in zip(
                cloth_handles, prepared_collections, prepared_pins,
                prepared_dynamic_meshes):
            for key in ("collision", "effector"):
                owner = owners[key]
                result = int(dll.GPUCloth_v3_collection_stage_snapshot(
                    transaction_id.value, cloth_handle,
                    pointer(owner["config"])))
                if result != CType.GPUCLOTH_ABI_OK:
                    raise RuntimeError(
                        f"{key} collection stage rejected with {result}")
            result = int(dll.GPUCloth_v3_collection_stage_pin_snapshot(
                transaction_id.value, cloth_handle,
                pointer(pin_owner["config"])))
            if result != CType.GPUCLOTH_ABI_OK:
                raise RuntimeError(
                    f"pin snapshot stage rejected with {result}")
            if dynamic_owner is not None:
                result = int(dll.GPUCloth_v3_collection_stage_mesh_state(
                    transaction_id.value, cloth_handle,
                    pointer(dynamic_owner["config"])))
                if result != CType.GPUCLOTH_ABI_OK:
                    raise RuntimeError(
                        f"dynamic mesh stage rejected with {result}")
        result = int(dll.GPUCloth_v3_collection_transaction_commit(
            g_runtime_handle, transaction_id.value))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"collection transaction commit rejected with {result}")
        committed = True
        # Native commit publishes both history and its source generation.
        _collider_history = next_collider_history
        _input_generation["value"] = int(source_generation)
    finally:
        if not committed:
            abort_result = int(dll.GPUCloth_v3_collection_transaction_abort(
                g_runtime_handle, transaction_id.value))
            if abort_result != CType.GPUCLOTH_ABI_OK:
                raise RuntimeError(
                    f"collection transaction abort rejected with "
                    f"{abort_result}")

    if not verify_committed_state:
        return None

    snapshots = []
    for cloth_handle, owners in zip(cloth_handles, prepared_collections):
        _effector_publication_state[
            "verification_round_trip_count"] += 1
        status = CType.GPUClothCollectionStatus()
        status.struct_size = sizeof(status)
        status.status_version = 1
        result = int(dll.GPUCloth_v3_cloth_get_collection_status(
            cloth_handle, pointer(status)))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"collection status rejected with {result}")
        collision = _query_collection_snapshot(
            dll, cloth_handle, owners["collision"], transaction_id.value)
        effector = _query_collection_snapshot(
            dll, cloth_handle, owners["effector"], transaction_id.value)
        _effector_publication_state[
            "verification_round_trip_count"] += (
                2 + int(bool(owners["collision"]["records"])) +
                int(bool(owners["effector"]["records"])))
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
            len(g_cloth_handles) == len(_pin_snapshot_states) ==
            len(_dynamic_mesh_states)):
        raise RuntimeError("frame input owners are not aligned")
    generation = max(
        int(_input_generation["value"]) + 1,
        int(_runtime_frame_generation) + 1)

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
            _collider_history, _effector_weight_states,
            owner_objects=g_simulationOBJs)
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
    # v3 transaction source generation must match runtime's published frame.
    _runtime_update(context.scene, generation=generation)
    committed_collections = _commit_frame_inputs(
        g_dll, g_cloth_handles, prepared_collections, prepared_pin_owners,
        prepared_dynamic_owners, generation)
    if committed_collections is not None:
        _collection_snapshots.clear()
        _collection_snapshots.extend(committed_collections)

    for state, owners in zip(
            _effector_weight_states, prepared_collections):
        state["snapshot_owner"] = owners["effector"]
        state["snapshot_signature"] = tuple(
            owners["effector"]["semantic_signature"])

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
    if not _runtime_handle_value():
        raise RuntimeError("v3 cache owner requires a live runtime")
    if not _cache_handle_value():
        if g_dll is None:
            raise RuntimeError("v3 cache owner is not live")
        _configure_cache_features(g_dll, scene)
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
    result = int(g_dll.GPUCloth_v3_cache_update_status(
        g_runtime_handle, _cache_handle_owner(), pointer(update)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            f"typed cache status update {operation} rejected with {result}")
    _cache_source_state['generation'] = generation
    return generation


def _query_cache_status():
    if not _runtime_handle_value() or not _cache_handle_value():
        raise RuntimeError("v3 cache owner is not live")
    status = CType.GPUClothCacheStatus()
    status.struct_size = sizeof(status)
    status.status_version = 1
    result = int(g_dll.GPUCloth_v3_cache_get_status(
        g_runtime_handle, _cache_handle_owner(), pointer(status)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"typed cache status query rejected with {result}")
    return status


def _sync_cache_status(scene):
    status = _query_cache_status()
    if int(status.source_generation):
        _cache_source_state['generation'] = int(status.source_generation)
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


def _runtime_create():
    """Create the native v3 owner exactly at simulation preparation."""
    global g_runtime_handle
    if g_dll is None:
        raise RuntimeError("GPUCloth DLL is not loaded")
    if _teardown_failure:
        raise RuntimeError("v3 runtime teardown recovery is required")
    if _runtime_handle_value():
        return g_runtime_handle

    config = CType.GPUClothV3RuntimeConfig()
    config.struct_size = sizeof(config)
    config.config_version = 1
    config.runtime_flags = CType.GPUCLOTH_V3_RUNTIME_NONE
    config.device_ordinal = _V3_RUNTIME_DEVICE_ORDINAL
    config.application_id = _V3_RUNTIME_APPLICATION_ID
    config.reserved[:] = (0, 0, 0, 0, 0)
    out_runtime = CType.GPUClothV3RuntimeHandle(0)
    result = int(g_dll.GPUCloth_v3_runtime_create(
        pointer(config), pointer(out_runtime)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"v3 runtime create rejected with {result}")
    if not out_runtime.value:
        raise RuntimeError("v3 runtime create returned a null handle")
    g_runtime_handle = out_runtime
    return g_runtime_handle


def _scene_fps_ratio(scene):
    """Convert Blender's fps/fps_base to a bounded exact v3 fraction."""
    fps = float(scene.render.fps)
    fps_base = float(scene.render.fps_base)
    if (not math.isfinite(fps) or not math.isfinite(fps_base)
            or fps <= 0.0 or fps_base <= 0.0):
        raise RuntimeError("Blender fps/fps_base must be finite and positive")

    def rationalize(value, label):
        tolerance = _V3_FPS_RATE_TOLERANCE * max(1.0, abs(value))
        for denominator in range(1, _V3_FPS_INPUT_DENOMINATOR_MAX + 1):
            numerator = int(round(value * denominator))
            if numerator <= 0:
                continue
            candidate = Fraction(numerator, denominator)
            if abs(float(candidate) - value) <= tolerance:
                return candidate
        raise RuntimeError(f"Blender {label} is not representable by v3 frame ABI")

    exact_rate = Fraction.from_float(fps) / Fraction.from_float(fps_base)
    candidate = rationalize(fps, "fps") / rationalize(fps_base, "fps_base")
    numerator = int(candidate.numerator)
    denominator = int(candidate.denominator)
    if (numerator <= 0 or numerator > _V3_FPS_NUMERATOR_MAX
            or denominator <= 0 or denominator > _V3_FPS_DENOMINATOR_MAX
            or abs(float(candidate) - float(exact_rate)) >
            _V3_FPS_RATE_TOLERANCE * max(1.0, abs(float(exact_rate)))):
        raise RuntimeError("Blender fps/fps_base is not representable by v3 frame ABI")
    return numerator, denominator


def _runtime_update(scene, generation=None):
    """Publish one Blender frame; reject duplicate/out-of-order generations."""
    global _runtime_frame_generation
    if g_dll is None or not _runtime_handle_value():
        raise RuntimeError("v3 runtime owner is not live")
    if generation is None:
        generation = _runtime_frame_generation + 1
    generation = int(generation)
    if generation <= _runtime_frame_generation:
        raise RuntimeError(
            f"v3 frame generation is not increasing: {generation} <= "
            f"{_runtime_frame_generation}")

    config = CType.GPUClothV3FrameConfig()
    config.struct_size = sizeof(config)
    config.config_version = 1
    config.frame = int(scene.frame_current)
    config.frame_flags = 0
    config.frame_generation = generation
    fps_numerator, fps_denominator = _scene_fps_ratio(scene)
    config.fps_numerator = fps_numerator
    config.fps_denominator = fps_denominator
    config.subframe = float(scene.frame_subframe)
    helper = scene.gpu_cloth_helper
    config.gravity[:] = (
        float(helper.gravity_x),
        float(helper.gravity_y),
        float(helper.gravity_z),
    )
    config.reserved[:] = (0, 0)
    result = int(g_dll.GPUCloth_v3_runtime_update(
        g_runtime_handle, pointer(config)))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"v3 runtime update rejected with {result}")
    _runtime_frame_generation = generation
    return config


def _ensure_runtime_for_scene(scene, generation=None):
    """Create and publish the first scene before any other native mutation."""
    _runtime_create()
    if generation is None:
        generation = _runtime_frame_generation + 1
    return _runtime_update(scene, generation=generation)


def _destroy_runtime(shutdown_runtime=False):
    """Destroy the v3 owner; retain the handle on every failure."""
    global g_runtime_handle, g_cache_handle, g_cache_owner
    if not _runtime_handle_value():
        return True
    if g_dll is None:
        raise RuntimeError("v3 runtime owner retained without DLL")
    if shutdown_runtime and _cache_handle_value():
        cache_result = int(g_dll.GPUCloth_v3_cache_destroy(
            g_runtime_handle, _cache_handle_owner()))
        if cache_result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"v3 cache destroy rejected with {cache_result}")
    try:
        result = int(g_dll.GPUCloth_v3_runtime_destroy(g_runtime_handle))
    except Exception:
        raise
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"v3 runtime destroy rejected with {result}")
    g_runtime_handle = CType.GPUClothV3RuntimeHandle(0)
    g_cache_handle = CType.GPUClothV3CacheHandle(0)
    g_cache_owner = None
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
    if g_dll is None or not _runtime_handle_value() or not _cache_handle_value():
        return False
    if not _cache_has_frame(scene, frame):
        return False
    updated = False
    for i, cloth_obj in enumerate(g_clothOBJs):
        cached = _cached_frame_positions(
            scene, frame, cloth_obj, None)
        if cached is not None:
            flat = np.frombuffer(cached, dtype=np.float32)
            cloth_obj.data.vertices.foreach_set("co", flat)
            cloth_obj.data.update()
            cloth_obj.data.update_tag()
            updated = True
    if updated:
        depsgraph.update()
    return updated


def _cached_frame_positions(scene, frame, cloth_obj, cache_dir):
    """Read one cached mesh without requiring a live solver owner."""
    if g_dll is None or not _runtime_handle_value() or not _cache_handle_value():
        return None
    nV = len(cloth_obj.data.vertices)
    pos = (c_float * (nV * 3))()
    request = _cache_frame_request(
        scene, frame, nV, CType.GPUCLOTH_V3_CACHE_FRAME_NONE)
    loaded = int(g_dll.GPUCloth_v3_cache_prefetch_frame(
        g_runtime_handle, _cache_handle_owner(), pointer(request)))
    if loaded != CType.GPUCLOTH_ABI_OK:
        return None
    request.frame_flags = CType.GPUCLOTH_V3_CACHE_FRAME_READ
    request.positions.struct_size = sizeof(CType.GPUClothBufferView)
    request.positions.element_type = CType.GPUCLOTH_ELEMENT_FLOAT3
    request.positions.element_count = nV
    request.positions.stride_bytes = sizeof(c_float) * 3
    request.positions.data_address = addressof(pos)
    request.positions.generation = int(request.frame_generation)
    loaded = int(g_dll.GPUCloth_v3_cache_read_frame(
        g_runtime_handle, _cache_handle_owner(), pointer(request)))
    if loaded != CType.GPUCLOTH_ABI_OK:
        return None
    return pos


def _cache_frame_request(scene, frame, vertex_count, flags):
    if not _cache_handle_value() or not _runtime_handle_value():
        raise RuntimeError("v3 cache owner is not live")
    request = CType.GPUClothV3CacheFrameConfig()
    request.struct_size = sizeof(request)
    request.config_version = 1
    request.frame_flags = int(flags)
    request.frame = int(frame)
    request.vertex_count = int(vertex_count)
    request.frame_generation = int(_runtime_frame_generation)
    request.cache_id = int(_cache_identity_value())
    request.reserved[:] = (0, 0, 0)
    return request


def _cache_identity_value():
    """Return the configure-time copied id; never dispatch a hot-path query."""
    owner = g_cache_owner
    if owner is None or not _cache_handle_value() or not _runtime_handle_value():
        return 0
    return int(owner["cache_id"])


def _cache_has_frame(scene, frame):
    cloth_count = len(g_simulationOBJs[0].data.vertices) if g_simulationOBJs else 0
    if cloth_count <= 0:
        return False
    request = _cache_frame_request(
        scene, frame, cloth_count, CType.GPUCLOTH_V3_CACHE_FRAME_NONE)
    present = c_uint(0)
    result = int(g_dll.GPUCloth_v3_cache_has_frame(
        g_runtime_handle, _cache_handle_owner(), pointer(request),
        pointer(present)))
    return result == CType.GPUCLOTH_ABI_OK and bool(present.value)


def _cache_buffer_address(values):
    ctypes_view = getattr(values, "ctypes", None)
    if ctypes_view is not None and hasattr(ctypes_view, "data"):
        return int(ctypes_view.data)
    return int(addressof(values))


def _cache_prefetch_frame(scene, frame, vertex_count):
    request = _cache_frame_request(
        scene, frame, vertex_count, CType.GPUCLOTH_V3_CACHE_FRAME_NONE)
    return int(g_dll.GPUCloth_v3_cache_prefetch_frame(
        g_runtime_handle, _cache_handle_owner(), pointer(request)))


def _cache_write_frame(scene, frame, values, vertex_count):
    request = _cache_frame_request(
        scene, frame, vertex_count, CType.GPUCLOTH_V3_CACHE_FRAME_WRITE)
    request.positions.struct_size = sizeof(CType.GPUClothBufferView)
    request.positions.element_type = CType.GPUCLOTH_ELEMENT_FLOAT3
    request.positions.element_count = int(vertex_count)
    request.positions.stride_bytes = sizeof(c_float) * 3
    request.positions.data_address = _cache_buffer_address(values)
    request.positions.generation = int(request.frame_generation)
    return int(g_dll.GPUCloth_v3_cache_write_frame_async(
        g_runtime_handle, _cache_handle_owner(), pointer(request)))


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
    global g_dll, g_cloth_handles, _cloth_input_owners
    global _readback_owners, _runtime_frame_generation, g_cache_handle
    global g_cache_owner
    global g_clothOBJs, g_simulationOBJs, g_clothCollisionOBJs, g_proxy_handles
    global _teardown_failure, _collider_history

    if g_dll is None and _runtime_owners_retained():
        print(
            "free_gpu_memory: native DLL unavailable while runtime or "
            "owners are retained")
        _teardown_failure = True
        return False

    if g_dll is not None:
        try:
            _destroy_runtime(shutdown_runtime=shutdown_runtime)
        except Exception as e:
            print(f"free_gpu_memory: v3 runtime teardown failed: {e}")
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

    g_cloth_handles      = []
    _cloth_input_owners  = []
    _readback_owners     = []
    g_clothOBJs          = []
    g_simulationOBJs     = []
    g_clothCollisionOBJs = []
    g_proxy_handles      = []
    g_cache_handle       = CType.GPUClothV3CacheHandle(0)
    g_cache_owner        = None
    _collision_keepalive.clear()
    _solver_diagnostics.clear()
    _pin_snapshot_states.clear()
    _dynamic_mesh_states.clear()
    _collection_snapshots.clear()
    _effector_weight_states.clear()
    _reset_effector_publication_metrics()
    _collider_history = {}
    _drape_status_by_uid.clear()
    _initial_positions.clear()
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
                f"Ожидаемый путь: {os.path.join(lib_dir, 'GPUCloth.dll')}. "
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

            _bind_gpucloth_v3_exports(g_dll)
            _validate_product_abi(g_dll)
            _validate_descriptor_layout(g_dll)

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

    @_guard_prepare_teardown
    def execute(self, context):
        global g_dll, g_cloth_handles
        global _cloth_input_owners, _readback_owners
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

        try:
            _reject_unsupported_v3_owners(context.scene, active_cloth)
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        helper = context.scene.gpu_cloth_helper
        external_playback = bool(
            getattr(helper, "use_external_cache", False) and
            getattr(helper, "playback_mode", False))
        if external_playback:
            try:
                _validate_external_cache_playback_source(context.scene)
            except (OSError, RuntimeError, ValueError) as exc:
                self.report({'ERROR'}, f"External cache preflight failed: {exc}")
                return {'CANCELLED'}

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

        # One generation owns the complete initial publication.  Keep the
        # runtime generation monotonic across destroy/recreate so a new
        # runtime cannot publish stale pin, collection, or mesh state.
        initial_generation = int(_runtime_frame_generation) + 1
        if initial_generation <= 0:
            raise RuntimeError("v3 initial generation overflowed")

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
            prepared_shrink_bounds = []
            prepared_self_collision_masks = []
            prepared_topology_generations = []
            prepared_dynamic_meshes = []
            prepared_constraint_networks = []
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
                prepared_shrink_bounds.append(
                    _capture_shrink_bounds(cloth_obj))
                prepared_self_collision_masks.append(
                    _capture_self_collision_mask(
                        cloth_obj, simulation_obj))
                prepared_pin_snapshots.append(_capture_pin_snapshot(
                    cloth_obj, simulation_obj, depsgraph,
                    topology_generation, initial_generation))
                prepared_dynamic_meshes.append(
                    _capture_dynamic_mesh_snapshot(
                        cloth_obj, simulation_obj, depsgraph,
                        topology_generation, initial_generation))
                solver_mask = {
                    "PD": CType.GPUCLOTH_SOLVER_PD,
                    "Mil2": CType.GPUCLOTH_SOLVER_MIL2,
                }.get(str(cloth_obj.GPUCloth.solver_type))
                if solver_mask is None:
                    raise RuntimeError(
                        f"unsupported v3 constraint-network solver: "
                        f"{cloth_obj.GPUCloth.solver_type}")
                prepared_constraint_networks.append(
                    _capture_constraint_network(
                        cloth_obj, simulation_obj, topology_generation, 1,
                        solver_mask))
                prepared_collision_configs.append(
                    _capture_cloth_collision_config(cloth_obj.GPUCloth))
            prepared_pin_owners = [
                prepare_pin_snapshot(CType, snapshot)
                for snapshot in prepared_pin_snapshots]
            prepared_dynamic_owners = [
                _prepare_dynamic_mesh_state(snapshot, None)
                for snapshot in prepared_dynamic_meshes]
            prepared_collections = _prepare_collection_snapshots(
                context, depsgraph, active_cloth, initial_generation, {},
                owner_objects=simulation_objects)
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

        # The first native mutation is owned by the v3 runtime update.  Keep
        # all Blender extraction above this boundary.
        try:
            _ensure_runtime_for_scene(
                context.scene, generation=initial_generation)
            if _cache_requested_for_prepare(context.scene):
                _configure_cache_features(g_dll, context.scene)
                # Recreated disk/external owners restore persisted status and
                # its source-generation baseline before handlers/playback.
                _sync_cache_status(context.scene)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, f"v3 runtime setup failed: {exc}")
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        if external_playback:
            # External playback owns only the v3 runtime/cache and Blender
            # mesh targets.  It must not allocate a cloth/solver owner.
            bpy.ops.object.mode_set(mode=mode)
            context.scene.gpu_cloth_springs_built = False
            _store_initial_positions()
            _bake_range['start'] = int(helper.bake_start)
            _bake_range['end'] = int(helper.bake_end)
            _simulation_frame_state['last_solved'] = max(
                1, int(helper.bake_start) - 1)
            return {'FINISHED'}

        try:
            for index, (cloth_obj, simulation_obj) in enumerate(zip(
                    g_clothOBJs, g_simulationOBJs)):
                backend = (
                    CType.GPUCLOTH_V3_BACKEND_FAST
                    if cloth_obj.GPUCloth.solver_type == "PD"
                    else CType.GPUCLOTH_V3_BACKEND_ACCURACY)
                owner = _create_v3_cloth_owner(
                    g_dll, cloth_obj, simulation_obj,
                    prepared_input_meshes[index],
                    prepared_topology_generations[index], backend, 1)
                g_cloth_handles.append(owner["handle"])
                _cloth_input_owners.append(owner)
                _readback_owners.append(owner)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, f"v3 cloth create failed: {exc}")
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        if not (len(g_clothOBJs) == len(g_simulationOBJs) ==
                len(g_cloth_handles) == len(_cloth_input_owners) ==
                len(_readback_owners)):
            self.report({'ERROR'}, "v3 cloth owners are not aligned")
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        # 7. Live simulation owns the native scene and solver state.
        for i in range(len(g_clothOBJs)):
            cloth_handle = g_cloth_handles[i]
            try:
                _configure_simulation_features(
                    g_dll, cloth_handle, context.scene,
                    g_clothOBJs[i].GPUCloth,
                    _cloth_input_owners[i]["object_id"],
                    _cloth_input_owners[i]["topology_generation"],
                    _cloth_input_owners[i]["geometry_generation"])
                _publish_material_features(
                    g_dll, cloth_handle, prepared_material_features[i])
                _publish_internal_springs_config(
                    g_dll, cloth_handle, prepared_internal_configs[i])
                _publish_pressure_features(
                    g_dll, cloth_handle, prepared_pressure_features[i])
                _publish_constraint_network(
                    g_dll, cloth_handle, prepared_constraint_networks[i])
                _publish_cloth_collision_config(
                    g_dll, cloth_handle, prepared_collision_configs[i])
                _configure_solver_diagnostics(g_dll, cloth_handle)
            except (OSError, RuntimeError) as exc:
                self.report({'ERROR'}, f"Simulation config failed: {exc}")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}
            # v3 snapshots operator-affecting typed inputs at build.  Keep
            # anisotropy, stiffness, rest shape, and shrink pre-build;
            # post-build uploads below are limited to state the ABI permits.
            try:
                _publish_material_features(
                    g_dll, cloth_handle, prepared_material_features[i],
                    publish_anisotropy=True)
                _upload_stiffness_channels(
                    g_dll, cloth_handle, prepared_stiffness_channels[i])
                _publish_shrink_bounds(
                    g_dll, cloth_handle, prepared_shrink_bounds[i],
                    CType.GPUCLOTH_SOLVER_PD
                    if g_clothOBJs[i].GPUCloth.solver_type == "PD"
                    else CType.GPUCLOTH_SOLVER_MIL2,
                    _cloth_input_owners[i]["object_id"],
                    _cloth_input_owners[i]["topology_generation"],
                    _cloth_input_owners[i]["geometry_generation"])
                _upload_rest_shape_key(
                    g_dll, cloth_handle, prepared_rest_shapes[i],
                    _cloth_input_owners[i]["object_id"],
                    _cloth_input_owners[i]["topology_generation"],
                    _cloth_input_owners[i]["geometry_generation"])
                _upload_shrink_weights(
                    g_dll, cloth_handle, g_clothOBJs[i], g_simulationOBJs[i])
            except (OSError, RuntimeError, VertexChannelError) as exc:
                self.report({'ERROR'}, f"Pre-build cloth data failed: {exc}")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}
            try:
                result = int(g_dll.GPUCloth_v3_cloth_build(cloth_handle))
                if result != CType.GPUCLOTH_ABI_OK:
                    raise RuntimeError(
                        f"v3 cloth build rejected with {result}")
                _validate_constraint_network_status(
                    g_dll, cloth_handle, prepared_constraint_networks[i])
                _validate_shrink_status(
                    g_dll, cloth_handle, prepared_shrink_bounds[i],
                    _cloth_input_owners[i]["object_id"],
                    _cloth_input_owners[i]["topology_generation"],
                    _cloth_input_owners[i]["geometry_generation"])
            except (OSError, RuntimeError) as exc:
                self.report({'ERROR'}, f"v3 cloth build failed: {exc}")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

            try:
                _upload_sewing(
                    g_dll, cloth_handle, g_clothOBJs[i], g_simulationOBJs[i])
                _upload_pressure_weights(
                    g_dll, cloth_handle, g_clothOBJs[i], g_simulationOBJs[i])
                _upload_object_collision_mask(
                    g_dll, cloth_handle, g_clothOBJs[i], g_simulationOBJs[i])
                _upload_self_collision_mask(
                    g_dll, cloth_handle,
                    prepared_self_collision_masks[i])
            except (OSError, RuntimeError, VertexChannelError) as exc:
                self.report({'ERROR'}, f"Cloth data upload failed: {exc}")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

        try:
            for cloth_handle, owners in zip(
                    g_cloth_handles, prepared_collections):
                _configure_effector_weights(
                    g_dll, cloth_handle, owners["effector_weights"])
            _collection_snapshots.extend(_commit_frame_inputs(
                g_dll, g_cloth_handles, prepared_collections,
                prepared_pin_owners, prepared_dynamic_owners,
                initial_generation,
                verify_committed_state=True))
            for index, cloth_handle in enumerate(g_cloth_handles):
                _validate_native_preparation(
                    g_dll, cloth_handle,
                    prepared_topology_generations[index], initial_generation)
            _effector_weight_states.extend({
                "collection_id": int(
                    owners["effector_weights"]["collection_id"]),
                "weights": tuple(
                    owners["effector_weights"]["weights"]),
                "weights_owner": owners["effector_weights"],
                "snapshot_owner": owners["effector"],
                "snapshot_signature": tuple(
                    owners["effector"]["semantic_signature"]),
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

        # 8. Create one typed v3 proxy owner per render/simulation binding.
        g_proxy_handles.clear()
        for index, cloth_obj in enumerate(g_clothOBJs):
            s = cloth_obj.GPUCloth
            if s.use_proxy and s.proxy_object is not None:
                proxy_owner = _create_v3_proxy_owner(
                    g_dll, _cloth_input_owners[index], cloth_obj,
                    g_simulationOBJs[index], s, _readback_owners[index])
                g_proxy_handles.append(proxy_owner)
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
        return {'FINISHED'}


# ===========================================================================
#  Оператор: обновление симуляции (один кадр)
# ===========================================================================

class GPUCloth_UpdateSimulation(bpy.types.Operator):
    """Просчитать один кадр симуляции и обновить меш в Blender"""
    bl_idname = "gpucloth.update_simulation"
    bl_label  = "Update GPUCloth Simulation"

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def _readback(self, index):
        if index < 0 or index >= len(_readback_owners):
            raise RuntimeError("v3 readback owner does not exist")
        owner = _readback_owners[index]
        result = int(g_dll.GPUCloth_v3_cloth_readback(
            owner["handle"], pointer(owner["readback"])))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(f"v3 cloth readback rejected with {result}")
        if (int(owner["readback"].frame_generation) !=
                int(_runtime_frame_generation)):
            raise RuntimeError("v3 readback generation differs from runtime")
        return owner["readback_positions"]

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
        global g_dll
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
        frame     = context.scene.frame_current

        # ── РЕЖИМ ВОСПРОИЗВЕДЕНИЯ из кэша ───────────────────────────────────
        #
        #   v3 cache prefetch/read owns the frame and returns a checked copy.
        #
        if scene_s.playback_mode and scene_s.is_baked:
            _load_cached_frame(
                context.scene, context.evaluated_depsgraph_get(), frame)
            # Prefetch следующего кадра пока пользователь смотрит текущий
            for cloth_obj in g_clothOBJs:
                _cache_prefetch_frame(
                    context.scene, frame + 1,
                    len(cloth_obj.data.vertices))
            return {'FINISHED'}

        last_solved = _simulation_frame_state['last_solved']
        if last_solved is not None and frame <= last_solved:
            _load_cached_frame(
                context.scene, context.evaluated_depsgraph_get(), frame)
            return {'FINISHED'}

        # ── РЕЖИМ ЖИВОЙ СИМУЛЯЦИИ ────────────────────────────────────────────
        #
        #   GPUCloth_v3_cloth_step(handle) [GPU]
        #   GPUCloth_v3_cloth_readback(handle) [D2H]
        #   foreach_set() [Blender ~<1мс]
        #   v3 cache write copies payload before return.
        #

        if not (len(g_clothOBJs) == len(g_simulationOBJs)
                == len(g_cloth_handles) == len(_readback_owners)):
            self.report({'ERROR'},
                f"Несоответствие размеров: clothOBJs={len(g_clothOBJs)}, "
                f"simulationOBJs={len(g_simulationOBJs)}, "
                f"cloth_handles={len(g_cloth_handles)}, "
                f"readback={len(_readback_owners)}")
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

                cloth_handle = g_cloth_handles[i]
                previous_status = CType.GPUClothV3ClothStatus()
                previous_status.struct_size = sizeof(previous_status)
                previous_status.status_version = 1
                result = int(g_dll.GPUCloth_v3_cloth_get_status(
                    cloth_handle, pointer(previous_status)))
                if result != CType.GPUCLOTH_ABI_OK:
                    raise RuntimeError(
                        f"v3 cloth status rejected with {result}")

                # 1. GPU simulation одного кадра.
                result = int(g_dll.GPUCloth_v3_cloth_step(cloth_handle))
                if result != CType.GPUCLOTH_ABI_OK:
                    _solver_diagnostic_snapshot(i)
                    self.report(
                        {'ERROR'},
                        f"v3 cloth step rejected {cloth_obj.name_full}: "
                        f"{result}")
                    bpy.ops.screen.animation_cancel()
                    return {'CANCELLED'}
                status = CType.GPUClothV3ClothStatus()
                status.struct_size = sizeof(status)
                status.status_version = 1
                result = int(g_dll.GPUCloth_v3_cloth_get_status(
                    cloth_handle, pointer(status)))
                if (result != CType.GPUCLOTH_ABI_OK or
                        int(status.solve_count) <=
                        int(previous_status.solve_count) or
                        int(status.accepted_frame_generation) !=
                        int(_runtime_frame_generation) or
                        int(status.state) != CType.GPUCLOTH_V3_CLOTH_RUNNABLE):
                    raise RuntimeError(
                        "v3 cloth status did not accept stepped frame")
                _accept_dynamic_mesh_snapshot(i)
                diagnostic = _solver_diagnostic_snapshot(i)
                if diagnostic is not None:
                    native_ms = diagnostic["status"]["execution_time_ms"]
                    print(
                        f"[GPUCloth] {cloth_obj.name_full}: "
                        f"native {native_ms:.3f} ms")

                # 2. Persistent v3 readback positions/velocities.
                simulation_pos = self._readback(i)

                # 3. Handle-scoped v3 proxy apply into the persistent output.
                proxy_owner = (g_proxy_handles[i]
                               if i < len(g_proxy_handles) else None)
                if proxy_owner is not None:
                    if int(proxy_owner["proxy_vertex_count"]) != n_sim:
                        raise RuntimeError(
                            "v3 proxy status count differs from simulation mesh")
                    self._apply_positions(simulation_obj, simulation_pos, n_sim)
                    pos = _apply_v3_proxy(
                        proxy_owner, _runtime_frame_generation)
                    nV = int(proxy_owner["render_vertex_count"])
                else:
                    pos = simulation_pos
                    nV = n_sim

                # 4. Обновляем меш в Blender (foreach_set, Blender 4.x safe)
                self._apply_positions(cloth_obj, pos, nV)

                # 5. Handle-scoped v3 cache write. Native copies the payload
                # before returning; Python owns no cache array after call.
                if (not scene_s.is_baked and
                        not bool(getattr(scene_s, "use_external_cache", False))
                        and _cache_handle_value()):
                    if (_cache_write_frame(
                            context.scene, frame, pos, nV) !=
                            CType.GPUCLOTH_ABI_OK):
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
    if index < 0 or index >= len(_readback_owners):
        raise RuntimeError("v3 drape readback owner does not exist")
    owner = _readback_owners[index]
    result = int(g_dll.GPUCloth_v3_cloth_readback(
        owner["handle"], pointer(owner["readback"])))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"v3 drape readback rejected with {result}")
    positions = owner["readback_positions"]
    flat = np.frombuffer(positions, dtype=np.float32)
    simulation_obj.data.vertices.foreach_set("co", flat)
    simulation_obj.data.update()

    proxy_owner = (
        g_proxy_handles[index] if index < len(g_proxy_handles) else None)
    if proxy_owner is None:
        if cloth_obj is not simulation_obj:
            cloth_obj.data.vertices.foreach_set("co", flat)
            cloth_obj.data.update()
        return
    if int(proxy_owner["proxy_vertex_count"]) != n_sim:
        raise RuntimeError("Drape v3 proxy vertex count changed")
    high_positions = _apply_v3_proxy(
        proxy_owner, _runtime_frame_generation)
    cloth_obj.data.vertices.foreach_set(
        "co", np.frombuffer(high_positions, dtype=np.float32))
    cloth_obj.data.update()


def _selected_drape_owner(context):
    index = _prepared_cloth_index(context.object)
    if g_dll is None or index < 0 or index >= len(g_cloth_handles):
        raise RuntimeError("selected cloth is not prepared")
    return index, context.object, g_cloth_handles[index]


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
            index, cloth_obj, cloth_handle = _selected_drape_owner(context)
            _publish_frame_inputs(
                context, context.evaluated_depsgraph_get())
            config = CType.GPUClothDrapeConfig()
            config.struct_size = sizeof(config)
            config.config_version = 1
            config.max_steps = 240
            config.convergence_window = 8
            config.convergence_tolerance = float(
                cloth_obj.GPUCloth.solver_convergence_tol)
            layer_owner = _capture_drape_triangle_layers(
                g_simulationOBJs[index])
            if layer_owner is not None:
                layers, generation = layer_owner
                config.drape_flags = CType.GPUCLOTH_DRAPE_USE_TRIANGLE_LAYERS
                _set_buffer_view(
                    config.triangle_layers, CType.GPUCLOTH_ELEMENT_UINT32,
                    len(layers), sizeof(c_uint), addressof(layers), generation)
            status = _new_drape_status()
            result = int(g_dll.GPUCloth_v3_cloth_begin_drape(
                cloth_handle, pointer(config), pointer(status)))
            _remember_drape_status(cloth_obj, status)
            if result != CType.GPUCLOTH_ABI_OK:
                witness = _invariant_witness_data(cloth_handle)
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
            index, cloth_obj, cloth_handle = _selected_drape_owner(context)
            status = _new_drape_status()
            limit = 240 if self.until_settled else 1
            for _ in range(limit):
                result = int(g_dll.GPUCloth_v3_cloth_step_drape(
                    cloth_handle, pointer(status)))
                state = _remember_drape_status(cloth_obj, status)
                if result != CType.GPUCLOTH_ABI_OK:
                    witness = _invariant_witness_data(cloth_handle)
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
            index, cloth_obj, cloth_handle = _selected_drape_owner(context)
            status = _new_drape_status()
            result = int(g_dll.GPUCloth_v3_cloth_apply_drape(
                cloth_handle, pointer(status)))
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
            index, cloth_obj, cloth_handle = _selected_drape_owner(context)
            status = _new_drape_status()
            result = int(g_dll.GPUCloth_v3_cloth_cancel_drape(
                cloth_handle, pointer(status)))
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
    _, cloth_obj, cloth_handle = _selected_drape_owner(context)
    payload = {
        "schema": "GPUClothInvariantWitness/1",
        "cloth": cloth_obj.name_full,
        "witness": _invariant_witness_data(cloth_handle),
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
      v3 cloth step → v3 readback → foreach_set() → cache
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
        if int(g_dll.GPUCloth_v3_cache_clear(
                g_runtime_handle, _cache_handle_owner())) != CType.GPUCLOTH_ABI_OK:
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
        self._frame = s.bake_start
        return True

    def _step_frame(self, context, frame):
        _cache_playback_guard['active'] = True
        try:
            context.scene.frame_set(frame)
        finally:
            _cache_playback_guard['active'] = False
        result = bpy.ops.gpucloth.update_simulation()
        return (
            'FINISHED' in result and
            _cache_has_frame(context.scene, frame)
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
            g_dll.GPUCloth_v3_cache_clear(
                g_runtime_handle, _cache_handle_owner())
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

        if int(g_dll.GPUCloth_v3_cache_clear(
                g_runtime_handle, _cache_handle_owner())) != CType.GPUCLOTH_ABI_OK:
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
                cached = _cached_frame_positions(
                    scene, frame, cloth_obj, cache_dir_bytes)
                if cached is not None:
                    flat = np.frombuffer(cached, dtype=np.float32)
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


def _setup_cloth(obj, solver='PD', material='COTTON'):
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

        _setup_cloth(cloth_obj, solver='PD', material='COTTON')

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

        _setup_cloth(cloth_obj, solver='PD', material='COTTON')

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

        _setup_cloth(cloth_obj, solver='PD', material='COTTON')
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

        _setup_cloth(cushion_obj, solver='PD', material='COTTON')

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
        _setup_cloth(upper, solver='PD', material='COTTON')
        upper.GPUCloth.use_self_collision = True
        upper.GPUCloth.ogc_radius         = 150.0
        upper.GPUCloth.ogc_friction       = 0.3
        upper.GPUCloth.show_ogc_bounds    = True

        # Activate OGC on the lower sheet as well
        lower.select_set(True)
        bpy.context.view_layer.objects.active = lower
        _setup_cloth(lower, solver='PD', material='COTTON')
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
    global g_dll
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
    _close_dll_directories()
    return True
