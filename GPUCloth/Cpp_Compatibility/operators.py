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
import math
import os
import sys
import subprocess
import time

import numpy as np
from ctypes import (
    addressof, cdll, windll, POINTER, pointer, cast,
    c_bool, c_float, c_short, c_int, c_uint, c_void_p, c_size_t, c_char_p,
    create_string_buffer, sizeof,
)

from . import cpp_types as CType
from .proxy_binding import ProxyBindingError, validate_proxy_binding
from .vertex_channels import (
    VertexChannelError, apply_float_channel, binary_exclusion_mask,
    binary_pin_weights,
    evaluated_local_positions, vertex_group_weights,
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
    root = bpy.path.abspath(helper.cache_dir)
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
    config.storage_mode = (
        CType.GPUCLOTH_CACHE_STORAGE_DISK
        if helper.use_disk_cache
        else CType.GPUCLOTH_CACHE_STORAGE_MEMORY)
    config.compression_mode = CType.GPUCLOTH_CACHE_COMPRESSION_NONE
    config.frame_start = int(helper.bake_start)
    config.frame_end = int(helper.bake_end)
    config.frame_step = 1
    config.cache_index = cache_index
    config.cache_flags = 0
    config.cache_id = _stable_cache_id(
        path_bytes + b'\0' + cache_index.to_bytes(4, 'little') +
        name_bytes)
    config.path_utf8_address = addressof(path_buffer)
    config.name_utf8_address = addressof(name_buffer)
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    storage_feature = (
        CType.GPUCLOTH_FEATURE_CACHE_DISK
        if helper.use_disk_cache
        else CType.GPUCLOTH_FEATURE_CACHE_MEMORY)
    for feature in (
            storage_feature,
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


def _configure_material_features(dll, clmd, settings):
    config = CType.GPUClothMaterialConfig()
    config.header.struct_size = sizeof(config)
    config.header.config_version = 1
    config.bending_model = 1 if settings.bending_model == 'ANGULAR' else 0
    config.stiffness[:] = (
        settings.tension,
        settings.compression,
        settings.shear,
        settings.bending_stiffness,
    )
    config.stiffness_max[:] = (
        settings.max_tension,
        settings.max_compression,
        settings.max_shear,
        settings.max_bend,
    )
    config.damping[:] = (
        settings.tension_damp,
        settings.compression_damp,
        settings.shear_damp,
        settings.bending_damping,
    )
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    for feature in (
            CType.GPUCLOTH_FEATURE_STRETCH,
            CType.GPUCLOTH_FEATURE_SHEAR):
        config.header.feature_id = feature
        result = int(dll.SIM_configure_cloth_feature(clmd, header))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"typed material feature {feature} rejected with {result}")
    return CType.GPUCLOTH_ABI_OK


def _configure_pressure_features(dll, clmd, settings):
    if not settings.use_pressure:
        return CType.GPUCLOTH_ABI_OK
    config = CType.GPUClothPressureConfig()
    config.header.struct_size = sizeof(config)
    config.header.config_version = 1
    config.pressure_flags = CType.GPUCLOTH_PRESSURE_ENABLED
    config.uniform_pressure_force = settings.uniform_pressure_force
    config.target_volume = settings.target_volume
    config.pressure_factor = settings.pressure_factor
    config.fluid_density = settings.fluid_density
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    features = [
        CType.GPUCLOTH_FEATURE_PRESSURE_UNIFORM,
        CType.GPUCLOTH_FEATURE_FLUID_DENSITY,
    ]
    if settings.target_volume > 0.0:
        features.append(CType.GPUCLOTH_FEATURE_PRESSURE_VOLUME)
    for feature in features:
        config.header.feature_id = feature
        result = int(dll.SIM_configure_cloth_feature(clmd, header))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"typed pressure feature {feature} rejected with {result}")
    return CType.GPUCLOTH_ABI_OK


def _blender_sewing_edges(settings_owner, simulation_obj):
    cloth_settings = next(
        (modifier.settings for modifier in settings_owner.modifiers
         if modifier.type == 'CLOTH'),
        None)
    if (cloth_settings is None or
            not bool(getattr(cloth_settings, "use_sewing_springs", False))):
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


def _upload_pin_weights(dll, clmd, settings_owner, simulation_obj):
    weights = binary_pin_weights(
        simulation_obj, settings_owner.GPUCloth.vgroup_mass)
    return apply_float_channel(
        dll, CType, clmd, CType.GPUCLOTH_FEATURE_PIN_GOAL,
        CType.GPUCLOTH_VERTEX_PIN_WEIGHT, 1, weights)


def _upload_pin_targets(dll, clmd, settings_owner, simulation_obj, depsgraph):
    if not settings_owner.GPUCloth.vgroup_mass:
        return CType.GPUCLOTH_ABI_OK
    positions = evaluated_local_positions(simulation_obj, depsgraph)
    return apply_float_channel(
        dll, CType, clmd, CType.GPUCLOTH_FEATURE_ANIMATED_PIN,
        CType.GPUCLOTH_VERTEX_PIN_TARGET_XYZ, 3, positions)


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

# Защита от GC для ctypes-массивов, переданных в Cache_write_frame_async
# C++ пишет в фоне — массив должен жить до завершения записи
_live_arrays         = []     # list[c_float array]

# Keepalive for effectors ctypes array
_effectors_keepalive = None


def _close_dll_directories():
    for directory_handle in _dll_directory_handles:
        directory_handle.close()
    _dll_directory_handles.clear()

# ─── Effector helpers ──────────────────────────────────────────────────────

def _make_effector_weights(ew):
    """Convert Blender GPUClothEffectorWeights PropertyGroup to ctypes float[15]."""
    w = (c_float * 15)()
    w[0]  = ew.weight_gravity
    w[1]  = ew.weight_wind
    w[2]  = ew.weight_vortex
    w[3]  = ew.weight_magnetic
    w[4]  = ew.weight_turbulence
    w[5]  = ew.weight_drag
    w[6]  = ew.weight_smoke_flow
    w[7]  = ew.weight_harmonic
    w[8]  = ew.weight_charge
    w[9]  = ew.weight_lennard_jones
    w[10] = ew.weight_texture
    w[11] = ew.weight_curve_guide
    w[12] = ew.weight_boid
    w[13] = ew.weight_fluid
    w[14] = ew.global_gravity
    return w


def _upload_effectors(operator, context, dll):
    """
    Scan Blender scene for objects with force field (FIELD type empties).
    Build GPUEffector array and upload via SIM_set_effectors.
    """
    global _effectors_keepalive

    cloth_settings = None
    for obj in g_clothOBJs:
        if hasattr(obj, 'GPUCloth'):
            cloth_settings = obj.GPUCloth
            break

    if cloth_settings is None:
        dll.SIM_set_effectors(None, 0, (c_float * 15)(*([0.0]*15)))
        return

    effectors = []
    for obj in context.scene.objects:
        if not obj or obj.type != 'EMPTY':
            continue
        if not obj.field or obj.field.type == 'NONE':
            continue

        fd = obj.field
        mw = obj.matrix_world
        imw = mw.inverted_safe()

        ef = CType.GPUEffector()
        ef.maxdist = fd.distance_max if fd.use_max_distance else 0.0
        ef.mindist = fd.distance_min if fd.use_min_distance else 0.0
        ef.f_power = fd.falloff_power
        ef.f_noise = fd.noise
        ef.seed = fd.seed
        ef.f_size = fd.size
        ef.f_damp = 0.0  # not directly exposed in Blender field settings

        # Map Blender field type to PFIELD enum
        _FIELD_TYPE_MAP = {
            'FORCE': 1,           # PFIELD_FORCE
            'WIND': 4,            # PFIELD_WIND
            'VORTEX': 2,          # PFIELD_VORTEX
            'MAGNET': 3,          # PFIELD_MAGNET
            'HARMONIC': 7,        # PFIELD_HARMONIC
            'CHARGE': 8,          # PFIELD_CHARGE
            'LENNARDJ': 9,        # PFIELD_LENNARDJ
            'TURBULENCE': 11,     # PFIELD_TURBULENCE
            'DRAG': 12,           # PFIELD_DRAG
            'TEXTURE': 6,         # PFIELD_TEXTURE
            'GUIDE': 5,           # PFIELD_GUIDE
            'FLUID': 13,          # PFIELD_FLUIDFLOW
            'BOID': 10,           # PFIELD_BOID
        }
        ef.type = _FIELD_TYPE_MAP.get(fd.type, 0)
        if ef.type == 0:
            continue  # skip unsupported types

        ef.strength = fd.strength
        ef.flow = fd.flow

        # falloff type
        ef.falloff_type = {'SPHERE': 0, 'TUBE': 1, 'CONE': 2}.get(fd.falloff_type, 0)
        ef.shape_type = {'POINT': 0, 'PLANE': 1, 'SURFACE': 2, 'POINTS': 3, 'LINE': 4}.get(fd.shape, 0)
        ef.zdir = {'BOTH': 0, 'POSITIVE': 1, 'NEGATIVE': 2}.get(fd.z_direction, 0)

        # world matrix (column-major flat)
        for col in range(4):
            for row in range(4):
                ef.obmat[col * 4 + row] = mw[row][col]
        # inverse world matrix (column-major flat)
        for col in range(4):
            for row in range(4):
                ef.imat[col * 4 + row] = imw[row][col]

        effectors.append(ef)

    num_effectors = len(effectors)
    if num_effectors > CType.MAX_EFFECTORS:
        num_effectors = CType.MAX_EFFECTORS
        effectors = effectors[:CType.MAX_EFFECTORS]

    # Build ctypes array
    if num_effectors > 0:
        eff_array = (CType.GPUEffector * num_effectors)(*effectors)
        _effectors_keepalive = eff_array
        weights_arr = _make_effector_weights(cloth_settings.effector_weights)
        dll.SIM_set_effectors(eff_array, num_effectors, weights_arr)
    else:
        dll.SIM_set_effectors(None, 0, (c_float * 15)(*([0.0]*15)))

_cache_playback_guard  = {'active': False}
_initial_positions     = []     # list[np.ndarray] — rest positions per cloth object
_bake_range            = {'start': 1, 'end': 250}
_simulation_frame_state = {'last_solved': None}
_cache_source_state = {'generation': 0}


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
            hasher, f"cloth[{index}].settings", obj.GPUCloth)
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


def _shutdown_runtime_if_initialized():
    global g_runtime_initialized

    if g_dll is None or not g_runtime_initialized:
        return
    try:
        g_dll.SIM_shutdown_runtime()
    except AttributeError:
        pass
    finally:
        g_runtime_initialized = False


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

def free_gpu_memory(context=None):
    """Освобождает GPU память, сбрасывает все глобальные массивы."""
    global g_dll, g_scene, g_obj, g_mesh, g_clmd
    global g_clothOBJs, g_simulationOBJs, g_clothCollisionOBJs, g_proxy_handles

    success = True
    if g_dll is not None:
        # Освобождаем ProxySim handles перед FreeSolverData
        for handle in g_proxy_handles:
            if handle is not None:
                try:
                    g_dll.ProxySim_free(handle)
                except Exception as exc:
                    print(f"free_gpu_memory: ProxySim_free() failed: {exc}")
                    success = False
        try:
            g_dll.FreeSolverData()
        except Exception as e:
            print(f"free_gpu_memory: FreeSolverData() failed: {e}")
            success = False

    g_scene              = None
    g_obj                = []
    g_clmd               = []
    g_mesh               = []
    g_clothOBJs          = []
    g_simulationOBJs     = []
    g_clothCollisionOBJs = []
    g_proxy_handles      = []
    _collision_keepalive.clear()
    _initial_positions.clear()
    _live_arrays.clear()
    _simulation_frame_state['last_solved'] = None
    _cache_source_state['generation'] = 0

    if context is not None and hasattr(context.scene, 'gpu_cloth_springs_built'):
        context.scene.gpu_cloth_springs_built = False

    return success


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
            if "CUDA Version" in result.stdout:
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
                candidates.append(os.path.join(cuda_path, "bin"))
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

            g_dll.SIM_solver.argtypes = []
            g_dll.SIM_solver.restype  = c_bool

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

            g_dll.AddCollisionObject.argtypes    = [POINTER(CType.Object)]
            g_dll.AddCollisionObject.restype     = c_bool

            g_dll.RemoveCollisionObject.argtypes = [POINTER(CType.Object)]
            g_dll.RemoveCollisionObject.restype  = c_bool

            g_dll.UpdateScene.argtypes = [POINTER(CType.Scene)]
            g_dll.UpdateScene.restype  = c_bool

            # ── Effector fields ──────────────────────────────────────────
            g_dll.SIM_set_effectors.argtypes = [
                POINTER(CType.GPUEffector),
                c_int,
                POINTER(c_float),
            ]
            g_dll.SIM_set_effectors.restype = c_bool

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

            g_dll.SIM_set_cloth_vertex_channel.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.GPUClothVertexChannelConfig),
            ]
            g_dll.SIM_set_cloth_vertex_channel.restype = c_uint

            # ── ProxySim API ─────────────────────────────────────────────────
            #
            #   Симуляция грубого proxy-меша + апсэмплинг до hi-res.
            #   GPU путь: SIM_solver() → scatter → ClothVertex.x (proxy)
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
        except AttributeError as e:
            self.report({'ERROR'}, f"DLL не содержит ожидаемой функции: {e}")
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
        try:
            _shutdown_runtime_if_initialized()
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
        finally:
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
        if scene.rigidbody_world is None:
            bpy.ops.rigidbody.world_add()
        g_scene = pointer(CType.Scene())
        g_scene.contents.flag = scene.rigidbody_world.enabled
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

    def setMesh(self, context, obj) -> POINTER(CType.Mesh):
        if not obj:
            raise ValueError("obj не должен быть None")
        try:
            mesh      = CType.Mesh()
            mesh.totedge = len(obj.edges)
            mesh.totvert = len(obj.vertices)
            mesh.totpoly = len(obj.polygons)
            mesh.totloop = len(obj.loops)

            # Вершины
            mvertType = CType.MVert * len(obj.vertices)
            mvert     = mvertType()
            for i, v in enumerate(obj.vertices):
                mvert[i].co   = (c_float * 3)(*v.co)
                mvert[i].flag = 0
            mesh.mvert = cast(mvert, POINTER(CType.MVert))

            # Рёбра (crease/bweight убраны в Blender 4.0 как прямые поля,
            # здесь используются для C++ структуры — всегда 0)
            medgeType = CType.MEdge * len(obj.edges)
            medge     = medgeType()
            for i in obj.edges:
                medge[i.index].v1      = i.vertices[0]
                medge[i.index].v2      = i.vertices[1]
                medge[i.index].crease  = 0
                medge[i.index].bweight = 0
                medge[i.index].flag    = 35
            mesh.medge = cast(medge, POINTER(CType.MEdge))

            # Полигоны
            mpolyType = CType.MPoly * len(obj.polygons)
            mpoly     = mpolyType()
            for i in obj.polygons:
                mpoly[i.index].loopstart = i.loop_start
                mpoly[i.index].totloop   = i.loop_total
            mesh.mpoly = cast(mpoly, POINTER(CType.MPoly))

            # Loops
            mloopType = CType.MLoop * len(obj.loops)
            mloop     = mloopType()
            for idx, loop in enumerate(obj.loops):
                mloop[idx].v = loop.vertex_index
                mloop[idx].e = loop.edge_index
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
        sim_parms.mingoal        = 0
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

        # ── Pressure ──────────────────────────────────────────────────────
        sim_parms.fluid_density    = gs.fluid_density
        sim_parms.pressure_factor  = gs.pressure_factor
        sim_parms.target_volume    = gs.target_volume
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
        if gs.use_dynamic_mesh:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_DYNAMIC_MESH

        # Модель изгиба
        sim_parms.bending_model = (
            CType.CLOTH_BENDING_ANGULAR
            if gs.bending_model == 'ANGULAR'
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
        coll_parms.contents.self_friction = 5.0
        coll_parms.contents.friction      = gs.collision_friction
        coll_parms.contents.damping       = gs.collision_damping
        coll_parms.contents.selfepsilon   = gs.selfepsilon
        coll_parms.contents.loop_count    = gs.collision_quality
        coll_parms.contents.group         = None  # TODO: resolve Collection ptr
        coll_parms.contents.vgroup_selfcol = 0
        object_collision_group = (
            gs.id_data.vertex_groups.get(gs.vgroup_objcol)
            if gs.vgroup_objcol else None)
        coll_parms.contents.vgroup_objcol = (
            object_collision_group.index + 1
            if object_collision_group is not None else 0)
        coll_parms.contents.clamp          = gs.clamp
        coll_parms.contents.self_clamp     = gs.self_clamp
        coll_parms.contents.flags = (
            CType.CLOTH_COLLSETTINGS_FLAG_ENABLED
            if gs.use_object_collision else 0)
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

    def execute(self, context):
        global g_dll, g_scene, g_obj, g_mesh, g_clmd
        global g_clothOBJs, g_simulationOBJs
        global g_clothCollisionOBJs, g_proxy_handles

        # 1. Загружаем DLL если нужно
        if g_dll is None:
            bpy.ops.gpucloth.load_dll()
            if g_dll is None:
                self.report({'ERROR'}, "Не удалось загрузить DLL")
                return {'CANCELLED'}

        # 2. Освобождаем старые данные если были
        if context.scene.gpu_cloth_springs_built:
            if not free_gpu_memory(context):
                self.report({'ERROR'}, "Не удалось освободить GPU память")
                return {'CANCELLED'}

        # 3. Сохраняем файл (DLL нужен путь к blend для ряда операций)
        if not bpy.data.is_saved:
            bpy.ops.wm.save_as_mainfile(
                filepath=bpy.app.tempdir + 'GPU_Cloth.blend',
                check_existing=False)
        elif bpy.data.is_dirty:
            bpy.ops.wm.save_as_mainfile(
                filepath=bpy.data.filepath, check_existing=False)

        # 4. Переходим в Object mode для корректного считывания данных
        mode = bpy.context.active_object.mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # 5. Collect render owners and validate proxy topology before any
        # native cloth/collision owner is created.
        active_cloth = [
            item for item in bpy.context.scene.objects
            if item is not None
            and hasattr(item, 'GPUCloth')
            and item.GPUCloth.is_active
        ]
        bindings = []
        try:
            for item in active_cloth:
                bindings.append(validate_proxy_binding(item, item.GPUCloth))
        except ProxyBindingError as exc:
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            self.report({'ERROR'}, f"Proxy configuration rejected: {exc}")
            return {'CANCELLED'}

        g_clothOBJs.extend(active_cloth)
        g_simulationOBJs.extend(
            binding["simulation_object"] for binding in bindings)

        for item in bpy.context.scene.objects:
            if item is None:
                continue
            if item in active_cloth:
                for modifier in item.modifiers:
                    if modifier.type == 'CLOTH':
                        modifier.show_viewport = False
                        modifier.show_render   = False
            else:
                for modif in item.modifiers:
                    if modif.type == 'COLLISION':
                        data_ptr = self.fill_Object(item)
                        if not data_ptr:
                            self.report({'ERROR'}, "NULL от fill_Object (collision)")
                            return {'CANCELLED'}
                        g_clothCollisionOBJs.append(data_ptr)

        # 6. Заполняем Mesh + ClothModifierData + Object для каждой ткани
        for cloth_obj, simulation_obj in zip(g_clothOBJs, g_simulationOBJs):
            if (cloth_obj is None or simulation_obj is None
                    or not hasattr(simulation_obj, 'data')):
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

            data_ptr = self.setMesh(context, simulation_obj.data)
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

        # 7. Загружаем сцену на GPU
        # Reject unsupported weights and evaluated topology before native cloth
        # allocation. BuildClothSprings has no rollback owner before AddCloth.
        try:
            depsgraph = context.evaluated_depsgraph_get()
            for cloth_obj, simulation_obj in zip(
                    g_clothOBJs, g_simulationOBJs):
                binary_pin_weights(
                    simulation_obj, cloth_obj.GPUCloth.vgroup_mass)
                binary_exclusion_mask(
                    simulation_obj, cloth_obj.GPUCloth.vgroup_objcol,
                    "object collision")
                vertex_group_weights(
                    simulation_obj, cloth_obj.GPUCloth.vgroup_shrink,
                    "shrink")
                _blender_sewing_edges(cloth_obj, simulation_obj)
                if cloth_obj.GPUCloth.vgroup_mass:
                    evaluated_local_positions(simulation_obj, depsgraph)
        except VertexChannelError as exc:
            self.report({'ERROR'}, f"Vertex channel preflight failed: {exc}")
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        self.fill_Scene(context)

        for coll_ptr in g_clothCollisionOBJs:
            try:
                if not g_dll.AddCollisionObject(coll_ptr):
                    self.report({'ERROR'}, "Ошибка AddCollisionObject")
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError в AddCollisionObject")
                return {'CANCELLED'}

        # 7.5. Scan scene for force field effectors
        _upload_effectors(self, context, g_dll)

        try:
            _configure_cache_features(g_dll, context.scene)
        except (OSError, RuntimeError) as exc:
            self.report({'ERROR'}, f"Cache config failed: {exc}")
            free_gpu_memory(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        if not g_dll.FillSolverData(g_scene):
            self.report({'ERROR'}, "FillSolverData вернул ошибку")
            return {'CANCELLED'}

        for i in range(len(g_clothOBJs)):
            try:
                _configure_simulation_features(
                    g_dll, g_clmd[i], context.scene,
                    g_clothOBJs[i].GPUCloth)
                _configure_material_features(
                    g_dll, g_clmd[i], g_clothOBJs[i].GPUCloth)
                _configure_pressure_features(
                    g_dll, g_clmd[i], g_clothOBJs[i].GPUCloth)
            except (OSError, RuntimeError) as exc:
                self.report({'ERROR'}, f"Simulation config failed: {exc}")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}
            try:
                if not g_clmd[i].contents.clothObject:
                    if not g_dll.BuildClothSprings(g_clmd[i], g_mesh[i]):
                        self.report({'ERROR'}, "BuildClothSprings вернул ошибку")
                        return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError в BuildClothSprings")
                return {'CANCELLED'}

            try:
                if g_clmd[i].contents.clothObject is None:
                    self.report({'ERROR'}, "clothObject is NULL после BuildClothSprings")
                    bpy.ops.screen.animation_cancel()
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError: clothObject is NULL")
                return {'CANCELLED'}

            try:
                depsgraph = context.evaluated_depsgraph_get()
                _upload_sewing(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i])
                _upload_pin_weights(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i])
                _upload_pin_targets(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i],
                    depsgraph)
                _upload_pressure_weights(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i])
                _upload_shrink_weights(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i])
                _upload_object_collision_mask(
                    g_dll, g_clmd[i], g_clothOBJs[i], g_simulationOBJs[i])
            except (OSError, VertexChannelError) as exc:
                self.report({'ERROR'}, f"Vertex channel upload failed: {exc}")
                free_gpu_memory(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

            try:
                if not g_dll.AddCloth(g_clmd[i], g_mesh[i], g_obj[i]):
                    self.report({'ERROR'}, "AddCloth вернул ошибку")
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError в AddCloth")
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
        #   SIM_solver() [GPU ~N мс]
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
            for i in range(len(g_clothOBJs)):
                cloth_obj = g_clothOBJs[i]
                simulation_obj = g_simulationOBJs[i]
                n_sim = len(simulation_obj.data.vertices)

                _upload_pin_targets(
                    g_dll, g_clmd[i], cloth_obj, simulation_obj, depsgraph)

                # 1. GPU симуляция одного кадра
                if not g_dll.SIM_solver():
                    self.report({'ERROR'}, "SIM_solver вернул ошибку")
                    bpy.ops.screen.animation_cancel()
                    return {'CANCELLED'}

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

        except (OSError, VertexChannelError) as err:
            print(f"GPUCloth_UpdateSimulation OSError: {err}")
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}

        _simulation_frame_state['last_solved'] = frame
        elapsed_ms = (time.perf_counter_ns() - total_t0) / 1_000_000
        print(f"[GPUCloth] кадр {frame}: {elapsed_ms:.2f} мс")
        return {'FINISHED'}


# ===========================================================================
#  Оператор: запекание (bake) симуляции
# ===========================================================================

class GPUCloth_BakeSimulation(bpy.types.Operator):
    """
    Просчитать симуляцию для всего диапазона кадров и записать кэш на диск.
    Паттерн: FLIP Fluids BakeFluidSimulation (modal с таймером).

    На каждом кадре:
      SIM_solver() → SIM_get_cloth_verts() → foreach_set() → Cache_write_frame_async()
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


def register():
    _close_dll_directories()
    for cls in _OPERATOR_CLASSES:
        bpy.utils.register_class(cls)

    # Сбрасываем глобальное состояние при регистрации
    global g_dll, g_runtime_initialized, g_scene, g_obj, g_mesh, g_clmd
    global g_clothOBJs, g_simulationOBJs
    global g_clothCollisionOBJs, g_proxy_handles
    g_dll                = None
    g_runtime_initialized = False
    g_scene              = None
    g_obj                = []
    g_clmd               = []
    g_mesh               = []
    g_clothOBJs          = []
    g_simulationOBJs     = []
    g_clothCollisionOBJs = []
    g_proxy_handles      = []
    _collision_keepalive.clear()
    _initial_positions.clear()
    _live_arrays.clear()

    if _frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_frame_change_handler)
    if _cache_input_change_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(
            _cache_input_change_handler)

    # OGC contact-bounds visualiser — register once, draw callback checks flag
    global _ogc_draw_handle
    if _ogc_draw_handle is None:
        _ogc_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _ogc_bounds_draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
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
    global g_dll, g_runtime_initialized, g_scene, g_obj, g_mesh, g_clmd
    global g_clothOBJs, g_simulationOBJs
    global g_clothCollisionOBJs, g_proxy_handles
    free_gpu_memory()
    _shutdown_runtime_if_initialized()
    g_dll                = None
    g_runtime_initialized = False
    _close_dll_directories()
    g_scene              = None
    g_obj                = []
    g_clmd               = []
    g_mesh               = []
    g_clothOBJs          = []
    g_simulationOBJs     = []
    g_clothCollisionOBJs = []
    g_proxy_handles      = []
    _collision_keepalive.clear()
    _initial_positions.clear()
    _live_arrays.clear()
