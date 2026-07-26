"""Non-destructive Blender Cloth to GPUCloth settings bridge."""

from __future__ import annotations

import json

import bpy


SCHEMA_VERSION = 1
_STATE_PREFIX = "gpucloth_cpu_modifier_"


CLOTH_SETTINGS_MAP = {
    "mass": "vertex_mass",
    "quality": "quality_step",
    "time_scale": "speed_multiplier",
    "air_damping": "air_viscosity",
    "tension_stiffness": "tension",
    "compression_stiffness": "compression",
    "shear_stiffness": "shear",
    "bending_stiffness": "bending_stiffness",
    "tension_stiffness_max": "max_tension",
    "compression_stiffness_max": "max_compression",
    "shear_stiffness_max": "max_shear",
    "bending_stiffness_max": "max_bend",
    "tension_damping": "tension_damp",
    "compression_damping": "compression_damp",
    "shear_damping": "shear_damp",
    "bending_damping": "bending_damping",
    "bending_model": "bending_model",
    "use_internal_springs": "use_internal_springs",
    "internal_spring_normal_check": "use_internal_springs_normal",
    "internal_spring_max_length": "internal_spring_max_length",
    "internal_spring_max_diversion": "internal_spring_max_diversion",
    "internal_tension_stiffness": "internal_tension",
    "internal_tension_stiffness_max": "max_internal_tension",
    "internal_compression_stiffness": "internal_compression",
    "internal_compression_stiffness_max": "max_internal_compression",
    "use_pressure": "use_pressure",
    "uniform_pressure_force": "uniform_pressure_force",
    "target_volume": "target_volume",
    "pressure_factor": "pressure_factor",
    "fluid_density": "fluid_density",
    "shrink_min": "shrink_min",
    "shrink_max": "shrink_max",
    "use_dynamic_mesh": "use_dynamic_mesh",
    "goal_spring": "goalspring",
    "goal_friction": "goalfrict",
    "goal_max": "maxgoal",
    "vertex_group_mass": "vgroup_mass",
}


COLLISION_SETTINGS_MAP = {
    "use_collision": "use_object_collision",
    "distance_min": "epsilon",
    "friction": "collision_friction",
    "damping": "collision_damping",
    "collision_quality": "collision_quality",
    "impulse_clamp": "clamp",
    "self_distance_min": "selfepsilon",
    "self_impulse_clamp": "self_clamp",
}


EFFECTOR_WEIGHTS_MAP = {
    "gravity": "global_gravity",
    "wind": "weight_wind",
    "vortex": "weight_vortex",
    "magnetic": "weight_magnetic",
    "turbulence": "weight_turbulence",
    "drag": "weight_drag",
    "smokeflow": "weight_smoke_flow",
    "harmonic": "weight_harmonic",
    "charge": "weight_charge",
    "lennardjones": "weight_lennard_jones",
    "texture": "weight_texture",
    "curve_guide": "weight_curve_guide",
    "boid": "weight_boid",
}

POINT_CACHE_RANGE_MAP = {
    "frame_start": "bake_start",
    "frame_end": "bake_end",
}

POINT_CACHE_RUNTIME_STATUS = (
    "is_baked", "is_baking", "is_outdated", "is_frame_skip", "info",
)


def find_cpu_cloth_modifier(obj):
    return next((modifier for modifier in obj.modifiers if modifier.type == "CLOTH"), None)


def _stored_cpu_cloth_modifier(obj):
    stored_uid = obj.get(_STATE_PREFIX + "uid")
    if stored_uid is not None:
        return next((
            modifier for modifier in obj.modifiers
            if modifier.type == "CLOTH"
            and getattr(modifier, "persistent_uid", None) == stored_uid
        ), None)
    stored_name = obj.get(_STATE_PREFIX + "name")
    if not stored_name:
        return None
    modifier = obj.modifiers.get(stored_name)
    return modifier if modifier is not None and modifier.type == "CLOTH" else None


def _public_writable_properties(owner):
    if owner is None or not hasattr(owner, "bl_rna"):
        return set()
    return {
        prop.identifier
        for prop in owner.bl_rna.properties
        if prop.identifier != "rna_type" and not prop.is_readonly and not prop.is_hidden
    }


def _is_non_default(owner, prop_name):
    prop = owner.bl_rna.properties[prop_name]
    value = getattr(owner, prop_name)
    if prop.type == 'POINTER':
        return value is not None
    if prop.type == 'COLLECTION':
        return len(value) != 0
    if getattr(prop, "is_array", False):
        return tuple(value) != tuple(prop.default_array)
    return value != prop.default


def _values_match(source, destination):
    if isinstance(source, float) or isinstance(destination, float):
        scale = max(1.0, abs(float(source)), abs(float(destination)))
        return abs(float(source) - float(destination)) <= 1e-6 * scale
    return source == destination


def _copy_group(source, destination, mapping, owner_name, copied, errors, rollback):
    for source_name, target_name in mapping.items():
        if not hasattr(source, source_name):
            errors.append(f"{owner_name}.{source_name}: source missing")
            continue
        if not hasattr(destination, target_name):
            errors.append(f"GPUCloth.{target_name}: destination missing")
            continue
        try:
            value = getattr(source, source_name)
            previous = getattr(destination, target_name)
            setattr(destination, target_name, value)
            actual = getattr(destination, target_name)
            if not _values_match(value, actual):
                setattr(destination, target_name, previous)
                errors.append(
                    f"{owner_name}.{source_name}: value {value!r} cannot be "
                    f"represented by GPUCloth.{target_name} (got {actual!r})")
                continue
            rollback.append((destination, target_name, previous))
            copied.append({
                "source": f"{owner_name}.{source_name}",
                "target": f"GPUCloth.{target_name}",
                "value": value if isinstance(value, (bool, int, float, str)) else repr(value),
            })
        except (AttributeError, TypeError, ValueError) as exc:
            errors.append(f"{owner_name}.{source_name}: {exc}")


def sync_cpu_to_gpu(obj, scene=None):
    """Copy only fields already consumed by addon production wiring."""
    modifier = find_cpu_cloth_modifier(obj)
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_modifier": modifier.name if modifier else None,
        "copied": [],
        "unsupported": [],
        "unsupported_non_default": [],
        "errors": [],
        "committed": False,
    }
    if modifier is None:
        result["errors"].append("CPU Cloth modifier not found")
        _store_report(obj, result)
        return result

    rollback = []
    gpu = obj.GPUCloth
    settings = modifier.settings
    collision = modifier.collision_settings
    _copy_group(
        settings, gpu, CLOTH_SETTINGS_MAP, "ClothSettings",
        result["copied"], result["errors"], rollback)
    _copy_group(
        collision, gpu, COLLISION_SETTINGS_MAP, "ClothCollisionSettings",
        result["copied"], result["errors"], rollback)
    if hasattr(settings, "effector_weights") and settings.effector_weights:
        _copy_group(
            settings.effector_weights, gpu.effector_weights,
            EFFECTOR_WEIGHTS_MAP, "EffectorWeights",
            result["copied"], result["errors"], rollback)
    point_cache = getattr(modifier, "point_cache", None)
    if scene is not None and point_cache is not None and hasattr(
            scene, "gpu_cloth_helper"):
        _copy_group(
            point_cache, scene.gpu_cloth_helper, POINT_CACHE_RANGE_MAP,
            "PointCache", result["copied"], result["errors"], rollback)
        frame_step = int(getattr(point_cache, "frame_step", 1))
        if frame_step != 1:
            result["errors"].append(
                f"PointCache.frame_step: {frame_step} is unsupported; "
                "GPUCloth calculate-to-frame requires step 1")
        else:
            result["copied"].append({
                "source": "PointCache.frame_step",
                "target": "GPUClothScene.frame_step",
                "value": 1,
            })
        if bool(getattr(point_cache, "use_disk_cache", False)):
            result["copied"].append({
                "source": "PointCache.use_disk_cache",
                "target": "GPUClothCache.storage_mode",
                "value": "DISK",
            })
        cache_index = int(getattr(point_cache, "index", 0))
        if cache_index not in (-1, 0):
            result["errors"].append(
                f"PointCache.index: {cache_index} is unsupported; "
                "GPUCloth owns only the active primary cache")
        else:
            result["copied"].append({
                "source": "PointCache.index",
                "target": "GPUClothCache.cache_index",
                "value": 0,
            })

    mapped = {
        "ClothSettings": set(CLOTH_SETTINGS_MAP),
        "ClothCollisionSettings": set(COLLISION_SETTINGS_MAP),
        "EffectorWeights": set(EFFECTOR_WEIGHTS_MAP),
        "PointCache": (
            set(POINT_CACHE_RANGE_MAP) |
            {"frame_step", "use_disk_cache", "index"}),
    }
    owners = {
        "ClothSettings": settings,
        "ClothCollisionSettings": collision,
        "EffectorWeights": getattr(settings, "effector_weights", None),
        "PointCache": point_cache,
    }
    for owner_name, owner in owners.items():
        for prop_name in sorted(_public_writable_properties(owner) - mapped[owner_name]):
            qualified_name = f"{owner_name}.{prop_name}"
            result["unsupported"].append(qualified_name)
            inactive_effector_setting = (
                owner_name == "EffectorWeights"
                and not _scene_has_effectors(scene))
            inactive_point_cache_setting = (
                owner_name == "PointCache"
                and prop_name == "use_library_path"
                and not bool(getattr(point_cache, "use_external", False)))
            if (_is_non_default(owner, prop_name)
                    and not inactive_effector_setting
                    and not inactive_point_cache_setting):
                result["unsupported_non_default"].append(qualified_name)
    if point_cache is not None:
        for prop_name in POINT_CACHE_RUNTIME_STATUS:
            qualified_name = f"PointCache.{prop_name}"
            result["unsupported"].append(qualified_name)

    if result["errors"] or result["unsupported_non_default"]:
        for destination, target_name, previous in reversed(rollback):
            setattr(destination, target_name, previous)
        result["copied"] = []
    elif scene is not None and hasattr(scene, "gpu_cloth_helper"):
        scene.gpu_cloth_helper.gravity_x = scene.gravity[0]
        scene.gpu_cloth_helper.gravity_y = scene.gravity[1]
        scene.gpu_cloth_helper.gravity_z = scene.gravity[2]
        result["copied"].append({
            "source": "Scene.gravity",
            "target": "GPUClothScene.gravity",
            "value": list(scene.gravity),
        })

    result["committed"] = not result["errors"] and not result["unsupported_non_default"]

    _store_report(obj, result)
    return result


def _scene_has_effectors(scene):
    if scene is None:
        return False
    return any(
        getattr(obj, "field", None) is not None
        and obj.field.type != 'NONE'
        for obj in scene.objects)


def _store_report(obj, result):
    gpu = obj.GPUCloth
    gpu.cpu_sync_copied = len(result["copied"])
    gpu.cpu_sync_unsupported = len(result["unsupported"])
    gpu.cpu_sync_blockers = len(result["unsupported_non_default"])
    gpu.cpu_sync_errors = len(result["errors"])
    gpu.cpu_sync_report = json.dumps(result, sort_keys=True)


def load_report(obj):
    payload = obj.GPUCloth.cpu_sync_report
    if not payload:
        return {"copied": [], "unsupported": [], "unsupported_non_default": [], "errors": []}
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return {
            "copied": [], "unsupported": [], "unsupported_non_default": [],
            "errors": ["invalid sync report"]}


def record_runtime_error(obj, message):
    """Attach activation failure to current transaction report."""
    result = load_report(obj)
    result.setdefault("errors", []).append(message)
    result["committed"] = False
    _store_report(obj, result)
    return result


def select_backend(obj, backend, scene=None):
    """Switch evaluator ownership while preserving CPU modifier configuration."""
    modifier = find_cpu_cloth_modifier(obj)
    result = None
    if backend == "GPU":
        result = sync_cpu_to_gpu(obj, scene) if modifier else None
        if result is not None and (
                result["errors"] or result["unsupported_non_default"]):
            return result
        apply_modifier_ownership(obj, backend)
        obj.GPUCloth.is_active = True
        return result

    from . import operators
    if obj in operators.g_clothOBJs:
        released = operators.free_gpu_memory()
        if scene is not None and hasattr(scene, "gpu_cloth_springs_built"):
            scene.gpu_cloth_springs_built = False
        if not released:
            record_runtime_error(
                obj, "GPUCloth native data release failed while returning to CPU")
    obj.GPUCloth.is_active = False
    apply_modifier_ownership(obj, backend)
    return result


def apply_modifier_ownership(obj, backend):
    """Toggle CPU modifier evaluation without touching GPU runtime state."""
    modifier = find_cpu_cloth_modifier(obj)
    if backend == "GPU":
        if modifier is not None:
            if _STATE_PREFIX + "name" not in obj:
                obj[_STATE_PREFIX + "name"] = modifier.name
                if hasattr(modifier, "persistent_uid"):
                    obj[_STATE_PREFIX + "uid"] = int(modifier.persistent_uid)
                obj[_STATE_PREFIX + "viewport"] = bool(modifier.show_viewport)
                obj[_STATE_PREFIX + "render"] = bool(modifier.show_render)
            modifier.show_viewport = False
            modifier.show_render = False
        return

    if (_STATE_PREFIX + "name" not in obj
            and _STATE_PREFIX + "uid" not in obj):
        return
    stored_modifier = _stored_cpu_cloth_modifier(obj)
    if stored_modifier is not None:
        stored_modifier.show_viewport = bool(obj.get(_STATE_PREFIX + "viewport", True))
        stored_modifier.show_render = bool(obj.get(_STATE_PREFIX + "render", True))
    for suffix in ("name", "uid", "viewport", "render"):
        key = _STATE_PREFIX + suffix
        if key in obj:
            del obj[key]


def restore_all_cpu_owners():
    """Restore CPU modifiers after addon reload, disable, or failed runtime."""
    restored = []
    for obj in bpy.data.objects:
        if (_STATE_PREFIX + "name" not in obj
                and _STATE_PREFIX + "uid" not in obj):
            continue
        modifier = _stored_cpu_cloth_modifier(obj)
        if modifier is not None:
            modifier.show_viewport = bool(
                obj.get(_STATE_PREFIX + "viewport", True))
            modifier.show_render = bool(
                obj.get(_STATE_PREFIX + "render", True))
            restored.append(obj.name)
        for suffix in ("name", "uid", "viewport", "render"):
            key = _STATE_PREFIX + suffix
            if key in obj:
                del obj[key]
    return restored
