"""Non-destructive Blender Cloth to GPUCloth settings bridge."""

from __future__ import annotations

import json
import math

import bpy


SCHEMA_VERSION = 1
_STATE_PREFIX = "gpucloth_cpu_modifier_"
_MODIFIER_VISIBILITY = (
    ("viewport", "show_viewport"),
    ("render", "show_render"),
    ("editmode", "show_in_editmode"),
    ("cage", "show_on_cage"),
)


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
    "sewing_force_max": "max_sewing",
    "use_sewing_springs": "use_sewing_springs",
    "use_pressure": "use_pressure",
    "use_pressure_volume": "use_pressure_volume",
    "uniform_pressure_force": "uniform_pressure_force",
    "pressure_factor": "pressure_factor",
    "fluid_density": "fluid_density",
    "shrink_min": "shrink_min",
    "shrink_max": "shrink_max",
    "use_dynamic_mesh": "use_dynamic_mesh",
    "goal_spring": "goalspring",
    "goal_friction": "goalfrict",
    "goal_min": "mingoal",
    "goal_max": "maxgoal",
    "goal_default": "defgoal",
    "vertex_group_mass": "vgroup_mass",
    "vertex_group_structural_stiffness": "vgroup_struct",
    "vertex_group_shear_stiffness": "vgroup_shear",
    "vertex_group_bending": "vgroup_bend",
    "vertex_group_intern": "vgroup_intern",
    "vertex_group_shrink": "vgroup_shrink",
    "vertex_group_pressure": "vgroup_pressure",
}


COLLISION_SETTINGS_MAP = {
    "collection": "collision_collection",
    "use_collision": "use_object_collision",
    "distance_min": "epsilon",
    "friction": "collision_friction",
    "damping": "collision_damping",
    "collision_quality": "collision_quality",
    "impulse_clamp": "clamp",
    "use_self_collision": "use_self_collision",
    "self_distance_min": "selfepsilon",
    "self_friction": "self_collision_friction",
    "self_impulse_clamp": "self_clamp",
    "vertex_group_object_collisions": "vgroup_objcol",
    "vertex_group_self_collisions": "vgroup_selfcol",
}


EFFECTOR_WEIGHTS_MAP = {
    "collection": "collection",
    "gravity": "global_gravity",
    "all": "weight_all",
    "force": "weight_force",
    "vortex": "weight_vortex",
    "magnetic": "weight_magnetic",
    "wind": "weight_wind",
    "curve_guide": "weight_curve_guide",
    "texture": "weight_texture",
    "harmonic": "weight_harmonic",
    "charge": "weight_charge",
    "lennardjones": "weight_lennard_jones",
    "boid": "weight_boid",
    "turbulence": "weight_turbulence",
    "drag": "weight_drag",
    "smokeflow": "weight_smoke_flow",
}


def capture_v3_velocity_damping(settings):
    """Capture the GPUCloth-owned global velocity damping exactly once.

    This is deliberately separate from Blender Cloth's CPU settings map:
    ``vel_damping`` is a shipped GPUCloth RNA setting and has its own v3
    typed owner, distinct from air and material damping.
    """
    try:
        value = float(settings.vel_damping)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "GPUCloth.vel_damping is unavailable") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            "GPUCloth.vel_damping must be finite and within [0, 1]")
    return value


def capture_v3_effector_scales(settings):
    """Capture the native global force/wind scales as one typed owner.

    These are independent from per-effector collection weights.  Capture is
    prepare-time only so native callers own the values for the full cloth
    lifetime and the frame loop performs no Python/ctypes allocation.
    """
    try:
        force_scale = float(settings.eff_force_scale)
        wind_scale = float(settings.eff_wind_scale)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "GPUCloth effector scales are unavailable") from exc
    for name, value in (
            ("eff_force_scale", force_scale),
            ("eff_wind_scale", wind_scale)):
        if not math.isfinite(value) or not 0.0 <= value <= 100000.0:
            raise ValueError(
                f"GPUCloth.{name} must be finite and within [0, 100000]")
    return force_scale, wind_scale


def capture_v3_constraint_network(settings):
    """Capture the supported typed constraint-network settings.

    The loose-edge records are captured by the evaluated-mesh owner. This
    helper owns only the four Blender RNA values and never retains a Blender
    or ctypes pointer.
    """
    try:
        enabled = bool(settings.use_constraint_network)
        phases = int(settings.cn_phases)
        sewing_speed = float(settings.cn_sewing_speed)
        seam_stiffness = float(settings.cn_seam_stiffness)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "GPUCloth constraint-network settings are unavailable") from exc
    if not enabled:
        return None
    if not 1 <= phases <= 10:
        raise ValueError("GPUCloth.cn_phases must be within [1, 10]")
    if (not math.isfinite(sewing_speed) or
            not 1.0 <= sewing_speed <= 100.0):
        raise ValueError(
            "GPUCloth.cn_sewing_speed must be finite and within [1, 100]")
    if (not math.isfinite(seam_stiffness) or
            not 0.0 <= seam_stiffness <= 5.0):
        raise ValueError(
            "GPUCloth.cn_seam_stiffness must be finite and within [0, 5]")
    return {
        "enabled": True,
        "phase_count": phases,
        "sewing_speed": sewing_speed,
        "seam_stiffness": seam_stiffness,
    }


def capture_v3_shrink_bounds(settings):
    """Capture Blender's ordered shrink endpoints for the v3 owner.

    Blender applies ``shrink_min`` at weight zero and ``shrink_max`` at
    weight one.  Blender's RNA enforces ``shrink_min <= shrink_max`` when
    either endpoint is assigned; reject malformed stand-ins instead of
    silently changing that product contract.
    """
    try:
        shrink_min = float(settings.shrink_min)
        shrink_max = float(settings.shrink_max)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "GPUCloth shrink bounds are unavailable") from exc
    for name, value in (
            ("shrink_min", shrink_min), ("shrink_max", shrink_max)):
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(
                f"GPUCloth.{name} must be finite and within [-1, 1]")
    if shrink_min > shrink_max:
        raise ValueError(
            "GPUCloth.shrink_min must be less than or equal to shrink_max")
    return shrink_min, shrink_max


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


def _copy_rest_shape_key(settings, gpu, copied, errors, rollback):
    if not hasattr(settings, "rest_shape_key"):
        errors.append("ClothSettings.rest_shape_key: source missing")
        return
    if not hasattr(gpu, "shapekey_rest"):
        errors.append("GPUCloth.shapekey_rest: destination missing")
        return
    try:
        source = settings.rest_shape_key
        value = "" if source is None else str(source.name)
        previous = gpu.shapekey_rest
        gpu.shapekey_rest = value
        actual = gpu.shapekey_rest
        if actual != value:
            gpu.shapekey_rest = previous
            errors.append(
                "ClothSettings.rest_shape_key: value "
                f"{value!r} cannot be represented by "
                f"GPUCloth.shapekey_rest (got {actual!r})")
            return
        rollback.append((gpu, "shapekey_rest", previous))
        copied.append({
            "source": "ClothSettings.rest_shape_key",
            "target": "GPUCloth.shapekey_rest",
            "value": value,
        })
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"ClothSettings.rest_shape_key: {exc}")


def _copy_pressure_volume(settings, gpu, copied, errors, rollback):
    if not hasattr(settings, "target_volume"):
        errors.append("ClothSettings.target_volume: source missing")
        return
    if not hasattr(settings, "use_pressure_volume"):
        errors.append("ClothSettings.use_pressure_volume: source missing")
        return
    if not hasattr(gpu, "target_volume"):
        errors.append("GPUCloth.target_volume: destination missing")
        return

    use_custom_volume = bool(settings.use_pressure_volume)
    source_volume = float(settings.target_volume)
    if use_custom_volume and source_volume <= 0.0:
        errors.append(
            "ClothSettings.use_pressure_volume: a zero custom target "
            "volume cannot be represented; disable custom volume to use "
            "the initial mesh volume")
        return

    # Blender ignores the stored target_volume value when custom volume is
    # disabled and derives the equilibrium volume from the initial mesh.
    # GPUCloth represents that mode with target_volume == 0.
    value = source_volume if use_custom_volume else 0.0
    previous = gpu.target_volume
    try:
        gpu.target_volume = value
        actual = float(gpu.target_volume)
        if not _values_match(value, actual):
            gpu.target_volume = previous
            errors.append(
                "ClothSettings.target_volume: effective value "
                f"{value!r} cannot be represented by "
                f"GPUCloth.target_volume (got {actual!r})")
            return
        rollback.append((gpu, "target_volume", previous))
        copied.extend(({
            "source": "ClothSettings.use_pressure_volume",
            "target": "GPUCloth.target_volume.mode",
            "value": use_custom_volume,
        }, {
            "source": "ClothSettings.target_volume",
            "target": "GPUCloth.target_volume",
            "value": value,
        }))
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"ClothSettings.target_volume: {exc}")


def _copy_point_cache(obj, scene, copied, errors):
    modifier = find_cpu_cloth_modifier(obj)
    if modifier is None or not hasattr(modifier, "point_cache"):
        errors.append("ClothModifier.point_cache: source missing")
        return
    if scene is None or not hasattr(scene, "gpu_cloth_helper"):
        errors.append("GPUClothScene.cache: destination missing")
        return

    point_cache = modifier.point_cache
    helper = scene.gpu_cloth_helper
    required = (
        "cache_index", "cache_name", "use_disk_cache",
        "use_external_cache", "external_cache_dir", "use_library_path",
        "cache_compression", "bake_start", "bake_end",
    )
    missing = [name for name in required if not hasattr(helper, name)]
    if missing:
        errors.append(
            "GPUClothScene.cache destination missing: " + ", ".join(missing))
        return

    cache_index = max(0, int(point_cache.index))
    cache_name = str(point_cache.name) if point_cache.name else "GPUCloth"
    use_external = bool(point_cache.use_external)
    use_library_path = bool(
        use_external and point_cache.use_library_path and obj.library)
    external_path = ""
    if use_external:
        library = obj.library if use_library_path else None
        external_path = bpy.path.abspath(
            str(point_cache.filepath), library=library)

    values = {
        "cache_index": cache_index,
        "cache_name": cache_name,
        "use_disk_cache": bool(point_cache.use_disk_cache),
        "use_external_cache": use_external,
        "external_cache_dir": external_path,
        "use_library_path": use_library_path,
        "cache_compression": str(point_cache.compression),
        "bake_start": int(point_cache.frame_start),
        "bake_end": int(point_cache.frame_end),
    }
    previous = {name: getattr(helper, name) for name in values}
    try:
        for name, value in values.items():
            setattr(helper, name, value)
            if not _values_match(value, getattr(helper, name)):
                raise ValueError(
                    f"{name} value {value!r} cannot be represented")
    except (AttributeError, TypeError, ValueError) as exc:
        for name, value in previous.items():
            setattr(helper, name, value)
        errors.append(f"ClothModifier.point_cache: {exc}")
        return

    copied.extend({
        "source": f"PointCache.{name}",
        "target": f"GPUClothScene.{name}",
        "value": value,
    } for name, value in values.items())


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
    _copy_rest_shape_key(
        settings, gpu, result["copied"], result["errors"], rollback)
    _copy_pressure_volume(
        settings, gpu, result["copied"], result["errors"], rollback)
    if hasattr(settings, "effector_weights") and settings.effector_weights:
        _copy_group(
            settings.effector_weights, gpu.effector_weights,
            EFFECTOR_WEIGHTS_MAP, "EffectorWeights",
            result["copied"], result["errors"], rollback)

    mapped = {
        "ClothSettings": (
            set(CLOTH_SETTINGS_MAP) |
            {"rest_shape_key", "target_volume"}),
        "ClothCollisionSettings": set(COLLISION_SETTINGS_MAP),
        "EffectorWeights": set(EFFECTOR_WEIGHTS_MAP),
    }
    owners = {
        "ClothSettings": settings,
        "ClothCollisionSettings": collision,
        "EffectorWeights": getattr(settings, "effector_weights", None),
    }
    for owner_name, owner in owners.items():
        for prop_name in sorted(_public_writable_properties(owner) - mapped[owner_name]):
            qualified_name = f"{owner_name}.{prop_name}"
            result["unsupported"].append(qualified_name)
            inactive_effector_setting = (
                owner_name == "EffectorWeights"
                and not _scene_has_effectors(scene))
            if _is_non_default(owner, prop_name) and not inactive_effector_setting:
                result["unsupported_non_default"].append(qualified_name)

    if result["errors"] or result["unsupported_non_default"]:
        for destination, target_name, previous in reversed(rollback):
            setattr(destination, target_name, previous)
        result["copied"] = []
    elif scene is not None and hasattr(scene, "gpu_cloth_helper"):
        _copy_point_cache(
            obj, scene, result["copied"], result["errors"])
        if result["errors"]:
            for destination, target_name, previous in reversed(rollback):
                setattr(destination, target_name, previous)
            result["copied"] = []
            result["committed"] = False
            _store_report(obj, result)
            return result
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
            for suffix, attribute in _MODIFIER_VISIBILITY:
                key = _STATE_PREFIX + suffix
                if key not in obj:
                    obj[key] = bool(getattr(modifier, attribute, False))
                setattr(modifier, attribute, False)
        return

    if (_STATE_PREFIX + "name" not in obj
            and _STATE_PREFIX + "uid" not in obj):
        return
    stored_modifier = _stored_cpu_cloth_modifier(obj)
    if stored_modifier is not None:
        for suffix, attribute in _MODIFIER_VISIBILITY:
            setattr(
                stored_modifier, attribute,
                bool(obj.get(
                    _STATE_PREFIX + suffix,
                    getattr(stored_modifier, attribute, False))))
    for suffix in (
            "name", "uid",
            *(suffix for suffix, _attribute in _MODIFIER_VISIBILITY)):
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
            for suffix, attribute in _MODIFIER_VISIBILITY:
                setattr(
                    modifier, attribute,
                    bool(obj.get(
                        _STATE_PREFIX + suffix,
                        getattr(modifier, attribute, False))))
            restored.append(obj.name)
        for suffix in (
                "name", "uid",
                *(suffix for suffix, _attribute in _MODIFIER_VISIBILITY)):
            key = _STATE_PREFIX + suffix
            if key in obj:
                del obj[key]
    return restored
