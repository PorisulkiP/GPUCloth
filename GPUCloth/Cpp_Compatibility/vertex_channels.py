"""Blender-owned per-vertex channels for the native cloth ABI."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass


class VertexChannelError(ValueError):
    pass


ABI_RESULT_NAMES = {
    0: "OK",
    1: "INVALID_ARGUMENT",
    2: "STRUCT_TOO_SMALL",
    3: "UNKNOWN_FEATURE",
    4: "UNSUPPORTED",
    5: "NOT_CONFIGURABLE",
    6: "VERSION_MISMATCH",
    7: "COUNT_MISMATCH",
    8: "INVALID_VALUE",
    9: "INVALID_STATE",
}


@dataclass(frozen=True)
class EvaluatedPinSnapshot:
    """One immutable Blender evaluation used by one native publication."""

    object_id: int
    topology_generation: int
    frame_generation: int
    group_present: bool
    membership: tuple[int, ...]
    raw_weights: tuple[float, ...]
    evaluated_targets: tuple[float, ...]
    goal_min: float
    goal_max: float
    goal_default: float
    goal_spring: float
    goal_damping: float

    @property
    def vertex_count(self):
        return len(self.membership)


@dataclass(frozen=True)
class MaterialCoordinateSnapshot:
    """One explicit, seam-free GPUCloth material-coordinate field."""

    topology_generation: int
    uv_map: str
    coordinates: tuple[float, ...]

    @property
    def vertex_count(self):
        return len(self.coordinates) // 2


def _live_object_id(obj, label):
    try:
        object_id = int(obj.session_uid)
    except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        raise VertexChannelError(f"{label} has no Blender session UID") from exc
    if object_id <= 0:
        raise VertexChannelError(f"{label} has no Blender session UID")
    return object_id


def _finite_setting(settings, name, fallback):
    value = float(getattr(settings, name, fallback))
    if not math.isfinite(value):
        raise VertexChannelError(f"pin setting {name} is not finite")
    return value


def _finite_float32(value, label):
    try:
        source = float(value)
        converted = float(ctypes.c_float(source).value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise VertexChannelError(f"{label} is not a float32 value") from exc
    if not math.isfinite(source) or not math.isfinite(converted):
        raise VertexChannelError(f"{label} is not finite")
    return converted


def capture_material_coordinates(obj, settings, topology_generation):
    """Capture an explicit named UV map as one exact float2 per vertex."""
    if not bool(getattr(settings, "use_anisotropy", False)):
        return None
    if int(topology_generation) <= 0:
        raise VertexChannelError(
            "material-coordinate topology generation must be positive")

    uv_map = str(getattr(settings, "anisotropy_uv_map", ""))
    if not uv_map:
        raise VertexChannelError(
            "anisotropy requires an explicit material-direction UV map")

    try:
        mesh = obj.data
        vertices = mesh.vertices
        loops = mesh.loops
        uv_layers = mesh.uv_layers
        layer = uv_layers.get(uv_map)
    except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        raise VertexChannelError(
            f"cannot read material-direction UV map on {obj.name!r}") from exc
    if layer is None:
        raise VertexChannelError(
            f"material-direction UV map {uv_map!r} is absent on "
            f"{obj.name!r}")

    vertex_count = len(vertices)
    loop_count = len(loops)
    if vertex_count <= 0 or loop_count <= 0:
        raise VertexChannelError(
            f"material-direction mesh for {obj.name!r} has no surface")
    if len(layer.data) != loop_count:
        raise VertexChannelError(
            f"material-direction UV map {uv_map!r} has "
            f"{len(layer.data)} corners; mesh has {loop_count}")

    coordinates = [None] * vertex_count
    seen_loops = set()
    for loop in loops:
        try:
            loop_index = int(loop.index)
            vertex_index = int(loop.vertex_index)
        except (AttributeError, TypeError, ValueError) as exc:
            raise VertexChannelError(
                "material-direction mesh contains an invalid loop") from exc
        if loop_index < 0 or loop_index >= loop_count:
            raise VertexChannelError(
                f"material-direction loop index {loop_index} is invalid")
        if loop_index in seen_loops:
            raise VertexChannelError(
                f"material-direction loop index {loop_index} is duplicated")
        seen_loops.add(loop_index)
        if vertex_index < 0 or vertex_index >= vertex_count:
            raise VertexChannelError(
                f"material-direction vertex index {vertex_index} is invalid")

        try:
            uv = layer.data[loop_index].uv
            coordinate = (
                _finite_float32(
                    uv[0],
                    f"material U at loop {loop_index}"),
                _finite_float32(
                    uv[1],
                    f"material V at loop {loop_index}"),
            )
        except (
                AttributeError, IndexError, ReferenceError, RuntimeError,
                TypeError) as exc:
            raise VertexChannelError(
                f"material-direction UV at loop {loop_index} is invalid") from exc
        previous = coordinates[vertex_index]
        if previous is not None and previous != coordinate:
            raise VertexChannelError(
                f"material-direction UV map {uv_map!r} has a seam at "
                f"vertex {vertex_index}; GPUCloth anisotropy requires one "
                "exact UV coordinate per vertex")
        coordinates[vertex_index] = coordinate

    if len(seen_loops) != loop_count:
        raise VertexChannelError(
            f"material-direction UV map {uv_map!r} has incomplete loops")
    missing = next(
        (index for index, value in enumerate(coordinates) if value is None),
        None)
    if missing is not None:
        raise VertexChannelError(
            f"material-direction UV map {uv_map!r} has no coordinate for "
            f"vertex {missing}")

    flattened = tuple(
        component for coordinate in coordinates for component in coordinate)
    return MaterialCoordinateSnapshot(
        topology_generation=int(topology_generation),
        uv_map=uv_map,
        coordinates=flattened,
    )


def capture_evaluated_pin_snapshot(
        obj, depsgraph, settings, topology_generation, frame_generation,
        expected_vertex_count=None, identity_obj=None):
    """Capture membership, raw weights, and targets from one evaluated mesh."""
    if int(topology_generation) <= 0 or int(frame_generation) <= 0:
        raise VertexChannelError(
            "pin topology/frame generations must be positive")
    identity_obj = obj if identity_obj is None else identity_obj
    object_id = _live_object_id(
        identity_obj, f"pin owner {identity_obj.name!r}")
    group_name = str(getattr(settings, "vgroup_mass", ""))

    try:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(
            preserve_all_data_layers=True, depsgraph=depsgraph)
    except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        raise VertexChannelError(
            f"cannot evaluate pin mesh for {obj.name!r}") from exc

    try:
        vertex_count = len(mesh.vertices)
        if expected_vertex_count is not None and (
                vertex_count != int(expected_vertex_count)):
            raise VertexChannelError(
                f"evaluated topology changed for {obj.name!r}: "
                f"{vertex_count} != {int(expected_vertex_count)}")
        if vertex_count <= 0:
            raise VertexChannelError(
                f"evaluated pin mesh for {obj.name!r} is empty")

        membership = [0] * vertex_count
        raw_weights = [0.0] * vertex_count
        targets = [0.0] * (vertex_count * 3)

        group = evaluated.vertex_groups.get(group_name) if group_name else None
        if group_name and group is None:
            raise VertexChannelError(
                f"pin vertex group {group_name!r} is absent on "
                f"{obj.name!r}")
        group_index = int(group.index) if group is not None else -1

        for vertex in mesh.vertices:
            index = int(vertex.index)
            if index < 0 or index >= vertex_count:
                raise VertexChannelError(
                    f"evaluated vertex index {index} is invalid for "
                    f"{obj.name!r}")
            offset = index * 3
            position = (
                float(vertex.co.x), float(vertex.co.y), float(vertex.co.z))
            if not all(math.isfinite(value) for value in position):
                raise VertexChannelError(
                    f"evaluated pin target at vertex {index} is not finite")
            targets[offset:offset + 3] = position

            if group is None:
                continue
            for assignment in vertex.groups:
                if int(assignment.group) != group_index:
                    continue
                weight = float(assignment.weight)
                if (not math.isfinite(weight) or
                        weight < 0.0 or weight > 1.0):
                    raise VertexChannelError(
                        f"pin weight at vertex {index} is outside [0, 1]")
                # Membership is distinct from a stored raw weight of zero.
                membership[index] = 1
                raw_weights[index] = weight
                break
    finally:
        evaluated.to_mesh_clear()

    snapshot = EvaluatedPinSnapshot(
        object_id=object_id,
        topology_generation=int(topology_generation),
        frame_generation=int(frame_generation),
        group_present=group is not None,
        membership=tuple(membership),
        raw_weights=tuple(raw_weights),
        evaluated_targets=tuple(targets),
        goal_min=_finite_setting(settings, "mingoal", 0.0),
        goal_max=_finite_setting(settings, "maxgoal", 1.0),
        goal_default=_finite_setting(settings, "defgoal", 0.0),
        goal_spring=_finite_setting(settings, "goalspring", 0.0),
        goal_damping=_finite_setting(settings, "goalfrict", 0.0),
    )
    if snapshot.goal_min > snapshot.goal_max:
        raise VertexChannelError(
            "pin goal minimum exceeds pin goal maximum")
    return snapshot


def prepare_pin_snapshot(types, snapshot):
    """Build one caller-owned descriptor and keep every payload alive."""
    vertex_count = snapshot.vertex_count
    membership = (ctypes.c_uint32 * vertex_count)(*snapshot.membership)
    raw_weights = (ctypes.c_float * vertex_count)(*snapshot.raw_weights)
    targets = (ctypes.c_float * (vertex_count * 3))(
        *snapshot.evaluated_targets)

    config = types.GPUClothPinSnapshotConfig()
    config.header.struct_size = ctypes.sizeof(config)
    config.header.feature_id = types.GPUCLOTH_FEATURE_ANIMATED_PIN
    config.header.config_version = 1
    config.object_id = snapshot.object_id
    config.topology_generation = snapshot.topology_generation
    config.frame_generation = snapshot.frame_generation
    config.pin_flags = (
        types.GPUCLOTH_PIN_GROUP_PRESENT
        if snapshot.group_present else 0)
    config.vertex_count = vertex_count
    config.goal_min = snapshot.goal_min
    config.goal_max = snapshot.goal_max
    config.goal_default = snapshot.goal_default
    config.goal_spring = snapshot.goal_spring
    config.goal_damping = snapshot.goal_damping

    views = [
        (config.evaluated_targets, types.GPUCLOTH_ELEMENT_FLOAT3,
         vertex_count, ctypes.sizeof(ctypes.c_float) * 3, targets),
    ]
    if snapshot.group_present:
        views[:0] = [
            (config.membership, types.GPUCLOTH_ELEMENT_UINT32,
             vertex_count, ctypes.sizeof(ctypes.c_uint32), membership),
            (config.raw_weights, types.GPUCLOTH_ELEMENT_FLOAT,
             vertex_count, ctypes.sizeof(ctypes.c_float), raw_weights),
        ]
    for view, element_type, element_count, stride, payload in views:
        view.struct_size = ctypes.sizeof(types.GPUClothBufferView)
        view.element_type = element_type
        view.element_count = element_count
        view.stride_bytes = stride
        view.data_address = ctypes.addressof(payload)
        view.generation = snapshot.frame_generation

    return {
        "config": config,
        "membership": membership,
        "raw_weights": raw_weights,
        "targets": targets,
        "snapshot": snapshot,
    }


def publish_pin_snapshot(dll, types, clmd, snapshot):
    """Compatibility direct publication; frame paths stage transactions."""
    owner = prepare_pin_snapshot(types, snapshot)
    result = int(dll.SIM_set_cloth_pin_snapshot(
        clmd, ctypes.byref(owner["config"])))
    if result != types.GPUCLOTH_ABI_OK:
        raise VertexChannelError(
            f"native pin snapshot rejected: "
            f"{ABI_RESULT_NAMES.get(result, result)}")
    return result


def binary_pin_weights(obj, group_name):
    vertex_count = len(obj.data.vertices)
    weights = [0.0] * vertex_count
    if not group_name:
        return weights
    group = obj.vertex_groups.get(group_name)
    if group is None:
        raise VertexChannelError(
            f"pin vertex group {group_name!r} is absent on {obj.name!r}")
    group_index = group.index
    for vertex in obj.data.vertices:
        weight = 0.0
        for membership in vertex.groups:
            if membership.group == group_index:
                weight = float(membership.weight)
                break
        if not math.isfinite(weight):
            raise VertexChannelError(
                f"pin weight at vertex {vertex.index} is not finite")
        if weight < 0.0 or weight > 1.0:
            raise VertexChannelError(
                f"pin weight {weight} at vertex {vertex.index} is outside "
                "[0, 1]")
        weights[vertex.index] = weight
    return weights


def evaluated_local_positions(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(
        preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        positions = [0.0] * (len(mesh.vertices) * 3)
        for vertex in mesh.vertices:
            offset = vertex.index * 3
            positions[offset:offset + 3] = (
                float(vertex.co.x), float(vertex.co.y), float(vertex.co.z))
    finally:
        evaluated.to_mesh_clear()
    if len(positions) != len(obj.data.vertices) * 3:
        raise VertexChannelError(
            f"evaluated topology changed for {obj.name!r}: "
            f"{len(positions) // 3} != {len(obj.data.vertices)}")
    if not all(math.isfinite(value) for value in positions):
        raise VertexChannelError(
            f"evaluated pin targets for {obj.name!r} are not finite")
    return positions


def apply_float_channel(dll, types, clmd, feature_id, channel,
                        element_width, values):
    if element_width <= 0 or len(values) % element_width:
        raise VertexChannelError("vertex channel width does not divide data")
    payload = (ctypes.c_float * len(values))(*values)
    config = types.GPUClothVertexChannelConfig()
    config.header.struct_size = ctypes.sizeof(config)
    config.header.feature_id = feature_id
    config.header.config_version = 1
    config.header.flags = 0
    config.channel = channel
    config.element_width = element_width
    config.element_count = len(values) // element_width
    config.data_address = ctypes.addressof(payload)
    result = int(dll.SIM_set_cloth_vertex_channel(clmd, ctypes.byref(config)))
    if result != types.GPUCLOTH_ABI_OK:
        raise VertexChannelError(
            f"native vertex channel rejected: "
            f"{ABI_RESULT_NAMES.get(result, result)}")
    return result
