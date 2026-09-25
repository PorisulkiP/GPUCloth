"""Blender-owned per-vertex channels for the native cloth ABI."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, replace

import numpy as np

from ..utils import version_compatibility_utils as vcu


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
    # C-contiguous arrays, not boxed Python sequences.  The publication wants
    # exactly these bytes, and on the 16 384-vertex drape fixture the former
    # tuple-of-floats form cost 4.6 ms per frame in boxing and slice
    # assignment against 0.03 ms for one copy each.  Same precedent as
    # ``MaterialCoordinateSnapshot.coordinates``.  Callers that still want a
    # Python sequence can iterate or unpack these unchanged.
    membership: "np.ndarray"
    raw_weights: "np.ndarray"
    evaluated_targets: "np.ndarray"
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
    """One GPUCloth material-coordinate field.

    v2 stores one coordinate per vertex.  v3 stores three coordinates per
    uploaded triangle, so UV seams are represented directly at corners.

    ``coordinates`` is one flat C-contiguous float32 array of interleaved U/V
    pairs - the exact layout the ABI's FLOAT2 element wants.  It is not a tuple
    of Python floats: the only consumer copies it into the
    ``GPUClothMaterialConfig.material_coordinates`` buffer, and building that
    buffer from a tuple boxes every component into a Python float only for the
    variadic constructor to unbox it again.  On the owner's scene shape that is
    193 548 components, measured at 30.1 ms through the tuple and 0.54 ms from
    the array.
    """

    topology_generation: int
    uv_map: str
    coordinates: "np.ndarray"

    @property
    def vertex_count(self):
        return len(self.coordinates) // 2

    @property
    def corner_count(self):
        return len(self.coordinates) // 2


def material_coordinate_payload(coordinates, label):
    """The flat float32 array the ABI's material-coordinate buffer wants.

    Every value is checked the way the per-element `float()`/`c_float()` round
    trip checked it, in one vectorised pass: the payload is already float32, so
    a finite value is unchanged by the round trip and a value outside float32's
    range or otherwise not finite is rejected here rather than silently
    published as an infinity.  The offending component is still named, with the
    loop and the U/V half rather than only the flat index.
    """
    array = np.ascontiguousarray(coordinates, dtype=np.float32).reshape(-1)
    non_finite = np.flatnonzero(~np.isfinite(array))
    if non_finite.size:
        offset = int(non_finite[0])
        raise VertexChannelError(
            f"{label} component {'U' if offset % 2 == 0 else 'V'} of loop "
            f"{offset // 2} is not finite")
    return array


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


def _automatic_triangle_coordinates(mesh):
    """Derive stable local-rest UVs when an imported mesh has no UV layer."""
    triangles = tuple(vcu.calc_mesh_loop_triangles(mesh))
    coordinates = []
    positions = mesh.vertices
    for triangle in triangles:
        loops = tuple(int(index) for index in triangle.loops)
        if len(loops) != 3:
            raise VertexChannelError("material triangle is not three corners")
        points = [positions[int(mesh.loops[index].vertex_index)].co
                  for index in loops]
        origin = np.asarray(points[0], dtype=np.float64)
        vectors = [np.asarray(point, dtype=np.float64) - origin
                   for point in points[1:]]
        normal = np.cross(vectors[0], vectors[1])
        normal_length = float(np.linalg.norm(normal))
        if not math.isfinite(normal_length) or normal_length <= 1.0e-12:
            raise VertexChannelError(
                "cannot derive material coordinates for a degenerate triangle")
        # A fixed local X reference projected onto the triangle tangent keeps
        # orientation stable across refinement; fall back to Y only when X is
        # parallel to the normal. This is an automatic orientation, not a
        # claim about imported fabric grain.
        reference = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        tangent = reference - normal * (np.dot(reference, normal) /
                                        (normal_length * normal_length))
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1.0e-12:
            reference = np.array((0.0, 1.0, 0.0), dtype=np.float64)
            tangent = reference - normal * (np.dot(reference, normal) /
                                            (normal_length * normal_length))
            tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1.0e-12:
            raise VertexChannelError("cannot orient material triangle")
        tangent /= tangent_length
        bitangent = np.cross(normal / normal_length, tangent)
        for vector in (np.zeros(3), vectors[0], vectors[1]):
            coordinate = (float(np.dot(vector, tangent)),
                          float(np.dot(vector, bitangent)))
            coordinates.extend(_finite_float32(value, "automatic material UV")
                               for value in coordinate)
    return tuple(coordinates), "<automatic-local-rest>"


def capture_material_coordinates(
        obj, settings, topology_generation, v3_corner=False):
    """Capture v2 vertex UVs or v3 uploaded-triangle corner UVs.

    FABRIC is the only material model, so the membrane directions always come
    from the UV map: ``use_anisotropy`` no longer decides whether the payload
    exists, only whether the same directions are also reflected onto the
    isotropic stiffness slots.  This function therefore no longer has a
    "no coordinates needed" early exit; a missing UV map is an error the caller
    has to resolve, not a silent fallback to an isotropic material.
    """
    if int(topology_generation) <= 0:
        raise VertexChannelError(
            "material-coordinate topology generation must be positive")

    try:
        mesh = obj.data
        vertices = mesh.vertices
        loops = mesh.loops
        uv_layers = mesh.uv_layers
        layer = None
    except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        raise VertexChannelError(
            f"cannot read material-direction UV map on {obj.name!r}") from exc
    uv_map = str(getattr(settings, "anisotropy_uv_map", ""))
    if not uv_map and getattr(uv_layers, "active", None) is not None:
        layer = uv_layers.active
        uv_map = str(getattr(layer, "name", "<active>"))
    elif uv_map:
        layer = uv_layers.get(uv_map)
    if layer is None:
        if v3_corner and not uv_map:
            coordinates, uv_map = _automatic_triangle_coordinates(mesh)
            return MaterialCoordinateSnapshot(
                topology_generation=int(topology_generation),
                uv_map=uv_map,
                coordinates=material_coordinate_payload(
                    coordinates, "automatic material UV"))
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

    if v3_corner:
        # One bulk read replaces 96 774 per-corner RNA accesses.  Blender stores
        # UVs as float32 and the gather below reproduces the former
        # triangle-then-corner order exactly, so the uploaded payload is
        # byte-identical.
        uv_values = np.empty(loop_count * 2, dtype=np.float32)
        layer.data.foreach_get("uv", uv_values)
        triangles = vcu.calc_mesh_loop_triangles(mesh)
        triangle_loops = np.empty(len(triangles) * 3, dtype=np.int32)
        triangles.foreach_get("loops", triangle_loops)
        if triangle_loops.size % 3:
            raise VertexChannelError("material triangle is not three corners")
        invalid = np.flatnonzero(
            (triangle_loops < 0) | (triangle_loops >= loop_count))
        if invalid.size:
            raise VertexChannelError(
                f"material-direction loop index "
                f"{int(triangle_loops[invalid[0]])} is invalid")
        # Gather the payload.  A loop triangle always owns three corners, and the
        # gather below reproduces the former triangle-then-corner order.
        gathered = uv_values.reshape(-1, 2)[triangle_loops].reshape(-1)
        non_finite = np.flatnonzero(~np.isfinite(gathered))
        if non_finite.size:
            corner = int(non_finite[0])
            raise VertexChannelError(
                f"material {'U' if corner % 2 == 0 else 'V'} at loop "
                f"{int(triangle_loops[corner // 2])} is not finite")
        if gathered.size == 0:
            raise VertexChannelError("material-direction mesh has no triangles")
        return MaterialCoordinateSnapshot(
            topology_generation=int(topology_generation), uv_map=uv_map,
            coordinates=material_coordinate_payload(
                gathered, f"material-direction UV map {uv_map!r}"))

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

    # The same interleaved U/V payload the nested-tuple flattening produced, as
    # one array: every component already went through `_finite_float32` above,
    # so the values are float32-exact and the array's bytes are the tuple's.
    flattened = np.asarray(coordinates, dtype=np.float32).reshape(-1)
    return MaterialCoordinateSnapshot(
        topology_generation=int(topology_generation),
        uv_map=uv_map,
        coordinates=flattened,
    )


def capture_evaluated_pin_snapshot(
        obj, depsgraph, settings, topology_generation, frame_generation,
        expected_vertex_count=None, identity_obj=None, channels=None,
        witness=None):
    """Capture membership, raw weights, and targets from one evaluated mesh.

    ``channels`` is an optional cache dict the caller keeps across frames and
    ``witness`` is what says whether it still describes this mesh: the caller
    hands over the cloth input fingerprint, which hashes every vertex-group
    name, member index and member weight of the object read here
    (``_cache_mesh_group_weights``).  While that fingerprint stands, the
    deform-layer walk below cannot produce anything else; when it moves, the
    walk runs again.  A caller that passes no cache - or a simulation object
    whose groups the fingerprint does not cover, which is the proxy case - gets
    the walk every time.

    Reusing the walk is the point of the parameter.  Membership and raw weights
    are deform-layer data: a *prepared* input, like the stiffness, pressure,
    shrink and self-collision channels captured once in the prepare operator,
    while the pin snapshot is recaptured every frame for its targets.  The walk
    is a Python pass over every vertex, measured at 19.5 ms median over nine
    frames on the owner's 16 384-vertex Drape scene - against 1.3 ms for the
    coordinate read it exists to serve and 2.0 ms for the whole collider
    capture.  It stays exact because those same values are hashed into the
    input fingerprint, so an edit to any of them is answered by a rebuilt
    owner before another frame is solved.
    """
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

        group = evaluated.vertex_groups.get(group_name) if group_name else None
        if group_name and group is None:
            raise VertexChannelError(
                f"pin vertex group {group_name!r} is absent on "
                f"{obj.name!r}")
        group_index = int(group.index) if group is not None else -1

        # foreach_get fills every coordinate in one C loop.  Reading the same
        # values through ``mesh.vertices`` in Python measured 43 ms at 16384
        # vertices, all of it bytecode.  Blender stores coordinates as float and
        # this buffer is float32, so the stored values are bit-identical to the
        # former float() round trip through Python floats.  The targets are what
        # this function is re-entered for on every frame, so they are always
        # read; the deform layer below is not.
        coordinates = np.empty(vertex_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coordinates)
        non_finite = np.flatnonzero(~np.isfinite(coordinates))
        if non_finite.size:
            raise VertexChannelError(
                f"evaluated pin target at vertex "
                f"{int(non_finite[0]) // 3} is not finite")

        cache_key = (witness, group_index, vertex_count)
        channels_hit = (
            channels is not None and witness is not None and
            channels.get("key") == cache_key)
        if channels_hit:
            membership = channels["membership"]
            raw_weights = channels["raw_weights"]
        else:
            membership = [0] * vertex_count
            raw_weights = [0.0] * vertex_count

            # Membership and raw weights are one scalar and one flag per vertex,
            # read from Blender's evaluated deform layer.  ``MeshVertex.groups``
            # exposes that same layer, so scanning it directly is the same read;
            # building a bmesh copy of the whole mesh to reach it is not.  On the
            # 16384-vertex drape fixture the bmesh route measured 9.9 ms/frame
            # against 3.7 ms for this scan, and every millisecond of it is host
            # time in front of the frame's first GPU work, so the device waits
            # for it.  The scan stays lazy: with no pin group configured it is
            # skipped entirely, which leaves the zero-initialised membership the
            # loop would have produced.
            if group is not None:
                for vertex in mesh.vertices:
                    for assignment in vertex.groups:
                        if int(assignment.group) != group_index:
                            continue
                        index = int(vertex.index)
                        weight = float(assignment.weight)
                        if (not math.isfinite(weight) or
                                weight < 0.0 or weight > 1.0):
                            raise VertexChannelError(
                                f"pin weight at vertex {index} is outside "
                                "[0, 1]")
                        # Membership is distinct from a stored raw weight of zero.
                        membership[index] = 1
                        raw_weights[index] = weight
                        break
            membership = np.asarray(membership, dtype=np.uint32)
            raw_weights = np.asarray(raw_weights, dtype=np.float32)
            if channels is not None and witness is not None:
                channels["key"] = cache_key
                channels["membership"] = membership
                channels["raw_weights"] = raw_weights

        targets = coordinates
    finally:
        evaluated.to_mesh_clear()

    snapshot = EvaluatedPinSnapshot(
        object_id=object_id,
        topology_generation=int(topology_generation),
        frame_generation=int(frame_generation),
        group_present=group is not None,
        membership=membership,
        raw_weights=raw_weights,
        evaluated_targets=targets,
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
    # One C copy per payload from the contiguous array the snapshot already
    # holds.  Boxed slice assignment - a list or tuple per payload - measured
    # 3.1 ms on the 16 384-vertex drape fixture against 0.02 ms for these
    # copies, and the stored bytes are the same.
    membership = (ctypes.c_uint32 * vertex_count).from_buffer_copy(
        np.ascontiguousarray(snapshot.membership, dtype=np.uint32))
    raw_weights = (ctypes.c_float * vertex_count).from_buffer_copy(
        np.ascontiguousarray(snapshot.raw_weights, dtype=np.float32))
    targets = (ctypes.c_float * (vertex_count * 3)).from_buffer_copy(
        np.ascontiguousarray(snapshot.evaluated_targets, dtype=np.float32))

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


def with_dragged_vertex_pin(snapshot, vertex_index, target):
    """Return a snapshot that hard-pins exactly one vertex to ``target``.

    The v3 product ABI owns every per-vertex pin value through this snapshot:
    ``GPUCLOTH_VERTEX_PIN_WEIGHT`` and ``GPUCLOTH_VERTEX_PIN_TARGET_XYZ`` are
    NOT_CONFIGURABLE vertex channels, and the native commit of this snapshot
    writes its targets into ``ClothVertex::xconst`` and raises
    ``CLOTH_VERT_FLAG_PINNED`` for members whose goal reaches SOFTGOALSNAP
    (``goal = weight ** 4``).  A full weight for one member therefore
    reproduces the transient per-frame pin the reference TestScene applies to
    the vertex under the cursor.

    Every other vertex keeps its membership, weight and target, so the drag
    constrains one vertex and leaves the rest of the cloth untouched.
    """
    index = int(vertex_index)
    if index < 0 or index >= snapshot.vertex_count:
        raise VertexChannelError(
            f"drag vertex index {index} is outside the pin snapshot")
    try:
        components = tuple(float(component) for component in target)
    except (TypeError, ValueError) as exc:
        raise VertexChannelError(
            "drag target is not a 3-component value") from exc
    if len(components) != 3:
        raise VertexChannelError("drag target is not a 3-component value")
    coordinates = tuple(
        _finite_float32(value, f"drag target component {axis}")
        for axis, value in enumerate(components))

    membership = list(snapshot.membership)
    raw_weights = list(snapshot.raw_weights)
    evaluated_targets = list(snapshot.evaluated_targets)
    membership[index] = 1
    raw_weights[index] = 1.0
    offset = index * 3
    evaluated_targets[offset:offset + 3] = coordinates
    return replace(
        snapshot,
        group_present=True,
        # A snapshot without a pin group carries no membership view and gives
        # every vertex goal 0.  Publishing membership for the dragged vertex
        # must not turn the configured default goal on for the others.
        goal_default=snapshot.goal_default if snapshot.group_present else 0.0,
        membership=tuple(membership),
        raw_weights=tuple(raw_weights),
        evaluated_targets=tuple(evaluated_targets),
    )


def publish_pin_snapshot(dll, types, cloth_handle, snapshot):
    """Compatibility direct publication; frame paths stage transactions."""
    owner = prepare_pin_snapshot(types, snapshot)
    result = int(dll.GPUCloth_v3_cloth_set_pin_snapshot(
        cloth_handle, ctypes.byref(owner["config"])))
    if result != types.GPUCLOTH_ABI_OK:
        raise VertexChannelError(
            f"native pin snapshot rejected: "
            f"{ABI_RESULT_NAMES.get(result, result)}")
    return result


def vertex_group_weights(obj, group_name, label):
    """Return raw base-mesh weights in vertex-index order.

    Empty group names mean that the optional channel is not configured.
    Named but absent groups are configuration errors.
    """
    if not group_name:
        return None
    try:
        vertices = obj.data.vertices
        vertex_count = len(vertices)
        group = obj.vertex_groups.get(group_name)
        object_name = obj.name
    except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        raise VertexChannelError(
            f"cannot read {label} vertex group") from exc
    if group is None:
        raise VertexChannelError(
            f"{label} vertex group {group_name!r} is absent on "
            f"{object_name!r}")

    weights = [0.0] * vertex_count
    seen = set()
    group_index = int(group.index)
    for vertex in vertices:
        try:
            vertex_index = int(vertex.index)
        except (AttributeError, TypeError, ValueError) as exc:
            raise VertexChannelError(
                f"{label} mesh has an invalid vertex index") from exc
        if vertex_index < 0 or vertex_index >= vertex_count:
            raise VertexChannelError(
                f"{label} vertex index {vertex_index} is invalid")
        if vertex_index in seen:
            raise VertexChannelError(
                f"{label} vertex index {vertex_index} is duplicated")
        seen.add(vertex_index)

        weight = 0.0
        for membership in vertex.groups:
            try:
                membership_group = int(membership.group)
            except (AttributeError, TypeError, ValueError) as exc:
                raise VertexChannelError(
                    f"{label} membership at vertex {vertex_index} "
                    "has an invalid group index") from exc
            if membership_group == group_index:
                try:
                    weight = float(membership.weight)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise VertexChannelError(
                        f"{label} weight at vertex {vertex_index} "
                        "is not numeric") from exc
                break
        if not math.isfinite(weight):
            raise VertexChannelError(
                f"{label} weight at vertex {vertex_index} is not finite")
        if weight < 0.0 or weight > 1.0:
            raise VertexChannelError(
                f"{label} weight {weight} at vertex {vertex_index} is outside "
                "[0, 1]")
        weights[vertex_index] = weight
    if len(seen) != vertex_count:
        raise VertexChannelError(
            f"{label} mesh vertex indices are incomplete")
    return weights


def binary_exclusion_mask(obj, group_name, label):
    """Map Blender's positive exclusion weights to native binary flags."""
    weights = vertex_group_weights(obj, group_name, label)
    if weights is None:
        return [0.0] * len(obj.data.vertices)
    return [1.0 if weight > 0.0 else 0.0 for weight in weights]


def binary_pin_weights(obj, group_name):
    weights = vertex_group_weights(obj, group_name, "pin")
    if weights is None:
        return [0.0] * len(obj.data.vertices)
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


def apply_float_channel(dll, types, cloth_handle, feature_id, channel,
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
    result = int(dll.GPUCloth_v3_cloth_set_vertex_channel(
        cloth_handle, ctypes.byref(config)))
    if result != types.GPUCLOTH_ABI_OK:
        raise VertexChannelError(
            f"native vertex channel rejected: "
            f"{ABI_RESULT_NAMES.get(result, result)}")
    return result
