"""Blender-owned per-vertex channels for the native cloth ABI."""

from __future__ import annotations

import ctypes
import math


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
        if weight not in (0.0, 1.0):
            raise VertexChannelError(
                f"soft pin weight {weight} at vertex {vertex.index} is unsupported; "
                "only exact 0 or 1 is representable")
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
