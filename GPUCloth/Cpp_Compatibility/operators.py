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
import ctypes
import hashlib
import json
import math
import os
import queue
import struct
import sys
import subprocess
import threading
import time
import traceback
from fractions import Fraction

import numpy as np
from bpy_extras.view3d_utils import (
    location_3d_to_region_2d, region_2d_to_location_3d)
from ctypes import (
    addressof, cdll, POINTER, pointer, cast,
    c_bool, c_float, c_int, c_uint, c_uint64, c_void_p, c_size_t,
    c_char_p,
    create_string_buffer, sizeof,
)
from mathutils import Matrix, Vector

from . import cpp_types as CType
from . import properties
from .proxy_binding import ProxyBindingError, validate_proxy_binding
from .vertex_channels import (
    VertexChannelError, apply_float_channel, binary_exclusion_mask,
    capture_evaluated_pin_snapshot, capture_material_coordinates,
    prepare_pin_snapshot,
    publish_pin_snapshot,
    vertex_group_weights,
    with_dragged_vertex_pin,
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

# ── Validity of ``g_clothOBJs`` ─────────────────────────────────────────────
#
#  Blender can free an object while this module still holds its Python wrapper:
#  the user removes the cloth object, or Ctrl+Z rewinds the scene past the
#  prepare that built it.  The wrapper is not a reference Blender honours - it
#  goes REMOVED, and from then on every read from it, ``.name`` and ``.data``
#  and ``is_valid`` alike, raises ``ReferenceError: StructRNA of type Object
#  has been removed``.  Measured on host Blender 5.2.1 for all of
#  ``is_valid``, ``as_pointer()``, ``name``, ``name_full``, ``bl_rna`` and
#  ``data``; the only reads that survive are ``type(x).__name__`` and ``is``.
#
#  So validity cannot be a property of an entry.  It is a property of the LIST,
#  and the owner of that property is `_live_cloth_objects`: the identity of an
#  entry is captured while the object is still alive - the name it had and the
#  database pointer it had - and afterwards only ``bpy.data`` is asked.  A
#  lookup keyed on the name string reads the database, never the stale wrapper,
#  and reports None for a freed object instead of raising; the pointer, taken
#  once at entry time, rejects a name that a different object has since reused.
_cloth_object_identity = []


def _cloth_object_key(cloth_obj):
    """The name an object is taken with, or None.

    ``None`` covers both "no name" and "no longer a live object": a removed
    wrapper raises ``ReferenceError`` for ``name``/``name_full`` exactly as it
    does for every other read, and a caller that cannot tell the two apart only
    ever needs "no identity", which is this.
    """
    try:
        key = getattr(cloth_obj, "name_full", None)
        if key is None:
            key = getattr(cloth_obj, "name", None)
    except (AttributeError, ReferenceError, RuntimeError):
        return None
    return key


def _live_name(value, fallback=None):
    """The name of a Blender ID, or ``fallback`` if it is gone.

    ``getattr(obj, "name_full", obj.name)`` is the idiom this module used for
    "name this thing", and it is wrong for an object Blender has freed: the
    default is only reached on ``AttributeError``, and a removed wrapper raises
    ``ReferenceError`` for both reads, so the fallback expression is evaluated
    first and the whole call raises.  Measured on host Blender 5.2.1.
    """
    name = _cloth_object_key(value)
    return fallback if name is None else name


def _cloth_object_pointer(cloth_obj):
    """The live database pointer of ``cloth_obj``, or None.

    ``None`` means the object is gone, or is not the object this module took.
    Only ``bpy.data`` and ``as_pointer()`` on the RESOLVED object are read, so
    this never dereferences a removed wrapper.
    """
    key = _cloth_object_key(cloth_obj)
    if key is None:
        return None
    live = bpy.data.objects.get(key)
    if live is None:
        return None
    return live.as_pointer()


def _take_cloth_object(cloth_obj):
    """Record one entry's identity in ``g_clothOBJs``.

    Every writer of the list goes through this, so an entry cannot exist
    without the identity that makes it prunable.  A caller that appends to the
    list directly still works - the entry is then resolved by name alone on the
    next ``_live_cloth_objects`` - but it loses the reuse protection above.
    """
    identity = _cloth_object_identity
    try:
        index = g_clothOBJs.index(cloth_obj)
    except (ValueError, AttributeError, ReferenceError, RuntimeError):
        index = None
    if index is None:
        g_clothOBJs.append(cloth_obj)
        identity.append(
            (_cloth_object_key(cloth_obj), _cloth_object_pointer(cloth_obj)))
        return
    identity[index] = (
        _cloth_object_key(cloth_obj), _cloth_object_pointer(cloth_obj))


def _dead_cloth_objects():
    """The indices of ``g_clothOBJs`` entries Blender has already freed.

    ``bpy.data`` is the only thing read that belongs to Blender: nothing here
    touches an object that may have been removed, and nothing here raises when
    one has been.
    """
    dead = []
    identity = _cloth_object_identity
    for index, cloth_obj in enumerate(g_clothOBJs):
        if cloth_obj is None:
            dead.append(index)
            continue
        recorded = identity[index] if index < len(identity) else None
        if recorded is None:
            recorded = (_cloth_object_key(cloth_obj), None)
        name, pointer = recorded
        live = bpy.data.objects.get(name) if name is not None else None
        if live is None:
            dead.append(index)
        elif pointer is not None and live.as_pointer() != pointer:
            dead.append(index)
    return dead


def _live_cloth_objects():
    """``g_clothOBJs`` with every entry Blender has freed removed from it.

    The one validity owner for the list.  Consumers that iterate it call this
    first, which keeps the list clean as a side effect of being read, so a stale
    entry cannot survive a single visit by any of them - the dependency-graph
    handler is enough on its own, because it runs on every scene update.

    Pruning cannot happen where the object is freed.  Blender frees it inside
    its own call (``bpy.data.objects.remove``, the end of an undo); no handler
    or callback of this add-on runs at that instant, and once it has happened
    the wrapper carries no readable name to prune BY - which is the whole
    reason this function exists.
    """
    dead = _dead_cloth_objects()
    if dead:
        for index in reversed(dead):
            del g_clothOBJs[index]
            if index < len(_cloth_object_identity):
                del _cloth_object_identity[index]
        print(
            f"GPUCloth: dropped {len(dead)} removed cloth object(s) from the "
            f"active list; prepare again to rebuild from the scene")
    return g_clothOBJs


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
# The payload each history entry was built from, keyed by the same
# `(cloth_owner_id, object_id, instance_id, modifier_index)`.  One entry per
# collider occurrence, so the bound is the collider count; it is cleared with
# `_collider_history` in `free_gpu_memory`, which is where a cloth owner is
# released, and it is never cleared while an entry's owner is live.
_collider_payload_cache = {}
_COLLIDER_PAYLOAD_CACHE_MAX = 64
_collider_payload_state = {'retain_count': 0, 'rebuild_count': 0}
# The triangulation and cloth-local transform each retained payload was
# built from, keyed by the same `(cloth_owner_id, object_id, instance_id,
# modifier_index)` occurrence key.  It is the payload cache's own lifetime -
# one entry per collider occurrence, cleared with it in `free_gpu_memory` -
# because it holds the inputs that payload was built from and nothing else.
_collider_geometry_cache = {}
# `reuse_count` is the frames served without re-triangulating the collider
# and `build_count` the frames that had to.  `rebuild_after_reuse_count` is
# the one failure this reuse can have - a frame that presented geometry the
# cache claimed to cover - and it is the number a probe has to read to know
# the gate was exercised rather than merely present.  See
# `_collider_geometry_witness`.
_collider_geometry_state = {'reuse_count': 0, 'build_count': 0,
                            'rebuild_after_reuse_count': 0}
# Memo for `_cached_collider_motion_certificate`.  Cleared with the rest of the
# per-run state in `free_gpu_memory`, and bounded while a run is live.
_collider_certificate_cache = {}
_COLLIDER_CERTIFICATE_CACHE_MAX = 32
_collider_certificate_state = {'fit_count': 0, 'reuse_count': 0,
                               'eviction_count': 0}
_drape_status_by_uid = {}
# Last completed Settle verdict per cloth: the drape's criterion is a position
# tolerance plus a simulated-seconds budget, so the panel cannot recompute the
# verdict from the native status alone and the operator that owns the run
# publishes it here.
_drape_settle_verdict_by_uid = {}
_teardown_failure = False
_pending_auto_prepare = None
_auto_prepare_timer_registered = False
_prepare_task = None
_prepare_timer_registered = False
_prepare_task_serial = 0
_stop_requested = False
# Timer tick for one in-flight prepare transaction.  This interval is not a
# period anything is scheduled at: the callback returns it only while it still
# has work to hand back, and the work it waits on is a worker thread running one
# native call (or the generator's own next segment, which is main-thread work
# already in progress).  Measured on a live GUI session, the 128x128 drape at
# 16 384 verts: the same prepare spends 909.6 ms asleep at 50 ms and 745.0 ms at
# 5 ms, and its wall clock falls from 1782.6 ms to 1508.8 ms.  The interval stays
# above zero because a callback returning 0.0 would run at the frame rate and
# drown the frame it is meant to free.
_PREPARE_POLL_INTERVAL = 0.005
# How long the deferred-prepare timer waits before retrying a mode switch
# Blender refused.  The refusal means a modal transform owns the object mode, and
# a transform lasts as long as the user holds the mouse, so the retry is paced
# well below the per-frame rate rather than at `_PREPARE_POLL_INTERVAL`: a
# tick-rate retry would attempt the switch - and report it - once per tick for the
# whole drag.  `_prepare_deferred_notified` keeps one run of retries to a single
# report, because the first already told the user what is happening.
_DEFERRED_PREPARE_RETRY = 0.25
_prepare_deferred_notified = False

# Per-group cost of the last live re-configure, and the gate cost of the frame
# that read it.  Off unless asked for: a live re-configure runs on the frame path,
# and the measurement is two `perf_counter_ns` calls per group.
_live_input_profile = {
    'enabled': False,
    'steps': [],
    'gate_ms': [],
    'changed_groups': [],
}

# Per-cloth, per-group fingerprints of the last live publication.  Cleared with
# the rest of the per-run state in `free_gpu_memory`, and by a prepare, because a
# rebuilt owner is holding freshly captured inputs rather than the live ones.
_live_group_state = {}
_MODIFIER_VISIBILITY = (
    "show_viewport", "show_render", "show_in_editmode", "show_on_cage")


def _effector_publication_metrics():
    return dict(_effector_publication_state)


def _reset_effector_publication_metrics():
    for name in _effector_publication_state:
        _effector_publication_state[name] = 0


def _collider_payload_metrics():
    metrics = dict(_collider_payload_state)
    metrics.update(_collider_geometry_state)
    return metrics


def _reset_collider_payload_metrics():
    for name in _collider_payload_state:
        _collider_payload_state[name] = 0
    for name in _collider_geometry_state:
        _collider_geometry_state[name] = 0


def _runtime_owners_retained():
    """True while state that needs the native library to be released is held.

    The prepared rest pose (``_initial_positions``) is deliberately NOT named
    here, and that is the whole of this predicate's contract: it is the set of
    state a native free retires, and nothing else.  The pose is scene state - the
    one a successful prepare validated and the one a refused rebuild is recovered
    from - so `_reset_owner_python_state` keeps it on every path, including the
    paths that destroy the runtime (see the note there).  Counting it here made
    this predicate true with no native owner left, which is the state the gate's
    teardown reaches: free -> `gpucloth.unload_dll` -> `addon.unregister()`.  The
    add-on then asked `free_gpu_memory` to release owners that did not exist,
    with no DLL left to release them, and the refusal that came back - "native
    DLL unavailable while runtime or owners are retained" - was about state no
    native call can release, so package unregister raised over a session that was
    already torn down.  Freeing the pose is not this predicate's to ask for: it
    is released by dropping the reference, and the frame path restores it.
    """
    return bool(
        _runtime_handle_value() or _cache_handle_value() or g_cloth_handles or
        _cloth_input_owners or _readback_owners or
        g_clothOBJs or g_simulationOBJs or g_clothCollisionOBJs or
        g_proxy_handles or _collision_keepalive or _solver_diagnostics or
        _pin_snapshot_states or _dynamic_mesh_states or
        _collection_snapshots or _effector_weight_states or
        _collider_history)


def prepare_task_active(obj=None):
    task = _prepare_task
    if task is None:
        return False
    return obj is None or task.get("object_name") == _live_name(obj)


def _prepare_native_worker_active():
    task = _prepare_task
    worker = task.get("worker") if task is not None else None
    return worker is not None and worker.is_alive()


def auto_prepare_pending(obj=None):
    if prepare_task_active(obj):
        return True
    request = _pending_auto_prepare
    if request is None:
        return False
    return obj is None or request.get("object_name") == _live_name(obj)


def cancel_auto_prepare():
    """Cancel the single deferred prepare request and its timer owner."""
    global _pending_auto_prepare, _auto_prepare_timer_registered
    _pending_auto_prepare = None
    if _auto_prepare_timer_registered:
        try:
            if bpy.app.timers.is_registered(_run_auto_prepare):
                bpy.app.timers.unregister(_run_auto_prepare)
        except (AttributeError, RuntimeError):
            pass
    _auto_prepare_timer_registered = False
    cancel_prepare_task()


def _run_auto_prepare():
    global _pending_auto_prepare, _auto_prepare_timer_registered
    request = _pending_auto_prepare
    if request is not None and prepare_task_active():
        return _PREPARE_POLL_INTERVAL
    _pending_auto_prepare = None
    _auto_prepare_timer_registered = False
    # The rebuild this request stands for is now either starting or refused;
    # either way it no longer blocks the frame path.  ``_simulation_frame_state``
    # is defined below this function, so read it back off the module rather than
    # closing over a name that is not bound yet.
    frame_state = globals().get('_simulation_frame_state')
    if frame_state is not None:
        frame_state['rebuild_pending'] = False
    if request is None or _stop_requested:
        return None
    scene = bpy.data.scenes.get(request["scene_name"])
    obj = bpy.data.objects.get(request["object_name"])
    if scene is None or obj is None:
        return None
    settings = getattr(obj, "GPUCloth", None)
    if (settings is None or settings.execution_backend != 'GPU' or
            not settings.is_active or
            (not settings.auto_prepare and
             not request.get("required", False))):
        return None
    owner = live_drape_sandbox_owner()
    if owner is not None:
        # A rebuild here would take the drape sandbox with it; hold the request
        # until the sandbox ends rather than starting it.
        _hold_deferred_prepare(request, owner)
        return None
    view_layer = scene.view_layers[0] if scene.view_layers else None
    helper = getattr(scene, "gpu_cloth_helper", None)
    if view_layer is None:
        if helper is not None:
            helper.memory_preflight_status = (
                "ERROR: deferred prepare has no scene view layer")
        return None
    previous_active = view_layer.objects.active
    previous_selected = tuple(
        candidate for candidate in view_layer.objects
        if candidate.select_get())
    try:
        view_layer.objects.active = obj
        obj.select_set(True)
        override = {
            "scene": scene,
            "view_layer": view_layer,
            "object": obj,
            "active_object": obj,
        }
        if bpy.context.window is not None:
            override["window"] = bpy.context.window
        with bpy.context.temp_override(
                **override):
            if not _start_prepare_task(bpy.context, automatic=True):
                raise RuntimeError("another preparation task is active")
    except (AttributeError, RuntimeError) as exc:
        message = f"ERROR: deferred prepare failed: {exc}"
        if helper is not None:
            helper.memory_preflight_status = message
        print(f"[GPUCloth] {message}")
    finally:
        view_layer.objects.active = previous_active
        for candidate in view_layer.objects:
            candidate.select_set(candidate in previous_selected)
    return None


def _defer_prepare_for_transform(self, exc):
    """Record a refused mode switch and pace the next attempt.

    Returns the status the prepare generator reports for this attempt.  The
    retry keeps the request queued without spending a timer tick on it, so a
    transform that lasts seconds costs one queue entry and one report rather
    than one of each per tick.
    """
    global _prepare_deferred_notified
    self._prepare_deferred = "{}".format(exc)
    _simulation_frame_state['rebuild_pending'] = True
    if not _prepare_deferred_notified:
        _prepare_deferred_notified = True
        self.report(
            {'INFO'},
            "GPUCloth preparation deferred until the current transform "
            f"finishes: {exc}")
    return {'FINISHED'}


def schedule_auto_prepare(obj, scene, required=False):
    """Queue one Blender-main-thread prepare after property callbacks return.

    ``required`` marks a prepare the user's own edit made necessary rather than
    an ``auto_prepare`` convenience, so the deferred run is not suppressed by
    that preference.
    """
    global _pending_auto_prepare, _auto_prepare_timer_registered, _stop_requested
    global _prepare_deferred_notified
    if obj is None or scene is None:
        return False
    _stop_requested = False
    if _pending_auto_prepare is None:
        # A fresh request, not one of the deferred timer's own retries: the next
        # deferral is a new event and is worth reporting again.
        _prepare_deferred_notified = False
    _pending_auto_prepare = {
        "object_name": _live_name(obj),
        "scene_name": _live_name(scene),
        "required": bool(required),
    }
    if _auto_prepare_timer_registered:
        return True
    try:
        bpy.app.timers.register(_run_auto_prepare, first_interval=0.0)
    except (AttributeError, RuntimeError):
        _pending_auto_prepare = None
        return False
    _auto_prepare_timer_registered = True
    return True


def live_drape_sandbox_owner():
    """The cloth whose live drape sandbox a rebuild would destroy, if any.

    A prepare replaces the native solver owner, and the drape sandbox lives in
    that owner: measured, a deferred prepare that ran under a live sandbox left
    the next drape step refused with ABI 10 and the panel's drape rows without a
    status.  A sandbox counts as live while it is ACTIVE or while the native
    status is the 240-step cap expiring - the drape keeps stepping in both.
    """
    for cloth_obj in _live_cloth_objects():
        status = get_drape_ui_status(cloth_obj)
        if status is None:
            continue
        flags = int(status["status_flags"])
        if flags & (CType.GPUCLOTH_DRAPE_STATUS_CANCELLED |
                    CType.GPUCLOTH_DRAPE_STATUS_APPLIED):
            continue
        if flags & CType.GPUCLOTH_DRAPE_STATUS_ACTIVE or (
                drape_not_settled(status)):
            return cloth_obj
    return None


def _hold_deferred_prepare(request, owner):
    """Keep a deferred prepare queued while a drape sandbox blocks it.

    This is the same answer the add-on already gives to a prepare it cannot
    serve yet (``_finish_prepare_task`` with ``deferred=True``): the request
    stays queued - so ``auto_prepare_pending()`` keeps saying so and the frame
    path keeps dropping frames it cannot solve - and the queue is re-armed by
    the operators that end a sandbox, Cancel and Apply.  Nothing polls it, so
    the hold cannot become a retry loop, and nothing drops it silently: the
    panel shows the queued prepare and its reason.
    """
    global _pending_auto_prepare, _auto_prepare_timer_registered
    global _prepare_deferred_notified
    _pending_auto_prepare = request
    _auto_prepare_timer_registered = False
    _simulation_frame_state['rebuild_pending'] = True
    message = (
        "Prepare held: the drape sandbox on "
        f"{_live_name(owner, 'the cloth')} is live and a rebuild "
        "would discard it; Cancel or Apply the drape to let it run")
    scene = bpy.data.scenes.get(request["scene_name"])
    if scene is not None:
        _set_prepare_status(scene, "QUEUED", 0, message)
    if not _prepare_deferred_notified:
        _prepare_deferred_notified = True
        print(f"[GPUCloth] {message}")


def resume_held_prepare():
    """Re-arm a prepare that was held for a live drape sandbox.

    Called by the operators that end a sandbox.  Those are the only events that
    can clear the hold, and the teardown paths (Stop, unregister) drop the
    request instead, so no timer has to poll for the condition.
    """
    global _auto_prepare_timer_registered
    if _pending_auto_prepare is None or _auto_prepare_timer_registered:
        return False
    try:
        bpy.app.timers.register(_run_auto_prepare, first_interval=0.0)
    except (AttributeError, RuntimeError):
        return False
    _auto_prepare_timer_registered = True
    return True


def _tag_prepare_redraw():
    try:
        windows = tuple(bpy.context.window_manager.windows)
    except (AttributeError, ReferenceError, RuntimeError):
        return
    for window in windows:
        try:
            areas = tuple(window.screen.areas)
        except (AttributeError, ReferenceError, RuntimeError):
            continue
        for area in areas:
            if getattr(area, "type", None) == 'PROPERTIES':
                area.tag_redraw()


def _set_prepare_status(scene, state, progress, message):
    helper = getattr(scene, "gpu_cloth_helper", None)
    if helper is not None:
        helper.prepare_state = str(state)
        helper.prepare_progress = max(0, min(100, int(progress)))
        helper.prepare_status = str(message)
    task = _prepare_task
    if task is not None:
        task["state"] = str(state)
        task["progress"] = max(0, min(100, int(progress)))
    _tag_prepare_redraw()


def _prepare_progress(progress, message):
    return {
        "kind": "progress",
        "progress": int(progress),
        "message": str(message),
    }


def _prepare_native(function, args, progress, message):
    return {
        "kind": "native",
        "function": function,
        "args": tuple(args),
        "progress": int(progress),
        "message": str(message),
    }


def _run_prepare_native(result_queue, token, function, args):
    """Run one pure native call; this worker never owns Blender data."""
    try:
        result = function(*args)
    except BaseException as exc:
        result_queue.put((token, False, exc))
    else:
        result_queue.put((token, True, result))


class _PrepareRunner:
    def __init__(self, scene_name):
        self.scene_name = scene_name
        self._native_prepare_mutated = False
        # Set when the prepare stopped because Blender refused the object-mode
        # switch a modal transform owns.  A deferred prepare is not a failed
        # one: it mutated nothing and its request has to be re-armed.
        self._prepare_deferred = None
        self.last_report = ""

    def report(self, levels, message):
        self.last_report = str(message)
        task = _prepare_task
        if task is None or task.get("scene_name") != self.scene_name:
            print(f"[GPUCloth] {message}")
            return
        scene = bpy.data.scenes.get(self.scene_name)
        if scene is not None:
            state = "ERROR" if 'ERROR' in levels else task["state"]
            _set_prepare_status(scene, state, task["progress"], message)
        print(f"[GPUCloth] {message}")


def _prepare_context(task):
    scene = bpy.data.scenes.get(task["scene_name"])
    obj = bpy.data.objects.get(task["object_name"])
    if scene is None or obj is None:
        raise RuntimeError("preparation scene or object no longer exists")
    view_layer = scene.view_layers.get(task["view_layer_name"])
    if view_layer is None:
        raise RuntimeError("preparation view layer no longer exists")
    return scene, obj, view_layer


def _resume_prepare_task(task):
    scene, obj, view_layer = _prepare_context(task)
    previous_active = view_layer.objects.active
    previous_selected = tuple(
        candidate for candidate in view_layer.objects
        if candidate.select_get())
    try:
        view_layer.objects.active = obj
        obj.select_set(True)
        override = {
            "scene": scene,
            "view_layer": view_layer,
            "object": obj,
            "active_object": obj,
        }
        if bpy.context.window is not None:
            override["window"] = bpy.context.window
        with bpy.context.temp_override(**override):
            if task["generator"] is None:
                task["generator"] = (
                    GPUCloth_PrepareSimulation._prepare_steps(
                        task["runner"], bpy.context))
            pending_exception = task.pop("pending_exception", None)
            if pending_exception is not None:
                return task["generator"].throw(pending_exception)
            if task.pop("send_ready", False):
                return task["generator"].send(task.pop("send_value", None))
            return next(task["generator"])
    finally:
        try:
            view_layer.objects.active = previous_active
            for candidate in view_layer.objects:
                candidate.select_set(candidate in previous_selected)
        except (AttributeError, ReferenceError, RuntimeError):
            pass


def _restore_prepare_mode(task):
    original_mode = task.get("original_mode")
    if original_mode is None:
        return
    try:
        scene, obj, view_layer = _prepare_context(task)
        previous_active = view_layer.objects.active
        previous_selected = tuple(
            candidate for candidate in view_layer.objects
            if candidate.select_get())
        view_layer.objects.active = obj
        obj.select_set(True)
        override = {
            "scene": scene,
            "view_layer": view_layer,
            "object": obj,
            "active_object": obj,
        }
        if bpy.context.window is not None:
            override["window"] = bpy.context.window
        try:
            with bpy.context.temp_override(**override):
                if obj.mode != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
        finally:
            view_layer.objects.active = previous_active
            for candidate in view_layer.objects:
                candidate.select_set(candidate in previous_selected)
    except (AttributeError, ReferenceError, RuntimeError):
        pass


def _finish_prepare_task(success=False, cancelled=False, error=None,
                         deferred=False):
    global _prepare_task, _prepare_timer_registered, _stop_requested
    task = _prepare_task
    if task is None:
        return
    worker = task.get("worker")
    if worker is not None and worker.is_alive():
        raise RuntimeError("cannot finish preparation while native worker runs")
    generator = task.get("generator")
    if generator is not None and not task.get("generator_finished"):
        try:
            generator.close()
        except (RuntimeError, ValueError):
            pass
    _restore_prepare_mode(task)
    scene = bpy.data.scenes.get(task["scene_name"])
    runner = task["runner"]
    native_mutated = bool(runner._native_prepare_mutated)
    if not success:
        if native_mutated:
            try:
                context_scene, context_obj, view_layer = _prepare_context(task)
                override = {
                    "scene": context_scene,
                    "view_layer": view_layer,
                    "object": context_obj,
                    "active_object": context_obj,
                }
                if bpy.context.window is not None:
                    override["window"] = bpy.context.window
                with bpy.context.temp_override(**override):
                    free_gpu_memory(bpy.context)
            except (AttributeError, ReferenceError, RuntimeError):
                free_gpu_memory()
        else:
            for modifier, visibility in task["modifier_state"]:
                try:
                    for attribute, value in zip(
                            _MODIFIER_VISIBILITY, visibility):
                        setattr(modifier, attribute, value)
                except (AttributeError, ReferenceError, RuntimeError):
                    pass
        if scene is not None and hasattr(scene, "gpu_cloth_springs_built"):
            scene.gpu_cloth_springs_built = (
                False if native_mutated else task["original_springs_built"])

    if scene is not None:
        if deferred:
            # Report the refusal Blender actually gave, not a restatement of it.
            message = runner._prepare_deferred
            state = "DEFERRED"
            progress = task["progress"]
        elif success:
            message = "GPUCloth preparation complete"
            state = "READY"
            progress = 100
        elif cancelled:
            message = "GPUCloth preparation cancelled"
            state = "CANCELLED"
            progress = task["progress"]
        else:
            detail = str(error or runner.last_report or "unknown error")
            message = detail if detail.startswith("ERROR:") else f"ERROR: {detail}"
            state = "ERROR"
            progress = task["progress"]
        _set_prepare_status(scene, state, progress, message)
        if success:
            # A landed prepare is the one moment the add-on knows, without
            # guessing, that the cloth now shows the start of a run: see
            # ``_place_playhead_at_simulation_start``.  Asked of the landing and
            # not of the caller, because every prepare that built an owner
            # arrives here - the panel's button, the auto-prepare a settings edit
            # schedules, and the bake's own re-prepare - and all three leave the
            # timeline describing a simulation that has just been replaced.
            _place_playhead_at_simulation_start(scene)
    _prepare_task = None
    _prepare_timer_registered = False
    # A deferred prepare is not a stop, and not a failure: the engine state is
    # untouched, so the frame path keeps stepping and the request is re-armed.
    #
    # A prepare landing no longer clears the switch at all, and that is half of
    # the defect-3 fix.  It used to be cleared here whenever the landing was not
    # a failure (``_stop_requested = not success and not deferred``, which is
    # False on a deferred landing), while ``_prepare_steps`` had already cleared
    # it for a prepare that passed its own preflight (operators.py:9651).  A
    # switch still set at this point was therefore set *after* that preflight -
    # by Stop - and clearing it here undid the owner's own stop: a refused
    # (deferred) rebuild keeps the engine live by design and its retry clears the
    # switch again (``schedule_auto_prepare``, operators.py:475), so the frame
    # path went back in flight a quarter of a second after Stop returned, on a
    # playback Stop had never managed to cancel (measured on the shipped build:
    # 259 frame changes after Stop returned, ``is_animation_playing`` still true
    # 12 s later).  Nothing is lost by not clearing it: the starts that should
    # clear it are the owner's own - Prepare through ``_prepare_steps`` and an
    # edit through ``schedule_auto_prepare``.
    if not success and not deferred:
        _stop_requested = True
    # A deferred prepare is re-armed *unless the session is stopped*.  The retry
    # is the add-on asking itself again for work the owner's earlier edit
    # required, not the owner starting anything: with the switch set, re-arming
    # it would clear the switch (``schedule_auto_prepare``, operators.py:475) and
    # put a prepare - and behind it the frame path - back in flight a quarter of
    # a second after Stop returned.  That is the loop the owner could not escape:
    # a refused (deferred) rebuild keeps the engine live by design, so every Stop
    # pressed while one was retrying landed in the same window.  The request is
    # not lost: the panel still shows the deferred status and its reason, and the
    # owner's next edit or Prepare re-arms it.
    if deferred and not _stop_requested:
        schedule_auto_prepare(
            bpy.data.objects.get(task["object_name"]),
            bpy.data.scenes.get(task["scene_name"]),
            required=True)
        # Pace the retry instead of letting it run at the timer's poll rate: the
        # refusal means a modal transform is running, and one attempt per poll
        # interval is one refused mode switch and one report per 50 ms for as long
        # as the user holds the mouse.
        if _auto_prepare_timer_registered:
            bpy.app.timers.unregister(_run_auto_prepare)
            bpy.app.timers.register(
                _run_auto_prepare, first_interval=_DEFERRED_PREPARE_RETRY)
    _tag_prepare_redraw()


def _advance_prepare_task():
    global _prepare_timer_registered
    task = _prepare_task
    if task is None:
        _prepare_timer_registered = False
        return None

    worker = task.get("worker")
    if worker is not None:
        if worker.is_alive():
            return _PREPARE_POLL_INTERVAL
        try:
            token, succeeded, payload = task["result_queue"].get_nowait()
        except queue.Empty:
            return _PREPARE_POLL_INTERVAL
        task["worker"] = None
        if token != task["token"]:
            _finish_prepare_task(error="native worker token mismatch")
            return None
        if task["cancel_requested"]:
            _finish_prepare_task(cancelled=True)
            return None
        if succeeded:
            task["send_value"] = payload
            task["send_ready"] = True
        else:
            task["pending_exception"] = payload

    if task["cancel_requested"]:
        _finish_prepare_task(cancelled=True)
        return None

    try:
        event = _resume_prepare_task(task)
    except StopIteration as completed:
        task["generator_finished"] = True
        result = completed.value
        runner = task["runner"]
        # A deferred generator returns FINISHED without having built anything:
        # the mode switch it could not perform is the whole of what it did.
        if result == {'FINISHED'} and runner._prepare_deferred:
            _finish_prepare_task(deferred=True)
        elif result == {'FINISHED'} and not _teardown_failure:
            _finish_prepare_task(success=True)
        else:
            _finish_prepare_task(error=runner.last_report or "preparation failed")
        return None
    except BaseException as exc:
        _finish_prepare_task(error=f"Prepare transaction failed: {exc}")
        return None

    if not isinstance(event, dict):
        _finish_prepare_task(error="invalid preparation task event")
        return None
    scene = bpy.data.scenes.get(task["scene_name"])
    if scene is None:
        _finish_prepare_task(error="preparation scene no longer exists")
        return None
    kind = event.get("kind")
    _set_prepare_status(
        scene, "BUILDING" if kind == "native" else "PREPARING",
        event.get("progress", task["progress"]), event.get("message", ""))
    if kind == "progress":
        # A progress publish is a status update, not a work boundary: the next
        # generator step is main-thread work or the next native dispatch, and
        # both are ready now.  It used to ask for a fixed 10 ms, which was two
        # poll intervals at the old 50 ms cadence but a pure 5 ms of added
        # latency per publish at the current one - 13 publishes on the owner's
        # scene shape.  The tick is the same tick the worker wait uses, so a
        # publish still hands the frame back to Blender before the next segment
        # starts, and a callback that returns 0.0 would run at the frame rate.
        return _PREPARE_POLL_INTERVAL
    if kind != "native":
        _finish_prepare_task(error=f"unsupported preparation event {kind!r}")
        return None
    result_queue = queue.Queue(maxsize=1)
    worker = threading.Thread(
        name=f"GPUClothPrepare-{task['token']}",
        target=_run_prepare_native,
        args=(result_queue, task["token"], event["function"], event["args"]),
        daemon=False,
    )
    task["result_queue"] = result_queue
    task["worker"] = worker
    worker.start()
    return _PREPARE_POLL_INTERVAL


def _start_prepare_task(context, automatic=False):
    global _prepare_task, _prepare_timer_registered, _prepare_task_serial
    if _prepare_task is not None:
        return False
    scene = getattr(context, "scene", None)
    obj = getattr(context, "object", None) or getattr(
        context, "active_object", None)
    view_layer = getattr(context, "view_layer", None)
    if scene is None or obj is None or view_layer is None:
        return False
    _prepare_task_serial += 1
    modifier_state = []
    for candidate in tuple(getattr(scene, "objects", ())):
        for modifier in tuple(getattr(candidate, "modifiers", ())):
            if getattr(modifier, "type", None) == 'CLOTH':
                modifier_state.append((
                    modifier,
                    tuple(bool(getattr(modifier, attribute, False))
                          for attribute in _MODIFIER_VISIBILITY),
                ))
    runner = _PrepareRunner(scene.name_full)
    _prepare_task = {
        "token": _prepare_task_serial,
        "scene_name": scene.name_full,
        "object_name": obj.name_full,
        "view_layer_name": view_layer.name,
        "automatic": bool(automatic),
        "runner": runner,
        "generator": None,
        "generator_finished": False,
        "worker": None,
        "cancel_requested": False,
        "state": "QUEUED",
        "progress": 0,
        "original_mode": getattr(obj, "mode", None),
        "original_springs_built": bool(getattr(
            scene, "gpu_cloth_springs_built", False)),
        "modifier_state": modifier_state,
    }
    _set_prepare_status(scene, "QUEUED", 0, "GPUCloth preparation queued")
    try:
        bpy.app.timers.register(_advance_prepare_task, first_interval=0.0)
    except (AttributeError, RuntimeError):
        _finish_prepare_task(error="failed to register preparation timer")
        return False
    _prepare_timer_registered = True
    return True


def cancel_prepare_task():
    task = _prepare_task
    if task is None:
        return False
    task["cancel_requested"] = True
    scene = bpy.data.scenes.get(task["scene_name"])
    if scene is not None:
        _set_prepare_status(
            scene, "CANCELLING", task["progress"],
            "Cancelling GPUCloth preparation")
    worker = task.get("worker")
    if worker is None or not worker.is_alive():
        _finish_prepare_task(cancelled=True)
    return True


def _reject_unsupported_v3_owners(scene, cloth_objects):
    """Reject owners without a v3 ABI surface before native mutation."""
    unsupported = []
    helper = scene.gpu_cloth_helper
    for cloth_obj in cloth_objects:
        settings = cloth_obj.GPUCloth
        solver = str(getattr(settings, "solver_type", ""))
        if solver == "Mil2":
            # Reachable only from a project stored while Mil2 was offered - it
            # is not in `_solver_type_items` any more (properties.py:253).  The
            # stored value is deliberately left alone and the refusal is named
            # here, because the alternative is the old outcome: an offered
            # solver that reached GPUCLOTH_V3_BACKEND_ACCURACY and died in
            # GPUCloth_v3_cloth_create with ABI 4, naming no cause.
            unsupported.append(
                f"solver:{cloth_obj.name_full}:Mil2 (the Mil2 backend is not "
                "in this build, so Solver no longer lists it; set Solver to "
                "'PD')")
        elif solver != "PD":
            unsupported.append(
                f"solver:{cloth_obj.name_full}:{settings.solver_type}")
        if (bool(getattr(settings, "use_dynamic_mesh", False)) and
                str(getattr(settings, "shapekey_rest", ""))):
            unsupported.append(
                f"rest_shape_key_dynamic_mesh:{cloth_obj.name_full}")
        if str(getattr(settings, "bending_model", "")) not in {
                "LINEAR", "ANGULAR"}:
            # The SDB bending model was removed from the addon: under FABRIC -
            # the only material model - it takes the whole bending payload and
            # cannot coexist with the v3 triangle membrane, and it was never
            # selectable in the shipped panel.  A project stored while it was
            # offered still carries the string, so it is refused by name here,
            # before any native mutation, exactly like a stored 'Mil2' solver
            # above.  Nothing is remapped to a neighbour value.
            unsupported.append(
                f"bending_model:{cloth_obj.name_full}:"
                f"{settings.bending_model} (the SDB bending model is no longer "
                "offered; set Bending Model to 'Angular' or 'Linear')")
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


def _estimate_gpu_memory_bytes(simulation_objects, solver_types):
    """Estimate explicitly described persistent v3 buffers, in bytes."""
    if len(simulation_objects) != len(solver_types):
        raise RuntimeError("GPU memory estimate inputs are inconsistent")
    position_components = 3
    edge_endpoints = 2
    face_indices = 2
    loop_indices = 2
    solver_state_vectors = {"PD": 2, "Mil2": 3}
    total = 0
    for obj, solver in zip(simulation_objects, solver_types):
        if solver not in solver_state_vectors:
            raise RuntimeError(f"GPU memory estimate solver is unsupported: {solver}")
        mesh = obj.data
        vertices = len(mesh.vertices)
        edges = len(mesh.edges)
        faces = len(mesh.polygons)
        loops = len(mesh.loops)
        vertex_state = vertices * position_components * sizeof(c_float)
        topology = (
            edges * sizeof(c_uint) * edge_endpoints +
            faces * sizeof(c_uint) * face_indices +
            loops * sizeof(c_uint) * loop_indices)
        solver_state = vertex_state * solver_state_vectors[solver]
        total += vertex_state + topology + solver_state
    return int(total)


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


# One NVML session per Blender session.  The reading itself costs ~10 us, but
# loading the library and `nvmlInit_v2` costs 13-36 ms, which is a large part of
# what the `nvidia-smi` spawn this replaces costs - so the session is opened once
# and the device handle kept.  It owns no allocation, no CUDA context and no
# device state; it is a read-only driver query handle, and `nvmlShutdown` is left
# to process exit because the driver reference-counts it.  A refused attempt is
# latched, so a machine without NVML pays for the failure once instead of on
# every prepare.  Owner: the GPU memory preflight.  Retirement condition: none
# needed - it is replaced wholesale if the ABI grows a memory query of its own.
_nvml_session = {'usable': None, 'handle': None, 'get_memory': None}


def _nvml_memory_bytes():
    """Device-0 free/total bytes through NVML, or None if it is not usable.

    NVML is the library `nvidia-smi` itself reads, so this is the same reading
    through a cheaper door rather than a different check.  Every step is
    fail-closed: a missing library, a refused init, a refused device handle or a
    refused query returns None, and the caller then runs the subprocess query.
    """
    session = _nvml_session
    if session['usable'] is False:
        return None
    if session['usable'] is None:
        try:
            library = ctypes.WinDLL("nvml.dll")
            init = library.nvmlInit_v2
            get_handle = library.nvmlDeviceGetHandleByIndex_v2
            get_memory = library.nvmlDeviceGetMemoryInfo
            handle = ctypes.c_void_p()
            if (int(init()) != 0 or
                    int(get_handle(0, ctypes.byref(handle))) != 0):
                raise OSError("NVML refused initialisation")
        except (OSError, AttributeError, TypeError, ValueError):
            session['usable'] = False
            return None
        session['handle'] = handle
        session['get_memory'] = get_memory
        session['usable'] = True
    try:
        memory = _NvmlMemory()
        if int(session['get_memory'](
                session['handle'], ctypes.byref(memory))) != 0:
            return None
        return int(memory.free), int(memory.total)
    except (OSError, AttributeError, TypeError, ValueError):
        return None


def _nvidia_smi_memory_bytes():
    """Read device-0 free/total memory by spawning `nvidia-smi`."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        first = next(line for line in result.stdout.splitlines() if line.strip())
        free_text, total_text = (item.strip() for item in first.split(",", 1))
        free_mib = int(free_text)
        total_mib = int(total_text)
    except (FileNotFoundError, subprocess.CalledProcessError,
            StopIteration, ValueError):
        return None
    return free_mib * 1024 * 1024, total_mib * 1024 * 1024


def _query_gpu_memory_bytes():
    """Read device-0 free/total memory, with the subprocess as the fallback.

    `nvidia-smi` is a process spawn and measured 76.5 ms inside a prepare on the
    owner's scene shape; the in-process NVML query answers the same question in
    ~10 us once its session is open.  The subprocess stays as the fallback, so a
    machine where NVML is absent or refuses still gets the existing check rather
    than a skipped one, and both routes pass the same plausibility gate - a
    missing, unreadable or implausible reading is still a refusal.
    """
    available = _nvml_memory_bytes()
    if available is None:
        available = _nvidia_smi_memory_bytes()
    if available is None:
        return None
    free_bytes, total_bytes = available
    if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
        return None
    return free_bytes, total_bytes


def _require_gpu_memory_preflight(simulation_objects, solver_types):
    """Admit only a visible lower-bound estimate; native allocation remains oracle."""
    estimate = _estimate_gpu_memory_bytes(simulation_objects, solver_types)
    required = estimate
    available = _query_gpu_memory_bytes()
    if available is None:
        raise RuntimeError(
            "GPU memory preflight unavailable: driver free/total query "
            "failed; native allocation not attempted")
    free_bytes, total_bytes = available
    if free_bytes < required:
        raise RuntimeError(
            "GPU memory preflight rejected: "
            f"free={free_bytes} bytes, lower-bound={required} bytes, "
            f"total={total_bytes} bytes")
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "estimate_bytes": estimate,
        "lower_bound_bytes": required,
    }


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


def ensure_native_teardown(shutdown_runtime=False):
    """Resolve retained native owners before package registration mutation."""
    if prepare_task_active():
        cancel_prepare_task()
        if prepare_task_active():
            return False
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
                # Writing a flag its current value would still tag the
                # dependency graph, so only a real change is written.
                if bool(getattr(modifier, attribute, False)) != bool(value):
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
                visibility = _modifier_visibility(modifier)
                states.append((modifier, visibility))
                for attribute, value in zip(_MODIFIER_VISIBILITY, visibility):
                    if value:
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
    from . import cloth_settings_bridge
    plans = []
    for binding_index, binding in enumerate(bindings):
        cloth_obj = binding["render_object"]
        simulation_obj = binding["simulation_object"]
        modifiers = tuple(getattr(cloth_obj, "modifiers", ()))
        cpu_cloth_modifiers = cloth_settings_bridge.find_cpu_cloth_modifiers(
            cloth_obj)
        if len(cpu_cloth_modifiers) != 1:
            raise RuntimeError(
                f"{cloth_obj.name_full!r} requires exactly one Cloth "
                "modifier")
        cloth_modifier = cpu_cloth_modifiers[0]
        cloth_index = modifiers.index(cloth_modifier)
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


def _mesh_topology_arrays(mesh):
    """Bulk-read the four topology payloads, one ``foreach_get`` each.

    Reading the same values through ``mesh.edges`` / ``mesh.polygons`` /
    ``mesh.loops`` in Python builds roughly 130 000 single-element RNA reads per
    mesh and cost 237 ms per capture on a 128x128 grid.  Blender stores all four
    payloads as integers, so the bulk buffers carry exactly the values the
    element walk produced.
    """
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    loop_start = np.empty(len(mesh.polygons), dtype=np.int32)
    loop_total = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loop_vertex = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex)
    loop_edge = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("edge_index", loop_edge)
    return edges, loop_start, loop_total, loop_vertex, loop_edge


def _mesh_record_table(payload, count, words=4):
    """One fixed-width uint32 record per row: the payload as the ABI's record.

    The record is `words` uint32 words of which only the first two carry the
    payload, so the table is zero-filled exactly as the per-element form left it.
    `words` is not a detail: `GPUClothV3MeshEdge` and `GPUClothV3MeshFace` are
    four-word records, `GPUClothV3MeshCorner` is a **two**-word record, and
    handing a corner a four-word table interleaves two zero words into every
    record - a payload the native builder rejects rather than ignores.
    """
    table = np.zeros((count, words), dtype=np.uint32)
    if count:
        table[:, :2] = np.asarray(payload, dtype=np.uint32).reshape(count, 2)
    return table


def _mesh_edge_table(edges, source_mesh, edge_count):
    """The edge records, including the loose flag the ABI reads from word 2.

    `is_loose` is one boolean per edge; asking each `MeshEdge` for it built
    32 512 RNA wrappers and 32 512 attribute reads per capture.  The collection
    answers the same question as one boolean array, and the values are the same
    because both read the edge's own flag.  A mesh whose edges collection cannot
    answer is not silently treated as all-tight: the element walk is the
    fallback.
    """
    table = _mesh_record_table(edges, edge_count)
    source_edges = getattr(source_mesh, "edges", None)
    if source_edges is None:
        return table
    source_count = len(source_edges)
    if not source_count:
        return table
    loose_flag = int(CType.GPUCLOTH_V3_MESH_EDGE_LOOSE)
    limit = min(edge_count, source_count)
    try:
        loose = np.empty(source_count, dtype=bool)
        source_edges.foreach_get("is_loose", loose)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        for index in range(limit):
            if bool(getattr(source_edges[index], "is_loose", False)):
                table[index, 2] = loose_flag
        return table
    table[:limit, 2] = np.where(loose[:limit], loose_flag, 0)
    return table


def _mesh_topology_arrays_pair(first, second):
    """Stack two per-element int32 buffers into one (n, 2) int32 array.

    The payloads are the same two vectors `zip()` produced, in the same order,
    with the same values; they are kept as one contiguous array so the consumer
    can hand them to the ABI without walking 130 000 Python ints back out.
    """
    paired = np.empty((first.size, 2), dtype=np.int32)
    paired[:, 0] = first
    paired[:, 1] = second
    return paired


def _mesh_topology_from_arrays(vertex_count, arrays):
    """The capture's topology payloads, as the arrays the ABI receives.

    The previous form built 113 000 Python tuples here (23.7 ms at 128x128) and
    the v3 cloth create then converted every one of them back into an integer
    array.  Both readers of this dictionary - the create below and nothing else -
    take arrays, so the tuples were a pure round trip.  The values and their
    order are unchanged; the consumer's own conversion is what proves it.
    """
    edges, loop_start, loop_total, loop_vertex, loop_edge = arrays
    return {
        "vertex_count": int(vertex_count),
        "edges": edges.reshape(-1, 2),
        "polygons": _mesh_topology_arrays_pair(loop_start, loop_total),
        "loops": _mesh_topology_arrays_pair(loop_vertex, loop_edge),
    }


def _mesh_topology_matches(
        vertex_count, arrays, other_vertex_count, other_arrays):
    """Whether two topology array sets describe the same topology.

    This is ``_mesh_topology_from_arrays(a) == _mesh_topology_from_arrays(b)``
    decided on the arrays instead of on two materialised dictionaries.  The
    dictionaries hold exactly these five payloads as Python tuples plus the
    vertex count, so comparing the payloads field by field decides the same
    question while skipping roughly 130 000 tuple elements per side.
    """
    if int(vertex_count) != int(other_vertex_count):
        return False
    for values, other in zip(arrays, other_arrays):
        if values.shape != other.shape or not np.array_equal(values, other):
            return False
    return True


def _mesh_topology(mesh):
    """The same dictionary the element walk produced, filled from bulk reads."""
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    loop_start = np.empty(len(mesh.polygons), dtype=np.int32)
    loop_total = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loop_vertex = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex)
    loop_edge = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("edge_index", loop_edge)
    return {
        "vertex_count": len(mesh.vertices),
        "edges": tuple(map(tuple, edges.reshape(-1, 2).tolist())),
        "polygons": tuple(zip(loop_start.tolist(), loop_total.tolist())),
        "loops": tuple(zip(loop_vertex.tolist(), loop_edge.tolist())),
    }


def _capture_modifier_input_mesh(obj, depsgraph, triangulate=False):
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
        vertex_count = len(mesh.vertices)
        if mesh is None or vertex_count == 0:
            raise RuntimeError(
                f"Cloth input mesh for {obj.name_full!r} is empty")
        mesh_arrays = _mesh_topology_arrays(mesh)
        # The writable mesh's topology is read as arrays too, and compared as
        # arrays: building its dictionary only to compare against it cost
        # ~62 ms per capture for a question the payloads answer directly.
        if not _mesh_topology_matches(
                vertex_count, mesh_arrays,
                len(obj.data.vertices), _mesh_topology_arrays(obj.data)):
            raise RuntimeError(
                f"modifier input topology for {obj.name_full!r} differs "
                "from the writable simulation mesh")
        # The dictionary form is this function's return value on *both* paths -
        # it is where ``edges``, ``polygons`` and ``loops`` come from for the
        # legacy, non-triangulated capture - so it is built here, once, and the
        # triangulated branch below reuses it.  ``_mesh_topology_matches`` above
        # is what spares the *comparison* its dictionary; it does not spare the
        # return value its own.
        topology = _mesh_topology_from_arrays(vertex_count, mesh_arrays)
        # foreach_get fills every coordinate in one C loop and Blender stores
        # them as float32, so this buffer already holds exactly the values the
        # former per-vertex float() round trip produced: float() widens a float32
        # exactly and c_float() narrows it back to the same bits, which makes the
        # 49 152-element Python pass below an identity on already-validated data.
        # It was 21.2 ms of the call at 16 641 vertices plus ~5 ms to convert its
        # tuple back to an array, and the vectorised `np.isfinite` scan is the
        # check that actually decides - it is stricter than a per-element
        # `math.isfinite` on the widened value, and it still names the offending
        # vertex rather than only the array.
        coordinates = np.empty(vertex_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coordinates)
        non_finite = np.flatnonzero(~np.isfinite(coordinates))
        if non_finite.size:
            raise RuntimeError(
                f"Cloth input vertex {int(non_finite[0]) // 3} contains a "
                "non-finite float")
        # The only consumer of this payload reshapes it flat and copies it to
        # the ABI, so it is kept as the (vertex, 3) float32 array rather than
        # rebuilt into 16 384 Python three-tuples.  The bytes handed over are
        # unchanged: float32 C-contiguous (n, 3) lists the same components in
        # the same order as the tuple of triples did, so reshape(-1).tobytes()
        # is byte-identical, and the finiteness check above is untouched.
        positions = coordinates.reshape(vertex_count, 3)
        upload_triangles = None
        if triangulate:
            vcu.calc_mesh_loop_triangles(mesh)
            triangle_loops = np.empty(
                len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("loops", triangle_loops)
            if triangle_loops.size % 3:
                raise RuntimeError(
                    f"Fabric input mesh for {obj.name_full!r} has a "
                    "non-triangular loop triangle")
            loop_vertex = mesh_arrays[3]
            triangle_vertices = loop_vertex[triangle_loops].astype(np.int64)
            triangle_vertices = triangle_vertices.reshape(-1, 3)
            # Same payload the tuple form carried, kept as the ABI's (n, 2)
            # integer array: this branch is not on the measured drape's path,
            # but the create below reads both forms and the values are the ones
            # the former `list(topology["edges"])` held.
            upload_edges = topology["edges"].astype(np.int64)
            # Each triangle corner owns the edge from its vertex to the next
            # one.  The former per-corner Python lookup cost 166 ms at 128x128;
            # the same assignment is computed here with a sorted code per edge
            # in triangle-then-corner order, which preserves both the existing
            # edge indices and the first-seen order of the new ones.
            stride = np.int64(vertex_count + 1)
            following = np.roll(triangle_vertices, -1, axis=1)
            corner_codes = (
                np.minimum(triangle_vertices, following) * stride +
                np.maximum(triangle_vertices, following)).reshape(-1)
            existing = upload_edges
            edge_index = np.zeros(corner_codes.shape, dtype=np.int64)
            if existing.size:
                existing_codes = (
                    np.minimum(existing[:, 0], existing[:, 1]) * stride +
                    np.maximum(existing[:, 0], existing[:, 1]))
                order = np.argsort(existing_codes, kind="stable")
                sorted_codes = existing_codes[order]
                unique_codes, first_index, counts = np.unique(
                    sorted_codes, return_index=True, return_counts=True)
                # A duplicated mesh edge is owned by its last occurrence, which
                # is what the former ``edge_lookup`` dictionary kept.
                lookup_index = order[first_index + counts - 1]
                slot = np.searchsorted(unique_codes, corner_codes)
                found = slot < unique_codes.size
                found[found] &= (
                    unique_codes[slot[found]] == corner_codes[found])
                edge_index[found] = lookup_index[slot[found]]
            else:
                found = np.zeros(corner_codes.shape, dtype=bool)
            missing = np.flatnonzero(~found)
            if missing.size:
                unique_missing, first_missing = np.unique(
                    corner_codes[missing], return_index=True)
                first_seen = missing[first_missing]
                seen_order = np.argsort(first_seen, kind="stable")
                new_codes = unique_missing[seen_order]
                base = int(upload_edges.shape[0])
                # The new edge for a corner position is the corner's own vertex
                # and the vertex that follows it; the former loop read both back
                # through int() one element at a time in this same order.
                chosen = first_seen[seen_order]
                flat_vertices = triangle_vertices.reshape(-1)
                flat_following = following.reshape(-1)
                upload_edges = np.concatenate((
                    upload_edges,
                    np.stack((flat_vertices[chosen],
                              flat_following[chosen]), axis=1)), axis=0)
                # ``np.searchsorted`` needs an *ascending* haystack, and
                # ``new_codes`` is deliberately permuted into first-seen order so
                # the appended edge records come out in that order.  Searching
                # the permuted array is wrong wherever the permutation is not
                # monotonic, and it returns ``new_codes.size`` whenever the
                # needle exceeds the array's last element - which is how corners
                # came to name edge ``base + 510 == edge_count`` itself, and the
                # native owner validation rejected the whole payload as an
                # out-of-range index.  That is why the Cushion scene could not be
                # created at all while the Drape, whose grid happens to make the
                # permutation monotonic here, was unaffected.  Search the
                # ascending unique array - the same pattern the branch above
                # already uses - and map the hit through the permutation so the
                # appended record order is exactly what it was.
                slot = np.searchsorted(unique_missing, corner_codes[missing])
                position = np.empty(seen_order.size, dtype=np.int64)
                position[seen_order] = np.arange(
                    seen_order.size, dtype=np.int64)
                edge_index[missing] = base + position[slot]
            upload_triangles = np.stack(
                (triangle_vertices, edge_index.reshape(-1, 3)), axis=-1)
            topology = dict(topology)
            topology["edges"] = upload_edges
    finally:
        evaluated.to_mesh_clear()
    return {
        **topology,
        "positions": positions,
        "upload_triangles": upload_triangles,
        "capture_space": "CLOTH_INPUT_LOCAL",
    }


def _create_v3_cloth_owner(
        dll, cloth_obj, simulation_obj, mesh_snapshot, topology_generation,
        backend, geometry_generation=1):
    """Create one native v3 cloth; retain every caller-owned input buffer."""
    vertex_count = int(mesh_snapshot["vertex_count"])
    edges = mesh_snapshot["edges"]
    polygon_count = 0
    corner_count = 0
    upload_triangles = mesh_snapshot.get("upload_triangles")
    if upload_triangles is not None and len(upload_triangles):
        # A triangle's corners are the same payload the nested tuples carried,
        # already in (triangle, corner, {vertex, edge}) order: one face per
        # triangle starting at every third corner loop, and one corner record
        # per corner.  The former form rebuilt 96 000 tuples to say exactly
        # this, and the create then converted them back to arrays.
        polygon_count = len(upload_triangles)
        corner_count = polygon_count * 3
        face_codes = np.empty((polygon_count, 2), dtype=np.uint32)
        face_codes[:, 0] = np.arange(
            polygon_count, dtype=np.uint32) * np.uint32(3)
        face_codes[:, 1] = np.uint32(3)
        corner_codes_array = upload_triangles.reshape(-1, 2)
    else:
        face_codes = mesh_snapshot["polygons"]
        corner_codes_array = mesh_snapshot["loops"]
        polygon_count = len(face_codes)
        corner_count = len(corner_codes_array)
    edge_count = len(edges)
    if not edge_count or not polygon_count or not corner_count:
        raise RuntimeError(
            "v3 cloth create requires non-empty vertices, edges, faces, "
            "and corners")
    object_id = _blender_session_uid(
        simulation_obj, "v3 cloth simulation object")
    topology_generation = int(topology_generation)
    geometry_generation = int(geometry_generation)
    if topology_generation <= 0 or geometry_generation <= 0:
        raise RuntimeError("v3 cloth generations must be positive")

    positions = (c_float * (vertex_count * 3)).from_buffer_copy(
        np.asarray(mesh_snapshot["positions"],
                   dtype=np.float32).reshape(-1).tobytes())
    # The three mesh payloads are fixed-width uint32 records, so each is built
    # as one structured buffer and handed over with a single from_buffer_copy
    # instead of a per-element assignment loop with a per-field int().  They are
    # already integer arrays - the capture keeps them that way - so this is a
    # dtype cast into the record table, not a walk out of Python objects.
    edge_payload = (CType.GPUClothV3MeshEdge * edge_count) \
        .from_buffer_copy(
            _mesh_edge_table(edges, simulation_obj.data, edge_count).tobytes())
    face_payload = (CType.GPUClothV3MeshFace * polygon_count) \
        .from_buffer_copy(
            _mesh_record_table(face_codes, polygon_count).tobytes())
    corner_payload = (CType.GPUClothV3MeshCorner * corner_count) \
        .from_buffer_copy(
            _mesh_record_table(
                corner_codes_array, corner_count, words=2).tobytes())

    object_matrix = _matrix_signature(
        simulation_obj.matrix_world,
        f"{simulation_obj.name_full!r} world transform")
    inverse_matrix = _matrix_signature(
        _matrix_inverse(
            simulation_obj.matrix_world,
            f"{simulation_obj.name_full!r} world transform",
            require_rigid_transform=(
                int(backend) == CType.GPUCLOTH_V3_BACKEND_FAST)),
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
    config.edge_count = edge_count
    config.face_count = polygon_count
    config.corner_count = corner_count
    _set_buffer_view(
        config.positions, CType.GPUCLOTH_ELEMENT_FLOAT3, vertex_count,
        sizeof(c_float) * 3, addressof(positions), geometry_generation)
    _set_buffer_view(
        config.edges, CType.GPUCLOTH_ELEMENT_MESH_EDGE, edge_count,
        sizeof(CType.GPUClothV3MeshEdge), addressof(edge_payload),
        topology_generation)
    _set_buffer_view(
        config.faces, CType.GPUCLOTH_ELEMENT_MESH_FACE, polygon_count,
        sizeof(CType.GPUClothV3MeshFace), addressof(face_payload),
        topology_generation)
    _set_buffer_view(
        config.corners, CType.GPUCLOTH_ELEMENT_MESH_CORNER, corner_count,
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


# One solver identity per loaded binary, for the process's lifetime.  The
# identity is the sha256 of the DLL that will solve the frames, and hashing
# 30 MB on every prepare would be a new cost on a path that has none today, so
# stat is the cheap key and the digest is the expensive value.
_solver_identity_cache = {'key': None, 'bytes': b''}


def _solver_identity_bytes():
    """The bytes that name the solver: where it is, and exactly which bytes it is.

    A cache is only valid for the solver that filled it, and nothing about the
    addon's own inputs can tell two solvers apart - the same scene, the same
    settings and the same mesh produce different physics under a different
    ``GPUCloth.dll``.  Measured before this change: a range baked by the retired
    pre-fix build was replayed verbatim by a new process running the shipped
    binary, 250 of 250 frames with the engine's ``solve_count`` never moving and
    mesh hashes equal to the old build's own output, because the identity was
    ``hash(path + cache_index + cache_name)`` while the on-disk status metadata
    carried a ``source_generation`` the recreated owner restored as its own
    baseline.

    Content, not just a version stamp: an ABI version does not change when the
    solver behind it does, and every physics change in this repository has
    shipped without one.  The digest is folded together with the resolved path
    and the file's size and mtime, so the identity is stable across sessions for
    one binary and different for any rebuild - including a rebuild in place that
    the content hash alone would catch only after re-reading 30 MB.

    Returns ``b''`` when no DLL can be resolved (external playback against a
    machine that has no library, or a teardown that already dropped it).  That
    is the pre-existing identity, not an error: it can only *widen* which frames
    are reachable, and only for a session that has no solver to attribute them
    to in the first place.
    """
    global g_dll
    path = getattr(g_dll, "_name", None) if g_dll is not None else None
    if not path:
        try:
            path = vcu.get_dll_path("GPUCloth.dll")
        except (OSError, RuntimeError, AttributeError, TypeError):
            path = None
    if not path:
        return b''
    try:
        info = os.stat(path)
    except OSError:
        return b''
    key = (str(path), int(info.st_size), int(info.st_mtime_ns))
    if _solver_identity_cache['key'] == key:
        return _solver_identity_cache['bytes']
    digest = hashlib.sha256()
    try:
        with open(path, 'rb') as library:
            for block in iter(lambda: library.read(1 << 20), b''):
                digest.update(block)
    except OSError:
        return b''
    identity = (
        f"solver:{key[0]}:{key[1]}:{key[2]}:{digest.hexdigest()}"
    ).encode('utf-8')
    _solver_identity_cache['key'] = key
    _solver_identity_cache['bytes'] = identity
    return identity


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
    # The solver that will fill this cache is part of the cache's identity: a
    # range built by a different ``GPUCloth.dll`` is a different simulation, and
    # without this the new binary replays the old one's frames out of the status
    # metadata it finds on disk.  See ``_solver_identity_bytes``.
    solver_identity = _solver_identity_bytes()
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
    # Where the run the cache stores actually begins.  Frame 1 is the rest state
    # and is never solved - the frame path's own rule for a range's first frame is
    # `range_first = max(2, bake_start)` (`_frame_change_handler`) - so the declared
    # range has to agree with it.  It did not: `bake_start`'s RNA default is 1, the
    # range was declared as 1..end while no frame can be written at 1, and the
    # engine's own completion check (`s_v3_cache_status_update`,
    # BAKE_COMPLETE -> `Cache_v3_inspect_range` with `missing_frame_count != 0`)
    # therefore rejected every bake of the default range with
    # "typed cache status update 2 rejected with 9".  The product ABI contract test
    # configures its own cache at `frame_start = 2` for the same reason.
    #
    # An external cache is the one range this module does not own: its frames and
    # its status metadata were written by whoever produced the file, the engine
    # matches that metadata by an exact frame_start/end/step
    # (`Cache_read_status_metadata`), and a read-only cache has nothing to solve -
    # so there the user's declaration is passed through untouched.  Every other
    # storage mode is written by a run of this module, and a run begins where the
    # frame path says it begins.
    config.frame_start = (
        int(helper.bake_start) if helper.use_external_cache
        else max(2, int(helper.bake_start)))
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
        name_bytes + b'\0' + solver_identity)
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
        "solver_identity": bytes(solver_identity),
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


# The scene helper's live-published scalars, and the ABI feature each one feeds.
#
# These are published through three separate paths that share no table
# (`_configure_simulation_features`, `_runtime_update`, `_set_scene_gravity`), so
# the staged digest could not be cross-checked against a publish list that did
# not exist.  This tuple IS that list: `_scene_live_setting_values` reads through
# it, so a name here that stops being published changes the values the solver
# receives, and `_assert_live_settings_are_excluded` fails loudly if the staged
# stream stops excluding it.
_LIVE_FEATURE_SCENE_SETTINGS = (
    ("gravity_x", CType.GPUCLOTH_FEATURE_GRAVITY_VECTOR),
    ("gravity_y", CType.GPUCLOTH_FEATURE_GRAVITY_VECTOR),
    ("gravity_z", CType.GPUCLOTH_FEATURE_GRAVITY_VECTOR),
)


def _scene_live_setting_names():
    """Names of the scene-helper scalars published live."""
    return tuple(name for name, _feature in _LIVE_FEATURE_SCENE_SETTINGS)


def _scene_live_setting_values(helper):
    """Read the live-published scalars, in declaration order."""
    return tuple(float(getattr(helper, name))
                 for name in _scene_live_setting_names())


def _configure_simulation_features(
        dll, cloth_handle, scene, settings,
        object_id=None, topology_generation=None, geometry_generation=None,
        live_only=False):
    """Publish the simulation config block.

    ``live_only`` publishes only the features ``GPUCloth_v3_cloth_configure``
    accepts on a built owner.  VELOCITY_DAMPING and EFFECTOR_SCALES answer
    INVALID_STATE once the owner is built (main.cpp:10530, main.cpp:10574) and
    an AREAL mass is rejected outright (main.cpp:11245), so a live re-configure
    carrying them would abort part-way through the block.
    """
    solver_mask = {
        'PD': CType.GPUCLOTH_SOLVER_PD,
        'Mil2': CType.GPUCLOTH_SOLVER_MIL2,
    }.get(settings.solver_type)
    if solver_mask is None:
        raise RuntimeError(
            f"typed simulation config owns only PD/Mil2; got "
            f"{settings.solver_type}")
    if not live_only and (
            object_id is None or topology_generation is None or
            geometry_generation is None or int(object_id) == 0 or
            int(topology_generation) == 0 or int(geometry_generation) == 0):
        raise RuntimeError(
            "typed effector scales require nonzero cloth identity/generations")
    # FABRIC is the only material model, and it owns the areal mass contract:
    # the anisotropic triangle-membrane payload is always published, so the
    # effective mass mode is always AREAL and `mass_mode` no longer selects
    # anything.  `_effective_material_model` still refuses the one native
    # combination that cannot be built, before any payload is constructed.
    material_model = _effective_material_model(settings)
    effective_mass_mode = (
        'AREAL' if material_model == 'FABRIC' else
        getattr(settings, "mass_mode", "VERTEX"))
    if (effective_mass_mode == 'AREAL' and
            float(getattr(getattr(scene, "unit_settings", None),
                          "scale_length", 1.0)) != 1.0):
        raise RuntimeError(
            "Fabric density requires Scene Units > Unit Scale = 1: "
            "GPUCloth geometry and gravity currently use metres")
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
    config.gravity[:] = _scene_live_setting_values(scene.gpu_cloth_helper)
    config.air_damping = settings.air_viscosity
    config.velocity_damping = (
        cloth_settings_bridge.capture_v3_velocity_damping(settings))
    config.simulation_flags = 0
    config.reserved[:] = (0, 0)
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    simulation_features = [
        CType.GPUCLOTH_FEATURE_TIMESTEP_SPEED,
        CType.GPUCLOTH_FEATURE_GRAVITY_VECTOR,
        CType.GPUCLOTH_FEATURE_AIR_DAMPING,
    ]
    if not live_only:
        simulation_features.append(CType.GPUCLOTH_FEATURE_VELOCITY_DAMPING)
    for feature in simulation_features:
        config.header.feature_id = feature
        config.header.config_version = (
            2 if feature == CType.GPUCLOTH_FEATURE_VELOCITY_DAMPING else 1)
        result = int(dll.GPUCloth_v3_cloth_configure(cloth_handle, header))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(
                f"typed simulation feature {feature} rejected with {result}")

    mass_mode = effective_mass_mode
    mass = None
    if mass_mode == 'AREAL':
        if live_only:
            # The areal density payload is staged (main.cpp:11245); publishing
            # the per-vertex block instead would silently change the mass model.
            mass = None
        else:
            mass = CType.GPUClothArealMassConfig()
            mass.header.struct_size = sizeof(mass)
            mass.header.feature_id = CType.GPUCLOTH_FEATURE_MATERIAL_MASS
            mass.header.config_version = 2
            mass.solver_mask = solver_mask
            mass.density_kg_m2 = float(settings.fabric_density) * 0.001
    elif mass_mode == 'VERTEX':
        mass = CType.GPUClothSimulationConfig.from_buffer_copy(config)
        mass.header.feature_id = CType.GPUCLOTH_FEATURE_MATERIAL_MASS
        mass.header.config_version = 1
    else:
        raise RuntimeError(f"unsupported mass mode: {mass_mode}")
    if mass is not None:
        result = int(dll.GPUCloth_v3_cloth_configure(
            cloth_handle,
            cast(pointer(mass), POINTER(CType.GPUClothFeatureConfigHeader))))
        if result != CType.GPUCLOTH_ABI_OK:
            raise RuntimeError(f"typed material mass rejected with {result}")

    quality = CType.GPUClothQualityConfig()
    quality.header.struct_size = sizeof(quality)
    quality.header.feature_id = (
        CType.GPUCLOTH_FEATURE_SIMULATION_QUALITY)
    quality.header.config_version = 2
    quality.solver_mask = solver_mask
    quality.quality_steps = int(settings.quality_step)
    quality.time_scale = config.time_scale
    quality.vertex_mass = config.vertex_mass
    quality.gravity[:] = config.gravity
    quality.air_damping = config.air_damping
    quality.velocity_damping = config.velocity_damping
    quality.simulation_flags = 0
    quality.simulation_reserved[:] = (0, 0)
    quality.solver_iterations = int(settings.solver_iterations)
    quality.reserved = 0
    quality.solver_krylov_iterations = int(settings.solver_krylov_iterations)
    result = int(dll.GPUCloth_v3_cloth_configure(
        cloth_handle,
        cast(pointer(quality), POINTER(CType.GPUClothFeatureConfigHeader))))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(
            "typed simulation quality feature rejected with "
            f"{result}")

    if live_only:
        return CType.GPUCLOTH_ABI_OK
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
        expected_config_version = 3 if name == "anisotropy" else 2 if name in (
            "velocity_damping", "constraint_network",
            "simulation_quality", "material_mass") else 1
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
        print(f"[GPUCloth] native preparation witness: {detail}")
        if witness_result == CType.GPUCLOTH_ABI_OK and witness.invariant:
            reason = _INVARIANT_NAMES.get(
                int(witness.invariant), "UNKNOWN").replace("_", " ").lower()
            raise RuntimeError(
                f"cloth preparation rejected: {reason}; "
                f"faces {witness.triangle_i}/{witness.triangle_j}, "
                f"vertex {witness.vertex_i}")
        raise RuntimeError(f"native hard preflight rejected cloth: {detail}")
    return status


_ABI_RESULT_NAMES = {
    CType.GPUCLOTH_ABI_INVALID_ARGUMENT: "INVALID_ARGUMENT",
    CType.GPUCLOTH_ABI_STRUCT_TOO_SMALL: "STRUCT_TOO_SMALL",
    CType.GPUCLOTH_ABI_UNKNOWN_FEATURE: "UNKNOWN_FEATURE",
    CType.GPUCLOTH_ABI_UNSUPPORTED: "UNSUPPORTED",
    CType.GPUCLOTH_ABI_NOT_CONFIGURABLE: "NOT_CONFIGURABLE",
    CType.GPUCLOTH_ABI_VERSION_MISMATCH: "VERSION_MISMATCH",
    CType.GPUCLOTH_ABI_COUNT_MISMATCH: "COUNT_MISMATCH",
    CType.GPUCLOTH_ABI_INVALID_VALUE: "INVALID_VALUE",
    CType.GPUCLOTH_ABI_INVALID_STATE: "INVALID_STATE",
    CType.GPUCLOTH_ABI_INVALID_HANDLE: "INVALID_HANDLE",
    CType.GPUCLOTH_ABI_ALREADY_EXISTS: "ALREADY_EXISTS",
    CType.GPUCLOTH_ABI_BUFFER_TOO_SMALL: "BUFFER_TOO_SMALL",
    CType.GPUCLOTH_ABI_BACKEND_UNAVAILABLE: "BACKEND_UNAVAILABLE",
    CType.GPUCLOTH_ABI_SOLVE_FAILED: "SOLVE_FAILED",
    CType.GPUCLOTH_ABI_INTERNAL_ERROR: "INTERNAL_ERROR",
}


def _abi_result_name(code):
    """Name one ABI result.  An unknown code names its number, never a dash."""
    value = int(code)
    if value == CType.GPUCLOTH_ABI_OK:
        return "OK"
    return _ABI_RESULT_NAMES.get(value, f"ABI_{value}")


# Which check inside the native drape sandbox refused the call.  The narrow
# refusal paths deliberately leave the invariant witness at its reset value
# (`GPUCLOTH_INVARIANT_NONE`), so the witness alone cannot say why `Begin` was
# rejected: this table is the diagnosis for those paths, and it is also what
# keeps the string "NONE" out of a message whose whole job is to be a reason.
# `wire` is the field the native status carries for that check, so the recorded
# number can be checked against the boundary it violated rather than trusted.
_BEGIN_DRAPE_STEP_REASONS = {
    CType.GPUCLOTH_ABI_OK: (
        "sandbox accepted",
        "the native sandbox accepted the drape",
    ),
    CType.GPUCLOTH_ABI_INVALID_ARGUMENT: (
        "config",
        "the drape configuration was refused before any state was read",
    ),
    CType.GPUCLOTH_ABI_STRUCT_TOO_SMALL: (
        "config.struct_size",
        "the configuration or status struct is smaller than the native "
        "contract requires",
    ),
    CType.GPUCLOTH_ABI_VERSION_MISMATCH: (
        "config.config_version",
        "the configuration or status struct version does not match the native "
        "contract",
    ),
    CType.GPUCLOTH_ABI_INVALID_STATE: (
        "accepted_generation",
        "the sandbox is not in a startable state: the prepared owner is not "
        "runnable, a drape sandbox is already open, the cloth arrays are gone, "
        "or a frame has already been accepted since the last Prepare.  The "
        "'accepted' counter in the Prepare row is this number, and Prepare "
        "resets it to 0",
    ),
    CType.GPUCLOTH_ABI_INVALID_VALUE: (
        "drape_flags/max_steps/convergence_window/tolerance",
        "an argument the native sandbox validates exactly was refused: one of "
        "the fixed step budget, the fixed convergence window, a non-zero "
        "reserved field, or a tolerance that is not finite and equal to the "
        "solver's own",
    ),
}

_BEGIN_DRAPE_WITNESS_STEP = "hard preflight"

# The drape *step* has one narrow refusal of its own: the sandbox is not in a
# state that accepts another step (no live sandbox, already converged, or the
# cloth arrays are gone).  Every other refusal comes back through the solver's
# own verdict, which does fill the invariant witness in.
_STEP_DRAPE_STEP_REASONS = {
    CType.GPUCLOTH_ABI_OK: (
        "sandbox accepted",
        "the native sandbox accepted the drape step"),
    CType.GPUCLOTH_ABI_INVALID_ARGUMENT: (
        "config",
        "the step call was refused before any state was read"),
    CType.GPUCLOTH_ABI_STRUCT_TOO_SMALL: (
        "status.struct_size",
        "the status struct is smaller than the native contract requires"),
    CType.GPUCLOTH_ABI_VERSION_MISMATCH: (
        "status.status_version",
        "the status struct version does not match the native contract"),
    CType.GPUCLOTH_ABI_INVALID_STATE: (
        "sandbox state",
        "the sandbox will not take another step: it is not live, it has "
        "already converged, or the cloth arrays are gone; the solver's own "
        "verdict for the refused step is in the invariant row",
    ),
}


def _invariant_witness_detail(witness):
    """The primitive a broken invariant names, or None when it names none.

    A witness that carries an invariant but no primitive is still a diagnosis -
    the invariant name is the reason - so this returns None rather than a
    placeholder, and the message simply carries no primitive detail.
    """
    primitives = (witness or {}).get("primitives") or {}
    parts = []
    if primitives.get("triangle_i", -1) >= 0:
        parts.append(f"triangle {primitives['triangle_i']}"
                     + (f"/{primitives['triangle_j']}"
                        if primitives.get("triangle_j", -1) >= 0 else ""))
    if primitives.get("vertex_i", -1) >= 0:
        parts.append(f"vertex {primitives['vertex_i']}")
    if primitives.get("edge_i", -1) >= 0:
        parts.append(f"edge {primitives['edge_i']}")
    layers = (witness or {}).get("layers") or [0, 0]
    if layers[0] or layers[1]:
        parts.append(f"layers {layers[0]}/{layers[1]}")
    return ", ".join(parts) if parts else None


def drape_refusal_message(operation, result, status, step, witness, detail=None):
    """One sentence naming what refused a drape action, and with what.

    ``operation`` is the user's action ("Drape Begin"), ``status`` the native
    drape status returned by the same call, ``step`` the name of the check the
    ABI result maps to, ``witness`` the invariant witness dict the native side
    filled in for this refusal, and ``detail`` what that check tests.

    The invariant is kept out of the sentence when it is `NONE`: a reset witness
    is the *absence* of a diagnosis, and printing it as one is what made the
    original defect unreadable.  When an invariant *is* recorded it leads the
    sentence, because it is the native side's own answer and the step name is
    then only the classification around it.
    """
    invariant_name = (witness or {}).get("invariant_name", "NONE")
    parts = []
    if invariant_name and invariant_name != "NONE":
        parts.append(f"{operation} rejected on invariant {invariant_name}")
        if detail is None:
            # The witness that named the invariant usually names the primitive
            # too; carrying it is what turns "an invariant broke" into "this
            # triangle broke it".
            detail = _invariant_witness_detail(witness)
    else:
        parts.append(f"{operation} rejected: {step}")
    parts[0] += f" [{_abi_result_name(result)}]"
    if detail:
        parts.append(detail)
    if status is not None:
        parts.append(
            f"step {status.get('step_count', 0)}, "
            f"L-inf {float(status.get('maximum_position_delta', 0.0)) * 1000.0:.3f} "
            f"mm/frame, result {int(status.get('result', 0))}")
    return "; ".join(parts)


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
        # Additive W2 diagnostic: selected joint-bank stable-orientation
        # conflicts for the last solved frame (0 = none reported).
        "joint_orientation_conflicts": int(status.joint_orientation_conflicts),
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
    """Return the float32 the ABI receives, after an inclusive bound check.

    The bounds are rounded to float32 before the comparison because float32 is
    the precision of this channel at both ends: Blender stores every
    ``FloatProperty`` value *and* its ``min``/``max`` as C ``float``, and the
    native guards compare the same C ``float`` fields.  A float64 bound that is
    not float32-representable therefore refuses the value it declares -- the
    nearest float32 to ``pi/4`` is 2.19e-08 above it, so
    ``internal_spring_max_diversion``, whose declared default and maximum are
    both ``pi/4``, was rejected at a value Blender itself produced.  Rounding
    the bound makes the accepted set equal to the set the channel can carry,
    and is the identity for every bound that is already float32-exact.
    """
    try:
        source = float(value)
        converted = float(c_float(source).value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not a float32 value") from exc
    if not math.isfinite(source) or not math.isfinite(converted):
        raise RuntimeError(f"{label} is not finite")
    try:
        lower = float(c_float(lower).value)
        upper = float(c_float(upper).value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} has a non-float32 bound") from exc
    if source < lower or source > upper:
        raise RuntimeError(f"{label} is outside [{lower:g}, {upper:g}]")
    return converted


def _effective_bending_model(settings):
    """The bending model of one owner, with SDB already removed.

    The addon offers ``LINEAR`` and ``ANGULAR`` only (properties.py:311).  The
    third value, ``SDB``, is gone from the enum, so the string can only arrive
    from a project stored while it was offered; it is refused here by name
    rather than remapped, and ``_reject_unsupported_v3_owners`` refuses it one
    step earlier, in the panel, before any native mutation.
    """
    solver_type = str(getattr(settings, "solver_type", ""))
    bending_model = str(getattr(settings, "bending_model", ""))
    if solver_type not in {'PD', 'Mil2'}:
        raise RuntimeError(f"unknown solver type {solver_type!r}")
    if bending_model == 'SDB':
        raise RuntimeError(
            "NOT_CONFIGURABLE: the SDB bending model was removed - under "
            "FABRIC, the only material model, it takes the whole bending "
            "payload the v3 triangle membrane owns. Set Bending Model to "
            "'Angular' or 'Linear'.")
    if bending_model not in {'LINEAR', 'ANGULAR'}:
        raise RuntimeError(f"unknown bending model {bending_model!r}")
    return bending_model


def _effective_material_model(settings):
    """The one material model: ``FABRIC``.

    ``FABRIC`` owns the anisotropic v3 payload
    (``GPUCLOTH_FEATURE_ANISOTROPY`` with ``config_version = 3`` and
    ``GPUCLOTH_MATERIAL_TRIANGLE_MEMBRANE``) unconditionally, and that is what
    ``_capture_material_features`` publishes.  There is no second model to
    demote to: a silent demotion is forbidden, and the Legacy model is gone.

    This function has no bending statement to make.  The only bending model
    that ever competed with FABRIC for the v3 payload was ``SDB``, and it is no
    longer part of the addon: ``_effective_bending_model`` above refuses the
    stored string, ``_reject_unsupported_v3_owners`` (operators.py:1044-1056)
    refuses it in the panel before any native mutation, and no publisher can
    emit ``GPUCLOTH_FEATURE_BENDING_SDB`` any more.  The solver's own
    matrix-free SDB route (``PD_PCG_REPORT_ROUTE_SDB``, ``route=2``) is not this
    addon option and is untouched.

    ``solver_type == 'Mil2'`` has no physical membrane backend at all.  It is
    refused before this point by ``_reject_unsupported_v3_owners``, and this
    function does not need a Mil2 statement of its own: the product build ships
    no Mil2 backend, and the solver list no longer offers one.
    """
    return "FABRIC"


def _capture_material_features(settings, material_coordinates):
    config = CType.GPUClothMaterialConfig()
    config.header.struct_size = sizeof(config)
    solver_type = str(getattr(settings, "solver_type", "PD"))
    fabric = _effective_material_model(settings) == "FABRIC"
    bending_model = _effective_bending_model(settings)
    config.bending_model = 1 if bending_model == 'ANGULAR' else 0

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
    damping_values = (
        (settings.tension_damp, "tension damping", 0.0, 50.0),
        (settings.compression_damp, "compression damping", 0.0, 50.0),
        (settings.shear_damp, "shear damping", 0.0, 50.0),
    )
    stiffness = (0.0, 0.0, _bounded_float32(
        getattr(settings, "fabric_shear_c66", 500.0),
        "fabric shear C66", 0.0, float("inf")), stiffness[3])
    stiffness_max = (0.0, 0.0, _bounded_float32(
        getattr(settings, "fabric_shear_c66_max", 500.0),
        "maximum fabric shear C66", 0.0, float("inf")), stiffness_max[3])
    if stiffness_max[2] < stiffness[2]:
        raise RuntimeError(
            "maximum fabric shear C66 is below its base value")
    damping_values = (
        (getattr(settings, "fabric_tensile_damping", 5.0),
         "fabric tensile damping", 0.0, float("inf")),
        (getattr(settings, "fabric_compression_damping", 5.0),
         "fabric compression damping", 0.0, float("inf")),
        (getattr(settings, "fabric_shear_damping", 1.0),
         "fabric shear damping", 0.0, float("inf")),
    )
    config.stiffness[:] = stiffness
    config.stiffness_max[:] = stiffness_max
    config.damping[:] = tuple(
        _bounded_float32(value, label, lower, upper)
        for value, label, lower, upper in damping_values + (
            (settings.bending_damping, "bending damping", 0.0, 1000.0),))

    coordinate_payload = None
    if fabric or bool(getattr(settings, "use_anisotropy", False)):
        if material_coordinates is None:
            raise RuntimeError(
                "anisotropy material coordinates were not captured")
        if fabric:
            directional_values = (
                (getattr(settings, "fabric_tensile_u", 10000.0), "fabric tensile U"),
                (getattr(settings, "fabric_tensile_v", 10000.0), "fabric tensile V"),
                (getattr(settings, "fabric_compression_u", 10000.0), "fabric compression U"),
                (getattr(settings, "fabric_compression_v", 10000.0), "fabric compression V"),
                (settings.bending_stiffness, "fabric bending U"),
                (settings.bending_stiffness, "fabric bending V"),
            )
            directional_max_values = (
                (getattr(settings, "fabric_tensile_u_max", 10000.0), "maximum fabric tensile U"),
                (getattr(settings, "fabric_tensile_v_max", 10000.0), "maximum fabric tensile V"),
                (getattr(settings, "fabric_compression_u_max", 10000.0), "maximum fabric compression U"),
                (getattr(settings, "fabric_compression_v_max", 10000.0), "maximum fabric compression V"),
                (settings.max_bend, "maximum fabric bending U"),
                (settings.max_bend, "maximum fabric bending V"),
            )
        else:
            directional_values = tuple(zip((
                settings.tension_u, settings.tension_v,
                settings.compression_u, settings.compression_v,
                settings.bending_u, settings.bending_v), (
                    "tension U stiffness", "tension V stiffness",
                    "compression U stiffness", "compression V stiffness",
                    "bending U stiffness", "bending V stiffness")))
            directional_max_values = tuple(zip((
                settings.max_tension_u, settings.max_tension_v,
                settings.max_compression_u, settings.max_compression_v,
                settings.max_bend_u, settings.max_bend_v), (
                    "maximum tension U stiffness", "maximum tension V stiffness",
                    "maximum compression U stiffness", "maximum compression V stiffness",
                    "maximum bending U stiffness", "maximum bending V stiffness")))
        directional = tuple(
            _bounded_float32(value, label, 0.0, float("inf"))
            for value, label in directional_values
        )
        directional_max = tuple(
            _bounded_float32(value, label, 0.0, float("inf"))
            for value, label in directional_max_values
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
        config.material_flags = CType.GPUCLOTH_MATERIAL_ANISOTROPY_ENABLED
        if fabric:
            config.material_flags |= CType.GPUCLOTH_MATERIAL_TRIANGLE_MEMBRANE

        # One byte copy of the payload the capture already holds as a
        # C-contiguous float32 array.  Splatting it through the variadic
        # constructor boxed every component into a Python float only to unbox it
        # again: on the owner's scene shape that is 193 548 components, measured
        # at 30.1 ms through the tuple against 0.54 ms here, and the two buffers
        # are bit-identical (`GPUClothMaterialConfig.material_coordinates` is
        # declared FLOAT2, i.e. two float32 words per element, which is exactly
        # the interleaving the array carries).
        components = material_coordinates.coordinates
        dimension_count = (
            material_coordinates.corner_count if fabric
            else material_coordinates.vertex_count)
        if len(components) != dimension_count * 2:
            raise RuntimeError(
                "material-coordinate payload is not two components per "
                "element")
        coordinate_payload = (c_float * len(components)).from_buffer_copy(
            components.tobytes())
        _set_buffer_view(
            config.material_coordinates,
            CType.GPUCLOTH_ELEMENT_FLOAT2,
            dimension_count,
            sizeof(c_float) * 2,
            addressof(coordinate_payload),
            material_coordinates.topology_generation)

    # FABRIC is the only material model, so the triangle membrane and its
    # per-corner coordinates are always the payload's owner.  `fabric` is kept
    # as a named bool because it decides three separate things below (which
    # directional stiffness slots are published, whether the v3 triangle flag is
    # set, and whether the coordinate stream is per corner or per vertex), and
    # `use_anisotropy` only widens the audience of an already-published payload.
    return {
        "config": config,
        "coordinates": coordinate_payload,
        "bending_model": bending_model,
        "anisotropy": fabric or bool(getattr(settings, "use_anisotropy", False)),
        "fabric": fabric,
        "legacy_tension_damp": getattr(settings, "tension_damp", 0.0),
        "legacy_compression_damp": getattr(settings, "compression_damp", 0.0),
        "legacy_shear_damp": getattr(settings, "shear_damp", 0.0),
        "legacy_bending_damping": getattr(settings, "bending_damping", 0.0),
    }


def _publish_material_features(
        dll, cloth_handle, owner, publish_anisotropy=False, live_only=False):
    source = owner["config"]
    if publish_anisotropy:
        if not owner["anisotropy"]:
            return CType.GPUCLOTH_ABI_OK
        features = (CType.GPUCLOTH_FEATURE_ANISOTROPY,)
    else:
        features = [CType.GPUCLOTH_FEATURE_MATERIAL_DAMPING]
        if not owner.get("fabric", False):
            features[0:0] = [
                CType.GPUCLOTH_FEATURE_STRETCH,
                CType.GPUCLOTH_FEATURE_COMPRESSION,
                CType.GPUCLOTH_FEATURE_SHEAR,
            ]
        # A table, not a fallback chain: `_effective_bending_model` guarantees
        # one of these two strings, and an unexpected one must fail here rather
        # than be published as LINEAR.
        bending_feature = {
            'ANGULAR': CType.GPUCLOTH_FEATURE_BENDING_ANGULAR,
            'LINEAR': CType.GPUCLOTH_FEATURE_BENDING_LINEAR,
        }[owner["bending_model"]]
        features.insert(len(features) - 1, bending_feature)
    if live_only:
        features = [feature for feature in features
                    if feature in _LIVE_MATERIAL_FEATURES]

    for feature in features:
        config = CType.GPUClothMaterialConfig.from_buffer_copy(bytes(source))
        config.header.feature_id = feature
        if feature == CType.GPUCLOTH_FEATURE_ANISOTROPY:
            config.header.config_version = 3 if owner.get("fabric", False) else 2
            config.material_flags = CType.GPUCLOTH_MATERIAL_ANISOTROPY_ENABLED
            if owner.get("fabric", False):
                config.material_flags |= CType.GPUCLOTH_MATERIAL_TRIANGLE_MEMBRANE
        else:
            config.header.config_version = 1
            config.material_flags = 0
            config.directional_stiffness[:] = (0.0,) * 6
            config.directional_stiffness_max[:] = (0.0,) * 6
            config.material_coordinates = CType.GPUClothBufferView()
            if feature == CType.GPUCLOTH_FEATURE_MATERIAL_DAMPING:
                config.stiffness[:] = (0.0,) * 4
                config.stiffness_max[:] = (0.0,) * 4
            else:
                owned_index = {
                    CType.GPUCLOTH_FEATURE_STRETCH: 0,
                    CType.GPUCLOTH_FEATURE_COMPRESSION: 1,
                    CType.GPUCLOTH_FEATURE_SHEAR: 2,
                    CType.GPUCLOTH_FEATURE_BENDING_LINEAR: 3,
                    CType.GPUCLOTH_FEATURE_BENDING_ANGULAR: 3,
                }[feature]
                owned_value = config.stiffness[owned_index]
                owned_max = config.stiffness_max[owned_index]
                config.stiffness[:] = (0.0, 0.0, 0.0, 0.0)
                config.stiffness_max[:] = (0.0, 0.0, 0.0, 0.0)
                config.stiffness[owned_index] = owned_value
                config.stiffness_max[owned_index] = owned_max
                config.damping[:] = (0.0,) * 4
            if (feature == CType.GPUCLOTH_FEATURE_MATERIAL_DAMPING and
                    owner.get("fabric", False)):
                # Fabric damping lives in the v3 triangle-membrane payload.
                # The retained legacy feature receives only its legacy fields.
                config.damping[:] = (
                    _bounded_float32(owner["legacy_tension_damp"],
                                     "legacy tension damping", 0.0, 50.0),
                    _bounded_float32(owner["legacy_compression_damp"],
                                     "legacy compression damping", 0.0, 50.0),
                    _bounded_float32(owner["legacy_shear_damp"],
                                     "legacy shear damping", 0.0, 50.0),
                    _bounded_float32(owner["legacy_bending_damping"],
                                     "legacy bending damping", 0.0, 1000.0))
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


# Constraint features the engine accepts on an owner that is already built.
# `GPUCLOTH_FEATURE_INTERNAL_SPRINGS` is absent for the reason main.cpp:7641
# gives: internal-spring enablement changes the spring topology that
# `BuildClothSprings` assembles, so a late update would mutate the host settings
# and leave the accepted operator unchanged.  It answers INVALID_STATE once a
# cloth manager or a spring array exists, which a built owner always has.
# Sewing travels in the same payload and is published with it.
_LIVE_CONSTRAINT_FEATURES = frozenset((
    CType.GPUCLOTH_FEATURE_SEWING,
))

# Pressure features the engine accepts on a built owner.  All three of
# `s_configure_pressure_feature`'s cases write `settings->` and return OK
# (main.cpp:7877-7913), so no state gate applies to them.
_LIVE_PRESSURE_FEATURES = frozenset((
    CType.GPUCLOTH_FEATURE_PRESSURE_UNIFORM,
    CType.GPUCLOTH_FEATURE_FLUID_DENSITY,
    CType.GPUCLOTH_FEATURE_PRESSURE_VOLUME,
))


def _publish_internal_springs_config(
        dll, cloth_handle, prepared_config, live_only=False):
    if live_only and prepared_config.header.feature_id not in _LIVE_CONSTRAINT_FEATURES:
        return CType.GPUCLOTH_ABI_OK
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


def _publish_pressure_features(dll, cloth_handle, prepared, live_only=False):
    if prepared is None:
        return CType.GPUCLOTH_ABI_OK
    prepared_config, features = prepared
    config = CType.GPUClothPressureConfig.from_buffer_copy(
        bytes(prepared_config))
    header = cast(
        pointer(config), POINTER(CType.GPUClothFeatureConfigHeader))
    for feature in features:
        if live_only and feature not in _LIVE_PRESSURE_FEATURES:
            continue
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
    vcu.calc_mesh_loop_triangles(mesh)
    triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.uint32)
    mesh.loop_triangles.foreach_get("vertices", triangles)
    return _topology_generation(
        _blender_session_uid(simulation_obj, "pin simulation object"),
        len(mesh.vertices), triangles)


def _capture_pin_snapshot(
        settings_owner, simulation_obj, depsgraph, topology_generation,
        frame_generation, channels=None, witness=None):
    return capture_evaluated_pin_snapshot(
        simulation_obj, depsgraph, settings_owner.GPUCloth,
        topology_generation, frame_generation,
        expected_vertex_count=len(simulation_obj.data.vertices),
        identity_obj=simulation_obj, channels=channels, witness=witness)


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
    """Hash one triangle payload.

    ``int(index).to_bytes(4, "little", signed=False)`` per corner and a
    little-endian uint32 buffer of the same indices are the same byte stream, so
    accepting an already-flattened index buffer costs nothing in digest
    stability.  Nested ``(a, b, c)`` sequences are flattened in the same
    triangle-then-corner order the former per-index loop used.
    """
    hasher = hashlib.blake2b(digest_size=8, person=b"GPUTopology")
    hasher.update(int(object_id).to_bytes(8, "little", signed=False))
    hasher.update(int(vertex_count).to_bytes(8, "little", signed=False))
    hasher.update(
        np.ascontiguousarray(triangles, dtype='<u4').reshape(-1).tobytes())
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
    """Validate and convert one float sequence, one element at a time.

    The per-element loop is what makes this report *which* element was bad,
    which a vectorised scan cannot; callers that only need the decision use
    ``np.isfinite`` on the buffer instead.  ``float()`` accepts a numpy float32
    and ``c_float()`` stores the bits an already-float32 value already has, so
    the loop is value-identical over a numpy buffer and over a Python list.
    """
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

    The three payloads this reads - the canonical triangles, the canonical
    coordinates, and the two vertex-group dictionaries - were element walks.
    On the isolated 5 048-vertex / 10 092-triangle collider they measured
    13.5 ms, 9.8 ms and 85.4 ms per call; the same reads through
    ``foreach_get`` and the same score computed from the two dictionaries
    measured 2.2 ms, 0.12 ms and 5.9 ms.  ``MeshVertex.groups`` has no bulk
    reader, so the deform-layer walk itself survives; what is removed is the
    per-corner ``MeshVertex`` RNA lookup and the per-assignment
    ``VertexGroupElement`` read paid for every corner whether or not any group
    was a bone at all.  With no armature - the shipped fixtures - the group
    dictionaries stay empty and no scan runs at all.
    """
    if getattr(source, "type", None) != 'MESH':
        return None
    try:
        mesh = source.data
        mesh.calc_loop_triangles()
        canonical_triangles = np.empty(
            len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", canonical_triangles)
        canonical_triangles = canonical_triangles.reshape(-1, 3)
        if (len(mesh.vertices) != int(vertex_count) or
                tuple(map(tuple, canonical_triangles.tolist())) !=
                tuple(evaluated_triangles)):
            return None
        # One bulk read replaces three per-vertex RNA reads (co.x/co.y/co.z
        # through a generator) plus a float()/c_float() pair per component.
        # Blender stores coordinates as float32 and the bulk buffer is float32,
        # so the stored values are the same values that round trip produced.
        canonical = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", canonical)
        non_finite = np.flatnonzero(~np.isfinite(canonical))
        if non_finite.size:
            raise RuntimeError(
                f"collider {source.name_full!r} canonical vertices contain a "
                "non-finite float")

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

        scores = _collider_vertex_group_scores(mesh, bone_group_indices)
        triangle_keys = []
        for triangle in canonical_triangles:
            score = {}
            for vertex_index in triangle.tolist():
                entry = scores.get(int(vertex_index))
                if entry is None:
                    continue
                for group, weight in entry:
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

        groups_np = np.asarray(triangle_groups, dtype='<u4')
        hasher = hashlib.blake2b(digest_size=8, person=b"GPUCollCert")
        hasher.update(int(base_generation).to_bytes(8, "little", signed=False))
        hasher.update(canonical.tobytes(order='C'))
        hasher.update(groups_np.tobytes(order='C'))
        topology_generation = int.from_bytes(
            hasher.digest(), "little") or 1
        return {
            "canonical": tuple(float(value) for value in canonical),
            "triangle_groups": triangle_groups,
            "group_count": len(unique_keys),
            "topology_generation": topology_generation,
        }
    except (
            AttributeError, ReferenceError, RuntimeError, TypeError,
            ValueError, OverflowError):
        return None


def _cached_collider_motion_certificate(
        canonical_topology, triangles, current_positions, next_positions):
    """``_fit_collider_motion_certificate`` for the case its inputs repeat.

    Fitting the certificate is an SVD per motion group per endpoint, and it is
    the largest single cost inside ``_capture_collider_payload``.  Its result is
    a pure function of ``(canonical_topology, triangles, current_positions,
    next_positions)`` and it returns immutable tuples, so a frame that presents
    equal values recomputes it for nothing.  That is the common case: a collider
    that does not move relative to the cloth presents ``current == next``, and
    `_collider_canonical_motion_topology` is guarded on the canonical triangle
    list matching the evaluated one, so it returns the same canonical vertices
    for every *rigid* collider transform.

    The key is the value bytes of the two position sequences, the triangle
    sequence and the canonical vertices - not their object identity, because
    ``_capture_collider_payload`` rebuilds those sequences every frame, which is
    what makes an identity key miss every time.  Any change to any fit input
    changes the key and forces a refit.  The cached value is the same immutable
    dict the uncached call returns and is never mutated by the caller.
    ``_fit_collider_motion_certificate`` itself is unchanged.
    """
    canonical_values = canonical_topology.get("canonical")
    if canonical_values is None:
        return _fit_collider_motion_certificate(
            canonical_topology, triangles, current_positions, next_positions)

    def payload(values):
        if hasattr(values, "tobytes"):
            return values.tobytes()
        return repr(values).encode("ascii")

    hasher = hashlib.blake2b(digest_size=16, person=b"GPCert")
    hasher.update(payload(triangles))
    for values in (canonical_values, current_positions, next_positions):
        hasher.update(payload(values))
    cache_key = hasher.digest()
    cached = _collider_certificate_cache.get(cache_key)
    if cached is not None:
        _collider_certificate_state['reuse_count'] += 1
        return cached
    fitted = _fit_collider_motion_certificate(
        canonical_topology, triangles, current_positions, next_positions)
    _collider_certificate_state['fit_count'] += 1
    if len(_collider_certificate_cache) >= _COLLIDER_CERTIFICATE_CACHE_MAX:
        _collider_certificate_cache.clear()
        _collider_certificate_state['eviction_count'] += 1
    _collider_certificate_cache[cache_key] = fitted
    return fitted


def _collider_vertex_group_scores(mesh, bone_group_indices):
    """Return ``{vertex_index: [(group, weight), ...]}`` for bone members only.

    ``MeshVertex.groups`` is the only route Blender offers to the deform layer -
    there is no ``foreach_get`` for it, it is not a mesh attribute, and the
    bmesh deform layer has no bulk read either - so this walk is intrinsic.  It
    is paid once per collider instead of once per triangle corner, and only for
    vertices that carry a group whose name matched a bone.  The ``weight >
    1.0e-8`` filter is the one the element walk applied, so a group whose weight
    is zero contributes no score and does not create a patch key.
    """
    if not bone_group_indices:
        return {}
    scores = {}
    for vertex in mesh.vertices:
        rows = []
        for entry in vertex.groups:
            group = int(entry.group)
            if group not in bone_group_indices:
                continue
            weight = float(entry.weight)
            if weight > 1.0e-8:
                rows.append((group, weight))
        if rows:
            scores[int(vertex.index)] = rows
    return scores


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
    # The self-collision response is the only selector of the Mil2 variant
    # (`main.cpp:8107-8121` writes `runtime->mil2_self_contact_variant` from this
    # field, and `Mil2_substep.cu:518` derives `use_barrier` from it).  The
    # barrier/NBD response is what runs the projected collision solve, and the
    # native validation accepts MIL2_NDB only for a Mil2 owner
    # (`main.cpp:8056-8062`).  Publishing OGC unconditionally left that variant
    # unreachable from every scene, so the Mil2 backend always ran the OGC
    # response.  Send the response that belongs to the selected solver.
    config.self_response = (
        CType.GPUCLOTH_SELF_RESPONSE_MIL2_NDB
        if settings.solver_type == "Mil2"
        else CType.GPUCLOTH_SELF_RESPONSE_OGC)
    return config


def _publish_cloth_collision_config(
        dll, cloth_handle, prepared_config, live_only=False):
    for feature_id in (
            CType.GPUCLOTH_FEATURE_STATIC_OBJECT_COLLISION,
            CType.GPUCLOTH_FEATURE_COLLISION_FRICTION_DAMPING,
            CType.GPUCLOTH_FEATURE_COLLISION_QUALITY_CLAMP,
            CType.GPUCLOTH_FEATURE_SELF_COLLISION,
            CType.GPUCLOTH_FEATURE_SELF_COLLISION_FRICTION):
        if live_only and feature_id not in _LIVE_COLLISION_FEATURES:
            continue
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


# Features the engine applies to collision_settings WITHOUT invalidating the prepared
# runtime.  main.cpp's GPUCLOTH_FEATURE_SELF_COLLISION handler clears
# runtime->preparation.runnable (and resets the preparation result), and
# COLLISION_COLLECTION changes what the prepared state owns, so those two legitimately
# require a re-prepare.  FRICTION_DAMPING, QUALITY_CLAMP and SELF_COLLISION_FRICTION
# only write fields, so a UI edit to them can follow the live handle.
_LIVE_COLLISION_FEATURES = (
    CType.GPUCLOTH_FEATURE_COLLISION_FRICTION_DAMPING,
    CType.GPUCLOTH_FEATURE_COLLISION_QUALITY_CLAMP,
    CType.GPUCLOTH_FEATURE_SELF_COLLISION_FRICTION,
)


def republish_live_collision_settings(obj):
    """Push live-safe collision settings onto an already prepared cloth.

    Called from property `update=` callbacks so friction, damping, quality/clamp and
    self-collision friction take effect without a re-prepare.  Enabling self-collision
    itself is deliberately NOT handled here: the engine invalidates the preparation for
    that feature, so a toggle still needs a prepare and the callback says so rather than
    appearing to work.
    """
    if obj is None or g_dll is None:
        return
    if prepare_task_active() or _teardown_failure or _stop_requested:
        return
    try:
        index = g_clothOBJs.index(obj)
    except ValueError:
        return
    if index >= len(g_cloth_handles):
        return
    try:
        prepared = _capture_cloth_collision_config(obj.GPUCloth)
        for feature_id in _LIVE_COLLISION_FEATURES:
            config = CType.GPUClothCollisionConfig.from_buffer_copy(
                bytes(prepared))
            config.header.feature_id = feature_id
            result = int(g_dll.GPUCloth_v3_cloth_configure(
                g_cloth_handles[index],
                cast(
                    pointer(config),
                    POINTER(CType.GPUClothFeatureConfigHeader))))
            if result != CType.GPUCLOTH_ABI_OK:
                print(
                    f"[GPUCloth] live collision setting {feature_id} rejected "
                    f"with {result}")
                return
    except (AttributeError, ReferenceError, RuntimeError, TypeError,
            ValueError) as exc:
        # A property callback must never raise into the UI; report and leave the
        # prepared runtime as it was.
        print(f"[GPUCloth] live collision update skipped: {exc}")


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


# Blender's matrix values are published as float32.  The largest expression
# below is a three-term dot product or a 3x3 determinant; 64 unit roundoffs
# (64 * 2**-23 ~= 7.63e-6) bounds their accumulated float32 error.  The
# explicit 1e-5 ceiling leaves a small conversion margin and is only a
# roundoff allowance; it does not rescale geometry or admit measurable scale.
_RIGID_TRANSFORM_TOLERANCE = 1.0e-5


def _linear_transform_determinant(values):
    columns = tuple(
        tuple(values[row * 4 + column] for row in range(3))
        for column in range(3))
    return (
        columns[0][0] * (
            columns[1][1] * columns[2][2] -
            columns[1][2] * columns[2][1]) -
        columns[1][0] * (
            columns[0][1] * columns[2][2] -
            columns[0][2] * columns[2][1]) +
        columns[2][0] * (
            columns[0][1] * columns[1][2] -
            columns[0][2] * columns[1][1]))


def _validate_rigid_transform(matrix, label):
    """Require an affine transform with unit scale and no shear."""
    values = _matrix_signature(matrix, label)
    tolerance = _RIGID_TRANSFORM_TOLERANCE
    if (
            abs(values[12]) > tolerance or
            abs(values[13]) > tolerance or
            abs(values[14]) > tolerance or
            abs(values[15] - 1.0) > tolerance):
        raise RuntimeError(
            f"{label} must be an affine rigid transform with applied scale")
    columns = tuple(
        tuple(values[row * 4 + column] for row in range(3))
        for column in range(3))
    for column in columns:
        if abs(sum(value * value for value in column) - 1.0) > tolerance:
            raise RuntimeError(
                f"{label} must be an affine rigid transform with applied scale")
    for first in range(3):
        for second in range(first + 1, 3):
            if abs(sum(
                    columns[first][index] * columns[second][index]
                    for index in range(3))) > tolerance:
                raise RuntimeError(
                    f"{label} must be an affine rigid transform with applied scale")
    determinant = _linear_transform_determinant(values)
    if abs(determinant - 1.0) > tolerance:
        raise RuntimeError(
            f"{label} must be an affine rigid transform with applied scale")
    return matrix


def _matrix_inverse(matrix, label, require_rigid_transform=False):
    if require_rigid_transform:
        _validate_rigid_transform(matrix, label)
    else:
        _matrix_signature(matrix, label)
    try:
        inverse = matrix.inverted()
    except (
            AttributeError, ReferenceError, RuntimeError, TypeError,
            ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"{label} is singular") from exc
    if require_rigid_transform:
        _validate_rigid_transform(inverse, f"{label} inverse")
    else:
        _matrix_signature(inverse, f"{label} inverse")
    return inverse


def _cloth_local_matrix(
        cloth_inverse, object_matrix, label, require_rigid_transform=False):
    try:
        relative = cloth_inverse @ object_matrix
    except (
            AttributeError, ReferenceError, RuntimeError, TypeError,
            ValueError) as exc:
        raise RuntimeError(
            f"cannot transform {label} into cloth-local space") from exc
    if require_rigid_transform:
        _validate_rigid_transform(relative, f"{label} cloth-local transform")
    else:
        _matrix_signature(relative, f"{label} cloth-local transform")
    return relative


def _same_float32_payload(left, right):
    """Value equality for two float32 payloads, without boxing either one.

    The readers of a collider history entry accept a sequence of floats; the
    writer now keeps the float32 buffer it already built.  Both sides are
    float32, so byte equality is exactly the ``tuple(...) != tuple(...)``
    decision it replaces.
    """
    left_array = np.asarray(left, dtype=np.float32).reshape(-1)
    right_array = np.asarray(right, dtype=np.float32).reshape(-1)
    if left_array.size != right_array.size:
        return False
    return bool(np.array_equal(left_array, right_array))


def _indexed_float32(values, order):
    """The walk's ``local_positions[index * 3 + component]`` read order.

    ``values`` is the ``(n, 3)`` float32 coordinate block a ``foreach_get``
    filled; ``order`` is the permutation that puts it in the evaluated mesh's
    own vertex order.  Identity is the common case and is returned as the flat
    view it already is, so the ordered path stays one copy.
    """
    if not np.array_equal(order, np.arange(values.shape[0])):
        return np.ascontiguousarray(values[order].reshape(-1))
    return values.reshape(-1)


def _collider_position_digest(positions, person):
    values = np.ascontiguousarray(
        positions, dtype=np.float32).reshape(-1)
    hasher = hashlib.blake2b(digest_size=16, person=person)
    hasher.update(int(values.size).to_bytes(8, "little", signed=False))
    hasher.update(values.tobytes(order="C"))
    return hasher.digest()


def _collider_geometry_fingerprint(cloth_local_positions):
    """Digest the evaluated collider geometry, in cloth-local space.

    This is the payload's retention key.  `_collider_motion_classification`
    decides STATIC/MOVING/DEFORMING from three inputs - `topology_generation` (a
    digest of `object_id`, `vertex_count` and the triangle indices), equality of
    the cloth-local positions, and equality of the cloth-local matrix - and this
    digest is taken over the same cloth-local array the position comparison
    reads, after the same transform.  Equal digests therefore imply a STATIC
    classification; the implication is one-way through a 2**-128 collision, not
    through a modelling assumption.

    It is what lets the payload be retained without re-deriving it: the
    comparison costs one `blake2b` over 3n float32 (0.06 ms at 1 826 vertices)
    where the payload costs 21.65 ms, and the materialisation cannot be skipped
    because judging that the collider did not move *is* reading where it is.

    See `_collider_world_fingerprint` for the cheaper key taken one step
    earlier, over the mesh's own coordinates and the cloth-local matrix, which is
    what lets the triangulation and the transform below it be skipped entirely
    on a collider that presents the geometry it did last frame.
    """
    return _collider_position_digest(cloth_local_positions, b"GPColGeom")


def _collider_world_fingerprint(coordinates, matrix_signature, vertex_count):
    """The frame's collider geometry key, before the cloth-local transform.

    Taken over the evaluated mesh's own float32 coordinates, the cloth-local
    matrix and the vertex count, so it is a function of exactly the inputs the
    cloth-local positions are a function of: a collider that presents the same
    coordinates under the same matrix presents the same cloth-local positions,
    and one that moved, deformed, was re-scaled or was replaced does not.
    """
    hasher = hashlib.blake2b(digest_size=16, person=b"GPColWld")
    hasher.update(int(vertex_count).to_bytes(8, "little", signed=False))
    hasher.update(int(coordinates.size).to_bytes(8, "little", signed=False))
    for value in tuple(matrix_signature):
        hasher.update(struct.pack("<d", float(value)))
    hasher.update(np.ascontiguousarray(
        coordinates, dtype=np.float32).reshape(-1).tobytes(order="C"))
    return hasher.digest()


def _collider_payload_retained(
        retained, fingerprint, topology_generation, matrix_signature):
    """May the retained payload stand for this frame's capture?

    Only when the frame presents the geometry the payload was built from, in
    full: the same evaluated cloth-local geometry digest, the same topology
    generation and the same cloth-local matrix.  Any difference at all - a move,
    a deform, a topology change, a re-prepare, a new collider - rebuilds.  There
    is no tolerance and no fallback here on purpose: a wrong answer publishes
    last frame's collision surface as this frame's.
    """
    if retained is None:
        return False
    if retained.get("fingerprint") != fingerprint:
        return False
    if int(retained.get("topology_generation", 0)) != int(topology_generation):
        return False
    return tuple(retained.get("matrix_signature", ())) == tuple(
        matrix_signature)


def _collider_geometry_retained(cached, geometry):
    """May the cached triangulation and transform stand for this frame?

    The question `_collider_payload_retained` asks of the built payload, asked
    one step earlier of the work that builds it.  It is the same decision on the
    same inputs: the world digest is a function of the coordinates and the
    matrix, the cloth-local digest a function of those and nothing else, and the
    face count is the topology the triangulation walks.  The triangle count
    cannot be the witness here - `calc_loop_triangles` is the expensive half of
    what this gate skips (1.39 ms on the Drape scene's 1 826-vertex sphere), so
    calling it to find out whether it may be skipped would defeat the gate - and
    a collider whose faces changed cannot report the same face count.  The
    triangle count is re-derived whenever this test misses, and the payload's
    own retention test below re-derives the topology generation with it.

    Reusing on a match is what makes a static collider cost one coordinate read
    and one 16-byte digest instead of a re-triangulation of the whole surface,
    two index-stream reads and a float64 transform: measured on the Drape
    scene's 1 826-vertex sphere, `calc_loop_triangles` 1.39 ms, the triangle
    index read 0.42 ms, the vertex index read 0.25 ms and the transform 0.03 ms,
    against 0.21 ms for the read and the digest that decide it.
    """
    if cached is None:
        return False
    if cached.get("world_fingerprint") != geometry["world_fingerprint"]:
        return False
    if int(cached.get("vertex_count", -1)) != int(geometry["vertex_count"]):
        return False
    if int(cached.get("polygon_count", -1)) != int(geometry["polygon_count"]):
        return False
    return tuple(cached.get("matrix_signature", ())) == tuple(
        geometry["matrix_signature"])


def _collider_geometry_witness(previous, geometry):
    """Say so when a frame rebuilds geometry the cache claimed to cover.

    A frame that had to build the collider is a frame that presented geometry the
    reuse test did not hold - that is what a miss means - so a miss over a live
    entry for the same occurrence is the one way this gate can be wrong.  It is
    counted on the build path, where the truth is re-derived, and printed once,
    because a silently reused payload would publish last frame's collision
    surface as this frame's.
    """
    if previous is None or _collider_geometry_retained(previous, geometry):
        return
    _collider_geometry_state["rebuild_after_reuse_count"] += 1
    if _collider_geometry_state["rebuild_after_reuse_count"] == 1:
        print(
            "[GPUCloth] collider geometry reuse missed on a build: the cache "
            f"held an entry for {geometry['vertex_count']} verts / "
            f"{geometry['polygon_count']} faces and this frame built "
            f"{int(geometry['vertex_count'])} / "
            f"{int(geometry['polygon_count'])}; the rebuilt capture is "
            "published", flush=True)


def _retain_collider_payload(
        history_key, payload, fingerprint, topology_generation,
        matrix_signature, geometry=None):
    """Store the built payload under the occurrence key that owns it.

    ``geometry`` is the triangulation and cloth-local transform the payload was
    built from, retained beside it so the next frame that presents the same
    collider geometry can be served without rebuilding either.  It travels with
    the payload under the same key, so the two can never be read apart.
    """
    cache = _collider_payload_cache
    if len(cache) >= _COLLIDER_PAYLOAD_CACHE_MAX and history_key not in cache:
        cache.clear()
        _collider_geometry_cache.clear()
    cache[history_key] = {
        "payload": payload,
        "fingerprint": fingerprint,
        "topology_generation": int(topology_generation),
        "matrix_signature": tuple(matrix_signature),
    }
    if geometry is not None:
        _collider_geometry_witness(_collider_geometry_cache.get(history_key),
                                   geometry)
        _collider_geometry_cache[history_key] = geometry


def _retarget_collider_payload_generations(payload, snapshot_generation):
    """Point a retained payload at this frame's generation.

    Every generation the payload publishes is the frame's, as it is on the
    rebuild path; the buffer views keep their addresses and their bytes, which
    is what "the payload is still current" means.  `_query_collection_snapshot`
    compares `record.geometry_generation` against the committed snapshot, so a
    retained payload that did not move its generation forward would be rejected
    by that check.
    """
    generation = int(snapshot_generation)
    config = payload["config"]
    config.geometry_generation = generation
    for attribute in (
            "positions_previous", "positions_current", "positions_next",
            "motion_group_canonical_to_world",
            "motion_group_endpoint_residual"):
        getattr(config, attribute).generation = generation
    payload["record"].geometry_generation = generation
    history = payload["history_next"]
    history["geometry_generation"] = generation
    return payload


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
        history_local = history["local_positions"]
        history_matrix = tuple(history["matrix_signature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("collider history is invalid") from exc
    if history_generation >= int(snapshot_generation):
        raise RuntimeError("collider history generation is not monotonic")
    # Byte equality on the same float32 buffers is the same decision the former
    # ``tuple(...) != tuple(...)`` made: both sides are float32 payloads, so two
    # buffers are equal exactly when their bytes are.  Boxing 3n floats per
    # frame to compare them measured 1.8 ms at 5 048 vertices.
    if (
            history_topology != int(topology_generation) or
            not _same_float32_payload(history_local, local_positions)):
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


def _resolve_collider_surface_contract(settings):
    """Map Blender's independent flags onto the finite native ABI."""
    single_sided = bool(settings.use_culling)
    # Blender 4.2 defaults to (True, False). The native one-sided contract
    # always uses surface normals, so Single Sided owns the mapping. Keep RNA
    # unchanged; Override Normals cannot change native sidedness.
    bool(settings.use_normal)
    return (
        CType.GPUCLOTH_COLLIDER_ONE_SIDED_NORMAL
        if single_sided else CType.GPUCLOTH_COLLIDER_TWO_SIDED)


def _capture_collider_payload(
    occurrence, modifier_index, depsgraph, cloth_owner_id, collection_id,
        snapshot_generation, collider_history, cloth_inverse,
        require_rigid_transform=False):
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
        vertex_count = len(mesh.vertices)
        polygon_count = len(mesh.polygons)
        if vertex_count == 0 or polygon_count == 0:
            raise RuntimeError(
                f"collider {occurrence['source_object'].name_full!r} "
                "has no evaluated collision surface")
        relative_matrix = _cloth_local_matrix(
            cloth_inverse, occurrence["matrix_world"],
            f"collider {occurrence['source_object'].name_full!r}")
        matrix_signature = tuple(_matrix_signature(
            relative_matrix,
            f"collider {occurrence['source_object'].name_full!r} "
            "cloth-local transform"))
        # ── the decision this frame's geometry answers ──────────────────────
        # A collider that presents the coordinates, the face count and the
        # cloth-local matrix it presented last frame does not need its surface
        # re-triangulated, its index streams re-read or its vertices
        # re-transformed: it needs the geometry those produced.  The test is the
        # frame's own key, taken here - before any of the work it judges - for
        # the same reason the classification reads the positions before building
        # the payload from them: judging that the collider did not move *is*
        # reading where it is.  See `_collider_geometry_retained`.
        #
        # The face count is the topology witness.  `calc_loop_triangles` is the
        # expensive half of what this gate skips (1.39 ms on the Drape scene's
        # 1 826-vertex sphere), so the count it would produce cannot be the thing
        # that decides whether to call it; the evaluated mesh's polygon count is
        # free and a collider whose faces changed cannot report the same one.
        # The triangle count is re-derived whenever this gate misses, so a
        # topology change still rebuilds the payload below.
        coordinates = np.empty(vertex_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coordinates)
        non_finite = np.flatnonzero(~np.isfinite(coordinates))
        if non_finite.size:
            raise RuntimeError(
                f"collider {occurrence['source_object'].name_full!r} "
                f"vertex {int(non_finite[0]) // 3} contains a "
                "non-finite float")
        geometry = {
            "world_fingerprint": _collider_world_fingerprint(
                coordinates, matrix_signature, vertex_count),
            "matrix_signature": matrix_signature,
            "vertex_count": int(vertex_count),
            "polygon_count": int(polygon_count),
            "local_positions": coordinates,
            "cloth_local_positions": None,
            "cloth_fingerprint": None,
            "triangle_indices": None,
            "vertex_indices": None,
        }
        history_key = (
            int(cloth_owner_id),
            int(occurrence["object_id"]),
            int(occurrence["instance_id"]),
            int(modifier_index),
        )
        history = collider_history.get(history_key)
        cached_geometry = _collider_geometry_cache.get(history_key)
        if _collider_geometry_retained(cached_geometry, geometry):
            _collider_geometry_state["reuse_count"] += 1
            geometry = cached_geometry
            local_positions = geometry["local_positions"]
            cloth_local_positions = geometry["cloth_local_positions"]
            fingerprint = geometry["cloth_fingerprint"]
            triangle_indices = geometry["triangle_indices"]
        else:
            _collider_geometry_state["build_count"] += 1
            # Two bulk reads replace the element walk that read ``vertex.index``
            # and ``vertex.co`` once per vertex and ran ``element_multiply`` per
            # vertex through a Matrix/Vector RNA round trip.  On the isolated
            # 5 048-vertex collider that walk measured 41.3 ms per capture; the
            # same coordinates and the same transform through ``foreach_get``
            # and one matmul measure under 2 ms, and the payloads are
            # byte-identical.
            loop_triangles = vcu.calc_mesh_loop_triangles(mesh)
            triangle_count = len(loop_triangles)
            if triangle_count == 0:
                raise RuntimeError(
                    f"collider {occurrence['source_object'].name_full!r} "
                    "has no evaluated collision surface")
            triangle_indices = np.empty(triangle_count * 3, dtype=np.int32)
            loop_triangles.foreach_get("vertices", triangle_indices)
            if ((triangle_indices < 0).any() or
                    (triangle_indices >= vertex_count).any()):
                raise RuntimeError(
                    f"collider {occurrence['source_object'].name_full!r} "
                    "has invalid evaluated triangles")
            vertex_indices = np.empty(vertex_count, dtype=np.int32)
            mesh.vertices.foreach_get("index", vertex_indices)
            indices = vertex_indices.astype(np.int64)
            out_of_range = np.flatnonzero(
                (indices < 0) | (indices >= vertex_count))
            if out_of_range.size:
                raise RuntimeError(
                    f"collider {occurrence['source_object'].name_full!r} "
                    f"has invalid evaluated vertex index "
                    f"{int(indices[out_of_range[0]])}")
            # Blender stores coordinates as float32 and this buffer is float32,
            # so the extracted components are the values the former float()
            # round trip produced.  The transform is evaluated in float64 to
            # match Blender's own Matrix @ Vector, then narrowed once, which is
            # what ``_finite_float32_tuple`` did per component.
            world_matrix = np.array(
                [[relative_matrix[row][column] for column in range(4)]
                 for row in range(4)], dtype=np.float64)
            points = np.empty((vertex_count, 4), dtype=np.float64)
            points[:, :3] = coordinates.reshape(-1, 3)
            points[:, 3] = 1.0
            cloth_local = (points @ world_matrix.T)[:, :3].astype(np.float32)
            non_finite = np.flatnonzero(
                ~np.isfinite(coordinates) |
                ~np.isfinite(cloth_local.reshape(-1)))
            if non_finite.size:
                raise RuntimeError(
                    f"collider {occurrence['source_object'].name_full!r} "
                    f"local vertex {int(non_finite[0]) // 3} contains a "
                    "non-finite float")
            # The walk published ``local_positions[index * 3 + component]``, so
            # the index stream is a permutation of the read order rather than an
            # assumption that Blender numbers its vertices 0..n-1.
            order = np.argsort(indices, kind="stable")
            # The decision that consumes these is value equality against the
            # next frame's buffer and a float32 buffer for the ABI, so the
            # float32 arrays stay arrays.  Boxing 3n floats into a Python tuple
            # per frame measured 1.8 ms at 5 048 vertices and 3n floats per
            # level here; the values are the same bits either way.
            local_positions = _indexed_float32(
                coordinates.reshape(-1, 3), order)
            cloth_local_positions = _indexed_float32(cloth_local, order)
            geometry["local_positions"] = local_positions
            geometry["cloth_local_positions"] = cloth_local_positions
            geometry["cloth_fingerprint"] = _collider_geometry_fingerprint(
                cloth_local_positions)
            geometry["triangle_indices"] = triangle_indices
            geometry["vertex_indices"] = vertex_indices
            fingerprint = geometry["cloth_fingerprint"]
    finally:
        if mesh_owner is not None:
            mesh_owner.to_mesh_clear()

    triangle_indices_flat = np.ascontiguousarray(
        triangle_indices.reshape(-1))
    # ── the decision, before the work it decides about ──────────────────────
    # A collider that presents the geometry its payload was built from does not
    # need its topology re-derived, its triangles re-boxed, its time levels
    # re-copied, its motion certificate re-fitted or its config re-filled: it
    # needs the payload it already has, pointed at this frame's generation.
    # `_topology_generation`, `_collider_canonical_motion_topology`,
    # `triangles` and the whole build below are all skipped on this path.
    #
    # The test is the classification's own, run on the classification's own
    # inputs.  Only the *order* changes: the classification's geometry half is
    # answered by the digest instead of by a `np.array_equal` over the same
    # buffer, so the two are the same decision at the same inputs.
    if history is not None:
        retained = _collider_payload_cache.get(history_key)
        history_matrix = tuple(history["matrix_signature"])
        if _collider_payload_retained(
                retained, fingerprint,
                int(history["topology_generation"]), history_matrix):
            if (require_rigid_transform and
                    _linear_transform_determinant(history_matrix) <= 0.0):
                raise RuntimeError(
                    f"collider {occurrence['source_object'].name_full!r} "
                    "cloth-local transform must preserve orientation")
            _collider_payload_state["retain_count"] += 1
            _collider_geometry_cache[history_key] = geometry
            return _retarget_collider_payload_generations(
                retained["payload"], snapshot_generation)
    _collider_payload_state["rebuild_count"] += 1
    triangle_array = (c_uint * int(triangle_indices_flat.size)).from_buffer_copy(
        np.ascontiguousarray(triangle_indices_flat, dtype=np.uint32))
    triangles = tuple(map(
        tuple, triangle_indices.reshape(-1, 3).tolist()))
    exact_topology_generation = _topology_generation(
        occurrence["object_id"], vertex_count, triangles)
    canonical_topology = _collider_canonical_motion_topology(
        source, vertex_count, triangles, exact_topology_generation)
    topology_generation = (
        int(canonical_topology["topology_generation"])
        if canonical_topology is not None
        else exact_topology_generation)
    collider_class, feature_id = _collider_motion_classification(
        history, topology_generation, local_positions, matrix_signature,
        snapshot_generation)
    if (require_rigid_transform and
            collider_class == CType.GPUCLOTH_COLLIDER_STATIC and
            _linear_transform_determinant(matrix_signature) <= 0.0):
        raise RuntimeError(
            f"collider {occurrence['source_object'].name_full!r} "
            "cloth-local transform must preserve orientation")
    # Static scale is already baked into cloth-local vertices. Motion still
    # requires rigid endpoints, including history admitted while static.
    if require_rigid_transform and (
            collider_class != CType.GPUCLOTH_COLLIDER_STATIC):
        if history is not None:
            try:
                history_matrix_values = tuple(history["matrix_signature"])
                if len(history_matrix_values) != 16:
                    raise ValueError
                history_matrix = Matrix(tuple(
                    tuple(history_matrix_values[row * 4 + column]
                          for column in range(4))
                    for row in range(4)))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("collider history matrix is invalid") from exc
            _validate_rigid_transform(
                history_matrix,
                f"collider {occurrence['source_object'].name_full!r} "
                "previous cloth-local transform")
        _validate_rigid_transform(
            relative_matrix,
            f"collider {occurrence['source_object'].name_full!r} "
            "cloth-local transform")

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
        # History levels are the same float32 buffers this frame publishes, so
        # they are re-used rather than re-validated into Python floats.  The
        # stored arrays are float32 by construction, which is the only value
        # the old round trip could produce.
        previous_positions = np.asarray(
            history["previous_positions"], dtype=np.float32)
        current_positions = np.asarray(
            history["current_positions"], dtype=np.float32)
    else:
        previous_positions = cloth_local_positions
        current_positions = cloth_local_positions
    next_positions = cloth_local_positions

    scalar_count = int(cloth_local_positions.size)
    position_type = c_float * scalar_count
    # One C copy per time level instead of boxing the floats into lists and
    # slice-assigning them.  The three levels measured 1.44 ms on the Drape
    # scene's collider against 0.02 ms for these copies, same bytes.
    previous_array = position_type.from_buffer_copy(
        np.ascontiguousarray(previous_positions, dtype=np.float32))
    current_array = position_type.from_buffer_copy(
        np.ascontiguousarray(current_positions, dtype=np.float32))
    next_array = position_type.from_buffer_copy(
        np.ascontiguousarray(next_positions, dtype=np.float32))
    if scalar_count and len({
            addressof(previous_array),
            addressof(current_array),
            addressof(next_array)}) != 3:
        raise RuntimeError("collider time-level buffers alias")

    motion_certificate = _cached_collider_motion_certificate(
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
    triangle_address = (
        addressof(triangle_array) if triangle_indices_flat.size else 0)
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
        config.sidedness = _resolve_collider_surface_contract(settings)
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
            f"collider {source_object.name_full!r} absorption is outside "
            "[0, 1]")
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
    payload = {
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
            # The retained levels are the float32 buffers this frame already
            # built; readers accept a float sequence and the classification
            # compares them by value.
            "local_positions": local_positions,
            "matrix_signature": tuple(
                value for value in matrix_signature),
            "previous_positions": current_positions,
            "current_positions": cloth_local_positions,
            "geometry_fingerprint": fingerprint,
        },
    }
    # The payload the next static frame will be served from, with the
    # triangulation and transform it was built from retained beside it.  It is
    # dropped by the same comparison that admits it: a frame presenting different
    # geometry fails `_collider_payload_retained` and rebuilds here, overwriting
    # both entries.
    _retain_collider_payload(
        history_key, payload, fingerprint, topology_generation,
        matrix_signature, geometry)
    return payload
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
        require_rigid_transform = (
            str(getattr(settings, "solver_type", "")) == "PD")
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
            f"cloth {cloth_obj.name_full!r} evaluated world transform",
            require_rigid_transform)
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
                snapshot_generation, collider_history, cloth_inverse,
                require_rigid_transform))

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
                # Retained as the float32 buffers the capture already built:
                # re-boxing 3n floats per collider per frame is what the
                # element walk cost, and every reader takes a float sequence.
                "local_positions": np.asarray(
                    candidate["local_positions"], dtype=np.float32),
                "matrix_signature": tuple(
                    candidate["matrix_signature"]),
                "previous_positions": np.asarray(
                    candidate["previous_positions"], dtype=np.float32),
                "current_positions": np.asarray(
                    candidate["current_positions"], dtype=np.float32),
                # The retention key travels with the history entry, so a
                # retained payload and the history it was admitted under can
                # never be read apart.
                "geometry_fingerprint": candidate["geometry_fingerprint"],
            }
    return history


def _commit_frame_inputs(
        dll, cloth_handles, prepared_collections, prepared_pins,
        prepared_dynamic_meshes, source_generation,
        verify_committed_state=False):
    global _collider_history, _live_group_state
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


def _publish_frame_inputs(context):
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
        # The pin targets are re-sampled every frame; the pin membership and raw
        # weights are not frame state at all.  They are the mesh's deform layer,
        # which the cloth input fingerprint hashes in full - every group name,
        # member index and member weight - so they are reused while that
        # fingerprint stands and rebuilt when it moves.  It is the contract the
        # stiffness, pressure, shrink and self-collision channels are already
        # captured under, and the fingerprint read is this frame path's own
        # memoised one, not a second fingerprint of the scene.  The fingerprint
        # covers the groups of the cloth objects, so a proxy simulation object,
        # whose groups it does not reach, is never given the cache.
        pin_witness = _cache_source_generation(context.scene)
        for index, (cloth_obj, simulation_obj) in enumerate(zip(
                g_clothOBJs, g_simulationOBJs)):
            pin_state = _pin_snapshot_states[index]
            prepared_pins.append(_vertex_drag_pin_snapshot(
                index,
                _capture_pin_snapshot(
                    cloth_obj, simulation_obj, capture_depsgraph,
                    pin_state["topology_generation"], generation,
                    channels=pin_state.setdefault("pin_channels", {}),
                    witness=(pin_witness
                             if simulation_obj is cloth_obj else None))))
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
_simulation_frame_state = {
    'last_solved': None,
    # Live simulation does not own a native cache unless bake/playback is
    # requested.  Retain reached render states so timeline rewinds still
    # publish the requested geometry in that mode.
    'positions': {},
    # A changed input rebuilt the owners from rest; no frame may be solved
    # against the abandoned owner until that preparation runs, because a solve
    # would write deformed coordinates into the mesh the rebuild captures.
    'rebuild_pending': False,
}
_cache_source_state = {
    'generation': 0,
    'staged': 0,
    'epoch': 0,
    # A dependency-graph notification named a watched input while this handler
    # could not act on it (a prepare was running, or the frame path held the
    # playback guard).  The change is not lost - the frame path consumes this -
    # and until it is consumed no frame may be written, because a frame of the
    # old inputs stamped with the new generation is a stale frame a reader
    # cannot tell from a current one.
    'deferred': False,
    # The first frame of the run the cache currently belongs to, or None when
    # no run owns the cache.  A frame joins the past only when a run that began
    # at the start of the range produced it; see _cache_run_belongs_to_the_start.
    'run_first': None,
    # The cache-input generation that run is a statement about, or None when no
    # run owns the cache.  A run's frames are geometry the inputs in force when
    # it opened produced, and the engine cannot tell afterwards which inputs a
    # stored frame states: the write carries the frame's own generation and the
    # engine rejects only a *lowering* of it (main.cpp:14150-14153), so a frame
    # written under one generation is accepted and then served under another.
    # The claim therefore carries the generation as well as the frame, and both
    # are dropped together (_clear_retained_frames).
    'run_generation': None,
}
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


def _cache_hash_array(hasher, label, values):
    """Hash one numeric payload as raw bytes instead of a Python tuple repr.

    ``_cache_hash_value`` hashes ``repr(tuple(value))``, which for the per-edge,
    per-polygon and per-vertex-group payloads of a 128x128 grid builds multi-megabyte
    strings.  The dtype and element count are hashed alongside the bytes so that two
    different payloads cannot collide by framing.
    """
    array = np.ascontiguousarray(values)
    hasher.update(label.encode('utf-8'))
    hasher.update(b'\0')
    hasher.update(array.dtype.str.encode('ascii'))
    hasher.update(b'\0')
    hasher.update(int(array.size).to_bytes(8, 'little'))
    hasher.update(array.tobytes())
    hasher.update(b'\0')


def _cache_mesh_group_weights(mesh, vertex_groups):
    """Return (name, member indices, member weights) per group in one vertex pass.

    Blender exposes vertex groups only through ``MeshVertex.groups``: there is no
    ``foreach_get`` for them, they are not mesh attributes, and the bmesh deform layer
    has no bulk read either.  The former form called ``VertexGroup.weight()`` once per
    vertex per group and caught the RuntimeError raised for every non-member - 37 ms for
    one two-member group on a 16384-vertex mesh.  One pass over the vertices collecting
    every group at once is the cheapest route the API offers.
    """
    names = tuple(group.name for group in vertex_groups)
    if not names:
        return ()
    indices = [[] for _ in names]
    values = [[] for _ in names]
    for vertex in mesh.vertices:
        index = int(vertex.index)
        for assignment in vertex.groups:
            slot = int(assignment.group)
            if 0 <= slot < len(names):
                indices[slot].append(index)
                values[slot].append(float(assignment.weight))
    return tuple(
        (name,
         np.asarray(indices[slot], dtype=np.int32),
         np.asarray(values[slot], dtype=np.float32))
        for slot, name in enumerate(names))


def _cache_hash_mesh_topology(hasher, label, mesh):
    """Hash edges and the polygon-to-vertex mapping through bulk reads."""
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    _cache_hash_array(hasher, f"{label}.edges", edges)

    loop_start = np.empty(len(mesh.polygons), dtype=np.int32)
    loop_total = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loop_vertex = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex)
    _cache_hash_array(hasher, f"{label}.loop_start", loop_start)
    _cache_hash_array(hasher, f"{label}.loop_total", loop_total)
    _cache_hash_array(hasher, f"{label}.loop_vertex", loop_vertex)


def _cache_material_coordinate_layer(settings, mesh):
    """Resolve the UV layer capture_material_coordinates would upload.

    Mirrors that function's own selection rule and its "no coordinates needed"
    early exit, so the fingerprint hashes exactly the payload that reaches the
    solver for FABRIC membrane directions: the per-loop UV values of the active
    layer, or of the named ``anisotropy_uv_map``.  The uploaded triangle-corner
    order is a deterministic permutation of this array whose order is already
    covered by the topology hash.

    FABRIC is the only material model, so the coordinates are needed whenever
    ``capture_material_coordinates`` is asked for them - ``use_anisotropy`` no
    longer widens what the engine receives (it only reflects the same payload
    onto the isotropic stiffness slots as well).
    """
    uv_map = str(getattr(settings, "anisotropy_uv_map", ""))
    uv_layers = getattr(mesh, "uv_layers", None)
    if uv_layers is None:
        return None
    if not uv_map:
        return getattr(uv_layers, "active", None)
    return uv_layers.get(uv_map)


def _cache_hash_material_coordinates(hasher, label, settings, mesh):
    layer = _cache_material_coordinate_layer(settings, mesh)
    if layer is None:
        return
    try:
        loop_count = len(mesh.loops)
        if len(layer.data) != loop_count:
            _cache_hash_value(hasher, f"{label}.uv_map", layer.name)
            _cache_hash_value(hasher, f"{label}.uv_corners", len(layer.data))
            return
        values = np.empty(loop_count * 2, dtype=np.float32)
        layer.data.foreach_get("uv", values)
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return
    _cache_hash_value(hasher, f"{label}.uv_map", layer.name)
    _cache_hash_array(hasher, f"{label}.uv", values)


# ---------------------------------------------------------------------------
# Identity of an animated or deformed input.
#
# The fingerprint is the *identity of the input*, not a value that was sampled at
# the frame the fingerprint happened to run on.  For a static object those are the
# same thing and the evaluated value is read directly.  For an input the timeline
# owns they are not: the evaluated transform, and the evaluated mesh, are functions
# of the frame, so hashing them makes the source generation move with the frame.
# A bake then fails at its own completion - ``GPUCLOTH_CACHE_STATUS_BAKE_COMPLETE``
# requires ``cache->source_generation == update->source_generation``
# (src/engine/source/main.cpp:11947-11957) - and, when no bake is running, every
# frame of a live session looks like an input change.
#
# What is hashed instead is the animation's own data: the action, its fcurves and
# keyframes, its frame range, the NLA strips and the drivers.  That is a property
# of the input, so it does not move when the timeline advances, and editing the
# animation moves it, which is the invalidation the user expects.
# ---------------------------------------------------------------------------

# The channels a manual transform writes.  They are hashed for an animated object
# so that a hand-move of a channel the timeline does not own still invalidates.
_TRANSFORM_CHANNEL_NAMES = (
    "location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle",
    "rotation_mode", "scale", "delta_location", "delta_rotation_euler",
    "delta_rotation_quaternion", "delta_scale",
)


def _cache_timeline_owns(animation_data):
    """True when ``animation_data`` carries data that decides values per frame."""
    if animation_data is None:
        return False
    try:
        if getattr(animation_data, "action", None) is not None:
            return True
        if bool(getattr(animation_data, "use_nla", False)) and len(
                tuple(getattr(animation_data, "nla_tracks", ()))):
            return True
        return len(animation_data.drivers) != 0
    except (AttributeError, ReferenceError, RuntimeError):
        return False


def _cache_action_fcurves(action):
    """Every fcurve of an action, in a deterministic order.

    Blender 4.4 introduced slotted actions; ``Action.fcurves`` is the legacy
    accessor for them and is empty when the action has no legacy slot, so the
    layered structure is walked as well.  Both hosts this add-on ships on (4.2.3
    and 5.2.1) are covered by the same call.
    """
    curves = list(getattr(action, "fcurves", ()))
    if not curves:
        for layer in getattr(action, "layers", ()):
            for strip in getattr(layer, "strips", ()):
                for channelbag in getattr(strip, "channelbags", ()):
                    curves.extend(getattr(channelbag, "fcurves", ()))
    return sorted(
        curves, key=lambda curve: (
            str(getattr(curve, "data_path", "")),
            int(getattr(curve, "array_index", 0))))


def _cache_hash_fcurve(hasher, label, fcurve):
    """The curve itself: path, index, keyframes, interpolation and handles.

    ``keyframe_points`` is a property collection, so its values are read in bulk
    (``spline.points.foreach_get("co", coords)`` is the documented fast route)
    rather than one ``Keyframe`` at a time.
    """
    _cache_hash_rna_scalars(hasher, label, fcurve)
    count = len(fcurve.keyframe_points)
    _cache_hash_value(hasher, f"{label}.keyframes", count)
    if not count:
        return
    for attribute, dtype in (("co", np.float32),
                             ("handle_left", np.float32),
                             ("handle_right", np.float32),
                             ("interpolation", np.int32),
                             ("easing", np.int32),
                             ("handle_left_type", np.int32),
                             ("handle_right_type", np.int32)):
        try:
            values = np.empty(count * (2 if dtype is np.float32 else 1),
                              dtype=dtype)
            fcurve.keyframe_points.foreach_get(attribute, values)
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            _cache_hash_value(
                hasher, f"{label}.{attribute}",
                tuple(_cache_keyframe_attribute(fcurve, attribute)))
            continue
        _cache_hash_array(hasher, f"{label}.{attribute}", values)


def _cache_keyframe_attribute(fcurve, attribute):
    for point in fcurve.keyframe_points:
        value = getattr(point, attribute, None)
        yield tuple(value) if hasattr(value, "__len__") else value


def _cache_hash_driver(hasher, label, fcurve):
    """The driver's own data, including the identity of every target."""
    driver = getattr(fcurve, "driver", None)
    if driver is None:
        return
    _cache_hash_rna_scalars(hasher, label, driver)
    for index, variable in enumerate(driver.variables):
        variable_label = f"{label}.variable[{index}]"
        _cache_hash_rna_scalars(hasher, variable_label, variable)
        for target_index, target in enumerate(variable.targets):
            target_label = f"{variable_label}.target[{target_index}]"
            _cache_hash_rna_scalars(hasher, target_label, target)
            _cache_hash_reference(
                hasher, f"{target_label}.id",
                getattr(target, "id", None))


def _cache_hash_reference(hasher, label, referenced):
    """Hash the identity of a referenced datablock, never its pointer."""
    _cache_hash_value(
        hasher, label,
        getattr(referenced, "name_full", None)
        or getattr(referenced, "name", None))


def _cache_hash_action(hasher, label, action):
    _cache_hash_reference(hasher, f"{label}.name", action)
    _cache_hash_rna_scalars(hasher, label, action)
    try:
        _cache_hash_value(
            hasher, f"{label}.frame_range", tuple(action.frame_range))
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    for index, fcurve in enumerate(_cache_action_fcurves(action)):
        _cache_hash_fcurve(hasher, f"{label}.fcurve[{index}]", fcurve)


def _cache_animation_identity(owner):
    """A digest of the timeline data that owns ``owner``, or None.

    None means the timeline does not own this datablock and its evaluated values
    are the input itself, which is the case every fingerprint read before this
    existed covered.
    """
    animation_data = getattr(owner, "animation_data", None)
    if not _cache_timeline_owns(animation_data):
        return None
    hasher = hashlib.blake2b(digest_size=8, person=b"GPUAnim ")
    _cache_hash_rna_scalars(hasher, "animation_data", animation_data)
    _cache_hash_action(hasher, "action", getattr(animation_data, "action", None))
    for index, track in enumerate(getattr(animation_data, "nla_tracks", ())):
        track_label = f"nla[{index}]"
        _cache_hash_rna_scalars(hasher, track_label, track)
        for strip_index, strip in enumerate(track.strips):
            strip_label = f"{track_label}.strip[{strip_index}]"
            _cache_hash_rna_scalars(hasher, strip_label, strip)
            strip_action = getattr(strip, "action", None)
            _cache_hash_reference(
                hasher, f"{strip_label}.action", strip_action)
            _cache_hash_action(
                hasher, f"{strip_label}.action_data", strip_action)
    for index, driver in enumerate(getattr(animation_data, "drivers", ())):
        driver_label = f"driver[{index}]"
        _cache_hash_fcurve(hasher, driver_label, driver)
        _cache_hash_driver(hasher, driver_label, driver)
    return int.from_bytes(hasher.digest(), "little") or 1


def _cache_driven_identifiers(animation_data):
    """{owner prefix: {identifier, ...}} for every channel the timeline drives.

    Blender writes an animated property back onto the original datablock when the
    frame changes - that is why the panel shows the animated value - so a channel
    the timeline owns is a function of the frame and only its animation identity
    may be hashed.  A channel the timeline does not own keeps whatever the user
    last set, which is the hand-move that must still invalidate.
    """
    driven = {}
    if animation_data is None:
        return driven
    paths = []
    action = getattr(animation_data, "action", None)
    if action is not None:
        paths.extend(
            curve.data_path for curve in _cache_action_fcurves(action))
    paths.extend(
        curve.data_path for curve in getattr(animation_data, "drivers", ()))
    for path in paths:
        head, _, tail = str(path).rpartition(".")
        if not tail:
            head, tail = "", str(path)
        driven.setdefault(head, set()).add(tail)
    return driven


def _cache_hand_edited_transform_channels(obj, frame=None):
    """Object-level transform channels the timeline drives but does not currently own.

    A driven channel's value is written back by the animation on every frame change,
    which is why ``_cache_hash_object_transform`` hashes the animation's *identity*
    rather than the value: hashing the value would move the source generation on every
    frame of a bake.  A hand edit is the exception that matters.  While the edited
    value stands, the pose the solver reads is not the pose the animation produces and
    the store stops describing the scene being simulated.  Measured on the shipped
    build: a keyframed collider moved by hand left the digest byte-identical
    (``digest_before_move == digest_after_move``) while the evaluated transform the
    solver consumes moved to 0.4375, the store survived, and the frame after a wrap was
    served from before the move.

    Returns the channel names whose current value differs from the action's own
    evaluation at ``frame``, so a caller can hash exactly those and nothing else - a
    channel the animation still owns keeps the digest still, which is what a playback
    or a bake needs.
    """
    animation_data = getattr(obj, "animation_data", None)
    action = getattr(animation_data, "action", None)
    if action is None:
        return ()
    if frame is None:
        scene = getattr(bpy.context, "scene", None)
        frame = int(getattr(scene, "frame_current", 0) or 0)
    frame = int(frame)
    edited = set()
    for curve in _cache_action_fcurves(action):
        head, _, tail = str(getattr(curve, "data_path", "")).rpartition(".")
        if head or tail not in _TRANSFORM_CHANNEL_NAMES:
            continue        # object-level transform curves only; arrays are not one
        try:
            current = getattr(obj, tail)
            index = int(getattr(curve, "array_index", 0))
            value = float(current[index])
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
        try:
            prescribed = float(curve.evaluate(frame))
        except (ReferenceError, RuntimeError, TypeError, ValueError):
            continue
        if not math.isclose(value, prescribed, rel_tol=1e-5, abs_tol=1e-6):
            edited.add(tail)
    return tuple(sorted(edited))


def _cache_hash_object_transform(hasher, label, obj, frame=None):
    """Hash the identity of an object's transform instead of one sampled value.

    A static object is read exactly as it always was - the evaluated
    ``matrix_world``, which is what a hand-move changes and what the live
    reconfiguration path depends on.  An object the timeline owns is read as its
    animation data plus the transform channels the timeline does *not* drive,
    because its evaluated matrix is a function of the frame: hashing that made the
    source generation move on every frame of a bake.

    A channel the timeline drives but whose current value is not the one the action
    prescribes is the third case, and it is a change like any other: the pose the
    solver consumes right now is not the pose the cache was built from.  It is hashed
    as itself (see ``_cache_hand_edited_transform_channels``), which moves the
    generation for as long as the edit stands and leaves it still once the animation
    writes the channel back - the only form that catches the hand move without
    reintroducing the per-frame movement the animation identity exists to prevent.
    """
    identity = _cache_animation_identity(obj)
    if identity is None:
        _cache_hash_value(
            hasher, f"{label}.matrix_world",
            tuple(tuple(row) for row in obj.matrix_world))
        return
    _cache_hash_value(hasher, f"{label}.animation", identity)
    driven = _cache_driven_identifiers(obj.animation_data)
    unowned = driven.get("", frozenset())
    _cache_hash_rna_selected(
        hasher, f"{label}.transform", obj,
        tuple(name for name in _TRANSFORM_CHANNEL_NAMES
              if name not in unowned))
    edited = _cache_hand_edited_transform_channels(obj, frame)
    if edited:
        _cache_hash_rna_selected(hasher, f"{label}.hand_edited", obj, edited)
    _cache_hash_reference(hasher, f"{label}.parent", obj.parent)
    _cache_hash_value(
        hasher, f"{label}.parent_inverse",
        tuple(tuple(row) for row in obj.matrix_parent_inverse))
    for index, constraint in enumerate(obj.constraints):
        constraint_label = f"{label}.constraint[{index}]"
        _cache_hash_rna_scalars(
            hasher, constraint_label, constraint,
            excluded=driven.get(f'constraints["{constraint.name}"]', ()))
        for referenced, referenced_name in (
                ("target", getattr(constraint, "target", None)),
                ("pole_target", getattr(constraint, "pole_target", None))):
            _cache_hash_reference(
                hasher, f"{constraint_label}.{referenced}", referenced_name)


def _cache_hash_bone_channels(hasher, label, armature):
    """The rest pose's channel values, read in bulk.

    These are the original (non-evaluated) pose channels, so a hand-posed
    armature moves them while an animated one does not - an animated pose is
    hashed through its animation identity instead.
    """
    bones = armature.pose.bones
    count = len(bones)
    _cache_hash_value(hasher, f"{label}.bones", count)
    if not count:
        return
    for attribute, width, dtype in (
            ("location", 3, np.float32),
            ("scale", 3, np.float32),
            ("rotation_quaternion", 4, np.float32),
            ("rotation_euler", 3, np.float32),
            ("rotation_axis_angle", 4, np.float32),
            ("rotation_mode", 1, np.int32)):
        try:
            values = np.empty(count * width, dtype=dtype)
            bones.foreach_get(attribute, values)
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            continue
        _cache_hash_array(hasher, f"{label}.{attribute}", values)


def _cache_modifier_stack(obj):
    """The visible modifiers that make the evaluated mesh differ from the data."""
    return tuple(
        modifier for modifier in obj.modifiers
        if modifier.type != 'COLLISION' and _modifier_visibility(modifier)[0])


def _cache_deformation_identity(obj, label):
    """(identity, timeline_owned) for the sources that deform this collider.

    ``identity`` is None when nothing deforms the object, which is the case the
    fingerprint has always covered by hashing the original vertex coordinates.
    ``timeline_owned`` says that at least one of those sources is driven by the
    timeline, so its contribution to the evaluated mesh is a function of the
    frame and only its animation identity may be hashed.
    """
    mesh = obj.data
    shape_keys = getattr(mesh, "shape_keys", None)
    modifiers = _cache_modifier_stack(obj)
    if shape_keys is None and not modifiers:
        return None, False
    hasher = hashlib.blake2b(digest_size=8, person=b"GPUDeform")
    timeline_owned = False
    if shape_keys is not None:
        _cache_hash_rna_scalars(hasher, f"{label}.shape_keys", shape_keys)
        key_blocks = tuple(shape_keys.key_blocks)
        driven = _cache_driven_identifiers(
            getattr(shape_keys, "animation_data", None))
        for index, key in enumerate(key_blocks):
            key_label = f"{label}.key[{index}]"
            _cache_hash_reference(hasher, f"{key_label}.name", key)
            _cache_hash_reference(
                hasher, f"{key_label}.relative_key",
                getattr(key, "relative_key", None))
            # A key's ``value`` is written back by the animation system whenever
            # the frame changes, so it is hashed only for the keys the timeline
            # does not drive; an animated value is covered by the Key
            # datablock's animation identity below.
            _cache_hash_rna_scalars(
                hasher, key_label, key, excluded=("value",))
            coords = np.empty(len(key.data) * 3, dtype=np.float32)
            key.data.foreach_get("co", coords)
            _cache_hash_array(hasher, f"{key_label}.coords", coords)
            if "value" not in driven.get(f'key_blocks["{key.name}"]', ()):
                _cache_hash_value(
                    hasher, f"{key_label}.value", float(key.value))
        key_animation = _cache_animation_identity(shape_keys)
        if key_animation is not None:
            timeline_owned = True
            _cache_hash_value(
                hasher, f"{label}.shape_key_animation", key_animation)
    for index, modifier in enumerate(modifiers):
        modifier_label = f"{label}.modifier[{index}]"
        _cache_hash_value(
            hasher, f"{modifier_label}.type", str(modifier.type))
        _cache_hash_rna_scalars(hasher, modifier_label, modifier)
        _cache_hash_reference(
            hasher, f"{modifier_label}.object",
            getattr(modifier, "object", None))
        if modifier.type == 'ARMATURE':
            armature = getattr(modifier, "object", None)
            if armature is None:
                continue
            _cache_hash_object_transform(
                hasher, f"{modifier_label}.armature", armature)
            pose_animation = _cache_animation_identity(armature)
            if pose_animation is None:
                _cache_hash_bone_channels(
                    hasher, f"{modifier_label}.pose", armature)
            else:
                timeline_owned = True
                _cache_hash_value(
                    hasher, f"{modifier_label}.pose_animation", pose_animation)
            if bool(getattr(modifier, "use_vertex_groups", False)):
                # The deform weights decide the deformation and are reachable
                # only through MeshVertex.groups; the cloth loop reads them the
                # same way.
                for group_index, (name, indices, weights) in enumerate(
                        _cache_mesh_group_weights(mesh, obj.vertex_groups)):
                    weights_label = f"{modifier_label}.vgroup[{group_index}]"
                    _cache_hash_value(hasher, f"{weights_label}.name", name)
                    _cache_hash_array(
                        hasher, f"{weights_label}.vertices", indices)
                    _cache_hash_array(
                        hasher, f"{weights_label}.weights", weights)
    return (int.from_bytes(hasher.digest(), "little") or 1), timeline_owned


def _cache_evaluated_depsgraph():
    """The view-layer dependency graph the collider capture itself evaluates.

    ``_publish_frame_inputs`` marshals the collider through
    ``context.evaluated_depsgraph_get()``, so the fingerprint reads the same
    graph rather than a second, possibly stale one.
    """
    try:
        return bpy.context.evaluated_depsgraph_get()
    except (AttributeError, ReferenceError, RuntimeError):
        return None


def _cache_hash_collider_geometry(hasher, label, obj, depsgraph, identity,
                                  timeline_owned):
    """Hash the collider's geometry the way the engine receives it.

    ``_capture_collider_payload`` marshals ``evaluated_get(depsgraph)``, so the
    original ``mesh.vertices`` is only the right read while nothing deforms the
    object.  Once something does, the deformation's own identity is hashed, and
    the evaluated coordinates are hashed as well *unless* the timeline owns the
    deformation - an animated pose or shape key makes the evaluated mesh a
    function of the frame, which is what moved the source generation on every
    baked frame.
    """
    mesh = obj.data
    if identity is None:
        coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        _cache_hash_array(hasher, f"{label}.coords", coords)
        return
    _cache_hash_value(hasher, f"{label}.deformation", identity)
    if timeline_owned or depsgraph is None:
        # Either the evaluated mesh is a function of the frame and only the
        # animation identity may be hashed, or there is no dependency graph to
        # evaluate against; the deformation identity above already covers the
        # shape keys, the modifier stack, the pose and the weights.
        return
    evaluated = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.data
    coords = np.empty(len(evaluated_mesh.vertices) * 3, dtype=np.float32)
    evaluated_mesh.vertices.foreach_get("co", coords)
    _cache_hash_array(hasher, f"{label}.evaluated_coords", coords)


_STAGED_SETTING_NAMES = (
    # Inputs GPUCloth_v3_cloth_configure rejects or cannot clear once the owner
    # is built, so only a rebuilt owner accepts them.
    "solver_type",              # backend owner: main.cpp:7115-7125 rejects
    "bending_model",            # angular/linear: staged owner, main.cpp:10624
    "use_anisotropy",           # stage: main.cpp:10368
    "anisotropy_uv_map",
    "mass_mode",                # AREAL: main.cpp:11245 rejects after build
    "fabric_density",
    "vel_damping",              # stage: main.cpp:10530
    "eff_force_scale",          # stage: main.cpp:10574
    "eff_wind_scale",
    "use_self_collision",       # main.cpp:7096 clears preparation.runnable
    "selfepsilon",
    "use_pressure",             # no ABI path clears the pressure flag
    "use_dynamic_mesh",         # stage: main.cpp:11826
    "use_proxy", "proxy_object", "hi_nx", "hi_ny", "proxy_nx", "proxy_ny",
    "num_sheets", "proxy_scene_type",
    "collision_collection",     # prepared collection snapshot
    "vgroup_mass", "vgroup_struct", "vgroup_bend", "vgroup_shear",
    "vgroup_intern", "vgroup_shrink", "vgroup_pressure",
    "vgroup_selfcol", "vgroup_objcol",
    "effector_weights",         # _publish_frame_inputs refuses a change
    "fabric_tensile_u", "fabric_tensile_v",
    "fabric_compression_u", "fabric_compression_v",
    "fabric_tensile_u_max", "fabric_tensile_v_max",
    "fabric_compression_u_max", "fabric_compression_v_max",
    "fabric_tensile_damping", "fabric_compression_damping",
    "fabric_shear_damping",
    "tension_u", "tension_v", "compression_u", "compression_v",
    "bending_u", "bending_v",
    "max_tension_u", "max_tension_v", "max_compression_u",
    "max_compression_v", "max_bend_u", "max_bend_v",
)
# With the FABRIC triangle-membrane payload the isotropic stretch/shear
# stiffnesses travel inside the staged anisotropy buffer, so they stop being
# live.  There is no second material model left to re-publish them onto
# `clmd->sim_parms`, so this extension is unconditional.
_STAGED_FABRIC_STIFFNESS_NAMES = (
    "tension", "compression", "shear",
    "max_tension", "max_compression", "max_shear",
)
_LIVE_COLLISION_FEATURES = _LIVE_COLLISION_FEATURES + (
    # main.cpp:7058 writes the enable flag and epsilon directly.
    CType.GPUCLOTH_FEATURE_STATIC_OBJECT_COLLISION,
)

# Material features the engine accepts on an owner that is already built, and
# therefore the only ones a live re-configure may carry.  The three that are
# absent are absent for one native reason each, all in `SIM_configure_cloth_feature`:
#
#   MATERIAL_DAMPING   main.cpp:7435 answers INVALID_STATE while a cloth manager
#                      exists (`s_find_cloth_manager`), and a built owner always
#                      has one.
#   BENDING_ANGULAR    main.cpp:7462 answers INVALID_STATE while a manager exists
#                      or the cloth already owns springs.
#   BENDING_SDB        is no longer published at all: the addon's SDB bending
#                      model was removed, so no owner reaches that feature.
#   ANISOTROPY         main.cpp:7498 answers INVALID_STATE unless the cloth owns
#                      vertices and springs, so it is not accepted before a build
#                      either; it is published pre-build on purpose.
#
# Carrying them in the live block is not a lost update but a lost *block*: the
# publisher raises on the first refusal, so every feature after it is skipped
# too.  The live path published damping first, which is why the refusal the user
# saw named feature 8 rather than a stiffness.
_LIVE_MATERIAL_FEATURES = frozenset((
    CType.GPUCLOTH_FEATURE_STRETCH,
    CType.GPUCLOTH_FEATURE_COMPRESSION,
    CType.GPUCLOTH_FEATURE_SHEAR,
    CType.GPUCLOTH_FEATURE_BENDING_LINEAR,
))


def _cache_hash_rna_selected(hasher, label, owner, names):
    if owner is None:
        return
    for name in names:
        if not hasattr(owner, name):
            continue
        value = getattr(owner, name)
        if hasattr(value, "name_full") or hasattr(value, "name"):
            value = ("<ref>", getattr(value, "name_full", None)
                     or getattr(value, "name", None))
        _cache_hash_value(hasher, f"{label}.{name}", value)


def _cache_staged_setting_names():
    # The FABRIC triangle membrane is the only material model, so its staged
    # isotropic stiffness mirror is part of every owner's staged set and no
    # owner setting can remove it.
    return tuple(_STAGED_SETTING_NAMES) + _STAGED_FABRIC_STIFFNESS_NAMES


def _assert_live_settings_are_excluded(staged_excluded):
    """Fail loudly if a live-published scalar is not excluded from staging.

    Without this the two halves can drift apart again exactly as they did for
    gravity: the scalar stays live in the publisher while the staged digest keeps
    watching it, and every edit to it silently costs a full prepare.
    """
    missing = sorted(set(_scene_live_setting_names()) - set(staged_excluded))
    if missing:
        raise RuntimeError(
            "live-published scene settings must not be staged: "
            + ", ".join(missing))


# The collider surface settings the engine is handed, and therefore the only ones
# the fingerprint has to watch.  They live on the OBJECT (``obj.collision``), not on
# the COLLISION modifier: on Blender 4.2.3 that modifier's own RNA exposes
# ``execution_time``, ``is_active``, ``is_override_data``, ``name``,
# ``persistent_uid``, the five ``show_*`` toggles, ``type``,
# ``use_apply_on_spline`` and ``use_pin_to_last`` - none of them a surface setting.
# The settings are reachable only through its ``settings`` POINTER, which
# ``_cache_hash_rna_scalars`` skips because it hashes scalar property types only,
# so hashing the modifier left every one of these values outside the fingerprint.
# ``modifier.settings`` and ``obj.collision`` are the same ``CollisionSettings``
# allocation (same ``as_pointer()``), so hashing this list from the object is one
# hash per value rather than two.
#
# The names are the ones ``_capture_collider_payload`` reads
# (``operators.py:4535-4539``).  ``use_normal`` is read there as well, and
# ``_resolve_collider_surface_contract`` states that it cannot change native
# sidedness; it is hashed anyway, because a collider edit a user makes on the
# object's Collision panel must not be silently ignored - that silence is the
# defect this list exists to close.
_COLLIDER_SETTING_NAMES = (
    "thickness_outer", "cloth_friction", "damping", "absorption",
    "use_culling", "use_normal",
)


# True while this module is reading the fingerprint's own dependency graph.
# The read is what makes the graph notify: ``_cache_evaluated_depsgraph``
# evaluates it, and the evaluation reports the cloth mesh this add-on itself
# wrote on the previous frame, so ``depsgraph_update_post`` runs
# ``_cache_input_change_handler`` with a watched ID and no change behind it.
# That handler cannot act while the frame path holds the playback guard, so it
# records a deferral, and every deferral costs the frame path one more full
# fingerprint.  Measured on the owner's scene before this flag existed: 59
# deferral writes over 13 ordinary frames, one whole extra `_cache_input_digests`
# per frame and one more from the gate reading it again, all of them confirming
# "nothing moved".  A notification that arrives while this flag is set is the
# read noticing the add-on's own output, not a change the handler failed to act
# on, and it is not lost either way: the fingerprint the read is computing is
# the check, and the frame path re-reads it before it writes or serves.
_cache_digest_read_in_progress = False


def _cache_input_digests(scene):
    """Return (full, staged) digests of the cache-affecting addon inputs.

    ``full`` is the generation the cache owner is stamped with and covers every
    input.  ``staged`` covers only the inputs a rebuilt owner is required for,
    so the frame path can tell "re-configure in place" from "prepare again"
    using the same single read of the mesh payloads.

    Membership of ``staged`` is what ``_stage_changed_inputs`` acts on, so an
    input that the live frame path already republishes does not belong in it:
    the collider's transform and vertex positions are the two that do not, and
    the reason is recorded at the collider loop below.

    This is a declaration around the read rather than a second copy of it: the
    digest itself is one function, and the flag exists only so the notification
    the read provokes is attributed to the read.
    """
    global _cache_digest_read_in_progress
    previous = _cache_digest_read_in_progress
    _cache_digest_read_in_progress = True
    try:
        return _cache_input_digests_read(scene)
    finally:
        _cache_digest_read_in_progress = previous


def _cache_input_digests_read(scene):
    hasher = hashlib.blake2b(digest_size=8, person=b"GPUCloth")
    staged = hashlib.blake2b(digest_size=8, person=b"GPUStaged")
    for hasher_ in (hasher, staged):
        helper = scene.gpu_cloth_helper
        # The staged stream answers "does a rebuilt owner have to be built".
        # A scalar the solver reads live cannot answer yes to that: the engine
        # picks it up on the next solve, so staging it would schedule a prepare
        # for an edit that needs none.  `_scene_live_setting_names()` is the one
        # declaration of which scalars those are.
        staged_excluded = {
            "cache_dir", "cache_index", "cache_name", "use_disk_cache",
            "use_external_cache", "external_cache_dir",
            "use_library_path", "cache_compression",
            "bake_start", "bake_end", "bake_progress",
            "is_baked", "is_baking", "is_outdated", "is_frame_skip",
            "cache_info", "cached_frame_count", "playback_mode",
            "memory_preflight_status", "prepare_state", "prepare_status",
            "prepare_progress",
        }
        if hasher_ is staged:
            staged_excluded.update(_scene_live_setting_names())
            _assert_live_settings_are_excluded(staged_excluded)
        _cache_hash_rna_scalars(
            hasher_, "scene", helper, excluded=staged_excluded)
        _cache_hash_value(hasher_, "render.fps", scene.render.fps)
        _cache_hash_value(hasher_, "render.fps_base", scene.render.fps_base)

    cloth_objects = sorted(
        (obj for obj in scene.objects
         if obj.type == 'MESH' and hasattr(obj, "GPUCloth")
         and obj.GPUCloth.is_active),
        key=lambda obj: obj.name_full)
    for index, obj in enumerate(cloth_objects):
        label = f"cloth[{index}]"
        for hasher_ in (hasher, staged):
            _cache_hash_value(hasher_, f"{label}.name", obj.name_full)
            _cache_hash_object_transform(
                hasher_, label, obj, scene.frame_current)
        mesh = obj.data
        _cache_hash_rna_scalars(
            hasher, f"{label}.settings", obj.GPUCloth,
            excluded={
                "cpu_sync_copied", "cpu_sync_unsupported",
                "cpu_sync_blockers", "cpu_sync_errors",
                "cpu_sync_report",
            })
        _cache_hash_rna_selected(
            staged, f"{label}.staged", obj.GPUCloth,
            _cache_staged_setting_names())
        # The effector weight group is reached only through a POINTER property,
        # so the scalar walk above cannot see its fifteen fields.
        effector_weights = getattr(obj.GPUCloth, "effector_weights", None)
        if effector_weights is not None:
            _cache_hash_rna_scalars(
                hasher, f"{label}.effector_weights", effector_weights)
            _cache_hash_rna_scalars(
                staged, f"{label}.effector_weights", effector_weights)
        # POINTER properties are skipped by the scalar walk; hash the resolved
        # identity of the referenced datablock rather than the pointer.
        for pointer_name in ("collision_collection", "proxy_object"):
            referenced = getattr(obj.GPUCloth, pointer_name, None)
            _cache_hash_value(
                hasher, f"{label}.{pointer_name}",
                getattr(referenced, "name_full", None)
                if referenced is not None else None)
        _cache_hash_material_coordinates(
            hasher, f"{label}.material", obj.GPUCloth, mesh)
        _cache_hash_material_coordinates(
            staged, f"{label}.material", obj.GPUCloth, mesh)
        if index < len(_initial_positions):
            rest = np.asarray(
                _initial_positions[index], dtype=np.float32).reshape(-1)
        else:
            rest = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", rest)
        _cache_hash_array(hasher, f"{label}.rest", rest)
        _cache_hash_mesh_topology(hasher, label, mesh)
        _cache_hash_mesh_topology(staged, label, mesh)
        for group_index, (name, indices, values) in enumerate(
                _cache_mesh_group_weights(mesh, obj.vertex_groups)):
            _cache_hash_value(
                hasher, f"{label}.vgroup[{group_index}].name", name)
            _cache_hash_array(
                hasher, f"{label}.vgroup[{group_index}].vertices", indices)
            _cache_hash_array(
                hasher, f"{label}.vgroup[{group_index}].weights", values)
            _cache_hash_value(
                staged, f"{label}.vgroup[{group_index}].name", name)
            _cache_hash_array(
                staged, f"{label}.vgroup[{group_index}].vertices", indices)
            _cache_hash_array(
                staged, f"{label}.vgroup[{group_index}].weights", values)

    collision_objects = sorted(
        (obj for obj in scene.objects
         if obj.type == 'MESH' and any(
             modifier.type == 'COLLISION' for modifier in obj.modifiers)),
        key=lambda obj: obj.name_full)
    for index, obj in enumerate(collision_objects):
        label = f"collider[{index}]"
        mesh = obj.data
        # A collider's transform and its vertex positions are geometry, and the
        # geometry is republished live: `_publish_frame_inputs` captures the
        # collider through `_prepare_collection_snapshots` and commits it with
        # `_commit_frame_inputs` on every stepped frame, which is what reaches
        # `sXY_apply_collision_collection` -> `XPBD_solver_set_collision_batch`
        # and the obstacle refit.  Staging them therefore bought nothing and cost
        # a full prepare: moving a collider moved the staged stream, and
        # `_stage_changed_inputs` scheduled a re-prepare of an owner whose cloth
        # had not changed at all.  They stay in the *full* stream, which is the
        # fingerprint the cache owner is stamped with, so an edit here still
        # invalidates a stale bake exactly as any other changed input does.
        #
        # What stays staged is what only a rebuilt owner can accept: the
        # collider's mesh topology and the six surface settings.  The engine
        # builds the obstacle record and its triangle ownership during
        # preparation, and `_prepare_collection_snapshots` refuses a changed
        # collection or surface contract with "reprepare is required".
        #
        # The transform is read as its identity rather than as one sampled
        # matrix, and the geometry as the geometry the capture marshals: an
        # animated collider must not move the fingerprint once per frame, and a
        # deformed one must not leave the collision surface the solver holds
        # stale.  Both are recorded where the values are read, and the frame is
        # passed so a channel the timeline drives can be told from one a hand edit
        # has left standing at a value the animation does not prescribe.
        _cache_hash_object_transform(hasher, label, obj, scene.frame_current)
        deformation, timeline_owned = _cache_deformation_identity(obj, label)
        _cache_hash_collider_geometry(
            hasher, label, obj, _cache_evaluated_depsgraph(), deformation,
            timeline_owned)
        settings = getattr(obj, "collision", None)
        for hasher_ in (hasher, staged):
            _cache_hash_mesh_topology(hasher_, label, mesh)
            # The engine-boundary consumer reads ``source_object.collision``, so the
            # surface settings are hashed from there.  The COLLISION modifier's own
            # scalars below carry no surface setting at all (see
            # ``_COLLIDER_SETTING_NAMES``); they are kept because its metadata is an
            # evaluated input too.
            _cache_hash_rna_selected(
                hasher_, f"{label}.collision", settings,
                _COLLIDER_SETTING_NAMES)
        for modifier in obj.modifiers:
            if modifier.type == 'COLLISION':
                _cache_hash_rna_scalars(
                    hasher, f"{label}.modifier", modifier)
                _cache_hash_rna_scalars(
                    staged, f"{label}.modifier", modifier)

    return (int.from_bytes(hasher.digest(), "little") or 1,
            int.from_bytes(staged.digest(), "little") or 1)


def _cache_source_generation(scene):
    """Stable fingerprint of cache-affecting addon inputs, never solved output.

    The depsgraph handler recomputes this before it compares generations, so it has to
    stay cheap; the whole point of the comparison is to avoid acting on the result.
    Every bulk input below is read in binary form rather than as a Python tuple of
    tuples.

    This is the memoised entry point (``_cache_input_digests_for_frame``): it costs
    a comparison unless the memo is cold, and the frame path keeps it primed once
    per frame.  A caller that must observe a change on its own rather than reuse the
    frame's reading asks ``_cache_source_generation_fresh`` instead - the
    dependency-graph handler and the sliced re-simulation's per-tick check are the
    two that do.
    """
    return _cache_input_digests_for_frame(scene)[0]


# The frame-path memo of the fingerprint.  Measured before it existed: 113 digest
# computations for one 13-frame pass, six call sites per frame, each a full read
# of every hashed input including the collider's evaluated vertices - and the
# biggest of them sat inside the `store_frame` stage, where the write gate read
# the whole fingerprint again to compare it with the one `_live_step_cloth_scene`
# had computed a few hundred microseconds earlier.
#
# The key is the frame's cheap revision (``_cache_fingerprint_key``) plus the
# depsgraph observation, and it cannot miss a change for a stated reason rather
# than a hopeful one:
#
#   * every ``GPUCloth`` scalar is declared with
#     ``update=_on_simulation_input_change`` (properties.py), and that is what
#     moves ``properties._simulation_input_epoch``;
#   * the two scalars the fingerprint reads straight off the Scene - the imported
#     frame rate the engine takes its timestep from - move no counter and send no
#     notification, so they are read into the key itself.  Measured on the
#     owner's route: with the epoch alone as the key, an fps edit left the memo
#     answering with the generation from before it, while every other
#     fingerprinted input probed there (a ``GPUCloth`` scalar, the helper's
#     gravity, a collider transform, a collider surface setting, a vertex-group
#     weight, a rename of the cloth) was reached by one of the other two routes;
#   * everything else the fingerprint reads - a collider's transform, its
#     ``COLLISION`` modifier, the cloth mesh the user edited - reaches
#     ``_cache_input_change_handler`` through ``depsgraph_update_post``, and that
#     handler clears this memo rather than reading it;
#   * the fingerprint *reads* the dependency graph, so the read itself is what
#     makes the graph report the add-on's own output; that report is suppressed
#     while the read is in flight (``_cache_digest_read_in_progress``) and never
#     reaches the memo.
#
# The gate reads this memo for the generation the frame was solved under - the
# value the frame path took at that frame's entry - and never as evidence that
# nothing has moved since: reuse alone would make that comparison a tautology, and
# "the inputs moved while the step ran" is exactly what the gate exists to refuse.
# What it compares against the value it captured is the key above and the
# notification channel (``_cache_input_change_handler``), which are the routes a
# change can take.
_cache_fingerprint_memo = {'key': None, 'full': 0, 'staged': 0}
_cache_digest_computations = 0
# How many watched notifications the handler had to record instead of acting on.
# The gate compares this across the step: during the step the handler can never act
# (the frame path holds the playback guard), so a real motion made mid-step cannot
# reach the cache and must refuse the frame.
_cache_unactionable_notifications = 0


def _cache_fingerprint_invalidate():
    """Forget the memo.  Called from every route that can see a change."""
    _cache_fingerprint_memo['key'] = None


def _cache_input_digests_for_frame(scene):
    """The fingerprint for one frame's production, computed once.

    Same function as ``_cache_input_digests``, with the frame-path memo in front
    of it.  The handler deliberately does not use this: its whole job is to
    compare against the value it last stored, so it must read.
    """
    global _cache_digest_computations
    key = _cache_fingerprint_key(scene)
    memo = _cache_fingerprint_memo
    if memo['key'] == key:
        return memo['full'], memo['staged']
    full, staged = _cache_input_digests(scene)
    memo['key'] = key
    memo['full'] = full
    memo['staged'] = staged
    _cache_digest_computations += 1
    return full, staged


def _cache_source_generation_fresh(scene):
    """The fingerprint, read now, and it also refreshes the frame-path memo.

    Used where the answer decides something on its own rather than being compared
    with a value captured earlier in the same frame: the sliced re-simulation's
    per-tick check is the case in point.  That check runs on the timer, outside any
    frame, and a memo of it would answer that check with the value
    from the previous tick - so a collider moved while the span was being produced
    would not be seen, which is the one thing the check exists to see.

    Every real read is counted, wherever it is taken from, because "one fingerprint
    per frame" is a claim about computations and not about call sites.
    """
    global _cache_digest_computations
    memo = _cache_fingerprint_memo
    full, staged = _cache_input_digests(scene)
    memo['key'] = _cache_fingerprint_key(scene)
    memo['full'] = full
    memo['staged'] = staged
    _cache_digest_computations += 1
    return full


def _cache_status_update(operation, scene, error_code=0, frame=-1,
                         generation=None):
    if not _runtime_handle_value():
        raise RuntimeError("v3 cache owner requires a live runtime")
    if not _cache_handle_value():
        if g_dll is None:
            raise RuntimeError("v3 cache owner is not live")
        _configure_cache_features(g_dll, scene)
    # A caller that already fingerprinted this unchanged scene passes the value in:
    # the cache-input handler computes it to decide whether to call this at all, and the
    # scene cannot change between the two reads.
    if generation is None:
        generation = _cache_source_generation(scene)
    else:
        generation = int(generation)
    if (operation == CType.GPUCLOTH_CACHE_STATUS_SOURCE_CHANGED and
            frame < 0):
        # The engine validates this frame against the range copied into
        # GPUClothCacheConfig at configure time, and bake_start is deliberately
        # not fingerprinted, so an edit to it must not turn a valid
        # invalidation into an INVALID_VALUE rejection.
        frame = int(scene.gpu_cloth_helper.bake_start)
        owner = g_cache_owner
        if owner is not None:
            configured = owner["config"]
            frame = min(
                max(frame, int(configured.frame_start)),
                int(configured.frame_end))
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
    if g_dll is None:
        raise RuntimeError("v3 native module is not loaded")
    if not _runtime_handle_value():
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
    config.gravity[:] = _scene_live_setting_values(scene.gpu_cloth_helper)
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
    # A runtime teardown is the one event that can invalidate a live step drive
    # from underneath it, and it has no other visible trace: the drive's next
    # tick would report a released runtime, which names the symptom and not the
    # cause.  Only the drive's own owner is printed, because that is the one
    # caller that cannot have intended it; the add-on's stdout is where this
    # file already puts what a user cannot see from the panel.
    if _infinite_is_running():
        print(
            "[GPUCloth] runtime teardown during the infinite simulation "
            "(shutdown_runtime=%s):\n%s"
            % (bool(shutdown_runtime),
               "".join(traceback.format_stack()[-6:-1])), flush=True)
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
    """Put the prepared rest pose back into the cloth meshes.

    A stored pose that does not fit the mesh it would be written into is left
    unwritten rather than forced in.  The rest belongs to the *prepared*
    topology, and the pose now survives a refused rebuild
    (``_reset_owner_python_state``), so the two can disagree: a topology edit
    between a successful prepare and the rebuild that follows it makes the
    stored array a different length, and ``foreach_set`` would then raise
    instead of restoring anything.  Skipping it is the fail-closed answer - the
    rebuild then captures the mesh as it stands, exactly as it does when no rest
    was ever captured.
    """
    for i, cloth_obj in enumerate(g_clothOBJs):
        if i >= len(_initial_positions):
            continue
        values = _initial_positions[i]
        if values.size != len(cloth_obj.data.vertices) * 3:
            continue
        cloth_obj.data.vertices.foreach_set("co", values)
        cloth_obj.data.update()
        cloth_obj.data.update_tag()


def _store_simulation_frame(frame):
    """Retain one reached render state for live timeline playback."""
    snapshots = []
    for cloth_obj in g_clothOBJs:
        # foreach_get writes into the caller's buffer and ``values`` is freshly
        # allocated inside this loop, so it is not aliased by anything else and the
        # former ``values.copy()`` was a second full-size memcpy per object per frame
        # (196 KB at 16384 vertices).
        values = np.empty(len(cloth_obj.data.vertices) * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", values)
        snapshots.append(values)
    _simulation_frame_state['positions'][int(frame)] = tuple(snapshots)


def _load_simulation_frame(frame, depsgraph):
    """Publish a retained live state when native cache is unavailable."""
    snapshots = _simulation_frame_state['positions'].get(int(frame))
    if snapshots is None:
        if int(frame) != int(_bake_range['start']) or not _initial_positions:
            return False
        _restore_initial_positions()
        depsgraph.update()
        return True
    if len(snapshots) != len(g_clothOBJs):
        return False
    for cloth_obj, values in zip(g_clothOBJs, snapshots):
        expected = len(cloth_obj.data.vertices) * 3
        if values.size != expected:
            return False
        cloth_obj.data.vertices.foreach_set("co", values)
        cloth_obj.data.update()
        cloth_obj.data.update_tag()
    depsgraph.update()
    return True


def _clear_retained_frames():
    """Drop the in-session live states that let a backward scrub replay the past.

    ``positions`` is one solved render state per reached frame and is what
    ``_load_simulation_frame`` publishes when the native cache misses;
    ``last_solved`` is the guard the frame path compares the requested frame
    against.  With both gone the frame path has nothing to replay, so the next
    requested frame re-simulates from the first frame of the range in order.
    ``rebuild_pending`` belongs to the same state and is reset with it.

    The run identity goes with them, and for the same reason: it is the record of
    *which* run the past belongs to, so a past that has just been dropped must not
    keep claiming one.  Only a path that starts a run over the whole range may
    claim one again (``_cache_run_open``), and no frame joins a past it did not
    open (``_cache_run_belongs_to_the_start``).  The claim carries the
    cache-input generation it was opened under as well as its first frame: both
    are the record of which run the past belongs to, so both are dropped here.
    """
    _simulation_frame_state['last_solved'] = None
    _simulation_frame_state['positions'].clear()
    _simulation_frame_state['rebuild_pending'] = False
    _cache_source_state['run_first'] = None
    _cache_source_state['run_generation'] = None
    _cache_source_state['deferred'] = False
    # A past that has just been dropped says nothing about which inputs it was
    # produced under, and the memo must not outlive it.
    _cache_fingerprint_invalidate()


def _cache_run_open(scene, frame):
    """Declare that the run now starting begins at ``frame``.

    Called by the two paths that can start a run over the whole range - the frame
    path's forward solve and the bake modal - because they are the ones that know
    where a run begins: the timeline never solves frame 1 (it is the rest state),
    while a bake may well start there.  A frame solved outside such a run joins no
    past, which is what makes "nothing is written until the simulation starts
    again from the beginning" true rather than hopeful.

    The claim records the cache-input generation in force when the run opens, and
    is **not** rebound while that generation is still current: the caller reaches
    this with ``first`` equal to the range's first frame only while the run has no
    past to advance from, so a later frame of the same run arrives with a ``first``
    of its own and must not move the anchor.  Rebinding moved the claim to the
    frontier, after which "did this run begin at the start of the range" was true
    of every run and the predicate could no longer refuse anything.

    That generation is read for real rather than taken from the frame-path memo,
    because the write gate now compares the frame's own entry-time read against
    this claim (``_frame_may_be_persisted``): a claim bound to a memo that no route
    had refreshed would describe a generation the inputs are no longer in, and the
    gate would admit a frame the previous code refused.  Measured - the
    cache-ownership probe's animated-collider case, where ``frame_set`` writes a
    hand-edited transform channel back and the fingerprint returns to its earlier
    value with neither route firing: with the claim bound from the memo the gate
    wrote the frame; with this read it refuses it, exactly as before.  A run opens
    once per span, so this costs one fingerprint per span, not one per frame.
    """
    generation = _cache_source_generation_fresh(scene)
    if (_cache_source_state['run_generation'] == generation):
        return
    _cache_source_state['run_first'] = int(frame)
    _cache_source_state['run_generation'] = generation


def _cache_run_belongs_to_the_start(scene, frame, generation=None):
    """Is this frame part of a run that began at the start of the range?

    A cache frame means "the simulation reached this frame", and a rewind is
    served from that claim.  Only a run opened at the start of the range owns such
    a claim; see ``_cache_run_open`` for where one is opened.

    The past is a statement about *inputs* as much as about frames, and it is the
    one thing the engine cannot re-check: a frame's generation is its own at the
    time it was written, and a reader cannot tell a frame the current inputs
    produced from one an earlier generation did.  So the claim also has to be the
    one the current inputs opened.

    ``generation`` is how a caller that already holds that value hands it in, and
    the write gate is the one that does: the frame path reads the fingerprint once
    at the frame's entry (``_live_step_cloth_scene``) and its gate passes that
    value here, beside the revision and the notification count that establish
    whether anything moved since (``_frame_may_be_persisted``).  Reading it again
    here would compute the value the frame path is already holding, at the cost of
    a whole fingerprint per frame - measured on the owner's route, 51-62 ms inside
    the ``store_frame`` stage.

    A caller with no such value passes nothing and gets the read for real
    (``_cache_source_generation_fresh``): the serve path (``_load_cached_frame``)
    reaches this outside a frame, where the memoised answer would be the value
    from before the change it is asked about.

    Both callers ask it - the write gate before a frame is stored, and
    ``_load_cached_frame`` before one is published - because a past that may not
    be written may not be read either.
    """
    run_first = _cache_source_state['run_first']
    if run_first is None:
        return False
    if generation is None:
        generation = _cache_source_generation_fresh(scene)
    if _cache_source_state['run_generation'] != int(generation):
        return False
    return int(frame) >= int(run_first)


def _cache_fingerprint_epoch():
    """The property revision: the trigger for whether a read is worth it.

    The property epoch moves for every ``GPUCloth`` scalar, because every one of
    them is declared with ``update=_on_simulation_input_change``; the depsgraph
    route reaches the memo by clearing it (``_cache_input_change_handler``), not
    by moving this.  ``_refresh_prepared_inputs`` compares it to decide whether the
    fingerprint is worth recomputing, and it is deliberately wider than that
    question needs: an input it misses there is still reached by the key below.
    """
    return int(properties._simulation_input_epoch['value'])


def _cache_fingerprint_key(scene):
    """The frame's cheap revision: the memo's key, and the gate's comparison.

    Everything the fingerprint reads carries one of three routes: a ``GPUCloth``
    scalar moves the property epoch, an object or mesh edit reaches
    ``_cache_input_change_handler`` through ``depsgraph_update_post``, and the two
    scalars that decide the timestep - ``render.fps`` and ``render.fps_base`` -
    carry neither, so they are read here.  Measured on the owner's route with the
    epoch alone as the key: an fps edit left the frame path's memoised read
    answering with the generation from before it, while every other fingerprinted
    input probed there (a ``GPUCloth`` scalar, the helper's gravity, a collider
    transform, a collider surface setting, a vertex-group weight, a rename of the
    cloth) was reached by one of the other two routes.
    """
    return (
        int(properties._simulation_input_epoch['value']),
        float(scene.render.fps),
        float(scene.render.fps_base),
    )


def _frame_may_be_persisted(scene, frame, run, generation, revision,
                           notifications):
    """May the frame just solved be written to the cache?

    The engine cannot tell afterwards whether a stored frame states the inputs in
    force now: the write carries the frame's own generation and the engine rejects
    only a *lowering* of that generation, so a frame whose inputs moved while it
    was being solved is accepted and then served back as though it belonged to the
    new inputs.  That is the splice the owner sees as a frame "from another part
    of the simulation" (round 2, item 4).

    Three conditions, and each is a refusal rather than a repair:

    * the run this frame belongs to did not begin at the start of the range, or is
      not the one the inputs this frame was solved under opened.  The first half is
      ``_cache_run_belongs_to_the_start``; the second half is the generation the
      frame path read at this frame's entry and handed in here, which is compared
      there against the run claim - no second read, so no second fingerprint;
    * the vertex grab is live.  Its frame is the user's hand rather than a step of
      the simulation.  That rule already exists as ``_vertex_grab_in_progress``;
      this function is that rule generalised, not a second mechanism beside it;
    * the inputs are not the ones the step began under.

    The third condition used to cost a full fingerprint here, on top of the one
    ``_live_step_cloth_scene`` had computed at the top of the same frame - measured
    at 113 computations for a 13-frame pass, with the larger share of the
    `store_frame` stage being this second read.  ``generation`` is now the value the
    frame path read once at this frame's entry (``_refresh_prepared_inputs``), and
    what this function has to establish is whether that value is still the one in
    force.  It is asked with two cheap revisions instead of a second read:

    * ``revision`` is the frame-path memo's own key: the property revision, which
      moves for every ``GPUCloth`` scalar (each carries
      ``update=_on_simulation_input_change``), plus the Scene scalars no callback
      and no notification reach (``_cache_fingerprint_key`` - the frame rate the
      engine takes its timestep from).  A step that began under one revision and
      writes under another is a step whose inputs moved;
    * ``notifications`` is the count of watched dependency-graph notifications the
      handler had to record instead of acting on (``_cache_playback_guard`` is
      held for the whole step, so during the step the handler can never act).  It
      is a counter rather than the ``deferred`` flag because the flag cannot
      survive the frame path's own next entry: measured, our own output's
      notification arrives *after* the guard is released, is recorded, and would
      then be indistinguishable from a user's hand.

    Either revision moving means the inputs moved, and the frame is refused and the
    deferral raised so the frame path drops the past at its next entry - the same
    refusal as before, decided by two cheap comparisons instead of a fingerprint.
    When neither moved there is nothing for a read to find: every input the
    fingerprint covers is reached by one of those two routes, and both are quiet.
    """
    if not _cache_run_belongs_to_the_start(scene, frame, generation):
        return False
    if _vertex_grab_in_progress():
        return False
    if (_cache_fingerprint_key(scene) != revision or
            int(_cache_unactionable_notifications) != int(notifications)):
        _cache_source_state['deferred'] = True
        return False
    return True


# True while the frame path is re-simulating a past that a scene change dropped.
# The panel reads it through ``resimulating()`` and shows it in the row the
# prepare already uses, which is the whole point: the re-run is Blender's own
# contract for an outdated cache, and it has to be visible rather than a window
# that appears to have hung.
_resimulate_state = {'active': False}
# The slice interval, and the one flag that says whether the timer is registered -
# the same discipline ``_prepare_timer_registered`` and
# ``_infinite_timer_registered`` use, and the same order of magnitude as
# ``_PREPARE_POLL_INTERVAL``.  A frame costs far more than this, so the interval
# decides only how quickly the event loop comes back, never the rate.
_RESIMULATE_INTERVAL = 0.005
_resimulate_timer_registered = False


def resimulating():
    """True while an invalidated cache is being re-simulated from the start.

    Asked by the panel, which shows the same progress row the prepare uses.  The
    owner's ruling asks for Blender's semantics *because they are predictable*:
    every scene change drops the past, and the simulation runs again from the
    beginning.  Predictable is only true if the user can see it happening.
    """
    return bool(_resimulate_state['active'])


def _resimulate_notice(scene, done, total):
    """Put the re-simulation on the panel, or clear it when it is over.

    ``total`` of zero clears.  The row is the prepare's own - ``prepare_state``,
    ``prepare_progress`` and ``prepare_status`` are all excluded from the cache
    fingerprint (``_cache_input_digests``), so reporting progress here cannot move
    the inputs it is reporting on.  ``_infinite_tag_redraw`` is the module's one
    repaint helper and is reused rather than duplicated, and it can only be seen
    because the work that calls this runs a frame per timer tick: inside one frame
    change nothing repaints, which is what the synchronous form of this measured.
    """
    active = int(total) > 0
    _resimulate_state['active'] = active
    helper = getattr(scene, "gpu_cloth_helper", None)
    if helper is not None:
        if active:
            helper.prepare_state = 'RUNNING'
            helper.prepare_progress = max(
                0, min(100, int(100 * int(done) / int(total))))
            helper.prepare_status = (
                "Re-simulating an invalidated cache: "
                f"frame {int(done)} of {int(total)}")
        else:
            # Always cleared, never left behind: the row belongs to the prepare as
            # well, and a re-simulation that ended while leaving "RUNNING 100%"
            # on it would disable the Prepare button for the rest of the session.
            helper.prepare_state = ''
            helper.prepare_progress = 0
            helper.prepare_status = ''
    _infinite_tag_redraw()


def _resimulate_start(scene, first, target):
    """Hand a re-simulation of ``first..target`` to the timer that owns it.

    A run that has to re-simulate a span can be as long as the range, and doing it
    inside one frame change freezes the window for as long as it takes: measured,
    3.39 s for thirteen frames with the event loop serviced zero times, so nothing
    repaints and the user cannot steer.  That is the freeze the owner reported as
    "всё зависло и пришло ждать, пока просчитается до 106 кадра".  The timer does
    one frame per tick instead; *what* may be produced or written is decided
    exactly where it was decided before, so this changes where the loop runs and
    nothing else.
    """
    global _resimulate_timer_registered
    state = _resimulate_state
    run = _cache_source_state['run_first']
    if (_resimulate_timer_registered and state['active'] and
            state['scene'] == scene.name and state['run'] == run and
            state['first'] == int(first) and int(target) >= int(state['frame'])):
        # A further request on the same run, on the way to where this one is
        # already going: the newest request wins by moving the target, and the
        # frames already produced on the way stay produced.
        state['target'] = int(target)
        _resimulate_notice(
            scene, int(state['frame']) - int(first), int(target) - int(first) + 1)
        return True
    if _resimulate_timer_registered and state['active']:
        _resimulate_finish(scene, "a newer request replaced it")
    state.update({
        'active': True,
        'scene': scene.name,
        'first': int(first),
        'frame': int(first),
        'target': int(target),
        # The run this work belongs to, as the write gate knows it: a change to
        # the inputs drops the run (_clear_retained_frames), and a dropped run's
        # frames must never be produced into the past.
        'run': run,
        # The inputs the span is being produced under, so a move made while the
        # span runs is noticed by this work itself and not only by the handler.
        'generation': _cache_source_generation_fresh(scene),
        'reason': None,
    })
    _resimulate_notice(scene, 0, int(target) - int(first) + 1)
    if _resimulate_timer_registered:
        return True
    try:
        bpy.app.timers.register(
            _advance_resimulation, first_interval=_RESIMULATE_INTERVAL)
    except (AttributeError, RuntimeError):
        _resimulate_timer_registered = False
        _resimulate_finish(scene, "the timer could not be registered")
        return False
    _resimulate_timer_registered = True
    return True


def _resimulate_finish(scene, reason):
    """End the sliced re-simulation and say why, in the state the panel reads."""
    global _resimulate_timer_registered
    _resimulate_timer_registered = False
    _resimulate_state.update({
        'active': False, 'scene': None, 'first': None, 'frame': None,
        'target': None, 'run': None, 'generation': None,
        'reason': str(reason)})
    if scene is not None:
        _resimulate_notice(scene, 0, 0)
    return None


def _advance_resimulation():
    """One frame of a re-simulation, from the timer that owns it.

    The prepare's discipline, reused rather than a third scheduling mechanism: a
    module-level callback that returns the next interval to stay registered and
    ``None`` to unregister itself, with one boolean recording whether it is
    registered.  There is no second liveness rule either - the run that owns this
    work is ``_cache_source_state['run_first']``, the identity the write gate
    already uses, so a change that drops the run stops this work by itself.
    """
    state = _resimulate_state
    if not state['active']:
        return _resimulate_finish(None, "nothing to do")
    scene = bpy.data.scenes.get(state['scene'] or "")
    if scene is None:
        return _resimulate_finish(None, "the scene is gone")
    if _stop_requested or _teardown_failure:
        return _resimulate_finish(scene, "the session was stopped")
    if _cache_source_state['run_first'] != state['run']:
        return _resimulate_finish(scene, "the run that owned this work is gone")
    if _cache_source_generation_fresh(scene) != state['generation']:
        # The inputs moved while the span was being re-simulated.  What has been
        # produced so far belongs to the old inputs and the frame path is about to
        # drop it; producing more would only extend a past that is no longer true.
        # The observation is left where the frame path consumes it before it
        # serves anything (_refresh_prepared_inputs), so the blind window stays
        # closed without this work needing a rule of its own.
        _cache_source_state['deferred'] = True
        return _resimulate_finish(scene, "the inputs changed")
    frame = int(state['frame'])
    if frame > int(state['target']):
        return _resimulate_finish(scene, "done")
    _resimulate_notice(
        scene, frame - int(state['first']) + 1,
        int(state['target']) - int(state['first']) + 1)
    _cache_playback_guard['active'] = True
    try:
        if bpy.context.scene.frame_current != frame:
            scene.frame_set(frame)
        try:
            result = bpy.ops.gpucloth.update_simulation()
        except RuntimeError:
            result = None
    finally:
        _cache_playback_guard['active'] = False
    if result is None or 'FINISHED' not in result:
        return _resimulate_finish(scene, f"frame {frame} did not finish")
    state['frame'] = frame + 1
    return _RESIMULATE_INTERVAL


def _load_cached_frame(scene, depsgraph, frame):
    """Publish one stored frame - if the current run owns the store it is in.

    This is the only reader of the store on the frame path, and until now it read
    whatever the store held: a stored frame is geometry, and the store cannot say
    which inputs produced it.  So the question the write gate already asks before
    a frame is stored (``_cache_run_belongs_to_the_start``) is asked here before
    one is published, and for the same reason - a frame is a statement about the
    inputs in force when its run opened, and a frame whose run is not the current
    one is a frame from another simulation.  Serving it is what the owner reported
    as "на 2 кадре она уже показывает какой-то кэш": the playhead wrapped, the
    frame was requested, and the store answered with a pose written before the
    collider moved.

    A refusal is not a hole: the caller falls through to the retained in-session
    states and then to solving the frame, which is the re-simulation the owner
    ruled for.  What it costs is the read; what it buys is that no frame is ever
    published from a store written under a different input generation.
    """
    if g_dll is None or not _runtime_handle_value() or not _cache_handle_value():
        return False
    if not _cache_run_belongs_to_the_start(scene, frame):
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


def _cache_input_id_key(id_block):
    """A comparable key for a Blender ID without touching its data.

    ``bpy_struct.__eq__`` compares property values recursively, which on a mesh
    datablock means walking the whole mesh for every update entry.  Identity by
    name and type is what the watch set needs and is O(1).
    """
    try:
        return (
            type(id_block).__name__,
            getattr(id_block, "name_full", None) or getattr(id_block, "name", None),
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return None


def _cache_input_watch_ids(scene):
    """The Blender IDs whose own edit can move a fingerprinted input.

    Two sources, and the fingerprint itself is the check on both:

    * every object that contributes a cloth input - an active ``GPUCloth`` object
      and its mesh.  A cloth object's transform is hashed, and so are its mesh
      coordinates and topology.
    * every object that can contribute a collider - an object carrying a
      ``COLLISION`` modifier, and its mesh.  A collider's ``matrix_world``, its
      evaluated vertices and its ``COLLISION`` modifier's scalars are hashed.

    Deliberately not "any ``bpy.types.Object``".  ``depsgraph_update_post`` fires
    for every evaluated object, including effectors, proxies and unrelated scene
    content, and each firing would otherwise cost one full fingerprint -
    measured at ~4 ms on this fixture (`runs/m2-fixed.json`), i.e. ~11 % of a
    37 ms live frame, and one fingerprint per frame through a timeline scrub
    (39 for a 1->20 round trip, `runs/m2-scrub-fixed.json`).  Narrowing the set to
    the objects the digest actually reads keeps the trigger cheap without keying
    it on anything about the cache's state, so the live unbaked session is still
    watched.  The cost of the set itself is two attribute reads per scene object.

    The Scene is not in the set even though three scene scalars (``render.fps``,
    ``render.fps_base``) and the whole ``gpu_cloth_helper`` group are hashed:
    ``GPUCloth`` edits move the property epoch and reach the fingerprint through
    ``_refresh_prepared_inputs``, and an fps edit is picked up by the frame path on
    the next requested frame.  Reacting to every Scene notification as well would
    restore the cost this function exists to avoid.
    """
    watched = set()
    for obj in tuple(getattr(scene, "objects", ())):
        try:
            settings = getattr(obj, "GPUCloth", None)
            active = bool(getattr(settings, "is_active", False))
            is_collider = any(
                getattr(modifier, "type", None) == 'COLLISION'
                for modifier in tuple(getattr(obj, "modifiers", ())))
        except (AttributeError, ReferenceError, RuntimeError):
            continue
        if active or is_collider:
            watched.add(_cache_input_id_key(obj))
            watched.add(_cache_input_id_key(getattr(obj, "data", None)))
            watched |= _cache_animation_watch_ids(obj)
    for obj in _live_cloth_objects():
        watched.add(_cache_input_id_key(obj))
        watched.add(_cache_input_id_key(getattr(obj, "data", None)))
        watched |= _cache_animation_watch_ids(obj)
    watched.discard(None)
    return watched


def _cache_animation_watch_ids(owner):
    """The IDs whose own edit moves what the digest reads about ``owner``.

    ``_cache_animation_identity`` hashes an animated owner by its *timeline data* -
    the action, its fcurves and their keyframes, NLA strips, drivers, and the
    shape-key animation - and by the transform channels the timeline does not
    drive.  A keyframe edit therefore moves a fingerprinted input while Blender
    notifies the ``Action``, not the object, and the watch set held only the object
    and its mesh: ``_cache_input_notification_is_watched`` said no, the handler
    returned before it read the fingerprint, and the store outlived the change, so
    a scene whose collider had just been given keyframes kept serving frames the
    previous animation produced.  Reported by the owner as "кэш не сбросился после
    изменения сцены (я добавил кейфреймы к объекту коллизий)" and reproduced on the
    installed payload before this existed.

    Watching what the digest reads closes it: the actions the identity hashes are
    watched exactly as the object and its mesh already were.  An ``Action`` key is
    ``('Action', name)`` and cannot collide with the object's ``('Object', name)``,
    and the set grows by a handful of entries per cloth or collider - the handler's
    cost is one membership test per notification, which is the cost this set exists
    to keep small.
    """
    keys = set()
    holders = [owner]
    data = getattr(owner, "data", None)
    if data is not None:
        holders.append(data)
        shape_keys = getattr(data, "shape_keys", None)
        if shape_keys is not None:
            holders.append(shape_keys)
    for holder in holders:
        try:
            animation_data = getattr(holder, "animation_data", None)
            if animation_data is None:
                continue
            actions = [getattr(animation_data, "action", None)]
            for track in tuple(getattr(animation_data, "nla_tracks", ())):
                for strip in tuple(getattr(track, "strips", ())):
                    actions.append(getattr(strip, "action", None))
        except (AttributeError, ReferenceError, RuntimeError):
            continue
        for action in actions:
            keys.add(_cache_input_id_key(action))
    keys.discard(None)
    return keys


def _cache_input_notification_is_watched(depsgraph, watched):
    """Do these dependency-graph notifications name an object the digest reads?

    This is the whole of the handler's cheap trigger, and it is its own function
    so that the trigger can be measured without a window: Blender reports no
    ``depsgraph.updates`` at all in ``--background`` on this host (measured:
    zero entries for a collider transform, a collider mesh edit and a collider
    COLLISION setting alike), so a probe that could only ask this question
    through a live event loop could not test the trigger at all.

    Kept narrow on purpose - see ``_cache_input_watch_ids`` for the cost it
    exists to avoid - and kept separate from the fingerprint, which stays the
    only decider of whether anything actually changed.
    """
    for update in depsgraph.updates:
        if _cache_input_id_key(update.id) in watched:
            return True
    return False


def _cache_input_change_handler(scene, depsgraph):
    """Act on a changed cache input that carries a dependency-graph notification.

    This is the only route that sees an edit outside ``GPUCloth``: a collider
    object's transform or its ``COLLISION`` modifier is not a GPUCloth property,
    so it moves no property epoch and the frame path's cheap trigger stays quiet
    for it.  The dependency graph is the notification Blender does deliver, so the
    fingerprint is compared here.

    The comparison used to be gated on ``cached_frame_count != 0``.  That gate made
    the handler unreachable in exactly the session the user reports from: a live,
    prepared, *unbaked* scene.  Measured on this fixture (a pinned PD grid over a
    sphere collider, collider transform edited after frame 4, gate as it was) the
    handler returned before it read the fingerprint, ``_invalidate_cache_for_change``
    never ran, no prepare was scheduled, the published fingerprint had moved, and
    the next cloth step was rejected with ``GPUCLOTH_ABI_SOLVE_FAILED`` (14): a
    fail-closed clearance rejection of a candidate the *stale* owner was asked to
    satisfy.

    The gate is removed rather than widened, and what stands in its place is a
    cheap one that cannot exclude the live session: the notification has to name an
    object ``_cache_input_watch_ids`` decides the fingerprint actually reads.  The
    fingerprint is then the only decider - it is read once and the handler returns
    if it has not moved.  What that cost buys is this handler's own result, so a
    handler that would have been wrong anyway does not get a gate in front of it.

    The decision below is not duplicated here - ``_apply_changed_cache_inputs``
    owns "the fingerprint moved: the past is gone, and re-prepare if the change is
    one only a rebuilt owner accepts", and the frame path calls the same function
    for the edits that do move the property epoch.

    The cheap trigger is read *before* the two gates that can make this handler
    unable to act, because a notification it cannot act on must not be a
    notification it never saw.  A user motion made while a prepare runs or while
    the frame path is inside its own solve used to return here unobserved; the
    frame being solved was then written to the cache stamped with the generation
    it had not been solved under.  Now such a change is *deferred*: the write gate
    refuses every frame until the frame path has consumed it, so the invariant
    holds whatever this handler could or could not do at the time - no frame is
    written, and none is served, that does not belong to the current inputs.

    One notification is not a change and is not deferred: the one the fingerprint
    read itself provokes.  ``_cache_digest_read_in_progress`` records that read,
    and the reason it must be excluded is that the read evaluates the dependency
    graph and the add-on's own output write from the previous frame is what the
    evaluation reports - the read, in other words, notifies this handler about a
    frame the add-on produced.  Recording that as a deferral made the frame path
    re-read the whole fingerprint to learn "nothing moved", on every frame.
    """
    if (g_dll is None or not g_clothOBJs or
            _cache_source_state['generation'] == 0):
        return
    watched = _cache_input_watch_ids(scene)
    if not _cache_input_notification_is_watched(depsgraph, watched):
        return
    if _cache_digest_read_in_progress:
        return
    if prepare_task_active() or _cache_playback_guard['active']:
        global _cache_unactionable_notifications
        _cache_unactionable_notifications += 1
        _cache_source_state['deferred'] = True
        return
    # This handler is the sensor and must read for real: its whole job is to
    # compare against the generation it last stored, and a notification is exactly
    # the event a memo must not answer.  The read is also what the frame path can
    # then reuse, because it is authoritative at the moment it is taken.
    _apply_changed_cache_inputs(scene, _cache_source_generation_fresh(scene))


def _apply_changed_cache_inputs(scene, generation):
    """One owner for "the fingerprint moved": act on it, or report no change.

    Both entry points that can notice an input change call this: the dependency
    graph handler for an edit outside ``GPUCloth`` (a collider object's transform
    or its ``COLLISION`` modifier), and the frame path for an edit to a GPUCloth
    property, whose change carries no depsgraph notification.  Keeping the decision
    in one place is the point - the two entry points had drifted into different
    gates over the same fingerprint, and the gate that was wrong is what left a
    changed collider unnoticed.

    Returns True when the change needs a rebuilt owner, so the caller can drop the
    frame it was asked for.

    What the change costs the cache no longer depends on the caller: every change
    drops the whole past (``_invalidate_cache_for_change``), and the class of the
    change decides only whether the owner is rebuilt or re-tuned in place.
    """
    # The observation is complete as soon as the fingerprint has been read, whether
    # or not it moved: a deferral that outlived its own check would refuse every
    # later write for a change that was never there.
    _cache_source_state['deferred'] = False
    if generation == _cache_source_state['generation']:
        return False
    # A change is in force.  The frame-path memo is dropped here, where the change
    # is known, so nothing downstream can be handed the value that was current
    # before it - and the memo is not keyed on this alone, because a change that
    # arrives while this handler is busy is recorded as a deferral rather than
    # applied, and the next read has to be a real one.
    _cache_fingerprint_invalidate()
    profile = _live_input_profile
    profile['gate_ms'] = []
    mark = (lambda label, started: profile['gate_ms'].append(
        (label, (time.perf_counter_ns() - started) / 1e6))) \
        if profile['enabled'] else (lambda label, started: None)
    started = time.perf_counter_ns()
    _cache_source_state['generation'] = generation
    _, staged_digest = _cache_input_digests_for_frame(scene)
    mark('digest', started)
    staged = staged_digest != _cache_source_state['staged']
    started = time.perf_counter_ns()
    _invalidate_cache_for_change(scene, generation, staged)
    mark('invalidate', started)
    if staged:
        _cache_source_state['staged'] = staged_digest
        started = time.perf_counter_ns()
        _stage_changed_inputs(scene, generation)
        mark('stage_and_apply', started)
        return True
    started = time.perf_counter_ns()
    _apply_live_inputs(scene)
    mark('apply_live_inputs', started)
    return False


def _invalidate_cache_for_change(scene, generation, staged=True):
    """Tell the cache that the simulation inputs changed: the past is gone.

    This is Blender's own physics semantics, and the owner ruled that we follow it
    exactly rather than refine it (round-2 hand test: "в MD и Vellum такой логики
    нет в принципе, так что делаем так как делает сам Blender ... если в сцене
    что-то меняется, то кэши сбрасываются и не пишутся пока не начнём с начала").
    So "what survives a change" has one answer for every class of change - nothing
    - and ``staged`` no longer decides that.  It decides only **how the change is
    applied**: a staged change needs a rebuilt owner (``_stage_changed_inputs``),
    a live-safe one is re-tuned in place (``_apply_live_inputs``).

    This replaces the prefix that used to survive a live-safe change.  That prefix
    had a real reason - clearing everything made every later backward request
    re-simulate the range from its first frame, the freeze the owner reported as
    "всё зависло и пришло ждать пока просчитается до 106 кадра" - but it was also
    the splice he reported as "12 и 14 кадр шли нормально, а 13 почему-то был уже
    из другого участка симуляции": a kept frame is geometry the *old* inputs
    produced, served under the new ones, and nothing downstream can tell it from a
    current frame.  A predictable re-simulation is the price of a past that is
    always true, and the re-simulation is made visible rather than silent - see
    ``_resimulate_notice``.

    The frames are **removed**, not marked stale.  The engine has a frame-granular
    stale marker (``outdated_from_frame``) but it is armed only against a baked
    baseline: ``s_v3_cache_frame_is_current`` (main.cpp:11921-11928) reads
    "current" for *every* frame while ``baked_source_generation`` is zero, which is
    the state of a live session that has not completed a bake.  So an unbaked cache
    cannot be told where it stopped being true, and the only identity that makes a
    frame from before the edit unreachable is its absence.

    Both stores go together, because a replay can be served from either and the two
    must not disagree about what the past is.

    Returns True when the cache was cleared.
    """
    _clear_retained_frames()
    if not _cache_handle_value():
        return False
    if int(g_dll.GPUCloth_v3_cache_clear(
            g_runtime_handle, _cache_handle_owner())) == CType.GPUCLOTH_ABI_OK:
        helper = scene.gpu_cloth_helper
        helper.bake_progress = 0
        helper.playback_mode = False
        try:
            _sync_cache_status(scene)
        except (OSError, RuntimeError) as exc:
            print(f"GPUCloth cache status refresh failed: {exc}")
            helper.is_baked = False
            helper.cached_frame_count = 0
        # The clear deleted the disk/external status metadata, so the engine has no
        # baseline left to compare a generation against - which is correct now: the
        # cache holds nothing, and nothing may be written into it again until the
        # simulation runs from the start of the range.  Stating that on the whole
        # range is also what raises the engine's own OUTDATED flag, which is
        # Blender's "the cache is out of date, simulate again" contract - and it is
        # raised *before* the status is read, or the flag lands after the read and
        # the panel never sees it (measured: is_outdated stayed False across a
        # change, with the whole re-simulation in front of the user).
        try:
            _cache_status_update(
                CType.GPUCLOTH_CACHE_STATUS_SOURCE_CHANGED, scene,
                generation=generation)
        except (OSError, RuntimeError) as exc:
            print(f"GPUCloth cache status note failed: {exc}")
        try:
            _sync_cache_status(scene)
        except (OSError, RuntimeError) as exc:
            print(f"GPUCloth cache status refresh failed: {exc}")
        return True
    print(
        "GPUCloth cache clear after an input change was refused; telling the "
        "cache owner its inputs changed instead")
    try:
        _cache_status_update(
            CType.GPUCLOTH_CACHE_STATUS_SOURCE_CHANGED, scene,
            generation=generation)
        _sync_cache_status(scene)
    except (OSError, RuntimeError):
        scene.gpu_cloth_helper.playback_mode = False
    return False


def _apply_live_inputs(scene):
    """Re-publish every live-safe configure that has actually changed.

    This is the whole fix for a live-safe edit: the engine reads the solver
    configuration out of ``clmd->sim_parms`` on every solve (main.cpp:9042-9064)
    and these features write it directly, so the very next frame integrates with
    the new values and the simulation keeps its state.  Nothing here restarts or
    rebuilds, which is the point - a settings change must not cost a prepare.

    The block used to republish every group whenever the *whole-scene*
    fingerprint moved, which is not the same thing: a collider transform moves
    the full digest, and the response was to re-capture and re-publish the
    cloth's own material features, internal springs, pressure and collision
    contract - none of which a collider can change.  That is why a collider move
    cost the material group's O(corners) capture at all.

    Each group therefore carries its own fingerprint, built the same way
    ``_cache_input_digests`` builds its two streams, and only a group whose
    fingerprint moved is republished.  Publishing unchanged inputs is not a
    contract; it is work with no effect.  The first call after a prepare
    publishes everything, because nothing has been published yet.
    """
    for index, cloth_obj in enumerate(g_clothOBJs):
        if index >= len(g_cloth_handles) or index >= len(_cloth_input_owners):
            continue
        settings = cloth_obj.GPUCloth
        handle = g_cloth_handles[index]
        owner = _cloth_input_owners[index]
        changed = _live_group_changes(scene, index, cloth_obj, settings)
        # Each group is published on its own.  A group that the engine refuses
        # must not hide the groups after it: the refusal that reached the user
        # named feature 8 only because the publisher raised on the first one and
        # never reached the stiffness features behind it.  Every refusal is
        # reported, and the owner is left with whatever the engine did accept.
        if _live_input_profile['enabled']:
            _live_input_profile['steps'] = []
            _live_input_profile['changed_groups'] = sorted(changed)
        steps = (
            ("simulation features", changed, lambda:
                _configure_simulation_features(
                    g_dll, handle, scene, settings, live_only=True)),
            ("material features", changed, lambda: _publish_material_features(
                g_dll, handle,
                _capture_material_features(
                    settings,
                    # FABRIC is the only material model, so the live path always
                    # speaks the per-corner (v3) coordinate stream.
                    capture_material_coordinates(
                        g_simulationOBJs[index], settings,
                        owner["topology_generation"],
                        v3_corner=True)),
                live_only=True)),
            ("internal springs", changed, lambda:
                _publish_internal_springs_config(
                    g_dll, handle, _capture_internal_springs_config(settings),
                    live_only=True)),
            ("pressure features", changed, lambda: _publish_pressure_features(
                g_dll, handle, _capture_pressure_features(settings),
                live_only=True)),
            ("cloth collision config", changed, lambda:
                _publish_cloth_collision_config(
                    g_dll, handle, _capture_cloth_collision_config(settings),
                    live_only=True)),
        )
        for label, changed_groups, publish in steps:
            if label not in changed_groups:
                continue
            step_started = time.perf_counter_ns()
            try:
                publish()
            except (OSError, RuntimeError, VertexChannelError) as exc:
                print(f"GPUCloth live reconfigure failed: {label}: {exc}")
            finally:
                if _live_input_profile['enabled']:
                    _live_input_profile['steps'].append(
                        (label, (time.perf_counter_ns() - step_started) / 1e6))


def _live_group_setting_names(settings):
    """Every GPUCloth scalar that feeds one live-published group, by group.

    ``_scene_live_setting_names`` is the same declaration for the scene helper
    and exists because "the two halves can drift apart" - a scalar stays live in
    the publisher while the digest keeps watching something else.  This is its
    per-group counterpart for the cloth's own settings: a name that decides a
    group's published payload appears in that group's list, and a name that does
    not decide anything does not.  The lists are folded into a fingerprint by
    name, so adding a setting to a publisher without adding it here makes that
    edit invisible to the live path - the drift this exists to prevent.

    FABRIC is the only material model, so the two groups it feeds - the areal
    mass payload and the triangle-membrane payload - always carry their FABRIC
    names; no owner setting can take them out.
    """
    return {
        # `_configure_simulation_features` reads these.
        "simulation features": (
            "solver_type", "quality_step", "speed_multiplier", "vertex_mass",
            "air_viscosity", "solver_iterations", "solver_krylov_iterations",
            "mass_mode", "fabric_density",
        ),
        # `_capture_material_features` reads these, and its directional and
        # damping blocks are fed from the same names, so nothing that reaches the
        # published payload is left out.  The material *coordinates* are derived
        # from the mesh, not from these; they are constant for a prepared owner,
        # which is why a per-frame fingerprint does not have to hash 96 774
        # corners to know the material group did not change.
        "material features": (
            "bending_model", "use_anisotropy",
            "anisotropy_uv_map", "bending_stiffness", "max_bend",
            "tension", "compression", "shear",
            "max_tension", "max_compression", "max_shear",
            "tension_damp", "compression_damp", "shear_damp",
            "bending_damping",
            "tension_u", "tension_v", "compression_u", "compression_v",
            "bending_u", "bending_v",
            "max_tension_u", "max_tension_v",
            "max_compression_u", "max_compression_v",
            "max_bend_u", "max_bend_v",
            "fabric_tensile_u", "fabric_tensile_v",
            "fabric_compression_u", "fabric_compression_v",
            "fabric_tensile_u_max", "fabric_tensile_v_max",
            "fabric_compression_u_max", "fabric_compression_v_max",
            "fabric_tensile_damping", "fabric_compression_damping",
            "fabric_shear_damping", "fabric_shear_c66", "fabric_shear_c66_max",
        ),
        "internal springs": (
            "use_internal_springs", "use_internal_springs_normal",
            "internal_spring_max_length", "internal_spring_max_diversion",
            "internal_tension", "max_internal_tension",
            "internal_compression", "max_internal_compression", "max_sewing",
        ),
        "pressure features": (
            "use_pressure", "use_pressure_volume", "pressure_factor",
            "uniform_pressure_force", "fluid_density", "target_volume",
        ),
        # `_capture_cloth_collision_config` reads these; the collider's own
        # surface settings are the collider's, and they reach the engine through
        # the collection transaction, not through this group.
        "cloth collision config": (
            "use_object_collision", "collision_friction", "collision_damping",
            "collision_quality", "epsilon", "self_collision_friction",
            "use_self_collision", "self_collision_quality",
            "self_collision_distance", "self_collision_impulse_clamp",
        ),
    }


def _live_group_fingerprints(scene, index, cloth_obj, settings):
    """One digest per live-published group, over the inputs that decide it."""
    names_by_group = _live_group_setting_names(settings)
    hasher = hashlib.blake2b(digest_size=8, person=b"GPCLive")
    scene_helper = scene.gpu_cloth_helper
    for name in _scene_live_setting_names():
        _cache_hash_value(hasher, f"scene.{name}", getattr(scene_helper, name))
    _cache_hash_value(hasher, "render.fps", scene.render.fps)
    _cache_hash_value(hasher, "render.fps_base", scene.render.fps_base)
    result = {}
    for group, names in names_by_group.items():
        group_hasher = hasher.copy()
        for name in names:
            if hasattr(settings, name):
                _cache_hash_value(
                    group_hasher, f"{group}.{name}", getattr(settings, name))
        # The collection a collider is selected through decides the collision
        # group, and it is a pointer property the scalar walk cannot see.
        if group == "cloth collision config":
            referenced = getattr(settings, "collision_collection", None)
            _cache_hash_value(
                group_hasher, f"{group}.collection",
                getattr(referenced, "name_full", None)
                if referenced is not None else None)
        result[group] = int.from_bytes(group_hasher.digest(), "little")
    return result


def _live_group_changes(scene, index, cloth_obj, settings):
    """Which live groups a group-carrying fingerprint says have moved.

    Returns every group on the first call after a prepare, because no group has
    been published on the live path yet and the engine is holding whatever the
    prepare left it with.
    """
    fingerprints = _live_group_fingerprints(scene, index, cloth_obj, settings)
    state = _live_group_state.get(index)
    if state is None:
        _live_group_state[index] = fingerprints
        return set(fingerprints)
    changed = {group for group, value in fingerprints.items()
               if state.get(group) != value}
    _live_group_state[index] = fingerprints
    return changed


def _stage_changed_inputs(scene, generation):
    """Route a change only a rebuilt owner can accept to the prepare path.

    The native owner integrates ``clmd->clothObject->verts`` in place and
    exposes no rest-reset entry point, so a staged input (anisotropy, the
    constraint network, shrink, rest shape, dynamic mesh, the AREAL mass, the
    vertex-damping and effector-scale stages, the self-collision toggle, the
    prepared collection snapshots, topology) can only take effect from a rebuild.
    ``_cache_input_digests`` already decided that this is the case; this function
    restores the rest shape so the rebuild captures it, and queues the prepare on
    the existing path.
    """
    _apply_live_inputs(scene)
    _clear_retained_frames()
    _simulation_frame_state['rebuild_pending'] = True
    if _initial_positions:
        _restore_initial_positions()
    cloth_objects = _live_cloth_objects()
    if cloth_objects:
        schedule_auto_prepare(cloth_objects[0], scene, required=True)


def _refresh_prepared_inputs(scene):
    """Act on a changed cache input while a run is prepared.

    A solver setting carries no depsgraph notification, so the frame path is the
    only place a user edit can be noticed.  The shared property epoch is a cheap
    trigger deciding when the fingerprint is worth recomputing; the fingerprint
    is the sole detector of *what* changed, and ``_apply_changed_cache_inputs``
    owns the action - so an edit that reaches this path and an edit that reaches
    the dependency graph handler are handled by the same decision.  Returns True
    when a rebuild was queued, so the caller can drop the frame it was asked for.

    This is also where a *deferred* observation is consumed: the dependency-graph
    handler cannot act while a prepare runs or while this path holds the playback
    guard, so it records the observation instead of dropping it, and this call is
    the frame path's next entry.  Until it is consumed no frame may be written
    (``_frame_may_be_persisted``), which is what makes a motion made during a solve
    unable to persist anything.

    The read is the frame's one read.  It happens when the property epoch has
    moved since this path last looked, and what it produces is primed into the
    frame-path memo, which is what ``_live_step_cloth_scene`` and the write gate
    then use.  The deferral flag is deliberately *not* a trigger here: it is the
    write gate's signal, and it is raised by the add-on's own output as well as by
    a user's hand - measured, gating this read on it made every frame read twice,
    the second time to learn what the first read had already established.

    Nothing here is keyed on time.  The epoch is the property revision, and the
    dependency-graph route does not need a re-read to be safe: that handler reads
    the fingerprint itself, every time it is allowed to act, and applies the change
    in the same call - so by the time this path runs there is nothing left for it
    to discover.
    """
    epoch = _cache_fingerprint_epoch()
    if epoch == _cache_source_state['epoch']:
        return False
    _cache_source_state['epoch'] = epoch
    if not _runtime_handle_value() or not g_clothOBJs:
        return False
    generation = _cache_source_generation_fresh(scene)
    return _apply_changed_cache_inputs(scene, generation)


# Blender's playback writes ``frame_current`` itself, so a playhead move made
# while it runs is invisible unless this handler compares the frame it is handed
# against the frame playback's own last step would have produced.  The recorded
# pair lives beside the other handler state; ``frame`` is updated on every change
# this handler sees, including the ones its own forward-solve loop and the bake
# modal make, so a legitimate jump is only ever misread once.
_playhead_watch = {'frame': None, 'playing': False}


def _playhead_moved(scene):
    """True when this frame change is a move of the playhead, not a playback step.

    ``_infinite_advance`` already owns this rule for the other driver: "the mode's
    premise is that the timeline owns nothing, so any movement of the frame
    counter is a stop condition with a reason, never silent".  Under Blender's
    playback the same rule applies to the playhead - a scrub while playing must
    stop the drive and leave the frame where the owner put it, instead of being
    overwritten by the next tick and run forward from.

    Playback's own step is the successor frame, or the wrap back to
    ``frame_start``; anything else while it is running is a move.  The test is
    gated on ``sync_mode == 'NONE'`` because that is the only mode where a step
    is a successor: under ``FRAME_DROP`` playback legitimately skips frames to
    hold the frame rate, and a successor rule would cancel it for keeping up.
    Modes this add-on cannot classify are left alone - reporting nothing is
    better than stopping a user's playback for a reason that is not true.
    """
    current = int(scene.frame_current)
    playing = bool(getattr(
        getattr(bpy.context, "screen", None), "is_animation_playing", False))
    previous = _playhead_watch['frame']
    was_playing = _playhead_watch['playing']
    _playhead_watch['frame'] = current
    _playhead_watch['playing'] = playing
    if not (playing and was_playing) or previous is None:
        return False
    if str(getattr(scene, "sync_mode", "NONE")) != 'NONE':
        return False
    expected = int(previous) + 1
    if expected > int(scene.frame_end):
        expected = int(scene.frame_start)
    return current != expected


def _place_playhead_at_simulation_start(scene):
    """Put the playhead back on the frame the rebuilt simulation starts from.

    The owner: "После изменения настроек и повторного Prepare каретка должна
    возвращаться в начало симуляции.  Сейчас не так."  A prepare publishes the
    cloth at the rest state it captured, and a run's first frame is
    ``max(2, int(bake_start))`` - the frame path's own ``range_first`` rule, and
    the very frame ``last_solved`` is left one before (``max(1, bake_start - 1)``,
    the prepare path).  So the frame the cloth now shows is the start of the
    range, and that is where the carriage belongs: a playhead left where it was
    would be describing a frame of a simulation that no longer exists.

    The move is the add-on's own and is marked as such.  ``_cache_playback_guard``
    is that mark and this is a reader of it rather than a second flag: the frame
    handler returns the moment it sees it, so the move cannot be mistaken for the
    user's hand - and, since a bare move no longer simulates
    (``_frame_producer_running``), a move that *was* mistaken for one would land
    on a frame nothing had produced and leave the mesh where it stood.  That is
    the failure this guard exists to prevent here, not a formality.

    Nothing else about the frame is touched: no frame is produced, no cache is
    read, and the frontier is not moved.  The cloth is already the geometry of
    this frame - the prepare published it - so producing it would be a step the
    owner did not ask for, which is the whole of the scrub ruling.
    """
    if scene is None or not g_clothOBJs:
        return False
    target = max(2, int(_bake_range['start']))
    if int(scene.frame_current) == target:
        return False
    previous = _cache_playback_guard['active']
    _cache_playback_guard['active'] = True
    try:
        scene.frame_set(target)
    finally:
        _cache_playback_guard['active'] = previous
    print(f"[GPUCloth] playhead returned to the start of the simulation, "
          f"frame {target}")
    return True


# The frame the owner moved to while the drive was up.  The move's cancel does
# not take effect inside the run of playback steps it is issued from - measured on
# a live session, those steps keep arriving, one per dispatch, and each one is a
# frame change this handler would otherwise solve and show.  That is why the
# timeline used to end up past the frame the owner put it on.  The hold is what
# makes those steps not the answer; ``_settle_playhead_hold`` is what ends it,
# because the drive stops between dispatches and its last step leaves no frame
# change behind that could notice.
_playhead_hold = {'frame': None}
# The hold's release cadence, and nothing else: while the hold is set the frame is
# held by the handler, not by how often this runs.  The drive steps once per
# playback dispatch, so a poll of that order sees it go down promptly.
_PLAYHEAD_HOLD_POLL = 0.05


def _stop_animation_playback():
    """Ask Blender's playback to stop; report whether it was still up to ask.

    One ask does not carry far on the measured path - the cancel is answered only
    after the steps already under way have run - so a caller that needs the drive
    actually down has to keep asking rather than treat the first ask as the stop.
    Measured on the owner's own scene (build/r22-defect6/live, control arm): the
    ask from a plain timer callback takes effect in **56 ms**, and from inside a
    frame change of the drive it took **1.054 s**, with the already-dispatched
    steps walking the playhead 8 -> 23 in between.  So this returns "the drive
    was up when asked", which is a request, not a state change, and the caller
    owns holding whatever it promised until the drive answers.

    ``restore_frame=False`` is the add-on's own definition of the drive stopping:
    the playhead stays where the drive was stopped, which is what a pause means,
    and it is the flag the hold path depends on - a restore would put the frame
    back where playback started and take the owner's moved-to frame away.

    This is the one authority for the ask.  A caller with its own idea of it -
    ``bpy.ops.screen.animation_cancel()`` with another flag, or a second copy of
    the try/except - is a second stop path that can disagree with this one.
    """
    if not _animation_is_playing(bpy.context):
        return False
    try:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        pass
    return True


def _settle_playhead_hold():
    """Hold the moved-to frame until the drive lets go, then produce it there.

    The hold drops every step of the drive the move interrupted, so none of them
    is solved and the geometry never leaves the frame the owner asked for.  Once
    the screen says the drive is down the timeline goes back to that frame and the
    frame path produces it; it is that frame's first produce, because the dropped
    steps were dropped in front of it.
    """
    held = _playhead_hold['frame']
    if held is None:
        return None
    if _animation_is_playing(bpy.context):
        _stop_animation_playback()
        return _PLAYHEAD_HOLD_POLL
    _playhead_hold['frame'] = None
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        scene.frame_set(int(held))
    return None


def _frame_producer_running():
    """True while something that owns frame production is driving the playhead.

    The one rule behind "no Space, no simulation": a frame change is produced
    only while a driver is running, and the two drivers are the two Blender
    already names - Blender's own playback, which is the owner's Space, and the
    infinite mode, whose steps publish without a frame change at all.

    The three other frame paths are deliberately *not* asked here, and the
    reason is the same for all three: each of them is already the producer, so
    asking would answer itself.  ``_playhead_hold`` produces the one frame a
    move asked for and reaches this handler only after its own hold has been
    drained; the re-simulation timer and the bake modal both call the step
    operator directly, which publishes without returning here.
    ``_cache_playback_guard`` in particular cannot be a term of this rule even
    though it means "a frame the add-on set itself": this handler sets it
    itself, before it asks - measured on the probe's own live arm, the guard
    read ``True`` at the gate on a bare scrub, which turned the gate into a
    no-op.  A flag that is true whenever the question is asked answers nothing.

    So a bare move of the playhead has neither driver.  It is not a request to
    simulate: it is a request to *look*, so the frame path serves what the run
    already owns and otherwise leaves the mesh exactly as it was.  The two rules
    the movement of the playhead already had are untouched and run before this -
    ``_playhead_hold`` produces the one frame a move asked for, and
    ``_playhead_moved`` is the one owner of "is this a step or a move".
    """
    if _animation_is_playing(bpy.context):
        return True
    return _infinite_is_running()


def playhead_frame_status(scene):
    """One pair of strings, or None, for the panel's "not simulated" row.

    The frame handler's own decision, asked rather than restated: a frame is
    simulated when the request is at or below the frontier the current run
    reached (``last_solved``), and outside the baked range there is no
    simulation to speak of.  A move onto a frame past that frontier produces
    nothing (``_frame_producer_running``), so the panel has to say which of the
    two the owner is looking at instead of letting an unchanged mesh imply the
    simulation ran.  The row clears by itself on the next frame that is
    produced, because the answer is computed from the frontier and not stored.
    """
    helper = getattr(scene, "gpu_cloth_helper", None)
    if helper is None or not g_clothOBJs:
        return None
    if bool(getattr(helper, "is_baked", False)):
        # A baked session's playback owns every frame of the range, and the
        # frame path's baked branch never reaches the producer question.
        return None
    frame = int(scene.frame_current)
    if frame < max(2, int(_bake_range['start'])):
        return None
    if frame > int(getattr(helper, "bake_end", _bake_range['end'])):
        return None
    last_solved = _simulation_frame_state['last_solved']
    if last_solved is not None and frame <= int(last_solved):
        return None
    if (int(frame) == max(2, int(_bake_range['start'])) and
            int(last_solved or 0) < int(_bake_range['start'])):
        # The first frame of a run that has not stepped yet: the solver is
        # holding exactly the state this frame is, because a prepare publishes
        # the rest pose it captured and ``last_solved`` is left one before the
        # range.  Nothing was solved at this frame and nothing needs to be - it
        # is what ``_place_playhead_at_simulation_start`` lands the playhead on
        # - so calling it unproduced would report a hole where there is none.
        return None
    if _frame_producer_running():
        return None
    return _t_infinite(
        f"Frame {frame}: not simulated - the simulation is at frame "
        f"{int(last_solved) if last_solved is not None else 1}.  "
        f"Space starts it.",
        f"Кадр {frame}: не просчитан - симуляция на кадре "
        f"{int(last_solved) if last_solved is not None else 1}.  "
        f"Запускает Space.")


def _frame_change_handler(scene, depsgraph):
    if prepare_task_active() or _teardown_failure or _stop_requested:
        return
    if _cache_playback_guard['active']:
        # A frame this add-on set itself is not a playhead move, but it still
        # advances the watch, or the next playback tick after a bounded forward
        # solve would be compared against a frame from before it.
        _playhead_watch['frame'] = int(scene.frame_current)
        _playhead_watch['playing'] = bool(getattr(
            getattr(bpy.context, "screen", None), "is_animation_playing",
            False))
        return
    held = _playhead_hold['frame']
    moved = _playhead_moved(scene)
    if held is not None and int(scene.frame_current) != held:
        # A move armed a hold and the drive is letting go of the frame one step at
        # a time.  ``_playhead_moved`` owns the one rule that tells a step of the
        # drive from the owner's hand - a step is the successor of the frame
        # playback last produced, his move is anything else - so the hold asks it
        # rather than keeping a second copy of that rule.  A step is dropped; a
        # newer answer, and anything arriving once the drive is down, supersedes
        # the hold and is produced.
        if moved or not _animation_is_playing(bpy.context):
            _playhead_hold['frame'] = int(scene.frame_current)
        else:
            return
    elif held is None and moved:
        # Stop the drive, and still produce the frame the owner moved to: the move
        # is a request for one frame, not a request for nothing.  The hold keeps
        # the steps the cancel has not stopped yet from taking that frame away.
        _playhead_hold['frame'] = int(scene.frame_current)
        if not bpy.app.timers.is_registered(_settle_playhead_hold):
            try:
                bpy.app.timers.register(
                    _settle_playhead_hold, first_interval=_PLAYHEAD_HOLD_POLL)
            except (AttributeError, RuntimeError):
                pass
        _stop_animation_playback()
        print(
            f"[GPUCloth] playback stopped: the playhead moved to frame "
            f"{int(scene.frame_current)}")
    # This handler's own body reads ``g_clothOBJs`` once (the liveness test
    # below), but the functions it drives do not: ``_restore_initial_positions``
    # and ``_load_simulation_frame`` walk it, ``_publish_frame_inputs`` reads
    # its length against ``g_simulationOBJs``, and ``_live_step_cloth_scene``
    # indexes it.  None of them can tell a live entry from a freed one, so the
    # prune happens here, once, before any of them runs: the dependency-graph
    # handler has usually pruned already - Blender sends that update before it
    # sends the frame change - and this call closes the order left when it has
    # not.
    _live_cloth_objects()
    if _refresh_prepared_inputs(scene) or (
            _simulation_frame_state['rebuild_pending']):
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
                if not _load_cached_frame(scene, depsgraph, frame):
                    _load_simulation_frame(frame, depsgraph)
                return
            if frame > scene_s.bake_end:
                return
            # A frame the current run has already produced is a read, not a
            # re-solve - and the run's own store is where it is.  Without this the
            # decision was made by direction: a request at or below the frontier
            # was served with one store read, while a request ahead of it was
            # handed to the re-simulation timer and solved one frame per tick.  On
            # the owner's dense scene that is 100-400 ms per frame, which is what
            # he reported as the cache loading "only after I let go of the mouse":
            # the two directions differed by *how* the frame was reached, not by
            # whether anything was cached.  ``_load_cached_frame`` answers only for
            # a frame the current run owns under the inputs in force, so a hit here
            # is the same frame the re-simulation would have reproduced, reached
            # without re-solving the span in front of it.  A miss is where the
            # request stops being a read.
            if _load_cached_frame(scene, depsgraph, frame):
                _simulation_frame_state['last_solved'] = max(
                    int(_simulation_frame_state['last_solved'] or 0), int(frame))
                _store_simulation_frame(frame)
                return
            if not _frame_producer_running():
                # The read missed and nothing is driving the timeline, so this
                # frame was never produced - and a move of the playhead is not a
                # request to produce it.  The owner's ruling is the reason the
                # handler stops here instead of handing the span to the
                # re-simulation: "если с подготовленной тканью я перемещаю
                # каретку, то надо ждать, когда ткань досимулирует до нового
                # кадра.  Так не надо.  Не нажимали Space - не запускаем
                # симуляцию".  Before this, a scrub past the frontier started
                # ``_resimulate_start`` and solved every frame in between, one
                # per tick, at 100-400 ms each on the owner's route.  Nothing is
                # published here either: the mesh keeps the pose it had, because
                # restoring the rest pose is a change the owner did not ask for
                # and is farther from the simulation than the pose he is looking
                # at.  What he gets instead is the panel's own statement that the
                # frame was not simulated (``playhead_frame_status``), so "not
                # simulated yet" cannot be read as "this is the simulation".
                # Pressing Space lands on the same branch with a driver running,
                # and the span is produced from ``last_solved + 1`` exactly as it
                # was before - which is what makes the fix not take playback away.
                return
            range_first = max(2, int(scene_s.bake_start))
            first = max(
                range_first,
                (last_solved + 1) if last_solved is not None else range_first)
            if int(first) == range_first:
                # This span starts where the range starts, so it is a run beginning
                # at the beginning - the only kind of run that may own a past.
                # Asked of the *span* rather than of "is there a past": a prepare
                # leaves `last_solved` at the frame before the range (there is
                # nothing to serve, but there is a number), so asking the old way
                # never opened a run at all.  Measured with the old way: every
                # forward request re-solved 2..N from scratch - `кадр 2`, then
                # `кадр 2, 3`, then `кадр 2, 3, 4`, ... - because `last_solved` is
                # only advanced for a frame of an open run, and the cache was never
                # written, because a frame of no run may not be persisted.
                _cache_run_open(scene, first)
            if frame - first + 1 > 1:
                # A span longer than one frame - the re-run after a change, or a
                # scrub that jumps ahead - is handed to the timer instead of being
                # solved here.  Solving it here is a frozen window for as long as
                # it takes, which is the defect the owner reported as "всё зависло
                # и пришло ждать, пока просчитается до 106 кадра".  The timer
                # produces the same frames in the same order, one per tick, and
                # every decision about what may be produced or written stays where
                # it was.
                _resimulate_start(scene, first, frame)
                return
            if frame < first:
                # Nothing to produce: the request is before the frame the range
                # starts at (frame 1 is the rest state and is not solved).  The
                # forward solve this replaced did nothing here either, and a frame
                # the range never covers must not be produced into the past.
                return
            # One frame: the ordinary advance, produced here and now.
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

def _destroy_session_cloth_owners():
    """Destroy this prepare's cloth owners, leaving the runtime alive.

    The v3 runtime is a session owner: the add-on creates it once when the DLL
    loads and destroys it when the session ends.  A re-prepare does not need a
    new one - it needs the *owners* built on top of it retired - and
    ``GPUCloth_v3_cloth_destroy`` is exactly that operation: it aborts the
    runtime's collection transactions, destroys the proxies that belong to the
    cloth, destroys the cloth owner and erases it from the runtime's map.  A
    later create with the same ``object_id`` is therefore admitted, which is what
    the prepare path needs.

    Ordering is by index, because that is the order the lists are filled in
    (``g_cloth_handles``, ``_cloth_input_owners``, ``_readback_owners`` are
    appended together per cloth).  A destroy that fails is reported and the
    walk continues to the remaining owners; returning False leaves the caller to
    fall back to the full teardown, so a refused destroy can never produce a
    partially-retired owner set that the next prepare would build on.
    """
    if g_dll is None:
        return False
    if not g_cloth_handles:
        # Nothing was built: the raw owners this prepare created (if any) are
        # not reachable by handle and the runtime's own destroy covers them.
        return True
    ok = True
    for index in range(len(g_cloth_handles) - 1, -1, -1):
        handle = g_cloth_handles[index]
        try:
            result = int(g_dll.GPUCloth_v3_cloth_destroy(handle))
        except Exception as exc:                              # noqa: BLE001
            print(f"free_gpu_memory: cloth destroy raised: {exc}")
            ok = False
            continue
        if result != CType.GPUCLOTH_ABI_OK:
            print(f"free_gpu_memory: cloth destroy rejected with {result}")
            ok = False
    return ok


def _reset_owner_python_state(context=None):
    """Reset the Python-side state that describes the owners just released.

    This is hygiene, not teardown: it owns no device resource, so it runs on
    both paths.  It deliberately does NOT touch `_runtime_frame_generation` or
    `_cache_source_state['generation']`: both are baselines that describe the
    *runtime*, not the owners.  `_runtime_frame_generation` must stay monotonic
    while a runtime survives - `_runtime_update` refuses a generation that does
    not increase, and the one-shot pre-warm already created the runtime - and
    `_cache_source_state['generation']` is the cache session's baseline, which
    retiring owners is not allowed to invalidate.  Both resets therefore live in
    `free_gpu_memory`, the path that destroys the runtime they describe.
    """
    global g_cloth_handles, _cloth_input_owners
    global _readback_owners, g_cache_handle, g_cache_owner
    global g_clothOBJs, g_simulationOBJs, g_clothCollisionOBJs, g_proxy_handles
    global _collider_history

    g_cloth_handles      = []
    _cloth_input_owners  = []
    _readback_owners     = []
    g_clothOBJs          = []
    _cloth_object_identity.clear()
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
    # The retained payloads are only valid under the history entry that admits
    # them, so they are released with the owners that history describes - here,
    # and nowhere later.  `release_prepare_owners` deliberately does not clear
    # either one: a reused runtime keeps its history.
    _collider_payload_cache.clear()
    _collider_geometry_cache.clear()
    _reset_collider_payload_metrics()
    _live_group_state = {}
    _collider_certificate_cache.clear()
    _collider_certificate_state['fit_count'] = 0
    _collider_certificate_state['reuse_count'] = 0
    _collider_certificate_state['eviction_count'] = 0
    _drape_status_by_uid.clear()
    _drape_settle_verdict_by_uid.clear()
    # The prepared rest pose is deliberately NOT cleared here.  It is scene
    # state - the pose a successful prepare validated and rebuilt from - and it
    # does not depend on the native owners this function retires.  Clearing it
    # with them is what made a refused rebuild unrecoverable: the preparation
    # preflight refuses a pose the simulation itself produced (measured: the
    # drag's own fold, witness `self intersection; faces 3722/4488`), the refusal
    # releases the owners through this function, and the one pose that could have
    # been rebuilt from went with them - so every later prepare re-captured the
    # same refused mesh and the session could not come back.  `_store_initial
    # _positions` is the only writer: it clears and refills the list at the end
    # of a prepare that succeeded, so what survives is always a pose a
    # preparation accepted.
    _clear_retained_frames()
    _cache_source_state['generation'] = 0
    _resimulate_state['active'] = False
    _input_generation['value'] = 0
    if context is not None and hasattr(context.scene, 'gpu_cloth_springs_built'):
        context.scene.gpu_cloth_springs_built = False


def release_prepare_owners(context=None):
    """Retire this prepare's owners; the session's runtime stays alive.

    The v3 runtime is a SCENE owner, not a per-prepare one.  It carries the CUDA
    context, the OptiX device context and the feature detection, which cost
    80 ms (warm) / 114 ms (cold) to build and are created once when the DLL
    loads.  What a re-prepare actually needs is for the owners built on top of it
    to be retired so the rebuild starts from a clean state, and that is exactly
    what `GPUCloth_v3_cloth_destroy` does: it aborts the runtime's collection
    transactions, destroys the proxies belonging to the cloth, destroys the cloth
    owner and erases it from the runtime's map, so a later create with the same
    `object_id` is admitted.

    The frame generation is deliberately left alone: it grows monotonically
    across the reuse, and `_runtime_update` refuses a generation that does not
    increase, so a reused runtime keeps advancing rather than restarting.

    A refused destroy returns False and leaves `_teardown_failure` clear; the
    caller (`prepare_simulation`) treats that as "cannot start from a clean
    state" and cancels, rather than building on a partially retired owner set.
    """
    global _teardown_failure

    if _prepare_native_worker_active():
        print("release_prepare_owners: native preparation worker is active")
        return False

    if g_dll is None:
        if _runtime_owners_retained():
            print(
                "release_prepare_owners: native DLL unavailable while runtime "
                "or owners are retained")
            _teardown_failure = True
            return False
        _reset_owner_python_state(context)
        return True

    if not _destroy_session_cloth_owners():
        print(
            "release_prepare_owners: cloth owner teardown refused; the "
            "preparation cannot start from a clean owner set")
        return False

    _reset_owner_python_state(context)
    _teardown_failure = False
    return True


def free_gpu_memory(context=None, shutdown_runtime=False):
    """Release owners only after native solver teardown is confirmed."""
    global g_dll
    global _teardown_failure

    if _prepare_native_worker_active():
        print("free_gpu_memory: native preparation worker is active")
        return False

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

    _reset_owner_python_state(context)
    # The runtime is gone, so the baselines that describe it restart.  These two
    # used to be cleared unconditionally; they now belong to the path that
    # actually destroys the runtime, which is what lets owner-only retirement
    # reuse the runtime without restarting the frame generation.
    _cache_source_state['generation'] = 0
    _input_generation['value'] = 0
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
        global _stop_requested
        was_preparing = prepare_task_active()
        cancel_auto_prepare()
        _stop_requested = True
        # The drive is asked down before any branch below can return, because
        # Stop is the one place that promises the simulation stops and the
        # promise cannot depend on which branch ran.  The preparation branch used
        # to return before this ask, and that is a measured defect-3 path: the
        # owner presses pause, the add-on answers "preparation cancellation
        # requested", and playback - which nobody cancelled - keeps stepping the
        # frame path until the prepare lands and re-arms it.
        #
        # One ask is what this operator can make synchronously, and an ask is not
        # a state change: the cancel's answer arrives only after the steps already
        # dispatched have run (1.054 s measured after a scrub, 56 ms from a plain
        # timer).  What makes the ask terminal is below - the dead-man switch is
        # not cleared while a drive is still owed.
        _stop_animation_playback()
        if was_preparing and prepare_task_active():
            self.report({'INFO'}, "GPUCloth preparation cancellation requested")
            return {'FINISHED'}
        # A live drape sandbox owns the Begin snapshot that can undo its preview
        # pose, and the native owner integrates `clmd->clothObject->verts` in
        # place, so while the sandbox is open the pose every frame shows *is*
        # the sandbox's.  Freeing the owners without closing it first leaves
        # that pose as the only state there is: the next Prepare captures it as
        # the rest shape, and the frame path has no owner left to step.  Closing
        # it here is the same Cancel the panel's own button performs.
        try:
            abort_live_drape_sandbox(context)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"[GPUCloth] drape sandbox was not closed before Stop: "
                  f"{type(exc).__name__}: {exc}")
        if not free_gpu_memory(context, shutdown_runtime=True):
            self.report({'ERROR'}, "Не удалось освободить GPU память.")
            return {'CANCELLED'}
        # Stop has now released every owner the dead-man switch exists to
        # protect, and each guard that walks those owners tests `g_dll` /
        # `g_clothOBJs` directly (`_frame_change_handler`,
        # `GPUCloth_UpdateSimulation.execute`, `_live_cloth_objects`,
        # `_realtime_state_reason`).  Leaving the switch on past a *successful*
        # release is what made "cannot start the simulation even after Stop"
        # permanent: the frame handler returned without a word, so the cloth
        # stayed on the pose Stop had just failed to clear.  A failed release
        # still leaves it set, and so does a cancelled preparation.
        #
        # It is therefore cleared here only when there is nothing left that
        # could step the simulation by itself: a drive that has not answered yet
        # leaves steps in flight (each one a frame change the frame path would
        # solve the moment an owner exists again), and a preparation in flight
        # will clear this switch itself when it lands - see `_finish_prepare_task`
        # for why it no longer does.  Either way the owner's next start clears
        # it: Prepare with `_prepare_steps`, an edit with `schedule_auto_prepare`.
        if _animation_is_playing(context) or prepare_task_active():
            self.report(
                {'INFO'},
                "GPUCloth: the drive is still stopping; prepare again to start")
            return {'FINISHED'}
        _stop_requested = False
        self.report({'INFO'}, "GPU память освобождена.")
        return {'FINISHED'}


# ===========================================================================
#  Оператор: загрузка DLL
# ===========================================================================

def _is_windows_host():
    """True on the Windows host; False on every POSIX host."""
    return sys.platform == "win32"


def _native_library_name(dll_name="GPUCloth.dll"):
    """Native library filename on this host: .dll / .dylib / .so."""
    if _is_windows_host():
        return dll_name
    if sys.platform == "darwin":
        return dll_name.replace(".dll", ".dylib")
    return dll_name.replace(".dll", ".so")


def _load_gpucloth_dll_native(filename):
    """Load and validate the DLL without touching Blender's Python API."""
    # The nvidia-smi probe is the legacy Windows/CUDA prerequisite.  It is not a
    # Vulkan-or-POSIX readiness test, so it must not gate a POSIX load: there the
    # native load, ABI validation and runtime creation are authoritative.
    if _is_windows_host():
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("nvidia-smi CUDA check failed") from exc
        if ("CUDA Version" not in result.stdout and
                "CUDA UMD Version" not in result.stdout):
            raise RuntimeError("NVIDIA driver does not report CUDA support")

    dll_dir = os.path.dirname(filename)
    directory_handles = []
    try:
        # Windows dependency search must be explicit.  On POSIX the loader
        # resolves packaged dependencies through dyld/ld.so, so process search
        # paths are left alone.
        if _is_windows_host():
            if dll_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = (
                    dll_dir + os.pathsep + os.environ.get("PATH", ""))
        if _is_windows_host() and hasattr(os, "add_dll_directory"):
            candidates = [dll_dir]
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                candidates.extend((
                    os.path.join(cuda_path, "bin", "x64"),
                    os.path.join(cuda_path, "bin"),
                ))
            for directory in dict.fromkeys(
                    os.path.abspath(path) for path in candidates):
                if os.path.isdir(directory):
                    directory_handles.append(os.add_dll_directory(directory))
        dll = cdll.LoadLibrary(filename)
        _bind_gpucloth_v3_exports(dll)
        _validate_product_abi(dll)
        _validate_descriptor_layout(dll)
    except BaseException:
        for handle in reversed(directory_handles):
            try:
                handle.close()
            except OSError:
                pass
        raise
    return dll, tuple(directory_handles)


# ---------------------------------------------------------------------------
# One-shot native pre-warm.
#
# Owner of what this pre-pays: g_runtime_handle (created exactly as the prepare
# path creates it: _runtime_create -> SIM_initialize_runtime -> CUDAFeatures and
# OptixFeatures init), plus g_dll and _dll_directory_handles.  Release is the
# existing one and is unchanged: free_gpu_memory -> _destroy_runtime from the
# prepare entry, gpucloth.destroy_simulation_data, gpucloth.unload_dll, and
# ensure_native_teardown(shutdown_runtime=True) on unregister and on Blender's
# exit hook (bpy.utils._on_exit, Cpp_Compatibility/__init__.py).
# ---------------------------------------------------------------------------

_warmup_state = "idle"          # idle -> done | failed; never re-entered


def _warmup_native_runtime():
    """Pay the one-time DLL load and CUDA/OptiX initialisation before Prepare.

    This only pays cost earlier: it validates nothing, mutates no Blender data
    and cannot change a verdict.  It is not called on a worker thread, so the
    process-wide CUDA primary context it creates is usable by every later
    call -- a context created on the main thread is what the prepare worker
    already uses today.
    """
    global g_dll
    if g_dll is not None and _runtime_handle_value():
        return True
    if g_dll is None:
        filename = vcu.get_dll_path("GPUCloth.dll")
        if filename is None:
            raise RuntimeError("GPUCloth.dll was not found for pre-warm")
        loaded_dll, directory_handles = _load_gpucloth_dll_native(filename)
        # Published only after the load and ABI validation succeeded, so a
        # failed pre-warm leaves g_dll exactly as Prepare expects to find it.
        g_dll = loaded_dll
        _dll_directory_handles.extend(directory_handles)
    _runtime_create()
    return True


def _warmup_native_handler(*_args):
    """Pre-warm once, on the first scene that owns an active cloth object.

    Registered on both load_post (a file is opened) and depsgraph_update_post
    (the add-on is enabled while a file is already open); both run on the main
    thread, so no thread and no UI change is involved.  A scene without an
    active GPUCloth object is left untouched, so opening an unrelated file pays
    nothing.

    A failed pre-warm is recorded and printed, never raised: Prepare then runs
    its own load and validation path exactly as it does without this hook.
    """
    global _warmup_state
    if _warmup_state != "idle" or prepare_task_active():
        return
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return
    for obj in getattr(scene, "objects", ()):
        settings = getattr(obj, "GPUCloth", None)
        if settings is not None and bool(settings.is_active):
            break
    else:
        return              # nothing to warm for this file; stay idle
    _warmup_state = "done"
    try:
        _warmup_native_runtime()
    except BaseException as exc:  # noqa: BLE001 - a warm-up must never raise
        _warmup_state = "failed"
        print(f"[GPUCloth] native pre-warm skipped: {exc!r}")


def _build_v3_cloth_native(
        dll, cloth_handle, prepared_constraint_network, prepared_shrink_bounds,
        object_id, topology_generation, geometry_generation):
    """Build one native cloth and validate its accepted typed inputs."""
    result = int(dll.GPUCloth_v3_cloth_build(cloth_handle))
    if result != CType.GPUCLOTH_ABI_OK:
        raise RuntimeError(f"v3 cloth build rejected with {result}")
    _validate_constraint_network_status(
        dll, cloth_handle, prepared_constraint_network)
    _validate_shrink_status(
        dll, cloth_handle, prepared_shrink_bounds,
        object_id, topology_generation, geometry_generation)
    return result

class GPUCloth_LoadDLL(bpy.types.Operator):
    """Загрузить нативную библиотеку GPUCloth (DLL / .dylib / .so)"""
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

    def _platform_gate(self):
        """Run the platform-only preflight; load/ABI/runtime prove readiness."""
        # The CUDA/nvidia-smi prerequisite belongs to the legacy Windows backend.
        # POSIX readiness is established by native load, ABI validation, then
        # runtime creation, so this is not a Vulkan device probe.
        if _is_windows_host():
            return self.check_cuda_support()
        return True

    # ── Загрузка библиотеки и привязка функций ───────────────────────────────

    def load_dll(self):
        global g_dll, _dll_directory_handles
        if g_dll is not None:
            return True  # уже загружена

        if not self._platform_gate():
            return False

        # Ищем native library через vcu (относительно директории аддона)
        filename = vcu.get_dll_path("GPUCloth.dll")
        expected_name = _native_library_name()
        if filename is None:
            lib_dir   = vcu.get_lib_directory()
            addon_dir = vcu.get_addon_directory()
            self.report({'ERROR'},
                f"{expected_name} не найдена. "
                f"Искали в: {lib_dir} , {addon_dir} , {addon_dir}\\build\\ . "
                f"Скопируйте {expected_name} в {lib_dir}")
            return False

        # Windows dependency search is explicit; POSIX leaves dyld/ld.so alone.
        if _is_windows_host():
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
                for directory in dict.fromkeys(
                        os.path.abspath(path) for path in candidates):
                    if os.path.isdir(directory):
                        _dll_directory_handles.append(
                            os.add_dll_directory(directory))

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
        return g_dll is not None and not prepare_task_active()

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
            if not _is_windows_host():
                # Explicit dlclose can crash when ctypes/native static state
                # outlives Python.  Keep one POSIX handle process-resident;
                # load_dll reuses it on later registration/load operations.
                self.report(
                    {'INFO'},
                    "Нативная библиотека оставлена загруженной после teardown.")
                return {'FINISHED'}

            # Windows: FreeLibrary через kernel32
            handle = c_void_p(g_dll._handle)
            result = ctypes.WinDLL("kernel32").FreeLibrary(handle)
            if result == 0:
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
        return not prepare_task_active()

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def execute(self, context):
        if not _start_prepare_task(context, automatic=False):
            self.report({'WARNING'}, "GPUCloth preparation is already active")
            return {'CANCELLED'}
        self.report({'INFO'}, "GPUCloth preparation started")
        return {'FINISHED'}

    def _prepare_steps(self, context):
        global g_dll, g_cloth_handles
        global _dll_directory_handles
        global _cloth_input_owners, _readback_owners
        global g_clothOBJs, g_simulationOBJs
        global g_clothCollisionOBJs, g_proxy_handles
        global _stop_requested

        self._native_prepare_mutated = False
        self._prepare_deferred = None

        if _teardown_failure:
            if not release_prepare_owners(context):
                self.report(
                    {'ERROR'},
                    "Native teardown retry failed; retained owners block "
                    "prepare")
                return {'CANCELLED'}

        yield _prepare_progress(2, "Loading GPUCloth runtime")

        # 1. Загружаем DLL если нужно
        if g_dll is None:
            filename = vcu.get_dll_path("GPUCloth.dll")
            if filename is None:
                self.report({'ERROR'}, "GPUCloth.dll was not found")
                return {'CANCELLED'}
            try:
                loaded_dll, directory_handles = yield _prepare_native(
                    _load_gpucloth_dll_native, (filename,), 5,
                    "Loading and validating GPUCloth.dll")
            except (AttributeError, OSError, RuntimeError) as exc:
                self.report({'ERROR'}, f"GPUCloth DLL load failed: {exc}")
                return {'CANCELLED'}
            g_dll = loaded_dll
            _dll_directory_handles.extend(directory_handles)

        yield _prepare_progress(10, "Saving Blender inputs")

        # 2. Shall GPUCloth save the document?  No - and it never needed to.
        #
        # This step used to call `wm.save_as_mainfile` twice: on an unsaved
        # document it wrote `bpy.app.tempdir + 'GPU_Cloth.blend'`, and on a dirty
        # one it wrote `bpy.data.filepath` - the user's own file, unasked.  The
        # stated reason was "the DLL needs a path to the blend for some
        # operations", but that reason is not supported by anything in this
        # repository: the engine never mentions a blend path (`grep` over
        # `src/engine` finds no `blend_path`, `blend_file` or `GPU_Cloth.blend`),
        # the add-on never reads `bpy.data.filepath` except as the argument to
        # the second save, and no path is ever handed to native.
        #
        # The side effects were real and were measured on Blender 5.2.1 with an
        # unsaved document: pressing Prepare gave the document a filename and a
        # location the user never chose, inside a temp directory Blender deletes,
        # and cleared `dirty`, so the unsaved-work indicator disappeared and a
        # later Ctrl+S would have written into that temp directory silently.
        #
        # Deleting the save is the fix.  Prepare needs the document's *contents*,
        # which it reads from the mesh and the depsgraph; it needs no path, and if
        # it ever did, the caller that wants one should pass it explicitly rather
        # than have Prepare silently re-identify the user's document.

        # 3. Переходим в Object mode для корректного считывания данных.
        #
        # Blender refuses this while a modal operator is running ("Unable to
        # change object mode while transforming"), and a transform is exactly
        # when a deferred prepare is most likely to fire: the timer queue runs
        # between modal events.  Nothing native has been mutated at this point,
        # so a refusal is deferred rather than failed - the request is re-armed
        # and the next attempt runs once the transform has finished.  Failing
        # here instead turned a legitimate rebuild into an error the user could
        # do nothing about, and it is the second line of the reported defect.
        #
        # Only the mode switch itself is covered.  A prepare with no active
        # object is a different failure and must still be reported as one.
        mode = bpy.context.active_object.mode
        if mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError as exc:
                return _defer_prepare_for_transform(self, exc)

        # 4. Preflight every Blender-owned input before native mutation.
        yield _prepare_progress(15, "Reading Blender cloth inputs")
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
            if not release_prepare_owners(context):
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
        solver_types = [
            str(item.GPUCloth.solver_type) for item in active_cloth]
        try:
            memory = _require_gpu_memory_preflight(
                simulation_objects, solver_types)
        except RuntimeError as exc:
            helper.memory_preflight_status = str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        helper.memory_preflight_status = (
            f"PASS: lower bound {memory['lower_bound_bytes']} bytes; "
            f"free {memory['free_bytes']} bytes")
        _stop_requested = False

        yield _prepare_progress(25, "Capturing evaluated cloth geometry")

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
                # FABRIC is the only material model: the triangle membrane needs
                # a triangulated input mesh and a per-corner coordinate stream.
                # That is now unconditional - the one bending model that
                # competed with it for the payload (SDB) was removed from the
                # addon.
                prepared_input_meshes.append(
                    _capture_modifier_input_mesh(
                        simulation_obj, depsgraph, triangulate=True))
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
                    topology_generation,
                    v3_corner=True)
                material_features = _capture_material_features(
                    cloth_obj.GPUCloth, material_coordinates)
                prepared_material_features.append(material_features)
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

        yield _prepare_progress(40, "Creating GPUCloth runtime owners")

        # 5. Native mutation starts only after complete preflight.
        self._native_prepare_mutated = True
        if (_runtime_owners_retained() or
                context.scene.gpu_cloth_springs_built):
            if not release_prepare_owners(context):
                self.report({'ERROR'}, "Не удалось освободить GPU память")
                return {'CANCELLED'}

        for cloth_obj in active_cloth:
            _take_cloth_object(cloth_obj)
        g_simulationOBJs.extend(simulation_objects)
        try:
            from . import cloth_settings_bridge
            for cloth_obj in _live_cloth_objects():
                cloth_settings_bridge.apply_modifier_ownership(
                    cloth_obj, "GPU")
        except (
                AttributeError, ReferenceError, RuntimeError,
                TypeError) as exc:
            self.report(
                {'ERROR'}, f"CPU Cloth ownership failed: {exc}")
            release_prepare_owners(context)
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
            release_prepare_owners(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        yield _prepare_progress(48, "Creating native cloth objects")

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
            release_prepare_owners(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        if not (len(g_clothOBJs) == len(g_simulationOBJs) ==
                len(g_cloth_handles) == len(_cloth_input_owners) ==
                len(_readback_owners)):
            self.report({'ERROR'}, "v3 cloth owners are not aligned")
            release_prepare_owners(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        yield _prepare_progress(55, "Configuring native cloth solvers")

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
                release_prepare_owners(context)
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
                release_prepare_owners(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}
            cloth_progress = 60 + int(
                (20 * i) / max(1, len(g_clothOBJs)))
            try:
                result = yield _prepare_native(
                    _build_v3_cloth_native,
                    (
                        g_dll, cloth_handle,
                        prepared_constraint_networks[i],
                        prepared_shrink_bounds[i],
                        _cloth_input_owners[i]["object_id"],
                        _cloth_input_owners[i]["topology_generation"],
                        _cloth_input_owners[i]["geometry_generation"],
                    ),
                    cloth_progress,
                    f"Building cloth {i + 1}/{len(g_clothOBJs)} on GPU",
                )
                if result != CType.GPUCLOTH_ABI_OK:
                    raise RuntimeError(f"v3 cloth build returned {result}")
            except (OSError, RuntimeError) as exc:
                self.report({'ERROR'}, f"v3 cloth build failed: {exc}")
                release_prepare_owners(context)
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
                release_prepare_owners(context)
                bpy.ops.object.mode_set(mode=mode)
                return {'CANCELLED'}

            yield _prepare_progress(
                60 + int((20 * (i + 1)) / max(1, len(g_clothOBJs))),
                f"Cloth {i + 1}/{len(g_clothOBJs)} built")

        yield _prepare_progress(84, "Publishing initial GPU state")
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
                # The host self-intersection preflight inside this call is
                # 470 ms of CPU-only work at 16384 vertices.  It owns no Blender
                # data, so it belongs on the worker thread the prepare task
                # already owns: the wait is unchanged, the event loop is not.
                yield _prepare_native(
                    _validate_native_preparation,
                    (
                        g_dll, cloth_handle,
                        prepared_topology_generations[index],
                        initial_generation,
                    ),
                    90,
                    "Validating initial cloth state",
                )
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
            release_prepare_owners(context)
            bpy.ops.object.mode_set(mode=mode)
            return {'CANCELLED'}

        yield _prepare_progress(94, "Finalizing Blender ownership")

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
        # The owners now hold exactly these inputs.  Baseline the generation and
        # the property epoch here so _refresh_prepared_inputs compares the next
        # edit against what was actually built, and cannot re-prepare in a loop.
        _cache_source_state['generation'], _cache_source_state['staged'] = (
            _cache_input_digests_for_frame(context.scene))
        _cache_source_state['epoch'] = _cache_fingerprint_epoch()
        _simulation_frame_state['rebuild_pending'] = False
        # A rebuilt owner holds freshly captured inputs, not the live ones, so
        # the first live call after this republishes every group.
        _live_group_state.clear()
        return {'FINISHED'}


# ===========================================================================
#  Оператор: обновление симуляции (один кадр)
# ===========================================================================

# ===========================================================================
#  Бесконечная симуляция (MD-style): драйвер шагов, чекпоинт старта, Ctrl+Z
# ===========================================================================
#
#  Требование владельца: «режим бесконечной симуляции (активируется как в MD
#  по нажатию пробела с включённым "Move Cloth by vertex") ткани без анимаций
#  в сцене с сохранением чекпоинтов перед стартом симуляции с работами через
#  ctrl + z».
#
#  Форма, которую это принимает:
#
#    * Space потребляет уже вооружённый модальный обработчик инструмента
#      (``GPUCloth_MoveClothByVertex.modal``, ветка до гейта).  Глобальной
#      привязки клавиши нет: она затеняла бы Space у всех пользователей
#      аддона в любом файле, а вооружённый инструмент — видимое и обратимое
#      состояние, которое панель показывает.
#    * Драйвер — таймер ``bpy.app.timers``, а не модальный оператор: новый
#      bl_idname увеличил бы ``TrustedAddonInventoryCount`` (22) и сломал бы
#      trust anchor гейта, а новый класс не нужен — шаг делает
#      ``_live_step_cloth_scene``.
#    * Счётчик кадров Blender заморожен: ``frame_set`` не вызывается.
#      Собственные часы режима — ``_infinite_sim_state['steps']``.
#    * Чекпоинт C0 = «перед стартом симуляции»: rest-поза, которую уже снял
#      prepare, плюс номера генераций, к которым она относится.  Восстановление
#      — существующий путь пересборки, поэтому оно не может испортить нативного
#      владельца (Design C, k = 0).
#    * Ctrl+Z наблюдается, но не вызывается: одна граница ``undo_push`` на
#      входе в режим и обработчики ``undo_post``/``redo_post``, которые
#      останавливают драйвер и требуют пересборки (fail closed).
#      ``bpy.ops.ed.undo()`` не вызывается нигде.
#
#  Состояние живёт в модуле рядом с состоянием инструмента
#  (``_vertex_drag_state``) — установленный в файле образец.  Всё, что панель
#  показывает, читается через ``infinite_sim_state()``; поля
#  ``bake_progress``/``prepare_progress`` этот режим не пишет (bake_progress
#  принадлежит запеканию, и UI ключует на нём показ кнопки Bake).

_infinite_sim_state = {
    'running': False,       # драйвер владеет циклом шагов
    'scene_name': None,     # сцена, чей кадр заморожен; разрешается по имени
    'steps': 0,             # собственные часы режима, не frame_current
    'step_ms': 0.0,         # длительность последнего шага, для панели
    'frame': None,          # замороженный frame_current, снятый на входе
    'checkpoint': None,     # C0: поза на момент старта
    'status': "",           # одна строка для панели
    # Consecutive refused steps with no accepted one between them.  A refusal is
    # not progress, so this is the clock that bounds the retry below; an accepted
    # step zeroes it.
    'refusals': 0,
}
_infinite_timer_registered = False
# One undo boundary per session, and never cleared while the add-on is loaded:
# see _infinite_start.
_infinite_state = {'undo_boundary_pushed': False}

# One step per timer tick.  A tick is "as soon as the previous step returned":
# the requested interval is 1 ms and a step costs tens of milliseconds, so the
# solver, not this number, sets the rate.  0.0 would ask Blender to spin with no
# delay at all, which buys nothing and makes a runaway loop harder to see.
_INFINITE_SIM_INTERVAL = 0.001
# How many refusals in a row the drive retries through before it stops.
#
# A refused solve is not a dead session, and the engine says so itself: the
# refused call commits its rejected candidate to the host array, clears the
# runtime latch and arms a gravity-suppressed retry for the *next* call
# (main.cpp:13460-13527), so the attempt after a refusal is the one that path
# exists to make succeed.  The frame path never had to know this, because the
# next frame change asks again; this driver's next ask is its own, and latching
# on the first refusal is what made one refused solve cost the user a fresh
# press.  The budget cannot be borrowed from the native side either: each
# refusal changes the pose the next attempt starts from, which is what resets
# ``kRecoveryStallAttempts`` (main.cpp:879), so the streak is not bounded by
# four.  Measured on the Drape scene with the collider moved into the cloth: one
# retry was enough for a direct call (twice, identical - the first-solve probe),
# and the streaks the driver met before a step landed were three.  Eight is more
# than twice the worst streak observed, and it still terminates: a session that
# cannot step stops here, with the count and the native cause, instead of
# spinning for ever.
_INFINITE_REFUSAL_RETRY_LIMIT = 8
# The one sentence the drive shows while it is taking the timeline from Blender's
# playback: the start asks the cancel, and the steps already dispatched keep
# arriving until it lands (measured on the owner's route: 146.2 ms for the first
# step, the frame advancing during it).  It is a constant because the step path
# has to recognise it - the mode is not "taking" anything once it is stepping.
_INFINITE_TAKEOVER_STATUS = (
    "Taking the timeline from the animation playback; the drive steps once it "
    "is down")


def _infinite_step_timer_register():
    """Register the one window-manager timer the driver owns.

    ``bpy.app.timers`` rather than ``wm.event_timer_add`` on purpose: the
    driver must not need a window handle, and ``_advance_prepare_task`` (the
    repo's other long-running poller, :577) uses the same surface, so there is
    one timer discipline in this module instead of two.
    """
    global _infinite_timer_registered
    if _infinite_timer_registered:
        return False
    try:
        bpy.app.timers.register(
            _infinite_advance, first_interval=_INFINITE_SIM_INTERVAL)
    except (AttributeError, RuntimeError):
        _infinite_timer_registered = False
        return False
    _infinite_timer_registered = True
    return True


def _infinite_step_timer_unregister():
    """Remove the timer.  Idempotent, and safe to call after unregister()."""
    global _infinite_timer_registered
    if not _infinite_timer_registered:
        return
    try:
        if bpy.app.timers.is_registered(_infinite_advance):
            bpy.app.timers.unregister(_infinite_advance)
    except (AttributeError, RuntimeError):
        pass
    _infinite_timer_registered = False


def _infinite_is_running():
    """True while the driver owns a step loop.  Read by the drag gate.

    The timer registration, not the flag, is the authority: this predicate
    decides whether the drag gate opens and whether the panel says the mode is
    running, and a flag that disagrees with the timer would claim a drive that
    has already ended.  ``is_registered`` is one registry lookup, and it is the
    same call ``_infinite_step_timer_unregister`` already makes.
    """
    if not _infinite_sim_state['running']:
        return False
    try:
        return bool(bpy.app.timers.is_registered(_infinite_advance))
    except (AttributeError, RuntimeError):
        return False


def _infinite_stop(reason):
    """Stop the drive.  The checkpoint, if any, survives (Space restarts).

    Idempotent.  Releases a live grab, because the Mode's premise was that the
    driver steps, and once it does not the grab gate is false again.  The
    reason is printed as well as stored: a drive that stops is the one event a
    user cannot see the cause of from the panel alone, and the add-on's own
    stdout is where every other diagnostic of this kind already goes.
    """
    global _infinite_timer_registered
    state = _infinite_sim_state
    was_running = bool(state['running'])
    state['running'] = False
    state['status'] = _infinite_text(reason)
    _infinite_step_timer_unregister()
    _infinite_timer_registered = False
    if _vertex_drag_state['dragging']:
        _vertex_drag_release("grab released: infinite simulation stopped")
    if was_running:
        print(
            f"[GPUCloth] infinite simulation stopped after "
            f"{state['steps']} step(s): {state['status']}", flush=True)
    return was_running


def _infinite_timeline_owners(scene):
    """Watched objects the timeline owns, and why.

    Returns ``(blocking, warning)`` lists of ``{"object", "source", "detail"}``.
    ``blocking`` is non-empty exactly when the mode must refuse: some watched
    object's inputs are evaluated *from the frame*, so the pose being simulated
    is the timeline's, not the scene's.  That is the owner's own precondition
    ("ткани без анимаций в сцене") and it is also the honest boundary - if the
    timeline owns an input, the frame-indexed cache is the right tool and a
    mode built on a frozen timeline is the wrong one.

    ``warning`` is currently always empty: every source below *does* touch
    something the simulation reads, so there is nothing to downgrade to a
    warning.  The list exists as the seam for a source that provably does not
    (for example an action whose channels are all outside the fingerprint's
    watched set), and so the refusal report has one shape.

    Only ``action`` / ``nla_track`` / ``driver`` on a *watched* ID are looked
    for - the same watched set ``_cache_input_watch_ids`` (:6635) already
    defines, so "an animation on an unrelated object" cannot refuse the mode.
    """
    blocking = []
    warning = []
    seen = set()

    def _record(obj, owner_label, source, detail):
        key = (owner_label, source, detail)
        if key in seen:
            return
        seen.add(key)
        blocking.append({
            "object": owner_label,
            "source": source,
            "detail": detail,
        })

    for obj in tuple(getattr(scene, "objects", ())):
        watched = []
        try:
            settings = getattr(obj, "GPUCloth", None)
            is_cloth = bool(getattr(settings, "is_active", False))
            is_collider = any(
                getattr(modifier, "type", None) == 'COLLISION'
                for modifier in tuple(getattr(obj, "modifiers", ())))
        except (AttributeError, ReferenceError, RuntimeError):
            continue
        if is_cloth or is_collider:
            watched.append((obj, obj.name_full))
        mesh = getattr(obj, "data", None)
        if mesh is not None and (is_cloth or is_collider):
            watched.append((mesh, f"{obj.name_full} (mesh)"))
        for owner, label in watched:
            try:
                animation = getattr(owner, "animation_data", None)
            except (AttributeError, ReferenceError, RuntimeError):
                continue
            if animation is None:
                continue
            action = getattr(animation, "action", None)
            if action is not None:
                _record(
                    owner, label, "action",
                    f"{getattr(action, 'name', '?')!r} on {label}")
            tracks = tuple(getattr(animation, "nla_tracks", ()) or ())
            if tracks:
                names = ", ".join(
                    repr(getattr(track, "name", "?")) for track in tracks[:3])
                _record(
                    owner, label, "nla_track",
                    f"{len(tracks)} NLA track(s) [{names}] on {label}")
            drivers = tuple(getattr(animation, "drivers", ()) or ())
            for driver in drivers:
                if not bool(getattr(driver, "mute", False)):
                    _record(
                        owner, label, "driver",
                        f"driver on {getattr(driver, 'data_path', '?')} "
                        f"of {label}")
    return blocking, warning


def _infinite_capable(scene):
    """Every precondition of the mode, in one place, with the reason.

    Returns ``{"ok": True}`` or ``{"ok": False, "reason": "<code>",
    "message": "<user text>"}``.  Nothing here mutates.
    """
    if g_dll is None:
        return {
            "ok": False, "reason": "no_dll",
            "message": _t_infinite(
                "Load the native module first",
                "Сначала загрузите нативный модуль"),
        }
    if scene is None:
        return {
            "ok": False, "reason": "no_scene",
            "message": _t_infinite("No scene", "Нет сцены"),
        }
    if not bool(getattr(scene, "gpu_cloth_springs_built", False)):
        return {
            "ok": False, "reason": "not_prepared",
            "message": _t_infinite(
                "Prepare the simulation first",
                "Сначала выполните Prepare"),
        }
    if prepare_task_active():
        return {
            "ok": False, "reason": "prepare_active",
            "message": _t_infinite(
                "Preparation is running",
                "Идёт подготовка"),
        }
    if _teardown_failure or _stop_requested:
        return {
            "ok": False, "reason": "stopped",
            "message": _t_infinite(
                "Simulation data is stopped; prepare again",
                "Данные симуляции остановлены; выполните Prepare"),
        }
    # The scene-level precondition the owner asked for comes before the
    # owner/handle checks and before the frame floor.  It is the answer to
    # "may this mode run in this scene at all", independent of whether the
    # simulation is prepared or which frame the timeline is parked on, and
    # reporting `not_prepared` for a scene that is timeline-driven would send
    # the user to Prepare for a problem Prepare cannot fix.
    blocking, _warning = _infinite_timeline_owners(scene)
    if blocking:
        first = blocking[0]
        return {
            "ok": False, "reason": "timeline_owned",
            "owners": blocking,
            "message": _t_infinite(
                f"The timeline drives {first['detail']}; infinite mode "
                f"simulates a scene without animation",
                f"Таймлайн управляет {first['detail']}; бесконечный режим "
                f"работает со сцены без анимации"),
        }
    if int(scene.frame_current) < 2:
        # The shared step core has no such rule; the *operator* entry point
        # refuses frames below 2 (:8331), and a live run that starts at frame 1
        # would then be unable to hand its result back to the timeline path.
        return {
            "ok": False, "reason": "frame_below_two",
            "message": _t_infinite(
                "The current frame must be 2 or later",
                "Текущий кадр должен быть 2 или больше"),
        }
    helper = getattr(scene, "gpu_cloth_helper", None)
    if helper is not None:
        if bool(getattr(helper, "is_baked", False)):
            return {
                "ok": False, "reason": "baked",
                "message": _t_infinite(
                    "The timeline is baked; free the cache first",
                    "Таймлайн запечён; сначала очистите кэш"),
            }
        if bool(getattr(helper, "is_baking", False)):
            return {
                "ok": False, "reason": "baking",
                "message": _t_infinite(
                    "A bake is running", "Идёт запекание"),
            }
        if bool(getattr(helper, "playback_mode", False)):
            return {
                "ok": False, "reason": "playback",
                "message": _t_infinite(
                    "Cache playback is on; turn it off first",
                    "Включено воспроизведение кэша; выключите его"),
            }
        if bool(getattr(helper, "use_external_cache", False)):
            return {
                "ok": False, "reason": "external_cache",
                "message": _t_infinite(
                    "External cache is on; the mode writes no cache",
                    "Включён внешний кэш; режим не пишет кэш"),
            }
    if not _vertex_drag_simulation_ready(scene):
        return {
            "ok": False, "reason": "not_ready",
            "message": _t_infinite(
                "A live simulation owner is required",
                "Нужен живой владелец симуляции"),
        }
    return {"ok": True}


def _infinite_report_target(context):
    """Where the mode's refusal/stop text goes when there is no operator.

    The panel already renders ``infinite_sim_state()['status']`` and the scene
    helper already has a ``prepare_status`` line the preparation sub-panel
    shows; mirroring into it keeps a stop reason visible after the modal that
    reported it has moved on.  It is a *status* field, not a progress field:
    ``bake_progress`` and ``prepare_progress`` are never written here.
    """
    helper = getattr(getattr(context, "scene", None), "gpu_cloth_helper", None)
    if helper is not None:
        try:
            helper.prepare_status = _infinite_text(
                _infinite_sim_state['status'])
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            pass


def _t_infinite(english, russian):
    """The panel idiom: English and Russian label variants."""
    return english, russian


def _infinite_text(value):
    """One string out of a message that may be an ``(en, ru)`` pair.

    ``_t_infinite`` returns a pair because the panel picks a language per
    label; a report or a status line needs one string, and it takes English -
    the same choice ``ui.py`` makes when the add-on runs on a non-Russian
    locale.
    """
    if isinstance(value, (tuple, list)):
        return str(value[0]) if value else ""
    return str(value)


def infinite_sim_state(context=None):
    """Read-only mode state for the panel.  One meaning per field."""
    context = bpy.context if context is None else context
    state = _infinite_sim_state
    checkpoint = state['checkpoint']
    return {
        "running": _infinite_is_running(),
        "steps": int(state['steps']),
        "step_ms": float(state['step_ms']),
        "frame": int(state['frame']) if state['frame'] is not None else -1,
        "status": state['status'] or "",
        "has_checkpoint": checkpoint is not None,
        "checkpoint_steps": (
            int(checkpoint["steps"]) if checkpoint is not None else 0),
        "capable": _infinite_capable(
            getattr(context, "scene", None)),
        "space_hint": _infinite_space_hint(context),
    }


def _infinite_space_hint(context):
    """The single sentence the panel shows for Space."""
    state = _infinite_sim_state
    if _infinite_is_running():
        return _t_infinite(
            "Space: stop the infinite simulation",
            "Пробел: остановить бесконечную симуляцию")
    if state['checkpoint'] is not None:
        return _t_infinite(
            "Space: run again  |  Ctrl+Z: back to the start checkpoint",
            "Пробел: запустить снова  |  Ctrl+Z: вернуться к чекпоинту старта")
    return _t_infinite(
        "Space: start the infinite simulation",
        "Пробел: запустить бесконечную симуляцию")


def _infinite_checkpoint_capture(context):
    """Capture C0 = "before the simulation starts" (md_caching_gap.md §5.7).

    C0 owns the *bookkeeping* of the run: the step index it starts from, the
    frozen frame, and the generations the state belongs to.  The pose itself is
    ``_initial_positions`` - the rest pose the prepare captured and validated -
    and it is deliberately not copied here: Ctrl+Z restores that pose
    (``_infinite_restore_start``), because it is the only one a rebuild can be
    built from, and a copy of the live mesh would be a second, unvalidated idea
    of where the cloth is.  Two copies were carried for one reader and neither
    was read, so the checkpoint no longer carries a pose at all.

    Restore is the existing rebuild path, so at ``k = 0`` it is exact by
    construction: ``_restore_initial_positions`` + ``_clear_retained_frames`` +
    ``_invalidate_cache_for_change`` + ``schedule_auto_prepare(required=True)``
    is the same sequence an input change already takes
    (``_stage_changed_inputs``, :7011).
    """
    if not g_clothOBJs:
        return None
    state = _infinite_sim_state
    return {
        "steps": int(state['steps']),
        "frame": int(context.scene.frame_current),
        "cache_source_generation": int(_cache_source_state['generation']),
        "runtime_generation": int(_runtime_frame_generation),
    }


def _infinite_start(context):
    """Start the drive: preconditions, checkpoint C0, undo boundary, timer."""
    if _infinite_is_running():
        return {
            "ok": True, "already_running": True,
            "message": _t_infinite(
                "Infinite simulation is already running",
                "Бесконечная симуляция уже запущена"),
        }
    capable = _infinite_capable(getattr(context, "scene", None))
    if not capable["ok"]:
        return capable
    state = _infinite_sim_state
    # The scene is held by NAME, not by reference and not as a saved context:
    # the drive outlives the operator invocation that started it, and the repo
    # already treats a stored bpy reference across such a boundary as a hazard
    # (``_original_object``, :899; name-based re-lookup, :220).  ``_advance_prepare_task``
    # resolves its scene the same way (:631).
    state['scene_name'] = getattr(context.scene, "name_full", None)
    state['frame'] = int(context.scene.frame_current)
    checkpoint = _infinite_checkpoint_capture(context)
    if checkpoint is None:
        state['scene_name'] = None
        return {
            "ok": False, "reason": "no_rest_positions",
            "message": _t_infinite(
                "No captured rest pose; prepare again",
                "Нет сохранённой rest-позы; выполните Prepare"),
        }
    # The counter is zeroed only once the start can actually happen.  Zeroing
    # it before the checkpoint would erase "how many steps the previous run
    # took" from the panel when a start is refused, which is the one number a
    # user needs in order to understand the refusal.
    state['steps'] = 0
    state['refusals'] = 0
    state['checkpoint'] = checkpoint
    # The undo boundary of §5.10: one push per session, the first time the mode
    # starts.  undo_push only *records* a step - it is the one Blender undo call
    # the add-on makes, and it is never undo().  It is NOT repeated per start:
    # re-entering Blender's undo machinery on every restart adds a second undo
    # entry for the same boundary, which the user would have to press Ctrl+Z
    # through twice, and the boundary it records is the same state either way.
    if not _infinite_state['undo_boundary_pushed']:
        try:
            bpy.ops.ed.undo_push(
                message="GPUCloth: infinite simulation started")
        except (AttributeError, RuntimeError, TypeError) as exc:
            print(f"GPUCloth infinite simulation: undo_push refused: {exc}")
        _infinite_state['undo_boundary_pushed'] = True
    if not _infinite_step_timer_register():
        state['scene_name'] = None
        return {
            "ok": False, "reason": "timer",
            "message": _t_infinite(
                "Cannot start the step timer",
                "Не удалось запустить таймер шагов"),
        }
    # The mode steps the solver itself, so no other driver may be stepping the
    # timeline beside it, and there are exactly two that can be up when Space is
    # pressed.  Blender's own playback is one: a start that leaves it up is a
    # start the drive loses on its first tick - measured on this fixture
    # (playback at 24 fps, the first step 146.2 ms where the owner's was), the
    # frame advances *during* that step and the next tick stops the mode with
    # "timeline moved to frame N+1; the drive stopped (it started at frame N)",
    # which is the owner's console, six starts in a row.  The frame path's
    # re-simulation is the other: it produces frames itself, one per tick, and
    # moves the frame to do it under ``_cache_playback_guard`` - measured, a
    # drive started during one saw every one of those moves as a user's hand and
    # stopped at the next tick, and the guard is set only for the duration of the
    # ``frame_set``, so the driver's own check cannot tell them apart afterwards.
    # Both are the timeline's work, and the mode is taking the timeline; the asks
    # are the existing ones (``_stop_animation_playback``, ``_resimulate_finish``)
    # rather than a third way to stop either.
    _stop_animation_playback()
    if _resimulate_state['active']:
        _resimulate_finish(
            getattr(context, "scene", None), "the infinite drive took the timeline")
    state['running'] = True
    state['status'] = (
        f"running from frame {state['frame']}, checkpoint step 0")
    return {"ok": True, "message": _t_infinite(
        "Infinite simulation started",
        "Бесконечная симуляция запущена")}


def _infinite_step_reason(scene):
    """Why the drive must stop before the next step, or None.

    The order is the timeline operator's own guard order (:8325-8337): the
    cheap flags first, then the fingerprint.  ``_refresh_prepared_inputs`` is
    the one owner of "an input changed and needs a rebuilt owner" and it also
    clears ``rebuild_pending`` when it has queued that rebuild, so asking it
    before reading the flag is what keeps a *live-safe* edit from stopping the
    drive.  Asking the flag first would stop the mode on every settings tweak,
    which is not what "the timeline changed" means.
    """
    if prepare_task_active() or _teardown_failure or _stop_requested:
        return _t_infinite(
            "Simulation data is stopped; prepare is required",
            "Данные симуляции остановлены; нужен Prepare")
    if _refresh_prepared_inputs(scene) or (
            _simulation_frame_state['rebuild_pending']):
        return _t_infinite(
            "The simulation inputs changed; the drive stopped and a rebuild "
            "is queued",
            "Входы симуляции изменились; драйвер остановлен, пересборка в "
            "очереди")
    if not _runtime_handle_value():
        # Reached only if something released the native owner while the drive
        # was running.  The step core would refuse anyway; stopping here means
        # the refusal names the cause instead of reporting "runtime owner is
        # not live" from three frames deeper, and it stops the timer instead of
        # retrying a step that cannot succeed.
        return _t_infinite(
            "The native runtime owner is gone; prepare again",
            "Нативный runtime недоступен; выполните Prepare")
    helper = getattr(scene, "gpu_cloth_helper", None)
    if helper is not None:
        if bool(getattr(helper, "is_baked", False)) or bool(
                getattr(helper, "is_baking", False)):
            return _t_infinite(
                "A bake took the timeline",
                "Запекание забрало таймлайн")
        if bool(getattr(helper, "playback_mode", False)):
            return _t_infinite(
                "Cache playback was switched on",
                "Включено воспроизведение кэша")
    return None


def _infinite_refusal_is_retryable(step):
    """True when a retry is the thing this refusal was built to survive.

    Exactly one refusal is retryable and it is the solver's own:
    ``step_rejected`` with ``GPUCLOTH_ABI_SOLVE_FAILED``.  For that one the
    engine keeps the candidate, drops the latch and arms the next attempt
    (main.cpp:13460-13527), so the call after it starts from the pose this one
    published and is the attempt the recovery exists to make succeed.

    Every other refusal is structural - a misaligned owner set, a status read the
    ABI rejected, a step whose own status never accepted the frame, or an ABI
    result the recovery does not cover - and a retry would reproduce it byte for
    byte.  Those keep the fail-closed stop they have always had: the budget below
    is what bounds a solver that will not step, not a licence to retry anything.
    """
    if step.get("reason") != "step_rejected":
        return False
    return int(step.get("abi_result", -1)) == CType.GPUCLOTH_ABI_SOLVE_FAILED


def _infinite_refusal_detail(step):
    """The native cause of a refused step, for the line that reports it.

    ``step['message']`` names the object and the ABI result; the diagnostics add
    the solver's own words - ``last_error`` and the last event - so a stop says
    why rather than only that it happened.  ``_live_step_cloth_scene`` already
    read this snapshot on its own rejection path, so this is the same read and
    not a second source of truth about the same call.
    """
    index = step.get("obj_index")
    if index is None:
        return ""
    try:
        snapshot = _solver_diagnostic_snapshot(int(index))
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        return ""
    if not snapshot:
        return ""
    status = snapshot.get("status") or {}
    detail = f"last_error={status.get('last_error')}"
    events = snapshot.get("events") or []
    if events:
        last = events[-1]
        detail += (f" event_type={last.get('event_type')}"
                   f" result={last.get('result')}"
                   f" error_code={last.get('error_code')}")
    return detail


def _infinite_advance():
    """One timer tick: validate the invariant, take one step, reschedule.

    The Blender frame counter is NOT the clock of this mode.  ``frame_set`` is
    never called, ``scene.frame_current`` stays where the user left it, and the
    imported fps still defines the timestep - the native side reads the frame
    time from ``framelen``/``frs_sec`` (main.cpp:12140-12141), and
    ``scene.r.cfra`` (main.cpp:12138) is written but read by nothing else.  The
    mode's own clock is ``state['steps']``.
    """
    state = _infinite_sim_state
    try:
        if not state['running']:
            return None
        scene = bpy.data.scenes.get(state['scene_name'] or "")
        if scene is None:
            _infinite_stop("the driving scene disappeared")
            _infinite_report_target(bpy.context)
            _infinite_tag_redraw()
            return None
        # The frozen-frame contract, checked rather than assumed: the mode's
        # premise is that the timeline owns nothing, so any movement of the
        # frame counter is a stop condition with a reason (never silent).
        current_frame = int(scene.frame_current)
        if current_frame != int(state['frame']):
            if _animation_is_playing(bpy.context):
                # The cancel the start asked for is still landing, so the steps
                # the playback had already dispatched are not a playhead move -
                # they are the mode's own taking of the timeline, and reading
                # them as the user's hand is what stopped the owner's drive one
                # step after every start.  Nothing is stepped while it lands (the
                # frame path owns the timeline until it is quiet), the anchor
                # follows the frames that arrive, and the ask is repeated because
                # one ask is a request, not a state change.  Once the screen says
                # the drive is down, a moved frame is a move again and stops the
                # mode exactly as before.
                state['frame'] = current_frame
                if state['status'] != _INFINITE_TAKEOVER_STATUS:
                    state['status'] = _INFINITE_TAKEOVER_STATUS
                    _infinite_report_target(bpy.context)
                    _infinite_tag_redraw()
                _stop_animation_playback()
                return _INFINITE_SIM_INTERVAL
            _infinite_stop(
                f"timeline moved to frame {current_frame}; the drive stopped "
                f"(it started at frame {state['frame']})")
            _infinite_report_target(bpy.context)
            _infinite_tag_redraw()
            return None
        reason = _infinite_step_reason(scene)
        if reason is not None:
            _infinite_stop(_infinite_text(reason))
            _infinite_report_target(bpy.context)
            _infinite_tag_redraw()
            return None
        # Cache writes belong to the timeline path: the frame index repeats in
        # this mode, so a write would overwrite one file per step (see
        # _live_step_cloth_scene's write_cache note).
        step_t0 = time.perf_counter()
        step = _live_step_cloth_scene(
            scene, state['frame'], write_cache=False, stage_marks=None)
        step_ms = (time.perf_counter() - step_t0) * 1000.0
        state['step_ms'] = step_ms
        if step.get("warning"):
            print(f"GPUCloth infinite simulation: {step['warning']}")
        if not step.get("ok"):
            # A refusal is reported, and a solver refusal is retried: this tick
            # returns the ordinary interval, the next tick asks the same question
            # again from the pose the refusal published, and that is the attempt
            # the native recovery was built for (see the limit's own note).
            # Nothing else about the mode changes - the clock does not advance
            # (a refused step is not a step), the frame stays frozen, and the
            # retry is bounded and announced.
            detail = _infinite_refusal_detail(step)
            retryable = _infinite_refusal_is_retryable(step)
            refusals = int(state['refusals']) + 1
            state['refusals'] = refusals
            if retryable and refusals < _INFINITE_REFUSAL_RETRY_LIMIT:
                # The panel's line and the console's say the same thing, and the
                # console is where a retry streak is visible while it happens.
                state['status'] = (
                    f"step {state['steps']} refused "
                    f"({refusals}/{_INFINITE_REFUSAL_RETRY_LIMIT}); retrying"
                    + (f" [{detail}]" if detail else ""))
                print(f"[GPUCloth] infinite simulation: {state['status']}",
                      flush=True)
                _infinite_report_target(bpy.context)
                _infinite_tag_redraw()
                return _INFINITE_SIM_INTERVAL
            if retryable:
                why = f"refused {refusals} times in a row"
            else:
                why = "refused, and this refusal is not one a retry changes"
            _infinite_stop(
                f"step {state['steps']} {why}: {step['message']}"
                + (f" [{detail}]" if detail else ""))
            _infinite_report_target(bpy.context)
            _infinite_tag_redraw()
            return None
        refused_before = int(state['refusals'])
        state['refusals'] = 0
        state['steps'] = int(state['steps']) + 1
        if state['status'] == _INFINITE_TAKEOVER_STATUS or refused_before:
            # The takeover is over: the timeline is the mode's, and the panel
            # stops saying otherwise at the first step that proves it.  A retry
            # notice is the same kind of line - it describes a state this accepted
            # step has just ended - so it goes back to the run's line as well.
            state['status'] = (
                f"running from frame {state['frame']}, step {state['steps']}")
        # The first ten steps and any slow one are printed: the mode's rate is
        # a number the owner will ask for, and a step that suddenly costs
        # seconds is the difference between "the drive is running" and "the
        # drive is wedged".  Two lines per run in the steady state.
        if state['steps'] <= 10 or step_ms > 5000.0:
            print(
                f"[GPUCloth] infinite step {state['steps']}: "
                f"{step_ms:.1f} ms", flush=True)
        _infinite_tag_redraw()
        return _INFINITE_SIM_INTERVAL
    except BaseException as exc:  # noqa: BLE001 - a timer must not leak a loop
        print(
            f"GPUCloth infinite simulation: driver failed at step "
            f"{state['steps']}: {type(exc).__name__}: {exc}")
        _infinite_stop(f"driver failed: {type(exc).__name__}: {exc}")
        try:
            _infinite_report_target(bpy.context)
            _infinite_tag_redraw()
        except BaseException:  # noqa: BLE001
            pass
        return None


def _infinite_tag_redraw():
    """Repaint the panel and viewports that show the mode's own counter."""
    try:
        windows = tuple(bpy.context.window_manager.windows)
    except (AttributeError, ReferenceError, RuntimeError):
        return
    for window in windows:
        try:
            areas = tuple(window.screen.areas)
        except (AttributeError, ReferenceError, RuntimeError):
            continue
        for area in areas:
            if getattr(area, "type", None) in ('PROPERTIES', 'VIEW_3D'):
                area.tag_redraw()


def _infinite_restore_start(context):
    """Ctrl+Z: return the cloth to the pose the current preparation was built from.

    Design C at ``k = 0`` (md_caching_gap.md §5.7): the restore *is* the
    existing rebuild path, so it cannot corrupt the native owner.  The drive is
    stopped first and never resumed by this call - a restore is not a start.

    The pose it publishes is ``_initial_positions`` - the rest pose the last
    successful prepare captured and validated (:7984).  That is the checkpoint
    pose whenever the run began from it, which is the mode's own workflow (Space
    on a prepared, unstepped scene): the checkpoint *is* "before the simulation
    starts".  When the timeline had already produced frames, the pose at Space is
    one the simulation produced, and a rebuild cannot be built from it: the
    native preparation preflight refuses such a pose with its own invariant
    (measured on this fixture - a dragged pose, ``self intersection; faces
    3722/4488``, and on the mode's own oracle, ``self intersection; faces
    506/758`` and ``external clearance; vertex 320``).  The restore then left a
    session with no runnable owner and no way back, which is the owner's report
    («симуляция сломалась и отказывается возвращаться в исходное положение»).

    So the restore publishes the pose the engine can actually be rebuilt from,
    and states that in the status text rather than promising the driven pose.
    The mesh the publish is compared against is the one the prepare captured, so
    a restore that lands is also a restore the rebuild can be built from.
    """
    scene = getattr(context, "scene", None)
    _infinite_stop("returned to the start checkpoint")
    if scene is None:
        return {"ok": False, "message": _t_infinite(
            "No scene to restore", "Нет сцены для восстановления")}
    checkpoint = _infinite_sim_state['checkpoint']
    if checkpoint is None:
        return {"ok": False, "message": _t_infinite(
            "No start checkpoint was captured",
            "Чекпоинт старта не был сохранён")}
    if not g_clothOBJs or not _initial_positions:
        return {"ok": False, "message": _t_infinite(
            "No captured rest pose; prepare again",
            "Нет сохранённой rest-позы; выполните Prepare")}
    for index, cloth_obj in enumerate(g_clothOBJs):
        if index >= len(_initial_positions):
            return {"ok": False, "message": _t_infinite(
                "The cloth objects changed since the checkpoint; prepare again",
                "Объекты ткани изменились с момента чекпоинта; выполните "
                "Prepare")}
        values = _initial_positions[index]
        try:
            if values.size != len(cloth_obj.data.vertices) * 3:
                return {"ok": False, "message": _t_infinite(
                    "The cloth topology changed since the checkpoint; "
                    "prepare again",
                    "Топология ткани изменилась с момента чекпоинта; "
                    "выполните Prepare")}
            cloth_obj.data.vertices.foreach_set("co", values)
            cloth_obj.data.update()
            cloth_obj.data.update_tag()
        except (AttributeError, ReferenceError, RuntimeError, ValueError) as exc:
            return {"ok": False, "message": _t_infinite(
                f"Restore failed: {exc}", f"Восстановление не удалось: {exc}")}
    # Retained frames and the native cache transaction go with the restore: the
    # live-path history describes the trajectory that has just been rewound, so
    # keeping it would let a later scrub replay it.  ``_clear_retained_frames``
    # also clears ``rebuild_pending``, which belongs to the retained group but
    # is set here deliberately - this restore needs a rebuilt owner, because the
    # native state is at the driven pose and the mesh has just been moved back
    # under it.
    _clear_retained_frames()
    _invalidate_cache_for_change(
        scene, int(checkpoint["cache_source_generation"]))
    _simulation_frame_state['rebuild_pending'] = True
    cloth_objects = _live_cloth_objects()
    if cloth_objects:
        schedule_auto_prepare(cloth_objects[0], scene, required=True)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        depsgraph.update()
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    _infinite_sim_state['steps'] = 0
    _infinite_sim_state['refusals'] = 0
    _infinite_sim_state['status'] = (
        "restored to the prepared rest pose (where this run started); a "
        "prepare is queued, then press Space to run again")
    _infinite_report_target(context)
    _infinite_tag_redraw()
    return {"ok": True, "message": _t_infinite(
        "Restored to the start checkpoint",
        "Возврат к чекпоинту старта")}


def _infinite_toggle(context):
    """Space, once: stop the drive if it runs, otherwise start it."""
    if _infinite_is_running():
        steps = int(_infinite_sim_state['steps'])
        _infinite_stop(f"stopped by Space after {steps} step(s)")
        _infinite_report_target(context)
        _infinite_tag_redraw()
        return {"ok": True, "stopped": True, "steps": steps}
    result = _infinite_start(context)
    if result.get("ok"):
        _infinite_tag_redraw()
    else:
        _infinite_sim_state['status'] = _infinite_status_text(result)
        _infinite_report_target(context)
        _infinite_tag_redraw()
    return result


def _infinite_status_text(result):
    """One string for a refusal, whichever shape its message has."""
    message = _infinite_text(result.get("message", ""))
    reason = result.get("reason")
    return f"{message} [{reason}]" if reason else message


# ── Blender-undo boundary and observer (md_caching_gap.md §5.10) ────────────
#
#  Ctrl+Z is *observed*, never invoked.  bpy.ops.ed.undo() must never be
#  called from an operator, a modal handler or a timer: it re-enters Blender's
#  file-read path while this module's state - and the native owner it points
#  at, which undo cannot see at all - is in flight.  undo_push (the boundary
#  recorded at mode entry) only records a step, which is the safe direction.

def _infinite_undo_post_handler(*_args):
    """Fail closed after an undo the add-on did not initiate.

    The mode's premise is "the scene owns the inputs", and undo is the one
    event that can change all of them at once - including the rest shape the
    checkpoint recorded.  So: stop the drive, drop the checkpoint, drop the
    retained frames and the native cache transaction, and require a rebuild
    instead of continuing from a state whose inputs were rewound underneath.
    """
    _infinite_after_external_rewind("undo")


def _infinite_redo_post_handler(*_args):
    _infinite_after_external_rewind("redo")


def _infinite_after_external_rewind(label):
    state = _infinite_sim_state
    running = bool(state['running'])
    checkpoint = state['checkpoint']
    if not running and checkpoint is None and not _vertex_drag_state['armed']:
        return
    _infinite_stop(f"Blender {label} rewound the scene; the drive stopped")
    state['checkpoint'] = None
    state['steps'] = 0
    try:
        scene = getattr(bpy.context, "scene", None)
    except (AttributeError, ReferenceError, RuntimeError):
        scene = None
    if scene is None:
        return
    try:
        _invalidate_cache_for_change(
            scene, int(_cache_source_state['generation']))
    except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
        print(f"GPUCloth infinite simulation: cache drop after {label}: {exc}")
    cloth_objects = _live_cloth_objects()
    if cloth_objects:
        # The rebuild this queues must be built from a pose the preparation
        # preflight accepts, and the one pose known to be that is the rest the
        # last successful prepare captured: the mesh an undo restores is a state
        # the simulation produced (or the snapshot Blender took while it held
        # one), and the preflight refuses such a pose with its own invariant -
        # measured, the drag's own fold, `self intersection; faces 3722/4488`.
        # This is the sequence `_stage_changed_inputs` already uses for the same
        # reason, and it is what makes the rewind recoverable instead of terminal.
        if _initial_positions:
            _restore_initial_positions()
        _simulation_frame_state['rebuild_pending'] = True
        schedule_auto_prepare(cloth_objects[0], scene, required=True)
    state['status'] = (
        f"Blender {label} rewound the scene: the cloth was put back at the "
        f"prepared rest pose and a rebuild is queued (the checkpoint was "
        f"dropped)")
    _infinite_report_target(bpy.context)
    _infinite_tag_redraw()


# ===========================================================================
#  Единственный владелец живого шага симуляции
# ===========================================================================
#
#  Тело шага жило внутри ``GPUCloth_UpdateSimulation.execute``, и это делало
#  таймлайн-оператора единственным возможным шагающим владельцем.  Именно
#  поэтому «бесконечный» режим не сводится к привязке клавиши: ``execute``
#  возвращается раньше, когда ``frame_current < 2``, и для кадра на уже
#  решённом фронтире переигрывает удержанное состояние вместо шага.
#
#  Тело вынесено сюда и вызывается из ДВУХ мест:
#
#    * ``GPUCloth_UpdateSimulation.execute`` — путь таймлайна, со всей своей
#      логикой удержанных кадров и ``frame_current < 2`` (не изменена);
#    * ``_infinite_advance`` (см. «Бесконечная симуляция») — драйвер режима,
#      у которого свой монотонный счётчик шагов и замороженный счётчик кадров.
#
#  Ровно один владелец шага и два вызывающих — правило репозитория; две
#  независимые реализации шага были бы именно тем дефектом, от которого этот
#  вынос избавляет.
#
#  Здесь НЕТ политики: не решается, *когда* шагать, не трогается
#  ``_simulation_frame_state`` и не пишется кэш по своей воле.

def _readback(index):
    """Прочитать persistent v3 readback owner одного объекта ткани."""
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


def _apply_positions(blender_obj, pos, nVerts):
    """
    Применяет плоский float3 массив к мешу Blender через foreach_set.
    foreach_set — единственный корректный и быстрый способ в Blender 4.x.
    Прямое присваивание vertices[i].co = ... работает медленно и
    требует mesh.update() для отображения.
    """
    flat = np.frombuffer(pos, dtype=np.float32)
    blender_obj.data.vertices.foreach_set("co", flat)
    blender_obj.data.update()


def _readback_rejected_pose(index):
    """Publish the pose the engine committed for a REFUSED frame.

    A refused frame still leaves a pose: the solver ran, the audit rejected
    its candidate, and the engine keeps that candidate as the committed host
    state so the next attempt continues from it instead of from the pose it
    started from.  That is the engine's own documented intent
    (PD_frame_runtime.cu: "so the visual follows the simulated cloth even on
    rejected frames"), and showing it is why the display should follow rather
    than sit at the untouched pre-frame pose.

    This is deliberately NOT the accepted-state readback.  The equality check
    in `_readback` asserts "the accepted generation is the generation you are
    rendering", and on a refused frame that claim is false by construction:
    the accepted generation must NOT advance.  Rather than advance it (which
    would break that contract) or invent a second stamp, this path drops the
    one assertion that does not apply and keeps every other one:

      * the readback result must be ABI_OK, and
      * every position copied must be finite.

    The engine's readback copies the host vertex array on every call
    (main.cpp:13099-13107) rather than returning a retained buffer, so there
    is no staleness for the dropped check to catch — and the caller does not
    use the returned generation for anything.
    """
    if index < 0 or index >= len(_readback_owners):
        return False
    owner = _readback_owners[index]
    result = int(g_dll.GPUCloth_v3_cloth_readback(
        owner["handle"], pointer(owner["readback"])))
    if result != CType.GPUCLOTH_ABI_OK:
        return False
    positions = owner["readback_positions"]
    flat = np.frombuffer(positions, dtype=np.float32)
    if not np.isfinite(flat).all():
        return False
    simulation_obj = g_simulationOBJs[index]
    cloth_obj = g_clothOBJs[index]
    simulation_obj.data.vertices.foreach_set("co", flat)
    simulation_obj.data.update()
    if cloth_obj is not simulation_obj:
        cloth_obj.data.vertices.foreach_set("co", flat)
        cloth_obj.data.update()
    return True


def _vertex_grab_in_progress():
    """True while the vertex grab is live on the simulation mesh this scene owns.

    The grab is the only input that reaches the solver without being part of the
    prepared configuration: ``_vertex_drag_pin_snapshot`` folds its target into
    the pin snapshot of the frame being published, so a frame solved during a
    grab is an interactive position rather than a step of the simulation.  The
    cache write asks this so the editing process is never what a later playback
    replays; ``_vertex_drag_grab_is_live`` is the grab's own liveness rule -
    "the held vertex still exists in the mesh the solver owns" - and is reused
    rather than re-derived.
    """
    if not _vertex_drag_state['dragging']:
        return False
    return bool(_vertex_drag_grab_is_live())


def _vertex_grab_drops_the_cache(scene):
    """An edit of the cloth is a change the simulation depends on.  Drop the past.

    The owner's requirement for this tool is explicit: "Действия с Move Cloth by
    Vertex по прежнему кэшируются, а не сбрасывают кэш и не переходят в режим
    бесконечной симуляции" - a grab must drop the cache and leave a session that
    caches nothing, rather than an edit the cache quietly absorbs.

    The drop is the one every other change already uses
    (``_invalidate_cache_for_change``): the retained in-session states go
    (``_clear_retained_frames``), the native store is cleared, and the range is
    restated so the engine raises its own OUTDATED flag.  What the grab adds is
    only *when* - at the moment the vertex is taken, before the first edited frame
    is produced - because the frames on either side of it are not comparable: the
    stored ones were produced with the cloth free and the grabbed ones with it
    pinned.

    Dropping the run claim is what makes "caches nothing" true rather than
    hopeful, and it needs no rule of its own: ``_clear_retained_frames`` retires
    ``run_first``/``run_generation``, and only a run opened at the start of the
    range may claim one again, so nothing produced during the grab can be written
    (``_frame_may_be_persisted``) and nothing the previous run wrote can be served
    (``_load_cached_frame``).  The claim is retired for the rest of the session:
    it comes back only when the simulation next runs from the beginning of the
    range, which is where the owner's "пока не начнём с начала" ends.

    The generation is read for real here for the same reason the dependency-graph
    handler reads it: this is a change that decides something on its own, and the
    frame-path memo would answer with the value from before the edit.
    """
    generation = _cache_source_generation_fresh(scene)
    _cache_source_state['generation'] = generation
    _invalidate_cache_for_change(scene, generation, staged=False)
    return True


def _live_step_cloth_scene(scene, frame, write_cache=True, stage_marks=None):
    """Один живой шаг симуляции для всех объектов ткани в сцене.

    ``scene`` — сцена, чей кадр шагается.  Контекст здесь не нужен:
    ``_publish_frame_inputs`` и ``_cache_write_frame`` читают ``scene``
    (первая берёт граф зависимостей через ``bpy.context``), и шаг вызывается
    ещё и таймером, у которого нет контекста вызывающего.  Таймлайн-оператор
    передаёт ``context.scene``.

    ``frame`` — индекс кадра для записи в кэш и для единственной строки
    «кадр N».  Вызывающий берёт его из ``scene.frame_current``.

    ``write_cache=False`` использует драйвер бесконечного режима: он никогда
    не пишет покадровый кэш.  Таймлайн в этом режиме заморожен, поэтому индекс
    кадра повторялся бы и каждый шаг перезаписывал бы один файл, а нативный
    гейт отвергает только *понижение* generation записи — повтор его проходит.

    ``stage_marks`` — список, в который складываются ``(label, ns)`` замеры
    стадий, когда ``GPUCLOTH_FRAME_PROFILE`` включён.  Профайлер остаётся делом
    вызывающего: это диагностика, а не шаг.

    Возвращает словарь: ``ok``, ``cancelled``, ``reason`` (``owners_misaligned``
    / ``status_in`` / ``step_rejected`` / ``status_not_accepted`` /
    ``exception``), ``message``, ``object_name``, ``abi_result``, ``warning``,
    ``write_rejected``.
    """
    if not (len(g_clothOBJs) == len(g_simulationOBJs)
            == len(g_cloth_handles) == len(_readback_owners)):
        return {
            "ok": False, "cancelled": True, "reason": "owners_misaligned",
            "message": (
                f"Несоответствие размеров: clothOBJs={len(g_clothOBJs)}, "
                f"simulationOBJs={len(g_simulationOBJs)}, "
                f"cloth_handles={len(g_cloth_handles)}, "
                f"readback={len(_readback_owners)}"),
        }

    # The inputs this frame is being solved under, read once here and handed to the
    # write gate below: a frame is a statement about these and nothing else, and the
    # engine cannot tell afterwards whether they moved.  The read goes through the
    # frame-path memo, so this frame path - the gate included - is not a second read
    # of every hashed input; the gate compares the value it is handed against the
    # run claim, not against a fresh fingerprint of its own.
    # The revision is read beside it because it is the cheap revision the gate
    # compares against: it moves for any ``GPUCloth`` scalar and for the Scene
    # scalars no callback reaches, and everything else reaches the memo through the
    # notification channel, so a step that began under one revision and writes
    # under another is a step whose inputs moved.
    generation = _cache_source_generation(scene)
    revision = _cache_fingerprint_key(scene)
    notifications = int(_cache_unactionable_notifications)
    run = {'generation': generation}
    profiled = stage_marks is not None
    stage_t = time.perf_counter_ns()

    def _mark_stage(label):
        nonlocal stage_t
        if not profiled:
            return
        now = time.perf_counter_ns()
        stage_marks.append((label, now - stage_t))
        stage_t = now

    try:
        # This used to evaluate the dependency graph here and pass it to
        # _publish_frame_inputs, which never read it - that function takes
        # its own graph from context.evaluated_depsgraph_get() after the
        # modifier isolation below.  Nothing between here and there reads
        # evaluated data, so the eager evaluation only added a second full
        # evaluation of the mesh the previous frame wrote.
        _publish_frame_inputs(bpy.context)
        _mark_stage("publish_inputs")
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
                return {
                    "ok": False, "cancelled": True, "reason": "status_in",
                    "object_name": cloth_obj.name_full, "abi_result": result,
                    "message": f"v3 cloth status rejected with {result}",
                }
            _mark_stage("get_status_in")

            # 1. GPU simulation одного кадра.
            result = int(g_dll.GPUCloth_v3_cloth_step(cloth_handle))
            _mark_stage("cloth_step")
            if result != CType.GPUCLOTH_ABI_OK:
                _solver_diagnostic_snapshot(i)
                # A refused frame still committed a pose: the engine keeps the
                # rejected candidate as its host state so the next attempt
                # continues from it.  Publish that pose so the display follows
                # the solver instead of sitting at the untouched pre-frame
                # geometry.  This does not accept the frame - nothing below
                # runs, the accepted generation does not advance, and the
                # caller still reports the rejection and cancels.
                _mark_stage("rejected_pose_readback")
                warning = None
                try:
                    _readback_rejected_pose(i)
                except (OSError, RuntimeError, AttributeError) as exc:
                    warning = (
                        f"could not publish the refused pose for "
                        f"{cloth_obj.name_full}: {exc}")
                return {
                    "ok": False, "cancelled": True, "reason": "step_rejected",
                    "object_name": cloth_obj.name_full, "abi_result": result,
                    "obj_index": i, "warning": warning,
                    "message": (
                        f"v3 cloth step rejected {cloth_obj.name_full}: "
                        f"{result}"),
                }
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
                return {
                    "ok": False, "cancelled": True,
                    "reason": "status_not_accepted",
                    "object_name": cloth_obj.name_full, "abi_result": result,
                    "message": "v3 cloth status did not accept stepped frame",
                }
            _accept_dynamic_mesh_snapshot(i)
            # The "native X ms" line is the only consumer of this snapshot, and it
            # costs one get_diagnostics ABI call plus a ~30-key dict every frame per
            # object.  GPUClothV3ClothStatus has no execution_time_ms field, so the
            # value cannot be taken from the status read a few lines above; instead
            # the call is opt-in, on the same flag as the stage profiler.
            if profiled:
                diagnostic = _solver_diagnostic_snapshot(i)
                if diagnostic is not None:
                    native_ms = diagnostic["status"]["execution_time_ms"]
                    print(
                        f"[GPUCloth] {cloth_obj.name_full}: "
                        f"native {native_ms:.3f} ms")
            _mark_stage("status_accept_diag")

            # 2. Persistent v3 readback positions/velocities.
            simulation_pos = _readback(i)
            _mark_stage("readback")

            # 3. Handle-scoped v3 proxy apply into the persistent output.
            proxy_owner = (g_proxy_handles[i]
                           if i < len(g_proxy_handles) else None)
            if proxy_owner is not None:
                if int(proxy_owner["proxy_vertex_count"]) != n_sim:
                    raise RuntimeError(
                        "v3 proxy status count differs from simulation mesh")
                _apply_positions(simulation_obj, simulation_pos, n_sim)
                pos = _apply_v3_proxy(
                    proxy_owner, _runtime_frame_generation)
                nV = int(proxy_owner["render_vertex_count"])
            else:
                pos = simulation_pos
                nV = n_sim

            # 4. Обновляем меш в Blender (foreach_set, Blender 4.x safe)
            _apply_positions(cloth_obj, pos, nV)
            _mark_stage("apply_positions")

            # 5. Handle-scoped v3 cache write. Native copies the payload
            # before returning; Python owns no cache array after call.
            # A frame produced while the vertex grab is live is not written.
            # The grab folds its target into the pin snapshot this step
            # published (`_vertex_drag_pin_snapshot`), so the geometry is the
            # *editing*, not the simulation: keeping it would make a later
            # playback replay the drag and a bake would freeze a transient
            # hand position into the archive.  The result of the edit is not
            # lost: the solved state is retained below exactly as any other
            # frame's is, so the look stays on screen and a rewind inside this
            # session republishes it from the retained store
            # (`_load_simulation_frame`).
            if (write_cache and
                    _frame_may_be_persisted(
                        scene, frame, run, generation, revision,
                        notifications) and
                    not scene.gpu_cloth_helper.is_baked and
                    not bool(getattr(
                        scene.gpu_cloth_helper,
                        "use_external_cache", False)) and
                    _cache_handle_value()):
                if (_cache_write_frame(
                        scene, frame, pos, nV) !=
                        CType.GPUCLOTH_ABI_OK):
                    return {
                        "ok": True, "cancelled": False,
                        "write_rejected": True,
                        "message": f"Cache write rejected at frame {frame}",
                    }

        # The retained store is refreshed in place and is only reachable through
        # ``last_solved``, which a run that began at the start of the range owns
        # (see the update operator).  So it needs no gate of its own: a rewind can
        # only reach frames the current run has already produced, and producing a
        # frame overwrites what the previous run left here.
        _store_simulation_frame(frame)
        _mark_stage("store_frame")

    except (OSError, RuntimeError, VertexChannelError) as err:
        print(f"GPUCloth_UpdateSimulation failed: {err}")
        return {
            "ok": False, "cancelled": True, "reason": "exception",
            "message": f"GPUCloth_UpdateSimulation failed: {err}",
        }

    return {"ok": True, "cancelled": False}

class GPUCloth_UpdateSimulation(bpy.types.Operator):
    """Просчитать один кадр симуляции и обновить меш в Blender"""
    bl_idname = "gpucloth.update_simulation"
    bl_label  = "Update GPUCloth Simulation"

    # ── Вспомогательные методы ───────────────────────────────────────────────
    #
    #  Все три делегируют модульным функциям того же имени, потому что тело
    #  шага теперь вызывается ещё и драйвером бесконечного режима, у которого
    #  нет экземпляра оператора.  Делегаты оставлены намеренно: имена
    #  ``GPUCloth_UpdateSimulation._readback`` / ``._apply_positions`` /
    #  ``._readback_rejected_pose`` используются существующими пробами и
    #  гейтами, и переименование сломало бы их без причины.

    def _readback(self, index):
        return _readback(index)

    def _apply_positions(self, blender_obj, pos, nVerts):
        return _apply_positions(blender_obj, pos, nVerts)

    def _readback_rejected_pose(self, index):
        return _readback_rejected_pose(index)

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

        if prepare_task_active() or _teardown_failure or _stop_requested:
            self.report(
                {'ERROR'},
                "Simulation is stopped; prepare is required before update")
            return {'CANCELLED'}

        if g_dll is None or context.scene.frame_current < 2:
            return {'FINISHED'}
        if not self.validate_objects():
            return {'FINISHED'}
        if _refresh_prepared_inputs(context.scene) or (
                _simulation_frame_state['rebuild_pending']):
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
            depsgraph = context.evaluated_depsgraph_get()
            if not _load_cached_frame(context.scene, depsgraph, frame):
                _load_simulation_frame(frame, depsgraph)
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

        # Per-stage frame timing, opt-in.  The addon already prints the frame total at
        # the end of this method; this breaks that total into the stages, because the
        # engine-side measurement shows the solver is only a small part of the wall time
        # a user sees in Blender - on DrapeOnSphere the native frame is ~160 ms while the
        # addon path is over a second.  Set GPUCLOTH_FRAME_PROFILE=1 to enable; disabled
        # it costs one attribute lookup per stage.  The list is only handed to the step
        # core when profiling is on, which is also how the core decides whether the
        # per-object diagnostics snapshot is worth its ABI call.
        _stage_profile = (os.environ.get("GPUCLOTH_FRAME_PROFILE", "") not in ("", "0"))
        _stage_marks = [] if _stage_profile else None

        # ── ЖИВОЙ ШАГ ────────────────────────────────────────────────────────
        #
        #   GPUCloth_v3_cloth_step(handle) [GPU]
        #   GPUCloth_v3_cloth_readback(handle) [D2H]
        #   foreach_set() [Blender ~<1мс]
        #   v3 cache write copies payload before return.
        #
        #   Тело шага живёт в _live_step_cloth_scene, потому что у него есть
        #   второй вызывающий — драйвер бесконечного режима.  Здесь остаётся
        #   вся политика таймлайна: ранние выходы выше, удержанные кадры,
        #   запись в кэш и last_solved ниже.
        step = _live_step_cloth_scene(
            context.scene, frame, write_cache=True, stage_marks=_stage_marks)

        if step.get("warning"):
            self.report({'WARNING'}, step["warning"])
        if not step.get("ok"):
            self.report({'ERROR'}, step["message"])
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}
        if step.get("write_rejected"):
            self.report({'ERROR'}, step["message"])
            return {'CANCELLED'}

        # The past is the cache: it may only record frames of a run that began at
        # the start of the range.  A run that lost its cache to a scene change and
        # kept stepping from wherever it was has produced real geometry, but it is
        # geometry of a simulation that never ran from the beginning, so it must
        # not become the thing a rewind is served from.  Leaving ``last_solved``
        # alone is what makes the next request re-simulate from the start instead,
        # which is Blender's behaviour and the owner's ruling.
        if _cache_source_state['run_first'] is not None:
            _simulation_frame_state['last_solved'] = frame
        elapsed_ms = (time.perf_counter_ns() - total_t0) / 1_000_000
        print(f"[GPUCloth] кадр {frame}: {elapsed_ms:.2f} мс")
        if _stage_marks:
            # Printed on its own line so the existing "[GPUCloth] кадр N: X мс" line keeps
            # its format.  Compare the cloth_step entry against the native time the native
            # layer reports: if cloth_step is the bulk of the frame then the solver owns
            # the wall time and the addon's Python path is not the place to optimise.
            _parts = " ".join(
                f"{label}={ns / 1e6:.1f}" for label, ns in _stage_marks)
            print(f"[GPUCloth] stage frame {frame}: {_parts} "
                  f"(total {elapsed_ms:.1f} ms)")
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


def abort_live_drape_sandbox(context):
    """Close every live drape sandbox and republish its Begin snapshot.

    A live sandbox is the state in which the cloth's published mesh is the
    sandbox's *preview* pose rather than any pose the timeline or the solver
    owns: `_readback_drape_preview` writes it on every Settle tick and on every
    Step.  Cancel exists to undo exactly that, and Stop destroys the owners the
    snapshot lives in, so every path that closes a sandbox without applying it
    has to go through here first - otherwise the preview pose is left published
    with no owner left to restore it, and the next Prepare captures that pose as
    the rest shape.

    Returns the number of sandboxes closed.  A sandbox that cannot be reached
    is reported and left alone rather than half-closed: the one thing this
    function must never do is free without restoring.
    """
    if g_dll is None:
        return 0
    closed = 0
    for index, cloth_obj in enumerate(list(g_clothOBJs)):
        if index >= len(g_cloth_handles):
            break
        try:
            uid = _blender_session_uid(cloth_obj, "drape cloth")
        except RuntimeError:
            continue
        status = _drape_status_by_uid.get(uid)
        if status is None:
            # No memo means no sandbox this session opened; the native owner is
            # the only remaining authority and Cancel is refused when it is not
            # active, so an unread memo is not a reason to guess.
            continue
        flags = int(status.get("status_flags", 0))
        if not flags & CType.GPUCLOTH_DRAPE_STATUS_ACTIVE:
            continue
        if flags & CType.GPUCLOTH_DRAPE_STATUS_CONVERGED:
            # A converged sandbox is the user's to Apply; closing it here would
            # discard the result the Apply button is waiting for.
            continue
        try:
            handle = g_cloth_handles[index]
            native_status = _new_drape_status()
            result = int(g_dll.GPUCloth_v3_cloth_cancel_drape(
                handle, pointer(native_status)))
            _remember_drape_status(cloth_obj, native_status)
            if result != CType.GPUCLOTH_ABI_OK:
                print(
                    f"[GPUCloth] drape sandbox on {cloth_obj.name_full} was not "
                    f"closed before teardown: Cancel rejected with "
                    f"{_abi_result_name(result)}")
                continue
            _readback_drape_preview(index)
            closed += 1
            print(
                f"[GPUCloth] drape sandbox on {cloth_obj.name_full} cancelled; "
                "the Begin snapshot is published again")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(
                f"[GPUCloth] drape sandbox on {cloth_obj.name_full} was not "
                f"closed before teardown: {type(exc).__name__}: {exc}")
    return closed


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
            _publish_frame_inputs(context)
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
            state = _remember_drape_status(cloth_obj, status)
            remember_drape_action(cloth_obj)
            if result != CType.GPUCLOTH_ABI_OK:
                # The native side now fills the witness in for every refusal it
                # can name, so the witness *is* a diagnosis when it carries one
                # and `drape_refusal_message` leads with it.  When it carries
                # none, the check is named from the ABI result, because the
                # witness at its reset value says nothing.
                witness = _invariant_witness_data(cloth_handle)
                detail = None
                if witness["invariant"] != CType.GPUCLOTH_INVARIANT_NONE:
                    step = _BEGIN_DRAPE_WITNESS_STEP
                else:
                    step, detail = _BEGIN_DRAPE_STEP_REASONS.get(
                        result, ("unknown native check",
                                 "the native sandbox refused the drape without "
                                 "naming a check"))
                if result == CType.GPUCLOTH_ABI_INVALID_STATE:
                    # Which of the guard's alternatives fired is otherwise
                    # decidable only from this counter - it is the field the
                    # guard itself tests, and the Prepare row shows the same one.
                    preparation = get_preparation_ui_status(cloth_obj)
                    accepted = (
                        preparation["accepted_generation"]
                        if preparation is not None else "unreadable")
                    detail = (
                        f"a frame has already been accepted since the last "
                        f"Prepare: prepared accepted_generation={accepted}; "
                        "Prepare resets it to 0")
                self.report(
                    {'ERROR'},
                    drape_refusal_message(
                        "Drape Begin", result, state, step, witness, detail))
                return {'CANCELLED'}
            self.report({'INFO'}, "Drape sandbox started")
            return {'FINISHED'}
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, f"Drape Begin failed: {exc}")
            return {'CANCELLED'}


# ── Drape Settle: a bounded, timer-driven advance ───────────────────────────
# Settle used to run all 240 native drape steps inside one execute() call on the
# main thread, so Blender serviced no event until the drape budget ran out and
# the window looked frozen.  It now advances from the event timer, exactly as
# the bake operator does.  One tick is bounded twice - by step count and by wall
# clock - because a native step costs whatever the cloth costs: a fixed count
# alone would still hold the thread for seconds on a dense cloth.
_DRAPE_SETTLE_TIMER_S = 0.01
_DRAPE_SETTLE_STEPS_PER_TICK = 4
_DRAPE_SETTLE_TICK_BUDGET_S = 0.05
# Consecutive frames at or below the drape's position tolerance that count as
# "at rest".  The native status counts its own window with the same length, so
# the drape does not invent a second window shape; only the criterion differs.
_DRAPE_SETTLE_WINDOW = 8


def drape_not_settled(state):
    """True when the native status reports its own step budget exhausted.

    ``GPUCLOTH_DRAPE_STATUS_FAILED`` with ``GPUCLOTH_DRAPE_RESULT_NOT_CONVERGED``
    is the native sandbox saying its 240-step cap expired while the cloth was
    still moving.  It is not a fault and it is not the drape's verdict: the
    sandbox keeps accepting steps, so the panel row stays live and the Settle
    keeps advancing until its own position criterion or its own budget decides.
    """
    return bool(
        state["status_flags"] & CType.GPUCLOTH_DRAPE_STATUS_FAILED and
        state["result"] == CType.GPUCLOTH_DRAPE_RESULT_NOT_CONVERGED)


def _drape_invariant_name(state):
    """Name the invariant a rejected drape reported, without a witness dump."""
    return _INVARIANT_NAMES.get(
        int(state["last_error"]), f"UNKNOWN_{int(state['last_error'])}")


def drape_frame_seconds(scene):
    """Simulated seconds one native drape step advances.

    A drape step is one ``SIM_solver_cloth`` frame - the substep count is
    ``stepsPerFrame`` inside that call - so it advances ``fps_base / fps``
    seconds.  Measured on the free-fall fixture: 240 steps moved the cloth to
    the depth free fall reaches in 9.84 s, against 10.0 s from 240 frames at
    24 fps.
    """
    render = getattr(scene, "render", None)
    fps = float(getattr(render, "fps", 0) or 0)
    base = float(getattr(render, "fps_base", 1.0) or 1.0)
    if fps <= 0.0:
        return 1.0 / 24.0
    return base / fps


def drape_criterion(cloth_obj, scene):
    """The drape's own criterion: (position tolerance m/frame, budget seconds).

    Owned by the drape path.  The solver's ``solver_convergence_tol`` is a force
    tolerance and is deliberately not read here.
    """
    settings = getattr(cloth_obj, "GPUCloth", None)
    tolerance = float(getattr(
        settings, "drape_position_tolerance", 0.0015) or 0.0015)
    budget = float(getattr(settings, "drape_budget_s", 24.0) or 24.0)
    return max(tolerance, 1e-6), max(budget, drape_frame_seconds(scene))


def _remember_drape_verdict(cloth_obj, verdict):
    uid = _blender_session_uid(cloth_obj, "drape cloth")
    _drape_settle_verdict_by_uid[uid] = verdict
    return verdict


def get_drape_settle_verdict(cloth_obj):
    """The last completed Settle verdict for this cloth, or None."""
    try:
        uid = _blender_session_uid(cloth_obj, "drape cloth")
    except RuntimeError:
        return None
    return _drape_settle_verdict_by_uid.get(uid)


def remember_drape_action(cloth_obj):
    """Drop a stale Settle verdict: any new drape action supersedes it."""
    try:
        uid = _blender_session_uid(cloth_obj, "drape cloth")
    except RuntimeError:
        return
    _drape_settle_verdict_by_uid.pop(uid, None)


def _drape_refusal(cloth_obj, cloth_handle, result, status):
    """Classify a refused native drape step: (report_kind, message, return).

    A step can be refused because the sandbox ended, and that is not a fault.
    The status memo is the truthful owner of "the sandbox ended": it says so for
    an explicit Cancel or Apply (the user's own doing, INFO) and it is gone
    entirely when a rebuild replaced the solver owner (WARNING, and Begin starts
    a fresh sandbox).  Only a refusal with the sandbox still live is a real
    rejection, and that one names the check from the ABI result and adds the
    native witness when - and only when - an invariant was actually broken.  The
    line is printed because a modal handler's report does not reach the console,
    so a refused settle would otherwise leave no trace there at all.
    """
    prior = get_drape_ui_status(cloth_obj)
    if prior is None:
        print("[GPUCloth] drape step refused: the drape sandbox was replaced "
              "by a rebuild")
        return (
            {'WARNING'},
            "Settle stopped: the drape sandbox was replaced by a rebuild; "
            "press Begin to start a new drape sandbox",
            {'CANCELLED'})
    flags = int(prior.get("status_flags", 0))
    if flags & (CType.GPUCLOTH_DRAPE_STATUS_CANCELLED |
                CType.GPUCLOTH_DRAPE_STATUS_APPLIED):
        print(f"[GPUCloth] drape step refused: the sandbox ended at step "
              f"{prior.get('step_count', 0)}")
        return (
            {'INFO'},
            f"Drape stopped at step {prior.get('step_count', 0)}: the drape "
            "sandbox ended",
            {'CANCELLED'})
    witness = _invariant_witness_data(cloth_handle)
    if witness["invariant"] != CType.GPUCLOTH_INVARIANT_NONE:
        step, detail = "solver verdict", None
    else:
        step, detail = _STEP_DRAPE_STEP_REASONS.get(
            int(result), ("unknown native check",
                          "the native sandbox refused the step without naming "
                          "a check"))
    message = drape_refusal_message(
        "Drape frame", result, status, step, witness, detail)
    print(f"[GPUCloth] drape step refused: {message}")
    return ({'ERROR'}, message, {'CANCELLED'})


def drape_settle_running(context):
    """True while a Settle modal owns a timer on this window.

    Blender's own modal bookkeeping is the source of truth: a module flag would
    stay set if a window closed under a running timer, which would refuse every
    later Settle.  When the window does not expose the collection the guard
    allows the call rather than inventing a refusal.
    """
    window = getattr(context, "window", None)
    modal = getattr(window, "modal_operators", None)
    if modal is None:
        return False
    return any(isinstance(operator, GPUCloth_StepDrape)
               for operator in modal)


class GPUCloth_StepDrape(bpy.types.Operator):
    bl_idname = "gpucloth.step_drape"
    bl_label = "Step Drape"
    bl_description = (
        "Advance Drape without moving timeline or writing cache. The Settle "
        "form advances on the event timer and stops on ESC")

    until_settled: bpy.props.BoolProperty(
        name="Until settled", default=False)

    _timer = None
    _index = -1
    _cloth_obj = None
    _cloth_handle = None

    @classmethod
    def poll(cls, context):
        status = (get_drape_ui_status(context.object)
                  if context.object is not None else None)
        if not status:
            return False
        flags = status["status_flags"]
        if flags & CType.GPUCLOTH_DRAPE_STATUS_CONVERGED:
            return False
        # A drape that ran out of steps without coming to rest is not wedged:
        # the native sandbox still accepts steps, so the row stays live.
        return bool(
            flags & CType.GPUCLOTH_DRAPE_STATUS_ACTIVE or
            drape_not_settled(status))

    def invoke(self, context, event):
        if not self.until_settled:
            return self.execute(context)
        window = getattr(context, "window", None)
        window_manager = getattr(context, "window_manager", None)
        if window is None or window_manager is None:
            self.report({'ERROR'}, "Settle needs a window")
            return {'CANCELLED'}
        if drape_settle_running(context):
            self.report({'INFO'}, "Settle is already running")
            return {'CANCELLED'}
        try:
            self._index, self._cloth_obj, self._cloth_handle = (
                _selected_drape_owner(context))
            self._tolerance, self._budget_s = drape_criterion(
                self._cloth_obj, context.scene)
            self._frame_seconds = drape_frame_seconds(context.scene)
            # The budget is a duration, so the step ceiling follows from the
            # scene's frame rate instead of being a bare cap.  One step per
            # frame means ceil() of the frames the budget buys.
            self._step_ceiling = max(
                1, int(math.ceil(self._budget_s / self._frame_seconds)))
            self._stable_steps = 0
            self._timer = window_manager.event_timer_add(
                _DRAPE_SETTLE_TIMER_S, window=window)
            window_manager.modal_handler_add(self)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._finish(context)
            self.report({'ERROR'}, f"Drape Step failed: {exc}")
            return {'CANCELLED'}
        print(
            f"[GPUCloth] drape settle started: tol "
            f"{self._tolerance * 1000.0:.3f} mm/frame, budget "
            f"{self._budget_s:.1f} s simulated ({self._step_ceiling} steps at "
            f"{self._frame_seconds:.4f} s/frame)")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            try:
                outcome = self._advance()
                if outcome is not None:
                    _readback_drape_preview(self._index)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self._finish(context)
                self.report({'ERROR'}, f"Drape Step failed: {exc}")
                return {'CANCELLED'}
            self._tag_redraw(context)
            if outcome is None:
                return {'RUNNING_MODAL'}
            self._finish(context)
            kind, message, result = outcome
            self.report(kind, message)
            return result
        if event.type == 'ESC':
            # ESC stops the advance and keeps the sandbox: the Begin snapshot
            # is still there for Cancel, and the panel still owns the drape.
            state = get_drape_ui_status(self._cloth_obj) or {}
            step = int(state.get("step_count", 0))
            simulated = step * self._frame_seconds
            self._finish(context)
            try:
                _readback_drape_preview(self._index)
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
            self._tag_redraw(context)
            print(f"[GPUCloth] drape settle stopped by ESC at step {step} "
                  f"({simulated:.1f} s simulated)")
            self.report(
                {'INFO'},
                f"Settle stopped at {simulated:.1f} s of simulated drape; the "
                "drape sandbox is kept")
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def _advance(self):
        """One bounded batch of native drape steps; ``None`` while running.

        The wall-clock budget is checked before every step after the first, so
        the batch is at least one step but stops as soon as the cloth has cost
        the tick its time.  On the fixture measured for this fix the first step
        of a drape carries the solver warm-up and costs an order of magnitude
        more than the ones after it; without the pre-check that warm-up step
        dragged three more steps into the same tick.

        The verdict is the drape's own: 8 consecutive frames at or below the
        drape's position tolerance, or the native sandbox's own window.  The
        native FAILED flag is used for exactly one thing - remembering that the
        sandbox's 240-step cap expired - and never as "not settled", because the
        cloth is usually still draping at that point and the budget in simulated
        seconds is what decides.

        What advances between ticks is the drape *status*, which is what the
        panel's step and progress rows read.  The mesh is published once, on the
        tick that ends the settle: writing it every tick puts a dependency-graph
        evaluation of the whole cloth between every batch, and measured on the
        fixture that cost 2.8x the wall time and 66 frames over 50 ms in one
        settle against 2.
        """
        deadline = time.perf_counter() + _DRAPE_SETTLE_TICK_BUDGET_S
        for step_index in range(_DRAPE_SETTLE_STEPS_PER_TICK):
            if step_index and time.perf_counter() >= deadline:
                break
            status = _new_drape_status()
            result = int(g_dll.GPUCloth_v3_cloth_step_drape(
                self._cloth_handle, pointer(status)))
            if result != CType.GPUCLOTH_ABI_OK:
                return _drape_refusal(
                    self._cloth_obj, self._cloth_handle, result, status)
            state = _remember_drape_status(self._cloth_obj, status)
            delta = float(state["maximum_position_delta"])
            simulated = float(state["step_count"]) * self._frame_seconds
            if delta <= self._tolerance:
                self._stable_steps += 1
            else:
                self._stable_steps = 0
            if state["status_flags"] & CType.GPUCLOTH_DRAPE_STATUS_CONVERGED:
                return self._settled(state, simulated, "native window")
            if self._stable_steps >= _DRAPE_SETTLE_WINDOW:
                return self._settled(state, simulated, "position")
            if (state["step_count"] >= self._step_ceiling or
                    simulated >= self._budget_s):
                return self._budget_exhausted(state, simulated)
        return None

    def _settled(self, state, simulated, how):
        delta_mm = float(state["maximum_position_delta"]) * 1000.0
        tolerance_mm = self._tolerance * 1000.0
        print(f"[GPUCloth] drape settled: {simulated:.2f} s simulated, step "
              f"{state['step_count']}, {delta_mm:.3f} mm/frame <= "
              f"{tolerance_mm:.3f} mm/frame ({how})")
        _remember_drape_verdict(self._cloth_obj, {
            "settled": True,
            "reason": how,
            "step_count": int(state["step_count"]),
            "simulated_s": round(simulated, 4),
            "maximum_position_delta_m": float(
                state["maximum_position_delta"]),
            "position_tolerance_m": float(self._tolerance),
            "budget_s": float(self._budget_s),
        })
        return (
            {'INFO'},
            f"Drape settled after {simulated:.1f} s of simulated drape "
            f"(moving {delta_mm:.2f} mm/frame against {tolerance_mm:.2f} "
            "mm/frame)",
            {'FINISHED'})

    def _budget_exhausted(self, state, simulated):
        """Report a Settle that ran out of its own allowance, with the numbers.

        The old sentence said "after N s of simulated drape", and the wording is
        what was wrong rather than the number: `N` is the budget being
        announced, and reading it as an elapsed measurement is the mistake the
        owner made.  Worse, the sentence carried nothing the reader could check
        it against - no step count, no comparison point - so a settle stopped
        early by anything at all read exactly like one that spent its allowance.

        What is decidable is now stated: the step count reached, the ceiling
        that count is measured against, and the budget in simulated seconds.
        Those three together answer "was the settle stopped by its budget, or
        did something cap it sooner?" without inventing a distinction the
        arithmetic does not have - `_step_ceiling` is `ceil(budget_s /
        frame_seconds)`, so the run that reaches the ceiling has, by
        construction, spent the budget, and the step count is the budget
        expressed in frames.
        """
        delta_mm = float(state["maximum_position_delta"]) * 1000.0
        tolerance_mm = self._tolerance * 1000.0
        ceiling = int(self._step_ceiling)
        spent = (state["step_count"] >= ceiling)
        print(f"[GPUCloth] drape not settled: still moving {delta_mm:.2f} "
              f"mm/frame against {tolerance_mm:.2f} mm/frame after "
              f"{simulated:.2f} s of simulated drape spent (step "
              f"{state['step_count']} of {ceiling}); the sandbox is kept")
        _remember_drape_verdict(self._cloth_obj, {
            "settled": False,
            "reason": "budget",
            "step_count": int(state["step_count"]),
            "simulated_s": round(simulated, 4),
            "maximum_position_delta_m": float(
                state["maximum_position_delta"]),
            "position_tolerance_m": float(self._tolerance),
            "budget_s": float(self._budget_s),
            "step_ceiling": ceiling,
            "allowance_exhausted": bool(spent),
        })
        return (
            {'WARNING'},
            f"Drape did not settle: still moving {delta_mm:.1f} mm/frame "
            f"against {tolerance_mm:.1f} mm/frame after {simulated:.1f} s of "
            f"simulated drape spent (step {state['step_count']} of {ceiling} "
            f"at {1.0 / self._frame_seconds:g} fps); the sandbox "
            "is kept - Step or Settle to advance it again, or Cancel to "
            "restore the Begin snapshot",
            {'FINISHED'})

    def _finish(self, context):
        timer = self._timer
        self._timer = None
        window_manager = getattr(context, "window_manager", None)
        if timer is None or window_manager is None:
            return
        try:
            window_manager.event_timer_remove(timer)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _tag_redraw(self, context):
        screen = getattr(context, "screen", None)
        if screen is None:
            return
        for area in screen.areas:
            area.tag_redraw()

    def execute(self, context):
        """Advance the drape by exactly one native step.

        This is the synchronous path, so it never loops: a batch here would
        hold the main thread for as long as the cloth takes, which is the
        freeze being fixed.  The Settle form reaches the timer through
        ``invoke``, where the main thread is free between ticks.
        """
        try:
            index, cloth_obj, cloth_handle = _selected_drape_owner(context)
            tolerance, budget_s = drape_criterion(cloth_obj, context.scene)
            frame_seconds = drape_frame_seconds(context.scene)
            remember_drape_action(cloth_obj)
            status = _new_drape_status()
            result = int(g_dll.GPUCloth_v3_cloth_step_drape(
                cloth_handle, pointer(status)))
            # The refusal is classified before the memo is written: a refused
            # call leaves out_status untouched, so remembering it here would
            # replace the real drape progress in the panel with zeros and hide
            # the very state the refusal has to be read against.
            if result != CType.GPUCLOTH_ABI_OK:
                kind, message, result_set = _drape_refusal(
                    cloth_obj, cloth_handle, result, status)
                self.report(kind, message)
                return result_set
            state = _remember_drape_status(cloth_obj, status)
            _readback_drape_preview(index)
            delta_mm = float(state["maximum_position_delta"]) * 1000.0
            simulated = float(state["step_count"]) * frame_seconds
            tolerance_mm = tolerance * 1000.0
            if state["status_flags"] & CType.GPUCLOTH_DRAPE_STATUS_CONVERGED:
                self.report(
                    {'INFO'},
                    f"Drape converged at {simulated:.1f} s of simulated drape")
                return {'FINISHED'}
            self.report(
                {'INFO'},
                f"Drape step {state['step_count']} ({simulated:.1f} s "
                f"simulated; moving {delta_mm:.2f} mm/frame against "
                f"{tolerance_mm:.2f} mm/frame, budget {budget_s:.1f} s)")
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
            remember_drape_action(cloth_obj)
            if result != CType.GPUCLOTH_ABI_OK:
                raise RuntimeError(f"native Apply rejected with {result}")
            # The sandbox is closed, so a prepare held for it can run now.
            resume_held_prepare()
            _readback_drape_preview(index)
            _store_initial_positions()
            _clear_retained_frames()
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
            remember_drape_action(cloth_obj)
            if result != CType.GPUCLOTH_ABI_OK:
                raise RuntimeError(f"native Cancel rejected with {result}")
            # The sandbox is closed, so a prepare held for it can run now.
            resume_held_prepare()
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
        "drape_settle": get_drape_settle_verdict(cloth_obj),
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
        _clear_retained_frames()
        # A bake runs the range in order from its start, so the run begins here:
        # every frame it writes belongs to a simulation that ran from the
        # beginning, which is the only kind of frame the cache may hold.
        _cache_run_open(context.scene, int(s.bake_start))
        _bake_range['start'] = s.bake_start
        _bake_range['end'] = s.bake_end
        _simulation_frame_state['last_solved'] = max(
            1, int(s.bake_start) - 1)
        self._frame = s.bake_start
        return True

    def _step_frame(self, context, frame):
        helper = context.scene.gpu_cloth_helper
        _cache_playback_guard['active'] = True
        try:
            context.scene.frame_set(frame)
        finally:
            _cache_playback_guard['active'] = False
        if int(frame) < max(2, int(helper.bake_start)):
            # Frame 1 is the rest state and is never solved; the frame path's own
            # rule for where a range starts is `range_first = max(2, bake_start)`
            # (see the frame-change handler, `_frame_change_handler`).  A bake whose
            # range starts at 1 therefore has nothing to step there, and the frame
            # it would have to find in the cache cannot exist - the step operator
            # returns without solving below frame 2 (`GPUCloth_UpdateSimulation.
            # execute`).  Requiring it anyway cancelled *every* bake of the default
            # range with "Cache write failed at frame 1", because `bake_start`'s own
            # RNA default is 1 (`properties.py`), so `is_baked` could never become
            # true and the Cache panel never reached the state that offers playback
            # from cache, the exports, or Clear Cache.  Measured on the shipped
            # build: `bpy.ops.gpucloth.bake_simulation()` -> RuntimeError: Error:
            # Cache write failed at frame 1.
            return True
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

# The store's own file names on disk.  ``GPUCloth_v3_cache_clear`` owns this set
# for a live cache owner - ``Cache_v3_clear`` (cache.cu:582-627) deletes every
# ``frame_*.bin`` and the status metadata - and these are the same names, because
# the states the owner reports are the ones no live owner can be asked about: a
# store left by an earlier run, under a path this session never configured, or a
# session whose cache owner was already released.  A second name for the same
# file would be a second idea of what the cache is.
_CACHE_FRAME_PREFIX = "frame_"
_CACHE_FRAME_SUFFIX = ".bin"
_CACHE_STATUS_FILE = "gpucloth_cache_status.bin"
_CACHE_STATUS_TEMPORARY = "gpucloth_cache_status.bin.tmp"


def _cache_store_files(scene):
    """The files the active cache store owns on disk, and where they are.

    Returns ``(path, files)``.  ``path`` is the path the scene's cache settings
    resolve to, or None when they resolve to nothing (an empty ``cache_dir`` on a
    document that was never saved, or settings the ABI rejects).  ``files`` are
    the store's own files in it: the frame payloads and the status metadata,
    including the temporary the atomic status write leaves behind when it is
    interrupted.  Anything else in the directory belongs to someone else and is
    never listed here, so the count this returns is the count a delete may take.
    """
    try:
        path = _active_cache_path(scene)
    except (OSError, RuntimeError):
        return None, []
    if not path or not os.path.isdir(path):
        return (path or None), []
    files = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                name = entry.name
                if name in (_CACHE_STATUS_FILE, _CACHE_STATUS_TEMPORARY):
                    files.append(entry.path)
                elif (name.startswith(_CACHE_FRAME_PREFIX)
                        and name.endswith(_CACHE_FRAME_SUFFIX)):
                    files.append(entry.path)
    except OSError:
        return path, []
    return path, sorted(files)


def cache_store_summary(scene):
    """What Clear Cache would delete for ``scene``, and where it lives.

    One owner for the question the panel and the operator both ask, because
    defect 7 was two different answers to it: the panel offered the button only
    for ``is_baked``, and the operator's poll asked the native status only.  A
    live session keeps its frames in the native owner *and* its reached states in
    ``_simulation_frame_state`` while reporting ``is_baked False``,
    ``cached_frame_count 0`` for both - the state the owner measured - and a
    store from an earlier run exists as nothing but files.  Reading all three
    sources here is what lets the row appear exactly when one of them is
    non-empty, and never as a bare button over nothing.

    Never raises: a panel draw calls it, and the panel's poll is a poll.
    """
    helper = getattr(scene, "gpu_cloth_helper", None)
    native_frames = int(getattr(helper, "cached_frame_count", 0) or 0)
    flags = tuple(
        name for name in ("is_baked", "is_outdated", "is_frame_skip")
        if bool(getattr(helper, name, False)))
    try:
        retained = len(_simulation_frame_state['positions'])
    except (AttributeError, KeyError, TypeError):
        retained = 0
    path = None
    files = []
    if not bool(getattr(helper, "use_external_cache", False)):
        # An external cache is read-only for this add-on (the native clear
        # refuses that storage mode), so its files are not this row's to delete.
        path, files = _cache_store_files(scene)
    disk_bytes = 0
    for name in files:
        try:
            disk_bytes += os.path.getsize(name)
        except OSError:
            pass
    return {
        "path": path,
        "disk_files": len(files),
        "disk_bytes": disk_bytes,
        "native_frames": native_frames,
        "flags": flags,
        "retained_states": retained,
        "has_store": bool(files or native_frames or flags or retained),
    }


def _delete_cache_store(scene):
    """Delete the store's own files, and report what was removed and refused.

    The native clear owns this for a live owner; this is the same file set for
    the states it cannot be asked about.  Only files ``_cache_store_files``
    lists are touched - the directory is not swept - and the directory itself is
    removed only when the store created it for its own identity
    (``cache_<index>_<hash>``) and it is left empty: the configured root can hold
    other caches and is never removed.
    """
    path, files = _cache_store_files(scene)
    removed = 0
    removed_bytes = 0
    refusals = []
    for name in files:
        try:
            size = os.path.getsize(name)
        except OSError:
            size = 0
        try:
            os.remove(name)
        except OSError as exc:
            refusals.append(f"{os.path.basename(name)}: {exc}")
            continue
        removed += 1
        removed_bytes += size
    if (path and removed and not refusals
            and os.path.basename(path).startswith("cache_")):
        try:
            if not os.listdir(path):
                os.rmdir(path)
        except OSError as exc:
            refusals.append(f"{path}: {exc}")
    return removed, removed_bytes, refusals


class GPUCloth_FreeCache(bpy.types.Operator):
    """Удалить кэш симуляции: кадры на диске, состояние владельца и прошлое сессии"""
    bl_idname = "gpucloth.free_cache"
    bl_label  = "Очистить кэш"

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        if scene is None or not hasattr(scene, "gpu_cloth_helper"):
            return False
        # Offered exactly when there is something to delete - the frames on
        # disk, the native owner's frames, or the reached states of this session
        # - and never as a bare button over an empty store.
        return bool(cache_store_summary(scene)["has_store"])

    def execute(self, context):
        scene     = context.scene
        s         = scene.gpu_cloth_helper

        # The native owner is asked while it still exists: only it can free the
        # frames it holds in memory, and a refusal there is reported as it always
        # was rather than papered over by the disk delete below.
        if g_dll is not None and _cache_handle_value():
            if int(g_dll.GPUCloth_v3_cache_clear(
                    g_runtime_handle, _cache_handle_owner())) != CType.GPUCLOTH_ABI_OK:
                self.report({'ERROR'}, "Не удалось очистить кэш")
                return {'CANCELLED'}

        # The past this session reached is part of what the owner calls the
        # cache: it is what a rewind is served from when the native store misses
        # (`_load_simulation_frame`), so a clear that left it would hand back
        # geometry the button had just deleted.
        _clear_retained_frames()
        removed, removed_bytes, refusals = _delete_cache_store(scene)

        s.bake_progress = 0
        s.playback_mode = False
        try:
            _sync_cache_status(scene)
        except (OSError, RuntimeError) as exc:
            # The owners are gone or the store was just deleted, so the cached
            # status is not readable any more: state that as the empty store it
            # is instead of leaving the panel's last flags standing.
            s.is_baked = False
            s.is_outdated = False
            s.is_frame_skip = False
            s.cached_frame_count = 0
            # A refresh that cannot read the owner is not a failed clear.  The
            # clear is judged by whether anything is left, and this branch is
            # reached in a state that has nothing left: the row was offered while
            # the session still held reached states (`cache_store_summary`), the
            # past is dropped above, and the native owner is gone because the
            # store it held is what was just deleted.  Measured on the integrated
            # tree: after an invalidation dropped the store, poll() was true
            # through those reached states and this branch turned a finished clear
            # into an error dialog (`v3 cache owner is not live`; receipts
            # build/r23-acceptance/scenario-acc-live.json and -acc-pause.json).
            # A store that is still there keeps the failure it earned.
            if not bool(cache_store_summary(scene)["has_store"]):
                self.report(
                    {'INFO'},
                    "Кэш очищен: владелец кэша уже освобождён, "
                    "удалять больше нечего.")
                return {'FINISHED'}
            self.report({'ERROR'}, f"Cache status refresh failed: {exc}")
            return {'CANCELLED'}

        if refusals:
            self.report(
                {'ERROR'},
                "Cache files could not be deleted: " + ", ".join(refusals[:3]))
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Кэш очищен: удалено файлов {removed} "
            f"({removed_bytes / (1024.0 * 1024.0):.2f} МиБ).")
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
            for i, cloth_obj in enumerate(_live_cloth_objects()):
                cached = _cached_frame_positions(
                    scene, frame, cloth_obj, cache_dir_bytes)
                if cached is not None:
                    flat = np.frombuffer(cached, dtype=np.float32)
                    cloth_obj.data.vertices.foreach_set("co", flat)
                    cloth_obj.data.update()
                    updated = True
            if updated:
                for cloth_obj in _live_cloth_objects():
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

# ── C++ TestScene scene registry (authoritative parameters) ────────────────
# enum class SceneType {
#     DrapeOnSphere = 0, TwistTest = 1, MultiLayerDrop = 2,
#     CushionDrop = 3, CapeProject = 4, MDHorizontalContact = 5 };
#     src/engine/TestScene/TestScene.h:159
# static const char* kSceneNames[6] = {...};
#     src/engine/TestScene/TestScene_UI.cpp:1321
#
# Every numeric scene parameter below cites its C++ definition.  Solver type,
# substep/iteration counts and frame timing deliberately stay on the add-on
# defaults shared by all six operators; only scene authoring is mirrored.
TESTSCENE_CLOTH_NX = 128            # TestScene.h:443-444 m_clothNX = m_clothNY
TESTSCENE_GRID_QUADS = TESTSCENE_CLOTH_NX - 1   # TestScene.cpp:2287 NX=m_clothNX-1
TESTSCENE_CLOTH_HALF_SIZE = 3.0     # TestScene.h:54  CLOTH_HALF_SIZE
TESTSCENE_INIT_Z = 4.0              # TestScene.cpp:2289 initZ
TESTSCENE_TWIST_INIT_Z = 0.0        # TestScene.cpp:2289 TwistTest initZ
TESTSCENE_SPHERE_RADIUS = 1.8       # TestScene.h:55  SPHERE_RADIUS
TESTSCENE_SPHERE_CZ = 0.5           # TestScene.h:58  SPHERE_CZ
TESTSCENE_SPHERE_RINGS = 10         # TestScene.h:59  SPHERE_RINGS
TESTSCENE_SPHERE_SECTORS = 12       # TestScene.h:60  SPHERE_SECTORS
# The Drape scene's collider sphere is deliberately denser than TestScene.h:59-60
# mirror, and this is the one place the mirror is overridden on purpose: the
# owner subdivided the sphere by two in his own scene ("я применил subdivision
# surface на 2") and asked for that density to be what the scene builder produces
# and what Prepare is measured against, with the target relaxed from 500 ms to
# 1 s to match the heavier collider.  Kept as its own override rather than folded
# into the two numbers above, so the mirror of the C++ fixture stays a mirror.
TESTSCENE_SPHERE_SUBDIV = 2
TESTSCENE_CYLINDER_RADIUS = 1.5     # TestScene.h:64  CYLINDER_RADIUS
TESTSCENE_CYLINDER_HALF_LEN = 4.5   # TestScene.h:65  CYLINDER_HALF_LEN
TESTSCENE_CYLINDER_CZ = 0.3         # TestScene.h:68  CYLINDER_CZ
TESTSCENE_CYLINDER_RINGS = 20       # TestScene.h:69  CYLINDER_RINGS
TESTSCENE_CYLINDER_STACKS = 10      # TestScene.h:70  CYLINDER_STACKS
TESTSCENE_FLOOR_Z = -2.0            # TestScene.h:71  FLOOR_Z
TESTSCENE_FLOOR_HALF = 12.0         # TestScene.h:72  FLOOR_HALF
TESTSCENE_MULTILAYER_BASE_Z = 4.0   # TestScene.h:74  MULTILAYER_BASE_Z
TESTSCENE_MULTILAYER_LAYER_DZ = 0.15  # TestScene.h:75 MULTILAYER_LAYER_DZ
TESTSCENE_MULTILAYER_COUNT = 2      # TestScene.h:490 m_layerCount{2}
TESTSCENE_CUSHION_DOME_H = TESTSCENE_CLOTH_HALF_SIZE * 0.25   # TestScene.h:81
TESTSCENE_CUSHION_INIT_SEP = 0.02   # TestScene.h:82  CUSHION_INIT_SEP
TESTSCENE_CUSHION_INIT_Z = (        # TestScene.h:83  CUSHION_INIT_Z
    TESTSCENE_FLOOR_Z + TESTSCENE_CUSHION_DOME_H
    + TESTSCENE_CUSHION_INIT_SEP * 0.5 + 2.0)
TESTSCENE_CUSHION_PRESSURE_RATIO = 1.3   # TestScene.cpp:2634 pressure.ratio
TESTSCENE_GRAVITY_Z = -9.81         # TestScene.cpp:1908 gravity.z (m/s^2)
TESTSCENE_CAPE_CONTACT_THICKNESS = 0.003    # TestScene.cpp:2532
TESTSCENE_MD_CONTACT_THICKNESS = 0.0025     # TestScene.cpp:2532
TESTSCENE_CONTACT_FRICTION = 0.5            # TestScene.cpp:2535
TESTSCENE_COLLISION_LOOPS = 4               # TestScene.cpp:2541
TESTSCENE_CAPE_OGC_RADIUS_MM = 3.0          # TestScene.cpp:5099
TESTSCENE_CAPE_OGC_KC = 1000.0              # TestScene.cpp:5100
TESTSCENE_CAPE_OGC_FRICTION = 0.3           # TestScene.cpp:5101
# MD fixture guard: TestScene.cpp:421-422.
TESTSCENE_MD_EXPECTED = {
    "nv": 9149, "nt": 17994, "nseam": 0, "npin": 0,
    "body_nv": 4, "body_nt": 2,
}
# OGC auto sizing for generated grid scenes, TestScene_UI.cpp:2531-2576:
#   r_m = clamp(radius_frac * avg_spring_len, 0.002, 0.08)
#   kc  = clamp(20 / r_m, 10, 100000)
TESTSCENE_MULTILAYER_OGC_RADIUS_FRAC = 0.60   # TestScene_UI.cpp:2542
TESTSCENE_OGC_FRICTION = 0.3                  # TestScene_UI.cpp:2567
TESTSCENE_OGC_GAMMA_P = 0.45                  # TestScene_UI.cpp:2425

# Cloth/material parameters, TestScene.cpp:1870-1907 / 1433-1448.  They are
# applied explicitly so a preset (PD/COTTON would raise tension to 30) or a
# future RNA default change cannot drift away from the C++ scene.
#
# `vel_damping` is the one deliberate exception: the shipped default is 1.0
# (properties.py) and this mirror carries that shipped value rather than the
# C++ scene's 0.0 (TestScene.cpp:1896).  Measured on this route, the 0.0
# operating point left the Drape-on-Sphere sheet creeping at L-inf 0.0273
# m/frame, 27x the 0.001 auto-stop tolerance, so the drape sandbox never filled
# its 8-step window and Settle ended NOT_CONVERGED at maximum_position_delta
# 0.0344; at 1.0 the step-matched residual falls to 0.0097 m/frame (tail kinetic
# energy 100.8 -> 16.4, probe units) and stops growing, but that 0.001 window
# still does not fill on this preset.  The native TestScene default was left
# alone on purpose - no gate, probe or comparison reads this parameter from
# either default (there is no TESTSCENE_CLOTH_VEL_DAMPING override, the
# ClothGeometryTests fixtures take vel_damping from the DNA defaults, and every
# tool that touches it sets it explicitly), while the 23 Full-tier TestScene
# scenes do run at that native default, so moving it would re-baseline their
# trajectories with no oracle.  A Blender-vs-native comparison on this parameter
# must therefore pin the value.
TESTSCENE_MATERIAL = {
    "vertex_mass": 0.3,             # RegisterParam("mass", 0.3f)
    "tension": 15.0,                # cloth.tension
    "compression": 15.0,            # cloth.compression
    "shear": 5.0,                   # cloth.shear
    "bending_stiffness": 0.5,       # cloth.bending
    "tension_damp": 5.0,            # cloth.tension_damp
    "compression_damp": 5.0,        # cloth.compression_damp
    "shear_damp": 5.0,              # cloth.shear_damp
    "bending_damping": 0.5,         # cloth.bending_damp
    "vel_damping": 1.0,             # cloth.vel_damping; see the deviation note above
    "max_tension": 500.0,           # cloth.max_tension
    "max_compression": 500.0,       # cloth.max_compression
    "max_shear": 500.0,             # cloth.max_shear
    "max_bend": 100.0,              # cloth.max_bend
}


# The owner's accepted solver effort for the two acceptance scenes.
#
# `quality_step` (substeps per frame) and `solver_krylov_iterations` (the PD
# global solve's update ceiling) are the two knobs that decide what a frame
# costs.  On this route the shipped defaults are 5 and 240, and at 5 substeps
# x 240 updates the Drape On Sphere scene costs 63.7 ms of operator time and
# 49.7 ms of native solve per frame.  The owner's call - the one decision in
# this project that is explicitly his - is that these two scenes run one
# substep per frame with a 50-update Krylov ceiling and accept the measured
# quality cost of that, which takes the same scene to 23.4-25.8 ms.
#
# Measured on the shipped Product DLL, Drape On Sphere, PD/SDB/OGC, 12
# measured frames after 3 warm-up, quiet machine: the native solve falls
# 49.67 -> 9.94 ms and the `cloth_step` stage with it (58.1 -> 19.1 ms), while
# every host stage is unchanged; the worst native spring moves from 1.75x to
# 5.08x its rest length and the springs outside the solver's own strain
# annulus [0.90, 1.03] from 193 of 48 641 to 1402; and every Krylov system now
# exits on the update budget (budget = hard_cap = 50, updates_sum = 50 on
# every frame) instead of on the convergence criterion (CONVERGED at 240).
# Cushion Drop goes 86.6 -> 44.8 ms and stays healthy on both arms.  The full
# stage split and every receipt are in build/perf-achievement-33ms/REPORT.md.
#
# Stated here, and applied only by those two builders, for two reasons:
#
#   * the same reason `TESTSCENE_MATERIAL` is stated: a mirror of a measured
#     scene must own the values it is measured at instead of inheriting them.
#     `_setup_testscene_cloth` leaves both fields at whatever properties.py
#     ships, so without this the scenes would silently follow a panel default;
#   * the accepted cost was measured on these two scenes and nowhere else.
#     `quality_step` is one property shared with Mil2, a route never measured
#     at this operating point, and `solver_krylov_iterations` reaches every PD
#     cloth a user configures, so moving either *shipped default* would spend
#     the owner's accepted cost on routes and users his decision does not
#     cover.  The other TestScene builders keep the shipped defaults for the
#     same reason: no quality cost has been measured for them.
#
# Both fields are published through `GPUClothQualityConfig` (`quality_steps`
# -> `stepsPerFrame`, `solver_krylov_iterations` -> `cfg->krylov_update_cap`),
# so a caller that wants the shipped ceiling back sets the two properties
# after the operator runs - which is what the drape shape acceptance arm does
# with its own `--krylov 240`.
TESTSCENE_ACCEPTANCE_QUALITY_STEPS = 1
TESTSCENE_ACCEPTANCE_KRYLOV_ITERATIONS = 50


def _testscene_avg_edge(half_size, quads):
    """Mean structural rest length of a uniform grid.

    ``avg_spring_len`` is the arithmetic mean over structural springs
    (src/engine/source/kernel/intern/cloth.cu:3461-3483); on a regular grid
    every structural spring is one cell long, so the mean is exact.
    """
    return 2.0 * half_size / float(quads)


def _testscene_ogc_auto(avg_edge, radius_frac):
    """OGC auto radius/stiffness for generated grid scenes (radii in mm)."""
    r_m = max(0.002, min(radius_frac * avg_edge, 0.08))
    return r_m * 1000.0, max(10.0, min(20.0 / r_m, 100000.0))


def _setup_testscene_cloth(obj, solver='PD'):
    """Use TestScene stiffness with area-based fabric mass in Blender."""
    _setup_cloth(obj, solver=solver, material='CUSTOM')
    for prop_name, value in TESTSCENE_MATERIAL.items():
        setattr(obj.GPUCloth, prop_name, value)
    # Imported MD scene density contract: 0.300 kg/m^2. Mesh refinement must
    # change per-vertex mass, not multiply the weight of the fabric.
    obj.GPUCloth.mass_mode = 'AREAL'
    obj.GPUCloth.fabric_density = 300.0
    obj.GPUCloth.use_object_collision = True   # TestScene.h:387
    return obj


def _apply_acceptance_solver_effort(obj):
    """Apply the owner's accepted solver effort to an acceptance scene.

    Called by the two builders the owner measured - Drape On Sphere and
    Cushion Drop - after `_setup_testscene_cloth` has applied the material
    mirror and before any prepare reads the fields, so the values travel the
    scene's own `GPUClothQualityConfig` rather than a default.  See
    `TESTSCENE_ACCEPTANCE_QUALITY_STEPS` for the decision and its cost.
    """
    obj.GPUCloth.quality_step = TESTSCENE_ACCEPTANCE_QUALITY_STEPS
    obj.GPUCloth.solver_krylov_iterations = (
        TESTSCENE_ACCEPTANCE_KRYLOV_ITERATIONS)
    return obj


def _set_scene_gravity(context, x=0.0, y=0.0, z=TESTSCENE_GRAVITY_Z):
    helper = context.scene.gpu_cloth_helper
    helper.gravity_x = x
    helper.gravity_y = y
    helper.gravity_z = z


def _imported_cloth_edges(asset):
    """Mesh edge list for an imported cloth asset, with seams appended.

    The asset edge tail reproduces the native ``medge`` array; each seam pair
    becomes one extra loose edge, which the add-on uploads as a sewing record
    with rest length 0 (``_upload_sewing``) — exactly the explicit seam spring
    the C++ builder appends (TestScene.cpp:2570-2580: ij/kl pair, restlen
    CAPE_SEAM_WELD_LENGTH_M).
    """
    edges = []
    seen = set()
    for index in range(0, len(asset["edges"]), 2):
        key = (int(asset["edges"][index]), int(asset["edges"][index + 1]))
        edges.append(key)
        seen.add(key)
    for index in range(0, len(asset["seams"]), 2):
        a, b = int(asset["seams"][index]), int(asset["seams"][index + 1])
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    return edges


def _closed_mesh_volume(mesh):
    """|signed volume| of a closed mesh (TestScene.cpp:2614-2629).

    ``V0`` is the pressure reference volume; the C++ builder sums the signed
    tetrahedron volumes and negates the result when it comes out negative.
    """
    total = 0.0
    for triangle in vcu.calc_mesh_loop_triangles(mesh):
        i0, i1, i2 = (int(index) for index in triangle.vertices)
        p0 = mesh.vertices[i0].co
        p1 = mesh.vertices[i1].co
        p2 = mesh.vertices[i2].co
        total += (p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
                  + p0[1] * (p1[2] * p2[0] - p1[0] * p2[2])
                  + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0])) / 6.0
    return abs(total)


def _grid_verts_faces(nx, ny, half_size, height):
    """Shared grid topology; one layer at height. Returns (verts, faces)."""
    sx = nx + 1
    verts = []
    for row in range(ny + 1):
        for col in range(nx + 1):
            x = -half_size + 2.0 * half_size * col / nx
            y = -half_size + 2.0 * half_size * row / ny
            verts.append((x, y, height))

    faces = []
    for row in range(ny):
        for col in range(nx):
            i = row * sx + col
            faces.append((i, i + 1, i + sx + 1, i + sx))
    return verts, faces


def _ensure_generated_material_uv(
        mesh_data, name="GPUClothMaterial", grid_uv=False):
    """Assign deterministic per-corner local UVs to generated cloth meshes."""
    layer = mesh_data.uv_layers.get(name) or mesh_data.uv_layers.new(name=name)
    if grid_uv:
        xs = [float(vertex.co.x) for vertex in mesh_data.vertices]
        ys = [float(vertex.co.y) for vertex in mesh_data.vertices]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax > xmin and ymax > ymin:
            for loop in mesh_data.loops:
                point = mesh_data.vertices[loop.vertex_index].co
                layer.data[loop.index].uv = (
                    (float(point.x) - xmin) / (xmax - xmin),
                    (float(point.y) - ymin) / (ymax - ymin))
            return layer
    for polygon in mesh_data.polygons:
        corners = tuple(polygon.loop_indices)
        if len(corners) < 3:
            continue
        origin = mesh_data.vertices[mesh_data.loops[corners[0]].vertex_index].co
        p1 = mesh_data.vertices[mesh_data.loops[corners[1]].vertex_index].co
        p2 = mesh_data.vertices[mesh_data.loops[corners[2]].vertex_index].co
        u = (p1 - origin)
        if u.length <= 1.0e-12:
            continue
        u.normalize()
        normal = (p1 - origin).cross(p2 - origin)
        if normal.length <= 1.0e-12:
            continue
        normal.normalize()
        v = normal.cross(u)
        for loop_index in corners:
            point = mesh_data.vertices[mesh_data.loops[loop_index].vertex_index].co
            delta = point - origin
            layer.data[loop_index].uv = (delta.dot(u), delta.dot(v))
    return layer


def _make_grid_mesh(name, nx, ny, half_size, height, pin_corners=False):
    """Create a subdivided grid mesh and return (obj, mesh_data)."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    verts, faces = _grid_verts_faces(nx, ny, half_size, height)

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    _ensure_generated_material_uv(mesh_data, grid_uv=True)

    if pin_corners:
        # Native DrapeOnSphere/TwistTest pin the two top corners
        # vi(0, NY) + vi(NX, NY) with goal 1. Same corners here.
        sx = nx + 1
        pin_group = obj.vertex_groups.new(name="Pin")
        pin_group.add([(ny * sx) + 0, (ny * sx) + nx], 1.0, 'REPLACE')

    return obj, mesh_data


def _make_multilayer_mesh(name, nx, ny, half_size, base_z, dz, layers):
    """Stack N grid layers base_z + k*dz into one mesh, native layout."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    layer_verts, layer_faces = _grid_verts_faces(nx, ny, half_size, base_z)
    per_layer = len(layer_verts)
    verts = []
    faces = []
    for k in range(layers):
        z = base_z + k * dz
        verts.extend((x, y, z) for x, y, _ in layer_verts)
        base = k * per_layer
        faces.extend((a + base, b + base, c + base, d + base)
                     for a, b, c, d in layer_faces)

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    _ensure_generated_material_uv(mesh_data, grid_uv=True)
    return obj, mesh_data


def _subdivide_collider(obj, levels):
    """Subdivide a collider mesh in place, so its density is the real one.

    Applied instead of left as a modifier: the collider's triangle count is the
    cost this scene exists to measure, and the native side builds its BVH and
    runs its clearance audit on the *evaluated* mesh, which would make the count
    a function of viewport versus render settings rather than of the scene.

    ``levels`` of 0 leaves the primitive exactly as Blender created it, which is
    what the C++ TestScene fixture mirrors; the Drape scene asks for more
    (``TESTSCENE_SPHERE_SUBDIV``).
    """
    if levels <= 0:
        return obj
    modifier = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    modifier.subdivision_type = 'CATMULL_CLARK'
    modifier.levels = levels
    modifier.render_levels = levels
    bpy.context.view_layer.objects.active = obj
    for candidate in bpy.context.view_layer.objects:
        candidate.select_set(candidate is obj)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.update()
    return obj


def _make_uv_sphere(name, radius, cx, cy, cz, rings=10, sectors=12):
    """Create a UV sphere collision object and return it.

    Uses the Blender primitive: hand-rolled grids duplicate pole verts,
    which makes zero-area fan triangles that the native hard preflight
    (DEGENERATE_TRIANGLE) correctly rejects.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=sectors, ring_count=rings, radius=radius,
        location=(cx, cy, cz))
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = name + "_mesh"
    return obj


def _make_cylinder_floor(name, radius, half_len, cx, cy, cz, floor_z, floor_half,
                         rings=TESTSCENE_CYLINDER_RINGS,
                         stacks=TESTSCENE_CYLINDER_STACKS):
    """Create a cylinder + floor collision object (native CollisionCylinder)."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    verts = []
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
            c = (s + 1) * w + r
            d = (s + 1) * w + r + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    body_verts = len(verts) - 4
    f0, f1, f2, f3 = body_verts, body_verts + 1, body_verts + 2, body_verts + 3
    faces.append((f0, f1, f2))
    faces.append((f0, f2, f3))

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    _ensure_generated_material_uv(mesh_data)
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
                face = (i, i + 1, i + sx + 1, i + sx)
                faces.append(face[::-1] if sh == 0 else face)

    # Side walls closing the volume, native CushionMesh layout: quads join
    # the boundary loops of both sheets so pressure sees a closed mesh.
    def _gvi(sh, col, row):
        return sh * sx * sy + row * sx + col

    for col in range(nx):
        faces.append((_gvi(0, col, 0), _gvi(0, col + 1, 0),
                      _gvi(1, col + 1, 0), _gvi(1, col, 0)))
        faces.append((_gvi(0, col + 1, ny), _gvi(0, col, ny),
                      _gvi(1, col, ny), _gvi(1, col + 1, ny)))
    for row in range(ny):
        faces.append((_gvi(0, 0, row + 1), _gvi(0, 0, row),
                      _gvi(1, 0, row), _gvi(1, 0, row + 1)))
        faces.append((_gvi(0, nx, row), _gvi(0, nx, row + 1),
                      _gvi(1, nx, row + 1), _gvi(1, nx, row)))

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    return obj


# ── Sewn cushion: two convex panels joined by perimeter sewing springs ──────
# The owner asked for a second cushion in the same scene whose two convex
# panels are *stitched* at the rim instead of welded, so the seam path is
# exercised too ("вторая сшитые по краям 2 выпуклые плоскости").  This is the
# MD seam mechanism, and on this product it is carried by the native sewing
# springs (`use_sewing_springs`), not by the constraint network: the two
# mechanisms are mutually exclusive (`_capture_constraint_network`).
#
# Density is this builder's own choice, not a TestScene.h mirror: the scene has
# no sewn cushion to mirror, and 128 quads for a second cloth would double the
# scene to ~33 k vertices.  32 quads resolves the same dome the welded cushion
# uses at 1/16 the solver cost, which is what a seam oracle needs.
TESTSCENE_SEWN_CUSHION_QUADS = 32
# Side-by-side in x so the two cushions cannot contact each other; the welded
# cushion stays exactly where it was, at x = 0.  Both footprints are 6 m wide,
# so 6.5 m leaves a 0.5 m gap at rest and stays inside the 12 m half-floor.
TESTSCENE_SEWN_CUSHION_CX = 6.5


def _make_sewn_cushion_mesh(
        name, n, half_size, cx, cz, dome_height):
    """Two convex panels whose rims are separate but coincide in space.

    Returns ``(obj, seam_pairs)`` where ``seam_pairs`` is the ordered list of
    ``(panel_a_rim_index, panel_b_rim_index)`` pairs, one per rim vertex, in
    the same order the rim runs around the perimeter.

    Layout, mirroring the welded cushion's two-sheet convention:

    * panel A (``sh == 0``) is the *lower* cap and bulges **down** from the
      shared rim plane ``z = cz``;
    * panel B (``sh == 1``) is the *upper* cap and bulges **up**.

    Both caps are triangulated with the *same* cell layout, but the upper one
    takes the opposite quad diagonal and is wound outward on it, so the closed
    shell is consistently oriented and the two panels traverse the shared rim
    edge in opposite directions.  Both halves of that are mandatory, not
    cosmetic; the ``faces`` comment below states which check each one satisfies.
    With pressure enabled the native shell-closure test runs on the *seam
    quotient* - the union-find classes of every zero-rest-length
    ``CLOTH_SPRING_TYPE_SEWING`` spring (``main.cpp:1747-1826``) - and
    ``main.cpp:1817`` fails preparation with
    ``GPUCLOTH_INVARIANT_INCONSISTENT_WINDING`` when two triangles traverse one
    quotient edge in the *same* direction.  The quotient is also what makes the
    sewn shell read as *closed*: without it every rim edge would be used once
    and preparation would raise ``GPUCLOTH_INVARIANT_PRESSURE_OPEN_SHELL``.

    The rim edges themselves stay in the triangle list (the two panels are not
    welded), so the native reference volume ``g_ensure_pressure`` sums is the
    full closed shell (``main.cpp:5982-6030``).

    KNOWN LIMITATION, measured on the real binary.  Because the two rim loops
    are separate vertices, every rim triangle edge is used *once* rather than
    twice: this mesh's own histogram is ``{1: 256, 2: 6016}``.  Preparation
    accepts that (it tests the seam quotient above), but the **run-time**
    pressure owner does not - ``g_ensure_pressure`` (``main.cpp:5988-6060``)
    runs its own closure test on raw triangle edges, so the first solved frame
    is rejected with ``GPUCLOTH_ABI_SOLVE_FAILED`` and the diagnosis
    ``GPUCLOTH_DIAGNOSTICS_ERROR_PRESSURE_STATE``.  The cloth then never moves
    at all.  Verified by control: the same domes with the rim vertices *shared*
    (one welded shell, identical triangle count) steps cleanly, and the same
    sewn mesh with ``use_pressure`` off moves normally.  So the sewn cushion is
    built and prepared exactly as asked, and it is the engine's run-time
    pressure gate - not this builder - that stops it inflating.
    """
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    sx = n + 1
    verts = []
    for sh in range(2):
        for row in range(sx):
            for col in range(sx):
                pu = col / n
                pv = row / n
                # sin(pi*pu)*sin(pi*pv) vanishes on all four edges, so the rim
                # is the flat rectangle z = cz and the two caps share every rim
                # vertex *position* exactly - the vertices are duplicated, not
                # welded, which is what leaves a seam for the sewing springs.
                dome = (math.sin(math.pi * pu) * math.sin(math.pi * pv)
                        * dome_height)
                verts.append((
                    cx - half_size + 2.0 * half_size * pu,
                    -half_size + 2.0 * half_size * pv,
                    (cz - dome) if sh == 0 else (cz + dome)))

    def _gvi(sh, col, row):
        return sh * sx * sx + row * sx + col

    # One rim walk: bottom row left->right in x, up the right column, back
    # along the top row, down the left column.  Rim vertex i of one panel is
    # stitched to rim vertex i of the other.
    rim = [_gvi(0, col, 0) for col in range(sx)]
    rim += [_gvi(0, n, row) for row in range(1, sx)]
    rim += [_gvi(0, col, n) for col in range(n - 1, -1, -1)]
    rim += [_gvi(0, 0, row) for row in range(n - 1, 0, -1)]
    seam_pairs = [(index, index + sx * sx) for index in rim]

    # Both panels are wound outward, and each is a proper triangulation of its
    # quads - but on *opposite* diagonals:
    #
    #   lower panel: (i, i+sx+1)      upper panel: (i+1, i+sx)
    #
    # Three properties are needed together and only this combination has all
    # three; each was measured with ``tools/perf/cushion_seam_mesh_oracle.py``,
    # which replays ``main.cpp:1784-1826`` on this exact mesh:
    #
    # * the shell must close.  With pressure enabled the closure test runs on
    #   the seam quotient, and a cell diagonal that lands on the *other*
    #   panel's rim edge makes some quotient edge used four times instead of
    #   twice => PRESSURE_OPEN_SHELL.  Opposite diagonals are what prevent it:
    #   the rim-adjacent cell of one panel then never spans the corner the
    #   other panel's rim edge uses;
    # * the two panels must traverse the shared rim edge in *opposite*
    #   directions, or ``main.cpp:1817`` fails preparation with
    #   GPUCLOTH_INVARIANT_INCONSISTENT_WINDING;
    # * the shell must be oriented outward so the signed volume is +V and not
    #   -V, which is what makes ``target_volume = 1.3 * V0`` the ratio the
    #   owner asked for rather than a ratio about a negative V0.
    #
    # Simplifying either panel to match the other breaks one of these; the
    # oracle rejects the naive forms on a 2x2 panel before any solver step.
    faces = []
    for row in range(n):
        for col in range(n):
            i = _gvi(0, col, row)
            faces.append((i, i + sx, i + sx + 1))
            faces.append((i, i + sx + 1, i + 1))
    for row in range(n):
        for col in range(n):
            i = _gvi(1, col, row)
            faces.append((i, i + 1, i + sx))
            faces.append((i + 1, i + sx + 1, i + sx))

    # Loose edges ARE the seam mechanism: `_upload_sewing` turns every loose
    # edge of the evaluated mesh into one GPUClothSewingRecord with
    # stiffness 1.0 / rest_length 0.0, and a zero rest length is exactly what
    # `cloth_constraint_has_zero_rest` needs to unite the two rim classes.
    edges = [(a, b) for a, b in seam_pairs]
    mesh_data.from_pydata(verts, edges, faces)
    mesh_data.update()
    _ensure_generated_material_uv(mesh_data)
    return obj, seam_pairs


def _make_two_sided_collider(obj):
    """Explicit two-sided surface contract for scene collider builders.

    Matches the Smoke fixture (tools/blender_product_gate.py): product
    preflight requires an explicit surface contract, while Blender 4.2
    defaults (culling on) resolve to ONE_SIDED_NORMAL.
    """
    obj.collision.use_culling = False
    obj.collision.use_normal = False


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

        # TestScene.cpp:2540-2542 grid = ClothGrid(NX, NY, CLOTH_HALF_SIZE,
        # CLOTH_HALF_SIZE, initZ) at 128x128 verts; pins vi(0,NY)/vi(NX,NY)
        # at TestScene.cpp:2646-2656.
        cloth_obj, _ = _make_grid_mesh(
            "DrapeCloth", TESTSCENE_GRID_QUADS, TESTSCENE_GRID_QUADS,
            TESTSCENE_CLOTH_HALF_SIZE, TESTSCENE_INIT_Z, pin_corners=True)
        sphere_obj = _make_uv_sphere(
            "CollisionSphere", TESTSCENE_SPHERE_RADIUS, 0.0, 0.0,
            TESTSCENE_SPHERE_CZ, TESTSCENE_SPHERE_RINGS,
            TESTSCENE_SPHERE_SECTORS)
        # Before the Collision modifier is added: the subdivision is applied, so
        # the saved mesh is the dense one and nothing downstream sees a modifier
        # it would have to evaluate.
        _subdivide_collider(sphere_obj, TESTSCENE_SPHERE_SUBDIV)

        sphere_obj.modifiers.new(name="Collision", type='COLLISION')
        # The Drape sphere is a closed solid, and the one-sided contract is the only
        # one that can expel a vertex which ends up inside it: the exact FP64 plane
        # signed distance, the certified thickness and the joint active-plane set all
        # live on that path, and it is also the only path that counts external
        # surface crossings, which is what the in-frame closure repairs.  Measured on
        # this scene with the collider switched to ONE_SIDED: a cloth buried in the
        # sphere comes back out (inside 1454 -> 0 at frame 14, worst 0.0) and rests
        # ON it, while the two-sided contract only ever certifies the buried pose as
        # clear — `inside` 4088 -> 4619 with the collision response proving every
        # frame that nothing is wrong.  Two-sided is the thin-shell contract, and this
        # collider is not a shell.
        sphere_obj.collision.use_culling = True
        sphere_obj.collision.use_normal = False

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_testscene_cloth(cloth_obj, solver='PD')
        _apply_acceptance_solver_effort(cloth_obj)
        cloth_obj.GPUCloth.vgroup_mass = "Pin"

        _set_scene_gravity(context)

        self.report({'INFO'}, "DrapeOnSphere test scene created")
        return {'FINISHED'}


class GPUCloth_TestTwist(bpy.types.Operator):
    """Create TwistTest scene: cloth pinned at top corners"""
    bl_idname = "gpucloth.test_twist"
    bl_label = "Twist Test"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        # Native TwistTest pins BOTH corner pairs with goal 1
        # (TestScene.cpp:2735-2760): the top pair vi(0, simNY)/vi(simNX, simNY)
        # stays where it is, and the bottom pair vi(0, 0)/vi(simNX, 0) is the
        # rotating one, whose original positions are kept for the per-step
        # update.  `pin_corners=True` pins the top pair (`_make_grid_mesh`,
        # :13978-13983) and the `add` below pins the bottom pair, so the four
        # corners are the native four, all at weight 1.
        cloth_obj, _ = _make_grid_mesh(
            "TwistCloth", TESTSCENE_GRID_QUADS, TESTSCENE_GRID_QUADS,
            TESTSCENE_CLOTH_HALF_SIZE, TESTSCENE_TWIST_INIT_Z,
            pin_corners=True)
        pin_group = cloth_obj.vertex_groups.get("Pin")
        pin_group.add([0, TESTSCENE_GRID_QUADS], 1.0, 'REPLACE')

        # Native moves those two pins on every sim step (TestScene.cpp:3351-3408):
        #
        #   theta += twist.speed * SIM_FIXED_DT                  (:1856, :3356)
        #   C      = (p0 + p1) / 2, over the ORIGINAL positions  (:3035-3037)
        #   n      = normalize(cross(p1 - p0, (0, 0, 1)))        (:3043-3052)
        #   target = C + Rodrigues(v - C, n, theta)              (:3369-3400)
        #
        # and re-asserts it after the readback (:3788-3807).  The add-on's pin
        # owner reads pin targets off the *evaluated* mesh once per frame
        # (`capture_evaluated_pin_snapshot`, vertex_channels.py:366-399), so a
        # driven shape key is how the same target reaches the solver here.
        #
        # WHY THE KEYS LOOK LIKE THIS.  For this sheet d = v - C is perpendicular
        # to n, so Rodrigues collapses to `C + d cos(theta) + (n x d) sin(theta)`
        # - a plain non-negative blend of two fixed shapes, which is the only kind
        # available: a relative shape key's `value` is clamped to [0, 1], so a key
        # cannot SUBTRACT.  Measured on Blender 5.2.1 by assigning -0.5 and -1.0
        # to a key's value: both read back 0.0, through RNA and through a driver
        # alike.  The 0.5 offsets are what keep every weight in [1/8, 7/8], and
        # `TwistBias` carries the constant term they introduce.
        #
        # The x component of the bias is stored as the ABSOLUTE coordinate
        # -0.5*amplitude, not as a displacement of that size - Blender adds each
        # key's offset FROM THE BASIS, so storing the absolute value makes the
        # offset `-0.5A - x`, and that -x is exactly what cancels the basis term.
        # Written as a delta the same line gives `x (1 + cos(theta))`: 2x at
        # theta = 0, which is a collapse, not a rotation.  The z line needs no
        # such care only because every bottom vertex starts at z = 0.
        cloth_obj.shape_key_add(name="Basis")
        bias_key = cloth_obj.shape_key_add(name="TwistBias")
        cos_key = cloth_obj.shape_key_add(name="TwistCos")
        sin_key = cloth_obj.shape_key_add(name="TwistSin")
        bias_key.value = 1.0
        bottom_indices = (0, TESTSCENE_GRID_QUADS)
        for index in bottom_indices:
            source = cloth_obj.data.vertices[index].co.copy()
            amplitude = source.x * (8.0 / 3.0)
            cos_key.data[index].co.x += amplitude
            sin_key.data[index].co.z += amplitude
            bias_key.data[index].co.x = -0.5 * amplitude
            bias_key.data[index].co.z -= 0.5 * amplitude

        scene = context.scene
        theta = "0.8*(frame_current-frame_start)*fps_base/fps"
        for key, expression in ((cos_key, f"0.5+0.375*cos({theta})"),
                                (sin_key, f"0.5+0.375*sin({theta})")):
            driver = key.driver_add("value").driver
            driver.expression = expression
            for name, path in (
                ("frame_current", "frame_current"),
                ("frame_start", "frame_start"),
                ("fps", "render.fps"),
                ("fps_base", "render.fps_base"),
            ):
                variable = driver.variables.new()
                variable.name = name
                variable.type = 'SINGLE_PROP'
                target = variable.targets[0]
                target.id_type = 'SCENE'
                target.id = scene
                target.data_path = path

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_testscene_cloth(cloth_obj, solver='PD')
        cloth_obj.GPUCloth.vgroup_mass = "Pin"

        # TestScene.cpp:3434-3435: gravity is forced to zero for TwistTest.
        _set_scene_gravity(context, 0.0, 0.0, 0.0)

        self.report({'INFO'}, "TwistTest scene created")
        return {'FINISHED'}


class GPUCloth_TestMultiLayerDrop(bpy.types.Operator):
    """Create MultiLayerDrop scene: multiple cloth layers falling on cylinder"""
    bl_idname = "gpucloth.test_multi_layer_drop"
    bl_label = "Multi Layer Drop"
    bl_options = {'REGISTER', 'UNDO'}

    # Native bounds: MULTILAYER_COUNT_MIN/MAX/STEP = 1/50/1, default 2.
    layer_count: bpy.props.IntProperty(
        name="Layers",
        description="Cloth layer count (native default 2, dz 0.15)",
        default=TESTSCENE_MULTILAYER_COUNT,
        min=1,
        max=50,
    )

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        # No pins: all N layers fall freely, OGC owns inter-layer contact.
        cloth_obj, _ = _make_multilayer_mesh(
            "MultiLayerCloth", TESTSCENE_GRID_QUADS, TESTSCENE_GRID_QUADS,
            TESTSCENE_CLOTH_HALF_SIZE, TESTSCENE_MULTILAYER_BASE_Z,
            TESTSCENE_MULTILAYER_LAYER_DZ, self.layer_count)
        collision_obj = _make_cylinder_floor(
            "CollisionCylinder",
            TESTSCENE_CYLINDER_RADIUS, TESTSCENE_CYLINDER_HALF_LEN,
            0.0, 0.0, TESTSCENE_CYLINDER_CZ,
            TESTSCENE_FLOOR_Z, TESTSCENE_FLOOR_HALF,
            TESTSCENE_CYLINDER_RINGS, TESTSCENE_CYLINDER_STACKS)

        collision_obj.modifiers.new(name="Collision", type='COLLISION')
        _make_two_sided_collider(collision_obj)

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_testscene_cloth(cloth_obj, solver='PD')
        # Inter-layer contact is OGC self-collision; the C++ auto sizer
        # (TestScene_UI.cpp:2531-2576) derives radius and kc from the grid's
        # mean structural spring length.
        avg_edge = _testscene_avg_edge(
            TESTSCENE_CLOTH_HALF_SIZE, TESTSCENE_GRID_QUADS)
        ogc_radius_mm, ogc_kc = _testscene_ogc_auto(
            avg_edge, TESTSCENE_MULTILAYER_OGC_RADIUS_FRAC)
        cloth_obj.GPUCloth.use_self_collision = True
        cloth_obj.GPUCloth.ogc_radius = ogc_radius_mm
        cloth_obj.GPUCloth.ogc_kc = ogc_kc
        cloth_obj.GPUCloth.ogc_friction = TESTSCENE_OGC_FRICTION
        cloth_obj.GPUCloth.ogc_gamma_p = TESTSCENE_OGC_GAMMA_P

        _set_scene_gravity(context)

        self.report({'INFO'}, "MultiLayerDrop test scene created")
        return {'FINISHED'}


def _apply_cushion_pressure(obj):
    """Pressure at ratio 1.3 on a closed cushion shell.

    ``pressure.ratio 1.3`` (TestScene.cpp:2634) is expressed as
    ``target_volume = 1.3 * V0``, the product RNA's only owner of
    ``Pressure_create``'s ``pressure_ratio``.  ``V0`` is the tetrahedron sum
    over the *rest* mesh, the same sum ``g_ensure_pressure`` performs on
    ``xrest`` (``main.cpp:6001-6014``).  Shared by both cushions of the
    CushionDrop scene; neither value is a new physics constant.
    """
    obj.GPUCloth.use_pressure = True
    obj.GPUCloth.use_pressure_volume = True
    obj.GPUCloth.target_volume = (
        TESTSCENE_CUSHION_PRESSURE_RATIO * _closed_mesh_volume(obj.data))
    return obj.GPUCloth.target_volume


class GPUCloth_TestCushionDrop(bpy.types.Operator):
    """Create CushionDrop scene: two-layer cushion falling on floor"""
    bl_idname = "gpucloth.test_cushion_drop"
    bl_label = "Cushion Drop"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        # TestScene.cpp:2410-2431: dome = 0.25 * CLOTH_HALF_SIZE, sep 0.02,
        # initZ = FLOOR_Z + dome + sep/2 + 2.
        cushion_obj = _make_cushion_mesh(
            "CushionCloth", TESTSCENE_GRID_QUADS, TESTSCENE_GRID_QUADS,
            TESTSCENE_CLOTH_HALF_SIZE, TESTSCENE_CUSHION_INIT_Z,
            TESTSCENE_CUSHION_INIT_SEP, TESTSCENE_CUSHION_DOME_H)

        # Second cushion: the same dome shape, but its two convex caps are
        # *sewn* at the rim instead of welded into one solid, so the scene
        # covers the seam path too.  Same rim plane and dome height as the
        # welded cushion, shifted in x so the two cannot contact each other.
        sewn_obj, _ = _make_sewn_cushion_mesh(
            "CushionSeamCloth", TESTSCENE_SEWN_CUSHION_QUADS,
            TESTSCENE_CLOTH_HALF_SIZE, TESTSCENE_SEWN_CUSHION_CX,
            TESTSCENE_CUSHION_INIT_Z, TESTSCENE_CUSHION_DOME_H)

        # TestScene.cpp:2501-2508: the visible floor quad (FLOOR_Z, half
        # FLOOR_HALF).  The C++ object also carries a degenerate 1x1 cylinder
        # 100 m below the floor; that sliver cannot contact the cloth and its
        # coincident vertices are rejected by the Blender preflight.
        floor_mesh = bpy.data.meshes.new("Floor_mesh")
        floor_obj = bpy.data.objects.new("Floor", floor_mesh)
        bpy.context.collection.objects.link(floor_obj)
        fh = TESTSCENE_FLOOR_HALF
        fz = TESTSCENE_FLOOR_Z
        floor_verts = [(-fh, -fh, fz), (fh, -fh, fz), (fh, fh, fz), (-fh, fh, fz)]
        floor_faces = [(0, 1, 2, 3)]
        floor_mesh.from_pydata(floor_verts, [], floor_faces)
        floor_mesh.update()
        floor_obj.modifiers.new(name="Collision", type='COLLISION')
        _make_two_sided_collider(floor_obj)

        bpy.context.view_layer.objects.active = cushion_obj
        cushion_obj.select_set(True)

        _setup_testscene_cloth(cushion_obj, solver='PD')
        _apply_acceptance_solver_effort(cushion_obj)
        # Closed side-wall volume: pressure.ratio 1.3 (TestScene.cpp:2634) is
        # expressed as target_volume = 1.3 * V0, the product RNA's only owner
        # of Pressure_create's pressure_ratio.
        _apply_cushion_pressure(cushion_obj)

        # The sewn cushion carries the same pressure contract.  Its two rim
        # loops are not welded, so the shell only reads as closed through the
        # native seam quotient; the loose rim edges of its mesh are uploaded as
        # sewing springs with rest length 0, which is what unites them.
        _setup_testscene_cloth(sewn_obj, solver='PD')
        _apply_acceptance_solver_effort(sewn_obj)
        _apply_cushion_pressure(sewn_obj)
        sewn_obj.GPUCloth.use_sewing_springs = True

        _set_scene_gravity(context)

        self.report({'INFO'}, "CushionDrop test scene created")
        return {'FINISHED'}


# --- Imported cloth asset layout (mirrors TestScene.cpp LoadCapeClothBin) ---
# Cape and MD scenes share the binary layout; only Cape consumes
# seams/pins/animation (TestScene.h:166-178).  Asset directories mirror the
# executable-relative catalogs CapeAssetPath/MDAssetPath (TestScene.cpp:135-151).
_CAPE_MAGIC = b"CCAPE002"
_CAPE_VERSION = 2
_CAPE_ASSET_SUBDIR = "cape_import"
_CAPE_ASSET_FILE = "cape.clothbin"
_MD_ASSET_SUBDIR = "md_comparison"
_MD_ASSET_FILE = "MDHorizontalContact.clothbin"


def _find_cloth_asset_dir(subdir, filename, explicit=""):
    """Locate a directory holding ``filename``, or return None.

    The release stage puts both catalogs beside the add-on package
    (tools/build_blender_release.py), so the add-on root and the source-tree
    root are both searched before giving up.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    try:
        addon_dir = os.path.dirname(os.path.abspath(__file__))
        for steps in ("..", os.path.join("..", ".."),
                      os.path.join("..", "..", "..")):
            candidates.append(os.path.join(addon_dir, steps, subdir))
    except Exception:
        pass
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, filename)):
            return os.path.normpath(candidate)
    return None


def _find_cape_asset_dir(explicit=""):
    """Locate cape_import/ holding cape.clothbin, or return None."""
    return _find_cloth_asset_dir(
        _CAPE_ASSET_SUBDIR, _CAPE_ASSET_FILE, explicit)


def _find_md_asset_dir(explicit=""):
    """Locate md_comparison/ holding MDHorizontalContact.clothbin, or None."""
    return _find_cloth_asset_dir(
        _MD_ASSET_SUBDIR, _MD_ASSET_FILE, explicit)


def _read_cape_cloth_bin(path, require_seams=True):
    """Parse CCAPE002 asset with the native header guards, or return None."""
    try:
        with open(path, "rb") as handle:
            payload = handle.read()
    except OSError:
        return None
    if len(payload) < 48 or payload[0:8] != _CAPE_MAGIC:
        return None
    (version, panels, nv, nt, ne, nseam, npin,
     body_nv, body_nt, scale) = struct.unpack("<9If", payload[8:48])
    if (version != _CAPE_VERSION or panels == 0 or nv == 0
            or nt == 0 or ne == 0 or (require_seams and nseam == 0)
            or body_nv == 0 or body_nt == 0 or scale <= 0.0):
        return None
    counts = (nv * 3, nt * 3, ne * 2, nseam * 2, npin,
              body_nv * 3, body_nt * 3)
    formats = ("f", "I", "I", "I", "I", "f", "I")
    if len(payload) != 48 + sum(counts) * 4:
        return None
    parts = []
    offset = 48
    for count, kind in zip(counts, formats):
        parts.append(struct.unpack("<%d%s" % (count, kind),
                                   payload[offset:offset + count * 4])
                     if count else ())
        offset += count * 4
    keys = ("verts", "tris", "edges", "seams", "pins",
            "body_verts", "body_tris")
    asset = dict(zip(keys, parts))
    asset.update(panels=panels, nv=nv, nt=nt, ne=ne, nseam=nseam,
                 npin=npin, body_nv=body_nv, body_nt=body_nt, scale=scale)
    return asset


def _read_md_cloth_bin(path):
    """Parse the MD horizontal-contact fixture, or return None.

    The fixture has no seams and no pins, and its body is a two-triangle
    plate; the counts are a hard guard (TestScene.cpp:421-422).
    """
    asset = _read_cape_cloth_bin(path, require_seams=False)
    if asset is None:
        return None
    for key, expected in TESTSCENE_MD_EXPECTED.items():
        if int(asset[key]) != int(expected):
            return None
    return asset


def _collapse_imported_zero_length_edges(asset):
    """Collapse exact zero-length panel edges, never coincident seam pairs.

    The Cape export contains two collapsed triangles. Only vertices already
    connected by a solid triangle edge can merge; separate panels keep their
    sewing constraints and no position or nonzero-area surface is changed.
    """
    positions = tuple(zip(*[iter(asset["verts"])] * 3))
    triangles = tuple(zip(*[iter(asset["tris"])] * 3))
    parent = list(range(asset["nv"]))

    def root(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for triangle in triangles:
        for a, b in zip(triangle, triangle[1:] + triangle[:1]):
            if positions[a] == positions[b]:
                a, b = root(a), root(b)
                parent[max(a, b)] = min(a, b)
    roots = [root(vertex) for vertex in range(asset["nv"])]
    if all(vertex == representative
           for vertex, representative in enumerate(roots)):
        return asset
    kept = [vertex for vertex, representative in enumerate(roots)
            if vertex == representative]
    indices = {vertex: index for index, vertex in enumerate(kept)}
    remap = [indices[representative] for representative in roots]
    cleaned = dict(asset)
    cleaned["verts"] = tuple(value for vertex in kept
                             for value in positions[vertex])
    cleaned["tris"] = tuple(
        remap[vertex] for triangle in triangles
        if len({remap[vertex] for vertex in triangle}) == 3
        for vertex in triangle)
    for name in ("edges", "seams"):
        pairs = dict.fromkeys(
            tuple(sorted((remap[a], remap[b])))
            for a, b in zip(*[iter(asset[name])] * 2)
            if remap[a] != remap[b])
        cleaned[name] = tuple(vertex for pair in pairs for vertex in pair)
    cleaned["pins"] = tuple(dict.fromkeys(remap[v] for v in asset["pins"]))
    for count, name, stride in (("nv", "verts", 3), ("nt", "tris", 3),
                                ("ne", "edges", 2), ("nseam", "seams", 2),
                                ("npin", "pins", 1)):
        cleaned[count] = len(cleaned[name]) // stride
    return cleaned


def _imported_cloth_objects(asset):
    """Build (cloth_obj, body_obj) from a parsed cloth asset, unscaled."""
    scale = asset["scale"]
    verts = [(asset["verts"][i] * scale,
              asset["verts"][i + 1] * scale,
              asset["verts"][i + 2] * scale)
             for i in range(0, asset["nv"] * 3, 3)]
    faces = [(asset["tris"][i], asset["tris"][i + 1], asset["tris"][i + 2])
             for i in range(0, asset["nt"] * 3, 3)]
    body_verts = [(asset["body_verts"][i] * scale,
                   asset["body_verts"][i + 1] * scale,
                   asset["body_verts"][i + 2] * scale)
                  for i in range(0, asset["body_nv"] * 3, 3)]
    body_faces = [(asset["body_tris"][i], asset["body_tris"][i + 1],
                   asset["body_tris"][i + 2])
                  for i in range(0, asset["body_nt"] * 3, 3)]
    return verts, faces, body_verts, body_faces


def _apply_imported_contact_settings(cloth_obj, contact_thickness):
    """C++ contact thickness/friction/convergence for imported garments.

    TestScene.cpp:2525-2542: epsilon = selfepsilon = contact thickness,
    friction 0.5, and four collision-iteration passes.
    """
    settings = cloth_obj.GPUCloth
    settings.epsilon = contact_thickness
    settings.selfepsilon = contact_thickness
    settings.collision_friction = TESTSCENE_CONTACT_FRICTION
    settings.collision_quality = TESTSCENE_COLLISION_LOOPS


class GPUCloth_TestCape(bpy.types.Operator):
    """Create CapeProject scene: sewn garment panels over body collider"""
    bl_idname = "gpucloth.test_cape"
    bl_label = "Cape Project"
    bl_options = {'REGISTER', 'UNDO'}

    asset_dir: bpy.props.StringProperty(
        name="Cape Asset Dir",
        description="Directory holding cape.clothbin (default: addon cape_import/)",
        default="",
    )

    collar_pins: bpy.props.BoolProperty(
        name="Collar Pins",
        description=(
            "Pin the imported garment at the collar.  Native default is off "
            "(TESTSCENE_CAPE_COLLAR_PINS=0), i.e. an MD-style free garment"),
        default=False,
    )

    def execute(self, context):
        asset_dir = _find_cape_asset_dir(self.asset_dir)
        if asset_dir is None:
            self.report({'ERROR'},
                        "cape.clothbin not found: set asset_dir or ship "
                        "cape_import/ beside the addon")
            return {'CANCELLED'}
        asset = _read_cape_cloth_bin(
            os.path.join(asset_dir, _CAPE_ASSET_FILE))
        if asset is None:
            self.report({'ERROR'}, "invalid cape.clothbin header or payload")
            return {'CANCELLED'}

        asset = _collapse_imported_zero_length_edges(asset)

        bpy.ops.object.select_all(action='DESELECT')
        verts, faces, body_verts, body_faces = _imported_cloth_objects(asset)
        cloth_mesh = bpy.data.meshes.new("CapeCloth_mesh")
        cloth_obj = bpy.data.objects.new("CapeCloth", cloth_mesh)
        bpy.context.collection.objects.link(cloth_obj)
        # Explicit sewing springs: one loose edge per imported seam pair
        # (TestScene.cpp:2554-2584).
        cloth_mesh.from_pydata(verts, _imported_cloth_edges(asset), faces)
        cloth_mesh.update()
        # The binary stores XYZ only, so this is an automatic local-rest
        # orientation and carries no claim about MD garment grain.
        _ensure_generated_material_uv(cloth_mesh)

        body_mesh = bpy.data.meshes.new("CapeBody_mesh")
        body_obj = bpy.data.objects.new("CapeBody", body_mesh)
        bpy.context.collection.objects.link(body_obj)
        body_mesh.from_pydata(body_verts, [], body_faces)
        body_mesh.update()
        body_obj.modifiers.new(name="Collision", type='COLLISION')
        _make_two_sided_collider(body_obj)

        # Imported garments use millimetre contact gaps. Blender's 20 mm
        # default shell overlaps that rest pose; use the product minimum.
        body_obj.collision.thickness_outer = 0.001

        if asset["npin"]:
            pin_group = cloth_obj.vertex_groups.new(name="CapePin")
            pin_group.add(list(asset["pins"]), 1.0, 'REPLACE')

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_testscene_cloth(cloth_obj, solver='PD')
        _apply_imported_contact_settings(
            cloth_obj, TESTSCENE_CAPE_CONTACT_THICKNESS)
        cloth_obj.GPUCloth.use_self_collision = True
        cloth_obj.GPUCloth.ogc_radius = TESTSCENE_CAPE_OGC_RADIUS_MM
        cloth_obj.GPUCloth.ogc_kc = TESTSCENE_CAPE_OGC_KC
        cloth_obj.GPUCloth.ogc_friction = TESTSCENE_CAPE_OGC_FRICTION
        cloth_obj.GPUCloth.use_sewing_springs = True
        if self.collar_pins and asset["npin"]:
            cloth_obj.GPUCloth.vgroup_mass = "CapePin"

        _set_scene_gravity(context)

        self.report({'INFO'},
                    "CapeProject test scene created: %d panels, %d verts, "
                    "%d tris, %d seams, %d pins%s" % (
                        asset["panels"], asset["nv"], asset["nt"],
                        asset["nseam"], asset["npin"],
                        "" if self.collar_pins else " (collar pins off)"))
        return {'FINISHED'}


class GPUCloth_TestMDHorizontalContact(bpy.types.Operator):
    """Create MDHorizontalContact scene: MD-imported cloth on a static plate"""
    bl_idname = "gpucloth.test_md_horizontal_contact"
    bl_label = "MD Horizontal Contact"
    bl_options = {'REGISTER', 'UNDO'}

    asset_dir: bpy.props.StringProperty(
        name="MD Asset Dir",
        description=(
            "Directory holding MDHorizontalContact.clothbin "
            "(default: addon md_comparison/)"),
        default="",
    )

    def execute(self, context):
        asset_dir = _find_md_asset_dir(self.asset_dir)
        if asset_dir is None:
            self.report({'ERROR'},
                        "MDHorizontalContact.clothbin not found: set "
                        "asset_dir or ship md_comparison/ beside the addon")
            return {'CANCELLED'}
        asset = _read_md_cloth_bin(os.path.join(asset_dir, _MD_ASSET_FILE))
        if asset is None:
            self.report({'ERROR'},
                        "invalid MDHorizontalContact.clothbin header, "
                        "payload, or fixture counts")
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        verts, faces, body_verts, body_faces = _imported_cloth_objects(asset)
        cloth_mesh = bpy.data.meshes.new("MDCloth_mesh")
        cloth_obj = bpy.data.objects.new("MDCloth", cloth_mesh)
        bpy.context.collection.objects.link(cloth_obj)
        cloth_mesh.from_pydata(verts, _imported_cloth_edges(asset), faces)
        cloth_mesh.update()
        # The binary stores XYZ only, so this is an automatic local-rest
        # orientation and carries no claim about MD garment grain.
        _ensure_generated_material_uv(cloth_mesh)

        body_mesh = bpy.data.meshes.new("MDContactBody_mesh")
        body_obj = bpy.data.objects.new("MDContactBody", body_mesh)
        bpy.context.collection.objects.link(body_obj)
        body_mesh.from_pydata(body_verts, [], body_faces)
        body_mesh.update()
        body_obj.modifiers.new(name="Collision", type='COLLISION')
        _make_two_sided_collider(body_obj)

        # The fixture starts 10 mm above the plate. The default 20 mm
        # Blender shell overlaps it before the cloth's 2.5 mm margin.
        body_obj.collision.thickness_outer = 0.001

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        # TestScene.cpp:5108-5125: no proxy, no SDB bending route, no self
        # collision, PD only.  No pins and no seams (TestScene.cpp:2728-2731).
        _setup_testscene_cloth(cloth_obj, solver='PD')
        _apply_imported_contact_settings(
            cloth_obj, TESTSCENE_MD_CONTACT_THICKNESS)
        cloth_obj.GPUCloth.use_self_collision = False
        cloth_obj.GPUCloth.use_proxy = False

        _set_scene_gravity(context)

        self.report({'INFO'},
                    "MDHorizontalContact test scene created: %d verts, "
                    "%d tris, static body %d verts" % (
                        asset["nv"], asset["nt"], asset["body_nv"]))
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
#  Interaction: move the cloth by dragging one vertex (MD-style grab)
# ===========================================================================
#
#  Reference behaviour (C++ TestScene application):
#    * TestScene_UI.cpp:3435-3470 — a scene click not consumed by the UI grabs
#      the nearest cloth vertex; releasing the button clears the grab.
#    * TestScene_UI.cpp:2716-2731 — the cursor delta is converted with
#      worldPerPx = depth * tanHalfFov * 2 / H and applied along the camera
#      right / screen-up axes, so the target stays in the camera plane at the
#      grabbed vertex's depth and the vertex follows the cursor.
#    * TestScene.cpp:3491-3504 — while a grab is live the drag target is
#      written into that vertex's positional constraint
#      (cv.xconst = m_dragTarget, CLOTH_VERT_FLAG_PINNED) once per frame, and
#      restored after the solver step, so the cloth is driven by the solver
#      rather than by directly written coordinates.
#
#  The add-on cannot write xconst directly: GPUCLOTH_VERTEX_PIN_WEIGHT and
#  GPUCLOTH_VERTEX_PIN_TARGET_XYZ are rejected as NOT_CONFIGURABLE, and the
#  animated pin snapshot owns all per-vertex pin state.  The grab therefore
#  folds its target into the pin snapshot that _publish_frame_inputs already
#  commits once per simulated frame, which the native commit turns into
#  exactly PINNED + xconst for that one vertex.  The tool is inert outside
#  playback: the gate below is the only place that lets the modal touch the
#  mouse, and every other event is passed through to Blender unchanged.

_VERTEX_DRAG_POLL_INTERVAL = 0.02
_VERTEX_DRAG_PICK_RADIUS_PX = 14.0
_VERTEX_DRAG_MARKER_FRACTION = 0.02

_vertex_drag_state = {
    'armed': False,        # the tool was switched on from the panel
    'modal_live': False,   # a modal handler owns the timer right now
    'dragging': False,     # one vertex is currently held
    'object_index': -1,    # index into g_clothOBJs / g_simulationOBJs
    'object_uid': 0,       # Blender session UID of the grabbed cloth object
    'vertex_index': -1,
    'target_world': None,  # mathutils.Vector; the live constraint target
    'last_mouse': (0.0, 0.0),
    'area': None,          # viewport owning the current drag
    'region': None,
    'timer': None,
    'status': "",
}
_vertex_drag_draw_handle = None


def _animation_is_playing(context):
    """True only while Blender plays back the animation (live sim drive)."""
    screen = getattr(context, "screen", None)
    if screen is None:
        return False
    try:
        return bool(screen.is_animation_playing)
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False


def _vertex_drag_simulation_ready(scene):
    """True when a live (non-baked) simulation can consume a drag target."""
    if g_dll is None or not g_clothOBJs or not g_simulationOBJs:
        return False
    if scene is None or not bool(
            getattr(scene, "gpu_cloth_springs_built", False)):
        return False
    helper = getattr(scene, "gpu_cloth_helper", None)
    if helper is not None and bool(getattr(helper, "is_baked", False)):
        return False
    if prepare_task_active() or _teardown_failure or _stop_requested:
        return False
    return True


def _vertex_drag_frame_will_step(scene):
    """True when the frame path will step the live solver from here.

    The grab has no other route to the solver: the drag target is folded into
    the pin snapshot that ``_publish_frame_inputs`` commits, and that function
    runs only from ``update_simulation``, which returns before publishing when
    the requested frame is one the live path only replays from retained or
    cached state (``frame <= _simulation_frame_state['last_solved']``), when
    the frame is outside the bake range, or when a rebuilt owner is pending and
    no frame may be solved into it.  Baked cache playback owns every frame
    outright and ``_vertex_drag_simulation_ready`` already excludes it.

    A frame past the frontier is stepped only while a producer is running
    (``_frame_producer_running``): a bare move of the playhead onto it leaves
    the mesh where it was, so there is no step for the grab to ride and the gate
    must say so rather than take the mouse for a frame nothing will produce.

    A replayed frame is not a live simulation either: the animation runs, but
    every frame is a state the solver already reached, so a grab there is
    silently inert - nothing pins the vertex and the cloth keeps replaying.
    Mirroring the frame path's own decision here is what keeps the gate from
    taking the mouse for a frame the drag cannot reach.
    """
    if scene is None:
        return False
    if _simulation_frame_state['rebuild_pending']:
        return False
    helper = getattr(scene, "gpu_cloth_helper", None)
    frame = int(scene.frame_current)
    if frame < int(_bake_range['start']):
        return False
    if frame >= int(getattr(helper, "bake_end", _bake_range['end'])):
        return False
    if not _frame_producer_running():
        return False
    last_solved = _simulation_frame_state['last_solved']
    return last_solved is None or frame >= int(last_solved)


def _vertex_drag_gate(context):
    """The only condition under which the tool may consume the mouse.

    Outside playback, and outside a frame the live solver will actually step,
    this returns False and the modal handler passes every event through, so
    clicks keep selecting objects and mesh elements exactly as Blender
    normally does.

    Two live drivers exist and they are alternatives, not a union of
    accidents: Blender's own playback, and the infinite-simulation step timer.
    While the infinite drive runs the solver takes a step on every tick, so
    the gate must be open for the whole time it runs.  Everything below the
    driver test is shared: ``_vertex_drag_frame_will_step(scene)`` is exactly
    "the driver will step", which is false for a replayed frame and true for
    every tick of the infinite drive.
    """
    if not _vertex_drag_state['armed']:
        return False
    if not (_animation_is_playing(context) or _infinite_is_running()):
        return False
    scene = getattr(context, "scene", None)
    if not _vertex_drag_simulation_ready(scene):
        return False
    return _vertex_drag_frame_will_step(scene)


def _vertex_drag_region(context, event):
    """Resolve the viewport from window coordinates, not the invoking panel.

    Blender retains the Properties context in this window modal handler.
    Once grabbed, keep the same projection even outside the viewport.
    """
    window = getattr(context, "window", None)
    if window is None:
        return None, None, None
    state = _vertex_drag_state
    try:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            if state['dragging'] and area != state['area']:
                continue
            for region in area.regions:
                if region.type != 'WINDOW':
                    continue
                if state['dragging']:
                    if region != state['region']:
                        continue
                elif not (region.x <= event.mouse_x < region.x + region.width
                          and region.y <= event.mouse_y < region.y + region.height):
                    continue
                # Resolve the region's own projection, including quad views.
                with context.temp_override(window=window, area=area, region=region):
                    rv3d = context.region_data
                if rv3d is not None:
                    return area, region, rv3d
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        pass
    return None, None, None


def _vertex_drag_release(reason):
    """Drop the grabbed vertex so the cloth stops being constrained."""
    state = _vertex_drag_state
    if not state['dragging']:
        return False
    state['dragging'] = False
    state['object_index'] = -1
    state['object_uid'] = 0
    state['vertex_index'] = -1
    state['target_world'] = None
    state['last_mouse'] = (0.0, 0.0)
    state['area'] = None
    state['region'] = None
    state['status'] = reason
    return True


def _vertex_drag_grab_is_live():
    """The held vertex still exists in the mesh the solver owns."""
    state = _vertex_drag_state
    index = state['object_index']
    if index < 0 or index >= len(g_simulationOBJs):
        return False
    obj = g_simulationOBJs[index]
    if obj is None:
        return False
    try:
        vertex_count = len(obj.data.vertices)
        object_uid = int(obj.session_uid)
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False
    if object_uid != state['object_uid']:
        return False
    return 0 <= state['vertex_index'] < vertex_count


def _simulation_vertex_world_positions(obj, depsgraph):
    """Homogeneous world-space positions of one simulation mesh, as (n, 4)."""
    try:
        vertices = obj.evaluated_get(depsgraph).data.vertices
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        try:
            vertices = obj.data.vertices
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            return None
    vertex_count = len(vertices)
    if vertex_count == 0:
        return None
    local = np.empty(vertex_count * 3, dtype=np.float64)
    vertices.foreach_get("co", local)
    try:
        matrix = np.array(obj.matrix_world, dtype=np.float64)
    except (ReferenceError, RuntimeError, TypeError, ValueError):
        return None
    positions = np.empty((vertex_count, 4), dtype=np.float64)
    positions[:, :3] = local.reshape(vertex_count, 3)
    positions[:, 3] = 1.0
    return positions @ matrix.T


def _vertex_drag_pick(region, rv3d, depsgraph, mouse):
    """Nearest simulation vertex under the cursor, within the pick radius.

    Mirrors TestScene::PickClosestVertex: project every candidate into the
    region and keep the closest one, or nothing when the cursor is not close
    to any cloth vertex.
    """
    try:
        projection = np.array(rv3d.perspective_matrix, dtype=np.float64)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    half_width = float(region.width) * 0.5
    half_height = float(region.height) * 0.5
    best = None
    best_distance = _VERTEX_DRAG_PICK_RADIUS_PX ** 2
    for index, obj in enumerate(g_simulationOBJs):
        positions = _simulation_vertex_world_positions(obj, depsgraph)
        if positions is None:
            continue
        clip = positions @ projection.T
        w = clip[:, 3]
        valid = w > 1.0e-6
        if not bool(valid.any()):
            continue
        inverse_w = np.where(valid, 1.0 / np.where(valid, w, 1.0), 0.0)
        px = half_width + half_width * clip[:, 0] * inverse_w
        py = half_height + half_height * clip[:, 1] * inverse_w
        distance = (px - mouse[0]) ** 2 + (py - mouse[1]) ** 2
        distance = np.where(valid, distance, np.inf)
        candidate = int(np.argmin(distance))
        if float(distance[candidate]) >= best_distance:
            continue
        best_distance = float(distance[candidate])
        best = (
            index,
            candidate,
            Vector(positions[candidate, :3]),
        )
    return best


def _vertex_drag_local_target(index):
    """The live drag target expressed in the simulation object's space."""
    state = _vertex_drag_state
    if not state['dragging'] or state['target_world'] is None:
        return None
    if index < 0 or index >= len(g_simulationOBJs):
        return None
    obj = g_simulationOBJs[index]
    if obj is None:
        return None
    try:
        matrix = obj.matrix_world.inverted()
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        _vertex_drag_release("grab rejected: cloth transform is singular")
        return None
    return matrix @ state['target_world']


def _vertex_drag_pin_snapshot(index, snapshot):
    """Fold the live drag target into one frame's pin snapshot.

    Called once per simulated frame from _publish_frame_inputs, so the
    constraint target is re-sampled every frame and released as soon as the
    grab ends.
    """
    state = _vertex_drag_state
    if (not state['dragging'] or state['object_index'] != index or
            state['vertex_index'] < 0):
        return snapshot
    if not _vertex_drag_grab_is_live():
        _vertex_drag_release("grab released: cloth topology changed")
        return snapshot
    target = _vertex_drag_local_target(index)
    if target is None:
        return snapshot
    try:
        return with_dragged_vertex_pin(
            snapshot, state['vertex_index'], tuple(target))
    except VertexChannelError as exc:
        _vertex_drag_release(f"grab rejected: {exc}")
        return snapshot


def _vertex_drag_begin(context, event):
    """Grab the nearest cloth vertex under the cursor."""
    area, region, rv3d = _vertex_drag_region(context, event)
    if region is None:
        return False
    mouse = (float(event.mouse_x - region.x), float(event.mouse_y - region.y))
    try:
        depsgraph = context.evaluated_depsgraph_get()
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return False
    hit = _vertex_drag_pick(region, rv3d, depsgraph, mouse)
    if hit is None:
        return False
    index, vertex_index, world = hit
    try:
        object_uid = int(g_simulationOBJs[index].session_uid)
    except (AttributeError, IndexError, ReferenceError, RuntimeError, TypeError):
        return False
    state = _vertex_drag_state
    state['dragging'] = True
    state['object_index'] = index
    state['object_uid'] = object_uid
    state['vertex_index'] = vertex_index
    state['target_world'] = world
    state['last_mouse'] = mouse
    state['area'] = area
    state['region'] = region
    # The edit is in force from this moment: the cloth about to be simulated is
    # pinned by the user's hand, so the past belongs to a simulation that was not.
    _vertex_grab_drops_the_cache(context.scene)
    state['status'] = f"dragging vertex {vertex_index}"
    return True


def _vertex_drag_update(context, event):
    """Move the constraint target with the cursor, in the camera plane."""
    state = _vertex_drag_state
    _area, region, rv3d = _vertex_drag_region(context, event)
    if region is None or state['target_world'] is None:
        return False
    mouse = (float(event.mouse_x - region.x), float(event.mouse_y - region.y))
    previous = state['last_mouse']
    delta_x = mouse[0] - previous[0]
    delta_y = mouse[1] - previous[1]
    state['last_mouse'] = mouse
    if delta_x == 0.0 and delta_y == 0.0:
        return False
    target = state['target_world']
    try:
        base = region_2d_to_location_3d(region, rv3d, mouse, target)
        right = region_2d_to_location_3d(
            region, rv3d, (mouse[0] + 1.0, mouse[1]), target)
        up = region_2d_to_location_3d(
            region, rv3d, (mouse[0], mouse[1] + 1.0), target)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    if base is None or right is None or up is None:
        return False
    # Per-pixel world vectors along the camera right / screen-up axes at the
    # grabbed vertex's depth; this is TestScene's worldPerPx conversion
    # (depth * tanHalfFov * 2 / H) taken from the region projection, so it
    # also holds for orthographic views.
    state['target_world'] = (
        target + (right - base) * delta_x + (up - base) * delta_y)
    return True


def _vertex_drag_tag_redraw(context):
    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in getattr(screen, "areas", ()):
        if getattr(area, "type", None) == 'VIEW_3D':
            area.tag_redraw()


def _vertex_drag_timer_remove(window_manager):
    state = _vertex_drag_state
    timer = state['timer']
    state['timer'] = None
    if timer is None or window_manager is None:
        return
    try:
        window_manager.event_timer_remove(timer)
    except (AttributeError, RuntimeError, TypeError):
        pass


def _vertex_drag_arm(context, operator):
    """Arm the tool and register the modal handler that owns its events."""
    state = _vertex_drag_state
    if state['armed']:
        return True
    if state['modal_live']:
        # The previous session still owns its timer and is leaving on its next
        # poll; a second handler would orphan that timer.
        return False
    window = getattr(context, "window", None)
    window_manager = getattr(context, "window_manager", None)
    if window is None or window_manager is None:
        return False
    try:
        state['timer'] = window_manager.event_timer_add(
            _VERTEX_DRAG_POLL_INTERVAL, window=window)
        window_manager.modal_handler_add(operator)
    except (AttributeError, RuntimeError, TypeError) as exc:
        _vertex_drag_timer_remove(window_manager)
        print(f"GPUCloth vertex drag: cannot start modal tool: {exc}")
        return False
    state['armed'] = True
    state['modal_live'] = True
    state['status'] = "armed: play the animation, then drag a cloth vertex"
    return True


def _vertex_drag_teardown(context):
    """Symmetric counterpart of _vertex_drag_arm."""
    state = _vertex_drag_state
    _vertex_drag_release("grab released: tool off")
    state['armed'] = False
    _vertex_drag_timer_remove(getattr(context, "window_manager", None))
    state['modal_live'] = False
    _vertex_drag_tag_redraw(context)


def _vertex_drag_disarm(context, status):
    """Leave the tool switched on as a modal handler but stop its grab.

    The armed modal handler removes itself, and its timer, on its next poll;
    removing the timer here would leave it unable to observe the flag.
    """
    state = _vertex_drag_state
    _vertex_drag_release(status)
    state['armed'] = False
    state['status'] = status


def vertex_drag_tool_state(context=None):
    """Read-only tool state for the panel."""
    context = bpy.context if context is None else context
    state = _vertex_drag_state
    dragging = bool(state['dragging'])
    object_name = ""
    if dragging and 0 <= state['object_index'] < len(g_clothOBJs):
        try:
            object_name = g_clothOBJs[state['object_index']].name
        except (AttributeError, IndexError, ReferenceError, RuntimeError):
            object_name = ""
    return {
        "armed": bool(state['armed']),
        "dragging": dragging,
        "vertex_index": int(state['vertex_index']) if dragging else -1,
        "object_name": object_name,
        "playing": _animation_is_playing(context),
        "live": _vertex_drag_frame_will_step(
            getattr(context, "scene", None)),
        "available": _vertex_drag_simulation_ready(
            getattr(context, "scene", None)),
        "status": state['status'] or "",
    }


def _vertex_drag_draw():
    """Viewport overlay: grabbed vertex, live constraint target."""
    state = _vertex_drag_state
    if not state['dragging'] or state['target_world'] is None:
        return
    index = state['object_index']
    vertex_index = state['vertex_index']
    if index < 0 or index >= len(g_simulationOBJs):
        return
    obj = g_simulationOBJs[index]
    if obj is None:
        return
    try:
        import gpu
        from gpu_extras.batch import batch_for_shader
    except ImportError:
        return
    try:
        vertices = obj.data.vertices
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return
    if vertex_index < 0 or vertex_index >= len(vertices):
        return
    try:
        right = Vector((1.0, 0.0, 0.0))
        up = Vector((0.0, 0.0, 1.0))
        region_data = bpy.context.region_data
        size = max(
            abs(float(region_data.view_distance)) * _VERTEX_DRAG_MARKER_FRACTION,
            1.0e-5)
        right = region_data.view_rotation @ right
        up = region_data.view_rotation @ up
    except (AttributeError, RuntimeError, TypeError, ValueError):
        right = Vector((1.0, 0.0, 0.0))
        up = Vector((0.0, 0.0, 1.0))
        size = 0.01
    try:
        vertex_world = obj.matrix_world @ vertices[vertex_index].co
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return

    target = state['target_world']
    target_lines = [
        target - right * size, target + right * size,
        target - up * size, target + up * size,
    ]
    vertex_lines = [
        vertex_world - right * size - up * size,
        vertex_world + right * size - up * size,
        vertex_world + right * size - up * size,
        vertex_world + right * size + up * size,
        vertex_world + right * size + up * size,
        vertex_world - right * size + up * size,
        vertex_world - right * size + up * size,
        vertex_world - right * size - up * size,
    ]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    shader.bind()
    shader.uniform_float("color", (1.0, 0.30, 0.30, 1.0))
    batch = batch_for_shader(
        shader, 'LINES',
        {"pos": [tuple(point) for point in vertex_lines]})
    batch.draw(shader)
    shader.uniform_float("color", (1.0, 0.85, 0.10, 1.0))
    batch = batch_for_shader(
        shader, 'LINES',
        {"pos": [tuple(point) for point in target_lines]})
    batch.draw(shader)
    gpu.state.blend_set('NONE')


def _vertex_drag_load_post_handler(*_args):
    """A new file owns no native grab and no native step drive."""
    _infinite_stop("a new file was loaded")
    _infinite_sim_state['checkpoint'] = None
    _vertex_drag_shutdown()


def _vertex_drag_shutdown():
    """Release every transient surface the tool owns (unregister path)."""
    global _vertex_drag_draw_handle
    _vertex_drag_teardown(bpy.context)
    if _vertex_drag_draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(
                _vertex_drag_draw_handle, 'WINDOW')
        except (AttributeError, RuntimeError, TypeError) as exc:
            print(f"GPUCloth vertex drag: overlay teardown failed: {exc}")
        _vertex_drag_draw_handle = None


def _vertex_drag_add_draw_handler():
    global _vertex_drag_draw_handle
    if _vertex_drag_draw_handle is None:
        _vertex_drag_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _vertex_drag_draw, (), 'WINDOW', 'POST_VIEW')


class GPUCloth_MoveClothByVertex(bpy.types.Operator):
    """Move the cloth by dragging one vertex while the simulation plays

    The grab is active only during animation playback: while the animation is
    stopped every event is handed back to Blender, so clicking keeps
    selecting as usual.
    """
    bl_idname = "gpucloth.move_cloth_by_vertex"
    bl_label = "Move Cloth By Vertex (MD-style)"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return (
            _vertex_drag_state['armed'] or
            _vertex_drag_simulation_ready(getattr(context, "scene", None))
        )

    def invoke(self, context, event):
        state = _vertex_drag_state
        if state['armed']:
            _vertex_drag_disarm(context, "tool off")
            self.report({'INFO'}, "Move Cloth By Vertex: off")
            return {'FINISHED'}
        if state['modal_live']:
            self.report(
                {'INFO'},
                "Move Cloth By Vertex is still stopping; try again")
            return {'CANCELLED'}
        if not _vertex_drag_simulation_ready(getattr(context, "scene", None)):
            self.report(
                {'ERROR'},
                "Prepare the simulation before moving cloth by vertex")
            return {'CANCELLED'}
        if not _vertex_drag_arm(context, self):
            self.report({'ERROR'}, "Move Cloth By Vertex needs a window")
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            "Move Cloth By Vertex armed: play the animation, then drag a "
            "cloth vertex")
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if _vertex_drag_state['armed']:
            _vertex_drag_disarm(context, "tool off")
            return {'FINISHED'}
        bpy.ops.gpucloth.move_cloth_by_vertex('INVOKE_DEFAULT')
        return {'FINISHED'}

    def modal(self, context, event):
        state = _vertex_drag_state

        # ── Бесконечная симуляция: Space и Ctrl+Z (ДО гейта) ────────────────
        #
        #  Ветка обязана быть до гейта: в покое гейт ложен
        #  (``is_animation_playing`` в статичной сцене False, драйвер ещё не
        #  запущен), поэтому за гейтом Space никогда бы не дошёл.  Событие
        #  потребляется только когда режим действительно может что-то сделать;
        #  иначе оно уходит в Blender, и Space остаётся обычным Space.
        if event.type == 'SPACE' and event.value == 'PRESS':
            if _infinite_is_running():
                result = _infinite_toggle(context)
                self.report(
                    {'INFO'},
                    "Infinite simulation stopped after "
                    f"{result.get('steps', 0)} step(s) (Space)")
                return {'RUNNING_MODAL'}
            capable = _infinite_capable(getattr(context, "scene", None))
            if capable.get("ok"):
                result = _infinite_start(context)
                if result.get("ok"):
                    self.report(
                        {'INFO'},
                        "Infinite simulation started (Space); press Space "
                        "again to stop, Ctrl+Z to return to the start")
                else:
                    self.report({'ERROR'}, _infinite_status_text(result))
                return {'RUNNING_MODAL'}

        if (event.type == 'Z' and event.value == 'PRESS' and
                getattr(event, "ctrl", False) and
                _infinite_sim_state['checkpoint'] is not None):
            # Одна клавиша — одно значение.  Пока режим держит чекпоинт,
            # Ctrl+Z возвращает к нему и НЕ пересылается в Blender: иначе
            # undo выполнился бы дважды с разным смыслом.
            result = _infinite_restore_start(context)
            if result.get("ok"):
                self.report({'INFO'}, _infinite_text(result.get("message")))
            else:
                self.report({'ERROR'}, _infinite_status_text(result))
            return {'RUNNING_MODAL'}

        if event.type == 'TIMER':
            if not state['armed']:
                _infinite_stop("the vertex drag tool was switched off")
                _vertex_drag_teardown(context)
                return {'CANCELLED'}
            if state['dragging'] and (
                    not _vertex_drag_gate(context) or
                    not _vertex_drag_grab_is_live()):
                _vertex_drag_release("grab released: simulation is not stepping")
                _vertex_drag_tag_redraw(context)
            return {'PASS_THROUGH'}

        if not _vertex_drag_gate(context):
            # Not playing and not driving, or a frame the live solver will not
            # step: never take the mouse, let Blender select normally.
            if state['dragging']:
                _vertex_drag_release("grab released: simulation is not stepping")
                _vertex_drag_tag_redraw(context)
            return {'PASS_THROUGH'}

        if event.type == 'ESC':
            # ESC while the drive runs stops the drive first; the next ESC
            # disarms the tool.  Two states, two meanings, in that order.
            if _infinite_is_running():
                _infinite_toggle(context)
                self.report({'INFO'}, "Infinite simulation stopped (ESC)")
                return {'RUNNING_MODAL'}
            if state['dragging']:
                _vertex_drag_release("grab cancelled")
                _vertex_drag_tag_redraw(context)
                return {'RUNNING_MODAL'}
            _vertex_drag_disarm(context, "tool off")
            return {'RUNNING_MODAL'}

        if state['dragging']:
            if event.type == 'MOUSEMOVE':
                if _vertex_drag_update(context, event):
                    _vertex_drag_tag_redraw(context)
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                _vertex_drag_release("grab released")
                _vertex_drag_tag_redraw(context)
                return {'RUNNING_MODAL'}
            if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
                _vertex_drag_release("grab cancelled")
                _vertex_drag_tag_redraw(context)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if _vertex_drag_begin(context, event):
                _vertex_drag_tag_redraw(context)
                return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}


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

    for cloth_obj in _live_cloth_objects():
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


class GPUCloth_ToggleInfiniteSimulation(bpy.types.Operator):
    """Start or stop the infinite simulation (the no-animation drive).

    The mode's only controls were Space *inside* the vertex-drag tool's modal
    (``operators.py:14197`` and ``14252``) and the tool being switched off.  Once the
    pointer left the viewport there was nothing to press: the panel drew the state
    and the Space hint, and no operator.  That is the owner's «буквально никакие
    действия не в силах остановить расчёты» on the loop that has no timeline to
    pause, and half of «не переходят в режим бесконечной симуляции» - the mode could
    not be entered except through a key that only reaches the modal.

    The preconditions are the mode's own (``_infinite_capable``), so the button is
    offered exactly when starting it would work, and stopping is always offered while
    it runs - a mode that can be entered but not left is the defect.
    """
    bl_idname = "gpucloth.toggle_infinite_simulation"
    bl_label  = "Infinite simulation"

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        if scene is None:
            return False
        if _infinite_is_running():
            return True
        return bool(_infinite_capable(scene).get("ok"))

    def execute(self, context):
        result = _infinite_toggle(context)
        if result.get("ok"):
            steps = int(result.get("steps", 0))
            if result.get("stopped"):
                self.report(
                    {'INFO'},
                    _t_infinite(
                        f"Infinite simulation stopped after {steps} step(s)",
                        f"Бесконечная симуляция остановлена после {steps} шаг(ов)")[0])
            else:
                self.report(
                    {'INFO'},
                    _t_infinite("Infinite simulation started",
                                "Бесконечная симуляция запущена")[0])
            return {'FINISHED'}
        self.report({'ERROR'}, _infinite_status_text(result))
        return {'CANCELLED'}


_OPERATOR_CLASSES = [
    GPUCloth_SyncCPUSettings,
    GPUCloth_ShowCPUSyncReport,
    GPUCloth_FreeVRAM,
    GPUCloth_ToggleInfiniteSimulation,
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
    GPUCloth_MoveClothByVertex,
    GPUCloth_FreeCache,
    GPUCloth_ExportAlembic,
    GPUCloth_ExportUSD,
    GPUCloth_TestDrapeOnSphere,
    GPUCloth_TestTwist,
    GPUCloth_TestMultiLayerDrop,
    GPUCloth_TestCushionDrop,
    GPUCloth_TestCape,
    GPUCloth_TestMDHorizontalContact,
    GPUCloth_TestOGCBounds,
]


def _register_operator_surfaces():
    global _ogc_draw_handle
    registered_classes = []
    frame_handler_added = False
    cache_handler_added = False
    draw_handler_added = False
    vertex_drag_draw_handler_added = False
    load_handler_added = False
    warmup_load_handler_added = False
    warmup_depsgraph_handler_added = False
    undo_handler_added = False
    redo_handler_added = False
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
        # The undo observer of the infinite mode.  Registered for the whole
        # add-on lifetime, not for the mode's: an undo can land while the mode
        # is idle, and a handler that only exists while the drive runs cannot
        # see the undo that would invalidate the held checkpoint.
        if (_infinite_undo_post_handler not in bpy.app.handlers.undo_post):
            bpy.app.handlers.undo_post.append(_infinite_undo_post_handler)
            undo_handler_added = True
        if (_infinite_redo_post_handler not in bpy.app.handlers.redo_post):
            bpy.app.handlers.redo_post.append(_infinite_redo_post_handler)
            redo_handler_added = True
        if _ogc_draw_handle is None:
            _ogc_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                _ogc_bounds_draw, (), 'WINDOW', 'POST_VIEW')
            draw_handler_added = True
        if _vertex_drag_draw_handle is None:
            _vertex_drag_add_draw_handler()
            vertex_drag_draw_handler_added = True
        if (_vertex_drag_load_post_handler not in
                bpy.app.handlers.load_post):
            bpy.app.handlers.load_post.append(_vertex_drag_load_post_handler)
            load_handler_added = True
        if _warmup_native_handler not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(_warmup_native_handler)
            warmup_load_handler_added = True
        if _warmup_native_handler not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(_warmup_native_handler)
            warmup_depsgraph_handler_added = True
    except Exception:
        if redo_handler_added and (_infinite_redo_post_handler in
                bpy.app.handlers.redo_post):
            bpy.app.handlers.redo_post.remove(_infinite_redo_post_handler)
        if undo_handler_added and (_infinite_undo_post_handler in
                bpy.app.handlers.undo_post):
            bpy.app.handlers.undo_post.remove(_infinite_undo_post_handler)
        if warmup_depsgraph_handler_added and (_warmup_native_handler in
                bpy.app.handlers.depsgraph_update_post):
            bpy.app.handlers.depsgraph_update_post.remove(_warmup_native_handler)
        if warmup_load_handler_added and (_warmup_native_handler in
                bpy.app.handlers.load_post):
            bpy.app.handlers.load_post.remove(_warmup_native_handler)
        if load_handler_added and (_vertex_drag_load_post_handler in
                bpy.app.handlers.load_post):
            bpy.app.handlers.load_post.remove(_vertex_drag_load_post_handler)
        if vertex_drag_draw_handler_added:
            _vertex_drag_shutdown()
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
    global _stop_requested, _prepare_timer_registered
    cancel_auto_prepare()
    _stop_requested = True
    if not ensure_native_teardown(shutdown_runtime=True):
        print(
            "GPUCloth unregister retained handlers, classes, DLL, and "
            "owners because native teardown failed")
        return False
    try:
        if bpy.app.timers.is_registered(_advance_prepare_task):
            bpy.app.timers.unregister(_advance_prepare_task)
    except (AttributeError, RuntimeError):
        pass
    _prepare_timer_registered = False

    global _ogc_draw_handle
    # The vertex drag owns a modal handler, a timer, an overlay and at most
    # one live positional constraint.  Release all of them before the
    # operator classes they reference are unregistered.
    #
    # The infinite drive is stopped first: its timer calls the step core, which
    # dereferences g_dll and the native owners that the teardown just above has
    # already released, and a timer that survives unregister() would fire once
    # more from Blender's loop.
    _infinite_stop("the add-on was unregistered")
    _infinite_sim_state['checkpoint'] = None
    _infinite_sim_state['scene_name'] = None
    _vertex_drag_shutdown()
    for handler, collection in (
            (_infinite_undo_post_handler, bpy.app.handlers.undo_post),
            (_infinite_redo_post_handler, bpy.app.handlers.redo_post)):
        if handler in collection:
            collection.remove(handler)
    if _vertex_drag_load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_vertex_drag_load_post_handler)
    if _warmup_native_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_warmup_native_handler)
    if _warmup_native_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_warmup_native_handler)

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
    # Windows can release the ctypes handle at module teardown; POSIX keeps the
    # single process-resident CDLL reference deliberately (see GPUCloth_UnloadDLL).
    if _is_windows_host():
        g_dll = None
    _close_dll_directories()
    return True
