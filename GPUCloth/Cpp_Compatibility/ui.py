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
from ..utils.version_compatibility_utils import _t
from . import cloth_settings_bridge
from . import operators
from .proxy_binding import ProxyBindingError, validate_proxy_binding


# ===========================================================================
#  Icon names
# ===========================================================================
#
#  A layout call converts its ``icon`` argument through RNA, and converting a
#  name the running Blender does not have raises *inside* ``draw()``, which
#  Blender answers by not drawing the panel at all - the owner lost the whole
#  GPU Cloth panel to ``LOOP_FORWARD`` (this enum spells it ``LOOP_FORWARDS``;
#  the traceback was ui.py:204, the label below) rather than losing one glyph.
#  Every icon argument in this file therefore goes through ``_icon``: a name
#  this build has is passed through unchanged, anything else becomes ``NONE``,
#  and a literal that names an icon the running Blender lacks fails
#  tools/blender_ui_contract_gate.py instead of shipping.

_ICON_NAMES = None


def _icon_names():
    """The icon identifiers this Blender's layout RNA accepts, or None.

    Read from the same RNA the layout calls convert their argument with, so
    this is the authority rather than a second, staler copy of the list.  An
    enum that cannot be read caches as empty: ``_icon`` then answers ``NONE``
    for every name - a panel without glyphs is still a panel - and the UI
    contract audit fails on the unreadable enum, so a silent iconless UI is a
    red build and not a quiet regression.
    """
    global _ICON_NAMES
    if _ICON_NAMES is None:
        try:
            parameter = bpy.types.UILayout.bl_rna.functions[
                "label"].parameters["icon"]
            names = frozenset(item.identifier for item in parameter.enum_items)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            names = frozenset()
        _ICON_NAMES = names
    return _ICON_NAMES


def _icon(name):
    """``name`` if this Blender has that icon, else ``NONE`` (no icon).

    The glyph is decoration and the label is content, so an unknown name must
    cost the glyph and nothing else: this is the one place that decides it.
    """
    return name if name in _icon_names() else 'NONE'


# ===========================================================================
#  Main panel: GPU Cloth
#  Properties → Physics → GPU Cloth
# ===========================================================================

class GPUCLOTH_PT_main(bpy.types.Panel):
    bl_label       = "GPU Cloth"
    bl_idname      = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == 'MESH'
        )

    def draw(self, context):
        layout = self.layout
        obj    = context.object
        scene  = context.scene

        if not hasattr(obj, 'GPUCloth'):
            layout.label(
                text=_t("Enable GPUCloth for this object",
                        "Включите GPUCloth для этого объекта"),
                icon=_icon('INFO'))
            return

        settings = obj.GPUCloth
        layout.prop(settings, "execution_backend", expand=True)
        cpu_modifier = cloth_settings_bridge.find_cpu_cloth_modifier(obj)
        row = layout.row(align=True)
        row.enabled = cpu_modifier is not None
        row.operator(
            "gpucloth.sync_cpu_settings", text="", icon=_icon('FILE_REFRESH'))
        if cpu_modifier is None:
            row.label(
                text=_t(
                    "Cloth will be created on GPU activation",
                    "Cloth будет создан при включении GPU"),
                icon=_icon('INFO'))
        elif settings.cpu_sync_errors or settings.cpu_sync_blockers:
            if settings.cpu_sync_errors:
                row.label(
                    text=_t(
                        f"{settings.cpu_sync_errors} errors",
                        f"Ошибок: {settings.cpu_sync_errors}"),
                    icon=_icon('ERROR'))
            if settings.cpu_sync_blockers:
                row.label(
                    text=_t(
                        f"{settings.cpu_sync_blockers} blocking",
                        f"Блокирует: {settings.cpu_sync_blockers}"),
                    icon=_icon('ERROR'))
        else:
            row.label(
                text=_t(
                    f"{settings.cpu_sync_copied} imported",
                    f"Перенесено: {settings.cpu_sync_copied}"),
                icon=_icon('CHECKMARK'))
            if settings.cpu_sync_unsupported:
                row.label(
                    text=_t(
                        f"{settings.cpu_sync_unsupported} unsupported",
                        f"Не поддержано: {settings.cpu_sync_unsupported}"),
                    icon=_icon('QUESTION'))
        if settings.cpu_sync_report:
            row.operator(
                "gpucloth.show_cpu_sync_report", text="", icon=_icon('INFO'))

        if settings.execution_backend != 'GPU' or not settings.is_active:
            return

        prepared = (
            scene.gpu_cloth_springs_built
            and obj in operators.g_clothOBJs)
        preparing = (operators.prepare_task_active(obj)
                     or operators.resimulating())

        layout.separator()

        status = layout.row(align=True)
        status.label(
            text=_t("Ready", "Готово") if prepared else
                 (_t("Preparing", "Подготовка") if preparing else
                  _t("Not prepared", "Не подготовлено")),
            icon=_icon(
                'CHECKMARK' if prepared else
                ('TIME' if preparing else 'INFO')))
        prepare = status.row(align=True)
        prepare.enabled = not prepared and not preparing
        prepare.operator(
            "gpucloth.prepare_simulation",
            text=_t("Prepare", "Подготовить"), icon=_icon('PLAY'))
        layout.prop(
            settings, "auto_prepare",
            text=_t("Auto Prepare", "Автоподготовка"))
        helper = scene.gpu_cloth_helper
        if helper.prepare_state in {'ERROR', 'CANCELLED'}:
            layout.label(
                text=helper.prepare_status,
                icon=_icon(
                    'ERROR' if helper.prepare_state == 'ERROR' else 'CANCEL'))
        if getattr(scene.gpu_cloth_helper, "memory_preflight_status", ""):
            layout.label(
                text=scene.gpu_cloth_helper.memory_preflight_status,
                icon=(
                    _icon('CHECKMARK'
                    if scene.gpu_cloth_helper.memory_preflight_status.startswith(
                        "PASS:")
                    else 'ERROR')))
        stop = status.row(align=True)
        stop.enabled = prepared or operators.auto_prepare_pending(obj)
        stop.operator(
            "gpucloth.destroy_simulation_data",
            text=_t("Stop", "Остановить"), icon=_icon('CANCEL'))

        # The playhead on a frame nothing has produced.  A move of the playhead
        # is a request to look, not a request to simulate, so the frame path
        # leaves the mesh exactly as it was - and an unchanged mesh reads as
        # "this is the simulation" unless the panel says otherwise.  One row,
        # from the one function that asks the same question the frame handler
        # asks; it disappears by itself on the next frame that is produced.
        if prepared:
            frame_status = operators.playhead_frame_status(scene)
            if frame_status:
                layout.label(text=_t(*frame_status), icon=_icon('INFO'))

        # Preparation progress, in the shape bake already uses for the same job
        # (Cache panel: one slider carrying its own percentage and phase text).
        # The value is the phase the preparation generator published and
        # nothing else - operators._set_prepare_status writes it at a phase
        # boundary (operators.py:8338-8817) and the driver tags the redraw - so
        # a bar that has moved is a phase that finished.  This readout replaces
        # the percentage that used to be drawn next to the mouse pointer:
        # wm.progress_* drew those digits, and no add-on-drivable bar exists at
        # the bottom of this build's window.
        #
        # It sits here, above the boxes this panel closes with, and not as the
        # panel's last row: a row appended after the last box is not measured
        # into the panel, so the first sub-panel header ("Prepare / Drape") is
        # drawn over it (measured in a live 5.2.1 session, receipts in
        # build/ui-r21/verify).  Below this row the panel keeps the same
        # element order it already rendered correctly.
        if preparing:
            progress = layout.row()
            progress.prop(
                helper, "prepare_progress",
                text=helper.prepare_status or _t("Preparing", "Подготовка"),
                slider=True)

        # MD-style grab: armed here, active only while the simulation plays.
        tool = operators.vertex_drag_tool_state(context)
        box = layout.box()
        box.label(
            text=_t("Move Cloth By Vertex (MD-style)",
                    "Перемещение ткани за вершину (MD)"),
            icon=_icon('HAND'))
        button = box.row(align=True)
        button.operator(
            "gpucloth.move_cloth_by_vertex",
            text=_t("Move Cloth By Vertex (MD-style)",
                    "Перемещение ткани за вершину (MD)"),
            icon=_icon('CANCEL' if tool["armed"] else 'TRIA_RIGHT'),
            depress=tool["armed"])
        if not tool["armed"]:
            box.label(
                text=(
                    _t("Inactive - arm it, then play and drag a vertex",
                       "Выключено - включите, запустите анимацию и тяните")
                    if tool["available"] else
                    _t("Inactive - prepare the simulation first",
                       "Выключено - сначала выполните Prepare")),
                icon=_icon('INFO'))
        elif tool["dragging"]:
            box.label(
                text=_t(
                    f"Dragging vertex {tool['vertex_index']} of "
                    f"{tool['object_name']}",
                    f"Тянем вершину {tool['vertex_index']} объекта "
                    f"{tool['object_name']}"),
                icon=_icon('PLAY'))
        elif tool["live"]:
            box.label(
                text=_t("Active - click a cloth vertex and drag it",
                        "Активно - потяните вершину ткани"),
                icon=_icon('PLAY'))
        elif tool["playing"]:
            box.label(
                text=_t(
                    "Waiting - the timeline is replaying frames the "
                    "simulation already solved; the drag resumes when the "
                    "simulation steps again",
                    "Ожидание - таймлайн воспроизводит уже просчитанные "
                    "кадры; перетаскивание возобновится, когда симуляция "
                    "снова начнёт считать"),
                icon=_icon('PAUSE'))
        else:
            box.label(
                text=_t("Active - waiting for playback",
                        "Активно - ожидание воспроизведения"),
                icon=_icon('PAUSE'))

        # Infinite simulation: the same armed tool, driven by its own step
        # timer instead of the timeline.  Space has exactly one meaning here
        # and the row below states it; the mode reports its own step counter,
        # and never borrows bake_progress (that field belongs to the bake and
        # the Cache panel keys the Bake button on it).
        infinite = operators.infinite_sim_state(context)
        box.separator()
        box.label(
            text=_t("Infinite simulation (no animation)",
                    "Бесконечная симуляция (без анимации)"),
            icon=_icon(
                'LOOP_FORWARDS' if infinite["running"] else 'LOOP_BACK'))
        if infinite["running"]:
            box.label(
                text=_t(
                    f"Running - step {infinite['steps']} (frame frozen at "
                    f"{infinite['frame']})",
                    f"Идёт - шаг {infinite['steps']} (кадр заморожен на "
                    f"{infinite['frame']})"),
                icon=_icon('PLAY'))
        elif infinite["has_checkpoint"]:
            box.label(
                text=_t(
                    f"Stopped - {infinite['steps']} step(s) since the "
                    f"checkpoint",
                    f"Остановлено - {infinite['steps']} шаг(ов) с момента "
                    f"чекпоинта"),
                icon=_icon('PAUSE'))
        capability = infinite["capable"]
        if not capability.get("ok"):
            box.label(text=capability["message"][0], icon=_icon('ERROR'))
        elif not tool["armed"]:
            box.label(
                text=_t("Arm Move Cloth By Vertex to use Space",
                        "Включите Move Cloth By Vertex, чтобы работать "
                        "пробелом"),
                icon=_icon('INFO'))
        box.label(text=infinite["space_hint"][0], icon=_icon('INFO'))
        # The control the mode was missing: Space only reaches the vertex-drag tool's
        # modal, so with the pointer anywhere else this loop could be started and not
        # stopped.  Offered exactly when the mode would accept a start, and always
        # offered while it runs.
        if infinite["running"] or capability.get("ok"):
            box.operator(
                "gpucloth.toggle_infinite_simulation",
                text=(_t("Stop infinite simulation",
                         "Остановить бесконечную симуляцию")
                      if infinite["running"] else
                      _t("Start infinite simulation",
                         "Запустить бесконечную симуляцию")),
                icon=_icon('PAUSE' if infinite["running"] else 'PLAY'))
        if infinite["status"]:
            box.label(text=infinite["status"], icon=_icon('INFO'))


# ===========================================================================
#  Sub-panel: Solver
# ===========================================================================

class GPUCLOTH_PT_preparation(bpy.types.Panel):
    bl_label = "Prepare / Drape"
    bl_idname = "GPUCLOTH_PT_preparation"
    bl_parent_id = "GPUCLOTH_PT_main"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "physics"

    @classmethod
    def poll(cls, context):
        return bool(
            context.object is not None and
            context.object.type == 'MESH' and
            hasattr(context.object, 'GPUCloth') and
            context.object.GPUCloth.is_active)

    def draw(self, context):
        layout = self.layout
        obj = context.object
        prepared = bool(
            context.scene.gpu_cloth_springs_built and
            obj in operators.g_clothOBJs)
        layout.enabled = prepared
        if not prepared:
            layout.label(text=_t(
                "Prepare simulation first", "Сначала выполните Prepare"),
                icon=_icon('INFO'))
            return

        preparation = operators.get_preparation_ui_status(obj)
        if preparation is not None:
            layout.label(
                text=(f"Prepare generation "
                      f"{preparation['preparation_generation']}; "
                      f"accepted {preparation['accepted_generation']}"),
                icon=_icon('CHECKMARK'))

        drape = operators.get_drape_ui_status(obj)
        flags = drape["status_flags"] if drape else 0
        active = bool(flags & operators.CType.GPUCLOTH_DRAPE_STATUS_ACTIVE)
        converged = bool(
            flags & operators.CType.GPUCLOTH_DRAPE_STATUS_CONVERGED)
        failed = bool(flags & operators.CType.GPUCLOTH_DRAPE_STATUS_FAILED)
        applied = bool(flags & operators.CType.GPUCLOTH_DRAPE_STATUS_APPLIED)
        # The drape owns its criterion: a position tolerance in metres and a
        # budget in simulated seconds.  The solver's force tolerance is not it,
        # and the native 240-step cap is only the sandbox's own limit.
        tolerance, budget_s = operators.drape_criterion(obj, context.scene)
        frame_seconds = operators.drape_frame_seconds(context.scene)
        settling = operators.drape_settle_running(context)
        verdict = operators.get_drape_settle_verdict(obj)

        if drape:
            box = layout.box()
            delta_mm = drape["maximum_position_delta"] * 1000.0
            tolerance_mm = tolerance * 1000.0
            simulated = drape["step_count"] * frame_seconds
            if settling:
                box.label(
                    text=(f"Settling: {simulated:.1f} / {budget_s:.1f} s "
                          f"simulated (step {drape['step_count']})"),
                    icon=_icon('TIME'))
            elif verdict is not None and not verdict.get("settled"):
                box.label(
                    text=(f"Drape did not settle: still moving "
                          f"{verdict['maximum_position_delta_m'] * 1000.0:.1f} "
                          f"mm/frame against {tolerance_mm:.1f} mm/frame after "
                          f"{verdict['simulated_s']:.1f} s"),
                    icon=_icon('INFO'))
            elif verdict is not None and verdict.get("settled"):
                box.label(
                    text=(f"Drape settled after {verdict['simulated_s']:.1f} s "
                          f"of simulated drape"),
                    icon=_icon('CHECKMARK'))
            else:
                box.label(
                    text=(f"Drape step {drape['step_count']} "
                          f"({simulated:.1f} s simulated)"),
                    icon=_icon(
                        'ERROR' if failed and
                        not operators.drape_not_settled(drape) else
                        ('CHECKMARK' if converged else 'TIME')))
            box.label(
                text=(f"L∞ {delta_mm:.2f} mm/frame; drape tol "
                      f"{tolerance_mm:.2f} mm/frame; budget {budget_s:.1f} s; "
                      f"step {drape['step_count']}"))
            criterion = box.column(align=True)
            criterion.prop(
                obj.GPUCloth, "drape_position_tolerance",
                text="Drape tol (m/frame)")
            criterion.prop(
                obj.GPUCloth, "drape_budget_s", text="Drape budget (s)")

        row = layout.row(align=True)
        if not active and not failed and not applied:
            row.operator(
                "gpucloth.begin_drape", text="Begin", icon=_icon('PLAY'))
        if (active or (drape and operators.drape_not_settled(drape))) and (
                not converged):
            row.operator(
                "gpucloth.step_drape", text="Step", icon=_icon('FRAME_NEXT'))
            settle = row.operator(
                "gpucloth.step_drape", text="Settle", icon=_icon('PLAY'))
            settle.until_settled = True
        if converged:
            row.operator(
                "gpucloth.apply_drape", text="Apply", icon=_icon('CHECKMARK'))
        if active or failed:
            row.operator(
                "gpucloth.cancel_drape", text="Cancel", icon=_icon('X'))

        witness = operators.get_invariant_ui_status(obj)
        if witness is not None:
            box = layout.box()
            # NOT_CONVERGED is emitted by the drape step budget alone; it names
            # no broken invariant, so it must not be dressed as one.
            budget_exhausted = (
                witness["invariant"] ==
                operators.CType.GPUCLOTH_INVARIANT_NOT_CONVERGED)
            box.label(
                text=("No invariant broken: the drape is still moving"
                      if budget_exhausted else
                      f"Invariant: {witness['invariant_name']}"),
                icon=_icon(
                    'CHECKMARK' if witness["invariant"] ==
                    operators.CType.GPUCLOTH_INVARIANT_NONE else
                    ('INFO' if budget_exhausted else 'ERROR')))
            primitives = witness["primitives"]
            if witness["invariant"] not in (
                    operators.CType.GPUCLOTH_INVARIANT_NONE,
                    operators.CType.GPUCLOTH_INVARIANT_NOT_CONVERGED):
                box.label(
                    text=(f"tri {primitives['triangle_i']} / "
                          f"{primitives['triangle_j']}; "
                          f"edge {primitives['edge_i']}; "
                          f"vertex {primitives['vertex_i']}"))
            row = box.row(align=True)
            row.operator(
                "gpucloth.copy_invariant_diagnostics",
                text="Copy JSON", icon=_icon('COPYDOWN'))
            row.operator(
                "gpucloth.save_invariant_diagnostics",
                text="Save JSON", icon=_icon('FILE_TICK'))


class GPUCLOTH_PT_solver(bpy.types.Panel):
    bl_label       = "Solver"
    bl_idname      = "GPUCLOTH_PT_solver"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == 'MESH'
            and hasattr(context.object, 'GPUCloth')
            and context.object.GPUCloth.is_active
        )

    def draw(self, context):
        layout  = self.layout
        s       = context.object.GPUCloth
        scene_s = context.scene.gpu_cloth_helper

        col = layout.column()
        col.prop(s, "solver_type")
        col.separator()

        # FABRIC is the only material model, so its physical-membrane preset is
        # the one the Solver panel offers; `material_preset` (the isotropic
        # stiffness/damping mirror the TestScene fixtures write) is not a user
        # choice any more and is no longer exposed.
        col.prop(s, "fabric_preset")
        col.separator()

        col.prop(s, "quality_step")
        col.prop(s, "speed_multiplier")
        col.prop(s, "bending_model")

        col.separator()
        col.label(text=_t("Gravity (m/s\u00b2):", "Гравитация (м/с\u00b2):"))
        row = col.row(align=True)
        row.prop(scene_s, "gravity_x", text="X")
        row.prop(scene_s, "gravity_y", text="Y")
        row.prop(scene_s, "gravity_z", text="Z")


# ===========================================================================
#  Sub-panel: Material (stiffness & damping)
# ===========================================================================

class GPUCLOTH_PT_material(bpy.types.Panel):
    bl_label       = "Material"
    bl_idname      = "GPUCLOTH_PT_material"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == 'MESH'
            and hasattr(context.object, 'GPUCloth')
            and context.object.GPUCloth.is_active
        )

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth

        col = layout.column(align=True)
        if s.solver_type == 'PD':
            col.label(text=_t("Fabric Physical Membrane (N/m):", "Физическая мембрана ткани (Н/м):"))
            for name, label in (
                    ("fabric_tensile_u", "Tensile U"),
                    ("fabric_tensile_v", "Tensile V"),
                    ("fabric_compression_u", "Compression U"),
                    ("fabric_compression_v", "Compression V"),
                    ("fabric_shear_c66", "Shear C66")):
                col.prop(s, name, text=_t(label, label))
            col.separator()
            col.label(text=_t("Fabric Maximums (N/m):", "Максимумы ткани (Н/м):"))
            for name, label in (
                    ("fabric_tensile_u_max", "Max Tensile U"),
                    ("fabric_tensile_v_max", "Max Tensile V"),
                    ("fabric_compression_u_max", "Max Compression U"),
                    ("fabric_compression_v_max", "Max Compression V"),
                    ("fabric_shear_c66_max", "Max Shear C66")):
                col.prop(s, name, text=_t(label, label))
            col.label(text=_t("Fabric Damping (N·s/m):", "Демпфирование ткани (Н·с/м):"))
            for name, label in (
                    ("fabric_tensile_damping", "Tensile Damping"),
                    ("fabric_compression_damping", "Compression Damping"),
                    ("fabric_shear_damping", "Shear Damping")):
                col.prop(s, name, text=_t(label, label))
            layout.separator()
        col.label(text=_t("Stiffness:", "Жёсткость:"))
        col.prop(s, "tension",           text=_t("Tension",     "Растяжение"))
        col.prop(s, "compression",       text=_t("Compression", "Сжатие"))
        col.prop(s, "shear",             text=_t("Shear",       "Сдвиг"))
        col.prop(s, "bending_stiffness", text=_t("Bending",     "Изгиб"))

        # ── Anisotropy ─────────────────────────────────────────────────────
        layout.separator()
        col = layout.column(align=True)
        col.prop(s, "use_anisotropy", text=_t("Anisotropic Stiffness", "Анизотропная жёсткость"))
        if s.use_anisotropy:
            simulation_data = (
                s.proxy_object.data
                if s.use_proxy and s.proxy_object is not None
                else context.object.data)
            col.prop_search(
                s, "anisotropy_uv_map", simulation_data, "uv_layers",
                text=_t("Material UV", "UV материала"))
            col.prop(s, "tension_u",    text=_t("Tension U (Warp)", "Растяжение U"))
            col.prop(s, "tension_v",    text=_t("Tension V (Weft)", "Растяжение V"))
            col.prop(s, "compression_u", text=_t("Compression U (Warp)", "Сжатие U"))
            col.prop(s, "compression_v", text=_t("Compression V (Weft)", "Сжатие V"))
            col.prop(s, "bending_u",    text=_t("Bending U (Warp)", "Изгиб U"))
            col.prop(s, "bending_v",    text=_t("Bending V (Weft)", "Изгиб V"))
            col.separator()
            col.prop(s, "max_tension_u", text=_t("Max Tension U", "Макс. растяжение U"))
            col.prop(s, "max_tension_v", text=_t("Max Tension V", "Макс. растяжение V"))
            col.prop(s, "max_compression_u", text=_t("Max Compression U", "Макс. сжатие U"))
            col.prop(s, "max_compression_v", text=_t("Max Compression V", "Макс. сжатие V"))
            col.prop(s, "max_bend_u", text=_t("Max Bending U", "Макс. изгиб U"))
            col.prop(s, "max_bend_v", text=_t("Max Bending V", "Макс. изгиб V"))

        layout.separator()

        col = layout.column(align=True)
        col.label(text=_t("Damping:", "Демпфирование:"))
        col.prop(s, "tension_damp",     text=_t("Tension",     "Растяжение"))
        col.prop(s, "compression_damp", text=_t("Compression", "Сжатие"))
        col.prop(s, "shear_damp",       text=_t("Shear",       "Сдвиг"))
        col.prop(s, "bending_damping",  text=_t("Bending",     "Изгиб"))


# ===========================================================================
#  Sub-panel: Physical Properties (clamping, advanced damping)
# ===========================================================================

class GPUCLOTH_PT_physical(bpy.types.Panel):
    bl_label       = "Physical Properties"
    bl_idname      = "GPUCLOTH_PT_physical"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and hasattr(obj, 'GPUCloth') and obj.GPUCloth.is_active

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth

        col = layout.column(align=True)
        # FABRIC is the only material model, so the mass input is always the
        # areal fabric density; `mass_mode` no longer selects a branch, and the
        # per-vertex mass stays available as the resolved value the engine
        # receives (`_configure_simulation_features` publishes AREAL mass).
        fabric_mode = s.solver_type == 'PD'
        if fabric_mode:
            col.prop(s, "fabric_density",
                     text=_t("Fabric Density (g/m²)", "Плотность ткани (г/м²)"))
        else:
            col.prop(s, "mass_mode", text=_t("Mass Mode", "Режим массы"))
        if not fabric_mode and s.mass_mode == 'AREAL':
            col.prop(s, "fabric_density",
                     text=_t("Fabric Density (g/m²)", "Плотность ткани (г/м²)"))
        elif not fabric_mode:
            col.prop(s, "vertex_mass",
                     text=_t("Vertex Mass", "Масса вершины"))

        layout.separator()
        col = layout.column(align=True)
        col.prop(s, "air_viscosity",  text=_t("Air Viscosity",  "Вязкость воздуха"))
        col.prop(s, "vel_damping",    text=_t("Velocity Damping", "Демпфирование скорости"))

        layout.separator()
        col = layout.column(align=True)
        col.label(text=_t("Stiffness Clamping:", "Ограничение жёсткости:"))
        col.prop(s, "max_tension",    text=_t("Max Tension",     "Макс. растяжение"))
        col.prop(s, "max_compression",text=_t("Max Compression", "Макс. сжатие"))
        col.prop(s, "max_shear",      text=_t("Max Shear",       "Макс. сдвиг"))
        col.prop(s, "max_bend",       text=_t("Max Bending",     "Макс. изгиб"))
        col.prop(s, "max_sewing",     text=_t("Max Sewing",      "Макс. шов"))
        col.prop(s, "use_sewing_springs",
                 text=_t("Sew Cloth", "Сшивать ткань"))


# ===========================================================================
#  Sub-panel: Internal Springs
# ===========================================================================

class GPUCLOTH_PT_internal_springs(bpy.types.Panel):
    bl_label       = "Internal Springs"
    bl_idname      = "GPUCLOTH_PT_internal_springs"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and hasattr(obj, 'GPUCloth') and obj.GPUCloth.is_active

    def draw_header(self, context):
        self.layout.prop(context.object.GPUCloth, "use_internal_springs", text="")

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth
        layout.active = s.use_internal_springs

        col = layout.column(align=True)
        col.prop(s, "use_internal_springs_normal",
                 text=_t("Check Surface Normals", "Проверять нормали поверхности"))

        layout.separator()
        col = layout.column(align=True)
        col.prop(s, "internal_spring_max_length",
                 text=_t("Max Spring Length", "Макс. длина пружины"))
        col.prop(s, "internal_spring_max_diversion",
                 text=_t("Max Normal Diversion", "Макс. отклонение от нормали"))

        layout.separator()
        col = layout.column(align=True)
        col.label(text=_t("Stiffness:", "Жёсткость:"))
        col.prop(s, "internal_tension",
                 text=_t("Tension", "Растяжение"))
        col.prop(s, "internal_compression",
                 text=_t("Compression", "Сжатие"))

        col.separator()
        col.prop(s, "max_internal_tension",
                 text=_t("Max Tension", "Макс. растяжение"))
        col.prop(s, "max_internal_compression",
                 text=_t("Max Compression", "Макс. сжатие"))

        layout.separator()
        col = layout.column()
        col.prop_search(s, "vgroup_intern", context.object,
                        "vertex_groups",
                        text=_t("Vertex Group", "Группа вершин"))


# ===========================================================================
#  Sub-panel: Pressure
# ===========================================================================

class GPUCLOTH_PT_pressure(bpy.types.Panel):
    bl_label       = "Pressure"
    bl_idname      = "GPUCLOTH_PT_pressure"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and hasattr(obj, 'GPUCloth') and obj.GPUCloth.is_active

    def draw_header(self, context):
        self.layout.prop(context.object.GPUCloth, "use_pressure", text="")

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth
        layout.active = s.use_pressure

        col = layout.column(align=True)
        col.prop(s, "uniform_pressure_force",
                 text=_t("Pressure", "Давление"))
        col.prop(s, "pressure_factor",
                 text=_t("Factor", "Множитель"))
        col.prop(s, "use_pressure_volume",
                 text=_t("Use Custom Volume", "Использовать заданный объём"))
        volume_col = col.column()
        volume_col.active = s.use_pressure_volume
        volume_col.prop(s, "target_volume",
                        text=_t("Target Volume", "Целевой объём"))
        col.prop(s, "fluid_density",
                 text=_t("Fluid Density", "Плотность флюида"))

        layout.separator()
        col = layout.column()
        col.prop_search(s, "vgroup_pressure", context.object,
                        "vertex_groups",
                        text=_t("Vertex Group", "Группа вершин"))


# ===========================================================================
#  Sub-panel: Shape / Pinning
# ===========================================================================

class GPUCLOTH_PT_shape(bpy.types.Panel):
    bl_label       = "Shape"
    bl_idname      = "GPUCLOTH_PT_shape"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and hasattr(obj, 'GPUCloth') and obj.GPUCloth.is_active

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth

        col = layout.column(align=True)
        col.prop_search(s, "vgroup_mass", context.object,
                        "vertex_groups",
                        text=_t("Pin Group", "Группа закрепления"))

        if s.vgroup_mass:
            col.prop(s, "goalspring",
                     text=_t("Stiffness", "Жёсткость"))
            col.prop(s, "goalfrict",
                     text=_t("Friction", "Трение"))
            col.prop(s, "mingoal",
                     text=_t("Min Goal", "Мин. цель"))
            col.prop(s, "maxgoal",
                     text=_t("Max Goal", "Макс. цель"))
            col.prop(s, "defgoal",
                     text=_t("Default Goal", "Цель по умолчанию"))

        layout.separator()
        col = layout.column(align=True)
        col.label(text=_t("Shrinking:", "Сжатие:"))
        col.prop(s, "shrink_min",
                 text=_t("Min", "Мин."))
        col.prop(s, "shrink_max",
                 text=_t("Max", "Макс."))

        layout.separator()
        col = layout.column()
        col.prop(s, "use_dynamic_mesh",
                 text=_t("Dynamic Mesh", "Динамический меш"),
                 toggle=True)

        if context.object.data.shape_keys:
            col.prop_search(s, "shapekey_rest",
                            context.object.data.shape_keys,
                            "key_blocks",
                            text=_t("Rest Shape Key", "Ключ формы покоя"))


# ===========================================================================
#  Sub-panel: Object Collision
# ===========================================================================

class GPUCLOTH_PT_object_collision(bpy.types.Panel):
    bl_label       = "Object Collision"
    bl_idname      = "GPUCLOTH_PT_object_collision"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and hasattr(obj, 'GPUCloth') and obj.GPUCloth.is_active

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth

        layout.prop(s, "use_object_collision", text="")
        body = layout.column()
        body.active = s.use_object_collision
        row = body.row(align=True)
        row.prop(s, "collision_quality")
        row.prop(s, "collision_friction")
        row.prop(s, "collision_damping")

        col = body.column(align=True)
        col.label(text=_t("Distance:", "Дистанция:"))
        col.prop(s, "epsilon",
                 text=_t("Object", "Объект"))
        col.prop(s, "selfepsilon",
                 text=_t("Self", "Самоколлизия"))

        body.separator()
        col = body.column(align=True)
        col.label(text=_t("Impulse Clamping:", "Ограничение импульса:"))
        col.prop(s, "clamp",
                 text=_t("Object", "Объект"))
        col.prop(s, "self_clamp",
                 text=_t("Self", "Самоколлизия"))

        body.separator()
        col = body.column()
        col.prop(s, "collision_collection",
                 text=_t("Collision Collection", "Коллекция коллизий"))

        col.separator()
        col.prop_search(s, "vgroup_objcol", context.object,
                        "vertex_groups",
                        text=_t("Exclude Objects", "Исключить объекты"))
        col.prop_search(s, "vgroup_selfcol", context.object,
                        "vertex_groups",
                        text=_t("Exclude Self", "Исключить самоколлизию"))


# ===========================================================================
#  Sub-panel: Property Weights (stiffness scaling vertex groups)
# ===========================================================================

class GPUCLOTH_PT_property_weights(bpy.types.Panel):
    bl_label       = "Property Weights"
    bl_idname      = "GPUCLOTH_PT_property_weights"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and hasattr(obj, 'GPUCloth') and obj.GPUCloth.is_active

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth

        col = layout.column(align=True)
        col.label(text=_t("Stiffness Scaling Groups:",
                          "Группы масштабирования жёсткости:"))
        col.prop_search(s, "vgroup_struct", context.object,
                        "vertex_groups",
                        text=_t("Structural", "Структурная"))
        col.prop_search(s, "vgroup_shear", context.object,
                        "vertex_groups",
                        text=_t("Shear", "Сдвиг"))
        col.prop_search(s, "vgroup_bend", context.object,
                        "vertex_groups",
                        text=_t("Bending", "Изгиб"))
        col.prop_search(s, "vgroup_shrink", context.object,
                        "vertex_groups",
                        text=_t("Shrinking", "Сжатие"))


# ===========================================================================
#  Sub-panel: Field Weights
# ===========================================================================

class GPUCLOTH_PT_field_weights(bpy.types.Panel):
    bl_label       = "Field Weights"
    bl_idname      = "GPUCLOTH_PT_field_weights"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and hasattr(obj, 'GPUCloth') and obj.GPUCloth.is_active

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth
        fw     = s.effector_weights
        layout.prop(fw, "collection", text="Collection")

        col = layout.column(align=True)
        col.prop(s, "eff_force_scale",
                 text=_t("Effector Force", "Сила эффектора"))
        col.prop(s, "eff_wind_scale",
                 text=_t("Effector Wind", "Сила ветра"))

        layout.separator()
        col = layout.column(align=True)
        col.label(text=_t("Field Weights:", "Веса полей:"))
        col.prop(fw, "global_gravity",
                 text=_t("Gravity", "Гравитация"))
        col.prop(fw, "weight_all",
                 text=_t("All", "Все"))
        col.prop(fw, "weight_force",
                 text=_t("Force", "Сила"))
        col.prop(fw, "weight_vortex",
                 text=_t("Vortex", "Вихрь"))
        col.prop(fw, "weight_magnetic",
                 text=_t("Magnetic", "Магнитное"))
        col.prop(fw, "weight_wind",
                 text=_t("Wind", "Ветер"))
        col.prop(fw, "weight_curve_guide",
                 text=_t("Curve Guide", "Напр. кривой"))
        col.prop(fw, "weight_texture",
                 text=_t("Texture", "Текстура"))
        col.prop(fw, "weight_harmonic",
                 text=_t("Harmonic", "Гармоническое"))
        col.prop(fw, "weight_charge",
                 text=_t("Charge", "Заряд"))
        col.prop(fw, "weight_lennard_jones",
                 text=_t("Lennard-Jones", "Леннард-Джонс"))
        col.prop(fw, "weight_boid",
                 text=_t("Boid", "Боид"))
        col.prop(fw, "weight_turbulence",
                 text=_t("Turbulence", "Турбулентность"))
        col.prop(fw, "weight_drag",
                 text=_t("Drag", "Торможение"))
        col.prop(fw, "weight_smoke_flow",
                 text=_t("Fluid Flow", "Поток жидкости"))


# ===========================================================================
#  Sub-panel: Self-Collision (OGC)
# ===========================================================================

class GPUCLOTH_PT_collision(bpy.types.Panel):
    bl_label       = "Self-Collision (OGC)"
    bl_idname      = "GPUCLOTH_PT_collision"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == 'MESH'
            and hasattr(context.object, 'GPUCloth')
            and context.object.GPUCloth.is_active
        )

    def draw_header(self, context):
        self.layout.prop(context.object.GPUCloth, "use_self_collision", text="")

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth
        layout.active = s.use_self_collision

        col = layout.column(align=True)
        col.prop(s, "ogc_radius")
        col.prop(s, "ogc_kc")
        col.prop(s, "ogc_friction")
        col.prop(s, "ogc_gamma_p")
        col.prop(
            s, "self_collision_friction",
            text=_t("Blender Self Friction", "Трение самоколлизии Blender"))

        layout.separator()

        row = layout.row()
        row.prop(
            s, "show_ogc_bounds",
            text=_t("Show Contact Bounds", "Показать границы коллизии"),
            toggle=True,
            icon=_icon('SPHERE'),
        )


# ===========================================================================
#  Sub-panel: Proxy-res simulation
# ===========================================================================

class GPUCLOTH_PT_proxy(bpy.types.Panel):
    bl_label       = "Proxy Simulation"
    bl_idname      = "GPUCLOTH_PT_proxy"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == 'MESH'
            and hasattr(context.object, 'GPUCloth')
            and context.object.GPUCloth.is_active
        )

    def draw_header(self, context):
        self.layout.prop(context.object.GPUCloth, "use_proxy", text="")

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth
        layout.active = s.use_proxy

        col = layout.column()
        col.prop(s, "proxy_object")

        if s.proxy_object is None:
            col.label(
                text=_t("Select a coarse mesh object",
                        "Выберите объект с грубой сеткой"),
                icon=_icon('ERROR' if s.use_proxy else 'INFO'),
            )
        elif s.use_proxy:
            try:
                binding = validate_proxy_binding(context.object, s)
                col.label(
                    text=(
                        f"{binding['simulation_vertex_count']} -> "
                        f"{binding['render_vertex_count']} vertices"),
                    icon=_icon('CHECKMARK'),
                )
            except ProxyBindingError:
                col.label(
                    text=_t("Grid counts do not match meshes",
                            "Размеры сетки не совпадают с мешами"),
                    icon=_icon('ERROR'),
                )

        col.separator()

        box = col.box()
        box.label(text=_t("Proxy grid (coarse):", "Proxy-сетка (грубая):"),
                  icon=_icon('MESH_GRID'))
        row = box.row(align=True)
        row.prop(s, "proxy_nx", text="NX")
        row.prop(s, "proxy_ny", text="NY")

        box = col.box()
        box.label(text=_t("Hi-res grid (render):", "Hi-res сетка (рендер):"),
                  icon=_icon('MESH_GRID'))
        row = box.row(align=True)
        row.prop(s, "hi_nx", text="NX")
        row.prop(s, "hi_ny", text="NY")

        col.separator()
        col.prop(s, "num_sheets")
        col.prop(s, "proxy_scene_type")


# ===========================================================================
#  Sub-panel: Simulation cache
# ===========================================================================

def _cache_store_detail(store):
    """One line naming exactly what Clear Cache would delete.

    Defect 7 asked for cache management in this tab and got a button that was
    only drawn for a baked store and, when drawn, cleared less than the owner
    means by the cache.  The row therefore states its own contents - the files
    and their size on disk, where they are, and what the native owner and this
    session hold beside them - so pressing it is an informed action and not a
    bare button.  ``operators.cache_store_summary`` owns the numbers; this owns
    only the wording, in the panel's own two languages.
    """
    where = store["path"] or _t("memory only", "только в памяти")
    return _t(
        f"Clear: {store['disk_files']} frame file(s), "
        f"{store['disk_bytes'] / (1024.0 * 1024.0):.2f} MiB in {where}; "
        f"{store['native_frames']} native frame(s), "
        f"{store['retained_states']} reached frame(s) in memory",
        f"Очистить: файлов кадров {store['disk_files']}, "
        f"{store['disk_bytes'] / (1024.0 * 1024.0):.2f} МиБ в {where}; "
        f"кадров в native {store['native_frames']}, "
        f"достигнутых кадров в памяти {store['retained_states']}")


class GPUCLOTH_PT_cache(bpy.types.Panel):
    bl_label       = "Cache"
    bl_idname      = "GPUCLOTH_PT_cache"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == 'MESH'
            and hasattr(context.object, 'GPUCloth')
            and context.object.GPUCloth.is_active
        )

    def draw(self, context):
        layout  = self.layout
        scene_s = context.scene.gpu_cloth_helper

        col = layout.column()
        col.prop(scene_s, "cache_dir")

        row = col.row(align=True)
        row.prop(scene_s, "bake_start",
                 text=_t("Start", "От"))
        row.prop(scene_s, "bake_end",
                 text=_t("End", "До"))

        col.separator()

        if scene_s.is_baked:
            col.label(
                text=_t("Status: baked", "Статус: запечено"),
                icon=_icon('CHECKMARK'))

            col.prop(scene_s, "playback_mode",
                     text=_t("Play from Cache", "Воспроизводить из кэша"),
                     toggle=True,
                     icon=_icon('PLAY' if scene_s.playback_mode else 'PAUSE'))

            col.separator()

            box = col.box()
            box.label(
                text=_t("Export Simulation:", "Экспорт симуляции:"),
                icon=_icon('EXPORT'))
            row = box.row(align=True)
            row.operator("gpucloth.export_alembic",
                         text="Alembic (.abc)", icon=_icon('FILE'))
            row.operator("gpucloth.export_usd",
                         text="USD (.usdc)",    icon=_icon('FILE'))

        else:
            if 0 < scene_s.bake_progress < 100:
                col.prop(scene_s, "bake_progress",
                         text=_t("Progress", "Прогресс"), slider=True)
                col.label(
                    text=_t("Press ESC to cancel",
                            "Нажмите ESC для отмены"),
                    icon=_icon('INFO'))
            else:
                col.operator("gpucloth.bake_simulation",
                             text=_t("Bake Simulation",
                                     "Запечь симуляцию"),
                             icon=_icon('REC'))

        # The store's row is drawn wherever there is a store - a live session's
        # frames, the reached states behind a rewind, a range an earlier run left
        # on disk - and not only inside the baked branch above.  That branch is
        # what made the owner's own stale cache undeletable: every state it does
        # not cover (never baked, invalidated, half written) hid the only button
        # that could remove it.  The row is disabled while a bake transaction
        # owns the store, because clearing under it would refuse the frame the
        # bake is on rather than cancel it.
        store = operators.cache_store_summary(context.scene)
        if store["has_store"]:
            col.separator()
            col.label(text=_cache_store_detail(store), icon=_icon('INFO'))
            clear = col.row(align=True)
            clear.enabled = not scene_s.is_baking
            clear.operator("gpucloth.free_cache",
                           text=_t("Delete Cache", "Удалить кэш"),
                           icon=_icon('TRASH'))


# ===========================================================================
#  Panel: Constraint Network (parked - see _PARKED_PANEL_CLASSES)
# ===========================================================================
#
# Hidden by the owner's ruling, not deleted.  The class, the property it draws
# and everything behind it stay exactly as they were; only its registration is
# gone, so nothing in the panel reaches ``use_constraint_network`` any more.
#
# Why it is hidden rather than exposed: there is no authorable input for it and
# no test scene, so on a real garment the only thing it can do is refuse.  The
# algorithm also does not distinguish a button from a seam - the v3 record has
# no type field at all (``GPUCLOTH_ELEMENT_CONSTRAINT_NETWORK_RECORD`` carries
# the constraint and nothing that names its kind), so every entry the capture
# publishes is one kind to the engine.  Anyone re-exposing this should know that
# before writing the panel's labels.

class GPUCLOTH_PT_constraint_network(bpy.types.Panel):
    bl_label = "Constraint Network"
    bl_parent_id = "GPUCLOTH_PT_main"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == 'MESH'
        )

    def draw_header(self, context):
        obj = context.object
        if hasattr(obj, 'GPUCloth'):
            layout = self.layout
            layout.prop(obj.GPUCloth, "use_constraint_network", text="")

    def draw(self, context):
        obj = context.object
        if not hasattr(obj, 'GPUCloth'):
            return
        gs = obj.GPUCloth
        layout = self.layout
        layout.active = gs.use_constraint_network

        layout.prop(gs, "cn_phases")
        layout.prop(gs, "cn_sewing_speed")
        layout.separator()
        stiffness_row = layout.row()
        stiffness_row.enabled = gs.solver_type != 'PD'
        stiffness_row.prop(gs, "cn_seam_stiffness")
        if gs.solver_type == 'PD':
            layout.label(text="PD seam stiffness is fixed at 1.0")
        layout.separator()
        layout.prop(gs, "cn_enable_selfcoll_stitching")


# ===========================================================================
#  Sub-panel: Advanced Solver
# ===========================================================================

class GPUCLOTH_PT_solver_advanced(bpy.types.Panel):
    bl_label       = "Advanced Solver"
    bl_idname      = "GPUCLOTH_PT_solver_advanced"
    bl_parent_id   = "GPUCLOTH_PT_main"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "physics"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == 'MESH'
            and hasattr(context.object, 'GPUCloth')
            and context.object.GPUCloth.is_active
        )

    def draw(self, context):
        layout = self.layout
        s      = context.object.GPUCloth
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(s, "solver_iterations")
        col.prop(s, "solver_omega")
        krylov_row = col.row()
        krylov_row.active = s.solver_type == 'PD'
        krylov_row.prop(s, "solver_krylov_iterations")

        layout.separator()
        # Adaptive convergence is not carried by the product route, and the
        # panel must say so rather than invite an edit that cannot act.  The v3
        # feature that owns the solver budget is GPUClothQualityConfig, and it
        # carries solver_iterations and the Krylov ceiling only
        # (product_abi.h:323-341; main.cpp:7594-7622 writes exactly those three
        # slots).  The native adaptive early exit and the iteration cap read
        # the legacy DNA slots (main.cpp:9521-9523), and the product route's
        # ClothSimSettings comes from DNA_DEFAULT_ClothSimSettings, which names
        # none of them - so they are 0, adaptive is off, and the cap is always
        # the shipped fallback of 100.  solver_convergence_tol is worse than
        # inert here: the drape accepts its config only while that RNA value
        # equals the same shipped 1e-3 (main.cpp:6724-6734), so moving it makes
        # Begin Drape reject instead of tuning anything.
        box = layout.box()
        box.label(text="Adaptive Convergence (not carried by this route)")
        carried = box.column(align=True)
        carried.enabled = False
        carried.prop(s, "use_adaptive")
        carried.prop(s, "solver_max_iterations")
        carried.prop(s, "solver_convergence_tol", text="Convergence Tol")
        box.label(
            text="The solver runs Iterations steps with the shipped caps: "
                 "adaptive early exit off, max 100, force tol 1e-3 N",
            icon=_icon('INFO'))
        box.label(
            text="Convergence Tol is the drape sandbox's per-frame position "
                 "limit and must stay at its shipped value; the drape's own "
                 "limit is Drape tol in Prepare / Drape",
            icon=_icon('INFO'))


# ===========================================================================
#  Developer panel: Create Test Scene (3D Viewport sidebar)
#  Deliberately NOT in _PANEL_CLASSES: the product-surface audit forbids
#  developer operators in Properties panels, so this dev surface owns
#  the one-button scene builders instead.
# ===========================================================================

class GPUCLOTH_PT_test_scenes(bpy.types.Panel):
    bl_label       = "Create Test Scene"
    bl_idname      = "GPUCLOTH_PT_test_scenes"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "GPUCloth"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("gpucloth.test_drape_on_sphere",
                     text="1: Drape on Sphere", icon=_icon('MESH_GRID'))
        col.operator("gpucloth.test_twist",
                     text="2: Twist / Self-Col",
                     icon=_icon('MOD_SIMPLEDEFORM'))
        col.operator("gpucloth.test_multi_layer_drop",
                     text="3: Multi-Layer Drop", icon=_icon('MOD_CLOTH'))
        col.operator("gpucloth.test_cushion_drop",
                     text="4: Cushion (Pressure)", icon=_icon('MESH_UVSPHERE'))
        col.operator("gpucloth.test_cape",
                     text="5: Cape ZPRJ (panels + seams)",
                     icon=_icon('OUTLINER_OB_MESH'))
        col.operator("gpucloth.test_md_horizontal_contact",
                     text="6: MD Horizontal Contact", icon=_icon('MESH_PLANE'))
        col.separator()
        col.operator("gpucloth.test_ogc_bounds",
                     text="OGC Bounds Viz", icon=_icon('MESH_CIRCLE'))


_DEV_PANEL_CLASSES = [
    GPUCLOTH_PT_test_scenes,
]

# Panels that exist in the file and are deliberately not registered.  A parked
# panel is not a deleted one: the class, its properties, the capture and the
# publish path are all still here, and bringing it back to the surface is moving
# the name from this list to ``_PANEL_CLASSES`` below.
#
# ``GPUCLOTH_PT_constraint_network`` is parked by the owner's ruling because
# there is no authorable input for it and no test scene, so on a real garment it
# can only ever refuse ("constraint network requires at least one loose mesh
# edge").  Hiding it is what makes the feature *off and unreachable* rather than
# merely unlabelled: ``use_constraint_network`` has no writer anywhere in the
# add-on - no preset, no scene builder, no migration, no default-applying helper
# (checked across ``src/python``; the only other readers are
# ``operators._capture_constraint_network`` and
# ``cloth_settings_bridge.capture_v3_constraint_network``), and its default is
# False, so with the header toggle gone the feature cannot be switched on and
# nothing is left on with no way to switch it off.
_PARKED_PANEL_CLASSES = [
    GPUCLOTH_PT_constraint_network,
]


# ===========================================================================
#  Registration
# ===========================================================================

_PANEL_CLASSES = [
    GPUCLOTH_PT_main,
    GPUCLOTH_PT_preparation,
    GPUCLOTH_PT_solver,
    GPUCLOTH_PT_material,
    GPUCLOTH_PT_physical,
    GPUCLOTH_PT_solver_advanced,
    GPUCLOTH_PT_internal_springs,
    GPUCLOTH_PT_pressure,
    GPUCLOTH_PT_shape,
    GPUCLOTH_PT_object_collision,
    GPUCLOTH_PT_property_weights,
    GPUCLOTH_PT_field_weights,
    GPUCLOTH_PT_collision,
    GPUCLOTH_PT_proxy,
    GPUCLOTH_PT_cache,
]


def register():
    for cls in _PANEL_CLASSES:
        bpy.utils.register_class(cls)
    for cls in _DEV_PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_DEV_PANEL_CLASSES):
        bpy.utils.unregister_class(cls)
    for cls in reversed(_PANEL_CLASSES):
        bpy.utils.unregister_class(cls)
