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
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None
            and context.object.type == 'MESH'
        )

    def draw_header(self, context):
        layout = self.layout
        obj    = context.object
        if hasattr(obj, 'GPUCloth'):
            layout.prop(obj.GPUCloth, "is_active", text="")

    def draw(self, context):
        layout = self.layout
        obj    = context.object
        scene  = context.scene

        if not hasattr(obj, 'GPUCloth') or not obj.GPUCloth.is_active:
            layout.label(
                text=_t("Enable GPUCloth for this object",
                        "Включите GPUCloth для этого объекта"),
                icon='INFO')
            return

        # ── DLL ──────────────────────────────────────────────────────────────
        box = layout.box()
        box.label(text=_t("Simulation Library", "Библиотека симуляции"),
                  icon='PLUGIN')
        row = box.row(align=True)
        row.operator("gpucloth.load_dll",
                     text=_t("Load DLL", "Загрузить DLL"), icon='IMPORT')
        row.operator("gpucloth.unload_dll",
                     text=_t("Unload", "Выгрузить"), icon='X')

        layout.separator()

        # ── Simulation controls ──────────────────────────────────────────────
        col = layout.column(align=True)
        col.operator("gpucloth.prepare_simulation",
                     text=_t("Prepare Simulation", "Подготовить симуляцию"),
                     icon='PLAY')
        col.operator("gpucloth.destroy_simulation_data",
                     text=_t("Free GPU Memory", "Освободить GPU память"),
                     icon='TRASH')

        if scene.gpu_cloth_springs_built:
            layout.label(
                text=_t("Simulation ready", "Симуляция готова"),
                icon='CHECKMARK')
        else:
            layout.label(
                text=_t("Simulation not initialized",
                        "Симуляция не инициализирована"),
                icon='ERROR')

        # ── Test scenes ────────────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text=_t("Test Scenes", "Тестовые сцены"),
                  icon='EXPERIMENTAL')
        col = box.column(align=True)
        col.operator("gpucloth.test_drape_on_sphere",
                     text=_t("Drape On Sphere", "Драпировка на сфере"))
        col.operator("gpucloth.test_twist",
                     text=_t("Twist Test", "Тест скручивания"))
        col.operator("gpucloth.test_multi_layer_drop",
                     text=_t("Multi Layer Drop", "Многослойное падение"))
        col.operator("gpucloth.test_cushion_drop",
                     text=_t("Cushion Drop", "Падение подушки"))


# ===========================================================================
#  Sub-panel: Solver
# ===========================================================================

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

        col.prop(s, "material_preset")
        col.separator()

        col.prop(s, "quality_step")
        col.prop(s, "speed_multiplier")
        col.prop(s, "vertex_mass")
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
        col.label(text=_t("Stiffness:", "Жёсткость:"))
        col.prop(s, "tension",           text=_t("Tension",     "Растяжение"))
        col.prop(s, "compression",       text=_t("Compression", "Сжатие"))
        col.prop(s, "shear",             text=_t("Shear",       "Сдвиг"))
        col.prop(s, "bending_stiffness", text=_t("Bending",     "Изгиб"))

        layout.separator()

        col = layout.column(align=True)
        col.label(text=_t("Damping:", "Демпфирование:"))
        col.prop(s, "tension_damp",     text=_t("Tension",     "Растяжение"))
        col.prop(s, "compression_damp", text=_t("Compression", "Сжатие"))
        col.prop(s, "shear_damp",       text=_t("Shear",       "Сдвиг"))
        col.prop(s, "bending_damping",  text=_t("Bending",     "Изгиб"))


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
                icon='ERROR' if s.use_proxy else 'INFO',
            )

        col.separator()

        box = col.box()
        box.label(text=_t("Proxy grid (coarse):", "Proxy-сетка (грубая):"),
                  icon='MESH_GRID')
        row = box.row(align=True)
        row.prop(s, "proxy_nx", text="NX")
        row.prop(s, "proxy_ny", text="NY")

        box = col.box()
        box.label(text=_t("Hi-res grid (render):", "Hi-res сетка (рендер):"),
                  icon='MESH_GRID')
        row = box.row(align=True)
        row.prop(s, "hi_nx", text="NX")
        row.prop(s, "hi_ny", text="NY")

        col.separator()
        col.prop(s, "num_sheets")
        col.prop(s, "proxy_scene_type")


# ===========================================================================
#  Sub-panel: Simulation cache
# ===========================================================================

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
                icon='CHECKMARK')

            col.prop(scene_s, "playback_mode",
                     text=_t("Play from Cache", "Воспроизводить из кэша"),
                     toggle=True,
                     icon='PLAY' if scene_s.playback_mode else 'PAUSE')

            col.separator()

            box = col.box()
            box.label(
                text=_t("Export Simulation:", "Экспорт симуляции:"),
                icon='EXPORT')
            row = box.row(align=True)
            row.operator("gpucloth.export_alembic",
                         text="Alembic (.abc)", icon='FILE')
            row.operator("gpucloth.export_usd",
                         text="USD (.usdc)",    icon='FILE')

            col.separator()
            col.operator("gpucloth.free_cache",
                         text=_t("Clear Cache", "Очистить кэш"),
                         icon='TRASH')

        else:
            if 0 < scene_s.bake_progress < 100:
                col.prop(scene_s, "bake_progress",
                         text=_t("Progress", "Прогресс"), slider=True)
                col.label(
                    text=_t("Press ESC to cancel",
                            "Нажмите ESC для отмены"),
                    icon='INFO')
            else:
                col.operator("gpucloth.bake_simulation",
                             text=_t("Bake Simulation",
                                     "Запечь симуляцию"),
                             icon='REC')


# ===========================================================================
#  Registration
# ===========================================================================

_PANEL_CLASSES = [
    GPUCLOTH_PT_main,
    GPUCLOTH_PT_solver,
    GPUCLOTH_PT_material,
    GPUCLOTH_PT_proxy,
    GPUCLOTH_PT_cache,
]


def register():
    for cls in _PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_PANEL_CLASSES):
        bpy.utils.unregister_class(cls)
