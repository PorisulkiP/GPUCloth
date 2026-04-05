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
from ..utils import version_compatibility_utils as vcu


# ===========================================================================
#  Главная панель GPU Cloth
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
            layout.label(text="Включите GPUCloth для этого объекта", icon='INFO')
            return

        # ── DLL управление ───────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Библиотека симуляции", icon='PLUGIN')
        row = box.row(align=True)
        row.operator("gpucloth.load_dll",   text="Загрузить DLL", icon='IMPORT')
        row.operator("gpucloth.unload_dll", text="Выгрузить",     icon='X')

        layout.separator()

        # ── Управление симуляцией ────────────────────────────────────────────
        col = layout.column(align=True)
        col.operator("gpucloth.prepare_simulation",
                     text="Подготовить симуляцию", icon='PLAY')
        col.operator("gpucloth.destroy_simulation_data",
                     text="Освободить GPU память",  icon='TRASH')

        # Статус
        if scene.gpu_cloth_springs_built:
            layout.label(text="Симуляция готова", icon='CHECKMARK')
        else:
            layout.label(text="Симуляция не инициализирована", icon='ERROR')


# ===========================================================================
#  Подпанель: Солвер
# ===========================================================================

class GPUCLOTH_PT_solver(bpy.types.Panel):
    bl_label       = "Солвер"
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
        layout = self.layout
        obj    = context.object
        s      = obj.GPUCloth
        scene_s = context.scene.gpu_cloth_helper

        col = layout.column()

        # Тип солвера — главный выбор
        col.prop(s, "solver_type")
        col.separator()

        # ── Пресет материала ─────────────────────────────────────────────────
        col.prop(s, "material_preset")
        col.separator()

        # Параметры времени/качества
        col.prop(s, "quality_step")
        col.prop(s, "speed_multiplier")
        col.prop(s, "vertex_mass")
        col.prop(s, "bending_model")

        col.separator()
        col.label(text="Гравитация (м/с²):")
        row = col.row(align=True)
        row.prop(scene_s, "gravity_x", text="X")
        row.prop(scene_s, "gravity_y", text="Y")
        row.prop(scene_s, "gravity_z", text="Z")

        # Краткая справка по солверу
        col.separator()
        solver = s.solver_type
        if solver == 'XPBD':
            col.label(text="Extended PBD (быстро, устойчиво)", icon='INFO')
        elif solver == 'PD':
            col.label(text="Projective Dynamics + Chebyshev", icon='INFO')
        elif solver == 'MGPBD':
            col.label(text="Многоуровневый PBD с AMG (мягкая ткань)", icon='INFO')
        elif solver == 'Mil2':
            col.label(text="Non-distance barriers + subspace reuse", icon='INFO')
        elif solver == 'OGC':
            col.label(text="Offset Geometric Contact (самостолкновение)", icon='INFO')


# ===========================================================================
#  Подпанель: Материал (жёсткость и демпфирование)
# ===========================================================================

class GPUCLOTH_PT_material(bpy.types.Panel):
    bl_label       = "Материал"
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

        # ── Жёсткость ───────────────────────────────────────────────────────
        col = layout.column(align=True)
        col.label(text="Жёсткость:")
        col.prop(s, "tension",           text="Растяжение")
        col.prop(s, "compression",       text="Сжатие")
        col.prop(s, "shear",             text="Сдвиг")
        col.prop(s, "bending_stiffness", text="Изгиб")

        layout.separator()

        # ── Демпфирование ────────────────────────────────────────────────────
        col = layout.column(align=True)
        col.label(text="Демпфирование:")
        col.prop(s, "tension_damp",     text="Растяжение")
        col.prop(s, "compression_damp", text="Сжатие")
        col.prop(s, "shear_damp",       text="Сдвиг")
        col.prop(s, "bending_damping",  text="Изгиб")


# ===========================================================================
#  Подпанель: Proxy-res симуляция
# ===========================================================================

class GPUCLOTH_PT_proxy(bpy.types.Panel):
    bl_label       = "Proxy симуляция"
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
                text="Выберите объект с грубой сеткой",
                icon='ERROR' if s.use_proxy else 'INFO',
            )

        col.separator()

        # Размеры сеток
        box = col.box()
        box.label(text="Proxy-сетка (грубая):", icon='MESH_GRID')
        row = box.row(align=True)
        row.prop(s, "proxy_nx", text="NX")
        row.prop(s, "proxy_ny", text="NY")

        box = col.box()
        box.label(text="Hi-res сетка (рендер):", icon='MESH_GRID')
        row = box.row(align=True)
        row.prop(s, "hi_nx", text="NX")
        row.prop(s, "hi_ny", text="NY")

        col.separator()
        col.prop(s, "num_sheets")
        col.prop(s, "proxy_scene_type")

        # Пояснение архитектуры
        col.separator()
        col.label(text="GPU путь:", icon='INFO')
        col.label(text="scatter → ClothVertex.x")
        col.label(text="ProxySim_apply → hi-res позиции")
        col.label(text="cudaMemcpyAsync D2H → foreach_set")


# ===========================================================================
#  Подпанель: Кэш симуляции
# ===========================================================================

class GPUCLOTH_PT_cache(bpy.types.Panel):
    bl_label       = "Кэш"
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

        # Директория кэша
        col.prop(scene_s, "cache_dir")

        # Диапазон кадров
        row = col.row(align=True)
        row.prop(scene_s, "bake_start", text="От")
        row.prop(scene_s, "bake_end",   text="До")

        col.separator()

        if scene_s.is_baked:
            # ── Кэш записан ─────────────────────────────────────────────────
            row = col.row()
            row.label(text="Статус: запечено", icon='CHECKMARK')

            col.prop(scene_s, "playback_mode",
                     text="Воспроизводить из кэша", toggle=True,
                     icon='PLAY' if scene_s.playback_mode else 'PAUSE')

            col.separator()

            # ── Экспорт ─────────────────────────────────────────────────────
            box = col.box()
            box.label(text="Экспорт симуляции:", icon='EXPORT')
            row = box.row(align=True)
            row.operator("gpucloth.export_alembic",
                         text="Alembic (.abc)", icon='FILE')
            row.operator("gpucloth.export_usd",
                         text="USD (.usdc)",    icon='FILE')

            col.separator()
            col.operator("gpucloth.free_cache",
                         text="Очистить кэш", icon='TRASH')

            # Архитектура чтения
            if scene_s.playback_mode:
                box = col.box()
                box.label(text="GPU-direct (zero-copy):", icon='INFO')
                box.label(text="NVMe → D3D12 VRAM")
                box.label(text="CUDA ext. memory → scatter")
                box.label(text="D2H → foreach_set")

        else:
            # ── Кэш не записан ──────────────────────────────────────────────

            # Прогресс (если запекание идёт)
            if 0 < scene_s.bake_progress < 100:
                col.prop(scene_s, "bake_progress",
                         text="Прогресс", slider=True)
                col.label(text="Нажмите ESC для отмены", icon='INFO')
            else:
                col.operator("gpucloth.bake_simulation",
                             text="Запечь симуляцию", icon='REC')

            # Пояснение архитектуры записи
            box = col.box()
            box.label(text="Запись кэша (async):", icon='INFO')
            box.label(text="SIM_solver → SIM_get_cloth_verts")
            box.label(text="foreach_set + Cache_write_frame_async")
            box.label(text="pinned RAM → DMA → NVMe (фон)")


# ===========================================================================
#  Регистрация
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
