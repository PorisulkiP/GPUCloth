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
                icon='INFO')
            return

        settings = obj.GPUCloth
        layout.prop(settings, "execution_backend", expand=True)
        cpu_modifier = cloth_settings_bridge.find_cpu_cloth_modifier(obj)
        row = layout.row(align=True)
        row.enabled = cpu_modifier is not None
        row.operator("gpucloth.sync_cpu_settings", text="", icon='FILE_REFRESH')
        if cpu_modifier is None:
            row.label(
                text=_t("CPU Cloth not found", "CPU Cloth не найден"),
                icon='ERROR')
        elif settings.cpu_sync_errors or settings.cpu_sync_blockers:
            if settings.cpu_sync_errors:
                row.label(
                    text=_t(
                        f"{settings.cpu_sync_errors} errors",
                        f"Ошибок: {settings.cpu_sync_errors}"),
                    icon='ERROR')
            if settings.cpu_sync_blockers:
                row.label(
                    text=_t(
                        f"{settings.cpu_sync_blockers} blocking",
                        f"Блокирует: {settings.cpu_sync_blockers}"),
                    icon='ERROR')
        else:
            row.label(
                text=_t(
                    f"{settings.cpu_sync_copied} imported",
                    f"Перенесено: {settings.cpu_sync_copied}"),
                icon='CHECKMARK')
            if settings.cpu_sync_unsupported:
                row.label(
                    text=_t(
                        f"{settings.cpu_sync_unsupported} unsupported",
                        f"Не поддержано: {settings.cpu_sync_unsupported}"),
                    icon='QUESTION')
        if settings.cpu_sync_report:
            row.operator("gpucloth.show_cpu_sync_report", text="", icon='INFO')

        if settings.execution_backend != 'GPU' or not settings.is_active:
            return

        prepared = (
            scene.gpu_cloth_springs_built
            and obj in operators.g_clothOBJs)

        layout.separator()

        status = layout.row(align=True)
        status.label(
            text=_t("Ready", "Готово") if prepared else
                 _t("Not prepared", "Не подготовлено"),
            icon='CHECKMARK' if prepared else 'INFO')
        prepare = status.row(align=True)
        prepare.enabled = operators.g_dll is not None and not prepared
        prepare.operator(
            "gpucloth.prepare_simulation",
            text=_t("Prepare", "Подготовить"), icon='PLAY')
        stop = status.row(align=True)
        stop.enabled = operators.g_dll is not None and prepared
        stop.operator(
            "gpucloth.destroy_simulation_data",
            text=_t("Stop", "Остановить"), icon='CANCEL')


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

        # ── Anisotropy ─────────────────────────────────────────────────────
        layout.separator()
        col = layout.column(align=True)
        col.prop(s, "use_anisotropy", text=_t("Anisotropic Stiffness", "Анизотропная жёсткость"))
        if s.use_anisotropy:
            col.prop(s, "tension_u",    text=_t("Tension U (Warp)", "Растяжение U"))
            col.prop(s, "tension_v",    text=_t("Tension V (Weft)", "Растяжение V"))
            col.prop(s, "bending_u",    text=_t("Bending U (Warp)", "Изгиб U"))
            col.prop(s, "bending_v",    text=_t("Bending V (Weft)", "Изгиб V"))

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
        col.prop(s, "air_viscosity",  text=_t("Air Viscosity",  "Вязкость воздуха"))
        col.prop(s, "vel_damping",    text=_t("Velocity Damping", "Демпфирование скорости"))

        layout.separator()
        col = layout.column(align=True)
        col.label(text=_t("Stiffness Clamping:", "Ограничение жёсткости:"))
        col.prop(s, "max_tension",    text=_t("Max Tension",     "Макс. растяжение"))
        col.prop(s, "max_compression",text=_t("Max Compression", "Макс. сжатие"))
        col.prop(s, "max_shear",      text=_t("Max Shear",       "Макс. сдвиг"))
        col.prop(s, "max_bend",       text=_t("Max Bending",     "Макс. изгиб"))
        col.prop(s, "max_struct",     text=_t("Max Structural",  "Макс. структура"))
        col.prop(s, "max_sewing",     text=_t("Max Sewing",      "Макс. шов"))

        layout.separator()
        col = layout.column(align=True)
        col.prop(s, "structural",     text=_t("Structural (Linear)",
                                              "Структурная жёсткость (Linear)"))
        col.active = (s.bending_model == 'LINEAR')


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
        col.prop(s, "target_volume",
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
            col.prop(s, "maxgoal",
                     text=_t("Max Goal", "Макс. цель"))

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
        col.prop(fw, "weight_gravity",
                 text=_t("Gravity", "Гравитация"))
        col.prop(fw, "weight_wind",
                 text=_t("Wind", "Ветер"))
        col.prop(fw, "weight_vortex",
                 text=_t("Vortex", "Вихрь"))
        col.prop(fw, "weight_magnetic",
                 text=_t("Magnetic", "Магнитное"))
        col.prop(fw, "weight_turbulence",
                 text=_t("Turbulence", "Турбулентность"))
        col.prop(fw, "weight_drag",
                 text=_t("Drag", "Торможение"))
        col.prop(fw, "weight_smoke_flow",
                 text=_t("Smoke Flow", "Поток дыма"))
        col.prop(fw, "weight_harmonic",
                 text=_t("Harmonic", "Гармоническое"))
        col.prop(fw, "weight_charge",
                 text=_t("Charge", "Заряд"))
        col.prop(fw, "weight_lennard_jones",
                 text=_t("Lennard-Jones", "Леннард-Джонс"))
        col.prop(fw, "weight_texture",
                 text=_t("Texture", "Текстура"))
        col.prop(fw, "weight_curve_guide",
                 text=_t("Curve Guide", "Напр. кривой"))
        col.prop(fw, "weight_boid",
                 text=_t("Boid", "Боид"))
        col.prop(fw, "weight_fluid",
                 text=_t("Fluid", "Флюид"))


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

        layout.separator()

        row = layout.row()
        row.prop(
            s, "show_ogc_bounds",
            text=_t("Show Contact Bounds", "Показать границы коллизии"),
            toggle=True,
            icon='SPHERE',
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
                icon='ERROR' if s.use_proxy else 'INFO',
            )
        elif s.use_proxy:
            try:
                binding = validate_proxy_binding(context.object, s)
                col.label(
                    text=(
                        f"{binding['simulation_vertex_count']} -> "
                        f"{binding['render_vertex_count']} vertices"),
                    icon='CHECKMARK',
                )
            except ProxyBindingError:
                col.label(
                    text=_t("Grid counts do not match meshes",
                            "Размеры сетки не совпадают с мешами"),
                    icon='ERROR',
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

        status_icon = (
            'ERROR' if scene_s.is_outdated or scene_s.is_frame_skip
            else ('TIME' if scene_s.is_baking
                  else ('CHECKMARK' if scene_s.is_baked else 'INFO')))
        col.label(text=scene_s.cache_info, icon=status_icon)

        if scene_s.is_baked:
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
            if scene_s.cached_frame_count > 0:
                col.operator("gpucloth.free_cache",
                             text="Clear Cache",
                             icon='TRASH')


# ===========================================================================
#  Panel: Constraint Network
# ===========================================================================

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
        layout.prop(gs, "cn_seam_stiffness")
        layout.prop(gs, "cn_button_stiffness")
        layout.prop(gs, "cn_zipper_stiffness")
        layout.prop(gs, "cn_dart_stiffness")
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
        col.prop(s, "use_small_steps")

        layout.separator()
        box = layout.box()
        box.label(text="Adaptive Convergence (Mil2/PD)")
        box.prop(s, "use_adaptive")
        sub = box.column()
        sub.active = s.use_adaptive
        sub.prop(s, "solver_max_iterations")
        sub.prop(s, "solver_convergence_tol")

        layout.separator()
        box = layout.box()
        box.label(text="Per-Type Budget (XPBD only)")
        box.prop(s, "use_per_type_budget")
        sub = box.column()
        sub.active = s.use_per_type_budget
        sub.prop(s, "ptb_stretch")
        sub.prop(s, "ptb_bending")
        sub.prop(s, "ptb_shear")
        sub.prop(s, "ptb_seam")


# ===========================================================================
#  Registration
# ===========================================================================

_PANEL_CLASSES = [
    GPUCLOTH_PT_main,
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
    GPUCLOTH_PT_constraint_network,
    GPUCLOTH_PT_cache,
]


def register():
    for cls in _PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_PANEL_CLASSES):
        bpy.utils.unregister_class(cls)
