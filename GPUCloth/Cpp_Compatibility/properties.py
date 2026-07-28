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
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    StringProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup

from ..utils import version_compatibility_utils as vcu
from ..utils.version_compatibility_utils import _t


def _on_is_active_change(self, context):
    if self.is_active and self.execution_backend != 'GPU':
        self.execution_backend = 'GPU'
        if self.execution_backend != 'GPU':
            self.is_active = False
            return
    elif not self.is_active and self.execution_backend != 'CPU':
        self.execution_backend = 'CPU'
    if self.is_active:
        from . import operators as ops
        if ops.g_dll is None:
            try:
                load_result = bpy.ops.gpucloth.load_dll()
            except Exception as exc:
                load_result = {'CANCELLED'}
                load_error = f"GPUCloth DLL activation failed: {exc}"
            else:
                load_error = "GPUCloth DLL activation failed"
            if 'FINISHED' not in load_result:
                from . import cloth_settings_bridge
                cloth_settings_bridge.record_runtime_error(
                    self.id_data, load_error)
                self.is_active = False
                self.execution_backend = 'CPU'
                cloth_settings_bridge.apply_modifier_ownership(
                    self.id_data, 'CPU')


_backend_switch_active = False


def _on_execution_backend_change(self, context):
    global _backend_switch_active
    if _backend_switch_active:
        return
    obj = self.id_data
    if obj is None:
        return
    from . import cloth_settings_bridge
    _backend_switch_active = True
    try:
        result = cloth_settings_bridge.select_backend(
            obj, self.execution_backend, context.scene if context else None)
        if result is not None and (
                result["errors"] or result["unsupported_non_default"]):
            self.execution_backend = 'CPU'
            cloth_settings_bridge.select_backend(
                obj, 'CPU', context.scene if context else None)
    finally:
        _backend_switch_active = False


# ===========================================================================
#  Материальные пресеты — значения параметров ткани для каждого солвера
# ===========================================================================
#
#  Источники значений:
#    XPBD  — Macklin & Müller 2016 "XPBD: Position-Based Simulation of
#             Compliant Constrained Dynamics", Table 1 (compliance → stiffness)
#    PD    — Bouaziz et al. 2014 "Projective Dynamics", §5 (projection weights);
#             глобальный solve позволяет бóльшую жёсткость при меньшем числе шагов
#    MGPBD — Xian et al. 2019 "A Scalable Galerkin Multigrid Method for
#             Real-time Simulation", §4; AMG-ускорение → меньше подшагов
#    Mil2  — Li et al. 2020 "Incremental Potential Contact", §3;
#             barrier-based, диапазон аналогичен XPBD
#    OGC   — Chen et al. 2025 "Offset Geometric Contact", §3.6;
#             базовые параметры как у XPBD + OGC-специфичные (radius, friction)
#
#  Физические ориентиры (реальные ткани):
#    Silk:    density ~1.3 g/cm³, thickness ~0.1 mm, very light and flowing
#    Cotton:  ~1.5 g/cm³, ~0.3 mm, moderate drape
#    Denim:   ~1.5 g/cm³, ~1 mm, stiff dense weave
#    Leather: ~0.9 g/cm³, ~1–2 mm, stiff in stretch, high bending
#    Rubber:  ~1.5 g/cm³, ~1–3 mm, elastic, low bending, high friction

MATERIAL_PRESETS = {
    'XPBD': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 10, 'bending_model': 'ANGULAR',
            'tension': 4.0,  'compression': 2.5,  'shear': 1.5,  'bending_stiffness': 0.03,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.5,   'bending_damping': 0.05,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 5, 'bending_model': 'ANGULAR',
            'tension': 15.0, 'compression': 15.0, 'shear': 5.0,  'bending_stiffness': 0.5,
            'tension_damp': 5.0, 'compression_damp': 5.0,
            'shear_damp': 5.0,   'bending_damping': 0.5,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 40.0, 'compression': 40.0, 'shear': 20.0, 'bending_stiffness': 5.0,
            'tension_damp': 10.0, 'compression_damp': 10.0,
            'shear_damp': 8.0,    'bending_damping': 2.0,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 80.0, 'compression': 80.0, 'shear': 10.0, 'bending_stiffness': 15.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 5.0,    'bending_damping': 5.0,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 8, 'bending_model': 'LINEAR',
            'tension': 50.0, 'compression': 50.0, 'shear': 25.0, 'bending_stiffness': 1.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 12.0,   'bending_damping': 1.0,
        },
    },
    'PD': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 8.0,  'compression': 5.0,  'shear': 3.0,  'bending_stiffness': 0.05,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.5,   'bending_damping': 0.05,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 30.0, 'compression': 30.0, 'shear': 10.0, 'bending_stiffness': 1.0,
            'tension_damp': 5.0, 'compression_damp': 5.0,
            'shear_damp': 5.0,   'bending_damping': 0.5,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 80.0, 'compression': 80.0, 'shear': 35.0, 'bending_stiffness': 10.0,
            'tension_damp': 10.0, 'compression_damp': 10.0,
            'shear_damp': 8.0,    'bending_damping': 2.0,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 150.0, 'compression': 150.0, 'shear': 20.0, 'bending_stiffness': 25.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 5.0,    'bending_damping': 5.0,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 5, 'bending_model': 'ANGULAR',
            'tension': 100.0, 'compression': 100.0, 'shear': 50.0, 'bending_stiffness': 2.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 12.0,   'bending_damping': 1.0,
        },
    },
    'MGPBD': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 3.0,  'compression': 2.0,  'shear': 1.0,  'bending_stiffness': 0.02,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.3,   'bending_damping': 0.03,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 3, 'bending_model': 'ANGULAR',
            'tension': 10.0, 'compression': 10.0, 'shear': 3.0,  'bending_stiffness': 0.3,
            'tension_damp': 3.0, 'compression_damp': 3.0,
            'shear_damp': 2.0,   'bending_damping': 0.3,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 30.0, 'compression': 30.0, 'shear': 12.0, 'bending_stiffness': 3.0,
            'tension_damp': 8.0, 'compression_damp': 8.0,
            'shear_damp': 6.0,   'bending_damping': 1.5,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 5, 'bending_model': 'ANGULAR',
            'tension': 60.0, 'compression': 60.0, 'shear': 8.0,  'bending_stiffness': 10.0,
            'tension_damp': 12.0, 'compression_damp': 12.0,
            'shear_damp': 4.0,    'bending_damping': 4.0,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 5, 'bending_model': 'LINEAR',
            'tension': 40.0, 'compression': 40.0, 'shear': 18.0, 'bending_stiffness': 0.5,
            'tension_damp': 12.0, 'compression_damp': 12.0,
            'shear_damp': 8.0,    'bending_damping': 0.5,
        },
    },
    'Mil2': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 10, 'bending_model': 'ANGULAR',
            'tension': 5.0,  'compression': 3.0,  'shear': 1.5,  'bending_stiffness': 0.03,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.5,   'bending_damping': 0.05,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 5, 'bending_model': 'ANGULAR',
            'tension': 15.0, 'compression': 15.0, 'shear': 5.0,  'bending_stiffness': 0.5,
            'tension_damp': 5.0, 'compression_damp': 5.0,
            'shear_damp': 5.0,   'bending_damping': 0.5,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 40.0, 'compression': 40.0, 'shear': 20.0, 'bending_stiffness': 5.0,
            'tension_damp': 10.0, 'compression_damp': 10.0,
            'shear_damp': 8.0,    'bending_damping': 2.0,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 80.0, 'compression': 80.0, 'shear': 10.0, 'bending_stiffness': 15.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 5.0,    'bending_damping': 5.0,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 8, 'bending_model': 'ANGULAR',
            'tension': 50.0, 'compression': 50.0, 'shear': 25.0, 'bending_stiffness': 1.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 12.0,   'bending_damping': 1.0,
        },
    },
    'OGC': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 10, 'bending_model': 'ANGULAR',
            'tension': 4.0,  'compression': 2.5,  'shear': 1.5,  'bending_stiffness': 0.03,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.5,   'bending_damping': 0.05,
            'use_self_collision': True, 'ogc_radius': 80.0, 'ogc_friction': 0.1,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 5, 'bending_model': 'ANGULAR',
            'tension': 15.0, 'compression': 15.0, 'shear': 5.0,  'bending_stiffness': 0.5,
            'tension_damp': 5.0, 'compression_damp': 5.0,
            'shear_damp': 5.0,   'bending_damping': 0.5,
            'use_self_collision': True, 'ogc_radius': 150.0, 'ogc_friction': 0.3,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 40.0, 'compression': 40.0, 'shear': 20.0, 'bending_stiffness': 5.0,
            'tension_damp': 10.0, 'compression_damp': 10.0,
            'shear_damp': 8.0,    'bending_damping': 2.0,
            'use_self_collision': True, 'ogc_radius': 200.0, 'ogc_friction': 0.5,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 80.0, 'compression': 80.0, 'shear': 10.0, 'bending_stiffness': 15.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 5.0,    'bending_damping': 5.0,
            'use_self_collision': True, 'ogc_radius': 250.0, 'ogc_friction': 0.6,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 8, 'bending_model': 'LINEAR',
            'tension': 50.0, 'compression': 50.0, 'shear': 25.0, 'bending_stiffness': 1.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 12.0,   'bending_damping': 1.0,
            'use_self_collision': True, 'ogc_radius': 300.0, 'ogc_friction': 0.8,
        },
    },
}


# ===========================================================================
#  Коллбэки обновления пресетов
# ===========================================================================

def _apply_preset(self, context):
    """Apply the selected material preset for the current solver."""
    preset_name = self.material_preset
    if preset_name == 'CUSTOM':
        return
    data = MATERIAL_PRESETS.get(self.solver_type, {}).get(preset_name)
    if data is None:
        return
    for prop, value in data.items():
        setattr(self, prop, value)


def _on_solver_change(self, context):
    """Re-apply preset when solver changes (values differ per solver)."""
    if self.material_preset != 'CUSTOM':
        _apply_preset(self, context)
    elif self.solver_type == 'Mil2' or (
            self.solver_type == 'PD' and
            self.bending_model not in {'ANGULAR', 'SDB'}):
        self.bending_model = 'ANGULAR'
    elif self.solver_type != 'PD' and self.bending_model == 'SDB':
        self.bending_model = 'ANGULAR'


# ===========================================================================
#  Локализованные items для EnumProperty (вызываются при каждом открытии)
# ===========================================================================

def _bending_model_items(self, context):
    if self.solver_type == 'PD':
        return [
            ('ANGULAR', _t("Angular", "Угловой"),
             _t("Dihedral-angle bending constraint",
                "Ограничение изгиба по двугранному углу")),
            ('SDB', "Stable Discrete Bending (SDB)",
             _t("Stable Discrete Bending operator for Projective Dynamics",
                "Оператор Stable Discrete Bending для Projective Dynamics")),
        ]
    if self.solver_type == 'Mil2':
        return [
            ('ANGULAR', _t("Angular", "Угловой"),
             _t("Standard Mil2 dihedral-angle bending constraint",
                "Стандартное ограничение изгиба Mil2 по двугранному углу")),
        ]
    return [
        ('LINEAR',  _t("Linear",  "Линейный"),  _t("Linear bending stiffness", "Линейная жёсткость изгиба")),
        ('ANGULAR', _t("Angular", "Угловой"),    _t("Angular bending stiffness (more realistic)", "Угловая жёсткость изгиба (реалистичнее)")),
    ]


def _solver_type_items(self, context):
    return [
        ('XPBD',  "XPBD",  "Extended Position-Based Dynamics (Macklin 2016)"),
        ('PD',    "PD",     _t("Projective Dynamics with Chebyshev-Jacobi acceleration",
                               "Projective Dynamics с Chebyshev-Jacobi ускорением")),
        ('MGPBD', "MGPBD",  _t("Multigrid PBD with Algebraic Multigrid",
                               "Многоуровневый PBD с Algebraic Multigrid")),
        ('Mil2',  "Mil2",   "Non-distance barriers + Subspace Reuse"),
        ('OGC',   "OGC",    _t("Offset Geometric Contact — self-collision",
                               "Offset Geometric Contact — самостолкновение")),
    ]


def _material_preset_items(self, context):
    return [
        ('CUSTOM',  _t("Custom",  "Свой"),    _t("Manual parameter tuning",             "Ручная настройка параметров")),
        ('SILK',    _t("Silk",    "Шёлк"),    _t("Light, flowing, minimal bending",     "Лёгкий, текучий, минимальный изгиб")),
        ('COTTON',  _t("Cotton",  "Хлопок"),  _t("Moderate drape, medium stiffness",    "Умеренная драпировка, средняя жёсткость")),
        ('DENIM',   _t("Denim",   "Деним"),   _t("Dense fabric, stiff bending & shear", "Плотная ткань, жёсткий изгиб и сдвиг")),
        ('LEATHER', _t("Leather", "Кожа"),    _t("Heavy, very stiff in stretch",        "Тяжёлая, очень жёсткая на растяжение")),
        ('RUBBER',  _t("Rubber",  "Резина"),  _t("Elastic, heavy, low bending",         "Упругая, тяжёлая, низкий изгиб")),
    ]


# ===========================================================================
#  PropertyGroup for field weights
# ===========================================================================

class GPUClothEffectorWeights(PropertyGroup):
    """Per-field-type weights mirrored by the native EffectorWeights array."""

    collection: PointerProperty(
        name="Effector Collection",
        description="Restrict force fields to objects in this collection",
        type=bpy.types.Collection,
    )

    global_gravity: FloatProperty(
        name="Gravity",
        description="Global gravity weight",
        default=1.0,
        min=-200.0,
        max=200.0,
    )

    weight_all: FloatProperty(
        name="All", description="All effectors weight", default=1.0, min=-200.0, max=200.0)
    weight_force: FloatProperty(
        name="Force", description="Force effector weight", default=1.0, min=-200.0, max=200.0)
    weight_wind: FloatProperty(
        name="Wind", description="Wind effector weight", default=1.0, min=-200.0, max=200.0)
    weight_vortex: FloatProperty(
        name="Vortex", description="Vortex effector weight", default=1.0, min=-200.0, max=200.0)
    weight_magnetic: FloatProperty(
        name="Magnetic", description="Magnetic effector weight", default=1.0, min=-200.0, max=200.0)
    weight_turbulence: FloatProperty(
        name="Turbulence", description="Turbulence effector weight", default=1.0, min=-200.0, max=200.0)
    weight_drag: FloatProperty(
        name="Drag", description="Drag effector weight", default=1.0, min=-200.0, max=200.0)
    weight_smoke_flow: FloatProperty(
        name="Fluid Flow", description="Fluid Flow effector weight", default=1.0, min=-200.0, max=200.0)
    weight_harmonic: FloatProperty(
        name="Harmonic", description="Harmonic effector weight", default=1.0, min=-200.0, max=200.0)
    weight_charge: FloatProperty(
        name="Charge", description="Charge effector weight", default=1.0, min=-200.0, max=200.0)
    weight_lennard_jones: FloatProperty(
        name="Lennard-Jones", description="Lennard-Jones effector weight", default=1.0, min=-200.0, max=200.0)
    weight_texture: FloatProperty(
        name="Texture", description="Texture effector weight", default=1.0, min=-200.0, max=200.0)
    weight_curve_guide: FloatProperty(
        name="Curve Guide", description="Curve Guide effector weight", default=1.0, min=-200.0, max=200.0)
    weight_boid: FloatProperty(
        name="Boid", description="Boid effector weight", default=1.0, min=-200.0, max=200.0)


# ===========================================================================
#  PropertyGroup для объекта — настройки ткани
# ===========================================================================

class GPUClothObjectSettings(PropertyGroup):
    """Per-object GPU cloth simulation settings (OBJ.GPUCloth)."""

    # ── Basics ─────────────────────────────────────────────────────────────
    is_active: BoolProperty(
        name="Enable GPUCloth",
        description="Simulate this object as GPU cloth",
        default=False,
        update=_on_is_active_change,
    )

    execution_backend: EnumProperty(
        name="Simulation Backend",
        description="Select CPU Cloth or GPUCloth evaluation",
        items=(
            ('CPU', "CPU Cloth", "Use Blender Cloth modifier"),
            ('GPU', "GPUCloth", "Use GPUCloth with imported CPU Cloth settings"),
        ),
        default='CPU',
        update=_on_execution_backend_change,
    )

    cpu_sync_copied: IntProperty(default=0, options={'HIDDEN'})
    cpu_sync_unsupported: IntProperty(default=0, options={'HIDDEN'})
    cpu_sync_blockers: IntProperty(default=0, options={'HIDDEN'})
    cpu_sync_errors: IntProperty(default=0, options={'HIDDEN'})
    cpu_sync_report: StringProperty(default="", options={'HIDDEN'})

    vertex_mass: FloatProperty(
        name="Vertex Mass",
        description="Mass of a single cloth vertex (kg)",
        default=0.3,
        min=0.001,
        max=10.0,
        unit='MASS',
    )

    quality_step: IntProperty(
        name="Quality Steps",
        description="Solver substeps per frame",
        default=5,
        min=1,
        max=80,
    )

    speed_multiplier: FloatProperty(
        name="Speed Multiplier",
        description="Simulation time scale",
        default=1.0,
        min=0.0,
        max=10.0,
    )

    bending_model: EnumProperty(
        name="Bending Model",
        description="Cloth bending constraint type",
        items=_bending_model_items,
    )

    # ── Solver ─────────────────────────────────────────────────────────────
    solver_type: EnumProperty(
        name="Solver",
        description="GPU cloth simulation algorithm",
        items=_solver_type_items,
        update=_on_solver_change,
    )

    # ── Material preset ────────────────────────────────────────────────────
    material_preset: EnumProperty(
        name="Material",
        description="Material physics preset tuned for the current solver",
        items=_material_preset_items,
        update=_apply_preset,
    )

    # ── Material parameters (stiffness) ────────────────────────────────────
    tension: FloatProperty(
        name="Tension",
        description="Stretch stiffness",
        default=15.0,
        min=0.0,
        max=10000.0,
    )

    compression: FloatProperty(
        name="Compression",
        description="Compression stiffness",
        default=15.0,
        min=0.0,
        max=10000.0,
    )

    shear: FloatProperty(
        name="Shear",
        description="Shear stiffness",
        default=5.0,
        min=0.0,
        max=10000.0,
    )

    bending_stiffness: FloatProperty(
        name="Bending",
        description="Bending stiffness",
        default=0.5,
        min=0.0,
        max=10000.0,
    )

    # ── Material parameters (damping) ─────────────────────────────────────
    tension_damp: FloatProperty(
        name="Tension Damping",
        description="Tension oscillation damping",
        default=5.0,
        min=0.0,
        max=50.0,
    )

    compression_damp: FloatProperty(
        name="Compression Damping",
        description="Compression oscillation damping",
        default=5.0,
        min=0.0,
        max=50.0,
    )

    shear_damp: FloatProperty(
        name="Shear Damping",
        description="Shear oscillation damping",
        default=5.0,
        min=0.0,
        max=50.0,
    )

    bending_damping: FloatProperty(
        name="Bending Damping",
        description="Bending oscillation damping",
        default=0.5,
        min=0.0,
        max=1000.0,
    )

    # ── Self-collision (OGC) ──────────────────────────────────────────────
    use_self_collision: BoolProperty(
        name="Self-Collision",
        description="Enable Offset Geometric Contact (OGC, Chen et al. 2025) penetration-free self-collision",
        default=False,
    )

    ogc_radius: FloatProperty(
        name="Contact Radius",
        description="OGC contact zone radius (mm). Larger = layers stay further apart. Auto: r = 16.6 * avg_edge^2",
        default=150.0,
        min=1.0,
        max=500.0,
        subtype='NONE',
        unit='NONE',
    )

    ogc_kc: FloatProperty(
        name="Contact Stiffness",
        description="OGC contact stiffness (kc, PD mode). Lower = softer & more stable. Auto: kc = 1.15 / radius_m",
        default=7.0,
        min=1.0,
        max=100000.0,
    )

    ogc_friction: FloatProperty(
        name="Layer Friction",
        description="Friction between cloth layers (OGC). 0 = free sliding, 1 = no sliding. Recommended 0.3-0.6",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    ogc_gamma_p: FloatProperty(
        name="Gamma P",
        description="OGC correction fraction per pass (Mil2 mode). Must be strictly < 0.5. 0.45 = default, 0.48 = dense twist",
        default=0.45,
        min=0.05,
        max=0.49,
    )

    show_ogc_bounds: BoolProperty(
        name="Show Contact Bounds",
        description="Draw OGC contact-radius spheres at each cloth vertex in the 3D viewport (two axis-aligned circles per vertex)",
        default=False,
    )

    # ── Constraint network ─────────────────────────────────────────────────
    use_constraint_network: BoolProperty(
        name="Use Constraint Network",
        description="Animate sewing seams, zippers, and buttons with phased initialization",
        default=False,
    )

    cn_phases: IntProperty(
        name="Phases",
        description="Number of activation phases (sewing stages)",
        default=1, min=1, max=10,
    )

    cn_sewing_speed: FloatProperty(
        name="Sewing Speed",
        description="Frames to fully tighten a seam phase",
        default=10.0, min=1.0, max=100.0,
    )

    cn_seam_stiffness: FloatProperty(
        name="Seam Stiffness",
        description="Stiffness multiplier for seam constraints",
        default=1.0, min=0.0, max=5.0, soft_max=2.0,
    )

    cn_button_stiffness: FloatProperty(
        name="Button Stiffness",
        description="Stiffness multiplier for button constraints",
        default=1.0, min=0.0, max=5.0, soft_max=2.0,
    )

    cn_zipper_stiffness: FloatProperty(
        name="Zipper Stiffness",
        description="Stiffness multiplier for zipper constraints",
        default=1.0, min=0.0, max=5.0, soft_max=2.0,
    )

    cn_dart_stiffness: FloatProperty(
        name="Dart Stiffness",
        description="Stiffness multiplier for dart constraints",
        default=1.0, min=0.0, max=5.0, soft_max=2.0,
    )

    cn_enable_selfcoll_stitching: BoolProperty(
        name="Self-Collision During Stitching",
        description="Enable self-collision while seams are being tightened",
        default=True,
    )

    # ── Physical Properties: Damping & Clamping ────────────────────────────
    air_viscosity: FloatProperty(
        name="Air Viscosity",
        description="Viscosity of the surrounding medium (air damping factor)",
        default=1.0,
        min=0.0,
        max=100.0,
    )

    structural: FloatProperty(
        name="Structural Stiffness",
        description="Structural spring stiffness for Linear Bending model",
        default=15.0,
        min=0.0,
        max=500.0,
    )

    max_tension: FloatProperty(
        name="Max Tension",
        description="Maximum tension stiffness clamping value",
        default=500.0,
        min=0.0,
        max=10000.0,
    )

    max_compression: FloatProperty(
        name="Max Compression",
        description="Maximum compression stiffness clamping value",
        default=500.0,
        min=0.0,
        max=10000.0,
    )

    max_shear: FloatProperty(
        name="Max Shear",
        description="Maximum shear stiffness clamping value",
        default=500.0,
        min=0.0,
        max=10000.0,
    )

    max_bend: FloatProperty(
        name="Max Bending",
        description="Maximum bending stiffness clamping value",
        default=100.0,
        min=0.0,
        max=10000.0,
    )

    max_struct: FloatProperty(
        name="Max Structural",
        description="Maximum structural stiffness clamping value",
        default=500.0,
        min=0.0,
        max=500.0,
    )

    max_sewing: FloatProperty(
        name="Max Sewing Force",
        description="Maximum sewing force clamping value",
        default=500.0,
        min=0.0,
        max=10000.0,
    )

    use_sewing_springs: BoolProperty(
        name="Sew Cloth",
        description="Pull loose Blender cloth edges together",
        default=False,
    )

    vel_damping: FloatProperty(
        name="Velocity Damping",
        description="Damp velocity to speed up convergence to rest pose",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    # ── Internal Springs ───────────────────────────────────────────────────
    use_internal_springs: BoolProperty(
        name="Internal Springs",
        description="Create internal springs to resist compression",
        default=False,
    )

    use_internal_springs_normal: BoolProperty(
        name="Check Surface Normals",
        description="Create internal springs only between points with opposite normals",
        default=False,
    )

    internal_spring_max_length: FloatProperty(
        name="Max Spring Length",
        description="Maximum length an internal spring can have during creation",
        default=0.0,
        min=0.0,
        max=1000.0,
    )

    internal_spring_max_diversion: FloatProperty(
        name="Max Normal Diversion",
        description="Maximum angle diversion from vertex normal during internal spring creation (radians)",
        default=0.7853981633974483,  # π/4
        min=0.0,
        max=0.7853981633974483,      # pi/4
        subtype='ANGLE',
    )

    internal_tension: FloatProperty(
        name="Tension",
        description="Tension stiffness for internal springs",
        default=15.0,
        min=0.0,
        max=10000.0,
    )

    internal_compression: FloatProperty(
        name="Compression",
        description="Compression stiffness for internal springs",
        default=15.0,
        min=0.0,
        max=10000.0,
    )

    max_internal_tension: FloatProperty(
        name="Max Internal Tension",
        description="Maximum tension stiffness clamping for internal springs",
        default=500.0,
        min=0.0,
        max=10000.0,
    )

    max_internal_compression: FloatProperty(
        name="Max Internal Compression",
        description="Maximum compression stiffness clamping for internal springs",
        default=500.0,
        min=0.0,
        max=10000.0,
    )

    # ── Anisotropy (warp/weft directional stiffness) ─────────────────────────
    use_anisotropy: BoolProperty(
        name="Anisotropic Stiffness",
        description=(
            "Enable per-direction warp/weft tension, compression, and "
            "bending stiffness"),
        default=False,
    )
    anisotropy_uv_map: StringProperty(
        name="Material Direction UV Map",
        description=(
            "Explicit seam-free UV map used by GPUCloth as warp (U) and "
            "weft (V) material coordinates"),
        default="",
    )
    tension_u: FloatProperty(
        name="Tension U (Warp)",
        description="Stretch stiffness along warp (U) direction. 0 = use isotropic tension",
        default=0.0, min=0.0, max=10000.0,
    )
    tension_v: FloatProperty(
        name="Tension V (Weft)",
        description="Stretch stiffness along weft (V) direction. 0 = use isotropic tension",
        default=0.0, min=0.0, max=10000.0,
    )
    compression_u: FloatProperty(
        name="Compression U (Warp)",
        description="Compression stiffness along warp. 0 = use isotropic compression",
        default=0.0, min=0.0, max=10000.0,
    )
    compression_v: FloatProperty(
        name="Compression V (Weft)",
        description="Compression stiffness along weft. 0 = use isotropic compression",
        default=0.0, min=0.0, max=10000.0,
    )
    bending_u: FloatProperty(
        name="Bending U (Warp)",
        description="Bending stiffness along warp. 0 = use isotropic bending",
        default=0.0, min=0.0, max=10000.0,
    )
    bending_v: FloatProperty(
        name="Bending V (Weft)",
        description="Bending stiffness along weft. 0 = use isotropic bending",
        default=0.0, min=0.0, max=10000.0,
    )
    max_tension_u: FloatProperty(
        name="Max Tension U",
        description="Maximum tension clamping along warp",
        default=500.0, min=0.0, max=10000.0,
    )
    max_tension_v: FloatProperty(
        name="Max Tension V",
        description="Maximum tension clamping along weft",
        default=500.0, min=0.0, max=10000.0,
    )
    max_compression_u: FloatProperty(
        name="Max Compression U",
        description="Maximum compression clamping along warp",
        default=500.0, min=0.0, max=10000.0,
    )
    max_compression_v: FloatProperty(
        name="Max Compression V",
        description="Maximum compression clamping along weft",
        default=500.0, min=0.0, max=10000.0,
    )
    max_bend_u: FloatProperty(
        name="Max Bend U",
        description="Maximum bending clamping along warp",
        default=500.0, min=0.0, max=10000.0,
    )
    max_bend_v: FloatProperty(
        name="Max Bend V",
        description="Maximum bending clamping along weft",
        default=500.0, min=0.0, max=10000.0,
    )

    # ── Advanced Solver Config ──────────────────────────────────────────────
    solver_iterations: IntProperty(
        name="Iterations",
        description="XPBD/PD/Mil2 iterations per substep",
        default=10, min=1, max=200,
    )

    solver_omega: FloatProperty(
        name="Chebyshev ω",
        description="Chebyshev acceleration omega (1.0 = off, 1.5 = default)",
        default=1.5, min=0.5, max=2.0,
    )

    use_small_steps: BoolProperty(
        name="Small Steps",
        description="XPBD: combine substeps×iterations into single-iteration small steps",
        default=False,
    )

    use_adaptive: BoolProperty(
        name="Adaptive Convergence",
        description="Early exit when position change < tolerance (Mil2/PD)",
        default=False,
    )

    solver_max_iterations: IntProperty(
        name="Max Iterations",
        description="Safety cap for adaptive mode",
        default=100, min=10, max=500,
    )

    solver_convergence_tol: FloatProperty(
        name="Convergence Tol",
        description="L∞ norm of position change (meters) for adaptive early exit",
        default=0.001, min=0.0001, max=0.1, soft_max=0.01,
    )

    use_per_type_budget: BoolProperty(
        name="Per-Type Budget",
        description="XPBD: separate iteration count per constraint type",
        default=False,
    )

    ptb_stretch: IntProperty(
        name="Stretch Iter", default=10, min=1, max=100,
    )
    ptb_bending: IntProperty(
        name="Bending Iter", default=5, min=1, max=100,
    )
    ptb_shear: IntProperty(
        name="Shear Iter", default=5, min=1, max=100,
    )
    ptb_seam: IntProperty(
        name="Seam Iter", default=20, min=1, max=100,
    )

    vgroup_intern: StringProperty(
        name="Vertex Group",
        description="Vertex group for scaling internal spring stiffness",
        default="",
    )

    # ── Pressure ───────────────────────────────────────────────────────────
    use_pressure: BoolProperty(
        name="Pressure",
        description="Enable internal pressure simulation",
        default=False,
    )

    uniform_pressure_force: FloatProperty(
        name="Pressure",
        description="Uniform pressure force constantly applied to the mesh (can be negative)",
        default=0.0,
        min=-100.0,
        max=100.0,
    )

    target_volume: FloatProperty(
        name="Target Volume",
        description="Equilibrium volume the mesh wants to expand to (0 = use rest volume)",
        default=0.0,
        min=0.0,
        max=1000.0,
    )

    use_pressure_volume: BoolProperty(
        name="Use Custom Volume",
        description="Use Target Volume instead of the initial mesh volume",
        default=False,
    )

    pressure_factor: FloatProperty(
        name="Factor",
        description="Pressure scaling factor: pressure = ((V/V0 - 1) + uniform) * factor",
        default=1.0,
        min=0.0,
        max=100.0,
    )

    fluid_density: FloatProperty(
        name="Fluid Density",
        description="Density of the fluid inside/outside for hydrostatic pressure gradient",
        default=0.0,
        min=-10.0,
        max=10.0,
    )

    vgroup_pressure: StringProperty(
        name="Pressure Vertex Group",
        description="Vertex group for scaling pressure",
        default="",
    )

    # ── Shape / Pinning ────────────────────────────────────────────────────
    vgroup_mass: StringProperty(
        name="Pin Group",
        description="Vertex group for pinning vertices (zero weight = free, full weight = pinned)",
        default="",
    )

    goalspring: FloatProperty(
        name="Pin Stiffness",
        description="Stiffness of goal springs (pinning force)",
        default=1.0,
        min=0.0,
        max=100.0,
    )

    goalfrict: FloatProperty(
        name="Pin Friction",
        description="Friction/damping applied to pinned vertices",
        default=0.0,
        min=0.0,
        max=1000.0,
    )

    mingoal: FloatProperty(
        name="Min Goal Factor",
        description="Minimum Blender goal factor",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    maxgoal: FloatProperty(
        name="Max Goal Factor",
        description="Maximum pin goal factor",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    defgoal: FloatProperty(
        name="Default Goal Factor",
        description="Goal factor for vertices absent from the pin group",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    shrink_min: FloatProperty(
        name="Shrink Min",
        description="Min shrinkage factor: 0=none, 1=shrink to nothing, -1=double edge length",
        default=0.0,
        min=-1.0,
        max=1.0,
    )

    shrink_max: FloatProperty(
        name="Shrink Max",
        description="Max shrinkage factor: 0=none, 1=shrink to nothing, -1=double edge length",
        default=0.0,
        min=-1.0,
        max=1.0,
    )

    shapekey_rest: StringProperty(
        name="Rest Shape Key",
        description="Shape key used as rest configuration for the cloth simulation",
        default="",
    )

    use_dynamic_mesh: BoolProperty(
        name="Dynamic Mesh",
        description="Allow the base mesh to deform in real-time during simulation",
        default=False,
    )

    # ── Object Collision ────────────────────────────────────────────────────
    use_object_collision: BoolProperty(
        name="Object Collision",
        description="Enable collision against Blender collision objects",
        default=True,
    )

    collision_friction: FloatProperty(
        name="Friction",
        description="Object collision friction",
        default=5.0,
        min=0.0,
        max=80.0,
    )

    collision_damping: FloatProperty(
        name="Damping",
        description="Object collision damping",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    collision_quality: IntProperty(
        name="Collision Quality",
        description="Collision iterations per simulation step",
        default=2,
        min=1,
        max=32767,
    )

    epsilon: FloatProperty(
        name="Distance",
        description="Minimum distance for object collisions (m)",
        default=0.015,
        min=0.001,
        max=1.0,
        subtype='DISTANCE',
    )

    selfepsilon: FloatProperty(
        name="Self Distance",
        description="Minimum distance for self-collisions (m)",
        default=0.015,
        min=0.001,
        max=0.1,
        subtype='DISTANCE',
    )

    self_collision_friction: FloatProperty(
        name="Self Friction",
        description="Blender self-collision friction before native scaling",
        default=5.0,
        min=0.0,
        max=80.0,
    )

    clamp: FloatProperty(
        name="Object Impulse Clamp",
        description="Maximum impulse for object collision correction",
        default=0.0,
        min=0.0,
        max=100.0,
    )

    self_clamp: FloatProperty(
        name="Self Impulse Clamp",
        description="Maximum impulse for self-collision correction",
        default=0.0,
        min=0.0,
        max=100.0,
    )

    collision_collection: PointerProperty(
        name="Collision Collection",
        description="Restrict object collisions to objects in this collection",
        type=bpy.types.Collection,
    )

    vgroup_objcol: StringProperty(
        name="Exclude Objects VGroup",
        description="Vertex group excluding vertices from object collisions (0=excluded, 1=fully collide)",
        default="",
    )

    vgroup_selfcol: StringProperty(
        name="Exclude Self VGroup",
        description=(
            "Vertices with any positive group weight are excluded from "
            "self-collisions"),
        default="",
    )

    # ── Property Weights (stiffness scaling groups) ──────────────────────────
    vgroup_struct: StringProperty(
        name="Structural Group",
        description="Vertex group for scaling structural stiffness",
        default="",
    )

    vgroup_bend: StringProperty(
        name="Bending Group",
        description="Vertex group for scaling bending stiffness",
        default="",
    )

    vgroup_shear: StringProperty(
        name="Shear Group",
        description="Vertex group for scaling shear stiffness",
        default="",
    )

    vgroup_shrink: StringProperty(
        name="Shrinking Group",
        description="Vertex group for shrinking cloth",
        default="",
    )

    # ── Field Weights ──────────────────────────────────────────────────────
    effector_weights: PointerProperty(
        name="Field Weights",
        description="Per-field-type effector weights for cloth simulation",
        type=GPUClothEffectorWeights,
    )

    eff_force_scale: FloatProperty(
        name="Effector Force",
        description="Scaling of effector forces",
        default=1000.0,
        min=0.0,
        max=100000.0,
    )

    eff_wind_scale: FloatProperty(
        name="Effector Wind",
        description="Scaling of effector wind forces",
        default=250.0,
        min=0.0,
        max=100000.0,
    )

    # ── Proxy-res simulation ──────────────────────────────────────────────
    use_proxy: BoolProperty(
        name="Proxy Simulation",
        description="Simulate coarse proxy mesh, then upsample to hi-res render mesh on GPU",
        default=False,
    )

    proxy_object: PointerProperty(
        name="Proxy Mesh",
        description="Object with coarse mesh for simulation (fewer vertices = faster)",
        type=bpy.types.Object,
    )

    proxy_nx: IntProperty(
        name="Proxy NX",
        description="Proxy grid cells along X axis",
        default=8,
        min=2,
        max=256,
    )

    proxy_ny: IntProperty(
        name="Proxy NY",
        description="Proxy grid cells along Y axis",
        default=8,
        min=2,
        max=256,
    )

    hi_nx: IntProperty(
        name="Hi-res NX",
        description="Render grid cells along X axis",
        default=32,
        min=2,
        max=1024,
    )

    hi_ny: IntProperty(
        name="Hi-res NY",
        description="Render grid cells along Y axis",
        default=32,
        min=2,
        max=1024,
    )

    num_sheets: IntProperty(
        name="Cloth Layers",
        description="Number of layers in multi-layer system",
        default=1,
        min=1,
        max=8,
    )

    proxy_scene_type: IntProperty(
        name="Scene Type",
        description="ProxySim test scene: 0=DrapeOnSphere, 1=TwistTest, 2=MultiLayerDrop, 3=CushionDrop",
        default=0,
        min=0,
        max=3,
    )


# ===========================================================================
#  PropertyGroup для сцены — гравитация + кэш
# ===========================================================================

class GPUClothSceneSettings(PropertyGroup):
    """Scene-level GPUCloth settings (context.scene.gpu_cloth_helper)."""

    # ── Gravity ──────────────────────────────────────────────────────────────
    gravity_x: FloatProperty(
        name="Gravity X",
        description="Gravitational acceleration along X (m/s^2)",
        default=0.0,
    )

    gravity_y: FloatProperty(
        name="Gravity Y",
        description="Gravitational acceleration along Y (m/s^2)",
        default=0.0,
    )

    gravity_z: FloatProperty(
        name="Gravity Z",
        description="Gravitational acceleration along Z (m/s^2)",
        default=-9.81,
    )

    # ── Simulation cache ─────────────────────────────────────────────────────

    cache_dir: StringProperty(
        name="Cache Directory",
        description="Directory for per-frame simulation cache files",
        default="//gpucloth_cache/",
        subtype='DIR_PATH',
        options=vcu.get_dir_path_property_options(),
    )

    is_baked: BoolProperty(
        name="Baked",
        description="True if simulation cache is fully written to disk",
        default=False,
    )

    bake_start: IntProperty(
        name="Start Frame",
        description="First frame of bake range",
        default=1,
        min=0,
    )

    bake_end: IntProperty(
        name="End Frame",
        description="Last frame of bake range",
        default=250,
        min=1,
    )

    bake_progress: IntProperty(
        name="Progress",
        description="Current bake progress percentage",
        default=0,
        min=0,
        max=100,
        subtype='PERCENTAGE',
    )

    playback_mode: BoolProperty(
        name="Cache Playback",
        description="Read vertex positions from cache (GPU-direct, zero-copy) instead of live simulation",
        default=False,
    )


# ===========================================================================
#  Регистрация
# ===========================================================================

_PROPERTY_CLASSES = [
    GPUClothEffectorWeights,
    GPUClothObjectSettings,
    GPUClothSceneSettings,
]


def register():
    for cls in _PROPERTY_CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Object.GPUCloth = PointerProperty(
        name="GPU Cloth Settings",
        type=GPUClothObjectSettings,
    )
    bpy.types.Scene.gpu_cloth_helper = PointerProperty(
        name="GPU Cloth Scene Settings",
        type=GPUClothSceneSettings,
    )
    bpy.types.Scene.gpu_cloth_springs_built = BoolProperty(
        name="Cloth Springs Built",
        default=False,
    )


def unregister():
    if hasattr(bpy.types.Scene, "gpu_cloth_springs_built"):
        del bpy.types.Scene.gpu_cloth_springs_built
    if hasattr(bpy.types.Scene, "gpu_cloth_helper"):
        del bpy.types.Scene.gpu_cloth_helper
    if hasattr(bpy.types.Object, "GPUCloth"):
        del bpy.types.Object.GPUCloth

    for cls in reversed(_PROPERTY_CLASSES):
        bpy.utils.unregister_class(cls)
