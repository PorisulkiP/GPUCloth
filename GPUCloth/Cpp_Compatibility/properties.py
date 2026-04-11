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
    if self.is_active:
        from . import operators as ops
        if ops.g_dll is None:
            loader = ops.GPUCloth_LoadDLL()
            loader.load_dll()


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
            'vertex_mass': 1.2, 'quality_step': 5, 'bending_model': 'LINEAR',
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
            'vertex_mass': 1.2, 'quality_step': 8, 'bending_model': 'LINEAR',
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


# ===========================================================================
#  Локализованные items для EnumProperty (вызываются при каждом открытии)
# ===========================================================================

def _bending_model_items(self, context):
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
        max=500.0,
    )

    compression: FloatProperty(
        name="Compression",
        description="Compression stiffness",
        default=15.0,
        min=0.0,
        max=500.0,
    )

    shear: FloatProperty(
        name="Shear",
        description="Shear stiffness",
        default=5.0,
        min=0.0,
        max=500.0,
    )

    bending_stiffness: FloatProperty(
        name="Bending",
        description="Bending stiffness",
        default=0.5,
        min=0.0,
        max=100.0,
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
        max=50.0,
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
