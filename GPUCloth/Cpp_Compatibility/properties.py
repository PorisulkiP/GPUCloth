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
#    Шёлк:  плотность ~1.3 г/см³, толщина ~0.1 мм, очень лёгкий и текучий
#    Хлопок: ~1.5 г/см³, ~0.3 мм, умеренная драпировка
#    Деним: ~1.5 г/см³, ~1 мм, жёсткое плотное плетение
#    Кожа:  ~0.9 г/см³, ~1–2 мм, жёсткая на растяжение, высокий изгиб
#    Резина: ~1.5 г/см³, ~1–3 мм, упругая, низкий изгиб, высокое трение

MATERIAL_PRESETS = {
    # ── XPBD (Macklin 2016) ─────────────────────────────────────────────────
    # Compliance α = 1/(k·Δt²).  Больше stiffness → жёстче.
    # Требует больше подшагов для жёстких материалов.
    'XPBD': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 10, 'bending_model': 'ANGULAR',
            'tension': 4.0,  'compression': 2.5,  'shear': 1.5,  'bending_stiffness':0.03,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.5,   'bending_damping': 0.05,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 5, 'bending_model': 'ANGULAR',
            'tension': 15.0, 'compression': 15.0, 'shear': 5.0,  'bending_stiffness':0.5,
            'tension_damp': 5.0, 'compression_damp': 5.0,
            'shear_damp': 5.0,   'bending_damping': 0.5,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 40.0, 'compression': 40.0, 'shear': 20.0, 'bending_stiffness':5.0,
            'tension_damp': 10.0, 'compression_damp': 10.0,
            'shear_damp': 8.0,    'bending_damping': 2.0,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 80.0, 'compression': 80.0, 'shear': 10.0, 'bending_stiffness':15.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 5.0,    'bending_damping': 5.0,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 8, 'bending_model': 'LINEAR',
            'tension': 50.0, 'compression': 50.0, 'shear': 25.0, 'bending_stiffness':1.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 12.0,   'bending_damping': 1.0,
        },
    },

    # ── PD (Bouaziz 2014) ───────────────────────────────────────────────────
    # Глобальный solve → сходимость быстрее → можно выше stiffness при меньшем
    # числе подшагов.  Веса ≈ 2× XPBD для эквивалентного поведения.
    'PD': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 8.0,  'compression': 5.0,  'shear': 3.0,  'bending_stiffness':0.05,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.5,   'bending_damping': 0.05,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 30.0, 'compression': 30.0, 'shear': 10.0, 'bending_stiffness':1.0,
            'tension_damp': 5.0, 'compression_damp': 5.0,
            'shear_damp': 5.0,   'bending_damping': 0.5,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 80.0, 'compression': 80.0, 'shear': 35.0, 'bending_stiffness':10.0,
            'tension_damp': 10.0, 'compression_damp': 10.0,
            'shear_damp': 8.0,    'bending_damping': 2.0,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 150.0, 'compression': 150.0, 'shear': 20.0, 'bending_stiffness':25.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 5.0,    'bending_damping': 5.0,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 5, 'bending_model': 'LINEAR',
            'tension': 100.0, 'compression': 100.0, 'shear': 50.0, 'bending_stiffness':2.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 12.0,   'bending_damping': 1.0,
        },
    },

    # ── MGPBD (Xian 2019) ──────────────────────────────────────────────────
    # AMG-ускорение → меньше подшагов, лучше для мягких тканей.
    # Значения ≈ 0.7× XPBD.
    'MGPBD': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 3.0,  'compression': 2.0,  'shear': 1.0,  'bending_stiffness':0.02,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.3,   'bending_damping': 0.03,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 3, 'bending_model': 'ANGULAR',
            'tension': 10.0, 'compression': 10.0, 'shear': 3.0,  'bending_stiffness':0.3,
            'tension_damp': 3.0, 'compression_damp': 3.0,
            'shear_damp': 2.0,   'bending_damping': 0.3,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 4, 'bending_model': 'ANGULAR',
            'tension': 30.0, 'compression': 30.0, 'shear': 12.0, 'bending_stiffness':3.0,
            'tension_damp': 8.0, 'compression_damp': 8.0,
            'shear_damp': 6.0,   'bending_damping': 1.5,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 5, 'bending_model': 'ANGULAR',
            'tension': 60.0, 'compression': 60.0, 'shear': 8.0,  'bending_stiffness':10.0,
            'tension_damp': 12.0, 'compression_damp': 12.0,
            'shear_damp': 4.0,    'bending_damping': 4.0,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 5, 'bending_model': 'LINEAR',
            'tension': 40.0, 'compression': 40.0, 'shear': 18.0, 'bending_stiffness':0.5,
            'tension_damp': 12.0, 'compression_damp': 12.0,
            'shear_damp': 8.0,    'bending_damping': 0.5,
        },
    },

    # ── Mil2 (Li 2020) ─────────────────────────────────────────────────────
    # Barrier-based, диапазон жёсткости аналогичен XPBD.
    'Mil2': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 10, 'bending_model': 'ANGULAR',
            'tension': 5.0,  'compression': 3.0,  'shear': 1.5,  'bending_stiffness':0.03,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.5,   'bending_damping': 0.05,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 5, 'bending_model': 'ANGULAR',
            'tension': 15.0, 'compression': 15.0, 'shear': 5.0,  'bending_stiffness':0.5,
            'tension_damp': 5.0, 'compression_damp': 5.0,
            'shear_damp': 5.0,   'bending_damping': 0.5,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 40.0, 'compression': 40.0, 'shear': 20.0, 'bending_stiffness':5.0,
            'tension_damp': 10.0, 'compression_damp': 10.0,
            'shear_damp': 8.0,    'bending_damping': 2.0,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 80.0, 'compression': 80.0, 'shear': 10.0, 'bending_stiffness':15.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 5.0,    'bending_damping': 5.0,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 8, 'bending_model': 'LINEAR',
            'tension': 50.0, 'compression': 50.0, 'shear': 25.0, 'bending_stiffness':1.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 12.0,   'bending_damping': 1.0,
        },
    },

    # ── OGC (Chen 2025) ────────────────────────────────────────────────────
    # Базовые параметры как у XPBD + OGC-специфичные (radius, friction).
    'OGC': {
        'SILK': {
            'vertex_mass': 0.04, 'quality_step': 10, 'bending_model': 'ANGULAR',
            'tension': 4.0,  'compression': 2.5,  'shear': 1.5,  'bending_stiffness':0.03,
            'tension_damp': 0.5, 'compression_damp': 0.5,
            'shear_damp': 0.5,   'bending_damping': 0.05,
            'use_self_collision': True, 'ogc_radius': 80.0, 'ogc_friction': 0.1,
        },
        'COTTON': {
            'vertex_mass': 0.3, 'quality_step': 5, 'bending_model': 'ANGULAR',
            'tension': 15.0, 'compression': 15.0, 'shear': 5.0,  'bending_stiffness':0.5,
            'tension_damp': 5.0, 'compression_damp': 5.0,
            'shear_damp': 5.0,   'bending_damping': 0.5,
            'use_self_collision': True, 'ogc_radius': 150.0, 'ogc_friction': 0.3,
        },
        'DENIM': {
            'vertex_mass': 0.5, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 40.0, 'compression': 40.0, 'shear': 20.0, 'bending_stiffness':5.0,
            'tension_damp': 10.0, 'compression_damp': 10.0,
            'shear_damp': 8.0,    'bending_damping': 2.0,
            'use_self_collision': True, 'ogc_radius': 200.0, 'ogc_friction': 0.5,
        },
        'LEATHER': {
            'vertex_mass': 0.8, 'quality_step': 6, 'bending_model': 'ANGULAR',
            'tension': 80.0, 'compression': 80.0, 'shear': 10.0, 'bending_stiffness':15.0,
            'tension_damp': 15.0, 'compression_damp': 15.0,
            'shear_damp': 5.0,    'bending_damping': 5.0,
            'use_self_collision': True, 'ogc_radius': 250.0, 'ogc_friction': 0.6,
        },
        'RUBBER': {
            'vertex_mass': 1.2, 'quality_step': 8, 'bending_model': 'LINEAR',
            'tension': 50.0, 'compression': 50.0, 'shear': 25.0, 'bending_stiffness':1.0,
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
    """Применить выбранный пресет материала для текущего солвера."""
    preset_name = self.material_preset
    if preset_name == 'CUSTOM':
        return
    data = MATERIAL_PRESETS.get(self.solver_type, {}).get(preset_name)
    if data is None:
        return
    for prop, value in data.items():
        setattr(self, prop, value)


def _on_solver_change(self, context):
    """При смене солвера — переприменить пресет (значения отличаются)."""
    if self.material_preset != 'CUSTOM':
        _apply_preset(self, context)


# ===========================================================================
#  PropertyGroup для объекта — настройки ткани
# ===========================================================================

class GPUClothObjectSettings(PropertyGroup):
    """Настройки симуляции ткани для конкретного объекта (OBJ.GPUCloth)."""

    # ── Основные ────────────────────────────────────────────────────────────
    is_active: BoolProperty(
        name="Включить GPUCloth",
        description="Симулировать этот объект как ткань на GPU",
        default=False,
    )

    vertex_mass: FloatProperty(
        name="Масса вершины",
        description="Масса одной вершины ткани (кг)",
        default=0.3,
        min=0.001,
        max=10.0,
        unit='MASS',
    )

    quality_step: IntProperty(
        name="Шаги качества",
        description="Количество подшагов солвера за один кадр",
        default=5,
        min=1,
        max=80,
    )

    speed_multiplier: FloatProperty(
        name="Множитель скорости",
        description="Масштаб времени симуляции (time_scale)",
        default=1.0,
        min=0.0,
        max=10.0,
    )

    bending_model: EnumProperty(
        name="Модель изгиба",
        description="Тип ограничения на изгиб ткани",
        items=[
            ('LINEAR',  "Линейный",  "Линейная жёсткость изгиба"),
            ('ANGULAR', "Угловой",   "Угловая жёсткость изгиба (реалистичнее)"),
        ],
        default='ANGULAR',
    )

    # ── Солвер ──────────────────────────────────────────────────────────────
    solver_type: EnumProperty(
        name="Солвер",
        description="Алгоритм GPU-симуляции ткани",
        items=[
            ('XPBD',  "XPBD",  "Extended Position-Based Dynamics (Macklin 2016)"),
            ('PD',    "PD",    "Projective Dynamics с Chebyshev-Jacobi ускорением"),
            ('MGPBD', "MGPBD", "Многоуровневый PBD с Algebraic Multigrid"),
            ('Mil2',  "Mil2",  "Non-distance barriers + Subspace Reuse"),
            ('OGC',   "OGC",   "Offset Geometric Contact — самостолкновение"),
        ],
        default='XPBD',
        update=_on_solver_change,
    )

    # ── Пресет материала ───────────────────────────────────────────────────
    material_preset: EnumProperty(
        name="Материал",
        description=(
            "Пресет физических свойств материала. "
            "Значения подобраны под текущий солвер"
        ),
        items=[
            ('CUSTOM',  "Свой",    "Ручная настройка параметров"),
            ('SILK',    "Шёлк",    "Лёгкий, текучий, минимальный изгиб"),
            ('COTTON',  "Хлопок",  "Умеренная драпировка, средняя жёсткость"),
            ('DENIM',   "Деним",   "Плотная ткань, жёсткий изгиб и сдвиг"),
            ('LEATHER', "Кожа",    "Тяжёлая, очень жёсткая на растяжение"),
            ('RUBBER',  "Резина",  "Упругая, тяжёлая, низкий изгиб"),
        ],
        default='CUSTOM',
        update=_apply_preset,
    )

    # ── Параметры материала (жёсткость) ────────────────────────────────────
    tension: FloatProperty(
        name="Растяжение",
        description="Жёсткость ткани на растяжение (stretch stiffness)",
        default=15.0,
        min=0.0,
        max=500.0,
    )

    compression: FloatProperty(
        name="Сжатие",
        description="Жёсткость ткани на сжатие (compression stiffness)",
        default=15.0,
        min=0.0,
        max=500.0,
    )

    shear: FloatProperty(
        name="Сдвиг",
        description="Жёсткость ткани на сдвиг (shear stiffness)",
        default=5.0,
        min=0.0,
        max=500.0,
    )

    bending_stiffness: FloatProperty(
        name="Изгиб",
        description="Жёсткость изгиба ткани (bending stiffness)",
        default=0.5,
        min=0.0,
        max=100.0,
    )

    # ── Параметры материала (демпфирование) ────────────────────────────────
    tension_damp: FloatProperty(
        name="Демпфирование растяжения",
        description="Затухание колебаний при растяжении",
        default=5.0,
        min=0.0,
        max=50.0,
    )

    compression_damp: FloatProperty(
        name="Демпфирование сжатия",
        description="Затухание колебаний при сжатии",
        default=5.0,
        min=0.0,
        max=50.0,
    )

    shear_damp: FloatProperty(
        name="Демпфирование сдвига",
        description="Затухание колебаний при сдвиге",
        default=5.0,
        min=0.0,
        max=50.0,
    )

    bending_damping: FloatProperty(
        name="Демпфирование изгиба",
        description="Затухание колебаний изгиба",
        default=0.5,
        min=0.0,
        max=50.0,
    )

    # ── Самостолкновения (OGC) ────────────────────────────────────────────
    use_self_collision: BoolProperty(
        name="Самостолкновения",
        description=(
            "Включить Offset Geometric Contact (OGC, Chen et al. 2025) — "
            "penetration-free самостолкновения. Работает с любым солвером"
        ),
        default=False,
    )

    ogc_radius: FloatProperty(
        name="Радиус контакта",
        description=(
            "Радиус зоны контакта OGC (мм). "
            "Больше — слои держатся дальше друг от друга. "
            "Авто: r = 16.6 × avg_edge²"
        ),
        default=150.0,
        min=1.0,
        max=500.0,
        subtype='NONE',
        unit='NONE',
    )

    ogc_kc: FloatProperty(
        name="Жёсткость контакта",
        description=(
            "Жёсткость OGC-контакта (kc, PD-режим). "
            "Меньше — мягче и стабильнее. "
            "Авто: kc = 1.15 / radius_m"
        ),
        default=7.0,
        min=1.0,
        max=100000.0,
    )

    ogc_friction: FloatProperty(
        name="Трение слоёв",
        description=(
            "Трение между слоями ткани (OGC). "
            "0 = скользят свободно, 1 = не скользят. "
            "Рекомендуется 0.3–0.6 для twist-сцен"
        ),
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    ogc_gamma_p: FloatProperty(
        name="Gamma P",
        description=(
            "Доля коррекции за проход OGC (Mil²-режим, параметр §3.6). "
            "Должен быть строго < 0.5. "
            "0.45 = стандарт, 0.48 = для плотного скручивания"
        ),
        default=0.45,
        min=0.05,
        max=0.49,
    )

    # ── Proxy-res симуляция ─────────────────────────────────────────────────
    use_proxy: BoolProperty(
        name="Proxy симуляция",
        description=(
            "Симулировать грубый proxy-меш, затем апсэмплировать до hi-res рендерного меша. "
            "GPU выполняет scatter → ClothVertex.x → ProxySim_apply → hi-res позиции"
        ),
        default=False,
    )

    proxy_object: PointerProperty(
        name="Proxy меш",
        description="Объект с грубой сеткой для симуляции (меньше вершин = быстрее)",
        type=bpy.types.Object,
    )

    proxy_nx: IntProperty(
        name="Proxy NX",
        description="Количество ячеек proxy-сетки по оси X",
        default=8,
        min=2,
        max=256,
    )

    proxy_ny: IntProperty(
        name="Proxy NY",
        description="Количество ячеек proxy-сетки по оси Y",
        default=8,
        min=2,
        max=256,
    )

    hi_nx: IntProperty(
        name="Hi-res NX",
        description="Количество ячеек рендерной сетки по оси X",
        default=32,
        min=2,
        max=1024,
    )

    hi_ny: IntProperty(
        name="Hi-res NY",
        description="Количество ячеек рендерной сетки по оси Y",
        default=32,
        min=2,
        max=1024,
    )

    num_sheets: IntProperty(
        name="Слоёв ткани",
        description="Количество слоёв в многослойной системе",
        default=1,
        min=1,
        max=8,
    )

    proxy_scene_type: IntProperty(
        name="Тип сцены",
        description=(
            "Тип тестовой сцены для ProxySim_create: "
            "0=DrapeOnSphere, 1=TwistTest, 2=MultiLayerDrop, 3=CushionDrop"
        ),
        default=0,
        min=0,
        max=3,
    )


# ===========================================================================
#  PropertyGroup для сцены — гравитация + кэш
# ===========================================================================

class GPUClothSceneSettings(PropertyGroup):
    """Настройки GPUCloth уровня сцены (context.scene.gpu_cloth_helper)."""

    # ── Гравитация ───────────────────────────────────────────────────────────
    gravity_x: FloatProperty(
        name="Gravity X",
        description="Ускорение свободного падения по X (м/с²)",
        default=0.0,
    )

    gravity_y: FloatProperty(
        name="Gravity Y",
        description="Ускорение свободного падения по Y (м/с²)",
        default=0.0,
    )

    gravity_z: FloatProperty(
        name="Gravity Z",
        description="Ускорение свободного падения по Z (м/с²)",
        default=-9.81,
    )

    # ── Кэш симуляции ────────────────────────────────────────────────────────
    #
    # Архитектура кэша:
    #   ЗАПИСЬ (Phase 1, CPU-destination):
    #     SIM_solver() → SIM_get_cloth_verts() → float32[] → Cache_write_frame_async()
    #     C++ пишет в фоне: pinned RAM → DMA → NVMe
    #
    #   ЧТЕНИЕ (Phase 2, GPU-destination, zero-copy):
    #     NVMe → DMA → D3D12 resource (VRAM) → CUDA external memory (view)
    #     → scatter_gpu_kernel → ClothVertex.x
    #     → cudaMemcpyAsync D2H → h_flat → foreach_set (viewport)

    cache_dir: StringProperty(
        name="Директория кэша",
        description="Папка для хранения файлов кэша симуляции (по кадрам)",
        default="//gpucloth_cache/",
        subtype='DIR_PATH',
        options=vcu.get_dir_path_property_options(),
    )

    is_baked: BoolProperty(
        name="Запечено",
        description="True если кэш симуляции полностью записан на диск",
        default=False,
    )

    bake_start: IntProperty(
        name="Начальный кадр",
        description="Первый кадр диапазона запекания",
        default=1,
        min=0,
    )

    bake_end: IntProperty(
        name="Конечный кадр",
        description="Последний кадр диапазона запекания",
        default=250,
        min=1,
    )

    bake_progress: IntProperty(
        name="Прогресс",
        description="Прогресс текущего запекания в процентах",
        default=0,
        min=0,
        max=100,
        subtype='PERCENTAGE',
    )

    playback_mode: BoolProperty(
        name="Воспроизведение из кэша",
        description=(
            "Читать позиции вершин из кэша (GPU-direct, zero-copy) "
            "вместо живой симуляции. Требует is_baked=True"
        ),
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

    # Привязка к типам Blender
    bpy.types.Object.GPUCloth = PointerProperty(
        name="GPU Cloth Settings",
        type=GPUClothObjectSettings,
    )
    bpy.types.Scene.gpu_cloth_helper = PointerProperty(
        name="GPU Cloth Scene Settings",
        type=GPUClothSceneSettings,
    )
    # Флаг «пружины построены» — остаётся прямым свойством сцены
    # для обратной совместимости со старым кодом
    bpy.types.Scene.gpu_cloth_springs_built = BoolProperty(
        name="Cloth Springs Built",
        default=False,
    )


def unregister():
    # Удаляем в обратном порядке
    if hasattr(bpy.types.Scene, "gpu_cloth_springs_built"):
        del bpy.types.Scene.gpu_cloth_springs_built
    if hasattr(bpy.types.Scene, "gpu_cloth_helper"):
        del bpy.types.Scene.gpu_cloth_helper
    if hasattr(bpy.types.Object, "GPUCloth"):
        del bpy.types.Object.GPUCloth

    for cls in reversed(_PROPERTY_CLASSES):
        bpy.utils.unregister_class(cls)
