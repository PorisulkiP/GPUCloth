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
        ],
        default='XPBD',
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
