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


# One shared trigger for the cache-input fingerprint.  A callback-less RNA
# property edit produces no depsgraph notification in Blender 4.2 or 5.2, so the
# only cheap way to learn that a solver input was touched is an ``update=``
# callback.  Every fingerprinted property carries this one callback; it does
# nothing but move a counter, and operators._refresh_prepared_inputs decides
# from that counter whether it is worth recomputing the fingerprint.  The
# fingerprint stays the sole owner of *what* changed.
_simulation_input_epoch = {'value': 0}


def _on_simulation_input_change(self, context):
    _simulation_input_epoch['value'] += 1


def _on_is_active_change(self, context):
    _on_simulation_input_change(self, context)
    if self.is_active and self.execution_backend != 'GPU':
        self.execution_backend = 'GPU'
        if self.execution_backend != 'GPU':
            self.is_active = False
            return
    elif not self.is_active and self.execution_backend != 'CPU':
        self.execution_backend = 'CPU'


_backend_switch_active = False


def _on_execution_backend_change(self, context):
    global _backend_switch_active
    _on_simulation_input_change(self, context)
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
        elif self.execution_backend == 'GPU' and context is not None:
            from . import operators as ops
            if bool(getattr(self, "auto_prepare", True)):
                ops.schedule_auto_prepare(obj, context.scene)
    finally:
        _backend_switch_active = False


# ===========================================================================
#  Материальные пресеты — значения параметров ткани для каждого солвера
# ===========================================================================
#
#  Источники значений:
#    PD    — Bouaziz et al. 2014 "Projective Dynamics", §5 (projection weights)
#    Mil2  — Li et al. 2020 "Incremental Potential Contact", §3
#
#  Физические ориентиры (реальные ткани):
#    Silk:    density ~1.3 g/cm³, thickness ~0.1 mm, very light and flowing
#    Cotton:  ~1.5 g/cm³, ~0.3 mm, moderate drape
#    Denim:   ~1.5 g/cm³, ~1 mm, stiff dense weave
#    Leather: ~0.9 g/cm³, ~1–2 mm, stiff in stretch, high bending
#    Rubber:  ~1.5 g/cm³, ~1–3 mm, elastic, low bending, high friction

MATERIAL_PRESETS = {
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
}



# ===========================================================================
#  Пресеты физической мембраны FABRIC
# ===========================================================================
#
#  Единственная модель материала — физическая треугольная мембрана FABRIC.
#  Эти пресеты заполняют её поля.  Каждое число здесь — либо измеренное
#  значение из архива реверс-инжиниринга Marvelous Designer, либо отсутствует.
#  Придуманных констант нет и быть не может.
#
#  Источники (архив D:\worktrees\CUDACloth-md-recon, только чтение):
#
#    Rib_2X2_468gsm
#      Разбор плаща: tmp\md-ui-handshake-20260908\cape-fabric-values\receipt.json
#      — GetFabricInformation.Weight = "468.182 g/m2" для обеих тканей плаща;
#      MD-имя физического пресета Rib_2X2_468gsm — tmp\luna-md-pd-experiments-
#      20260908\cape-material-audit\REPORT_RU.md:10-16.  В CUDACloth это же
#      число уже зафиксировано как 0.46818185 кг/м²
#      (tools\build_cape_scene.py:20-21, operators.py TESTSCENE_MATERIAL).
#
#    "Default" — горизонтальный контрольный материал
#      tmp\luna-md-pd-experiments-20260908\cape-material-audit\REPORT_RU.md:45-62
#      — physical preset `Default`, fDensity = 0.000300000014 г/мм², то есть
#      300.0 г/м².
#
#  Чего в архиве НЕТ (поля пресета намеренно не заполнены):
#
#    Жёсткости.  Экспорт .zfab даёт их под внутренними именами MD (fSuK, fSvK,
#    fHK, fBuK, fBvK, fBhK и *_v2) — cape-material-audit\REPORT_RU.md:26-34,
#    52-62.  Их единицы совпадают с нашими (Н/м: измеренные кривые силы в
#    REPORT_RU.md:38-43 записаны в ньютонах), но взаимно однозначного
#    отображения «коэффициент MD → направленный модуль нашего треугольника»
#    архив не даёт, и он прямо предупреждает против такого присваивания
#    (REPORT_RU.md:98-108: «assigning them to warp/weft/shear/bending UI
#    sliders would be speculation»).  Заполнение этих полей «похожими» числами
#    запрещено правилами проекта: это были бы скрытые пороги и выдуманные
#    константы.  Нужна либо калибровка (протокол сравнения на одном меше), либо
#    отдельный источник единиц — ни того, ни другого в архиве нет.
#
#    Толщина/плотность на единицу объёма.  fThickness = 1.49 мм и
#    fThicknessDensity = 9.27 зарегистрированы (REPORT_RU.md:13-14, 22-25), но
#    их SI-смысл не установлен (REPORT_RU.md:100-102).
#
#    Демпфирование.  Отдельного поля демпфирования ткани в .zfab нет;
#    fAirDamping — глобальная настройка симуляции (REPORT_RU.md:105-106).
#
#    Двухфазная жёсткость/гистерезис.  Кривые «длина → сила» и
#    fFabricBendingType экспортированы (REPORT_RU.md:26, 36-43) и в нашем
#    payload не выражаются.
#
#  Единицы: `fabric_density` в интерфейсе — г/м², и поле пресета называется так
#  же, потому что пишется прямо в RNA-свойство; значение совпадает с записью MD.
#  Никаких мм/с здесь нет: MD-гравитация −9800 мм/с² в наших полях не участвует.
#
#  ---- Второй источник: восемь тканей Houdini Vellum (SideFX Content Library)
#  ----
#
#  Файл: SideFX Content Library «Fabric Samples»
#      https://www.sidefx.com/contentlibrary/fabric-samples/  (417.1 KB, HIP)
#      внутри: VellumMaterialExamples/VellumMaterialExamples.hip
#      автор Andriy Bilichenko, создан в Houdini 18.5, опубликован 20.10.2020;
#      страница помощи Houdini «Downloadable fabrics» (houdini/help/vellum.zip
#      -> fabric.txt:6-28) перечисляет те же восемь тканей.
#
#  Один и тот же объект несёт два узла Vellum Constraints: служебный
#  (сетка/прокси) и тот, что описывает физику ткани.  Числа ниже прочитаны с
#  ФИЗИЧЕСКОГО узла — он и есть материал:
#
#      ткань    узел                          density (raw)   -> г/м²
#      Silk     /obj/sim/vellumcloth8         '0.04'             40
#      Raincoat /obj/sim/vellumcloth3         '0.25'            250
#      Jersey   /obj/sim/vellumcloth6         '0.04'             40
#      Lace     /obj/sim/vellumcloth11        '0.04'             40
#      Velvet   /obj/sim/vellumcloth9         '0.02'             20
#      Wool     /obj/sim/vellumcloth_wool     '0.04'             40
#      Leather  /obj/sim/vellumcloth10        '0.4'             400
#      Jeans    /obj/sim/vellumcloth15        '0.4'             400
#
#  Единица плотности измерена, а не выведена из имени параметра.  У каждого из
#  этих узлов Mass = Calculate Varying, а help Houdini
#  (houdini/help/vellum.zip -> mass_thickness.txt:3) говорит, что масса
#  считается через «the area of a triangle ... and the density of the material
#  to compute the mass».  Прогон узла Silk через hython и замер суммы
#  `mass` по всей геометрии против её площади даёт ровно density:
#      total_mass = 0.0128413 кг, total_area = 0.321032 м²
#      => 0.0128413 / 0.321032 = 0.04 кг/м² = 40 г/м²  == parameter `density`
#  Значит `density` в Vellum — это кг/м² (поверхностная плотность ткани), то
#  есть ровно наша `fabric_density` в г/м² после умножения на 1000.  Пересчёт
#  тривиален и точен: 1 кг/м² = 1000 г/м².  Сама сцена — в метрах и
#  килограммах (unitlength = 1 m, unitmass = 1 kg, scale объекта = 1).
#
#  Чего Vellum для наших полей НЕ даёт (проверено, а не предположено):
#
#    Жёсткости.  Vellum хранит НЕ физический модуль, а безразмерную жёсткость
#    XPBD-ограничения.  Панель задаёт её парой «Stiffness» + «Stiffness
#    Exponent», и закон проверен замером: при exponent = 0, 3, 6, 10 атрибут
#    `stiffness` примитива-ограничения равен ровно 1e0, 1e3, 1e6, 1e10, то
#    есть  k_ограничения = stiffness * 10**exponent.  Дальше решатель переводит
#    её в множитель XPBD по alpha = 1/(stiffness * dt^2) — это записано в
#    архиве реконструкции Vellum: EQUATIONS.md:16, и там же прямо сказано
#    («does not attach a global unit system to these kernels», EQUATIONS.md:4-5;
#    «its units beyond the source equation above» — EQUATIONS.md:170), что
#    единицы жёсткости за пределами этого уравнения источнику неизвестны.
#    Чтобы получить из k размерность Н/м, нужен dt и масштаб масс решателя,
#    которых хост не публикует.  Кроме того сама k — жёсткость РЕБРА, а не
#    материала: справка SideFX по Vellum Constraints SOP
#    (https://www.sidefx.com/docs/houdini/nodes/sop/vellumconstraints.html,
#    раздел Bend) прямо требует поднимать bend stiffness при повышении
#    плотности сетки, то есть k зависит от разрешения меша.  Поэтому
#    «stiffness из Vellum» в наши Н/м не переводится, и поля
#    fabric_tensile_*/fabric_compression_*/fabric_shear_c66 остаются пустыми:
#    подставить сюда число значило бы выдумать коэффициент пересчёта.
#
#    Демпфирование.  Vellum хранит Damping Ratio — безразмерную долю (< 1;
#    в примерах 0.001..0.1), тогда как наши fabric_tensile_damping,
#    fabric_compression_damping и fabric_shear_damping заданы в Н·с/м.  Это
#    разные величины, коэффициента между ними источник не даёт.
#
#    Толщина.  Vellum кладёт в `pscale` (Thickness * Thickness Scale), но
#    поля толщины ткани у нас нет вовсе, а сам pscale служит радиусом
#    столкновений, а не толщиной мембраны.
#
#  Итог по совместимости единиц: единственное поле, которое переносится
#  однозначно, — `fabric_density`.  Каждый пресет Vellum ниже заполняет ТОЛЬКО
#  его и оставляет остальные поля как есть; это не «пресет с несогласованными
#  единицами», а честно неполный пресет с указанным пробелом.
#
#  ---- Третий источник: Luible 2008, Annex G (жёсткости в Н/м) ----
#
#  Luible, C. «Study of mechanical properties in the simulation of 3D
#  garments», These no. 678, Universite de Geneve, 2008, URN
#  urn:nbn:ch:unige-6880, DOI 10.13097/archive-ouverte/unige:688.
#  Таблица — «Annex G: Linear derived fabric input parameters», PDF стр. 244-249
#  (стр. тезиса 231-236).  Это ровно тот источник, которого не хватало: он
#  даёт жёсткости НЕ в безразмерных единицах решателя, а сразу в Н/м.
#
#  Почему единицы Annex G — это наши Н/м.  Тезис сам фиксирует приведение
#  (Annex F, PDF стр. 240-241):
#      «it is important that the units of the measured data are converted to
#       match the units of the computation system: N, mm -> N/m»
#      «KES-f returns the shear rigidity G as characteristic value. However
#       ... If the shear strain is taken instead of the shear angle ... the
#       values is equal to shear modulus and can be taken as linear input
#       parameter.»
#      «G (tan O) = 57.3 G (O degree)»
#  Сырые единицы KES-F там же: «F; Tensile force per unit width. (gf/cm)»,
#  «G: Shear rigidity (gf/cm * degree)», «B: Bending rigidity per unit length».
#  То есть колонка «Elasticity N/m» — это сила на единицу ширины (форма E*t),
#  ровно то, что наш код называет «Physical warp/weft tensile modulus»
#  (properties.py:1160, :1165), а колонка «Shear N/m» — это G, приведённая к
#  деформации сдвига, то есть наш `fabric_shear_c66` (properties.py:1179).
#  Проверка согласованности колонок по самому тезису: значение фланели
#  «118.13 N/m (Mean value of weft and warp G (tan O))» совпадает с ячейкой
#  фланели в Annex G («11_Flannel ... 118 N/m»), то есть колонка Shear — это
#  действительно G(tan O), а не G(degree).  Колонки Bending в этих пресетах НЕ
#  используются: их единица — мкН·м (жёсткость на изгиб), а не Н/м, и поля
#  под неё в payload нет.
#
#  Оси: в Annex G порядок колонок Elasticity — «Weft, Warp, Shear».  В нашем
#  коде U — это warp, V — это weft (properties.py:1080-1081, :1087, :1093,
#  :1160, :1165), поэтому Weft -> fabric_tensile_v, Warp -> fabric_tensile_u.
#
#  Плотность в Annex G — из заголовка образца («... 120 g/m2, 0.61 mm»), это
#  тоже поверхностная плотность в г/м², то есть наша же величина без пересчёта.
#
#  Что заполнено и что нет:
#    * `fabric_density`, `fabric_tensile_u/v`, `fabric_shear_c66` — измеренные
#      значения Annex G.
#    * `<поле>_max` выставлены РАВНЫМИ базовому значению того же измерения.
#      Источник не даёт зависимости от скорости деформации, поэтому потолок
#      ставится ровно на измеренную величину: это не добавляет скрытого
#      запаса, но и не даёт нашему default-потолку молча срезать измеренную
#      величину (у shear потолок по умолчанию 500 Н/м, а у кожи измерено
#      1792.9 Н/м — без этой пары пресет обещал бы больше, чем решатель
#      получает).
#    * `fabric_compression_u/v` — НЕ заполнены и остаются на default.  В KES-F
#      и в Annex G измеряется растяжение (F) и сдвиг (G); отдельной ветви
#      «сжатие» у ткани в этом источнике нет.  Подставить сюда растяжение
#      значило бы стереть разницу между растяжением и сжатием, то есть
#      выдумать отсутствующее измерение.
#    * `fabric_*_damping` — НЕ заполнены: в Annex G есть только коэффициент
#      трения (безразмерный), а не вязкость в Н·с/м.
FABRIC_PRESETS = {
    'MD_DEFAULT': {
        'fabric_density': 300.0,
    },
    'MD_RIB_2X2_468GSM': {
        'fabric_density': 468.18185,
    },
    # Vellum, SideFX Content Library «Fabric Samples» (Houdini 18.5).
    # Плотность — параметр `density` физического узла Vellum Constraints,
    # кг/м², умноженный на 1000.
    'VELLUM_SILK': {
        'fabric_density': 40.0,        # vellumcloth8.density = 0.04
    },
    'VELLUM_RAINCOAT': {
        'fabric_density': 250.0,       # vellumcloth3.density = 0.25
    },
    'VELLUM_JERSEY': {
        'fabric_density': 40.0,        # vellumcloth6.density = 0.04
    },
    'VELLUM_LACE': {
        'fabric_density': 40.0,        # vellumcloth11.density = 0.04
    },
    'VELLUM_VELVET': {
        'fabric_density': 20.0,        # vellumcloth9.density = 0.02
    },
    'VELLUM_WOOL': {
        'fabric_density': 40.0,        # vellumcloth_wool.density = 0.04
    },
    'VELLUM_LEATHER': {
        'fabric_density': 400.0,       # vellumcloth10.density = 0.4
    },
    'VELLUM_DENIM': {
        'fabric_density': 400.0,       # vellumcloth15.density = 0.4
    },
    # Luible 2008, Annex G.  Комментарий у каждого поля — номер образца и
    # ячейка таблицы, чтобы значение проверялось без повторного разбора PDF.
    'LUI_COTTON_SHIRTING': {           # 02, 120 g/m2
        'fabric_density': 120.0,
        'fabric_tensile_u': 2000.0,    # warp
        'fabric_tensile_v': 3000.0,    # weft
        'fabric_shear_c66': 20.0,
        'fabric_tensile_u_max': 2000.0,
        'fabric_tensile_v_max': 3000.0,
        'fabric_shear_c66_max': 20.0,
    },
    'LUI_DENIM': {                     # 01, 380 g/m2
        'fabric_density': 380.0,
        'fabric_tensile_u': 1800.0,
        'fabric_tensile_v': 3000.0,
        'fabric_shear_c66': 102.0,
        'fabric_tensile_u_max': 1800.0,
        'fabric_tensile_v_max': 3000.0,
        'fabric_shear_c66_max': 102.0,
    },
    'LUI_STRETCH_DENIM': {             # 12, 275 g/m2
        'fabric_density': 275.0,
        'fabric_tensile_u': 4000.0,
        'fabric_tensile_v': 400.0,
        'fabric_shear_c66': 71.0,
        'fabric_tensile_u_max': 4000.0,
        'fabric_tensile_v_max': 400.0,
        'fabric_shear_c66_max': 71.0,
    },
    'LUI_SILK_MULBERRY': {             # 07, 15 g/m2
        'fabric_density': 15.0,
        'fabric_tensile_u': 6000.0,
        'fabric_tensile_v': 4500.0,
        'fabric_shear_c66': 10.0,
        'fabric_tensile_u_max': 6000.0,
        'fabric_tensile_v_max': 4500.0,
        'fabric_shear_c66_max': 10.0,
    },
    'LUI_SILK_BOURETTE': {             # 08, 150 g/m2
        'fabric_density': 150.0,
        'fabric_tensile_u': 1500.0,
        'fabric_tensile_v': 1500.0,
        'fabric_shear_c66': 38.0,
        'fabric_tensile_u_max': 1500.0,
        'fabric_tensile_v_max': 1500.0,
        'fabric_shear_c66_max': 38.0,
    },
    'LUI_SILK_TUSSAH': {               # 09, 80 g/m2
        'fabric_density': 80.0,
        'fabric_tensile_u': 4500.0,
        'fabric_tensile_v': 10000.0,
        'fabric_shear_c66': 80.0,
        'fabric_tensile_u_max': 4500.0,
        'fabric_tensile_v_max': 10000.0,
        'fabric_shear_c66_max': 80.0,
    },
    'LUI_WOOL_GABARDINE': {            # 05, 175 g/m2
        'fabric_density': 175.0,
        'fabric_tensile_u': 3000.0,
        'fabric_tensile_v': 2500.0,
        'fabric_shear_c66': 32.0,
        'fabric_tensile_u_max': 3000.0,
        'fabric_tensile_v_max': 2500.0,
        'fabric_shear_c66_max': 32.0,
    },
    'LUI_WOOL_SUITING': {              # 34, 232 g/m2, herringbone
        'fabric_density': 232.0,
        'fabric_tensile_u': 2000.0,
        'fabric_tensile_v': 400.0,
        'fabric_shear_c66': 32.9,
        'fabric_tensile_u_max': 2000.0,
        'fabric_tensile_v_max': 400.0,
        'fabric_shear_c66_max': 32.9,
    },
    'LUI_JERSEY_SINGLE': {             # 21, 172 g/m2, 98% CLY + 2% EL
        'fabric_density': 172.0,
        'fabric_tensile_u': 120.0,
        'fabric_tensile_v': 50.0,
        'fabric_shear_c66': 23.0,
        'fabric_tensile_u_max': 120.0,
        'fabric_tensile_v_max': 50.0,
        'fabric_shear_c66_max': 23.0,
    },
    'LUI_VELVET': {                    # 15, 300 g/m2, 92% CO + 8% CMD
        'fabric_density': 300.0,
        'fabric_tensile_u': 2000.0,
        'fabric_tensile_v': 1000.0,
        'fabric_shear_c66': 58.0,
        'fabric_tensile_u_max': 2000.0,
        'fabric_tensile_v_max': 1000.0,
        'fabric_shear_c66_max': 58.0,
    },
    'LUI_LEATHER': {                   # 32, 815 g/m2, garment leather
        'fabric_density': 815.0,
        'fabric_tensile_u': 3000.0,
        'fabric_tensile_v': 2400.0,
        'fabric_shear_c66': 1792.9,
        'fabric_tensile_u_max': 3000.0,
        'fabric_tensile_v_max': 2400.0,
        'fabric_shear_c66_max': 1792.9,
    },
}


def _apply_fabric_preset(self, context):
    """Apply the selected FABRIC material preset.

    ``CUSTOM`` is a no-op: it means the user owns every field.  A preset that is
    not in the table is a hard error rather than a silent no-op, because a
    preset which reports itself applied and writes nothing is exactly the
    "hidden fallback" the project forbids (a stored project from an older build
    is the reachable case).
    """
    _on_simulation_input_change(self, context)
    preset_name = self.fabric_preset
    if preset_name == 'CUSTOM':
        return
    data = FABRIC_PRESETS.get(str(preset_name))
    if data is None:
        raise RuntimeError(f"unknown FABRIC material preset {preset_name!r}")
    for prop_name, value in data.items():
        if not hasattr(self, prop_name):
            raise RuntimeError(
                f"FABRIC material preset {preset_name!r} names absent "
                f"property {prop_name!r}")
        setattr(self, prop_name, value)


# ===========================================================================
#  Коллбэки обновления пресетов
# ===========================================================================

def _apply_preset(self, context):
    """Apply the selected material preset for the current solver."""
    _on_simulation_input_change(self, context)
    preset_name = self.material_preset
    if preset_name == 'CUSTOM':
        return
    data = MATERIAL_PRESETS.get(self.solver_type, {}).get(preset_name)
    if data is None:
        return
    for prop, value in data.items():
        setattr(self, prop, value)
    # Presets own base stiffness; the native preflight owns max >= base.
    # The CPU bridge may leave max clamps at CPU defaults below the new
    # base, so raise lagging clamps instead of breaking preparation.
    for base_prop, max_prop in (
            ('tension', 'max_tension'),
            ('compression', 'max_compression'),
            ('shear', 'max_shear'),
            ('bending_stiffness', 'max_bend')):
        if getattr(self, max_prop) < getattr(self, base_prop):
            setattr(self, max_prop, getattr(self, base_prop))


def _on_solver_change(self, context):
    """Re-apply preset when solver changes (values differ per solver)."""
    _on_simulation_input_change(self, context)
    if self.material_preset != 'CUSTOM':
        _apply_preset(self, context)


def _on_live_collision_change(self, context):
    """Push a collision setting onto a live prepared cloth without a re-prepare.

    Only the settings the engine applies without invalidating preparation are sent -
    friction, damping, quality/clamp and self-collision friction.  Enabling
    self-collision itself is not among them: main.cpp's GPUCLOTH_FEATURE_SELF_COLLISION
    handler clears runtime->preparation.runnable, so that toggle needs a prepare by the
    engine's own contract and this callback intentionally leaves it alone.
    """
    from . import operators
    _on_simulation_input_change(self, context)
    obj = self.id_data
    if obj is None:
        return
    operators.republish_live_collision_settings(obj)


# ===========================================================================
#  Локализованные items для EnumProperty (вызываются при каждом открытии)
# ===========================================================================

def _bending_model_items(self, context):
    if self.solver_type == 'PD':
        return [
            ('ANGULAR', _t("Angular", "Угловой"),
             _t("Dihedral-angle bending constraint",
                "Ограничение изгиба по двугранному углу")),
            ('LINEAR', _t("Linear", "Линейный"),
             _t("Linear bending stiffness",
                "Линейная жёсткость изгиба")),
        ]
    if self.solver_type == 'Mil2':
        return [
            ('LINEAR', _t("Linear", "Линейный"),
             _t("Linear bending stiffness",
                "Линейная жёсткость изгиба")),
            ('ANGULAR', _t("Angular", "Угловой"),
             _t("Standard Mil2 dihedral-angle bending constraint",
                "Стандартное ограничение изгиба Mil2 по двугранному углу")),
        ]
    return [
        ('LINEAR',  _t("Linear",  "Линейный"),  _t("Linear bending stiffness", "Линейная жёсткость изгиба")),
        ('ANGULAR', _t("Angular", "Угловой"),    _t("Angular bending stiffness (more realistic)", "Угловая жёсткость изгиба (реалистичнее)")),
    ]


def _solver_type_items(self, context):
    """The shipped solver selection.

    PD is the only entry because PD is the only solver the product build ships
    a backend for.  ``GPUCLOTH_PRODUCT_BUILD`` removes every Mil2 site in
    main.cpp (:4587-4598, :9926-9932, :10271-10289, :11135-11138 plus the
    host-state sites), so the Product DLL exports no
    ``compute_eigenvectors_lanczos`` and no ``Mil2_cloth_solve_config_reported``
    - and the DLL's own feature manifest says as much: ``backend_mil2``
    (main.cpp:390-392) reports status ``GPUCLOTH_FEATURE_MISSING``,
    ``supported_solver_mask``/``proven_solver_mask`` both
    ``GPUCLOTH_SOLVER_NONE``, detail "Mil2/Accuracy is available only in the
    Internal build."  The Internal build declares the same feature PROVEN over
    ``GPUCLOTH_SOLVER_MIL2`` (:394-396).

    Mil2 therefore used to be offered and then refused: selecting it reached
    ``GPUCLOTH_V3_BACKEND_ACCURACY`` and ``GPUCloth_v3_cloth_create`` answered
    ABI 4 (``GPUCLOTH_ABI_UNSUPPORTED``), so a prepare failed with
    ``status="v3 cloth create rejected with 4"``.  An option that cannot
    succeed is worse than an absent one, so the entry is gone rather than
    gated: ``_validate_product_abi`` does read ``backend_mil2``'s status
    (operators.py:3030-3060), but only to *check* it - the values are dropped
    and no accessor publishes them, and this callback runs per panel redraw and
    during ``.blend`` load, when no DLL need be loaded at all.  Gating here
    would mean inventing a capability store, which is not this change.

    A stored project whose ``solver_type`` is already ``'Mil2'`` keeps that
    value - nothing remaps it silently - and ``_reject_unsupported_v3_owners``
    refuses the prepare in the panel with text naming Mil2.
    """
    return [
        ('PD',    "PD",     _t("Projective Dynamics with Chebyshev-Jacobi acceleration",
                               "Projective Dynamics с Chebyshev-Jacobi ускорением")),
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


def _fabric_preset_items(self, context):
    """FABRIC membrane presets, named after the fabric they were read from.

    Ten entries.  Two come from the Marvelous Designer archive, eight from the
    SideFX Content Library Vellum fabric samples.  Every field these presets do
    not name is stated in the ``FABRIC_PRESETS`` header as not recorded;
    nothing is interpolated, and every preset fills the same single field
    ``fabric_density`` because that is the only quantity whose units the two
    sources state unambiguously (see the header).
    """
    return [
        ('CUSTOM', _t("Custom", "Свой"),
         _t("Manual parameter tuning",
            "Ручная настройка параметров")),
        ('MD_DEFAULT', _t("MD Default (300 g/m²)", "MD Default (300 г/м²)"),
         _t("Marvelous Designer horizontal control material, `Default` "
            "physical preset: 300 g/m²",
            "Горизонтальный контрольный материал Marvelous Designer, "
            "физический пресет `Default`: 300 г/м²")),
        ('MD_RIB_2X2_468GSM',
         _t("MD Rib 2X2 (468 g/m²)", "MD Rib 2X2 (468 г/м²)"),
         _t("Marvelous Designer `Rib_2X2_468gsm`: 468.182 g/m²",
            "Marvelous Designer `Rib_2X2_468gsm`: 468.182 г/м²")),
        ('VELLUM_SILK', _t("Vellum Silk (40 g/m²)", "Vellum шёлк (40 г/м²)"),
         _t("Houdini Vellum fabric sample `Silk`: 40 g/m²",
            "Образец ткани Houdini Vellum `Silk`: 40 г/м²")),
        ('VELLUM_RAINCOAT',
         _t("Vellum Raincoat (250 g/m²)", "Vellum плащёвка (250 г/м²)"),
         _t("Houdini Vellum fabric sample `Raincoat`: 250 g/m²",
            "Образец ткани Houdini Vellum `Raincoat`: 250 г/м²")),
        ('VELLUM_JERSEY',
         _t("Vellum Jersey (40 g/m²)", "Vellum трикотаж (40 г/м²)"),
         _t("Houdini Vellum fabric sample `Jersey`: 40 g/m²",
            "Образец ткани Houdini Vellum `Jersey`: 40 г/м²")),
        ('VELLUM_LACE',
         _t("Vellum Lace (40 g/m²)", "Vellum кружево (40 г/м²)"),
         _t("Houdini Vellum fabric sample `tulle with embroidery`: 40 g/m²",
            "Образец ткани Houdini Vellum `tulle with embroidery`: 40 г/м²")),
        ('VELLUM_VELVET',
         _t("Vellum Velvet (20 g/m²)", "Vellum вельвет (20 г/м²)"),
         _t("Houdini Vellum fabric sample `Velvet`: 20 g/m²",
            "Образец ткани Houdini Vellum `Velvet`: 20 г/м²")),
        ('VELLUM_WOOL',
         _t("Vellum Wool (40 g/m²)", "Vellum шерсть (40 г/м²)"),
         _t("Houdini Vellum fabric sample `Wool`: 40 g/m²",
            "Образец ткани Houdini Vellum `Wool`: 40 г/м²")),
        ('VELLUM_LEATHER',
         _t("Vellum Leather (400 g/m²)", "Vellum кожа (400 г/м²)"),
         _t("Houdini Vellum fabric sample `Leather`: 400 g/m²",
            "Образец ткани Houdini Vellum `Leather`: 400 г/м²")),
        ('VELLUM_DENIM',
         _t("Vellum Denim (400 g/m²)", "Vellum деним (400 г/м²)"),
         _t("Houdini Vellum fabric sample `Jeans (denim)`: 400 g/m²",
            "Образец ткани Houdini Vellum `Jeans (denim)`: 400 г/м²")),
        # Luible 2008, Annex G: density, tensile and shear in N/m.
        ('LUI_COTTON_SHIRTING',
         _t("Cotton shirting (120, full)",
            "Хлопок, сорочечная (120, полный)"),
         _t("Luible 2008 Annex G sample 02: combined twill, 120 g/m², "
            "tensile 2000/3000 N/m, shear 20 N/m",
            "Luible 2008, Annex G, образец 02: combined twill, 120 г/м², "
            "растяжение 2000/3000 Н/м, сдвиг 20 Н/м")),
        ('LUI_DENIM',
         _t("Denim (380, full)", "Деним (380, полный)"),
         _t("Luible 2008 Annex G sample 01: 100% CO twill, 380 g/m², "
            "tensile 1800/3000 N/m, shear 102 N/m",
            "Luible 2008, Annex G, образец 01: 100% CO twill, 380 г/м², "
            "растяжение 1800/3000 Н/м, сдвиг 102 Н/м")),
        ('LUI_STRETCH_DENIM',
         _t("Stretch denim (275, full)", "Стрейч-деним (275, полный)"),
         _t("Luible 2008 Annex G sample 12: 62% PES, 35% CO, 3% EL, "
            "275 g/m², tensile 4000/400 N/m, shear 71 N/m",
            "Luible 2008, Annex G, образец 12: 62% PES, 35% CO, 3% EL, "
            "275 г/м², растяжение 4000/400 Н/м, сдвиг 71 Н/м")),
        ('LUI_SILK_MULBERRY',
         _t("Silk mulberry (15, full)", "Шёлк тутовый (15, полный)"),
         _t("Luible 2008 Annex G sample 07: 100% SE plain weave, 15 g/m², "
            "tensile 6000/4500 N/m, shear 10 N/m",
            "Luible 2008, Annex G, образец 07: 100% SE, полотняное, 15 г/м², "
            "растяжение 6000/4500 Н/м, сдвиг 10 Н/м")),
        ('LUI_SILK_BOURETTE',
         _t("Silk bourette (150, full)", "Шёлк бурет (150, полный)"),
         _t("Luible 2008 Annex G sample 08: 100% SE plain weave, 150 g/m², "
            "tensile 1500/1500 N/m, shear 38 N/m",
            "Luible 2008, Annex G, образец 08: 100% SE, полотняное, 150 г/м², "
            "растяжение 1500/1500 Н/м, сдвиг 38 Н/м")),
        ('LUI_SILK_TUSSAH',
         _t("Silk tussah (80, full)", "Шёлк туссор (80, полный)"),
         _t("Luible 2008 Annex G sample 09: 100% SE plain weave, 80 g/m², "
            "tensile 4500/10000 N/m, shear 80 N/m",
            "Luible 2008, Annex G, образец 09: 100% SE, полотняное, 80 г/м², "
            "растяжение 4500/10000 Н/м, сдвиг 80 Н/м")),
        ('LUI_WOOL_GABARDINE',
         _t("Wool gabardine (175, full)", "Шерсть габардин (175, полный)"),
         _t("Luible 2008 Annex G sample 05: 100% WO twill, 175 g/m², "
            "tensile 3000/2500 N/m, shear 32 N/m",
            "Luible 2008, Annex G, образец 05: 100% WO, саржа, 175 г/м², "
            "растяжение 3000/2500 Н/м, сдвиг 32 Н/м")),
        ('LUI_WOOL_SUITING',
         _t("Wool suiting (232, full)", "Шерсть костюмная (232, полный)"),
         _t("Luible 2008 Annex G sample 34: 100% WO herringbone, 232 g/m², "
            "tensile 2000/400 N/m, shear 32.9 N/m",
            "Luible 2008, Annex G, образец 34: 100% WO, «ёлочка», 232 г/м², "
            "растяжение 2000/400 Н/м, сдвиг 32.9 Н/м")),
        ('LUI_JERSEY_SINGLE',
         _t("Jersey single knit (172, full)",
            "Трикотаж кулирный (172, полный)"),
         _t("Luible 2008 Annex G sample 21: 98% CLY, 2% EL single jersey, "
            "172 g/m², tensile 120/50 N/m, shear 23 N/m",
            "Luible 2008, Annex G, образец 21: 98% CLY, 2% EL, кулирная гладь, "
            "172 г/м², растяжение 120/50 Н/м, сдвиг 23 Н/м")),
        ('LUI_VELVET',
         _t("Velvet (300, full)", "Вельвет/бархат (300, полный)"),
         _t("Luible 2008 Annex G sample 15: 92% CO, 8% CMD velvet, 300 g/m², "
            "tensile 2000/1000 N/m, shear 58 N/m",
            "Luible 2008, Annex G, образец 15: 92% CO, 8% CMD, 300 г/м², "
            "растяжение 2000/1000 Н/м, сдвиг 58 Н/м")),
        ('LUI_LEATHER',
         _t("Leather garment (815, full)", "Кожа одежная (815, полный)"),
         _t("Luible 2008 Annex G sample 32: 100% leather, 815 g/m², "
            "tensile 3000/2400 N/m, shear 1792.9 N/m",
            "Luible 2008, Annex G, образец 32: 100% кожа, 815 г/м², "
            "растяжение 3000/2400 Н/м, сдвиг 1792.9 Н/м")),
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
        update=_on_simulation_input_change,
    )

    global_gravity: FloatProperty(
        name="Gravity",
        description="Global gravity weight",
        default=1.0,
        min=-200.0,
        max=200.0,
        update=_on_simulation_input_change,
    )

    weight_all: FloatProperty(
        name="All", description="All effectors weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_force: FloatProperty(
        name="Force", description="Force effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_wind: FloatProperty(
        name="Wind", description="Wind effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_vortex: FloatProperty(
        name="Vortex", description="Vortex effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_magnetic: FloatProperty(
        name="Magnetic", description="Magnetic effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_turbulence: FloatProperty(
        name="Turbulence", description="Turbulence effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_drag: FloatProperty(
        name="Drag", description="Drag effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_smoke_flow: FloatProperty(
        name="Fluid Flow", description="Fluid Flow effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_harmonic: FloatProperty(
        name="Harmonic", description="Harmonic effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_charge: FloatProperty(
        name="Charge", description="Charge effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_lennard_jones: FloatProperty(
        name="Lennard-Jones", description="Lennard-Jones effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_texture: FloatProperty(
        name="Texture", description="Texture effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_curve_guide: FloatProperty(
        name="Curve Guide", description="Curve Guide effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )
    weight_boid: FloatProperty(
        name="Boid", description="Boid effector weight", default=1.0, min=-200.0, max=200.0,
        update=_on_simulation_input_change,
    )


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

    auto_prepare: BoolProperty(
        name="Auto Prepare",
        description="Defer GPU preparation after switching from CPU Cloth",
        default=True,
        update=_on_simulation_input_change,
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
        update=_on_simulation_input_change,
    )

    mass_mode: EnumProperty(
        name="Mass Mode",
        description="Choose per-vertex mass or areal fabric density",
        items=(
            ('VERTEX', "Vertex", "Use the mass assigned to each cloth vertex"),
            ('AREAL', "Areal", "Compute mass from fabric density and surface area"),
        ),
        default='VERTEX',
        update=_on_simulation_input_change,
    )

    fabric_density: FloatProperty(
        name="Fabric Density",
        description=(
            "Fabric mass per unit surface area (g/m²). The FABRIC presets "
            "carry measured values only: 300 g/m² and 468.182 g/m² from "
            "Marvelous Designer, eight Houdini Vellum fabric samples (Silk 40, "
            "Raincoat 250, Jersey 40, Lace 40, Velvet 20, Wool 40, Leather "
            "400, Denim 400 g/m²), and thirteen Luible 2008 Annex G samples "
            "(cotton shirting 120, denim 380, stretch denim 275, silk 15/150/"
            "80, wool 175/232, jersey 172, velvet 300, leather 815 g/m²)"),
        default=300.0,
        min=0.001,
        max=100000.0,
        update=_on_simulation_input_change,
    )

    quality_step: IntProperty(
        name="Quality Steps",
        description="Solver substeps per frame",
        default=5,
        min=1,
        max=80,
        update=_on_simulation_input_change,
    )

    speed_multiplier: FloatProperty(
        name="Speed Multiplier",
        description="Simulation time scale",
        default=1.0,
        min=0.0,
        max=10.0,
        update=_on_simulation_input_change,
    )

    bending_model: EnumProperty(
        name="Bending Model",
        description="Cloth bending constraint type",
        items=_bending_model_items,
        update=_on_simulation_input_change,
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

    # ── FABRIC physical membrane preset ────────────────────────────────────
    # FABRIC is the only material model, so this is the preset that feeds the
    # material payload the solver actually receives.  `material_preset` above
    # stays as it was: it is the isotropic stiffness/damping mirror used by the
    # TestScene fixtures and the scenario gates, and it is not part of the
    # FABRIC membrane.
    fabric_preset: EnumProperty(
        name="Fabric Preset",
        description=(
            "Physical membrane preset read from a Marvelous Designer material "
            "record; fields the MD analysis did not record stay untouched"),
        items=_fabric_preset_items,
        update=_apply_fabric_preset,
    )

    # ── Material parameters (stiffness) ────────────────────────────────────
    tension: FloatProperty(
        name="Tension",
        description="Stretch stiffness",
        default=15.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    compression: FloatProperty(
        name="Compression",
        description="Compression stiffness",
        default=15.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    shear: FloatProperty(
        name="Shear",
        description="Shear stiffness",
        default=5.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    bending_stiffness: FloatProperty(
        name="Bending",
        description="Bending stiffness",
        default=0.5,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    # ── Material parameters (damping) ─────────────────────────────────────
    tension_damp: FloatProperty(
        name="Tension Damping",
        description="Tension oscillation damping",
        default=5.0,
        min=0.0,
        max=50.0,
        update=_on_simulation_input_change,
    )

    compression_damp: FloatProperty(
        name="Compression Damping",
        description="Compression oscillation damping",
        default=5.0,
        min=0.0,
        max=50.0,
        update=_on_simulation_input_change,
    )

    shear_damp: FloatProperty(
        name="Shear Damping",
        description="Shear oscillation damping",
        default=5.0,
        min=0.0,
        max=50.0,
        update=_on_simulation_input_change,
    )

    bending_damping: FloatProperty(
        name="Bending Damping",
        description="Bending oscillation damping",
        default=0.5,
        min=0.0,
        max=1000.0,
        update=_on_simulation_input_change,
    )

    # ── Self-collision (OGC) ──────────────────────────────────────────────
    use_self_collision: BoolProperty(
        name="Self-Collision",
        description="Enable Offset Geometric Contact (OGC, Chen et al. 2025) penetration-free self-collision",
        default=False,
        update=_on_simulation_input_change,
    )

    ogc_radius: FloatProperty(
        name="Contact Radius",
        description="OGC contact zone radius (mm). Larger = layers stay further apart. Auto: r = 16.6 * avg_edge^2",
        default=150.0,
        min=1.0,
        max=500.0,
        subtype='NONE',
        unit='NONE',
        update=_on_simulation_input_change,
    )

    ogc_kc: FloatProperty(
        name="Contact Stiffness",
        description="OGC contact stiffness (kc, PD mode). Lower = softer & more stable. Auto: kc = 1.15 / radius_m",
        default=7.0,
        min=1.0,
        max=100000.0,
        update=_on_simulation_input_change,
    )

    ogc_friction: FloatProperty(
        name="Layer Friction",
        description="Friction between cloth layers (OGC). 0 = free sliding, 1 = no sliding. Recommended 0.3-0.6",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_on_simulation_input_change,
    )

    ogc_gamma_p: FloatProperty(
        name="Gamma P",
        description="OGC correction fraction per pass (Mil2 mode). Must be strictly < 0.5. 0.45 = default, 0.48 = dense twist",
        default=0.45,
        min=0.05,
        max=0.49,
        update=_on_simulation_input_change,
    )

    show_ogc_bounds: BoolProperty(
        name="Show Contact Bounds",
        description="Draw OGC contact-radius spheres at each cloth vertex in the 3D viewport (two axis-aligned circles per vertex)",
        default=False,
        update=_on_simulation_input_change,
    )

    # ── Constraint network ─────────────────────────────────────────────────
    use_constraint_network: BoolProperty(
        name="Use Constraint Network",
        description="Animate sewing seams, zippers, and buttons with phased initialization",
        default=False,
        update=_on_simulation_input_change,
    )

    cn_phases: IntProperty(
        name="Phases",
        description="Number of activation phases (sewing stages)",
        default=1, min=1, max=10,
        update=_on_simulation_input_change,
    )

    cn_sewing_speed: FloatProperty(
        name="Sewing Speed",
        description="Frames to fully tighten a seam phase",
        default=10.0, min=1.0, max=100.0,
        update=_on_simulation_input_change,
    )

    cn_seam_stiffness: FloatProperty(
        name="Seam Stiffness",
        description=(
            "Mil2 seam-stiffness multiplier in [0.25, 5.0]; "
            "PD uses the fixed value 1.0"),
        default=1.0, min=0.25, max=5.0, soft_max=2.0,
        update=_on_simulation_input_change,
    )

    cn_enable_selfcoll_stitching: BoolProperty(
        name="Self-Collision During Stitching",
        description="Enable self-collision while seams are being tightened",
        default=True,
        update=_on_simulation_input_change,
    )

    # ── Physical Properties: Damping & Clamping ────────────────────────────
    air_viscosity: FloatProperty(
        name="Air Viscosity",
        description="Viscosity of the surrounding medium (air damping factor)",
        default=1.0,
        min=0.0,
        max=100.0,
        update=_on_simulation_input_change,
    )

    max_tension: FloatProperty(
        name="Max Tension",
        description="Maximum tension stiffness clamping value",
        default=500.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    max_compression: FloatProperty(
        name="Max Compression",
        description="Maximum compression stiffness clamping value",
        default=500.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    max_shear: FloatProperty(
        name="Max Shear",
        description="Maximum shear stiffness clamping value",
        default=500.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    max_bend: FloatProperty(
        name="Max Bending",
        description="Maximum bending stiffness clamping value",
        default=100.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    max_sewing: FloatProperty(
        name="Max Sewing Force",
        description="Maximum sewing force clamping value",
        default=500.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    use_sewing_springs: BoolProperty(
        name="Sew Cloth",
        description="Pull loose Blender cloth edges together",
        default=False,
        update=_on_simulation_input_change,
    )

    vel_damping: FloatProperty(
        name="Velocity Damping",
        description="Damp velocity to speed up convergence to rest pose",
        # Shipped default 1.0.  Measured on the Drape-on-Sphere preset over the
        # FABRIC (membrane/SDB) route with self-collision off: at 0.0 the sheet
        # crept at L-inf 0.0273 m/frame, 27x the 0.001 auto-stop tolerance on
        # `maximum_position_delta`, so the sandbox never filled its 8-step window
        # and Settle ended NOT_CONVERGED at 0.0344.  At 1.0 the same scene's
        # residual falls to L-inf ~0.010 m/frame and stops growing, but the
        # 0.001 window still does not fill: 1.0 shrinks the creep, it does not by
        # itself close the auto-stop.  Presets that want the old 0.0 set it
        # explicitly; a saved .blend keeps its stored value.
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_on_simulation_input_change,
    )

    # ── Internal Springs ───────────────────────────────────────────────────
    use_internal_springs: BoolProperty(
        name="Internal Springs",
        description="Create internal springs to resist compression",
        default=False,
        update=_on_simulation_input_change,
    )

    use_internal_springs_normal: BoolProperty(
        name="Check Surface Normals",
        description="Create internal springs only between points with opposite normals",
        default=False,
        update=_on_simulation_input_change,
    )

    internal_spring_max_length: FloatProperty(
        name="Max Spring Length",
        description="Maximum length an internal spring can have during creation",
        default=0.0,
        min=0.0,
        max=1000.0,
        update=_on_simulation_input_change,
    )

    internal_spring_max_diversion: FloatProperty(
        name="Max Normal Diversion",
        description="Maximum angle diversion from vertex normal during internal spring creation (radians)",
        default=0.7853981633974483,  # π/4
        min=0.0,
        max=0.7853981633974483,      # pi/4
        subtype='ANGLE',
        update=_on_simulation_input_change,
    )

    internal_tension: FloatProperty(
        name="Tension",
        description="Tension stiffness for internal springs",
        default=15.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    internal_compression: FloatProperty(
        name="Compression",
        description="Compression stiffness for internal springs",
        default=15.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    max_internal_tension: FloatProperty(
        name="Max Internal Tension",
        description="Maximum tension stiffness clamping for internal springs",
        default=500.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    max_internal_compression: FloatProperty(
        name="Max Internal Compression",
        description="Maximum compression stiffness clamping for internal springs",
        default=500.0,
        min=0.0,
        max=10000.0,
        update=_on_simulation_input_change,
    )

    # ── Anisotropy (warp/weft directional stiffness) ─────────────────────────
    use_anisotropy: BoolProperty(
        name="Anisotropic Stiffness",
        description=(
            "Enable per-direction warp/weft tension, compression, and "
            "bending stiffness"),
        default=False,
        update=_on_simulation_input_change,
    )
    anisotropy_uv_map: StringProperty(
        name="Material Direction UV Map",
        description=(
            "Explicit seam-free UV map used by GPUCloth as warp (U) and "
            "weft (V) material coordinates"),
        default="",
        update=_on_simulation_input_change,
    )
    tension_u: FloatProperty(
        name="Tension U (Warp)",
        description="Stretch stiffness along warp (U) direction. 0 = use isotropic tension",
        default=0.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    tension_v: FloatProperty(
        name="Tension V (Weft)",
        description="Stretch stiffness along weft (V) direction. 0 = use isotropic tension",
        default=0.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    compression_u: FloatProperty(
        name="Compression U (Warp)",
        description="Compression stiffness along warp. 0 = use isotropic compression",
        default=0.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    compression_v: FloatProperty(
        name="Compression V (Weft)",
        description="Compression stiffness along weft. 0 = use isotropic compression",
        default=0.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    bending_u: FloatProperty(
        name="Bending U (Warp)",
        description="Bending stiffness along warp. 0 = use isotropic bending",
        default=0.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    bending_v: FloatProperty(
        name="Bending V (Weft)",
        description="Bending stiffness along weft. 0 = use isotropic bending",
        default=0.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    max_tension_u: FloatProperty(
        name="Max Tension U",
        description="Maximum tension clamping along warp",
        default=500.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    max_tension_v: FloatProperty(
        name="Max Tension V",
        description="Maximum tension clamping along weft",
        default=500.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    max_compression_u: FloatProperty(
        name="Max Compression U",
        description="Maximum compression clamping along warp",
        default=500.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    max_compression_v: FloatProperty(
        name="Max Compression V",
        description="Maximum compression clamping along weft",
        default=500.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    max_bend_u: FloatProperty(
        name="Max Bend U",
        description="Maximum bending clamping along warp",
        default=500.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )
    max_bend_v: FloatProperty(
        name="Max Bend V",
        description="Maximum bending clamping along weft",
        default=500.0, min=0.0, max=10000.0,
        update=_on_simulation_input_change,
    )

    # ── Fabric physical membrane parameters (N/m) ─────────────────────────
    fabric_tensile_u: FloatProperty(
        name="Fabric Tensile U (N/m)", description="Physical warp tensile modulus",
        default=10000.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_tensile_v: FloatProperty(
        name="Fabric Tensile V (N/m)", description="Physical weft tensile modulus",
        default=10000.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_compression_u: FloatProperty(
        name="Fabric Compression U (N/m)", description="Physical warp compression modulus",
        default=10000.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_compression_v: FloatProperty(
        name="Fabric Compression V (N/m)", description="Physical weft compression modulus",
        default=10000.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_shear_c66: FloatProperty(
        name="Fabric Shear C66 (N/m)", description="Physical in-plane shear modulus C66",
        default=500.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_tensile_u_max: FloatProperty(
        name="Fabric Max Tensile U (N/m)", default=10000.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_tensile_v_max: FloatProperty(
        name="Fabric Max Tensile V (N/m)", default=10000.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_compression_u_max: FloatProperty(
        name="Fabric Max Compression U (N/m)", default=10000.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_compression_v_max: FloatProperty(
        name="Fabric Max Compression V (N/m)", default=10000.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_shear_c66_max: FloatProperty(
        name="Fabric Max Shear C66 (N/m)", default=500.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_tensile_damping: FloatProperty(
        name="Fabric Tensile Damping (N·s/m)", default=5.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_compression_damping: FloatProperty(
        name="Fabric Compression Damping (N·s/m)", default=5.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )
    fabric_shear_damping: FloatProperty(
        name="Fabric Shear Damping (N·s/m)", default=1.0, min=0.0, max=1.0e20,
        update=_on_simulation_input_change,
    )

    # ── Advanced Solver Config ──────────────────────────────────────────────
    solver_iterations: IntProperty(
        name="Iterations",
        description="PD/Mil2 iterations per substep",
        default=10, min=1, max=200,
        update=_on_simulation_input_change,
    )

    solver_omega: FloatProperty(
        name="Chebyshev ω",
        description="Chebyshev acceleration omega (1.0 = off, 1.5 = default)",
        default=1.5, min=0.5, max=2.0,
        update=_on_simulation_input_change,
    )

    use_adaptive: BoolProperty(
        name="Adaptive Convergence",
        description=(
            "Enable adaptive early exit for supported solver modes"),
        default=False,
        update=_on_simulation_input_change,
    )

    solver_max_iterations: IntProperty(
        name="Max Iterations",
        description="Safety cap for adaptive mode",
        default=100, min=10, max=500,
        update=_on_simulation_input_change,
    )

    solver_convergence_tol: FloatProperty(
        name="Convergence Tol",
        description=(
            "PD Fabric free-force L2 residual tolerance (N) for adaptive early "
            "exit"),
        default=0.001, min=0.0001, max=0.1, soft_max=0.01,
        update=_on_simulation_input_change,
    )

    # ── Drape sandbox criterion ────────────────────────────────────────────
    # The drape sandbox compared a per-frame *position* change against
    # `solver_convergence_tol`, which is a *force* tolerance in newtons.  That
    # is a dimensional mismatch, and on the owner's drape the damped residual
    # sits 8-12x above it, so the sandbox could never settle.  The drape now
    # owns its criterion: a position tolerance in metres, and a budget in
    # simulated seconds rather than a bare step cap.  The solver's own
    # criterion and `solver_convergence_tol` are untouched.
    drape_position_tolerance: FloatProperty(
        name="Drape Position Tolerance (m/frame)",
        description=(
            "Largest per-frame cloth movement that still counts as at rest for "
            "the drape sandbox, in metres of position change per frame. The "
            "drape counts a drape settled after 8 consecutive frames at or "
            "below this. Default 1 mm/frame is measured on the Drape On Sphere "
            "preset: its residual never falls below 2.98 mm/frame in 40 s of "
            "simulated draping, so the default cannot declare a still-moving "
            "cloth settled"),
        default=0.001, min=0.00001, max=0.05, soft_max=0.01, precision=6,
        unit='LENGTH',
    )

    drape_budget_s: FloatProperty(
        name="Drape Budget (simulated seconds)",
        description=(
            "Simulated seconds of draping one Settle may spend before it "
            "reports the drape as not settled. One native drape step advances "
            "one frame, so the step ceiling is this budget times the scene "
            "frame rate. The sandbox is kept, so pressing Settle again "
            "continues the drape, and ESC stops a run at any time. Default "
            "12 s is 1.2x the native 240-step cap at 24 fps and 3x that cap at "
            "60 fps, and costs about 40 s of wall time on the 16k-vertex Drape "
            "On Sphere preset"),
        default=12.0, min=0.5, max=240.0, soft_max=60.0,
    )

    solver_krylov_iterations: IntProperty(
        name="Krylov Iterations",
        description=(
            "Per-call Krylov (cooperative PCG) update ceiling for the PD "
            "global solve on the physical membrane route. The linear solve "
            "stops after this many updates even when the residual tolerance "
            "is unmet; the nonlinear generation schedule is unchanged. 240 is "
            "the shipped ceiling - the smallest measured ceiling at which the "
            "convergence criterion, and not this ceiling, ends the majority of "
            "contact solves on the drape acceptance scene; 1000 is the "
            "reference's safety cap and this parameter's maximum"),
        default=240, min=1, max=1000,
        update=_on_simulation_input_change,
    )

    vgroup_intern: StringProperty(
        name="Vertex Group",
        description="Vertex group for scaling internal spring stiffness",
        default="",
        update=_on_simulation_input_change,
    )

    # ── Pressure ───────────────────────────────────────────────────────────
    use_pressure: BoolProperty(
        name="Pressure",
        description="Enable internal pressure simulation",
        default=False,
        update=_on_simulation_input_change,
    )

    uniform_pressure_force: FloatProperty(
        name="Pressure",
        description="Uniform pressure force constantly applied to the mesh (can be negative)",
        default=0.0,
        min=-100.0,
        max=100.0,
        update=_on_simulation_input_change,
    )

    target_volume: FloatProperty(
        name="Target Volume",
        description="Equilibrium volume the mesh wants to expand to (0 = use rest volume)",
        default=0.0,
        min=0.0,
        max=1000.0,
        update=_on_simulation_input_change,
    )

    use_pressure_volume: BoolProperty(
        name="Use Custom Volume",
        description="Use Target Volume instead of the initial mesh volume",
        default=False,
        update=_on_simulation_input_change,
    )

    pressure_factor: FloatProperty(
        name="Factor",
        description="Scales volume feedback: pressure = uniform + (target volume / current volume - 1) * factor",
        default=1.0,
        min=0.0,
        max=100.0,
        update=_on_simulation_input_change,
    )

    fluid_density: FloatProperty(
        name="Fluid Density",
        description="Density of the fluid inside/outside for hydrostatic pressure gradient",
        default=0.0,
        min=-10.0,
        max=10.0,
        update=_on_simulation_input_change,
    )

    vgroup_pressure: StringProperty(
        name="Pressure Vertex Group",
        description="Vertex group for scaling pressure",
        default="",
        update=_on_simulation_input_change,
    )

    # ── Shape / Pinning ────────────────────────────────────────────────────
    vgroup_mass: StringProperty(
        name="Pin Group",
        description="Vertex group for pinning vertices (zero weight = free, full weight = pinned)",
        default="",
        update=_on_simulation_input_change,
    )

    goalspring: FloatProperty(
        name="Pin Stiffness",
        description="Stiffness of goal springs (pinning force)",
        default=1.0,
        min=0.0,
        max=100.0,
        update=_on_simulation_input_change,
    )

    goalfrict: FloatProperty(
        name="Pin Friction",
        description="Friction/damping applied to pinned vertices",
        default=0.0,
        min=0.0,
        max=1000.0,
        update=_on_simulation_input_change,
    )

    mingoal: FloatProperty(
        name="Min Goal Factor",
        description="Minimum Blender goal factor",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_on_simulation_input_change,
    )

    maxgoal: FloatProperty(
        name="Max Goal Factor",
        description="Maximum pin goal factor",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_on_simulation_input_change,
    )

    defgoal: FloatProperty(
        name="Default Goal Factor",
        description="Goal factor for vertices absent from the pin group",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_on_simulation_input_change,
    )

    shrink_min: FloatProperty(
        name="Shrink Min",
        description="Min shrinkage factor: 0=none, 1=shrink to nothing, -1=double edge length",
        default=0.0,
        min=-1.0,
        max=1.0,
        update=_on_simulation_input_change,
    )

    shrink_max: FloatProperty(
        name="Shrink Max",
        description="Max shrinkage factor: 0=none, 1=shrink to nothing, -1=double edge length",
        default=0.0,
        min=-1.0,
        max=1.0,
        update=_on_simulation_input_change,
    )

    shapekey_rest: StringProperty(
        name="Rest Shape Key",
        description="Shape key used as rest configuration for the cloth simulation",
        default="",
        update=_on_simulation_input_change,
    )

    use_dynamic_mesh: BoolProperty(
        name="Dynamic Mesh",
        description="Allow the base mesh to deform in real-time during simulation",
        default=False,
        update=_on_simulation_input_change,
    )

    # ── Object Collision ────────────────────────────────────────────────────
    use_object_collision: BoolProperty(
        name="Object Collision",
        description="Enable collision against Blender collision objects",
        default=True,
        update=_on_simulation_input_change,
    )

    collision_friction: FloatProperty(
        name="Friction",
        description="Object collision friction",
        default=5.0,
        min=0.0,
        max=80.0,
        update=_on_live_collision_change,
    )

    collision_damping: FloatProperty(
        name="Damping",
        description="Object collision damping",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_on_live_collision_change,
    )

    collision_quality: IntProperty(
        name="Collision Quality",
        description="Collision iterations per simulation step",
        default=2,
        min=1,
        max=32767,
        update=_on_live_collision_change,
    )

    epsilon: FloatProperty(
        name="Distance",
        description="Minimum distance for object collisions (m)",
        default=0.015,
        min=0.001,
        max=1.0,
        subtype='DISTANCE',
        update=_on_simulation_input_change,
    )

    selfepsilon: FloatProperty(
        name="Self Distance",
        description="Minimum distance for self-collisions (m)",
        default=0.015,
        min=0.001,
        max=0.1,
        subtype='DISTANCE',
        update=_on_simulation_input_change,
    )

    self_collision_friction: FloatProperty(
        name="Self Friction",
        description="Blender self-collision friction before native scaling",
        default=5.0,
        min=0.0,
        max=80.0,
        update=_on_live_collision_change,
    )

    clamp: FloatProperty(
        name="Object Impulse Clamp",
        description="Maximum impulse for object collision correction",
        default=0.0,
        min=0.0,
        max=100.0,
        update=_on_live_collision_change,
    )

    self_clamp: FloatProperty(
        name="Self Impulse Clamp",
        description="Maximum impulse for self-collision correction",
        default=0.0,
        min=0.0,
        max=100.0,
        update=_on_live_collision_change,
    )

    collision_collection: PointerProperty(
        name="Collision Collection",
        description="Restrict object collisions to objects in this collection",
        type=bpy.types.Collection,
        update=_on_simulation_input_change,
    )

    vgroup_objcol: StringProperty(
        name="Exclude Objects VGroup",
        description="Vertex group excluding vertices from object collisions (0=excluded, 1=fully collide)",
        default="",
        update=_on_simulation_input_change,
    )

    vgroup_selfcol: StringProperty(
        name="Exclude Self VGroup",
        description=(
            "Vertices with any positive group weight are excluded from "
            "self-collisions"),
        default="",
        update=_on_simulation_input_change,
    )

    # ── Property Weights (stiffness scaling groups) ──────────────────────────
    vgroup_struct: StringProperty(
        name="Structural Group",
        description="Vertex group for scaling structural stiffness",
        default="",
        update=_on_simulation_input_change,
    )

    vgroup_bend: StringProperty(
        name="Bending Group",
        description="Vertex group for scaling bending stiffness",
        default="",
        update=_on_simulation_input_change,
    )

    vgroup_shear: StringProperty(
        name="Shear Group",
        description="Vertex group for scaling shear stiffness",
        default="",
        update=_on_simulation_input_change,
    )

    vgroup_shrink: StringProperty(
        name="Shrinking Group",
        description="Vertex group for shrinking cloth",
        default="",
        update=_on_simulation_input_change,
    )

    # ── Field Weights ──────────────────────────────────────────────────────
    effector_weights: PointerProperty(
        name="Field Weights",
        description="Per-field-type effector weights for cloth simulation",
        type=GPUClothEffectorWeights,
        update=_on_simulation_input_change,
    )

    eff_force_scale: FloatProperty(
        name="Effector Force",
        description="Scaling of effector forces",
        default=1000.0,
        min=0.0,
        max=100000.0,
        update=_on_simulation_input_change,
    )

    eff_wind_scale: FloatProperty(
        name="Effector Wind",
        description="Scaling of effector wind forces",
        default=250.0,
        min=0.0,
        max=100000.0,
        update=_on_simulation_input_change,
    )

    # ── Proxy-res simulation ──────────────────────────────────────────────
    use_proxy: BoolProperty(
        name="Proxy Simulation",
        description="Simulate coarse proxy mesh, then upsample to hi-res render mesh on GPU",
        default=False,
        update=_on_simulation_input_change,
    )

    proxy_object: PointerProperty(
        name="Proxy Mesh",
        description="Object with coarse mesh for simulation (fewer vertices = faster)",
        type=bpy.types.Object,
        update=_on_simulation_input_change,
    )

    proxy_nx: IntProperty(
        name="Proxy NX",
        description="Proxy grid cells along X axis",
        default=8,
        min=2,
        max=256,
        update=_on_simulation_input_change,
    )

    proxy_ny: IntProperty(
        name="Proxy NY",
        description="Proxy grid cells along Y axis",
        default=8,
        min=2,
        max=256,
        update=_on_simulation_input_change,
    )

    hi_nx: IntProperty(
        name="Hi-res NX",
        description="Render grid cells along X axis",
        default=32,
        min=2,
        max=1024,
        update=_on_simulation_input_change,
    )

    hi_ny: IntProperty(
        name="Hi-res NY",
        description="Render grid cells along Y axis",
        default=32,
        min=2,
        max=1024,
        update=_on_simulation_input_change,
    )

    num_sheets: IntProperty(
        name="Cloth Layers",
        description="Number of layers in multi-layer system",
        default=1,
        min=1,
        max=8,
        update=_on_simulation_input_change,
    )

    proxy_scene_type: IntProperty(
        name="Scene Type",
        description="Typed proxy mode: 0/1=local-frame interpolation, 3=direct barycentric; 2 is unsupported",
        default=0,
        min=0,
        max=3,
        update=_on_simulation_input_change,
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
        update=_on_simulation_input_change,
    )

    gravity_y: FloatProperty(
        name="Gravity Y",
        description="Gravitational acceleration along Y (m/s^2)",
        default=0.0,
        update=_on_simulation_input_change,
    )

    gravity_z: FloatProperty(
        name="Gravity Z",
        description="Gravitational acceleration along Z (m/s^2)",
        default=-9.81,
        update=_on_simulation_input_change,
    )

    # ── Simulation cache ─────────────────────────────────────────────────────

    cache_dir: StringProperty(
        name="Cache Directory",
        description="Directory for per-frame simulation cache files",
        default="//gpucloth_cache/",
        subtype='DIR_PATH',
        options=vcu.get_dir_path_property_options(),
    )

    cache_index: IntProperty(
        name="Cache Index",
        description="Blender PointCache identity index",
        default=0,
        min=0,
    )

    cache_name: StringProperty(
        name="Cache Name",
        description="Blender PointCache identity name",
        default="GPUCloth",
    )

    use_disk_cache: BoolProperty(
        name="Disk Cache",
        description="Persist simulation frames instead of session memory only",
        default=True,
    )

    use_external_cache: BoolProperty(
        name="External Cache",
        description="Read a Blender-selected external cache without mutation",
        default=False,
    )

    external_cache_dir: StringProperty(
        name="External Cache Directory",
        description="Resolved external PointCache directory",
        default="",
        subtype='DIR_PATH',
        options=vcu.get_dir_path_property_options(),
    )

    use_library_path: BoolProperty(
        name="Use Library Path",
        description="Resolve external cache relative to a linked blend file",
        default=False,
    )

    cache_compression: EnumProperty(
        name="Cache Compression",
        description="Lossless native frame compression",
        items=[
            ('NO', "None", "Store uncompressed float32 frames"),
            ('LIGHT', "Light", "Use light lossless compression"),
            ('HEAVY', "Heavy", "Use heavy lossless compression"),
        ],
        default='NO',
    )

    is_baked: BoolProperty(
        name="Baked",
        description="True if simulation cache is fully written to disk",
        default=False,
    )

    is_baking: BoolProperty(
        name="Baking",
        description="Native cache transaction is active",
        default=False,
    )

    is_outdated: BoolProperty(
        name="Outdated",
        description="Cached source generation differs from current inputs",
        default=False,
    )

    is_frame_skip: BoolProperty(
        name="Frame Missing",
        description="Configured range contains a missing or invalid frame",
        default=False,
    )

    cached_frame_count: IntProperty(
        name="Cached Frames",
        description="Number of structurally valid frames in active cache",
        default=0,
        min=0,
    )

    cache_info: StringProperty(
        name="Cache Status",
        description="Native cache status detail",
        default="",
    )

    memory_preflight_status: StringProperty(
        name="GPU Memory Preflight",
        description="Visible lower-bound GPU memory admission diagnostic",
        default="",
        options={'HIDDEN'},
    )

    prepare_state: StringProperty(
        name="Prepare State",
        description="Transient asynchronous preparation state",
        default="IDLE",
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    prepare_status: StringProperty(
        name="Prepare Status",
        description="Current asynchronous preparation phase",
        default="",
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    prepare_progress: IntProperty(
        name="Prepare Progress",
        description="Current asynchronous preparation percentage",
        default=0,
        min=0,
        max=100,
        subtype='PERCENTAGE',
        options={'HIDDEN', 'SKIP_SAVE'},
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
