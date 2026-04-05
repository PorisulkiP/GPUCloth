# Blender GPUCloth Add-on
# Copyright (C) 2023 Bubnov Aleksey
#
# Based on patterns from Blender FLIP Fluids addon by Ryan L. Guy
# https://github.com/rlguy/Blender-FLIP-Fluids
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
import os


# ===========================================================================
#  Версионные проверки
#  Паттерн взят из FLIP Fluids version_compatibility_utils.py
# ===========================================================================

def is_blender_28():
    return bpy.app.version >= (2, 80, 0)

def is_blender_293():
    return bpy.app.version >= (2, 93, 0)

def is_blender_30():
    return bpy.app.version >= (3, 0, 0)

def is_blender_32():
    return bpy.app.version >= (3, 2, 0)

def is_blender_40():
    return bpy.app.version >= (4, 0, 0)

def is_blender_41():
    return bpy.app.version >= (4, 1, 0)

def is_blender_42():
    return bpy.app.version >= (4, 2, 0)

def is_blender_43():
    return bpy.app.version >= (4, 3, 0)

def is_blender_44():
    return bpy.app.version >= (4, 4, 0)

def is_blender_45():
    return bpy.app.version >= (4, 5, 0)

def is_blender_50():
    return bpy.app.version >= (5, 0, 0)


# ===========================================================================
#  Пути
# ===========================================================================

def get_addon_directory():
    """
    Возвращает корневую директорию аддона GPUCloth.
    Используется вместо несуществующего get_script_paths_pref() из FLIP Fluids.
    Структура: <addon_root>/src/python/utils/version_compatibility_utils.py
    """
    utils_dir  = os.path.dirname(os.path.realpath(__file__))   # .../src/python/utils
    python_dir = os.path.dirname(utils_dir)                     # .../src/python
    src_dir    = os.path.dirname(python_dir)                    # .../src
    addon_dir  = os.path.dirname(src_dir)                       # <addon_root>
    return addon_dir


def get_lib_directory():
    """Возвращает директорию с нативными библиотеками (DLL / .so / .dylib)."""
    return os.path.join(get_addon_directory(), "lib")


def get_dll_path(dll_name="GPUCloth.dll"):
    """
    Ищет DLL сначала рядом с аддоном, затем в стандартных местах сборки.
    Поддерживает Windows (.dll) / Linux (.so) / macOS (.dylib).
    """
    import platform
    system = platform.system()

    # Платформо-зависимое имя библиотеки
    if system == "Windows":
        lib_filename = dll_name
    elif system == "Darwin":
        lib_filename = dll_name.replace(".dll", ".dylib")
    else:
        lib_filename = dll_name.replace(".dll", ".so")

    # 1. <addon_root>/lib/
    candidate = os.path.join(get_lib_directory(), lib_filename)
    if os.path.isfile(candidate):
        return candidate

    # 2. Рядом с директорией аддона (удобно при разработке)
    candidate = os.path.join(get_addon_directory(), lib_filename)
    if os.path.isfile(candidate):
        return candidate

    return None


# ===========================================================================
#  Совместимость Mesh API
# ===========================================================================

def calc_mesh_loop_triangles(mesh):
    """
    Совместимое получение loop_triangles.
    В Blender 4.1+ требуется явный вызов calc_loop_triangles() перед доступом
    к mesh.loop_triangles, иначе возвращается пустой массив.
    """
    if is_blender_41():
        mesh.calc_loop_triangles()
    return mesh.loop_triangles


def mesh_vertices_foreach_set(mesh, positions_flat):
    """
    Устанавливает позиции вершин меша через foreach_set.
    Это единственный надёжный способ в Blender 4.x
    (прямое присваивание vertices[i].co = ... работает медленно и
    требует mesh.update() для отображения).

    :param mesh: bpy.types.Mesh
    :param positions_flat: плоский список/массив float длиной nVerts * 3 (x,y,z,x,y,z,...)
    """
    mesh.vertices.foreach_set("co", positions_flat)
    mesh.update()


# ===========================================================================
#  Совместимость StringProperty опций
# ===========================================================================

def get_dir_path_property_options():
    """
    Возвращает options= для StringProperty(subtype='DIR_PATH').
    PATH_SUPPORTS_BLEND_RELATIVE появился в Blender 4.5 и позволяет
    сохранять относительные пути (// нотация) вместе с .blend файлом.
    """
    opts = set()
    if is_blender_45():
        opts.add('PATH_SUPPORTS_BLEND_RELATIVE')
    return opts


def get_file_path_property_options():
    """
    Возвращает options= для StringProperty(subtype='FILE_PATH').
    Аналогично get_dir_path_property_options.
    """
    opts = set()
    if is_blender_45():
        opts.add('PATH_SUPPORTS_BLEND_RELATIVE')
    return opts


# ===========================================================================
#  Утилиты для объектов / меша
#  (по паттерну FLIP Fluids version_compatibility_utils.py)
# ===========================================================================

def get_active_object(context=None):
    if context is None:
        context = bpy.context
    return context.view_layer.objects.active


def set_active_object(obj, context=None):
    if context is None:
        context = bpy.context
    context.view_layer.objects.active = obj


def select_get(obj):
    return obj.select_get()


def select_set(obj, value):
    obj.select_set(value)


def object_to_depsgraph_mesh(obj, context=None):
    """
    Возвращает вычисленный (evaluated) меш объекта с учётом модификаторов.
    Нужно для корректного получения triangulated меша под симуляцию.
    """
    if context is None:
        context = bpy.context
    depsgraph = context.evaluated_depsgraph_get()
    obj_eval  = obj.evaluated_get(depsgraph)
    return obj_eval.to_mesh()


def get_blender_version_string():
    """Возвращает строку вида '4.1.0' для отображения в UI."""
    v = bpy.app.version
    return f"{v[0]}.{v[1]}.{v[2]}"


# ===========================================================================
#  Математика / матрицы
#  (по паттерну FLIP Fluids version_compatibility_utils.py)
# ===========================================================================

# ===========================================================================
#  Локализация (i18n)
# ===========================================================================

def _t(en, ru):
    """
    Возвращает локализованную строку на основе языка интерфейса Blender.
    Вызывается в draw-time, поэтому реагирует на смену языка без перезапуска.

    :param en: English text
    :param ru: Russian text
    :return: строка на нужном языке
    """
    try:
        return ru if bpy.app.translations.locale.startswith('ru') else en
    except Exception:
        return en


# ===========================================================================
#  Математика / матрицы
#  (по паттерну FLIP Fluids version_compatibility_utils.py)
# ===========================================================================

def element_multiply(m, v):
    """
    Умножение матрицы на вектор (трансформация позиции).
    В Blender 4.x используется оператор @.
    В Blender < 2.80 использовался оператор * (удалён в 2.80+).

    :param m: Matrix (4x4) — обычно object.matrix_world
    :param v:  Vector (3D или 4D)
    :return:   Vector — трансформированный вектор
    """
    return m @ v
