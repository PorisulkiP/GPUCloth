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
    Структура установки: <addon>/utils/version_compatibility_utils.py
    Ищет __init__.py с bl_info, поднимаясь от текущего файла вверх.
    """
    current = os.path.dirname(os.path.realpath(__file__))
    searched = []
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            break
        init_path = os.path.join(current, "__init__.py")
        searched.append(init_path)
        if os.path.isfile(init_path):
            try:
                with open(init_path, "r", encoding="utf-8") as f:
                    if "bl_info" in f.read():
                        print(f"[GPUCloth] get_addon_directory() -> {current}")
                        return current
            except Exception:
                pass
        current = parent
    print(f"[GPUCloth] WARNING: bl_info not found. Searched: {searched}")
    fallback = os.path.dirname(os.path.realpath(__file__))
    print(f"[GPUCloth] get_addon_directory() fallback -> {fallback}")
    return fallback


def get_lib_directory():
    """Возвращает директорию с нативными библиотеками (DLL / .so / .dylib)."""
    return os.path.join(get_addon_directory(), "lib")


def get_dll_path(dll_name="GPUCloth.dll"):
    """
    Ищет библиотеку только в каталоге lib установленного аддона.
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

    addon_dir = get_addon_directory()
    lib_dir   = get_lib_directory()

    candidates = [os.path.join(lib_dir, lib_filename)]

    for candidate in candidates:
        resolved = os.path.normpath(candidate)
        print(f"[GPUCloth] get_dll_path() checking: {resolved}")
        if os.path.isfile(resolved):
            print(f"[GPUCloth] get_dll_path() FOUND: {resolved}")
            return resolved

    print(f"[GPUCloth] get_dll_path() NOT FOUND. addon_dir={addon_dir}, lib_dir={lib_dir}")
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
