"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

Задача данного файла – подключение к blender, настройка UI, регистрации аддона.
"""

# импортируем API для работы с blender
import bpy
import sys
import os

# Добавляем папку с проектом в поле зрения blender
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
    sys.path.append(dir + "\\python")
    print(sys.path)

# импорт файлов с функциями
import setup
import physic

bl_info = {
    "name": "GPUCloth",
    "author": "PorisulkiP",
    "version": (0, 0, 1),
    "blender": (2, 93, 0),
    "location": "",
    "warning": "",
    "description": "Cloth simulation on GPU",
    "doc_url": "",
    "category": "System",
}

# ------------------------------------------------------------------------
#    UI
# ------------------------------------------------------------------------



# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_class(setup.SetUp)
    bpy.utils.register_class(physics.Physics)

def unregister():
    bpy.utils.unregister_class(setup.SetUp)
    bpy.utils.unregister_class(physics.Physics)

if __name__ == "__main__":
    register()