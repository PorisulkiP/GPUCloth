"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

В данном файле происходит запуск файлов создания сцены и запуска симуляции
"""

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
import physics
  
if __name__ == "__main__":
    # Переход на первый кадр
    bpy.context.scene.frame_current = bpy.context.scene.frame_start

    setup.SetUp()
    backUp = physics.backupSet()
    physics.Physics(backUp)

    # Запуск симуляции
    bpy.ops.screen.animation_play(reverse=False, sync=False)