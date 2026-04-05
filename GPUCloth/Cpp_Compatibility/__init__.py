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

# Поддержка hot-reload в режиме разработки (паттерн FLIP Fluids)
if "bpy" in locals():
    import importlib

    reloadable_modules = [
        'properties',
        'operators',
        'ui',
    ]
    for module_name in reloadable_modules:
        if module_name in locals():
            importlib.reload(locals()[module_name])

from . import properties
from . import operators
from . import ui


def register():
    # Порядок важен: PropertyGroup-ы регистрируются ДО операторов,
    # которые читают поля типа OBJ.GPUCloth.*
    properties.register()
    operators.register()
    ui.register()


def unregister():
    # Порядок обратный: UI первым (ссылается на операторы),
    # затем операторы (ссылаются на PropertyGroup),
    # затем PropertyGroup-ы.
    ui.unregister()
    operators.unregister()
    properties.unregister()
