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

bl_info = {
    "name": "GPUCloth",
    "author": "Bubnov Aleksey (PorisulkiP)",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "Properties > Physics > GPU Cloth",
    "description": "Cloth simulation on GPU (CUDA)",
    "doc_url": "https://github.com/PorisulkiP/GPUCloth",
    "category": "Animation",
}

# Поддержка hot-reload в режиме разработки
if "bpy" in locals():
    import importlib
    if "Cpp_Compatibility" in locals():
        importlib.reload(Cpp_Compatibility)

import bpy
from . import Cpp_Compatibility


def register():
    Cpp_Compatibility.register()


def unregister():
    if Cpp_Compatibility.unregister() is False:
        raise RuntimeError(
            "GPUCloth unregister blocked by retained native owners")


if __name__ == "__main__":
    register()
