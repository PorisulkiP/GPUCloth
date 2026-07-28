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
        'cloth_settings_bridge',
        'proxy_binding',
        'properties',
        'operators',
        'ui',
    ]
    for module_name in reloadable_modules:
        if module_name in locals():
            importlib.reload(locals()[module_name])

from . import cloth_settings_bridge
from . import proxy_binding
from . import properties
from . import operators
from . import ui


_REGISTER_TEARDOWN_FAILURE = (
    "GPUCloth package register blocked by retained native owners")
_UNREGISTER_TEARDOWN_FAILURE = (
    "GPUCloth package unregister blocked by retained native owners")


def get_solver_diagnostics(cloth=None):
    return operators.get_solver_diagnostics(cloth)


def _rollback_registered_modules(modules):
    for module in reversed(modules):
        try:
            module.unregister()
        except Exception as exc:
            print(
                f"GPUCloth registration rollback failed for "
                f"{module.__name__}: {exc}")


def register():
    if not operators.ensure_native_teardown():
        raise RuntimeError(_REGISTER_TEARDOWN_FAILURE)

    registered = []
    # Порядок важен: PropertyGroup-ы регистрируются ДО операторов,
    # которые читают поля типа OBJ.GPUCloth.*
    try:
        properties.register()
        registered.append(properties)
        cloth_settings_bridge.restore_all_cpu_owners()
        operators.register()
        registered.append(operators)
        ui.register()
        registered.append(ui)
    except Exception:
        _rollback_registered_modules(registered)
        raise


def unregister():
    if not operators.ensure_native_teardown(shutdown_runtime=True):
        print(_UNREGISTER_TEARDOWN_FAILURE)
        return False

    # Порядок обратный: UI первым (ссылается на операторы),
    # затем операторы (ссылаются на PropertyGroup),
    # затем PropertyGroup-ы.
    cloth_settings_bridge.restore_all_cpu_owners()
    ui.unregister()
    if operators.unregister() is False:
        raise RuntimeError(_UNREGISTER_TEARDOWN_FAILURE)
    properties.unregister()
    return True
