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

import bpy


_REGISTER_TEARDOWN_FAILURE = (
    "GPUCloth package register blocked by retained native owners")
_UNREGISTER_TEARDOWN_FAILURE = (
    "GPUCloth package unregister blocked by retained native owners")
_EXIT_TEARDOWN_FAILURE = (
    "GPUCloth exit teardown retained native owners")

# Blender's own exit hook, chained rather than replaced, and the once-only guard
# for it: ``WM_exit_ex`` runs that hook, so calling it back from inside the hook
# would run Blender's whole disable-all pass a second time over classes this
# package has already unregistered.
_exit_hook_previous = None
_exit_teardown_done = False


def _exit_teardown():
    """Release the CUDA owners inside Blender's exit, before driver teardown.

    ``bpy.utils._on_exit`` is what ``WM_exit_ex`` runs to disable add-ons, and
    it runs while the CUDA driver is still usable.  A prepared simulation that
    is still owning managed memory when the process reaches static destruction
    frees it after the driver has begun shutting down, which aborts the process
    (``GPUassert: driver shutting down``).  Running the addon's existing
    teardown here releases that memory while the driver can still serve it.

    The wrapped hook is not called from here: Blender already runs it, and it
    performs its add-on disable pass over objects this teardown has released.
    """
    global _exit_teardown_done
    if _exit_teardown_done:
        return
    _exit_teardown_done = True
    try:
        if unregister() is False:
            print(_EXIT_TEARDOWN_FAILURE)
    except Exception as exc:  # an exit path must not raise into Blender's exit
        print(f"GPUCloth exit teardown failed: {exc}")


def _install_exit_hook():
    """Chain this package's teardown onto Blender's exit hook, once."""
    global _exit_hook_previous
    previous = getattr(bpy.utils, "_on_exit", None)
    if previous is None or previous is _exit_teardown:
        return False
    _exit_hook_previous = previous
    bpy.utils._on_exit = _exit_teardown
    return True


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
    _install_exit_hook()


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
