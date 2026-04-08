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
import os
import sys
import subprocess
import time

import numpy as np
from ctypes import (
    cdll, windll, POINTER, pointer, cast,
    c_bool, c_float, c_short, c_int, c_void_p, c_size_t, c_char_p,
)

from . import cpp_types as CType
from ..utils import version_compatibility_utils as vcu

debug = False
if sys.gettrace() is not None:
    debug = True

# ===========================================================================
#  Глобальное состояние симуляции
# ===========================================================================

g_dll                = None   # Загруженная DLL / .so
g_scene              = None   # Указатель на CType.Scene
g_obj                = []     # list[POINTER(CType.Object)]  — объекты ткани
g_clmd               = []     # list[POINTER(CType.ClothModifierData)]
g_mesh               = []     # list[POINTER(CType.Mesh)]
g_clothOBJs          = []     # list[bpy.types.Object]  — Blender-объекты ткани
g_clothCollisionOBJs = []     # list[POINTER(CType.Object)] — объекты столкновения

# Proxy-res: один handle на объект ткани (None если proxy не активен)
g_proxy_handles      = []     # list[c_void_p | None]
_collision_keepalive  = []     # prevent GC of collision ctypes data

# Защита от GC для ctypes-массивов, переданных в Cache_write_frame_async
# C++ пишет в фоне — массив должен жить до завершения записи
_live_arrays         = []     # list[c_float array]

_cache_playback_guard  = {'active': False}
_initial_positions     = []     # list[np.ndarray] — rest positions per cloth object
_bake_range            = {'start': 1, 'end': 250}


def _store_initial_positions():
    global _initial_positions
    _initial_positions.clear()
    for cloth_obj in g_clothOBJs:
        nV = len(cloth_obj.data.vertices)
        pos = np.empty(nV * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", pos)
        _initial_positions.append(pos.copy())


def _restore_initial_positions():
    for i, cloth_obj in enumerate(g_clothOBJs):
        if i < len(_initial_positions):
            cloth_obj.data.vertices.foreach_set("co", _initial_positions[i])
            cloth_obj.data.update()
            cloth_obj.data.update_tag()


def _frame_change_handler(scene, depsgraph):
    if _cache_playback_guard['active']:
        return
    scene_s = scene.gpu_cloth_helper
    if g_dll is None or not g_clothOBJs:
        return
    _cache_playback_guard['active'] = True
    try:
        frame = scene.frame_current

        if frame < _bake_range['start']:
            if _initial_positions:
                _restore_initial_positions()
                depsgraph.update()
            return

        if scene_s.is_baked and scene_s.playback_mode:
            cache_dir = scene_s.cache_dir
            cache_dir_bytes = bpy.path.abspath(cache_dir).encode('utf-8')

            if not g_dll.Cache_has_frame(frame, cache_dir_bytes):
                if _initial_positions:
                    _restore_initial_positions()
                    depsgraph.update()
                return

            updated = False
            for i, cloth_obj in enumerate(g_clothOBJs):
                if i >= len(g_clmd):
                    break
                nV = len(cloth_obj.data.vertices)
                pos = (c_float * (nV * 3))()
                if g_dll.Cache_load_frame_gpu(
                        frame, g_clmd[i], c_size_t(nV), cache_dir_bytes):
                    if g_dll.Cache_get_frame_positions(frame, pos, c_size_t(nV)):
                        flat = np.frombuffer(pos, dtype=np.float32)
                        cloth_obj.data.vertices.foreach_set("co", flat)
                        cloth_obj.data.update()
                        updated = True
                elif g_dll.Cache_prefetch_frame(frame, c_size_t(nV), cache_dir_bytes):
                    if g_dll.Cache_get_frame_positions(frame, pos, c_size_t(nV)):
                        flat = np.frombuffer(pos, dtype=np.float32)
                        cloth_obj.data.vertices.foreach_set("co", flat)
                        cloth_obj.data.update()
                        updated = True
            if updated:
                for cloth_obj in g_clothOBJs:
                    cloth_obj.data.update_tag()
                depsgraph.update()
        elif not scene_s.is_baked:
            try:
                bpy.ops.gpucloth.update_simulation()
            except RuntimeError:
                pass
    finally:
        _cache_playback_guard['active'] = False


# ===========================================================================
#  Вспомогательные функции
# ===========================================================================

def free_gpu_memory(context=None):
    """Освобождает GPU память, сбрасывает все глобальные массивы."""
    global g_dll, g_scene, g_obj, g_mesh, g_clmd
    global g_clothOBJs, g_clothCollisionOBJs, g_proxy_handles

    if g_dll is not None:
        # Освобождаем ProxySim handles перед FreeSolverData
        for handle in g_proxy_handles:
            if handle is not None:
                try:
                    g_dll.ProxySim_free(handle)
                except Exception:
                    pass
        try:
            g_dll.FreeSolverData()
        except Exception as e:
            print(f"free_gpu_memory: FreeSolverData() failed: {e}")
            return False

    g_scene              = None
    g_obj                = []
    g_clmd               = []
    g_mesh               = []
    g_clothOBJs          = []
    g_clothCollisionOBJs = []
    g_proxy_handles      = []
    _collision_keepalive.clear()
    _initial_positions.clear()
    _live_arrays.clear()

    if context is not None and hasattr(context.scene, 'gpu_cloth_springs_built'):
        context.scene.gpu_cloth_springs_built = False

    return True


# ===========================================================================
#  Оператор: освобождение VRAM
# ===========================================================================

class GPUCloth_FreeVRAM(bpy.types.Operator):
    """Освободить память GPU от данных симуляции"""
    bl_idname = "gpucloth.destroy_simulation_data"
    bl_label  = "Free GPU Memory"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        if not free_gpu_memory(context):
            self.report({'ERROR'}, "Не удалось освободить GPU память.")
            return {'CANCELLED'}
        self.report({'INFO'}, "GPU память освобождена.")
        return {'FINISHED'}


# ===========================================================================
#  Оператор: загрузка DLL
# ===========================================================================

class GPUCloth_LoadDLL(bpy.types.Operator):
    """Загрузить нативную библиотеку GPUCloth (DLL / .so)"""
    bl_idname = "gpucloth.load_dll"
    bl_label  = "Load GPUCloth DLL"

    # ── Проверка CUDA ────────────────────────────────────────────────────────

    def check_cuda_support(self):
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True,
            )
            if "CUDA Version" in result.stdout:
                return True
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"nvidia-smi завершился с ошибкой: {e}")
        except FileNotFoundError:
            self.report({'ERROR'},
                "nvidia-smi не найден. Убедитесь что установлены драйверы NVIDIA.")
        return False

    # ── Загрузка библиотеки и привязка функций ───────────────────────────────

    def load_dll(self):
        global g_dll
        if g_dll is not None:
            return True  # уже загружена

        if not self.check_cuda_support():
            return False

        # Ищем DLL через vcu (относительно директории аддона, без хардкода)
        filename = vcu.get_dll_path("GPUCloth.dll")
        if filename is None:
            lib_dir   = vcu.get_lib_directory()
            addon_dir = vcu.get_addon_directory()
            self.report({'ERROR'},
                f"GPUCloth.dll не найдена. "
                f"Искали в: {lib_dir} , {addon_dir} , {addon_dir}\\build\\ . "
                f"Скопируйте GPUCloth.dll в {lib_dir}")
            return False

        dll_dir = os.path.dirname(filename)
        if dll_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_dir)

        try:
            g_dll = cdll.LoadLibrary(filename)
            self.report({'INFO'}, f"DLL загружена: {filename}")

            # ── Существующие функции ─────────────────────────────────────────

            g_dll.FillSolverData.argtypes = [POINTER(CType.Scene)]
            g_dll.FillSolverData.restype  = c_bool

            g_dll.FreeSolverData.argtypes = []
            g_dll.FreeSolverData.restype  = c_bool

            g_dll.BuildClothSprings.argtypes = [
                POINTER(CType.ClothModifierData), POINTER(CType.Mesh)]
            g_dll.BuildClothSprings.restype = c_bool

            g_dll.SIM_solver.argtypes = []
            g_dll.SIM_solver.restype  = c_bool

            g_dll.AddCloth.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.Mesh),
                POINTER(CType.Object),
            ]
            g_dll.AddCloth.restype = c_bool

            g_dll.RemoveCloth.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.Mesh),
                POINTER(CType.Object),
            ]
            g_dll.RemoveCloth.restype = c_bool

            g_dll.AddCollisionObject.argtypes    = [POINTER(CType.Object)]
            g_dll.AddCollisionObject.restype     = c_bool

            g_dll.RemoveCollisionObject.argtypes = [POINTER(CType.Object)]
            g_dll.RemoveCollisionObject.restype  = c_bool

            g_dll.UpdateScene.argtypes = [POINTER(CType.Scene)]
            g_dll.UpdateScene.restype  = c_bool

            # ── Readback позиций вершин ──────────────────────────────────────
            #
            #   void SIM_get_cloth_verts(const ClothModifierData* clmd,
            #                            ClothVertex* out_verts, size_t count)
            #   Вместо прямого чтения mesh_ptr.contents.mvert — единственно
            #   корректный способ получить симулированные позиции.

            g_dll.SIM_get_cloth_verts.argtypes = [
                POINTER(CType.ClothModifierData),
                POINTER(CType.ClothVertex),
                c_size_t,
            ]
            g_dll.SIM_get_cloth_verts.restype = None

            # ── ProxySim API ─────────────────────────────────────────────────
            #
            #   Симуляция грубого proxy-меша + апсэмплинг до hi-res.
            #   GPU путь: SIM_solver() → scatter → ClothVertex.x (proxy)
            #             → ProxySim_apply → hi-res позиции → foreach_set

            g_dll.ProxySim_create.argtypes = [
                c_int, c_int,           # hi_NX,    hi_NY
                c_int, c_int,           # proxy_NX, proxy_NY
                c_int, c_int,           # num_sheets, scene_type
                POINTER(c_float),       # proxy_rest_pos  [nProxy*3]
                POINTER(c_float),       # hi_rest_pos     [nHi*3]
                c_int, c_int,           # nProxy, nHi
            ]
            g_dll.ProxySim_create.restype = c_void_p  # ProxySimHandle*

            g_dll.ProxySim_apply.argtypes = [
                c_void_p,           # ProxySimHandle*
                POINTER(c_float),   # proxy_pos  [nProxy*3]  — вход
                POINTER(c_float),   # hi_out     [nHi*3]     — выход
            ]
            g_dll.ProxySim_apply.restype = None

            g_dll.ProxySim_hi_count.argtypes    = [c_void_p]
            g_dll.ProxySim_hi_count.restype     = c_int

            g_dll.ProxySim_proxy_count.argtypes = [c_void_p]
            g_dll.ProxySim_proxy_count.restype  = c_int

            g_dll.ProxySim_free.argtypes = [c_void_p]
            g_dll.ProxySim_free.restype  = None

            # ── Cache API ────────────────────────────────────────────────────
            #
            #   ЗАПИСЬ (Phase 1, CPU-destination):
            #     SIM_get_cloth_verts → float[] → Cache_write_frame_async
            #     C++ пишет в фоне: pinned RAM → DMA → NVMe
            #
            #   ЧТЕНИЕ (Phase 2, GPU-destination, zero-copy):
            #     Cache_load_frame_gpu: NVMe → DMA → D3D12 resource (VRAM)
            #       → CUDA external memory (view, не копирование)
            #       → scatter_gpu_kernel → ClothVertex.x
            #       → cudaMemcpyAsync D2H → h_flat → foreach_set (viewport)

            g_dll.Cache_write_frame_async.argtypes = [
                c_int,              # frame
                POINTER(c_float),   # positions [nVerts*3]
                c_size_t,           # nVerts
                c_char_p,           # cache_dir (UTF-8)
            ]
            g_dll.Cache_write_frame_async.restype = c_bool

            g_dll.Cache_load_frame_gpu.argtypes = [
                c_int,                              # frame
                POINTER(CType.ClothModifierData),   # clmd (для scatter в ClothVertex.x)
                c_size_t,                           # nVerts
                c_char_p,                           # cache_dir
            ]
            g_dll.Cache_load_frame_gpu.restype = c_bool

            g_dll.Cache_prefetch_frame.argtypes = [
                c_int,      # frame
                c_size_t,   # nVerts
                c_char_p,   # cache_dir
            ]
            g_dll.Cache_prefetch_frame.restype = c_bool

            g_dll.Cache_is_frame_ready.argtypes = [c_int]
            g_dll.Cache_is_frame_ready.restype  = c_bool

            # После Cache_load_frame_gpu (GPU-direct): D2H для foreach_set
            g_dll.Cache_get_frame_positions.argtypes = [
                c_int,              # frame
                POINTER(c_float),   # out_positions [nVerts*3]
                c_size_t,           # nVerts
            ]
            g_dll.Cache_get_frame_positions.restype = c_bool

            g_dll.Cache_free_frame.argtypes = [c_int]
            g_dll.Cache_free_frame.restype  = c_bool

            g_dll.Cache_clear_all.argtypes = [c_char_p]
            g_dll.Cache_clear_all.restype  = c_bool

            g_dll.Cache_has_frame.argtypes = [c_int, c_char_p]
            g_dll.Cache_has_frame.restype  = c_bool

        except OSError as e:
            self.report({'ERROR'}, f"Не удалось загрузить DLL: {e}")
            g_dll = None
        except AttributeError as e:
            self.report({'ERROR'}, f"DLL не содержит ожидаемой функции: {e}")
            g_dll = None

        return g_dll is not None

    def execute(self, context):
        if not self.load_dll():
            self.report({'ERROR'}, "Не удалось загрузить DLL")
            return {'CANCELLED'}
        return {'FINISHED'}


# ===========================================================================
#  Оператор: выгрузка DLL
# ===========================================================================

class GPUCloth_UnloadDLL(bpy.types.Operator):
    """Выгрузить нативную библиотеку GPUCloth из Blender"""
    bl_idname = "gpucloth.unload_dll"
    bl_label  = "Unload GPUCloth DLL"

    @classmethod
    def poll(cls, context):
        return g_dll is not None

    def execute(self, context):
        global g_dll
        if g_dll is None:
            self.report({'WARNING'}, "DLL не загружена.")
            return {'CANCELLED'}
        try:
            # Windows: FreeLibrary через kernel32
            handle = c_void_p(g_dll._handle)
            result = windll.kernel32.FreeLibrary(handle)
            if result == 0:
                import ctypes
                raise ctypes.WinError()
            self.report({'INFO'}, "DLL успешно выгружена.")
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка при выгрузке DLL: {e}")
            return {'CANCELLED'}
        finally:
            g_dll = None
        return {'FINISHED'}


# ===========================================================================
#  Оператор: подготовка симуляции
# ===========================================================================

class GPUCloth_PrepareSimulation(bpy.types.Operator):
    """Подготовить данные и загрузить ткань на GPU"""
    bl_idname = "gpucloth.prepare_simulation"
    bl_label  = "Prepare GPUCloth Simulation"

    @classmethod
    def poll(cls, context):
        return True

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def fill_MVertTri_from_Object(self, obj: bpy.types.Object):
        """Извлекает треугольную топологию из меша объекта.
        Совместимо с Blender 4.1+ через vcu.calc_mesh_loop_triangles()."""
        if obj.type != 'MESH':
            return None
        mesh = obj.data
        # В Blender 4.1+ нужен явный вызов calc_loop_triangles()
        loop_tris = vcu.calc_mesh_loop_triangles(mesh)
        mvert_tris = (CType.MVertTri * len(loop_tris))()
        for i, tri in enumerate(loop_tris):
            mvert_tris[i].tri[0] = tri.vertices[0]
            mvert_tris[i].tri[1] = tri.vertices[1]
            mvert_tris[i].tri[2] = tri.vertices[2]
        return mvert_tris

    def fill_Scene(self, context):
        global g_scene
        scene = context.scene
        if scene.rigidbody_world is None:
            bpy.ops.rigidbody.world_add()
        g_scene = pointer(CType.Scene())
        g_scene.contents.flag = scene.rigidbody_world.enabled
        g_scene.contents.r = CType.RenderData(
            cfra=int(scene.frame_current),
            subframe=float(scene.frame_subframe),
            framelen=float(scene.render.frame_map_old),
            frs_sec=c_short(scene.render.fps),
        )
        g_scene.contents.physics_settings = CType.PhysicsSettings(
            gravity=(c_float * 3)(*scene.gravity),
            flag=CType.PHYS_GLOBAL_GRAVITY,
        )
        self.report({'INFO'}, "Scene заполнен")

    def fill_Object(self, OBJ: bpy.types.Object) -> POINTER(CType.Object):
        import numpy as np
        new_object = CType.Object()
        obmat = np.array(OBJ.matrix_world, dtype=np.float32)
        imat  = np.array(OBJ.matrix_world.inverted(), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                new_object.obmat[i][j] = obmat[i][j]
                new_object.imat[i][j]  = imat[i][j]
        new_object.pd = None

        for modif in OBJ.modifiers:
            if modif.type == 'COLLISION':
                tmp_collision = CType.CollisionModifierData()
                mvertType = CType.MVert * len(OBJ.data.vertices)
                mvert     = mvertType()
                for i, mv in enumerate(OBJ.data.vertices):
                    v = vcu.element_multiply(OBJ.matrix_world, mv.co)
                    mvert[i].co   = (c_float * 3)(*v)
                    mvert[i].flag = 0

                mvert_tri = self.fill_MVertTri_from_Object(OBJ)

                tmp_collision.x               = cast(mvert, POINTER(CType.MVert))
                tmp_collision.xnew            = cast(mvert, POINTER(CType.MVert))
                tmp_collision.xold            = cast(mvert, POINTER(CType.MVert))
                tmp_collision.current_xnew    = cast(mvert, POINTER(CType.MVert))
                tmp_collision.current_x       = cast(mvert, POINTER(CType.MVert))
                tmp_collision.current_v       = cast(mvert, POINTER(CType.MVert))
                tmp_collision.tri             = cast(mvert_tri, POINTER(CType.MVertTri))
                tmp_collision.mvert_num       = len(OBJ.data.vertices)
                loop_tris = vcu.calc_mesh_loop_triangles(OBJ.data)
                tmp_collision.tri_num         = len(loop_tris)
                tmp_collision.time_x          = -1000
                tmp_collision.time_xnew       = -1000
                tmp_collision.is_static       = True
                tmp_collision.bvhtree         = None
                new_object.modifiers          = pointer(tmp_collision)

                _collision_keepalive.extend(
                    [new_object, tmp_collision, mvert, mvert_tri])

        return pointer(new_object)

    def setMesh(self, context, obj) -> POINTER(CType.Mesh):
        if not obj:
            raise ValueError("obj не должен быть None")
        try:
            mesh      = CType.Mesh()
            mesh.totedge = len(obj.edges)
            mesh.totvert = len(obj.vertices)
            mesh.totpoly = len(obj.polygons)
            mesh.totloop = len(obj.loops)

            # Вершины
            mvertType = CType.MVert * len(obj.vertices)
            mvert     = mvertType()
            for i, v in enumerate(obj.vertices):
                mvert[i].co   = (c_float * 3)(*v.co)
                mvert[i].flag = 0
            mesh.mvert = cast(mvert, POINTER(CType.MVert))

            # Рёбра (crease/bweight убраны в Blender 4.0 как прямые поля,
            # здесь используются для C++ структуры — всегда 0)
            medgeType = CType.MEdge * len(obj.edges)
            medge     = medgeType()
            for i in obj.edges:
                medge[i.index].v1      = i.vertices[0]
                medge[i.index].v2      = i.vertices[1]
                medge[i.index].crease  = 0
                medge[i.index].bweight = 0
                medge[i.index].flag    = 35
            mesh.medge = cast(medge, POINTER(CType.MEdge))

            # Полигоны
            mpolyType = CType.MPoly * len(obj.polygons)
            mpoly     = mpolyType()
            for i in obj.polygons:
                mpoly[i.index].loopstart = i.loop_start
                mpoly[i.index].totloop   = i.loop_total
            mesh.mpoly = cast(mpoly, POINTER(CType.MPoly))

            # Loops
            mloopType = CType.MLoop * len(obj.loops)
            mloop     = mloopType()
            for idx, loop in enumerate(obj.loops):
                mloop[idx].v = loop.vertex_index
                mloop[idx].e = loop.edge_index
            mesh.mloop = cast(mloop, POINTER(CType.MLoop))

            return pointer(mesh)
        except Exception as e:
            raise RuntimeError(f"setMesh: {e}") from e

    def setClothModifierData(self, context, OBJ) -> POINTER(CType.ClothModifierData):
        clmd     = CType.ClothModifierData()
        sim_parms = CType.ClothSimSettings()

        # ── Базовые параметры ─────────────────────────────────────────────
        sim_parms.mingoal        = 0
        sim_parms.Cvi            = 1.0
        sim_parms.Cdis           = 1.0
        sim_parms.gravity[0]     = context.scene.gpu_cloth_helper.gravity_x
        sim_parms.gravity[1]     = context.scene.gpu_cloth_helper.gravity_y
        sim_parms.gravity[2]     = context.scene.gpu_cloth_helper.gravity_z
        sim_parms.mass           = OBJ.GPUCloth.vertex_mass
        sim_parms.structural     = 0
        sim_parms.shear          = OBJ.GPUCloth.shear
        sim_parms.bending        = OBJ.GPUCloth.bending_stiffness
        sim_parms.vgroup_mass    = 0
        sim_parms.stepsPerFrame  = OBJ.GPUCloth.quality_step
        sim_parms.maxgoal        = 1.0
        sim_parms.velocity_smooth= 0.0
        sim_parms.collider_friction = 0.0
        sim_parms.shrink_min     = 0.0
        sim_parms.shrink_max     = 0.0
        sim_parms.vgroup_bend    = 0
        sim_parms.vgroup_struct  = 0
        sim_parms.vgroup_shear   = 0
        sim_parms.vgroup_shrink  = 0
        sim_parms.bending_damping= OBJ.GPUCloth.bending_damping
        sim_parms.voxel_cell_size= 0.1
        sim_parms.tension        = OBJ.GPUCloth.tension
        sim_parms.compression    = OBJ.GPUCloth.compression
        sim_parms.tension_damp   = OBJ.GPUCloth.tension_damp
        sim_parms.compression_damp = OBJ.GPUCloth.compression_damp
        sim_parms.shear_damp     = OBJ.GPUCloth.shear_damp
        sim_parms.internal_spring_max_length     = 10
        sim_parms.internal_spring_max_diversion  = 0.7853981633974483  # pi/4
        sim_parms.vgroup_intern  = 0
        sim_parms.internal_tension     = 15.0
        sim_parms.internal_compression = 15.0
        sim_parms.max_internal_tension     = 15.0
        sim_parms.max_internal_compression = 15.0
        sim_parms.eff_force_scale  = 1000.0
        sim_parms.eff_wind_scale   = 250.0
        sim_parms.effector_weights = None
        sim_parms.reset            = 0
        sim_parms.presets          = 2
        sim_parms.shapekey_rest    = 0
        sim_parms.fluid_density    = 0.0
        sim_parms.pressure_factor  = 1.0
        sim_parms.target_volume    = 0.0
        sim_parms.uniform_pressure_force = 0.0
        sim_parms.time_scale       = OBJ.GPUCloth.speed_multiplier
        sim_parms.timescale        = 1.0
        sim_parms.dt               = 1
        sim_parms.avg_spring_len   = 0.0
        sim_parms.goalfrict        = 0.0
        sim_parms.goalspring       = 1.0
        sim_parms.flags            = CType.CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS_NORMAL

        # Модель изгиба
        sim_parms.bending_model = (
            CType.CLOTH_BENDING_ANGULAR
            if OBJ.GPUCloth.bending_model == 'ANGULAR'
            else CType.CLOTH_BENDING_LINEAR
        )

        # ── Тип солвера (НОВОЕ) ───────────────────────────────────────────
        # Маппинг строки EnumProperty → int константа C++
        _SOLVER_MAP = {
            'XPBD':  CType.SOLVER_XPBD,
            'PD':    CType.SOLVER_PD,
            'MGPBD': CType.SOLVER_MGPBD,
            'Mil2':  CType.SOLVER_Mil2,
            'OGC':   CType.SOLVER_OGC,
        }
        sim_parms.solver_type = _SOLVER_MAP.get(
            OBJ.GPUCloth.solver_type, CType.SOLVER_XPBD
        )

        clmd.sim_parms    = pointer(sim_parms)
        clmd.clothObject  = None

        # ── Параметры столкновений ────────────────────────────────────────
        coll_parms = pointer(CType.ClothCollSettings())
        coll_parms.contents.epsilon       = 0.015
        coll_parms.contents.self_friction = 5.0
        coll_parms.contents.friction      = 5.0
        coll_parms.contents.damping       = 0.0
        coll_parms.contents.selfepsilon   = 0.015
        coll_parms.contents.loop_count    = 2
        coll_parms.contents.group         = None
        coll_parms.contents.vgroup_selfcol = 0
        coll_parms.contents.vgroup_objcol  = 0
        coll_parms.contents.clamp          = 0.0
        coll_parms.contents.self_clamp     = 0.0
        coll_parms.contents.flags          = CType.CLOTH_COLLSETTINGS_FLAG_ENABLED
        clmd.coll_parms = coll_parms

        # ── Результат солвера ─────────────────────────────────────────────
        solver_result = CType.ClothSolverResult()
        solver_result.status         = 0
        solver_result.max_iterations = 0
        solver_result.avg_iterations = 0
        solver_result.max_error      = 0.0
        solver_result.min_error      = 0.0
        solver_result.avg_error      = 0.0
        clmd.solver_result = pointer(solver_result)

        return pointer(clmd)

    # ── Execute ──────────────────────────────────────────────────────────────

    def execute(self, context):
        global g_dll, g_scene, g_obj, g_mesh, g_clmd
        global g_clothOBJs, g_clothCollisionOBJs, g_proxy_handles

        # 1. Загружаем DLL если нужно
        if g_dll is None:
            bpy.ops.gpucloth.load_dll()
            if g_dll is None:
                self.report({'ERROR'}, "Не удалось загрузить DLL")
                return {'CANCELLED'}

        # 2. Освобождаем старые данные если были
        if context.scene.gpu_cloth_springs_built:
            if not free_gpu_memory(context):
                self.report({'ERROR'}, "Не удалось освободить GPU память")
                return {'CANCELLED'}

        # 3. Сохраняем файл (DLL нужен путь к blend для ряда операций)
        if not bpy.data.is_saved:
            bpy.ops.wm.save_as_mainfile(
                filepath=bpy.app.tempdir + 'GPU_Cloth.blend',
                check_existing=False)
        elif bpy.data.is_dirty:
            bpy.ops.wm.save_as_mainfile(
                filepath=bpy.data.filepath, check_existing=False)

        # 4. Переходим в Object mode для корректного считывания данных
        mode = bpy.context.active_object.mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # 5. Собираем объекты ткани и столкновения
        for item in bpy.context.scene.objects:
            if item is None:
                continue
            if hasattr(item, 'GPUCloth') and item.GPUCloth.is_active:
                g_clothOBJs.append(item)
                for modifier in item.modifiers:
                    if modifier.type == 'CLOTH':
                        modifier.show_viewport = False
                        modifier.show_render   = False
            else:
                for modif in item.modifiers:
                    if modif.type == 'COLLISION':
                        data_ptr = self.fill_Object(item)
                        if not data_ptr:
                            self.report({'ERROR'}, "NULL от fill_Object (collision)")
                            return {'CANCELLED'}
                        g_clothCollisionOBJs.append(data_ptr)

        # 6. Заполняем Mesh + ClothModifierData + Object для каждой ткани
        for cloth_obj in g_clothOBJs:
            if cloth_obj is None or not hasattr(cloth_obj, 'data'):
                return {'CANCELLED'}

            data_ptr = self.setMesh(context, cloth_obj.data)
            if not data_ptr:
                self.report({'ERROR'}, "NULL от setMesh")
                return {'CANCELLED'}
            g_mesh.append(data_ptr)

            data_ptr = self.setClothModifierData(context, cloth_obj)
            if not data_ptr:
                self.report({'ERROR'}, "NULL от setClothModifierData")
                return {'CANCELLED'}
            g_clmd.append(data_ptr)

            data_ptr = self.fill_Object(cloth_obj)
            if not data_ptr:
                self.report({'ERROR'}, "NULL от fill_Object (cloth)")
                return {'CANCELLED'}
            g_obj.append(data_ptr)

        if len(g_clothOBJs) != len(g_clmd):
            self.report({'ERROR'},
                f"Несоответствие объектов: clothOBJs={len(g_clothOBJs)}, clmd={len(g_clmd)}")
            return {'CANCELLED'}

        # 7. Загружаем сцену на GPU
        self.fill_Scene(context)

        for coll_ptr in g_clothCollisionOBJs:
            try:
                if not g_dll.AddCollisionObject(coll_ptr):
                    self.report({'ERROR'}, "Ошибка AddCollisionObject")
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError в AddCollisionObject")
                return {'CANCELLED'}

        if not g_dll.FillSolverData(g_scene):
            self.report({'ERROR'}, "FillSolverData вернул ошибку")
            return {'CANCELLED'}

        for i in range(len(g_clothOBJs)):
            try:
                if not g_clmd[i].contents.clothObject:
                    if not g_dll.BuildClothSprings(g_clmd[i], g_mesh[i]):
                        self.report({'ERROR'}, "BuildClothSprings вернул ошибку")
                        return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError в BuildClothSprings")
                return {'CANCELLED'}

            try:
                if g_clmd[i].contents.clothObject is None:
                    self.report({'ERROR'}, "clothObject is NULL после BuildClothSprings")
                    bpy.ops.screen.animation_cancel()
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError: clothObject is NULL")
                return {'CANCELLED'}

            try:
                if not g_dll.AddCloth(g_clmd[i], g_mesh[i], g_obj[i]):
                    self.report({'ERROR'}, "AddCloth вернул ошибку")
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError в AddCloth")
                return {'CANCELLED'}

        # 8. Инициализируем Proxy-res для объектов с use_proxy=True (НОВОЕ)
        g_proxy_handles.clear()
        for cloth_obj in g_clothOBJs:
            s = cloth_obj.GPUCloth
            if s.use_proxy and s.proxy_object is not None:
                nProxy = len(s.proxy_object.data.vertices)
                nHi    = len(cloth_obj.data.vertices)

                proxy_rest = (c_float * (nProxy * 3))(
                    *[c for v in s.proxy_object.data.vertices for c in v.co])
                hi_rest = (c_float * (nHi * 3))(
                    *[c for v in cloth_obj.data.vertices for c in v.co])

                handle = g_dll.ProxySim_create(
                    s.hi_nx,   s.hi_ny,
                    s.proxy_nx, s.proxy_ny,
                    s.num_sheets, s.proxy_scene_type,
                    proxy_rest, hi_rest,
                    nProxy, nHi,
                )
                g_proxy_handles.append(handle)
                if handle is None:
                    self.report({'WARNING'},
                        f"ProxySim_create вернул NULL для {cloth_obj.name}")
            else:
                g_proxy_handles.append(None)

        # 9. Возвращаем режим редактирования
        bpy.ops.object.mode_set(mode=mode)
        context.scene.gpu_cloth_springs_built = True
        _store_initial_positions()
        return {'FINISHED'}


# ===========================================================================
#  Оператор: обновление симуляции (один кадр)
# ===========================================================================

class GPUCloth_UpdateSimulation(bpy.types.Operator):
    """Просчитать один кадр симуляции и обновить меш в Blender"""
    bl_idname = "gpucloth.update_simulation"
    bl_label  = "Update GPUCloth Simulation"

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def _get_positions(self, clmd_ptr, nVerts):
        """
        Читает позиции вершин из GPU через SIM_get_cloth_verts.
        Возвращает плоский ctypes массив float[nVerts*3] (x0,y0,z0,x1,y1,z1,...)
        или None при ошибке.
        """
        buf = (CType.ClothVertex * nVerts)()
        g_dll.SIM_get_cloth_verts(clmd_ptr, buf, c_size_t(nVerts))
        pos = (c_float * (nVerts * 3))()
        for i in range(nVerts):
            pos[i * 3]     = buf[i].x[0]
            pos[i * 3 + 1] = buf[i].x[1]
            pos[i * 3 + 2] = buf[i].x[2]
        return pos

    def _apply_positions(self, blender_obj, pos, nVerts):
        """
        Применяет плоский float3 массив к мешу Blender через foreach_set.
        foreach_set — единственный корректный и быстрый способ в Blender 4.x.
        Прямое присваивание vertices[i].co = ... работает медленно и
        требует mesh.update() для отображения.
        """
        flat = np.frombuffer(pos, dtype=np.float32)
        blender_obj.data.vertices.foreach_set("co", flat)
        blender_obj.data.update()

    def validate_objects(self):
        for obj in g_clothOBJs:
            if obj is None or obj.name not in bpy.data.objects:
                self.report({'ERROR'}, f"Объект {obj} не существует в сцене.")
                return False
            if obj.type != 'MESH':
                self.report({'ERROR'}, f"Объект {obj.name} не является MESH.")
                return False
        return True

    # ── Execute ──────────────────────────────────────────────────────────────

    def execute(self, context):
        global g_dll, g_obj, g_mesh, g_clmd, g_clothOBJs

        if g_dll is None or context.scene.frame_current < 2:
            return {'FINISHED'}
        if not self.validate_objects():
            return {'FINISHED'}

        scene_s   = context.scene.gpu_cloth_helper
        cache_dir = bpy.path.abspath(scene_s.cache_dir).encode('utf-8')
        frame     = context.scene.frame_current

        # ── РЕЖИМ ВОСПРОИЗВЕДЕНИЯ из кэша ───────────────────────────────────
        #
        #   Phase 2 (GPU-destination, zero-copy):
        #     Cache_load_frame_gpu: NVMe → D3D12 VRAM → CUDA external memory
        #     → scatter_gpu_kernel → ClothVertex.x
        #     Cache_get_frame_positions: D2H → h_flat → foreach_set
        #
        if scene_s.playback_mode and scene_s.is_baked:
            for i, cloth_obj in enumerate(g_clothOBJs):
                nV  = len(cloth_obj.data.vertices)
                pos = (c_float * (nV * 3))()
                # GPU-direct load для данного кадра
                if g_dll.Cache_load_frame_gpu(
                        frame, g_clmd[i], c_size_t(nV), cache_dir):
                    # D2H для foreach_set (viewport)
                    if g_dll.Cache_get_frame_positions(frame, pos, c_size_t(nV)):
                        self._apply_positions(cloth_obj, pos, nV)
            # Prefetch следующего кадра пока пользователь смотрит текущий
            for cloth_obj in g_clothOBJs:
                g_dll.Cache_prefetch_frame(
                    frame + 1,
                    c_size_t(len(cloth_obj.data.vertices)),
                    cache_dir,
                )
            return {'FINISHED'}

        # ── РЕЖИМ ЖИВОЙ СИМУЛЯЦИИ ────────────────────────────────────────────
        #
        #   SIM_solver() [GPU ~N мс]
        #   SIM_get_cloth_verts() [memcpy D2H ~<1мс]
        #   foreach_set() [Blender ~<1мс]
        #   Cache_write_frame_async() ← возвращает немедленно, пишет в фоне
        #

        if not (len(g_clothOBJs) == len(g_clmd) == len(g_obj) == len(g_mesh)):
            self.report({'ERROR'},
                f"Несоответствие размеров: clothOBJs={len(g_clothOBJs)}, "
                f"clmd={len(g_clmd)}, obj={len(g_obj)}, mesh={len(g_mesh)}")
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}

        total_t0 = time.perf_counter_ns()

        try:
            for i in range(len(g_clothOBJs)):
                cloth_obj = g_clothOBJs[i]
                nV        = len(cloth_obj.data.vertices)

                # 1. GPU симуляция одного кадра
                if not g_dll.SIM_solver():
                    self.report({'ERROR'}, "SIM_solver вернул ошибку")
                    bpy.ops.screen.animation_cancel()
                    return {'CANCELLED'}

                # 2. Readback позиций (D2H через SIM_get_cloth_verts)
                pos = self._get_positions(g_clmd[i], nV)
                if pos is None:
                    self.report({'ERROR'}, "SIM_get_cloth_verts вернул ошибку")
                    bpy.ops.screen.animation_cancel()
                    return {'CANCELLED'}

                # 3. Proxy upsampling (если активен)
                #    proxy_pos заполняется из результатов симуляции proxy-меша
                handle = (g_proxy_handles[i]
                          if i < len(g_proxy_handles) else None)
                if handle is not None:
                    nP = g_dll.ProxySim_proxy_count(handle)
                    proxy_pos = (c_float * (nP * 3))()
                    # Позиции proxy-меша берутся из _get_positions proxy clmd
                    # (предполагается, что proxy симулируется отдельным AddCloth)
                    out_hi = (c_float * (g_dll.ProxySim_hi_count(handle) * 3))()
                    g_dll.ProxySim_apply(handle, proxy_pos, out_hi)
                    pos = out_hi
                    nV  = g_dll.ProxySim_hi_count(handle)

                # 4. Обновляем меш в Blender (foreach_set, Blender 4.x safe)
                self._apply_positions(cloth_obj, pos, nV)

                # 5. Асинхронная запись кэша
                #    Phase 1 (CPU-destination): C++ пишет в фоне
                #    pinned RAM → DMA → NVMe
                #    _live_arrays защищает pos от GC пока C++ работает
                if not scene_s.is_baked:
                    _live_arrays.append(pos)
                    g_dll.Cache_write_frame_async(
                        frame, pos, c_size_t(nV), cache_dir)

        except OSError as err:
            print(f"GPUCloth_UpdateSimulation OSError: {err}")
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}

        elapsed_ms = (time.perf_counter_ns() - total_t0) / 1_000_000
        print(f"[GPUCloth] кадр {frame}: {elapsed_ms:.2f} мс")
        return {'FINISHED'}


# ===========================================================================
#  Оператор: запекание (bake) симуляции
# ===========================================================================

class GPUCloth_BakeSimulation(bpy.types.Operator):
    """
    Просчитать симуляцию для всего диапазона кадров и записать кэш на диск.
    Паттерн: FLIP Fluids BakeFluidSimulation (modal с таймером).

    На каждом кадре:
      SIM_solver() → SIM_get_cloth_verts() → foreach_set() → Cache_write_frame_async()
    """
    bl_idname = "gpucloth.bake_simulation"
    bl_label  = "Запечь симуляцию GPU Cloth"
    _timer    = None

    @classmethod
    def poll(cls, context):
        return (
            g_dll is not None
            and context.scene.gpu_cloth_springs_built
            and not context.scene.gpu_cloth_helper.is_baked
        )

    def modal(self, context, event):
        s = context.scene.gpu_cloth_helper

        if event.type == 'TIMER':
            if self._frame > s.bake_end:
                self._finish(context, success=True)
                return {'FINISHED'}

            # Просчёт кадра (UpdateSimulation пишет кэш async внутри)
            context.scene.frame_set(self._frame)
            bpy.ops.gpucloth.update_simulation()

            # Прогресс
            total = max(s.bake_end - s.bake_start + 1, 1)
            s.bake_progress = int(100 * (self._frame - s.bake_start) / total)
            self._frame += 1

            for area in context.screen.areas:
                area.tag_redraw()

        if event.type == 'ESC':
            self._finish(context, success=False)
            self.report({'INFO'}, "Запекание отменено (ESC)")
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _finish(self, context, success: bool):
        context.window_manager.event_timer_remove(self._timer)
        s = context.scene.gpu_cloth_helper
        s.is_baked      = success
        s.bake_progress = 100 if success else 0
        if success:
            s.playback_mode = True
            _bake_range['start'] = s.bake_start
            _bake_range['end']   = s.bake_end
        _live_arrays.clear()
        if success:
            self.report({'INFO'}, "Запекание завершено.")

    def invoke(self, context, event):
        s = context.scene.gpu_cloth_helper
        self._frame = s.bake_start
        _live_arrays.clear()
        self._timer = context.window_manager.event_timer_add(
            0.001, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


# ===========================================================================
#  Оператор: очистка кэша
# ===========================================================================

class GPUCloth_FreeCache(bpy.types.Operator):
    """Удалить все файлы кэша симуляции с диска"""
    bl_idname = "gpucloth.free_cache"
    bl_label  = "Очистить кэш"

    @classmethod
    def poll(cls, context):
        return (
            g_dll is not None
            and context.scene.gpu_cloth_helper.is_baked
        )

    def execute(self, context):
        s         = context.scene.gpu_cloth_helper
        cache_dir = bpy.path.abspath(s.cache_dir).encode('utf-8')

        if not g_dll.Cache_clear_all(cache_dir):
            self.report({'ERROR'}, "Не удалось очистить кэш")
            return {'CANCELLED'}

        s.is_baked      = False
        s.bake_progress = 0
        s.playback_mode = False
        self.report({'INFO'}, "Кэш очищен.")
        return {'FINISHED'}


# ===========================================================================
#  Хелпер: frame_change_post обработчик для экспорта из кэша
# ===========================================================================
#
#   При экспорте Alembic/USD Blender внутренне вызывает scene.frame_set()
#   для каждого кадра.  frame_change_post обработчик загружает позиции
#   из кэша и записывает их в меш, после чего вызывает depsgraph.update()
#   чтобы экспортёр увидел актуальную геометрию.

def _make_cache_handler(cache_dir_bytes):
    """Создаёт frame_change_post обработчик для загрузки кэша при экспорте."""
    _guard = {'active': False}

    def _handler(scene, depsgraph):
        if _guard['active']:
            return
        _guard['active'] = True
        try:
            frame = scene.frame_current
            updated = False
            for i, cloth_obj in enumerate(g_clothOBJs):
                if i >= len(g_clmd):
                    break
                nV  = len(cloth_obj.data.vertices)
                pos = (c_float * (nV * 3))()
                if g_dll.Cache_load_frame_gpu(
                        frame, g_clmd[i], c_size_t(nV), cache_dir_bytes):
                    if g_dll.Cache_get_frame_positions(frame, pos, c_size_t(nV)):
                        flat = np.frombuffer(pos, dtype=np.float32)
                        cloth_obj.data.vertices.foreach_set("co", flat)
                        cloth_obj.data.update()
                        updated = True
            if updated:
                for cloth_obj in g_clothOBJs:
                    cloth_obj.data.update_tag()
                depsgraph.update()
        finally:
            _guard['active'] = False

    return _handler


# ===========================================================================
#  Оператор: экспорт Alembic (.abc)
# ===========================================================================

class GPUCloth_ExportAlembic(bpy.types.Operator):
    """Экспорт запечённой симуляции в Alembic (.abc)"""
    bl_idname  = "gpucloth.export_alembic"
    bl_label   = "Экспорт Alembic"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(
        name="Путь",
        subtype='FILE_PATH',
    )
    filename_ext = ".abc"
    filter_glob: bpy.props.StringProperty(
        default="*.abc",
        options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return (
            g_dll is not None
            and context.scene.gpu_cloth_springs_built
            and context.scene.gpu_cloth_helper.is_baked
            and len(g_clothOBJs) > 0
        )

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "//gpucloth_export.abc"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene_s   = context.scene.gpu_cloth_helper
        cache_dir = bpy.path.abspath(scene_s.cache_dir).encode('utf-8')

        # Выделяем только объекты ткани для экспорта
        prev_selection = [o for o in context.scene.objects if o.select_get()]
        bpy.ops.object.select_all(action='DESELECT')
        for obj in g_clothOBJs:
            obj.select_set(True)

        handler = _make_cache_handler(cache_dir)
        bpy.app.handlers.frame_change_post.append(handler)
        try:
            filepath = bpy.path.abspath(self.filepath)
            if not filepath.lower().endswith('.abc'):
                filepath += '.abc'
            bpy.ops.wm.alembic_export(
                'EXEC_DEFAULT',
                filepath=filepath,
                start=scene_s.bake_start,
                end=scene_s.bake_end,
                selected=True,
                visible_objects_only=False,
                export_hair=False,
                export_particles=False,
                as_background_job=False,
            )
            self.report({'INFO'}, f"Alembic экспортирован: {filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка экспорта Alembic: {e}")
            return {'CANCELLED'}
        finally:
            if handler in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.remove(handler)
            # Восстанавливаем выделение
            bpy.ops.object.select_all(action='DESELECT')
            for obj in prev_selection:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)

        return {'FINISHED'}


# ===========================================================================
#  Оператор: экспорт USD (.usd / .usdc / .usda)
# ===========================================================================

class GPUCloth_ExportUSD(bpy.types.Operator):
    """Экспорт запечённой симуляции в Universal Scene Description (.usd)"""
    bl_idname  = "gpucloth.export_usd"
    bl_label   = "Экспорт USD"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(
        name="Путь",
        subtype='FILE_PATH',
    )
    filename_ext = ".usdc"
    filter_glob: bpy.props.StringProperty(
        default="*.usd;*.usdc;*.usda",
        options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return (
            g_dll is not None
            and context.scene.gpu_cloth_springs_built
            and context.scene.gpu_cloth_helper.is_baked
            and len(g_clothOBJs) > 0
        )

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "//gpucloth_export.usdc"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene_s   = context.scene.gpu_cloth_helper
        cache_dir = bpy.path.abspath(scene_s.cache_dir).encode('utf-8')

        prev_selection = [o for o in context.scene.objects if o.select_get()]
        bpy.ops.object.select_all(action='DESELECT')
        for obj in g_clothOBJs:
            obj.select_set(True)

        handler = _make_cache_handler(cache_dir)
        bpy.app.handlers.frame_change_post.append(handler)
        try:
            filepath = bpy.path.abspath(self.filepath)
            valid_ext = ('.usd', '.usdc', '.usda')
            if not any(filepath.lower().endswith(ext) for ext in valid_ext):
                filepath += '.usdc'
            bpy.ops.wm.usd_export(
                'EXEC_DEFAULT',
                filepath=filepath,
                selected_objects_only=True,
                visible_objects_only=False,
                export_animation=True,
                export_hair=False,
            )
            self.report({'INFO'}, f"USD экспортирован: {filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка экспорта USD: {e}")
            return {'CANCELLED'}
        finally:
            if handler in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.remove(handler)
            bpy.ops.object.select_all(action='DESELECT')
            for obj in prev_selection:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)

        return {'FINISHED'}


# ===========================================================================
#  Регистрация
# ===========================================================================

_OPERATOR_CLASSES = [
    GPUCloth_FreeVRAM,
    GPUCloth_LoadDLL,
    GPUCloth_UnloadDLL,
    GPUCloth_PrepareSimulation,
    GPUCloth_UpdateSimulation,
    GPUCloth_BakeSimulation,
    GPUCloth_FreeCache,
    GPUCloth_ExportAlembic,
    GPUCloth_ExportUSD,
]


def register():
    for cls in _OPERATOR_CLASSES:
        bpy.utils.register_class(cls)

    # Сбрасываем глобальное состояние при регистрации
    global g_dll, g_scene, g_obj, g_mesh, g_clmd
    global g_clothOBJs, g_clothCollisionOBJs, g_proxy_handles
    g_dll                = None
    g_scene              = None
    g_obj                = []
    g_clmd               = []
    g_mesh               = []
    g_clothOBJs          = []
    g_clothCollisionOBJs = []
    g_proxy_handles      = []
    _collision_keepalive.clear()
    _initial_positions.clear()
    _live_arrays.clear()

    if _frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_frame_change_handler)


def unregister():
    if _frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_frame_change_handler)

    for cls in reversed(_OPERATOR_CLASSES):
        bpy.utils.unregister_class(cls)

    # Очищаем состояние
    global g_dll, g_scene, g_obj, g_mesh, g_clmd
    global g_clothOBJs, g_clothCollisionOBJs, g_proxy_handles
    g_dll                = None
    g_scene              = None
    g_obj                = []
    g_clmd               = []
    g_mesh               = []
    g_clothOBJs          = []
    g_clothCollisionOBJs = []
    g_proxy_handles      = []
    _collision_keepalive.clear()
    _initial_positions.clear()
    _live_arrays.clear()
