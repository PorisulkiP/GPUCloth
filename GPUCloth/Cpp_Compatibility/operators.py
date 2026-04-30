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
import math
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
        gs       = OBJ.GPUCloth
        gs_scene = context.scene.gpu_cloth_helper

        # ── Базовые параметры ─────────────────────────────────────────────
        sim_parms.mingoal        = 0
        sim_parms.Cvi            = 1.0
        sim_parms.Cdis           = gs.air_viscosity
        sim_parms.gravity[0]     = gs_scene.gravity_x
        sim_parms.gravity[1]     = gs_scene.gravity_y
        sim_parms.gravity[2]     = gs_scene.gravity_z
        sim_parms.mass           = gs.vertex_mass
        sim_parms.structural     = gs.structural
        sim_parms.shear          = gs.shear
        sim_parms.bending        = gs.bending_stiffness
        sim_parms.vgroup_mass    = 0  # set via vertex group data injection
        sim_parms.stepsPerFrame  = gs.quality_step
        sim_parms.maxgoal        = gs.maxgoal
        sim_parms.velocity_smooth= 0.0
        sim_parms.collider_friction = 0.0
        sim_parms.shrink_min     = gs.shrink_min
        sim_parms.shrink_max     = gs.shrink_max
        sim_parms.vgroup_bend    = 0
        sim_parms.vgroup_struct  = 0
        sim_parms.vgroup_shear   = 0
        sim_parms.vgroup_shrink  = 0
        sim_parms.bending_damping= gs.bending_damping
        sim_parms.voxel_cell_size= 0.1
        sim_parms.tension        = gs.tension
        sim_parms.compression    = gs.compression
        sim_parms.tension_damp   = gs.tension_damp
        sim_parms.compression_damp = gs.compression_damp
        sim_parms.shear_damp     = gs.shear_damp
        sim_parms.max_tension    = gs.max_tension
        sim_parms.max_compression = gs.max_compression
        sim_parms.max_shear      = gs.max_shear
        sim_parms.max_bend       = gs.max_bend
        sim_parms.max_struct     = gs.max_struct
        sim_parms.max_sewing     = gs.max_sewing
        sim_parms.vel_damping    = gs.vel_damping

        # ── Internal Springs ───────────────────────────────────────────────
        sim_parms.internal_spring_max_length     = gs.internal_spring_max_length
        sim_parms.internal_spring_max_diversion  = gs.internal_spring_max_diversion
        sim_parms.vgroup_intern  = 0
        sim_parms.internal_tension     = gs.internal_tension
        sim_parms.internal_compression = gs.internal_compression
        sim_parms.max_internal_tension     = gs.max_internal_tension
        sim_parms.max_internal_compression = gs.max_internal_compression

        # ── Effector forces ────────────────────────────────────────────────
        sim_parms.eff_force_scale  = gs.eff_force_scale
        sim_parms.eff_wind_scale   = gs.eff_wind_scale
        sim_parms.effector_weights = None  # allocated separately if needed
        sim_parms.reset            = 0
        sim_parms.presets          = 2
        sim_parms.shapekey_rest    = 0

        # ── Pressure ──────────────────────────────────────────────────────
        sim_parms.fluid_density    = gs.fluid_density
        sim_parms.pressure_factor  = gs.pressure_factor
        sim_parms.target_volume    = gs.target_volume
        sim_parms.uniform_pressure_force = gs.uniform_pressure_force

        # ── Timing ────────────────────────────────────────────────────────
        sim_parms.time_scale       = gs.speed_multiplier
        sim_parms.timescale        = 1.0
        sim_parms.dt               = 1
        sim_parms.avg_spring_len   = 0.0
        sim_parms.goalfrict        = gs.goalfrict
        sim_parms.goalspring       = gs.goalspring

        # ── Flags ─────────────────────────────────────────────────────────
        sim_parms.flags = 0
        if gs.use_internal_springs:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS
        if gs.use_internal_springs_normal:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS_NORMAL
        if gs.use_pressure:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_PRESSURE
        if gs.use_dynamic_mesh:
            sim_parms.flags |= CType.CLOTH_SIMSETTINGS_FLAG_DYNAMIC_MESH

        # Модель изгиба
        sim_parms.bending_model = (
            CType.CLOTH_BENDING_ANGULAR
            if gs.bending_model == 'ANGULAR'
            else CType.CLOTH_BENDING_LINEAR
        )

        # ─�� Тип солвера ───────────────────────────────────────────────────
        _SOLVER_MAP = {
            'XPBD':  CType.SOLVER_XPBD,
            'PD':    CType.SOLVER_PD,
            'MGPBD': CType.SOLVER_MGPBD,
            'Mil2':  CType.SOLVER_Mil2,
            'OGC':   CType.SOLVER_OGC,
        }
        sim_parms.solver_type = _SOLVER_MAP.get(
            gs.solver_type, CType.SOLVER_XPBD
        )

        clmd.sim_parms    = pointer(sim_parms)
        clmd.clothObject  = None

        # ── Параметры столкновений ────────────────────────────────────────
        coll_parms = pointer(CType.ClothCollSettings())
        coll_parms.contents.epsilon       = gs.epsilon
        coll_parms.contents.self_friction = 5.0
        coll_parms.contents.friction      = 5.0
        coll_parms.contents.damping       = 0.0
        coll_parms.contents.selfepsilon   = gs.selfepsilon
        coll_parms.contents.loop_count    = 2
        coll_parms.contents.group         = None  # TODO: resolve Collection ptr
        coll_parms.contents.vgroup_selfcol = 0
        coll_parms.contents.vgroup_objcol  = 0
        coll_parms.contents.clamp          = gs.clamp
        coll_parms.contents.self_clamp     = gs.self_clamp
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
#  Test scene operators
# ===========================================================================

import bmesh


def _make_grid_mesh(name, nx, ny, half_size, height, pin_corners=False):
    """Create a subdivided grid mesh and return (obj, mesh_data)."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    sx = nx + 1
    sy = ny + 1
    verts = []
    for row in range(sy):
        for col in range(sx):
            x = -half_size + 2.0 * half_size * col / nx
            y = -half_size + 2.0 * half_size * row / ny
            verts.append((x, y, height))

    faces = []
    for row in range(ny):
        for col in range(nx):
            i = row * sx + col
            faces.append((i, i + 1, i + sx + 1, i + sx))

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    if pin_corners:
        for v in obj.data.vertices:
            pinned = False
            if (abs(v.co.x - (-half_size)) < 0.01 and abs(v.co.y - half_size) < 0.01):
                pinned = True
            if (abs(v.co.x - half_size) < 0.01 and abs(v.co.y - half_size) < 0.01):
                pinned = True
            if pinned:
                v.co.z += 0.0
        mesh_data.update()

    return obj, mesh_data


def _make_uv_sphere(name, radius, cx, cy, cz, rings=10, sectors=12):
    """Create a UV sphere collision object and return it."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    segs_loop = sectors
    segs_ring = rings
    import math
    for i in range(segs_ring + 1):
        phi = math.pi * i / segs_ring
        for j in range(segs_loop + 1):
            theta = 2.0 * math.pi * j / segs_loop
            x = cx + radius * math.sin(phi) * math.cos(theta)
            y = cy + radius * math.sin(phi) * math.sin(theta)
            z = cz + radius * math.cos(phi)
            bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()
    w = segs_loop + 1
    for i in range(segs_ring):
        for j in range(segs_loop):
            a = i * w + j
            b = i * w + j + 1
            c = (i + 1) * w + j + 1
            d = (i + 1) * w + j
            bm.faces.new([bm.verts[a], bm.verts[b], bm.verts[c], bm.verts[d]])
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()
    return obj


def _make_cylinder_floor(name, radius, half_len, cx, cy, cz, floor_z, floor_half):
    """Create a cylinder + floor collision object."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    verts = []
    rings = 20
    stacks = 10
    for s in range(stacks + 1):
        y = cy - half_len + 2.0 * half_len * s / stacks
        for r in range(rings + 1):
            theta = 2.0 * math.pi * r / rings
            verts.append((cx + radius * math.cos(theta), y, cz + radius * math.sin(theta)))
    fh = floor_half
    verts.append((-fh, -fh, floor_z))
    verts.append((fh, -fh, floor_z))
    verts.append((fh, fh, floor_z))
    verts.append((-fh, fh, floor_z))

    faces = []
    w = rings + 1
    for s in range(stacks):
        for r in range(rings):
            a = s * w + r
            b = s * w + r + 1
            c = (s + 1) * w + r + 1
            d = (s + 1) * w + r
            faces.append((a, c, b))
            faces.append((b, c, d))
    body_verts = len(verts) - 4
    f0, f1, f2, f3 = body_verts, body_verts + 1, body_verts + 2, body_verts + 3
    faces.append((f0, f1, f2))
    faces.append((f0, f2, f3))

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    return obj


def _make_cushion_mesh(name, nx, ny, half_size, init_z, sep, dome_height):
    """Create a two-sheet cushion mesh."""
    mesh_data = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    sx = nx + 1
    sy = ny + 1
    verts = []
    for sh in range(2):
        for row in range(sy):
            for col in range(sx):
                x = -half_size + 2.0 * half_size * col / nx
                y = -half_size + 2.0 * half_size * row / ny
                pu = col / nx
                pv = row / ny
                dome = math.sin(math.pi * pu) * math.sin(math.pi * pv) * dome_height
                z = (init_z - sep * 0.5 - dome) if sh == 0 else (init_z + sep * 0.5 + dome)
                verts.append((x, y, z))

    faces = []
    for sh in range(2):
        base = sh * sx * sy
        for row in range(ny):
            for col in range(nx):
                i = base + row * sx + col
                faces.append((i, i + 1, i + sx + 1, i + sx))

    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    return obj


def _setup_cloth(obj, solver='XPBD', material='COTTON'):
    """Enable GPUCloth on object with given solver and material preset."""
    obj.GPUCloth.is_active = True
    obj.GPUCloth.solver_type = solver
    obj.GPUCloth.material_preset = material


class GPUCloth_TestDrapeOnSphere(bpy.types.Operator):
    """Create DrapeOnSphere test scene: cloth pinned at top, draped over sphere"""
    bl_idname = "gpucloth.test_drape_on_sphere"
    bl_label = "Drape On Sphere"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        cloth_obj, _ = _make_grid_mesh("DrapeCloth", 64, 64, 3.0, 4.0)
        sphere_obj = _make_uv_sphere("CollisionSphere", 1.8, 0.0, 0.0, 0.5)

        col_mod = sphere_obj.modifiers.new(name="Collision", type='COLLISION')

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_cloth(cloth_obj, solver='XPBD', material='COTTON')

        context.scene.gpu_cloth_helper.gravity_z = -9.81

        self.report({'INFO'}, "DrapeOnSphere test scene created")
        return {'FINISHED'}


class GPUCloth_TestTwist(bpy.types.Operator):
    """Create TwistTest scene: cloth pinned at top corners"""
    bl_idname = "gpucloth.test_twist"
    bl_label = "Twist Test"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        cloth_obj, _ = _make_grid_mesh("TwistCloth", 64, 64, 3.0, 0.0)

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_cloth(cloth_obj, solver='XPBD', material='COTTON')

        context.scene.gpu_cloth_helper.gravity_x = 0.0
        context.scene.gpu_cloth_helper.gravity_y = 0.0
        context.scene.gpu_cloth_helper.gravity_z = 0.0

        self.report({'INFO'}, "TwistTest scene created")
        return {'FINISHED'}


class GPUCloth_TestMultiLayerDrop(bpy.types.Operator):
    """Create MultiLayerDrop scene: multiple cloth layers falling on cylinder"""
    bl_idname = "gpucloth.test_multi_layer_drop"
    bl_label = "Multi Layer Drop"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        cloth_obj, _ = _make_grid_mesh("MultiLayerCloth", 64, 64, 3.0, 4.0)
        collision_obj = _make_cylinder_floor(
            "CollisionCylinder",
            1.5, 4.5, 0.0, 0.0, 0.3, -2.0, 12.0)

        col_mod = collision_obj.modifiers.new(name="Collision", type='COLLISION')

        bpy.context.view_layer.objects.active = cloth_obj
        cloth_obj.select_set(True)

        _setup_cloth(cloth_obj, solver='OGC', material='COTTON')
        cloth_obj.GPUCloth.use_self_collision = True
        cloth_obj.GPUCloth.ogc_radius = 150.0
        cloth_obj.GPUCloth.ogc_friction = 0.3

        context.scene.gpu_cloth_helper.gravity_z = -9.81

        self.report({'INFO'}, "MultiLayerDrop test scene created")
        return {'FINISHED'}


class GPUCloth_TestCushionDrop(bpy.types.Operator):
    """Create CushionDrop scene: two-layer cushion falling on floor"""
    bl_idname = "gpucloth.test_cushion_drop"
    bl_label = "Cushion Drop"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        init_z = -2.0 + 3.0 * 0.25 + 0.02 * 0.5 + 2.0
        cushion_obj = _make_cushion_mesh(
            "CushionCloth", 32, 32, 3.0, init_z, 0.02, 3.0 * 0.25)

        floor_mesh = bpy.data.meshes.new("Floor_mesh")
        floor_obj = bpy.data.objects.new("Floor", floor_mesh)
        bpy.context.collection.objects.link(floor_obj)
        fh = 12.0
        fz = -2.0
        floor_verts = [(-fh, -fh, fz), (fh, -fh, fz), (fh, fh, fz), (-fh, fh, fz)]
        floor_faces = [(0, 1, 2, 3)]
        floor_mesh.from_pydata(floor_verts, [], floor_faces)
        floor_mesh.update()
        col_mod = floor_obj.modifiers.new(name="Collision", type='COLLISION')

        bpy.context.view_layer.objects.active = cushion_obj
        cushion_obj.select_set(True)

        _setup_cloth(cushion_obj, solver='XPBD', material='COTTON')

        context.scene.gpu_cloth_helper.gravity_z = -9.81

        self.report({'INFO'}, "CushionDrop test scene created")
        return {'FINISHED'}


class GPUCloth_TestOGCBounds(bpy.types.Operator):
    """Create OGC Bounds test scene: two cloth sheets with self-collision and contact-bounds visualisation"""
    bl_idname = "gpucloth.test_ogc_bounds"
    bl_label  = "OGC Bounds Viz"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')

        # Two horizontal cloth sheets stacked 60 mm apart so they collide
        upper, _ = _make_grid_mesh("OGCCloth_Upper", 32, 32, 1.5,  0.06)
        lower, _ = _make_grid_mesh("OGCCloth_Lower", 32, 32, 1.5, -0.06)

        # Activate OGC on the upper sheet, show bounds immediately
        bpy.context.view_layer.objects.active = upper
        upper.select_set(True)
        _setup_cloth(upper, solver='OGC', material='COTTON')
        upper.GPUCloth.use_self_collision = True
        upper.GPUCloth.ogc_radius         = 150.0
        upper.GPUCloth.ogc_friction       = 0.3
        upper.GPUCloth.show_ogc_bounds    = True

        # Activate OGC on the lower sheet as well
        lower.select_set(True)
        bpy.context.view_layer.objects.active = lower
        _setup_cloth(lower, solver='OGC', material='COTTON')
        lower.GPUCloth.use_self_collision = True
        lower.GPUCloth.ogc_radius         = 150.0
        lower.GPUCloth.ogc_friction       = 0.3
        lower.GPUCloth.show_ogc_bounds    = True

        context.scene.gpu_cloth_helper.gravity_z = -9.81

        bpy.context.view_layer.objects.active = upper
        self.report({'INFO'}, "OGC Bounds Viz scene created")
        return {'FINISHED'}


# ===========================================================================
#  OGC contact-bounds visualiser (SpaceView3D draw callback)
# ===========================================================================

_ogc_draw_handle = None


def _ogc_bounds_draw():
    """
    Draw two axis-aligned circles (XY, XZ) of radius=ogc_radius around every
    cloth vertex to visualise the OGC contact-offset sphere.

    Called by Blender for every viewport redraw; skips silently when no cloth
    object has show_ogc_bounds=True or self-collision is disabled.
    """
    try:
        import gpu
        from gpu_extras.batch import batch_for_shader
    except ImportError:
        return

    for cloth_obj in g_clothOBJs:
        if cloth_obj is None:
            continue
        s = getattr(cloth_obj, 'GPUCloth', None)
        if s is None or not s.show_ogc_bounds or not s.use_self_collision:
            continue

        radius = s.ogc_radius * 0.001  # mm → m
        mesh   = cloth_obj.data
        nv     = len(mesh.vertices)
        if nv == 0:
            continue

        # Fast bulk position readback (avoids per-vertex Python overhead)
        co = np.empty(nv * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', co)
        co = co.reshape(nv, 3)  # (nv, 3)

        SEGS = 16
        a    = np.linspace(0.0, 2.0 * math.pi, SEGS, endpoint=False, dtype=np.float32)
        ca   = np.cos(a)  # (SEGS,)
        sa   = np.sin(a)

        i0 = np.arange(SEGS)
        i1 = (i0 + 1) % SEGS

        # ── XY circle  (cx + r·cos, cy + r·sin, cz) ──────────────────────
        xy_x = co[:, 0:1]  # (nv,1)
        xy_y = co[:, 1:2]
        xy_z = np.repeat(co[:, 2:3], SEGS, axis=1)  # (nv, SEGS)

        p0_xy = np.stack([xy_x + radius * ca[i0],
                           xy_y + radius * sa[i0],
                           xy_z], axis=-1)  # (nv, SEGS, 3)
        p1_xy = np.stack([xy_x + radius * ca[i1],
                           xy_y + radius * sa[i1],
                           xy_z], axis=-1)

        # ── XZ circle  (cx + r·cos, cy, cz + r·sin) ──────────────────────
        xz_x = co[:, 0:1]
        xz_y = np.repeat(co[:, 1:2], SEGS, axis=1)  # (nv, SEGS)
        xz_z = co[:, 2:3]

        p0_xz = np.stack([xz_x + radius * ca[i0],
                           xz_y,
                           xz_z + radius * sa[i0]], axis=-1)
        p1_xz = np.stack([xz_x + radius * ca[i1],
                           xz_y,
                           xz_z + radius * sa[i1]], axis=-1)

        # Interleave p0/p1 into line-segment pairs: (nv, SEGS, 2, 3)→(N, 3)
        pts_xy = np.stack([p0_xy, p1_xy], axis=2).reshape(-1, 3)
        pts_xz = np.stack([p0_xz, p1_xz], axis=2).reshape(-1, 3)
        pts    = np.concatenate([pts_xy, pts_xz], axis=0)  # (nv*SEGS*4, 3)

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch  = batch_for_shader(shader, 'LINES', {"pos": pts})

        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(1.0)
        shader.bind()
        shader.uniform_float("color", (0.15, 0.90, 0.35, 0.40))
        batch.draw(shader)
        gpu.state.blend_set('NONE')


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
    GPUCloth_TestDrapeOnSphere,
    GPUCloth_TestTwist,
    GPUCloth_TestMultiLayerDrop,
    GPUCloth_TestCushionDrop,
    GPUCloth_TestOGCBounds,
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

    # OGC contact-bounds visualiser — register once, draw callback checks flag
    global _ogc_draw_handle
    if _ogc_draw_handle is None:
        _ogc_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _ogc_bounds_draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _ogc_draw_handle
    if _ogc_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_ogc_draw_handle, 'WINDOW')
        _ogc_draw_handle = None

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
