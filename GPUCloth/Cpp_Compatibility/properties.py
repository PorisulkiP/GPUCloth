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
import sys
import subprocess
import numpy as np

import time

from ctypes import *
from . import cpp_types as CType
from . import pycloth as lib
from . import tests
from ..utils import version_compatibility_utils as vcu

debug = False
if sys.gettrace() is not None:
    debug = True

g_dll = None
g_scene = None
g_obj = []
g_clmd = []
g_mesh = []
g_clothOBJs = []
g_clothCollisionOBJs = []

def free_gpu_memory(context=None):
    global g_dll
    global g_scene
    global g_obj
    global g_mesh
    global g_clmd
    global g_clothOBJs
    global g_clothCollisionOBJs

    # Вызываем FreeSolverData() из g_dll
    if g_dll is not None:
        try:
            g_dll.FreeSolverData()
        except Exception as e:
            print(f"Failed to free solver data: {e}")
            return False

    # Очищаем глобальные списки
    g_scene = None
    g_obj = []
    g_clmd = []
    g_mesh = []
    g_clothOBJs = []
    g_clothCollisionOBJs = []

    # Сбрасываем флаг и счетчики, если к ним есть доступ
    if context is not None and hasattr(context.scene, 'gpu_cloth_springs_built'):
        context.scene.gpu_cloth_springs_built = False

    return True

class GPUCloth_FreeVRAM(bpy.types.Operator):
    """Освободить память GPU от данных"""
    bl_idname = "gpucloth.destroy_simulation_data"
    bl_label = "Free GPU memory from data"

    @classmethod
    def poll(cls, context):
        return True
        
    def execute(self, context):
        success = free_gpu_memory(context)
        if not success:
            self.report({'ERROR'}, "Failed to free GPU memory.")
            return {'CANCELLED'}
        self.report({'INFO'}, "GPU memory successfully freed.")
        return {'FINISHED'}

class GPUCloth_LoadDLL(bpy.types.Operator):
    """Загрузить DLL для GPUCloth"""
    bl_idname = "gpucloth.load_dll"
    bl_label = "Load GPUCloth DLL"

    def check_cuda_support(self):
        try:
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            if "CUDA Version" in result.stdout:
                return True
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, "nvidia-smi execution failed:", e)
        except FileNotFoundError:
            self.report({'ERROR'}, "nvidia-smi is not found. Ensure that NVIDIA drivers are installed and path is configured.")
        
        return False    

    def load_dll(self):
        '''
        Здась происходит загрузка DLL файла сгенерированного из .cu/.cuh файла

        bpy.data.filepath = Blend file path location
        bpy.utils.user_resource = Addons installation path location (APPDATA/Blender/.../scripts/addons/)
        '''
        global g_dll  # Используйте глобальную переменную для хранения экземпляра DLL
        if g_dll is None:
            if not self.check_cuda_support():
                return False

            # addons_path = vcu.get_script_paths_pref()
            # filename = addons_path + "\\engine\\build\\GPUCloth.dll"
            filename = "E:\\source\\repos\\CUDACloth\\build\\GPUCloth.dll"
            try:
                g_dll = cdll.LoadLibrary(filename)
                self.report({'INFO'}, "The DLL has been loaded successfully")   

                # Заполнение данных для вычислителя
                g_dll.FillSolverData.argtypes = [POINTER(CType.Scene)]
                g_dll.FillSolverData.restype = c_bool

                # Очистка данных на GPU из вычислителя
                g_dll.FreeSolverData.argtypes = []
                g_dll.FreeSolverData.restype = c_bool

                # Создание пружин ткани
                g_dll.BuildClothSprings.argtypes = [POINTER(CType.ClothModifierData), POINTER(CType.Mesh)]
                g_dll.BuildClothSprings.restype = c_bool

                # Вычисления для одного кадра
                g_dll.SIM_solver.argtypes = []
                g_dll.SIM_solver.restype = c_bool

                # Добавление одного объекта ткани в массив
                g_dll.AddCloth.argtypes = [POINTER(CType.ClothModifierData), POINTER(CType.Mesh), POINTER(CType.Object)]
                g_dll.AddCloth.restype = c_bool

                # Удаление одного объекта ткани из массива
                g_dll.RemoveCloth.argtypes = [POINTER(CType.ClothModifierData), POINTER(CType.Mesh), POINTER(CType.Object)]
                g_dll.RemoveCloth.restype = c_bool

                # Добавление одного объекта столкновения в массив
                g_dll.AddCollisionObject.argtypes = [POINTER(CType.Object)]
                g_dll.AddCollisionObject.restype = c_bool

                # Удаление одного объекта столкновения из массива
                g_dll.RemoveCollisionObject.argtypes = [POINTER(CType.Object)]
                g_dll.RemoveCollisionObject.restype = c_bool

                # Обновление данных сцены
                g_dll.UpdateScene.argtypes = [POINTER(CType.Scene)]
                g_dll.UpdateScene.restype = c_bool

            except OSError as e:
                self.report({'ERROR'}, e.strerror)
                self.report({'ERROR'}, "\nНе удаётся установить соединение с DLL файлом")
                self.report({'ERROR'}, "Unable to set up a connection with the DLL file\n")
                g_dll = None
            except AttributeError:
                self.report({'ERROR'}, "\nНе удаётся вызвать функцию из DLL файла")
                self.report({'ERROR'}, "Unable to call function from DLL file\n")
                g_dll = None
        return False if g_dll is None else True

    def execute(self, context):
        if not self.load_dll():
            self.report({'ERROR'}, "Не удалось загрузить DLL")
            return {'CANCELLED'}
        return {'FINISHED'}

class GPUCloth_UnloadDLL(bpy.types.Operator):
    """Выгрузить dll файл из blender"""
    bl_idname = "gpucloth.unload_dll"
    bl_label = "To download a dll file from blender"

    @classmethod
    def poll(cls, context):
        return g_dll is not None

    def execute(self, context):
        global g_dll
        if g_dll is not None:
            try:
                # Получаем дескриптор DLL
                handle = c_void_p(g_dll._handle)
                # Освобождаем DLL (для Windows)
                result = windll.kernel32.FreeLibrary(handle)
                if result == 0:
                    raise ctypes.WinError()
                self.report({'INFO'}, "The DLL has been successfully unloaded.")
            except Exception as e:
                self.report({'ERROR'}, f"Error when unloading DLL: {e}")
                return {'CANCELLED'}
            finally:
                g_dll = None
        else:
            self.report({'WARNING'}, "The DLL is not loaded.")
            return {'CANCELLED'}
        return {'FINISHED'}

class GPUCloth_PrepareSimulation(bpy.types.Operator):
    """Подготовить данные для симуляции GPUCloth"""
    bl_idname = "gpucloth.prepare_simulation"
    bl_label = "Prepare GPUCloth Simulation"

    @classmethod
    def poll(cls, context):
        return True
        
    def fill_MVertTri_from_Object(self, obj:bpy.types.Object):
        # Убедимся, что объект - это меш
        if obj.type != 'MESH':
            return None
            
        # Получаем данные меша
        mesh = obj.data
        
        # Создаем экземпляры MVertTri для каждого треугольника в меше
        mvert_tris = (CType.MVertTri * len(mesh.loop_triangles))()
        
        for i, tri in enumerate(mesh.loop_triangles):
            # print(f"tri.vertices[{i}] = {tri.vertices[0]}, {tri.vertices[1]}, {tri.vertices[2]}")
            # Заполняем данные о треугольнике
            mvert_tris[i].tri[0] = tri.vertices[0]
            mvert_tris[i].tri[1] = tri.vertices[1]
            mvert_tris[i].tri[2] = tri.vertices[2]

        return mvert_tris

    def fill_Scene(self, context):
        scene = context.scene

        global g_scene

        if scene.rigidbody_world is None:
            bpy.ops.rigidbody.world_add()

        # Заполняем поля Scene
        g_scene = pointer(CType.Scene())
        g_scene.contents.flag = scene.rigidbody_world.enabled

        g_scene.contents.r = CType.RenderData(cfra=int(scene.frame_current), subframe=float(scene.frame_subframe),
                                framelen=float(scene.render.frame_map_old), frs_sec=c_short(scene.render.fps))
        g_scene.contents.physics_settings = CType.PhysicsSettings(gravity=(c_float*3)(*scene.gravity), flag=CType.PHYS_GLOBAL_GRAVITY)

        self.report({'INFO'}, "Scene filled in")

    def fill_Object(self, OBJ:bpy.types.Object) -> POINTER(CType.Object):
        new_object = CType.Object()

        # Поле ID генерируется в dll файле
        # # Заполняем поля объекта ID
        # new_object.id = tests.fill_ID(OBJ)

        # # Заполняем данные объекта
        # new_object.data = OBJ.data.as_pointer()

        # Заполняем матрицы объекта
        obmat = np.array(OBJ.matrix_world, dtype=np.float32)
        imat = np.array(OBJ.matrix_world.inverted(), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                new_object.obmat[i][j] = obmat[i][j]
                new_object.imat[i][j] = imat[i][j]

        # # Заполняем систему частиц объекта
        # new_object.particlesystem = tests.fill_ListBase(0)

        # Заполняем поля PartDeflect, если доступны
        new_object.pd = None
        # if hasattr(OBJ, 'pd'):
        #     new_object.pd = fill_PartDeflect(OBJ.pd)

        for modif in OBJ.modifiers:
            if modif.type == 'COLLISION':
                tmp_collision = CType.CollisionModifierData()
                    
                mvertType = CType.MVert * len(OBJ.data.vertices)
                mvert = mvertType()

                for i, mv in enumerate(OBJ.data.vertices):
                    v = vcu.element_multiply(OBJ.matrix_world, mv.co)
                    mvert[i].co = (c_float * 3)(*v)
                    mvert[i].flag = 0

                mvert_xnew = mvert
                mvert_xold = mvert
                mvert_current_xnew = mvert
                mvert_current_x = mvert
                mvert_current_v = mvert
                
                mvert_tri = self.fill_MVertTri_from_Object(OBJ)

                tmp_collision.x = cast(mvert, POINTER(CType.MVert))
                tmp_collision.xnew = cast(mvert_xnew, POINTER(CType.MVert))
                tmp_collision.xold = cast(mvert_xold, POINTER(CType.MVert))
                tmp_collision.current_xnew = cast(mvert_current_xnew, POINTER(CType.MVert))
                tmp_collision.current_x = cast(mvert_current_x, POINTER(CType.MVert))
                tmp_collision.current_v = cast(mvert_current_v, POINTER(CType.MVert))
                tmp_collision.tri = cast(mvert_tri, POINTER(CType.MVertTri))
                tmp_collision.mvert_num = len(OBJ.data.vertices)
                tmp_collision.tri_num = len(OBJ.data.loop_triangles)
                tmp_collision.time_x = -1000
                tmp_collision.time_xnew = -1000
                tmp_collision.is_static = True
                tmp_collision.bvhtree = None
                new_object.modifiers = pointer(tmp_collision)

        return pointer(new_object)

    def fill_PartDeflect(self, context, field) -> POINTER(CType.PartDeflect):
        # Заполняем поля PartDeflect на основе объекта field
        pd = CType.PartDeflect()
        if field.type == 'NONE':
            pd.flag = 0
        elif field.type == 'BOID':
            pd.flag = int(field.strength)
        elif field.type == 'CHARGE':
            pd.flag = int(field.strength)
        elif field.type == 'GUIDE':
            pd.flag = int(field.guide_clump_amount)
        elif field.type == 'DRAG':
            pd.flag = int(field.linear_drag)
        elif field.type == 'FLUID_FLOW':
            pd.flag = int(field.flow_rate)
        elif field.type == 'FORCE':
            pd.flag = int(field.strength)
        elif field.type == 'HARMONIC':
            pd.flag = int(field.harmonic_damping)
        elif field.type == 'LENNARDJ':
            pd.flag = int(field.strength)
        elif field.type == 'MAGNET':
            pd.flag = int(field.strength)
        elif field.type == 'TEXTURE':
            pd.flag = int(field.texture_mode)
        elif field.type == 'TURBULENCE':
            pd.flag = int(field.strength)
        elif field.type == 'VORTEX':
            pd.flag = int(field.strength)
        elif field.type == 'WIND':
            pd.flag = int(field.strength)

        pd.deflect = 0
        if field.falloff_type == 'SPHERE':
            pd.falloff = 0
        elif field.falloff_type == 'CONE':
            pd.falloff = 1
        elif field.falloff_type == 'TUBE':
            pd.falloff = 2

        if field.shape == 'POINT':
            pd.shape = 0
        elif field.shape == 'PLANE':
            pd.shape = 1
        elif field.shape == 'SURFACE':
            pd.shape = 2
        elif field.shape == 'POINTS':
            pd.shape = 3
        elif field.shape == 'LINE':
            pd.shape = 4

        # pd.tex_mode = field.texture_mode

        if field.guide_kink_type == 'NONE':
            pd.kink = 0
        elif field.guide_kink_type == 'CURL':
            pd.kink = 1
        elif field.guide_kink_type == 'RADIAL':
            pd.kink = 2
        elif field.guide_kink_type == 'WAVE':
            pd.kink = 3
        elif field.guide_kink_type == 'BRAID':
            pd.kink = 4
        elif field.guide_kink_type == 'ROTATION':
            pd.kink = 5
        elif field.guide_kink_type == 'ROLL':
            pd.kink = 6

        if field.guide_kink_axis == 'X':
            pd.kink_axis = 0
        elif field.guide_kink_axis == 'Y':
            pd.kink_axis = 1
        elif field.guide_kink_axis == 'Z':
            pd.kink_axis = 2

        if field.z_direction == 'BOTH':
            pd.z_direction = 0
        elif field.z_direction == 'POSITIVE':
            pd.z_direction = 1
        elif field.z_direction == 'NEGATIVE':
            pd.z_direction = 2

        pd.f_strength = field.strength
        pd.f_damp = field.harmonic_damping
        pd.f_flow = field.flow
        pd.f_wind_factor = field.wind_factor
        pd.f_size = field.size
        pd.f_power = field.falloff_power
        pd.f_power_r = field.radial_falloff
        pd.f_noise = field.noise
        pd.f_source = fill_Object(CType.Object()) if field.source_object else None
        pd.maxdist = field.distance_max
        pd.mindist = field.distance_min
        pd.maxrad = field.radial_max
        pd.minrad = field.radial_min
        pd.pdef_damp = 0
        pd.pdef_rdamp = 0
        pd.pdef_perm = 0
        pd.pdef_frict = 0
        pd.pdef_rfrict = 0
        pd.pdef_cfrict = 0
        pd.pdef_stickness = 0
        pd.absorption = field.use_absorption
        pd.pdef_sbdamp = 0
        pd.pdef_sbift = 0
        pd.pdef_sboft = 0
        pd.clump_fac = 0
        pd.clump_pow = 0
        pd.kink_freq = field.guide_kink_frequency
        pd.kink_shape = field.guide_kink_shape
        pd.kink_amp = field.guide_kink_amplitude
        pd.free_end = 0
        pd.rng = CType.fill_RandomNumberGenerator(field)
        pd.seed = field.seed
        
        return pointer(pd)

    def setMesh(self, context, obj) -> POINTER(CType.Mesh):
        if not obj:
            raise ValueError("Object should not be None")

        try:
            mesh = CType.Mesh()
            mesh.totedge = len(obj.edges)
            mesh.totvert = len(obj.vertices)
            mesh.totpoly = len(obj.polygons)
            mesh.totloop = len(obj.loops)

            mvertType = CType.MVert * len(obj.vertices)
            mvert = mvertType()

            for i, v in enumerate(obj.vertices):
                mvert[i].co = (c_float * 3)(*v.co)
                mvert[i].flag = 0

            mesh.mvert = cast(mvert, POINTER(CType.MVert))

            medgeType = CType.MEdge * len(obj.edges)
            medge = medgeType()
            for i in obj.edges:
                medge[i.index].v1 = i.vertices[0]
                medge[i.index].v2 = i.vertices[1]
                medge[i.index].crease = 0 #int(i.crease)
                medge[i.index].bweight = 0 #int(i.bevel_weight)
                medge[i.index].flag = 35 # 
        
            mesh.medge = cast(medge, POINTER(CType.MEdge))

            mpolyType = CType.MPoly * len(obj.polygons)
            mpoly = mpolyType()
            for i in obj.polygons:
                mpoly[i.index].loopstart = i.loop_start
                mpoly[i.index].totloop = i.loop_total
                # mpoly[i.index].mat_nr = i.material_index
                # mpoly[i.index].flag = 0
                # if (i.use_smooth):
                #     mpoly[i.index].flag = 1
                
            mesh.mpoly = cast(mpoly, POINTER(CType.MPoly))

            mloopType = CType.MLoop * len(obj.loops)
            mloop = mloopType()
            loop_idx = 0
            for loop in obj.loops:
                mloop[loop_idx].v = loop.vertex_index
                mloop[loop_idx].e = loop.edge_index
                loop_idx += 1
                
            mesh.mloop = cast(mloop, POINTER(CType.MLoop))

            return pointer(mesh)
        except Exception as e:
            raise RuntimeError("An error occurred while setting the mesh: {}".format(e))

    def setClothModifierData(self, context, OBJ) -> POINTER(CType.ClothModifierData):
        clmd = CType.ClothModifierData()

        # Заполняем параметры
        # Часть из них стандартная. см. _DNA_DEFAULT_ClothSimSettings
        sim_parms = CType.ClothSimSettings()
        sim_parms.mingoal = 0
        sim_parms.Cvi = 1.0
        sim_parms.Cdis = 1.0
        sim_parms.gravity[0] = context.scene.gpu_cloth_helper.gravity_x
        sim_parms.gravity[1] = context.scene.gpu_cloth_helper.gravity_y
        sim_parms.gravity[2] = context.scene.gpu_cloth_helper.gravity_z
        sim_parms.mass = OBJ.GPUCloth.vertex_mass # cloth_settings.mass
        sim_parms.structural = 0 # cloth_settings.tension_stiffness
        sim_parms.shear = 5.0 # cloth_settings.shear_stiffness
        sim_parms.bending = 0.5 
        sim_parms.vgroup_mass = 0 # OBJ.vertex_groups.find(cloth_settings.vertex_group_mass)
        sim_parms.stepsPerFrame = OBJ.GPUCloth.quality_step # cloth_settings.quality
        sim_parms.maxgoal = 1.0 # cloth_settings.goal_max
        sim_parms.velocity_smooth = 0.0 # cloth_settings.air_damping
        sim_parms.collider_friction = 0.0 #  cloth_settings.collider_friction
        sim_parms.shrink_min = 0.0 # cloth_settings.shrink_min
        sim_parms.shrink_max = 0.0 # cloth_settings.shrink_max
        sim_parms.vgroup_bend = 0 # OBJ.vertex_groups.find(cloth_settings.vertex_group_bending)
        sim_parms.vgroup_struct = 0 # OBJ.vertex_groups.find(cloth_settings.vertex_group_structural_stiffness)
        sim_parms.vgroup_shear = 0 # OBJ.vertex_groups.find(cloth_settings.vertex_group_shear_stiffness)
        sim_parms.vgroup_shrink = 0 # OBJ.vertex_groups.find(cloth_settings.vertex_group_shrink)
        sim_parms.bending_damping = 0.5 # cloth_settings.bending_damping
        sim_parms.voxel_cell_size = 0.1 # cloth_settings.voxel_cell_size
        sim_parms.tension = 15.0 # cloth_settings.tension_stiffness
        sim_parms.compression = 15.0 # cloth_settings.compression_stiffness
        sim_parms.tension_damp = 5.0 # cloth_settings.tension_damping
        sim_parms.compression_damp = 5.0 # cloth_settings.compression_damping
        sim_parms.shear_damp = 5.0 # cloth_settings.shear_damping
        sim_parms.internal_spring_max_length =  10 # cloth_settings.internal_spring_max_length
        sim_parms.internal_spring_max_diversion = 0.78539816339744830962 # pi/4  #cloth_settings.internal_spring_max_diversion
        sim_parms.vgroup_intern = 0 # OBJ.vertex_groups.find(cloth_settings.vertex_group_intern)
        sim_parms.internal_tension = 15.0 # cloth_settings.internal_tension_stiffness
        sim_parms.internal_compression = 15.0 # cloth_settings.internal_compression_stiffness
        sim_parms.max_internal_tension = 15.0 # cloth_settings.internal_tension_stiffness_max
        sim_parms.max_internal_compression = 15.0 # cloth_settings.internal_compression_stiffness_max
        sim_parms.eff_force_scale = 1000.0
        sim_parms.eff_wind_scale = 250.0
        sim_parms.effector_weights = None
        sim_parms.reset = 0
        sim_parms.presets = 2
        sim_parms.shapekey_rest = 0
        sim_parms.fluid_density = 0.0
        sim_parms.pressure_factor = 1.0
        sim_parms.target_volume = 0.0
        sim_parms.uniform_pressure_force = 0.0
        sim_parms.time_scale = OBJ.GPUCloth.speed_multiplier
        sim_parms.timescale = 1.0
        sim_parms.dt = 1 #0.01
        sim_parms.avg_spring_len = 0.0
        sim_parms.goalfrict = 0.0
        sim_parms.goalspring = 1.0
        sim_parms.flags = CType.CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS_NORMAL
        sim_parms.bending_model = CType.CLOTH_BENDING_ANGULAR if OBJ.GPUCloth.bending_model == 'ANGULAR' else CType.CLOTH_BENDING_LINEAR

        clmd.sim_parms = pointer(sim_parms)

        coll_parms = pointer(CType.ClothCollSettings())
        clmd.clothObject = None

        # Заполняем поля coll_parms
        coll_parms.contents.epsilon = 0.015
        coll_parms.contents.self_friction = 5.0
        coll_parms.contents.friction = 5.0
        coll_parms.contents.damping = 0.0
        coll_parms.contents.selfepsilon = 0.015
        coll_parms.contents.loop_count = 2
        coll_parms.contents.group = None
        coll_parms.contents.vgroup_selfcol = 0
        coll_parms.contents.vgroup_objcol = 0
        coll_parms.contents.clamp = 0.0
        coll_parms.contents.self_clamp = 0.0
        coll_parms.contents.flags = CType.CLOTH_COLLSETTINGS_FLAG_ENABLED

        clmd.coll_parms = coll_parms

        # clmd.point_cache = PointCache()
        # clmd.ptcaches = ListBase()
        solver_result = CType.ClothSolverResult()

        # Инициализация полей с некоторыми значениями по умолчанию
        solver_result.status = 0
        solver_result.max_iterations = 0
        solver_result.avg_iterations = 0
        solver_result.max_error = 0.0
        solver_result.min_error = 0.0
        solver_result.avg_error = 0.0

        # Получение указателя на объект ClothSolverResult
        clmd.solver_result = pointer(solver_result)

        return pointer(clmd)

    def execute(self, context):
        '''
        Здесь готовятся данные для симуляции.
        1.Проверяется загружена ли DLL
        2.Собираются объекты для симуляции
        3.Данные передаются в DLL файл
        '''
        global g_dll
        global g_scene
        global g_obj
        global g_mesh
        global g_clmd
        global g_clothOBJs
        global g_clothCollisionOBJs
        
        # Проверяется загружена ли DLL
        if g_dll is None: 
            bpy.ops.gpucloth.load_dll()
            if g_dll is None: 
                self.report({'ERROR'}, "Не удаётся загрузить dll.\n")
                print("Не удаётся загрузить dll.")
                return {'CANCELLED'}

        # Если пружины уже построены, очищаем данные и вызываем FreeSolverData()
        if context.scene.gpu_cloth_springs_built:
            success = free_gpu_memory(context)
            if not success:
                self.report({'ERROR'}, "Failed to free GPU memory.")
                return {'CANCELLED'}
        
        # Если файл не сохранён, то сохраняем во временную папку
        if (not bpy.data.is_saved):
            bpy.ops.wm.save_as_mainfile(filepath=bpy.app.tempdir + 'GPU_Cloth.blend', check_existing=False)
        
        # Если файл был изменён после сохранения, то просто сохраняем
        if (bpy.data.is_dirty):
            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath, check_existing=False)

        # we need to switch from Edit mode to Object mode so the selection gets updated
        mode = bpy.context.active_object.mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Собираются все объекты, которые ткань
        # All objects that are fabric are collected
        for item in bpy.context.scene.objects:
            if item is not None and hasattr(item, 'GPUCloth') and item.GPUCloth.is_active:
                g_clothOBJs.append(item)
                for modifier in item.modifiers:
                    if modifier.type == 'CLOTH':
                        modifier.show_viewport = False
                        modifier.show_render = False
            else: 
                # Собираются все объекты, которые взаимодействуют с тканью
                # All objects that interact with the fabric are collected
                for modif in item.modifiers:
                    if modif.type == 'COLLISION':
                        data_ptr = self.fill_Object(item)
                        if not data_ptr:
                            self.report({'ERROR'}, "Получен NULL указатель от fill_Object")
                            return {'CANCELLED'}
                        else:
                            g_clothCollisionOBJs.append(data_ptr)

        # ClothModifierData и Mesh заполняются данными
        # ClothModifierData and Mesh is filled with data
        for cloth_obj in g_clothOBJs:
            if cloth_obj is None or not hasattr(cloth_obj, 'data'):
                return {'CANCELLED'}
            
            data_ptr = self.setMesh(context, cloth_obj.data)
            if not data_ptr:
                self.report({'ERROR'}, "Получен NULL указатель от setMesh")
                return {'CANCELLED'}
            else:
                g_mesh.append(data_ptr)

            data_ptr = self.setClothModifierData(context, cloth_obj)
            if not data_ptr:
                self.report({'ERROR'}, "Получен NULL указатель от setClothModifierData")
                return {'CANCELLED'}
            else:
                g_clmd.append(data_ptr)
            
            data_ptr = self.fill_Object(cloth_obj)
            if not data_ptr:
                self.report({'ERROR'}, "Получен NULL указатель от fill_Object")
                return {'CANCELLED'}
            else:
                g_obj.append(data_ptr)

        # Создаём связи для симуляции
        # Creating connections for simulation
        if (len(g_clothOBJs) != len(g_clmd)):
            self.report({'ERROR'}, "Что-то не так собрали")
            print(f"Что-то не так собрали len(g_clothOBJs)={len(g_clothOBJs)}, len(g_clmd)={len(g_clmd)}")
            return {'CANCELLED'}

        self.fill_Scene(context)
        
        for i in range(0, len(g_clothCollisionOBJs)):
            try:                    
                # Если получилось, то добавляем объект на GPU
                if(not g_dll.AddCollisionObject(g_clothCollisionOBJs[i])):
                    self.report({'ERROR'}, "Failed on AddCollisionObject")
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError in AddCollisionObject.")
                return {'CANCELLED'}

        if (not g_dll.FillSolverData(g_scene)):
            self.report({'ERROR'}, "Failed to load scene data")
            print("Failed to load scene data from FillSolverData()")
            return {'CANCELLED'}

        for i in range(0, len(g_clothOBJs)):
            try:
                # Если нет объекта такни, то создаём его вместе с пружинами
                if (not g_clmd[i].contents.clothObject):
                    if (not g_dll.BuildClothSprings(g_clmd[i], g_mesh[i])):
                        self.report({'ERROR'}, "Failed to create cloth spring connections")
                        print("Failed to create cloth spring connections")
                        return {'CANCELLED'}

            except OSError:
                self.report({'ERROR'}, "Failed to create cloth spring connections.")
                print("Failed to create cloth spring connections.")
                return {'CANCELLED'}

            try:
                # Если не получилось, то останавливаем весь кардебалет
                if g_clmd[i].contents.clothObject is None:
                    print("clothObject is a NULL pointer.")
                    self.report({'ERROR'}, "ClothObject is a NULL pointer")
                    print("ClothObject is a NULL pointer")
                    bpy.ops.screen.animation_cancel()
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError: ClothObject is a NULL pointer.")
                print("OSError: ClothObject is a NULL pointer.")
                return {'CANCELLED'}
                    
            try:
                # Если получилось, то добавляем объект на GPU
                if(not g_dll.AddCloth(g_clmd[i], g_mesh[i], g_obj[i])):
                    self.report({'ERROR'}, "Failed on AddCloth")
                    print("Failed on AddCloth")
                    return {'CANCELLED'}
            except OSError:
                self.report({'ERROR'}, "OSError on AddCloth.")
                print("OSError on AddCloth.")
                return {'CANCELLED'}

        # back to whatever mode we were in
        bpy.ops.object.mode_set(mode=mode)

        # Сбрасываем флаг
        context.scene.gpu_cloth_springs_built = True

        return {'FINISHED'}

class GPUCloth_UpdateSimulation(bpy.types.Operator):
    """Обновить симуляцию GPUCloth"""
    bl_idname = "gpucloth.update_simulation"
    bl_label = "Update GPUCloth Simulation"

    def updateBlenderMesh(self, mesh_ptr: POINTER(CType.Mesh), blender_obj: bpy.types.Object):
        update_start_time = time.perf_counter_ns()
        if not blender_obj or not mesh_ptr :
            self.report({'ERROR'}, f"Neither the mesh pointer nor the Blender object should be None")
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}

        if not hasattr(blender_obj, 'data'):
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}
            
        mesh_data = mesh_ptr.contents
        try:
            if blender_obj.type != 'MESH':
                self.report({'ERROR'}, f"The Blender object should be of type 'MESH'")
                bpy.ops.screen.animation_cancel()
                return {'CANCELLED'}

            mesh = blender_obj.data
            
            if len(mesh.vertices) != mesh_data.totvert:
                self.report({'ERROR'}, f"Vertex count mismatch between Blender object and mesh data")
                bpy.ops.screen.animation_cancel()
                return {'CANCELLED'}

            for i in range(len(mesh.vertices)):
                mesh_vertex = mesh_data.mvert[i]
                mesh.vertices[i].co = (
                    mesh_vertex.co[0],
                    mesh_vertex.co[1],
                    mesh_vertex.co[2]
                )

            # mesh.update()
            # blender_obj.update_from_editmode()
            # blender_obj.update_tag()
            # bpy.context.view_layer.update()

        except ReferenceError as e:
            self.report({'ERROR'}, f"ReferenceError: {e}")
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}
            
        update_end_time = time.perf_counter_ns()
        update_elapsed_ms = (update_end_time - update_start_time) / 1_000_000  # Перевод из наносекунд в миллисекунды
        print(f"updateBlenderMesh для объекта {blender_obj.name} заняло {update_elapsed_ms} миллисекунд.")

    def validate_objects(self):
        for obj in g_clothOBJs:
            if obj is None or obj.name not in bpy.data.objects:
                self.report({'ERROR'}, f"Объект {obj} не существует в сцене.")
                return False
            if obj.type != 'MESH':
                self.report({'ERROR'}, f"Объект {obj.name} не является типом 'MESH'.")
                return False
        return True

    def execute(self, context):
        '''
        Здесь запускается симуляция.
        0. Подготовка данных
        1. Запуск просчёта
        2. Получение данных
        3. Установка новых координат
        '''
        global g_dll
        global g_obj
        global g_mesh
        global g_clmd
        global g_clothOBJs

        if g_dll is None or context.scene.frame_current < 2 or not self.validate_objects(): 
            return {'FINISHED'}

        total_start_time = time.perf_counter_ns()
        try:
            if (len(g_clothOBJs) == len(g_clmd) == len(g_obj) == len(g_mesh)):
                for i in range(len(g_clothOBJs)):
                    if (g_dll.SIM_solver()): # 
                        self.updateBlenderMesh(g_mesh[i], g_clothOBJs[i])
                    else:
                        self.report({'ERROR'}, f"Error in SIM_solver")  
                        bpy.ops.screen.animation_cancel()
                        return {'CANCELLED'}

                # # Обновляем  меш и перерисовываем сцену
                # try:
                #     for clothOBJ in g_clothOBJs:
                #         clothOBJ.update_tag()
                # except ReferenceError as e:
                #     self.report({'ERROR'}, f"ReferenceError: {e}")
                #     # Обновим тогда пружины, чтобы не было проблем в дальнейшем
                #     context.scene.gpu_cloth_springs_built = True
                #     bpy.ops.screen.animation_cancel()
                #     return {'CANCELLED'}
                # bpy.context.view_layer.update()
                # if debug:
                #     print("Frame was updated")
            else:
                self.report({'ERROR'}, f"len(g_clothOBJs): {len(g_clothOBJs)}")  
                self.report({'ERROR'}, f"len(g_clmd): {len(g_clmd)}")  
                self.report({'ERROR'}, f"len(g_obj): {len(g_obj)}")  
                self.report({'ERROR'}, f"len(g_mesh): {len(g_mesh)}")  
                bpy.ops.screen.animation_cancel()
                return {'CANCELLED'}
        except OSError as err:
            print(err)
            bpy.ops.screen.animation_cancel()
            return {'CANCELLED'}

        total_end_time = time.perf_counter_ns()
        total_elapsed_ms = (total_end_time - total_start_time) / 1_000_000  # Перевод из наносекунд в миллисекунды
        print(f"Выполнение оператора заняло {total_elapsed_ms} миллисекунд.")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(GPUCloth_LoadDLL)
    bpy.utils.register_class(GPUCloth_UnloadDLL)
    bpy.utils.register_class(GPUCloth_FreeVRAM)
    bpy.utils.register_class(GPUCloth_PrepareSimulation)
    bpy.utils.register_class(GPUCloth_UpdateSimulation)

    bpy.types.Scene.gpu_cloth_springs_built = bpy.props.BoolProperty(name="Cloth Springs Built", default=False)

    global g_dll
    global g_obj
    global g_mesh
    global g_clmd
    global g_scene
    global g_clothOBJs
    global g_clothCollisionOBJs

    g_dll = None
    g_scene = None
    g_obj = []
    g_clmd = []
    g_mesh = []
    g_clothOBJs = []
    g_clothCollisionOBJs = []

def unregister():
    bpy.utils.unregister_class(GPUCloth_LoadDLL)
    bpy.utils.unregister_class(GPUCloth_UnloadDLL)
    bpy.utils.unregister_class(GPUCloth_FreeVRAM)
    bpy.utils.unregister_class(GPUCloth_PrepareSimulation)
    bpy.utils.unregister_class(GPUCloth_UpdateSimulation)
    
    del bpy.types.Scene.gpu_cloth_springs_built

    global g_dll
    global g_obj
    global g_mesh
    global g_clmd
    global g_scene
    global g_clothOBJs
    global g_clothCollisionOBJs

    g_dll = None
    g_scene = None
    g_obj = []
    g_clmd = []
    g_mesh = []
    g_clothOBJs = []
    g_clothCollisionOBJs = []

