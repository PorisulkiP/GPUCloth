"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

В данном файле происходит запуск файлов создания сцены и запуска симуляции
"""
import numpy as np
import sys
import os
import _bpy as bpy

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

# Добавляем папку с проектом в поле зрения blender


def importForDebugg():
    dir = os.path.dirname(bpy.data.filepath)
    if not dir in sys.path:
        sys.path.append(dir)
        sys.path.append(dir + "\\python")
        sys.path.append(dir + "\\python" + "\\work version 0.0.1")
        # print(sys.path)


def loadDLL():
    """
    Здась происходит загрузка DLL файла сгенерированного из cuda файла
    """
    from ctypes import cdll
    try:
        dirname = os.path.dirname(__file__)
        # Для VS Code
        # filename = os.path.join((dirname[::-1][dirname[::-1].index("/"):][::-1]),
        #                         "С\\main\\x64\\Release\\main.dll")
        # Для Blender
        filename = os.path.join((dirname[::-1][dirname[::-1].index("\\"):][::-1]),
                                "С\\main\\x64\\Release\\main.dll")
        lib = cdll.LoadLibrary(filename)
        lib.print_info()
    except OSError:
        print("Не удаётся установить соединение с DLL файлом")


def isCUDAAvailable():
    """
    Пробуем импортировать PyCuda для проверки поддержки данной технологии
    Надо исправить, потому что у человека может быть просто не установлен пакет
    для питона
    """
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
    except ModuleNotFoundError:
        print("Технология CUDA не поддерживается на данном устройстве")


class Physics():
    def __init__(self, backUp):
        print("Запуск")
        # Создаём атрибуты плоскости
        self.mesh = bpy.data.objects["Plane"]
        self.mass = 0.3
        self.gravity = bpy.context.scene.gravity
        self.fps = bpy.context.scene.render.fps

        # Создаём атрибут бэкапа
        self.backUp = backUp

        # Собираем все объекты в сцене, способные к столкновению
        self.collision_objects = self.collisionOBJ()

        bpy.app.handlers.frame_change_pre.clear()
        bpy.app.handlers.frame_change_pre.append(self.physicsCalculation())

    def physicsCalculation(self):
        """
        Данная функция вызывается каждый кадр.
        запускает ещё несколько функция для расчёта физики

        mesh - объект, с симуляциет ткани

        collision_objects - объект(ы) столкновения

        backUp - бэкап ткани до симуляции
        """
        def gravityCalc(scene):
            """
            Функция вычисляет, насколько должна переместиться точка,
            при заданной массе, гравитации и ускорении
            Для вычисления используется формула уменьшенного веса.
            Р - вес / m - масса / g - гравитация
            Р = m(g-a)
            """
            flagNum = 0
            flags = []

            # i - одна вершина из всех, что есть в объекте, с симуляциет ткани
            for i in range(0, len(self.mesh.data.vertices)-1):

                # obj_vert - меш из списка объектов столкновений
                for obj_vert in self.collision_objects:

                    # о - одна вершина из всех, что есть в объекте столкновения
                    for o in range(0, len(obj_vert.data.vertices)-1):

                        # 1-я вершина из отрезка
                        mesh_local = self.mesh.data.vertices[i].co
                        collision_objects_local = obj_vert.data.vertices[o].co

                        # попытка обратиться к 2-ой вершине из отрезка
                        # работает если кол-во точек чётно
                        try:
                            mesh_local_second = self.mesh.data.vertices[i+1].co
                            collision_objects_local_second = obj_vert.data.vertices[o+1].co
                        except IndexError:
                            break

                        # глобализация координат первой вершины
                        mesh_global = self.mesh.matrix_world @ mesh_local
                        collision_objects_global = obj_vert.matrix_world @ collision_objects_local

                        # глобализация координат второй вершины
                        mesh_global_second = self.mesh.matrix_world @ mesh_local_second
                        collision_objects_global_second = obj_vert.matrix_world @ collision_objects_local_second

                        # [0] - x, [1] - y, [2] - z
                        flags.append(self.cross(mesh_global, collision_objects_global,
                                               [mesh_global[0] - mesh_global_second[0],
                                                mesh_global[1] - mesh_global_second[1],
                                                mesh_global[2] - mesh_global_second[2]],
                                               [collision_objects_global[0] - collision_objects_global_second[0],
                                                collision_objects_global[1] - collision_objects_global_second[1],
                                                collision_objects_global[2] - collision_objects_global_second[2]]))
                        # print("flags[flagNum] = ", flags[flagNum])
                        flagNum += 1

            if ([i for i in flags].count(True) == 0):
                for newCoor in self.mesh.data.vertices:
                    # print("mass = ", self.mass)
                    # print("gravity_x = ", self.gravity[0])
                    # print("gravity_y = ", self.gravity[1])
                    # print("gravity_z = ", self.gravity[2])
                    # print("acceleration_x = ",
                    #       self.acceleration(newCoor.co)[0])
                    # print("acceleration_y = ",
                    #       self.acceleration(newCoor.co)[1])
                    # print("acceleration_z = ",
                    #       self.acceleration(newCoor.co)[2])
                    # print("newCoor.co[2] = ", ((self.mass*self.gravity[2] - self.acceleration(newCoor.co)[2]) / self.fps))
                    newCoor.co[0] += ((self.mass * (self.gravity[0] -
                                                    self.acceleration(newCoor.co))) / self.fps)
                    newCoor.co[1] += ((self.mass * (self.gravity[1] -
                                                    self.acceleration(newCoor.co))) / self.fps)
                    newCoor.co[2] += ((self.mass * (self.gravity[2] -
                                                    self.acceleration(newCoor.co))) / self.fps)
            else:
                # if flag:
                #     for newCoor in self.mesh.data.vertices:
                #         newCoor.co[0] -= (bpy.context.scene.gravity[0] / bpy.context.scene.render.fps)
                #         newCoor.co[1] -= (bpy.context.scene.gravity[1] / bpy.context.scene.render.fps)
                #         newCoor.co[2] -= (bpy.context.scene.gravity[2] / bpy.context.scene.render.fps)
                # else:
                for newCoor in self.mesh.data.vertices:
                    newCoor.co[0] += 0
                    newCoor.co[1] += 0
                    newCoor.co[2] += 0
        return gravityCalc

    def lenOfTwoPoints(self, a, b, distance_min=0.15):
        """ 
        Находим расскояние между двух точек,
        через формулу определение длинны вектора
        """
        x = (abs(b[0]) - abs(a[0]))**2
        y = (abs(b[1]) - abs(a[1]))**2
        z = (abs(b[2]) - abs(a[2]))**2
        d = (x+y+z)**0.5

        return d

    def velocity(self, a):
        """
        Возвращает скорость точки, 
        используя её массу и гравитацию в сцене
        V = S/t
        S = a - a_second
        t = 1/fps
        """
        V = (self.lenOfTwoPoints(a, [(a[0] + (self.mass * self.gravity[0] / self.fps)),
                                     (a[1] + (self.mass * self.gravity[1] / self.fps)),
                                     (a[2] + (self.mass * self.gravity[2] / self.fps))]) / (1 / self.fps))
        # print("V = ", V)
        return V

    def acceleration(self, a):
        """
        Функция возвращает ускорение для одной точки,
        в виде массива по 3 коордиинатам
        a = (V1-V2)/t
        t = 1/fps
        """
        V1 = self.velocity(a)
        V2 = self.lenOfTwoPoints([(a[0] + (self.mass * self.gravity[0] / self.fps)),
                                  (a[1] + (self.mass * self.gravity[1] / self.fps)),
                                  (a[2] + (self.mass * self.gravity[2] / self.fps))],
                                 [(a[0] + (self.mass * self.gravity[0] / self.fps)**2),
                                  (a[1] + (self.mass * self.gravity[1] / self.fps)**2),
                                  (a[2] + (self.mass * self.gravity[2] / self.fps)**2)]) / (1 / self.fps)

        a = (V1 - V2) / (1 / self.fps)
        # print("a = ", a)
        return a

    def cross(self, a, b, c, d, distance_min=0.15):
        """
        Задача данной функции – определить пересечения векторов двух точек,
        т.е. найти пересечения вершин, чтобы геометрия не пересекалась друг с другом
        Пераметр distance_min задаётся в настройках ткани во вкладке Collision ==> Object collision 
        ==> Distanse и по умолчанию равен 0.15
        Находится по формуле вычисления точек пересечения прямых в пространстве.
        """
        # print("a = ", a)
        # print("b = ", b)
        # print("c = ", c)
        # print("d = ", d)

        # vector - направляющие векторы прямых segment
        vector = np.array([
                            [c[1], -c[0], 0],
                            [0, c[2], -c[0]],
                            [d[1], -d[0], 0],
                            [0, d[2], -d[0]]
                         ])
                         
        # segment - прямые
        segment = np.array([
                            c[1] * a[0] - c[0] * a[1],
                            c[2] * a[1] - c[1] * a[2],
                            d[1] * b[0] - d[0] * b[1],
                            d[2] * b[1] - d[1] * b[2]
                         ])

        # print("Запуск просчёта")
        # print("vector = ", vector)
        # print("segment = ", segment)

        lstsq = np.linalg.lstsq(vector, segment, rcond=None)
        answer = lstsq[0] # Решение задачи наименьших квадратов для системы уравнений 
        residuals = lstsq[1] # Суммы "расстояний" для каждого столбца b - a*x.
        rank = lstsq[2] # Ранг матрицы answer

        # print("cross = ", answer[0:3]) 
        # print("residuals = ", residuals) 
        # print("rank = ", rank) 

        if(answer.all()):            
            return False
        return True

    def backupGet(self, backUp):
        """
        Установка координат для точек из бэкапа
        """
        for newVert in self.mesh.data.vertices:
            for oldVert in backUp:
                newVert.co.x = oldVert[0]
                newVert.co.y = oldVert[1]
                newVert.co.z = oldVert[2]
                break

    def collisionOBJ(self):
        """
        Данная функция возвращает массив объектов с которыми пересекается ткань
        """
        # numOfOBJ - объекты сцены
        numOfOBJ = np.array([bpy.data.objects[element]
                             for element in range(0, len(bpy.data.objects))])

        # collisionOBJ - список объектов способных к столкновению
        collisionOBJ = []
        
        for col in numOfOBJ:            
            try:
                col.modifiers['Collision'].settings.use == True
                collisionOBJ.append(col)
            except KeyError:
                pass

        return collisionOBJ


def backupSet():
    """    Создание бэкапа     """
    
    mesh = bpy.data.objects["Plane"].data

    # список всех вершин изначального меша ткани
    backupVert = []
    for NumVert in range(0, len(mesh.vertices)):
        # print(mesh.vertices[NumVert].co[0:3])
        backupVert.append(mesh.vertices[NumVert].co)

    return backupVert


# В данном класе создаётся сцена для тестировния алгоритма физики ткани
class SetUp():
    def __init__(self):
        self.opening_scene()
        # self.pipInstall("numba") # установка внешних пакетов

    def opening_scene(self):
        """
        Создаёт сцену с кубом и плоскостью
        """
        # Пробуем взять данные о плоскости(объект)
        try:
            self.mesh = bpy.data.objects["Plane"]
            # KeyError это стандартное исключение, так что просто создаём всё,
            # что нам нужно для дальнейшей работы
        except NameError:
            raise KeyError
        except KeyError:
            print("Scene creating!")
            # Создаём плоскость
            bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False,
                                             align='WORLD', location=(0, 0, 1),
                                             scale=(1, 1, 1))

            # bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

            # Подразделяем для симуляции ткани
            # bpy.ops.object.subdivision_set(level=5, relative=False)

            # Изменяем подразделение на "простое"
            # bpy.context.object.modifiers["Subdivision"].subdivision_type = 'SIMPLE'

            # Применяем модификатор
            # bpy.ops.object.modifier_apply(modifier="Subdivision")

            # Сглаживаем плоскость
            # bpy.ops.object.shade_smooth()

            # Назначаем на плоскость физику ткани
            bpy.ops.object.modifier_add(type='CLOTH')
            bpy.context.object.modifiers["Cloth"].collision_settings.use_self_collision = True

            # И записываем в файл
            self.fill_val_cloth_seettings()

            # Затем убираем её
            bpy.ops.object.modifier_remove(modifier="Cloth")

            # Создаём куб на который будет падать ткань
            bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False,
                                            align='WORLD', location=(0, 0, 0), scale=(0.8, 0.8, 0.8))
            # Уменьшает куб
            bpy.ops.transform.resize(value=(0.5, 0.5, 0.5))

            # Назначаем его объектом столкновения
            bpy.ops.object.modifier_add(type='COLLISION')

            print("Scene created!")

    def fill_val_cloth_seettings(self):
        """
        Данная функция записывает значения параметров физики ткани в отдельный файл

        В будущем должна вызываться каждый раз, когда пользователь нажимает на 'Bake on GPU'
        """

        self.clothSettings_list = dir(
            bpy.data.objects["Plane"].modifiers['Cloth'].settings)  # список параметров
        self.clothSettings = bpy.data.objects["Plane"].modifiers['Cloth'].settings

        value = []
        value.append("air_damping = ")
        value.append(self.clothSettings.air_damping)

        value.append("bending damping = ")
        value.append(self.clothSettings.bending_damping)
        value.append("bending model = ")
        value.append(self.clothSettings.bending_model)
        value.append("bending stiffness = ")
        value.append(self.clothSettings.bending_stiffness)
        value.append("bending stiffness max = ")
        value.append(self.clothSettings.bending_stiffness_max)

        value.append("collider friction = ")
        value.append(self.clothSettings.collider_friction)

        value.append("density target = ")
        value.append(self.clothSettings.compression_damping)

        value.append("fluid_density = ")
        value.append(self.clothSettings.fluid_density)

        value.append("goal_default = ")
        value.append(self.clothSettings.goal_default)

        value.append("goal_friction = ")
        value.append(self.clothSettings.goal_friction)

        value.append("goal_max = ")
        value.append(self.clothSettings.goal_max)

        value.append("goal_min = ")
        value.append(self.clothSettings.goal_min)

        value.append("goal_spring = ")
        value.append(self.clothSettings.goal_spring)

        value.append("gravity = ")
        value.append(self.clothSettings.gravity)

        value.append("internal_compression_stiffness = ")
        value.append(self.clothSettings.internal_compression_stiffness)

        value.append("internal_friction = ")
        value.append(self.clothSettings.internal_friction)

        value.append("internal_spring_max_diversion = ")
        value.append(self.clothSettings.internal_spring_max_diversion)
        value.append("internal_spring_max_length = ")
        value.append(self.clothSettings.internal_spring_max_length)
        value.append("internal_spring_normal_check = ")
        value.append(self.clothSettings.internal_spring_normal_check)
        value.append("internal_tension_stiffness = ")
        value.append(self.clothSettings.internal_tension_stiffness)
        value.append("internal_tension_stiffness_max = ")
        value.append(self.clothSettings.internal_tension_stiffness_max)
        value.append("mass = ")
        value.append(self.clothSettings.mass)

        value.append("pin_stiffness = ")
        value.append(self.clothSettings.pin_stiffness)

        value.append("pressure_factor = ")
        value.append(self.clothSettings.pressure_factor)

        value.append("quality = ")
        value.append(self.clothSettings.quality)

        value.append("rest_shape_key = ")
        value.append(self.clothSettings.rest_shape_key)

        value.append("sewing_force_max = ")
        value.append(self.clothSettings.sewing_force_max)

        value.append("shear_damping = ")
        value.append(self.clothSettings.shear_damping)
        value.append("shear_stiffness = ")
        value.append(self.clothSettings.shear_stiffness)
        value.append("shear_stiffness_max = ")
        value.append(self.clothSettings.shear_stiffness_max)

        value.append("shrink_max = ")
        value.append(self.clothSettings.shrink_max)
        value.append("shrink_min = ")
        value.append(self.clothSettings.shrink_min)

        value.append("target_volume = ")
        value.append(self.clothSettings.target_volume)

        value.append("tension_damping = ")
        value.append(self.clothSettings.tension_damping)
        value.append("tension_stiffness = ")
        value.append(self.clothSettings.tension_stiffness)
        value.append("tension_stiffness_max = ")
        value.append(self.clothSettings.tension_stiffness_max)

        value.append("time_scale = ")
        value.append(self.clothSettings.time_scale)

        value.append("uniform_pressure_force = ")
        value.append(self.clothSettings.uniform_pressure_force)

        value.append("use_dynamic_mesh = ")
        value.append(self.clothSettings.use_dynamic_mesh)

        value.append("use_internal_springs = ")
        value.append(self.clothSettings.use_internal_springs)

        value.append("use_pressure = ")
        value.append(self.clothSettings.use_pressure)
        value.append("use_pressure_volume = ")
        value.append(self.clothSettings.use_pressure_volume)

        value.append("use_sewing_springs = ")
        value.append(self.clothSettings.use_sewing_springs)

        value.append("vertex_group_bending = ")
        value.append(self.clothSettings.vertex_group_bending)
        value.append("vertex_group_intern = ")
        value.append(self.clothSettings.vertex_group_intern)
        value.append("vertex_group_mass = ")
        value.append(self.clothSettings.vertex_group_mass)
        value.append("vertex_group_pressure = ")
        value.append(self.clothSettings.vertex_group_pressure)
        value.append("vertex_group_shear_stiffness = ")
        value.append(self.clothSettings.vertex_group_shear_stiffness)
        value.append("vertex_group_shrink = ")
        value.append(self.clothSettings.vertex_group_shrink)
        value.append("vertex_group_structural_stiffness = ")
        value.append(self.clothSettings.vertex_group_structural_stiffness)

        value.append("voxel_cell_size = ")
        value.append(self.clothSettings.voxel_cell_size)

        file = open(
            'C:\\Users\\alex1\\Desktop\\CUDACloth\\clothSettings.txt', 'w')
        file.write(str(value).replace("]", "").replace("[", "").replace(
            ", '", "\n").replace("'", "").replace(",", ""))
        file.close()

    def pipInstall(self, pack):
        """
        В blender не так просто установить какой-либо пакет,
        потому я сделал функцию для установки через pip! 
        Достаточно просто передать параметр в виде имени пакета и раскоментировать строку в __init__
        """
        import sys
        import subprocess
        subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'ensurepip'])
        subprocess.call([sys.exec_prefix + '\\bin\\python.exe',
                                            '-m', 'pip', 'install'
                                            , 'numba'])


if __name__ == "__main__":
    # Проверка работоспособности CUDA
    # isCUDAAvailable()

    # Проверка dll для запуска симуляции
    # loadDLL()

    # Переход на первый кадр
    bpy.context.scene.frame_current = bpy.context.scene.frame_start

    SetUp()
    backUp = backupSet()
    Physics(backUp)

    # Запуск симуляции
    bpy.ops.screen.animation_play(reverse=False, sync=False)
