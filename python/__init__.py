"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

Задача данного файла – подключение к blender, настройка UI, регистрации аддона.
"""
# импортируем API для работы с blender
import bpy
import numpy as np
import sys
import os
import collections
import mathutils
import time
# import sympy


# Добавляем папку с проектом в поле зрения blender
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
    sys.path.append(dir + "\\python")
    print(sys.path)

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


# В данном класе создаётся сцена для тестировния алгоритма физики ткани
class SetUp:
    def __init__(self):
        self.opening_scene()
        # self.pipInstall("numba") # установка внешних пакетов

    def opening_scene(self):
        '''
        Создаёт сцену с кубом и плоскостью
        '''
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
            # bpy.ops.object.subdivision_set(level=3, relative=False)

            # Изменяем подразделение на "простое"
            # bpy.context.object.modifiers["Subdivision"].subdivision_type = 'SIMPLE'

            # Применяем модификатор
            # bpy.ops.object.modifier_apply(modifier="Subdivision")

            # Создаём куб на который будет падать ткань
            bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False,
                                            align='WORLD', location=(0, 0, 0), scale=(0.8, 0.8, 0.8))
            # Уменьшает куб
            bpy.ops.transform.resize(value=(0.5, 0.5, 0.5))

            # Назначаем его объектом столкновения
            bpy.ops.object.modifier_add(type='COLLISION')

            print("Scene created!")


    def pipInstall(self, pack):
        '''
        В blender не так просто установить какой-либо пакет,
        потому я сделал функцию для установки через pip! 
        Достаточно просто передать параметр в виде имени пакета и раскоментировать строку в __init__
        '''
        import sys
        import subprocess
        subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'ensurepip'])
        subprocess.call([sys.exec_prefix + '\\bin\\python.exe',
                                            '-m', 'pip', 'install'
                                            , 'numba'])

SetUp()

Points = collections.namedtuple('Point', ['velocity', 'acceleration', 'coordinates', 'is_collide'])

class Point:
    '''
    Класс реализуется по шаблону "Стратегия"
    Суть класса в усовершенствовании базовых точек из блендера, для более точной симуляции ткани
    number_of_points - кол-во точек на ткани
    mass_of_point - масса каждой точки
    [0] velocity_of_point - скорость каждой точки
    [1] acceleration_of_point - ускорение каждой точки
    [2] coordinates_of_point - координаты каждой точки
    [3] is_collide - сталкивается ли точка с другими объектами

    :Point.get all()[0] = [0.003, 0.001, 0.002] - масса всех точек
    :Point.get all()[0][0] = 0.003
    '''
    number_of_points = len(bpy.data.objects["Plane"].data.vertices)

    mass_of_point = 0.3 / number_of_points
    velocity_of_point  = [[0,0,0]] * number_of_points
    acceleration_of_point  = [[0,0,0]] * number_of_points
    # coordinates_of_point  = [bpy.data.objects["Plane"].data.vertices[i].co for i in range(0, number_of_points)]
    coordinates_of_point  = [bpy.data.objects["Plane"].matrix_world @ bpy.data.objects["Plane"].data.vertices[i].co for i in range(0, number_of_points)]
    is_collide_of_point  = [False] * number_of_points

    distance_min = 0.0001

    __slots__ = ['__vertG', '__xyz']

    # , number_of_points, mass_of_point, velocity, acceleration, coordinates, is_collide
    def __init__(self):

        # Этот вариант более подходящий, но выполняется слишком долго :(
        # self.__vertG = [Points(velocity, acceleration, coordinates) 
        #                         for velocity in self.velocity for acceleration in self.acceleration 
        #                         for coordinates in self.coordinates]

        self.__vertG = Points([v for v in self.velocity_of_point], 
                              [a for a in self.acceleration_of_point], 
                              [coo for coo in self.coordinates_of_point],
                              [col for col in self.is_collide_of_point])
        
        # print("__vertG = ", self.__vertG, "number_of_points = ", self.number_of_points, sep="\n")

    def __len__(self) -> int:
        ''' Возвращяет кол-во точек на ткани'''
        return self.number_of_points

    # def __repr__(self):
    #     return 'Point: ' % (self.__vertG)

    def __getiitem__(self, position:int) -> list:
        ''' Возвращает параметры конкретной точки'''
        return self.__vertG[position]

    # ------------------------------------------------------------------------
    #    CALCULATIONS
    # ------------------------------------------------------------------------
 
    def __add_masses(self, position1:int, position2:int) -> float:
        ''' Складывает массы двух точек'''
        return __vertG[position1][0] + __vertG[position2][0]

    def __subtract_masses(self, position1:int, position2:int) -> float:
        ''' Вычитает массы двух точек'''
        return __vertG[position1][0] - __vertG[position2][0]
    
    def find_momentum(self, position1:int) -> list:
        ''' 
        Возвращает список значений, который является импульсом точки.
        Номер является осью, а значение силой импульса по конкретной оси.
        
        :find_momentum(i) >> [0, 1.2356, 0.001235]
        '''
        return [self.mass_of_point * self.get_acceleration(position1) 
                for i in range(0, 3)]

    def find_distance(self, position1:int, position2:int) -> float:
        ''' 
        Находим расскояние между двух точек,
        через формулу определение длинны вектора
        '''
        __xyz = [(b[i] - a[i])**2 for i in range(0, 3)]

        return (__xyz[0] + __xyz[1] + __xyz[2])**0.5

    def find_velocity(self, position1:int) -> list:
        '''
        Возвращает скорость точки, 
        используя её массу и гравитацию в сцене
        V = S/t
        S = a - a_second
        t = 1/fps
        '''
        V = [self.get_coo(i) - self.get_coo(i) - (self.gravity[i]/(60/self.fps)) for i in range(0, 3)]
        # print("V = ", V)
        return V

    def find_acceleration(self, position1:int) -> list:
        '''
        Функция возвращает ускорение для одной точки,
        в виде массива по 3 коордиинатам
        a = (V1-V2)/t
        t = 1/fps
        '''
        V1 = self.velocity(a)
        V2 = self.velocity(self.velocity(a))
        a = [(V1[i] - V2[i])/(60/self.fps) for i in range(0, 3)]
        return a

    # ------------------------------------------------------------------------
    #    GETTERS
    # ------------------------------------------------------------------------
    
    def get_velocity(self, position:int) -> list:
        return self.__vertG[0][position]
    
    def get_acceleration(self, position:int) -> list:
        return self.__vertG[1][position]
        
    def get_coo(self, position:int) -> list:
        return self.__vertG[2][position]
    
    def get_is_collide(self, position:int) -> None:
        return self.__vertG[3][position] 
 
    @property   
    def all(self):
        return self.__vertG

    @property 
    def mass(self) -> float:
        return self.mass_of_point

    @property
    def get_all_coo(self):
        return self.__vertG[2]

    # ------------------------------------------------------------------------
    #    SETTERS
    # ------------------------------------------------------------------------

    # Не удалось выставить сеттеры с помощью property, 
    # поскольку property не даёт передавать больше одного параметра

    def set_velocity(self, position:int, velocity:list) -> None:
        self.__vertG[0][position] = velocity
    
    def set_acceleration(self, position:int, acceleration:list) -> None:
        self.__vertG[1][position] = acceleration

    def set_coo(self, position:int, new_coo:list) -> None:
        new_coo = mathutils.Vector(new_coo)
        self.__vertG[2][position] += new_coo
        bpy.data.objects["Plane"].data.vertices[position].co.xyz += new_coo

    def set_is_collide(self, position:int, is_collide:bool) -> None:
        self.__vertG[3][position] = is_collide

class BackUp(Point):
    def __init__(self, mesh:Point):
        self.mesh = mesh
        self.backUp = []

    def __del__(self):
        'Пока на объект есть хоть одна ссылка, то бэкап будет существовать'
        # print("Удаление бэкапа", self.mesh)
        del self.mesh
        del self.backUp

    def backupSet(self) -> list:
        ''' Создание бэкапа
        Возвразает список координат в виде двухмерного массива

        :BackUp.backupSet() >>  [[0, 1.2356, 0.001235], [0,0,0], ...]
        '''    
        # список всех вершин изначального меша ткани
        backUp = []
        for NumVert in range(0, len(self.mesh.vertices)):
            # print(self.mesh.vertices[NumVert].co[0:3])
            backUp.append(self.mesh.vertices[NumVert].co)
        return backUp

    def backupGet(self) -> None:
        '''  Установка координат из бэкапа   '''
        for i in range(0, len(self.mesh.vertices)):

            # print("До = ", self.mesh.vertices[i].co[:3])
            # print("backUp = ", backUp[i][:3])

            self.mesh.vertices[i].co.x = backUp[i][0]
            self.mesh.vertices[i].co.y = backUp[i][1]
            self.mesh.vertices[i].co.z = backUp[i][2]

            # print("После = ",self.mesh.vertices[i].co[:3])
            # print("После backUp = ", backUp[i][:3])

class Physics(Point):
    def __init__(self, vertexes_of_cloth:Point):
        # Создаём атрибуты сцены для дальнейших вычислений
        self.gravity = bpy.context.scene.gravity
        self.fps = bpy.context.scene.render.fps      
        self.mesh = vertexes_of_cloth

        # Создаём атрибут бэкапа
        # self.backUp = backUp
        # print("__init__ backUp = ", self.backUp[:4][:3])

        # Собираем все объекты в сцене, способные к столкновению
        self.collision_objects = self.collisionOBJ()
        self.cloth_objects = clothOBJ()

    def start_sim(self):
        ''' Функция запускает просчёты перед показом каждого кадра '''
        bpy.app.handlers.frame_change_pre.clear()
        bpy.app.handlers.frame_change_pre.append(self.physicsCalculation())

    def physicsCalculation(self):
        '''
        Данная функция вызывается каждый кадр.
        запускает ещё несколько функция для расчёта физики

        mesh - объект, с симуляциет ткани

        collision_objects - объект(ы) столкновения

        backUp - бэкап ткани до симуляции
        '''
        def gravityCalc(scene):
            '''
            Функция вычисляет, насколько должна переместиться точка,
            при заданной массе, гравитации и ускорении
            Для вычисления используется формула уменьшенного веса.
            Р - вес / m - масса / g - гравитация
            Р = m(g-a)
            '''
            flagNum = 0
            flags = []
            # if bpy.context.scene.frame_current == bpy.context.scene.frame_start:
            #     backupGet(self.backUp)   
            # i - одна вершина из всех, что есть в объекте, с симуляциет ткани
            for i in range(0, len(self.mesh)-1):

                # obj_vert - меш из списка объектов столкновений
                for obj_vert in self.collision_objects:

                    # о - одна вершина из всех, что есть в объекте столкновения
                    for o in range(0, len(obj_vert.data.vertices)):

                        # 1-я вершина из отрезка
                        global_col_point = obj_vert.matrix_world @ obj_vert.data.vertices[o].co

                        # [0] - x, [1] - y, [2] - z
                        flags.append(self.cross(self.mesh.get_coo(i), global_col_point, i))
                        flagNum += 1

            # print("True in flags = ", True in flags)
            if not(True in flags):
                self.cloth_deformation()
            else:
                # if (True in flags):
                #     for newCoor in self.mesh.get_all_coo:
                #         for i in range(0, 3):  
                #             newCoor[i] -= (self.gravity[i] / (60 / self.fps))
                # else:
                for newCoor in self.mesh.get_all_coo:
                    for i in range(0, 3):  
                        newCoor[i] += 0
        return gravityCalc

    def lenOfTwoPoints(self, a, b):
        '''
        Находим расскояние между двух точек,
        через формулу определение длинны вектора
        '''
        x = (b[0] - a[0])**2
        y = (b[1] - a[1])**2
        z = (b[2] - a[2])**2

        return (x+y+z)**0.5

    def cross(self, a, b, num_a, distance_min = 0.1) -> bool:
        '''
        Задача данной функции – определить пересечения векторов двух точек,
        т.е. найти пересечения вершин, чтобы геометрия не пересекалась друг с другом
        Пераметр distance_min задаётся в настройках ткани во вкладке Collision ==> Object collision 
        ==> Distanse и по умолчанию равен 0.15
        Находится по формуле вычисления точек пересечения прямых в пространстве.
        '''
         # Версия с a-b и c-d
        if (self.lenOfTwoPoints(a, b) >= distance_min):
            collisionX = a[0] < b[0]
            collisionY = a[1] > b[1]
            collisionZ = a[2] < b[2]
            if (collisionX and collisionY and collisionZ):
                self.mesh.set_is_collide(num_a, True)
                return True
        return False

    def cloth_deformation(self):
        for num in range(0, len(self.mesh)):
            acceleration = self.mesh.get_acceleration(num)
            res = [((self.mesh.mass * (self.gravity[i] - acceleration[i])) / self.fps)
                     for i in range(0, 3)]
            self.mesh.set_coo(num, res)

    def collisionOBJ(self):
        '''
        Данная функция возвращает массив объектов с которыми пересекается ткань
        '''
        # numOfOBJ - объекты сцены
        numOfOBJ = np.array([bpy.data.objects[element]
                             for element in range(0, len(bpy.data.objects))])

        # collisionOBJs - список объектов способных к столкновению
        collisionOBJs = []
        
        for col in numOfOBJ:            
            try:
                col.modifiers['Collision'].settings.use
                collisionOBJs.append(col)
            except KeyError:
                pass
        return collisionOBJs
    
    @staticmethod
    def clothOBJ():

        # numOfOBJ - объекты сцены
        numOfOBJ = np.array([bpy.data.objects[element]
                             for element in range(0, len(bpy.data.objects))])

        # clothOBJs - список объектов способных к столкновению
        clothOBJs = []
        
        for clo in numOfOBJ:            
            try:
                clo.modifiers['GPUCloth'].settings.use
                clothOBJs.append(clo)
            except KeyError:
                pass
        print(clothOBJs)
        return clothOBJs

# Добавляем папку с проектом в поле зрения blender
def importForDebugg():
    dir = os.path.dirname(bpy.data.filepath)
    if not dir in sys.path:
        sys.path.append(dir)
        sys.path.append(dir + "\\python")
        sys.path.append(dir + "\\python" + "\\work version 0.0.1")
        # print(sys.path)

def loadDLL():
    '''
    Здась происходит загрузка DLL файла сгенерированного из cuda файла
    '''
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
    '''
    Пробуем импортировать PyCuda для проверки поддержки данной технологии
    Надо исправить, потому что у человека может быть просто не установлен пакет
    для питона
    '''
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
    except ModuleNotFoundError:
        print("Технология CUDA не поддерживается на данном устройстве")


# ------------------------------------------------------------------------
#    UI
# ------------------------------------------------------------------------


from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       PointerProperty
                       )

from bpy.types import (Panel,
                       Operator,
                       AddonPreferences,
                       PropertyGroup,
                       )
 

#    --- Properties for UI ---

class MySettings(PropertyGroup):

    my_bool_object_coll : BoolProperty(
        name="Enable or Disable",
        description="A bool property",
        default = False
        )
        
    my_bool_self_coll : BoolProperty(
        name="Enable or Disable",
        description="A bool property",
        default = False
        )

    my_int : IntProperty(
        name = "Set a value",
        description="A integer property",
        default = 23,
        min = 10,
        max = 100
        )

    my_float_vertex : FloatProperty(
        name = "Set a value",
        description = "A float property",
        default = 0.33,
        min = 0.001,
        max = 30.0
        )

    my_float_speed : FloatProperty(
        name = "Set a value",
        description = "A float property",
        default = 0.1000,
        min = 0.001,
        max = 30.0
        )

    my_float_air : FloatProperty(
        name = "Set a value",
        description = "A float property",
        default = 0.1000,
        min = 0.001,
        max = 30.0
        )
        
    my_string : StringProperty(
        name = "Set a value",
        description = "A string property",
        default = "None"
        )

#    --- GPUCloth in Properties window ---

class UV_PT_GPUCloth(Panel):
    bl_idname = "UV_PT_GPUCloth"
    bl_label = "GPUCloth"
    bl_category = "My Category"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool

        # display the properties
        layout.prop(mytool, "my_float_vertex", text="Vertex Mass")
        layout.prop(mytool, "my_float_speed", text="Speed Multiplier")
        layout.prop(mytool, "my_float_air", text="Air Viscosity")

        layout.prop(mytool, "my_bool_object_coll", text="Object Collisions")
        layout.prop(mytool, "my_bool_self_coll", text="Self Collisions")

        # check if bool property is enabled
        if (mytool.my_bool_object_coll == True):
            print ("Property Enabled")
        else:
            print ("Property Disabled")

        if (mytool.my_bool_self_coll == True):
            print ("Property Enabled")
        else:
            print ("Property Disabled")

#     --- Registration UI Panel ---

classes = (
    MySettings,
    UV_PT_GPUCloth,
)


# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Scene.my_tool = PointerProperty(type=MySettings)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Scene.my_tool

if __name__ == "__main__":
    register()
    # Проверка работоспособности CUDA
    # isCUDAAvailable()

    # Проверка dll для запуска симуляции
    # loadDLL()

    # Переход на первый кадр
    bpy.context.scene.frame_current = bpy.context.scene.frame_start

    # 
    cloth = Point()
    # print( '\n', "p.get_all()[0][0] = ",p.get_all()[0][0], '\n')
    backUp = BackUp(cloth).backUp
    sim = Physics(cloth)

    sim.start_sim()

    # Запуск симуляции
    bpy.ops.screen.animation_play(reverse=False, sync=False)
