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

import numpy as np
# import sympy
import sys
import os

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
    def __init__(self):
        # Создаём атрибуты сцены для дальнейших вычислений
        self.gravity = bpy.context.scene.gravity
        self.fps = bpy.context.scene.render.fps      
        self.mesh = bpy.data.objects["Plane"]

        # Создаём атрибут бэкапа
        # self.backUp = backUp
        # print("__init__ backUp = ", self.backUp[:4][:3])

        # Собираем все объекты в сцене, способные к столкновению
        self.collision_objects = self.collisionOBJ()

    def start_sim(self):
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
            # if bpy.context.scene.frame_current == bpy.context.scene.frame_start:
            #     backupGet(self.backUp)        

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
                        flags.append(self.cross(mesh_global, collision_objects_global))
                        # print("flags[flagNum] = ", flags[flagNum])
                        flagNum += 1

            print("True in flags = ", True in flags)
            if not(True in flags):
                self.cloth_deformation()
            else:
                if flags[flagNum-1]:
                    for newCoor in self.mesh.data.vertices:
                        for i in range(0, 3):  
                            newCoor.co[i] -= (self.gravity[i] / self.fps)
                else:
                    for newCoor in self.mesh.data.vertices:
                        for i in range(0, 3):  
                            newCoor.co[i] += 0
        return gravityCalc

    def lenOfTwoPoints(self, a, b, distance_min=0.15):
        """ 
        Находим расскояние между двух точек,
        через формулу определение длинны вектора
        """
        x = (b[0] - a[0])**2
        y = (b[1] - a[1])**2
        z = (b[2] - a[2])**2

        return (x+y+z)**0.5

    def velocity(self, a=[0.123, 0.512, 0.362], mass = 0.3):
        """
        Возвращает скорость точки, 
        используя её массу и гравитацию в сцене
        V = S/t
        S = a - a_second
        t = 1/fps
        """
        # F = mass * gravity
        # a = F / mass
        # V = a * 1 / self.fps
        # t = 1/24

        # dx = sympy.Symbol('dx')
        # dy = sympy.Symbol('dy')
        # dz = sympy.Symbol('dz')

        # dt= sympy.Symbol('dt')

        # Vx = sympy.diff(dx, a[0])/sympy.diff(dt, t)
        # Vy = sympy.diff(dy, a[1])/sympy.diff(dt, t)
        # Vz = sympy.diff(dz, a[2])/sympy.diff(dt, t)

        V = [a[i] - a[i]-(self.gravity[i]/(60/self.fps)) for i in range(0, 3)]
        # print("Vx = ", V[0])
        # print("Vy = ", V[1])
        # print("Vz = ", V[2])
        return V

    def acceleration(self, a):
        """
        Функция возвращает ускорение для одной точки,
        в виде массива по 3 коордиинатам
        a = (V1-V2)/t
        t = 1/fps
        """
        V1 = self.velocity(a)
        V2 = self.velocity(self.velocity(a))
        a = [(V1[i] - V2[i])/(60/self.fps) for i in range(0, 3)]
        return a

    def cross(self, a, b, distance_min = 0.00001):
        """
        Задача данной функции – определить пересечения векторов двух точек,
        т.е. найти пересечения вершин, чтобы геометрия не пересекалась друг с другом
        Пераметр distance_min задаётся в настройках ткани во вкладке Collision ==> Object collision 
        ==> Distanse и по умолчанию равен 0.15
        Находится по формуле вычисления точек пересечения прямых в пространстве.
        """
        
        # Версия с a-b и b-c
        # collisionX = a[0] + self.lenOfTwoPoints(a, b) >= b[0] and b[0] + self.lenOfTwoPoints(b, c) >= c[0]
        # collisionY = a[1] + self.lenOfTwoPoints(a, b) >= b[1] and b[1] + self.lenOfTwoPoints(b, c) >= c[1]
        # collisionZ = a[2] + self.lenOfTwoPoints(a, b) >= b[2] and b[2] + self.lenOfTwoPoints(b, c) >= c[2]

        # Версия с a-b и c-d
        if (self.lenOfTwoPoints(a, b) > distance_min):
            collisionX = a[0] < b[0]
            collisionY = a[1] > b[1]
            collisionZ = a[2] < b[2]
            if collisionX and collisionY and collisionZ:
                return True

        return False

    def cloth_deformation(self, mass = 0.3):
        for newCoor in self.mesh.data.vertices:
            newCoor.co.x += ((mass * (self.gravity[0] - self.acceleration(newCoor.co)[0])) / self.fps)
            newCoor.co.y += ((mass * (self.gravity[1] - self.acceleration(newCoor.co)[1])) / self.fps)
            newCoor.co.z += ((mass * (self.gravity[2] - self.acceleration(newCoor.co)[2])) / self.fps)
            # newCoor.co.x += self.gravity[0] / self.fps
            # newCoor.co.y += self.gravity[1] / self.fps

    def collisionOBJ(self):

        """
        Данная функция возвращает массив объектов с которыми пересекается ткань
        """
        # numOfOBJ - объекты сцены
        numOfOBJ = np.array([bpy.data.objects[element]
                             for element in range(0, len(bpy.data.objects))])

        # collisionOBJs - список объектов способных к столкновению
        collisionOBJs = []
        
        for col in numOfOBJ:            
            try:
                col.modifiers['Collision'].settings.use == True
                collisionOBJs.append(col)
            except KeyError:
                pass

        return collisionOBJs
 
"""
Что за -> None?
Это PEP 3107 -- Function Annotations
Это подсказка для программистов о том, какой тип данных будет возвращён функцией
"""
class BackUp:
    def __init__(self):
        self.mesh = bpy.data.objects["Plane"].data
        self.backUp = []

    def __del__(self):
        "Пока на объект есть хоть одна ссылка, то бэкап будет существовать"
        print("Удаление бэкапа", self.mesh)

    def backupSet(self) -> list:
        """Создание бэкапа
        Возвразает список координат в виде двухмерного массива

        :BackUp.backupSet() >>  [[0, 1.2356, 0.001235], [0,0,0], ...]
        """    
        # список всех вершин изначального меша ткани
        backUp = []
        for NumVert in range(0, len(self.mesh.vertices)):
            print(self.mesh.vertices[NumVert].co[0:3])
            backUp.append(self.mesh.vertices[NumVert].co)
        return backUp

    def backupGet(self) -> None:
        """   Установка координат из бэкапа   """

        for i in range(0, len(self.mesh.vertices)):

            print("До = ", self.mesh.vertices[i].co[:3])
            print("backUp = ", backUp[i][:3])

            self.mesh.vertices[i].co.x = backUp[i][0]
            self.mesh.vertices[i].co.y = backUp[i][1]
            self.mesh.vertices[i].co.z = backUp[i][2]

            print("После = ",self.mesh.vertices[i].co[:3])
            print("После backUp = ", backUp[i][:3])

import collections

Point = collections.namedtuple('Vetex', ['mass', 'velocity', 'acceleration', 'coordinat', 'is_collide'])

class Point():
    
    number_of_points = len(bpy.data.objects["Plane"].data.vertices)
    mass = [0.3] * number_of_points
    velocity = [0] * number_of_points
    acceleration = [0] * number_of_points
    coordinat = [bpy.data.objects["Plane"].data.vertices[i].co for i in range(0, number_of_points)]
    is_collide = [False]

    def __init__(self):
        self._vertG = [Vertex(mass, velocity, acceleration, coordinat, is_collide) for mass in self.mass
                                                                                   for velocity in self.velocity 
                                                                                   for acceleration in self.acceleration 
                                                                                   for coordinat in self.coordinat
                                                                                   for is_collide in self.is_collide]

    def __len__(self):
        ''' Возвращяет кол-во точек на ткани'''
        return len(self._vertG)

    def __repr_(self):
        return 'Point %r' % (self._vertG)

    def __getiitem__(self, position):
        ''' Возвращает параметры конкретной точки'''
        return self._vertG[position]
    
    def add_masses(self, position1, position2):
        ''' Складывает массы двух точек'''
        return _vertG[position1][0] + _vertG[position2][0]

    def subtract_masses(self, position1, position2):
        ''' Вычитает массы двух точек'''
        return _vertG[position1][0] - _vertG[position2][0]
    
    def find_momentum(self, position1):
        ''' 
        Возвращает список значений, который является импульсом точки.
        Номер является осью, а значение силой импульса по конкретной оси.
        
        :find_momentum(i) >> [0, 1.2356, 0.001235]
        '''
        return [_vertG[position1][0][i] * _vertG[position1][1][i] for i in range(0, 3)]

    def len_of_two_points(self, position1, position2, distance_min=0.15):
        """ 
        Находим расскояние между двух точек,
        через формулу определение длинны вектора
        """
        x = (b[0] - a[0])**2
        y = (b[1] - a[1])**2
        z = (b[2] - a[2])**2

        return (x+y+z)**0.5

    def velocity(self, position1):
        """
        Возвращает скорость точки, 
        используя её массу и гравитацию в сцене
        V = S/t
        S = a - a_second
        t = 1/fps
        """
        V = (self.lenOfTwoPoints(a, [(a[i] + (self.mass * self.gravity[i] / self.fps))
                                     for i in range(0, 3)]) / (1 / self.fps))
        # print("V = ", V)
        return V

    def acceleration(self, position1d):
        """
        Функция возвращает ускорение для одной точки,
        в виде массива по 3 коордиинатам
        a = (V1-V2)/t
        t = 1/fps
        """
        V1 = self.velocity(a)
        V2 = self.lenOfTwoPoints([(a[i] + (self.mass * self.gravity[i] / self.fps))
                                  for i in range(0, 3)],
                                 [(a[i] + (self.mass * self.gravity[i] / self.fps)**2)
                                  for i in range(0, 3)]) / (1 / self.fps)
        ax, ay, az = [(V1 - V2) / (1 / self.fps)
                       for i in range(0, 3)]

        # ax, ay, az = [(V1[i] - V2[i]) / (1 / self.fps)
        #                for i in range(0, 3)]
        return ax, ay, az



# В данном класе создаётся сцена для тестировния алгоритма физики ткани
class SetUp:
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
            bpy.ops.object.subdivision_set(level=3, relative=False)

            # Изменяем подразделение на "простое"
            bpy.context.object.modifiers["Subdivision"].subdivision_type = 'SIMPLE'

            # Применяем модификатор
            bpy.ops.object.modifier_apply(modifier="Subdivision")

            # Создаём куб на который будет падать ткань
            bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False,
                                            align='WORLD', location=(0, 0, 0), scale=(0.8, 0.8, 0.8))
            # Уменьшает куб
            bpy.ops.transform.resize(value=(0.5, 0.5, 0.5))

            # Назначаем его объектом столкновения
            bpy.ops.object.modifier_add(type='COLLISION')

            print("Scene created!")


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
    isCUDAAvailable()

    # Проверка dll для запуска симуляции
    loadDLL()

    # Переход на первый кадр
    bpy.context.scene.frame_current = bpy.context.scene.frame_start

    SetUp()
    # backUp = backupSet()
    Physics().start_sim()

    # Запуск симуляции
    bpy.ops.screen.animation_play(reverse=False, sync=False)
