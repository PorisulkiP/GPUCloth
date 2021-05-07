"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

Задача данного файла – подключение к blender, настройка UI, регистрации аддона.
Что за -> None?
Это PEP 3107 -- Function Annotations
Это подсказка для программистов о том, какой тип данных будет возвращён функцией

Что за :int / :list?
Это тоже подсказка для программистов о том, какой тип данных нужно передавать.
Он носит рекомендательных характер, но конечно лучше это всё соблюдать

Чем отличается __repr__ от __str__?
__repr__ - должен быть написан так, чтобы РАЗРАБОТЧИК смог получить всю нужную ему информацию
__str__ - должен быть написан так, чтобы ПОЛЬЗОВАТЛЬ смог понять информацию о классе и его данных

Почему у переменой __vert_append два подчёркивания?
это нужно для защиты переменной от внешнего воздействия, т.к благодаря __ переменная стала приватной

Что за знак @?
Это PEP 465. Его задача – перемножить матрицы

Виды пружин:
-пружины сжатия/растяжения         (STRUCTURAL)
-пружины кручения                  (SHEAR)
-пружины изгиба.                   (BEND)
"""
# импортируем API для работы с blender
import bpy
import numpy as np
import sys, os, math, collections, mathutils, time
from mathutils import Vector

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

bl_info = {
    "name": "GPUCloth",
    "author": "PorisulkiP, GoldBath, Vi, Victus",
    "version": (0, 1, 0),
    "blender": (2, 93, 0),
    "location": "",
    "warning": "",
    "description": "Cloth simulation on GPU",
    "doc_url": "https:#github.com/PorisulkiP/GPUCloth",
    "category": "System",
}

blender_version = bpy.app.version

# В данном класе создаётся сцена для тестировния алгоритма физики ткани
class SetUp:
    def __init__(self):
        self.opening_scene()
        # self.pipInstall("pycuda") # установка внешних пакетов

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

            # Подразделяем для симуляции ткани
            # bpy.ops.object.subdivision_set(level=2, relative=False)

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
                                            '-m', 'pip', 'install', pack])


SetUp()


class Cloth:
    def __init__(self):
        super().__init__()
        self.mass = self.meshResolution*0.3

    @property
    def meshResolution(self):
        ''' Кол-во точек на ткани '''
        return len(bpy.data.objects["Plane"].data.vertices)
    
    @property
    def get_all_coord(self) -> list:
        ''' Возвращает список(list) всех локальных координат '''
        return self.__local_vertexPosition

    @property
    def cloth_lenght(self):
        # Длинна ткани по x
        sim_u = int(self.meshResolution**0.5) # Горизонталь
        return sim_u

    @property
    def cloth_wight(self):
        # Длинна ткани по y
        sim_v = int(self.meshResolution**0.5) # Вертикаль
        return sim_v
    

class Point:
    '''
    Класс реализуется по шаблону "Стратегия"
    Суть класса в усовершенствовании базовых точек из блендера, для более точной симуляции ткани
    _meshResolution  - кол-во точек на ткани
    __vertexMass - масса каждой точки
    [0] vertexVelocity- скорость каждой точки
    [1] vertexAcceleration - ускорение каждой точки
    __vertexPosition - координаты каждой точки
    [2] vertex_Is_collide - сталкивается ли точка с другими объектами

    '''

    def __init__(self, objID, vId, pos_x, pos_y, velocity= [0,0,0], 
                acceleration = [0,0,0], hasCollision = False):
        self.clothProp = Cloth()
        self.objID = objID
        self.id = vId
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.velocity= velocity
        self.acceleration = acceleration
        self.hasCollision = hasCollision

    def __len__(self) -> int:
        ''' Возвращяет кол-во точек на ткани'''
        return self.clothProp.meshResolution
       
    def __repr__(self):
        return f'''
        All properties of points: 
        id = {self.id}
        V = {self.velocity}
        a = {self.acceleration}
        m = {self.clothProp.mass}
        hasCollision = {self.hasCollision}
        '''

    # ------------------------------------------------------------------------
    #    GETTERS
    # ------------------------------------------------------------------------

    @property
    def V(self) -> list:
        ''' Возвращает список(list) скорости каждой точки '''
        return self.velocity

    @property
    def a(self) -> list:
        ''' Возвращает список(list) ускорения каждой точки '''
        return self.acceleration
    
    @property
    def local_position(self) -> list:
        ''' Возвращает список(list) координат конкретной точки '''
        return bpy.data.objects["Plane"].data.vertices[self.id].co

    @property
    def global_position(self) -> list:
        ''' Возвращает список(list) координат конкретной точки '''
        x = bpy.data.objects["Plane"].matrix_local @ bpy.data.objects["Plane"].data.vertices[self.id].co
        return x
    
    @property
    def has_collision(self) -> bool:
        ''' Возвращает факт(bool) пересейчения каждой точки '''
        return self.hasCollision

    @property
    def m(self) -> float:
        ''' Возвращает массу(float) точек '''
        return 0.3 /  self.clothProp.meshResolution

    # ------------------------------------------------------------------------
    #    SETTERS
    # ------------------------------------------------------------------------

    def set_velocity(self, velocity:list) -> None:
        ''' Устновка скорости для конкретных точек '''
        self.velocity = velocity
    
    def set_acceleration(self, acceleration:list) -> None:
        ''' Установка ускорения для конкретных точек '''
        self.acceleration = acceleration

    def set_coo(self, new_coo:Vector) -> None:
        ''' Установка новых координат '''
        bpy.data.objects["Plane"].data.vertices[self.id].co = new_coo


    def set_is_collide(self, hasCollision:bool) -> None:
        ''' Установка параметра столкновения '''
        self.hasCollision = hasCollision


class BackUp(Cloth):
    def __init__(self):
        super.__init__()
    
    def set_backUp(self, backUp:list) -> None:
        ''' Устновка координат точек из бэкапа '''
        for i in range(0, self.meshResolution):bpy.data.objects["Plane"].data.vertices[i].co = mathutils.Vector(backUp[i])
    
    @property
    def creating_backUp(self) -> list:
        ''' Возвращает список(list) бэкап точек '''        
        return [bpy.data.objects["Plane"].matrix_local @ bpy.data.objects["Plane"].data.vertices[i].co for i in range(0, self.meshResolution)]


class Spring:
    '''
    Класс пружины(нитки, но не совсем)
    pos_x: первая точка
    pos_y: вторая точка
    ks: постоянная пружины
    kd: коэффициент трения скольжения
    rest_length(l0): длина пружины в покое
    STIFFNESS: жёсткость
    '''
    STIFFNESS = 1

    def __init__(self, a, b, ks, kd, rest_length, spring_type):
        self.pos_x = a
        self.pos_y = b
        self.ks = ks
        self.kd = kd
        self.rest_length = rest_length
        self.spring_type = spring_type

    def __repr__(self):
        return f'''Spring: \nFirst position: {self.pos_x} \nSecond position: {self.pos_y} \
        \nks: {self.ks}\nkd: {self.kd} \nRest length: {self.rest_length} \nSpring type: {self.spring_type}       
                '''


class Collision:

    def __init__(self):
        super().__init__()
        # num_of_OBJ - объекты сцены
        self.__num_of_OBJ = np.array([bpy.data.objects[element]
                             for element in range(0, len(bpy.data.objects))])

        # collisionOBJs - список объектов способных к столкновению
        self.__collisionOBJs = []

        # clothOBJs - список объектов способных к столкновению
        self.__clothOBJs = []

    def __collisionOBJ(self) -> None:
        ''' Добавляет в массив объекты, с которыми пересекается ткань '''
        for col in self.__num_of_OBJ:            
            try:
                col.modifiers['Collision'].settings.use
                self.__collisionOBJs.append(col)
            except KeyError:
                pass

    def __clothOBJ(self) -> None:
        ''' Добавляет в массив объекты, на которых накинут можификатор ткани '''
        for clo in self.__num_of_OBJ:            
            try:
                clo.modifiers['GPUCloth'].settings.use
                self.__clothOBJs.append(clo)
            except KeyError:
                pass

    @property
    def collide_objs(self) -> list:
        ''' Возвращает массив объектов с которыми может столкнуться ткань '''
        self.__collisionOBJ()
        return self.__collisionOBJs
        
    @property
    def gpu_cloth_obj(self) -> list:
        ''' Возвращает массив объектов, на которых накинут можификатор ткани  '''
        self.__clothOBJ()
        return self.__clothOBJs

    def get_depth(self, a, b):
        ''' Возвращает глубину проникновения в объект '''
        depth = float(Vector(a) - Vector(b))
        return depth


class Physics(Cloth, Collision):

    GRAVITY = bpy.context.scene.gravity
    FPS = bpy.context.scene.render.fps
    TIME_STEP = 0.01
    # TIME_STEP = 0.001 / FPS
    print(TIME_STEP)
   
    STRUCTURAL_SPRING_TYPE = 0
    SHEAR_SPRING_TYPE = 1
    BEND_SPRING_TYPE = 2

    DEFAULT_DAMPING = -15
    VERTEX_SIZE = 4
    VERTEX_SIZE_HALF = 2.0

    # ks - постоянная пружины (Коэффициент упругости)
    # Отвечает за растяжение, чем больше, тем длиннее связи
    KS_STRUCTURAL = 50.75
    KD_STRUCTURAL = -0.25

    # Отвечает за кручение
    KS_SHEAR = 50.75
    KD_SHEAR = -0.25

    # Отвечает за изгиб
    KS_BEND = 50.95
    KD_BEND = -0.25

    def __init__(self):
        super().__init__()
        # Дельта времени
        self.dt = self.TIME_STEP

        # Создаём бэкап
        self.backUp = BackUp.creating_backUp

        # Создаём переменную где хранятся все связи(нитки, но не совсем) ткани
        self.protect_vertices = [] # точки которые не должны двигаться
        self.vertices = [] 
        self.last_coo = []
        self.springs = []
        self.preparation()

    def start_sim(self) -> None:
        ''' Функция запускает просчёты перед показом каждого кадра '''
        bpy.app.handlers.frame_change_pre.clear()
        bpy.app.handlers.frame_change_pre.append(self.сalculation())

    def сalculation(self) -> None:
        '''
        Данная функция вызывается каждый кадр.
        запускает ещё несколько функция для расчёта физики

        mesh - объект, с симуляциет ткани

        collide_objs - объект(ы) столкновения

        backUp - бэкап ткани до симуляции
        '''
        def checkCollision(scene):
            '''
            Функция вычисляет, насколько должна переместиться точка,
            при заданной массе, гравитации и ускорении
            Для вычисления используется формула уменьшенного веса.
            Р - вес / m - масса / g - гравитация
            Р = m(g-a)

            План такой:
            -Подготавливаем сцену
            -Расчитываем силы влияющие на движение точек
            -Расчитываем новые координаты точек
            -Проверяем новые координаты на пересечения
            -Выставляем координаты

            '''
            # if bpy.context.scene.frame_current == bpy.context.scene.frame_start:
            #     BackUp.set_backUp(self.backUp)

            id = 0
            # Запускаем цикл просчёта здесь
            for x in range(self.cloth_lenght):
                for y in range(0, self.cloth_wight):
                    self.vertices[id].set_coo(self.ComputeForces(x, y, id))
                    id += 1

            # if not(True in flags):
            #     self.ComputeForces()
            #     self.IntegrateVerlet()
            # else:
            #     for i in range(0, self.meshResolution):
            #         for num in self.vertices[i].global_position:
            #             self.vertices[i].set_coo([self.vertices[i].local_position[xyz]
            #                         for xyz in range(3)])
        return checkCollision
        
    def lenOfTwoPoints(self, a, b) -> float:
        '''
        Находим расскояние между двух точек,
        через формулу определение длинны вектора
        '''
        x = (b[0] - a[0])**2
        y = (b[1] - a[1])**2
        z = (b[2] - a[2])**2

        return (x+y+z)**0.5
    
    def cross(self, a, b, num_a, distance_min = 0.0001) -> bool:
        '''
        Задача данной функции – определить пересечения векторов двух точек,
        т.е. найти пересечения вершин, чтобы геометрия не пересекалась друг с другом
        Пераметр distance_min задаётся в настройках ткани во вкладке Collision ==> Object collision 
        ==> Distanse и по умолчанию равен 0.15
        Находится по формуле вычисления точек пересечения прямых в пространстве.
        '''
        # Версия с a-b и c-d
        # print(a, b, sep="\n")

        """  
            void clothObjCollision( Pos, & NextPos, 
                                        unsigned int x, unsigned int y) {

                float r = 0.005f

                for (int i = 0 i < objVar.nTrig - 1 i++) {
                    #triangle strip
                     A = readObjVbo(i, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos)
                     B = readObjVbo(i + 1 + (i + 1) % 2, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos)
                     C = readObjVbo(i + 1 + i % 2, objBuff, objIndBuff, objVar.vboStrdFlt, objVar.OffstPos)
                     n = objN[i]

                    if (sphrTrigCollision(Pos, NextPos, r, A, B, C, n)) {
                        #fix the damping bug
                        colFlag[x * self.cloth_wight + y] = true

                        float dn = getPerpDist(NextPos, r, A, n)
                        NextPos += 11f * (-dn) * n
                        break
                    }
                    else { 
                        if (colFlag[x * self.cloth_wight + y]) { #check last frame collision
                            writeToVBO(0, ppWriteBuff, x, y, fxVar.OffstVel)
                        }
                        colFlag[x * self.cloth_wight + y] = false
                    }
                }
            }  
        """
        if (self.lenOfTwoPoints(a, b) >= distance_min):
            collisionX = a[0] < b[0]
            collisionY = a[1] > b[1]
            collisionZ = a[2] < b[2]
            if (collisionX and collisionY and collisionZ):
                return True
        return False

    def plane_collision(self) -> None:
        flags = []
        flag_num = 0
        # i - одна вершина из всех, что есть в объекте, с симуляциет ткани
        for i in range(0, self.meshResolution-1):

            # obj_vert - меш из списка объектов столкновений
            for obj_vert in self.collide_objs:

                # о - одна вершина из всех, что есть в объекте столкновения
                for o in range(0, len(obj_vert.data.vertices)):

                    # 1-я вершина из отрезка
                    global_col_point = obj_vert.matrix_world @ obj_vert.data.vertices[o].co

                    # [0] - x, [1] - y, [2] - z
                    flags.append(self.cross(self.vertices[i].global_position, global_col_point, i))
                    flag_num += 1
        return flags, flag_num

    def add_point(self, objID, vId, pos_x, pos_y, velocity= [0,0,0], acceleration = [0,0,0],
                                                hasCollision = False) -> None:
        # Добавляем класс точки в список
        vert = Point(objID, vId, pos_x, pos_y, velocity, acceleration, hasCollision)
        self.vertices.append(vert)

    def add_spring(self, a, b, ks, kd, spring_type) -> None:
        # Добавляем класс пружин в список
        s = Spring(a, b, ks, kd, a - b, spring_type)
        self.springs.append(s)

    def IntegrateVerlet(self, pos, last_pos, a) -> list:
        ''' 
            Метод Стёрмера — Верле(Интеграция Верле)
            https:#ru.wikipedia.org/wiki/Метод_Стёрмера_—_Верле
             С помощью интеграции верлета находим дифференциал скорости
            
            1.Вычисляются новые положения тел.
            2.Для каждой связи удовлетворяется соответствующее ограничение, 
            то есть расстояние между точками делается таким, каким оно должно быть.
            3.Шаг 2 повторяется несколько раз, тем самым все условия удовлетворяются (разрешается система условий).
        '''
        return 2.0 * pos - last_pos + a * self.dt**2

    def preparation(self) -> None:
        '''
            Подготавливанем миллион данных для корректной работы симуляции
        '''

        i = 0
        for x in range(self.cloth_lenght):
            for y in range(0, self.cloth_wight):
                self.add_point(0, i, x, y)
                i+=1

        self.last_coo = [self.vertices[i].local_position for i in range(0, self.meshResolution)]

        # Добавление структурных пружин
        # Горизонтальные точки
        for i in range(0, self.cloth_lenght):
            for j in range(0, self.cloth_wight - 1):
                self.add_spring((i * self.cloth_wight) + j, 
                                (i * self.cloth_wight) + j + 1, 
                                self.KS_STRUCTURAL, self.KD_STRUCTURAL, 
                                self.STRUCTURAL_SPRING_TYPE)

        # Вертикальные точки
        for i in range(0, self.cloth_wight):
            for j in range(0, self.cloth_lenght - 1):
                self.add_spring((j * self.cloth_wight) + i, 
                                ((j + 1) * self.cloth_wight) + i, 
                                self.KS_STRUCTURAL, self.KD_STRUCTURAL, 
                                self.STRUCTURAL_SPRING_TYPE)

        # Добавление сдвиговых пружин
        for i in range(0, self.cloth_lenght - 1):
            for j in range(0, self.cloth_wight - 1):
                self.add_spring( (i * self.cloth_wight) + j, 
                                ((i + 1) * self.cloth_wight) + j + 1, 
                                self.KS_SHEAR, self.KD_SHEAR, 
                                self.SHEAR_SPRING_TYPE)
                self.add_spring( ((i + 1) * self.cloth_wight) + j, 
                                (i * self.cloth_wight) + j + 1, 
                                self.KS_SHEAR, self.KD_SHEAR, 
                                self.SHEAR_SPRING_TYPE)

        # Добавление сдвигающих пружин по горизонтали
        for i in range(0, self.cloth_lenght ):
            for j in range(0, self.cloth_wight - 2):
                self.add_spring((i * self.cloth_wight) + j, 
                                (i * self.cloth_wight) + j + 2, 
                                self.KS_BEND, self.KD_BEND, 
                                self.BEND_SPRING_TYPE)
            self.add_spring((i * self.cloth_wight) + (self.cloth_wight - 3), 
                            (i * self.cloth_wight) + (self.cloth_wight - 1), 
                            self.KS_BEND, self.KD_BEND, self.BEND_SPRING_TYPE)

        # Добавление сгибающихся пружин по вертикали
        for i in range(0, self.cloth_wight):
            for j in range(0, self.cloth_lenght - 2):
                self.add_spring((j * self.cloth_wight) + i, 
                                ((j + 2) * self.cloth_wight) + i, 
                                self.KS_BEND, self.KD_BEND, 
                                self.BEND_SPRING_TYPE)
            self.add_spring(((self.cloth_lenght - 3) * self.cloth_wight) + i, 
                            ((self.cloth_lenght - 1) * self.cloth_wight) + i, 
                            self.KS_BEND, self.KD_BEND, self.BEND_SPRING_TYPE)

    def normalize(self, v:list) -> list:
        '''
            Скомунизженно из функции glm::normalize()
        '''
        len_of_v = math.sqrt( (v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]) )
        if len_of_v == 0:
            return Vector([0, 0, 0])
        else:
            return Vector([v[0] / len_of_v, v[1] / len_of_v, v[2] / len_of_v])

    def constraintForce(self, velocity, SPRING_TYPE, index):
        # SPRING_TYPE = 1.0f: structure
        # SPRING_TYPE = 1.41421356237f: shear
        # SPRING_TYPE = 2.0f: bend
        len_vel = velocity[0] + velocity[1] + velocity[2]

        MxL = 0.03 # молю бога узнать, что же это
        k = 100 # жёскость
        rLen = 0.02 # rest len - длинна в покое
        rLen = self.springs[index].rest_length

        # print(f"return = {Vector(self.normalize(velocity)) * k * (len_vel - SPRING_TYPE * rLen)}")
        # a = Vector(self.normalize(velocity)) * float((k * (len_vel * \
        #                     MxL - len_vel - SPRING_TYPE * \
        #                     self.springs[index].rest_length) + \
        #                     k * 1.4 * (len_vel - SPRING_TYPE * MxL)))
        # print(f"return_2 = {a}")

        if len_vel < (SPRING_TYPE * MxL):
            return self.normalize(velocity) * k * (len_vel - SPRING_TYPE * rLen)
        else:
            return self.normalize(velocity) * float((k * (len_vel * \
                            MxL - len_vel - SPRING_TYPE * rLen) + \
                            k * 1.4 * (len_vel - SPRING_TYPE * MxL)))

    def computeInnerForce(self, i, x, y):
        # х - позиция по длинне ткани
        # у - позиция по ширене ткани
        #                ^ y-
        #                |
        #                |
        #      x- -------*-------> x+
        #                |
        #                |
        #                | y+

        curPos = Vector(self.vertices[i].local_position)
        innF = Vector([0, 0, 0])
        tempV = 0

        sp = self.springs[i]
        print("len(self.springs) = ",len(self.springs))
        print(sp)

        id = (x * self.cloth_wight) + y                  
        ######## structure ########
        #left
        if x < 1:
            pass
        else:
            for o in self.vertices:
                if o.pos_x - 1 == x and o.pos_y == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = second_pos - curPos
                    innF += self.constraintForce(tempV, 1, i)

                    print(f"\nspring_types = {sp.spring_type}" )
                    print(f"second_pos = {second_pos}")
                    print(f"curPos = {curPos}")
                    print(f"tempV = {tempV}")
                    print(f"innF = {innF}\n")

        #right
        if (x + 1) > (self.cloth_lenght - 1): 
            pass 
        else:
            for o in self.vertices:
                if o.pos_x + 1 == x and o.pos_y == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 1, i)

                    print(f"\nspring_types = {sp.spring_type}" )
                    print(f"second_pos = {second_pos}")
                    print(f"curPos = {curPos}")
                    print(f"tempV = {tempV}")
                    print(f"innF = {innF}\n")
        
        #up
        if y < 1: 
            pass 
        else:
            for o in self.vertices:
                if o.pos_x == x and o.pos_y - 1 == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 1, i)

                    print(f"\nspring_types = {sp.spring_type}" )
                    print(f"second_pos = {second_pos}")
                    print(f"curPos = {curPos}")
                    print(f"tempV = {tempV}")
                    print(f"innF = {innF}\n")

        
        #down
        if (y + 1) > (self.cloth_wight - 1): 
            pass
        else:
            for o in self.vertices:
                if o.pos_x + 1 == x and o.pos_y + 1 == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 1, i)

                    print(f"\nspring_types = {sp.spring_type}" )
                    print(f"second_pos = {second_pos}")
                    print(f"curPos = {curPos}")
                    print(f"tempV = {tempV}")
                    print(f"innF = {innF}\n")
        

        ######## shear neighbor ########
        #left up
        if (x) < 1 or (y < 1):
            pass 
        else:
            for o in self.vertices:
                if o.pos_x - 1 == x and o.pos_y - 1 == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 1.41421356237, i)

                    print(f"\nspring_types = {sp.spring_type}" )
                    print(f"second_pos = {second_pos}")
                    print(f"curPos = {curPos}")
                    print(f"tempV = {tempV}")
                    print(f"innF = {innF}\n")
        
        #left down
        if (x < 1) or (y + 1 > self.cloth_wight - 1):
            pass 
        else:
            for o in self.vertices:
                if o.pos_x - 1 == x and o.pos_y + 1 == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 1.41421356237, i)

                    print(f"\nspring_types = {sp.spring_type}" )
                    print(f"second_pos = {second_pos}")
                    print(f"curPos = {curPos}")
                    print(f"tempV = {tempV}")
                    print(f"innF = {innF}\n")
        
        #right up
        if (x + 1 > self.cloth_lenght - 1) or (y < 1):
            pass 
        else:
            for o in self.vertices:
                if o.pos_x + 1 == x and o.pos_y - 1 == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 1.41421356237, i)

                    print(f"\nspring_types = {sp.spring_type}" )
                    print(f"second_pos = {second_pos}")
                    print(f"curPos = {curPos}")
                    print(f"tempV = {tempV}")
                    print(f"innF = {innF}\n")
        
        #right down
        if (x + 1 > self.cloth_lenght - 1) or (y + 1 > self.cloth_wight - 1):
            pass 
        else: 
            for o in self.vertices:
                if o.pos_x + 1 == x and o.pos_y + 1 == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 1.41421356237, i)        

        ######## bend neighbor ########
        #left 2
        if x < 2:
            pass 
        else:
            for o in self.vertices:
                if o.pos_x - 2 == x and o.pos_y == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 2, i)
        
        #right 2
        if (x + 2) > self.cloth_lenght - 1:
            pass 
        else: 
            for o in self.vertices:
                if o.pos_x + 2 == x and o.pos_y == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 2, i)
        
        #up 2
        if y < 2:
            pass 
        else: 
            for o in self.vertices:
                if o.pos_x == x and o.pos_y - 2 == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 2, i)
        
        #down 2
        if ((y + 2) > self.cloth_wight - 1):
            pass 
        else: 
            for o in self.vertices:
                if o.pos_x == x and o.pos_y + 2 == y:
                    second_pos = Vector(self.vertices[o.pos_x].local_position)
                    tempV = list(second_pos - curPos )
                    innF += self.constraintForce(tempV, 2, i)

        return innF

    # def computeInnerForce(self, i) :
        # #                ^ y-
        # #                |
        # #                |
        # #      x- -------*-------> x+
        # #                |
        # #                |
        # #                | y+

        # x = self.vertices[i].pos_x
        # y = self.vertices[i].pos_y
        # curPos = Vector(self.vertices[i].local_position)
        # innF = Vector([0, 0, 0])
        # tempV = 0

        # sp = self.springs[i]
        # # print(sp.spring_type)

        # ######## structure ########
        # #left
        # if x < 1:
        #     pass
        # else:
        #     for o in self.springs:
        #         if o.pos_x - 1 == x and o.pos_y == y:

        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = second_pos - curPos
        #             innF += self.constraintForce(tempV, 1, i)

        #             print(f"\nspring_types = {sp.spring_type}" )
        #             print(f"second_pos = {second_pos}")
        #             print(f"curPos = {curPos}")
        #             print(f"tempV = {tempV}")
        #             print(f"innF = {innF}\n")

        # #right
        # if (x + 1) > (self.cloth_lenght - 1): 
        #     pass 
        # else:
        #     for o in self.springs:
        #         if o.pos_x + 1 == x and o.pos_y == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 1, i)

        #             print(f"\nspring_types = {sp.spring_type}" )
        #             print(f"second_pos = {second_pos}")
        #             print(f"curPos = {curPos}")
        #             print(f"tempV = {tempV}")
        #             print(f"innF = {innF}\n")
        
        # #up
        # if y < 1: 
        #     pass 
        # else:
        #     for o in self.springs:
        #         if o.pos_x == x and o.pos_y - 1 == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 1, i)

        #             print(f"\nspring_types = {sp.spring_type}" )
        #             print(f"second_pos = {second_pos}")
        #             print(f"curPos = {curPos}")
        #             print(f"tempV = {tempV}")
        #             print(f"innF = {innF}\n")

        
        # #down
        # if (y + 1) > (self.cloth_wight - 1): 
        #     pass
        # else:
        #     for o in self.springs:
        #         if o.pos_x + 1 == x and o.pos_y + 1 == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 1, i)

        #             print(f"\nspring_types = {sp.spring_type}" )
        #             print(f"second_pos = {second_pos}")
        #             print(f"curPos = {curPos}")
        #             print(f"tempV = {tempV}")
        #             print(f"innF = {innF}\n")
        

        # ######## shear neighbor ########
        # #left up
        # if (x) < 1 or (y < 1):
        #     pass 
        # else:
        #     for o in self.springs:
        #         if o.pos_x - 1 == x and o.pos_y - 1 == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 1.41421356237, i)

        #             print(f"\nspring_types = {sp.spring_type}" )
        #             print(f"second_pos = {second_pos}")
        #             print(f"curPos = {curPos}")
        #             print(f"tempV = {tempV}")
        #             print(f"innF = {innF}\n")
        
        # #left down
        # if (x < 1) or (y + 1 > self.cloth_wight - 1):
        #     pass 
        # else:
        #     for o in self.springs:
        #         if o.pos_x - 1 == x and o.pos_y + 1 == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 1.41421356237, i)

        #             print(f"\nspring_types = {sp.spring_type}" )
        #             print(f"second_pos = {second_pos}")
        #             print(f"curPos = {curPos}")
        #             print(f"tempV = {tempV}")
        #             print(f"innF = {innF}\n")
        
        # #right up
        # if (x + 1 > self.cloth_lenght - 1) or (y < 1):
        #     pass 
        # else:
        #     for o in self.springs:
        #         if o.pos_x + 1 == x and o.pos_y - 1 == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 1.41421356237, i)

        #             print(f"\nspring_types = {sp.spring_type}" )
        #             print(f"second_pos = {second_pos}")
        #             print(f"curPos = {curPos}")
        #             print(f"tempV = {tempV}")
        #             print(f"innF = {innF}\n")
        
        # #right down
        # if (x + 1 > self.cloth_lenght - 1) or (y + 1 > self.cloth_wight - 1):
        #     pass 
        # else: 
        #     for o in self.springs:
        #         if o.pos_x + 1 == x and o.pos_y + 1 == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 1.41421356237, i)        

        # ######## bend neighbor ########
        # #left 2
        # if x < 2:
        #     pass 
        # else:
        #     for o in self.springs:
        #         if o.pos_x - 2 == x and o.pos_y == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 2, i)
        
        # #right 2
        # if (x + 2) > self.cloth_lenght - 1:
        #     pass 
        # else: 
        #     for o in self.springs:
        #         if o.pos_x + 2 == x and o.pos_y == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 2, i)
        
        # #up 2
        # if y < 2:
        #     pass 
        # else: 
        #     for o in self.springs:
        #         if o.pos_x == x and o.pos_y - 2 == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 2, i)
        
        # #down 2
        # if ((y + 2) > self.cloth_wight - 1):
        #     pass 
        # else: 
        #     for o in self.springs:
        #         if o.pos_x == x and o.pos_y + 2 == y:
        #             second_pos = Vector(self.vertices[o.pos_x].local_position)
        #             tempV = list(second_pos - curPos )
        #             innF += self.constraintForce(tempV, 2, i)

        # return innF

    def computeForceNet(self, x, y, pov) -> list:
        '''
            pov - position of the vertex
            F = m*g + Fwind - air * vel* vel + innF - damp = m*Acc
        '''
        # print("\n\tcomputeForceNet")

        # Тут и считаются ограничения на растягивание
        inner_forse = self.computeInnerForce(pov, x, y)
        print(f"inner_forse = {inner_forse}")

        vel = self.vertices[pov].velocity

        # Длинна вектора
        len_vel = vel[0] + vel[1] + vel[2]

        # print(f"len_vel = {len_vel}")

        acceleration = self.vertices[pov].acceleration
        print(f"acceleration = {acceleration}")

        wind_forse = [0, 0, 0]

        net_forse = [
                        +1 * self.mass * self.GRAVITY[xyz] + \
                        +1 * wind_forse[xyz] + \
                        -1 * vel[xyz] * acceleration[xyz] * len_vel + \
                        +1 * inner_forse[xyz] + \
                        -1 * self.DEFAULT_DAMPING * vel[xyz] * len_vel
                             for xyz in range(3)
                     ]
            
        return net_forse

    def ComputeForces(self, x, y, id) -> list:
        '''
            Начинает просчёт сил действующих на точки
            pov - position_of_vertex
        '''
        # print("\n\tComputeForces")
        # Берём позицию точки и предыдущую позицию
        pos = self.vertices[id].local_position
        if id >= 1:
            pos_last = self.last_coo[id]
        else:
            pos_last = self.last_coo[id]

        # print(f"pos = {pos}")
        # print(f"pos_last = {pos_last}")

        net_forse = Vector(self.computeForceNet(x, y, id))
        print(f"net_forse = {net_forse}")

        # print("\n\tComputeForces")
        acceleration = net_forse / self.mass
        print(f"acceleration_f = {acceleration}")

        vel = Vector(self.vertices[id].velocity) + Vector(acceleration) * self.dt
        print(f"vel_f = {vel}")

        next_pos = self.vertices[id].local_position
        id = self.vertices[id].id

        if id == 25:
            # self.vertices[id].has_collision == True
            next_pos[0] += 0
            next_pos[1] += 0
            next_pos[2] += 0
        else:
            next_pos = self.IntegrateVerlet(pos, pos_last, acceleration)

        print(f"pos = {pos}")
        print(f"next_pos = {next_pos}")

        return next_pos
        
    ############################################ РАБОТАЕТ НА КОРРЕКТНО!!! ############################################
            # # Создаём локальную переменную для хранения скорости точки
            # vel = self.get_Vertlet_velocity(self.vertices[num].local_position, self.last_coo[num])

            # if num != 0 and num != self.cloth_wight:
            #     self.vertices[num].velocity= [1000 * self.GRAVITY[xyz] * self.mass for xyz in range(3)]

            # self.vertices[num].velocity= [self.vertices[num].velocity[xyz] + \
            #                                 self.DEFAULT_DAMPING * vel[xyz] 
            #                                 for xyz in range(3)]       
                
            # # Бежим по всем пужинам в поисках счастья
            # for i in range(0, len(self.springs)):

            #     # Ныняшняя позиция перой точки
            #     p_1 = self.vertices[self.springs[i].pos_x].local_position

            #     # Прошлая позиция второй точки
            #     p_1_last = self.last_coo[self.springs[i].pos_x]

            #     p_2 = self.vertices[self.springs[i].pos_y].local_position
            #     p_2_last = self.last_coo[self.springs[i].pos_y]

            #     # Скорость первой и второй точки
            #     v_1 = self.get_Vertlet_velocity(p_1, p_1_last)
            #     v_2 = self.get_Vertlet_velocity(p_2, p_2_last)

            #     # ForceNet - Равнодействующая сила
            #     forceNet = computeForceNet(num)

            #     # Дельта расстояния
            #     delta_p = [p_1[xyz] - p_2[xyz] 
            #                for xyz in range(3)] # p1 - p2

            #     # Дельта скорости
            #     delta_v = [v_1[xyz] - v_2[xyz] 
            #                for xyz in range(3)] # v1 - v2

            #     dist = math.sqrt(delta_p[0] * delta_p[0] + \
            #                      delta_p[1] * delta_p[1] + \
            #                      delta_p[2] * delta_p[2])
                
            #     if dist != 0:
            #         # Сила упругости  F = -k*dl
            #         left_term = -(self.springs[i].ks * (dist - self.springs[i].rest_length))/10

            #         # print("ks = ", self.springs[i].ks)
            #         # print("rest_length = ", self.springs[i].rest_length)
            #         # print("left_term = ", left_term)

            #         # Сила трения 
            #         right_term = self.springs[i].kd * ((delta_p[0] * delta_v[0] + delta_p[1] * delta_v[1] + delta_p[2] * delta_v[2]) / dist)
                    
            #         # print("delta_p[0]*+ = ", delta_p[0] * delta_v[0] + delta_p[1] * delta_v[1] + delta_p[2] * delta_v[2])
            #         # print("kd = ", self.springs[i].kd)
            #         # print("right_term = ", right_term)                
                    
            #         # (delta_p[xyz]/dist) - относительное удлиннение
            #         # (left_term + right_term) - механическое напряжение
            #         spring_force = [(left_term + right_term) * (delta_p[xyz]/dist) for xyz in range(3)]

            #         # print("spring_force = ", spring_force, "\n\n")

            #         if self.springs[i].pos_x != 0 and self.springs[i].pos_x != self.cloth_wight:
            #             self.vertices[self.springs[i].pos_x].set_velocity(
            #                                                                 [
            #                                                                     self.vertices[self.springs[i].pos_x].velocity[xyz] + \
            #                                                                     spring_force[xyz]
            #                                                                     for xyz in range(3)
            #                                                                 ]
            #                                                             )
                    
            #         if self.springs[i].pos_y != 0 and self.springs[i].pos_y != self.cloth_wight:
            #             self.vertices[self.springs[i].pos_y].set_velocity(
            #                                                                 [
            #                                                                     self.vertices[self.springs[i].pos_y].velocity[xyz] - \
            #                                                                     spring_force[xyz]
            #                                                                     for xyz in range(3)
            #                                                                 ]
            #                                                             )
    ############################################ РАБОТАЕТ НА КОРРЕКТНО!!! ############################################

    def cloth_deformation(self, K = 2, Cd = 0.0007) -> list:
        ''' 
        Здесь производится деформация ткнани 

        Пружинные силы(Spring forces) - при заданной пружине, 
        соединяющей две частицы, расположенные в точках p и q, 
        с жесткостью K и длиной покоя L0, сила пружины, действующая на p.

        K - это vec3, где K [0] , K [1] и K [2] обозначают жесткость всех структурных, 
                                        сдвигающих и изгибающих пружин, соответственно.
        L - вычисляем через функция len_of_two_points

        F(spring) = K*(L0−abs(p−q))*(p−q/abs(p−q)).

        Гравитация(Gravity) - self.gravity

        Затухание(Damping) - типа трения о воздух

        F(Damping) = -Cd * V.
        Cd - это коэффициент силы сопротивления движению
        хз, где его взять, так что пусть это будет 0.0007

        Вязкая жидкость(Viscous fluid) - чтобы справиться с вязким поведением ткани, 
                    мы предполагаем, что каждая частица со скоростью v выталкивается 
                    воображаемой вязкой жидкостью.
        '''

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
        global lib
        lib = cdll.LoadLibrary(filename)
        lib.print_work_info()
    except OSError:
        print("Не удаётся установить соединение с DLL файлом")

# ------------------------------------------------------------------------
#    UI
# ------------------------------------------------------------------------


class GPUCloth_Settings(PropertyGroup):
    '''
    This class defines the values ​​of each input. 
    We can change them byreferring to specific variables corresponding to a given input.
    There is also a description for each input.
    '''
    bool_object_coll : BoolProperty(
        name="Enable or Disable",
        description="A bool property",
        default = False
        )

    #    Object_Collizions = bpy.context.scene.tool.bool_object_coll
     
    bool_self_coll : BoolProperty(
        name="Enable or Disable",
        description="A bool property",
        default = False
        )
    
    #    Self_Collizions = bpy.context.scene.tool.bool_self_coll

    int : IntProperty(
        name = "Set a value",
        description="A integer property",
        default = 23,
        min = 10,
        max = 100
        )

    # Object_Collizions = bpy.context.scene.tool.bool_object_coll

    float_vertex : FloatProperty(
        name = "Vertex Mass",
        description = "The mass of each vertex on the cloth material",
        default = 0.33,
        min = 0.001,
        max = 30.0
        )

    #    Vertex_Mass = bpy.context.scene.tool.float_vertex

    float_speed : FloatProperty(
        name = "Speed Multiplier",
        description = "Close speed is multiplied bythis value",
        default = 0.1000,
        min = 0.001,
        max = 30.0
        )

    #    Speed_Multiplier = bpy.context.scene.tool.float_speed

    float_air : FloatProperty(
        name = "Air Viscosity",
        description = "Air has some thickness which slows falling things down",
        default = 0.1000,
        min = 0.001,
        max = 30.0
        )

    #    Air_Viscosity = bpy.context.scene.tool.float_air
    
    float_distance : FloatProperty(
        name = "Minimal Distance",
        description = "The distance another object must get to the cloth for the simulation to repel the cloth out of the way",
        default = 0.015,
        min = 0.001,
        max = 30.0
        ) 
        
    #    Minimal_Distance = bpy.context.scene.tool.float_distance
    
    float_distance_self : FloatProperty(
        name = "Self Minimal Distance",
        description = "The distance another object must get to the cloth for the simulation to repel the cloth out of the way",
        default = 0.015,
        min = 0.001,
        max = 30.0
        ) 
    
    #    Self_Minimal_Distance = bpy.context.scene.tool.float_distance_self

    float_friction : FloatProperty(
        name = "Friction",
        description = "A coefficient for how slippery the cloth is when it collides with itself",
        default = 5.000,
        min = 0.001,
        max = 30.0
        ) 

    #    Friction = bpy.context.scene.tool.float_friction

    float_impulse : FloatProperty(
        name = "Impulse Clamp",
        description = "Prevents explosions in tight and complicated collision situations by restricting the amount of movement after a collision",
        default = 0.000,
        min = 0.000,
        max = 30.0
        ) 

    #    Impulse_Clamp = bpy.context.scene.tool.float_impulse
    

    float_impulse_self : FloatProperty(
        name = "Self Impulse Clamp",
        description = "Prevents explosions in tight and complicated collision situations by restricting the amount of movement after a collision.",
        default = 0.000,
        min = 0.000,
        max = 30.0
        ) 

    #    Self_Impulse_Clamp = bpy.context.scene.tool.float_impulse_self

    string : StringProperty(
        name = "Set a value",
        description = "A string property",
        default = "None"
        )

    # Self_Impulse_Clamp = bpy.context.scene.tool.float_impulse_self

#    --- GPUCloth in Properties window ---

class UV_PT_GPUCloth(Panel):
    '''
    This is a main (parent) Panel on T-panel. 
    The main interface and nested panels of the created fabric are assembled here.
    '''
    bl_idname = "UV_PT_GPUCloth"
    bl_label = "GPUCloth"
    bl_category = "My Category"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.tool
        split = layout.split(factor=0.4)
        col_1 = split.column()
        col_2 = split.column()
        
        # display the properties
        col_1.label(text="Vertex Mass")
        col_2.prop(mytool, "float_vertex", text="")
        col_1.label(text="Speed Multiplier")
        col_2.prop(mytool, "float_speed", text="")
        col_1.label(text="Air Viscosity")
        col_2.prop(mytool, "float_air", text="")
        
class UV_PT_GPUCloth_ObjColl(Panel):
    '''
    This is a child Panel on GPUCloth.
    This panel is checkbox.
    '''
    bl_idname = "UV_PT_GPUCloth_ObjColl"
    bl_label = ""
    bl_parent_id = "UV_PT_GPUCloth"
    bl_category = "My Category"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self,context):
        self.layout.prop(context.scene.render, "use_border", text="Object Collizions")
        
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.tool
        split = layout.split(factor=0.4)
        col_1 = split.column()
        col_2 = split.column()

        # display the properties
        col_1.label(text="Distance (min)")
        col_2.prop(mytool, "float_distance", text="")
        col_1.label(text="Impulse Clamp")
        col_2.prop(mytool, "float_impulse", text="")
        
        if bpy.context.scene.render.use_border == False:
            col_1.enabled = False
            col_2.enabled = False
            
        if bpy.context.scene.render.use_border == True:
            col_1.enabled = True
            col_2.enabled = True

class UV_PT_GPUCloth_SelfColl(Panel):
    '''
    This is a child Panel on GPUCloth.
    This panel is checkbox.
    '''
    bl_idname = "UV_PT_GPUCloth_SelfColl"
    bl_label = ""
    bl_parent_id = "UV_PT_GPUCloth"
    bl_category = "My Category"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self,context):
        self.layout.prop(context.scene.render, "use_placeholder", text="Self Collizions")
        
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.tool
        split = layout.split(factor=0.4)
        col_1 = split.column()
        col_2 = split.column()

        # display the properties
        col_1.label(text="Friction")
        col_2.prop(mytool, "float_friction", text="")
        col_1.label(text="Distance (m)")
        col_2.prop(mytool, "float_distance_self", text="")
        col_1.label(text="Impulse Clamp")
        col_2.prop(mytool, "float_impulse_self", text="")
        
        if bpy.context.scene.render.use_placeholder == False:
            col_1.enabled = False
            col_2.enabled = False
            
        if bpy.context.scene.render.use_placeholder == True:
            col_1.enabled = True
            col_2.enabled = True

#     --- Registration UI Panel ---

classes = (
    GPUCloth_Settings,
    UV_PT_GPUCloth,
    UV_PT_GPUCloth_ObjColl,
    UV_PT_GPUCloth_SelfColl,
)

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Scene.tool = PointerProperty(type=GPUCloth_Settings)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Scene.tool


if __name__ == "__main__":
    register()

    # Проверка работоспособности CUDA
    # Проверка dll для запуска симуляции
    # loadDLL()
    print("\n\n\n\n\n\n\n\n\n\n\n")

    # Переход на первый кадр
    bpy.context.scene.frame_current = bpy.context.scene.frame_start

    # Создание экземпляра класса физики
    sim = Physics()

    # Запуск симуляции физики
    sim.start_sim()

    # Запуск анимации
    for i in range(10):
            bpy.context.scene.frame_current = i

    # bpy.ops.screen.animation_play(reverse=False, sync=False)
