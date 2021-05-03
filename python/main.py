'''
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

В данном файле происходит запуск файлов создания сцены и запуска симуляции

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
-пружины сжатия/растяжения;         (STRUCTURAL)
-пружины кручения;                  (SHEAR)
-пружины изгиба.                    (BEND)

'''

bl_info = {
    "name": "GPUCloth",
    "author": "PorisulkiP, GoldBath, Vi, Victus",
    "version": (0, 1, 0),
    "blender": (2, 93, 0),
    "location": "",
    "warning": "",
    "description": "Cloth simulation on GPU",
    "doc_url": "https://github.com/PorisulkiP/GPUCloth",
    "category": "System",
}

# импортируем API для работы с blender
import bpy
import numpy as np
import sys, os, math, collections, mathutils, time

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

            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

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

Points = collections.namedtuple('Points', ['velocity', 'acceleration', 'is_collide'])

class Point:
    '''
    Класс реализуется по шаблону "Стратегия"
    Суть класса в усовершенствовании базовых точек из блендера, для более точной симуляции ткани
    _meshResolution  - кол-во точек на ткани
    __vertexMass - масса каждой точки
    [0] vertexVelocity - скорость каждой точки
    [1] vertexAcceleration - ускорение каждой точки
    __vertexPosition - координаты каждой точки
    [2] vertex_Is_collide - сталкивается ли точка с другими объектами

    :Point.get all()[0] = [0.003, 0.001, 0.002] - масса всех точек
    :Point.get all()[0][0] = 0.003
    '''
    _meshResolution  = len(bpy.data.objects["Plane"].data.vertices)

    __vertexMass = 0.3 / _meshResolution 
    __vertexVelocity  = [[0,0,0]] * _meshResolution 
    __vertexAcceleration  = [[0,0,0]] * _meshResolution 
    __vertexIsCollide  = [False] * _meshResolution

    # Длинна ткани по x и по y
    _sim_u = int(_meshResolution**0.5) # Горизонталь
    _sim_v = int(_meshResolution**0.5) # Вертикаль
    
    distance_min = 0.0001

    # __slots__ = ['__vert_append', '__xyz', '__vertexPosition']

    def __init__(self):
        super().__init__()

        # Этот вариант более подходящий, но выполняется слишком долго :(
        self.__vert_append = [Points(velocity, acceleration, is_collide) 
                                     for velocity in self.__vertexVelocity 
                                     for acceleration in self.__vertexAcceleration 
                                     for is_collide in self.__vertexIsCollide]

        self.__vertexPosition = [bpy.data.objects["Plane"].matrix_world @ 
                                        bpy.data.objects["Plane"].data.vertices[i].co 
                                        for i in range(0, self._meshResolution)]

        # print("__vert_append = ", self.__vert_append, "__vertexPosition = ", self.__vertexPosition, sep="\n")

    def __len__(self) -> int:
        ''' Возвращяет кол-во точек на ткани'''
        return self._meshResolution 
       
    def __repr__(self):
        return 'All properties of points: ' % self.__vert_append

    def __getiitem__(self, position:int) -> list:
        ''' Возвращает параметры конкретной точки'''
        return self.__vert_append[position]

    # ------------------------------------------------------------------------
    #    GETTERS
    # ------------------------------------------------------------------------

    
    def get_velocity(self, position:int) -> list:
        ''' Возвращает список(list) скорости каждой точки '''
        return self.__vert_append[position].velocity
    
    def get_acceleration(self, position:int) -> list:
        ''' Возвращает список(list) ускорения каждой точки '''
        return self.__vert_append[position].acceleration
        
    def get_position(self, position:int) -> list:
        ''' Возвращает список(list) координат конкретной точки '''
        return self.__vertexPosition[position]
    
    def get_is_collide(self, position:int) -> bool:
        ''' Возвращает факт(bool) пересейчения каждой точки '''
        return self.__vert_append[position].is_collide

    @property 
    def get_sim_u(self) -> int:
        ''' Возвращает кол-во(int) точек по горизонтали '''
        return self._sim_u

    @property 
    def get_sim_v(self) -> int:
        ''' Возвращает кол-во(int) точек по вертикали '''
        return self._sim_v

    @property   
    def all_append(self) -> list:
        ''' Возвращает список(list) всех свойств точек'''
        return self.__vert_append

    @property 
    def mass(self) -> float:
        ''' Возвращает массу(float) точек '''
        return self.__vertexMass

    @property
    def get_all_coord(self) -> list:
        ''' Возвращает список(list) всех доступных координат '''
        return self.__vertexPosition

    @property
    def creating_backUp(self) -> list:
        ''' Возвращает список(list) бэкап точек '''
        return [self.get_all_coord[i] for i in range(0, self._meshResolution )]

    @property
    def get_meshResolution(self) -> int:
        ''' 
        Возвращает число точек на ткани
        
        :self.mesh._meshResolution >> 8 
        '''
        return self._meshResolution 

    # ------------------------------------------------------------------------
    #    SETTERS
    # ------------------------------------------------------------------------

    # Не удалось выставить сеттеры с помощью property, 
    # поскольку property не даёт передавать больше одного параметра
    def set_velocity(self, position:int, velocity:list) -> None:
        ''' Устновка скорости для конкретных точек '''
        self.__vert_append[position]._replace(velocity=velocity)
    
    def set_acceleration(self, position:int, acceleration:list) -> None:
        ''' Установка ускорения для конкретных точек '''
        self.__vert_append[position]._replace(acceleration=acceleration)

    def set_coo(self, position:int, new_coo:list) -> None:
        ''' Добавленик к существующим координатам значений '''
        new_coo = mathutils.Vector(new_coo)
        self.__vertexPosition[position] = new_coo
        bpy.data.objects["Plane"].data.vertices[position].co = new_coo

    def add_coo(self, position:int, new_coo:list) -> None:
        ''' Добавленик к существующим координатам значений '''
        new_coo = mathutils.Vector(new_coo)
        self.__vertexPosition[position] += new_coo
        bpy.data.objects["Plane"].data.vertices[position].co += new_coo

    def set_backUp(self, backUp:list) -> None:
        ''' Устновка координат точек из бэкапа '''
        print("\n\nЗапуск бэкапа\n\n")
        print("meshResolution  = ", self.get_meshResolution)
        print("vertexPosition = ", self.__vertexPosition)
        print("backUp = ", backUp)
        print("pointCo" , bpy.data.objects["Plane"].data.vertices[0].co)
        for i in range(0, self.get_meshResolution ):
            self.vertexPosition[i] = backUp[i]
            print("backUp[i] = ",backUp[i])
            
            bpy.data.objects["Plane"].data.vertices[i].co = backUp[i]

    def set_is_collide(self, position:int, is_collide:bool) -> None:
        ''' Установка параметра столкновения '''
        self.__vert_append[position]._replace(is_collide=is_collide)


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

    def get_collide_obj(self) -> list:
        ''' Возвращает массив объектов с которыми может столкнуться ткань '''
        self.__collisionOBJ()
        return self.__collisionOBJs

    def get_gpu_cloth_obj(self) -> list:
        ''' Возвращает массив объектов, на которых накинут можификатор ткани  '''
        self.__clothOBJ()
        return self.__clothOBJs


class Spring:
    '''
    Класс пружины(нитки, но не совсем)
    pos_a: первая позиция
    pos_b: вторая позиция
    ks: постоянная пружины
    kd: типа, отклонение от постоянной, но это не точно...
    rest_length(l0): длина пружины в покое
    STIFFNESS: жёсткость
    '''
    STIFFNESS = 1

    def __init__(self, a, b, ks, kd, rest_length, spring_type):
        self.pos_a = a
        self.pos_b = b
        self.ks = ks
        self.kd = kd
        self.rest_length = rest_length
        self.spring_type = spring_type

    def __repr__(self):
        return f'''\nSpring: \n
                    Current position: {self.pos_a}\n
                    Previous position: {self.pos_b}\n
                    ks: {self.ks}\n
                    kd: {self.kd}\n
                    Rest length: {self.rest_length}\n
                    Spring type: {self.spring_type}\n       
                '''


class Accurate:
    def __init__(self, stepSize = 1):
        super().__init__()
        # h - кол-во шагов, т.е stepSize
        self.h = stepSize

    def _solved_equation(self, xi, yi):
        '''  '''
        return (xi**2-xi)/yi

    def _next_y(self, xi, yi):
        """
        Считает y[i+1] следующим образом:
            y[i+1] = f(x[i+1])
        P.S.
        Функция вынесена таким образом, чтобы в след. методах (классах)
        можно было просто перегрузить ее и получить новый метод не дублируя код.
        :param xi: x[i]
        :param yi: y[i]
        :return: y[i+1]
        """
        return self._solved_equation(xi)

    def calculate(self, x0, y0, xf):
        """
        Вычисляет значения на промежуте [x0;xf] с шагом h выражения f
        :param x0:
        :param y0:
        :param xf:
        :return: список значений приближения для промежутка [x0;xf]
        """
        ys = []
        xs = np.arange(x0 + self.h, xf + self.h, self.h)  # вектор всех значений x
        y = y0
        for x in xs:
            ys.append(y)
            y = self._next_y(x, y)
        return ys


class Euler(Accurate):
    def _next_y(self, xi, yi):
        """
        Считает y[i+1] исходя из x[i] и y[i] следующим образом:
            y[i+1] = y[i] + h * f(xi, yi)
        :param xi: x[i]
        :param yi: y[i]
        :return: y[i+1]
        """
        return yi + self.h * self._solved_equation(xi, yi)


class RungeKutta(Euler):
    def _next_y(self, xi, yi):
        """
        Считает y[i+1] исходя из x[i] и y[i] следующим образом:
            y[i+1] = y[i] + h/6 * (k1 + 2k2+ 2k3 + k4)
            k1 = f(xi, yi)
            k2 = f(xi + h/2, yi + h/2 * k1)
            k3 = f(xi + h/2, yi + h/2 * k2)
            k4 = f(xi + h, yi + h * k3)
        :param xi: x[i]
        :param yi: y[i]
        :return: y[i+1]
        """
        h2 = self.h / 2
        k1 = self._solved_equation(xi, yi)
        k2 = self._solved_equation(xi + h2, yi + h2 * k1)
        k3 = self._solved_equation(xi + h2, yi + h2 * k2)
        k4 = self._solved_equation(xi + self.h, yi + self.h * k3)
        return yi + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class Physics(RungeKutta, Point, Collision):
   
    STRUCTURAL_SPRING_TYPE = 0
    SHEAR_SPRING_TYPE = 1
    BEND_SPRING_TYPE = 2

    DEFAULT_DAMPING = -0.0125
    VERTEX_SIZE = 4
    VERTEX_SIZE_HALF = 2.0

    # ks - постоянная пружины (Коэффициент упругости)
    KS_STRUCTURAL = 50.75
    KD_STRUCTURAL = -0.25

    KS_SHEAR = 50.75
    KD_SHEAR = -0.25

    KS_BEND = 50.95
    KD_BEND = -0.25

    def __init__(self, vertexes_of_cloth:Point, 
                        backUp:Point.creating_backUp):
        super().__init__()

        # Создаём атрибуты сцены для дальнейших вычислений
        self.gravity = bpy.context.scene.gravity
        self.fps = bpy.context.scene.render.fps
         
        self.mesh = vertexes_of_cloth

        # Так же dt(дельта t) это и есть TIME_STEP
        self.TIME_STEP = 1.0 / self.fps

        # Создание экземпляра метод Рунге-Кутты
        self.kutt = RungeKutta()

        # Создаём атрибут бэкапа
        self.backUp = backUp

        # Создаём переменную где хранятся все связи(нитки, но не совсем) ткани
        self.springs = []
        self.vertices_last = []

        # Собираем все объекты в сцене, способные к столкновению
        self.collision = Collision()
        self.collision_objects = self.collision.get_collide_obj()
        self.gpu_cloth = self.collision.get_gpu_cloth_obj()

    def start_sim(self) -> None:
        ''' Функция запускает просчёты перед показом каждого кадра '''
        bpy.app.handlers.frame_change_pre.clear()
        bpy.app.handlers.frame_change_pre.append(self.сalculation(self.TIME_STEP))

    def сalculation(self, dt) -> None:
        '''
        Данная функция вызывается каждый кадр.
        запускает ещё несколько функция для расчёта физики

        mesh - объект, с симуляциет ткани

        collision_objects - объект(ы) столкновения

        backUp - бэкап ткани до симуляции
        '''
        def checkCollision(scene):
            '''
            Функция вычисляет, насколько должна переместиться точка,
            при заданной массе, гравитации и ускорении
            Для вычисления используется формула уменьшенного веса.
            Р - вес / m - масса / g - гравитация
            Р = m(g-a)
            '''
            flag_num = 0
            flags = []
            # if bpy.context.scene.frame_current == bpy.context.scene.frame_start:
            #     self.mesh.set_backUp(self.backUp)
            # i - одна вершина из всех, что есть в объекте, с симуляциет ткани
            for i in range(0, self.mesh.get_meshResolution-1):

                # obj_vert - меш из списка объектов столкновений
                for obj_vert in self.collision_objects:

                    # о - одна вершина из всех, что есть в объекте столкновения
                    for o in range(0, len(obj_vert.data.vertices)):

                        # 1-я вершина из отрезка
                        global_col_point = obj_vert.matrix_world @ obj_vert.data.vertices[o].co

                        # [0] - x, [1] - y, [2] - z
                        flags.append(self.cross(self.mesh.get_position(i), global_col_point, i))
                        flag_num += 1

            # print("True in flags = ", True in flags)
            if not(True in flags):
                self.preparation()
                self.ComputeForces(dt)
                self.IntegrateVerlet(dt)
            else:
                # if (True in flags):
                #     for newCoor in self.mesh.get_all_coord:
                #         for i in range(0, 3):  
                #             newCoor[i] -= (self.gravity[i] / (60 / self.fps))
                # else:
                for newCoor in self.mesh.get_all_coord:
                    for i in range(0, 3):  
                        newCoor[i] += 0
        return checkCollision

    def add_spring(self, a, b, ks, kd, spring_type):
        s = Spring(a, b, ks, kd, a - b, spring_type)
        self.springs.append(s)

    def lenOfTwoPoints(self, a, b) -> float:
        '''
        Находим расскояние между двух точек,
        через формулу определение длинны вектора
        '''
        x = (b[0] - a[0])**2
        y = (b[1] - a[1])**2
        z = (b[2] - a[2])**2

        return (x+y+z)**0.5

    def get_momentum(self, position1:int) -> list:
        ''' 
        Возвращает список значений, который является импульсом точки.
        Номер является осью, а значение силой импульса по конкретной оси.
        
        :find_momentum(i) >> [0, 1.2356, 0.001235]
        '''
        return [self.__vertexMass * self.get_acceleration(position1) 
                for i in range(0, 3)]

    def get_Vertlet_velocity(self, v_i, v_i_last, dt) -> list:
        '''
        С помощью интеграции верлета находим дифференциал скорости
        '''
        return [(v_i[i] - v_i_last[i]) / float(dt) for i in range(0, 3)]

    def get_Vertlet_acceleration(self, v_i, v_i_last, dt) -> list:
        '''
        Функция возвращает ускорение для одной точки,
        в виде массива по 3 коордиинатам
        a = (V1-V2)/t
        t = 1/fps
        '''
        V1 = self.get_Vertlet_velocity(v_i, v_i_last, dt)
        V2 = self.get_Vertlet_velocity(self.get_Vertlet_velocity(v_i, v_i_last, dt))
        a = [(V1[i] - V2[i])/(60/self.fps) for i in range(0, 3)]
        return a


    def cross(self, a, b, num_a, distance_min = 0.1) -> bool:
        '''
        Задача данной функции – определить пересечения векторов двух точек,
        т.е. найти пересечения вершин, чтобы геометрия не пересекалась друг с другом
        Пераметр distance_min задаётся в настройках ткани во вкладке Collision ==> Object collision 
        ==> Distanse и по умолчанию равен 0.15
        Находится по формуле вычисления точек пересечения прямых в пространстве.
        '''
        # Версия с a-b и c-d
        # print(a, b, sep="\n")
        if (self.lenOfTwoPoints(a, b) >= distance_min):
            collisionX = a[0] < b[0]
            collisionY = a[1] > b[1]
            collisionZ = a[2] < b[2]
            if (collisionX and collisionY and collisionZ):
                self.mesh.set_is_collide(num_a, True)
                return True
        return False

    def preparation(self) -> None:
        '''
        Подготавливанем миллион данных для корректной работы симуляции

        '''
        # Добавляем позицию вершин в список.
        # Из данных в этом списке будем высчитывать скорость методом Стёрмера — Верле
        self.vertices_last = self.mesh.get_all_coord

        # Добавление структурных пружин
        # Горизонтальные точки
        for i in range(0, self.mesh.get_sim_v):
            for j in range(0, self.mesh.get_sim_u - 1):
                self.add_spring((i * self.mesh.get_sim_u) + j, 
                                (i * self.mesh.get_sim_u) + j + 1, 
                                self.KS_STRUCTURAL, self.KD_STRUCTURAL, 
                                self.STRUCTURAL_SPRING_TYPE)

        # Вертикальные точки
        for i in range(0, self.mesh.get_sim_u):
            for j in range(0, self.mesh.get_sim_v - 1):
                self.add_spring((j * self.mesh.get_sim_u) + i, 
                                ((j + 1) * self.mesh.get_sim_u) + i, 
                                self.KS_STRUCTURAL, self.KD_STRUCTURAL, 
                                self.STRUCTURAL_SPRING_TYPE)

        # Добавление сдвиговых пружин
        for i in range(0, self.mesh.get_sim_v - 1):
            for j in range(0, self.mesh.get_sim_u - 1):
                self.add_spring( (i * self.mesh.get_sim_u) + j, 
                                ((i + 1) * self.mesh.get_sim_u) + j + 1, 
                                self.KS_SHEAR, self.KD_SHEAR, 
                                self.SHEAR_SPRING_TYPE)
                self.add_spring( ((i + 1) * self.mesh.get_sim_u) + j, 
                                (i * self.mesh.get_sim_u) + j + 1, 
                                self.KS_SHEAR, self.KD_SHEAR, 
                                self.SHEAR_SPRING_TYPE)

        # Добавление сдвигающих пружин по горизонтали
        for i in range(0, self.mesh.get_sim_v ):
            for j in range(0, self.mesh.get_sim_u - 2):
                self.add_spring((i * self.mesh.get_sim_u) + j, 
                                (i * self.mesh.get_sim_u) + j + 2, 
                                self.KS_BEND, self.KD_BEND, 
                                self.BEND_SPRING_TYPE)
            self.add_spring((i * self.mesh.get_sim_u) + (self.mesh.get_sim_u - 3), 
                            (i * self.mesh.get_sim_u) + (self.mesh.get_sim_u - 1), 
                            self.KS_BEND, self.KD_BEND, self.BEND_SPRING_TYPE)

        # Добавление сгибающихся пружин по вертикали
        for i in range(0, self.mesh.get_sim_u):
            for j in range(0, self.mesh.get_sim_v - 2):
                self.add_spring((j * self.mesh.get_sim_u) + i, 
                                ((j + 2) * self.mesh.get_sim_u) + i, 
                                self.KS_BEND, self.KD_BEND, 
                                self.BEND_SPRING_TYPE)
            self.add_spring(((self.mesh.get_sim_v - 3) * self.mesh.get_sim_u) + i, 
                            ((self.mesh.get_sim_v - 1) * self.mesh.get_sim_u) + i, 
                            self.KS_BEND, self.KD_BEND, self.BEND_SPRING_TYPE)

    def IntegrateVerlet(self, dt):
        ''' 
        Метод Стёрмера — Верле(Интеграция Верле)
        https://ru.wikipedia.org/wiki/Метод_Стёрмера_—_Верле
        
        1.Вычисляются новые положения тел.
        2.Для каждой связи удовлетворяется соответствующее ограничение, 
          то есть расстояние между точками делается таким, каким оно должно быть.
        3.Шаг 2 повторяется несколько раз, тем самым все условия удовлетворяются (разрешается система условий).
        
        '''
        dt_2_mass = float(dt * dt) / float(self.mesh.mass)

        for i in range(0, self.mesh.get_meshResolution):
            buffer = self.mesh.get_position(i)[:]
            force = [dt_2_mass * self.mesh.get_velocity(i)[xyz] 
                    for xyz in range(0, 3)]

            diff = [self.mesh.get_position(i)[xyz] - self.vertices_last[i][xyz]
                    for xyz in range(0, 3)]

            self.mesh.set_velocity(i, [self.mesh.get_velocity(i)[xyz] + diff[xyz] + force[xyz]
                                       for xyz in range(0, 3)])

            self.vertices_last[i] = buffer
 
        if self.mesh.get_position(i)[1] < 0.0:
            self.mesh.get_position(i)[1] = 0
    
    def ComputeForces(self, dt):
        for i in range(0, self.mesh.get_meshResolution-1):
            # Создаём локальную переменную для хранения скорости точки
            vel = self.get_Vertlet_velocity(self.mesh.get_position(i), self.vertices_last[i], dt)

            if i != 0 and i != self.mesh.get_sim_u:
                self.mesh.get_velocity(i)[1] = 1000.0 * self.gravity[2] * float(self.mesh.mass) #y

            self.mesh.set_velocity(i, [self.mesh.get_velocity(i)[xyz] + self.DEFAULT_DAMPING * vel[xyz]
                                       for xyz in range(0, 3)])
            
        for i in range(0, len(self.springs)):
            p_1 = self.mesh.get_position(self.springs[i].pos_a)[:]
            p_1_last = self.vertices_last[self.springs[i].pos_a][:]
            p_2 = self.mesh.get_position(self.springs[i].pos_b)[:]
            p_2_last = self.vertices_last[self.springs[i].pos_b][:]

            v_1 = self.get_Vertlet_velocity(p_1, p_1_last, dt)
            v_2 = self.get_Vertlet_velocity(p_2, p_2_last, dt)

            # Дельта импульса
            delta_p = [p_1[xyz] - p_2[xyz] 
                       for xyz in range(0, 3)] # p1 - p2

            # Дельта скорости
            delta_v = [v_1[xyz] - v_2[xyz] 
                       for xyz in range(0, 3)] # v1 - v2

            dist = math.sqrt(delta_p[0] * delta_p[0] + delta_p[1] * delta_p[1] +  delta_p[2] * delta_p[2])
                             
            left_term = -self.springs[i].ks * (dist - self.springs[i].rest_length)
            right_term = self.springs[i].kd * ((delta_p[0] * delta_v[0] + delta_p[1] * delta_v[1] + delta_p[2] * delta_v[2]) / dist)
            
            spring_force = [(left_term + right_term) * (delta_p[xyz]/dist) 
                            for xyz in range(0, 3)]

            if self.springs[i].pos_a != 0 and self.springs[i].pos_a != self.mesh.get_sim_u:
                self.mesh.set_velocity(self.springs[i].pos_b, 
                                      [self.mesh.get_velocity(i)[xyz] + spring_force[xyz]
                                       for xyz in range(0, 3)])

            if self.springs[i].pos_b != 0 and self.springs[i].pos_b != self.mesh.get_sim_u:
                self.mesh.set_velocity(self.springs[i].pos_b, 
                                      [self.mesh.get_velocity(i)[xyz] - spring_force[xyz]
                                       for xyz in range(0, 3)])

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

        cloth_matrix = np.array([
                                    [i]
                                    for i in self.mesh.get_all_coord
                                ])

        cloth_matrix = np.reshape(cloth_matrix, (int(self.mesh.get_meshResolution**0.5), int(self.mesh.get_meshResolution**0.5), 3))


        structural_springs = np.array([[
                [cloth_matrix[i][j], cloth_matrix[i][j+1]], 
                [cloth_matrix[i][j], cloth_matrix[i+1][j]]
              ] for i in range(0, np.shape(cloth_matrix)[0]-1) 
                for j in range(0, np.shape(cloth_matrix)[1]-1)])

        print(np.shape(structural_springs))

        print("structural_springs = \n", structural_springs)

        for i in range(0, self.mesh.meshget_meshResolutionResolution-1):
            for o in range(0, 2):
                print(f'structural_springs{i}{o}{o} = ', structural_springs[i][o][o])
                self.mesh.set_coo(i, [self.kutt._next_y(structural_springs[i][o][o][a], structural_springs[i][o][o][a]) for a in range(0, 3)])

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

if __name__ == "__main__":
    # Проверка работоспособности CUDA
    # isCUDAAvailable()

    # Проверка dll для запуска симуляции
    # loadDLL()

    # Переход на первый кадр
    bpy.context.scene.frame_current = bpy.context.scene.frame_start

    # Создания экземпляра класса точек ткани
    cloth = Point()
    backUp = cloth.creating_backUp

    # Создание экземпляра класса физики
    sim = Physics(cloth, backUp)

    # Запуск симуляции физики
    sim.start_sim()

    # Запуск анимации
    # bpy.ops.screen.animation_play(reverse=False, sync=False)