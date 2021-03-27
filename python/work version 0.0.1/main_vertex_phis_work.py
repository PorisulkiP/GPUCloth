"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

В этом файле создаётся плоскость с множеством полигонов.
К этой плоскости применяется физика ткани. Данные о типе ткани 
записываются в файл Cloth.txt
"""

# импортируем API для работы с blender
import bpy
#import numpy as np

# Переход на первый кадр
bpy.context.scene.frame_current = bpy.context.scene.frame_start

# Гравитация -0,01, чтобы плоскоть медленне падала
bpy.context.scene.gravity[2] = -0.01

# В данном класе создаётся сцена для тестировния алгоритма физики ткани
class SetUp():
    def __init__(self):
        # запускаем создание сцены
        self.opening_scene()
        # self.pipInstall("numba") # установка внешних пакетов

    def opening_scene(self):
        """
        Создаёт сцену с кубом и плоскостью
        Накидывает на неё модификатор ткани и собирает все параметры
        Следом вызывает функцию записи всех этих параметров в файл
        """
        # Пробуем взять данные о плоскости(объект)
        try:
            self.mesh_data = bpy.data.objects["Plane"].data
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
        
        self.clothSettings_list = dir(bpy.data.objects["Plane"].modifiers['Cloth'].settings) # список параметров
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

#        file = open('C:\\Users\\alex1\\Desktop\\CUDACloth\\clothSettings.txt','w')
#        file.write(str(value).replace("]", "").replace("[", "").replace(", '", "\n").replace("'", "").replace(",", "")) 
#        file.close()
        
    def pipInstall(self, pack):
        """
        В blender не так просто установить какой-либо пакет,
        потому я сделал функцию для установки через pip! 
        Достаточно просто передать параметр в виде имени пакета и раскоментировать строку в __init__
        """
        import sys
        import subprocess
        subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'ensurepip'])
        subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'numba'])


class Phisic():
    def __init__(self, backUp):
        # Засовываем данные о плоскости в переменную
        self.mesh_data = bpy.data.objects["Plane"].data
        self.mesh = bpy.data.objects["Plane"]
        
        # Засовываем данные о кубе в переменную
        self.cube = bpy.data.objects["Cube"]
        self.cube_data = bpy.data.objects["Cube"].data

        self.backUp = backUp

        bpy.app.handlers.frame_change_pre.clear() 
        bpy.app.handlers.frame_change_pre.append(self.physicsCalculation(self.mesh_data, self.cube_data, self.mesh, self.cube, self.backUp))
        
    def physicsCalculation(self, mesh_data, cube_data, mesh, cube, backUp):
        '''
        Данная функция вызывается каждый кадр.
        запускает ещё несколько функция для расчёта физики
        '''
        def gravityCalc(scene):
            
            # Перебираем все точки плоскости и куба
            # len(mesh_data.vertices) это колличество точек у данной геометрии
            for i in range(0, len(mesh_data.vertices)-1):
                for o in range(0, len(cube_data.vertices)-1):
                    mesh_local = mesh_data.vertices[i].co
                    cube_local = cube_data.vertices[o].co
                    
                    # пробуем обратиться ко второй точке, чтобы сделать отрезок
                    try:
                        mesh_local_second = mesh_data.vertices[i+1].co
                        cube_local_second = cube_data.vertices[o+1].co
                    except IndexError:
                        break                        
                    
                    # Делаем транспонирование матриц, 
                    # чтобы превратить локальные координаты и глобальные
                    mesh_global = mesh.matrix_world @ mesh_local
                    cube_global = cube.matrix_world @ cube_local
                    
                    mesh_global_second = mesh.matrix_world @ mesh_local_second
                    cube_global_second = cube.matrix_world @ cube_local_second
                    
                    # Первая точка
                    global_x_mesh = mesh_global[0]
                    global_y_mesh = mesh_global[1]
                    global_z_mesh = mesh_global[2]
                    
                    # торая точка и вместе они создают отрезок с которым мы работаем
                    global_x_mesh_second = mesh_global_second[0]
                    global_y_mesh_second = mesh_global_second[1]
                    global_z_mesh_second = mesh_global_second[2]
                    
                    global_x_cube = cube_global[0]
                    global_y_cube = cube_global[1]
                    global_z_cube = cube_global[2]
                    
                    global_x_cube_second = cube_global_second[0]
                    global_y_cube_second = cube_global_second[1]
                    global_z_cube_second = cube_global_second[2]
                    
                    # Флаг пересечений, где True = пересекаются
                    flag = self.cross(global_x_mesh, global_y_mesh, global_z_mesh,
                                 global_x_mesh_second, global_y_mesh_second, global_z_mesh_second,
                                 global_x_cube, global_y_cube, global_z_cube,
                                 global_x_cube_second, global_y_cube_second, global_z_cube_second)

                    # Расстояние между объектами
                    distance_min = 0.0005
                    
                    # Расстояние между точками
                    x, y, z = self.lenOfTwoPoints(global_x_mesh, global_y_mesh, global_z_mesh,
                                                  global_x_cube, global_y_cube, global_z_cube)

                    #Делаем проверку на расстояние между точками и если они не пересекаются, то опускаем дальше
                    if (x > distance_min and y > distance_min and z > distance_min):
                        for newCoor in mesh_data.vertices:
                            newCoor.co[0] += (bpy.context.scene.gravity[0] / bpy.context.scene.render.fps)
                            newCoor.co[1] += (bpy.context.scene.gravity[1] / bpy.context.scene.render.fps)
                            newCoor.co[2] += (bpy.context.scene.gravity[2] / bpy.context.scene.render.fps)
                    else:
                        # Если есть пересечения, то возвращаем назад
                        if flag:
                            for newCoor in mesh_data.vertices:
                                newCoor.co[0] -= (bpy.context.scene.gravity[0] / bpy.context.scene.render.fps)
                                newCoor.co[1] -= (bpy.context.scene.gravity[1] / bpy.context.scene.render.fps)
                                newCoor.co[2] -= (bpy.context.scene.gravity[2] / bpy.context.scene.render.fps)
                        else:
                            # Если нет пересечений, но в следующем кадре они есть, то стоим на месте
                            for newCoor in mesh_data.vertices:
                                newCoor.co[0] += 0
                                newCoor.co[1] += 0
                                newCoor.co[2] += 0                        
        return gravityCalc
    
    def lenOfTwoPoints(self, x1, y1, z1, x2, y2, z2, distance_min = 0.15):
        """
        Определяем расстояние между точками
        Не подходит для данной реализации(
        """
        a = (abs(x1) - abs(x2))**2
        b = (abs(y1) - abs(y2))**2
        c = (abs(z1) - abs(z2))**2
#        d = (a+b+c)**0.5
        return a, b, c

    def cross(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, distance_min = 0.15):
        """
        Задача данной функции определить пересечения векторов двух точке,
        т.е. найти пересечения вершин, чтобы геометрия не пересекалась друг с другом
        Пераметр distance_min задаётся в настройках ткани во вкладке Collision ==> Object collision 
        ==> Distanse и по умолчанию равен 0.15
        """
        if(y2 - y1 != 0):
            q = (x2 - x1) / (y1 - y2)
            sn = (x3-x4)+(y3-y4)*q
            if (sn<=0):
                return False
            else:
                fn = (x3-x1)+ (y3-y1)*q
                n=fn/sn
        else:
            if ((y3-y4)<=0):
                return False
            if ((y3-y4)<=distance_min):
                return True
            else:
                n = (y3-y1)/(y3-y4)                
        return True
        
    def backupGet(self, backUp):
        """
        Тут мы переносим все точки из бэкапа в нынешнюю геометрию
        """
        for newVert in self.mesh_data.vertices:
            for oldVert in backUp:
                newVert.co.x = oldVert[0]
                newVert.co.y = oldVert[1]
                newVert.co.z = oldVert[2]
    
    def collisionOBJ(self):
        """
        Собираем список всех объектов в сцене и проверяем их 
        на способность к обработке столкновений
        """
        pass

def backupSet():
    """
    Тут мы делаем бэкап перед запуском симуляции
    """
    mesh_data = bpy.data.objects["Plane"].data
    
    backupVert = []
    for NumVert in range(0, len(mesh_data.vertices)-1):
        backupVert.append(mesh_data.vertices[NumVert].co)
    
    return backupVert
    
if __name__ == "__main__":
    
    # Создание сцены
    setup = SetUp()
    
    # Создание бэкапа
    backUp = backupSet()
    
    # Запуск симуляции
    phisic = Phisic(backUp)

    # Запуск анимации
    bpy.ops.screen.animation_play(reverse=False, sync=False)