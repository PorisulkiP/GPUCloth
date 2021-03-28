"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

В этом файле обрабатывается вся физика.
В __init__() передаётся бэкап из главного файла, чтобы во время перехода в Edit mode и/или в первый кадр
всё возвращалось в изначальное положение.

Меш (анг. Mesh) - это сетка полигонов, из которых состоит любой 3D объект 
Коллизии - столкновение
Коллайдер - один из объектов столкновения
"""

# импортируем API для работы с blender
import bpy
import numpy

class Physics():
    def __init__(self, backUp):
        # Создаём атрибуты плоскости
        self.mesh_data = bpy.data.objects["Plane"].data
        self.mesh = bpy.data.objects["Plane"]
        
        # Создаём атрибут бэкапа
        self.backUp = backUp

        # Собираем все объекты в сцене, способные к столкновению
        self.collision_objects, self.collision_objects_data = self.collisionOBJ()

        bpy.app.handlers.frame_change_pre.clear() 
        bpy.app.handlers.frame_change_pre.append(self.physicsCalculation())        

    def physicsCalculation(self):
        """
        Данная функция вызывается каждый кадр.
        запускает ещё несколько функция для расчёта физики

        mesh - объект, с симуляциет ткани
        mesh_data - данные об объекте, с симуляциет ткани

        collision_objects - объект(ы) столкновения
        collision_objects_data -  данные об объек(те)тах, столкновения

        backUp - бэкап ткани до симуляции
        """
#        if bpy.context.scene.frame_current == bpy.context.scene.frame_start:
#            self.backupGet(backUp)
        def gravityCalc(scene):
            # i - одна вершина из всех, что есть в объекте, с симуляциет ткани
            for i in range(0, len(self.mesh_data.vertices)-1):
                # obj_vert - меш из списка объектов столкновений
                for obj_vert in self.collision_objects:
                    # obj_vert - данные об меше из списка объектов столкновений
                    for obj_vert_data in self.collision_objects_data:
                        # о - одна вершина из всех, что есть в объекте столкновения
                        for o in range(0, len(obj_vert_data.vertices)-1):
                            # 1-я вершина из отрезка
                            mesh_local = self.mesh_data.vertices[i].co
                            collision_objects_local = obj_vert_data.vertices[o].co
                            
                            # попытка обратиться к 2-ой вершине из отрезка
                            # работает если кол-во точек чётно
                            try:
                                mesh_local_second = self.mesh_data.vertices[i+1].co
                                collision_objects_local_second = obj_vert_data.vertices[o+1].co
                            except IndexError:
                                break                        
                            
                            # глобализация координат первой вершины
                            mesh_global = self.mesh.matrix_world @ mesh_local
                            collision_objects_global = obj_vert.matrix_world @ collision_objects_local
                            
                            # глобализация координат второй вершины
                            mesh_global_second = self.mesh.matrix_world @ mesh_local_second
                            collision_objects_global_second = obj_vert.matrix_world @ collision_objects_local_second
                            
                            # [0] - x, [1] - y, [2] - z
                            global_x_mesh = mesh_global[0]
                            global_y_mesh = mesh_global[1]
                            global_z_mesh = mesh_global[2]
                            
                            global_x_mesh_second = mesh_global_second[0]
                            global_y_mesh_second = mesh_global_second[1]
                            global_z_mesh_second = mesh_global_second[2]
                            
                            global_x_collision_objects = collision_objects_global[0]
                            global_y_collision_objects = collision_objects_global[1]
                            global_z_collision_objects = collision_objects_global[2]
                            
                            global_x_collision_objects_second = collision_objects_global_second[0]
                            global_y_collision_objects_second = collision_objects_global_second[1]
                            global_z_collision_objects_second = collision_objects_global_second[2]
                            
                            flag = self.cross(global_x_mesh, global_y_mesh, global_z_mesh,
                                        global_x_mesh_second, global_y_mesh_second, global_z_mesh_second,
                                        global_x_collision_objects, global_y_collision_objects, global_z_collision_objects,
                                        global_x_collision_objects_second, global_y_collision_objects_second, global_z_collision_objects_second)
                            
                        #    print("X = ",  global_x_mesh - global_x_collision_objects)
                        #    print("Y = ",  global_y_mesh - global_y_collision_objects)
                        #    print("Z = ",  global_z_mesh - global_z_collision_objects)
                            distance_min = 0.00005
                            if (flag):
                                for newCoor in self.mesh_data.vertices:
                                    newCoor.co[0] += (bpy.context.scene.gravity[0] / bpy.context.scene.render.fps)
                                    newCoor.co[1] += (bpy.context.scene.gravity[1] / bpy.context.scene.render.fps)
                                    newCoor.co[2] += (bpy.context.scene.gravity[2] / bpy.context.scene.render.fps)
                            else:
                                if flag:
                                    for newCoor in self.mesh_data.vertices:
                                        newCoor.co[0] -= (bpy.context.scene.gravity[0] / bpy.context.scene.render.fps)
                                        newCoor.co[1] -= (bpy.context.scene.gravity[1] / bpy.context.scene.render.fps)
                                        newCoor.co[2] -= (bpy.context.scene.gravity[2] / bpy.context.scene.render.fps)
                                else:
                                    for newCoor in self.mesh_data.vertices:
                                        newCoor.co[0] += 0
                                        newCoor.co[1] += 0
                                        newCoor.co[2] += 0
                        
        return gravityCalc
    
    def crossing(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, distance_min = 0.15):
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
        Установка координат для точек из бэкапа
        """
        for newVert in self.mesh_data.vertices:
            for oldVert in backUp:
                newVert.co.x = oldVert[0]
                newVert.co.y = oldVert[1]
                newVert.co.z = oldVert[2]
    
    def collisionOBJ(self):
        """
        Данная функция возвращает массив объектов с которыми пересекается ткань
        """
        numOfOBJ = [bpy.data.objects[element] for element in range(0, len(bpy.data.objects)-1)]
        collision = []
        for col in numOfOBJ:
            try:
                col.modifiers['Collision'].settings.use == True
                collision.append(col)
            except KeyError:
                pass

        collisionOBJ_data = [element.data for element in collision]
        collisionOBJ = [element for element in collision]

        return collisionOBJ, collisionOBJ_data
    
def backupSet():
    """
    Создание бэкапа 
    """

    mesh_data = bpy.data.objects["Plane"].data
    
    # список всех вершин изначального меша ткани
    backupVert = []
    for NumVert in range(0, len(mesh_data.vertices)-1):
        backupVert.append(mesh_data.vertices[NumVert].co)
    
    return backupVert