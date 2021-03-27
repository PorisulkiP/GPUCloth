"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

В этом файле обрабатывается вся физика.
В __init__() передаётся бэкап из главного файла, чтобы во время перехода в Edit mode и/или в первый кадр
всё возвращалось в изначальное положение.
"""

# импортируем API для работы с blender
import bpy
import numpy

class Physics():
    def __init__(self, backUp):
        # Засовываем данные о плоскости в переменную
        self.mesh_data = bpy.data.objects["Plane"].data
        self.mesh = bpy.data.objects["Plane"]
        self.cube = bpy.data.objects["Cube"]
        self.cube_data = bpy.data.objects["Cube"].data
#        self.collisionOBJ = [bpy.data.objects[i] for i in len(bpy.data.objects)]
#        self.collisionOBJ = 
        
        self.backUp = backUp
#        self.backupGet(backUp)
        
#        print(self.collisionOBJ)

        bpy.app.handlers.frame_change_pre.clear() 
        bpy.app.handlers.frame_change_pre.append(self.physicsCalculation(self.mesh_data, 
                                                                         self.cube_data, 
                                                                         self.mesh, 
                                                                         self.cube, 
                                                                         self.backUp))
        

    def physicsCalculation(self, mesh_data, cube_data, mesh, cube, backUp):
        '''
        Данная функция вызывается каждый кадр.
        запускает ещё несколько функция для расчёта физики
        '''
#        if bpy.context.scene.frame_current == bpy.context.scene.frame_start:
#            self.backupGet(backUp)
        def gravityCalc(scene):
            for i in range(0, len(mesh_data.vertices)-1):
                for o in range(0, len(cube_data.vertices)-1):
                    mesh_local = mesh_data.vertices[i].co
                    cube_local = cube_data.vertices[o].co
                    
                    try:
                        mesh_local_second = mesh_data.vertices[i+1].co
                        cube_local_second = cube_data.vertices[o+1].co
                    except IndexError:
                        break                        
                    
                    mesh_global = mesh.matrix_world @ mesh_local
                    cube_global = cube.matrix_world @ cube_local
                    
                    mesh_global_second = mesh.matrix_world @ mesh_local_second
                    cube_global_second = cube.matrix_world @ cube_local_second
                    
                    global_x_mesh = mesh_global[0]
                    global_y_mesh = mesh_global[1]
                    global_z_mesh = mesh_global[2]
                    
                    global_x_mesh_second = mesh_global_second[0]
                    global_y_mesh_second = mesh_global_second[1]
                    global_z_mesh_second = mesh_global_second[2]
                    
                    global_x_cube = cube_global[0]
                    global_y_cube = cube_global[1]
                    global_z_cube = cube_global[2]
                    
                    global_x_cube_second = cube_global_second[0]
                    global_y_cube_second = cube_global_second[1]
                    global_z_cube_second = cube_global_second[2]
                    
                    flag = self.cross(global_x_mesh, global_y_mesh, global_z_mesh,
                                 global_x_mesh_second, global_y_mesh_second, global_z_mesh_second,
                                 global_x_cube, global_y_cube, global_z_cube,
                                 global_x_cube_second, global_y_cube_second, global_z_cube_second)
                    
#                    print("X = ",  global_x_mesh - global_x_cube)
#                    print("Y = ",  global_y_mesh - global_y_cube)
#                    print("Z = ",  global_z_mesh - global_z_cube)
#                    
#                    print("Debug = ",  global_z_mesh - global_z_cube)

#                        (abs(global_x_mesh - global_x_cube) >= 0.405 and 
#                            (global_y_mesh - global_y_cube) >= 0.405 and 
#                            (global_z_mesh - global_z_cube) >= 0.405)

                    distance_min = 0.00005
#                    x, y, z = self.lenOfTwoPoints(global_x_mesh, global_y_mesh, global_z_mesh,
#                                                  global_x_cube, global_y_cube, global_z_cube)
#                    print("distance = ", distance)
#                    print("distance = ", distance > 0.4)
                    if (flag):
                        # print("first flag = ", flag)
                        for newCoor in mesh_data.vertices:
                            newCoor.co[0] += (bpy.context.scene.gravity[0] / bpy.context.scene.render.fps)
                            newCoor.co[1] += (bpy.context.scene.gravity[1] / bpy.context.scene.render.fps)
                            newCoor.co[2] += (bpy.context.scene.gravity[2] / bpy.context.scene.render.fps)
                    else:
                        # print("second flag = ", flag)
                        if flag:
                            for newCoor in mesh_data.vertices:
                                newCoor.co[0] -= (bpy.context.scene.gravity[0] / bpy.context.scene.render.fps)
                                newCoor.co[1] -= (bpy.context.scene.gravity[1] / bpy.context.scene.render.fps)
                                newCoor.co[2] -= (bpy.context.scene.gravity[2] / bpy.context.scene.render.fps)
                        else:
                            for newCoor in mesh_data.vertices:
#                                print("newCoor = ", newCoor.co)
                                newCoor.co[0] += 0
                                newCoor.co[1] += 0
                                newCoor.co[2] += 0
                        
        return gravityCalc
    
    def crossing(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, distance_min = 0.15):
        a = (abs(x1) - abs(x2))**2
        b = (abs(y1) - abs(y2))**2
        c = (abs(z1) - abs(z2))**2
#        d = (a+b+c)**0.5
#        print("a = ", a)
#        print("b = ", b)
#        print("c = ", c)
#        print("d = ", d)
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
#                print("newVert.co.x: ", newVert.co.x)
#                print("oldVert[0]: ", oldVert[0])
                newVert.co.x = oldVert[0]
                newVert.co.y = oldVert[1]
                newVert.co.z = oldVert[2]
    
    def collisionOBJ(self):
        pass

def backupSet():
    """
    Создание бэкапа 
    """
    mesh_data = bpy.data.objects["Plane"].data
    
    backupVert = []
    for NumVert in range(0, len(mesh_data.vertices)-1):
        backupVert.append(mesh_data.vertices[NumVert].co)
    
    return backupVert