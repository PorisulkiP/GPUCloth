"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

В этом файле создаётся плоскость с множеством полигонов.
К этой плоскости применяется физика ткани. Данные о типе ткани 
записываются в файл Cloth.txt
"""

# импортируем API для работы с blender
import bpy

# ------------------------------------------------------------------------
#    Scene Properties
# ------------------------------------------------------------------------

# В данном класе создаётся сцена для тестировния алгоритма физики ткани
class SetUp():
    def __init__(self):
        self.opening_scene()
#        self.pipInstall("numba") # установка внешних пакетов

    def opening_scene(self):
        """
        Создаёт сцену с кубом и плоскостью
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
            
#            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

            # Подразделяем для симуляции ткани
#            bpy.ops.object.subdivision_set(level=5, relative=False)
            
            # Изменяем подразделение на "простое"
#            bpy.context.object.modifiers["Subdivision"].subdivision_type = 'SIMPLE'
            
            # Применяем модификатор
#            bpy.ops.object.modifier_apply(modifier="Subdivision")
            
            # Сглаживаем плоскость
#            bpy.ops.object.shade_smooth()

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

# ------------------------------------------------------------------------
#    Writing to the file the cloth settings
# ------------------------------------------------------------------------

        
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

        file = open('C:\\Users\\alex1\\Desktop\\CUDACloth\\clothSettings.txt','w')
        file.write(str(value).replace("]", "").replace("[", "").replace(", '", "\n").replace("'", "").replace(",", "")) 
        file.close()
        

# ------------------------------------------------------------------------
#    Pip Install for Blender
# ------------------------------------------------------------------------

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


    