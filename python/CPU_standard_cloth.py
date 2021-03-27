"""
Файл запускать только в blender 2.93.x, в рабочей области "Scripting"!

В этом файле создаётся плоскость с множеством полигонов.
Координаты точек записываются в файл data.txt.
Это пример того, что можно забирать данные(в нашем случае координаты вершини) 
из любого объекта и передавать в другой файл.

К этой плоскости применяется физика ткани. Данные о типе ткани 
записываются в файл Cloth.txt
"""

# импортируем API для работы с blender
import bpy

# Переход на первый кадр
bpy.context.scene.frame_current = bpy.context.scene.frame_start

def createPlane(): # Создаём плоскость
    # Создаём меш(объект и меш)
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, 
                                    align='WORLD', location=(0, 0, 1), 
                                    scale=(1, 1, 1))

    # Подразделяем для симуляции ткани
    bpy.ops.object.subdivision_set(level=5, relative=False)
    
    # Изменяем подразделение на "простое"
    bpy.context.object.modifiers["Subdivision"].subdivision_type = 'SIMPLE'
    
    # Применяем модификатор
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    
    # Сглаживаем плоскость
    bpy.ops.object.shade_smooth()

    # Засовываем данные о плоскости в переменную
    mesh_data = bpy.data.objects["Plane"].data
    mesh = bpy.data.objects["Plane"]
    
    # Назначаем на плоскость физику ткани
    bpy.ops.object.modifier_add(type='CLOTH') 
    
    bpy.context.object.modifiers["Cloth"].collision_settings.use_self_collision = True

    return mesh_data, mesh

def fillValClothSet(clothSetings, clothSetings_list): # записываем значения настроект в отдельный файл
    value = []
    value.append("air_damping = ")
    value.append(clothSetings.voxel_cell_size)
    
    value.append("bending damping = ")
    value.append(clothSetings.bending_damping)
    value.append("bending model = ")
    value.append(clothSetings.bending_model)
    value.append("bending stiffness = ")
    value.append(clothSetings.bending_stiffness)
    value.append("bending stiffness max = ")
    value.append(clothSetings.bending_stiffness_max)
    
    value.append("collider friction = ")
    value.append(clothSetings.collider_friction)
    
    value.append("density target = ")
    value.append(clothSetings.compression_damping)
    
    value.append("fluid_density = ")
    value.append(clothSetings.fluid_density)
    
    value.append("goal_default = ")
    value.append(clothSetings.goal_default)
    
    value.append("goal_friction = ")
    value.append(clothSetings.goal_friction)
    
    value.append("goal_max = ")
    value.append(clothSetings.goal_max)
    
    value.append("goal_min = ")
    value.append(clothSetings.goal_min)
    
    value.append("goal_spring = ")
    value.append(clothSetings.goal_spring)
    
    value.append("gravity = ")
    value.append(clothSetings.gravity)
    
    value.append("internal_compression_stiffness = ")
    value.append(clothSetings.internal_compression_stiffness)
    
    value.append("internal_friction = ")
    value.append(clothSetings.internal_friction)
    
    value.append("internal_spring_max_diversion = ")
    value.append(clothSetings.internal_spring_max_diversion)
    value.append("internal_spring_max_length = ")
    value.append(clothSetings.internal_spring_max_length)
    value.append("internal_spring_normal_check = ")
    value.append(clothSetings.internal_spring_normal_check)
    value.append("internal_tension_stiffness = ")
    value.append(clothSetings.internal_tension_stiffness)
    value.append("internal_tension_stiffness_max = ")
    value.append(clothSetings.internal_tension_stiffness_max)
    value.append("mass = ")
    value.append(clothSetings.mass)
    
    value.append("pin_stiffness = ")
    value.append(clothSetings.pin_stiffness)
    
    value.append("pressure_factor = ")
    value.append(clothSetings.pressure_factor)
    
    value.append("quality = ")
    value.append(clothSetings.quality)
    
    value.append("rest_shape_key = ")
    value.append(clothSetings.rest_shape_key)
    
    value.append("sewing_force_max = ")
    value.append(clothSetings.sewing_force_max)
    
    value.append("shear_damping = ")
    value.append(clothSetings.shear_damping)
    value.append("shear_stiffness = ")
    value.append(clothSetings.shear_stiffness)
    value.append("shear_stiffness_max = ")
    value.append(clothSetings.shear_stiffness_max)
    
    value.append("shrink_max = ")
    value.append(clothSetings.shrink_max)
    value.append("shrink_min = ")
    value.append(clothSetings.shrink_min)

    value.append("target_volume = ")
    value.append(clothSetings.target_volume)
    
    value.append("tension_damping = ")
    value.append(clothSetings.tension_damping)
    value.append("tension_stiffness = ")
    value.append(clothSetings.tension_stiffness)
    value.append("tension_stiffness_max = ")
    value.append(clothSetings.tension_stiffness_max)
    
    value.append("time_scale = ")
    value.append(clothSetings.time_scale)
    
    value.append("uniform_pressure_force = ")
    value.append(clothSetings.uniform_pressure_force)
    
    value.append("use_dynamic_mesh = ")
    value.append(clothSetings.use_dynamic_mesh)
    
    value.append("use_internal_springs = ")
    value.append(clothSetings.use_internal_springs)
    
    value.append("use_pressure = ")
    value.append(clothSetings.use_pressure)
    value.append("use_pressure_volume = ")
    value.append(clothSetings.use_pressure_volume)
    
    value.append("use_sewing_springs = ")
    value.append(clothSetings.use_sewing_springs)
    
    value.append("vertex_group_bending = ")
    value.append(clothSetings.vertex_group_bending)
    value.append("vertex_group_intern = ")
    value.append(clothSetings.vertex_group_intern)
    value.append("vertex_group_mass = ")
    value.append(clothSetings.vertex_group_mass)
    value.append("vertex_group_pressure = ")
    value.append(clothSetings.vertex_group_pressure)
    value.append("vertex_group_shear_stiffness = ")
    value.append(clothSetings.vertex_group_shear_stiffness)
    value.append("vertex_group_shrink = ")
    value.append(clothSetings.vertex_group_shrink)
    value.append("vertex_group_structural_stiffness = ")
    value.append(clothSetings.vertex_group_structural_stiffness)
    
    value.append("voxel_cell_size = ")
    value.append(clothSetings.voxel_cell_size)

    file = open('C:\\Users\\alex1\\Desktop\\CUDACloth\\clothSetings.txt','w')
    file.write(str(value).replace("]", "").replace("[", "").replace(", '", "\n").replace("'", "").replace(",", "")) 
    file.close()

# Пробуем взять данные о плоскости(объект)
try:
    mesh_data = bpy.data.objects["Plane"].data
    mesh = bpy.data.objects["Plane"]
    # KeyError это стандартное исключение, так что просто создаём всё,
    # что нам нужно для дальнейшей работы
except KeyError:
   # Создаём плоскость
   mesh_data, mesh = createPlane()
   
   # Создаём куб на который будет падать ткань
   bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, 
                               align='WORLD', location=(0, 0, 0))
   # Уменьшает куб
   bpy.ops.transform.resize(value=(0.5, 0.5, 0.5))
   
   # Назначаем его объектом столкновения
   bpy.ops.object.modifier_add(type='COLLISION')

vrts = [tuple(x.co) for x in mesh_data.vertices]

# Здесь замените на путь к своему проекту, там появится текстовый документ,
# в котором будут записаны координаты вершин.
# Открываем файл для записи
file = open('C:\\Users\\alex1\\Desktop\\CUDACloth\\data.txt','w')

# Записываем в файл точки
for i in range(len(vrts)):
    file.write(str(vrts[i]) + "\n")
    i+=1

file.close() # закрываем файл

clothSetings_list = dir(mesh.modifiers['Cloth'].settings) # список параметров
clothSetings = mesh.modifiers['Cloth'].settings

value = fillValClothSet(clothSetings, clothSetings_list)

# Запуск симуляции
bpy.ops.screen.animation_play(reverse=False, sync=False)

