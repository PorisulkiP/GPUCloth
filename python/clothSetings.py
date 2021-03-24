"""
Файл запускать только в blender 2.83.x, в рабочей области "Scripting"!

Данный файл создаёт тектовый документ с настройками такни, указанными во вкладке физики
"""

# импортируем API для работы с blender
import bpy

def fillValClothSet(clothSetings):
    value = []
    value.append("air damping = ")
    value.append(clothSetings.air_damping)
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
    value.append("effector weights = ")
    value.append(clothSetings.effector_weights)
    value.append("fluid_density = ")
    value.append(clothSetings.fluid_density)
    value.append("goal_default = ")
    value.append(clothSetings.goal_default)
    value.append("goal_friction = ")
    value.append(clothSetings.goal_friction)
    print(value)
    file = open('clothSetings.txt','w')

    file.write(str(value).replace("]", "").replace("[", "").replace("',", "").replace("', '", "\n")) 
        
    file.close()

if __name__ == "__main__":
    fillValClothSet()



