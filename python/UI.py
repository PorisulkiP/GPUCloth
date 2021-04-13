bl_info = {
    "name": "GPUCloth",
    "description": "User UI for User Cloth ",
    "author": "Victus",
    "version": (0, 0, 1),
    "blender": (2, 93, 0),
}
 
import bpy

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

class GPUCloth_Settings(PropertyGroup):

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
        name = "Vertex Mass",
        description = "The mass of each vertex on the cloth material",
        default = 0.33,
        min = 0.001,
        max = 30.0
        )

    my_float_speed : FloatProperty(
        name = "Speed Multiplier",
        description = "Close speed is multiplied by this value",
        default = 0.1000,
        min = 0.001,
        max = 30.0
        )

    my_float_air : FloatProperty(
        name = "Air Viscosity",
        description = "Air has some thickness which slows falling things down",
        default = 0.1000,
        min = 0.001,
        max = 30.0
        )

    my_float_distance : FloatProperty(
        name = "Minimal Distance",
        description = "The distance another object must get to the cloth for the simulation to repel the cloth out of the way",
        default = 0.015,
        min = 0.001,
        max = 30.0
        ) 
        
    my_float_distance_self : FloatProperty(
        name = "Self Minimal Distance",
        description = "The distance another object must get to the cloth for the simulation to repel the cloth out of the way",
        default = 0.015,
        min = 0.001,
        max = 30.0
        ) 
    
    my_float_friction : FloatProperty(
        name = "Friction",
        description = "A coefficient for how slippery the cloth is when it collides with itself",
        default = 5.000,
        min = 0.001,
        max = 30.0
        ) 

    my_float_impulse : FloatProperty(
        name = "Impulse Clamp",
        description = "Prevents explosions in tight and complicated collision situations by restricting the amount of movement after a collision",
        default = 0.000,
        min = 0.000,
        max = 30.0
        ) 

    my_float_impulse_self : FloatProperty(
        name = "Self Impulse Clamp",
        description = "Prevents explosions in tight and complicated collision situations by restricting the amount of movement after a collision.",
        default = 0.000,
        min = 0.000,
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
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        split = layout.split(factor=0.4)
        col_1 = split.column()
        col_2 = split.column()
        
        # display the properties
        col_1.label(text="Vertex Mass")
        col_2.prop(mytool, "my_float_vertex", text="")
        col_1.label(text="Speed Multiplier")
        col_2.prop(mytool, "my_float_speed", text="")
        col_1.label(text="Air Viscosity")
        col_2.prop(mytool, "my_float_air", text="")
        layout.operator("wm.hello_world")
class UV_PT_GPUCloth_ObjColl(Panel):
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
        mytool = scene.my_tool
        split = layout.split(factor=0.4)
        col_1 = split.column()
        col_2 = split.column()

        # display the properties
        col_1.label(text="Distance (min)")
        col_2.prop(mytool, "my_float_distance", text="")
        col_1.label(text="Impulse Clamp")
        col_2.prop(mytool, "my_float_impulse", text="")
        
        if bpy.context.scene.render.use_border == False:
            col_1.enabled = False
            col_2.enabled = False
            
        if bpy.context.scene.render.use_border == True:
            col_1.enabled = True
            col_2.enabled = True

class UV_PT_GPUCloth_SelfColl(Panel):
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
        mytool = scene.my_tool
        split = layout.split(factor=0.4)
        col_1 = split.column()
        col_2 = split.column()

        # display the properties
        col_1.label(text="Friction")
        col_2.prop(mytool, "my_float_friction", text="")
        col_1.label(text="Distance (m)")
        col_2.prop(mytool, "my_float_distance_self", text="")
        col_1.label(text="Impulse Clamp")
        col_2.prop(mytool, "my_float_impulse_self", text="")
        
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

    bpy.types.Scene.my_tool = PointerProperty(type=GPUCloth_Settings)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Scene.my_tool


if __name__ == "__main__":
    register()