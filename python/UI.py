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

class MySettings(PropertyGroup):

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
        name = "Set a value",
        description = "A float property",
        default = 0.33,
        min = 0.001,
        max = 30.0
        )

    my_float_speed : FloatProperty(
        name = "Set a value",
        description = "A float property",
        default = 0.1000,
        min = 0.001,
        max = 30.0
        )

    my_float_air : FloatProperty(
        name = "Set a value",
        description = "A float property",
        default = 0.1000,
        min = 0.001,
        max = 30.0
        )

    my_float_distance : FloatProperty(
        name = "Set a value",
        description = "A float property",
        default = 0.015,
        min = 0.001,
        max = 30.0
        ) 
        
    my_float_distance_self : FloatProperty(
        name = "Set a value",
        description = "A float property",
        default = 0.015,
        min = 0.001,
        max = 30.0
        ) 
    
    my_float_friction : FloatProperty(
        name = "Set a value",
        description = "A float property",
        default = 5.000,
        min = 0.001,
        max = 30.0
        ) 

    my_float_impulse : FloatProperty(
        name = "Set a value",
        description = "A float property",
        default = 0.000,
        min = 0.000,
        max = 30.0
        ) 

    my_float_impulse_self : FloatProperty(
        name = "Set a value",
        description = "A float property",
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

        # display the properties
        layout.prop(mytool, "my_float_vertex", text="Vertex Mass")
        layout.prop(mytool, "my_float_speed", text="Speed Multiplier")
        layout.prop(mytool, "my_float_air", text="Air Viscosity")


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

        # display the properties
        layout.prop(mytool, "my_float_distance", text="Distance (m)")
        layout.prop(mytool, "my_float_impulse", text="Impulse Clamp")

class UV_PT_GPUCloth_SelfColl(Panel):
    bl_idname = "UV_PT_GPUCloth_SelfColl"
    bl_label = ""
    bl_parent_id = "UV_PT_GPUCloth"
    bl_category = "My Category"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self,context):
        self.layout.prop(context.scene.render, "use_border", text="Self Collizions")
        
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool

        # display the properties
        layout.prop(mytool, "my_float_friction", text="Friction")
        layout.prop(mytool, "my_float_distance_self", text="Distance (m)")
        layout.prop(mytool, "my_float_impulse_self", text="Impulse Clamp")
#     --- Registration UI Panel ---

#class Open_Panel_GPUCloth(bpy.types.Operator):
#    bl_idname = "Open_Panel_GPUCloth"
#    bl_label = "GPUCloth"
#    bl_options = {"REGISTER", "UNDO"}
# 
#    def execute(self, context):
#        UV_PT_GPUCloth.bl_options = "True"
#        return {"FINISHED"}

classes = (
    MySettings,
    UV_PT_GPUCloth,
    UV_PT_GPUCloth_ObjColl,
    UV_PT_GPUCloth_SelfColl,
#    Open_Panel_GPUCloth
)

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Scene.my_tool = PointerProperty(type=MySettings)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Scene.my_tool


if __name__ == "__main__":
    register()