bl_info = {
    'name': 'Creating GPUCloth',
    'category': 'All'
}
 
import bpy
 
class qualitySteps(bpy.types.Operator):
    bl_idname = 'mesh.quality_steps'
    bl_label = 'Quality Steps'
    bl_options = {"REGISTER", "UNDO"}
 
    def execute(self, context):
        print("Accept!")
        return {"FINISHED"}
 
class panel1(bpy.types.Panel):
    bl_idname = "CLOTH_PT_Cloth"
    bl_label = "GPUCloth"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_category = "GPUCloth"
 
    def draw(self, context):
        self.layout.operator("mesh.quality_steps", text="Quality Steps")
 
#       split = layout.split() # frame
#       col = split.column(aligh=True) # Elements on frame
#       col.operator('mesh.quality_steps', text="", icon="") # element
#       col.operator('mesh.quality_steps', text="", icon="") # element

def register() :
    bpy.utils.register_class(qualitySteps)
    bpy.utils.register_class(panel1)
 
def unregister() :
    bpy.utils.unregister_class(qualitySteps)
    bpy.utils.unregister_class(panel1)
 
if __name__ == "__main__" :
    register()