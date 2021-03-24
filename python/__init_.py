bl_info = {
    "name": "GPUCloth",
    "author": "PorisulkiP",
    "version": (0, 0, 1),
    "blender": (2, 80, 0),
    "location": "",
    "warning": "",
    "description": "Mesh modelling toolkit. Several tools to aid modelling",
    "doc_url": "{BLENDER_MANUAL_URL}/addons/mesh/edit_mesh_tools.html",
    "category": "System",
}

# Import From Files
if "bpy" in locals():
    import importlib
    importlib.reload(fillValClothSet)


else:
    from . import fillValClothSet

