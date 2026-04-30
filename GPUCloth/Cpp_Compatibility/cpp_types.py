# Blender GPUCloth Add-on
# Copyright (C) 2023 Bubnov Aleksey
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ctypes import c_float, c_short, c_char, c_char_p, c_uint, c_uint16, c_uint32, c_int64, c_uint64, c_int, c_void_p, c_ubyte, c_size_t, Structure, POINTER

PHYS_GLOBAL_GRAVITY = 1

DAG_EVAL_VIEWPORT   = 0
DAG_EVAL_RENDER     = 1

CLOTH_COLLSETTINGS_FLAG_ENABLED = 2
CLOTH_COLLSETTINGS_FLAG_SELF    = 4

CLOTH_BENDING_LINEAR    = 0
CLOTH_BENDING_ANGULAR   = 1

CLOTH_SIMSETTINGS_FLAG_COLLOBJ                = (1 << 2)   #   4
CLOTH_SIMSETTINGS_FLAG_GOAL                   = (1 << 3)   #   8
CLOTH_SIMSETTINGS_FLAG_TEARING               = (1 << 4)   #  16
CLOTH_SIMSETTINGS_FLAG_PRESSURE              = (1 << 5)   #  32
CLOTH_SIMSETTINGS_FLAG_PRESSURE_VOL          = (1 << 6)   #  64
CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS      = (1 << 7)   # 128
CLOTH_SIMSETTINGS_FLAG_SCALING               = (1 << 8)   # 256
CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS_NORMAL = (1 << 9)   # 512
CLOTH_SIMSETTINGS_FLAG_CCACHE_EDIT           = (1 << 12)  # 4096
CLOTH_SIMSETTINGS_FLAG_RESIST_SPRING_COMPRESS = (1 << 13)  # 8192
CLOTH_SIMSETTINGS_FLAG_SEW                   = (1 << 14)  # 16384
CLOTH_SIMSETTINGS_FLAG_DYNAMIC_BASEMESH      = (1 << 15)  # 32768
CLOTH_SIMSETTINGS_FLAG_DYNAMIC_MESH           = CLOTH_SIMSETTINGS_FLAG_DYNAMIC_BASEMESH  # alias for Blender RNA compatibility

# ---------------------------------------------------------------------------
#  Типы GPU-солверов ткани
#  Зеркало C++ enum SolverType в src/engine/source/DNA/cloth_types.cuh
# ---------------------------------------------------------------------------
SOLVER_XPBD   = 0   # Extended Position-Based Dynamics (Macklin 2016/2019)
SOLVER_PD     = 1   # Projective Dynamics с Chebyshev-Jacobi ускорением
SOLVER_MGPBD  = 2   # Многоуровневый PBD с Algebraic Multigrid (AMG)
SOLVER_Mil2   = 3   # Non-distance barriers + subspace reuse (Mil²)
SOLVER_OGC    = 4   # Offset Geometric Contact — самостолкновение ткани
# OGC — солвер самостолкновений, не основной физический солвер.
# Подключается к любому из вышеперечисленных через SIM_set_self_collision_params.

# ---------------------------------------------------------------------------
#  Непрозрачный указатель на C++ объект ProxySimHandle
#  (создаётся через ProxySim_create, освобождается через ProxySim_free)
# ---------------------------------------------------------------------------
ProxySimHandle = c_void_p

class MVert(Structure):
    _fields_ = [
        ("co", c_float*3),
        ("flag", c_char)
    ]

class Edge(Structure): _fields_ = [ ("v_low", c_uint), ("v_high", c_uint)]

class EdgeSet(Structure):
    _fields_ = [
        ("entries", POINTER(Edge)), 
        ("map", POINTER(c_int)), 
        ("slot_mask", c_uint), 
        ("capacity_exp", c_uint),
        ("length", c_uint)
    ]

class MVertTri(Structure): _fields_ = [ ("tri", c_uint*3)]

class MTFace(Structure): _fields_ = [ ("uv", (c_float*3) * 2)]

class MEdge(Structure):
    _fields_ = [
        ("v1", c_uint), 
        ("v2", c_uint), 
        ("crease", c_char), 
        ("bweight", c_char),
        ("flag", c_short)
    ]

class MPoly(Structure):
    _fields_ = [
        ("loopstart", c_int), 
        ("totloop", c_int) 
        # ("mat_nr", c_short), 
        # ("flag", c_char),
        # ("_pad", c_char)
    ]

class MDeformWeight(Structure):
    _fields_ = [
        ("def_nr", c_uint), 
        ("weight", c_float)
    ]

class MDeformVert(Structure):
    _fields_ = [
        ("dw", POINTER(MDeformWeight)), 
        ("totweight", c_int),
        ("flag", c_int)
    ]

class MLoop(Structure):
    _fields_ = [("v", c_uint),  ("e", c_uint)]

class BVHNode(Structure):
    pass

BVHNode._fields_ = [
        ("children", POINTER(POINTER(BVHNode))), 
        ("parent", POINTER(BVHNode)), 
        ("bv", POINTER(c_float)), 
        ("index", c_int), 
        ("totnode", c_char), 
        ("main_axis", c_char)
    ]

class BVHTree(Structure):
    _fields_ = [
        ("nodes", POINTER(POINTER(BVHNode))), 
        ("nodearray", POINTER(BVHNode)), 
        ("nodechild", POINTER(POINTER(BVHNode))), 
        ("nodebv", POINTER(c_float)), 
        ("epsilon", c_float), 
        ("totleaf", c_int), 
        ("totbranch", c_int),
        ("start_axis", c_ubyte),
        ("stop_axis", c_ubyte),
        ("axis", c_ubyte),
        ("tree_type", c_char)
    ]

class ListBase(Structure):
    _fields_ = [
        ("first", c_void_p), 
        ("last", c_void_p)
    ]

class RandomNumberGenerator(Structure): _fields_ = [("x_", c_uint64)]

class Object(Structure):
    pass

class ID(Structure):
    pass

class PointCache(Structure):
    pass

class LinkNode(Structure):
    pass

class ModifierData(Structure):
    pass

ID._fields_ = [("name", c_char*66), ("id",c_uint), ("session_uuid", c_uint)]

LinkNode._fields_ = [("next", POINTER(LinkNode)), ("link", c_void_p)]

ModifierData._fields_ = [("next", POINTER(ModifierData)),
                ("prev", POINTER(ModifierData)),
                ("type", c_int),
                ("mode", c_int),
                ("id", c_uint),
                ("name", c_char*64)]

PointCache._fields_ = [("next", POINTER(PointCache)),
            ("prev", POINTER(PointCache)),
            ("flag", c_int),
            ("step", c_int),
            ("simframe", c_int),
            ("startframe", c_int),
            ("endframe", c_int),
            ("editframe", c_int),
            ("last_exact", c_int),
            ("last_valid", c_int),
            ("totpoint", c_int),
            ("index", c_int),
            ("cached_frames", c_char_p),
            ("cached_frames_len", c_int)]

class PartDeflect(Structure):
    _fields_ = [
        ("flag", c_int), 
        ("deflect", c_short), 
        ("forcefield", c_short), 
        ("falloff", c_short), 
        ("shape", c_short),
        ("tex_mode", c_short), 
        ("kink", c_short), 
        ("kink_axis", c_short), 
        ("zdir", c_short), 
        ("f_strength", c_float), 
        ("f_damp", c_float), 
        ("f_flow", c_float), 
        ("f_wind_factor", c_float), 
        ("f_size", c_float), 
        ("f_power", c_float), 
        ("maxdist", c_float), 
        ("mindist", c_float), 
        ("f_power_r", c_float), 
        ("maxrad", c_float), 
        ("minrad", c_float), 
        ("pdef_damp", c_float), 
        ("pdef_rdamp", c_float), 
        ("pdef_perm", c_float), 
        ("pdef_frict", c_float), 
        ("pdef_rfrict", c_float), 
        ("pdef_stickness", c_float), 
        ("absorption", c_float), 
        ("pdef_sbdamp", c_float), 
        ("pdef_sbift", c_float), 
        ("pdef_sboft", c_float), 
        ("clump_fac", c_float), 
        ("clump_pow", c_float), 
        ("kink_freq", c_float), 
        ("kink_shape", c_float), 
        ("kink_amp", c_float), 
        ("free_end", c_float), 
        ("rng", POINTER(RandomNumberGenerator)),
        ("f_noise", c_float), 
        ("seed", c_int), 
        ("pdef_cfrict", c_float)
    ]

class Collection(Structure):
    _fields_ = [
        ("id", ID), 
        ("gobject", ListBase), 
        ("children", ListBase)
    ]

class EffectorWeights(Structure):
    _fields_ = [
        ("weight", c_float*14), 
        ("global_gravity", c_float)
    ]
    def __init__(self):
        # Инициализируем все поля нулями
        self.weight = (c_float * 14)(*([0] * 14))
        self.global_gravity = c_float(0)

class fmatrix3x3(Structure):
    _fields_ = [
        ("m", c_float * 3 * 3),  # 3x3 matrix
        ("c", c_uint),  # column number
        ("r", c_uint),  # row number
        ("n1", c_float),  # three normal vectors for collision constraints
        ("n2", c_float),
        ("n3", c_float),
        ("vcount", c_uint),  # vertex count
        ("scount", c_uint)  # spring count
    ]

class Implicit_Data(Structure):
    _fields_ = [
        ("bigI", POINTER(fmatrix3x3)),  # identity (constant)
        ("tfm", POINTER(fmatrix3x3)),  # local coordinate transform
        ("M", POINTER(fmatrix3x3)),  # masses
        ("F", POINTER(c_float * 3)),  # forces
        ("dFdV", POINTER(fmatrix3x3)),  # force jacobians
        ("dFdX", POINTER(fmatrix3x3)),
        ("num_blocks", c_int),  # number of off-diagonal blocks (springs)
        ("X", POINTER(c_float * 3)),  # positions
        ("Xnew", POINTER(c_float * 3)),
        ("V", POINTER(c_float * 3)),  # velocities
        ("Vnew", POINTER(c_float * 3)),
        ("B", POINTER(c_float * 3)),  # B for A*dV = B
        ("A", POINTER(fmatrix3x3)),  # A for A*dV = B
        ("dV", POINTER(c_float * 3)),  # velocity change (solution of A*dV = B)
        ("z", POINTER(c_float * 3)),  # target velocity in constrained directions
        ("S", POINTER(fmatrix3x3)),  # filtering matrix for constraints
        ("P", POINTER(fmatrix3x3)),  # pre-conditioning matrix
        ("Pinv", POINTER(fmatrix3x3))
    ]

class ClothSimSettings(Structure):
    _fields_ = [
            ("mingoal", c_float),
            ("Cdis", c_float),
            ("Cvi", c_float),
            ("gravity", c_float*3),
            ("dt", c_float),
            ("mass", c_float),
            ("structural", c_float),
            ("shear", c_float),
            ("bending", c_float),
            ("max_bend", c_float),
            ("max_struct", c_float),
            ("max_shear", c_float),
            ("max_sewing", c_float),
            ("avg_spring_len", c_float),
            ("timescale", c_float),
            ("time_scale", c_float),
            ("maxgoal", c_float),
            ("eff_force_scale", c_float),
            ("eff_wind_scale", c_float),
            ("sim_time_old", c_float),
            ("defgoal", c_float),
            ("goalspring", c_float),
            ("goalfrict", c_float),
            ("velocity_smooth", c_float),
            ("density_target", c_float),
            ("density_strength", c_float),
            ("collider_friction", c_float),
            ("vel_damping", c_float),
            ("shrink_min", c_float),
            ("shrink_max", c_float),
            ("uniform_pressure_force", c_float),
            ("target_volume", c_float),
            ("pressure_factor", c_float),
            ("fluid_density", c_float),
            ("vgroup_pressure", c_short),
            ("bending_damping", c_float),
            ("voxel_cell_size", c_float),
            ("stepsPerFrame", c_int),
            ("flags", c_int),
            ("preroll", c_int),
            ("maxspringlen", c_int),
            ("solver_type", c_short),
            ("vgroup_bend", c_short),
            ("vgroup_mass", c_short),
            ("vgroup_struct", c_short),
            ("vgroup_shrink", c_short),
            ("shapekey_rest", c_short),
            ("presets", c_short),
            ("reset", c_short),
            ("effector_weights", POINTER(EffectorWeights)),
            ("bending_model", c_short),
            ("vgroup_shear", c_short),
            ("tension", c_float),
            ("compression", c_float),
            ("max_tension", c_float),
            ("max_compression", c_float),
            ("tension_damp", c_float),
            ("compression_damp", c_float),
            ("shear_damp", c_float),
            ("internal_spring_max_length", c_float),
            ("internal_spring_max_diversion", c_float),
            ("vgroup_intern", c_short),
            ("internal_tension", c_float),
            ("internal_compression", c_float),
            ("max_internal_tension", c_float),
            ("max_internal_compression", c_float)
]

    def __init__(self):
        # Инициализация всех полей значением 0 или NULL
        for field_name, field_type in self._fields_:
            if hasattr(field_type, "_length_"):
                setattr(self, field_name, field_type(*([0] * field_type._length_)))
            elif field_type == POINTER(EffectorWeights):
                setattr(self, field_name, None)
            else:
                setattr(self, field_name, field_type(0))

class ClothCollSettings(Structure):
    _fields_ = [
            ("epsilon",         c_float),
            ("self_friction",   c_float),
            ("friction",        c_float),
            ("damping",         c_float),
            ("selfepsilon",     c_float),
            ("flags",           c_int),
            ("loop_count",      c_short),
            ("group",           POINTER(Collection)),
            ("vgroup_selfcol",  c_short),
            ("vgroup_objcol",   c_short),
            ("clamp",           c_float),
            ("self_clamp",      c_float)
]

class ClothSolverResult(Structure):
    _fields_ = [
        ("status", c_int), 
        ("max_iterations", c_int), 
        ("min_iterations", c_int), 
        ("avg_iterations", c_float), 
        ("max_error", c_float), 
        ("min_error", c_float), 
        ("avg_error", c_float)
    ]

class ClothVertex(Structure):
    _fields_ = [
        ("flags",           c_int), 
        ("v",               c_float*3),
        ("xconst",          c_float*3), 
        ("x",               c_float*3), 
        ("xold",            c_float*3), 
        ("tx",              c_float*3), 
        ("txold",           c_float*3), 
        ("tv",              c_float*3), 
        ("mass",            c_float), 
        ("goal",            c_float), 
        ("impulse",         c_float*3), 
        ("xrest",           c_float*3), 
        ("dcvel",           c_float*3), 
        ("impulse_count",   c_uint), 
        ("avg_spring_len",  c_float), 
        ("struct_stiff",    c_float), 
        ("bend_stiff",      c_float), 
        ("shear_stiff",     c_float), 
        ("spring_count",    c_int), 
        ("shrink_factor",   c_float), 
        ("internal_stiff",  c_float), 
        ("pressure_factor", c_float)
    ]

class Cloth(Structure):
    _fields_ = [
        ("verts", POINTER(ClothVertex)), 
        ("springs", POINTER(LinkNode)), 
        ("numsprings", c_uint), 
        ("mvert_num", c_uint),
        ("primitive_num", c_uint),
        ("bvhtree", POINTER(BVHTree)),
        ("bvhselftree", POINTER(BVHTree)),
        ("tri", POINTER(MVertTri)),
        ("implicit", POINTER(Implicit_Data)),
        ("edgeset", POINTER(EdgeSet)),
        ("last_frame", c_int),
        ("initial_mesh_volume", c_float),
        ("average_acceleration", c_float*3),
        ("edges", POINTER(MEdge)),
        ("sew_edge_graph", POINTER(EdgeSet))
    ]

class IDNode(Structure):
    _fields_ = [("id_orig", POINTER(ID)),
                ("id_orig_session_uuid", c_uint),
                ("id_cow", POINTER(ID)),
                ("linked_state", c_uint)]

class PhysicsSettings(Structure):
    _fields_ = [("gravity", c_float*3), ("flag", c_int)]

class RenderData(Structure):
    _fields_ = [("cfra",        c_int),
                ("subframe",    c_float),
                ("framelen",    c_float),
                ("frs_sec",     c_short)
    ]

class Scene(Structure):
    pass

class CollisionModifierData(Structure):
    _fields_ = [
        ("x", POINTER(MVert)),
        ("xnew", POINTER(MVert)),
        ("xold", POINTER(MVert)),
        ("current_xnew", POINTER(MVert)),
        ("current_x", POINTER(MVert)),
        ("current_v", POINTER(MVert)),
        ("tri", POINTER(MVertTri)),
        ("mvert_num", c_uint),
        ("tri_num", c_uint),
        ("time_x", c_float),
        ("time_xnew", c_float),
        ("is_static", c_char),
        ("bvhtree", POINTER(BVHTree)),
    ]

Object._fields_ = [("id", ID),
                # ("data", c_void_p),
                ("modifiers", POINTER(CollisionModifierData)),
                ("obmat", c_float*4*4),
                ("imat", c_float*4*4),
                # ("particlesystem", ListBase),
                ("pd", POINTER(PartDeflect)),
                ]

Scene._fields_ = [("id", ID),
                  ("flag", c_short),
                  ("r", RenderData),
                  ("physics_settings", PhysicsSettings)
                ]

# # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # #
# Непосредственно классы для самой симуляции
# Directly classes for the simulation itself
# # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # 

# Из него для симуляции используем только scene, physics_relations, id_hash, 
# scene -> physics_settings(flag, gravity, RenderData(r).frs_sec)
class Depsgraph(Structure):
    _fields_ = [
        ("Scene", POINTER(Scene)),
        ("mode", c_short),
        ("ctime", c_float),
        ("lock", c_uint)
    ]

    def __init__(self):
        # Установка всех указателей в NULL
        self.Scene = POINTER(Scene)()
        
        # Установка примитивных типов в нулевые значения
        self.mode = 0
        self.ctime = 0.0
        self.lock = 0

class Mesh(Structure):
    _fields_ = [
        ("ID", ID),
        ("mvert", POINTER(MVert)),
        ("medge", POINTER(MEdge)),
        ("mpoly", POINTER(MPoly)),
        ("mloop", POINTER(MLoop)),
        ("totvert", c_int),
        ("totedge", c_int),
        ("totpoly", c_int),
        ("totloop", c_int),
        ("dvert", POINTER(MDeformVert)),
        ("runtime", c_char*8)
    ]

class ClothModifierData(Structure):
    _fields_ = [
        # from ModifierData, 'cause struct ClothModifierData : ModifierData
        ("next", POINTER(ModifierData)),
        ("prev", POINTER(ModifierData)),
        ("id", c_uint),
        ("type", c_int),
        ("mode", c_int),
        ("name", c_char*64),
        # Уже переменные самого класса
        ("modifier", ModifierData),
        ("clothObject", POINTER(Cloth)),
        ("sim_parms", POINTER(ClothSimSettings)),
        ("coll_parms", POINTER(ClothCollSettings)),
        ("point_cache", POINTER(PointCache)),
        ("ptcaches", ListBase),
        ("solver_result", POINTER(ClothSolverResult))
    ]