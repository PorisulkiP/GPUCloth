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

MAX_EFFECTORS = 256

class GPUEffector(Structure):
    _fields_ = [
        ("type", c_int),
        ("strength", c_float),
        ("flow", c_float),
        ("maxdist", c_float),
        ("mindist", c_float),
        ("f_power", c_float),
        ("f_noise", c_float),
        ("f_damp", c_float),
        ("f_size", c_float),
        ("seed", c_int),
        ("falloff_type", c_int),
        ("shape_type", c_int),
        ("zdir", c_int),
        ("obmat", c_float * 16),
        ("imat", c_float * 16),
    ]


GPUCLOTH_FEATURE_MISSING = 0
GPUCLOTH_FEATURE_PARTIAL = 1
GPUCLOTH_FEATURE_IMPLEMENTED = 2
GPUCLOTH_FEATURE_PROVEN = 3

GPUCLOTH_FEATURE_BLENDER_CORE = 1 << 0
GPUCLOTH_FEATURE_RELEASE_REQUIRED = 1 << 1
GPUCLOTH_FEATURE_FUTURE_CONFIG = 1 << 2
GPUCLOTH_FEATURE_EXTENSION = 1 << 3

GPUCLOTH_CONFIG_NONE = 0
GPUCLOTH_CONFIG_SIMULATION = 1 << 0
GPUCLOTH_CONFIG_MATERIAL = 1 << 1
GPUCLOTH_CONFIG_PIN = 1 << 2
GPUCLOTH_CONFIG_CONSTRAINT = 1 << 3
GPUCLOTH_CONFIG_PRESSURE = 1 << 4
GPUCLOTH_CONFIG_COLLISION = 1 << 5
GPUCLOTH_CONFIG_COLLIDER = 1 << 6
GPUCLOTH_CONFIG_MESH_STATE = 1 << 7
GPUCLOTH_CONFIG_EFFECTOR = 1 << 8
GPUCLOTH_CONFIG_EFFECTOR_WEIGHTS = 1 << 9
GPUCLOTH_CONFIG_CACHE = 1 << 10
GPUCLOTH_CONFIG_SEWING = 1 << 11
GPUCLOTH_CONFIG_VERTEX_CHANNEL = 1 << 12
GPUCLOTH_CONFIG_COLLISION_FILTER = 1 << 13
GPUCLOTH_CONFIG_PROXY = 1 << 14
GPUCLOTH_CONFIG_DIAGNOSTICS = 1 << 15

GPUCLOTH_ABI_UNSUPPORTED = 4
GPUCLOTH_ABI_OK = 0
GPUCLOTH_ABI_INVALID_ARGUMENT = 1
GPUCLOTH_ABI_STRUCT_TOO_SMALL = 2
GPUCLOTH_ABI_UNKNOWN_FEATURE = 3
GPUCLOTH_ABI_NOT_CONFIGURABLE = 5
GPUCLOTH_ABI_VERSION_MISMATCH = 6
GPUCLOTH_ABI_COUNT_MISMATCH = 7
GPUCLOTH_ABI_INVALID_VALUE = 8
GPUCLOTH_ABI_INVALID_STATE = 9

GPUCLOTH_FEATURE_PIN_GOAL = 10
GPUCLOTH_FEATURE_ANIMATED_PIN = 11
GPUCLOTH_FEATURE_TIMESTEP_SPEED = 2
GPUCLOTH_FEATURE_MATERIAL_MASS = 3
GPUCLOTH_FEATURE_STRETCH = 4
GPUCLOTH_FEATURE_SHEAR = 6
GPUCLOTH_FEATURE_PRESSURE_VOLUME = 18
GPUCLOTH_FEATURE_GRAVITY_VECTOR = 50
GPUCLOTH_FEATURE_SIMULATION_QUALITY = 51
GPUCLOTH_FEATURE_AIR_DAMPING = 52

GPUCLOTH_SOLVER_XPBD = 1 << 0
GPUCLOTH_SOLVER_PD = 1 << 1
GPUCLOTH_SOLVER_MIL2 = 1 << 2

GPUCLOTH_PRESSURE_ENABLED = 1 << 0

GPUCLOTH_VERTEX_PIN_WEIGHT = 2
GPUCLOTH_VERTEX_PIN_TARGET_XYZ = 3


class GPUClothABIVersion(Structure):
    _fields_ = [
        ("struct_size", c_uint),
        ("abi_major", c_uint),
        ("abi_minor", c_uint),
        ("abi_patch", c_uint),
        ("feature_schema_version", c_uint),
        ("feature_count", c_uint),
        ("reserved", c_uint * 2),
    ]


class GPUClothFeatureInfo(Structure):
    _fields_ = [
        ("struct_size", c_uint),
        ("feature_id", c_uint),
        ("status", c_uint),
        ("supported_solver_mask", c_uint),
        ("proven_solver_mask", c_uint),
        ("flags", c_uint),
        ("config_version", c_uint),
        ("config_kind_mask", c_uint),
        ("name", c_char * 64),
        ("detail", c_char * 192),
    ]


class GPUClothHostLayout(Structure):
    _fields_ = [
        ("struct_size", c_uint),
        ("schema_version", c_uint),
        ("cloth_vertex_size", c_uint),
        ("cloth_vertex_x_offset", c_uint),
        ("mesh_size", c_uint),
        ("mesh_mlooptri_offset", c_uint),
        ("mesh_runtime_offset", c_uint),
        ("cloth_modifier_data_size", c_uint),
        ("cloth_modifier_data_cloth_offset", c_uint),
        ("cloth_modifier_data_sim_offset", c_uint),
        ("cloth_modifier_data_coll_offset", c_uint),
        ("collision_modifier_data_size", c_uint),
        ("collision_modifier_data_bvh_offset", c_uint),
        ("object_size", c_uint),
        ("object_modifiers_offset", c_uint),
        ("object_pd_offset", c_uint),
        ("scene_size", c_uint),
        ("cloth_sim_settings_size", c_uint),
        ("cloth_coll_settings_size", c_uint),
        ("mvert_size", c_uint),
        ("medge_size", c_uint),
        ("mpoly_size", c_uint),
        ("mloop_size", c_uint),
        ("mvert_tri_size", c_uint),
        ("reserved", c_uint * 2),
    ]


class GPUClothFeatureConfigHeader(Structure):
    _fields_ = [
        ("struct_size", c_uint),
        ("feature_id", c_uint),
        ("config_version", c_uint),
        ("flags", c_uint),
    ]


class GPUClothVertexChannelConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("channel", c_uint),
        ("element_width", c_uint),
        ("element_count", c_uint64),
        ("data_address", c_uint64),
    ]


class GPUClothCollisionFilterConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("collection_id", c_uint64),
        ("object_mask_channel", c_uint),
        ("self_mask_channel", c_uint),
    ]


class GPUClothBufferView(Structure):
    _fields_ = [
        ("struct_size", c_uint),
        ("element_type", c_uint),
        ("element_count", c_uint64),
        ("stride_bytes", c_uint64),
        ("data_address", c_uint64),
        ("generation", c_uint64),
    ]


class GPUClothNamedValue(Structure):
    _fields_ = [
        ("name", c_char * 48),
        ("value_type", c_uint),
        ("flags", c_uint),
        ("value_bits", c_uint64),
    ]


class GPUClothSimulationConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("solver_mask", c_uint),
        ("quality_steps", c_uint),
        ("time_scale", c_float),
        ("vertex_mass", c_float),
        ("gravity", c_float * 3),
        ("air_damping", c_float),
        ("simulation_flags", c_uint),
        ("reserved", c_uint * 3),
    ]


class GPUClothMaterialConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("bending_model", c_uint),
        ("material_flags", c_uint),
        ("stiffness", c_float * 4),
        ("stiffness_max", c_float * 4),
        ("damping", c_float * 4),
        ("reserved", c_uint * 2),
    ]


class GPUClothPinConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("goal_min", c_float),
        ("goal_max", c_float),
        ("goal_default", c_float),
        ("goal_stiffness", c_float),
        ("goal_damping", c_float),
        ("pin_stiffness", c_float),
        ("reserved", c_uint * 2),
    ]


class GPUClothConstraintConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("constraint_flags", c_uint),
        ("reserved0", c_uint),
        ("sewing_force_max", c_float),
        ("internal_spring_max_length", c_float),
        ("internal_spring_max_diversion", c_float),
        ("internal_tension_stiffness", c_float),
        ("internal_tension_stiffness_max", c_float),
        ("internal_compression_stiffness", c_float),
        ("internal_compression_stiffness_max", c_float),
        ("reserved", c_uint * 3),
    ]


class GPUClothPressureConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("pressure_flags", c_uint),
        ("reserved0", c_uint),
        ("uniform_pressure_force", c_float),
        ("target_volume", c_float),
        ("pressure_factor", c_float),
        ("fluid_density", c_float),
        ("reserved", c_uint * 2),
    ]


class GPUClothCollisionConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("collision_flags", c_uint),
        ("collision_quality", c_uint),
        ("distance_min", c_float),
        ("friction", c_float),
        ("damping", c_float),
        ("impulse_clamp", c_float),
        ("self_distance_min", c_float),
        ("self_friction", c_float),
        ("self_impulse_clamp", c_float),
        ("reserved", c_uint * 3),
    ]


class GPUClothColliderConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("object_id", c_uint64),
        ("collection_id", c_uint64),
        ("topology_generation", c_uint64),
        ("geometry_generation", c_uint64),
        ("collider_flags", c_uint),
        ("vertex_count", c_uint),
        ("triangle_count", c_uint),
        ("reserved0", c_uint),
        ("positions_previous", GPUClothBufferView),
        ("positions_current", GPUClothBufferView),
        ("positions_next", GPUClothBufferView),
        ("triangles", GPUClothBufferView),
        ("thickness_outer", c_float),
        ("friction", c_float),
        ("damping", c_float),
        ("effector_absorption", c_float),
        ("reserved", c_uint * 4),
    ]


class GPUClothMeshStateConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("object_id", c_uint64),
        ("topology_generation", c_uint64),
        ("geometry_generation", c_uint64),
        ("rest_generation", c_uint64),
        ("mesh_flags", c_uint),
        ("reserved0", c_uint),
        ("positions_previous", GPUClothBufferView),
        ("positions_current", GPUClothBufferView),
        ("rest_positions", GPUClothBufferView),
        ("normals", GPUClothBufferView),
        ("triangles", GPUClothBufferView),
    ]


class GPUClothEffectorConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("object_id", c_uint64),
        ("collection_id", c_uint64),
        ("generation", c_uint64),
        ("effector_flags", c_uint),
        ("field_type", c_uint),
        ("shape_type", c_uint),
        ("falloff_type", c_uint),
        ("named_value_count", c_uint),
        ("reserved0", c_uint),
        ("named_values_address", c_uint64),
        ("object_matrix", GPUClothBufferView),
        ("inverse_matrix", GPUClothBufferView),
        ("reserved", c_uint64 * 1),
    ]


class GPUClothEffectorWeightsConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("collection_id", c_uint64),
        ("weight_flags", c_uint),
        ("weight_count", c_uint),
        ("weights", c_float * 15),
        ("reserved", c_uint),
    ]


class GPUClothCacheConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("storage_mode", c_uint),
        ("compression_mode", c_uint),
        ("frame_start", c_int),
        ("frame_end", c_int),
        ("frame_step", c_int),
        ("cache_index", c_uint),
        ("cache_flags", c_uint),
        ("reserved0", c_uint),
        ("cache_id", c_uint64),
        ("path_utf8_address", c_uint64),
        ("name_utf8_address", c_uint64),
        ("reserved", c_uint64 * 3),
    ]


class GPUClothSewingRecord(Structure):
    _fields_ = [
        ("seam_id", c_uint64),
        ("vertex_a", c_uint),
        ("vertex_b", c_uint),
        ("stiffness", c_float),
        ("rest_length", c_float),
        ("activation", c_float),
        ("flags", c_uint),
    ]


class GPUClothSewingConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("records", GPUClothBufferView),
        ("phase_count", c_uint),
        ("sewing_flags", c_uint),
        ("activation_speed", c_float),
        ("reserved", c_uint * 3),
    ]


class GPUClothProxyConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("render_object_id", c_uint64),
        ("proxy_object_id", c_uint64),
        ("topology_generation", c_uint64),
        ("render_x_count", c_uint),
        ("render_y_count", c_uint),
        ("proxy_x_count", c_uint),
        ("proxy_y_count", c_uint),
        ("render_vertex_count", c_uint),
        ("proxy_vertex_count", c_uint),
        ("proxy_flags", c_uint),
        ("reserved0", c_uint),
        ("render_rest_positions", GPUClothBufferView),
        ("proxy_rest_positions", GPUClothBufferView),
        ("reserved", c_uint64 * 1),
    ]


class GPUClothDiagnosticsConfig(Structure):
    _fields_ = [
        ("header", GPUClothFeatureConfigHeader),
        ("diagnostics_flags", c_uint),
        ("event_capacity", c_uint),
        ("minimum_severity", c_uint),
        ("reserved0", c_uint),
        ("callback_address", c_uint64),
        ("callback_user_data", c_uint64),
        ("event_buffer", GPUClothBufferView),
        ("reserved", c_uint64 * 1),
    ]


class GPUClothDescriptorLayout(Structure):
    _fields_ = [
        ("struct_size", c_uint),
        ("schema_version", c_uint),
        ("feature_config_header_size", c_uint),
        ("buffer_view_size", c_uint),
        ("named_value_size", c_uint),
        ("simulation_config_size", c_uint),
        ("material_config_size", c_uint),
        ("pin_config_size", c_uint),
        ("constraint_config_size", c_uint),
        ("pressure_config_size", c_uint),
        ("collision_config_size", c_uint),
        ("collider_config_size", c_uint),
        ("mesh_state_config_size", c_uint),
        ("effector_config_size", c_uint),
        ("effector_weights_config_size", c_uint),
        ("cache_config_size", c_uint),
        ("sewing_record_size", c_uint),
        ("sewing_config_size", c_uint),
        ("vertex_channel_config_size", c_uint),
        ("collision_filter_config_size", c_uint),
        ("proxy_config_size", c_uint),
        ("diagnostics_config_size", c_uint),
        ("reserved", c_uint * 3),
    ]

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
            ("max_internal_compression", c_float),
            ("tension_u",               c_float),
            ("tension_v",               c_float),
            ("compression_u",           c_float),
            ("compression_v",           c_float),
            ("bending_u",               c_float),
            ("bending_v",               c_float),
            ("max_tension_u",           c_float),
            ("max_tension_v",           c_float),
            ("max_compression_u",       c_float),
            ("max_compression_v",       c_float),
            ("max_bend_u",              c_float),
            ("max_bend_v",              c_float),
            ("cn_phases", c_short),
            ("cn_sewing_speed", c_float),
            ("cn_seam_stiffness", c_float),
            ("cn_button_stiffness", c_float),
            ("cn_zipper_stiffness", c_float),
            ("cn_dart_stiffness", c_float),
            ("cn_enable_selfcoll_stitching", c_short),
            ("solver_substeps", c_short),
            ("solver_iterations", c_short),
            ("solver_omega", c_float),
            ("solver_drag", c_float),
            ("solver_vel_damp", c_float),
            ("solver_small_steps", c_short),
            ("solver_adaptive", c_short),
            ("solver_max_iterations", c_short),
            ("solver_convergence_tol", c_float),
            ("solver_ptb_stretch", c_short),
            ("solver_ptb_bending", c_short),
            ("solver_ptb_shear", c_short),
            ("solver_ptb_seam", c_short),
            ("solver_use_pt_budget",         c_short),
            ("use_anisotropy",               c_short),
            ("_pad_aniso2",                  c_short)
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
        ("pressure_factor", c_float),
        ("coll_thickness",  c_float),
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
        ("mlooptri", c_void_p),
        ("mlooptri_num", c_int),
        ("runtime", c_void_p),
    ]

class ClothModifierData(Structure):
    _fields_ = [
        # from ModifierData, 'cause struct ClothModifierData : ModifierData
        ("next", POINTER(ModifierData)),
        ("prev", POINTER(ModifierData)),
        ("type", c_int),
        ("mode", c_int),
        ("id", c_uint),
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
