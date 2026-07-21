#pragma once

#include <cstddef>
#include <cstdint>

enum GPUClothSolverMask : uint32_t {
    GPUCLOTH_SOLVER_NONE = 0,
    GPUCLOTH_SOLVER_XPBD = 1u << 0,
    GPUCLOTH_SOLVER_PD = 1u << 1,
    GPUCLOTH_SOLVER_MIL2 = 1u << 2,
    GPUCLOTH_SOLVER_ALL = GPUCLOTH_SOLVER_XPBD | GPUCLOTH_SOLVER_PD | GPUCLOTH_SOLVER_MIL2,
};

enum GPUClothFeatureStatus : uint32_t {
    GPUCLOTH_FEATURE_MISSING = 0,
    GPUCLOTH_FEATURE_PARTIAL = 1,
    GPUCLOTH_FEATURE_IMPLEMENTED = 2,
    GPUCLOTH_FEATURE_PROVEN = 3,
};

enum GPUClothFeatureFlags : uint32_t {
    GPUCLOTH_FEATURE_BLENDER_CORE = 1u << 0,
    GPUCLOTH_FEATURE_RELEASE_REQUIRED = 1u << 1,
    GPUCLOTH_FEATURE_FUTURE_CONFIG = 1u << 2,
    GPUCLOTH_FEATURE_EXTENSION = 1u << 3,
};

enum GPUClothConfigKindMask : uint32_t {
    GPUCLOTH_CONFIG_NONE = 0,
    GPUCLOTH_CONFIG_SIMULATION = 1u << 0,
    GPUCLOTH_CONFIG_MATERIAL = 1u << 1,
    GPUCLOTH_CONFIG_PIN = 1u << 2,
    GPUCLOTH_CONFIG_CONSTRAINT = 1u << 3,
    GPUCLOTH_CONFIG_PRESSURE = 1u << 4,
    GPUCLOTH_CONFIG_COLLISION = 1u << 5,
    GPUCLOTH_CONFIG_COLLIDER = 1u << 6,
    GPUCLOTH_CONFIG_MESH_STATE = 1u << 7,
    GPUCLOTH_CONFIG_EFFECTOR = 1u << 8,
    GPUCLOTH_CONFIG_EFFECTOR_WEIGHTS = 1u << 9,
    GPUCLOTH_CONFIG_CACHE = 1u << 10,
    GPUCLOTH_CONFIG_SEWING = 1u << 11,
    GPUCLOTH_CONFIG_VERTEX_CHANNEL = 1u << 12,
    GPUCLOTH_CONFIG_COLLISION_FILTER = 1u << 13,
    GPUCLOTH_CONFIG_PROXY = 1u << 14,
    GPUCLOTH_CONFIG_DIAGNOSTICS = 1u << 15,
};

enum GPUClothFeatureId : uint32_t {
    GPUCLOTH_FEATURE_BASIC_DYNAMICS = 1,
    GPUCLOTH_FEATURE_TIMESTEP_SPEED,
    GPUCLOTH_FEATURE_MATERIAL_MASS,
    GPUCLOTH_FEATURE_STRETCH,
    GPUCLOTH_FEATURE_COMPRESSION,
    GPUCLOTH_FEATURE_SHEAR,
    GPUCLOTH_FEATURE_BENDING_LINEAR,
    GPUCLOTH_FEATURE_BENDING_ANGULAR,
    GPUCLOTH_FEATURE_MATERIAL_DAMPING,
    GPUCLOTH_FEATURE_PIN_GOAL,
    GPUCLOTH_FEATURE_ANIMATED_PIN,
    GPUCLOTH_FEATURE_SEWING,
    GPUCLOTH_FEATURE_SHRINK,
    GPUCLOTH_FEATURE_DYNAMIC_MESH,
    GPUCLOTH_FEATURE_REST_SHAPE_KEY,
    GPUCLOTH_FEATURE_INTERNAL_SPRINGS,
    GPUCLOTH_FEATURE_PRESSURE_UNIFORM,
    GPUCLOTH_FEATURE_PRESSURE_VOLUME,
    GPUCLOTH_FEATURE_FLUID_DENSITY,
    GPUCLOTH_FEATURE_PRESSURE_VERTEX_GROUP,
    GPUCLOTH_FEATURE_ANISOTROPY,
    GPUCLOTH_FEATURE_STATIC_OBJECT_COLLISION,
    GPUCLOTH_FEATURE_MOVING_OBJECT_COLLISION,
    GPUCLOTH_FEATURE_DEFORMING_OBJECT_COLLISION,
    GPUCLOTH_FEATURE_COLLISION_FRICTION_DAMPING,
    GPUCLOTH_FEATURE_COLLISION_QUALITY_CLAMP,
    GPUCLOTH_FEATURE_COLLISION_COLLECTION,
    GPUCLOTH_FEATURE_COLLISION_VERTEX_GROUP,
    GPUCLOTH_FEATURE_SELF_COLLISION,
    GPUCLOTH_FEATURE_SELF_COLLISION_FRICTION,
    GPUCLOTH_FEATURE_SELF_COLLISION_VERTEX_GROUP,
    GPUCLOTH_FEATURE_EFFECTORS,
    GPUCLOTH_FEATURE_EFFECTOR_WEIGHTS,
    GPUCLOTH_FEATURE_EFFECTOR_COLLECTION,
    GPUCLOTH_FEATURE_CACHE_DISK,
    GPUCLOTH_FEATURE_CACHE_MEMORY,
    GPUCLOTH_FEATURE_CACHE_EXTERNAL,
    GPUCLOTH_FEATURE_CACHE_MULTIPLE,
    GPUCLOTH_FEATURE_CACHE_COMPRESSION,
    GPUCLOTH_FEATURE_BAKE_RANGE,
    GPUCLOTH_FEATURE_CALCULATE_TO_FRAME,
    GPUCLOTH_FEATURE_PROXY,
    GPUCLOTH_FEATURE_READBACK,
    GPUCLOTH_FEATURE_BACKEND_XPBD,
    GPUCLOTH_FEATURE_BACKEND_PD,
    GPUCLOTH_FEATURE_BACKEND_MIL2,
    GPUCLOTH_FEATURE_LIFECYCLE,
    GPUCLOTH_FEATURE_MASS_VERTEX_GROUP,
    GPUCLOTH_FEATURE_STIFFNESS_VERTEX_GROUPS,
    GPUCLOTH_FEATURE_GRAVITY_VECTOR,
    GPUCLOTH_FEATURE_SIMULATION_QUALITY,
    GPUCLOTH_FEATURE_AIR_DAMPING,
    GPUCLOTH_FEATURE_COLLIDER_SURFACE_CONTROLS,
    GPUCLOTH_FEATURE_MODIFIER_EVALUATION,
    GPUCLOTH_FEATURE_SOLVER_DIAGNOSTICS,
    GPUCLOTH_FEATURE_CACHE_STATUS,
};

enum GPUClothABIResult : uint32_t {
    GPUCLOTH_ABI_OK = 0,
    GPUCLOTH_ABI_INVALID_ARGUMENT = 1,
    GPUCLOTH_ABI_STRUCT_TOO_SMALL = 2,
    GPUCLOTH_ABI_UNKNOWN_FEATURE = 3,
    GPUCLOTH_ABI_UNSUPPORTED = 4,
    GPUCLOTH_ABI_NOT_CONFIGURABLE = 5,
    GPUCLOTH_ABI_VERSION_MISMATCH = 6,
    GPUCLOTH_ABI_COUNT_MISMATCH = 7,
    GPUCLOTH_ABI_INVALID_VALUE = 8,
    GPUCLOTH_ABI_INVALID_STATE = 9,
};

struct GPUClothABIVersion {
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t abi_patch;
    uint32_t feature_schema_version;
    uint32_t feature_count;
    uint32_t reserved[2];
};

struct GPUClothFeatureInfo {
    uint32_t struct_size;
    uint32_t feature_id;
    uint32_t status;
    uint32_t supported_solver_mask;
    uint32_t proven_solver_mask;
    uint32_t flags;
    uint32_t config_version;
    uint32_t config_kind_mask;
    char name[64];
    char detail[192];
};

struct GPUClothHostLayout {
    uint32_t struct_size;
    uint32_t schema_version;
    uint32_t cloth_vertex_size;
    uint32_t cloth_vertex_x_offset;
    uint32_t mesh_size;
    uint32_t mesh_mlooptri_offset;
    uint32_t mesh_runtime_offset;
    uint32_t cloth_modifier_data_size;
    uint32_t cloth_modifier_data_cloth_offset;
    uint32_t cloth_modifier_data_sim_offset;
    uint32_t cloth_modifier_data_coll_offset;
    uint32_t collision_modifier_data_size;
    uint32_t collision_modifier_data_bvh_offset;
    uint32_t object_size;
    uint32_t object_modifiers_offset;
    uint32_t object_pd_offset;
    uint32_t scene_size;
    uint32_t cloth_sim_settings_size;
    uint32_t cloth_coll_settings_size;
    uint32_t mvert_size;
    uint32_t medge_size;
    uint32_t mpoly_size;
    uint32_t mloop_size;
    uint32_t mvert_tri_size;
    uint32_t reserved[2];
};

// Common prefix for future typed configuration descriptors. Unsupported
// descriptors must return an explicit result instead of being silently ignored.
struct GPUClothFeatureConfigHeader {
    uint32_t struct_size;
    uint32_t feature_id;
    uint32_t config_version;
    uint32_t flags;
};

enum GPUClothBufferElementType : uint32_t {
    GPUCLOTH_ELEMENT_FLOAT = 1,
    GPUCLOTH_ELEMENT_FLOAT3,
    GPUCLOTH_ELEMENT_UINT32,
    GPUCLOTH_ELEMENT_UINT3,
    GPUCLOTH_ELEMENT_SEWING_RECORD,
};

struct GPUClothBufferView {
    uint32_t struct_size;
    uint32_t element_type;
    uint64_t element_count;
    uint64_t stride_bytes;
    uint64_t data_address;
    uint64_t generation;
};

enum GPUClothNamedValueType : uint32_t {
    GPUCLOTH_VALUE_BOOL = 1,
    GPUCLOTH_VALUE_INT32,
    GPUCLOTH_VALUE_UINT32,
    GPUCLOTH_VALUE_FLOAT32,
    GPUCLOTH_VALUE_FLOAT64,
    GPUCLOTH_VALUE_OBJECT_ID,
};

struct GPUClothNamedValue {
    char name[48];
    uint32_t value_type;
    uint32_t flags;
    uint64_t value_bits;
};

struct GPUClothSimulationConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t solver_mask;
    uint32_t quality_steps;
    float time_scale;
    float vertex_mass;
    float gravity[3];
    float air_damping;
    uint32_t simulation_flags;
    uint32_t reserved[3];
};

struct GPUClothMaterialConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t bending_model;
    uint32_t material_flags;
    float stiffness[4];
    float stiffness_max[4];
    float damping[4];
    uint32_t reserved[2];
};

struct GPUClothPinConfig {
    GPUClothFeatureConfigHeader header;
    float goal_min;
    float goal_max;
    float goal_default;
    float goal_stiffness;
    float goal_damping;
    float pin_stiffness;
    uint32_t reserved[2];
};

struct GPUClothConstraintConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t constraint_flags;
    uint32_t reserved0;
    float sewing_force_max;
    float internal_spring_max_length;
    float internal_spring_max_diversion;
    float internal_tension_stiffness;
    float internal_tension_stiffness_max;
    float internal_compression_stiffness;
    float internal_compression_stiffness_max;
    uint32_t reserved[3];
};

struct GPUClothPressureConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t pressure_flags;
    uint32_t reserved0;
    float uniform_pressure_force;
    float target_volume;
    float pressure_factor;
    float fluid_density;
    uint32_t reserved[2];
};

struct GPUClothCollisionConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t collision_flags;
    uint32_t collision_quality;
    float distance_min;
    float friction;
    float damping;
    float impulse_clamp;
    float self_distance_min;
    float self_friction;
    float self_impulse_clamp;
    uint32_t reserved[3];
};

struct GPUClothColliderConfig {
    GPUClothFeatureConfigHeader header;
    uint64_t object_id;
    uint64_t collection_id;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    uint32_t collider_flags;
    uint32_t vertex_count;
    uint32_t triangle_count;
    uint32_t reserved0;
    GPUClothBufferView positions_previous;
    GPUClothBufferView positions_current;
    GPUClothBufferView positions_next;
    GPUClothBufferView triangles;
    float thickness_outer;
    float friction;
    float damping;
    float effector_absorption;
    uint32_t reserved[4];
};

struct GPUClothMeshStateConfig {
    GPUClothFeatureConfigHeader header;
    uint64_t object_id;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    uint64_t rest_generation;
    uint32_t mesh_flags;
    uint32_t reserved0;
    GPUClothBufferView positions_previous;
    GPUClothBufferView positions_current;
    GPUClothBufferView rest_positions;
    GPUClothBufferView normals;
    GPUClothBufferView triangles;
};

struct GPUClothEffectorConfig {
    GPUClothFeatureConfigHeader header;
    uint64_t object_id;
    uint64_t collection_id;
    uint64_t generation;
    uint32_t effector_flags;
    uint32_t field_type;
    uint32_t shape_type;
    uint32_t falloff_type;
    uint32_t named_value_count;
    uint32_t reserved0;
    uint64_t named_values_address;
    GPUClothBufferView object_matrix;
    GPUClothBufferView inverse_matrix;
    uint64_t reserved[1];
};

struct GPUClothEffectorWeightsConfig {
    GPUClothFeatureConfigHeader header;
    uint64_t collection_id;
    uint32_t weight_flags;
    uint32_t weight_count;
    float weights[15];
    uint32_t reserved;
};

enum GPUClothVertexChannel : uint32_t {
    GPUCLOTH_VERTEX_MASS = 1,
    GPUCLOTH_VERTEX_PIN_WEIGHT,
    GPUCLOTH_VERTEX_PIN_TARGET_XYZ,
    GPUCLOTH_VERTEX_SHRINK_WEIGHT,
    GPUCLOTH_VERTEX_PRESSURE_WEIGHT,
    GPUCLOTH_VERTEX_STRUCTURAL_STIFFNESS,
    GPUCLOTH_VERTEX_SHEAR_STIFFNESS,
    GPUCLOTH_VERTEX_BENDING_STIFFNESS,
    GPUCLOTH_VERTEX_INTERNAL_STIFFNESS,
    GPUCLOTH_VERTEX_OBJECT_COLLISION_MASK,
    GPUCLOTH_VERTEX_SELF_COLLISION_MASK,
};

struct GPUClothVertexChannelConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t channel;
    uint32_t element_width;
    uint64_t element_count;
    uint64_t data_address;
};

struct GPUClothCollisionFilterConfig {
    GPUClothFeatureConfigHeader header;
    uint64_t collection_id;
    uint32_t object_mask_channel;
    uint32_t self_mask_channel;
};

struct GPUClothCacheConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t storage_mode;
    uint32_t compression_mode;
    int32_t frame_start;
    int32_t frame_end;
    int32_t frame_step;
    uint32_t cache_index;
    uint32_t cache_flags;
    uint32_t reserved0;
    uint64_t cache_id;
    uint64_t path_utf8_address;
    uint64_t name_utf8_address;
    uint64_t reserved[3];
};

struct GPUClothSewingRecord {
    uint64_t seam_id;
    uint32_t vertex_a;
    uint32_t vertex_b;
    float stiffness;
    float rest_length;
    float activation;
    uint32_t flags;
};

struct GPUClothSewingConfig {
    GPUClothFeatureConfigHeader header;
    GPUClothBufferView records;
    uint32_t phase_count;
    uint32_t sewing_flags;
    float activation_speed;
    uint32_t reserved[3];
};

struct GPUClothProxyConfig {
    GPUClothFeatureConfigHeader header;
    uint64_t render_object_id;
    uint64_t proxy_object_id;
    uint64_t topology_generation;
    uint32_t render_x_count;
    uint32_t render_y_count;
    uint32_t proxy_x_count;
    uint32_t proxy_y_count;
    uint32_t render_vertex_count;
    uint32_t proxy_vertex_count;
    uint32_t proxy_flags;
    uint32_t reserved0;
    GPUClothBufferView render_rest_positions;
    GPUClothBufferView proxy_rest_positions;
    uint64_t reserved[1];
};

struct GPUClothDiagnosticsConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t diagnostics_flags;
    uint32_t event_capacity;
    uint32_t minimum_severity;
    uint32_t reserved0;
    uint64_t callback_address;
    uint64_t callback_user_data;
    GPUClothBufferView event_buffer;
    uint64_t reserved[1];
};

struct GPUClothDescriptorLayout {
    uint32_t struct_size;
    uint32_t schema_version;
    uint32_t feature_config_header_size;
    uint32_t buffer_view_size;
    uint32_t named_value_size;
    uint32_t simulation_config_size;
    uint32_t material_config_size;
    uint32_t pin_config_size;
    uint32_t constraint_config_size;
    uint32_t pressure_config_size;
    uint32_t collision_config_size;
    uint32_t collider_config_size;
    uint32_t mesh_state_config_size;
    uint32_t effector_config_size;
    uint32_t effector_weights_config_size;
    uint32_t cache_config_size;
    uint32_t sewing_record_size;
    uint32_t sewing_config_size;
    uint32_t vertex_channel_config_size;
    uint32_t collision_filter_config_size;
    uint32_t proxy_config_size;
    uint32_t diagnostics_config_size;
    uint32_t reserved[3];
};

static_assert(sizeof(GPUClothABIVersion) == 32, "GPUClothABIVersion ABI drift");
static_assert(sizeof(GPUClothFeatureInfo) == 288, "GPUClothFeatureInfo ABI drift");
static_assert(sizeof(GPUClothHostLayout) == 104, "GPUClothHostLayout ABI drift");
static_assert(sizeof(GPUClothFeatureConfigHeader) == 16, "GPUClothFeatureConfigHeader ABI drift");
static_assert(sizeof(GPUClothBufferView) == 40, "GPUClothBufferView ABI drift");
static_assert(sizeof(GPUClothNamedValue) == 64, "GPUClothNamedValue ABI drift");
static_assert(sizeof(GPUClothSimulationConfig) == 64, "GPUClothSimulationConfig ABI drift");
static_assert(sizeof(GPUClothMaterialConfig) == 80, "GPUClothMaterialConfig ABI drift");
static_assert(sizeof(GPUClothPinConfig) == 48, "GPUClothPinConfig ABI drift");
static_assert(sizeof(GPUClothConstraintConfig) == 64, "GPUClothConstraintConfig ABI drift");
static_assert(sizeof(GPUClothPressureConfig) == 48, "GPUClothPressureConfig ABI drift");
static_assert(sizeof(GPUClothCollisionConfig) == 64, "GPUClothCollisionConfig ABI drift");
static_assert(sizeof(GPUClothColliderConfig) == 256, "GPUClothColliderConfig ABI drift");
static_assert(sizeof(GPUClothMeshStateConfig) == 256, "GPUClothMeshStateConfig ABI drift");
static_assert(sizeof(GPUClothEffectorConfig) == 160, "GPUClothEffectorConfig ABI drift");
static_assert(sizeof(GPUClothEffectorWeightsConfig) == 96, "GPUClothEffectorWeightsConfig ABI drift");
static_assert(sizeof(GPUClothCacheConfig) == 96, "GPUClothCacheConfig ABI drift");
static_assert(sizeof(GPUClothSewingRecord) == 32, "GPUClothSewingRecord ABI drift");
static_assert(sizeof(GPUClothSewingConfig) == 80, "GPUClothSewingConfig ABI drift");
static_assert(sizeof(GPUClothVertexChannelConfig) == 40, "GPUClothVertexChannelConfig ABI drift");
static_assert(sizeof(GPUClothCollisionFilterConfig) == 32, "GPUClothCollisionFilterConfig ABI drift");
static_assert(sizeof(GPUClothProxyConfig) == 160, "GPUClothProxyConfig ABI drift");
static_assert(sizeof(GPUClothDiagnosticsConfig) == 96, "GPUClothDiagnosticsConfig ABI drift");
static_assert(sizeof(GPUClothDescriptorLayout) == 100, "GPUClothDescriptorLayout ABI drift");
