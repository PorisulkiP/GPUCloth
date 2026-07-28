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
    GPUCLOTH_CONFIG_COLLECTION = 1u << 16,
};

enum GPUClothPressureFlags : uint32_t {
    GPUCLOTH_PRESSURE_ENABLED = 1u << 0,
};

enum GPUClothCacheStorageMode : uint32_t {
    GPUCLOTH_CACHE_STORAGE_DISK = 1,
    GPUCLOTH_CACHE_STORAGE_MEMORY = 2,
    GPUCLOTH_CACHE_STORAGE_EXTERNAL = 3,
};

enum GPUClothCacheFlags : uint32_t {
    GPUCLOTH_CACHE_FLAG_EXTERNAL_READ_ONLY = 1u << 0,
    GPUCLOTH_CACHE_FLAG_LIBRARY_PATH = 1u << 1,
};

enum GPUClothCacheCompressionMode : uint32_t {
    GPUCLOTH_CACHE_COMPRESSION_NONE = 0,
    GPUCLOTH_CACHE_COMPRESSION_LIGHT = 1,
    GPUCLOTH_CACHE_COMPRESSION_HEAVY = 2,
};

enum GPUClothCacheStatusFlags : uint32_t {
    GPUCLOTH_CACHE_STATUS_CONFIGURED = 1u << 0,
    GPUCLOTH_CACHE_STATUS_BAKING = 1u << 1,
    GPUCLOTH_CACHE_STATUS_BAKED = 1u << 2,
    GPUCLOTH_CACHE_STATUS_OUTDATED = 1u << 3,
    GPUCLOTH_CACHE_STATUS_FRAME_SKIP = 1u << 4,
    GPUCLOTH_CACHE_STATUS_ERROR = 1u << 5,
};

enum GPUClothCacheStatusOperation : uint32_t {
    GPUCLOTH_CACHE_STATUS_BAKE_BEGIN = 1,
    GPUCLOTH_CACHE_STATUS_BAKE_COMPLETE,
    GPUCLOTH_CACHE_STATUS_BAKE_CANCEL,
    GPUCLOTH_CACHE_STATUS_SOURCE_CHANGED,
};

enum GPUClothCacheInfoCode : uint32_t {
    GPUCLOTH_CACHE_INFO_EMPTY = 0,
    GPUCLOTH_CACHE_INFO_READY,
    GPUCLOTH_CACHE_INFO_BAKING,
    GPUCLOTH_CACHE_INFO_BAKED,
    GPUCLOTH_CACHE_INFO_OUTDATED,
    GPUCLOTH_CACHE_INFO_FRAME_SKIP,
    GPUCLOTH_CACHE_INFO_ERROR,
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
    GPUCLOTH_ELEMENT_DIAGNOSTIC_EVENT,
    GPUCLOTH_ELEMENT_COLLECTION_RECORD,
    GPUCLOTH_ELEMENT_FLOAT2,
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

enum GPUClothMaterialFlags : uint32_t {
    GPUCLOTH_MATERIAL_ANISOTROPY_ENABLED = 1u << 0,
};

struct GPUClothMaterialConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t bending_model;
    uint32_t material_flags;
    float stiffness[4];
    float stiffness_max[4];
    float damping[4];
    // ABI v2 suffix used only by GPUCLOTH_FEATURE_ANISOTROPY:
    // tension U/V, compression U/V, bending U/V.
    float directional_stiffness[6];
    float directional_stiffness_max[6];
    GPUClothBufferView material_coordinates;
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

enum GPUClothPinSnapshotFlags : uint32_t {
    GPUCLOTH_PIN_GROUP_PRESENT = 1u << 0,
};

// One evaluated-frame pin publication. Membership, raw Blender group weights,
// evaluated targets, and goal settings become visible atomically. The native
// owner retains the last accepted target snapshot as frame history.
struct GPUClothPinSnapshotConfig {
    GPUClothFeatureConfigHeader header;
    uint64_t object_id;
    uint64_t topology_generation;
    uint64_t frame_generation;
    uint32_t pin_flags;
    uint32_t vertex_count;
    float goal_min;
    float goal_max;
    float goal_default;
    float goal_spring;
    float goal_damping;
    uint32_t reserved0;
    GPUClothBufferView membership;
    GPUClothBufferView raw_weights;
    GPUClothBufferView evaluated_targets;
    uint64_t reserved[1];
};

enum GPUClothConstraintFlags : uint32_t {
    GPUCLOTH_CONSTRAINT_INTERNAL_SPRINGS = 1u << 0,
    GPUCLOTH_CONSTRAINT_INTERNAL_NORMAL_CHECK = 1u << 1,
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

enum GPUClothCollisionFlags : uint32_t {
    GPUCLOTH_COLLISION_OBJECT_ENABLED = 1u << 0,
    GPUCLOTH_COLLISION_SELF_ENABLED = 1u << 1,
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

enum GPUClothColliderFlags : uint32_t {
    GPUCLOTH_COLLIDER_STATIC = 1u << 0,
    GPUCLOTH_COLLIDER_MOVING = 1u << 1,
    GPUCLOTH_COLLIDER_DEFORMING = 1u << 2,
    GPUCLOTH_COLLIDER_USE_CULLING = 1u << 3,
    GPUCLOTH_COLLIDER_USE_NORMAL = 1u << 4,
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

enum GPUClothMeshStateFlags : uint32_t {
    // Per-frame evaluated base-mesh positions. The accepted topology remains
    // fixed; only xrest/rest lengths/rest angles may change.
    GPUCLOTH_MESH_DYNAMIC_BASE = 1u << 0,
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

enum GPUClothEffectorFlags : uint32_t {
    GPUCLOTH_EFFECTOR_USE_ABSORPTION = 1u << 0,
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

enum GPUClothCollectionKind : uint32_t {
    GPUCLOTH_COLLECTION_COLLISION = 1,
    GPUCLOTH_COLLECTION_EFFECTOR = 2,
};

enum GPUClothCollectionObjectType : uint32_t {
    GPUCLOTH_COLLECTION_OBJECT_MESH = 1,
    GPUCLOTH_COLLECTION_OBJECT_CURVE = 2,
    GPUCLOTH_COLLECTION_OBJECT_EMPTY = 3,
    GPUCLOTH_COLLECTION_OBJECT_OTHER = 255,
};

enum GPUClothCollectionRecordFlags : uint32_t {
    GPUCLOTH_COLLECTION_RECORD_INSTANCE = 1u << 0,
    GPUCLOTH_COLLECTION_RECORD_EVALUATED = 1u << 1,
    GPUCLOTH_COLLECTION_RECORD_VIEWPORT_ENABLED = 1u << 2,
    GPUCLOTH_COLLECTION_RECORD_RENDER_ENABLED = 1u << 3,
};

enum GPUClothCollectionStatusFlags : uint32_t {
    GPUCLOTH_COLLECTION_STATUS_CONFIGURED = 1u << 0,
    GPUCLOTH_COLLECTION_STATUS_STAGED = 1u << 1,
};

// Ordered Blender dependency-graph occurrence. payload_address points to a
// GPUClothColliderConfig or GPUClothEffectorConfig during the synchronous
// stage call and is always deep-copied by the native owner.
struct GPUClothCollectionRecord {
    uint32_t struct_size;
    uint32_t record_version;
    uint32_t object_type;
    uint32_t record_flags;
    uint64_t object_id;
    uint64_t instance_id;
    uint64_t source_collection_id;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    uint32_t modifier_index;
    uint32_t payload_kind;
    uint64_t payload_address;
    uint64_t reserved[1];
};

struct GPUClothCollectionSnapshotConfig {
    GPUClothFeatureConfigHeader header;
    uint64_t cloth_id;
    uint64_t collection_id;
    uint64_t snapshot_generation;
    uint32_t collection_kind;
    uint32_t collection_flags;
    uint32_t record_count;
    uint32_t reserved0;
    GPUClothBufferView records;
    uint64_t reserved[8];
};

struct GPUClothCollectionTransactionConfig {
    uint32_t struct_size;
    uint32_t transaction_version;
    uint32_t transaction_flags;
    uint32_t reserved0;
    uint64_t source_generation;
    uint64_t requested_transaction_id;
    uint64_t reserved[2];
};

// Caller supplies records_address/record_capacity. The query always returns
// required_count and never publishes a partial record set.
struct GPUClothCollectionQuery {
    uint32_t struct_size;
    uint32_t query_version;
    uint32_t collection_kind;
    uint32_t query_flags;
    uint64_t cloth_id;
    uint64_t collection_id;
    uint64_t snapshot_generation;
    uint64_t transaction_id;
    uint32_t record_capacity;
    uint32_t record_count;
    uint32_t required_count;
    uint32_t reserved0;
    uint64_t records_address;
    uint64_t reserved[2];
};

struct GPUClothCollectionStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t status_flags;
    uint32_t last_error;
    uint64_t transaction_id;
    uint64_t snapshot_generation;
    uint64_t collection_id;
    uint32_t collision_record_count;
    uint32_t effector_record_count;
    uint32_t record_count;
    uint32_t reserved0;
    uint64_t commit_generation;
    uint64_t reserved[4];
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

struct GPUClothCacheStatusUpdate {
    GPUClothFeatureConfigHeader header;
    uint32_t operation;
    int32_t frame;
    uint32_t error_code;
    uint32_t reserved0;
    uint64_t source_generation;
    uint64_t reserved[2];
};

struct GPUClothCacheStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t flags;
    uint32_t info_code;
    int32_t frame_start;
    int32_t frame_end;
    int32_t frame_step;
    int32_t last_exact;
    int32_t last_valid;
    int32_t outdated_from_frame;
    uint32_t cached_frame_count;
    uint32_t missing_frame_count;
    uint32_t error_code;
    uint64_t source_generation;
    uint64_t baked_source_generation;
    uint64_t status_generation;
    char info[96];
    uint64_t reserved[2];
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

enum GPUClothDiagnosticsFlags : uint32_t {
    GPUCLOTH_DIAGNOSTICS_STATUS = 1u << 0,
    GPUCLOTH_DIAGNOSTICS_EVENTS = 1u << 1,
};

enum GPUClothDiagnosticsSeverity : uint32_t {
    GPUCLOTH_DIAGNOSTICS_INFO = 1,
    GPUCLOTH_DIAGNOSTICS_WARNING,
    GPUCLOTH_DIAGNOSTICS_ERROR,
};

enum GPUClothDiagnosticsEventType : uint32_t {
    GPUCLOTH_DIAGNOSTICS_CONFIGURED = 1,
    GPUCLOTH_DIAGNOSTICS_SOLVE_STARTED,
    GPUCLOTH_DIAGNOSTICS_SOLVE_SUCCEEDED,
    GPUCLOTH_DIAGNOSTICS_SOLVE_FAILED,
};

enum GPUClothDiagnosticsResult : uint32_t {
    GPUCLOTH_DIAGNOSTICS_RESULT_NONE = 0,
    GPUCLOTH_DIAGNOSTICS_RESULT_SUCCESS,
    GPUCLOTH_DIAGNOSTICS_RESULT_FAILED,
};

enum GPUClothDiagnosticsError : uint32_t {
    GPUCLOTH_DIAGNOSTICS_ERROR_NONE = 0,
    GPUCLOTH_DIAGNOSTICS_ERROR_RUNTIME_INIT,
    GPUCLOTH_DIAGNOSTICS_ERROR_NO_CLOTH,
    GPUCLOTH_DIAGNOSTICS_ERROR_INVALID_CLOTH,
    GPUCLOTH_DIAGNOSTICS_ERROR_EMPTY_CLOTH,
    GPUCLOTH_DIAGNOSTICS_ERROR_SOLVER_STATE,
    GPUCLOTH_DIAGNOSTICS_ERROR_PRESSURE_STATE,
    GPUCLOTH_DIAGNOSTICS_ERROR_DEVICE_CLOTH,
    GPUCLOTH_DIAGNOSTICS_ERROR_BACKEND_UNAVAILABLE,
    GPUCLOTH_DIAGNOSTICS_ERROR_PIN_STATE,
    GPUCLOTH_DIAGNOSTICS_ERROR_DYNAMIC_MESH_STATE,
};

enum GPUClothDiagnosticsStatusFlags : uint32_t {
    GPUCLOTH_DIAGNOSTICS_STATUS_CONFIGURED = 1u << 0,
    GPUCLOTH_DIAGNOSTICS_STATUS_RUNNING = 1u << 1,
    GPUCLOTH_DIAGNOSTICS_STATUS_HAS_RESULT = 1u << 2,
    GPUCLOTH_DIAGNOSTICS_STATUS_HAS_ERROR = 1u << 3,
};

enum GPUClothSolverResultStatus : uint32_t {
    GPUCLOTH_SOLVER_RESULT_SUCCESS = 1u << 0,
    GPUCLOTH_SOLVER_RESULT_NUMERICAL_ISSUE = 1u << 1,
    GPUCLOTH_SOLVER_RESULT_NO_CONVERGENCE = 1u << 2,
    GPUCLOTH_SOLVER_RESULT_INVALID_INPUT = 1u << 3,
};

enum GPUClothConvergenceMetric : uint32_t {
    GPUCLOTH_CONVERGENCE_LINF_POSITION_DELTA = 1,
};

// Caller-owned event records. A configured buffer remains live until
// RemoveCloth/FreeSolverData and is written only by synchronous API calls.
struct GPUClothDiagnosticsEvent {
    uint32_t struct_size;
    uint32_t event_type;
    uint32_t severity;
    uint32_t result;
    uint32_t error_code;
    uint32_t solver_mask;
    uint64_t sequence;
};

struct GPUClothDiagnosticsStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t status_flags;
    uint32_t last_result;
    uint32_t last_error;
    uint32_t requested_solver_mask;
    uint32_t backend_solver_mask;
    uint32_t solver_result_status;
    uint32_t convergence_metric;
    uint32_t event_capacity;
    uint32_t event_count;
    uint32_t dropped_event_count;
    uint32_t event_write_index;
    uint32_t substep_count;
    uint32_t min_iterations;
    uint32_t max_iterations;
    float avg_iterations;
    float min_error_value;
    float max_error_value;
    float avg_error_value;
    float last_error_value;
    float convergence_tolerance;
    uint64_t total_iterations;
    uint64_t execution_time_ns;
    uint64_t sequence;
    uint64_t solve_count;
    uint64_t failure_count;
    uint64_t event_generation;
    uint64_t reserved[2];
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
    uint32_t cache_status_update_size;
    uint32_t cache_status_size;
    uint32_t sewing_record_size;
    uint32_t sewing_config_size;
    uint32_t vertex_channel_config_size;
    uint32_t collision_filter_config_size;
    uint32_t proxy_config_size;
    uint32_t diagnostics_config_size;
    // ABI 1.25 and earlier exposed this word as reserved at offset 96.
    uint32_t legacy_reserved0;
    uint32_t diagnostics_event_size;
    uint32_t diagnostics_status_size;
    uint32_t pin_snapshot_config_size;
    uint32_t collection_record_size;
    uint32_t collection_snapshot_config_size;
    uint32_t collection_transaction_config_size;
    uint32_t collection_query_size;
    uint32_t collection_status_size;
    uint32_t reserved[3];
};

static_assert(sizeof(GPUClothABIVersion) == 32, "GPUClothABIVersion ABI drift");
static_assert(sizeof(GPUClothFeatureInfo) == 288, "GPUClothFeatureInfo ABI drift");
static_assert(sizeof(GPUClothHostLayout) == 104, "GPUClothHostLayout ABI drift");
static_assert(sizeof(GPUClothFeatureConfigHeader) == 16, "GPUClothFeatureConfigHeader ABI drift");
static_assert(sizeof(GPUClothBufferView) == 40, "GPUClothBufferView ABI drift");
static_assert(sizeof(GPUClothNamedValue) == 64, "GPUClothNamedValue ABI drift");
static_assert(sizeof(GPUClothSimulationConfig) == 64, "GPUClothSimulationConfig ABI drift");
static_assert(sizeof(GPUClothMaterialConfig) == 160, "GPUClothMaterialConfig ABI drift");
static_assert(sizeof(GPUClothPinConfig) == 48, "GPUClothPinConfig ABI drift");
static_assert(sizeof(GPUClothPinSnapshotConfig) == 200, "GPUClothPinSnapshotConfig ABI drift");
static_assert(sizeof(GPUClothConstraintConfig) == 64, "GPUClothConstraintConfig ABI drift");
static_assert(sizeof(GPUClothPressureConfig) == 48, "GPUClothPressureConfig ABI drift");
static_assert(sizeof(GPUClothCollisionConfig) == 64, "GPUClothCollisionConfig ABI drift");
static_assert(sizeof(GPUClothColliderConfig) == 256, "GPUClothColliderConfig ABI drift");
static_assert(sizeof(GPUClothMeshStateConfig) == 256, "GPUClothMeshStateConfig ABI drift");
static_assert(sizeof(GPUClothEffectorConfig) == 160, "GPUClothEffectorConfig ABI drift");
static_assert(sizeof(GPUClothEffectorWeightsConfig) == 96, "GPUClothEffectorWeightsConfig ABI drift");
static_assert(sizeof(GPUClothCollectionRecord) == 80, "GPUClothCollectionRecord ABI drift");
static_assert(sizeof(GPUClothCollectionSnapshotConfig) == 160, "GPUClothCollectionSnapshotConfig ABI drift");
static_assert(sizeof(GPUClothCollectionTransactionConfig) == 48, "GPUClothCollectionTransactionConfig ABI drift");
static_assert(sizeof(GPUClothCollectionQuery) == 88, "GPUClothCollectionQuery ABI drift");
static_assert(sizeof(GPUClothCollectionStatus) == 96, "GPUClothCollectionStatus ABI drift");
static_assert(sizeof(GPUClothCacheConfig) == 96, "GPUClothCacheConfig ABI drift");
static_assert(sizeof(GPUClothCacheStatusUpdate) == 56, "GPUClothCacheStatusUpdate ABI drift");
static_assert(sizeof(GPUClothCacheStatus) == 192, "GPUClothCacheStatus ABI drift");
static_assert(sizeof(GPUClothSewingRecord) == 32, "GPUClothSewingRecord ABI drift");
static_assert(sizeof(GPUClothSewingConfig) == 80, "GPUClothSewingConfig ABI drift");
static_assert(sizeof(GPUClothVertexChannelConfig) == 40, "GPUClothVertexChannelConfig ABI drift");
static_assert(sizeof(GPUClothCollisionFilterConfig) == 32, "GPUClothCollisionFilterConfig ABI drift");
static_assert(sizeof(GPUClothProxyConfig) == 160, "GPUClothProxyConfig ABI drift");
static_assert(sizeof(GPUClothDiagnosticsConfig) == 96, "GPUClothDiagnosticsConfig ABI drift");
static_assert(sizeof(GPUClothDiagnosticsEvent) == 32, "GPUClothDiagnosticsEvent ABI drift");
static_assert(sizeof(GPUClothDiagnosticsStatus) == 152, "GPUClothDiagnosticsStatus ABI drift");
static_assert(sizeof(GPUClothDescriptorLayout) == 144, "GPUClothDescriptorLayout ABI drift");
