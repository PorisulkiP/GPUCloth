#pragma once

#include <cstddef>
#include <cstdint>

enum GPUClothSolverMask : uint32_t {
    GPUCLOTH_SOLVER_NONE = 0,
    GPUCLOTH_SOLVER_PD = 1u << 0,
    GPUCLOTH_SOLVER_MIL2 = 1u << 1,
    GPUCLOTH_SOLVER_ALL = GPUCLOTH_SOLVER_PD | GPUCLOTH_SOLVER_MIL2,
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
    GPUCLOTH_FEATURE_EXTENSION = 1u << 2,
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
    // Explicit PD-only Stable Discrete Bending owner.  This is intentionally
    // appended so existing feature ids remain ABI-stable.
    GPUCLOTH_FEATURE_BENDING_SDB,
    // Global velocity damping is a product setting distinct from air and
    // material damping.  Append it so all existing feature ids stay stable.
    GPUCLOTH_FEATURE_VELOCITY_DAMPING,
    // Global effector force and wind scales are one typed owner.  Append this
    // feature so all existing ids remain ABI-stable.
    GPUCLOTH_FEATURE_EFFECTOR_SCALES,
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
    GPUCLOTH_ABI_INVALID_HANDLE = 10,
    GPUCLOTH_ABI_ALREADY_EXISTS = 11,
    GPUCLOTH_ABI_BUFFER_TOO_SMALL = 12,
    GPUCLOTH_ABI_BACKEND_UNAVAILABLE = 13,
    GPUCLOTH_ABI_SOLVE_FAILED = 14,
    GPUCLOTH_ABI_INTERNAL_ERROR = 15,
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
    GPUCLOTH_ELEMENT_MESH_EDGE,
    GPUCLOTH_ELEMENT_MESH_FACE,
    GPUCLOTH_ELEMENT_MESH_CORNER,
    GPUCLOTH_ELEMENT_MATERIAL_SPRING,
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
    float velocity_damping;
    uint32_t simulation_flags;
    uint32_t reserved[2];
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

enum GPUClothSelfResponse : uint32_t {
    GPUCLOTH_SELF_RESPONSE_OGC = 1,
    GPUCLOTH_SELF_RESPONSE_MIL2_NDB = 2,
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
    uint32_t self_response;
    uint32_t reserved[2];
};

enum GPUClothColliderFlags : uint32_t {
    GPUCLOTH_COLLIDER_STATIC = 1u << 0,
    GPUCLOTH_COLLIDER_MOVING = 1u << 1,
    GPUCLOTH_COLLIDER_DEFORMING = 1u << 2,
};

enum GPUClothColliderSidedness : uint32_t {
    GPUCLOTH_COLLIDER_ONE_SIDED_NORMAL = 1,
    GPUCLOTH_COLLIDER_TWO_SIDED = 2,
};

enum GPUClothColliderMotionCertificateFlags : uint32_t {
    // Optional accelerator for plain Mil2 only.  Collision truth remains the
    // evaluated positions/triangles above; missing or unusable certificates
    // therefore select the exact evaluated-mesh path.
    GPUCLOTH_COLLIDER_MOTION_CERTIFICATE_PRESENT = 1u << 0,
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
    uint32_t sidedness;
    GPUClothBufferView positions_previous;
    GPUClothBufferView positions_current;
    GPUClothBufferView positions_next;
    GPUClothBufferView triangles;
    float thickness_outer;
    float friction;
    float damping;
    float effector_absorption;
    uint32_t motion_group_count;
    uint32_t motion_certificate_flags;
    GPUClothBufferView canonical_positions;
    GPUClothBufferView triangle_motion_groups;
    GPUClothBufferView motion_group_canonical_to_world;
    GPUClothBufferView motion_group_endpoint_residual;
    uint32_t reserved[2];
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
    uint64_t effector_state_generation;
    uint64_t effector_applied_generation;
    uint64_t effector_upload_count;
    uint64_t effector_allocation_count;
};

// ABI v3 exposes product modes, not implementation solver names. Fast is
// backed by PD; Accuracy is backed by Mil2. XPBD is not a product backend.
enum GPUClothV3Backend : uint32_t {
    GPUCLOTH_V3_BACKEND_NONE = 0,
    GPUCLOTH_V3_BACKEND_FAST = 1,
    GPUCLOTH_V3_BACKEND_ACCURACY = 2,
};

enum GPUClothV3BackendMask : uint32_t {
    GPUCLOTH_V3_BACKEND_MASK_NONE = 0,
    GPUCLOTH_V3_BACKEND_MASK_FAST = 1u << 0,
    GPUCLOTH_V3_BACKEND_MASK_ACCURACY = 1u << 1,
    GPUCLOTH_V3_BACKEND_MASK_ALL =
        GPUCLOTH_V3_BACKEND_MASK_FAST |
        GPUCLOTH_V3_BACKEND_MASK_ACCURACY,
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

enum GPUClothInvariantStage : uint32_t {
    GPUCLOTH_INVARIANT_STAGE_NONE = 0,
    GPUCLOTH_INVARIANT_STAGE_PREPARE,
    GPUCLOTH_INVARIANT_STAGE_DRAPE,
    GPUCLOTH_INVARIANT_STAGE_RUNTIME,
};

enum GPUClothInvariantResult : uint32_t {
    GPUCLOTH_INVARIANT_RESULT_NONE = 0,
    GPUCLOTH_INVARIANT_RESULT_PASS,
    GPUCLOTH_INVARIANT_RESULT_FAILED,
};

enum GPUClothInvariant : uint32_t {
    GPUCLOTH_INVARIANT_NONE = 0,
    GPUCLOTH_INVARIANT_INVALID_INDEX,
    GPUCLOTH_INVARIANT_NONFINITE_STATE,
    GPUCLOTH_INVARIANT_DEGENERATE_TRIANGLE,
    GPUCLOTH_INVARIANT_INCONSISTENT_WINDING,
    GPUCLOTH_INVARIANT_INVALID_SEAM,
    GPUCLOTH_INVARIANT_SELF_INTERSECTION,
    GPUCLOTH_INVARIANT_EXTERNAL_INTERSECTION,
    GPUCLOTH_INVARIANT_EXTERNAL_CLEARANCE,
    GPUCLOTH_INVARIANT_PRESSURE_OPEN_SHELL,
    GPUCLOTH_INVARIANT_PRESSURE_VOLUME,
    GPUCLOTH_INVARIANT_CONTACT_OVERFLOW,
    GPUCLOTH_INVARIANT_STALE_GENERATION,
    GPUCLOTH_INVARIANT_CUDA_ERROR,
    GPUCLOTH_INVARIANT_GRAPH_ERROR,
    GPUCLOTH_INVARIANT_NOT_CONVERGED,
};

enum GPUClothIntersectionType : uint32_t {
    GPUCLOTH_INTERSECTION_NONE = 0,
    GPUCLOTH_INTERSECTION_EDGE_FACE,
    GPUCLOTH_INTERSECTION_ENDPOINT,
    GPUCLOTH_INTERSECTION_COPLANAR_OVERLAP,
};

enum GPUClothPreparationFlags : uint32_t {
    GPUCLOTH_PREPARATION_CONFIGURED = 1u << 0,
};

enum GPUClothPreparationStatusFlags : uint32_t {
    GPUCLOTH_PREPARATION_STATUS_VALIDATED = 1u << 0,
    GPUCLOTH_PREPARATION_STATUS_RUNNABLE = 1u << 1,
    GPUCLOTH_PREPARATION_STATUS_FAILED = 1u << 2,
};

enum GPUClothPreparationResult : uint32_t {
    GPUCLOTH_PREPARATION_RESULT_NONE = 0,
    GPUCLOTH_PREPARATION_RESULT_READY,
    GPUCLOTH_PREPARATION_RESULT_REJECTED,
};

enum GPUClothDrapeFlags : uint32_t {
    GPUCLOTH_DRAPE_USE_TRIANGLE_LAYERS = 1u << 0,
};

enum GPUClothDrapeStatusFlags : uint32_t {
    GPUCLOTH_DRAPE_STATUS_ACTIVE = 1u << 0,
    GPUCLOTH_DRAPE_STATUS_CONVERGED = 1u << 1,
    GPUCLOTH_DRAPE_STATUS_APPLIED = 1u << 2,
    GPUCLOTH_DRAPE_STATUS_CANCELLED = 1u << 3,
    GPUCLOTH_DRAPE_STATUS_FAILED = 1u << 4,
};

enum GPUClothDrapeResult : uint32_t {
    GPUCLOTH_DRAPE_RESULT_NONE = 0,
    GPUCLOTH_DRAPE_RESULT_RUNNING,
    GPUCLOTH_DRAPE_RESULT_CONVERGED,
    GPUCLOTH_DRAPE_RESULT_APPLIED,
    GPUCLOTH_DRAPE_RESULT_CANCELLED,
    GPUCLOTH_DRAPE_RESULT_NOT_CONVERGED,
    GPUCLOTH_DRAPE_RESULT_REJECTED,
};

// Always-on first failure record.  Pair tracing may add detail elsewhere, but
// never changes ordering, math, retry policy, or this deterministic witness.
struct GPUClothInvariantWitness {
    uint32_t struct_size;
    uint32_t witness_version;
    uint32_t stage;
    uint32_t result;
    uint32_t invariant;
    int32_t frame;
    int32_t substep;
    uint32_t solver_mask;
    uint64_t candidate_generation;
    uint64_t detection_generation;
    uint64_t contact_generation;
    uint64_t apply_generation;
    uint64_t cloth_id;
    uint64_t other_object_id;
    int32_t cloth_layer;
    int32_t other_layer;
    int32_t triangle_i;
    int32_t triangle_j;
    int32_t edge_i;
    int32_t edge_j;
    int32_t vertex_i;
    int32_t vertex_j;
    uint32_t intersection_type;
    uint32_t vf_count;
    uint32_t ee_count;
    uint32_t ef_count;
    uint32_t accepted_owner_count;
    uint32_t overflow_count;
    int32_t cuda_status;
    int32_t graph_status;
    float minimum_clearance;
    float maximum_penetration;
    float aabb_min[3];
    float aabb_max[3];
    float maximum_velocity;
    uint32_t reserved0;
    uint64_t frame_wall_ns;
    uint64_t reserved[8];
};

struct GPUClothPreparationConfig {
    uint32_t struct_size;
    uint32_t config_version;
    uint32_t preparation_flags;
    uint32_t reserved0;
    uint64_t topology_generation;
    uint64_t requested_generation;
    uint64_t reserved[8];
};

struct GPUClothPreparationStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t status_flags;
    uint32_t result;
    uint32_t last_error;
    uint32_t reserved0;
    uint64_t preparation_generation;
    uint64_t topology_generation;
    uint64_t accepted_generation;
    uint64_t reserved[6];
};

struct GPUClothDrapeConfig {
    uint32_t struct_size;
    uint32_t config_version;
    uint32_t drape_flags;
    uint32_t max_steps;
    uint32_t convergence_window;
    uint32_t reserved0;
    float convergence_tolerance;
    float reserved1;
    GPUClothBufferView triangle_layers;
    uint64_t reserved[3];
};

struct GPUClothDrapeStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t status_flags;
    uint32_t result;
    uint32_t last_error;
    uint32_t step_count;
    uint32_t consecutive_converged_steps;
    uint32_t reserved0;
    float maximum_position_delta;
    float convergence_tolerance;
    uint64_t begin_generation;
    uint64_t current_generation;
    uint64_t reserved[5];
};

// -------------------------------------------------------------------------
// ABI v3 stable host boundary
// -------------------------------------------------------------------------
// All public identities are monotonic 64-bit tokens. No Blender DNA or C++
// object address crosses the DLL boundary.
using GPUClothV3RuntimeHandle = uint64_t;
using GPUClothV3ClothHandle = uint64_t;
using GPUClothV3ProxyHandle = uint64_t;
using GPUClothV3TransactionHandle = uint64_t;
using GPUClothV3CacheHandle = uint64_t;

enum GPUClothV3RuntimeFlags : uint32_t {
    GPUCLOTH_V3_RUNTIME_NONE = 0,
};

enum GPUClothV3ClothFlags : uint32_t {
    GPUCLOTH_V3_CLOTH_NONE = 0,
};

enum GPUClothV3ClothState : uint32_t {
    GPUCLOTH_V3_CLOTH_CREATED = 1,
    GPUCLOTH_V3_CLOTH_BUILT = 2,
    GPUCLOTH_V3_CLOTH_RUNNABLE = 3,
};

enum GPUClothProxyFlags : uint32_t {
    GPUCLOTH_PROXY_LOCAL_FRAME = 0u,
    GPUCLOTH_PROXY_DIRECT_BARYCENTRIC = 1u << 0,
};

enum GPUClothV3ProxyState : uint32_t {
    GPUCLOTH_V3_PROXY_READY = 1u,
};

struct GPUClothV3ABIInfo {
    uint32_t struct_size;
    uint32_t struct_version;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t abi_patch;
    uint32_t feature_schema_version;
    uint32_t feature_count;
    uint32_t backend_mask;
    uint32_t pointer_width_bits;
    uint32_t little_endian;
    uint32_t export_manifest_version;
    uint32_t reserved0;
    uint64_t reserved[2];
};

struct GPUClothV3RuntimeConfig {
    uint32_t struct_size;
    uint32_t config_version;
    uint32_t runtime_flags;
    uint32_t device_ordinal;
    uint64_t application_id;
    uint64_t reserved[5];
};

struct GPUClothV3FrameConfig {
    uint32_t struct_size;
    uint32_t config_version;
    int32_t frame;
    uint32_t frame_flags;
    uint64_t frame_generation;
    uint32_t fps_numerator;
    uint32_t fps_denominator;
    float subframe;
    float gravity[3];
    uint64_t reserved[2];
};

struct GPUClothV3MeshEdge {
    uint32_t vertex_a;
    uint32_t vertex_b;
    uint32_t edge_flags;
    uint32_t reserved;
};

// Versioned, cloth-owned global effector scales.  These values are distinct
// from per-effector weights and are consumed by the existing PD/Mil2 force
// stage through the persistent native simulation state.
struct GPUClothEffectorScaleConfig {
    GPUClothFeatureConfigHeader header;
    uint32_t solver_mask;
    uint32_t reserved0;
    uint64_t object_id;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    float force_scale;
    float wind_scale;
    uint64_t reserved[2];
};

struct GPUClothV3MeshFace {
    uint32_t first_corner;
    uint32_t corner_count;
    uint32_t face_flags;
    uint32_t reserved;
};

struct GPUClothV3MeshCorner {
    uint32_t vertex_index;
    uint32_t edge_index;
};

struct GPUClothV3ClothCreateConfig {
    uint32_t struct_size;
    uint32_t config_version;
    uint32_t cloth_flags;
    uint32_t backend;
    uint64_t object_id;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    uint32_t vertex_count;
    uint32_t edge_count;
    uint32_t face_count;
    uint32_t corner_count;
    GPUClothBufferView positions;
    GPUClothBufferView edges;
    GPUClothBufferView faces;
    GPUClothBufferView corners;
    float object_to_world[16];
    float world_to_object[16];
    uint64_t reserved[3];
};

enum GPUClothV3ReadbackFlags : uint32_t {
    GPUCLOTH_V3_READBACK_POSITIONS = 1u << 0,
    GPUCLOTH_V3_READBACK_VELOCITIES = 1u << 1,
};

struct GPUClothV3ReadbackConfig {
    uint32_t struct_size;
    uint32_t config_version;
    uint32_t readback_flags;
    uint32_t reserved0;
    uint64_t frame_generation;
    GPUClothBufferView positions;
    GPUClothBufferView velocities;
    uint64_t reserved[2];
};

// Cache configuration reuses GPUClothCacheConfig: its header, fixed-width
// fields, and UTF-8 addresses are already versioned and size-gated.  The v3
// handle-scoped entry points below never expose those addresses as an
// unowned global descriptor; native code copies both strings before return.
enum GPUClothV3CacheFrameFlags : uint32_t {
    GPUCLOTH_V3_CACHE_FRAME_NONE = 0,
    GPUCLOTH_V3_CACHE_FRAME_WRITE = 1u << 0,
    GPUCLOTH_V3_CACHE_FRAME_READ = 1u << 1,
};

struct GPUClothV3CacheFrameConfig {
    uint32_t struct_size;
    uint32_t config_version;
    uint32_t frame_flags;
    uint32_t reserved0;
    int32_t frame;
    uint32_t vertex_count;
    uint64_t frame_generation;
    uint64_t cache_id;
    GPUClothBufferView positions;
    uint64_t reserved[3];
};

struct GPUClothV3ClothStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t state;
    uint32_t backend;
    uint64_t cloth_handle;
    uint64_t object_id;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    uint64_t accepted_frame_generation;
    uint32_t vertex_count;
    uint32_t edge_count;
    uint32_t face_count;
    uint32_t corner_count;
    uint32_t sewing_record_count;
    uint32_t last_error;
    uint64_t solve_count;
    uint64_t reserved[3];
};

// Handle-scoped proxy lifecycle/status.  The owner retains the validated
// topology and rest payload; only opaque handles and typed counts cross ABI.
struct GPUClothV3ProxyStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t state;
    uint32_t last_result;
    uint64_t runtime_handle;
    uint64_t cloth_handle;
    uint64_t proxy_handle;
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
    uint64_t apply_count;
    uint64_t last_generation;
};

// Handle-scoped status for the typed SDB owner.  Configuration is accepted
// before build; applied becomes true only after the owning PD instance has
// consumed the setting through PD_solver_set_sdb_enabled().
struct GPUClothV3SDBStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t backend;
    uint32_t requested;
    uint32_t configured;
    uint32_t applied;
    uint32_t solver_mask;
    uint32_t last_result;
    uint64_t object_id;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    uint64_t apply_count;
};

// Handle-scoped status for the global velocity-damping owner.  The value is
// accepted before build and marked applied only after a successful PD/Mil2
// solve consumes it through the persistent solver state.
struct GPUClothV3VelocityDampingStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t backend;
    uint32_t requested;
    uint32_t configured;
    uint32_t applied;
    uint32_t solver_mask;
    uint32_t last_result;
    float velocity_damping;
    uint32_t reserved0;
    uint64_t object_id;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    uint64_t apply_count;
};

// Handle-scoped status for the global effector-scale owner.  Values become
// applied only after a successful PD/Mil2 solve consumes the persistent state.
struct GPUClothV3EffectorScaleStatus {
    uint32_t struct_size;
    uint32_t status_version;
    uint32_t backend;
    uint32_t requested;
    uint32_t configured;
    uint32_t applied;
    uint32_t solver_mask;
    uint32_t last_result;
    float force_scale;
    float wind_scale;
    uint64_t object_id;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    uint64_t apply_count;
};

// Lossless, handle-scoped material spring record. The complete native spring
// type carries base kind plus WARP/WEFT direction bits; no host pointer crosses
// the ABI.
struct GPUClothV3MaterialSpringRecord {
    uint32_t spring_index;
    uint32_t endpoint_a;
    uint32_t endpoint_b;
    uint32_t spring_type;
    uint32_t spring_flags;
    float linear_stiffness;
    float angular_stiffness;
    uint32_t reserved[2];
};

// Versioned in/out query. Caller supplies record capacity/stride/address and
// receives accepted material state plus exact spring records after build.
// directional_* and the anisotropy material flag are populated only when
// anisotropy was configured; bending_model and spring records are populated
// for every built material owner. BUFFER_TOO_SMALL reports
// required_record_count without writing records.
struct GPUClothV3MaterialStateQuery {
    uint32_t struct_size;
    uint32_t query_version;
    uint32_t record_element_type;
    uint32_t flags;
    uint64_t topology_generation;
    uint64_t geometry_generation;
    uint64_t required_record_count;
    uint64_t returned_record_count;
    uint64_t record_capacity;
    uint64_t record_stride_bytes;
    uint64_t records_address;
    float directional_stiffness[6];
    float directional_stiffness_max[6];
    uint32_t bending_model;
    uint32_t material_flags;
    uint32_t reserved[2];
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
    uint32_t material_state_query_size;
    uint32_t diagnostics_event_size;
    uint32_t diagnostics_status_size;
    uint32_t pin_snapshot_config_size;
    uint32_t collection_record_size;
    uint32_t collection_snapshot_config_size;
    uint32_t collection_transaction_config_size;
    uint32_t collection_query_size;
    uint32_t collection_status_size;
    uint32_t invariant_witness_size;
    uint32_t preparation_config_size;
    uint32_t preparation_status_size;
    uint32_t drape_config_size;
    uint32_t drape_status_size;
    uint32_t sdb_status_size;
    uint32_t proxy_status_size;
    uint32_t velocity_damping_status_size;
    uint32_t effector_scales_config_size;
    uint32_t effector_scales_status_size;
    uint32_t reserved[1];
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
static_assert(sizeof(GPUClothColliderConfig) == 416, "GPUClothColliderConfig ABI drift");
static_assert(sizeof(GPUClothMeshStateConfig) == 256, "GPUClothMeshStateConfig ABI drift");
static_assert(sizeof(GPUClothEffectorConfig) == 160, "GPUClothEffectorConfig ABI drift");
static_assert(sizeof(GPUClothEffectorWeightsConfig) == 96, "GPUClothEffectorWeightsConfig ABI drift");
static_assert(sizeof(GPUClothEffectorScaleConfig) == 72,
    "GPUClothEffectorScaleConfig ABI drift");
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
static_assert(sizeof(GPUClothInvariantWitness) == 256, "GPUClothInvariantWitness ABI drift");
static_assert(sizeof(GPUClothPreparationConfig) == 96, "GPUClothPreparationConfig ABI drift");
static_assert(sizeof(GPUClothPreparationStatus) == 96, "GPUClothPreparationStatus ABI drift");
static_assert(sizeof(GPUClothDrapeConfig) == 96, "GPUClothDrapeConfig ABI drift");
static_assert(sizeof(GPUClothDrapeStatus) == 96, "GPUClothDrapeStatus ABI drift");
static_assert(sizeof(GPUClothV3ABIInfo) == 64, "GPUClothV3ABIInfo ABI drift");
static_assert(sizeof(GPUClothV3RuntimeConfig) == 64, "GPUClothV3RuntimeConfig ABI drift");
static_assert(sizeof(GPUClothV3FrameConfig) == 64, "GPUClothV3FrameConfig ABI drift");
static_assert(sizeof(GPUClothV3MeshEdge) == 16, "GPUClothV3MeshEdge ABI drift");
static_assert(sizeof(GPUClothV3MeshFace) == 16, "GPUClothV3MeshFace ABI drift");
static_assert(sizeof(GPUClothV3MeshCorner) == 8, "GPUClothV3MeshCorner ABI drift");
static_assert(sizeof(GPUClothV3ClothCreateConfig) == 368, "GPUClothV3ClothCreateConfig ABI drift");
static_assert(sizeof(GPUClothV3ReadbackConfig) == 120, "GPUClothV3ReadbackConfig ABI drift");
static_assert(sizeof(GPUClothV3CacheFrameConfig) == 104, "GPUClothV3CacheFrameConfig ABI drift");
static_assert(sizeof(GPUClothV3ClothStatus) == 112, "GPUClothV3ClothStatus ABI drift");
static_assert(sizeof(GPUClothV3ProxyStatus) == 112,
    "GPUClothV3ProxyStatus ABI drift");
static_assert(sizeof(GPUClothV3SDBStatus) == 64,
    "GPUClothV3SDBStatus ABI drift");
static_assert(sizeof(GPUClothV3VelocityDampingStatus) == 72,
    "GPUClothV3VelocityDampingStatus ABI drift");
static_assert(sizeof(GPUClothV3EffectorScaleStatus) == 72,
    "GPUClothV3EffectorScaleStatus ABI drift");
static_assert(sizeof(GPUClothV3MaterialSpringRecord) == 36,
    "GPUClothV3MaterialSpringRecord ABI drift");
static_assert(sizeof(GPUClothV3MaterialStateQuery) == 136,
    "GPUClothV3MaterialStateQuery ABI drift");
static_assert(sizeof(GPUClothDescriptorLayout) == 176, "GPUClothDescriptorLayout ABI drift");
