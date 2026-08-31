#pragma once

#include "product_abi.h"

#ifndef GPUCLOTH_V3_API
#  if defined(_WIN32)
#    if defined(GPUCLOTH_V3_EXPORTS)
#      define GPUCLOTH_V3_API __declspec(dllexport)
#    else
#      define GPUCLOTH_V3_API __declspec(dllimport)
#    endif
#  else
#    define GPUCLOTH_V3_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

GPUCLOTH_V3_API uint32_t GPUCloth_v3_get_abi_info(
    GPUClothV3ABIInfo* out_info);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_get_descriptor_layout(
    GPUClothDescriptorLayout* out_layout);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_get_feature_count(
    uint64_t* out_count);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_get_feature_info(
    uint64_t index, GPUClothFeatureInfo* out_info);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_query_feature(
    uint32_t feature_id, GPUClothFeatureInfo* out_info);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_runtime_create(
    const GPUClothV3RuntimeConfig* config,
    GPUClothV3RuntimeHandle* out_runtime);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_runtime_update(
    GPUClothV3RuntimeHandle runtime,
    const GPUClothV3FrameConfig* config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_runtime_destroy(
    GPUClothV3RuntimeHandle runtime);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_create(
    GPUClothV3RuntimeHandle runtime,
    const GPUClothV3ClothCreateConfig* config,
    GPUClothV3ClothHandle* out_cloth);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_configure(
    GPUClothV3ClothHandle cloth,
    const GPUClothFeatureConfigHeader* config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_set_vertex_channel(
    GPUClothV3ClothHandle cloth,
    const GPUClothVertexChannelConfig* config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_set_shrink_config(
    GPUClothV3ClothHandle cloth,
    const GPUClothShrinkConfig* config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_set_pin_snapshot(
    GPUClothV3ClothHandle cloth,
    const GPUClothPinSnapshotConfig* config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_collection_transaction_begin(
    GPUClothV3RuntimeHandle runtime,
    const GPUClothCollectionTransactionConfig* config,
    GPUClothV3TransactionHandle* out_transaction);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_collection_stage_snapshot(
    GPUClothV3TransactionHandle transaction,
    GPUClothV3ClothHandle cloth,
    const GPUClothCollectionSnapshotConfig* config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_collection_stage_pin_snapshot(
    GPUClothV3TransactionHandle transaction,
    GPUClothV3ClothHandle cloth,
    const GPUClothPinSnapshotConfig* config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_collection_stage_mesh_state(
    GPUClothV3TransactionHandle transaction,
    GPUClothV3ClothHandle cloth,
    const GPUClothMeshStateConfig* config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_collection_transaction_commit(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3TransactionHandle transaction);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_collection_transaction_abort(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3TransactionHandle transaction);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_query_collection(
    GPUClothV3ClothHandle cloth,
    GPUClothCollectionQuery* query);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_collection_status(
    GPUClothV3ClothHandle cloth,
    GPUClothCollectionStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_diagnostics(
    GPUClothV3ClothHandle cloth,
    GPUClothDiagnosticsStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_validate_initial_state(
    GPUClothV3ClothHandle cloth,
    const GPUClothPreparationConfig* config,
    GPUClothPreparationStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_preparation_status(
    GPUClothV3ClothHandle cloth,
    GPUClothPreparationStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_invariant_status(
    GPUClothV3ClothHandle cloth,
    GPUClothInvariantWitness* out_witness);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_begin_drape(
    GPUClothV3ClothHandle cloth,
    const GPUClothDrapeConfig* config,
    GPUClothDrapeStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_step_drape(
    GPUClothV3ClothHandle cloth,
    GPUClothDrapeStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_apply_drape(
    GPUClothV3ClothHandle cloth,
    GPUClothDrapeStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_cancel_drape(
    GPUClothV3ClothHandle cloth,
    GPUClothDrapeStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_build(
    GPUClothV3ClothHandle cloth);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_destroy(
    GPUClothV3ClothHandle cloth);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_step(
    GPUClothV3ClothHandle cloth);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_readback(
    GPUClothV3ClothHandle cloth,
    GPUClothV3ReadbackConfig* config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_status(
    GPUClothV3ClothHandle cloth,
    GPUClothV3ClothStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_shrink_status(
    GPUClothV3ClothHandle cloth,
    GPUClothV3ShrinkStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_proxy_create(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3ClothHandle cloth,
    const GPUClothProxyConfig* config,
    GPUClothV3ProxyHandle* out_proxy);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_proxy_apply(
    GPUClothV3ProxyHandle proxy,
    const GPUClothBufferView* proxy_positions,
    GPUClothBufferView* render_positions);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_proxy_get_status(
    GPUClothV3ProxyHandle proxy,
    GPUClothV3ProxyStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_proxy_destroy(
    GPUClothV3ProxyHandle proxy);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_sdb_status(
    GPUClothV3ClothHandle cloth,
    GPUClothV3SDBStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_velocity_damping_status(
    GPUClothV3ClothHandle cloth,
    GPUClothV3VelocityDampingStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_effector_scales_status(
    GPUClothV3ClothHandle cloth,
    GPUClothV3EffectorScaleStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_get_constraint_network_status(
    GPUClothV3ClothHandle cloth,
    GPUClothV3ConstraintNetworkStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cloth_query_material_state(
    GPUClothV3ClothHandle cloth,
    GPUClothV3MaterialStateQuery* query);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_configure(
    GPUClothV3RuntimeHandle runtime,
    const GPUClothCacheConfig* config,
    GPUClothV3CacheHandle* out_cache);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_query(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache,
    GPUClothCacheConfig* out_config);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_destroy(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_update_status(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache,
    const GPUClothCacheStatusUpdate* update);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_get_status(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache,
    GPUClothCacheStatus* out_status);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_write_frame_async(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache,
    const GPUClothV3CacheFrameConfig* request);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_prefetch_frame(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache,
    const GPUClothV3CacheFrameConfig* request);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_is_frame_ready(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache,
    const GPUClothV3CacheFrameConfig* request,
    uint32_t* out_ready);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_read_frame(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache,
    GPUClothV3CacheFrameConfig* request);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_free_frame(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache,
    const GPUClothV3CacheFrameConfig* request);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_has_frame(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache,
    const GPUClothV3CacheFrameConfig* request,
    uint32_t* out_present);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_clear(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache);
GPUCLOTH_V3_API uint32_t GPUCloth_v3_cache_flush(
    GPUClothV3RuntimeHandle runtime,
    GPUClothV3CacheHandle cache);

#ifdef __cplusplus
}
#endif
