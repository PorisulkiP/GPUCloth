#pragma once

#include "gpucloth_cuda_backend.h"

// Private host-side vtable for the optional CUDA capability-probe module.
// GPUCloth.dll can own this table without linking against CUDA, OptiX, or the
// probe's C++ implementation.  Call GPUCloth_probe_cuda_backend instead of
// invoking probe directly: it serializes calls with unload.
struct GPUClothCudaBackendVTable {
    void* module;
    uint32_t (*probe)(const GPUClothCudaProbeRequest*,
                      GPUClothCudaProbeResult*);
};

bool GPUCloth_load_cuda_backend(const char* module_path,
                                GPUClothCudaBackendVTable* out_vtable,
                                char* diagnostic,
                                uint32_t diagnostic_size);
uint32_t GPUCloth_probe_cuda_backend(
    GPUClothCudaBackendVTable* vtable,
    const GPUClothCudaProbeRequest* request,
    GPUClothCudaProbeResult* result);
// The vtable must remain loaded until all synchronized probe calls return.
// Direct calls through vtable->probe are not synchronized with unload.
void GPUCloth_unload_cuda_backend(GPUClothCudaBackendVTable* vtable);
