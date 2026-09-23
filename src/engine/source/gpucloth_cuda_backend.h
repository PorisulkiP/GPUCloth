#pragma once

// Internal boundary for the optional CUDA/OptiX capability probe.
//
// This header deliberately has no CUDA, OptiX, or Vulkan SDK dependency.  The
// implementation resolves the CUDA driver and OptiX entry points at runtime.
// Consequently the Vulkan/core target can be built and loaded without an
// NVIDIA driver or CUDA runtime being present.  These declarations are not
// part of the frozen GPUCloth v3 ABI.

#include <cstdint>

#ifndef GPUCLOTH_WITH_CUDA
#  define GPUCLOTH_WITH_CUDA 0
#endif

enum GPUClothCudaProbeState : uint32_t {
    GPUCLOTH_CUDA_PROBE_COMPILED = 1u << 0,
    // The CUDA driver shared library (nvcuda/libcuda) was opened.
    GPUCLOTH_CUDA_PROBE_MODULE_LOADED = 1u << 1,
    GPUCLOTH_CUDA_PROBE_DRIVER_AVAILABLE = 1u << 2,
    GPUCLOTH_CUDA_PROBE_DEVICE_READY = 1u << 3,
    // Preflight only: no resource import/export has been attempted yet.
    GPUCLOTH_CUDA_PROBE_INTEROP_PREFLIGHT_READY = 1u << 4,
    GPUCLOTH_CUDA_PROBE_OPTIX_READY = 1u << 5,
};

enum GPUClothCudaProbeResultCode : uint32_t {
    GPUCLOTH_CUDA_PROBE_OK = 0u,
    GPUCLOTH_CUDA_PROBE_INVALID_ARGUMENT = 1u,
    GPUCLOTH_CUDA_PROBE_UNAVAILABLE = 2u,
    GPUCLOTH_CUDA_PROBE_OPTIX_UNAVAILABLE = 3u,
};

// The masks are Vulkan external-handle types, intentionally kept opaque at
// this private boundary so the core does not need Vulkan headers.  A non-zero
// matching pair is required before reporting interop *preflight* readiness.
enum GPUClothVulkanExternalHandleType : uint32_t {
    GPUCLOTH_VULKAN_EXTERNAL_HANDLE_OPAQUE_FD = 1u << 0,
    GPUCLOTH_VULKAN_EXTERNAL_HANDLE_OPAQUE_WIN32 = 1u << 1,
    GPUCLOTH_VULKAN_EXTERNAL_HANDLE_DMA_BUF = 1u << 2,
};

struct GPUClothVulkanInteropProbe {
    uint32_t struct_size;
    uint32_t uuid_valid;
    uint32_t external_memory_handle_types;
    uint32_t external_semaphore_handle_types;
    uint8_t device_uuid[16];
};

struct GPUClothCudaProbeRequest {
    uint32_t struct_size;
    // Ordinal is only a hint when Vulkan identity data is supplied.
    uint32_t device_ordinal_hint;
    uint32_t require_interop;
    uint32_t require_optix;
    const GPUClothVulkanInteropProbe* vulkan;
};

struct GPUClothCudaProbeResult {
    uint32_t struct_size;
    uint32_t state;
    uint32_t device_count;
    uint32_t device_ordinal;
    uint8_t device_uuid[16];
    char diagnostic[512];
};

// Keep this symbol hidden when linked into GPUCloth itself.  A future
// standalone GPUClothCudaProbe module can define
// GPUCLOTH_CUDA_BACKEND_BUILD to export the exact same C entry point.
#if defined(_WIN32)
#  if defined(GPUCLOTH_CUDA_BACKEND_BUILD)
#    define GPUCLOTH_CUDA_BACKEND_API __declspec(dllexport)
#  else
#    define GPUCLOTH_CUDA_BACKEND_API
#  endif
#elif defined(GPUCLOTH_CUDA_BACKEND_BUILD)
#  define GPUCLOTH_CUDA_BACKEND_API __attribute__((visibility("default")))
#else
#  define GPUCLOTH_CUDA_BACKEND_API __attribute__((visibility("hidden")))
#endif

extern "C" GPUCLOTH_CUDA_BACKEND_API uint32_t
GPUCloth_cuda_backend_probe(const GPUClothCudaProbeRequest* request,
                            GPUClothCudaProbeResult* result);

#undef GPUCLOTH_CUDA_BACKEND_API
