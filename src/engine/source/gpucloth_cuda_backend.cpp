#include "gpucloth_cuda_backend.h"

#include <algorithm>
#include <cstring>
#include <mutex>

#if GPUCLOTH_WITH_CUDA && __has_include(<cuda.h>)
#  include <cuda.h>
#  define GPUCLOTH_HAVE_CUDA_DRIVER 1
#  if defined(CUDA_VERSION) && CUDA_VERSION >= 11030
#    define GPUCLOTH_HAVE_CU_GET_PROC_ADDRESS 1
#  else
#    define GPUCLOTH_HAVE_CU_GET_PROC_ADDRESS 0
#  endif
#else
#  define GPUCLOTH_HAVE_CUDA_DRIVER 0
#  define GPUCLOTH_HAVE_CU_GET_PROC_ADDRESS 0
#endif

// Use NVIDIA's official OptiX stubs.  The only OptiX entry point involved is
// optixQueryFunctionTable; optixInit is never looked up as a DLL symbol.
#if GPUCLOTH_HAVE_CUDA_DRIVER && defined(_WIN32) && \
    __has_include(<optix_stubs.h>) && \
    __has_include(<optix_function_table_definition.h>)
#  include <optix_stubs.h>
#  include <optix_function_table_definition.h>
#  define GPUCLOTH_HAVE_OPTIX 1
#else
#  define GPUCLOTH_HAVE_OPTIX 0
#endif

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#endif

namespace {

constexpr uint32_t kRequestSize = sizeof(GPUClothCudaProbeRequest);
#if GPUCLOTH_HAVE_CUDA_DRIVER
constexpr uint32_t kInteropSize = sizeof(GPUClothVulkanInteropProbe);
#endif
constexpr uint32_t kResultSize = sizeof(GPUClothCudaProbeResult);
std::mutex g_module_mutex;

#if GPUCLOTH_HAVE_CUDA_DRIVER
using CuInitFn = decltype(&cuInit);
#if GPUCLOTH_HAVE_CU_GET_PROC_ADDRESS
using CuGetProcAddressFn = decltype(&cuGetProcAddress);
#endif
using CuDeviceGetCountFn = decltype(&cuDeviceGetCount);
using CuDeviceGetFn = decltype(&cuDeviceGet);
using CuDeviceGetUuidFn = decltype(&cuDeviceGetUuid);

struct DriverApi {
    void* module = nullptr;
    CuInitFn init = nullptr;
#if GPUCLOTH_HAVE_CU_GET_PROC_ADDRESS
    CuGetProcAddressFn get_proc_address = nullptr;
#endif
    CuDeviceGetCountFn get_device_count = nullptr;
    CuDeviceGetFn get_device = nullptr;
    CuDeviceGetUuidFn get_device_uuid = nullptr;
};

// Process-owned: this module is never unloaded while its function pointers
// can be in use.  Publication and first load are mutex-protected.
DriverApi g_driver;
#endif

#if GPUCLOTH_HAVE_OPTIX
void* g_optix_module = nullptr;
#endif

void set_diagnostic(GPUClothCudaProbeResult* result, const char* message)
{
    if (!result) return;
    std::strncpy(result->diagnostic, message,
                 sizeof(result->diagnostic) - 1u);
    result->diagnostic[sizeof(result->diagnostic) - 1u] = '\0';
}

#if GPUCLOTH_HAVE_CUDA_DRIVER
template <typename Function>
Function lookup_function(void* module, const char* name)
{
#if defined(_WIN32)
    return module ? reinterpret_cast<Function>(GetProcAddress(
        static_cast<HMODULE>(module), name)) : nullptr;
#else
    (void)module;
    (void)name;
    return nullptr;
#endif
}

void close_module(void* module)
{
#if defined(_WIN32)
    if (module) FreeLibrary(static_cast<HMODULE>(module));
#else
    (void)module;
#endif
}

void* load_driver_module()
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryExW(
        L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
#else
    // Linux driver resolution is intentionally deferred: a bare dlopen of
    // libcuda.so.1 would make LD_LIBRARY_PATH a code-loading boundary.  A
    // future implementation must use a documented, trusted system-driver
    // path before enabling this branch.
    return nullptr;
#endif
}

template <typename Function>
Function symbol_as(void* symbol)
{
    // Function's type comes from cuda.h and therefore preserves CUDAAPI on
    // Windows (including its calling convention).
    return reinterpret_cast<Function>(symbol);
}

template <typename Function>
Function query_or_lookup(const DriverApi& api, const char* name,
                         int requested_cuda_version)
{
#if GPUCLOTH_HAVE_CU_GET_PROC_ADDRESS
    if (api.get_proc_address) {
        void* pointer = nullptr;
        CUdriverProcAddressQueryResult query_result =
            CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        if (api.get_proc_address(name, &pointer, requested_cuda_version,
                                 CU_GET_PROC_ADDRESS_DEFAULT,
                                 &query_result) == CUDA_SUCCESS && pointer &&
            query_result ==
                CU_GET_PROC_ADDRESS_SUCCESS) {
            return symbol_as<Function>(pointer);
        }
    }
#endif
    // cuGetProcAddress is newer than the base driver ABI.  Use the exact
    // exported symbol as a compatibility fallback for older drivers.
    return lookup_function<Function>(api.module, name);
}

bool acquire_driver(DriverApi* out)
{
    std::lock_guard<std::mutex> lock(g_module_mutex);
    if (g_driver.module) {
        *out = g_driver;
        return true;
    }

    void* module = load_driver_module();
    if (!module) return false;
    DriverApi candidate;
    candidate.module = module;
    // cuGetProcAddress itself is obtained from the stable export; all other
    // entries use it when available.
#if GPUCLOTH_HAVE_CU_GET_PROC_ADDRESS
    candidate.get_proc_address = lookup_function<CuGetProcAddressFn>(
        module, "cuGetProcAddress");
#endif
    candidate.init = query_or_lookup<CuInitFn>(candidate, "cuInit", 1000);
    candidate.get_device_count = query_or_lookup<CuDeviceGetCountFn>(
        candidate, "cuDeviceGetCount", 1000);
    candidate.get_device = query_or_lookup<CuDeviceGetFn>(
        candidate, "cuDeviceGet", 1000);
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
    constexpr int kDeviceUuidApiVersion = 12000;
#else
    // cuDeviceGetUuid's original ABI was introduced in CUDA 9.2.  Keep the
    // requested version tied to this typedef, never to the toolkit version.
    constexpr int kDeviceUuidApiVersion = 9020;
#endif
    candidate.get_device_uuid = query_or_lookup<CuDeviceGetUuidFn>(
        candidate, "cuDeviceGetUuid", kDeviceUuidApiVersion);

    if (!candidate.init || !candidate.get_device_count ||
        !candidate.get_device) {
        close_module(module);
        return false;
    }
    g_driver = candidate;
    *out = g_driver;
    return true;
}

bool uuid_equal(const uint8_t* lhs, const uint8_t* rhs)
{
    return std::memcmp(lhs, rhs, 16u) == 0;
}

void copy_uuid(uint8_t* destination, const CUuuid& source)
{
    std::memcpy(destination, source.bytes, 16u);
}
#endif

#if GPUCLOTH_HAVE_CUDA_DRIVER
bool handle_types_are_concrete(const GPUClothVulkanInteropProbe& vulkan)
{
    if (!vulkan.uuid_valid || vulkan.external_memory_handle_types == 0u ||
        vulkan.external_semaphore_handle_types == 0u) {
        return false;
    }
#if defined(_WIN32)
    constexpr uint32_t supported = GPUCLOTH_VULKAN_EXTERNAL_HANDLE_OPAQUE_WIN32;
#elif defined(__linux__)
    constexpr uint32_t supported = GPUCLOTH_VULKAN_EXTERNAL_HANDLE_OPAQUE_FD |
        GPUCLOTH_VULKAN_EXTERNAL_HANDLE_DMA_BUF;
#else
    constexpr uint32_t supported = 0u;
#endif
    return (vulkan.external_memory_handle_types & supported) != 0u &&
           (vulkan.external_semaphore_handle_types & supported) != 0u;
}
#endif

#if GPUCLOTH_HAVE_OPTIX
bool acquire_optix()
{
    std::lock_guard<std::mutex> lock(g_module_mutex);
    if (g_optix_module) return true;
    void* handle = nullptr;
    const OptixResult status = optixInitWithHandle(&handle);
    if (status != OPTIX_SUCCESS || !handle) {
        if (handle) optixUninitWithHandle(handle);
        return false;
    }
    // Keep the official function table and module alive for the process.
    g_optix_module = handle;
    return true;
}
#endif

} // namespace

extern "C" uint32_t
GPUCloth_cuda_backend_probe(const GPUClothCudaProbeRequest* request,
                            GPUClothCudaProbeResult* result)
{
    if (!request || !result || request->struct_size != kRequestSize ||
        result->struct_size != kResultSize) {
        return GPUCLOTH_CUDA_PROBE_INVALID_ARGUMENT;
    }
    result->state = 0u;
    result->device_count = 0u;
    result->device_ordinal = request->device_ordinal_hint;
    std::memset(result->device_uuid, 0, sizeof(result->device_uuid));
    std::memset(result->diagnostic, 0, sizeof(result->diagnostic));

#if !GPUCLOTH_HAVE_CUDA_DRIVER
    set_diagnostic(result,
                   "CUDA backend not compiled with CUDA Driver API headers");
    return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
#else
    result->state |= GPUCLOTH_CUDA_PROBE_COMPILED;
    const GPUClothVulkanInteropProbe* vulkan = request->vulkan;
    if (vulkan && vulkan->struct_size != kInteropSize) {
        set_diagnostic(result, "Vulkan interop descriptor size is invalid");
        return GPUCLOTH_CUDA_PROBE_INVALID_ARGUMENT;
    }
    if (request->require_interop &&
        !vulkan) {
        set_diagnostic(result, "Vulkan interop descriptor is required");
        return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
    }

    DriverApi driver;
    if (!acquire_driver(&driver)) {
#if defined(_WIN32)
        set_diagnostic(result, "CUDA driver module or entry points unavailable");
#else
        set_diagnostic(result,
                       "CUDA capability probing is disabled on this platform; secure driver resolution is not implemented");
#endif
        return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
    }
    result->state |= GPUCLOTH_CUDA_PROBE_MODULE_LOADED;
    if (driver.init(0u) != CUDA_SUCCESS) {
        set_diagnostic(result, "cuInit failed");
        return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
    }
    result->state |= GPUCLOTH_CUDA_PROBE_DRIVER_AVAILABLE;

    int device_count = 0;
    if (driver.get_device_count(&device_count) != CUDA_SUCCESS ||
        device_count <= 0) {
        result->device_count = static_cast<uint32_t>(std::max(device_count, 0));
        set_diagnostic(result, "CUDA reports no usable devices");
        return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
    }
    result->device_count = static_cast<uint32_t>(device_count);

    const bool match_uuid = vulkan && vulkan->uuid_valid;
    const bool need_uuid = match_uuid || (vulkan != nullptr);
    if (need_uuid && !driver.get_device_uuid) {
        set_diagnostic(result, "CUDA driver does not expose cuDeviceGetUuid");
        return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
    }

    CUdevice selected_device = 0;
    uint32_t selected_ordinal = request->device_ordinal_hint;
    bool selected = false;
    if (match_uuid) {
        // The Vulkan ordinal is only a hint; UUID identity is authoritative.
        for (int ordinal = 0; ordinal < device_count; ++ordinal) {
            CUdevice candidate = 0;
            CUuuid candidate_uuid{};
            if (driver.get_device(&candidate, ordinal) != CUDA_SUCCESS ||
                driver.get_device_uuid(&candidate_uuid, candidate) != CUDA_SUCCESS) {
                continue;
            }
            uint8_t bytes[16];
            copy_uuid(bytes, candidate_uuid);
            if (uuid_equal(bytes, vulkan->device_uuid)) {
                selected_device = candidate;
                selected_ordinal = static_cast<uint32_t>(ordinal);
                std::memcpy(result->device_uuid, bytes, sizeof(bytes));
                selected = true;
                break;
            }
        }
        if (!selected) {
            set_diagnostic(result,
                           "no CUDA device matches the Vulkan device UUID");
            return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
        }
    } else {
        if (selected_ordinal >= static_cast<uint32_t>(device_count) ||
            driver.get_device(&selected_device,
                              static_cast<int>(selected_ordinal)) != CUDA_SUCCESS) {
            set_diagnostic(result, "requested CUDA device is unavailable");
            return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
        }
        if (need_uuid) {
            CUuuid selected_uuid{};
            if (driver.get_device_uuid(&selected_uuid, selected_device) !=
                CUDA_SUCCESS) {
                set_diagnostic(result, "CUDA device UUID query failed");
                return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
            }
            copy_uuid(result->device_uuid, selected_uuid);
        }
        selected = true;
    }
    if (!selected) return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
    result->device_ordinal = selected_ordinal;
    result->state |= GPUCLOTH_CUDA_PROBE_DEVICE_READY;

    if (vulkan) {
        const bool preflight = handle_types_are_concrete(*vulkan) &&
                               uuid_equal(result->device_uuid,
                                          vulkan->device_uuid);
        if (preflight) {
            result->state |= GPUCLOTH_CUDA_PROBE_INTEROP_PREFLIGHT_READY;
        }
        if (request->require_interop && !preflight) {
            set_diagnostic(result,
                           "CUDA/Vulkan UUID or external handle capabilities do not match");
            return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
        }
    }

    if (request->require_optix) {
#if GPUCLOTH_HAVE_OPTIX
        if (!acquire_optix()) {
            set_diagnostic(result,
                           "OptiX official function-table initialization failed");
            return GPUCLOTH_CUDA_PROBE_OPTIX_UNAVAILABLE;
        }
        result->state |= GPUCLOTH_CUDA_PROBE_OPTIX_READY;
#else
        set_diagnostic(result, "OptiX headers/stubs are not available");
        return GPUCLOTH_CUDA_PROBE_OPTIX_UNAVAILABLE;
#endif
    }
    set_diagnostic(result, "CUDA device probe succeeded");
    return GPUCLOTH_CUDA_PROBE_OK;
#endif
}
