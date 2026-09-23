#include "gpucloth_backend_loader.h"

#include <cerrno>
#include <cwchar>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#  include <dlfcn.h>
#  include <limits.h>
#  include <stdlib.h>
#endif

namespace {

struct LoadedModule {
    void* handle = nullptr;
    uint32_t (*probe)(const GPUClothCudaProbeRequest*,
                      GPUClothCudaProbeResult*) = nullptr;
    uint32_t references = 0u;
};

std::mutex g_loader_mutex;
std::unordered_map<std::string, LoadedModule> g_modules;

void set_diagnostic(char* diagnostic, uint32_t size, const std::string& message)
{
    if (!diagnostic || size == 0u) return;
    std::strncpy(diagnostic, message.c_str(), size - 1u);
    diagnostic[size - 1u] = '\0';
}

std::string os_error()
{
#if defined(_WIN32)
    const DWORD error = GetLastError();
    char buffer[256] = {};
    const DWORD length = FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, error, 0, buffer, static_cast<DWORD>(sizeof(buffer)), nullptr);
    if (length != 0u) {
        std::string text(buffer, length);
        while (!text.empty() && (text.back() == '\r' || text.back() == '\n')) {
            text.pop_back();
        }
        return text;
    }
    return "Windows error " + std::to_string(error);
#elif defined(__unix__) || defined(__APPLE__)
    const char* loader_error = dlerror();
    if (loader_error && loader_error[0]) return loader_error;
    return std::strerror(errno);
#else
    return "unsupported operating system";
#endif
}

#if defined(_WIN32)
bool utf8_to_wide(const char* source, std::wstring* destination)
{
    if (!source || !destination) return false;
    const int source_length = static_cast<int>(std::strlen(source));
    if (source_length <= 0) return false;
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                           source, source_length, nullptr, 0);
    if (length <= 0) return false;
    destination->resize(static_cast<size_t>(length));
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, source,
                               source_length, destination->data(), length) == length;
}

bool canonical_windows_path(const char* source, std::wstring* destination,
                            std::string* error)
{
    std::wstring utf8_path;
    if (!utf8_to_wide(source, &utf8_path)) {
        *error = "module path is not valid UTF-8";
        return false;
    }
    const bool drive_absolute = utf8_path.size() >= 3u &&
        ((utf8_path[0] >= L'A' && utf8_path[0] <= L'Z') ||
         (utf8_path[0] >= L'a' && utf8_path[0] <= L'z')) &&
        utf8_path[1] == L':' &&
        (utf8_path[2] == L'\\' || utf8_path[2] == L'/');
    const bool unc_absolute = utf8_path.size() >= 2u &&
        (utf8_path[0] == L'\\' && utf8_path[1] == L'\\');
    if (!drive_absolute && !unc_absolute) {
        *error = "module path must be absolute";
        return false;
    }
    const DWORD required = GetFullPathNameW(utf8_path.c_str(), 0, nullptr, nullptr);
    if (required == 0u) {
        *error = "could not make module path absolute: " + os_error();
        return false;
    }
    std::wstring absolute(static_cast<size_t>(required) + 1u, L'\0');
    if (GetFullPathNameW(utf8_path.c_str(), required + 1u, absolute.data(), nullptr) == 0u) {
        *error = "could not make module path absolute: " + os_error();
        return false;
    }
    absolute.resize(std::wcslen(absolute.c_str()));
    HANDLE file = CreateFileW(absolute.c_str(), GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        *error = "could not resolve module path: " + os_error();
        return false;
    }
    const DWORD final_length = GetFinalPathNameByHandleW(file, nullptr, 0,
                                                         FILE_NAME_NORMALIZED);
    if (final_length == 0u) {
        const std::string detail = os_error();
        CloseHandle(file);
        *error = "could not canonicalize module path: " + detail;
        return false;
    }
    std::wstring canonical(static_cast<size_t>(final_length) + 1u, L'\0');
    const DWORD written = GetFinalPathNameByHandleW(file, canonical.data(),
                                                    final_length + 1u,
                                                    FILE_NAME_NORMALIZED);
    CloseHandle(file);
    if (written == 0u || written > final_length) {
        *error = "could not canonicalize module path: " + os_error();
        return false;
    }
    canonical.resize(static_cast<size_t>(written));
    *destination = std::move(canonical);
    return true;
}

std::string wide_to_utf8(const std::wstring& source)
{
    if (source.empty()) return {};
    const int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                           source.c_str(), static_cast<int>(source.size()),
                                           nullptr, 0, nullptr, nullptr);
    if (length <= 0) return {};
    std::string destination(static_cast<size_t>(length), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, source.c_str(),
                            static_cast<int>(source.size()), destination.data(),
                            length, nullptr, nullptr) != length) return {};
    return destination;
}
#endif

bool canonical_path(const char* source, std::string* path, std::string* error)
{
    if (!source || source[0] == '\0') {
        *error = "module path is empty";
        return false;
    }
#if defined(_WIN32)
    std::wstring canonical;
    if (!canonical_windows_path(source, &canonical, error)) return false;
    *path = wide_to_utf8(canonical);
    if (path->empty()) {
        *error = "canonical module path is not valid UTF-8";
        return false;
    }
    return true;
#elif defined(__unix__) || defined(__APPLE__)
    if (source[0] != '/') {
        *error = "module path must be absolute";
        return false;
    }
    char* resolved = realpath(source, nullptr);
    if (!resolved) {
        *error = "could not canonicalize module path: " + os_error();
        return false;
    }
    *path = resolved;
    std::free(resolved);
    return true;
#else
    (void)path;
    *error = "unsupported operating system";
    return false;
#endif
}

void* open_module(const std::string& path)
{
#if defined(_WIN32)
    std::wstring wide;
    std::string ignored;
    if (!utf8_to_wide(path.c_str(), &wide)) return nullptr;
    return reinterpret_cast<void*>(LoadLibraryExW(
        wide.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
#elif defined(__unix__) || defined(__APPLE__)
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#else
    (void)path;
    return nullptr;
#endif
}

void close_module(void* module)
{
#if defined(_WIN32)
    if (module) FreeLibrary(static_cast<HMODULE>(module));
#elif defined(__unix__) || defined(__APPLE__)
    if (module) dlclose(module);
#else
    (void)module;
#endif
}

void* lookup(void* module, const char* symbol)
{
#if defined(_WIN32)
    return module ? reinterpret_cast<void*>(GetProcAddress(
        static_cast<HMODULE>(module), symbol)) : nullptr;
#elif defined(__unix__) || defined(__APPLE__)
    return module ? dlsym(module, symbol) : nullptr;
#else
    (void)module;
    (void)symbol;
    return nullptr;
#endif
}

} // namespace

bool GPUCloth_load_cuda_backend(const char* module_path,
                                GPUClothCudaBackendVTable* out_vtable,
                                char* diagnostic,
                                uint32_t diagnostic_size)
{
    if (!out_vtable) {
        set_diagnostic(diagnostic, diagnostic_size,
                       "CUDA capability-probe output table is null");
        return false;
    }
    *out_vtable = {};

    std::string path;
    std::string error;
    if (!canonical_path(module_path, &path, &error)) {
        set_diagnostic(diagnostic, diagnostic_size, error);
        return false;
    }

    std::lock_guard<std::mutex> lock(g_loader_mutex);
    auto existing = g_modules.find(path);
    if (existing != g_modules.end()) {
        ++existing->second.references;
        out_vtable->module = existing->second.handle;
        out_vtable->probe = existing->second.probe;
        set_diagnostic(diagnostic, diagnostic_size,
                       "CUDA capability-probe module already loaded");
        return true;
    }

    void* module = open_module(path);
    if (!module) {
        set_diagnostic(diagnostic, diagnostic_size,
                       "CUDA capability-probe module could not be loaded: " + os_error());
        return false;
    }
    auto probe = reinterpret_cast<decltype(out_vtable->probe)>(
        lookup(module, "GPUCloth_cuda_backend_probe"));
    if (!probe) {
        const std::string detail = os_error();
        close_module(module);
        set_diagnostic(diagnostic, diagnostic_size,
                       "CUDA capability-probe symbol is missing: " + detail);
        return false;
    }
    g_modules.emplace(path, LoadedModule{module, probe, 1u});
    out_vtable->module = module;
    out_vtable->probe = probe;
    set_diagnostic(diagnostic, diagnostic_size,
                   "CUDA capability-probe module loaded");
    return true;
}

uint32_t GPUCloth_probe_cuda_backend(
    GPUClothCudaBackendVTable* vtable,
    const GPUClothCudaProbeRequest* request,
    GPUClothCudaProbeResult* result)
{
    if (!vtable || !vtable->module || !vtable->probe || !request || !result) {
        return GPUCLOTH_CUDA_PROBE_INVALID_ARGUMENT;
    }
    // Hold the registry lock across the call.  Unload therefore cannot close
    // the shared object until this invocation has returned.
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    for (const auto& entry : g_modules) {
        if (entry.second.handle == vtable->module &&
            entry.second.probe == vtable->probe) {
            return entry.second.probe(request, result);
        }
    }
    return GPUCLOTH_CUDA_PROBE_UNAVAILABLE;
}

void GPUCloth_unload_cuda_backend(GPUClothCudaBackendVTable* vtable)
{
    if (!vtable || !vtable->module) {
        if (vtable) *vtable = {};
        return;
    }
    // Callers must use GPUCloth_probe_cuda_backend; this lock also ensures
    // that an in-flight synchronized probe finishes before the last close.
    std::lock_guard<std::mutex> lock(g_loader_mutex);
    for (auto it = g_modules.begin(); it != g_modules.end(); ++it) {
        if (it->second.handle != vtable->module) continue;
        if (--it->second.references == 0u) {
            close_module(it->second.handle);
            g_modules.erase(it);
        }
        *vtable = {};
        return;
    }
    // Unknown handles are not ours; never call dlclose/FreeLibrary on them.
    *vtable = {};
}
