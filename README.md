# GPUCloth

Public Blender add-on and stable C ABI for the GPUCloth native runtime.

This repository contains only the client boundary:

- `GPUCloth/` — Blender 4.x add-on and `ctypes` bindings;
- `include/GPUCloth/gpucloth.h` — 57 callable ABI v3 entry points;
- `include/GPUCloth/product_abi.h` — versioned ABI layouts and handles;
- `abi/GPUClothV3.exports.allowlist` — frozen 57-symbol export contract.

The native solver implementation and its tests live in the separate
GPUCloth-Core repository. A compatible release of `GPUCloth.dll` is required
at runtime; implementation sources are not part of this repository.

## Install

1. Install the NVIDIA driver and compatible CUDA 13.3 runtime libraries.
2. Remove any legacy add-on installed as `scripts/addons/python`.
3. Download `gpucloth-<version>-windows-x64.zip` from the release.
4. In Blender 4.2, open **Edit > Preferences > Get Extensions**.
5. Open the menu, choose **Install from Disk**, and select the ZIP.
6. Enable **GPUCloth** if Blender does not enable it automatically.

Do not extract or rearrange the ZIP. It contains the Python add-on,
`GPUCloth.dll`, DirectStorage dependencies, and the required OptiX PTX files.
The CUDA runtime must provide `cublas64_13.dll`, `cusparse64_12.dll`, and
`cusolver64_12.dll` together through `CUDA_PATH/bin/x64` or `PATH`.

The loader fails closed when the DLL is absent, an export is missing, or ABI
layout/version checks do not match.

GPU activation queues preparation without blocking the property callback.
Blender shows phase and percentage progress in the GPU Cloth panel and status
bar. **Stop** requests cancellation; native teardown waits for the active DLL
call to finish.

Collider sidedness follows Blender's **Single Sided** setting. Blender 4.2's
default `(use_culling=True, use_normal=False)` maps automatically to the native
one-sided-normal contract. Disabling **Single Sided** selects two-sided contact.

## Validate the public boundary

```powershell
py -3 -m unittest discover -s tests -v
```

The check parses source without importing Blender. It verifies Python syntax,
the exact ordered export list, header parity, loader binding order, and release
contract.

Build an official Blender extension archive from a compatible Core build:

```powershell
py -3 tools/build_blender_release.py `
  --core-build-dir <GPUCloth-Core-build> `
  --blender "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
```

Validate installation, registration, ABI metadata, restart, and removal in an
isolated Blender profile:

```powershell
py -3 tools/test_blender_release.py `
  --package dist/gpucloth-0.1.3-windows-x64.zip `
  --blender "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
```

## Platform

- Windows x64
- Blender 4.2 LTS
- NVIDIA driver and CUDA 13.3 runtime compatible with the supplied native release

## License

See `LICENSE` and the notices retained in the source files.
