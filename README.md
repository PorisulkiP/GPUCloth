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

1. Obtain a `GPUCloth.dll` release built for the matching ABI v3 contract.
2. Place it at `GPUCloth/lib/GPUCloth.dll`.
3. Install the `GPUCloth` directory as a Blender add-on.
4. Enable **GPUCloth** under Blender preferences.

The loader fails closed when the DLL is absent, an export is missing, or ABI
layout/version checks do not match.

## Validate the public boundary

```powershell
py -3 -m unittest discover -s tests -v
```

The check parses source without importing Blender. It verifies Python syntax,
the exact ordered export list, header parity, and loader binding order.

## Platform

- Windows x64
- Blender 4.0 or newer
- NVIDIA runtime compatible with the supplied native release

## License

See `LICENSE` and the notices retained in the source files.
