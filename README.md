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

## 🧪 Test scenes

Built-in builders mirror the native `TestScene` set.
Every `SceneType` has one builder; remaining gaps are listed below.

| # | Scene | Blender operator | Status |
|---|---|---|---|
| 1 | DrapeOnSphere | `gpucloth.test_drape_on_sphere` | Pins wired (`Pin` group, 2 top corners) |
| 2 | TwistTest | `gpucloth.test_twist` | Static pins wired; rotating pair stays solver-side |
| 3 | MultiLayerDrop | `gpucloth.test_multi_layer_drop` | N layers (default 2, range 1–50, dz 0.15); OGC auto sizing |
| 4 | CushionDrop | `gpucloth.test_cushion_drop` | Closed side walls; pressure ratio 1.3 via target volume |
| 5 | CapeProject | `gpucloth.test_cape` | Static body; seams wired as loose sewing edges |
| 6 | MDHorizontalContact | `gpucloth.test_md_horizontal_contact` | MD fixture garment on a static plate |

All builders are one-button: 3D Viewport sidebar `GPUCloth > Create Test Scene`.
A headless audit (no DLL/GPU) executes every operator and checks the mesh and
parity contracts.

Known gaps to full parity (tracked for next slices):

- Twist: missing rotating bottom-pin pair animation (needs solver animated-pin path).
- Cushion: pressure reference volume is the closed mesh as authored (18.72 m³);
  the native pressure builder swaps the lower sheet and uses the enclosed
  volume (22.60 m³). The exposed `pressure.ratio` 1.3 is reproduced.
- Cape/MD: imported areal-density vertex mass (32.0 mg / 16.4 mg per vertex)
  is below the `vertex_mass` RNA floor of 1 g, so the builders cannot set it.
- Cape: missing animated body (`CBODY002` sampling per frame).
- Collision: the native contact-convergence loop count is reproducible through
  `collision_quality`; the C++ `max_sewing = 0` override has no product owner.

> All 6 scenes exist as builders; remaining gaps are runtime-side
> (twist rotation, animated body) plus the RNA-owned values above.

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
