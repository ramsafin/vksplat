# Windows Training Quick Start

These steps assume you built the project with CMake Method 2 on Windows and want to run `simple_trainer.py`.

## Local Tested Setup

This guide was prepared against the following Windows setup:

- OS: Windows 10.0.26200.8457
- Primary training GPU: NVIDIA GeForce RTX 4070 SUPER, 12 GB VRAM, NVIDIA driver 596.49
- Secondary GPU: Intel UHD Graphics 770
- Vulkan: instance 1.4.350; RTX 4070 SUPER device API 1.4.329; Intel UHD Graphics 770 device API 1.4.323
- Build tools: CMake 4.2.3 from `D:\Tools\CMake\bin\cmake.exe`, MSVC 14.29.30133 from Visual Studio 2019 Professional
- Python environment manager: micromamba from `D:\micromamba\condabin\micromamba.bat`
- CMake output: `D:\Projects\vksplat\vksplat\build\Release\vksplat.cp311-win_amd64.pyd`
- Python ABI: CPython 3.11 (`cp311`)

## 1. Create a micromamba environment

The CMake build in this workspace produced `build\Release\vksplat.cp311-win_amd64.pyd`, so use Python 3.11 unless you rebuild against a different Python.

```powershell
micromamba create -n vksplat -c conda-forge python=3.11 numpy opencv tqdm
micromamba activate vksplat
```

Optional evaluation dependencies:

```powershell
python -m pip install torch "torchmetrics[image]>=1.0.1"
```

Training itself uses `numpy`, `opencv-python`/`cv2`, and `tqdm`. Evaluation uses `torch` and `torchmetrics`.

## 2. Build with the active environment

If you already built with this Python 3.11 environment, skip this section. Otherwise rebuild so the `.pyd` matches the environment:

```powershell
cd D:\Projects\vksplat\vksplat
cmake -B build -DPython_EXECUTABLE="$env:CONDA_PREFIX\python.exe"
cmake --build build --config Release
```

## 3. Check dataset layout

`simple_trainer.py` expects a COLMAP-style dataset:

```text
D:\vksplat\data\flowers
  images_4\
  sparse\0\
    cameras.bin
    images.bin
    points3D.bin
```

The checked-in defaults currently use:

```python
output_dir = r"D:\vksplat\output"
dataset_dir = r"D:\vksplat\data\flowers"
image_dir = "images_4"
sparse_dir = "sparse/0"
train_steps = 30_000
```

For a quick smoke test, temporarily set `train_steps = 100`. For a normal training run, use the checked-in `train_steps = 30_000`.

## 4. Run Training

Run from the inner `vksplat` folder so `from build.Release import vksplat` resolves the CMake-built extension:

```powershell
cd D:\Projects\vksplat\vksplat
python .\simple_trainer.py
```

The script prints available Vulkan devices at startup. Leave `TRAIN_DEVICE = -1` for auto-selection, or set it to a printed device id.

Outputs are written under `D:\vksplat\output\<timestamp>_flowers\`, including `config.json`, `train.json`, `splat.ply`, validation renders, and `eval.json` when evaluation dependencies are installed.

## 5. Visualize Training

`simple_trainer.py` includes a bundled browser viewer served by the training process. To enable it, set this in `TrainerConfig`:

```python
enable_viewer: bool = True
viewer_port: int = 7007
```

Start training normally:

```powershell
cd D:\Projects\vksplat\vksplat
python .\simple_trainer.py
```

When training starts, the terminal should print:

```text
Viewer at  http://0.0.0.0:7007/
```

Open this URL on the same Windows machine:

```text
http://localhost:7007/
```

The viewer is local to the running training process. It uses the current model state to render requested camera views, so keep the training process running while the browser is open. Very short smoke tests such as `train_steps = 100` can finish before there is much time to inspect the viewer; use a larger temporary value such as `1000` when testing visualization.

The viewer does not require CUDA. It uses the Vulkan training backend plus the existing Python dependencies (`numpy`, `opencv-python`/`cv2`, and `tqdm`).

## Troubleshooting

- If import fails, confirm the environment Python version matches the `.pyd` tag, such as `cp311` for Python 3.11.
- If the viewer does not load, confirm training is still running and open `http://localhost:7007/`, not `http://0.0.0.0:7007/`.
- If port 7007 is already in use, change `viewer_port` in `TrainerConfig`.
- If Vulkan reports `WARNING: To use this device, shaders must be compiled with USE_EMULATED_F32_ATOMIC=1`, do not pass `-DUSE_EMULATED_F32_ATOMIC=1` to CMake. This is a Slang shader macro. Set it in `D:\Projects\vksplat\vksplat\slang\config.slang`:

  ```c
  #define USE_EMULATED_F32_ATOMIC 1
  ```

  Then recompile the checked-in SPIR-V shaders from the repository root:

  ```powershell
  cd D:\Projects\vksplat
  python .\compile_shaders.py --force
  ```

  Rebuilding the C++/Python extension is not required for this shader-only change.
- If Vulkan asks for `USE_EMULATED_INT64=1`, apply the same process for `USE_EMULATED_INT64` in `vksplat\slang\config.slang`.
- If evaluation is skipped, install `torch` and `torchmetrics[image]`; training can still complete without them.
