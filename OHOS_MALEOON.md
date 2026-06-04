# OHOS Maleoon 910 bring-up notes

Target profile confirmed for the initial OHOS ARM64 training bring-up:

- native subgroup size: `64`
- `shaderInt64`: unavailable, so compile shaders and C++ with `USE_EMULATED_INT64=1`
- `VK_EXT_shader_atomic_float`: unavailable, so compile shaders and C++ with `USE_EMULATED_F32_ATOMIC=1`
- `VK_EXT_subgroup_size_control`: unavailable, so compile shaders and C++ with `SUBGROUP_SIZE=64`
- `maxComputeWorkGroupInvocations`: `512`, so compile shaders and C++ with `MAX_WORKGROUP_INVOCATIONS=512`
- queue timestamps are usable (`timestamp_valid_bits=64`)

## Shader variant

Build the first Maleoon shader variant into a separate directory:

```bash
python3 compile_shaders.py --force \
  --variant ohos_maleoon_sg64_wg512_i64emul_f32atomicemul \
  -D SUBGROUP_SIZE=64 \
  -D MAX_WORKGROUP_INVOCATIONS=512 \
  -D SH_REORDER_SIZE=64 \
  -D USE_EMULATED_INT64=1 \
  -D USE_EMULATED_F32_ATOMIC=1
```

The native runner auto-selects this variant when it is present under the shader root and the probed device matches the Maleoon profile.

## Native runner

Build the native runner with matching C++ compile-time constants:

```bash
cmake -B build-ohos-runner \
  -DVKSPLAT_BUILD_OHOS_RUNNER=ON \
  -DSUBGROUP_SIZE=64 \
  -DMAX_WORKGROUP_INVOCATIONS=512 \
  -DSH_REORDER_SIZE=64 \
  -DUSE_EMULATED_INT64=1 \
  -DUSE_EMULATED_F32_ATOMIC=1 \
  vksplat
cmake --build build-ohos-runner --target vksplat_ohos_runner
```

Expected raw filesystem layout on device:

```text
/data/local/tmp/vksplat/
  shaders/
    ohos_maleoon_sg64_wg512_i64emul_f32atomicemul/
      generated/*.spv
      radix_sort/*.spv
  dataset/
    sparse/ or sparse/0/
    images/
  output/
```

Bring-up sequence:

```bash
/data/local/tmp/vksplat/vksplat_ohos_runner --mode probe
/data/local/tmp/vksplat/vksplat_ohos_runner --mode forward
/data/local/tmp/vksplat/vksplat_ohos_runner --mode train --steps 1
/data/local/tmp/vksplat/vksplat_ohos_runner --mode train --steps 2
```

The runner writes `device_info.json`, the forward smoke-test buffer, and training outputs under `/data/local/tmp/vksplat/output` by default.

Viewer, browser serving, torchmetrics evaluation, MCMC, packaged asset loading, and memory pooling are intentionally excluded from this first bring-up.
