#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifndef MAX_WORKGROUP_INVOCATIONS
#define MAX_WORKGROUP_INVOCATIONS 1024
#endif

#ifndef MORTON_SORT_STATS_BLOCK_SIZE
#define MORTON_SORT_STATS_BLOCK_SIZE min(MAX_WORKGROUP_INVOCATIONS, SUBGROUP_SIZE*SUBGROUP_SIZE)
#endif

#ifndef MORTON_SORT_APPLY_BLOCK_SIZE
#define MORTON_SORT_APPLY_BLOCK_SIZE MAX_WORKGROUP_INVOCATIONS
#endif

#ifndef CUMSUM_BLOCK_SIZE
#define CUMSUM_BLOCK_SIZE MAX_WORKGROUP_INVOCATIONS
#endif

#ifndef SUM_BLOCK_SIZE
#define SUM_BLOCK_SIZE MAX_WORKGROUP_INVOCATIONS
#endif

#define ALPHA_THRESHOLD (1.f/255.f)

// 0: vanilla 3DGS
// 1: Mip-splatting
#ifndef ANTIALIASE_MODE
#define ANTIALIASE_MODE 0
#endif
#define AA_EPS2D 0.3

#define TILE_HEIGHT 16
#define TILE_WIDTH 16
#define TILE_SIZE (TILE_HEIGHT*TILE_WIDTH)

// 0: none
// 1: gsplat
// 2: all nonzero
#define TILE_CULLING_MODE 2

// 0.2 in Inria, 0.01 in GSplat
#define NEAR_CLIP 0.01f


#define EXPORT_MODE_VULKAN_FORWARD 0
#define EXPORT_MODE_VULKAN_BACKWARD 1

#define sortingKey_t uint32_t
// #define sortingKey_t int64_t

// SH ordering
// Degree 3 SH gives (3+1)^2=16 coefficients, or 16×3=48 floats
// 16×float3: [000111222...15]×N
// 12×float4: [000011112222...11]×N (default)
// Reordered 12×float4 coalesced memory access:
// [[0000...×W][1111...×W]×12]×ceil(N/W)

#ifndef SH_REORDER_SIZE
#define SH_REORDER_SIZE SUBGROUP_SIZE
#endif
// #define SH_REORDER_SIZE 1


// 0.99 for Inria, 0.999 for GSplat
#define ALPHA_BLEND_MAX_OPACITY 0.999f

// subtile culling in rasterization
#if TILE_HEIGHT >= 16 && TILE_WIDTH >= 16
    #define SUBTILE_HEIGHT 4
    #define SUBTILE_WIDTH 8
#else
    #define SUBTILE_HEIGHT 0
    #define SUBTILE_WIDTH 0
#endif
#define SUBTILE_SIZE (SUBTILE_HEIGHT*SUBTILE_WIDTH)


// optional, not in Inria or GSplat
#define NORMALIZE_QUATS_IN_OPTIMIZER 0


static const float M_PI = 3.14159265358979;

#ifndef USE_EMULATED_INT64
#define USE_EMULATED_INT64 0
#endif

#ifndef USE_EMULATED_F32_ATOMIC
#define USE_EMULATED_F32_ATOMIC 0
#endif

#if USE_EMULATED_F32_ATOMIC
void AtomicAddFloat(RWByteAddressBuffer buffer, uint byteAddress, float valueToAdd) {
    uint oldInt = buffer.Load(byteAddress);
    uint newInt;
    uint assumed;

    do {
        assumed = oldInt;
        float fAssumed = asfloat(assumed);
        float fNew = fAssumed + valueToAdd;
        newInt = asuint(fNew);

        uint resultInt;
        buffer.InterlockedCompareExchange(byteAddress, assumed, newInt, resultInt);
        oldInt = resultInt;
    } while (assumed != oldInt);
}
#endif
