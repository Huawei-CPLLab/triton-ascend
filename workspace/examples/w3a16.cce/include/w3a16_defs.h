/**
 * Architectural constants, quantization geometry, derived memory maps, and
 * internal module declarations for W3A16 group-wise dequant + FP16 GEMV/GEMM
 * on HiSilicon DaVinci dav-l310 (Kirin 9020).
 *
 * Weight quantization format (Q3_0-style symmetric 3-bit, group_size = 32):
 *   - x_gm  : [M, K]      FP16 activations (row-major)
 *   - qh_gm : [N, K / 8]  int8 packed high 1-bit plane (bit b -> group sub-block)
 *   - qs_gm : [N, K / 4]  int8 packed low  2-bit plane (2 bits per weight)
 *   - sc_gm : [N, K / 32] FP16 per-group scales
 *   - shape : [3]         int32 {M, N, K}
 *   - out_gm: [M, N]      FP16 output (row-major)
 *
 * Dequantization formula for weight (n, k) with group g = k / 32,
 * sub-plane p = (k % 32) / 8 in {0..3}, lane l = k % 8 in {0..7}:
 *   qh_byte = qh_gm[n, g * 4 + (p >> 1)]
 *   qh_bit  = (qh_byte >> ((p & 1) * 4 + l)) & 1          // in {0, 1}
 *   qs_byte = qs_gm[n, g * 8 + l]
 *   qs_2bit = (qs_byte >> (p * 2)) & 3                    // in {0..3}
 *   w_fp16  = half(((qh_bit << 2) | qs_2bit) - 4) * sc_gm[n, g]
 */

#pragma once

#include <stdint.h>

// ============================================================================
// 1. DaVinci dav-l310 Hardware Capacities & Geometry
// ============================================================================

constexpr uint64_t UB_CAPACITY_BYTES       = 120 * 1024; // 120 KB usable UB (128 KB physical - 8 KB reserve)
constexpr uint64_t L1_CAPACITY_BYTES       = 512 * 1024; // 512 KB L1 (CBUF)
constexpr uint64_t L0C_CAPACITY_BYTES      = 32 * 1024;  // 32 KB L0C accumulator

constexpr int64_t  UB_BLOCK_BYTES          = 32;         // 32 B UB / DMA alignment unit
constexpr int64_t  FP16_BYTES              = static_cast<int64_t>(sizeof(half));
constexpr int64_t  FP16_PER_BLOCK          = UB_BLOCK_BYTES / FP16_BYTES; // 16

constexpr int64_t  VEC_BYTES               = 128;        // 128 B SIMD vector register
constexpr int64_t  VEC_FP16_LANES          = VEC_BYTES / FP16_BYTES; // 64

constexpr int64_t  CUBE_FRACTAL            = 16;         // 16 x 16 Cube fractal
constexpr int64_t  FRACTAL_ELEMS_F16       = CUBE_FRACTAL * CUBE_FRACTAL;     // 256 halfs
constexpr uint64_t FRACTAL_BYTES_F16       = FRACTAL_ELEMS_F16 * FP16_BYTES;  // 512 B
constexpr uint64_t L0C_PAIR_BYTES          = 2 * FRACTAL_BYTES_F16;           // 1024 B ping-pong pair

constexpr int64_t  L0AB_PINGPONG_ELEMS     = 8192;                            // 16 KB = 8192 halfs
constexpr uint64_t L0AB_PINGPONG_BYTES     = L0AB_PINGPONG_ELEMS * FP16_BYTES;

// SPR.CTRL bit positions (HIVM architectural specification)
constexpr int32_t  CTRL_BIT_FP_SATURATION  = 48; // FP16/BF16/FP8 saturation mode
constexpr int32_t  CTRL_BIT_VF_SAT_OVERRIDE = 60; // Vector-thread cast saturation override

// Cube / Fixpipe unit-flag synchronization modes
constexpr uint8_t  UF_MODE_ACCUM           = 2;
constexpr uint8_t  UF_MODE_FINAL           = 3;

// ============================================================================
// 2. W3A16 Quantization Format Constants (group_size = 32)
// ============================================================================

constexpr int32_t  QUANT_GROUP_SHIFT       = 5;
constexpr int64_t  QUANT_GROUP_SIZE        = 1LL << QUANT_GROUP_SHIFT; // 32 weights/group
constexpr int8_t   QUANT_ZERO_POINT        = 4;                        // [0..7] - 4 -> [-4..+3]
constexpr int8_t   QUANT_LOW2_MASK         = 0x3;                      // 2-bit low plane mask
constexpr int8_t   QUANT_HIGH_BIT_VAL      = 0x4;                      // 1 << 2 high bit value

constexpr uint64_t QH_BYTES_PER_GROUP      = QUANT_GROUP_SIZE / 8;     // 4 B  (32 x 1-bit)
constexpr uint64_t QS_BYTES_PER_GROUP      = QUANT_GROUP_SIZE / 4;     // 8 B  (32 x 2-bit)
constexpr uint64_t SC_BYTES_PER_GROUP      = FP16_BYTES;               // 2 B  (1 x FP16)
constexpr uint64_t ACT_BYTES_PER_GROUP     = QUANT_GROUP_SIZE * FP16_BYTES; // 64 B (32 x FP16)

// ============================================================================
// 3. Coupled M == 1 GEMV Geometry & Derived Memory Map
// ============================================================================

constexpr uint64_t GEMV_N_TILE             = 32; // 2 Cube N-fractals per output panel
constexpr uint64_t GEMV_N_FRACTALS         = GEMV_N_TILE / CUBE_FRACTAL; // 2

constexpr int32_t  GEMV_K_ALIGN            = 128;
constexpr uint64_t GEMV_K_TAIL             = 128;
constexpr uint64_t GEMV_K_STEP             = 256;
constexpr uint64_t GEMV_K_TAIL_GROUPS      = GEMV_K_TAIL / QUANT_GROUP_SIZE; // 4
constexpr uint64_t GEMV_K_STEP_GROUPS      = GEMV_K_STEP / QUANT_GROUP_SIZE; // 8

constexpr int32_t  GEMV_PANEL_MAX_K        = 1024;
constexpr uint64_t GEMV_PANEL_MAX_GROUPS   = GEMV_PANEL_MAX_K / QUANT_GROUP_SIZE; // 32
constexpr uint64_t GEMV_PANEL_MAX_Q4       = GEMV_PANEL_MAX_K / GEMV_K_ALIGN;     // 8
constexpr int32_t  GEMV_STREAM_MAX_K       = 32768;

constexpr uint64_t GEMV_PINGPONG_SLOTS     = 2;
constexpr uint64_t GEMV_RING_SLOTS         = 4;

// Strided NZ UB output geometry for N_TILE = 32 (2 fractals -> 32 UB blocks per k1 step)
constexpr uint16_t GEMV_NZ_BLK_STRIDE      = GEMV_N_FRACTALS * CUBE_FRACTAL;           // 32 blocks
constexpr uint32_t GEMV_NZ_VSSTB_CFG       = static_cast<uint32_t>(GEMV_NZ_BLK_STRIDE) << 16;
constexpr int64_t  GEMV_NZ_VSSTB_ELEMS     = 4 * GEMV_N_FRACTALS * FRACTAL_ELEMS_F16;  // 2048 halfs

// Fast-Path 1 (K <= 1024): 2 x 16 KB padded input panels + 4 x 16 KB NZ output ring
constexpr uint64_t GEMV_P1024_QH_ROW_BYTES = GEMV_PANEL_MAX_GROUPS * QH_BYTES_PER_GROUP; // 128 B
constexpr uint64_t GEMV_P1024_QS_ROW_BYTES = GEMV_PANEL_MAX_GROUPS * QS_BYTES_PER_GROUP; // 256 B
constexpr uint64_t GEMV_P1024_SC_ROW_BYTES = GEMV_PANEL_MAX_GROUPS * SC_BYTES_PER_GROUP; //  64 B

constexpr uint64_t GEMV_P1024_QH_OFF       = 0;
constexpr uint64_t GEMV_P1024_QS_OFF       = GEMV_P1024_QH_OFF + GEMV_N_TILE * GEMV_P1024_QH_ROW_BYTES; //  4096 B
constexpr uint64_t GEMV_P1024_SC_OFF       = GEMV_P1024_QS_OFF + GEMV_N_TILE * GEMV_P1024_QS_ROW_BYTES; // 12288 B
constexpr uint64_t GEMV_P1024_IN_STRIDE    = 16 * 1024;                                                 // 16384 B

constexpr uint64_t GEMV_WNZ_SLOT_BYTES     = GEMV_N_TILE * GEMV_K_STEP * FP16_BYTES;                    // 16384 B
constexpr uint64_t GEMV_P1024_WNZ_BASE     = GEMV_PINGPONG_SLOTS * GEMV_P1024_IN_STRIDE;                // 32768 B

// Fast-Path 2 (1024 < K <= 32768): 4 x 24 KB streaming UB slots (packed + NZ in each slot)
constexpr uint64_t GEMV_STREAM_QH_ROW_BYTES = GEMV_K_STEP_GROUPS * QH_BYTES_PER_GROUP;                  // 32 B
constexpr uint64_t GEMV_STREAM_QS_ROW_BYTES = GEMV_K_STEP_GROUPS * QS_BYTES_PER_GROUP;                  // 64 B
constexpr uint64_t GEMV_STREAM_SC_ROW_BYTES = UB_BLOCK_BYTES;                                           // 32 B (16 B padded to 1 block)

constexpr uint64_t GEMV_STREAM_QH_OFF      = 0;
constexpr uint64_t GEMV_STREAM_QS_OFF      = GEMV_STREAM_QH_OFF + GEMV_N_TILE * GEMV_STREAM_QH_ROW_BYTES; // 1024 B
constexpr uint64_t GEMV_STREAM_SC_OFF      = GEMV_STREAM_QS_OFF + GEMV_N_TILE * GEMV_STREAM_QS_ROW_BYTES; // 3072 B
constexpr uint64_t GEMV_STREAM_WNZ_OFF     = GEMV_STREAM_SC_OFF + GEMV_N_TILE * GEMV_STREAM_SC_ROW_BYTES; // 4096 B
constexpr int64_t  GEMV_STREAM_WNZ_ELEMS   = GEMV_STREAM_WNZ_OFF / FP16_BYTES;                            // 2048 halfs
constexpr uint64_t GEMV_STREAM_SLOT_BYTES  = 24 * 1024;                                                   // 24576 B

// VSELR byte-gather lookup tables in UB (6 x 128 B = 768 B immediately above 96 KB workspace)
constexpr uint64_t GEMV_LUT_UB_BASE        = GEMV_RING_SLOTS * GEMV_STREAM_SLOT_BYTES;                    // 98304 B
constexpr int32_t  GEMV_LUT_OFF_QH_IDX     = 0 * VEC_BYTES; //   0
constexpr int32_t  GEMV_LUT_OFF_QS_IDX     = 1 * VEC_BYTES; // 128
constexpr int32_t  GEMV_LUT_OFF_QH_SHIFT   = 2 * VEC_BYTES; // 256
constexpr int32_t  GEMV_LUT_OFF_QS_SHIFT   = 3 * VEC_BYTES; // 384
constexpr int32_t  GEMV_LUT_OFF_SC_IDX     = 4 * VEC_BYTES; // 512
constexpr int32_t  GEMV_LUT_OFF_MASK3      = 5 * VEC_BYTES; // 640

// Coupled GEMV L1, L0A/L0B, and 32-slot circular L0C layout
constexpr uint64_t GEMV_AL1_BASE           = 0;
constexpr uint64_t GEMV_AL1_MAX_BYTES      = static_cast<uint64_t>(GEMV_STREAM_MAX_K) * FP16_BYTES; // 65536 B
constexpr uint64_t GEMV_BL1_BASE           = GEMV_AL1_BASE + GEMV_AL1_MAX_BYTES;                    // 65536 B
constexpr uint64_t GEMV_BL1_STRIDE         = GEMV_WNZ_SLOT_BYTES;                                   // 16384 B

constexpr uint64_t GEMV_L0A_SLOT_BYTES     = CUBE_FRACTAL * GEMV_K_STEP * FP16_BYTES;               //  8192 B
constexpr uint64_t GEMV_L0B_SLOT_BYTES     = GEMV_N_TILE * GEMV_K_STEP * FP16_BYTES;                // 16384 B
constexpr uint64_t GEMV_L0C_SLOT_BYTES     = GEMV_N_FRACTALS * FRACTAL_BYTES_F16;                   //  1024 B
constexpr uint64_t GEMV_L0C_RING_SLOTS     = L0C_CAPACITY_BYTES / GEMV_L0C_SLOT_BYTES;              //    32 slots
constexpr uint64_t GEMV_L0C_NCOL_MASK      = (GEMV_L0C_RING_SLOTS - 1) * GEMV_N_TILE;               //   992

static_assert(GEMV_P1024_WNZ_BASE + GEMV_RING_SLOTS * GEMV_WNZ_SLOT_BYTES <= GEMV_LUT_UB_BASE,
              "Coupled GEMV P1024 NZ ring overlaps LUT table in UB");
static_assert(GEMV_RING_SLOTS * GEMV_STREAM_SLOT_BYTES <= GEMV_LUT_UB_BASE,
              "Coupled GEMV streaming slots overlap LUT table in UB");
static_assert(GEMV_LUT_UB_BASE + 6 * VEC_BYTES <= UB_CAPACITY_BYTES,
              "Coupled GEMV LUT table exceeds UB capacity");
static_assert(GEMV_BL1_BASE + GEMV_RING_SLOTS * GEMV_BL1_STRIDE <= L1_CAPACITY_BYTES,
              "Coupled GEMV BL1 ring exceeds L1 capacity");

// ============================================================================
// 4. General Tiled GEMM (16 x 512) Geometry & Derived Memory Map
// ============================================================================

constexpr int64_t  GEMM_M_TILE             = CUBE_FRACTAL;                       //  16
constexpr int64_t  GEMM_N_TILE             = CUBE_FRACTAL;                       //  16
constexpr int64_t  GEMM_K_TILE             = 512;                                // 512
constexpr int64_t  GEMM_K_TILE_GROUPS      = GEMM_K_TILE / QUANT_GROUP_SIZE;     //  16
constexpr int64_t  GEMM_QH_TILE_COLS       = GEMM_K_TILE / 8;                    //  64 B/row
constexpr int64_t  GEMM_QS_TILE_COLS       = GEMM_K_TILE / 4;                    // 128 B/row

constexpr uint16_t GEMM_NZ_BLK_STRIDE      = CUBE_FRACTAL;                       //  16 blocks
constexpr uint32_t GEMM_NZ_VSSTB_CFG       = static_cast<uint32_t>(GEMM_NZ_BLK_STRIDE) << 16;
constexpr int64_t  GEMM_NZ_VSSTB_ELEMS     = 4 * FRACTAL_ELEMS_F16;              // 1024 halfs

// 4-slot 16 x 512 UB weight ring + tightly packed activation UB ping-pong
constexpr uint64_t GEN_W_RING_SLOTS        = 4;

constexpr uint64_t GEN_W_SC_BYTES          = GEMM_N_TILE * GEMM_K_TILE_GROUPS * FP16_BYTES; //   512 B
constexpr uint64_t GEN_W_QH_BYTES          = GEMM_N_TILE * GEMM_QH_TILE_COLS;               //  1024 B
constexpr uint64_t GEN_W_QS_BYTES          = GEMM_N_TILE * GEMM_QS_TILE_COLS;               //  2048 B
constexpr int64_t  GEN_W_NZ_ELEMS          = GEMM_N_TILE * GEMM_K_TILE;                     //  8192 halfs
constexpr uint64_t GEN_W_NZ_BYTES          = GEN_W_NZ_ELEMS * FP16_BYTES;                   // 16384 B

constexpr uint64_t GEN_W_SC_OFF            = 0;
constexpr uint64_t GEN_W_QH_OFF            = GEN_W_SC_OFF + GEN_W_SC_BYTES;      //   512 B
constexpr uint64_t GEN_W_QS_OFF            = GEN_W_QH_OFF + GEN_W_QH_BYTES;      //  1536 B
constexpr uint64_t GEN_W_NZ_OFF            = GEN_W_QS_OFF + GEN_W_QS_BYTES;      //  3584 B
constexpr uint64_t GEN_W_SLOT_USED_BYTES   = GEN_W_NZ_OFF + GEN_W_NZ_BYTES;      // 19968 B
constexpr uint64_t GEN_W_SLOT_BYTES        = GEN_W_SLOT_USED_BYTES;              // 19968 B tight stride

// Activation ping-pong [16, 512] FP16 tiles placed immediately after Weight Slot 3
constexpr uint64_t GEN_ACT_TILE_BYTES      = GEMM_M_TILE * GEMM_K_TILE * FP16_BYTES; // 16384 B
constexpr uint64_t GEN_ACT_UB_PING         = (GEN_W_RING_SLOTS - 1) * GEN_W_SLOT_BYTES + GEN_W_SLOT_USED_BYTES; // 79872 B
constexpr uint64_t GEN_ACT_UB_PONG         = GEN_ACT_UB_PING + GEN_ACT_TILE_BYTES;                              // 96256 B

// L1 (512 KB) layout across streaming, stationary, and hybrid caching modes
constexpr uint64_t GEN_AL1_TILE_BYTES      = GEN_ACT_TILE_BYTES;                 // 16384 B
constexpr uint64_t GEN_BL1_BASE            = GEN_AL1_TILE_BYTES;                 // 16384 B (ping=16K, pong=48K)
constexpr uint64_t GEN_BL1_STRIDE          = 2 * GEN_AL1_TILE_BYTES;             // 32768 B

constexpr uint64_t GEN_AL1_STATIC_BASE     = 4 * GEN_AL1_TILE_BYTES;             // 65536 B (above [0..64K) BL1/stream)
constexpr uint64_t GEN_AL1_STATIC_BUDGET   = L1_CAPACITY_BYTES - GEN_AL1_STATIC_BASE; // 458752 B (448 KB)

// Stationary K-tile capacities derived from the 448 KB stationary AL1 budget
constexpr uint64_t GEMM_M1_K_TILE_BYTES    = GEMM_K_TILE * FP16_BYTES;           // 1024 B (1 x 512 halfs)
constexpr int32_t  GEMM_M1_MAX_K_TILES     = static_cast<int32_t>(GEN_AL1_STATIC_BUDGET / GEMM_M1_K_TILE_BYTES); // 448

constexpr int32_t  GEMM_M128_TILES         = 8;                                  // 8 x 16 = 128 M-rows
constexpr int32_t  GEMM_M128_MAX_K_TILES   = static_cast<int32_t>((GEN_AL1_STATIC_BUDGET / GEMM_M128_TILES) / GEN_AL1_TILE_BYTES); // 3
constexpr uint64_t GEMM_M128_AL1_STRIDE    = static_cast<uint64_t>(GEMM_M128_MAX_K_TILES) * GEN_AL1_TILE_BYTES; // 49152 B

constexpr int32_t  GEMM_M32_TILES          = 2;                                  // 2 x 16 = 32 M-rows
constexpr int32_t  GEMM_M32_MAX_K_TILES    = static_cast<int32_t>((GEN_AL1_STATIC_BUDGET / GEMM_M32_TILES) / GEN_AL1_TILE_BYTES); // 14
constexpr uint64_t GEMM_M32_AL1_TILE0_BASE = GEN_AL1_STATIC_BASE;                //  65536 B
constexpr uint64_t GEMM_M32_AL1_TILE1_BASE = GEN_AL1_STATIC_BASE + static_cast<uint64_t>(GEMM_M32_MAX_K_TILES) * GEN_AL1_TILE_BYTES; // 294912 B

// Hybrid M=32 mode: [0..96 KB) holds interleaved A0/BL1/A1 streaming ping-pong; [96..512 KB) caches K-tiles
constexpr uint64_t GEMM_HYBRID_A0_PING     = 0 * GEN_AL1_TILE_BYTES;             //      0 B
constexpr uint64_t GEMM_HYBRID_A0_PONG     = 2 * GEN_AL1_TILE_BYTES;             //  32768 B
constexpr uint64_t GEMM_HYBRID_A1_PING     = 4 * GEN_AL1_TILE_BYTES;             //  65536 B
constexpr uint64_t GEMM_HYBRID_A1_PONG     = 5 * GEN_AL1_TILE_BYTES;             //  81920 B
constexpr uint64_t GEMM_HYBRID_STREAM_END  = 6 * GEN_AL1_TILE_BYTES;             //  98304 B
constexpr int32_t  GEMM_HYBRID_CACHED_K    = static_cast<int32_t>(((L1_CAPACITY_BYTES - GEMM_HYBRID_STREAM_END) / GEMM_M32_TILES) / GEN_AL1_TILE_BYTES); // 13
constexpr uint64_t GEMM_HYBRID_CACHE0_BASE = GEMM_HYBRID_STREAM_END;             //  98304 B
constexpr uint64_t GEMM_HYBRID_CACHE1_BASE = GEMM_HYBRID_CACHE0_BASE + static_cast<uint64_t>(GEMM_HYBRID_CACHED_K) * GEN_AL1_TILE_BYTES; // 311296 B

static_assert(GEN_ACT_UB_PONG + GEN_ACT_TILE_BYTES <= UB_CAPACITY_BYTES,
              "General GEMM activation pong buffer exceeds UB capacity");
static_assert(GEMM_HYBRID_CACHE1_BASE + static_cast<uint64_t>(GEMM_HYBRID_CACHED_K) * GEN_AL1_TILE_BYTES <= L1_CAPACITY_BYTES,
              "General GEMM hybrid AL1 cache exceeds L1 capacity");

// ============================================================================
// 5. Internal Cross-Module Execution Tier Entry Points
// ============================================================================

extern "C" [aicore] __attribute__((always_inline)) void
gemv_m1_panel_k1024(__gm__ half*   x_gm,
                    __gm__ int8_t* qh_gm,
                    __gm__ int8_t* qs_gm,
                    __gm__ half*   sc_gm,
                    __gm__ half*   out_gm,
                    uint64_t       n_total,
                    uint64_t       k_total);

extern "C" [aicore] __attribute__((noinline)) void
gemv_m1_streaming(__gm__ half*   x_gm,
                  __gm__ int8_t* qh_gm,
                  __gm__ int8_t* qs_gm,
                  __gm__ half*   sc_gm,
                  __gm__ half*   out_gm,
                  uint64_t       n_total,
                  uint64_t       k_total);

extern "C" [aicore] __attribute__((noinline)) void
gemm_tiled_fallback(__gm__ half*    x_gm,
                    __gm__ int8_t*  qh_gm,
                    __gm__ int8_t*  qs_gm,
                    __gm__ half*    sc_gm,
                    __gm__ int32_t* shape_ptr,
                    __gm__ half*    out_gm);

// Multi-row (M >= 32) general GEMM sub-paths with L0B weight reuse across M-tiles
extern "C" [aicore] __attribute__((always_inline)) int32_t
gemm_m128_stationary(__gm__ half* x_gm, __gm__ int8_t* qh_gm,
                     __gm__ int8_t* qs_gm, __gm__ half* sc_gm,
                     __gm__ half* out_gm,
                     int32_t full_m_t16, int32_t n_tiles_raw,
                     int32_t num_n, int32_t num_k, int32_t last_k_idx,
                     int64_t k_tail, int64_t k_i64, int64_t n_i64,
                     int64_t k_groups, int32_t sc_n_step_i32,
                     uint64_t& w_iter, uint64_t& c_iter);

extern "C" [aicore] __attribute__((always_inline)) void
gemm_m32_stationary(__gm__ half* x_gm, __gm__ int8_t* qh_gm,
                    __gm__ int8_t* qs_gm, __gm__ half* sc_gm,
                    __gm__ half* out_gm,
                    int32_t m_limit_2, int32_t n_tiles_raw,
                    int32_t num_n, int32_t num_k, int32_t last_k_idx,
                    int64_t k_tail, int64_t k_i64, int64_t n_i64,
                    int64_t x_16row, int64_t out_16row,
                    int64_t k_groups, int32_t sc_n_step_i32,
                    uint64_t& w_iter, uint64_t& c_iter);

extern "C" [aicore] __attribute__((always_inline)) void
gemm_m32_hybrid(__gm__ half* x_gm, __gm__ int8_t* qh_gm,
                __gm__ int8_t* qs_gm, __gm__ half* sc_gm,
                __gm__ half* out_gm,
                int32_t m_limit_2, int32_t n_tiles_raw,
                int32_t num_n, int32_t num_k, int32_t last_k_idx,
                int64_t k_tail, int64_t k_i64, int64_t n_i64,
                int64_t x_16row, int64_t out_16row,
                int64_t k_groups, int32_t sc_n_step_i32,
                uint64_t& w_iter, uint64_t& c_iter);
