/**
 * SIMD Vector Thread (__simd_vf__) kernels for General Tiled GEMM:
 *   - vf_dequant_general_16x512 : 2-row VINTLV dequantizer for 16 x 512 weight tiles
 *   - vf_zero_tile_16x16        : 16 x 16 FP16 zero-tile initializer for K <= 0
 */

#pragma once

#include "w3a16_defs.h"

static inline __simd_callee__ [aicore] __attribute__((always_inline)) void
vf_emit_half_row_256(vector_s8 wa, vector_s8 wb, vector_f16 sc_x8,
                     __ubuf__ half* dst_base, vector_bool p8, vector_bool p16) {
  vector_f16 s_a, s_b, s0, s1, s2, s3;
  vintlv(s_a, s_b, sc_x8, sc_x8);
  vintlv(s0, s1, s_a, s_a);
  vintlv(s2, s3, s_b, s_b);

  vector_s8 u0, u1;
  vintlv(u0, u1, wa, wb);

  vector_f16 f0 = __builtin_cce_vcvtif_s82f16_x(u0, p8, 0);
  vector_f16 f1 = __builtin_cce_vcvtif_s82f16_x(u1, p8, 0);
  vector_f16 f2 = __builtin_cce_vcvtif_s82f16_x(u0, p8, 1);
  vector_f16 f3 = __builtin_cce_vcvtif_s82f16_x(u1, p8, 1);

  vmul(f0, f0, s0, p16);
  vmul(f1, f1, s1, p16);
  vmul(f2, f2, s2, p16);
  vmul(f3, f3, s3, p16);

  vsstb(f0, dst_base + 0 * GEMM_NZ_VSSTB_ELEMS, GEMM_NZ_VSSTB_CFG, p16);
  vsstb(f1, dst_base + 1 * GEMM_NZ_VSSTB_ELEMS, GEMM_NZ_VSSTB_CFG, p16);
  vsstb(f2, dst_base + 2 * GEMM_NZ_VSSTB_ELEMS, GEMM_NZ_VSSTB_CFG, p16);
  vsstb(f3, dst_base + 3 * GEMM_NZ_VSSTB_ELEMS, GEMM_NZ_VSSTB_CFG, p16);
}

/**
 * General GEMM 16 x 512 VINTLV weight dequantizer:
 * Processes 2 N-rows (r, r+1) per iteration across all 8 iterations (16 rows x 512 cols).
 *   qh : [16,  64] int8  -> 128 B per 2 rows
 *   qs : [16, 128] int8  -> 128 B per row
 *   sc : [16,  16] half  ->  64 B per 2 rows (broadcast x32 via VINTLV tree)
 *   dst: [32, 16, 16] NZ half (16 KB, stored via VSSTB block_stride = 16 blocks = 512 B)
 */
static inline __simd_vf__ [aicore] __attribute__((noinline)) void
vf_dequant_general_16x512(__ubuf__ int8_t* __restrict__ qh,
                          __ubuf__ int8_t* __restrict__ qs,
                          __ubuf__ half*   __restrict__ sc,
                          __ubuf__ half*   __restrict__ dst) {
  vector_bool p16 = pset_b16(PAT_VL64);
  vector_bool p8  = pset_b8(PAT_VL128);

  vector_s8 c4, c3;
  vdup(c4, QUANT_ZERO_POINT, p8, MODE_ZEROING);
  vdup(c3, QUANT_LOW2_MASK,  p8, MODE_ZEROING);

  for (int32_t r = 0; r < static_cast<int32_t>(GEMM_N_TILE); r += 2) {
    int64_t r0 = static_cast<int64_t>(r);
    int64_t r1 = r0 + 1;

    // 1. Load 2 rows of qh (128 B) and split low/high nibbles into row r and row r+1
    vector_s8 qh_raw, qh_r0, qh_r1;
    vector_u8 qh_hi_nib;
    vlds(qh_raw, qh + r0 * GEMM_QH_TILE_COLS, 0, NORM);
    vshrs(qh_hi_nib, (vector_u8)qh_raw, (int16_t)4, p8);
    vintlv(qh_r0, qh_r1, qh_raw, (vector_s8)qh_hi_nib);

    // 2. Load row r and row r+1 of qs (128 B each)
    vector_s8 qs_r0, qs_r1;
    vlds(qs_r0, qs + r0 * GEMM_QS_TILE_COLS, 0, NORM);
    vlds(qs_r1, qs + r1 * GEMM_QS_TILE_COLS, 0, NORM);

    // Sub-plane 0 (bits 0..1 of qs, bit 0 of qh nibble)
    vector_s8 lo_a0, lo_a1, qa0, qa1;
    vector_u8 hi_a0, hi_a1;
    vand(lo_a0, qs_r0, c3, p8);
    vand(lo_a1, qs_r1, c3, p8);
    vshls(hi_a0, (vector_u8)qh_r0, (int16_t)2, p8);
    vshls(hi_a1, (vector_u8)qh_r1, (int16_t)2, p8);
    vand(qa0, (vector_s8)hi_a0, c4, p8);
    vand(qa1, (vector_s8)hi_a1, c4, p8);
    vadif(qa0, lo_a0, c4, p8);
    vadif(qa1, lo_a1, c4, p8);

    // Sub-plane 2 (bits 4..5 of qs, bit 2 of qh nibble)
    vector_u8 qs_shr4_0, qs_shr4_1;
    vector_s8 lo_c0, lo_c1, qc0, qc1;
    vshrs(qs_shr4_0, (vector_u8)qs_r0, (int16_t)4, p8);
    vshrs(qs_shr4_1, (vector_u8)qs_r1, (int16_t)4, p8);
    vand(lo_c0, (vector_s8)qs_shr4_0, c3, p8);
    vand(lo_c1, (vector_s8)qs_shr4_1, c3, p8);
    vand(qc0, qh_r0, c4, p8);
    vand(qc1, qh_r1, c4, p8);
    vadif(qc0, lo_c0, c4, p8);
    vadif(qc1, lo_c1, c4, p8);

    vector_s8 qac0_0, qac0_1, qac1_0, qac1_1;
    vintlv(qac0_0, qac0_1, qa0, qc0);
    vintlv(qac1_0, qac1_1, qa1, qc1);

    // Sub-plane 1 (bits 2..3 of qs, bit 1 of qh nibble)
    vector_u8 qs_shr2_0, qs_shr2_1, hi_b0, hi_b1;
    vector_s8 lo_b0, lo_b1, qb0, qb1;
    vshrs(qs_shr2_0, (vector_u8)qs_r0, (int16_t)2, p8);
    vshrs(qs_shr2_1, (vector_u8)qs_r1, (int16_t)2, p8);
    vand(lo_b0, (vector_s8)qs_shr2_0, c3, p8);
    vand(lo_b1, (vector_s8)qs_shr2_1, c3, p8);
    vshls(hi_b0, (vector_u8)qh_r0, (int16_t)1, p8);
    vshls(hi_b1, (vector_u8)qh_r1, (int16_t)1, p8);
    vand(qb0, (vector_s8)hi_b0, c4, p8);
    vand(qb1, (vector_s8)hi_b1, c4, p8);
    vadif(qb0, lo_b0, c4, p8);
    vadif(qb1, lo_b1, c4, p8);

    // Sub-plane 3 (bits 6..7 of qs, bit 3 of qh nibble)
    vector_u8 lo_d0, lo_d1, qh_shr1_0, qh_shr1_1;
    vector_s8 qd0, qd1;
    vshrs(lo_d0, (vector_u8)qs_r0, (int16_t)6, p8);
    vshrs(lo_d1, (vector_u8)qs_r1, (int16_t)6, p8);
    vshrs(qh_shr1_0, (vector_u8)qh_r0, (int16_t)1, p8);
    vshrs(qh_shr1_1, (vector_u8)qh_r1, (int16_t)1, p8);
    vand(qd0, (vector_s8)qh_shr1_0, c4, p8);
    vand(qd1, (vector_s8)qh_shr1_1, c4, p8);
    vadif(qd0, (vector_s8)lo_d0, c4, p8);
    vadif(qd1, (vector_s8)lo_d1, c4, p8);

    vector_s8 qbd0_0, qbd0_1, qbd1_0, qbd1_1;
    vintlv(qbd0_0, qbd0_1, qb0, qd0);
    vintlv(qbd1_0, qbd1_1, qb1, qd1);

    // Interleave sub-planes into ordered K chunks
    vector_s8 w0_0, w0_1, w0_2, w0_3, w1_0, w1_1, w1_2, w1_3;
    vintlv(w0_0, w0_1, qac0_0, qbd0_0);
    vintlv(w0_2, w0_3, qac0_1, qbd0_1);
    vintlv(w1_0, w1_1, qac1_0, qbd1_0);
    vintlv(w1_2, w1_3, qac1_1, qbd1_1);

    // 3. Load 2 rows of group scales (32 halfs = 64 B) and broadcast x32 via VINTLV
    int64_t row0_off = r0 * GEMM_K_TILE_GROUPS;
    int64_t row1_off = r1 * GEMM_K_TILE_GROUPS;
    vector_f16 sc_raw, sc_x2, sc_unused, sc_r0_x4, sc_r1_x4;
    vlds(sc_raw, sc + row0_off, 0, NORM);
    vintlv(sc_x2, sc_unused, sc_raw, sc_raw);
    vintlv(sc_r0_x4, sc_r1_x4, sc_x2, sc_x2);

    vector_f16 sc_r0_lo, sc_r0_hi, sc_r1_lo, sc_r1_hi;
    vintlv(sc_r0_lo, sc_r0_hi, sc_r0_x4, sc_r0_x4);
    vintlv(sc_r1_lo, sc_r1_hi, sc_r1_x4, sc_r1_x4);

    constexpr int64_t half_tile_nz_off = 4 * GEMM_NZ_VSSTB_ELEMS; // 4096 halfs (cols 256..511)
    vf_emit_half_row_256(w0_0, w0_1, sc_r0_lo, dst + row0_off,                    p8, p16);
    vf_emit_half_row_256(w0_2, w0_3, sc_r0_hi, dst + row0_off + half_tile_nz_off, p8, p16);
    vf_emit_half_row_256(w1_0, w1_1, sc_r1_lo, dst + row1_off,                    p8, p16);
    vf_emit_half_row_256(w1_2, w1_3, sc_r1_hi, dst + row1_off + half_tile_nz_off, p8, p16);
  }
}

/**
 * Zeroes a 16 x 16 FP16 UB tile (512 B) for the K <= 0 edge case
 */
static inline __simd_vf__ [aicore] __attribute__((noinline)) void
vf_zero_tile_16x16(__ubuf__ half* __restrict__ dst) {
  vector_bool p64 = pset_b16(PAT_VL64);
  vector_bool p16 = pset_b16(PAT_VL16);
  vector_f16 zero;
  vdup(zero, static_cast<half>(0.0f), p64, MODE_ZEROING);
  for (int32_t r = 0; r < static_cast<int32_t>(CUBE_FRACTAL); ++r) {
    vsts(zero, dst + static_cast<int64_t>(r) * CUBE_FRACTAL, 0, NORM_B16, p16);
  }
}
