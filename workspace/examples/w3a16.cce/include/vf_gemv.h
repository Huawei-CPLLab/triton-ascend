/**
 * SIMD Vector Thread (__simd_vf__) kernels for coupled M == 1 GEMV fast-paths:
 *   - vf_subbyte_lut_init   : Initializes 6 x 128 B VSELR byte-gather LUTs in UB
 *   - vf_dequant_p1024_t128 : Padded-panel VSELR dequantizer for K-tail = 128
 *   - vf_dequant_p1024_t256 : Padded-panel VSELR dequantizer for K-tile = 256
 *   - vf_dequant_t128       : Streaming 24 KB slot VSELR dequantizer for K-tail = 128
 *   - vf_dequant_t256       : Streaming 24 KB slot VSELR dequantizer for K-tile = 256
 */

#pragma once

#include "w3a16_defs.h"

/**
 * Initializes the 6 x 128-byte VSELR lookup tables in UB for coupled M=1 GEMV:
 *   lut0 (+0)  : qh byte gather index   (((i >> 4) & 7) + ((i & 1) << 3))
 *   lut1 (+128): qs byte gather index   (((i >> 2) & 30) + ((i & 1) << 5))
 *   lut2 (+256): qh right-shift amount  (((i >> 1) & 7))
 *   lut3 (+384): qs right-shift amount  (i & 6)
 *   lut4 (+512): sc byte gather index   ((i & 1) + (((i >> 5) & 2)))
 *   lut5 (+640): constant mask 3        (also shifted to derive 1, 4, 16, 64)
 */
static inline __simd_vf__ [aicore] __attribute__((noinline)) void
vf_subbyte_lut_init(__ubuf__ int8_t* __restrict__ lut0,
                    __ubuf__ int8_t* __restrict__ lut1,
                    __ubuf__ int8_t* __restrict__ lut2,
                    __ubuf__ int8_t* __restrict__ lut3,
                    __ubuf__ int8_t* __restrict__ lut4,
                    __ubuf__ int8_t* __restrict__ lut5) {
  vector_bool p8 = pset_b8(PAT_ALL);
  vector_s8 idx, c1, c3, c7, c15;
  vector_u8 c2, c6, c8, c30;

  vci(idx, (int8_t)0);
  vdup(c1, (int8_t)1, p8, MODE_ZEROING);
  vshls(c2,  (vector_u8)c1,  (int16_t)1, p8);
  vadd(c3,   c1,             (vector_s8)c2, p8);
  vshls(c6,  (vector_u8)c3,  (int16_t)1, p8);
  vadd(c7,   (vector_s8)c6,  c1, p8);
  vshls(c8,  (vector_u8)c1,  (int16_t)3, p8);
  vadd(c15,  c7,             (vector_s8)c8, p8);
  vshls(c30, (vector_u8)c15, (int16_t)1, p8);

  vector_s8 odd_bit;
  vector_u8 odd_sh3, odd_sh5;
  vand(odd_bit,  idx,                c1, p8);
  vshls(odd_sh3, (vector_u8)odd_bit, (int16_t)3, p8);
  vshls(odd_sh5, (vector_u8)odd_bit, (int16_t)5, p8);

  vector_u8 shr4, shr2, shr1, shr5;
  vshrs(shr4, (vector_u8)idx, (int16_t)4, p8);
  vshrs(shr2, (vector_u8)idx, (int16_t)2, p8);
  vshrs(shr1, (vector_u8)idx, (int16_t)1, p8);
  vshrs(shr5, (vector_u8)idx, (int16_t)5, p8);

  vector_s8 v_lut0, v_lut1, v_lut2, v_lut3, v_lut4;
  vand(v_lut0, (vector_s8)shr4, c7, p8);
  vadd(v_lut0, v_lut0, (vector_s8)odd_sh3, p8);

  vand(v_lut1, (vector_s8)shr2, (vector_s8)c30, p8);
  vadd(v_lut1, v_lut1, (vector_s8)odd_sh5, p8);

  vand(v_lut2, (vector_s8)shr1, c7, p8);
  vand(v_lut3, idx, (vector_s8)c6, p8);

  vand(v_lut4, (vector_s8)shr5, (vector_s8)c2, p8);
  vadd(v_lut4, odd_bit, v_lut4, p8);

  vsts(v_lut0, lut0, 0, NORM_B16, p8);
  vsts(v_lut1, lut1, 0, NORM_B16, p8);
  vsts(v_lut2, lut2, 0, NORM_B16, p8);
  vsts(v_lut3, lut3, 0, NORM_B16, p8);
  vsts(v_lut4, lut4, 0, NORM_B16, p8);
  vsts(c3,     lut5, 0, NORM_B16, p8);
}

static inline [aicore] __attribute__((always_inline)) void
init_gemv_subbyte_luts() {
  vf_subbyte_lut_init(
      reinterpret_cast<__ubuf__ int8_t*>(GEMV_LUT_UB_BASE + GEMV_LUT_OFF_QH_IDX),
      reinterpret_cast<__ubuf__ int8_t*>(GEMV_LUT_UB_BASE + GEMV_LUT_OFF_QS_IDX),
      reinterpret_cast<__ubuf__ int8_t*>(GEMV_LUT_UB_BASE + GEMV_LUT_OFF_QH_SHIFT),
      reinterpret_cast<__ubuf__ int8_t*>(GEMV_LUT_UB_BASE + GEMV_LUT_OFF_QS_SHIFT),
      reinterpret_cast<__ubuf__ int8_t*>(GEMV_LUT_UB_BASE + GEMV_LUT_OFF_SC_IDX),
      reinterpret_cast<__ubuf__ int8_t*>(GEMV_LUT_UB_BASE + GEMV_LUT_OFF_MASK3));
}

// Loads all 6 VSELR byte-gather lookup vectors from UB
static inline __simd_callee__ [aicore] __attribute__((always_inline)) void
vf_load_gemv_luts(__ubuf__ int8_t* lut,
                  vector_u8& qh_idx0, vector_u8& qs_idx0,
                  vector_u8& qh_shift, vector_u8& qs_shift,
                  vector_u8& sc_base_idx, vector_u8& mask3) {
  auto* lut_u8 = reinterpret_cast<__ubuf__ uint8_t*>(lut);
  vlds(qh_idx0,     lut_u8, GEMV_LUT_OFF_QH_IDX,   NORM);
  vlds(qs_idx0,     lut_u8, GEMV_LUT_OFF_QS_IDX,   NORM);
  vlds(qh_shift,    lut_u8, GEMV_LUT_OFF_QH_SHIFT, NORM);
  vlds(qs_shift,    lut_u8, GEMV_LUT_OFF_QS_SHIFT, NORM);
  vlds(sc_base_idx, lut_u8, GEMV_LUT_OFF_SC_IDX,   NORM);
  vlds(mask3,       lut_u8, GEMV_LUT_OFF_MASK3,    NORM);
}

// Unpacks 128 3-bit weights (4 groups of 32) into signed int8 [-4..+3] via VSELR gather
static inline __simd_callee__ [aicore] __attribute__((always_inline)) vector_s8
vf_unpack_w3_chunk128(vector_s8 qh_raw, vector_s8 qs_raw,
                      vector_u8 qh_idx, vector_u8 qs_idx,
                      vector_u8 qh_shift, vector_u8 qs_shift,
                      vector_u8 mask3, vector_u8 c4, vector_bool p8) {
  vector_s8 qh_b, q_hi, qs_b, q_lo;
  vector_u8 qh_shl2;

  vselr(qh_b, qh_raw, qh_idx);
  vshr(qh_b, qh_b, (vector_s8)qh_shift, p8);
  vshls(qh_shl2, (vector_u8)qh_b, (int16_t)2, p8);
  vand(q_hi, (vector_s8)qh_shl2, (vector_s8)c4, p8);

  vselr(qs_b, qs_raw, qs_idx);
  vshr(qs_b, qs_b, (vector_s8)qs_shift, p8);
  vand(q_lo, qs_b, (vector_s8)mask3, p8);

  vadif(q_hi, q_lo, (vector_s8)c4, p8);
  return q_hi;
}

// Converts 128 int8 weights -> 2 x 64 FP16, applies gathered group scales, and stores NZ.
// Keeps (dst_base + row_off) + chunk_off GEP order so vloop-lowering hoists zero extra pointers.
static inline __simd_callee__ [aicore] __attribute__((always_inline)) void
vf_scale_and_store_nz_chunk128(vector_s8 q_chunk, vector_u8 sc_raw,
                               vector_u8 sc_idx_even, vector_u8 sc_idx_odd,
                               __ubuf__ half* dst_base, int64_t row_off,
                               int64_t chunk_off, vector_bool p16) {
  vector_f16 f_even, f_odd;
  vfcvt(f_even, q_chunk, PART_EVEN);
  vfcvt(f_odd,  q_chunk, PART_ODD);

  vector_u8 sc_even, sc_odd;
  vselr(sc_even, sc_raw, sc_idx_even);
  vselr(sc_odd,  sc_raw, sc_idx_odd);

  vmul(f_even, f_even, (vector_f16)sc_even, p16);
  vmul(f_odd,  f_odd,  (vector_f16)sc_odd,  p16);

  vsstb(f_even, dst_base + row_off + chunk_off,                         GEMV_NZ_VSSTB_CFG, p16);
  vsstb(f_odd,  dst_base + row_off + (chunk_off + GEMV_NZ_VSSTB_ELEMS), GEMV_NZ_VSSTB_CFG, p16);
}

// Dequantizes 256 3-bit weights (8 groups of 32) from preloaded (qh_raw, qs_raw, sc_raw)
// and stores 4 strided NZ FP16 chunks, keeping all loads & gathers scheduled ahead of VMUL.
static inline __simd_callee__ [aicore] __attribute__((always_inline)) void
vf_dequant_and_store_nz_row256(vector_s8 qh_raw, vector_s8 qs_raw, vector_u8 sc_raw,
                               vector_u8 qh_idx0, vector_u8 qh_idx1,
                               vector_u8 qs_idx0, vector_u8 qs_idx1,
                               vector_u8 qh_shift, vector_u8 qs_shift,
                               vector_u8 mask3, vector_u8 c4,
                               vector_u8 sc_idx0, vector_u8 sc_idx1,
                               vector_u8 sc_idx2, vector_u8 sc_idx3,
                               __ubuf__ half* dst_base, int64_t row_off,
                               int64_t base_off, vector_bool p8, vector_bool p16) {
  vector_s8 qh_b0, qh_b1, q0, q1;
  vector_u8 qh_sh0, qh_sh1;
  vselr(qh_b0, qh_raw, qh_idx0);
  vselr(qh_b1, qh_raw, qh_idx1);
  vshr(qh_b0, qh_b0, (vector_s8)qh_shift, p8);
  vshr(qh_b1, qh_b1, (vector_s8)qh_shift, p8);
  vshls(qh_sh0, (vector_u8)qh_b0, (int16_t)2, p8);
  vshls(qh_sh1, (vector_u8)qh_b1, (int16_t)2, p8);
  vand(q0, (vector_s8)qh_sh0, (vector_s8)c4, p8);
  vand(q1, (vector_s8)qh_sh1, (vector_s8)c4, p8);

  vector_s8 qs_b0, qs_b1, lo0, lo1;
  vselr(qs_b0, qs_raw, qs_idx0);
  vselr(qs_b1, qs_raw, qs_idx1);
  vshr(qs_b0, qs_b0, (vector_s8)qs_shift, p8);
  vshr(qs_b1, qs_b1, (vector_s8)qs_shift, p8);
  vand(lo0, qs_b0, (vector_s8)mask3, p8);
  vand(lo1, qs_b1, (vector_s8)mask3, p8);

  vadif(q0, lo0, (vector_s8)c4, p8);
  vadif(q1, lo1, (vector_s8)c4, p8);

  vector_f16 f0, f1, f2, f3;
  vfcvt(f0, q0, PART_EVEN);
  vfcvt(f1, q0, PART_ODD);
  vfcvt(f2, q1, PART_EVEN);
  vfcvt(f3, q1, PART_ODD);

  vector_u8 sb0, sb1, sb2, sb3;
  vselr(sb0, sc_raw, sc_idx0);
  vselr(sb1, sc_raw, sc_idx1);
  vselr(sb2, sc_raw, sc_idx2);
  vselr(sb3, sc_raw, sc_idx3);

  vmul(f0, f0, (vector_f16)sb0, p16);
  vmul(f1, f1, (vector_f16)sb1, p16);
  vmul(f2, f2, (vector_f16)sb2, p16);
  vmul(f3, f3, (vector_f16)sb3, p16);

  vsstb(f0, dst_base + row_off + (base_off + 0 * GEMV_NZ_VSSTB_ELEMS), GEMV_NZ_VSSTB_CFG, p16);
  vsstb(f1, dst_base + row_off + (base_off + 1 * GEMV_NZ_VSSTB_ELEMS), GEMV_NZ_VSSTB_CFG, p16);
  vsstb(f2, dst_base + row_off + (base_off + 2 * GEMV_NZ_VSSTB_ELEMS), GEMV_NZ_VSSTB_CFG, p16);
  vsstb(f3, dst_base + row_off + (base_off + 3 * GEMV_NZ_VSSTB_ELEMS), GEMV_NZ_VSSTB_CFG, p16);
}

/**
 * Coupled GEMV (K <= 1024) padded-panel VSELR dequantizer for K-tail = 128 (4 groups x rows)
 */
static inline __simd_vf__ [aicore] __attribute__((noinline)) void
vf_dequant_p1024_t128(__ubuf__ int8_t* __restrict__ qh,
                      __ubuf__ int8_t* __restrict__ qs,
                      __ubuf__ int8_t* __restrict__ sc,
                      __ubuf__ half*   __restrict__ dst,
                      __ubuf__ int8_t* __restrict__ lut,
                      int32_t rows,
                      uint8_t sc_byte_off) {
  vector_bool p8  = pset_b8(PAT_ALL);
  vector_bool p16 = pset_b16(PAT_ALL);

  vector_u8 lut0, lut1, lut2, lut3, lut4, lut5;
  vf_load_gemv_luts(lut, lut0, lut1, lut2, lut3, lut4, lut5);

  vector_u8 c1, c4, sc_idx0, sc_idx1;
  vshrs(c1, lut5, (int16_t)1, p8);
  vshls(c4, c1,   (int16_t)2, p8);
  vadds(sc_idx0, lut4, sc_byte_off, p8);
  vadd(sc_idx1,  sc_idx0, c4, p8);

  int32_t r = 0;
  do {
    vector_s8 qh_raw, qs_raw;
    vector_u8 sc_raw;
    vlds(qh_raw, qh, r * static_cast<int32_t>(GEMV_P1024_QH_ROW_BYTES), BLK);
    vlds(qs_raw, qs, r * static_cast<int32_t>(GEMV_P1024_QS_ROW_BYTES), US_B8);
    vlds(sc_raw, reinterpret_cast<__ubuf__ uint8_t*>(sc),
         r * static_cast<int32_t>(GEMV_P1024_SC_ROW_BYTES), BLK);

    vector_s8 q0 = vf_unpack_w3_chunk128(qh_raw, qs_raw, lut0, lut1, lut2, lut3, lut5, c4, p8);
    int64_t row_off = static_cast<int64_t>(r) * FP16_PER_BLOCK;
    vf_scale_and_store_nz_chunk128(q0, sc_raw, sc_idx0, sc_idx1, dst, row_off, 0, p16);
    ++r;
  } while (r < rows);
}

/**
 * Coupled GEMV (K <= 1024) padded-panel VSELR dequantizer for K-tile = 256 (8 groups x rows)
 */
static inline __simd_vf__ [aicore] __attribute__((noinline)) void
vf_dequant_p1024_t256(__ubuf__ int8_t* __restrict__ qh,
                      __ubuf__ int8_t* __restrict__ qs,
                      __ubuf__ int8_t* __restrict__ sc,
                      __ubuf__ half*   __restrict__ dst,
                      __ubuf__ int8_t* __restrict__ lut,
                      int32_t rows,
                      uint8_t sc_byte_off) {
  vector_bool p8  = pset_b8(PAT_ALL);
  vector_bool p16 = pset_b16(PAT_ALL);

  vector_u8 lut0, lut1, lut2, lut3, lut4, lut5;
  vf_load_gemv_luts(lut, lut0, lut1, lut2, lut3, lut4, lut5);

  vector_u8 c1, c4, sc_idx0, sc_idx1, c16, c64, qh_idx1, qs_idx1, sc_idx2, sc_idx3;
  vshrs(c1, lut5, (int16_t)1, p8);
  vshls(c4, c1,   (int16_t)2, p8);
  vadds(sc_idx0, lut4, sc_byte_off, p8);
  vadd(sc_idx1,  sc_idx0, c4, p8);
  vshls(c16, c4,  (int16_t)2, p8);
  vshls(c64, c16, (int16_t)2, p8);
  vadd(qh_idx1, lut0, c16, p8);
  vadd(qs_idx1, lut1, c64, p8);
  vadd(sc_idx2, sc_idx1, c4, p8);
  vadd(sc_idx3, sc_idx2, c4, p8);

  int32_t r = 0;
  do {
    vector_s8 qh_raw, qs_raw;
    vector_u8 sc_raw;
    vlds(qh_raw, qh, r * static_cast<int32_t>(GEMV_P1024_QH_ROW_BYTES), BLK);
    vlds(qs_raw, qs, r * static_cast<int32_t>(GEMV_P1024_QS_ROW_BYTES), US_B8);
    vlds(sc_raw, reinterpret_cast<__ubuf__ uint8_t*>(sc),
         r * static_cast<int32_t>(GEMV_P1024_SC_ROW_BYTES), BLK);

    int64_t row_off = static_cast<int64_t>(r) * FP16_PER_BLOCK;
    vf_dequant_and_store_nz_row256(
        qh_raw, qs_raw, sc_raw, lut0, qh_idx1, lut1, qs_idx1,
        lut2, lut3, lut5, c4, sc_idx0, sc_idx1, sc_idx2, sc_idx3,
        dst, row_off, 0, p8, p16);
    ++r;
  } while (r < rows);
}

/**
 * Coupled GEMV (K > 1024) streaming 24 KB slot VSELR dequantizer for K-tail = 128
 */
static inline __simd_vf__ [aicore] __attribute__((noinline)) void
vf_dequant_t128(__ubuf__ half*   __restrict__ slot_ub,
                __ubuf__ int8_t* __restrict__ lut,
                int32_t rows) {
  vector_bool p8  = pset_b8(PAT_ALL);
  vector_bool p16 = pset_b16(PAT_ALL);

  vector_u8 lut0, lut1, lut2, lut3, lut4, lut5;
  vf_load_gemv_luts(lut, lut0, lut1, lut2, lut3, lut4, lut5);

  vector_u8 c1, c4, sc_idx1;
  vshrs(c1, lut5, (int16_t)1, p8);
  vshls(c4, c1,   (int16_t)2, p8);
  vadd(sc_idx1, lut4, c4, p8);

  __ubuf__ int8_t*  s8_base = reinterpret_cast<__ubuf__ int8_t*>(slot_ub);
  __ubuf__ uint8_t* u8_base = reinterpret_cast<__ubuf__ uint8_t*>(slot_ub);

  int32_t r = 0;
  do {
    vector_s8 qh_raw, qs_raw;
    vector_u8 sc_raw;
    vlds(qh_raw, s8_base, r * static_cast<int32_t>(GEMV_STREAM_QH_ROW_BYTES) + static_cast<int32_t>(GEMV_STREAM_QH_OFF), BLK);
    vlds(qs_raw, s8_base, r * static_cast<int32_t>(GEMV_STREAM_QS_ROW_BYTES) + static_cast<int32_t>(GEMV_STREAM_QS_OFF), US_B8);
    vlds(sc_raw, u8_base, r * static_cast<int32_t>(GEMV_STREAM_SC_ROW_BYTES) + static_cast<int32_t>(GEMV_STREAM_SC_OFF), BLK);

    vector_s8 q0 = vf_unpack_w3_chunk128(qh_raw, qs_raw, lut0, lut1, lut2, lut3, lut5, c4, p8);
    int64_t row_off = static_cast<int64_t>(r) * FP16_PER_BLOCK;
    vf_scale_and_store_nz_chunk128(q0, sc_raw, lut4, sc_idx1, slot_ub, row_off,
                                   GEMV_STREAM_WNZ_ELEMS, p16);
    ++r;
  } while (r < rows);
}

/**
 * Coupled GEMV (K > 1024) streaming 24 KB slot VSELR dequantizer for K-tile = 256
 */
static inline __simd_vf__ [aicore] __attribute__((noinline)) void
vf_dequant_t256(__ubuf__ half*   __restrict__ slot_ub,
                __ubuf__ int8_t* __restrict__ lut,
                int32_t rows) {
  vector_bool p8  = pset_b8(PAT_ALL);
  vector_bool p16 = pset_b16(PAT_ALL);

  vector_u8 lut0, lut1, lut2, lut3, lut4, lut5;
  vf_load_gemv_luts(lut, lut0, lut1, lut2, lut3, lut4, lut5);

  vector_u8 c1, c4, sc_idx1, c16, c64, qh_idx1, qs_idx1, sc_idx2, sc_idx3;
  vshrs(c1, lut5, (int16_t)1, p8);
  vshls(c4, c1,   (int16_t)2, p8);
  vadd(sc_idx1, lut4, c4, p8);
  vshls(c16, c4,  (int16_t)2, p8);
  vshls(c64, c16, (int16_t)2, p8);
  vadd(qh_idx1, lut0, c16, p8);
  vadd(qs_idx1, lut1, c64, p8);
  vadd(sc_idx2, sc_idx1, c4, p8);
  vadd(sc_idx3, sc_idx2, c4, p8);

  __ubuf__ int8_t*  s8_base = reinterpret_cast<__ubuf__ int8_t*>(slot_ub);
  __ubuf__ uint8_t* u8_base = reinterpret_cast<__ubuf__ uint8_t*>(slot_ub);

  int32_t r = 0;
  do {
    vector_s8 qh_raw, qs_raw;
    vector_u8 sc_raw;
    vlds(qh_raw, s8_base, r * static_cast<int32_t>(GEMV_STREAM_QH_ROW_BYTES) + static_cast<int32_t>(GEMV_STREAM_QH_OFF), BLK);
    vlds(qs_raw, s8_base, r * static_cast<int32_t>(GEMV_STREAM_QS_ROW_BYTES) + static_cast<int32_t>(GEMV_STREAM_QS_OFF), US_B8);
    vlds(sc_raw, u8_base, r * static_cast<int32_t>(GEMV_STREAM_SC_ROW_BYTES) + static_cast<int32_t>(GEMV_STREAM_SC_OFF), BLK);

    int64_t row_off = static_cast<int64_t>(r) * FP16_PER_BLOCK;
    vf_dequant_and_store_nz_row256(
        qh_raw, qs_raw, sc_raw, lut0, qh_idx1, lut1, qs_idx1,
        lut2, lut3, lut5, c4, lut4, sc_idx1, sc_idx2, sc_idx3,
        slot_ub, row_off, GEMV_STREAM_WNZ_ELEMS, p8, p16);
    ++r;
  } while (r < rows);
}
