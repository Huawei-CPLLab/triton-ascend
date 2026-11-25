/**
 * Tiled GEMM DMA, ND2NZ layout conversion, Cube MMA, and Fixpipe primitives
 * for 16 x 512 tiles on dav-l310.
 */

#pragma once

#include "w3a16_defs.h"
#include "l310_gemv_ops.h"

template <typename T>
static inline [aicore] __attribute__((always_inline)) void
dma_load_gm_to_ub_2d(__gm__ T* src, int64_t gm_offset,
                     int64_t rows, int64_t cols, int64_t gm_stride,
                     __ubuf__ T* dst, int64_t ub_stride_cols = -1) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  constexpr int64_t elem_bytes = static_cast<int64_t>(sizeof(T));
  int64_t dst_cols          = (ub_stride_cols > 0) ? ub_stride_cols : cols;
  int64_t row_bytes         = cols * elem_bytes;
  int64_t aligned_row_bytes = (row_bytes + UB_BLOCK_BYTES - 1) & ~(UB_BLOCK_BYTES - 1);
  int64_t dst_gap_bytes     = dst_cols * elem_bytes - aligned_row_bytes;
  int64_t dst_gap_blocks    = (dst_gap_bytes > 0) ? (dst_gap_bytes / UB_BLOCK_BYTES) : 0;
  copy_gm_to_ubuf_align(
      dst, src + gm_offset, 0,
      static_cast<uint16_t>(rows),
      static_cast<uint32_t>(row_bytes),
      0, 0,
      static_cast<uint64_t>((gm_stride - cols) * elem_bytes),
      static_cast<uint32_t>(dst_gap_blocks));
}

static inline [aicore] __attribute__((always_inline)) void
dma_load_gm_to_l1_1d(__gm__ half* src, int64_t num_elems, __cbuf__ half* dst) {
  if (num_elems <= 0) {
    return;
  }
  copy_gm_to_cbuf_align(
      dst, src, 0, 1,
      static_cast<uint32_t>(num_elems * FP16_BYTES),
      0, 0, 0, 0);
}

static inline [aicore] __attribute__((always_inline)) void
dma_copy_ub_to_l1_1d(__ubuf__ half* src, __cbuf__ half* dst, int64_t num_elems) {
  if (num_elems <= 0) {
    return;
  }
  uint16_t burst_blocks = static_cast<uint16_t>((num_elems + FP16_PER_BLOCK - 1) / FP16_PER_BLOCK);
  copy_ubuf_to_cbuf(dst, src, 0, 1, burst_blocks, 0, 0);
}

// Converts row-major [rows, 512] in UB to NZ [32, 1, 16, 16] in L1
static inline [aicore] __attribute__((always_inline)) void
dma_copy_ub_to_l1_nd2nz(__ubuf__ half* src, int64_t rows, __cbuf__ half* dst) {
  constexpr uint16_t num_fractals = static_cast<uint16_t>(GEMM_K_TILE / FP16_PER_BLOCK);     // 32
  constexpr uint16_t len_burst    = 1;
  constexpr uint16_t dst_stride   = static_cast<uint16_t>(CUBE_FRACTAL - len_burst);         // 15

  if (rows == 1) {
    copy_ubuf_to_cbuf(dst, src, 0, num_fractals, len_burst, 0, dst_stride);
    return;
  }
#pragma clang loop unroll(disable)
  for (int64_t r = 0; r < rows; ++r) {
    copy_ubuf_to_cbuf(dst + r * FP16_PER_BLOCK, src + r * GEMM_K_TILE, 0,
                      num_fractals, len_burst, 0, dst_stride);
  }
}

// Single-K-tile (k <= 512, n == 16, B transposed) L1 -> L0A/L0B -> L0C Cube MMA
static inline [aicore] __attribute__((always_inline)) void
cube_mma_tile_f16_tb(__cbuf__ half* al1, __cbuf__ half* bl1,
                     bool init_c, int64_t m, int64_t k_actual,
                     __cc__ half* l0c, int64_t l0c_m_blocks,
                     int64_t kloop_db_cond, int64_t db_ev0, int64_t db_ev1,
                     uint8_t unit_flag, bool skip_l0b_load = false,
                     int64_t l0b_db_cond = -1) {
  int64_t k_ceil     = ((k_actual + CUBE_FRACTAL - 1) / CUBE_FRACTAL) * CUBE_FRACTAL;
  int64_t l0c_m_size = (m == 1) ? 1 : (l0c_m_blocks * CUBE_FRACTAL);

  int64_t mte1_mad_ping = 1 - (kloop_db_cond & 1);
  event_t ev            = static_cast<event_t>(mte1_mad_ping ? db_ev0 : db_ev1);
  int64_t l0b_ping      = (l0b_db_cond >= 0) ? (1 - (l0b_db_cond & 1)) : mte1_mad_ping;

  __ca__ half* l0a = reinterpret_cast<__ca__ half*>(
      static_cast<uintptr_t>(static_cast<uint64_t>(1 - mte1_mad_ping) * L0AB_PINGPONG_BYTES));
  __cb__ half* l0b = reinterpret_cast<__cb__ half*>(
      static_cast<uintptr_t>(static_cast<uint64_t>(1 - l0b_ping) * L0AB_PINGPONG_BYTES));

  wait_flag(PIPE_M, PIPE_MTE1, ev);

  uint16_t a_fractals = (l0c_m_size == 1)
                            ? static_cast<uint16_t>((static_cast<uint64_t>(k_ceil) + FRACTAL_ELEMS_F16 - 1) / FRACTAL_ELEMS_F16)
                            : static_cast<uint16_t>((k_ceil * CUBE_FRACTAL) / FRACTAL_ELEMS_F16);
  load_cbuf_to_ca(l0a, al1, 0, 0, 1, a_fractals, 1, 1, false);

  if (!skip_l0b_load) {
    uint16_t b_fractals = static_cast<uint16_t>((k_ceil * CUBE_FRACTAL) / FRACTAL_ELEMS_F16);
    load_cbuf_to_cb(l0b, bl1, 0, 0, 1, b_fractals, 1, 1, false);
  }

  set_flag(PIPE_MTE1, PIPE_M, ev);
  wait_flag(PIPE_MTE1, PIPE_M, ev);

  mad(l0c, l0a, l0b,
      static_cast<uint16_t>(l0c_m_size),
      static_cast<uint16_t>(k_actual),
      static_cast<uint16_t>(CUBE_FRACTAL),
      unit_flag, false, false, static_cast<uint8_t>(init_c));

  constexpr int64_t min_blocks_no_barrier = 10;
  if ((l0c_m_size / CUBE_FRACTAL) < min_blocks_no_barrier) {
    pipe_barrier(PIPE_M);
  }
  set_flag(PIPE_M, PIPE_MTE1, ev);
}

static inline [aicore] __attribute__((always_inline)) void
fixpipe_store_tile_nz2nd(__cc__ half* src_l0c, __gm__ half* out_gm, int64_t gm_offset,
                         int64_t m_rows, int64_t n_cols, int64_t gm_stride_n,
                         uint8_t unit_flag) {
  set_fixpipe_loop3_unit();
  copy_matrix_cc_to_gm(
      out_gm + gm_offset, src_l0c,
      0,                                  // sid
      static_cast<uint16_t>(n_cols),      // n_size (16)
      static_cast<uint16_t>(m_rows),      // m_size (1..16)
      static_cast<uint32_t>(gm_stride_n), // dst_D  (N)
      static_cast<uint16_t>(CUBE_FRACTAL),
      0,                                  // clip_relu_pre
      unit_flag,
      static_cast<uint64_t>(QuantMode_t::NoQuant),
      0, false, true,
      0, 0, false, false, 0, false, false, false, false, false, false);
}

static inline [aicore] __attribute__((always_inline)) void
dma_store_ub_to_gm_2d(__ubuf__ half* src_ub, int64_t ub_stride,
                      __gm__ half* dst_gm, int64_t gm_offset,
                      int64_t rows, int64_t cols, int64_t gm_stride) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  set_mov_pad_zero();
  if (rows == 1 && cols == 1) {
    copy_ubuf_to_gm_align(dst_gm + gm_offset, src_ub, 0, 1, FP16_BYTES, 0, 0);
    return;
  }
  int64_t ub_gap_bytes = (ub_stride - cols) * FP16_BYTES;
  int64_t gm_gap_bytes = (gm_stride - cols) * FP16_BYTES;
  if (gm_stride < cols || ub_gap_bytes < 0) {
    for (int64_t r = 0; r < rows; ++r) {
      copy_ubuf_to_gm_align(
          dst_gm + gm_offset + r * gm_stride,
          src_ub + r * ub_stride,
          0, 1, static_cast<uint32_t>(cols * FP16_BYTES), 0, 0);
    }
  } else {
    copy_ubuf_to_gm_align(
        dst_gm + gm_offset, src_ub, 0,
        static_cast<uint16_t>(rows),
        static_cast<uint32_t>(cols * FP16_BYTES),
        static_cast<uint32_t>(ub_gap_bytes / UB_BLOCK_BYTES),
        static_cast<uint64_t>(gm_gap_bytes));
  }
}
