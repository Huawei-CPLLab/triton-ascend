/**
 * Hardware primitives and pipeline synchronization helpers for coupled M == 1
 * GEMV fast-paths on dav-l310: DMA, Cube Load2D, MAD, and NZ2ND Fixpipe.
 */

#pragma once

#include "w3a16_defs.h"

static inline [aicore] __attribute__((always_inline)) void
set_mov_pad_zero() {
  set_padding(0ULL);
}

static inline [aicore] __attribute__((always_inline)) void
set_fixpipe_loop3_unit() {
  set_loop3_para(1ULL | (1ULL << 16) | (1ULL << 32));
}

static inline [aicore] __attribute__((always_inline)) void
dma_load_gm_to_l1_bytes(__cbuf__ half* dst, __gm__ half* src, uint64_t len_bytes) {
  copy_gm_to_cbuf_align(dst, src, 0, 1, static_cast<uint32_t>(len_bytes), 0, 0, 0, 0);
}

template <typename T>
static inline [aicore] __attribute__((always_inline)) void
dma_load_gm_to_ub_rows(__ubuf__ T* dst, __gm__ T* src,
                       uint64_t n_burst, uint64_t row_bytes,
                       uint64_t src_gap_bytes, uint64_t dst_gap_blocks) {
  copy_gm_to_ubuf_align(dst, src, 0,
                        static_cast<uint16_t>(n_burst),
                        static_cast<uint32_t>(row_bytes),
                        0, 0,
                        src_gap_bytes,
                        static_cast<uint32_t>(dst_gap_blocks));
}

static inline [aicore] __attribute__((always_inline)) void
dma_copy_ub_to_l1_blocks(__cbuf__ half* dst, __ubuf__ half* src, uint64_t num_blocks) {
  copy_ubuf_to_cbuf(dst, src, 0, 1, static_cast<uint16_t>(num_blocks), 0, 0);
}

static inline [aicore] __attribute__((always_inline)) void
cube_gemv_load2d_f16(__cbuf__ half* al1, __cbuf__ half* bl1,
                     uint64_t ab_ping, uint64_t k_blocks_16,
                     uint64_t cur_n, uint64_t bl1_n_fractals) {
  __ca__ half* l0a = reinterpret_cast<__ca__ half*>(
      static_cast<uintptr_t>(ab_ping * GEMV_L0A_SLOT_BYTES));
  __cb__ half* l0b = reinterpret_cast<__cb__ half*>(
      static_cast<uintptr_t>(ab_ping * GEMV_L0B_SLOT_BYTES));

  load_cbuf_to_ca(l0a, al1, 0, 0, 1, static_cast<uint16_t>(k_blocks_16), 1, 1, false);
  uint16_t n_step = (cur_n > static_cast<uint64_t>(CUBE_FRACTAL)) ? 2 : 1;
  load_cbuf_to_cb(l0b, bl1, 0, 0, n_step,
                  static_cast<uint16_t>(k_blocks_16),
                  static_cast<uint16_t>(bl1_n_fractals), n_step, false);
}

static inline [aicore] __attribute__((always_inline)) void
cube_gemv_mad_f16(__cc__ half* l0c, uint64_t ab_ping, uint64_t k_elems,
                  uint64_t cur_n, bool init_c, uint8_t unit_flag) {
  __ca__ half* l0a = reinterpret_cast<__ca__ half*>(
      static_cast<uintptr_t>(ab_ping * GEMV_L0A_SLOT_BYTES));
  __cb__ half* l0b = reinterpret_cast<__cb__ half*>(
      static_cast<uintptr_t>(ab_ping * GEMV_L0B_SLOT_BYTES));
  mad(l0c, l0a, l0b, 1,
      static_cast<uint16_t>(k_elems),
      static_cast<uint16_t>(cur_n),
      unit_flag, false, false, static_cast<uint8_t>(init_c));
  pipe_barrier(PIPE_M);
}

static inline [aicore] __attribute__((always_inline)) __cc__ half*
gemv_l0c_slot_ptr(uint64_t n_col) {
  constexpr uint64_t bytes_per_n_col = GEMV_L0C_SLOT_BYTES / GEMV_N_TILE; // 32 B
  return reinterpret_cast<__cc__ half*>((n_col & GEMV_L0C_NCOL_MASK) * bytes_per_n_col);
}

static inline [aicore] __attribute__((always_inline)) void
fixpipe_store_m1_nz2nd(__gm__ half* dst_gm, __cc__ half* src_l0c,
                       uint64_t cur_n, uint64_t dst_stride) {
  copy_matrix_cc_to_gm(
      dst_gm, src_l0c,
      0,                                 // sid
      static_cast<uint16_t>(cur_n),      // n_size
      1,                                 // m_size
      static_cast<uint32_t>(dst_stride), // dst_D
      static_cast<uint16_t>(CUBE_FRACTAL),
      0,                                 // clip_relu_pre
      UF_MODE_FINAL,
      static_cast<uint64_t>(QuantMode_t::NoQuant),
      0,                                 // pre_relu
      false,                             // channel_split
      true,                              // nz2nd_en
      0, 0, false, false, 0, false, false, false, false, false, false);
}

// Arms the 4-slot NZ/BL1 ring and 2-slot L0A/L0B ping-pong events for coupled GEMV
static inline [aicore] __attribute__((always_inline)) void
gemv_arm_downstream_rings() {
  set_flag(PIPE_MTE3, PIPE_V,    EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V,    EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V,    EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V,    EVENT_ID3);
  set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID3);
  set_flag(PIPE_M,    PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M,    PIPE_MTE1, EVENT_ID1);
}

// Drains the 4-slot NZ/BL1 ring and 2-slot L0A/L0B ping-pong events for coupled GEMV
static inline [aicore] __attribute__((always_inline)) void
gemv_drain_downstream_rings() {
  wait_flag(PIPE_MTE3, PIPE_V,    EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V,    EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V,    EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V,    EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_M,    PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M,    PIPE_MTE1, EVENT_ID1);
}
