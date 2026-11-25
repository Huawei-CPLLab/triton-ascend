/**
 * Shared 4-slot 16 x 512 weight-ring pipeline (GM -> UB -> VF dequant -> BL1)
 * and AL1 activation-pair staging helpers for General Tiled GEMM sub-paths.
 */

#pragma once

#include "w3a16_defs.h"
#include "gemm_tile_ops.h"
#include "vf_gemm.h"

static inline [aicore] __attribute__((always_inline)) void
weight_ring_load_tile(__gm__ int8_t* qh_gm, __gm__ int8_t* qs_gm, __gm__ half* sc_gm,
                      int64_t sc_gm_off, int64_t k_groups, uint64_t slot,
                      int64_t tile_groups = -1) {
  uint64_t ub_base = slot * GEN_W_SLOT_BYTES;
  constexpr int64_t qh_per_grp = static_cast<int64_t>(QH_BYTES_PER_GROUP);
  constexpr int64_t qs_per_grp = static_cast<int64_t>(QS_BYTES_PER_GROUP);
  int64_t cur_groups = (tile_groups > 0)
                           ? tile_groups
                           : ((k_groups < GEMM_K_TILE_GROUPS) ? k_groups : GEMM_K_TILE_GROUPS);

  wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(slot));
  dma_load_gm_to_ub_2d(
      sc_gm, sc_gm_off, GEMM_N_TILE, cur_groups, k_groups,
      reinterpret_cast<__ubuf__ half*>(ub_base + GEN_W_SC_OFF), GEMM_K_TILE_GROUPS);
  dma_load_gm_to_ub_2d(
      qh_gm, sc_gm_off * qh_per_grp, GEMM_N_TILE, cur_groups * qh_per_grp, k_groups * qh_per_grp,
      reinterpret_cast<__ubuf__ int8_t*>(ub_base + GEN_W_QH_OFF), GEMM_QH_TILE_COLS);
  dma_load_gm_to_ub_2d(
      qs_gm, sc_gm_off * qs_per_grp, GEMM_N_TILE, cur_groups * qs_per_grp, k_groups * qs_per_grp,
      reinterpret_cast<__ubuf__ int8_t*>(ub_base + GEN_W_QS_OFF), GEMM_QS_TILE_COLS);
  set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(slot));
}

static inline [aicore] __attribute__((always_inline)) void
weight_ring_prefetch_next(__gm__ int8_t* qh_gm, __gm__ int8_t* qs_gm, __gm__ half* sc_gm,
                          int32_t k_idx, int32_t num_k, bool has_next_n,
                          int64_t sc_n_base, int64_t next_sc_n_base,
                          int64_t k_groups, uint64_t w_iter) {
  constexpr uint64_t ring_mask = GEN_W_RING_SLOTS - 1;
  int32_t next_k = k_idx + 1;
  if (static_cast<uint32_t>(next_k) < static_cast<uint32_t>(num_k)) {
    uint64_t next_slot   = (w_iter + 1ULL) & ring_mask;
    int64_t  k_grp_off   = static_cast<int64_t>(next_k) * GEMM_K_TILE_GROUPS;
    int64_t  rem_groups  = k_groups - k_grp_off;
    int64_t  tile_groups = (rem_groups < GEMM_K_TILE_GROUPS) ? rem_groups : GEMM_K_TILE_GROUPS;
    weight_ring_load_tile(qh_gm, qs_gm, sc_gm,
                          sc_n_base + k_grp_off,
                          k_groups, next_slot, tile_groups);
  } else if (has_next_n) {
    uint64_t next_slot = (w_iter + 1ULL) & ring_mask;
    weight_ring_load_tile(qh_gm, qs_gm, sc_gm,
                          next_sc_n_base, k_groups, next_slot);
  }
}

static inline [aicore] __attribute__((always_inline)) void
weight_ring_dequant_slot(uint64_t slot) {
  uint64_t ub_base = slot * GEN_W_SLOT_BYTES;
  wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(slot));
  wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(slot));
  vf_dequant_general_16x512(
      reinterpret_cast<__ubuf__ int8_t*>(ub_base + GEN_W_QH_OFF),
      reinterpret_cast<__ubuf__ int8_t*>(ub_base + GEN_W_QS_OFF),
      reinterpret_cast<__ubuf__ half*>(ub_base + GEN_W_SC_OFF),
      reinterpret_cast<__ubuf__ half*>(ub_base + GEN_W_NZ_OFF));
  set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(slot));
  set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(slot));
}

static inline [aicore] __attribute__((always_inline)) __cbuf__ half*
weight_ring_stage_prev_to_bl1(uint64_t prev_w, uint64_t& out_mte3_ev) {
  constexpr uint64_t ring_mask = GEN_W_RING_SLOTS - 1;
  uint64_t prev_slot = prev_w & ring_mask;
  uint64_t prev_pp   = prev_w & 1ULL;
  out_mte3_ev        = prev_pp ^ 1ULL;

  __ubuf__ half* w_nz_ub = reinterpret_cast<__ubuf__ half*>(
      prev_slot * GEN_W_SLOT_BYTES + GEN_W_NZ_OFF);
  __cbuf__ half* bl1 = reinterpret_cast<__cbuf__ half*>(
      GEN_BL1_BASE + prev_pp * GEN_BL1_STRIDE);

  wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(prev_slot));
  dma_copy_ub_to_l1_1d(w_nz_ub, bl1, GEN_W_NZ_ELEMS);
  set_flag(PIPE_MTE3, PIPE_V,    static_cast<event_t>(prev_slot));
  set_flag(PIPE_MTE3, PIPE_MTE1, static_cast<event_t>(out_mte3_ev));
  wait_flag(PIPE_MTE3, PIPE_MTE1, static_cast<event_t>(out_mte3_ev));
  return bl1;
}

// Loads & ND2NZ-converts a pair of 16 x k_cur activation tiles from GM into AL1
static inline [aicore] __attribute__((always_inline)) void
load_act_pair_16x512_to_al1(__gm__ half* x_gm, int64_t gm_off0, int64_t gm_off1,
                            int64_t k_cur, int64_t k_stride,
                            __cbuf__ half* al1_0, __cbuf__ half* al1_1) {
  __ubuf__ half* ub_ping = reinterpret_cast<__ubuf__ half*>(GEN_ACT_UB_PING);
  __ubuf__ half* ub_pong = reinterpret_cast<__ubuf__ half*>(GEN_ACT_UB_PONG);

  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  dma_load_gm_to_ub_2d(x_gm, gm_off0, GEMM_M_TILE, k_cur, k_stride, ub_ping, GEMM_K_TILE);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);

  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  dma_load_gm_to_ub_2d(x_gm, gm_off1, GEMM_M_TILE, k_cur, k_stride, ub_pong, GEMM_K_TILE);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  dma_copy_ub_to_l1_nd2nz(ub_ping, GEMM_M_TILE, al1_0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  dma_copy_ub_to_l1_nd2nz(ub_pong, GEMM_M_TILE, al1_1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

static inline [aicore] __attribute__((always_inline)) void
select_hybrid_al1(int64_t k_i, uint64_t w_i,
                  __cbuf__ half*& a0, __cbuf__ half*& a1) {
  bool     cached  = (k_i < GEMM_HYBRID_CACHED_K);
  bool     odd_w   = ((w_i & 1ULL) == 1ULL);
  uint64_t k_bytes = static_cast<uint64_t>(k_i) * GEN_AL1_TILE_BYTES;
  a0 = reinterpret_cast<__cbuf__ half*>(
      cached ? (GEMM_HYBRID_CACHE0_BASE + k_bytes)
             : (odd_w ? GEMM_HYBRID_A0_PONG : GEMM_HYBRID_A0_PING));
  a1 = reinterpret_cast<__cbuf__ half*>(
      cached ? (GEMM_HYBRID_CACHE1_BASE + k_bytes)
             : (odd_w ? GEMM_HYBRID_A1_PONG : GEMM_HYBRID_A1_PING));
}
