#pragma once
#include <cuda_runtime.h>

#include "buf_mgmt.h"
#include "gcli_budget.h"
#include "pred.h"
#include "bitpacking.h"
#include "precinct.h"
#include "sigbuffer.h"
#include "gcli_methods.h"
#include "rate_control.h"
#include "libjxs.h"
#define PACKING_GENERATE_FRAGMENT_CODE

typedef struct line_idx_info_t
{
    int start_idx;
    int stop_idx;
} line_idx_info_t;

typedef struct gpu_unpacked_info_t
{
    uint8_t nb_subpkts;
    int Lprc;
    int quantization;
    int refinement;

    int uses_raw_fallback[MAX_NBANDS];
    int gtli_table_data[MAX_NBANDS];
    int gtli_table_gcli[MAX_NBANDS];
    int gcli_sb_methods[MAX_NBANDS];

    int gcli_len[MAX_SUBPKTS];
    int sign_len[MAX_SUBPKTS];
    int data_len[MAX_SUBPKTS];

    line_idx_info_t line_idxs[MAX_SUBPKTS];

    bitstream_info_t gcli_sb_methods_bitstream_info;

    bitstream_info_t significance_bitstream_infos[MAX_SUBPKTS];
    bitstream_info_t gcli_bitstream_infos[MAX_SUBPKTS];
    bitstream_info_t sign_bitstream_infos[MAX_SUBPKTS];
    bitstream_info_t data_bitstream_infos[MAX_SUBPKTS];

    uint32_t *gpu_inclusion_mask = NULL;
    gcli_data_t *gpu_gcli_buf = NULL;

} gpu_unpacked_info_t;

typedef struct gpu_xs_dec_context_t
{
    const xs_config_t *xs_config;
    ids_t ids;
    ids_t *gpu_ids;
    int level_count;
    int idx_from_level[MAX_PRECINCT_HEIGHT][MAX_PACKETS];
    sigbuffer_t *gclis_significance;
    int use_sign_subpkt;
    uint32_t enabled_methods;
    int is_init;
    uint32_t nb_subpkts;
    uint64_t max_size;
    int quant_type;
    uint32_t group_size;
    bit_unpacker_t *bitstream;
    bitstream_type *gpu_bitstream_ptr;
    uint8_t *gpu_lvl_gains;
    uint8_t *gpu_lvl_priorities;
    uint32_t *gpu_gclis_prefix_sum;
    uint32_t *gpu_gclis_sizes;
    uint32_t *gclis_sizes;
    uint32_t *gclis_prefix_sum;
#ifdef PACKING_GENERATE_FRAGMENT_CODE
    xs_fragment_info_cb_t fragment_info_cb;
    void *fragment_info_context;
    int fragment_cnt;
    xs_fragment_info_cb_t user_fragment_info_cb;
    void *user_fragment_info_context;
    xs_buffering_fragment_t fragment_info_buf;
#endif
} gpu_xs_dec_context_t;

void gpu_unpack_bitstream_info(gpu_xs_dec_context_t *ctx, void *bitstream_buf, uint64_t bitstream_buf_size, xs_image_t *image_out);
