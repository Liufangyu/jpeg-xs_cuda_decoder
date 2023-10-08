#include <cuda_runtime.h>

#include "buf_mgmt.h"
#include "gcli_budget.h"
#include "pred.h"
#include "bitpacking.h"
#include "precinct.h"
#include "sigbuffer.h"
#include "gcli_methods.h"
#include "rate_control.h"
#include "packing.cuh"
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include "libjxs.h"
#include "xs_config.h"
#include "xs_markers.h"
#include "common.h"
#include "precinct.h"
#include "buf_mgmt.h"
#include "bitpacking.h"
#include "budget.h"
#include "packing.h"
#include "quant.h"
#include "ids.h"
#include "dwt.h"
#include "mct.h"
#include "nlt.h"

/*

nvcc bitpacking.c budget.c buf_mgmt.c data_budget.c dwt.c gcli_budget.c gcli_methods.c ids.c image.c mct.c nlt.c packing.c precinct.c precinct_budget.c precinct_budget_table.c pred.c predbuffer.c quant.c rate_control.c sb_weighting.c sig_flags.c sigbuffer.c version.c xs_config.c xs_config_parser.c xs_dec.c xs_markers.c packing.cu xs_dec_main.c file_io.c cmdline_options.c file_sequence.c image_open.c v210.c rgb16.c yuv16.c planar.c uyvy8.c argb.c mono.c ppm.c pgx.c helpers.c -o jpegxs_decoder -w -rdc=true -gencode=arch=compute_61,code=compute_61
*/

#define SIGFLAGS_NEXTLVL_SIZE(w, g) (((w) + (g)-1) / (g))
#define MAXB (sizeof(uint64_t) * 8)

__global__ void kernel_convert_ipx_htobe64(uint64_t *bitstream_ptr, uint64_t len)
{
    const uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        uint64_t in = bitstream_ptr[tid];
        union
        {
            uint64_t integer;
            uint8_t bytes[8];
        } a, b;
        a.integer = in;
        for (int i = 0; i < 8; i++)
            b.bytes[i] = a.bytes[7 - i];
        bitstream_ptr[tid] = b.integer;
    }
}

void gpu_convert_ipx_htobe64(uint64_t *bitstream_ptr, uint64_t max_size)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (max_size + block_size - 1) / block_size;
    kernel_convert_ipx_htobe64<<<grid_size, block_size>>>(bitstream_ptr, max_size);
}
__global__ void kernel_compute_gtli_tables(gpu_unpacked_info_t *infos,
                                           const uint8_t *sb_gains, const uint8_t *sb_priority,
                                           const uint32_t n_lvls, const uint32_t column_num, const uint32_t len)
{

    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const uint32_t prec_y_idx = tid / (n_lvls * column_num);
        const uint32_t column = (tid / n_lvls) % column_num;
        const uint32_t lvl = tid % n_lvls;
        gpu_unpacked_info_t info_cur = infos[prec_y_idx * column_num + column];
        const int gain = sb_gains[lvl];
        const int scenario = info_cur.quantization;
        const int refinement = info_cur.refinement;
        int val = scenario - gain;
        const uint8_t add_1bp = (sb_priority[lvl] < refinement);
        if (add_1bp)
            val -= 1;
        val = MAX(val, 0);
        val = MIN(val, MAX_GCLI);
        info_cur.gtli_table_data[lvl] = info_cur.gtli_table_gcli[lvl] = val;
    }
}

inline int prec_y_idx_is_first_of_slice(const ids_t *ids, const uint32_t prec_y_idx, const uint32_t slice_height)
{
    assert(prec_y_idx >= 0 && slice_height > 0);
    return (((prec_y_idx * ids->ph) % slice_height) == 0);
}
inline int precinct_subpkt_of(const ids_t *ids, uint32_t position)
{
    return ids->pi[position].s;
}
__device__ __host__ void precinct_band_index_of(const ids_t *ids, uint32_t position, uint32_t *val)
{
    *val = ids->pi[position].b;
}
__device__ __host__ void precinct_ypos_of(const ids_t *ids, uint32_t position, uint32_t *val)
{
    *val = ids->pi[position].y - ids->l0[ids->pi[position].b];
}
__device__ __host__ void precinct_in_band_height_of(const ids_t *ids, const uint32_t prec_y_idx, uint32_t band_index, uint32_t *val)
{
    const int is_last_precinct_y = (prec_y_idx < (ids->npy - 1)) ? 0 : 1;
    *val = ids->l1[is_last_precinct_y][band_index] - ids->l0[band_index];
}
__device__ __host__ void precinct_is_line_present(const ids_t *ids, const uint32_t prec_y_idx, uint32_t lvl, uint32_t ypos, uint32_t *val)
{
    precinct_in_band_height_of(ids, prec_y_idx, lvl, val);
    *val = ypos < *val;
}
__device__ __host__ void precinct_gcli_width_of(uint32_t *gclis_sizes, int idx_from_level[MAX_PRECINCT_HEIGHT][MAX_PACKETS], uint32_t column, uint32_t npi, uint32_t band_index, uint32_t *val)
{
    int idx = idx_from_level[0][band_index];
    *val = gclis_sizes[column * npi + idx];
}
__device__ __host__ void precinct_gcli_offset_of(uint32_t *prefix_sum_of_size, int idx_from_level[MAX_PRECINCT_HEIGHT][MAX_PACKETS], uint32_t column, uint32_t npi, uint32_t band_index, uint32_t *val)
{
    int idx = idx_from_level[0][band_index];
    *val = prefix_sum_of_size[column * npi + idx];
}
__device__ void _device_read_n_bits(uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur, uint32_t bit_offset,
                                    uint64_t max_size, uint64_t *val, uint16_t nbits)
{
    bitstream_ptr_cur += bit_offset / MAXB;
    bit_offset %= MAXB;
    uint64_t temp = 0;
    int available0 = MAXB - bit_offset;
    int len0 = (available0 >= nbits) ? nbits : available0;
    int len1 = nbits - len0;
    if (len0)
    {
        temp |= (((*bitstream_ptr_cur) >> (available0 - len0)) << len1);
    }
    if (len1)
    {
        uint64_t cur = 0UL;
        bitstream_ptr_cur++;
        if ((uint64_t)(bitstream_ptr_cur + 1 - bitstream_ptr) * 8 <= max_size)
        {
            cur = *bitstream_ptr_cur;
        }
        else
        {
            for (uint64_t i = 0; i < (max_size % 8); ++i)
            {
                ((uint8_t *)&cur)[i] = ((uint8_t *)(bitstream_ptr_cur))[i];
            }
        }
        bit_offset = 0;
        temp |= ((cur) >> (MAXB - len1));
    }
    if (nbits < 64)
    {
        *val = temp & ((1ULL << nbits) - 1ULL);
    }
}
__device__ void _device_gcli_method_is_enabled(uint32_t enabled, int gcli_method, int precinct_group, int *ret)
{
#define is_run_enabled(run) ((1ULL << (run)) & (enabled_runs))
    const uint32_t enabled_alphabets = (enabled >> METHOD_ENABLE_MASK_ALPHABETS_OFFSET) & ((1UL << ALPHABET_COUNT) - 1);
    const uint32_t enabled_predictions = (enabled >> METHOD_ENABLE_MASK_PREDICTIONS_OFFSET) & ((1UL << PRED_COUNT) - 1);
    const uint32_t enabled_runs = (enabled >> METHOD_ENABLE_MASK_RUNS_OFFSET) & ((1UL << RUN_COUNT) - 1);

    const int alphabet = method_get_alphabet(gcli_method);
    const int pred = method_get_pred(gcli_method);
    const int run = method_get_run(gcli_method);

    if (!((1ULL << alphabet) & enabled_alphabets))
    {
        *ret = 0;
        return;
    }

    if (alphabet != ALPHABET_RAW_4BITS)
    {
        if (!is_run_enabled(run))
        {
            *ret = 0;
            return;
        }

        if (!((1ULL << pred) & enabled_predictions))
        {
            *ret = 0;
            return;
        }

        if (precinct_group == PRECINCT_FIRST_OF_SLICE && pred != PRED_NONE)
        {
            *ret = 0;
            return;
        }
    }
    else
    {
        if (run != RUN_NONE || pred != PRED_NONE)
        {
            *ret = 0;
            return;
        }
    }
    *ret = 1;
    return;
}
__device__ void _device_gcli_method_get_signaling(int gcli_method, uint32_t enabled_methods, int *ret)
{
    int signaling = 0;
    if (gcli_method == method_get_idx(ALPHABET_RAW_4BITS, 0, 0))
    {
        *ret = -1;
        return;
    }
    const int uses_run = ((method_get_run(gcli_method) == RUN_SIGFLAGS_ZRF) || (method_get_run(gcli_method) == RUN_SIGFLAGS_ZRCSF));
    *ret = (uses_run ? 0x2 : 0) | ((method_get_pred(gcli_method) == PRED_VER) ? 0x1 : 0);
    return;
}
__global__ void kernel_read_gcli_sb_methods(gpu_unpacked_info_t *infos, uint64_t *bitstream_ptr,
                                            uint64_t max_size, const int enabled_methods,
                                            const uint32_t bands_count, const uint32_t column_num,
                                            const uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const uint32_t prec_y_idx = tid / (bands_count * column_num);
        const uint32_t column = (tid / bands_count) % column_num;
        const uint32_t band_idx = tid % bands_count;
        uint64_t val;
        gpu_unpacked_info_t info = infos[prec_y_idx * column_num + column];
        uint64_t *bitstream_ptr_cur = bitstream_ptr + info.gcli_sb_methods_bitstream_info.ptr_diff;
        uint32_t bit_offset = info.gcli_sb_methods_bitstream_info.offset;
        _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset + GCLI_METHOD_NBITS * band_idx, max_size, &val, GCLI_METHOD_NBITS);

        for (int gcli_method = 0; gcli_method < GCLI_METHODS_NB; gcli_method++)
        {
            int ret = 0;
            _device_gcli_method_is_enabled(enabled_methods, gcli_method, PRECINCT_ALL, &ret);
            if (ret)
            {
                _device_gcli_method_get_signaling(gcli_method, enabled_methods, &ret);
                if (ret == (int)val)
                {
                    info.gcli_sb_methods[band_idx] = gcli_method;
                    return;
                }
            }
        }
        info.gcli_sb_methods[band_idx] = -1;
        return;
    }
}
__global__ void kernel_read_inclusion_mask(gpu_unpacked_info_t *info_cur, uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur,
                                           uint32_t bit_offset, uint32_t gcli_band_offset,
                                           uint64_t max_size,
                                           const uint32_t significance_group_size,
                                           const uint32_t len)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        uint64_t val = 0;
        _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset + tid, max_size, &val, 1);
        val = (!val) & 0x1;
        tid += gcli_band_offset;
        for (int i = 0; i < significance_group_size; i++)
        {
            info_cur->gpu_inclusion_mask[tid + i] = val;
        }
    }
}

__global__ void kernel_prepare_inclusion_mask(uint32_t *gclis_prefix_sum, uint32_t *gclis_sizes, gpu_unpacked_info_t *info_cur, int idx_from_level[MAX_PRECINCT_HEIGHT][MAX_PACKETS], uint64_t *bitstream_ptr,
                                              uint64_t max_size,
                                              ids_t *ids,
                                              const uint32_t significance_group_size,
                                              uint32_t prec_y_idx, uint32_t column, uint32_t subpkt,
                                              const uint32_t len)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const int idx = info_cur->line_idxs[subpkt].start_idx + tid;

        uint32_t lvl;
        precinct_band_index_of(ids, idx, &lvl);
        uint32_t ypos;
        precinct_ypos_of(ids, idx, &ypos);
        uint32_t is_present;
        precinct_is_line_present(ids, prec_y_idx, lvl, ypos, &is_present);
        if (is_present)
        {
            const int sb_gcli_method = info_cur->gcli_sb_methods[lvl];
            if (method_uses_sig_flags(sb_gcli_method))
            {
                const int block_size = BLOCK_SIZE;
                uint32_t gcli_width;
                // precinct_gcli_width_of(multi_buf_t * gclis_mb, int **idx_from_level, int band_index, int *val)
                precinct_gcli_width_of(gclis_sizes, idx_from_level, column, ids->npi, lvl, &gcli_width);
                uint64_t *bitstream_ptr_cur = info_cur->significance_bitstream_infos->ptr_diff + bitstream_ptr;
                uint32_t bit_offset = info_cur->significance_bitstream_infos->offset;
                uint32_t gcli_band_offset;
                precinct_gcli_offset_of(gclis_prefix_sum, idx_from_level, column, ids->npi, lvl, &gcli_band_offset);
                const int grid_size = (gcli_width / significance_group_size + block_size - 1) / block_size;
                kernel_read_inclusion_mask<<<grid_size, block_size>>>(info_cur, bitstream_ptr, bitstream_ptr_cur, bit_offset,
                                                                      gcli_band_offset, max_size, significance_group_size, gcli_width / significance_group_size);
            }
        }
    }
}

__global__ void kernel_unpack_gclis_significance(uint32_t *gclis_prefix_sum, uint32_t *gclis_sizes, gpu_unpacked_info_t *infos, int idx_from_level[MAX_PRECINCT_HEIGHT][MAX_PACKETS], uint64_t *bitstream_ptr,
                                                 ids_t *ids, const uint32_t significance_group_size, uint32_t nb_subpkts, uint32_t column_num, const uint32_t max_size, uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {

        const uint32_t subpkt = tid % nb_subpkts;
        const uint32_t column = (tid / nb_subpkts) % column_num;
        const uint32_t prec_y_idx = tid / (nb_subpkts * column_num);
        gpu_unpacked_info_t *info_cur = infos + prec_y_idx * column_num + column;
        const uint32_t block_size = BLOCK_SIZE;
        const uint32_t prepare_len = info_cur->line_idxs[subpkt].stop_idx - info_cur->line_idxs[subpkt].start_idx + 1;
        const uint32_t grid_size = (prepare_len + block_size - 1) / block_size; // ctx->xs_config->p.S_s
        kernel_prepare_inclusion_mask<<<grid_size, block_size>>>(gclis_prefix_sum, gclis_sizes, info_cur, idx_from_level, bitstream_ptr, max_size, ids, significance_group_size, prec_y_idx, column, subpkt, prepare_len);
    }
}

void gpu_unpack_gtlis_and_gclis_significance(gpu_xs_dec_context_t *ctx, gpu_unpacked_info_t *infos, const ids_t *ids,
                                             int nb_subpkts, const int n_precs, const int column_num)
{
    cudaStream_t streams[2];
    cudaStreamCreate(streams);
    cudaStreamCreate(streams + 1);
    const uint32_t block_size = BLOCK_SIZE;
    const uint32_t bands_count = ids->nbands * n_precs * column_num;
    uint32_t grid_size = (bands_count + block_size - 1) / block_size;
    kernel_read_gcli_sb_methods<<<grid_size, block_size, 0, streams[0]>>>(infos, ctx->gpu_bitstream_ptr, ctx->max_size, ctx->enabled_methods, ids->nbands, column_num, bands_count);

    // put to different stream
    const uint32_t gtli_tables_len = ctx->level_count * n_precs * column_num;
    grid_size = (gtli_tables_len + block_size - 1) / block_size;
    kernel_compute_gtli_tables<<<grid_size, block_size, 0, streams[1]>>>(infos, ctx->gpu_lvl_gains, ctx->gpu_lvl_priorities, ctx->level_count, column_num, gtli_tables_len);

    cudaDeviceSynchronize();

    const uint32_t significance_len = ctx->level_count * n_precs * column_num;
    grid_size = (significance_len + block_size - 1) / block_size;
    kernel_unpack_gclis_significance<<<grid_size, block_size>>>(ctx->gpu_gclis_prefix_sum, ctx->gpu_gclis_sizes, infos, ctx->idx_from_level, ctx->gpu_bitstream_ptr, ctx->gpu_ids, ctx->xs_config->p.S_s, nb_subpkts, column_num, ctx->max_size, significance_len);
    cudaDeviceSynchronize();
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
}

__global__ void kernel_unpack_raw_gclis(uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur, uint32_t bit_offset, uint64_t max_size, gcli_data_t *gclis, uint32_t gcli_width)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < gcli_width)
    {
        uint64_t val;
        _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset + tid * 4, max_size, &val, 4);
        gclis[tid] = (gcli_data_t)val;
    }
}

__device__ void _device_read_unary_unsigned(uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur, uint32_t bit_offset, uint64_t max_size, int8_t *ret, uint32_t *n_bits)
{
    uint64_t bit = 1;
    int val = -1;
    while (bit)
    {
        _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset++, max_size, &bit, 1);
        ++val;
    }
    *ret = val;
    *n_bits += val + 1;
}

__device__ void _device_read_unary_signed(uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur, uint32_t bit_offset, uint64_t max_size, int8_t *ret, unary_alphabet_t alphabet, uint32_t *n_bits)
{
    int val = -1;
    uint32_t bit_offset_flag = bit_offset;
    switch (alphabet)
    {
    case UNARY_ALPHABET_FULL:
    {
        uint64_t bit = 1;
        do
        {
            _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset++, max_size, &bit, 1);
            val++;
        } while (bit && val < 17);

        if (val == 1)
            val = -1;
        else if (val == 2)
            val = 1;
        else if (val == 3)
            val = -2;
        else if (val == 4)
            val = 2;
        else if (val > 4)
        {
            val -= 2;

            _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset++, max_size, &bit, 1);
            if (bit)
                val = -val;
        }
        *ret = val;
        break;
    }
    case UNARY_ALPHABET_4_CLIPPED:
    {
        uint64_t bit = 1;
        do
        {
            _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset++, max_size, &bit, 1);
            val++;
        } while (bit && val < 15);

        if (val == 1)
            val = -1;
        else if (val == 2)
            val = 1;
        else if (val == 3)
            val = -2;
        else if (val == 4)
            val = 2;

        if (val > 4)
        {
            val -= 2;
            if ((val) && (val != MAX_UNARY - 2))
                _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset++, max_size, &bit, 1);
            if (bit)
                val = -val;
        }
        *ret = val;
        break;
    }
    case UNARY_ALPHABET_0:
    {
        uint64_t bit = 1;
        while (bit && (val < MAX_UNARY))
        {
            _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset++, max_size, &bit, 1);
            val++;
        }
        if (val && (val != MAX_UNARY))
            _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset++, max_size, &bit, 1);
        if (bit)
            val = -val;
        *ret = val;
        break;
    }
    default:
        assert(!"invalid alphabet specified");
        return;
    }
    *n_bits += bit_offset - bit_offset_flag;
}
__device__ void _device_read_bounded_code(uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur, uint32_t bit_offset, uint64_t max_size,
                                          int8_t min_allowed, int8_t max_allowed,
                                          int8_t *val, uint32_t *n_bits)
{
    int8_t tmp;

    const int trigger = abs(min_allowed);

    _device_read_unary_unsigned(bitstream_ptr, bitstream_ptr_cur, bit_offset, max_size, &tmp, n_bits);
    if (tmp > 2 * trigger)
    {
        *val = tmp - trigger;
    }
    else
    {
        *val = (tmp + 1) / 2;
        if (tmp % 2)
        {
            *val = -*val;
        }
    }
}
__device__ void _device_unary_decode2(uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur, uint32_t bit_offset, uint64_t max_size, gcli_pred_t *gcli_pred_buf, uint32_t *inclusion_mask,
                                      int len, int no_sign, int sb_gcli_method, unary_alphabet_t alph, uint32_t *n_bits)
{
    for (int i = 0; i < len; i++)
    {
        if (!method_uses_sig_flags(sb_gcli_method) || inclusion_mask[i])
        {
            if (!no_sign)
            {
                _device_read_unary_signed(bitstream_ptr, bitstream_ptr_cur, bit_offset + *n_bits, max_size, gcli_pred_buf + i, alph, n_bits);
            }
            else
            {
                _device_read_unary_unsigned(bitstream_ptr, bitstream_ptr_cur, bit_offset + *n_bits, max_size, gcli_pred_buf + i, n_bits);
            }
        }
        else
        {
            gcli_pred_buf[i] = 0;
        }
    }
}

__device__ void _device_bounded_decode2(uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur, uint32_t bit_offset, uint64_t max_size,
                                        gcli_pred_t *gcli_pred_buf, uint32_t *inclusion_mask, gcli_data_t *gcli_top_buf,
                                        int sb_gcli_method, int gtli, int gtli_top, int len,
                                        uint32_t *n_bits)
{

    for (int i = 0; i < len; i++)
    {
        if (!method_uses_sig_flags(sb_gcli_method) || inclusion_mask[i])
        {
            int min_value = -20;
            int max_value = 20;
            if (gcli_top_buf)
            {
                int predictor = MAX(gcli_top_buf[i], MAX(gtli, gtli_top));
                min_value = -MAX(predictor - gtli, 0);
                max_value = MAX(MAX_GCLI - MAX(predictor, gtli), 0);
            }

            _device_read_bounded_code(bitstream_ptr, bitstream_ptr_cur, bit_offset, max_size, min_value, max_value, &gcli_pred_buf[i], n_bits);
        }
        else
        {
            gcli_pred_buf[i] = 0;
        }
    }
}
__global__ void kernel_tco_pred_ver_inverse(gcli_data_t *gclis_top, gcli_data_t *gclis, uint32_t *inclusion_mask, const int gtli, const int gtli_top, const int sig_flags_are_zrcsf, uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        int top = MAX(gclis_top[tid], MAX(gtli, gtli_top));
        gclis[tid] = top + gclis[tid];

        if (gclis[tid] <= gtli)
        {
            gclis[tid] -= gtli;
        }
        if (sig_flags_are_zrcsf && !inclusion_mask[tid])
        {
            gclis[tid] = 0;
        }
    }
}

__global__ void kernel_tco_pred_none_inverse(gcli_data_t *gclis, const int gtli, uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        gclis[tid] = (gclis[tid] > 0) ? gclis[tid] + gtli : 0;
    }
}
__global__ void kernel_unpack_gclis(gpu_unpacked_info_t *infos, uint64_t *bitstream_ptr, uint64_t max_size, uint32_t *gclis_prefix_sum, uint32_t *gclis_sizes,
                                    int idx_from_level[MAX_PRECINCT_HEIGHT][MAX_PACKETS],
                                    uint32_t len,
                                    const uint32_t slice_height, const uint32_t column_num,
                                    uint32_t nb_subpkts, const ids_t *gpu_ids)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const uint32_t column = tid % column_num;
        const uint32_t slice_prec_y_idx = slice_height * (tid / column_num); //????
        const uint32_t bands_count = gpu_ids->nbands;
        int *gtli_table_top = NULL;
        gcli_data_t *gclis_top = NULL;
        for (int line_idx = 0; line_idx < slice_height; line_idx += gpu_ids->ph)
        {
            const int is_first_of_slice = !line_idx;
            const int prec_y_idx = ((line_idx + slice_prec_y_idx) / gpu_ids->ph);

            gpu_unpacked_info_t unpack_cur = infos[tid];
            for (int subpkt = 0; subpkt < nb_subpkts; subpkt++)
            {
                const int uses_raw_fallback = unpack_cur.uses_raw_fallback[subpkt];
                const int idx_start = unpack_cur.line_idxs[subpkt].start_idx;
                const int idx_stop = unpack_cur.line_idxs[subpkt].stop_idx;
                uint64_t *bitstream_ptr_cur = bitstream_ptr + unpack_cur.gcli_bitstream_infos[subpkt].ptr_diff;
                uint32_t bit_offset = unpack_cur.gcli_bitstream_infos[subpkt].offset;
                for (int idx = idx_start; idx <= idx_stop; idx++)
                {
                    uint32_t lvl;
                    precinct_band_index_of(gpu_ids, idx, &lvl);
                    uint32_t ypos;
                    precinct_ypos_of(gpu_ids, idx, &ypos);
                    if (ypos == 0 && !is_first_of_slice)
                    {

                        const int is_last_precinct_y = ((prec_y_idx - 1) < (gpu_ids->npy - 1)) ? 0 : 1;
                        const int ylast = gpu_ids->l1[is_last_precinct_y][lvl] - gpu_ids->l0[lvl] - 1;
                        int temp_idx = idx_from_level[ylast][lvl];
                        uint32_t gcli_offset_top;
                        precinct_gcli_offset_of(gclis_prefix_sum, idx_from_level, column, gpu_ids->npi, temp_idx, &gcli_offset_top);
                        gclis_top += gcli_offset_top;
                    }
                    else if (ypos != 0)
                    {
                        const int ylast = ypos - 1;
                        int temp_idx = idx_from_level[ylast][lvl];
                        uint32_t gcli_offset_top;
                        precinct_gcli_offset_of(gclis_prefix_sum, idx_from_level, column, gpu_ids->npi, temp_idx, &gcli_offset_top);
                        gclis_top = unpack_cur.gpu_gcli_buf + gcli_offset_top;
                    }
                    else
                    {
                        gclis_top = NULL;
                    }
                    uint32_t is_present;
                    precinct_is_line_present(gpu_ids, prec_y_idx, lvl, ypos, &is_present);
                    if (is_present)
                    {
                        int sb_gcli_method = uses_raw_fallback ? method_get_idx(ALPHABET_RAW_4BITS, 0, 0) : unpack_cur.gcli_sb_methods[gpu_ids->pi[idx].b];
                        const int gtli = unpack_cur.gtli_table_gcli[lvl];
                        const int gtli_top = (ypos == 0) ? ((gtli_table_top != NULL) ? gtli_table_top[lvl] : gtli) : (gtli);
                        uint32_t gcli_width;
                        precinct_gcli_width_of(gclis_sizes, idx_from_level, column, gpu_ids->npi, lvl, &gcli_width);
                        uint32_t gcli_offset;
                        precinct_gcli_offset_of(gclis_prefix_sum, idx_from_level, column, gpu_ids->npi, lvl, &gcli_offset);
                        gcli_data_t *gclis = unpack_cur.gpu_gcli_buf + gcli_offset;
                        uint32_t *inclusion_mask = unpack_cur.gpu_inclusion_mask + gcli_offset;

                        unary_alphabet_t alph = FIRST_ALPHABET;

                        if (method_is_raw(sb_gcli_method))
                        {
                            const int block_size = BLOCK_SIZE;
                            const int grid_size = (gcli_width + block_size - 1) / block_size;
                            kernel_unpack_raw_gclis<<<grid_size, block_size>>>(bitstream_ptr, bitstream_ptr_cur, bit_offset, max_size, gclis, gcli_width);
                            bit_offset += gcli_width << 2;
                        }
                        else
                        {
                            uint32_t n_bits = 0;
                            int no_prediction = method_uses_no_pred(sb_gcli_method) || ((method_uses_ver_pred(sb_gcli_method) && method_get_alphabet(sb_gcli_method) == ALPHABET_UNARY_UNSIGNED_BOUNDED) && (gclis_top == NULL));
                            int sig_flags_are_zrcsf = (method_get_run(sb_gcli_method) == RUN_SIGFLAGS_ZRCSF);
                            if ((method_get_alphabet(sb_gcli_method) != ALPHABET_UNARY_UNSIGNED_BOUNDED) || no_prediction)
                            {
                                _device_unary_decode2(bitstream_ptr, bitstream_ptr_cur, bit_offset, max_size, gclis, inclusion_mask, gcli_width, no_prediction, sb_gcli_method, alph, &n_bits);
                            }
                            else
                            {
                                _device_bounded_decode2(bitstream_ptr, bitstream_ptr_cur, bit_offset, max_size, gclis, inclusion_mask, gclis_top, sb_gcli_method, gtli, gtli_top, gcli_width, &n_bits);
                            }
                            bit_offset += n_bits;
                            if ((method_uses_ver_pred(sb_gcli_method) && gclis_top))
                            {
                                const int block_size = BLOCK_SIZE;
                                const int grid_size = (gcli_width + block_size - 1) / block_size;
                                kernel_tco_pred_ver_inverse<<<grid_size, block_size>>>(gclis_top, gclis, inclusion_mask, gtli, gtli_top, sig_flags_are_zrcsf, gcli_width);
                            }
                            else if (method_uses_no_pred(sb_gcli_method) || (method_uses_ver_pred(sb_gcli_method) && gclis_top == NULL))
                            {
                                const int block_size = BLOCK_SIZE;
                                const int grid_size = (gcli_width + block_size - 1) / block_size;
                                kernel_tco_pred_none_inverse<<<grid_size, block_size>>>(gclis, gtli, gcli_width);
                            }
                        }
                    }
                }
            }

            gtli_table_top = unpack_cur.gtli_table_gcli;
            gclis_top = unpack_cur.gpu_gcli_buf;
        }
    }
}

void gpu_unpack_gclis(gpu_xs_dec_context_t *ctx, gpu_unpacked_info_t *infos, const uint32_t column_num)
{
    const uint32_t block_size = BLOCK_SIZE;
    const uint32_t slice_height = ctx->xs_config->p.slice_height;
    const uint32_t slice_size = SIGFLAGS_NEXTLVL_SIZE(ctx->ids.h, slice_height) * column_num;
    const uint32_t grid_size = (slice_size + block_size - 1) / block_size;
    kernel_unpack_gclis<<<grid_size, block_size>>>(infos, ctx->gpu_bitstream_ptr, ctx->max_size, ctx->gpu_gclis_prefix_sum, ctx->gpu_gclis_sizes, ctx->idx_from_level, slice_size, slice_height, column_num, ctx->nb_subpkts, ctx->gpu_ids);
}
__global__ void kernel_unpack_data(uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur, uint32_t bit_offset, uint64_t max_size,
                                   gcli_data_t *gclis, uint32_t *inclusion_mask,
                                   const int group_size, const int gtli, const uint8_t sign_packing,
                                   xs_data_in_t *image, uint64_t dst_inc, const int quant_type, const uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        // tid is
        const int i = tid % group_size;
        const int group = tid / group_size;
        const int gcli = gclis[group];
        uint32_t ret = 0;
        if (gcli > gtli)
        {
            uint64_t val;

            bit_offset += inclusion_mask[group - 1] * group_size;

            for (int bp = 0; bp < gcli - gtli; bp++)
            {
                _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset + group_size + i + group_size * bp, max_size, &val, 1);
                ret |= (sig_mag_data_t)((val & 0x01) << (gcli - 1 - bp));
            }
            if (sign_packing == 0)
            {
                _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset + i, max_size, &val, 1);
                ret |= (sig_mag_data_t)val << SIGN_BIT_POSITION;

                // dequant
                if (quant_type == 1)
                {
                    int sign = (ret & SIGN_BIT_MASK);
                    int phi = ret & ~SIGN_BIT_MASK;
                    int zeta = gcli - gtli + 1;
                    int rho = 0;
                    for (rho = 0; phi > 0; phi >>= zeta)
                        rho += phi;
                    ret = sign | rho;
                }
                else if (quant_type == 0)
                {
                    if (gtli > 0 && (ret & ~SIGN_BIT_MASK))
                        ret |= (1 << (gtli - 1));
                }
            }
        }
        image[tid * dst_inc] = ret;
    }
}

__global__ void kernel_unpack_sign(uint64_t *bitstream_ptr, uint64_t *bitstream_ptr_cur, uint32_t bit_offset, uint64_t max_size,
                                   gcli_data_t *gclis, uint32_t *inclusion_mask, const int group_size, const int gtli, const uint8_t sign_packing,
                                   xs_data_in_t *image, uint64_t dst_inc, const int quant_type, const uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        // tid is
        const int i = tid % group_size;
        const int group = tid / group_size;
        const int gcli = gclis[group];
        uint32_t ret = image[tid * dst_inc];
        if (ret)
        {
            uint64_t val;

            bit_offset += inclusion_mask[group - 1] * group_size;

            _device_read_n_bits(bitstream_ptr, bitstream_ptr_cur, bit_offset + i, max_size, &val, 1);
            ret |= (sig_mag_data_t)((val & 0x01) << SIGN_BIT_POSITION);

            // dequant
            if (quant_type == 1)
            {
                int sign = (ret & SIGN_BIT_MASK);
                int phi = ret & ~SIGN_BIT_MASK;
                int zeta = gcli - gtli + 1;
                int rho = 0;
                for (rho = 0; phi > 0; phi >>= zeta)
                    rho += phi;
                ret = sign | rho;
            }
            else if (quant_type == 0)
            {
                if (gtli > 0 && (ret & ~SIGN_BIT_MASK))
                    ret |= (1 << (gtli - 1));
            }
            image[tid * dst_inc] = ret;
        }
    }
}
// __global__ void kernel_prepare_data(bit_unpacker_t *bitstream, sig_mag_data_t *buf, int buf_len, gcli_data_t *gclis, int group_size, int gtli, const uint8_t sign_packing)
// {
//     const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < len)
//     {
//     }
// }

__device__ __host__ void precinct_ptr_for_line_of_band(const ids_t *ids, xs_image_t *image,
                                                       const int band_idx, const int in_band_ypos,
                                                       const uint32_t prec_y_idx, const uint32_t column, const uint32_t is_last_column,
                                                       xs_data_in_t **ptr, uint64_t *x_inc, uint64_t *len)
{

    const int c = ids->band_idx_to_c_and_b[band_idx].c;
    const int b = ids->band_idx_to_c_and_b[band_idx].b;

    // Handle precinct component base.
    xs_data_in_t *the_ptr = image->comps_array[c];
    // Handle start y position of band.
    the_ptr += ids->band_is_high[b].y * ((uint64_t)1 << (ids->band_d[c][b].y - 1)) * ids->comp_w[c];
    // Handle start x position of band.
    the_ptr += ids->band_is_high[b].x * ((uint64_t)1 << (ids->band_d[c][b].x - 1));

    // Handle precinct y index.
    the_ptr += (uint64_t)ids->comp_w[c] * (ids->ph >> (image->sy[c] - 1)) * prec_y_idx;
    // Go to ypos line in precinct.
    the_ptr += (uint64_t)ids->comp_w[c] * in_band_ypos * ((uint64_t)1 << ids->band_d[c][b].y);

    // Handle precinct x index.
    the_ptr += (ids->pw[0] >> (image->sx[c] - 1)) * column;

    *ptr = the_ptr;                            // first sample
    *x_inc = 1ull << ids->band_d[c][b].x;      // increment to next sample
    *len = ids->pwb[is_last_column][band_idx]; // number of samples in line of band in precinct
}

__global__ void kernel_unpack_data_and_sign(uint64_t *bitstream_ptr, const uint64_t max_size, gpu_unpacked_info_t *infos,
                                            uint32_t *gclis_prefix_sum, uint32_t *gclis_sizes, int idx_from_level[MAX_PRECINCT_HEIGHT][MAX_PACKETS],
                                            const uint32_t nb_subpkts, const uint32_t column_num, const int sign_packing,
                                            const uint32_t group_size, const uint32_t len, const int quant_type,
                                            xs_image_t *gpu_image, ids_t *gpu_ids)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const uint32_t subpkt = tid % nb_subpkts;
        const uint32_t column = (tid / nb_subpkts) % column_num;
        const uint32_t prec_y_idx = tid / (nb_subpkts * column_num);
        gpu_unpacked_info_t *info_cur = infos + prec_y_idx * column_num + column;
        uint32_t start_idx = info_cur->line_idxs[subpkt].start_idx;
        uint32_t stop_idx = info_cur->line_idxs[subpkt].stop_idx;
        uint32_t gcli_width;
        uint32_t gcli_offset;
        uint64_t *bitstream_ptr_cur = info_cur->data_bitstream_infos[subpkt].ptr_diff + bitstream_ptr;
        uint32_t bit_offset = info_cur->data_bitstream_infos[subpkt].offset;

        for (int idx = start_idx; idx <= stop_idx; idx++)
        {
            uint32_t lvl;
            precinct_band_index_of(gpu_ids, idx, &lvl);
            uint32_t ypos;
            precinct_ypos_of(gpu_ids, idx, &ypos);
            uint32_t is_present;
            precinct_is_line_present(gpu_ids, prec_y_idx, lvl, ypos, &is_present);
            if (is_present)
            {
                const int gtli = info_cur->gtli_table_gcli[lvl];

                precinct_gcli_width_of(gclis_sizes, idx_from_level, column, gpu_ids->npi, lvl, &gcli_width);
                precinct_gcli_offset_of(gclis_prefix_sum, idx_from_level, column, gpu_ids->npi, lvl, &gcli_offset);

                gcli_data_t *gclis = gcli_width + info_cur->gpu_gcli_buf;
                uint8_t offset_width = (sign_packing == 0) ? group_size : 0;
                uint32_t *inclusion_mask = info_cur->gpu_inclusion_mask + gcli_offset;
                inclusion_mask[0] = (gclis[0] > gtli) ? (gclis[0] - gtli + offset_width) : 0;

                for (int group = 1; group < gcli_width; group++)
                {
                    inclusion_mask[group] = inclusion_mask[group - 1];
                    if (gclis[group] > gtli)
                    {
                        inclusion_mask[group] += gclis[group] - gtli + offset_width;
                    }
                }
                uint32_t bits_sum = inclusion_mask[gcli_width - 1];
                const int coff_width = gcli_width * group_size;
                xs_data_in_t *dst;
                uint64_t dst_inc, dst_len;
                uint32_t is_last_column = (column == gpu_ids->npx - 1) ? 1 : 0;
                precinct_ptr_for_line_of_band(gpu_ids, gpu_image, lvl, ypos, prec_y_idx, column, is_last_column, &dst, &dst_inc, &dst_len);
                int block_size = BLOCK_SIZE;
                int grid_size = (coff_width + block_size - 1) / block_size;
                kernel_unpack_data<<<grid_size, block_size>>>(bitstream_ptr, bitstream_ptr_cur, bit_offset, max_size, gclis, inclusion_mask, group_size, gtli, sign_packing, dst, dst_inc, quant_type, coff_width);

                if (sign_packing)
                {
                    inclusion_mask = (uint32_t *)malloc(sizeof(uint32_t) * coff_width);
                    inclusion_mask[0] = (dst[0] != 0) ? 1 : 0;

                    for (int group = 1; group < coff_width; group++)
                    {
                        inclusion_mask[group] = inclusion_mask[group - 1];
                        if (dst[group * dst_inc] != 0)
                        {
                            inclusion_mask[group]++;
                        }
                    }
                    grid_size = (coff_width + block_size - 1) / block_size;
                    kernel_unpack_sign<<<grid_size, block_size>>>(bitstream_ptr, bitstream_ptr_cur, bit_offset, max_size, gclis, inclusion_mask, group_size, gtli, sign_packing, dst, dst_inc, quant_type, coff_width);
                    free(inclusion_mask);
                }
                bit_offset += bits_sum;
            }
        }
    }
}
void gpu_unpack_data_and_sign(gpu_xs_dec_context_t *ctx, gpu_unpacked_info_t *infos, xs_image_t *gpu_image, const uint32_t column_num)
{
    const uint32_t block_size = BLOCK_SIZE;
    const uint32_t n_precs = SIGFLAGS_NEXTLVL_SIZE(ctx->ids.h, ctx->ids.ph) * column_num;
    const uint32_t grid_size = (n_precs + block_size - 1) / block_size;
    kernel_unpack_data_and_sign<<<grid_size, block_size>>>(ctx->gpu_bitstream_ptr, ctx->max_size, infos,
                                                           ctx->gpu_gclis_prefix_sum, ctx->gpu_gclis_sizes, ctx->idx_from_level,
                                                           ctx->nb_subpkts, column_num, ctx->use_sign_subpkt,
                                                           ctx->group_size, n_precs, ctx->quant_type,
                                                           gpu_image, ctx->gpu_ids);
}
void unpack_bit_offset_info(gpu_xs_dec_context_t *ctx, bit_unpacker_t *bitstream, gpu_unpacked_info_t *info, gpu_unpacked_info_t *gpu_info,
                            uint32_t prec_y_idx, uint32_t column, int extra_bits_before_precinct)
{
    uint64_t val;
    int empty;
    int len_before_subpkt = 0;
    int gcli_sb_methods[MAX_NBANDS];
    int subpkt = 0;
    int subpkt_len;

    const uint32_t position_count = ctx->ids.npi;
    const bool use_long_precinct_headers = ctx->ids.use_long_precinct_headers;
    const int bitpos_prc_start = (int)bitunpacker_consumed_bits(bitstream);

    if (ctx->xs_config->verbose > 2)
    {
        fprintf(stderr, "(bitpos=%d) Precinct (bytepos=%d)\n", bitpos_prc_start, bitpos_prc_start >> 3);
    }
    cudaDeviceSynchronize();
    if (!ctx->is_init)
    {
        cudaMalloc((void **)&(info->gpu_inclusion_mask), sizeof(uint32_t) * 1);
        cudaMalloc((void **)&(info->gpu_gcli_buf), sizeof(gcli_type_t) * 1);
    }
    // Start of precinct.
    bitunpacker_read(bitstream, &val, PREC_HDR_PREC_SIZE);
    const int Lprc = info->Lprc = ((int)val << 3);
    bitunpacker_read(bitstream, &val, PREC_HDR_QUANTIZATION_SIZE);
    const int quantization = info->quantization = (int)val;
    bitunpacker_read(bitstream, &val, PREC_HDR_REFINEMENT_SIZE);
    const int refinement = info->refinement = (int)val;

    bitunpacker_set_info(bitstream, &(info->gcli_sb_methods_bitstream_info));
    const int bands_count = ctx->level_count;
    bitunpacker_skip(bitstream, bands_count * GCLI_METHOD_NBITS);
    // for (int band = 0; band < bands_count; ++band)
    // {
    //     bitunpacker_read(bitstream, &val, GCLI_METHOD_NBITS);
    //     gcli_sb_methods[band] = gcli_method_from_signaling((int)val, ctx->enabled_methods);
    // }
    bitunpacker_align(bitstream, PREC_HDR_ALIGNMENT);
    const int bitpos_at_prc_data = (int)bitunpacker_consumed_bits(bitstream);
#ifdef PACKING_GENERATE_FRAGMENT_CODE
    // Add precinct header bits.
    extra_bits_before_precinct += bitpos_at_prc_data - bitpos_prc_start;
#endif

    if (ctx->xs_config->verbose > 3)
    {
        fprintf(stderr, "(bitpos=%d) precinct header read (prec_len=%d quant=(%d,%d)\n", bitpos_at_prc_data, Lprc, quantization, refinement);
    }
    // compute gtli
    // gpu_compute_gtli_tables(quantization, refinement, ctx->level_count, ctx->xs_config->p.lvl_gains, ctx->xs_config->p.lvl_priorities, ctx->gtli_table_data, ctx->gtli_table_gcli, &empty);

    for (int idx_start = 0, idx_stop = 0; idx_stop < position_count; idx_stop++)
    {
        if ((idx_stop != (position_count - 1)) && (precinct_subpkt_of(&ctx->ids, idx_stop) == precinct_subpkt_of(&ctx->ids, idx_stop + 1)))
        {
            continue;
        }
        uint32_t lvl;
        precinct_band_index_of(&ctx->ids, idx_start, &lvl);
        uint32_t ypos;
        precinct_ypos_of(&ctx->ids, idx_start, &ypos);
        uint32_t is_present;
        precinct_is_line_present(&ctx->ids, prec_y_idx, lvl, ypos, &is_present);
        if (!is_present)
        {
            ++subpkt;
            idx_start = idx_stop + 1;
            continue;
        }

#ifdef PACKING_GENERATE_FRAGMENT_CODE
        const int bitpos_packet_start = (int)bitunpacker_consumed_bits(bitstream);
#endif

        // Start of packet.
        bitunpacker_read(bitstream, &val, 1);
        info->uses_raw_fallback[subpkt] = (int)val & 0x1;

        bitunpacker_read(bitstream, &val, use_long_precinct_headers ? PKT_HDR_DATA_SIZE_LONG : PKT_HDR_DATA_SIZE_SHORT);
        info->data_len[subpkt] = (int)val;

        bitunpacker_read(bitstream, &val, use_long_precinct_headers ? PKT_HDR_GCLI_SIZE_LONG : PKT_HDR_GCLI_SIZE_SHORT);
        info->gcli_len[subpkt] = (int)val;

        bitunpacker_read(bitstream, &val, use_long_precinct_headers ? PKT_HDR_SIGN_SIZE_LONG : PKT_HDR_SIGN_SIZE_SHORT);
        info->sign_len[subpkt] = (int)val;

        bitunpacker_align(bitstream, PKT_HDR_ALIGNMENT);

        if (ctx->xs_config->verbose > 2)
        {
            const int bitpos = (int)bitunpacker_consumed_bits(bitstream);
            fprintf(stderr, "(bitpos=%d) Subpacket DATALEN=%d GCLILEN=%d SIGNLEN=%d (force_raw%d)\n", bitpos,
                    info->data_len[subpkt],
                    info->gcli_len[subpkt],
                    info->sign_len[subpkt],
                    info->uses_raw_fallback[subpkt]);
        }

        // get significance group bitstreamm info
        bitunpacker_set_info(bitstream, info->significance_bitstream_infos + subpkt);
        if (!info->uses_raw_fallback[subpkt])
        {
            int skip_bits = 0;
            for (int idx = idx_start; idx <= idx_stop; idx++)
            {
                uint32_t lvl;
                precinct_band_index_of(&ctx->ids, idx, &lvl);
                uint32_t ypos;
                precinct_ypos_of(&ctx->ids, idx, &ypos);
                uint32_t is_present;
                precinct_is_line_present(&ctx->ids, prec_y_idx, lvl, ypos, &is_present);
                if (is_present)
                {
                    uint32_t gcli_width;
                    precinct_gcli_width_of(ctx->gclis_sizes, ctx->idx_from_level, column, ctx->ids.npi, lvl, &gcli_width);
                    const int significance_group_size = ctx->xs_config->p.S_s;
                    skip_bits += SIGFLAGS_NEXTLVL_SIZE(gcli_width, significance_group_size);
                }
            }
            bitunpacker_skip(bitstream, (skip_bits));
        }
        bitunpacker_align(bitstream, SUBPKT_ALIGNMENT);

        // get gcli bitstreamm info
        len_before_subpkt = (int)bitunpacker_consumed_bits(bitstream);
        bitunpacker_set_info(bitstream, info->gcli_bitstream_infos + subpkt);
        bitunpacker_skip(bitstream, (info->gcli_len[subpkt] << 3));
        bitunpacker_align(bitstream, SUBPKT_ALIGNMENT);

        // get data bitstreamm info
        bitunpacker_set_info(bitstream, info->data_bitstream_infos + subpkt);
        bitunpacker_skip(bitstream, (info->data_len[subpkt] << 3));
        bitunpacker_align(bitstream, SUBPKT_ALIGNMENT);

        // get sign bitstreamm info
        bitunpacker_set_info(bitstream, info->sign_bitstream_infos + subpkt);
        bitunpacker_skip(bitstream, (info->sign_len[subpkt] << 3));
        bitunpacker_align(bitstream, SUBPKT_ALIGNMENT);

#ifdef PACKING_GENERATE_FRAGMENT_CODE
        if (ctx->fragment_info_cb != NULL)
        {
            int n_gclis = 0;
            for (int idx = idx_start; idx <= idx_stop; idx++)
            {
                uint32_t val;
                precinct_band_index_of(&ctx->ids, idx, &val);
                precinct_gcli_width_of(ctx->gclis_sizes, ctx->idx_from_level, column, ctx->ids.npi, lvl, &val);
                n_gclis += val;
            }

            const int bitpos_packet_end = (int)bitunpacker_consumed_bits(bitstream);
            // account for EOC if really last fragment (the decoder will verify the actual EOC being present)
            const int _fragment_size = bitpos_packet_end - bitpos_packet_start + extra_bits_before_precinct;
            extra_bits_before_precinct = 0;
            ctx->fragment_info_cb(ctx->fragment_info_context, ctx->fragment_cnt, _fragment_size, n_gclis, 0);
            ++ctx->fragment_cnt;
        }
#endif

        ++subpkt;
        idx_start = idx_stop + 1;
    }
    ctx->nb_subpkts = subpkt;
    cudaMemcpyAsync(gpu_info, info, sizeof(gpu_unpacked_info_t), cudaMemcpyHostToDevice);
    const int padding_len = Lprc - ((int)bitunpacker_consumed_bits(bitstream) - bitpos_at_prc_data);
    assert(padding_len >= 0);
#ifdef PACKING_GENERATE_FRAGMENT_CODE
    if (ctx->fragment_info_cb != NULL && padding_len > 0)
    {
        // Late contribution (the padding), so refer to last fragment ID (value -1).
        ctx->fragment_info_cb(ctx->fragment_info_context, -1, 0, 0, padding_len);
    }
#endif
    bitunpacker_skip(bitstream, padding_len);
}

gpu_xs_dec_context_t *gpu_xs_dec_init(xs_config_t *xs_config, xs_image_t *image)
{
    gpu_xs_dec_context_t *ctx;

    ctx = (gpu_xs_dec_context_t *)malloc(sizeof(gpu_xs_dec_context_t));
    if (!ctx)
    {
        return NULL;
    }

    memset(ctx, 0, sizeof(gpu_xs_dec_context_t));
    ctx->xs_config = xs_config;

    assert(xs_config != NULL);
    if (!xs_config_validate(xs_config, image))
    {
        free(ctx);
        return NULL;
    }
    ctx->quant_type = xs_config->p.Qpih;
    ctx->group_size = xs_config->p.N_g;
    ids_construct(&ctx->ids, image, xs_config->p.NLx, xs_config->p.NLy, xs_config->p.Sd, xs_config->p.Cw, xs_config->p.Lh);
    cudaMalloc((void **)&ctx->gpu_ids, sizeof(ids_t));
    cudaMemcpyAsync(ctx->gpu_ids, &ctx->ids, sizeof(ids_t), cudaMemcpyHostToDevice);
    ctx->level_count = ctx->ids.nbands;
    ctx->gclis_sizes = (uint32_t *)malloc(sizeof(uint32_t) * (ctx->ids.npx * ctx->ids.npi));
    ctx->gclis_prefix_sum = (uint32_t *)malloc(sizeof(uint32_t) * (ctx->ids.npx * ctx->ids.npi));
    cudaMalloc((void **)&ctx->gpu_gclis_sizes, sizeof(uint32_t) * (ctx->ids.npx * ctx->ids.npi));
    cudaMalloc((void **)&ctx->gpu_gclis_prefix_sum, sizeof(uint32_t) * (ctx->ids.npx * ctx->ids.npi));
    for (int column = 0; column < ctx->ids.npx; column++)
    {

        int is_last_column = (column == ctx->ids.npx - 1) ? 1 : 0;
        for (int idx = 0; idx < ctx->ids.npi; ++idx)
        {
            const int band = ctx->ids.pi[idx].b;
            const int y = ctx->ids.pi[idx].y - ctx->ids.l0[band]; // relative Y in band

            ctx->idx_from_level[y][band] = idx;

            const int N_cg = (ctx->ids.pwb[is_last_column][band] + ctx->group_size - 1) / ctx->group_size;
            ctx->gclis_sizes[column * ctx->ids.npi + idx] = N_cg;
            ctx->gclis_prefix_sum[column * ctx->ids.npi + idx] = idx ? (ctx->gclis_sizes[column * ctx->ids.npi + idx - 1] + ctx->gclis_prefix_sum[column * ctx->ids.npi + idx - 1]) : 0;
        }
    }
    cudaMemcpyAsync(ctx->gpu_gclis_sizes, ctx->gclis_sizes, sizeof(uint32_t) * (ctx->ids.npx * ctx->ids.npi), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(ctx->gpu_gclis_prefix_sum, ctx->gclis_prefix_sum, sizeof(uint32_t) * (ctx->ids.npx * ctx->ids.npi), cudaMemcpyHostToDevice);
    ctx->bitstream = bitunpacker_init();
    return ctx;
}

void gpu_unpack_bitstream_infos(gpu_xs_dec_context_t *ctx, void *bitstream_buf, uint64_t bitstream_buf_size, xs_image_t *image_out)
{
    int slice_idx = 0;
    uint64_t bitstream_pos = 0;
    static gpu_unpacked_info_t *gpu_infos = NULL;
    static gpu_unpacked_info_t *cpu_info = NULL;
    if (!ctx->is_init)
    {
        if (gpu_infos == NULL)
        {
            cudaMalloc((void **)&gpu_infos, sizeof(gpu_unpacked_info_t) * ((ctx->ids.npx) * ((ctx->ids.h + ctx->ids.ph - 1) / ctx->ids.ph) + 10));
        }
        if (cpu_info == NULL)
        {
            cpu_info = (gpu_unpacked_info_t *)malloc(sizeof(gpu_unpacked_info_t));
        }
        if (ctx->gpu_bitstream_ptr == NULL)
        {
            cudaMalloc((void **)&ctx->gpu_bitstream_ptr, (bitstream_buf_size + 8) & (~0x7));
            ctx->max_size = bitstream_buf_size;
        }
    }
    cudaMemcpy(ctx->gpu_bitstream_ptr, bitstream_buf, (bitstream_buf_size + 8) & (~0x7), cudaMemcpyHostToDevice);
    // memset(cpu_info, 0, sizeof(gpu_unpacked_info_t));
    gpu_convert_ipx_htobe64(ctx->gpu_bitstream_ptr, (((bitstream_buf_size + 8) & (~0x7)) / sizeof(uint64_t)));
    bitunpacker_set_buffer(ctx->bitstream, bitstream_buf, bitstream_buf_size);

    xs_parse_head(ctx->bitstream, NULL, NULL);

#ifdef PACKING_GENERATE_FRAGMENT_CODE
    memset(&ctx->fragment_info_buf, 0, sizeof(xs_buffering_fragment_t));
    ctx->fragment_info_buf.id = -1;
#endif
    ids_t *ids = &ctx->ids;

    uint32_t prec_y_idx;
    for (int line_idx = 0; line_idx < ids->h; line_idx += ids->ph)
    {
        prec_y_idx = (line_idx / ids->ph);
        for (int column = 0; column < ids->npx; column++)
        {
            const int is_first_of_slice = prec_y_idx_is_first_of_slice(ids, prec_y_idx, ctx->xs_config->p.slice_height);
            if (is_first_of_slice && column == 0)
            {
                int slice_idx_check;
                xs_parse_slice_header(ctx->bitstream, &slice_idx_check);
                assert(slice_idx_check == (slice_idx++));
                if (ctx->xs_config->verbose > 1)
                {
                    fprintf(stderr, "Read Slice Header (slice_idx=%d)\n", slice_idx_check);
                }
            }
#ifdef PACKING_GENERATE_FRAGMENT_CODE
            const int extra_bits_before_precinct = (int)(bitunpacker_consumed_bits(ctx->bitstream) - bitstream_pos);
#else
            const int extra_bits_before_precinct = 0;
#endif
            unpack_bit_offset_info(ctx, ctx->bitstream, cpu_info, gpu_infos, prec_y_idx, column, extra_bits_before_precinct);
        }
    }
    const int n_precs = prec_y_idx;
    gpu_unpack_gtlis_and_gclis_significance(ctx, gpu_infos, &ctx->ids, ctx->nb_subpkts, n_precs, ids->npx);

    gpu_unpack_gclis(ctx, gpu_infos, ctx->ids.npx);

    ctx->is_init = 1;
}
