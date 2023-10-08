#include "dwt.cuh"
#include <assert.h>
#include <cuda_runtime.h>
#include "libjxs.h"
#include "ids.h"
typedef enum dwt_mode_t
{
    VERTICAL,
    HORIZONTAL,
} dwt_mode_t;

__global__ void kernel_dwt_inverse_low_pass(xs_data_in_t *const base, xs_data_in_t *const end, const ptrdiff_t x_inc, const ptrdiff_t y_inc,
                                            const int column_num, const dwt_mode_t mode,
                                            const int bound, const uint32_t len)
{

    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const int column = tid % column_num;
        const int raw = tid / column_num;
        xs_data_in_t *p = base;
        if (mode == VERTICAL)
        {

            if (raw == 0)
            {
                p += x_inc * column;
                if (p < end)
                {
                    *p -= (*(p + y_inc) + 1) >> 1;
                }
            }
            else
            {
                p += (y_inc << 1) * raw;
                if (p - base < bound - y_inc)
                {
                    p += x_inc * column;
                    if (p < end)
                    {
                        *p -= (*(p - y_inc) + *(p + y_inc) + 2) >> 2;
                    }
                }
                else if (p - base < bound)
                {
                    p += x_inc * column;
                    if (p < end)
                    {
                        *p -= (*(p - y_inc) + 1) >> 1;
                    }
                }
            }
        }
        else if (mode == HORIZONTAL)
        {

            if (column == 0)
            {
                p += y_inc * raw;
                if (p < end)
                {
                    *p -= (*(p + x_inc) + 1) >> 1;
                }
            }
            else
            {
                p += (x_inc << 1) * column;
                if (p - base < bound - x_inc)
                {
                    p += y_inc * raw;
                    if (p < end)
                    {
                        *p -= (*(p - x_inc) + *(p + x_inc) + 2) >> 2;
                    }
                }
                else if (p - base < bound)
                {
                    p += y_inc * raw;
                    if (p < end)
                    {
                        *p -= (*(p - x_inc) + 1) >> 1;
                    }
                }
            }
        }
    }
}

/*


 dwt的边界还没有计算
*/
__global__ void kernel_dwt_inverse_high_pass(xs_data_in_t *const base, xs_data_in_t *const end, const ptrdiff_t x_inc, const ptrdiff_t y_inc,
                                             const int column_num, const dwt_mode_t mode,
                                             const int bound, const uint32_t len)
{

    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const int column = tid % column_num;
        const int raw = tid / column_num;

        if (mode == VERTICAL)
        {
            xs_data_in_t *p = base + y_inc;
            p += (y_inc << 1) * raw;
            if (p - base < bound - y_inc)
            {
                p += x_inc * column;
                if (p < end)
                {
                    *p += (*(p - y_inc) + *(p + y_inc)) >> 1;
                }
            }
            else if (p - base < bound)
            {
                p += x_inc * column;
                if (p < end)
                {
                    *p += *(p - y_inc);
                }
            }
        }
        else if (mode == HORIZONTAL)
        {
            xs_data_in_t *p = base + x_inc;
            p += (x_inc << 1) * column;
            if (p - base < bound - x_inc)
            {
                p += y_inc * raw;
                if (p < end)
                {
                    *p += (*(p - x_inc) + *(p + x_inc)) >> 1;
                }
            }
            else if (p - base < bound)
            {
                p += y_inc * raw;
                if (p < end)
                {
                    *p += *(p - x_inc);
                }
            }
        }
    }
}

void gpu_dwt_inverse_horizontal(const ids_t *ids, xs_data_in_t **gpu_comps_array, const int k, const int h_level, const int v_level, cudaStream_t *stream)
{

    const ptrdiff_t x_inc = (ptrdiff_t)1 << h_level;
    const ptrdiff_t y_inc = (ptrdiff_t)ids->comp_w[k] << v_level;
    xs_data_in_t *base = gpu_comps_array[k];
    xs_data_in_t *const end = base + (size_t)ids->comp_w[k] * (size_t)ids->comp_h[k];
    const int column_num = ((ids->comp_w[k] + x_inc - 1) / x_inc);
    const int raw_num = (((size_t)ids->comp_w[k] * (size_t)ids->comp_h[k] + y_inc - 1) / y_inc);
    const int threads_num = raw_num * column_num;
    int block_size = BLOCK_SIZE;
    const int grid_size = (threads_num + block_size - 1) / block_size;
    kernel_dwt_inverse_low_pass<<<grid_size, block_size, 0, stream[k]>>>(base, end, x_inc, y_inc, column_num, HORIZONTAL, ids->comp_w[k], threads_num);
    kernel_dwt_inverse_high_pass<<<grid_size, block_size, 0, stream[k]>>>(base, end, x_inc, y_inc, column_num, HORIZONTAL, ids->comp_w[k], threads_num);
}
void gpu_dwt_inverse_vertical_(const ids_t *ids, xs_data_in_t **gpu_comps_array, const int k, const int h_level, const int v_level, cudaStream_t *stream)
{

    const ptrdiff_t x_inc = (ptrdiff_t)1 << h_level;
    const ptrdiff_t y_inc = (ptrdiff_t)ids->comp_w[k] << v_level;
    xs_data_in_t *base = gpu_comps_array[k];
    xs_data_in_t *const end = base + (size_t)ids->comp_w[k] * (size_t)ids->comp_h[k];
    const int column_num = ((ids->comp_w[k] + x_inc - 1) / x_inc);
    const int raw_num = (((size_t)ids->comp_w[k] * (size_t)ids->comp_h[k] + y_inc - 1) / y_inc);
    const int threads_num = raw_num * column_num;
    int block_size = BLOCK_SIZE;
    const int grid_size = (threads_num + block_size - 1) / block_size;
    kernel_dwt_inverse_low_pass<<<grid_size, block_size, 0, stream[k]>>>(base, end, x_inc, y_inc, column_num, VERTICAL, (size_t)ids->comp_w[k] * (size_t)ids->comp_h[k], threads_num);
    kernel_dwt_inverse_high_pass<<<grid_size, block_size, 0, stream[k]>>>(base, end, x_inc, y_inc, column_num, VERTICAL, (size_t)ids->comp_w[k] * (size_t)ids->comp_h[k], threads_num);
}

void gpu_dwt_inverse_transform(const ids_t *ids, xs_data_in_t **gpu_comps_array)
{
    cudaStream_t streams[N_STREAM];
    for (int i = 0; i < N_STREAM; i++)
    {
        cudaStreamCreate(streams + i);
    }
    for (int k = 0; k < ids->ncomps - ids->sd; ++k)
    {
        assert(ids->nlxyp[k].y <= ids->nlxyp[k].x);

        for (int d = ids->nlxyp[k].x - 1; d >= ids->nlxyp[k].y; --d)
        {
            gpu_dwt_inverse_horizontal(ids, gpu_comps_array, k, d, ids->nlxyp[k].y, streams);
        }

        for (int d = ids->nlxyp[k].y - 1; d >= 0; --d)
        {
            gpu_dwt_inverse_horizontal(ids, gpu_comps_array, k, d, d, streams);
            gpu_dwt_inverse_vertical_(ids, gpu_comps_array, k, d, d, streams);
        }
    }
    for (int i = 0; i < N_STREAM; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
}
