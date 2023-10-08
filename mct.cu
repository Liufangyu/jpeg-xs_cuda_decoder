#include "mct.cuh"

#include "common.h"
#include <assert.h>
#include <malloc.h>
#include <stdio.h>

__device__ __host__ void swap_ptr(xs_data_in_t **p1, xs_data_in_t **p2)
{
    xs_data_in_t *tmp = *p1;
    *p1 = *p2;
    *p2 = tmp;
}

__global__ void kernel_mct_inverse_rct(xs_image_t *gpu_image, uint32_t len)
{

    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        xs_data_in_t *c0 = gpu_image->comps_array[0] + tid;
        xs_data_in_t *c1 = gpu_image->comps_array[1] + tid;
        xs_data_in_t *c2 = gpu_image->comps_array[2] + tid;
        const xs_data_in_t tmp = *c0 - ((*c1 + *c2) >> 2);
        *c0 = tmp + *c2;
        *c2 = tmp + *c1;
        *c1 = tmp;
        ++c0;
        ++c1;
        ++c2;
    }
}
void gpu_mct_inverse_rct(xs_image_t *image, xs_image_t *gpu_image)
{
    const uint32_t len = image->width * image->height;

    const int block_size = BLOCK_SIZE;
    const int grid_size = (len + block_size - 1) / block_size;
    kernel_mct_inverse_rct<<<grid_size, block_size>>>(gpu_image, len);
}

__device__ __host__ void mct_tetrix_access(xs_image_t *im, const int c, const int Cf, const int Ct, const int rx, const int ry, const int x, const int y, xs_data_in_t *ret)
{
    // Stupid magic.
    assert(c >= 0 && c <= 3);
    assert(Cf == 0 || Cf == 3);
    assert(Ct == 0 || Ct == 1);
    assert(rx >= -1 && rx <= 1);
    assert(ry >= -1 && ry <= 1);
    int t_x = rx + ((Ct + c) & 1);
    int t_y = ry + (((~(c)) >> 1) & 1);
    assert(t_x >= -1 && t_x <= 2);
    assert(t_y >= -1 && t_y <= 2);
    const int k = ((((~(t_y)) << 1) & 2) | (((Ct) ^ (t_x)) & 1));
    assert(k >= 0 && k <= 3);
    if (Cf == 3)
    {
        t_y &= 1;
    }
    t_x += x << 1;
    t_y += y << 1;
    if (t_x < 0)
    {
        t_x += 2;
    }
    else if (t_x >= (im->width << 1))
    {
        t_x -= 2;
    }
    if (t_y < 0)
    {
        t_y += 2;
    }
    else if (t_y >= (im->height << 1))
    {
        t_y -= 2;
    }
    t_x >>= 1;
    t_y >>= 1;
    assert(t_x >= 0 && t_x < im->width);
    assert(t_y >= 0 && t_y < im->height);
    // assert(mct_tetrix_access_slow(im, c, Cf, Ct, rx, ry, x, y) == (im->comps_array[k] + t_y * im->width + t_x));
    *ret = im->comps_array[k][(size_t)t_y * (size_t)im->width + t_x];
}
__global__ void kernel_inverse_average(xs_image_t *gpu_image, int Cf, int Ct, int width, uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const int x = tid % width;
        const int y = tid / width;
        xs_data_in_t dtl, dtr, dbl, dbr;
        mct_tetrix_access(gpu_image, 0, Cf, Ct, -1, -1, x, y, &dtl);
        mct_tetrix_access(gpu_image, 0, Cf, Ct, 1, -1, x, y, &dtr);
        mct_tetrix_access(gpu_image, 0, Cf, Ct, -1, 1, x, y, &dbl);
        mct_tetrix_access(gpu_image, 0, Cf, Ct, 1, 1, x, y, &dbr);
        gpu_image->comps_array[0][tid] -= (dtl + dtr + dbl + dbr) >> 3;
    }
}
__global__ void kernel_inverse_delta(xs_image_t *gpu_image, int Cf, int Ct, int width, uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const int x = tid % width;
        const int y = tid / width;
        xs_data_in_t ytl, ytr, ybl, ybr;
        mct_tetrix_access(gpu_image, 3, Cf, Ct, -1, -1, x, y, &ytl);
        mct_tetrix_access(gpu_image, 3, Cf, Ct, 1, -1, x, y, &ytr);
        mct_tetrix_access(gpu_image, 3, Cf, Ct, -1, 1, x, y, &ybl);
        mct_tetrix_access(gpu_image, 3, Cf, Ct, 1, 1, x, y, &ybr);
        gpu_image->comps_array[3][tid] += (ytl + ytr + ybl + ybr) >> 2;
    }
}

__global__ void kernel_inverse_Y(xs_image_t *gpu_image, int Cf, int Ct, int e1, int e2, int width, uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const int x = tid % width;
        const int y = tid / width;

        xs_data_in_t bl, br, rt, rb;
        mct_tetrix_access(gpu_image, 0, Cf, Ct, -1, 0, x, y, &bl);
        mct_tetrix_access(gpu_image, 0, Cf, Ct, 1, 0, x, y, &br);
        mct_tetrix_access(gpu_image, 0, Cf, Ct, 0, -1, x, y, &rt);
        mct_tetrix_access(gpu_image, 0, Cf, Ct, 0, 1, x, y, &rb);
        gpu_image->comps_array[0][tid] -= (((bl + br) << e2) + ((rt + rb) << e1)) >> 3;

        xs_data_in_t bt, bb, rl, rr;

        mct_tetrix_access(gpu_image, 3, Cf, Ct, 0, -1, x, y, &bt);
        mct_tetrix_access(gpu_image, 3, Cf, Ct, 0, 1, x, y, &bb);
        mct_tetrix_access(gpu_image, 3, Cf, Ct, -1, 0, x, y, &rl);
        mct_tetrix_access(gpu_image, 3, Cf, Ct, 1, 0, x, y, &rr);
        gpu_image->comps_array[3][tid] -= (((bt + bb) << e2) + ((rl + rr) << e1)) >> 3;
    }
}

__global__ void kernel_inverse_CbCr(xs_image_t *gpu_image, int Cf, int Ct, int e1, int e2, int width, uint32_t len)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len)
    {
        const int x = tid % width;
        const int y = tid / width;

        xs_data_in_t gl, gr, gt, gb;
        mct_tetrix_access(gpu_image, 1, Cf, Ct, -1, 0, x, y, &gl);
        mct_tetrix_access(gpu_image, 1, Cf, Ct, 1, 0, x, y, &gr);
        mct_tetrix_access(gpu_image, 1, Cf, Ct, 0, -1, x, y, &gt);
        mct_tetrix_access(gpu_image, 1, Cf, Ct, 0, 1, x, y, &gb);

        gpu_image->comps_array[1][tid] += (gl + gr + gt + gb) >> 2;

        mct_tetrix_access(gpu_image, 2, Cf, Ct, 0, -1, x, y, &gl);
        mct_tetrix_access(gpu_image, 2, Cf, Ct, 0, 1, x, y, &gr);
        mct_tetrix_access(gpu_image, 2, Cf, Ct, -1, 0, x, y, &gt);
        mct_tetrix_access(gpu_image, 2, Cf, Ct, 1, 0, x, y, &gb);
        gpu_image->comps_array[2][tid] += (gl + gr + gt + gb) >> 2;
    }
}

__global__ void gpu_swap_image_ptr(xs_image_t *gpu_image)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0)
    {
        swap_ptr(&gpu_image->comps_array[0], &gpu_image->comps_array[2]);
        swap_ptr(&gpu_image->comps_array[1], &gpu_image->comps_array[3]);
    }
}
void gpu_mct_inverse_tetrix(xs_image_t *im, xs_image_t *gpu_image, const xs_cfa_pattern_t cfa_pattern, const xs_cts_parameters_t cts_parameters)
{
    assert(im->ncomps == 4);
    assert(im->sx[0] == im->sx[1] && im->sx[0] == im->sx[2] && im->sx[0] == im->sx[3]);
    assert(im->sy[0] == im->sy[1] && im->sy[0] == im->sy[2] && im->sy[0] == im->sy[3]);

    const int Cf = cts_parameters.Cf;
    const int Ct = (cfa_pattern == XS_CFA_RGGB || cfa_pattern == XS_CFA_BGGR) ? 0 : 1;
    const uint8_t e1 = cts_parameters.e1;
    const uint8_t e2 = cts_parameters.e2;

    // Inverse average.
    const int len = im->height * im->width;
    const int block_size = BLOCK_SIZE;
    const int grid_size = (len + block_size - 1) / block_size;
    kernel_inverse_average<<<grid_size, block_size>>>(gpu_image, Cf, Ct, im->width, len);
    cudaDeviceSynchronize();
    // Inverse delta.

    kernel_inverse_delta<<<grid_size, block_size>>>(gpu_image, Cf, Ct, im->width, len);
    cudaDeviceSynchronize();
    // Inverse Y.

    kernel_inverse_Y<<<grid_size, block_size>>>(gpu_image, Cf, Ct, e1, e2, im->width, len);
    cudaDeviceSynchronize();
    // Inverse CbCr.
    kernel_inverse_CbCr<<<grid_size, block_size>>>(gpu_image, Cf, Ct, e1, e2, im->width, len);
    cudaDeviceSynchronize();
    // Reassign component order.
    swap_ptr(&im->comps_array[0], &im->comps_array[2]);
    swap_ptr(&im->comps_array[1], &im->comps_array[3]);
    gpu_swap_image_ptr<<<1, 1>>>(gpu_image);
    cudaDeviceSynchronize();
}

void gpu_mct_inverse_transform(xs_image_t *image, xs_image_t *gpu_image, const xs_config_parameters_t *p)
{
    switch (p->color_transform)
    {
    case XS_CPIH_NONE:
    {
        break;
    }
    case XS_CPIH_RCT:
    {
        gpu_mct_inverse_rct(image, gpu_image);
        break;
    }
    case XS_CPIH_TETRIX:
    {
        gpu_mct_inverse_tetrix(image, gpu_image, p->cfa_pattern, p->tetrix_params);
        break;
    }
    default:
        assert(!"Unknown color transform");
        break;
    }
}
