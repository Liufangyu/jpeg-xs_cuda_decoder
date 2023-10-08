/***************************************************************************
** intoPIX SA & Fraunhofer IIS (hereinafter the "Software Copyright       **
** Holder") hold or have the right to license copyright with respect to   **
** the accompanying software (hereinafter the "Software").                **
**                                                                        **
** Copyright License for Evaluation and Testing                           **
** --------------------------------------------                           **
**                                                                        **
** The Software Copyright Holder hereby grants, to any implementer of     **
** this ISO Standard, an irrevocable, non-exclusive, worldwide,           **
** royalty-free, sub-licensable copyright licence to prepare derivative   **
** works of (including translations, adaptations, alterations), the       **
** Software and reproduce, display, distribute and execute the Software   **
** and derivative works thereof, for the following limited purposes: (i)  **
** to evaluate the Software and any derivative works thereof for          **
** inclusion in its implementation of this ISO Standard, and (ii)         **
** to determine whether its implementation conforms with this ISO         **
** Standard.                                                              **
**                                                                        **
** The Software Copyright Holder represents and warrants that, to the     **
** best of its knowledge, it has the necessary copyright rights to        **
** license the Software pursuant to the terms and conditions set forth in **
** this option.                                                           **
**                                                                        **
** No patent licence is granted, nor is a patent licensing commitment     **
** made, by implication, estoppel or otherwise.                           **
**                                                                        **
** Disclaimer: Other than as expressly provided herein, (1) the Software  **
** is provided �AS IS� WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING  **
** BUT NOT LIMITED TO, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A   **
** PARTICULAR PURPOSE AND NON-INFRINGMENT OF INTELLECTUAL PROPERTY RIGHTS **
** and (2) neither the Software Copyright Holder (or its affiliates) nor  **
** the ISO shall be held liable in any event for any damages whatsoever   **
** (including, without limitation, damages for loss of profits, business  **
** interruption, loss of information, or any other pecuniary loss)        **
** arising out of or related to the use of or inability to use the        **
** Software.�                                                             **
**                                                                        **
** RAND Copyright Licensing Commitment                                    **
** -----------------------------------                                    **
**                                                                        **
** IN THE EVENT YOU WISH TO INCLUDE THE SOFTWARE IN A CONFORMING          **
** IMPLEMENTATION OF THIS ISO STANDARD, PLEASE BE FURTHER ADVISED THAT:   **
**                                                                        **
** The Software Copyright Holder agrees to grant a copyright              **
** license on reasonable and non- discriminatory terms and conditions for **
** the purpose of including the Software in a conforming implementation   **
** of the ISO Standard. Negotiations with regard to the license are       **
** left to the parties concerned and are performed outside the ISO.       **
**                                                                        **
** No patent licence is granted, nor is a patent licensing commitment     **
** made, by implication, estoppel or otherwise.                           **
***************************************************************************/
#include "libjxs.h"
#include "common.h"
#include <assert.h>
#include <cuda_runtime.h>
#include "nlt.cuh"
__device__ void clamp(xs_data_in_t v, xs_data_in_t max_v, xs_data_in_t *ret)
{
    if (v > max_v)
    {
        v = max_v;
    }
    if (v < 0)
    {
        v = 0;
    }
    *ret = v;
}

__device__ void clamp64(int64_t v, int64_t max_v, int64_t *ret)
{
    if (v > max_v)
    {
        v = max_v;
    }
    if (v < 0)
    {
        v = 0;
    }
    *ret = v;
}

__global__ void kernel_nlt_inverse_linear(xs_image_t *gpu_image,
                                          const uint8_t s, const xs_data_in_t dclev_and_rounding,
                                          const xs_data_in_t max_val)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int pixels_num = 0;
    for (int c = 0; c < gpu_image->ncomps; ++c)
    {
        pixels_num = (size_t)(gpu_image->width / gpu_image->sx[c]) * (size_t)(gpu_image->height / gpu_image->sy[c]);
        if (tid < pixels_num)
        {
            xs_data_in_t *the_ptr = gpu_image->comps_array[c] + tid;
            clamp((*the_ptr + dclev_and_rounding) >> s, max_val, the_ptr);
            break;
        }
        tid -= pixels_num;
    }
}

void gpu_nlt_inverse_linear(xs_image_t *image, xs_image_t *gpu_image, const uint8_t Bw)
{
    const uint8_t s = Bw - (uint8_t)image->depth;
    const xs_data_in_t dclev_and_rounding = ((1 << Bw) >> 1) + ((1 << s) >> 1);
    const xs_data_in_t max_val = (1 << image->depth) - 1;
    int pixels_num = 0;
    for (int c = 0; c < image->ncomps; ++c)
    {
        pixels_num += (size_t)(image->width / image->sx[c]) * (size_t)(image->height / image->sy[c]);
    }
    const int block_size = BLOCK_SIZE;
    const int grid_size = (pixels_num + block_size - 1) / block_size;
    kernel_nlt_inverse_linear<<<grid_size, block_size>>>(gpu_image, s, dclev_and_rounding, max_val);
}
__global__ void kernel_nlt_inverse_quadratic(xs_image_t *gpu_image, const xs_data_in_t dclev,
                                             const uint8_t s, const xs_data_in_t vdco, const xs_data_in_t s_r,
                                             const xs_data_in_t max_val, const xs_data_in_t max_coef)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int pixels_num = 0;
    for (int c = 0; c < gpu_image->ncomps; ++c)
    {
        pixels_num = (size_t)(gpu_image->width / gpu_image->sx[c]) * (size_t)(gpu_image->height / gpu_image->sy[c]);
        if (tid < pixels_num)
        {
            xs_data_in_t *the_ptr = gpu_image->comps_array[c] + tid;
            int64_t v;
            clamp(*the_ptr + dclev, max_coef, (xs_data_in_t *)&v);
            clamp64(((v * v + s_r) >> s) + vdco, max_val, (int64_t *)the_ptr);
            break;
        }
        tid -= pixels_num;
    }
}

void gpu_nlt_inverse_quadratic(xs_image_t *image, xs_image_t *gpu_image, const uint8_t Bw, const xs_nlt_parameters_t nlt_parameters)
{
    const xs_data_in_t vdco = (xs_data_in_t)nlt_parameters.quadratic.alpha - (xs_data_in_t)nlt_parameters.quadratic.sigma * 32768;
    const uint8_t s = (Bw << 1) - (uint8_t)image->depth;
    const xs_data_in_t dclev = ((1 << Bw) >> 1);
    const xs_data_in_t s_r = (1 << s) >> 1;
    const xs_data_in_t max_val = (1 << image->depth) - 1;
    const xs_data_in_t max_coef = (1 << Bw) - 1;
    int pixels_num = 0;
    for (int c = 0; c < image->ncomps; ++c)
    {
        pixels_num += (size_t)(image->width / image->sx[c]) * (size_t)(image->height / image->sy[c]);
    }
    const int block_size = BLOCK_SIZE;
    const int grid_size = (pixels_num + block_size - 1) / block_size;
    kernel_nlt_inverse_quadratic<<<grid_size, block_size>>>(gpu_image, dclev, s, vdco, s_r, max_val, max_coef);
}

__global__ void kernel_nlt_inverse_extended(xs_image_t *gpu_image, const uint8_t Bw, const uint8_t e,
                                            uint32_t T1, uint32_t T2, const uint8_t s,
                                            const xs_data_in_t s_r, const xs_data_in_t max_val,
                                            const xs_data_in_t dclev, const xs_data_in_t max_coef,
                                            const int64_t A1, const int64_t A3,
                                            const int64_t B1, const int64_t B2, const int64_t B3)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int pixels_num = 0;
    for (int c = 0; c < gpu_image->ncomps; ++c)
    {
        pixels_num = (size_t)(gpu_image->width / gpu_image->sx[c]) * (size_t)(gpu_image->height / gpu_image->sy[c]);
        if (tid < pixels_num)
        {
            xs_data_in_t *the_ptr = gpu_image->comps_array[c] + tid;
            int64_t v = (int64_t)*the_ptr + dclev;
            if (v < T1)
            {
                clamp64(B1 - v, max_coef, &v);
                v = A1 - v * v;
            }
            else if (v < T2)
            {
                v = (v << e) + B2;
            }
            else
            {
                clamp64(v - B3, max_coef, &v);
                v = A3 + v * v;
            }
            clamp64((v + s_r) >> s, max_val, (int64_t *)the_ptr);
            break;
        }
        tid -= pixels_num;
    }
}

void gpu_nlt_inverse_extended(xs_image_t *image, xs_image_t *gpu_image, const uint8_t Bw, const xs_nlt_parameters_t nlt_parameters)
{
    const uint8_t e = Bw - nlt_parameters.extended.E;
    const int64_t B2 = (int64_t)nlt_parameters.extended.T1 * nlt_parameters.extended.T1;
    const int64_t A1 = B2 + ((int64_t)nlt_parameters.extended.T1 << e) + (1ll << (2ll * e - 2));
    const int64_t B1 = (int64_t)nlt_parameters.extended.T1 + (1ll << (e - 1));
    const int64_t A3 = B2 + ((int64_t)nlt_parameters.extended.T2 << e) - (1ll << (2ll * e - 2));
    const int64_t B3 = (int64_t)nlt_parameters.extended.T2 - (1ll << (e - 1));
    const uint8_t s = (Bw << 1) - (uint8_t)image->depth;
    const xs_data_in_t s_r = (1 << s) >> 1;
    const xs_data_in_t max_val = (1 << image->depth) - 1;
    const xs_data_in_t max_coef = (1 << Bw) - 1;
    const xs_data_in_t dclev = ((1 << Bw) >> 1);
    int pixels_num = 0;
    for (int c = 0; c < image->ncomps; ++c)
    {
        pixels_num += (size_t)(image->width / image->sx[c]) * (size_t)(image->height / image->sy[c]);
    }
    const int block_size = BLOCK_SIZE;
    const int grid_size = (pixels_num + block_size - 1) / block_size;
    kernel_nlt_inverse_extended<<<grid_size, block_size>>>(gpu_image, Bw, e, nlt_parameters.extended.T1, nlt_parameters.extended.T2, s, s_r, max_val, dclev, max_coef, A1, A3, B1, B2, B3);
}

void gpu_nlt_inverse_transform(xs_image_t *image, xs_image_t *gpu_image, const xs_config_parameters_t *p)
{
    switch (p->Tnlt)
    {
    case XS_NLT_NONE:
    {
        gpu_nlt_inverse_linear(image, gpu_image, p->Bw);
        break;
    }
    case XS_NLT_QUADRATIC:
    {
        gpu_nlt_inverse_quadratic(image, gpu_image, p->Bw, p->Tnlt_params);
        break;
    }
    case XS_NLT_EXTENDED:
    {
        gpu_nlt_inverse_extended(image, gpu_image, p->Bw, p->Tnlt_params);
        break;
    }
    default:
        assert(false);
        break;
    }
}
