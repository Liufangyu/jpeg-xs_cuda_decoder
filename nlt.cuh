#ifndef NLT_CUH
#define NLT_CUH

#include "libjxs.h"
#include "common.h"
#include <assert.h>
#include <cuda_runtime.h>

void gpu_nlt_inverse_transform(xs_image_t *im, xs_image_t *gpu_image, const xs_config_parameters_t *p);

#endif