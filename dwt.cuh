#ifndef DWT_CUH
#define DWT_CUH
#include "libjxs.h"
#include "ids.h"
#define N_STREAM 5

void gpu_dwt_inverse_transform(const ids_t *ids, xs_data_in_t **gpu_comps_array);
#endif
