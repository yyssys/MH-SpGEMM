#pragma once
#include <common.h>
#include "CSR.h"
class Tool
{
public:
    int *d_combined_mem;
    int d_combined_size;

    int *h_combined_mem;
    int h_combined_size;

    int *h_bin_size;
    int *h_bin_offset;
    int *count;

    int *d_bin_size;
    int *d_bin_offset;
    int *d_bins_B;
    int *d_bins_C;

    int *d_max_row_nnz;

    int *group_size;

    int *hash_conflict;

    cudaStream_t *streams;
    int *d_B_per_row_nnz;

    int *d_global_mem_pool;
    size_t global_mem_pool_size = 0;
    int d_global_mem_pool_flag = 0;

    size_t cub_temp_storage = 0;
    int *d_cub_storage;

    Tool() {}
    ~Tool();

    void allocate(const CSR &B, const CSR &C);
    void release();
};
