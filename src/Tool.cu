#include <thrust/scan.h>
#include "Tool.h"

void Tool::allocate(const CSR &B, const CSR &C)
{
    streams = new cudaStream_t[BIN_SIZE - 1];
    for (int i = 0; i < BIN_SIZE - 1; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }
    size_t tmp_B = 0, tmp_C = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_B, B.d_tileptr, B.d_tileptr, B.M + 1, 0);
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_C, C.d_ptr, C.d_ptr, C.M + 1, 0);
    cub_temp_storage = max(tmp_B, tmp_C);
    d_combined_size = BIN_SIZE * 2 + B.M * 2 + C.M + 2;

    if (B.M != C.M)
        d_combined_size += C.M;

    CHECK_ERROR(cudaMalloc(&d_combined_mem, d_combined_size * sizeof(int) + cub_temp_storage));

    d_bin_size = d_combined_mem;                                       // BIN_SIZE;
    d_bin_offset = d_combined_mem + BIN_SIZE;                          // BIN_SIZE;
    d_B_per_row_nnz = d_combined_mem + BIN_SIZE * 2;                   // B.M
    d_bins_B = d_combined_mem + BIN_SIZE * 2 + B.M;                    // B.M
    hash_conflict = d_combined_mem + BIN_SIZE * 2 + B.M * 2;           // 1
    group_size = d_combined_mem + BIN_SIZE * 2 + B.M * 2 + 1;          // C.M
    d_max_row_nnz = d_combined_mem + BIN_SIZE * 2 + B.M * 2 + C.M + 1; // 1
    if (B.M != C.M)
    {
        d_bins_C = d_combined_mem + BIN_SIZE * 2 + B.M * 2 + 2 + C.M;          // C.M
        d_cub_storage = d_combined_mem + BIN_SIZE * 2 + B.M * 2 + 2 + C.M * 2;
    }
    else
    {
        d_bins_C = d_bins_B;
        d_cub_storage = d_combined_mem + BIN_SIZE * 2 + B.M * 2 + C.M + 2;
    }

    h_combined_size = BIN_SIZE * 2 + 1;
    h_combined_mem = new int[h_combined_size];
    h_bin_size = h_combined_mem;
    h_bin_offset = h_combined_mem + BIN_SIZE;
    count = h_combined_mem + BIN_SIZE * 2;
}

void Tool::release()
{
    cudaFree(d_combined_mem);
    d_combined_mem = nullptr;
    if (d_global_mem_pool_flag)
    {
        cudaFree(d_global_mem_pool);
        d_global_mem_pool = nullptr;
    }
    if (streams != nullptr)
    {
        for (int i = 0; i < BIN_SIZE - 1; i++)
        {
            cudaStreamDestroy(streams[i]);
        }
        delete[] streams;
    }
    if (h_combined_mem != nullptr)
    {
        delete[] h_combined_mem;
        h_combined_mem = nullptr;
    }
}
Tool::~Tool()
{
    // release();
}