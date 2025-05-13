#include <thrust/scan.h>
#include <cmath>
#include "Form_mask_matrix_B.cuh"
#include "Calculate_C_nnz.cuh"
#include "numeric.cuh"
#include "binning.cuh"
#include "CSR.h"
#include "Tool.h"

__global__ void warm_up_gpu()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

void warm_gpu()
{
    int *d;
    CHECK_ERROR(cudaMalloc(&d, 512));
    warm_up_gpu<<<4096, 1024>>>();
    CHECK_ERROR(cudaFree(d));
    CHECK_ERROR(cudaDeviceSynchronize());
}
template <int TYPE>
void binning(Tool &tools, int *d_bins, int *flop, int M)
{
    int BS = 512;
    int GS = (M + BS - 1) / BS;
    CHECK_ERROR(cudaMemset(tools.d_bin_size, 0, BIN_SIZE * sizeof(int)));
    CHECK_ERROR(cudaMemset(tools.d_max_row_nnz, 0, sizeof(int)));
    k_binning1<TYPE><<<GS, BS>>>(flop, tools.d_bin_size, M, tools.d_max_row_nnz);
    CHECK_ERROR(cudaMemcpy(tools.h_bin_size, tools.d_bin_size, BIN_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemset(tools.d_bin_size, 0, BIN_SIZE * sizeof(int)));
    tools.h_bin_offset[0] = 0;
    for (int i = 0; i < BIN_SIZE - 1; i++)
    {
        tools.h_bin_offset[i + 1] = tools.h_bin_offset[i] + tools.h_bin_size[i];
    }
    CHECK_ERROR(cudaMemcpy(tools.d_bin_offset, tools.h_bin_offset, BIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    k_binning2<TYPE><<<GS, BS>>>(flop, M, d_bins, tools.d_bin_size, tools.d_bin_offset);
}
void Calculate_B_tilePtr(CSR &B, Tool &tools)
{
    int BS;
    int GS;
    if (tools.h_bin_size[8])
    {
        k_calculate_B_tilePtr_shared_hash_tb<8527><<<tools.h_bin_size[8], 512, 0, tools.streams[7]>>>(B.d_ptr, B.d_col, B.d_tileptr, tools.d_bins_B + tools.h_bin_offset[8]);
    }
    if (tools.h_bin_size[7])
    {
        k_calculate_B_tilePtr_shared_hash_tb<4259><<<tools.h_bin_size[7], 256, 0, tools.streams[6]>>>(B.d_ptr, B.d_col, B.d_tileptr, tools.d_bins_B + tools.h_bin_offset[7]);
    }
    if (tools.h_bin_size[5])
    {
        k_calculate_B_tilePtr_shared_hash_tb<1063><<<tools.h_bin_size[5], 64, 0, tools.streams[4]>>>(B.d_ptr, B.d_col, B.d_tileptr, tools.d_bins_B + tools.h_bin_offset[5]);
    }
    if (tools.h_bin_size[6])
    {
        k_calculate_B_tilePtr_shared_hash_tb<2131><<<tools.h_bin_size[6], 128, 0, tools.streams[5]>>>(B.d_ptr, B.d_col, B.d_tileptr, tools.d_bins_B + tools.h_bin_offset[6]);
    }
    if (tools.h_bin_size[9])
    {
        CHECK_ERROR(cudaFuncSetAttribute(k_calculate_B_tilePtr_max_shared,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
        k_calculate_B_tilePtr_max_shared<<<tools.h_bin_size[9], 1024, 101376, tools.streams[8]>>>(B.d_ptr, B.d_col, B.d_tileptr, tools.d_bins_B + tools.h_bin_offset[9]);
    }
    if (tools.h_bin_size[10])
    {
        k_calculate_B_tilePtr_global_mem<<<tools.h_bin_size[10], 1024, 0, tools.streams[9]>>>(B.d_ptr, B.d_col, B.d_tileptr, tools.d_global_mem_pool, tools.d_bins_B + tools.h_bin_offset[10], (B.N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }
    if (tools.h_bin_size[1])
    {
        BS = 512;
        GS = ((tools.h_bin_size[1]) + BS - 1) / BS;
        k_calculate_tilePtr_one_nnz<<<GS, BS, 0, tools.streams[0]>>>(tools.h_bin_size[1], B.d_tileptr, tools.d_bins_B + tools.h_bin_offset[1]);
    }
    if (tools.h_bin_size[2])
    {
        BS = 512;
        GS = ((tools.h_bin_size[2]) + BS - 1) / BS;
        k_calculate_tilePtr_two_three_nnz<2><<<GS, BS, 0, tools.streams[1]>>>(B.d_ptr, B.d_col, tools.h_bin_size[2], B.d_tileptr, tools.d_bins_B + tools.h_bin_offset[2]);
    }
    if (tools.h_bin_size[3])
    {
        BS = 512;
        GS = ((tools.h_bin_size[3]) + BS - 1) / BS;
        k_calculate_tilePtr_two_three_nnz<3><<<GS, BS, 0, tools.streams[2]>>>(B.d_ptr, B.d_col, tools.h_bin_size[3], B.d_tileptr, tools.d_bins_B + tools.h_bin_offset[3]);
    }
    if (tools.h_bin_size[4])
    {
        BS = PWARP_ROWS_FOR_B_TILEPTR * PWARP_FOR_B_TILEPTR;
        GS = ((tools.h_bin_size[4]) + PWARP_ROWS_FOR_B_TILEPTR - 1) / PWARP_ROWS_FOR_B_TILEPTR;
        k_calculate_B_tilePtr_shared_hash_pwarp<<<GS, BS, 0, tools.streams[3]>>>(B.d_ptr, B.d_col, tools.h_bin_size[4], B.d_tileptr, tools.d_bins_B + tools.h_bin_offset[4]);
    }
}

void Calculate_B_tileColAndtileMask(CSR &B, Tool &tools)
{
    int BS = 512;
    int GS;
    if (tools.h_bin_size[1])
    {
        GS = ((tools.h_bin_size[1]) + BS - 1) / BS;
        k_calculate_B_tileColAndtileMask_for_one_tile<<<GS, BS, 0, tools.streams[0]>>>(B.d_ptr, B.d_col, tools.h_bin_size[1], B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.d_bins_B + tools.h_bin_offset[1]);
    }
    if (tools.h_bin_size[2])
    {
        GS = ((tools.h_bin_size[2]) + BS - 1) / BS;
        k_calculate_B_tileColAndtileMask_for_two_tile<<<GS, BS, 0, tools.streams[1]>>>(B.d_ptr, B.d_col, tools.h_bin_size[2], B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.d_bins_B + tools.h_bin_offset[2]);
    }
    if (tools.h_bin_size[3])
    {
        BS = PWARP_ROWS_FOR_B_MASK * PWARP_FOR_B_MASK;
        GS = ((tools.h_bin_size[3]) + PWARP_ROWS_FOR_B_MASK - 1) / PWARP_ROWS_FOR_B_MASK;
        k_calculate_B_tileColAndtileMask_shared_hash_pwarp<<<GS, BS, 0, tools.streams[2]>>>(B.d_ptr, B.d_col, tools.h_bin_size[3], B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.d_bins_B + tools.h_bin_offset[3]);
    }
    if (tools.h_bin_size[4])
    {
        k_calculate_B_tileColAndtileMask_shared_hash_tb<523><<<tools.h_bin_size[4], 64, 0, tools.streams[3]>>>(B.d_ptr, B.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.d_bins_B + tools.h_bin_offset[4]);
    }
    if (tools.h_bin_size[5])
    {
        k_calculate_B_tileColAndtileMask_shared_hash_tb<1063><<<tools.h_bin_size[5], 128, 0, tools.streams[4]>>>(B.d_ptr, B.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.d_bins_B + tools.h_bin_offset[5]);
    }
    if (tools.h_bin_size[6])
    {
        k_calculate_B_tileColAndtileMask_shared_hash_tb<2131><<<tools.h_bin_size[6], 256, 0, tools.streams[5]>>>(B.d_ptr, B.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.d_bins_B + tools.h_bin_offset[6]);
    }
    if (tools.h_bin_size[7])
    {
        k_calculate_B_tileColAndtileMask_shared_hash_tb<4259><<<tools.h_bin_size[7], 512, 0, tools.streams[6]>>>(B.d_ptr, B.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.d_bins_B + tools.h_bin_offset[7]);
    }
    if (tools.h_bin_size[8])
    {
        CHECK_ERROR(cudaFuncSetAttribute(k_calculate_B_tileColAndtileMask_max_shared,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
        k_calculate_B_tileColAndtileMask_max_shared<<<tools.h_bin_size[8], 1024, 101376, tools.streams[7]>>>(B.d_ptr, B.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.d_bins_B + tools.h_bin_offset[8]);
    }
    if (tools.h_bin_size[9])
    {
        k_calculate_B_tileColAndtileMask_global_mem<<<tools.h_bin_size[10], 1024, 0, tools.streams[8]>>>(B.d_ptr, B.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, (MASK_TYPE*)tools.d_global_mem_pool, tools.d_bins_B + tools.h_bin_offset[10], (B.N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }
}

void Calculate_C_tilePtr(const CSR &A, const CSR &B, CSR &C, Tool &tools)
{
    int BS = 512;
    int GS;
    if (tools.h_bin_size[8])
    {
        k_calculate_C_tilePtr_global_mem<<<tools.h_bin_size[8], 1024, 0, tools.streams[7]>>>(
            A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, tools.d_bins_C + tools.h_bin_offset[8], tools.d_global_mem_pool, C.d_tileptr, (C.N + BLOCK_SIZE - 1) / BLOCK_SIZE, tools.group_size);
    }
    if (tools.h_bin_size[4])
    {
        k_calculate_C_tilePtr_shared_hash_tb<hash_size_C_tileptr[1]><<<tools.h_bin_size[4], 128, 0, tools.streams[3]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, tools.d_bins_C + tools.h_bin_offset[4], C.d_tileptr, tools.hash_conflict, tools.group_size);
    }
    if (tools.h_bin_size[5])
    {
        k_calculate_C_tilePtr_shared_hash_tb<hash_size_C_tileptr[2]><<<tools.h_bin_size[5], 256, 0, tools.streams[4]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, tools.d_bins_C + tools.h_bin_offset[5], C.d_tileptr, tools.hash_conflict, tools.group_size);
    }
    if (tools.h_bin_size[6])
    {
        k_calculate_C_tilePtr_shared_hash_tb<hash_size_C_tileptr[3]><<<tools.h_bin_size[6], 512, 0, tools.streams[5]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, tools.d_bins_C + tools.h_bin_offset[6], C.d_tileptr, tools.hash_conflict, tools.group_size);
    }
    if (tools.h_bin_size[7])
    {
        CHECK_ERROR(cudaFuncSetAttribute(k_calculate_C_tilePtr_max_shared,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
        k_calculate_C_tilePtr_max_shared<<<tools.h_bin_size[7], 1024, 101376, tools.streams[6]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, tools.d_bins_C + tools.h_bin_offset[7], C.d_tileptr, tools.hash_conflict, tools.group_size);
    }
    if (tools.h_bin_size[1])
    {
        GS = ((tools.h_bin_size[1]) + BS - 1) / BS;
        k_calculate_tilePtr_one_nnz<<<GS, BS, 0, tools.streams[0]>>>(tools.h_bin_size[1], C.d_tileptr, tools.d_bins_C + tools.h_bin_offset[1]);
    }
    if (tools.h_bin_size[2])
    {
        BS = PWARP_ROWS_FOR_C_TILEPTR * PWARP_FOR_C_TILEPTR;
        GS = ((tools.h_bin_size[2]) + PWARP_ROWS_FOR_C_TILEPTR - 1) / PWARP_ROWS_FOR_C_TILEPTR;
        k_calculate_C_tilePtr_shared_hash_pwarp<<<GS, BS, 0, tools.streams[1]>>>(A.d_ptr, A.d_col, tools.h_bin_size[2], B.d_tileptr, B.d_tilecol, tools.d_bins_C + tools.h_bin_offset[2], C.d_tileptr, tools.hash_conflict);
    }
    if (tools.h_bin_size[3])
    {
        k_calculate_C_tilePtr_shared_hash_tb<hash_size_C_tileptr[0]><<<tools.h_bin_size[3], 64, 0, tools.streams[2]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, tools.d_bins_C + tools.h_bin_offset[3], C.d_tileptr, tools.hash_conflict, tools.group_size);
    }
}

void Calculate_C_nnz_by_OR_CtileMask(const CSR &A, const CSR &B, CSR &C, Tool &tools)
{
    int BS = 512;
    int GS;
    if (tools.h_bin_size[3])
    {
        k_calculate_C_nnz_shared_hash_tb<hash_size_C_nnz[0]><<<tools.h_bin_size[3], 64, 0, tools.streams[2]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, C.d_tileptr, tools.d_bins_C + tools.h_bin_offset[3], C.d_ptr, tools.hash_conflict, tools.group_size);
    }
    if (tools.h_bin_size[4])
    {
        k_calculate_C_nnz_shared_hash_tb<hash_size_C_nnz[1]><<<tools.h_bin_size[4], 128, 0, tools.streams[3]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, C.d_tileptr, tools.d_bins_C + tools.h_bin_offset[4], C.d_ptr, tools.hash_conflict, tools.group_size);
    }
    if (tools.h_bin_size[5])
    {
        k_calculate_C_nnz_shared_hash_tb<hash_size_C_nnz[2]><<<tools.h_bin_size[5], 256, 0, tools.streams[4]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, C.d_tileptr, tools.d_bins_C + tools.h_bin_offset[5], C.d_ptr, tools.hash_conflict, tools.group_size);
    }
    if (tools.h_bin_size[1])
    {
        GS = ((tools.h_bin_size[1]) + BS - 1) / BS;
        k_calculate_C_nnz_for_one_tile<<<GS, BS, 0, tools.streams[0]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.h_bin_size[1], C.d_tileptr, tools.d_bins_C + tools.h_bin_offset[1], C.d_ptr);
    }
    if (tools.h_bin_size[2])
    {
        BS = PWARP_ROWS_FOR_C_NNZ * PWARP_FOR_C_NNZ;
        GS = ((tools.h_bin_size[2]) + PWARP_ROWS_FOR_C_NNZ - 1) / PWARP_ROWS_FOR_C_NNZ;
        k_calculate_C_nnz_shared_hash_pwarp<<<GS, BS, 0, tools.streams[1]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, tools.h_bin_size[2], C.d_tileptr, tools.d_bins_C + tools.h_bin_offset[2], C.d_ptr, tools.hash_conflict);
    }
    if (tools.h_bin_size[6])
    {
        k_calculate_C_nnz_shared_hash_tb<hash_size_C_nnz[3]><<<tools.h_bin_size[6], 512, 0, tools.streams[5]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, C.d_tileptr, tools.d_bins_C + tools.h_bin_offset[6], C.d_ptr, tools.hash_conflict, tools.group_size);
    }
    if (tools.h_bin_size[7])
    {
        CHECK_ERROR(cudaFuncSetAttribute(k_calculate_C_nnz_max_shared,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
        k_calculate_C_nnz_max_shared<<<tools.h_bin_size[7], 1024, 101376, tools.streams[6]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, C.d_tileptr, tools.d_bins_C + tools.h_bin_offset[7], C.d_ptr, tools.hash_conflict, tools.group_size);
    }
    if (tools.h_bin_size[8])
    {
        k_calculate_C_tileColAndtileMask_global_mem<<<tools.h_bin_size[8], 1024, 101376, tools.streams[7]>>>(A.d_ptr, A.d_col, B.d_tileptr, B.d_tilecol, B.d_tilemask, C.d_tileptr, tools.d_bins_C + tools.h_bin_offset[7], C.d_ptr, (C.N + BLOCK_SIZE - 1) / BLOCK_SIZE, tools.d_global_mem_pool, tools.group_size);
    }
}

void Form_mask_matrix_B(const CSR &A, CSR &B, CSR &C, Tool &tools)
{
    tools.allocate(B, C);

    CHECK_ERROR(cudaMalloc((void **)&C.d_ptr, (C.M + 1) * sizeof(int)));
    CHECK_ERROR(cudaMalloc((void **)&B.d_tileptr, (B.M + 1) * sizeof(int)));
    CHECK_ERROR(cudaMemset(B.d_tileptr, 0, (B.M + 1) * sizeof(int)));

    int BS = 512;
    int GS = (B.M + BS - 1) / BS;

    k_calculate_B_per_row_nnz<<<GS, BS>>>(B.d_ptr, tools.d_B_per_row_nnz, B.M);

    binning<0>(tools, tools.d_bins_B, tools.d_B_per_row_nnz, B.M);
    if (tools.h_bin_size[10] != 0)
    {
        tools.d_global_mem_pool_flag = 1;
        tools.global_mem_pool_size = tools.h_bin_size[10] * sizeof(int) * ((C.N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        CHECK_ERROR(cudaMalloc(&tools.d_global_mem_pool, tools.global_mem_pool_size));
    }
    Calculate_B_tilePtr(B, tools);
    // printf("%u %u %u %u %u %u %u %u %u %u\n", tools.h_bin_size[0], tools.h_bin_size[1], tools.h_bin_size[2], tools.h_bin_size[3], tools.h_bin_size[4], tools.h_bin_size[5], tools.h_bin_size[6], tools.h_bin_size[7], tools.h_bin_size[8], tools.h_bin_size[9]);
    binning<1>(tools, tools.d_bins_B, B.d_tileptr, B.M);
    // Using 4 threads to compute a row, a 512-size thread block can compute 128 rows
    GS = (A.M + 127) / 128;
    k_calculate_flop<<<GS, BS>>>(A.d_ptr, A.d_col, B.d_tileptr, A.M, C.d_ptr);

    cub::DeviceScan::ExclusiveSum(tools.d_cub_storage, tools.cub_temp_storage, B.d_tileptr, B.d_tileptr, B.M + 1, 0);

    CHECK_ERROR(cudaMemcpy(tools.count, B.d_tileptr + B.M, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMalloc(&B.d_tilecol, *tools.count * sizeof(int)));
    CHECK_ERROR(cudaMalloc(&B.d_tilemask, *tools.count * sizeof(MASK_TYPE)));
    if (tools.h_bin_size[9] != 0)
    {
        size_t global_size = tools.h_bin_size[9] * sizeof(MASK_TYPE) * ((C.N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (tools.d_global_mem_pool_flag)
        {
            if (global_size > tools.global_mem_pool_size)
            {
                CHECK_ERROR(cudaFree(tools.d_global_mem_pool));
                CHECK_ERROR(cudaMalloc(&tools.d_global_mem_pool, global_size));
            }
        }
        else
        {
            tools.d_global_mem_pool_flag = 1;
            tools.global_mem_pool_size = global_size;
            CHECK_ERROR(cudaMalloc(&tools.d_global_mem_pool, global_size));
        }
    }
    Calculate_B_tileColAndtileMask(B, tools);
}

void Calculate_C_nnz(const CSR &A, const CSR &B, CSR &C, Tool &tools)
{
    CHECK_ERROR(cudaMalloc((void **)&C.d_tileptr, (C.M + 1) * sizeof(int)));
    CHECK_ERROR(cudaMemset(C.d_tileptr, 0, (C.M + 1) * sizeof(int)));

    binning<2>(tools, tools.d_bins_C, C.d_ptr, C.M);

    if (tools.h_bin_size[8] != 0)
    {
        size_t global_size = tools.h_bin_size[8] * sizeof(int) * ((C.N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (tools.d_global_mem_pool_flag)
        {
            if (global_size > tools.global_mem_pool_size)
            {
                CHECK_ERROR(cudaFree(tools.d_global_mem_pool));
                CHECK_ERROR(cudaMalloc(&tools.d_global_mem_pool, global_size));
            }
        }
        else
        {
            tools.d_global_mem_pool_flag = 1;
            tools.global_mem_pool_size = global_size;
            CHECK_ERROR(cudaMalloc(&tools.d_global_mem_pool, global_size));
        }
    }
    int rows = C.M - tools.h_bin_offset[3];
#if ADAPTIVE_GROUPING
    k_init_group_size<0><<<(rows + 511) / 512, 512>>>(A.d_ptr, C.d_ptr, C.d_ptr, rows, tools.d_bins_C + tools.h_bin_offset[3], tools.group_size);
#endif
    Calculate_C_tilePtr(A, B, C, tools);

    binning<3>(tools, tools.d_bins_C, C.d_tileptr, C.M);
    // printf("%u %u %u %u %u %u %u %u %u %u\n", tools.h_bin_size[0], tools.h_bin_size[1], tools.h_bin_size[2], tools.h_bin_size[3], tools.h_bin_size[4], tools.h_bin_size[5], tools.h_bin_size[6], tools.h_bin_size[7], tools.h_bin_size[8], tools.h_bin_size[9]);
#if ADAPTIVE_GROUPING
    rows = C.M - tools.h_bin_offset[3];
    k_init_group_size<1><<<(rows + 511) / 512, 512>>>(A.d_ptr, C.d_tileptr, C.d_ptr, rows, tools.d_bins_C + tools.h_bin_offset[3], tools.group_size);
#endif
    cub::DeviceScan::ExclusiveSum(tools.d_cub_storage, tools.cub_temp_storage, C.d_tileptr, C.d_tileptr, C.M + 1, 0);
    CHECK_ERROR(cudaMemcpy(tools.count, C.d_tileptr + C.M, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaMemset(C.d_ptr, 0, (C.M + 1) * sizeof(int)));
    if (tools.h_bin_size[8] != 0)
    {
        size_t global_size = tools.h_bin_size[8] * ((C.N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(MASK_TYPE);
        if (tools.d_global_mem_pool_flag)
        {
            if (global_size > tools.global_mem_pool_size)
            {
                CHECK_ERROR(cudaFree(tools.d_global_mem_pool));
                CHECK_ERROR(cudaMalloc(&tools.d_global_mem_pool, global_size));
            }
        }
        else
        {
            tools.d_global_mem_pool_flag = 1;
            tools.global_mem_pool_size = global_size;
            CHECK_ERROR(cudaMalloc(&tools.d_global_mem_pool, global_size));
        }
    }
    Calculate_C_nnz_by_OR_CtileMask(A, B, C, tools);

    binning<4>(tools, tools.d_bins_C, C.d_ptr, C.M);
#if ADAPTIVE_GROUPING
    int GS = (A.M + 127) / 128;
    k_calculate_flop_tmp<<<GS, 512>>>(A.d_ptr, A.d_col, B.d_ptr, A.M, C.d_tileptr);
    rows = C.M - tools.h_bin_offset[3];
    k_init_group_size<2><<<(rows + 511) / 512, 512>>>(A.d_ptr, C.d_ptr, C.d_tileptr, rows, tools.d_bins_C + tools.h_bin_offset[3], tools.group_size);
#endif
    // printf("%u %u %u %u %u %u %u %u %u %u %u %u\n", tools.h_bin_size[0], tools.h_bin_size[1], tools.h_bin_size[2], tools.h_bin_size[3], tools.h_bin_size[4], tools.h_bin_size[5], tools.h_bin_size[6], tools.h_bin_size[7], tools.h_bin_size[8], tools.h_bin_size[9], tools.h_bin_size[10], tools.h_bin_size[11]);

    cub::DeviceScan::ExclusiveSum(tools.d_cub_storage, tools.cub_temp_storage, C.d_ptr, C.d_ptr, C.M + 1, 0);

    CHECK_ERROR(cudaMemcpy(tools.count, C.d_ptr + C.M, sizeof(int), cudaMemcpyDeviceToHost));
}

void h_numeric(const CSR &A, const CSR &B, CSR &C, Tool &tools)
{
    int BS = 512;
    int GS;
    if (tools.h_bin_size[1])
    {
        GS = (tools.h_bin_size[1] + BS - 1) / BS;
        k_numeric_for_one_nnz<<<GS, BS, 0, tools.streams[0]>>>(A.d_ptr, A.d_col, A.d_val, B.d_ptr, B.d_col, B.d_val, tools.h_bin_size[1], tools.d_bins_C + tools.h_bin_offset[1], C.d_ptr, C.d_col, C.d_val);
    }
    if (tools.h_bin_size[8])
    {
        CHECK_ERROR(cudaMemcpy(tools.count, tools.d_max_row_nnz, sizeof(int), cudaMemcpyDeviceToHost));
        printf("global row:%d\n", tools.h_bin_size[8]);
        int max_tsize = *tools.count * 2;
        size_t global_size = tools.h_bin_size[8] * max_tsize * (sizeof(VALUE_TYPE) + sizeof(int));
        if (tools.d_global_mem_pool_flag)
        {
            if (global_size > tools.global_mem_pool_size)
            {
                CHECK_ERROR(cudaFree(tools.d_global_mem_pool));
                CHECK_ERROR(cudaMalloc(&tools.d_global_mem_pool, global_size));
            }
        }
        else
        {
            tools.d_global_mem_pool_flag = 1;
            tools.global_mem_pool_size = global_size;
            CHECK_ERROR(cudaMalloc(&tools.d_global_mem_pool, global_size));
        }
        k_numeric_global_hash<<<tools.h_bin_size[8], 1024, 0, tools.streams[7]>>>(A.d_ptr, A.d_col, A.d_val, B.d_ptr, B.d_col, B.d_val, tools.d_bins_C + tools.h_bin_offset[8], C.d_ptr, C.d_col, C.d_val, tools.d_global_mem_pool, max_tsize, tools.group_size, tools.hash_conflict);
    }
    if (tools.h_bin_size[2])
    {
        BS = PWARP_FOR_C_NNZ * PWARP_ROWS_FOR_C_NNZ;
        GS = ((tools.h_bin_size[2]) + PWARP_ROWS_FOR_C_NNZ - 1) / PWARP_ROWS_FOR_C_NNZ;
        k_numeric_shared_hash_pwarp<<<GS, BS, 0, tools.streams[1]>>>(A.d_ptr, A.d_col, A.d_val, B.d_ptr, B.d_col, B.d_val, tools.h_bin_size[2], tools.d_bins_C + tools.h_bin_offset[2], C.d_ptr, C.d_col, C.d_val, tools.hash_conflict);
    }

    if (tools.h_bin_size[3])
    {
        k_numeric_shared_hash_tb<hash_size_numeric[0]><<<tools.h_bin_size[3], 64, 0, tools.streams[2]>>>(A.d_ptr, A.d_col, A.d_val, B.d_ptr, B.d_col, B.d_val, tools.d_bins_C + tools.h_bin_offset[3], C.d_ptr, C.d_col, C.d_val, tools.group_size, tools.hash_conflict);
    }
    if (tools.h_bin_size[4])
    {
        k_numeric_shared_hash_tb<hash_size_numeric[1]><<<tools.h_bin_size[4], 128, 0, tools.streams[3]>>>(A.d_ptr, A.d_col, A.d_val, B.d_ptr, B.d_col, B.d_val, tools.d_bins_C + tools.h_bin_offset[4], C.d_ptr, C.d_col, C.d_val, tools.group_size, tools.hash_conflict);
    }
    if (tools.h_bin_size[5])
    {
        k_numeric_shared_hash_tb<hash_size_numeric[2]><<<tools.h_bin_size[5], 256, 0, tools.streams[4]>>>(A.d_ptr, A.d_col, A.d_val, B.d_ptr, B.d_col, B.d_val, tools.d_bins_C + tools.h_bin_offset[5], C.d_ptr, C.d_col, C.d_val, tools.group_size, tools.hash_conflict);
    }
    if (tools.h_bin_size[6])
    {
        k_numeric_shared_hash_tb<hash_size_numeric[3]><<<tools.h_bin_size[6], 512, 0, tools.streams[5]>>>(A.d_ptr, A.d_col, A.d_val, B.d_ptr, B.d_col, B.d_val, tools.d_bins_C + tools.h_bin_offset[6], C.d_ptr, C.d_col, C.d_val, tools.group_size, tools.hash_conflict);
    }
    if (tools.h_bin_size[7])
    {
        CHECK_ERROR(cudaFuncSetAttribute(k_numeric_max_shared_hash_tb,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
        k_numeric_max_shared_hash_tb<<<tools.h_bin_size[7], 1024, 101376, tools.streams[6]>>>(A.d_ptr, A.d_col, A.d_val, B.d_ptr, B.d_col, B.d_val, tools.d_bins_C + tools.h_bin_offset[7], C.d_ptr, C.d_col, C.d_val, tools.group_size, tools.hash_conflict);
    }
}
