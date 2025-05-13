#include <unistd.h>
#include <fstream>
#include <iomanip>
#include "common.h"
#include "mmio_read.h"
#include "CSR.h"
#include "MH_spgemm.cuh"
#include "cusparse_spgemm.cuh"
#include "Timing.h"
#include "Tool.h"

void MH_spgemm(const CSR &A, CSR &B, CSR &C, Timing &Timing, Tool &tools)
{
    double t0;
    t0 = fast_clock_time();
    C.M = A.M;
    C.N = B.N;
    Form_mask_matrix_B(A, B, C, tools);
    CHECK_ERROR(cudaDeviceSynchronize());
    Timing.Form_mask_matrix_B = (fast_clock_time() - t0) * 1000;

    t0 = fast_clock_time();
    Calculate_C_nnz(A, B, C, tools);
    CHECK_ERROR(cudaDeviceSynchronize());
    Timing.Calculate_C_nnz = (fast_clock_time() - t0) * 1000;

    t0 = fast_clock_time();
    C.nnz = *tools.count;
    printf("C.nnz = %d\n", C.nnz);
    CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(int)));
    CHECK_ERROR(cudaMalloc(&C.d_val, C.nnz * sizeof(VALUE_TYPE)));
    Timing.Malloc_C_col_val = (fast_clock_time() - t0) * 1000;

    t0 = fast_clock_time();
    h_numeric(A, B, C, tools);
    CHECK_ERROR(cudaDeviceSynchronize());
    Timing.Numeric = (fast_clock_time() - t0) * 1000;

#if HASH_CONFLICT
    CHECK_ERROR(cudaMemcpy(tools.count, tools.hash_conflict, sizeof(int), cudaMemcpyDeviceToHost));
    printf("hash conflict: %u\n", *tools.count);
#endif
}

int main(int argc, char **argv)
{
    char *filename;
    if (argc == 2)
    {
        filename = argv[1];
    }
    else
    {
        puts("Invalid Arguments.");
        puts("Usage:\t ./spgemm <Input File>");
        return -1;
    }
    std::string matrix_name = extract_matrix_name(filename);
    CSR A, B, C;

    readMtxFile(A, filename);

    if (!AAT && A.M != A.N)
    {
        puts("C=AA must have rowA = colA. Exit.");
        return 0;
    }
    printf("--------------------------SpGEMM Start!!!--------------------------\n");
    if (AAT && !A.isSymmetric)
        matrix_transposition(A, B);
    else
        B = A;
    unsigned long long int int_result = 0;
    for (int i = 0; i < A.nnz; i++)
    {
        int rowidx = A.col[i];
        int_result += B.ptr[rowidx + 1] - B.ptr[rowidx];
    }
    C.ptr = new int[A.M + 1]();
    warm_gpu();
    A.H2D();
    B.H2D();
    int iter;
    double Gflops = 0;
#if SPGEMM
    printf("Matrix %s (%d , %d) nnz:%d\n", matrix_name.c_str(), A.M, B.N, A.nnz);
    printf("SpGEMM intermediate result = %lld\n", int_result);
    Timing timing, bench_timing;
    Tool tools;
    iter = 1;
    for (int i = 0; i < iter; i++)
    {
        MH_spgemm(A, B, C, timing, tools);
        bench_timing += timing;
        if (i < iter - 1)
        {
            C.d_release_csr();
            tools.release();
            B.d_release_tile();
            CHECK_ERROR(cudaFree(C.d_tileptr));
        }
    }
    bench_timing /= iter;
    bench_timing.print_step_time();
    Gflops = 2.0 * (double)int_result / (bench_timing.getTotal() * 1e6);
    printf("MH-SpGEMM runtime is %.3lfms, Gflops is %.2lf\n", bench_timing.getTotal(), Gflops);
    tools.release();
    B.d_release_tile();
#endif

#if CUSPARSE
    double cusparse_time;
    double Gflops_cu = 0;
    CSR cusparse_C;
    iter = 1;
    for (int i = 0; i < iter; i++)
    {
        cusparse_spgemm(&A, &B, &cusparse_C, &cusparse_time);
        if (i < iter - 1)
        {
            cusparse_C.release();
        }
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess || cusparse_C.nnz <= 0)
    {
        printf("cusparse is failed!\n");
        Gflops_cu = 0;
    }
    else
    {
        cusparse_C.D2H();
        Gflops_cu = 2.0 * (double)int_result / (cusparse_time * 1e6);
        printf("cusparse: %.3lfms, Gflops is %.2lf\n", cusparse_time, Gflops_cu);
    }
#if WRITE
    std::ofstream Throughput_CU("./data/Gflops_cu.csv", std::ios::app);
    if (!Throughput_CU)
    {
        std::cerr << "Unable to open Gflops_cu.csv" << std::endl;
        return 1;
    }
    Throughput_CU.seekp(0, std::ios::end);
    Throughput_CU << std::fixed << std::setprecision(3)
               << Gflops_cu << std::endl;
    Throughput_CU.close();
#endif
#endif
#if CHECK_RESULT && CUSPARSE
    C.D2H();
    if (C == cusparse_C)
    {
        printf("pass\n");
    }
    else
    {
        printf("error\n");
    }
    cusparse_C.release();
#endif

#if WRITE
    std::ofstream Throughput("./data/Gflops_MH-SpGEMM.csv", std::ios::app);

    if (!Throughput)
    {
        std::cerr << "Unable to open Gflops_MH-SpGEMM.csv" << std::endl;
        return 1;
    }
    Throughput.seekp(0, std::ios::end);
    Throughput << std::fixed << std::setprecision(2)
               << Gflops << std::endl;
    Throughput.close();
#endif

    printf("--------------------------SpGEMM   End!!!--------------------------\n");
    return 0;
}