
#include "kernel_wrapper.cuh"
#include <fstream>
#include <string>
#include <iomanip>
#include <cuda_profiler_api.h>
#include <cub/cub.cuh>
#include "cusparse_spgemm.h"
#include "Timings.h"

std::string extract_matrix_name(const std::string &path)
{
    size_t last_slash = path.find_last_of("/\\");
    std::string file_name = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);

    size_t last_dot = file_name.find_last_of('.');
    if (last_dot != std::string::npos)
    {
        file_name = file_name.substr(0, last_dot);
    }
    return file_name;
}
void opsparse(const CSR &A, const CSR &B, CSR &C, Meta &meta, Timings &timing)
{

    double t0, t1;
    t1 = t0 = fast_clock_time();
    C.M = A.M;
    C.N = B.N;
    C.nnz = 0;
    h_setup(A, B, C, meta, timing);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.setup = fast_clock_time() - t0;

    // symbolic binning
    t0 = fast_clock_time();
    h_symbolic_binning(C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic_binning = fast_clock_time() - t0;

    // symbolic phase
    t0 = fast_clock_time();
    h_symbolic(A, B, C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic = fast_clock_time() - t0;

    // numeric binning
    t0 = fast_clock_time();
    h_numeric_binning(C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.numeric_binning = fast_clock_time() - t0;

    // malloc C
    t0 = fast_clock_time();
    C.nnz = *meta.total_nnz;
    printf("C.nnz = %d\n", C.nnz);
    CHECK_ERROR(cudaMalloc(&C.d_val, C.nnz * sizeof(mdouble)));
    CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
    timing.allocate = fast_clock_time() - t0;

    // prefix sum and malloc
    t0 = fast_clock_time();
    cub::DeviceScan::ExclusiveSum(meta.d_cub_storage, meta.cub_storage_size, C.d_rpt, C.d_rpt, C.M + 1);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.prefix = fast_clock_time() - t0;

    // numeric
    t0 = fast_clock_time();
    h_numeric_full_occu(A, B, C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.numeric = fast_clock_time() - t0;

    // cleanup
    t0 = fast_clock_time();
    meta.release();
    timing.cleanup = fast_clock_time() - t0;
    timing.total = fast_clock_time() - t1;
}

int main(int argc, char **argv)
{
    char *filename = argv[1];

    CSR A, B;
    A.construct(filename);
    B = A;

    A.H2D();
    B.H2D();

    long total_flop = compute_flop(A, B);
    CSR C;
    cudaruntime_warmup();
    Meta meta;
    // {
    //     Timings timing;
    //     opsparse(A, B, C, meta, timing);
    //     C.release();
    // }
    mint iter = 1;
    Timings timing, bench_timing;
    for (mint i = 0; i < iter; i++)
    {
        opsparse(A, B, C, meta, timing);
        bench_timing += timing;
        if (i < iter - 1)
        {
            C.release();
        }
    }
    bench_timing /= iter;

    bench_timing.print(total_flop * 2);
    double Gflops = 2.0 * (double)total_flop / (bench_timing.total * 1e9);

    std::string matrix_name = extract_matrix_name(filename);
    std::ofstream outfile("408MatrixResults.csv", std::ios::app);

    if (!outfile)
    {
        std::cerr << "Unable to open 408MatrixResults.csv" << std::endl;
        return 1;
    }
    outfile.seekp(0, std::ios::end);

    outfile << std::fixed << std::setprecision(2)
            << Gflops << std::endl;
    outfile.close();

    // compare result

    // C.D2H();
    // CSR C_ref;
    // cusparse_spgemm(&A, &B, &C_ref);
    // C_ref.D2H();
    // if(C == C_ref){
    //     printf("pass\n");
    // }
    // else{
    //     printf("error\n");
    // }

    A.release();
    B.release();

    C.release();
    return 0;
}
