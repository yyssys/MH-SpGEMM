#include "kernel_wrapper.cuh"
#include <fstream>
#include <cuda_profiler_api.h>
#include <cub/cub.cuh>
#include "Timings.h"
#include "cusparse_spgemm.h"


int main(int argc, char **argv)
{
    char *filename = argv[1];

    CSR A, B;
    A.construct(filename);
    B = A;

    A.H2D();
    B.H2D();
    

    long total_flop = compute_flop(A, B);
    double total_flop_G = double(total_flop) * 2/1000000000;
 
    CSR C;
    double t0 = fast_clock_time(), t1;
    // cusparse_spgemm(&A, &B, &C);
    // C.release();

    int iter = 1;
    t1 = 0;
    for(int i = 0; i < iter; i++){
        t0 = fast_clock_time();
        cusparse_spgemm(&A, &B, &C);
        t1 += fast_clock_time() - t0;
        //printf("iter %d %le\n", i, fast_clock_time() - t0);
        if(i < iter - 1){
            C.release();
        }
    }
    t1 /= iter;
    printf("executione time %.3lf\n", t1 * 1000);
    // printf("%s %lf\n", mat1.c_str(), total_flop_G / t1);

    A.release();
    B.release();
    C.release();
    return 0;
}


