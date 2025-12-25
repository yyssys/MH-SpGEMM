#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>
#include <sys/time.h>

#define BIN_SIZE 13
#define VALUE_TYPE double
// #define index_t unsigned int
#define MASK_TYPE unsigned int

#define INT_MAX __INT_MAX__

#define ADAPTIVE_GROUPING 1
#define BITONIC_SORT 1
#define SQUARING 1
#define SQUARING_B_MASK 1
#define HASH_CONFLICT 0

#if SQUARING_B_MASK
constexpr int hash_size_B_tileptr[4] = {1063, 2131, 4259, 8527};
constexpr int hash_size_B_tileColAndtileMask[4] = {523, 1063, 2131, 4259};
#else
constexpr int hash_size_B_tileptr[4] = {1024, 2048, 4096, 8192};
constexpr int hash_size_B_tileColAndtileMask[4] = {512, 1024, 2048, 4096};
#endif

#if SQUARING
constexpr int hash_size_C_tileptr[4] = {1063, 2131, 4259, 8527};
constexpr int hash_size_C_nnz[4] = {523, 1063, 2131, 4259};
constexpr int hash_size_numeric[4] = {347, 691, 1376, 2843};
#else
constexpr int hash_size_C_tileptr[4] = {1024, 2048, 4096, 8192};
constexpr int hash_size_C_nnz[4] = {512, 1024, 2048, 4096};
constexpr int hash_size_numeric[4] = {256, 512, 1024, 2048};
#endif
#define AAT 0

#if SQUARING
#define PWARP_HASH_SIZE_FOR_CTILEPTR 59
#define PWARP_HASH_SIZE_FOR_C_NNZ 59
#define PWARP_HASH_SIZE_FOR_NUMERIC 43
#else
#define PWARP_HASH_SIZE_FOR_CTILEPTR 64
#define PWARP_HASH_SIZE_FOR_C_NNZ 64
#define PWARP_HASH_SIZE_FOR_NUMERIC 32
#endif

#define PWARP_FOR_B_TILEPTR 4
#define PWARP_ROWS_FOR_B_TILEPTR 128

#define PWARP_FOR_B_MASK 8
#define PWARP_ROWS_FOR_B_MASK 64

#if SQUARING_B_MASK
#define PWARP_HASH_SIZE_FOR_B_TILEPTR 59
#define PWARP_HASH_SIZE_FOR_B_MASK 59
#else
#define PWARP_HASH_SIZE_FOR_B_TILEPTR 64
#define PWARP_HASH_SIZE_FOR_B_MASK 64
#endif

#define PWARP_FOR_C_TILEPTR 4
#define PWARP_ROWS_FOR_C_TILEPTR 128

#define PWARP_FOR_C_NNZ 8
#define PWARP_ROWS_FOR_C_NNZ 64

#define PWARP_FOR_NUMERIC 8
#define PWARP_ROWS_FOR_NUMERIC 64

#define HASH_SCALE 107

#define BLOCK_SIZE_BIT 5 // 32
#define BLOCK_SIZE 32

#define SPGEMM 1
#define CUSPARSE 0
#define CHECK_RESULT 0
#define WRITE 0

// Error Detection
#define likely(x) __builtin_expect(x, 1)
#define unlikely(x) __builtin_expect(x, 0)
inline static void
checkCUDA(cudaError_t err, const char *file, int line)
{
    if (unlikely(err != cudaSuccess))
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        throw std::exception();
    }
}
#define CHECK_ERROR(err) (checkCUDA(err, __FILE__, __LINE__))

#define HP_TIMING_NOW(Var) \
    ({ unsigned int _hi, _lo; \
     asm volatile ("lfence\n\trdtsc" : "=a" (_lo), "=d" (_hi)); \
     (Var) = ((unsigned long long int) _hi << 32) | _lo; })

/* precision is 1 clock cycle.
 * execute time is roughly 50 or 140 cycles depends on cpu family */
inline void cpuid(int *info, int eax, int ecx = 0)
{
    int ax, bx, cx, dx;
    __asm__ __volatile__("cpuid" : "=a"(ax), "=b"(bx), "=c"(cx), "=d"(dx) : "a"(eax));

    info[0] = ax;
    info[1] = bx;
    info[2] = cx;
    info[3] = dx;
}

inline long get_tsc_freq()
{
    static long freq = 0;
    if (unlikely((freq == 0)))
    {
        int raw[4];
        cpuid(raw, 0x16); // get cpu freq
        freq = long(raw[0]) * 1000000;
        // printf("static first call %f\n", freq);
    }
    return freq;
}

inline double fast_clock_time()
{
    long counter;
    HP_TIMING_NOW(counter);
    return double(counter) / get_tsc_freq();
}
