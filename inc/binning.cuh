template <int TYPE>
__device__ __forceinline__ int get_range(int j)
{
    /* ---------------The bin range when calculating B.d_tileptr.--------------- */
    if constexpr (TYPE == 0)
    {
#if SQUARING_B_MASK
        //                hash_size =  0  1  2  3  59  1063 2131  4259  8527  25343  dense
        const int c_range[BIN_SIZE] = {0, 1, 2, 3, 49, 886, 1776, 3549, 7106, 21119, INT_MAX}; // 1.2
#else
        //                hash_size =  0  1  2  3  64  1024 2048  4096  8192  25343  dense
        const int c_range[BIN_SIZE] = {0, 1, 2, 3, 53, 853, 1706, 3413, 6827, 21119, INT_MAX}; // 1.2
#endif
        return c_range[j];
    }
    /* ---------------The bin range when calculating B.d_tilecol and B.d_tilemask.--------------- */
    else if constexpr (TYPE == 1)
    {
#if SQUARING_B_MASK
        //                hash_size =  0  0  0  59  523  1063 2131  4259  12671  dense
        const int c_range[BIN_SIZE] = {0, 1, 2, 49, 436, 886, 1776, 3549, 10559, INT_MAX}; // 1.2
#else
        //                hash_size =  0  0  0  64  512  1024 2048  4096  12671  dense
        const int c_range[BIN_SIZE] = {0, 1, 2, 53, 416, 853, 1706, 3413, 10559, INT_MAX}; // 1.2
#endif
        return c_range[j];
    }
    /* ---------------The bin range when calculating C.d_tileptr.--------------- */
    else if constexpr (TYPE == 2)
    {
#if SQUARING
        //                hash_size =  0  0  59  1063 2131  4259  8527  25343  dense
        const int c_range[BIN_SIZE] = {0, 1, 49, 886, 1776, 3549, 7106, 21119, INT_MAX}; // 1.2
#else
        //                 hash_size = 0  0  64  1024 2048  4096  8192  25343  dense
        const int c_range[BIN_SIZE] = {0, 1, 53, 853, 1706, 3413, 6827, 21119, INT_MAX}; // 1.2
#endif
        return c_range[j];
    }
    /* ---------------The bin range when calculating C nnz.--------------- */
    else if constexpr (TYPE == 3)
    {
#if SQUARING
        //                 hash_size = 0  0  59  523  1063 2131  4259  12671 dense
        const int c_range[BIN_SIZE] = {0, 1, 30, 262, 532, 1066, 2130, 6336, INT_MAX}; // 2
#else
        //                 hash_size = 0  0  64  512  1024 2048  4096  12672 dense
        const int c_range[BIN_SIZE] = {0, 1, 32, 256, 512, 1024, 2048, 6336, INT_MAX}; // 2
#endif
        return c_range[j];
    }
    /* ---------------The bin range when calculating C col and val.--------------- */
    else if constexpr (TYPE == 4)
    {
#if SQUARING
        //                hash_size =  0  0  43  347  691  1376 2843  8447  global
        const int c_range[BIN_SIZE] = {0, 1, 22, 174, 346, 684, 1422, 4224, INT_MAX};
#else
        //                hash_size =  0  0  32  256  512  1024 2048  8192  global
        const int c_range[BIN_SIZE] = {0, 1, 16, 128, 256, 512, 1024, 4096, INT_MAX}; // 2
#endif
        return c_range[j];
    }
    return INT_MAX;
}

template <int TYPE>
__global__ void k_binning1(
    const int *__restrict__ flop,
    int *__restrict__ d_bin_size,
    int M,
    int *__restrict__ d_max_row_nnz)
{
    __shared__ int shared_bin_size[BIN_SIZE];
    __shared__ int shared_max_row_nnz[1];
    if (threadIdx.x < BIN_SIZE)
        shared_bin_size[threadIdx.x] = 0;
    if (threadIdx.x == 0)
        shared_max_row_nnz[0] = 0;
    __syncthreads();
    int rowid = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowid < M)
    {
        int nnz = flop[rowid];
        atomicMax(&shared_max_row_nnz[0], nnz);

        for (int j = 0; j < BIN_SIZE; ++j)
        {
            if (nnz <= get_range<TYPE>(j))
            {
                atomicAdd(&shared_bin_size[j], 1);
                break;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < BIN_SIZE)
        atomicAdd(&d_bin_size[threadIdx.x], shared_bin_size[threadIdx.x]);
    if (threadIdx.x == 0)
    {
        atomicMax(d_max_row_nnz, shared_max_row_nnz[0]);
    }
}
template <int TYPE>
__global__ void k_binning2(
    int *__restrict__ flop,
    int M,
    int *__restrict__ d_bins,
    int *__restrict__ d_bin_size,
    int *__restrict__ d_bin_offset)
{
    __shared__ int shared_bin_size[BIN_SIZE];
    __shared__ int shared_bin_offset[BIN_SIZE];
    if (threadIdx.x < BIN_SIZE)
    {
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    int j, nnz;
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid < M)
    {
        nnz = flop[rowid];
        for (j = 0; j < BIN_SIZE; ++j)
        {
            if (nnz <= get_range<TYPE>(j))
            {
                atomicAdd(&shared_bin_size[j], 1);
                break;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < BIN_SIZE)
    {
        shared_bin_offset[threadIdx.x] = atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
        shared_bin_offset[threadIdx.x] += d_bin_offset[threadIdx.x];
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    int index;
    if (rowid < M)
    {
        for (j = 0; j < BIN_SIZE; j++)
        {
            if (nnz <= get_range<TYPE>(j))
            {
                index = atomicAdd(shared_bin_size + j, 1);
                d_bins[shared_bin_offset[j] + index] = rowid;
                return;
            }
        }
    }
}
