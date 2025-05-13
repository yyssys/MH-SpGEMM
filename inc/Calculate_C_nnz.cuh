__device__ __forceinline__ int round_to_nearest_pow2(unsigned int x)
{
    if (x <= 1)
        return 2;
    int lower = 1u << (31 - __clz(x));
    int upper = lower << 1;
    return (x - lower < upper - x) ? lower : upper;
}
template <int TYPE>
__device__ __forceinline__ int get_range_group_size(int j)
{
    // calculating C tileptr
    if constexpr (TYPE == 0)
    {
#if SQUARING
        const int c_range[5] = {886, 1776, 3549, 7106, 21119};
        return c_range[j];
#else
        const int c_range[5] = {853, 1706, 3413, 6826, 21119};
        return c_range[j];
#endif
    }
    // calculating C nnz
    else if constexpr (TYPE == 1)
    {
#if SQUARING
        const int c_range[5] = {262, 532, 1066, 2130, 6336};
        return c_range[j];
#else
        const int c_range[5] = {256, 512, 1024, 2048, 6336};
        return c_range[j];
#endif
    }
    // calculating C col and val
    else if constexpr (TYPE == 2)
    {
#if SQUARING
        const int c_range[5] = {174, 346, 684, 1422, 4224};
        return c_range[j];
#else
        const int c_range[5] = {128, 256, 512, 1024, 4096};
        return c_range[j];
#endif
    }
    return INT_MAX;
}
template <int TYPE>
__global__ void k_init_group_size(
    const int *ptrA,
    const int *ptrC,
    const int *ref_flop,
    int rows,
    const int *bins,
    int *d_group_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    if (i >= rows)
        return;
    int rowid = bins[i];

    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
    int Arow_nnz = Annz_end - Annz_start;
    int real_flop = ptrC[rowid];

    int block_size;
    for (j = 0; j < 5; j++)
    {
        if (real_flop <= get_range_group_size<TYPE>(j))
        {
            block_size = 64 << j;
            break;
        }
    }
    if (j == 5)
        block_size = 1024;

    int group_size = ref_flop[rowid] / Arow_nnz;
    if (group_size >= block_size)
        group_size = block_size;
    else
        group_size = round_to_nearest_pow2(group_size);
    while (block_size / group_size * 2 > Arow_nnz && group_size < block_size)
        group_size *= 2;
    d_group_size[rowid] = group_size;
}
__global__ void k_calculate_C_tilePtr_shared_hash_pwarp(
    const int *__restrict__ ptrA,
    const int *__restrict__ colA,
    int rows,
    const int *__restrict__ tileptrB,
    const int *__restrict__ tilecolB,
    const int *__restrict__ bins,
    int *__restrict__ tileptrC,
    int *__restrict__ conflict)
{
    int i, k;
    int row_offset = PWARP_ROWS_FOR_C_TILEPTR * blockIdx.x;
    int row_num = blockIdx.x == gridDim.x - 1 ? rows - row_offset : PWARP_ROWS_FOR_C_TILEPTR;
    int local_rowid = threadIdx.x / PWARP_FOR_C_TILEPTR;
    int tid = threadIdx.x & (PWARP_FOR_C_TILEPTR - 1);
    __shared__ int shared_mem[PWARP_ROWS_FOR_C_TILEPTR * PWARP_HASH_SIZE_FOR_CTILEPTR + PWARP_ROWS_FOR_C_TILEPTR];
    int *shared_table = shared_mem;
    int *tmp_num = shared_mem + PWARP_ROWS_FOR_C_TILEPTR * PWARP_HASH_SIZE_FOR_CTILEPTR;

    for (i = threadIdx.x; i < PWARP_ROWS_FOR_C_TILEPTR * PWARP_HASH_SIZE_FOR_CTILEPTR; i += blockDim.x)
    {
        shared_table[i] = -1;
    }
    if (threadIdx.x < PWARP_ROWS_FOR_C_TILEPTR)
    {
        tmp_num[threadIdx.x] = 0;
    }
    __syncthreads();
    if (local_rowid >= row_num)
        return;
    int *table = shared_table + local_rowid * PWARP_HASH_SIZE_FOR_CTILEPTR;
    int rowid = bins[row_offset + local_rowid];
    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
    int Bnnz_start, Bnnz_end;
    int acol, btilecol;
    int hash, old;
    for (i = Annz_start + tid; i < Annz_end; i += PWARP_FOR_C_TILEPTR)
    {
        acol = colA[i];
        Bnnz_start = tileptrB[acol];
        Bnnz_end = tileptrB[acol + 1];
        for (k = Bnnz_start; k < Bnnz_end; k++)
        {
            int j = 1;
            btilecol = tilecolB[k];
#if SQUARING
            hash = (btilecol * HASH_SCALE) % PWARP_HASH_SIZE_FOR_CTILEPTR;
#else
            hash = (btilecol * HASH_SCALE) & (PWARP_HASH_SIZE_FOR_CTILEPTR - 1);
#endif
            while (1)
            {
                old = atomicCAS(table + hash, -1, btilecol);
                if (old == -1)
                {
                    atomicAdd(tmp_num + local_rowid, 1);
                    break;
                }
                else if (old == btilecol)
                {
                    break;
                }
                else
                {
#if HASH_CONFLICT
                    atomicAdd(conflict, 1);
#endif
#if SQUARING
                    hash = (hash + j * j) % PWARP_HASH_SIZE_FOR_CTILEPTR;
                    j++;
#else
                    hash = (hash + 1) & (PWARP_HASH_SIZE_FOR_CTILEPTR - 1);
#endif
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        tileptrC[rowid] = tmp_num[local_rowid];
    }
}

template <int HASH_SIZE>
__global__ void k_calculate_C_tilePtr_shared_hash_tb(
    const int *__restrict__ ptrA,
    const int *__restrict__ colA,
    const int *__restrict__ tileptrB,
    const int *__restrict__ tilecolB,
    const int *__restrict__ bins,
    int *__restrict__ tileptrC,
    int *__restrict__ conflict,
    const int *__restrict__ d_group_size)
{
    int i, k;
    int row_offset = blockIdx.x;
    int rowid = bins[row_offset];
    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
#if ADAPTIVE_GROUPING
    int group_size = d_group_size[rowid];
#else
    int group_size = 32;
#endif
    int group_id = threadIdx.x / group_size;
    int group_num = blockDim.x / group_size;
    int tid = threadIdx.x & (group_size - 1);

    __shared__ int shared_table[HASH_SIZE];
    __shared__ int tmp_num;
    for (i = threadIdx.x; i < HASH_SIZE; i += blockDim.x)
    {
        shared_table[i] = -1;
    }
    if (threadIdx.x == 0)
    {
        tmp_num = 0;
    }
    __syncthreads();
    int Bnnz_start, Bnnz_end;
    int acol, btilecol;
    int hash, old;
    for (i = Annz_start + group_id; i < Annz_end; i += group_num)
    {
        acol = colA[i];
        Bnnz_start = tileptrB[acol];
        Bnnz_end = tileptrB[acol + 1];
        for (k = Bnnz_start + tid; k < Bnnz_end; k += group_size)
        {
            int j = 1;
            btilecol = tilecolB[k];
#if SQUARING
            hash = (btilecol * HASH_SCALE) % HASH_SIZE;
#else
            hash = (btilecol * HASH_SCALE) & (HASH_SIZE - 1);
#endif
            while (1)
            {
                old = atomicCAS(shared_table + hash, -1, btilecol);
                if (old == -1)
                {
                    atomicAdd(&tmp_num, 1);
                    break;
                }
                else if (old == btilecol)
                {
                    break;
                }
                else
                {
#if HASH_CONFLICT
                    atomicAdd(conflict, 1);
#endif
#if SQUARING
                    hash = (hash + j * j) % HASH_SIZE;
                    j++;
#else
                    hash = (hash + 1) & (HASH_SIZE - 1);
#endif
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        tileptrC[rowid] = tmp_num;
    }
}

__global__ void k_calculate_C_tilePtr_max_shared(
    const int *__restrict__ ptrA,
    const int *__restrict__ colA,
    const int *__restrict__ tileptrB,
    const int *__restrict__ tilecolB,
    const int *__restrict__ bins,
    int *__restrict__ tileptrC,
    int *__restrict__ conflict,
    int *__restrict__ d_group_size)
{
    int i, k;
    int row_offset = blockIdx.x;
    int rowid = bins[row_offset];
    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
#if ADAPTIVE_GROUPING
    int group_size = d_group_size[rowid];
#else
    int group_size = 32;
#endif
    int group_id = threadIdx.x / group_size;
    int group_num = blockDim.x / group_size;
    int tid = threadIdx.x & (group_size - 1);

    const int tsize = 25343;
    extern __shared__ int shared_mem[];
    int *shared_col = shared_mem;
    int *tmp_num = shared_mem + tsize;

    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        shared_col[i] = -1;
    }
    if (threadIdx.x == 0)
    {
        tmp_num[0] = 0;
    }
    __syncthreads();

    int Bnnz_start, Bnnz_end;
    int acol, btilecol;
    int hash, old;
    for (i = Annz_start + group_id; i < Annz_end; i += group_num)
    {
        acol = colA[i];
        Bnnz_start = tileptrB[acol];
        Bnnz_end = tileptrB[acol + 1];

        for (k = Bnnz_start + tid; k < Bnnz_end; k += group_size)
        {
            int j = 1;
            btilecol = tilecolB[k];
            hash = (btilecol * HASH_SCALE) % tsize;
            while (1)
            {
                old = atomicCAS(shared_col + hash, -1, btilecol);
                if (old == -1)
                {
                    atomicAdd(tmp_num, 1);
                    break;
                }
                else if (old == btilecol)
                {
                    break;
                }
                else
                {
#if HASH_CONFLICT
                    atomicAdd(conflict, 1);
#endif
#if SQUARING
                    hash = (hash + j * j) % tsize;
                    j++;
#else
                    hash = hash + 1 < tsize ? hash + 1 : 0;
#endif
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        tileptrC[rowid] = tmp_num[0];
    }
}

__global__ void k_calculate_C_tilePtr_global_mem(
    const int *__restrict__ ptrA,
    const int *__restrict__ colA,
    const int *__restrict__ tileptrB,
    const int *__restrict__ tilecolB,
    const int *__restrict__ bins,
    int *__restrict__ d_col_table,
    int *__restrict__ tileptrC,
    int col_size,
    const int *__restrict__ d_group_size)
{
    int row_offset = blockIdx.x;
    int rowid = bins[row_offset];
    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
#if ADAPTIVE_GROUPING
    int group_size = d_group_size[rowid];
#else
    int group_size = 32;
#endif
    int group_num = blockDim.x / group_size;
    int group_id = threadIdx.x / group_size;
    int tid = threadIdx.x & (group_size - 1);
    int j, k;
    __shared__ int shared_nnz[1];

    int *col_table = d_col_table + blockIdx.x * col_size;
    for (j = threadIdx.x; j < col_size; j += blockDim.x)
    {
        col_table[j] = -1;
    }
    if (threadIdx.x == 0)
    {
        shared_nnz[0] = 0;
    }
    __syncthreads();

    int Bnnz_start, Bnnz_end;
    int acol, btilecol;
    for (j = Annz_start + group_id; j < Annz_end; j += group_num)
    {
        acol = colA[j];
        Bnnz_start = tileptrB[acol];
        Bnnz_end = tileptrB[acol + 1];

        for (k = Bnnz_start + tid; k < Bnnz_end; k += group_size)
        {
            btilecol = tilecolB[k];
            if (atomicCAS(col_table + btilecol, -1, btilecol) == -1)
            {
                atomicAdd(shared_nnz, 1);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        tileptrC[rowid] = shared_nnz[0];
    }
}

__global__ void k_calculate_C_nnz_for_one_tile(
    const int *__restrict__ ptrA,
    const int *__restrict__ colA,
    const int *__restrict__ tileptrB,
    const int *__restrict__ tilecolB,
    const MASK_TYPE *__restrict__ tilemaskB,
    int M,
    const int *__restrict__ tileptrC,
    const int *__restrict__ bins,
    int *__restrict__ ptrC)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M)
        return;
    int rowid = bins[i];
    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
    int tilecol;
    MASK_TYPE tilemask = 0;
    int Bnnz_idx, col;
    col = colA[Annz_start];
    Bnnz_idx = tileptrB[col];
    tilecol = tilecolB[Bnnz_idx];
    tilemask |= tilemaskB[Bnnz_idx];
    for (i = Annz_start + 1; i < Annz_end; i++)
    {
        col = colA[i];
        Bnnz_idx = tileptrB[col];
        tilemask |= tilemaskB[Bnnz_idx];
    }
    ptrC[rowid] = __popc(tilemask);
}

__global__ void k_calculate_C_nnz_shared_hash_pwarp(
    const int *__restrict__ ptrA,
    const int *__restrict__ colA,
    const int *__restrict__ tileptrB,
    const int *__restrict__ tilecolB,
    const MASK_TYPE *__restrict__ tilemaskB,
    int M,
    const int *__restrict__ tileptrC,
    const int *__restrict__ bins,
    int *__restrict__ ptrC,
    int *__restrict__ conflict)
{
    int i, k;
    int row_offset = PWARP_ROWS_FOR_C_NNZ * blockIdx.x;
    int row_num = blockIdx.x == gridDim.x - 1 ? M - row_offset : PWARP_ROWS_FOR_C_NNZ;
    int local_rowid = threadIdx.x / PWARP_FOR_C_NNZ;
    int tid = threadIdx.x & (PWARP_FOR_C_NNZ - 1);
    const int tsize = PWARP_HASH_SIZE_FOR_C_NNZ * PWARP_ROWS_FOR_C_NNZ;
    __shared__ int shared_mem[tsize * (sizeof(int) + sizeof(MASK_TYPE)) / sizeof(int)];
    int *shared_col = shared_mem;
    MASK_TYPE *shared_mask = (MASK_TYPE *)(shared_mem + tsize);
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        shared_col[i] = -1;
        shared_mask[i] = 0;
    }
    __syncthreads();
    if (local_rowid >= row_num)
        return;
    int *table_shared_col = shared_col + local_rowid * PWARP_HASH_SIZE_FOR_C_NNZ;
    MASK_TYPE *table_shared_mask = shared_mask + local_rowid * PWARP_HASH_SIZE_FOR_C_NNZ;
    int rowid = bins[row_offset + local_rowid];
    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
    int Bnnz_start, Bnnz_end;
    int acol, btilecol;
    MASK_TYPE btilemask;
    int hash, old;
    for (i = Annz_start + tid; i < Annz_end; i += PWARP_FOR_C_NNZ)
    {
        acol = colA[i];
        Bnnz_start = tileptrB[acol];
        Bnnz_end = tileptrB[acol + 1];
        for (k = Bnnz_start; k < Bnnz_end; k++)
        {
            int j = 1;
            btilecol = tilecolB[k];
            btilemask = tilemaskB[k];
#if SQUARING
            hash = (btilecol * HASH_SCALE) % PWARP_HASH_SIZE_FOR_C_NNZ;
#else
            hash = (btilecol * HASH_SCALE) & (PWARP_HASH_SIZE_FOR_C_NNZ - 1);
#endif
            while (1)
            {
                old = atomicCAS(table_shared_col + hash, -1, btilecol);
                if (old == -1 || old == btilecol)
                {
                    atomicOr(table_shared_mask + hash, btilemask);
                    break;
                }
                else
                {
#if HASH_CONFLICT
                    atomicAdd(conflict, 1);
#endif
#if SQUARING
                    hash = (hash + j * j) % PWARP_HASH_SIZE_FOR_C_NNZ;
                    j++;
#else
                    hash = (hash + 1) & (PWARP_HASH_SIZE_FOR_C_NNZ - 1);
#endif
                }
            }
        }
    }
    __syncthreads();
    int sum = 0;
    for (i = tid; i < PWARP_HASH_SIZE_FOR_C_NNZ; i += PWARP_FOR_C_NNZ)
    {
        if (table_shared_col[i] != -1)
        {
            sum += __popc(table_shared_mask[i]);
        }
    }
    atomicAdd(ptrC + rowid, sum);
}

template <int HASH_SIZE>
__global__ void k_calculate_C_nnz_shared_hash_tb(
    const int *__restrict__ ptrA,
    const int *__restrict__ colA,
    const int *__restrict__ tileptrB,
    const int *__restrict__ tilecolB,
    const MASK_TYPE *__restrict__ tilemaskB,
    const int *__restrict__ tileptrC,
    const int *__restrict__ bins,
    int *__restrict__ ptrC,
    int *__restrict__ conflict,
    const int *__restrict__ d_group_size)
{
    int i, k;
    __shared__ MASK_TYPE shared_mask[HASH_SIZE];
    __shared__ int shared_col[HASH_SIZE];
    for (i = threadIdx.x; i < HASH_SIZE; i += blockDim.x)
    {
        shared_col[i] = -1;
        shared_mask[i] = 0;
    }
    __syncthreads();

    int rowid = bins[blockIdx.x];
    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
#if ADAPTIVE_GROUPING
    int group_size = d_group_size[rowid];
#else
    int group_size = 32;
#endif
    int group_id = threadIdx.x / group_size;
    int group_num = blockDim.x / group_size;
    int tid = threadIdx.x & (group_size - 1);
    int Bnnz_start, Bnnz_end;
    int acol, btilecol;
    MASK_TYPE btilemask;
    int hash, old;
    for (i = Annz_start + group_id; i < Annz_end; i += group_num)
    {
        acol = colA[i];
        Bnnz_start = tileptrB[acol];
        Bnnz_end = tileptrB[acol + 1];

        for (k = Bnnz_start + tid; k < Bnnz_end; k += group_size)
        {
            int j = 1;
            btilecol = tilecolB[k];
            btilemask = tilemaskB[k];
#if SQUARING
            hash = (btilecol * HASH_SCALE) % HASH_SIZE;
#else
            hash = (btilecol * HASH_SCALE) & (HASH_SIZE - 1);
#endif
            while (1)
            {
                old = atomicCAS(shared_col + hash, -1, btilecol);
                if (old == -1 || old == btilecol)
                {
                    atomicOr(shared_mask + hash, btilemask);
                    break;
                }
                else
                {
#if HASH_CONFLICT
                    atomicAdd(conflict, 1);
#endif
#if SQUARING
                    hash = (hash + j * j) % HASH_SIZE;
                    j++;
#else
                    hash = (hash + 1) & (HASH_SIZE - 1);
#endif
                }
            }
        }
    }
    __syncthreads();
    int local_sum = 0;
    for (i = threadIdx.x; i < HASH_SIZE; i += blockDim.x)
    {
        if (shared_col[i] != -1)
        {
            local_sum += __popc(shared_mask[i]);
        }
    }
    int tid_t = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warp_num = blockDim.x >> 5;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (tid_t == 0)
        shared_col[warp_id] = local_sum;
    __syncthreads();
    local_sum = (threadIdx.x < warp_num) ? shared_col[threadIdx.x] : 0;
    if (warp_id == 0)
    {
        for (int offset = 16; offset > 0; offset /= 2)
        {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }
    if (threadIdx.x == 0)
    {
        ptrC[rowid] = local_sum;
    }
}

__global__ void k_calculate_C_nnz_max_shared(
    const int *__restrict__ ptrA,
    const int *__restrict__ colA,
    const int *__restrict__ tileptrB,
    const int *__restrict__ tilecolB,
    const MASK_TYPE *__restrict__ tilemaskB,
    const int *__restrict__ tileptrC,
    const int *__restrict__ bins,
    int *__restrict__ ptrC,
    int *__restrict__ conflict,
    const int *__restrict__ d_group_size)
{
    int i, k;
#if SQUARING
    int tsize = 12671;
#else
    int tsize = 12672;
#endif
    extern __shared__ int shared_mem[];
    int *shared_col = shared_mem;
    MASK_TYPE *shared_mask = (MASK_TYPE *)(shared_col + tsize);
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        shared_col[i] = -1;
        shared_mask[i] = 0;
    }
    __syncthreads();
    int rowid = bins[blockIdx.x];
    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
#if ADAPTIVE_GROUPING
    int group_size = d_group_size[rowid];
#else
    int group_size = 32;
#endif
    int group_id = threadIdx.x / group_size;
    int group_num = blockDim.x / group_size;
    int tid = threadIdx.x & (group_size - 1);

    int Bnnz_start, Bnnz_end;
    int acol, btilecol;
    MASK_TYPE btilemask;
    int hash, old;
    for (i = Annz_start + group_id; i < Annz_end; i += group_num)
    {
        acol = colA[i];
        Bnnz_start = tileptrB[acol];
        Bnnz_end = tileptrB[acol + 1];

        for (k = Bnnz_start + tid; k < Bnnz_end; k += group_size)
        {
            int j = 1;
            btilecol = tilecolB[k];
            btilemask = tilemaskB[k];
            hash = (btilecol * HASH_SCALE) % tsize;
            while (1)
            {
                old = atomicCAS(shared_col + hash, -1, btilecol);
                if (old == -1 || old == btilecol)
                {
                    atomicOr(shared_mask + hash, btilemask);
                    break;
                }
                else
                {
#if HASH_CONFLICT
                    atomicAdd(conflict, 1);
#endif
#if SQUARING
                    hash = (hash + j * j) % tsize;
                    j++;
#else
                    hash = (hash + 1) < tsize ? (hash + 1) : 0;
#endif
                }
            }
        }
    }
    __syncthreads();
    int local_sum = 0;
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        if (shared_col[i] != -1)
        {
            local_sum += __popc(shared_mask[i]);
        }
    }
    int tid_t = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warp_num = blockDim.x >> 5;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (tid_t == 0)
        shared_col[warp_id] = local_sum;
    __syncthreads();
    local_sum = (threadIdx.x < warp_num) ? shared_col[threadIdx.x] : 0;
    if (warp_id == 0)
    {
        for (int offset = 16; offset > 0; offset /= 2)
        {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }
    if (threadIdx.x == 0)
    {
        ptrC[rowid] = local_sum;
    }
}

__global__ void k_calculate_C_tileColAndtileMask_global_mem(
    const int *__restrict__ ptrA,
    const int *__restrict__ colA,
    const int *__restrict__ tileptrB,
    const int *__restrict__ tilecolB,
    const MASK_TYPE *__restrict__ tilemaskB,
    const int *__restrict__ tileptrC,
    const int *__restrict__ bins,
    int *__restrict__ ptrC,
    int N,
    int *__restrict__ d_global_mem,
    const int *__restrict__ d_group_size)
{
    int i, k;
    int row_offset = blockIdx.x;
    int *local_mask_offset = d_global_mem + row_offset * N * sizeof(MASK_TYPE);
    __shared__ int shared_scan[32];

    for (i = threadIdx.x; i < N; i += blockDim.x)
    {
        local_mask_offset[i] = 0;
    }
    __syncthreads();
    int rowid = bins[row_offset];
    int Annz_start = ptrA[rowid];
    int Annz_end = ptrA[rowid + 1];
#if ADAPTIVE_GROUPING
    int group_size = d_group_size[rowid];
#else
    int group_size = 32;
#endif
    int group_id = threadIdx.x / group_size;
    int group_num = blockDim.x / group_size;
    int tid = threadIdx.x & (group_size - 1);
    int Bnnz_start, Bnnz_end;
    int acol;
    int hash, old;
    for (i = Annz_start + group_id; i < Annz_end; i += group_num)
    {
        acol = colA[i];
        Bnnz_start = tileptrB[acol];
        Bnnz_end = tileptrB[acol + 1];

        for (k = Bnnz_start + tid; k < Bnnz_end; k += group_size)
        {
            atomicOr(local_mask_offset + tilecolB[k], tilemaskB[k]);
        }
    }
    __syncthreads();
    int local_sum = 0;
    for (i = threadIdx.x; i < N; i += blockDim.x)
    {
        if (local_mask_offset[i] != 0)
        {
            local_sum += __popc(local_mask_offset[i]);
        }
    }
    __syncthreads();
    int tid_t = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warp_num = blockDim.x >> 5;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (tid_t == 0)
        shared_scan[warp_id] = local_sum;
    __syncthreads();
    local_sum = (threadIdx.x < warp_num) ? shared_scan[threadIdx.x] : 0;
    if (warp_id == 0)
    {
        for (int offset = 16; offset > 0; offset /= 2)
        {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }
    if (threadIdx.x == 0)
    {
        ptrC[rowid] = local_sum;
    }
}