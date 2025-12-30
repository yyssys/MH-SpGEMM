__global__ void k_calculate_B_per_row_nnz(
    const int *__restrict__ ptrB,
    int *__restrict__ per_row_nnz,
    int M)
{
    int rowid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowid >= M)
        return;

    int nnz = ptrB[rowid + 1] - ptrB[rowid];
    per_row_nnz[rowid] = nnz;
}

__global__ void k_calculate_flop(
    const int *__restrict__ d_arpt,
    const int *__restrict__ d_acol,
    const int *__restrict__ d_tileptrB,
    int M,
    int *__restrict__ d_row_flop)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    int global_rowid = i >> 2;
    int local_rowid = threadIdx.x >> 2;
    int tid = threadIdx.x & 3;

    __shared__ int row_flop[128];

    if (threadIdx.x < 128)
    {
        row_flop[threadIdx.x] = 0;
    }
    __syncthreads();

    if (global_rowid < M)
    {
        int arow_start = d_arpt[global_rowid];
        int arow_end = d_arpt[global_rowid + 1];
        int sum = 0, acol, max_bnnz = 0;
        for (j = arow_start + tid; j < arow_end; j += 4)
        {
            acol = d_acol[j];
            int nnz = d_tileptrB[acol];
            sum += nnz;
        }
        atomicAdd(&row_flop[local_rowid], sum);
        __syncthreads();

        if (tid == 0)
        {
            d_row_flop[global_rowid] = row_flop[local_rowid];
        }
    }
}

__global__ void k_calculate_flop_tmp(
    const int *__restrict__ d_arpt,
    const int *__restrict__ d_acol,
    const int *__restrict__ d_ptrB,
    int M,
    int *__restrict__ d_row_flop)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int global_rowid = i >> 2;
    int local_rowid = threadIdx.x >> 2;
    int tid = threadIdx.x & 3;

    __shared__ int row_flop[128];

    if (threadIdx.x < 128)
    {
        row_flop[threadIdx.x] = 0;
    }
    __syncthreads();

    if (global_rowid < M)
    {
        int arow_start = d_arpt[global_rowid];
        int arow_end = d_arpt[global_rowid + 1];
        int sum = 0, acol, max_bnnz = 0;
        for (int j = arow_start + tid; j < arow_end; j += 4)
        {
            acol = d_acol[j];
            int nnz = d_ptrB[acol + 1] - d_ptrB[acol];
            sum += nnz;
        }
        atomicAdd(&row_flop[local_rowid], sum);
        __syncthreads();

        if (tid == 0)
        {
            d_row_flop[global_rowid] = row_flop[local_rowid];
        }
    }
}

__global__ void k_calculate_tilePtr_one_nnz(
    int rows,
    int *__restrict__ ptr,
    const int *__restrict__ bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows)
    {
        ptr[bins[i]] = 1;
    }
}

template <int NNZ>
__global__ void k_calculate_tilePtr_two_three_nnz(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    int rows,
    int *__restrict__ ptr,
    const int *__restrict__ bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows)
        return;

    int rowid = bins[i];
    int start = ptrB[rowid];

    if constexpr (NNZ == 2)
    {
        int b0 = colB[start] >> BLOCK_SIZE_BIT;
        int b1 = colB[start + 1] >> BLOCK_SIZE_BIT;
        ptr[rowid] = (b0 == b1) ? 1 : 2;
    }
    else if constexpr (NNZ == 3)
    {
        int b0 = colB[start] >> BLOCK_SIZE_BIT;
        int b1 = colB[start + 1] >> BLOCK_SIZE_BIT;
        int b2 = colB[start + 2] >> BLOCK_SIZE_BIT;

        int count = 3;
        if (b0 == b1 && b1 == b2)
            count = 1;
        else if (b0 == b1 || b0 == b2 || b1 == b2)
            count = 2;
        ptr[rowid] = count;
    }
}

__global__ void k_calculate_B_tilePtr_shared_hash_pwarp(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    int rows,
    int *__restrict__ tileptrB,
    int *__restrict__ bins)
{
    int i;
    int row_offset = PWARP_ROWS_FOR_B_TILEPTR * blockIdx.x;
    int row_num = blockIdx.x == gridDim.x - 1 ? rows - row_offset : PWARP_ROWS_FOR_B_TILEPTR;
    int local_rowid = threadIdx.x >> 2;
    int tid = threadIdx.x & 3;
    const int tsize = PWARP_ROWS_FOR_B_TILEPTR * PWARP_HASH_SIZE_FOR_B_TILEPTR;
    __shared__ int shared_mem[tsize + PWARP_ROWS_FOR_B_TILEPTR];
    int *shared_table = shared_mem;
    int *tmp_num = shared_mem + tsize;

    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        shared_table[i] = -1;
    }
    if (threadIdx.x < PWARP_ROWS_FOR_B_TILEPTR)
    {
        tmp_num[threadIdx.x] = 0;
    }
    __syncthreads();
    if (local_rowid >= row_num)
        return;
    int *table = shared_table + local_rowid * PWARP_HASH_SIZE_FOR_B_TILEPTR;
    int rowid = bins[row_offset + local_rowid];
    int nnz_start = ptrB[rowid];
    int nnz_end = ptrB[rowid + 1];
    int c, tilecol;
    int hash, old;
    for (i = nnz_start + tid; i < nnz_end; i += PWARP_FOR_B_TILEPTR)
    {
        int j = 1;
        c = colB[i];
        tilecol = c >> BLOCK_SIZE_BIT;
#if SQUARING_B_MASK
        hash = (tilecol * HASH_SCALE) % PWARP_HASH_SIZE_FOR_B_TILEPTR;
#else
        hash = (tilecol * HASH_SCALE) & (PWARP_HASH_SIZE_FOR_B_TILEPTR - 1);
#endif
        while (1)
        {
            old = atomicCAS(table + hash, -1, tilecol);
            if (old == -1)
            {
                atomicAdd(tmp_num + local_rowid, 1);
                break;
            }
            else if (old == tilecol)
            {
                break;
            }
            else
            {
#if SQUARING_B_MASK
                hash = (hash + j * j) % PWARP_HASH_SIZE_FOR_B_TILEPTR;
                j++;
#else
                // hash = (hash + 1) & (PWARP_HASH_SIZE_FOR_B_TILEPTR - 1);
                hash = (hash + 1) < PWARP_HASH_SIZE_FOR_B_TILEPTR ? (hash + 1) : 0;
#endif
            }
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        tileptrB[rowid] = tmp_num[local_rowid];
    }
}

template <int HASH_SIZE>
__global__ void k_calculate_B_tilePtr_shared_hash_tb(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    int *__restrict__ tileptrB,
    int *__restrict__ bins)
{
    int row_offset = blockIdx.x;
    __shared__ int shared_table[HASH_SIZE];
    __shared__ int tmp_num[1];
    int i;
    for (i = threadIdx.x; i < HASH_SIZE; i += blockDim.x)
    {
        shared_table[i] = -1;
    }
    if (threadIdx.x == 0)
    {
        tmp_num[0] = 0;
    }
    __syncthreads();
    int rowid = bins[row_offset];
    int nnz_start = ptrB[rowid];
    int nnz_end = ptrB[rowid + 1];
    int c, tilecol;
    int hash, old;
    for (i = nnz_start + threadIdx.x; i < nnz_end; i += blockDim.x)
    {
        int j = 1;
        c = colB[i];
        tilecol = c >> BLOCK_SIZE_BIT;
#if SQUARING_B_MASK
        hash = (tilecol * HASH_SCALE) % HASH_SIZE;
#else
        hash = (tilecol * HASH_SCALE) & (HASH_SIZE - 1);
#endif
        while (1)
        {
            old = atomicCAS(shared_table + hash, -1, tilecol);
            if (old == -1)
            {
                atomicAdd(tmp_num, 1);
                break;
            }
            else if (old == tilecol)
            {
                break;
            }
            else
            {
#if SQUARING_B_MASK
                hash = (hash + j * j) % HASH_SIZE;
                j++;
#else
                // hash = (hash + 1) & (HASH_SIZE - 1);
                hash = (hash + 1) < HASH_SIZE ? (hash + 1) : 0;
#endif
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        tileptrB[rowid] = tmp_num[0];
    }
}

__global__ void k_calculate_B_tilePtr_max_shared(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    int *__restrict__ tileptrB,
    int *__restrict__ bins)
{
    int row_offset = blockIdx.x;
    const int tsize = 25343;
    extern __shared__ int shared_mem[];
    int *shared_col = shared_mem;
    int *tmp_num = shared_mem + tsize;
    int i, k;
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        shared_col[i] = -1;
    }
    if (threadIdx.x == 0)
    {
        tmp_num[0] = 0;
    }
    __syncthreads();
    int rowid = bins[row_offset];
    int nnz_start = ptrB[rowid];
    int nnz_end = ptrB[rowid + 1];
    int c, tilecol;
    int hash, old;
    for (i = nnz_start + threadIdx.x; i < nnz_end; i += blockDim.x)
    {
        int j = 1;
        c = colB[i];
        tilecol = c >> BLOCK_SIZE_BIT;
        hash = (tilecol * HASH_SCALE) % tsize;
        while (1)
        {
            old = atomicCAS(shared_col + hash, -1, tilecol);
            if (old == -1)
            {
                atomicAdd(tmp_num, 1);
                break;
            }
            else if (old == tilecol)
            {
                break;
            }
            else
            {
#if SQUARING_B_MASK
                hash = (hash + j * j) % tsize;
                j++;
#else
                hash = hash + 1 < tsize ? hash + 1 : 0;
#endif
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        tileptrB[rowid] = tmp_num[0];
    }
}

__global__ void k_calculate_B_tilePtr_global_mem(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    int *__restrict__ tileptrB,
    int *__restrict__ d_global_mem,
    int *__restrict__ bins,
    int N)
{
    int row_offset = blockIdx.x;
    int *shared_table = d_global_mem + N * row_offset;
    __shared__ int tmp_num[1];
    int i;
    for (i = threadIdx.x; i < N; i += blockDim.x)
    {
        shared_table[i] = -1;
    }
    if (threadIdx.x == 0)
    {
        tmp_num[0] = 0;
    }
    __syncthreads();
    int rowid = bins[row_offset];
    int nnz_start = ptrB[rowid];
    int nnz_end = ptrB[rowid + 1];
    int c, tilecol;
    for (i = nnz_start + threadIdx.x; i < nnz_end; i += blockDim.x)
    {
        int j = 1;
        c = colB[i];
        tilecol = c >> BLOCK_SIZE_BIT;
        if (atomicCAS(shared_table + tilecol, -1, tilecol) == -1)
        {
            atomicAdd(tmp_num, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        tileptrB[rowid] = tmp_num[0];
    }
}

__global__ void k_calculate_B_tileColAndtileMask_for_one_tile(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    int rows,
    const int *__restrict__ tileptrB,
    int *__restrict__ tilecolB,
    MASK_TYPE *__restrict__ tilemaskB,
    const int *__restrict__ bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows)
        return;

    int rowid = bins[i];
    int row_offset = ptrB[rowid];
    int row_end = ptrB[rowid + 1];

    MASK_TYPE mask = 0;
    int first_col = colB[row_offset];
    int tilecol = first_col >> BLOCK_SIZE_BIT;
    mask = (1UL << (first_col & (BLOCK_SIZE - 1)));

    for (int j = row_offset + 1; j < row_end; j++)
    {
        int bit_pos = colB[j] & (BLOCK_SIZE - 1);
        mask |= (1UL << bit_pos);
    }

    int offset = tileptrB[rowid];
    tilecolB[offset] = tilecol;
    tilemaskB[offset] = mask;
}

__global__ void k_calculate_B_tileColAndtileMask_for_two_tile(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    int rows,
    const int *__restrict__ tileptrB,
    int *__restrict__ tilecolB,
    MASK_TYPE *__restrict__ tilemaskB,
    const int *__restrict__ bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows)
        return;

    int bit_pos1, bit_pos2;
    int rowid = bins[i];
    int row_offset = ptrB[rowid];
    int row_end = ptrB[rowid + 1];

    MASK_TYPE mask1 = 0, mask2 = 0;
    int tilecol1 = colB[row_offset] >> BLOCK_SIZE_BIT;
    bit_pos1 = colB[row_offset] & (BLOCK_SIZE - 1);
    mask1 = (1UL << bit_pos1);
    int tilecol2 = colB[row_end - 1] >> BLOCK_SIZE_BIT;
    bit_pos2 = colB[row_end - 1] & (BLOCK_SIZE - 1);
    mask2 = (1UL << bit_pos2);

    for (int j = row_offset + 1; j < row_end - 1; j++)
    {
        int col = colB[j];
        int current_tilecol = col >> BLOCK_SIZE_BIT;
        bit_pos1 = col & (BLOCK_SIZE - 1);
        if (current_tilecol == tilecol1)
        {
            mask1 |= (1UL << bit_pos1);
        }
        else
        {
            mask2 |= (1UL << bit_pos1);
        }
    }

    int offset = tileptrB[rowid];

    tilecolB[offset] = tilecol1;
    tilemaskB[offset] = mask1;

    tilecolB[offset + 1] = tilecol2;
    tilemaskB[offset + 1] = mask2;
}

__global__ void k_calculate_B_tileColAndtileMask_shared_hash_pwarp(
    const int *ptrB,
    const int *colB,
    int rows,
    const int *tileptrB,
    int *tilecolB,
    MASK_TYPE *tilemaskB,
    const int *bins)
{
    int i;
    int row_offset = PWARP_ROWS_FOR_B_MASK * blockIdx.x;
    int row_num = blockIdx.x == gridDim.x - 1 ? rows - row_offset : PWARP_ROWS_FOR_B_MASK;
    int local_rowid = threadIdx.x >> 3;
    int tid = threadIdx.x & 7;
    const int tsize = PWARP_HASH_SIZE_FOR_B_MASK * PWARP_ROWS_FOR_B_MASK;
    __shared__ int shared_mem[tsize * (sizeof(int) + sizeof(MASK_TYPE)) / sizeof(int) + PWARP_ROWS_FOR_B_MASK];
    int *shared_col = (int *)shared_mem;
    int *tmp_num = shared_mem + tsize;
    MASK_TYPE *shared_mask = (MASK_TYPE *)(shared_mem + tsize + PWARP_ROWS_FOR_B_MASK);
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        shared_col[i] = -1;
        shared_mask[i] = 0;
    }
    if (threadIdx.x < PWARP_ROWS_FOR_B_MASK)
    {
        tmp_num[threadIdx.x] = 0;
    }
    __syncthreads();
    if (local_rowid >= row_num)
        return;
    int *table_shared_col = shared_col + local_rowid * PWARP_HASH_SIZE_FOR_B_MASK;
    MASK_TYPE *table_shared_mask = shared_mask + local_rowid * PWARP_HASH_SIZE_FOR_B_MASK;
    int rowid = bins[row_offset + local_rowid];
    int nnz_start = ptrB[rowid];
    int nnz_end = ptrB[rowid + 1];
    int c, tilecol;
    int old, hash;
    for (i = nnz_start + tid; i < nnz_end; i += PWARP_FOR_B_MASK)
    {
        MASK_TYPE tmp = 1;
        int j = 1;
        c = colB[i];
        tilecol = c >> BLOCK_SIZE_BIT;
        tmp = tmp << (c & BLOCK_SIZE - 1);
#if SQUARING_B_MASK
        hash = (tilecol * HASH_SCALE) % PWARP_HASH_SIZE_FOR_B_MASK;
#else
        hash = (tilecol * HASH_SCALE) & (PWARP_HASH_SIZE_FOR_B_MASK - 1);
#endif
        while (1)
        {
            old = atomicCAS(table_shared_col + hash, -1, tilecol);
            if (old == -1 || old == tilecol)
            {
                atomicOr(table_shared_mask + hash, tmp);
                break;
            }
            else
            {
#if SQUARING_B_MASK
                hash = (hash + j * j) % PWARP_HASH_SIZE_FOR_B_MASK;
                j++;
#else
                // hash = (hash + 1) & (PWARP_HASH_SIZE_FOR_B_MASK - 1);
                hash = (hash + 1) < PWARP_HASH_SIZE_FOR_B_MASK ? (hash + 1) : 0;
#endif
            }
        }
    }
    __syncthreads();
    int offset = tileptrB[rowid];
    int offset_t;
    for (i = tid; i < PWARP_HASH_SIZE_FOR_B_MASK; i += PWARP_FOR_B_MASK)
    {
        if (table_shared_col[i] != -1)
        {
            offset_t = atomicAdd(tmp_num + local_rowid, 1);
            tilecolB[offset + offset_t] = table_shared_col[i];
            tilemaskB[offset + offset_t] = table_shared_mask[i];
        }
    }
}

template <int HASH_SIZE>
__global__ void k_calculate_B_tileColAndtileMask_shared_hash_tb(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    const int *__restrict__ tileptrB,
    int *__restrict__ tilecolB,
    MASK_TYPE *__restrict__ tilemaskB,
    const int *__restrict__ bins)
{
    int row_offset = blockIdx.x;
    __shared__ int shared_col[HASH_SIZE];
    __shared__ MASK_TYPE shared_mask[HASH_SIZE];
    __shared__ int tmp_num[1];
    int i;
    for (i = threadIdx.x; i < HASH_SIZE; i += blockDim.x)
    {
        shared_col[i] = -1;
        shared_mask[i] = 0;
    }
    if (threadIdx.x == 0)
    {
        tmp_num[0] = 0;
    }
    __syncthreads();
    int rowid = bins[row_offset];
    int nnz_start = ptrB[rowid];
    int nnz_end = ptrB[rowid + 1];
    int c, tilecol;
    int old, hash;

    for (i = nnz_start + threadIdx.x; i < nnz_end; i += blockDim.x)
    {
        int j = 1;
        MASK_TYPE tmp = 1;
        c = colB[i];
        tilecol = c >> BLOCK_SIZE_BIT;
        tmp = tmp << (c & BLOCK_SIZE - 1);
#if SQUARING_B_MASK
        hash = (tilecol * HASH_SCALE) % HASH_SIZE;
#else
        hash = (tilecol * HASH_SCALE) & (HASH_SIZE - 1);
#endif
        while (1)
        {
            old = atomicCAS(shared_col + hash, -1, tilecol);
            if (old == -1 || old == tilecol)
            {
                atomicOr(shared_mask + hash, tmp);
                break;
            }
            else
            {
#if SQUARING_B_MASK
                hash = (hash + j * j) % HASH_SIZE;
                j++;
#else
                // hash = (hash + 1) & (HASH_SIZE - 1);
                hash = (hash + 1) < HASH_SIZE ? (hash + 1) : 0;
#endif
            }
        }
    }
    __syncthreads();
    int offset_t;
    int offset = tileptrB[rowid];
    for (i = threadIdx.x; i < HASH_SIZE; i += blockDim.x)
    {
        if (shared_col[i] != -1)
        {
            offset_t = atomicAdd(tmp_num, 1);
            tilecolB[offset + offset_t] = shared_col[i];
            tilemaskB[offset + offset_t] = shared_mask[i];
        }
    }
}

__global__ void k_calculate_B_tileColAndtileMask_max_shared(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    const int *__restrict__ tileptrB,
    int *__restrict__ tilecolB,
    MASK_TYPE *__restrict__ tilemaskB,
    const int *__restrict__ bins)
{
    int row_offset = blockIdx.x;
    const int tsize = 12671;
    extern __shared__ int shared_mem[];
    int *shared_col = shared_mem;
    int *tmp_num = shared_mem + tsize;
    MASK_TYPE *shared_mask = (MASK_TYPE *)(shared_mem + (tsize + 1));
    int i;
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        shared_col[i] = -1;
        shared_mask[i] = 0;
    }
    if (threadIdx.x == 0)
    {
        tmp_num[0] = 0;
    }
    __syncthreads();
    int rowid = bins[row_offset];
    int nnz_start = ptrB[rowid];
    int nnz_end = ptrB[rowid + 1];
    int c, tilecol;
    int old, hash;

    for (i = nnz_start + threadIdx.x; i < nnz_end; i += blockDim.x)
    {
        MASK_TYPE tmp = 1;
        int j = 1;
        c = colB[i];
        tilecol = c >> BLOCK_SIZE_BIT;
        tmp = tmp << (c & (BLOCK_SIZE - 1));
        hash = (tilecol * HASH_SCALE) % tsize;
        while (1)
        {
            old = atomicCAS(shared_col + hash, -1, tilecol);
            if (old == -1 || old == tilecol)
            {
                atomicOr(shared_mask + hash, tmp);
                break;
            }
            else
            {
#if SQUARING_B_MASK
                hash = (hash + j * j) % tsize;
                j++;
#else
                hash = hash + 1 < tsize ? hash + 1 : 0;
#endif
            }
        }
    }
    __syncthreads();
    int offset_t;
    int offset = tileptrB[rowid];
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        if (shared_col[i] != -1)
        {
            offset_t = atomicAdd(tmp_num, 1);
            tilecolB[offset + offset_t] = shared_col[i];
            tilemaskB[offset + offset_t] = shared_mask[i];
        }
    }
}
__global__ void k_calculate_B_tileColAndtileMask_global_mem(
    const int *__restrict__ ptrB,
    const int *__restrict__ colB,
    const int *__restrict__ tileptrB,
    int *__restrict__ tilecolB,
    MASK_TYPE *__restrict__ tilemaskB,
    MASK_TYPE *__restrict__ d_global_mem,
    const int *__restrict__ bins,
    int N)
{
    int row_offset = blockIdx.x;
    MASK_TYPE *shared_mask = d_global_mem + N * row_offset;
    __shared__ int tmp_num[1];
    int i;
    for (i = threadIdx.x; i < N; i += blockDim.x)
    {
        shared_mask[i] = 0;
    }
    if (threadIdx.x == 0)
    {
        tmp_num[0] = 0;
    }
    __syncthreads();
    int rowid = bins[row_offset];
    int nnz_start = ptrB[rowid];
    int nnz_end = ptrB[rowid + 1];
    int c, tilecol;
    for (i = nnz_start + threadIdx.x; i < nnz_end; i += blockDim.x)
    {
        MASK_TYPE tmp = 1;
        c = colB[i];
        tilecol = c >> BLOCK_SIZE_BIT;
        tmp = tmp << (c & BLOCK_SIZE - 1);
        atomicOr(shared_mask + tilecol, tmp);
    }
    __syncthreads();
    int offset_t;
    int offset = tileptrB[rowid];
    for (i = threadIdx.x; i < N; i += blockDim.x)
    {
        if (shared_mask[i] != 0)
        {
            offset_t = atomicAdd(tmp_num, 1);
            tilecolB[offset + offset_t] = i;
            tilemaskB[offset + offset_t] = shared_mask[i];
        }
    }
}