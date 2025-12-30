__global__ void k_numeric_for_one_nnz(
    const int *__restrict__ d_ptrA,
    const int *__restrict__ d_colA,
    const VALUE_TYPE *__restrict__ d_valA,
    const int *__restrict__ d_ptrB,
    const int *__restrict__ d_colB,
    const VALUE_TYPE *__restrict__ d_valB,
    int M,
    const int *__restrict__ d_bins,
    const int *__restrict__ d_ptrC,
    int *__restrict__ d_colC,
    VALUE_TYPE *__restrict__ d_valC)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= M)
        return;
    int rowid = d_bins[i];
    int Annz_start = d_ptrA[rowid];
    int Annz_end = d_ptrA[rowid + 1];

    int colA;
    VALUE_TYPE valA;
    int colC;
    VALUE_TYPE valC = 0;

    for (i = Annz_start; i < Annz_end; i++)
    {
        colA = d_colA[i];
        valA = d_valA[i];
        int start = d_ptrB[colA];
        int end = d_ptrB[colA + 1];
        if (end > start)
        {
            colC = d_colB[start];
            valC += d_valB[start] * valA;
        }
    }
    int offset = d_ptrC[rowid];
    d_colC[offset] = colC;
    d_valC[offset] = valC;
}

__global__ void k_numeric_shared_hash_pwarp(
    const int *__restrict__ d_ptrA,
    const int *__restrict__ d_colA,
    const VALUE_TYPE *__restrict__ d_valA,
    const int *__restrict__ d_ptrB,
    const int *__restrict__ d_colB,
    const VALUE_TYPE *__restrict__ d_valB,
    int M,
    const int *__restrict__ d_bins,
    const int *__restrict__ d_ptrC,
    int *__restrict__ d_colC,
    VALUE_TYPE *__restrict__ d_valC,
    int *__restrict__ conflict)
{
    int i, j, k;
    int row_offset = PWARP_ROWS_FOR_NUMERIC * blockIdx.x;
    int row_num = blockIdx.x == gridDim.x - 1 ? M - row_offset : PWARP_ROWS_FOR_NUMERIC;
    int local_rowid = threadIdx.x / PWARP_FOR_NUMERIC;
    int tid = threadIdx.x & (PWARP_FOR_NUMERIC - 1);
    const int tsize = PWARP_HASH_SIZE_FOR_NUMERIC * PWARP_ROWS_FOR_NUMERIC;
    __shared__ int shared_mem[tsize * (sizeof(int) + sizeof(VALUE_TYPE)) / sizeof(int) + PWARP_ROWS_FOR_NUMERIC];
    int *shared_col = shared_mem;
    int *shared_tmp_num = shared_mem + tsize;
    VALUE_TYPE *shared_val = (VALUE_TYPE *)(shared_col + tsize + PWARP_ROWS_FOR_NUMERIC);
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        shared_col[i] = -1;
        shared_val[i] = 0;
    }
    if (threadIdx.x < PWARP_ROWS_FOR_NUMERIC)
    {
        shared_tmp_num[threadIdx.x] = 0;
    }
    __syncthreads();
    if (local_rowid >= row_num)
        return;
    int *table_shared_col = shared_col + local_rowid * PWARP_HASH_SIZE_FOR_NUMERIC;
    VALUE_TYPE *table_shared_val = shared_val + local_rowid * PWARP_HASH_SIZE_FOR_NUMERIC;
    int rowid = d_bins[row_offset + local_rowid];
    int Annz_start = d_ptrA[rowid];
    int Annz_end = d_ptrA[rowid + 1];
    int Bnnz_start, Bnnz_end;
    int acol, bcol;
    VALUE_TYPE aval, bval;
    int hash, old;
    for (i = Annz_start + tid; i < Annz_end; i += PWARP_FOR_NUMERIC)
    {
        acol = d_colA[i];
        aval = d_valA[i];
        Bnnz_start = d_ptrB[acol];
        Bnnz_end = d_ptrB[acol + 1];
        for (k = Bnnz_start; k < Bnnz_end; k++)
        {
            j = 1;
            bcol = d_colB[k];
            bval = d_valB[k];
#if SQUARING
            hash = (int)(((uint64_t)bcol * HASH_SCALE) % PWARP_HASH_SIZE_FOR_NUMERIC);
            // hash = (bcol * HASH_SCALE) % PWARP_HASH_SIZE_FOR_NUMERIC;
#else
            // hash = (bcol * HASH_SCALE) & (PWARP_HASH_SIZE_FOR_NUMERIC - 1);
            hash = (int)(((uint64_t)bcol * HASH_SCALE) & (PWARP_HASH_SIZE_FOR_NUMERIC - 1));
#endif
            while (1)
            {
                old = atomicCAS(table_shared_col + hash, -1, bcol);
                if (old == -1 || old == bcol)
                {
                    atomicAdd(table_shared_val + hash, aval * bval);
                    break;
                }
                else
                {
#if HASH_CONFLICT
                    atomicAdd(conflict, 1);
#endif
#if SQUARING
                    hash = (hash + j * j) % PWARP_HASH_SIZE_FOR_NUMERIC;
                    j++;
#else
                    // hash = (hash + 1) & (PWARP_HASH_SIZE_FOR_NUMERIC - 1);
                    hash = (hash + 1) < PWARP_HASH_SIZE_FOR_NUMERIC ? hash + 1 : 0;
#endif
                }
            }
        }
    }
    __syncthreads();
    int c_offset = d_ptrC[rowid];
    int row_nnz = d_ptrC[rowid + 1] - d_ptrC[rowid];
    int offset;
    bool valid;
#pragma unroll
    for (i = 0; i < PWARP_HASH_SIZE_FOR_NUMERIC; i += PWARP_FOR_NUMERIC)
    {
        offset = i + tid;
        valid = offset < PWARP_HASH_SIZE_FOR_NUMERIC;
        if (valid)
        {
            acol = table_shared_col[offset];
            aval = table_shared_val[offset];
            if (acol != -1)
            {
                offset = atomicAdd(shared_tmp_num + local_rowid, 1);
            }
        }
        __syncthreads();
        if (valid && acol != -1)
        {
            table_shared_col[offset] = acol;
            table_shared_val[offset] = aval;
        }
    }
    __syncthreads();
    for (i = tid; i < row_nnz; i += PWARP_FOR_NUMERIC)
    {
        acol = table_shared_col[i];
        offset = 0;
        for (k = 0; k < row_nnz; k++)
        {
            offset += (unsigned int)(table_shared_col[k] - acol) >> 31;
        }
        d_colC[c_offset + offset] = table_shared_col[i];
        d_valC[c_offset + offset] = table_shared_val[i];
    }
}

template <int HASH_SIZE>
__global__ void k_numeric_shared_hash_tb(
    const int *__restrict__ d_ptrA,
    const int *__restrict__ d_colA,
    const VALUE_TYPE *__restrict__ d_valA,
    const int *__restrict__ d_ptrB,
    const int *__restrict__ d_colB,
    const VALUE_TYPE *__restrict__ d_valB,
    const int *__restrict__ d_bins,
    const int *__restrict__ d_ptrC,
    int *__restrict__ d_colC,
    VALUE_TYPE *__restrict__ d_valC,
    const int *__restrict__ d_group_size,
    int *__restrict__ conflict)
{
    int i, j, k;
    int row_offset = blockIdx.x;
    __shared__ int shared_col[HASH_SIZE];
    __shared__ VALUE_TYPE shared_val[HASH_SIZE];
    __shared__ int shared_tmp_num;
    for (i = threadIdx.x; i < HASH_SIZE; i += blockDim.x)
    {
        shared_col[i] = -1;
        shared_val[i] = 0;
    }
    if (threadIdx.x == 0)
    {
        shared_tmp_num = 0;
    }
    __syncthreads();
    int rowid = d_bins[row_offset];
    int Annz_start = d_ptrA[rowid];
    int Annz_end = d_ptrA[rowid + 1];
#if ADAPTIVE_GROUPING
    int group_size = d_group_size[rowid];
#else
    int group_size = 32;
#endif
    int group_id = threadIdx.x / group_size;
    int group_num = blockDim.x / group_size;
    int tid = threadIdx.x & (group_size - 1);
    int Bnnz_start, Bnnz_end;
    int acol, bcol;
    VALUE_TYPE aval, bval;
    int hash, old;
    for (i = Annz_start + group_id; i < Annz_end; i += group_num)
    {
        acol = d_colA[i];
        aval = d_valA[i];
        Bnnz_start = d_ptrB[acol];
        Bnnz_end = d_ptrB[acol + 1];

        for (k = Bnnz_start + tid; k < Bnnz_end; k += group_size)
        {
            j = 1;
            bcol = d_colB[k];
            bval = d_valB[k];

#if SQUARING
            // hash = (bcol * HASH_SCALE) % HASH_SIZE;
            hash = (int)(((uint64_t)bcol * HASH_SCALE) % HASH_SIZE);
#else
            // hash = (bcol * HASH_SCALE) & (HASH_SIZE - 1);
            hash = (int)(((uint64_t)bcol * HASH_SCALE) & (HASH_SIZE - 1));
#endif
            while (1)
            {
                old = atomicCAS(shared_col + hash, -1, bcol);
                if (old == -1 || old == bcol)
                {
                    atomicAdd(shared_val + hash, aval * bval);
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
                    // hash = (hash + 1) & (HASH_SIZE - 1);
                    hash = (hash + 1) < HASH_SIZE ? hash + 1 : 0;
#endif
                }
            }
        }
    }
    __syncthreads();
    int c_offset = d_ptrC[rowid];
    int row_nnz = d_ptrC[rowid + 1] - d_ptrC[rowid];
    int offset;
    bool valid;
#pragma unroll
    for (i = 0; i < HASH_SIZE; i += blockDim.x)
    {
        offset = i + threadIdx.x;
        valid = offset < HASH_SIZE;
        if (valid)
        {
            acol = shared_col[offset];
            aval = shared_val[offset];
            if (acol != -1)
            {
                offset = atomicAdd(&shared_tmp_num, 1);
            }
        }
        __syncthreads();
        if (valid && acol != -1)
        {
            shared_col[offset] = acol;
            shared_val[offset] = aval;
        }
    }
    __syncthreads();
    int count, target;
    for (j = threadIdx.x; j < row_nnz; j += blockDim.x)
    {
        target = shared_col[j];
        count = 0;
        for (k = 0; k < row_nnz; k++)
        {
            count += (unsigned int)(shared_col[k] - target) >> 31;
        }
        d_colC[c_offset + count] = shared_col[j];
        d_valC[c_offset + count] = shared_val[j];
    }
}

__global__ void k_numeric_max_shared_hash_tb(
    const int *__restrict__ d_ptrA,
    const int *__restrict__ d_colA,
    const VALUE_TYPE *__restrict__ d_valA,
    const int *__restrict__ d_ptrB,
    const int *__restrict__ d_colB,
    const VALUE_TYPE *__restrict__ d_valB,
    const int *__restrict__ d_bins,
    const int *__restrict__ d_ptrC,
    int *__restrict__ d_colC,
    VALUE_TYPE *__restrict__ d_valC,
    const int *__restrict__ d_group_size,
    int *__restrict__ conflict)
{
    int i, j, k;
    int row_offset = blockIdx.x;
#if SQUARING
    int tsize = 8447;
#else
    int tsize = 8192;
#endif
    extern __shared__ int shared_mem[];
    VALUE_TYPE *shared_val = (VALUE_TYPE *)shared_mem;
    int *shared_col = (int *)(shared_val + tsize);
    int *shared_tmp_num = (int *)(shared_col + tsize);

    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        shared_col[i] = -1;
        shared_val[i] = 0;
    }
    if (threadIdx.x == 0)
    {
        shared_tmp_num[0] = 0;
    }
    __syncthreads();
    int rowid = d_bins[row_offset];
    int Annz_start = d_ptrA[rowid];
    int Annz_end = d_ptrA[rowid + 1];

#if ADAPTIVE_GROUPING
    int group_size = d_group_size[rowid];
#else
    int group_size = 32;
#endif
    int group_id = threadIdx.x / group_size;
    int group_num = blockDim.x / group_size;
    int tid = threadIdx.x & (group_size - 1);

    int Bnnz_start, Bnnz_end;
    int acol, bcol;
    VALUE_TYPE aval, bval;
    int hash, old;
    for (i = Annz_start + group_id; i < Annz_end; i += group_num)
    {
        acol = d_colA[i];
        aval = d_valA[i];
        Bnnz_start = d_ptrB[acol];
        Bnnz_end = d_ptrB[acol + 1];

        for (k = Bnnz_start + tid; k < Bnnz_end; k += group_size)
        {
            j = 1;
            bcol = d_colB[k];
            bval = d_valB[k];

#if SQUARING
            // hash = (bcol * HASH_SCALE) % tsize;
            hash = (int)(((uint64_t)bcol * HASH_SCALE) % tsize);
#else
            hash = (int)(((uint64_t)bcol * HASH_SCALE) & (tsize - 1));
#endif
            while (1)
            {
                old = atomicCAS(shared_col + hash, -1, bcol);
                if (old == -1 || old == bcol)
                {
                    atomicAdd(shared_val + hash, aval * bval);
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
                    // hash = (hash + 1) & (tsize - 1);
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
#endif
                }
            }
        }
    }
    __syncthreads();
    int c_offset = d_ptrC[rowid];
    int row_nnz = d_ptrC[rowid + 1] - d_ptrC[rowid];
    int offset;
    bool valid;
#pragma unroll
    for (i = 0; i < tsize; i += blockDim.x)
    {
        offset = i + threadIdx.x;
        valid = offset < tsize;
        if (valid)
        {
            acol = shared_col[offset];
            aval = shared_val[offset];
            if (acol != -1)
            {
                offset = atomicAdd(shared_tmp_num, 1);
            }
        }
        __syncthreads();
        if (valid && acol != -1)
        {
            shared_col[offset] = acol;
            shared_val[offset] = aval;
        }
    }
    __syncthreads();

#if BITONIC_SORT
    if (row_nnz >= 2048)
    {
        int m = row_nnz / 2;
        for (i = 1; i < row_nnz; i *= 2)
        {
            for (j = i; j > 0; j /= 2)
            {
                for (k = threadIdx.x; k < m; k += blockDim.x)
                {
                    int a = 2 * j * (k / j);
                    int b = k % j;
                    int u = (j == i) ? (a + j - 1 - b) : (a + b);
                    int d = a + b + j;
                    if (d < row_nnz && shared_col[u] > shared_col[d])
                    {
                        int tmp_col = shared_col[u];
                        VALUE_TYPE tmp_val = shared_val[u];
                        shared_col[u] = shared_col[d];
                        shared_val[u] = shared_val[d];
                        shared_col[d] = tmp_col;
                        shared_val[d] = tmp_val;
                    }
                }
                __syncthreads();
            }
        }
        for (i = threadIdx.x; i < row_nnz; i += blockDim.x)
        {
            d_colC[c_offset + i] = shared_col[i];
            d_valC[c_offset + i] = shared_val[i];
        }
    }
    else
    {
        int count, target;
        for (j = threadIdx.x; j < row_nnz; j += blockDim.x)
        {
            target = shared_col[j];
            count = 0;
            for (k = 0; k < row_nnz; k++)
            {
                count += (unsigned int)(shared_col[k] - target) >> 31;
            }
            d_colC[c_offset + count] = shared_col[j];
            d_valC[c_offset + count] = shared_val[j];
        }
    }
#else
    int count, target;
    for (j = threadIdx.x; j < row_nnz; j += blockDim.x)
    {
        target = shared_col[j];
        count = 0;
        for (k = 0; k < row_nnz; k++)
        {
            count += (unsigned int)(shared_col[k] - target) >> 31;
        }
        d_colC[c_offset + count] = shared_col[j];
        d_valC[c_offset + count] = shared_val[j];
    }
#endif
}

__global__ void k_numeric_global_hash(
    const int *__restrict__ d_ptrA,
    const int *__restrict__ d_colA,
    const VALUE_TYPE *__restrict__ d_valA,
    const int *__restrict__ d_ptrB,
    const int *__restrict__ d_colB,
    const VALUE_TYPE *__restrict__ d_valB,
    const int *__restrict__ d_bins,
    const int *__restrict__ d_ptrC,
    int *__restrict__ d_colC,
    VALUE_TYPE *__restrict__ d_valC,
    int *__restrict__ d_global_mem,
    int max_tsize,
    const int *__restrict__ d_group_size,
    int *__restrict__ conflict)
{
    int i, j, k;
    int row_offset = blockIdx.x;
    int rowid = d_bins[row_offset];
    __shared__ int shared_tmp_num[1];

    int *table_col = d_global_mem + row_offset * max_tsize * ((sizeof(int) + sizeof(VALUE_TYPE)) / sizeof(int));
    VALUE_TYPE *table_val = (VALUE_TYPE *)(table_col + max_tsize);
    int c_offset = d_ptrC[rowid];
    int row_nnz = d_ptrC[rowid + 1] - c_offset;
    int tsize = max_tsize;
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        table_col[i] = -1;
        table_val[i] = 0;
    }
    if (threadIdx.x == 0)
    {
        shared_tmp_num[0] = 0;
    }
    __syncthreads();

    int Annz_start = d_ptrA[rowid];
    int Annz_end = d_ptrA[rowid + 1];

#if ADAPTIVE_GROUPING
    int group_size = d_group_size[rowid];
#else
    int group_size = 32;
#endif

    int group_id = threadIdx.x / group_size;
    int group_num = blockDim.x / group_size;
    int tid = threadIdx.x & (group_size - 1);

    int Bnnz_start, Bnnz_end, hash, old;
    int acol, bcol;
    VALUE_TYPE aval, bval;
    for (i = Annz_start + group_id; i < Annz_end; i += group_num)
    {
        acol = d_colA[i];
        aval = d_valA[i];
        Bnnz_start = d_ptrB[acol];
        Bnnz_end = d_ptrB[acol + 1];

        for (k = Bnnz_start + tid; k < Bnnz_end; k += group_size)
        {
            bcol = d_colB[k];
            bval = d_valB[k];
            // hash = (bcol * HASH_SCALE) % tsize;
            hash = (int)(((uint64_t)bcol * HASH_SCALE) % tsize);
            while (1)
            {
                old = atomicCAS(table_col + hash, -1, bcol);
                if (old == -1 || old == bcol)
                {
                    atomicAdd(table_val + hash, aval * bval);
                    break;
                }
                else
                {
#if HASH_CONFLICT
                    atomicAdd(conflict, 1);
#endif
                    hash = (hash + 1) < tsize ? hash + 1 : 0;
                }
            }
        }
    }
    __syncthreads();
    int offset;
    for (i = threadIdx.x; i < tsize; i += blockDim.x)
    {
        acol = table_col[i];
        aval = table_val[i];
        if (acol != -1)
        {
            offset = atomicAdd(shared_tmp_num, 1);
            d_colC[c_offset + offset] = acol;
            d_valC[c_offset + offset] = aval;
        }
    }
    __syncthreads();
    for (i = threadIdx.x; i < row_nnz; i += blockDim.x)
    {
        table_col[i] = d_colC[c_offset + i];
        table_val[i] = d_valC[c_offset + i];
    }

    __syncthreads();

#if BITONIC_SORT
    int m = row_nnz / 2;
    for (i = 1; i < row_nnz; i *= 2)
    {
        for (j = i; j > 0; j /= 2)
        {
            for (k = threadIdx.x; k < m; k += blockDim.x)
            {
                int a = 2 * j * (k / j);
                int b = k % j;
                int u = (j == i) ? (a + j - 1 - b) : (a + b);
                int d = a + b + j;
                if (d < row_nnz && table_col[u] > table_col[d])
                {
                    int tmp_col = table_col[u];
                    VALUE_TYPE tmp_val = table_val[u];
                    table_col[u] = table_col[d];
                    table_val[u] = table_val[d];
                    table_col[d] = tmp_col;
                    table_val[d] = tmp_val;
                }
            }
            __syncthreads();
        }
    }
    for (i = threadIdx.x; i < row_nnz; i += blockDim.x)
    {
        d_colC[c_offset + i] = table_col[i];
        d_valC[c_offset + i] = table_val[i];
    }
#else
    int count, target;
    for (i = threadIdx.x; i < row_nnz; i += blockDim.x)
    {
        target = table_col[i];
        count = 0;
        for (k = 0; k < row_nnz; k++)
        {
            count += (unsigned int)(table_col[k] - target) >> 31;
        }
        d_colC[c_offset + count] = table_col[i];
        d_valC[c_offset + count] = table_val[i];
    }
#endif
}