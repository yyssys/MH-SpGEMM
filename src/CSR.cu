#include <cassert>
#include "CSR.h"

void CSR::h_release_csr()
{
    delete[] ptr;
    ptr = nullptr;
    delete[] col;
    col = nullptr;
    delete[] val;
    val = nullptr;
}

void CSR::d_release_csr()
{
    CHECK_ERROR(cudaFree(d_ptr));
    d_ptr = nullptr;
    CHECK_ERROR(cudaFree(d_col));
    d_col = nullptr;
    CHECK_ERROR(cudaFree(d_val));
    d_val = nullptr;
}

void CSR::alloc(int r, int c, int n)
{
    M = r;
    N = c;
    nnz = n;
    ptr = new int[r + 1]();
    col = new int[n];
    val = new VALUE_TYPE[n];
}

CSR &CSR::operator=(const CSR &A)
{
    M = A.M;
    N = A.N;
    nnz = A.nnz;
    isSymmetric = A.isSymmetric;
    ptr = new int[M + 1];
    col = new int[nnz];
    val = new VALUE_TYPE[nnz];
    std::copy(A.ptr, A.ptr + (A.M + 1), ptr);
    std::copy(A.col, A.col + nnz, col);
    std::copy(A.val, A.val + nnz, val);
    return *this;
}
bool CSR::operator==(const CSR &C_tmp)
{
    if (nnz != C_tmp.nnz)
    {
        printf("nnz not equal %d %d\n", nnz, C_tmp.nnz);
        throw std::runtime_error("nnz not equal");
    }

    assert(M == C_tmp.M && "dimension not same");
    assert(N == C_tmp.N && "dimension not same");

    int error_num = 0;
    double epsilon = 1e-9;
    for (int i = 0; i < M; i++)
    {
        if (unlikely(error_num > 10))
            throw std::runtime_error("matrix compare: error num exceed threshold");
        if (unlikely(ptr[i] != C_tmp.ptr[i]))
        {
            printf("ptr not equal at %d rows, %d != %d\n", i, ptr[i], C_tmp.ptr[i]);
            error_num++;
        }
        for (int j = ptr[i]; j < ptr[i + 1]; j++)
        {
            if (unlikely(error_num > 10))
                throw std::runtime_error("matrix compare: error num exceed threshold");
            if (col[j] != C_tmp.col[j])
            {
                printf("col not equal at %d rows, index %d != %d\n", i, col[j], C_tmp.col[j]);
                error_num++;
            }
            if (!(std::fabs(val[j] - C_tmp.val[j]) < epsilon ||
                  std::fabs(val[j] - C_tmp.val[j]) < epsilon * std::fabs(val[j])))
            {
                printf("val not eqaul at %d rows, value %.18le != %.18le\n", i, val[j], C_tmp.val[j]);
                error_num++;
            }
        }
    }
    if (ptr[M] != C_tmp.ptr[M])
    {
        printf("ptr[M] not equal\n");
        throw std::runtime_error("matrix compare: error num exceed threshold");
    }
    if (error_num)
        return false;
    else
        return true;
}
void CSR::H2D()
{
    CHECK_ERROR(cudaMalloc((void **)&d_ptr, (M + 1) * sizeof(int)));
    CHECK_ERROR(cudaMalloc((void **)&d_col, nnz * sizeof(int)));
    CHECK_ERROR(cudaMalloc((void **)&d_val, nnz * sizeof(VALUE_TYPE)));
    CHECK_ERROR(cudaMemcpy(d_ptr, ptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_col, col, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_val, val, nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));
}

void CSR::D2H()
{
    ptr = new int[M + 1];
    col = new int[nnz];
    val = new VALUE_TYPE[nnz];
    CHECK_ERROR(cudaMemcpy(ptr, d_ptr, (M + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(col, d_col, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(val, d_val, nnz * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));
}
void CSR::release()
{
    h_release_csr();
    d_release_csr();
}

void CSR::d_release_tile()
{
    CHECK_ERROR(cudaFree(d_tileptr));
    d_tileptr = nullptr;
    CHECK_ERROR(cudaFree(d_tilecol));
    d_tilecol = nullptr;
    CHECK_ERROR(cudaFree(d_tilemask));
    d_tilemask = nullptr;
}

CSR::~CSR()
{
    release();
}