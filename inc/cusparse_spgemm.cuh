#include <cusparse.h>
#include "CSR.h"
#include <cuda_runtime.h>
#include "common.h"

void cusparse_spgemm_inner(int *d_row_ptr_A, int *d_col_idx_A, double *d_csr_values_A,
                           int *d_row_ptr_B, int *d_col_idx_B, double *d_csr_values_B,
                           int **d_row_ptr_C, int **d_col_idx_C, double **d_csr_values_C,
                           int M, int K, int N, int nnz_A, int nnz_B, int *nnz_C, double *time)
{
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t matA, matB, matC;

    cusparseCreateCsr(&matA, M, K, nnz_A,
                      d_row_ptr_A, d_col_idx_A, d_csr_values_A,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCsr(&matB, K, N, nnz_B,
                      d_row_ptr_B, d_col_idx_B, d_csr_values_B,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCsr(&matC, M, N, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    CHECK_ERROR(cudaDeviceSynchronize());
    double t0, t1;
    t0 = fast_clock_time();

    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_64F;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    double alpha = 1.0f;
    double beta = 0.0f;

    CHECK_ERROR(cudaMalloc((void **)d_row_ptr_C, (M + 1) * sizeof(int)));

    cusparseSpGEMMDescr_t spgemmDescr;
    cusparseSpGEMM_createDescr(&spgemmDescr);

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDescr, &bufferSize1, NULL);
    CHECK_ERROR(cudaMalloc((void **)&dBuffer1, bufferSize1));
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDescr, &bufferSize1, dBuffer1);
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDescr, &bufferSize2, NULL);

    CHECK_ERROR(cudaMalloc((void **)&dBuffer2, bufferSize2));

    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDescr, &bufferSize2, dBuffer2);

    int64_t M_C, N_C, nnz_C_64I;
    cusparseSpMatGetSize(matC, &M_C, &N_C, &nnz_C_64I);
    *nnz_C = nnz_C_64I;
    CHECK_ERROR(cudaMalloc((void **)d_col_idx_C, *nnz_C * sizeof(int)));
    CHECK_ERROR(cudaMalloc((void **)d_csr_values_C, *nnz_C * sizeof(double)));

    cusparseCsrSetPointers(matC, *d_row_ptr_C, *d_col_idx_C, *d_csr_values_C);

    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr);
    cusparseSpGEMM_destroyDescr(spgemmDescr);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);

    cudaFree(dBuffer1);
    cudaFree(dBuffer2);

    CHECK_ERROR(cudaDeviceSynchronize());
    t1 = fast_clock_time();
    *time = (t1 - t0) * 1000;
    if (nnz_C_64I <= 0)
    {
        throw std::exception();
    }
}

void cusparse_spgemm(CSR *a, CSR *b, CSR *c, double *time)
{
    int tmp_nnz;
    cusparse_spgemm_inner(a->d_ptr, a->d_col, a->d_val,
                          b->d_ptr, b->d_col, b->d_val,
                          &(c->d_ptr), &(c->d_col), &(c->d_val),
                          a->M, a->N, b->N, a->nnz, b->nnz, &(tmp_nnz), time);
    c->M = a->M;
    c->N = b->N;
    c->nnz = tmp_nnz;
}