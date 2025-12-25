#pragma once

#include <vector>
#include <algorithm>

#include "mmio.h"
#include "utils.h"

void sort_csr_col(int *ptr, int *col, VALUE_TYPE *val, const int m_tmp)
{
#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < m_tmp; row++)
    {
        int start = ptr[row];
        int end = ptr[row + 1];

        std::vector<std::pair<int, VALUE_TYPE>> row_data;
        for (int j = start; j < end; j++)
        {
            row_data.emplace_back(col[j], val[j]);
        }

        std::sort(row_data.begin(), row_data.end());

        for (int j = start, k = 0; j < end; j++, k++)
        {
            col[j] = row_data[k].first;
            val[j] = row_data[k].second;
        }
    }
}

// read matrix infomation from mtx file
int readMtxFile(CSR &A, const char *filename)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int m_tmp, n_tmp, nnz_tmp;
    int i;
    int mtx_report_nnz;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
    {
        printf("Could not open file %s.\n", filename);
        return -1;
    }
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -1;
    }

    int is_real = mm_is_real(matcode);
    int is_complex = mm_is_complex(matcode);
    int is_integer = mm_is_integer(matcode);
    int is_pattern = mm_is_pattern(matcode);
    int is_symmetric = mm_is_symmetric(matcode);
    int is_hermitian = mm_is_hermitian(matcode);

    A.isSymmetric = is_symmetric;

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &mtx_report_nnz)) != 0)
        return -1;

    /* reseve temp memory for matrices */
    int *ptr_tmp = new int[m_tmp + 1]();

    int *rowIdx_tmp = new int[mtx_report_nnz];
    int *colIdx_tmp = new int[mtx_report_nnz];
    VALUE_TYPE *val_tmp = new VALUE_TYPE[mtx_report_nnz];

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < mtx_report_nnz; i++)
    {
        double val_i;
        int int_val;

        if (is_real)
        {
            fscanf(f, "%d %d %lg\n", &rowIdx_tmp[i], &colIdx_tmp[i], &val_tmp[i]);
        }
        else if (is_integer)
        {
            fscanf(f, "%d %d %d\n", &rowIdx_tmp[i], &colIdx_tmp[i], &int_val);
            val_tmp[i] = int_val;
        }
        else if (is_pattern)
        {
            fscanf(f, "%d %d\n", &rowIdx_tmp[i], &colIdx_tmp[i]);
            val_tmp[i] = 1.0;
        }
        else if (is_complex)
        {
            fscanf(f, "%d %d %lg %lg\n", &rowIdx_tmp[i], &colIdx_tmp[i], &val_tmp[i], &val_i); // Complex numbers only store the real part.
        }

        rowIdx_tmp[i]--; /*adjust from 1-based to 0-based*/
        colIdx_tmp[i]--;

        ptr_tmp[rowIdx_tmp[i]]++; // Count the nonzero elements in each row.
    }

    if (f != stdin)
        fclose(f);

    // Count the nonzero elements in the symmetric positions of columns for symmetric or Hermitian matrices.
    if (is_hermitian || is_symmetric)
    {
        for (i = 0; i < mtx_report_nnz; i++)
        {
            if (rowIdx_tmp[i] != colIdx_tmp[i])
                ptr_tmp[colIdx_tmp[i]]++;
        }
    }

    /* compute ptr_tmp */
    exclusive_scan(ptr_tmp, m_tmp + 1);
    nnz_tmp = ptr_tmp[m_tmp];

    /* reseve memory for matrices */
    A.alloc(m_tmp, n_tmp, nnz_tmp);

    int *ptr_offset = new int[m_tmp]();

    for (i = 0; i < mtx_report_nnz; i++)
    {
        int offset = ptr_tmp[rowIdx_tmp[i]] + ptr_offset[rowIdx_tmp[i]];
        A.col[offset] = colIdx_tmp[i];
        A.val[offset] = val_tmp[i];
        ptr_offset[rowIdx_tmp[i]]++;
        if ((is_hermitian || is_symmetric) && (rowIdx_tmp[i] != colIdx_tmp[i]))
        {
            offset = ptr_tmp[colIdx_tmp[i]] + ptr_offset[colIdx_tmp[i]];
            A.col[offset] = rowIdx_tmp[i];
            A.val[offset] = val_tmp[i];
            ptr_offset[colIdx_tmp[i]]++;
        }
    }

    /* copy ptr_tmp to ptr */
    std::copy(ptr_tmp, ptr_tmp + (m_tmp + 1), A.ptr);

    sort_csr_col(A.ptr, A.col, A.val, m_tmp);

    delete[] ptr_tmp;
    delete[] rowIdx_tmp;
    delete[] colIdx_tmp;
    delete[] val_tmp;
    delete[] ptr_offset;

    return 0;
}