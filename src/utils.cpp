#include "utils.h"

void exclusive_scan(int *input, int length)
{
    if (length == 0 || length == 1)
        return;

    int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

void matrix_transposition(CSR const &A, CSR &B)
{
    // calculate the number of nonzero elements in each column
    B.alloc(A.N, A.M, A.nnz);
    for (int i = 0; i < A.nnz; i++)
    {
        B.ptr[A.col[i]]++;
    }

    exclusive_scan(B.ptr, A.N + 1);
    int *col_num = new int[A.N]();

    // copy non-zero element values and row indexes to CSC format
    for (int row = 0; row < A.M; row++)
    {
        for (int j = A.ptr[row]; j < A.ptr[row + 1]; j++)
        {
            int col = A.col[j];
            int pos = B.ptr[col] + col_num[col];
            B.col[pos] = row;
            B.val[pos] = A.val[j];
            col_num[col]++;
        }
    }
    delete[] col_num;
    col_num = nullptr;
}

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