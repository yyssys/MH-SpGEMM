#include "Timing.h"
#include <stdio.h>

void Timing::operator+=(const Timing &t)
{
    Form_mask_matrix_B += t.Form_mask_matrix_B;
    Calculate_C_nnz += t.Calculate_C_nnz;
    Malloc_C_col_val += t.Malloc_C_col_val;
    Numeric += t.Numeric;
}

void Timing::print_step_time()
{
    printf("Form_mask_matrix_B: \t%.3lfms\n", Form_mask_matrix_B);
    printf("Calculate_C_nnz: \t%.3lfms\n", Calculate_C_nnz);
    printf("Malloc_C_col_val: \t%.3lfms\n", Malloc_C_col_val);
    printf("Numeric: \t\t%.3lfms\n", Numeric);
}

void Timing::operator/=(const double x)
{
    Form_mask_matrix_B /= x;
    Calculate_C_nnz /= x;
    Malloc_C_col_val /= x;
    Numeric /= x;
}

double Timing::getTotal()
{
    return Calculate_C_nnz + Malloc_C_col_val + Numeric;
}