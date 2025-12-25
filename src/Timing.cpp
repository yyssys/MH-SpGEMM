#include "Timing.h"
#include <stdio.h>

void Timing::operator+=(const Timing &t)
{
    mem_alloc += t.mem_alloc;
    Form_mask_matrix_B += t.Form_mask_matrix_B;
    Calculate_C_nnz += t.Calculate_C_nnz;
    Malloc_C_col_val += t.Malloc_C_col_val;
    Numeric += t.Numeric;
    symbolic_binning += t.symbolic_binning;
    numeric_binning += t.numeric_binning;
}

void Timing::print_step_time()
{
    printf("  -------------time-------------\n");
    printf("    mem_alloc: \t\t%.3lfms\n", mem_alloc);
    printf("    form_mask_matrix_B: %.3lfms\n", Form_mask_matrix_B);
    printf("    symbolic_binning: \t%.3lfms\n", symbolic_binning);
    printf("    calculate_C_nnz: \t%.3lfms\n", Calculate_C_nnz);
    printf("    malloc_C_col_val: \t%.3lfms\n", Malloc_C_col_val);
    printf("    numeric_binning: \t%.3lfms\n", numeric_binning);
    printf("    numeric: \t\t%.3lfms\n", Numeric);
    printf("  ------------------------------\n");
}

void Timing::operator/=(const double x)
{
    mem_alloc /= x;
    Form_mask_matrix_B /= x;
    Calculate_C_nnz /= x;
    Malloc_C_col_val /= x;
    Numeric /= x;
    symbolic_binning /= x;
    numeric_binning /= x;
}

double Timing::getTotal()
{
    return Calculate_C_nnz + Malloc_C_col_val + Numeric + symbolic_binning + numeric_binning + mem_alloc;
}