#pragma once

class Timing
{
public:
    double mem_alloc;
    double Form_mask_matrix_B;
    double Calculate_C_nnz;
    double Malloc_C_col_val;
    double Numeric;
    double symbolic_binning;
    double numeric_binning;

    Timing() :mem_alloc(0), Form_mask_matrix_B(0), Calculate_C_nnz(0), Malloc_C_col_val(0), Numeric(0), symbolic_binning(0), numeric_binning(0){};

    void operator+=(const Timing &t);

    void operator/=(const double x);
    void print_step_time();
    double getTotal();
};