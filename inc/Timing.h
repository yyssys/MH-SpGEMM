#pragma once

class Timing
{
public:
    double Form_mask_matrix_B;
    double Calculate_C_nnz;
    double Malloc_C_col_val;
    double Numeric;

    Timing() : Form_mask_matrix_B(0), Calculate_C_nnz(0), Malloc_C_col_val(0), Numeric(0){};

    void operator+=(const Timing &t);

    void operator/=(const double x);
    void print_step_time();
    double getTotal();
};