#pragma once
#include "common.h"

class CSR
{
public:
    int M;
    int N;
    int nnz;

    int *ptr;
    int *col;
    VALUE_TYPE *val;

    int *d_ptr;
    int *d_col;
    VALUE_TYPE *d_val;

    int isSymmetric;

    int *tileptr;
    int *tilecol;
    MASK_TYPE *tilemask;

    int *d_tileptr;
    int *d_tilecol;
    MASK_TYPE *d_tilemask;

    CSR() : M(0), N(0), nnz(0), ptr(nullptr), col(nullptr), val(nullptr), isSymmetric(0), d_ptr(nullptr), d_col(nullptr), d_val(nullptr), tileptr(nullptr), tilecol(nullptr), tilemask(nullptr), d_tileptr(nullptr), d_tilecol(nullptr), d_tilemask(nullptr)
    {
    }
    ~CSR();

    void alloc(int r, int c, int n);
    CSR &operator=(const CSR &A);
    bool operator==(const CSR &A);
    void D2H();
    void H2D();

    void h_release_csr();
    void d_release_csr();
    void d_release_tile();
    void release();
};