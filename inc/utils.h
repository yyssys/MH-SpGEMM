#pragma once
#include <iostream>
#include <string>
#include "CSR.h"

void exclusive_scan(int *input, int length);

void matrix_transposition(CSR const &A, CSR &B);

std::string extract_matrix_name(const std::string &path);