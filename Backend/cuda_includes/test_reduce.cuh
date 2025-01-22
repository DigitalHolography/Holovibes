/*! \file
 *
 * \brief Declaration of test functions
 *
 * WARNING This file should only be included in the test_reduce.cu file
 *
 * Test files must be .cc
 * To be templatable our reduce_generic needs to be in a .cuhxx file
 * Including directly this file in a .cc would not make it compile with nvcc
 * The only solution is to create a .cu file that will be compiled with nvcc
 * Only then, include this file in our test_reduce.cc
 */

#pragma once

#include "common.cuh"

/*! \brief reduce_add wrapper */
void test_gpu_reduce_add(double* const output, const float* const input, const uint size);

/*! \brief reduce_min wrapper */
void test_gpu_reduce_min(double* const output, const double* const input, const uint size);

/*! \brief reduce_max for int values wrapper */
void test_gpu_reduce_max(int* const output, const int* const input, const uint size);

/*! \brief reduce_max for float values wrapper */
void test_gpu_reduce_max(float* const output, const float* const input, const uint size);
