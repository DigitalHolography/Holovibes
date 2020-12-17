/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "reduce.cuh"

void test_gpu_reduce_add(const float* const input,
                         double* const result,
                         const uint size)
{
    reduce_add(input, result, size);
}

void test_gpu_reduce_min(const double* const input,
                         double* const result,
                         const uint size)
{
    reduce_min(input, result, size);
}

void test_gpu_reduce_max(const int* const input,
                         int* const result,
                         const uint size)
{
    reduce_max(input, result, size);
}

void test_gpu_reduce_max(const float* const input,
                         float* const result,
                         const uint size)
{
    reduce_max(input, result, size);
}