/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "reduce.cuh"

void test_gpu_reduce_add(const float* const input, double* const result, const uint size)
{
    reduce_add(input, result, size);
}

void test_gpu_reduce_min(const double* const input, double* const result, const uint size)
{
    reduce_min(input, result, size);
}

void test_gpu_reduce_max(const int* const input, int* const result, const uint size)
{
    reduce_max(input, result, size);
}

void test_gpu_reduce_max(const float* const input, float* const result, const uint size)
{
    reduce_max(input, result, size);
}