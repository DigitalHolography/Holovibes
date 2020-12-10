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

/*! \file WARNING This file should only be included in the test_reduce.cu file
*
* Test files must be .cc
* To be templatable our reduce_generic needs to be in a .cuhxx file
* Including directly this file in a .cc would not make it compile with nvcc
* The only solution is to create a .cu file that will be compiled with nvcc
* Only then, include this file in our test_reduce.cc
*/

#pragma once

#include "Common.cuh"

/*! \brief reduce_add wrapper */
void test_gpu_reduce_add(const float* const input, double* const result, const uint size);

/*! \brief reduce_min wrapper */
void test_gpu_reduce_min(const double* const input, double* const result, const uint size);

/*! \brief reduce_max for int values wrapper */
void test_gpu_reduce_max(const int* const input, int* const result, const uint size);

/*! \brief reduce_max for float values wrapper */
void test_gpu_reduce_max(const float* const input, float* const result, const uint size);