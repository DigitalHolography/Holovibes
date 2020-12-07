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

#pragma once

#include <device_launch_parameters.h>
#include <string>
#include <exception>
#include <cublas_v2.h>
#include <cassert>
#include "cusolverDn.h"

#include "tools.cuh"
#include "compute_descriptor.hh"
#include "custom_exception.hh"

#ifndef M_PI
  #define M_PI       3.14159265358979323846   // pi
#endif
#ifndef M_PI_2
  #define M_PI_2     1.57079632679489661923   // pi/2
#endif
#define M_2PI		6.28318530717959f
#define THREADS_256	256
#define THREADS_128	128

static constexpr cudaStream_t default_cuda_stream = 0;

#ifndef _DEBUG
#define cudaCheckError()
#else
#define cudaCheckError()                                                     \
{                                                                            \
	auto e = cudaGetLastError();                                             \
	if (e != cudaSuccess)                                                    \
	{                                                                        \
		std::string error = "Cuda failure in ";                              \
		error += __FILE__;                                                   \
		error += " at line ";                                                \
		error += std::to_string(__LINE__);                                   \
		error += ": ";                                                       \
		error += cudaGetErrorString(e);                                      \
		throw holovibes::CustomException(error, holovibes::fail_cudaLaunch); \
	}												                         \
}
#endif

#ifndef _DEBUG
#define cudaSafeCall(ans) ans
#else
#define cudaSafeCall(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

#ifndef _DEBUG
#define cublasSafeCall(err) err
#else
static const char *_cudaGetCublasErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
		case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "Unknown cublas error";
}
#define cublasSafeCall(err)      __cublasSafeCall(err, __FILE__, __LINE__)
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if( CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n%s\nterminating!\n",
											__FILE__,
													__LINE__,
															_cudaGetCublasErrorEnum(err));
		cudaDeviceReset();
		assert(0);
    }
}
#endif

#ifndef _DEBUG
#define cusolverSafeCall(err) err
#else
static const char *_cudaGetCusolverErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
		case CUSOLVER_STATUS_NOT_INITIALIZED:
			return "CUSOLVER_STATUS_NOT_INITIALIZED";
		case CUSOLVER_STATUS_ALLOC_FAILED:
			return "CUSOLVER_STATUS_ALLOC_FAILED";
		case CUSOLVER_STATUS_INVALID_VALUE:
			return "CUSOLVER_STATUS_INVALID_VALUE";
		case CUSOLVER_STATUS_ARCH_MISMATCH:
			return "CUSOLVER_STATUS_ARCH_MISMATCH";
		case CUSOLVER_STATUS_EXECUTION_FAILED:
			return "CUSOLVER_STATUS_EXECUTION_FAILED";
		case CUSOLVER_STATUS_INTERNAL_ERROR:
			return "CUSOLVER_STATUS_INTERNAL_ERROR";
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "Unknown cusolver error";
}
#define cusolverSafeCall(err)      __cusolverSafeCall(err, __FILE__, __LINE__)
inline void __cusolverSafeCall(cusolverStatus_t err, const char *file, const int line)
{
    if( CUSOLVER_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n%s\nterminating!\n",
											__FILE__,
													__LINE__,
															_cudaGetCusolverErrorEnum(err));
		cudaDeviceReset();
		assert(0);
    }
}
#endif

#ifndef _DEBUG
#define cufftSafeCall(err) err
#else
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
		fprintf(stderr, "CUFFT error in file '%s', line %d\n%s\nterminating!\n",
											__FILE__,
													__LINE__,
															_cudaGetErrorEnum(err));
		cudaDeviceReset();
		assert(0);
    }
}
#endif

// atomicAdd with double is not defined if CUDA Version is not greater than or equal to 600
// So we use this macro to keep a fully compatible program
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif