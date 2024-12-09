/*! \file
 *
 * \brief class to use HoloVibes
 */
#pragma once

#include "common.cuh"

constexpr int CUDA_STREAM_READER_PRIORITY = 1;
constexpr int CUDA_STREAM_RECORDER_PRIORITY = 1;
constexpr int CUDA_STREAM_COMPUTE_PRIORITY = 0;

namespace holovibes
{

struct CudaStreams
{
    CudaStreams() { reload(); }

    /*! \brief Used when the device is reset. Recreate the streams.
     *
     * This might cause a small memory leak, but at least it doesn't cause a crash/segfault
     */
    void reload()
    {
        cudaSafeCall(cudaStreamCreateWithPriority(&reader_stream, cudaStreamDefault, CUDA_STREAM_READER_PRIORITY));
        cudaSafeCall(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, CUDA_STREAM_COMPUTE_PRIORITY));
        cudaSafeCall(cudaStreamCreateWithPriority(&recorder_stream, cudaStreamDefault, CUDA_STREAM_RECORDER_PRIORITY));
    }

    ~CudaStreams()
    {
        cudaSafeCall(cudaStreamDestroy(reader_stream));
        cudaSafeCall(cudaStreamDestroy(compute_stream));
        cudaSafeCall(cudaStreamDestroy(recorder_stream));
    }

    cudaStream_t reader_stream;
    cudaStream_t compute_stream;
    cudaStream_t recorder_stream;
};

} // namespace holovibes