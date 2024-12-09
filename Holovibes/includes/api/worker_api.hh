/*! \file worker_api.hh
 *
 * \brief Regroup all functions used to interact (start, stop) with the workers (compute, file read, camera read,
 * information, ...). Contains also cuda streams and cuda stream priority.
 */
#pragma once

#include "common_api.hh"

// Cuda includes
#include "common.cuh"

// Worker includes
#include "worker.hh"
#include "information_worker.hh"

// Threads priority
constexpr int THREAD_COMPUTE_PRIORITY = THREAD_PRIORITY_TIME_CRITICAL;
constexpr int THREAD_READER_PRIORITY = THREAD_PRIORITY_TIME_CRITICAL;
constexpr int THREAD_RECORDER_PRIORITY = THREAD_PRIORITY_TIME_CRITICAL;
constexpr int THREAD_DISPLAY_PRIORITY = THREAD_PRIORITY_TIME_CRITICAL;

// CUDA streams priority
// Lower numbers represent higher priorities
constexpr int CUDA_STREAM_QUEUE_PRIORITY = 1;
constexpr int CUDA_STREAM_WINDOW_PRIORITY = 1;
constexpr int CUDA_STREAM_READER_PRIORITY = 1;
constexpr int CUDA_STREAM_RECORDER_PRIORITY = 1;
constexpr int CUDA_STREAM_COMPUTE_PRIORITY = 0;

namespace holovibes::api
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

class WorkerApi : IApi
{
  public:
    WorkerApi() = default;

    inline const CudaStreams& get_cuda_streams() const { return cuda_streams_; }

    /*! \brief Sets the error callback function. This function will be called when an error occurs in a worker.
     *
     * \param[in] func the error callback function (takes an exception as argument)
     */
    void set_error_callback(std::function<void(const std::exception&)> func) { error_callback_ = func; }

    /*! \brief Start file read worker (image acquisition)
     *
     * \param callback
     */
    void start_file_frame_read();

    /*! \brief Sets the right camera settings, then starts the camera_read_worker (image acquisition)
     * TODO: refacto (see issue #22)
     *
     * \param camera_kind
     */
    void start_camera_frame_read(CameraKind camera_kind);

    /*! \brief Handle frame reading interruption
     *
     * Stops both read_worker, resets the active camera and store the input_queue
     */
    void stop_frame_read();

    void start_information_display();

    void stop_information_display();

    void start_frame_record(const std::function<void()>& callback = []() {});

    void stop_frame_record();

    void start_chart_record(const std::function<void()>& callback = []() {});

    void stop_chart_record();

    /*! \brief Start compute worker */
    void start_compute_worker();

    void start_compute();

    void stop_compute();

    // Always close the 3D cuts before calling this function
    void stop_all_worker_controller();

  private:
    CudaStreams cuda_streams_;

    /*! \brief Error callback function. This function will be called when an error occurs in a worker. */
    std::function<void(const std::exception&)> error_callback_;

    worker::ThreadWorkerController<worker::CameraFrameReadWorker> camera_read_worker_controller_;
    worker::ThreadWorkerController<worker::FileFrameReadWorker> file_read_worker_controller_;
    worker::ThreadWorkerController<worker::FrameRecordWorker> frame_record_worker_controller_;
    worker::ThreadWorkerController<worker::ChartRecordWorker> chart_record_worker_controller_;
    worker::ThreadWorkerController<worker::InformationWorker> info_worker_controller_;
    worker::ThreadWorkerController<worker::ComputeWorker> compute_worker_controller_;
}

} // namespace holovibes::api