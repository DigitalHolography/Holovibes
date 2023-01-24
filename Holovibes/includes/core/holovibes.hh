/*! \file
 *
 * \brief class to use HoloVibes
 */
#pragma once

#include "icamera.hh"
#include "pipe.hh"

// Worker & Controller
#include "thread_worker_controller.hh"
#include "file_frame_read_worker.hh"
#include "camera_frame_read_worker.hh"
#include "information_worker.hh"
#include "chart_record_worker.hh"
#include "frame_record_worker.hh"
#include "batch_gpib_worker.hh"
#include "compute_worker.hh"

#include "common.cuh"

// Enum
#include "import_struct.hh"

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

// FIXME API : NEED TO BE RECODED

namespace holovibes::gui
{
class MainWindow;
} // namespace holovibes::gui

/*! \brief Contains all function and structure needed to computes data */
namespace holovibes
{
class Queue;
class BatchInputQueue;

/*! \class Holovibes
 *
 * \brief Core class to use HoloVibes
 *
 * This class does not depends on the user interface (classes under the
 * holovibes namespace can be seen as a library).
 *
 * It contains high-level ressources (Pipe, Camera, Recorder ...). These
 * ressources are shared between threads and should be allocated in threads
 * themselves.
 */
class Holovibes
{
    struct CudaStreams
    {
        CudaStreams()
        {
            cudaSafeCall(cudaStreamCreateWithPriority(&reader_stream, cudaStreamDefault, CUDA_STREAM_READER_PRIORITY));
            cudaSafeCall(
                cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, CUDA_STREAM_COMPUTE_PRIORITY));
            cudaSafeCall(
                cudaStreamCreateWithPriority(&recorder_stream, cudaStreamDefault, CUDA_STREAM_RECORDER_PRIORITY));
        }

        /*! \brief Used when the device is reset. Recreate the streams.
         *
         * This might cause a small memory leak, but at least it doesn't cause a crash/segfault
         */
        void reload()
        {
            cudaSafeCall(cudaStreamCreateWithPriority(&reader_stream, cudaStreamDefault, CUDA_STREAM_READER_PRIORITY));
            cudaSafeCall(
                cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, CUDA_STREAM_COMPUTE_PRIORITY));
            cudaSafeCall(
                cudaStreamCreateWithPriority(&recorder_stream, cudaStreamDefault, CUDA_STREAM_RECORDER_PRIORITY));
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

  public:
    static Holovibes& instance();

    std::shared_ptr<BatchInputQueue>& get_gpu_input_queue() { return gpu_input_queue_; }
    std::shared_ptr<Queue>& get_gpu_output_queue() { return gpu_output_queue_; }
    std::shared_ptr<Pipe>& get_compute_pipe() { return compute_pipe_; }
    std::shared_ptr<camera::ICamera>& get_active_camera() { return active_camera_; }
    void set_active_camera(std::shared_ptr<camera::ICamera> camera) { active_camera_ = camera; }
    const CudaStreams& get_cuda_streams() const { return cuda_streams_; }

    // clang-format off
    worker::ThreadWorkerController<worker::FileFrameReadWorker>& get_file_frame_read_worker_controller() { return file_frame_read_worker_controller_; }
    worker::ThreadWorkerController<worker::CameraFrameReadWorker>& get_camera_read_worker_controller() { return camera_read_worker_controller_; }
    worker::ThreadWorkerController<worker::FrameRecordWorker>& get_frame_record_worker_controller() { return frame_record_worker_controller_; }
    worker::ThreadWorkerController<worker::ChartRecordWorker>& get_chart_record_worker_controller() { return chart_record_worker_controller_; }
    worker::ThreadWorkerController<worker::BatchGPIBWorker>& get_batch_gpib_worker_controller() { return batch_gpib_worker_controller_; }
    worker::ThreadWorkerController<worker::InformationWorker>& get_info_worker_controller() { return info_worker_controller_; }
    worker::ThreadWorkerController<worker::ComputeWorker>& get_compute_worker_controller() { return compute_worker_controller_; }
    // clang-format on

    void init_gpu_queues();
    void destroy_gpu_queues();

    void start_file_frame_read();
    void stop_file_frame_read();

    void start_camera_frame_read();
    void stop_camera_frame_read();

    void start_frame_record();
    void stop_frame_record();

    void start_chart_record();
    void stop_chart_record();

    void start_information_display();
    void stop_information_display();

    void start_compute();
    void stop_compute();

    void create_pipe();
    void sync_pipe();
    void destroy_pipe();

    /*! \brief Reload the cuda streams when the device is reset */
    void reload_streams();

    /*! \brief function called when some thread throws an exception */
    std::function<void(const std::exception&)> error_callback_;

    void set_error_callback(std::function<void(const std::exception&)> func) { error_callback_ = func; }

  private:
    /*! \brief Construct the holovibes object. */
    Holovibes() = default;

    worker::ThreadWorkerController<worker::FileFrameReadWorker> file_frame_read_worker_controller_{};
    worker::ThreadWorkerController<worker::CameraFrameReadWorker> camera_read_worker_controller_{};
    std::shared_ptr<camera::ICamera> active_camera_{nullptr};

    worker::ThreadWorkerController<worker::FrameRecordWorker> frame_record_worker_controller_{};
    worker::ThreadWorkerController<worker::ChartRecordWorker> chart_record_worker_controller_{};
    worker::ThreadWorkerController<worker::BatchGPIBWorker> batch_gpib_worker_controller_{};

    worker::ThreadWorkerController<worker::InformationWorker> info_worker_controller_{};

    worker::ThreadWorkerController<worker::ComputeWorker> compute_worker_controller_{};
    std::shared_ptr<Pipe> compute_pipe_{nullptr};

    std::shared_ptr<BatchInputQueue> gpu_input_queue_{nullptr};
    std::shared_ptr<Queue> gpu_output_queue_{nullptr};

    CudaStreams cuda_streams_;
};
} // namespace holovibes
