/*! \file
 *
 * \brief class to use HoloVibes
 */
#pragma once

#include "compute_descriptor.hh"
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
#include "enum_camera_kind.hh"
#include "enum_record_mode.hh"

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

/*! \brief Contains all function and structure needed to computes data */
namespace holovibes
{
namespace gui
{
class MainWindow;
}

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

    /*! \name Queue getters
     * \{
     */
    /*! \brief Used to record frames */
    std::shared_ptr<BatchInputQueue> get_gpu_input_queue();

    /*! \brief Used to display frames */
    std::shared_ptr<Queue> get_gpu_output_queue();
    /*! \} */

    /*! \name Getters/Setters
     * \{
     */
    std::shared_ptr<Pipe> get_compute_pipe();

    /*! \return Common ComputeDescriptor */
    ComputeDescriptor& get_cd();

    const CudaStreams& get_cuda_streams() const;

    /*! \brief Set ComputeDescriptor options
     *
     * Used when options are loaded from an INI file.
     *
     * \param cd ComputeDescriptor to load
     */
    void set_cd(const ComputeDescriptor& cd);

    /*! \return Corresponding Camera INI file path */
    const char* get_camera_ini_name() const;

    /*! \brief Get zb = N d^2 / lambda
     *
     * Is updated everytime the camera changes or lamdba changes
     * N = frame height
     * d = pixel size
     * lambda = wavelength
     *
     * \return const float
     */
    const float get_boundary();

    /*! \} */

    /*! \brief Initializes the input queue
     *
     * \param fd frame descriptor of the camera
     * \param input_queue_size size of the input queue
     */
    void init_input_queue(const camera::FrameDescriptor& fd, const unsigned int input_queue_size);

    /*! \brief Sets and starts the file_read_worker attribute
     *
     * \param file_path
     * \param loop
     * \param fps
     * \param first_frame_id
     * \param nb_frames_to_read
     * \param load_file_in_gpu
     * \param callback
     */
    void start_file_frame_read(
        const std::string& file_path,
        bool loop,
        unsigned int fps,
        unsigned int first_frame_id,
        unsigned int nb_frames_to_read,
        bool load_file_in_gpu,
        const std::function<void()>& callback = []() {});

    /*! \brief Sets the right camera settings, then starts the camera_read_worker (image acquisition)
     * TODO: refacto (see issue #22)
     *
     * \param camera_kind
     * \param callback
     */
    void start_camera_frame_read(
        CameraKind camera_kind, const std::function<void()>& callback = []() {});

    /*! \brief Handle frame reading interruption
     *
     * Stops both read_worker, resets the active camera and store the gpu_input_queue
     */
    void stop_frame_read();

    /*! \brief Initialize and start the frame record worker controller
     *
     * \param path
     * \param nb_frames_to_record
     * \param record_mode
     * \param nb_frames_skip
     * \param callback
     */
    void start_frame_record(
        const std::string& path,
        std::optional<unsigned int> nb_frames_to_record,
        RecordMode record_mode,
        unsigned int nb_frames_skip = 0,
        const std::function<void()>& callback = []() {});

    void stop_frame_record();

    void start_chart_record(
        const std::string& path,
        const unsigned int nb_points_to_record,
        const std::function<void()>& callback = []() {});

    void stop_chart_record();

    void start_batch_gpib(
        const std::string& batch_input_path,
        const std::string& output_path,
        unsigned int nb_frames_to_record,
        RecordMode record_mode,
        const std::function<void()>& callback = []() {});

    void stop_batch_gpib();

    void start_information_display(const std::function<void()>& callback = []() {});

    void stop_information_display();

    void start_compute(const std::function<void()>& callback = []() {});

    void stop_compute();

    // Always close the 3D cuts before calling this function
    void stop_all_worker_controller();

    void start_cli_record_and_compute(const std::string& path,
                                      std::optional<unsigned int> nb_frames_to_record,
                                      RecordMode record_mode,
                                      unsigned int nb_frames_skip);

    void init_pipe();

    /*! \brief Reload the cuda streams when the device is reset */
    void reload_streams();

    worker::ThreadWorkerController<worker::FrameRecordWorker> frame_record_worker_controller_;

  private:
    /*! \brief Construct the holovibes object. */
    Holovibes() = default;

    worker::ThreadWorkerController<worker::FileFrameReadWorker> file_read_worker_controller_;
    worker::ThreadWorkerController<worker::CameraFrameReadWorker> camera_read_worker_controller_;
    std::shared_ptr<camera::ICamera> active_camera_{nullptr};

    worker::ThreadWorkerController<worker::ChartRecordWorker> chart_record_worker_controller_;

    worker::ThreadWorkerController<worker::BatchGPIBWorker> batch_gpib_worker_controller_;

    worker::ThreadWorkerController<worker::InformationWorker> info_worker_controller_;

    worker::ThreadWorkerController<worker::ComputeWorker> compute_worker_controller_;
    std::atomic<std::shared_ptr<Pipe>> compute_pipe_{nullptr};

    /*! \name Frames queue (GPU)
     * \{
     */
    std::atomic<std::shared_ptr<BatchInputQueue>> gpu_input_queue_{nullptr};
    std::atomic<std::shared_ptr<Queue>> gpu_output_queue_{nullptr};
    /*! \} */

    /*! \brief Common compute descriptor shared between CLI/GUI and the Pipe. */
    ComputeDescriptor cd_;

    CudaStreams cuda_streams_;
};
} // namespace holovibes

#include "holovibes.hxx"
