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
#include "compute_worker.hh"

#include "common.cuh"
#include "settings/settings.hh"
#include "settings/settings_container.hh"
#include "utils/custom_type_traits.hh"

// Enum
#include "enum_camera_kind.hh"
#include "enum_record_mode.hh"

#include <spdlog/spdlog.h>
#include <string>

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::InputFPS,                 \
    holovibes::settings::InputFilePath,            \
    holovibes::settings::FileBufferSize,           \
    holovibes::settings::LoopOnInputFile,          \
    holovibes::settings::LoadFileInGPU,            \
    holovibes::settings::InputFileStartIndex,      \
    holovibes::settings::InputFileEndIndex,        \
    holovibes::settings::RecordFilePath,           \
    holovibes::settings::RecordFrameCount,         \
    holovibes::settings::RecordMode,               \
    holovibes::settings::RecordFrameSkip,          \
    holovibes::settings::OutputBufferSize,         \
    holovibes::settings::BatchEnabled,             \
    holovibes::settings::BatchFilePath,            \
    holovibes::settings::ImageType,                \
    holovibes::settings::X,                        \
    holovibes::settings::Y,                        \
    holovibes::settings::P,                        \
    holovibes::settings::Q,                        \
    holovibes::settings::XY,                       \
    holovibes::settings::XZ,                       \
    holovibes::settings::YZ,                       \
    holovibes::settings::Filter2d,                 \
    holovibes::settings::CurrentWindow,            \
    holovibes::settings::LensViewEnabled,          \
    holovibes::settings::ChartDisplayEnabled,      \
    holovibes::settings::Filter2dEnabled,          \
    holovibes::settings::Filter2dViewEnabled,      \
    holovibes::settings::FftShiftEnabled,          \
    holovibes::settings::RawViewEnabled,           \
    holovibes::settings::CutsViewEnabled,          \
    holovibes::settings::RenormEnabled,            \
    holovibes::settings::ReticleScale,             \
    holovibes::settings::ReticleDisplayEnabled,    \
    holovibes::settings::Filter2dN1,               \
    holovibes::settings::Filter2dN2,               \
    holovibes::settings::Filter2dSmoothLow,        \
    holovibes::settings::Filter2dSmoothHigh,       \
    holovibes::settings::FrameRecordEnabled,       \
    holovibes::settings::ChartRecordEnabled,       \
    holovibes::settings::DisplayRate,              \
    holovibes::settings::InputBufferSize,          \
    holovibes::settings::RecordBufferSize,         \
    holovibes::settings::ContrastLowerThreshold,   \
    holovibes::settings::RawBitshift,              \
    holovibes::settings::ContrastUpperThreshold,   \
    holovibes::settings::RenormConstant,           \
    holovibes::settings::CutsContrastPOffset,      \
    holovibes::settings::BatchSize,                \
    holovibes::settings::TimeStride,               \
    holovibes::settings::TimeTransformationSize,   \
    holovibes::settings::SpaceTransformation,      \
    holovibes::settings::TimeTransformation,       \
    holovibes::settings::Lambda,                   \
    holovibes::settings::ZDistance,                \
    holovibes::settings::ConvolutionEnabled,       \
    holovibes::settings::ConvolutionMatrix,        \
    holovibes::settings::DivideConvolutionEnabled, \
    holovibes::settings::ComputeMode,              \
    holovibes::settings::PixelSize,                \
    holovibes::settings::UnwrapHistorySize,        \
    holovibes::settings::IsComputationStopped,     \
    holovibes::settings::TimeTransformationCutsOutputBufferSize
     
#define ALL_SETTINGS REALTIME_SETTINGS

// clang-format on
#pragma endregion

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
    std::shared_ptr<Pipe> get_compute_pipe_nothrow();

    const CudaStreams& get_cuda_streams() const;

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

    /*! \brief Say if the worker recording raw/holo/cuts is running.
     *
     * \return bool true if recording, else false
     */
    bool is_recording() const;

    /*! \} */

    /*! \brief Initializes the input queue
     *
     * \param fd frame descriptor of the camera
     * \param input_queue_size size of the input queue
     */
    void init_input_queue(const camera::FrameDescriptor& fd, const unsigned int input_queue_size);

    /*! \brief Sets and starts the file_read_worker attribute
     *
     * \param callback
     */
    void start_file_frame_read(const std::function<void()>& callback = []() {});

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
    void start_frame_record(const std::function<void()>& callback = []() {});

    void stop_frame_record();

    void start_chart_record(const std::function<void()>& callback = []() {});

    void stop_chart_record();

    void start_information_display(const std::function<void()>& callback = []() {});

    void stop_information_display();

    /*! \brief Start compute worker */
    void start_compute_worker(const std::function<void()>& callback = []() {});

    void start_compute(const std::function<void()>& callback = []() {});

    void stop_compute();

    // Always close the 3D cuts before calling this function
    void stop_all_worker_controller();

    void init_pipe();

    /*! \brief Reload the cuda streams when the device is reset */
    void reload_streams();

    /*! \brief This value is set in start_gui or start_cli. It says if we are in cli or gui mode. This information is
     * used to know if queues have to keep contiguity or not. */
    bool is_cli;

    /*! \brief function called when some thread throws an exception */
    std::function<void(const std::exception&)> error_callback_;

    void set_error_callback(std::function<void(const std::exception&)> func) { error_callback_ = func; }

    /**
     * @brief Update a setting. The actual application of the update
     * might ve delayed until a certain event occurs.
     * @tparam T The type of tho update.
     * @param setting The new value of the setting.
     */
    template <typename T>
    inline void update_setting(T setting)
    {
        spdlog::info("[Holovibes] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
            realtime_settings_.update_setting(setting);

        if constexpr (has_setting<T, worker::FileFrameReadWorker>::value)
            file_read_worker_controller_.update_setting(setting);

        if constexpr (has_setting<T, worker::FrameRecordWorker>::value)
            frame_record_worker_controller_.update_setting(setting);

        if (compute_pipe_.load() != nullptr)
        {
            if constexpr (has_setting<T, Pipe>::value)
                compute_pipe_.load()->update_setting(setting);
        }
    }

    template <typename T>
    inline T get_setting()
    {
        auto all_settings = std::tuple_cat(realtime_settings_.settings_);
        return std::get<T>(all_settings);
    }

  private:
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_.settings_)>::value)
        {
            return realtime_settings_.get<T>().value;
        }
    }
    

    /*! \brief Construct the holovibes object. */
    Holovibes()
        : realtime_settings_(std::make_tuple(settings::InputFPS{60},
                                             settings::InputFilePath{std::string("")},
                                             settings::FileBufferSize{1024},
                                             settings::LoopOnInputFile{true},
                                             settings::LoadFileInGPU{false},
                                             settings::InputFileStartIndex{0},
                                             settings::InputFileEndIndex{60},
                                             settings::RecordFilePath{std::string("")},
                                             settings::RecordFrameCount{0},
                                             settings::RecordMode{RecordMode::NONE},
                                             settings::RecordFrameSkip{0},
                                             settings::OutputBufferSize{1024},
                                             settings::BatchEnabled{false},
                                             settings::BatchFilePath{std::string("")},
                                             settings::ImageType{ImgType::Modulus},
                                             settings::X{ViewXY{}},
                                             settings::Y{ViewXY{}},
                                             settings::P{ViewPQ{}},
                                             settings::Q{ViewPQ{}},
                                             settings::XY{ViewXYZ{}},
                                             settings::XZ{ViewXYZ{}},
                                             settings::YZ{ViewXYZ{}},
                                             settings::Filter2d{ViewWindow{}},
                                             settings::CurrentWindow{WindowKind::XYview},
                                             settings::LensViewEnabled{false},
                                             settings::ChartDisplayEnabled{false},
                                             settings::Filter2dEnabled{false},
                                             settings::Filter2dViewEnabled{false},
                                             settings::FftShiftEnabled{false},
                                             settings::RawViewEnabled{false},
                                             settings::CutsViewEnabled{false},
                                             settings::RenormEnabled{true},
                                             settings::ReticleScale{0.5f},
                                             settings::ReticleDisplayEnabled{false},
                                             settings::Filter2dN1{0},
                                             settings::Filter2dN2{1},
                                             settings::Filter2dSmoothLow{0},
                                             settings::Filter2dSmoothHigh{1},
                                             settings::FrameRecordEnabled{false},
                                             settings::ChartRecordEnabled{false},
                                             settings::DisplayRate{30},
                                             settings::InputBufferSize{512},
                                             settings::RecordBufferSize{1024},
                                             settings::ContrastLowerThreshold{0.5f},
                                             settings::RawBitshift{0},
                                             settings::ContrastUpperThreshold{99.5f},
                                             settings::RenormConstant{5},
                                             settings::CutsContrastPOffset{2},
                                             settings::BatchSize{1},
                                             settings::TimeStride{1},
                                             settings::TimeTransformationSize{1},
                                             settings::SpaceTransformation{SpaceTransformation::NONE},
                                             settings::TimeTransformation{TimeTransformation::NONE},
                                             settings::Lambda{852e-9f},
                                             settings::ZDistance{1.50f},
                                             settings::ConvolutionEnabled{false},
                                             settings::ConvolutionMatrix{std::vector<float>{}},
                                             settings::DivideConvolutionEnabled{false},
                                             settings::ComputeMode{Computation::Raw},
                                             settings::PixelSize{12.0f},
                                             settings::UnwrapHistorySize{1},
                                             settings::IsComputationStopped{true},
                                             settings::TimeTransformationCutsOutputBufferSize{512}))
    {
    }

    worker::ThreadWorkerController<worker::FileFrameReadWorker> file_read_worker_controller_;
    worker::ThreadWorkerController<worker::CameraFrameReadWorker> camera_read_worker_controller_;
    std::shared_ptr<camera::ICamera> active_camera_{nullptr};

    worker::ThreadWorkerController<worker::FrameRecordWorker> frame_record_worker_controller_;
    worker::ThreadWorkerController<worker::ChartRecordWorker> chart_record_worker_controller_;

    worker::ThreadWorkerController<worker::InformationWorker> info_worker_controller_;

    worker::ThreadWorkerController<worker::ComputeWorker> compute_worker_controller_;
    std::atomic<std::shared_ptr<Pipe>> compute_pipe_{nullptr};

    /*! \name Frames queue (GPU)
     * \{
     */
    std::atomic<std::shared_ptr<BatchInputQueue>> gpu_input_queue_{nullptr};
    std::atomic<std::shared_ptr<Queue>> gpu_output_queue_{nullptr};
    /*! \} */

    CudaStreams cuda_streams_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
};
} // namespace holovibes

#include "holovibes.hxx"