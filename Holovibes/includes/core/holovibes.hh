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
#include "benchmark_worker.hh"
#include "chart_record_worker.hh"
#include "frame_record_worker.hh"
#include "compute_worker.hh"

#include "common.cuh"
#include "settings/settings.hh"
#include "settings/settings_container.hh"
#include "custom_type_traits.hh"
#include "logger.hh"

// Enum
#include "enum_camera_kind.hh"
#include "enum_recorded_eye_type.hh"
#include "enum_record_mode.hh"
#include "enum_import_type.hh"
#include "enum_device.hh"
#include "enum_recorded_data_type.hh"

#include <string>

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                                        \
    holovibes::settings::InputFPS,                               \
    holovibes::settings::InputFilePath,                          \
    holovibes::settings::InputFd,                                \
    holovibes::settings::ImportType,                             \
    holovibes::settings::CameraKind,                             \
    holovibes::settings::FileBufferSize,                         \
    holovibes::settings::FileLoadKind,                           \
    holovibes::settings::InputFileStartIndex,                    \
    holovibes::settings::InputFileEndIndex,                      \
    holovibes::settings::RecordFilePath,                         \
    holovibes::settings::RecordFrameCount,                       \
    holovibes::settings::RecordMode,                             \
    holovibes::settings::RecordedEye,                            \
    holovibes::settings::RecordFrameOffset,                      \
    holovibes::settings::OutputBufferSize,                       \
    holovibes::settings::ImageType,                              \
    holovibes::settings::Unwrap2d,                               \
    holovibes::settings::X,                                      \
    holovibes::settings::Y,                                      \
    holovibes::settings::P,                                      \
    holovibes::settings::Q,                                      \
    holovibes::settings::XY,                                     \
    holovibes::settings::XZ,                                     \
    holovibes::settings::YZ,                                     \
    holovibes::settings::Filter2d,                               \
    holovibes::settings::XYContrastRange,                        \
    holovibes::settings::XZContrastRange,                        \
    holovibes::settings::YZContrastRange,                        \
    holovibes::settings::Filter2dContrastRange,                  \
    holovibes::settings::CurrentWindow,                          \
    holovibes::settings::LensViewEnabled,                        \
    holovibes::settings::ChartDisplayEnabled,                    \
    holovibes::settings::Filter2dEnabled,                        \
    holovibes::settings::Filter2dViewEnabled,                    \
    holovibes::settings::FftShiftEnabled,                        \
    holovibes::settings::RegistrationEnabled,                    \
    holovibes::settings::RawViewEnabled,                         \
    holovibes::settings::CutsViewEnabled,                        \
    holovibes::settings::RenormEnabled,                          \
    holovibes::settings::ReticleScale,                           \
    holovibes::settings::RegistrationZone,                       \
    holovibes::settings::ReticleDisplayEnabled,                  \
    holovibes::settings::Filter2dN1,                             \
    holovibes::settings::Filter2dN2,                             \
    holovibes::settings::Filter2dSmoothLow,                      \
    holovibes::settings::Filter2dSmoothHigh,                     \
    holovibes::settings::FilterFileName,                         \
    holovibes::settings::FrameAcquisitionEnabled,                \
    holovibes::settings::ChartRecordEnabled,                     \
    holovibes::settings::DisplayRate,                            \
    holovibes::settings::InputBufferSize,                        \
    holovibes::settings::RecordBufferSize,                       \
    holovibes::settings::ContrastLowerThreshold,                 \
    holovibes::settings::RawBitshift,                            \
    holovibes::settings::ContrastUpperThreshold,                 \
    holovibes::settings::RenormConstant,                         \
    holovibes::settings::CutsContrastPOffset,                    \
    holovibes::settings::BatchSize,                              \
    holovibes::settings::TimeStride,                             \
    holovibes::settings::TimeTransformationSize,                 \
    holovibes::settings::SpaceTransformation,                    \
    holovibes::settings::TimeTransformation,                     \
    holovibes::settings::Lambda,                                 \
    holovibes::settings::ZDistance,                              \
    holovibes::settings::ConvolutionMatrix,                      \
    holovibes::settings::DivideConvolutionEnabled,               \
    holovibes::settings::ConvolutionFileName,                    \
    holovibes::settings::ComputeMode,                            \
    holovibes::settings::PixelSize,                              \
    holovibes::settings::IsComputationStopped,                   \
    holovibes::settings::SignalZone,                             \
    holovibes::settings::NoiseZone,                              \
    holovibes::settings::CompositeZone,                          \
    holovibes::settings::ReticleZone,                            \
    holovibes::settings::InputFilter,                            \
    holovibes::settings::TimeTransformationCutsOutputBufferSize, \
    holovibes::settings::CompositeKind,                          \
    holovibes::settings::CompositeAutoWeights,                   \
    holovibes::settings::RGB,                                    \
    holovibes::settings::HSV,                                    \
    holovibes::settings::ZFFTShift,                              \
    holovibes::settings::RecordQueueLocation,                    \
    holovibes::settings::BenchmarkMode,                          \
    holovibes::settings::FrameSkip,                              \
    holovibes::settings::Mp4Fps,                                 \
    holovibes::settings::CameraFps,                              \
    holovibes::settings::DataType

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
        CudaStreams() { reload(); }

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
    /*! \brief Return the input queue (the one storing the input frames)
     *
     * \return std::shared_ptr<BatchInputQueue> The input queue
     */
    inline std::shared_ptr<BatchInputQueue> get_input_queue() const { return input_queue_.load(); }

    /*! \brief Return the record queue (the one storing the computed frames before saving)
     *
     * \return std::shared_ptr<Queue> The record queue
     */
    inline std::atomic<std::shared_ptr<Queue>> get_record_queue() const { return record_queue_.load(); }
    /*! \} */

    /*! \name Getters/Setters
     * \{
     */
    /*! \brief Return the compute pipe and throw if no pipe.
     * user.
     *
     * \return std::shared_ptr<Pipe> The compute pipe
     * \throw std::runtime_error If the compute pipe is not initialized
     */
    inline std::shared_ptr<Pipe> get_compute_pipe()
    {
        auto loaded = compute_pipe_.load();
        if (!loaded)
            throw std::runtime_error("Pipe is not initialized");

        return loaded;
    }

    /*! \brief Return the compute pipe.
     *
     * \return std::shared_ptr<Pipe> The compute pipe
     */
    inline std::shared_ptr<Pipe> get_compute_pipe_no_throw() const { return compute_pipe_.load(); }

    /*! \brief Return the cuda streams
     *
     * \return const CudaStreams& The cuda streams
     */
    inline const Holovibes::CudaStreams& get_cuda_streams() const { return cuda_streams_; }

    /*! \brief Return the path of the camera INI file used.
     *
     * \return const char* the path of the camera INI file of the current camera.
     */
    inline const char* get_camera_ini_name() const
    {
        if (!active_camera_)
            return "";
        return active_camera_->get_ini_name();
    }

    /*! \brief Return whether the recording worker is running or not
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

    /*!
     * \brief Initializes the record queue, depending on the record mode and the device (GPU or CPU)
     *
     */
    void init_record_queue();

    /*! \brief Sets and starts the file_read_worker attribute
     *
     * \param callback
     */
    void start_file_frame_read(const std::function<void()>& callback = []() {});

    /*! \brief Sets the right camera settings, then starts the camera_read_worker (image acquisition)
     * TODO: refacto (see issue #22)
     */
    void start_camera_frame_read();

    /*! \brief Handle frame reading interruption
     *
     * Stops both read_worker, resets the active camera and store the input_queue
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

    void start_benchmark();

    void stop_benchmark();

    void start_compute();

    void stop_compute();

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
        LOG_TRACE("[Holovibes] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting_v<T, decltype(realtime_settings_)>)
            realtime_settings_.update_setting(setting);

        if constexpr (has_setting_v<T, worker::FileFrameReadWorker>)
            file_read_worker_controller_.update_setting(setting);

        if constexpr (has_setting_v<T, worker::FrameRecordWorker>)
            frame_record_worker_controller_.update_setting(setting);

        if constexpr (has_setting_v<T, Pipe>)
            if (compute_pipe_.load() != nullptr)
                compute_pipe_.load()->update_setting(setting);
    }

    template <typename T>
    inline T get_setting()
    {
        return realtime_settings_.get<T>();
    }

  private:
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting_v<T, decltype(realtime_settings_.settings_)>)
            return realtime_settings_.get<T>().value;
    }

    /*! \brief Construct the holovibes object. */
    Holovibes()
        : realtime_settings_(std::make_tuple(settings::InputFPS{10000},
                                             settings::InputFilePath{std::string("")},
                                             settings::ImportType{ImportType::None},
                                             settings::InputFd{camera::FrameDescriptor{}},
                                             settings::CameraKind{CameraKind::NONE},
                                             settings::FileBufferSize{1024},
                                             settings::FileLoadKind{FileLoadKind::REGULAR},
                                             settings::InputFileStartIndex{0},
                                             settings::InputFileEndIndex{60},
                                             settings::RecordFilePath{std::string("")},
                                             settings::RecordFrameCount{std::nullopt},
                                             settings::RecordMode{RecordMode::RAW},
                                             settings::RecordedEye{RecordedEyeType::NONE},
                                             settings::RecordFrameOffset{0},
                                             settings::OutputBufferSize{1024},
                                             settings::ImageType{ImgType::Modulus},
                                             settings::Unwrap2d{false},
                                             settings::X{ViewXY{}},
                                             settings::Y{ViewXY{}},
                                             settings::P{ViewPQ{}},
                                             settings::Q{ViewPQ{}},
                                             settings::XY{ViewXYZ{}},
                                             settings::XZ{ViewXYZ{}},
                                             settings::YZ{ViewXYZ{}},
                                             settings::Filter2d{ViewWindow{}},
                                             settings::XYContrastRange{ContrastRange{}},
                                             settings::XZContrastRange{ContrastRange{}},
                                             settings::YZContrastRange{ContrastRange{}},
                                             settings::Filter2dContrastRange{ContrastRange{}},
                                             settings::CurrentWindow{WindowKind::XYview},
                                             settings::LensViewEnabled{false},
                                             settings::ChartDisplayEnabled{false},
                                             settings::Filter2dEnabled{false},
                                             settings::Filter2dViewEnabled{false},
                                             settings::FftShiftEnabled{false},
                                             settings::RegistrationEnabled{false},
                                             settings::RawViewEnabled{false},
                                             settings::CutsViewEnabled{false},
                                             settings::RenormEnabled{true},
                                             settings::ReticleScale{0.5f},
                                             settings::RegistrationZone{0.7f},
                                             settings::ReticleDisplayEnabled{false},
                                             settings::Filter2dN1{0},
                                             settings::Filter2dN2{1},
                                             settings::Filter2dSmoothLow{0},
                                             settings::Filter2dSmoothHigh{1},
                                             settings::FilterFileName{std::string("")},
                                             settings::FrameAcquisitionEnabled{false},
                                             settings::ChartRecordEnabled{false},
                                             settings::DisplayRate{24},
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
                                             settings::ZDistance{0.0f},
                                             settings::ConvolutionMatrix{std::vector<float>{}},
                                             settings::DivideConvolutionEnabled{false},
                                             settings::ConvolutionFileName{std::string("")},
                                             settings::ComputeMode{Computation::Raw},
                                             settings::PixelSize{12.0f},
                                             settings::IsComputationStopped{true},
                                             settings::TimeTransformationCutsOutputBufferSize{512},
                                             settings::SignalZone{units::RectFd{}},
                                             settings::NoiseZone{units::RectFd{}},
                                             settings::CompositeZone{units::RectFd{}},
                                             settings::ReticleZone{units::RectFd{}},
                                             settings::InputFilter{{}},
                                             settings::CompositeKind{CompositeKind::RGB},
                                             settings::CompositeAutoWeights{false},
                                             settings::RGB{CompositeRGB{}},
                                             settings::HSV{CompositeHSV{}},
                                             settings::ZFFTShift{false},
                                             settings::RecordQueueLocation{Device::CPU},
                                             settings::BenchmarkMode{false},
                                             settings::FrameSkip{0},
                                             settings::Mp4Fps{24},
                                             settings::CameraFps{0},
                                             settings::DataType{RecordedDataType::RAW}))
    {
    }

    worker::ThreadWorkerController<worker::FileFrameReadWorker> file_read_worker_controller_;
    worker::ThreadWorkerController<worker::CameraFrameReadWorker> camera_read_worker_controller_;

    worker::ThreadWorkerController<worker::FrameRecordWorker> frame_record_worker_controller_;
    worker::ThreadWorkerController<worker::ChartRecordWorker> chart_record_worker_controller_;

    worker::ThreadWorkerController<worker::BenchmarkWorker> benchmark_worker_controller_;

    worker::ThreadWorkerController<worker::ComputeWorker> compute_worker_controller_;
    std::atomic<std::shared_ptr<Pipe>> compute_pipe_{nullptr};

    /*! \name Frames queue (GPU)
     * \{
     */
    std::atomic<std::shared_ptr<BatchInputQueue>> input_queue_{nullptr};
    std::atomic<std::shared_ptr<Queue>> record_queue_{nullptr};
    /*! \} */

    CudaStreams cuda_streams_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;

  public:
    std::shared_ptr<camera::ICamera> active_camera_{nullptr};
};
} // namespace holovibes
