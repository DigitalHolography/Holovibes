/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#pragma region Settings configuration

#define REALTIME_SETTINGS holovibes::settings::InputFPS

#define ONRESTART_SETTINGS holovibes::settings::InputFilePath, holovibes::settings::FileBufferSize, holovibes::settings::LoopOnInputFile

#define ALL_SETTINGS REALTIME_SETTINGS, ONRESTART_SETTINGS

#pragma endregion

#include "frame_read_worker.hh"
#include "input_frame_file.hh"
#include "settings/settings_container.hh"
#include "settings/settings.hh"
#include "utils/custom_type_traits.hh"
#include "utils/fps_limiter.hh"

// Fast forward declarations
namespace holovibes
{
class Queue;
} // namespace holovibes

namespace holovibes::io_files
{
class InputFrameFile;
} // namespace holovibes::io_files

namespace holovibes::worker
{
/*! \class FileFrameReadWorker
 *
 * \brief    Class used to read frames from a file
 */
class FileFrameReadWorker final : public FrameReadWorker
{
  public:
    /*! \brief Constructor
     *
     * \param file_path  The file path
     * \param first_frame_id Id of the first frame to read
     * \param total_nb_frames_to_read Total number of frames to read
     * \param load_file_in_gpu Whether the file should be load in gpu
     * \param gpu_input_queue The input queue
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    FileFrameReadWorker(unsigned int first_frame_id,
                        unsigned int total_nb_frames_to_read,
                        bool load_file_in_gpu,
                        std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue,
                        InitSettings settings)
        : FrameReadWorker(gpu_input_queue)
        , fast_updates_entry_(GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::FILE_READ))
        , current_nb_frames_read_(fast_updates_entry_->first)
        , total_nb_frames_to_read_(fast_updates_entry_->second)
        , first_frame_id_(first_frame_id)
        , load_file_in_gpu_(load_file_in_gpu)
        , input_file_(nullptr)
        , frame_size_(0)
        , cpu_frame_buffer_(nullptr)
        , gpu_frame_buffer_(nullptr)
        , gpu_packed_buffer_(nullptr)
        , realtime_settings_(settings)
        , onrestart_settings_(settings)
    {
        current_nb_frames_read_ = 0;
        total_nb_frames_to_read_ = total_nb_frames_to_read;
    }

    void run() override;

    /**
     * @brief Update a setting. The actual application of the update
     * might ve delayed until a certain event occurs.
     * @tparam T The type of tho update.
     * @param setting The new value of the setting.
     */
    template <typename T>
    inline void update_setting(T setting)
    {
        spdlog::info("[FileFrameReadWorker] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            realtime_settings_.update_setting(setting);
        }

        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            onrestart_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief Init the cpu_buffer and gpu_buffer */
    bool init_frame_buffers();

    /*! \brief Load all the frames of the file in the gpu
     *
     * Read all the frames in cpu and copy them in gpu.
     * Then enqueue the frames one by one in the gpu_input_queue
     */
    void read_file_in_gpu();

    /*! \brief Load the frames of the file by batch into the gpu
     *
     * Read batch in cpu and copy it in gpu.
     * Then enqueue the frames one by one in the gpu_input_queue
     */
    void read_file_batch();

    /*! \brief Read frames in cpu and copy in gpu
     *
     * \param frames_to_read The number of frames to read
     * \return The number of frames read
     */
    size_t read_copy_file(size_t frames_to_read);

    /*! \brief Enqueue frames_read in the gpu_input_queue with a speed related to the given fps
     *
     * \param nb_frames_to_enqueue The number of frames to enqueue from gpu_buffer to gpu_input_queue
     */
    void enqueue_loop(size_t nb_frames_to_enqueue);

  private:
    FastUpdatesHolder<ProgressType>::Value fast_updates_entry_;

    /*! \brief Current number of frames read */
    std::atomic<unsigned int>& current_nb_frames_read_;
    /*! \brief Total number of frames to read at the beginning of the process */
    std::atomic<unsigned int>& total_nb_frames_to_read_;

    /*! \brief Id of the first frame to read */
    unsigned int first_frame_id_;
    /*! \brief Whether the entire file should be loaded in the gpu */
    bool load_file_in_gpu_;
    /*! \brief The input file in which the frames are read */
    std::unique_ptr<io_files::InputFrameFile> input_file_;
    /*! \brief Size of an input frame */
    size_t frame_size_;
    /*! \brief CPU buffer in which the frames are temporarly stored */
    char* cpu_frame_buffer_;
    /*! \brief GPU buffer in which the frames are temporarly stored */
    char* gpu_frame_buffer_;
    /*! \brief Tmp GPU buffer in which the frames are temporarly stored to convert data from packed bits to 16bit */
    char* gpu_packed_buffer_;

    FPSLimiter fps_limiter_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;

    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
};
} // namespace holovibes::worker