/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::InputFPS                  \

#define ONRESTART_SETTINGS                         \
    holovibes::settings::InputFilePath,            \
    holovibes::settings::FileBufferSize,           \
    holovibes::settings::LoopOnInputFile,          \
    holovibes::settings::LoadFileInGPU,            \
    holovibes::settings::InputFileStartIndex,      \
    holovibes::settings::InputFileEndIndex

#define ALL_SETTINGS REALTIME_SETTINGS, ONRESTART_SETTINGS

// clang-format on
#pragma endregion

#include "frame_read_worker.hh"
#include "input_frame_file.hh"
#include "settings/settings_container.hh"
#include "settings/settings.hh"
#include "global_state_holder.hh"
#include "utils/custom_type_traits.hh"
#include "utils/fps_limiter.hh"
#include <optional>
#include "logger.hh"

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
    FileFrameReadWorker(FileFrameReadWorker&) = delete;
    FileFrameReadWorker& operator=(FileFrameReadWorker&) = delete;
    FileFrameReadWorker(FileFrameReadWorker&&) = delete;
    FileFrameReadWorker& operator=(FileFrameReadWorker&&) = delete;

    /**
     * @brief Constructor.
     * @tparam InitSettings A tuple type that contains at least all the settings of the FileFrameReadWorker.
     * @param input_queue The queue where the frames should be copied.
     * This is the input queue of the compute pipeline, it should be allocated on GPU memory.
     * @param settings A tuple that contains the initial value of all settings used by the FileFrameReadWorker.
     * It should contain at least all the settings used, but it can carry more.
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    FileFrameReadWorker(std::atomic<std::shared_ptr<BatchInputQueue>>& input_queue, InitSettings settings)
        : FrameReadWorker(input_queue)
        , fast_updates_entry_(GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::FILE_READ))
        , current_nb_frames_read_(fast_updates_entry_->first)
        , total_nb_frames_to_read_(fast_updates_entry_->second)
        , realtime_settings_(settings)
        , onrestart_settings_(settings)
    {
        current_nb_frames_read_ = 0;
        total_nb_frames_to_read_ = static_cast<uint>(onrestart_settings_.get<settings::InputFileEndIndex>().value -
                                                     onrestart_settings_.get<settings::InputFileStartIndex>().value);
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
        LOG_TRACE("[FileFrameReadWorker] [update_setting] {}", typeid(T).name());

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
    /**
     * @brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            return realtime_settings_.get<T>().value;
        }

        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            return onrestart_settings_.get<T>().value;
        }
    }

    /*! \brief Sets the input file to the one in settings and fd + frame_size accordingly. */
    void open_file();

    /*! \brief Checks if the file is loaded in GPU and reads it accordingly. */
    void read_file();

    /*! \brief Init the cpu_buffer and gpu_buffer */
    bool init_frame_buffers();

    /**
     * @brief Free the cpu_buffer and gpu_buffer.
     */
    void free_frame_buffers();

    /**
     * @brief Creates entry in the fast update map to send informations
     * about this worker to the GSH.
     */
    void insert_fast_update_map_entries();

    /**
     * @brief Removes the workers entries from the fast update map of the GSH.
     */
    void remove_fast_update_map_entries();

    /*! \brief Load all the frames of the file in the gpu
     *
     * Read all the frames in cpu and copy them in gpu.
     * Then enqueue the frames one by one in the input_queue
     */
    void read_file_in_gpu();

    /*! \brief Load the frames of the file by batch into the gpu
     *
     * Read batch in cpu and copy it in gpu.
     * Then enqueue the frames one by one in the input_queue
     */
    void read_file_batch();

    /*! \brief Read frames in cpu and copy in gpu
     *
     * \param frames_to_read The number of frames to read
     * \return The number of frames read
     */
    size_t read_copy_file(size_t frames_to_read);

    /*! \brief Enqueue frames_read in the input_queue with a speed related to the given fps
     *
     * \param nb_frames_to_enqueue The number of frames to enqueue from gpu_buffer to input_queue
     */
    void enqueue_loop(size_t nb_frames_to_enqueue);

    /*! \brief Returns the number of frames to allocate depending on whether or not the file is loaded in GPU.
     *
     */
    size_t get_buffer_nb_frames();

  private:
    FastUpdatesHolder<ProgressType>::Value fast_updates_entry_;

    /**
     * @brief Current number of frames read
     */
    std::atomic<unsigned int>& current_nb_frames_read_;

    /**
     * @brief Total number of frames to read at the beginning of the process
     */
    std::atomic<unsigned int>& total_nb_frames_to_read_;

    /**
     * @brief The input file in which the frames are read
     */
    std::unique_ptr<io_files::InputFrameFile> input_file_;

    /**
     * @brief The frame descriptor associated with the opened file.
     */
    std::optional<camera::FrameDescriptor> fd_;

    /**
     * @brief Size of an input frame
     */
    size_t frame_size_;

    /**
     * @brief CPU buffer in which the frames are temporarly stored
     */
    char* cpu_frame_buffer_;

    /**
     * @brief GPU buffer in which the frames are temporarly stored
     */
    char* gpu_frame_buffer_;

    /**
     * @brief Tmp GPU buffer in which the frames are temporarly stored to convert
     * data from packed bits to 16bit
     */
    char* gpu_packed_buffer_;

    /**
     * @brief The Fps limiter used in the enqueue loop to limit the number of frames enqueued
     * per seconds.
     */
    FPSLimiter fps_limiter_;

    /**
     * @brief All the settings used by the FileFrameReadWorker that should be updated
     * in realtime.
     */
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;

    /**
     * @brief All the settings used by the FileFrameReadWorker that can be updated
     * only on restart.
     */
    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
};

} // namespace holovibes::worker

namespace holovibes
{
template <typename T>
struct has_setting<T, worker::FileFrameReadWorker> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
