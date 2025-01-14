/*! \file
 *
 * \brief Declaration of the FrameRecordWorker class.
 */
#pragma once

#include <array>
#include <atomic>
#include <cuda_runtime.h>
#include <optional>

#include "enum_record_mode.hh"
#include "logger.hh"
#include "output_frame_file.hh"
#include "queue.hh"
#include "settings/settings_container.hh"
#include "settings/settings.hh"
#include "worker.hh"

#pragma region Settings configuration
// clang-format off

#define ONRESTART_SETTINGS                    \
  holovibes::settings::RecordFilePath,        \
  holovibes::settings::RecordFrameCount,      \
  holovibes::settings::RecordMode,            \
  holovibes::settings::RecordFrameOffset,     \
  holovibes::settings::OutputBufferSize,      \
  holovibes::settings::FrameSkip,             \
  holovibes::settings::Mp4Fps

#define ALL_SETTINGS ONRESTART_SETTINGS

// clang-format on
#pragma endregion

#define FPS_LAST_X_VALUES 16

namespace holovibes
{
// Fast forward declarations
class Queue;
class ICompute;
class Holovibes;
} // namespace holovibes

namespace holovibes::worker
{
/*! \class FrameRecordWorker
 *
 * \brief Class used to record frames
 */
class FrameRecordWorker final : public Worker
{
  public:
    /*! \brief Constructor
     *
     * \param file_path The path of the file to record
     * \param nb_frames_to_record The number of frames to record
     * \param nb_frames_skip Number of frames to skip before starting
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    FrameRecordWorker(InitSettings settings, cudaStream_t stream, std::atomic<std::shared_ptr<Queue>>& record_queue)
        : Worker()
        , stream_(stream)
        , onrestart_settings_(settings)
        , record_queue_(record_queue)
    {
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

        if constexpr (has_setting_v<T, decltype(onrestart_settings_)>)
            onrestart_settings_.update_setting(setting);
    }

  private:
    /**
     * @brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting_v<T, decltype(onrestart_settings_)>)
            return onrestart_settings_.get<T>().value;
    }

    /*! \brief Open the output file
     *
     * \param[in] frame_count The number of frames to record
     *
     * \return The output file
     */
    io_files::OutputFrameFile* open_output_file(const uint frame_count);

    /*! \brief Reset the record queue to free memory. */
    void reset_record_queue();

    /*! \brief Integrate Input FPS in fps_buffers if relevant. */
    void integrate_fps_average();

    /*! \brief Check if all frames are saved.
     *
     * \return True if all frames are saved (acquired + saved), false otherwise.
     */
    bool all_frames_saved(uint frames_saved, uint total) const;

    /*! \brief Compute fps_buffer_ average on the correct number of value. */
    size_t compute_fps_average() const;

  private:
    // Average fps is computed with the last FPS_LAST_X_VALUES values of input fps.
    /*! \brief Useful for Input fps value. */
    unsigned int fps_current_index_ = 0;
    /*! \brief Useful for Input fps value. */
    std::array<unsigned int, FPS_LAST_X_VALUES> fps_buffer_ = {0};

    const cudaStream_t stream_;

    /**
     * @brief Contains all the settings of the worker that should be updated
     * on restart.
     */
    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
    /*! \brief The queue in which the frames are stored for record*/
    std::atomic<std::shared_ptr<Queue>>& record_queue_;
};
} // namespace holovibes::worker

namespace holovibes
{
template <typename T>
struct has_setting<T, worker::FrameRecordWorker> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
