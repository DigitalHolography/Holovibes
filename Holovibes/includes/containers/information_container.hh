/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "queue.hh"

namespace holovibes
{
namespace worker
{
class InformationWorker;
}

class InformationContainer
{
  public:
    using QueueType = Queue::QueueType;

    enum class IndicationType
    {
        IMG_SOURCE,

        INPUT_FORMAT,
        OUTPUT_FORMAT,

        CUTS_SLICE_CURSOR,
    };

    enum class FpsType
    {
        INPUT_FPS,
        OUTPUT_FPS,
        SAVING_FPS,
    };

    enum class ProgressType
    {
        FILE_READ,
        FRAME_RECORD,
        CHART_RECORD,
    };

    InformationContainer() = default;

    void set_display_info_text_function(
        const std::function<void(const std::string&)>& function);

    void set_update_progress_function(
        const std::function<void(ProgressType, size_t, size_t)>& function);

    void add_indication(IndicationType indication_type,
                        const std::string& indication);

    void add_processed_fps(FpsType fps_type,
                           std::atomic<unsigned int>& processed_fps);

    void add_queue_size(QueueType queue_type,
                        const std::atomic<unsigned int>& cur_size,
                        const std::atomic<unsigned int>& max_size);

    void add_progress_index(ProgressType progress_type,
                            const std::atomic<unsigned int>& cur_progress,
                            const std::atomic<unsigned int>& max_progress);

    void remove_indication(IndicationType info_type);

    void remove_processed_fps(FpsType fps_type);

    void remove_queue_size(QueueType queue_type);

    void remove_progress_index(ProgressType progress_type);

    void clear();

  private:
    friend class worker::InformationWorker;

    mutable std::mutex mutex_;

    static const std::unordered_map<IndicationType, std::string>
        indication_type_to_string_;

    std::map<IndicationType, std::string> indication_map_;

    static const std::unordered_map<FpsType, std::string> fps_type_to_string_;

    std::map<FpsType, std::atomic<unsigned int>*> fps_map_;

    static const std::unordered_map<QueueType, std::string>
        queue_type_to_string_;

    std::map<QueueType,
             std::pair<const std::atomic<unsigned int>*,
                       const std::atomic<unsigned int>*>>
        queue_size_map_;

    std::function<void(const std::string&)> display_info_text_function_ =
        [](const std::string&) {};

    std::unordered_map<ProgressType,
                       std::pair<const std::atomic<unsigned int>*,
                                 const std::atomic<unsigned int>*>>
        progress_index_map_;

    std::function<void(ProgressType, size_t, size_t)>
        update_progress_function_ = [](ProgressType, size_t, size_t) {};
};
} // namespace holovibes

#include "information_container.hxx"