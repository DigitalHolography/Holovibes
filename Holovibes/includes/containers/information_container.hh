/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

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

        void set_display_info_text_function(const std::function<void(const std::string&)>& function);

        void set_update_progress_function(const std::function<void(ProgressType, size_t, size_t)>& function);

        void add_indication(IndicationType indication_type, const std::string& indication);

        void add_processed_fps(FpsType fps_type, std::atomic<size_t>& processed_fps);

        void add_queue_size(QueueType queue_type, const std::atomic<unsigned int>& cur_size,
                                                    const std::atomic<unsigned int>& max_size);

        void add_progress_index(ProgressType progress_type, const std::atomic<size_t>& cur_progress,
                                                            const std::atomic<size_t>& max_progress);

        void remove_indication(IndicationType info_type);

        void remove_processed_fps(FpsType fps_type);

        void remove_queue_size(QueueType queue_type);

        void remove_progress_index(ProgressType progress_type);

        void clear();

    private:
        friend class worker::InformationWorker;

        mutable std::mutex mutex_;

        static const std::unordered_map<IndicationType, std::string> indication_type_to_string_;

        std::map<IndicationType, std::string> indication_map_;

        static const std::unordered_map<FpsType, std::string> fps_type_to_string_;

        std::map<FpsType, std::atomic<size_t>*> fps_map_;

        static const std::unordered_map<QueueType, std::string> queue_type_to_string_;

        std::map<QueueType, std::pair<const std::atomic<unsigned int>*, const std::atomic<unsigned int>*>> queue_size_map_;

        std::function<void(const std::string&)> display_info_text_function_ = [](const std::string&){};

        std::unordered_map<ProgressType, std::pair<const std::atomic<size_t>*, const std::atomic<size_t>*>> progress_index_map_;

        std::function<void(ProgressType, size_t, size_t)> update_progress_function_ = [](ProgressType, size_t, size_t){};
    };
} // namespace holovibes

#include "information_container.hxx"