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

#include "information_container.hh"
#include "queue.hh"

namespace holovibes
{
    using IndicationType = InformationContainer::IndicationType;
    using FpsType = InformationContainer::FpsType;
    using QueueType = Queue::QueueType;
    using ProgressType = InformationContainer::ProgressType;
    using MutexGuard = std::lock_guard<std::mutex>;

    const std::unordered_map<IndicationType, std::string> InformationContainer::indication_type_to_string_ =
    {
        { IndicationType::IMG_SOURCE, "Image Source" },

        { IndicationType::INPUT_FORMAT, "Input Format" },
        { IndicationType::OUTPUT_FORMAT, "Output Format" },

        { IndicationType::CUTS_SLICE_CURSOR, "Cuts Slice Cursor"},
    };

    const std::unordered_map<FpsType, std::string> InformationContainer::fps_type_to_string_ =
    {
        { FpsType::INPUT_FPS, "Input FPS" },
        { FpsType::OUTPUT_FPS, "Output FPS" },
        { FpsType::SAVING_FPS, "Saving FPS" },
    };

    const std::unordered_map<QueueType, std::string> InformationContainer::queue_type_to_string_ =
    {
        { QueueType::INPUT_QUEUE, "Input Queue" },
        { QueueType::OUTPUT_QUEUE, "Output Queue" },
        { QueueType::RECORD_QUEUE, "Record Queue" },
    };

    void InformationContainer::add_indication(IndicationType indication_type, const std::string& info)
    {
        MutexGuard m_guard(mutex_);
        indication_map_.insert_or_assign(indication_type, info);
    }

    void InformationContainer::add_processed_fps(FpsType fps_type, std::atomic<unsigned int>& processed_fps)
    {
        MutexGuard m_guard(mutex_);
        fps_map_.insert_or_assign(fps_type, &processed_fps);
    }

    void InformationContainer::add_queue_size(QueueType queue_type, const std::atomic<unsigned int>& cur_size,
                                                                    const std::atomic<unsigned int>& max_size)
    {
        if (queue_type == QueueType::UNDEFINED)
            return;

        MutexGuard m_guard(mutex_);
        queue_size_map_.insert_or_assign(queue_type, std::make_pair(&cur_size, &max_size));
    }

    void InformationContainer::add_progress_index(ProgressType progress_type, const std::atomic<unsigned int>& cur_progress,
                                                                            const std::atomic<unsigned int>& max_progress)
    {
        MutexGuard m_guard(mutex_);
        progress_index_map_.insert_or_assign(progress_type, std::make_pair(&cur_progress, &max_progress));
    }

    void InformationContainer::remove_indication(IndicationType indication_type)
    {
        MutexGuard m_guard(mutex_);
        indication_map_.erase(indication_type);
    }

    void InformationContainer::remove_processed_fps(FpsType fps_type)
    {
        MutexGuard m_guard(mutex_);
        fps_map_.erase(fps_type);
    }

    void InformationContainer::remove_queue_size(QueueType queue_type)
    {
        if (queue_type == QueueType::UNDEFINED)
            return;

        MutexGuard m_guard(mutex_);
        queue_size_map_.erase(queue_type);
    }

    void InformationContainer::remove_progress_index(ProgressType progress_type)
    {
        MutexGuard m_guard(mutex_);
        progress_index_map_.erase(progress_type);
    }

    void InformationContainer::clear()
    {
        MutexGuard m_guard(mutex_);
        indication_map_.clear();
        fps_map_.clear();
        queue_size_map_.clear();
        progress_index_map_.clear();
    }
} // namespace holovibes
