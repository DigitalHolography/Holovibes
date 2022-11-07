#include "chart_record_worker.hh"
#include "chart_point.hh"

#include "holovibes.hh"
#include "API.hh"
#include "icompute.hh"
#include "tools.hh"

namespace holovibes::worker
{
ChartRecordWorker::ChartRecordWorker(const std::string& path, const unsigned int nb_frames_to_record)
    : Worker()
    , path_(get_record_filename(path))
    , nb_frames_to_record_(nb_frames_to_record)
{
}

void ChartRecordWorker::run()
{
    std::ofstream of(path_);

    // Header displaying
    of << "[#img : " << GSH::instance().get_value<TimeTransformationSize>()
       << ", p : " << GSH::instance().get_value<ViewAccuP>().index
       << ", lambda : " << GSH::instance().get_value<Lambda>() << ", z : " << GSH::instance().get_value<ZDistance>()
       << "]" << std::endl;

    of << "["
       << "Column 1 : avg(signal), "
       << "Column 2 : avg(noise), "
       << "Column 3 : avg(signal) / avg(noise), "
       << "Column 4 : 10 * log10 (avg(signal) / avg(noise)), "
       << "Column 5 : std(signal), "
       << "Column 6 : std(signal) / avg(noise), "
       << "Column 7 : std(signal) / avg(signal)"
       << "]" << std::endl;

    {
        api::detail::change_value<ChartRecord>()->set_nb_points_to_record(nb_frames_to_record_);
    }
    while (api::get_compute_pipe().get_export_cache().has_change_requested() && !stop_requested_)
        continue;

    auto& chart_queue = *api::get_compute_pipe().get_chart_record_queue_ptr();

    auto entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::CHART_RECORD);

    std::atomic<unsigned int>& i = entry->first;
    std::atomic<unsigned int>& nb_frames_to_record = entry->second;
    i = 0;
    nb_frames_to_record = nb_frames_to_record_;

    for (; i < nb_frames_to_record_; ++i)
    {
        while (chart_queue.size() <= i && !stop_requested_)
            continue;
        if (stop_requested_)
            break;

        ChartPoint& point = chart_queue[i];
        of << std::fixed << std::setw(11) << std::setprecision(10) << std::setfill('0') << point.avg_signal << ","
           << point.avg_noise << "," << point.avg_signal_div_avg_noise << "," << point.log_avg_signal_div_avg_noise
           << "," << point.std_signal << "," << point.std_signal_div_avg_noise << "," << point.std_signal_div_avg_signal
           << std::endl;
    }

    {
        GSH::instance().change_value<ChartRecord>()->disable();
    }
    while (api::get_compute_pipe().get_export_cache().has_change_requested() && !stop_requested_)
        continue;

    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::CHART_RECORD);
}

} // namespace holovibes::worker
