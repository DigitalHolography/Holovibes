#include "chart_record_worker.hh"
#include "chart_point.hh"

#include "holovibes.hh"
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
    auto& cd = Holovibes::instance().get_cd();

    std::ofstream of(path_);

    // Header displaying
    of << "[#img : " << GSH::instance().time_transformation_size_query().value << ", p : " << cd.p.index << ", lambda : " << cd.lambda
       << ", z : " << cd.zdistance << "]" << std::endl;

    of << "["
       << "Column 1 : avg(signal), "
       << "Column 2 : avg(noise), "
       << "Column 3 : avg(signal) / avg(noise), "
       << "Column 4 : 10 * log10 (avg(signal) / avg(noise)), "
       << "Column 5 : std(signal), "
       << "Column 6 : std(signal) / avg(noise), "
       << "Column 7 : std(signal) / avg(signal)"
       << "]" << std::endl;

    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_record_chart(nb_frames_to_record_);
    while (pipe->get_chart_record_requested() != std::nullopt && !stop_requested_)
        continue;

    auto& chart_queue = *pipe->get_chart_record_queue();

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

    pipe->request_disable_record_chart();
    while (pipe->get_disable_chart_record_requested() && !stop_requested_)
        continue;

    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::CHART_RECORD);
}

} // namespace holovibes::worker
