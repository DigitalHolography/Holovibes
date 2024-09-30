#include "chart_record_worker.hh"
#include "chart_point.hh"

#include "API.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "tools.hh"

namespace holovibes::worker
{

void ChartRecordWorker::run()
{
    onrestart_settings_.apply_updates();
    std::ofstream of(setting<settings::RecordFilePath>());

    // Header displaying
    of << "[#img : " << setting<settings::TimeTransformationSize>() << ", p : " << setting<settings::P>().start
       << ", lambda : " << api::get_lambda() << ", z : " << setting<settings::ZDistance>() << "]" << std::endl;

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
    pipe->request_record_chart(static_cast<unsigned int>(setting<settings::RecordFrameCount>().value()));
    while (pipe->get_chart_record_requested() != std::nullopt && !stop_requested_)
        continue;

    auto& chart_queue = *pipe->get_chart_record_queue();

    auto entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::CHART_RECORD);

    std::atomic<unsigned int>& i = entry->first;
    std::atomic<unsigned int>& nb_frames_to_record = entry->second;
    i = 0;
    nb_frames_to_record = static_cast<unsigned int>(setting<settings::RecordFrameCount>().value());

    for (; i < setting<settings::RecordFrameCount>().value(); ++i)
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

    pipe->request(ICS::DisableChartRecord);
    while (pipe->is_requested(ICS::DisableChartRecord) && !stop_requested_)
        continue;

    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::CHART_RECORD);
}

} // namespace holovibes::worker
