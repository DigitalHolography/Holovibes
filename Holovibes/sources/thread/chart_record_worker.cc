#include "chart_record_worker.hh"
#include "chart_point.hh"

#include "holovibes.hh"
#include "API.hh"
#include "icompute.hh"
#include "tools.hh"

namespace holovibes::worker
{
ChartRecordWorker::ChartRecordWorker()
    : Worker()
    , env_(api::get_compute_pipe().get_chart_env())
    , export_cache_()
{
}

void ChartRecordWorker::run()
{
    std::ofstream of(export_cache_.get_value<Record>().file_path);

    // Header displaying
    of << "[#img : " << api::detail::get_value<TimeTransformationSize>()
       << ", p : " << api::detail::get_value<ViewAccuP>().start << ", lambda : " << api::detail::get_value<Lambda>()
       << ", z : " << api::detail::get_value<ZDistance>() << "]" << std::endl;

    of << "["
       << "Column 1 : avg(signal), "
       << "Column 2 : avg(noise), "
       << "Column 3 : avg(signal) / avg(noise), "
       << "Column 4 : 10 * log10 (avg(signal) / avg(noise)), "
       << "Column 5 : std(signal), "
       << "Column 6 : std(signal) / avg(noise), "
       << "Column 7 : std(signal) / avg(signal)"
       << "]" << std::endl;

    auto& nb_to_record = export_cache_.get_value<Record>().nb_to_record;
    env_.current_nb_points_recorded = 0;

    auto& entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::RECORD);
    entry.recorded = &env_.current_nb_points_recorded;
    entry.to_record = &nb_to_record;

    while (env_.current_nb_points_recorded < nb_to_record)
    {
        // FIXME : This should be a trigger
        while (env_.chart_record_queue_->size() <= env_.current_nb_points_recorded && !stop_requested_)
            continue;
        if (stop_requested_)
            break;

        ChartPoint& point = env_.chart_record_queue_->operator[](env_.current_nb_points_recorded);
        of << std::fixed << std::setw(11) << std::setprecision(10) << std::setfill('0') << point.avg_signal << ","
           << point.avg_noise << "," << point.avg_signal_div_avg_noise << "," << point.log_avg_signal_div_avg_noise
           << "," << point.std_signal << "," << point.std_signal_div_avg_noise << "," << point.std_signal_div_avg_signal
           << std::endl;

        // Maybe ?
        env_.current_nb_points_recorded++;
    }

    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::RECORD);
}

} // namespace holovibes::worker
