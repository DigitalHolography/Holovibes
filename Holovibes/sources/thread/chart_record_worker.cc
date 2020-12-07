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
    {}

    void ChartRecordWorker::run()
    {
		auto& cd = Holovibes::instance().get_cd();

		std::ofstream of(path_);

		// Header displaying
		of << "[#img : " << cd.time_transformation_size
			<< ", p : " << cd.pindex
			<< ", lambda : " << cd.lambda
			<< ", z : " << cd.zdistance
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

		auto pipe = Holovibes::instance().get_compute_pipe();
        pipe->request_record_chart(nb_frames_to_record_);
		while (pipe->get_chart_record_requested() != std::nullopt && !stop_requested_);

        auto& chart_queue = *pipe->get_chart_record_queue();

		std::atomic<size_t> i = 0;
		Holovibes::instance().get_info_container().add_progress_index(InformationContainer::ProgressType::CHART_RECORD,
			i, nb_frames_to_record_);

		for (; i < nb_frames_to_record_; ++i)
		{
			while (chart_queue.size() <= i && !stop_requested_);
			if (stop_requested_)
				break;

			ChartPoint& point = chart_queue[i];
			of << std::fixed << std::setw(11) << std::setprecision(10) << std::setfill('0')
				<< point.avg_signal << ","
				<< point.avg_noise << ","
				<< point.avg_signal_div_avg_noise << ","
				<< point.log_avg_signal_div_avg_noise << ","
				<< point.std_signal << ","
				<< point.std_signal_div_avg_noise << ","
				<< point.std_signal_div_avg_signal << std::endl;
		}

        pipe->request_disable_record_chart();
		while (pipe->get_disable_chart_record_requested() && !stop_requested_);

		Holovibes::instance().get_info_container().remove_progress_index(InformationContainer::ProgressType::CHART_RECORD);
    }

} // namespace holovibes::worker