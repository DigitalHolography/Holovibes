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

#include <iomanip>

#include <QProgressBar>
#include <QObject>
#include <QThread>

#include "thread_csv_record.hh"
#include "concurrent_deque.hh"
#include "holovibes.hh"
#include "info_manager.hh"

namespace holovibes
{
	namespace gui
	{
		ThreadCSVRecord::ThreadCSVRecord(Holovibes& holo,
			Deque& deque,
			const std::string path,
			const unsigned int nb_frames,
			QObject* parent)
			: QThread(parent)
			, holo_(holo)
			, deque_(deque)
			, path_(path)
			, nb_frames_(nb_frames)
			, record_(true)
			, old_progress_bar_max_(InfoManager::get_manager()->get_progress_bar()->maximum())
		{
		}

		ThreadCSVRecord::~ThreadCSVRecord()
		{
			this->stop();

			QProgressBar* progress_bar = InfoManager::get_manager()->get_progress_bar();
			progress_bar->setMaximum(old_progress_bar_max_);
			progress_bar->setValue(old_progress_bar_max_);
		}

		void ThreadCSVRecord::stop()
		{
			record_ = false;

			wait();
		}

		void ThreadCSVRecord::run()
		{
			QProgressBar* progress_bar = InfoManager::get_manager()->get_progress_bar();
			connect(this, SIGNAL(value_change(int)), progress_bar, SLOT(setValue(int)));
			progress_bar->setMaximum(nb_frames_);

			size_t new_elts = 0;
			do
			{
				emit value_change(static_cast<int>(new_elts));
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			} while ((new_elts = deque_.size()) < nb_frames_ && record_);

			std::ofstream of(path_);

			// Header displaying
			of << "[#img : " << holo_.get_cd().time_filter_size
				<< ", p : " << holo_.get_cd().pindex
				<< ", lambda : " << holo_.get_cd().lambda
				<< ", z : " << holo_.get_cd().zdistance
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

			for (uint i = 0; i < new_elts; ++i)
			{
				ChartPoint& point = deque_[i];
				of << std::fixed << std::setw(11) << std::setprecision(10) << std::setfill('0')
					<< point.avg_signal << ","
					<< point.avg_noise << ","
					<< point.avg_signal_div_avg_noise << ","
					<< point.log_avg_signal_div_avg_noise << ","
					<< point.std_signal << ","
					<< point.std_signal_div_avg_noise << ","
					<< point.std_signal_div_avg_signal << std::endl;
			}

			// It's the function display_info -> refactor it
			InfoManager::get_manager()->insert_info(InfoManager::InfoType::INFO, "Info", "Chart record done");
			InfoManager::get_manager()->startDelError("Info");
		}
	}
}