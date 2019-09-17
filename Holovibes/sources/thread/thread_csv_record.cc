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
		{}

		ThreadCSVRecord::~ThreadCSVRecord()
		{
			this->stop();
		}

		void ThreadCSVRecord::stop()
		{
			record_ = false;
		}

		void ThreadCSVRecord::run()
		{
			deque_.clear();
			QProgressBar*   progress_bar = InfoManager::get_manager()->get_progress_bar();
			connect(this, SIGNAL(value_change(int)), progress_bar, SLOT(setValue(int)));
			progress_bar->setMaximum(nb_frames_);
			holo_.get_pipe()->request_average_record(&deque_, nb_frames_);

			while (deque_.size() < nb_frames_ && record_)
			{
				if (deque_.size() <= nb_frames_)
					emit value_change(static_cast<int>(deque_.size()));
				continue;
			}
			emit value_change(nb_frames_);
			std::cout << path_ << std::endl;
			std::ofstream of(path_);

			// Header displaying
			of << "[Phase number : " << holo_.get_compute_desc().nSize
				<< ", p : " << holo_.get_compute_desc().pindex
				<< ", lambda : " << holo_.get_compute_desc().lambda
				<< ", z : " << holo_.get_compute_desc().zdistance
				<< "]" << std::endl;

			of << "[Column 1 : signal, Column 2 : noise, Column 3 : 10 * log10 (signal / noise)]" << std::endl;

			const unsigned int deque_size = static_cast<unsigned int>(deque_.size());
			unsigned int i = 0;
			while (i < deque_size && record_)
			{
				Tuple4f& tuple = deque_[i];
				of << std::fixed << std::setw(11) << std::setprecision(10) << std::setfill('0')
					<< std::get<0>(tuple) << ","
					<< std::get<1>(tuple) << ","
					<< std::get<2>(tuple) << std::endl;
				++i;
			}

			holo_.get_pipe()->request_average_stop();
		}
	}
}