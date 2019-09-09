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

#include "thread_recorder.hh"
#include "recorder.hh"
#include "queue.hh"

#include "info_manager.hh"

namespace holovibes
{
	namespace gui
	{
		ThreadRecorder::ThreadRecorder(
			Queue& queue,
			const std::string& filepath,
			const unsigned int n_images,
			QObject* parent)
			: QThread(parent)
			, queue_(queue)
			, recorder_(queue, filepath)
			, n_images_(n_images)
		{
		}

		ThreadRecorder::~ThreadRecorder()
		{
		}

		void ThreadRecorder::stop()
		{
			recorder_.stop();
		}

		void ThreadRecorder::run()
		{
			QProgressBar*   progress_bar = InfoManager::get_manager()->get_progress_bar();

			queue_.flush();
			progress_bar->setMaximum(n_images_);
			connect(&recorder_, SIGNAL(value_change(int)), progress_bar, SLOT(setValue(int)));
			recorder_.record(n_images_);
		}
	}
}