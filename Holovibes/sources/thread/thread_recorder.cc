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
			ComputeDescriptor& cd,
			QObject* parent)
			: QThread(parent)
			, recorder_(queue, filepath, cd)
		{
			if (!gui::InfoManager::is_cli())
			{
				QProgressBar*   progress_bar = InfoManager::get_manager()->get_progress_bar();

				progress_bar->setMaximum(cd.nb_frames_record);
				connect(&recorder_, SIGNAL(value_change(int)), progress_bar, SLOT(setValue(int)));
			}
		}

		ThreadRecorder::~ThreadRecorder()
		{}

		void ThreadRecorder::stop()
		{
			recorder_.stop();
		}

		void ThreadRecorder::run()
		{
			recorder_.record();
		}
	}
}