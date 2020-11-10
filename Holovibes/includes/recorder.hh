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

/*! \file
 *
 * Store frames of a given queue in a file. */
#pragma once

#include <fstream>
#include <string>

#include <QObject>
#include <QThread>

#include "json.hh"
using json = ::nlohmann::json;

 /* Forward declaration. */
namespace holovibes
{
	class Queue;
	class ComputeDescriptor;
}

namespace holovibes
{
	/*! \brief Store frames of given queue in file
	 *
	 * Image are stored in raw format at the given file path.
	 * Recorder is thread safe and you can stop record anytime.
	 *
	 * Usage:
	 * * Create Recorder
	 * * Use record to record n_images any times you need it
	 * * delete Recorder
	 */
	class Recorder : public QObject
	{
		Q_OBJECT

			signals :
		/*! \brief Inform that one frame has been recorded
		**
		** value is number of frames recorded*/
		void value_change(int value);

	public:
		/*! \brief Open the given filepath.
		 *
		 * \param queue The source queue.
		 * \param filepath The absolute path to the destination file.
		 * \param cd The compute descriptor (hold the recording description)
		 */
		Recorder(
			Queue& queue,
			const std::string& filepath,
			ComputeDescriptor& cd);

		~Recorder();

		/*! \brief record n_images to file
		 *
		 * Recorder is thread safe and you can stop this function anytime
		 * by using the "stop" button.
		 */
		void record();

		/*! \brief Stop current record */
		void stop();

	private:
		std::string execDir;
		Queue& queue_;
		bool stop_requested_;
		std::string output_path_;

		ComputeDescriptor& cd_;
	};
}
