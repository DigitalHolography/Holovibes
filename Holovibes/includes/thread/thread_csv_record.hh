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
 * Thread class used to record CSV files of ROI/chart computations. */
#pragma once

#include <string>

#include <QObject>
#include <QThread>

#include "pipe.hh"

namespace holovibes
{
	// Forward declarations.
	class Holovibes;
	template <class T> class ConcurrentDeque;

	namespace gui
	{
		/*! \brief Thread class used to record CSV files of ROI/chart computations.
		**
		** It inherits QThread because it is the GUI that needs to launch the record and it has
		** to know when it is finished (signals/slots system).
		*/
		class ThreadCSVRecord : public QThread
		{
			Q_OBJECT

				typedef ConcurrentDeque<ChartPoint> Deque;

		signals:

			void value_change(int value);

		public:
			/*! \brief ThreadCSVRecord constructor
			**
			** \param pipe pipe of the program, see holovibes::Holovibes::get_pipe()
			** \param deque concurrent Deque containing the chart values to record
			** \param path string containing output path of record
			** \param nb_frames number of frames i-e number of values to record
			** \param parent Qt parent (default is null)
			*/
			ThreadCSVRecord(Holovibes& holo,
				Deque& deque,
				const std::string path,
				const unsigned int nb_frames,
				QObject* parent = nullptr);

			~ThreadCSVRecord();

			public slots:
			/*! Stops the record by setting a flag */
			void stop();

		private:
			/*! \brief Overrided QThread run method, recording method
			**
			** Ensure to flush the Deque before using it in order to record the frames
			** from the moment the user started the record and not before.
			*/
			void run() override;

		private:
			/*! Reference to the core class of the program. */
			Holovibes& holo_;
			/*! Deque storing recorded data. */
			Deque& deque_;
			/*! Output record path */
			std::string path_;
			/*! Number of frames i-e number of values to record */
			unsigned int nb_frames_;
			/*! Flag used to stop recording */
			bool record_;
		};
	}
}
