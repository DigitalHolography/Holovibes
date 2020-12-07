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

#pragma once

#include "file_frame_read_worker.hh"

namespace holovibes::worker
{
	inline FileFrameReadWorker::FpsHandler::FpsHandler(unsigned int fps) :
		enqueue_interval_((1 / static_cast<double>(fps)))
	{}

	inline void FileFrameReadWorker::FpsHandler::begin()
	{
		begin_time_ = std::chrono::high_resolution_clock::now();
	}

	inline void FileFrameReadWorker::FpsHandler::wait()
	{
		/* end_time should only be being_time + enqueue_interval_ aka the time point for the next enqueue
		** However the wasted_time is substracted to get the correct next enqueue time point
		*/
		auto end_time = (begin_time_ + enqueue_interval_) - wasted_time_;

		// Wait until the next enqueue time point is reached
		while (std::chrono::high_resolution_clock::now() < end_time)
			continue;

		/* Wait is done, it might have been too long (descheduling...)
		**
		** Set the begin_time (now) for the next enqueue
		** And compute the wasted time (real time point - theoretical time point)
		*/
		auto now =  std::chrono::high_resolution_clock::now();
		wasted_time_ = now - end_time;
		begin_time_ = now;
	}
} // namespace holovibes
