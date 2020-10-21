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
 * Thread used to count the number of frames per second (fps)
*/

#pragma once

#include <thread>
#include <atomic>

namespace holovibes
{
	class ThreadTimer : public std::thread
    {
		public:
			/*! \brief ThreadRecorder constructor
			**
			** \param nb_frame_one_second The number of frames passed each second
			*/
			ThreadTimer(std::atomic<uint>& nb_frame_one_second);

            /*! \brief Join before destruction */
            virtual ~ThreadTimer();

            /*! \brief Stop the thread */
            void stop() { stop_ = true; }

		private:
			/*! \brief Method that will be run by the thread
			**
			** Thread function used by to count 1 second
            ** The calling thread increment the nb_frame_one_second every time one frame is copied
            ** This function prints on the IU the number of frames copied after 1 second
            **
            ** This thread exist to avoid interuptions in the calling thread doing work
            */
			void run();

		private:
			/*! \brief Variable shared by calling thread and this one.
			** Current number of frames within a second
			*/
            std::atomic<uint>& nb_frame_one_second_;

			/*! \brief Flag to stop the thread. */
            std::atomic<bool> stop_;
    };
}
