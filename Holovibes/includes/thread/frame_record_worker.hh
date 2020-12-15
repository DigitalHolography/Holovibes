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

#include "worker.hh"

namespace holovibes
{
    class Queue;
    class ICompute;

    namespace worker
    {
        class FrameRecordWorker : public Worker
        {
        public:
            FrameRecordWorker(const std::string &file_path,
                            std::optional<unsigned int> nb_frames_to_record,
                            bool raw_record,
                            bool square_output);

            void run() override;

        private:
            Queue& init_gpu_record_queue(std::shared_ptr<ICompute> pipe);

            void wait_for_frames(Queue& record_queue, std::shared_ptr<ICompute> pipe);

            void reset_gpu_record_queue(std::shared_ptr<ICompute> pipe);

            const std::string file_path_;

            std::optional<unsigned int> nb_frames_to_record_;

            std::atomic<unsigned int> processed_fps_;

            bool raw_record_;

            bool square_output_;
        };
    } // namespace worker
} // namespace holovibes
