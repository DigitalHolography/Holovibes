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
    class InformationContainer;

    namespace worker
    {
        class InformationWorker : public Worker
        {
        public:
            InformationWorker(bool is_cli, InformationContainer& info);

            void run() override;

        private:
            void compute_fps(long long waited_time);

            void compute_throughput(ComputeDescriptor& cd, unsigned int output_frame_res,
                                    unsigned int input_frame_size,
                                    unsigned int record_frame_size);

            void display_gui_information();

            bool is_cli_;

            InformationContainer& info_;

            unsigned int input_fps_ = 0;

            unsigned int output_fps_ = 0;

            unsigned int saving_fps_ = 0;

            unsigned int input_throughput_ = 0;

            unsigned int output_throughput_ = 0;

            unsigned int saving_throughput_ = 0;
        };
    }
} // namespace holovibes::worker