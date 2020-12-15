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

#include <deque>

#include "gpib_dll.hh"
#include "IVisaInterface.hh"
#include "gpib_controller.hh"
#include "gpib_exceptions.hh"

#include "frame_record_worker.hh"
#include "chart_record_worker.hh"

namespace holovibes::worker
{
    class BatchGPIBWorker : public Worker
    {
    public:
        BatchGPIBWorker(const std::string& batch_input_path,
                        const std::string& output_path,
                        unsigned int nb_frames_to_record,
                        bool chart_record,
                        bool raw_record_enabled,
                        bool square_output);

        void stop() override;

        void run() override;

    private:
        void parse_file(const std::string& batch_input_path);

        void execute_instrument_command(gpib::BatchCommand instrument_command);

        std::string format_batch_output(const unsigned int index);

    private:
        const std::string output_path_;

        const unsigned int nb_frames_to_record_;

        const bool chart_record_;

        const bool raw_record_;

        const bool square_output_;

        std::unique_ptr<FrameRecordWorker> frame_record_worker_;

        std::unique_ptr<ChartRecordWorker> chart_record_worker_;

        std::deque<gpib::BatchCommand> batch_cmds_;

        std::shared_ptr<gpib::IVisaInterface> gpib_interface_;
    };
} // namespace holovibes::worker