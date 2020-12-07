#include "batch_gpib_worker.hh"

#include <chrono>

#include "chart_record_worker.hh"
#include "logger.hh"

namespace holovibes::worker
{
    BatchGPIBWorker::BatchGPIBWorker(const std::string& batch_input_path,
                        const std::string& output_path,
                        const unsigned int nb_frames_to_record,
                        const bool chart_record,
                        const bool raw_record_enabled)
        : Worker()
        , batch_input_path_(batch_input_path)
        , output_path_(output_path)
        , nb_frames_to_record_(nb_frames_to_record)
        , chart_record_(chart_record)
        , raw_record_enabled_(raw_record_enabled)
        , frame_record_worker_(nullptr)
        , chart_record_worker_(nullptr)
    {
        try
        {
            gpib_interface_ = gpib::GpibDLL::load_gpib("gpib.dll", batch_input_path_);
        }
        catch (const std::exception& e)
        {
            gpib_interface_ = nullptr;
            LOG_ERROR(e.what());
        }
    }

    void BatchGPIBWorker::stop()
    {
        Worker::stop();

        if (frame_record_worker_)
            frame_record_worker_->stop();

        if (chart_record_worker_)
            chart_record_worker_->stop();
    }

    void BatchGPIBWorker::run()
    {
        unsigned int file_index = 1;

        if (gpib_interface_ == nullptr)
            return;

        try
        {
            while (true)
            {
                auto cmd = gpib_interface_->get_next_command();

                if (!cmd.has_value() || stop_requested_)
                    return;

                if (cmd->type == gpib::Command::INSTRUMENT_COMMAND)
                    gpib_interface_->execute_instrument_command(*cmd);
                else if (cmd->type == gpib::Command::CAPTURE)
                {
                    std::string formatted_path = format_batch_output(file_index);

                    if (chart_record_)
                    {
                        chart_record_worker_ = std::make_unique<ChartRecordWorker>(formatted_path,
                            nb_frames_to_record_);
                        chart_record_worker_->run();
                    }
                    else // Frame Record
                    {
                        frame_record_worker_ = std::make_unique<FrameRecordWorker>(formatted_path,
                            nb_frames_to_record_, raw_record_enabled_);
                        frame_record_worker_->run();
                    }

                    ++file_index;
                }
                else if (cmd->type == gpib::Command::WAIT)
                {
                    auto waiting_time = cmd->wait;

                    auto starting_time = std::chrono::high_resolution_clock::now();
                    while (!stop_requested_)
                    {
                        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - starting_time).count() >= waiting_time)
                            break;
                    }
                }

                gpib_interface_->pop_next_command();
            }
        }
        catch (const std::exception& e)
        {
            LOG_ERROR(e.what());
            return;
        }
    }

    std::string BatchGPIBWorker::format_batch_output(const unsigned int index)
    {
        std::string file_index;
        std::ostringstream convert;
        convert << std::setw(6) << std::setfill('0') << index;
        file_index = convert.str();

        std::vector<std::string> path_tokens;
        boost::split(path_tokens, output_path_, boost::is_any_of("."));

        std::string res = path_tokens[0] + "_" + file_index;
        if (path_tokens.size() > 1)
            res += "." + path_tokens[1];

        return res;
    }
} // namespace holovibes::worker