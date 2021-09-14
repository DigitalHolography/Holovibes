#include "batch_gpib_worker.hh"

#include <chrono>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>

#include "chart_record_worker.hh"
#include "logger.hh"
#include "gpib_exceptions.hh"

namespace holovibes::worker
{
BatchGPIBWorker::BatchGPIBWorker(const std::string& batch_input_path,
                                 const std::string& output_path,
                                 unsigned int nb_frames_to_record,
                                 RecordMode record_mode,
                                 bool square_output)
    : Worker()
    , output_path_(output_path)
    , nb_frames_to_record_(nb_frames_to_record)
    , record_mode_(record_mode)
    , square_output_(square_output)
    , frame_record_worker_(nullptr)
    , chart_record_worker_(nullptr)
{
    try
    {
        parse_file(batch_input_path);
    }
    catch (const std::exception& exception)
    {
        LOG_ERROR << exception.what();
        batch_cmds_.clear();
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

    try
    {
        while (true)
        {
            if (batch_cmds_.empty() || stop_requested_)
                return;

            auto cmd = batch_cmds_.back();

            if (cmd.type == gpib::BatchCommand::INSTRUMENT_COMMAND)
                execute_instrument_command(cmd);
            else if (cmd.type == gpib::BatchCommand::CAPTURE)
            {
                std::string formatted_path = format_batch_output(file_index);

                if (record_mode_ == RecordMode::CHART)
                {
                    chart_record_worker_ = std::make_unique<ChartRecordWorker>(
                        formatted_path,
                        nb_frames_to_record_);
                    chart_record_worker_->run();
                }
                else // Frame Record
                {
                    bool raw_record = record_mode_ == RecordMode::RAW;

                    frame_record_worker_ = std::make_unique<FrameRecordWorker>(
                        formatted_path,
                        nb_frames_to_record_,
                        raw_record,
                        square_output_,
                        0);
                    frame_record_worker_->run();
                }

                ++file_index;
            }
            else if (cmd.type == gpib::BatchCommand::WAIT)
            {
                auto waiting_time = cmd.wait;

                auto starting_time = std::chrono::high_resolution_clock::now();
                while (!stop_requested_)
                {
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() -
                            starting_time)
                            .count() >= waiting_time)
                        break;
                }
            }

            batch_cmds_.pop_back();
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        return;
    }
}

void BatchGPIBWorker::parse_file(const std::string& batch_input_path)
{
    std::ifstream in(batch_input_path);
    if (!in.is_open())
        throw std::exception("GPIB : Invalid Filepath");

    std::string line;
    unsigned int line_num = 0;
    unsigned int cur_address = 0;

    while (in >> line)
    {
        batch_cmds_.push_front(gpib::BatchCommand());
        gpib::BatchCommand& cmd = batch_cmds_.front();

        if (line.compare("#Block") == 0)
        {
            // Just a #Block : no special meaning.
            batch_cmds_.pop_front();
        }
        else if (line.compare("#InstrumentAddress") == 0)
        {
            // We change the address currently used for commands.
            try
            {
                in >> line;
                unsigned int address = boost::lexical_cast<unsigned int>(line);
                cur_address = address;
                batch_cmds_.pop_front();
            }
            catch (const boost::bad_lexical_cast& /*e*/)
            {
                throw gpib::GpibParseError(
                    boost::lexical_cast<std::string>(line_num),
                    gpib::GpibParseError::NoAddress);
            }
        }
        else if (line.compare("#WAIT") == 0)
        {
            // We insert a waiting action in the block.
            try
            {
                in >> line;
                unsigned int wait = boost::lexical_cast<unsigned int>(line);

                cmd.type = gpib::BatchCommand::WAIT;
                cmd.address = 0;
                cmd.command = "";
                cmd.wait = wait;
            }
            catch (const boost::bad_lexical_cast& /*e*/)
            {
                throw gpib::GpibParseError(
                    boost::lexical_cast<std::string>(line_num),
                    gpib::GpibParseError::NoWait);
            }
        }
        else if (line.compare("#Capture") == 0)
        {
            cmd.type = gpib::BatchCommand::CAPTURE;
            cmd.address = 0;
            cmd.command = "";
            cmd.wait = 0;
        }
        else
        {
            /* A command string, the validity of which can not be tested because
             * of the multiple interfaces to various existing instruments. */
            for (unsigned i = 0; i < line.size(); ++i)
                in.unget();
            std::getline(in, line, '\n');
            line.append(
                "\n"); // Don't forget the end-of-command character for VISA.

            cmd.type = gpib::BatchCommand::INSTRUMENT_COMMAND;
            cmd.address = cur_address;
            cmd.command = line;
            cmd.wait = 0;
        }
        ++line_num;
    }

    if (line_num == 0)
    {
        // We just read a blank file...
        throw gpib::GpibBlankFileError();
    }
}

void BatchGPIBWorker::execute_instrument_command(
    gpib::BatchCommand instrument_command)
{
    if (!gpib_interface_)
        gpib_interface_ = gpib::GpibDLL::load_gpib("gpib.dll");

    gpib_interface_->execute_instrument_command(instrument_command);
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
