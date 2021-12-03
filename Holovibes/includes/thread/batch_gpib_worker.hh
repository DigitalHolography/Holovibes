/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <deque>

#include "gpib_dll.hh"
#include "IVisaInterface.hh"
#include "gpib_controller.hh"
#include "gpib_exceptions.hh"

#include "enum_record_mode.hh"

#include "frame_record_worker.hh"
#include "chart_record_worker.hh"

namespace holovibes::worker
{
/*! \class BatchGPIBWorker
 *
 * \brief Class used for batch functionality (chart or frame record multiple times)
 */
class BatchGPIBWorker final : public Worker
{
  public:
    /*!
     * \param batch_input_path Batch input path that details commands to process
     * \param output_path Output record path
     * \param nb_frames_to_record Number of frames to record for a signle record command
     * \param record_mode The record mode used for all the batch (RAW, HOLOGRAM, CHART)
     * \param Option passed to the frame record: resize saved images (linear interpolation)
     */
    BatchGPIBWorker(const std::string& batch_input_path,
                    const std::string& output_path,
                    unsigned int nb_frames_to_record,
                    RecordMode record_mode,
                    const unsigned int output_buffer_size);

    void stop() override;

    void run() override;

  private:
    /*! \brief Parse the different commands in the file to put them in batch_cmds_
     *
     * \param batch_input_path File that contains all batch commands
     */
    void parse_file(const std::string& batch_input_path);

    /*! \brief Execute a single BatchCommand through the gpib_interface_
     *
     * \param instrument_command BatchCommand to execute
     */
    void execute_instrument_command(gpib::BatchCommand instrument_command);

    /*!
     * \brief Generate a new batch record output relative to the number of record that has already occured
     *
     * \param index Number of record that has already occured
     * \return The new generated batch record output
     */
    std::string format_batch_output(const unsigned int index);

  private:
    /*! \brief Output record path */
    const std::string output_path_;

    /*! \brief Number of frames to record for a signle record command */
    const unsigned int nb_frames_to_record_;

    /*! \brief The record mode used for all the batch (RAW, HOLOGRAM, CHART) */
    const RecordMode record_mode_;

    /*! \brief The max output buffer size */
    const unsigned int output_buffer_size_;

    /*! \brief Instance of the frame record worker used if RecordMode is RAW or HOLOGRAM */
    std::unique_ptr<FrameRecordWorker> frame_record_worker_;

    /*! \brief Instance of the chart record worker used if RecordMode is CHART */
    std::unique_ptr<ChartRecordWorker> chart_record_worker_;

    /*! \brief Batch commands that are parsed from the batch input file */
    std::deque<gpib::BatchCommand> batch_cmds_;

    /*! \brief Instance of the gpib interface used to communicate with external instruments with GPIB protocol */
    std::shared_ptr<gpib::IVisaInterface> gpib_interface_;
};
} // namespace holovibes::worker
