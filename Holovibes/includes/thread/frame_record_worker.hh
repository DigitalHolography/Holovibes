/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"

namespace holovibes
{
// Fast forward declarations
class Queue;
class ICompute;

namespace worker
{
/*! \class FrameRecordWorker
 *
 * \brief Class used to record frames
 */
class FrameRecordWorker : public Worker
{
  public:
    /*! \brief Constructor
     *
     * \param file_path The path of the file to record
     * \param nb_frames_to_record The number of frames to record
     * \param raw_record Whether raw images are recorded
     * \param nb_frames_skip Number of frames to skip before starting
     */
    FrameRecordWorker(const std::string& file_path,
                      std::optional<unsigned int> nb_frames_to_record,
                      bool raw_record,
                      unsigned int nb_frames_skip,
                      unsigned int output_buffer_size);

    void run() override;

  private:
    /*! \brief Init the record queue
     *
     * \return The record queue
     */
    Queue& init_gpu_record_queue(std::shared_ptr<ICompute> pipe);

    /*! \brief Wait for frames to be present in the record queue
     *
     * \param record_queue The record queue
     * \param pipe The compute pipe used to perform the operations
     */
    void wait_for_frames(Queue& record_queue, std::shared_ptr<ICompute> pipe);

    /*! \brief Reset the record queue to free memory
     *
     * \param pipe The compute pipe used to perform the operations
     */
    void reset_gpu_record_queue(std::shared_ptr<ICompute> pipe);

  private:
    /*! \brief The path of the file to record */
    const std::string file_path_;
    /*! \brief The number of frames to record */
    std::optional<unsigned int> nb_frames_to_record_;
    /*! \brief The number of frames to skip before starting the recording */
    unsigned int nb_frames_skip_;
    /*! \brief The current fps */
    std::atomic<unsigned int> processed_fps_;
    /*! \brief Whether the raw images are recorded */
    bool raw_record_;
    /*! \brief Output buffer size */
    unsigned int output_buffer_size_;

    const cudaStream_t stream_;
};
} // namespace worker
} // namespace holovibes
