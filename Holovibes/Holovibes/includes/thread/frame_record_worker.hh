#pragma once

#include "worker.hh"

namespace holovibes
{
// Fast forward declarations
class Queue;
class ICompute;

namespace worker
{
/*!
 *  \brief    Class used to record frames
 */
class FrameRecordWorker : public Worker
{
  public:
    /*!
     *  \brief    Constructor
     *
     *  \param    file_path             The path of the file to record
     *  \param    nb_frames_to_record   The number of frames to record
     *  \param    raw_record            Whether raw images are recorded
     *  \param    square_output         Whether the output should be a square
     *  \param    nb_frames_skip        Number of frames to skip before starting
     */
    FrameRecordWorker(const std::string& file_path,
                      std::optional<unsigned int> nb_frames_to_record,
                      bool raw_record,
                      bool square_output,
                      unsigned int nb_frames_skip);

    void run() override;

  private:
    /*!
     *  \brief    Init the record queue
     *
     *  \return   The record queue
     */
    Queue& init_gpu_record_queue(std::shared_ptr<ICompute> pipe);

    /*!
     *  \brief    Wait for frames to be present in the record queue
     *
     *  \param    record_queue   The record queue
     *  \param    pipe           The compute pipe used to perform the operations
     */
    void wait_for_frames(Queue& record_queue, std::shared_ptr<ICompute> pipe);

    /*!
     *  \brief    Reset the record queue to free memory
     *
     *  \param    pipe           The compute pipe used to perform the operations
     */
    void reset_gpu_record_queue(std::shared_ptr<ICompute> pipe);

  private:
    //! The path of the file to record
    const std::string file_path_;
    //! The number of frames to record
    std::optional<unsigned int> nb_frames_to_record_;
    //! The number of frames to skip before starting the recording
    unsigned int nb_frames_skip_;
    //! The current fps
    std::atomic<unsigned int> processed_fps_;
    //! Whether the raw images are recorded
    bool raw_record_;
    //! Whether the output should be a square
    bool square_output_;

    const cudaStream_t stream_;
};
} // namespace worker
} // namespace holovibes
