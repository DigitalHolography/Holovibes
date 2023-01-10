/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "frame_read_worker.hh"
#include "input_frame_file.hh"
#include "env_structs.hh"
#include "export_cache.hh"
#include "import_cache.hh"

// Fast forward declarations
namespace holovibes
{
class Queue;
} // namespace holovibes

namespace holovibes::io_files
{
class InputFrameFile;
} // namespace holovibes::io_files

namespace holovibes::worker
{
/*! \class FileFrameReadWorker
 *
 * \brief Class used to read frames from a file
 */
class FileFrameReadWorker final : public FrameReadWorker
{
  public:
    FileFrameReadWorker();
    ~FileFrameReadWorker();

    void run() override;

  private:
    class FpsHandler
    {
      public:
        FpsHandler(unsigned int fps)
            : enqueue_interval_((1 / static_cast<double>(fps)))
        {
        }

        /*! \brief Begin the process of fps handling */
        void begin() { begin_time_ = std::chrono::high_resolution_clock::now(); }

        /*! \brief Wait the correct time to simulate fps
         *
         * Between each frame enqueue, the waiting duration should be enqueue_interval_.
         * However the real waiting duration might be longer than the theoretical one (due to descheduling).
         * To cope with this issue, we compute the wasted time in order to take it into account for the next enqueue.
         * By doing so, the correct enqueuing time is computed, not doing so would create a lag.
         */
        void wait()
        {
            /* end_time should only be being_time + enqueue_interval_ aka the time point
             * for the next enqueue
             * However the wasted_time is substracted to get the correct next enqueue
             * time point
             */
            auto end_time = (begin_time_ + enqueue_interval_) - wasted_time_;

            // Wait until the next enqueue time point is reached
            while (std::chrono::high_resolution_clock::now() < end_time)
            {
            }

            /* Wait is done, it might have been too long (descheduling...)
             *
             * Set the begin_time (now) for the next enqueue
             * And compute the wasted time (real time point - theoretical time point)
             */
            auto now = std::chrono::high_resolution_clock::now();
            wasted_time_ = now - end_time;
            begin_time_ = now;
        }

      private:
        /*! \brief Theoretical time between 2 enqueues/waits */
        std::chrono::duration<double> enqueue_interval_;

        /*! \brief Begin time point of the wait */
        std::chrono::steady_clock::time_point begin_time_;

        /*! \brief Time wasted in last wait (if waiting was too long) */
        std::chrono::duration<double> wasted_time_{0};
    };

    /*! \brief Init the cpu_buffer and gpu_buffer */
    bool init_frame_buffers();

    /*! \brief Load all the frames of the file in the gpu
     *
     * Read all the frames in cpu and copy them in gpu.
     * Then enqueue the frames one by one in the gpu_input_queue
     */
    void read_file_in_gpu();

    /*! \brief Load the frames of the file by batch into the gpu
     *
     * Read batch in cpu and copy it in gpu.
     * Then enqueue the frames one by one in the gpu_input_queue
     */
    void read_file_batch();

    /*! \brief Read frames in cpu and copy in gpu
     *
     * \param frames_to_read The number of frames to read
     * \return The number of frames read
     */
    size_t read_copy_file(size_t frames_to_read);

    /*! \brief Enqueue frames_read in the gpu_input_queue with a speed related to the given fps
     *
     * \param nb_frames_to_enqueue The number of frames to enqueue from gpu_buffer to gpu_input_queue
     */
    void enqueue_loop(size_t nb_frames_to_enqueue);

  private:
    uint current_nb_frames_read_;

    FpsHandler fps_handler_;

    /*! \brief CPU buffer in which the frames are temporarly stored */
    char* cpu_frame_buffer_;
    /*! \brief GPU buffer in which the frames are temporarly stored */
    char* gpu_frame_buffer_;
    /*! \brief Tmp GPU buffer in which the frames are temporarly stored to convert data from packed bits to 16bit */
    char* gpu_packed_buffer_;

    std::unique_ptr<io_files::InputFrameFile> input_file_;
};
} // namespace holovibes::worker
