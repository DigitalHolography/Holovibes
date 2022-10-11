/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "frame_read_worker.hh"
#include "input_frame_file.hh"

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
 * \brief    Class used to read frames from a file
 */
class FileFrameReadWorker final : public FrameReadWorker
{
  public:
    /*! \brief Constructor
     *
     * \param file_path  The file path
     * \param loop Whether the reading should loop
     * \param first_frame_id Id of the first frame to read
     * \param total_nb_frames_to_read Total number of frames to read
     * \param load_file_in_gpu Whether the file should be load in gpu
     * \param gpu_input_queue The input queue
     */
    FileFrameReadWorker(const std::string& file_path,
                        bool loop,
                        unsigned int fps,
                        unsigned int first_frame_id,
                        unsigned int total_nb_frames_to_read,
                        bool load_file_in_gpu,
                        std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue);

    void run() override;

  private:
    class FpsHandler
    {
      public:
        FpsHandler(unsigned int fps);

        /*! \brief Begin the process of fps handling */
        void begin();

        /*! \brief Wait the correct time to simulate fps
         *
         * Between each frame enqueue, the waiting duration should be enqueue_interval_.
         * However the real waiting duration might be longer than the theoretical one (due to descheduling).
         * To cope with this issue, we compute the wasted time in order to take it into account for the next enqueue.
         * By doing so, the correct enqueuing time is computed, not doing so would create a lag.
         */
        void wait();

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
    FastUpdatesHolder<ProgressType>::Value fast_updates_entry_;

    /*! \brief Current number of frames read */
    std::atomic<unsigned int>& current_nb_frames_read_;
    /*! \brief Total number of frames to read at the beginning of the process */
    std::atomic<unsigned int>& total_nb_frames_to_read_;

    /*! \brief The file path */
    const std::string file_path_;
    /*! \brief Whether the reading should start over when meeting the end of the file */
    bool loop_;
    /*! \brief Object used to handle the fps */
    FpsHandler fps_handler_;
    /*! \brief Id of the first frame to read */
    unsigned int first_frame_id_;
    /*! \brief Whether the entire file should be loaded in the gpu */
    bool load_file_in_gpu_;
    /*! \brief The input file in which the frames are read */
    std::unique_ptr<io_files::InputFrameFile> input_file_;
    /*! \brief Size of an input frame */
    size_t frame_size_;
    /*! \brief CPU buffer in which the frames are temporarly stored */
    char* cpu_frame_buffer_;
    /*! \brief GPU buffer in which the frames are temporarly stored */
    char* gpu_frame_buffer_;
    /*! \brief Tmp GPU buffer in which the frames are temporarly stored to convert data from packed bits to 16bit */
    char* gpu_packed_buffer_;

    FileReadCache::Cache file_read_cache_;
};
} // namespace holovibes::worker

#include "file_frame_read_worker.hxx"
