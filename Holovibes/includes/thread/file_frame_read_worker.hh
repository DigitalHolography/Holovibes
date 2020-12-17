/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "frame_read_worker.hh"
#include "input_frame_file.hh"

namespace holovibes
{
class Queue;

namespace io_files
{
class InputFrameFile;
}

namespace worker
{
class FileFrameReadWorker : public FrameReadWorker
{
  public:
    FileFrameReadWorker(const std::string& file_path,
                        bool loop,
                        unsigned int fps,
                        unsigned int first_frame_id,
                        unsigned int total_nb_frames_to_read,
                        bool load_file_in_gpu,
                        std::atomic<std::shared_ptr<Queue>>& gpu_input_queue);

    void run() override;

  private:
    class FpsHandler
    {
      public:
        FpsHandler(unsigned int fps);

        /*! \brief Begin the process of fps handling. */
        void begin();

        /*! \brief Wait the correct time to simulate fps.
        **
        ** Between each frame enqueue, the waiting duration should be
        *enqueue_interval_
        ** However the real waiting duration might be longer than the
        *theoretical one (due to descheduling)
        ** To cope with this issue, we compute the wasted time in order to take
        *it into account for the next enqueue
        ** By doing so, the correct enqueuing time is computed, not doing so
        *would create a lag
        **/
        void wait();

      private:
        /*! \brief Theoretical time between 2 enqueues/waits */
        std::chrono::duration<double> enqueue_interval_;

        /*! \brief Begin time point of the wait */
        std::chrono::steady_clock::time_point begin_time_;

        /*! \brief Time wasted in last wait (if waiting was too long) */
        std::chrono::duration<double> wasted_time_{0};
    };

    bool init_frame_buffers();

    void read_file_in_gpu();

    void read_file_batch();

    size_t read_copy_file(size_t frames_to_read);

    void enqueue_loop(size_t nb_frames_to_enqueue);

  private:
    const std::string file_path_;

    bool loop_;

    FpsHandler fps_handler_;

    unsigned int first_frame_id_;

    std::atomic<unsigned int> current_nb_frames_read_;

    const std::atomic<unsigned int> total_nb_frames_to_read_;

    bool load_file_in_gpu_;

    std::unique_ptr<io_files::InputFrameFile> input_file_;

    size_t frame_size_;

    char* cpu_frame_buffer_;

    char* gpu_frame_buffer_;
};
} // namespace worker
} // namespace holovibes

#include "file_frame_read_worker.hxx"