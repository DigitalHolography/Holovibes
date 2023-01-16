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
#include "advanced_cache.hh"

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

class FileFrameReadWorker;

class FileRequestOnSync
{
  public:
    static void begin_requests() { need_refresh_ = false; }

    static bool do_need_refresh() { return need_refresh_; }
    static void need_refresh() { need_refresh_ = true; }

    static bool has_requests_fail() { return requests_fail_; }
    static void request_fail() { requests_fail_ = true; }

  private:
    static inline bool need_refresh_ = false;
    static inline bool requests_fail_ = false;
};

class AdvancedFileRequestOnSync : public FileRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, FileFrameReadWorker&)
    {
    }

    template <typename T>
    void on_sync(typename T::ConstRefType new_value,
                 [[maybe_unused]] typename T::ConstRefType,
                 FileFrameReadWorker& file_worker)
    {
        operator()<T>(new_value, file_worker);
    }

  public:
    template <>
    void operator()<FileBufferSize>(uint, FileFrameReadWorker&);
};

class ImportFileRequestOnSync : public FileRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, FileFrameReadWorker&)
    {
    }

    template <typename T>
    void on_sync(typename T::ConstRefType new_value,
                 [[maybe_unused]] typename T::ConstRefType,
                 FileFrameReadWorker& file_worker)
    {
        operator()<T>(new_value, file_worker);
    }

  public:
    template <>
    void operator()<ImportFrameDescriptor>(const FrameDescriptor&, FileFrameReadWorker&);
    template <>
    void operator()<InputFps>(uint, FileFrameReadWorker&);
    template <>
    void operator()<StartFrame>(uint, FileFrameReadWorker&);
    template <>
    void operator()<EndFrame>(uint, FileFrameReadWorker&);
    template <>
    void operator()<LoadFileInGpu>(bool, FileFrameReadWorker&);
};

class FpsHandler
{
  public:
    FpsHandler(uint fps)
        : enqueue_interval_((1 / static_cast<double>(fps)))
    {
    }

    void set_new_fps_target(uint fps)
    {
        enqueue_interval_ = std::chrono::duration<double>(1 / static_cast<double>(fps));
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

/*! \class FileFrameReadWorker
 *
 * \brief Class used to read frames from a file
 */
class FileFrameReadWorker final : public FrameReadWorker
{
  public:
    using FileImportCache = ImportCache::Cache<ImportFileRequestOnSync>;
    using FileAdvancedCache = AdvancedCache::Cache<AdvancedFileRequestOnSync>;

    FileFrameReadWorker();
    ~FileFrameReadWorker();

    void run() override;

    /*! \brief Init the cpu_buffer and gpu_buffer */
    bool init_frame_buffers();

    void refresh();

    FpsHandler& get_fps_handler() { return fps_handler_; }

    char* get_cpu_frame_buffer() { return cpu_frame_buffer_; }
    void set_cpu_frame_buffer(char* buffer) { cpu_frame_buffer_ = buffer; }

    char* get_gpu_frame_buffer() { return gpu_frame_buffer_; }
    void set_gpu_frame_buffer(char* buffer) { gpu_frame_buffer_ = buffer; }

    char* get_gpu_packed_buffer() { return gpu_packed_buffer_; }
    void set_gpu_packed_buffer(char* buffer) { gpu_packed_buffer_ = buffer; }

  private:
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
    char* cpu_frame_buffer_ = nullptr;
    /*! \brief GPU buffer in which the frames are temporarly stored */
    char* gpu_frame_buffer_ = nullptr;
    /*! \brief Tmp GPU buffer in which the frames are temporarly stored to convert data from packed bits to 16bit */
    char* gpu_packed_buffer_ = nullptr;

    FileImportCache import_cache_;
    FileAdvancedCache advanced_cache_;

    std::unique_ptr<io_files::InputFrameFile> input_file_;
};
} // namespace holovibes::worker
