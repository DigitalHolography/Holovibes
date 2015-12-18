#pragma once

# include "camera_dll.hh"
# include "thread_compute.hh"
# include "thread_capture.hh"
# include "thread_reader.hh"
# include "recorder.hh"
# include "compute_descriptor.hh"
# include "concurrent_deque.hh"

# include <memory>

/*! \brief Containt all function and structure needed to computes data */
namespace holovibes
{
  /*! \brief Core class to use HoloVibes
   *
   * This class does not depends on the user interface (classes under the
   * holovibes namespace can be seen as a library).
   *
   * It contains high-level ressources (Pipe, Camera, Recorder ...). These
   * ressources are shared between threads and should be allocated in threads
   * themselves. */
  class Holovibes
  {
  public:
    /*! \brief Available camera models
     *
     * Available Cameras are defined here, it is easier for user to select the
     * Camera he wish to use, instead of loading the corresponding DLL himself.
     *
     * The non hardcoded-way would be to search for DLL and build a list of
     * available cameras. */
    enum camera_type
    {
      NONE,
      ADIMEC,
      EDGE,
      IDS,
      IXON,
      PIKE,
      PIXELFLY,
      XIQ,
    };

    Holovibes();

    ~Holovibes();

    /*! \brief Open the camera and launch the ThreadCapture
     *
     * Launch the capture thread to continuously acquire frames in input
     * buffer. */
    void Holovibes::init_capture(const enum camera_type c,
      const unsigned int input_max_elts_,
      const unsigned int output_max_elts_);

    /*! \brief Request the capture thread to stop - Free ressources. */
    void dispose_capture();

    bool is_camera_initialized()
    {
      return camera_.operator bool();
    }

    /*! \{ \name Queue getters
     *
     * Used to display/record frames */
    Queue& get_capture_queue()
    {
      return *input_;
    }

    Queue& get_output_queue()
    {
      return *output_;
    }
    /*! \} */

    /*! \brief Launch the recorder
     *
     * \param filepath File path to record frames
     * \param rec_n_images Number of frames to record
     *
     * The Recorder is used only for CLI purpose, a thread is
     * available for GUI because it use QThread way (use slots and is
     * stoppable).
     * Recorder input queue depends on the mode :
     *
     * - direct: use input_ queue
     * - hologram: use output_ queue. */
    void recorder(
      const std::string& filepath,
      const unsigned int rec_n_images);

    /*! \brief Request the recorder thread to stop */
    void dispose_recorder();

    /*! \brief Launch the ThreadCompute
     * \see ThreadCompute
     * \see Pipe
     *
     * The pipe is allocated and his allocation can take some times so that
     * the method contains a lock to avoid conflicts between threads that would
     * use the Pipe before it finished the initialization. */
    void init_compute(
      const ThreadCompute::PipeType pipetype = ThreadCompute::PipeType::PIPE,
      const bool is_float_output_enabled = false,
      const std::string float_output_file_src = "",
      const unsigned int float_output_nb_frame = 0);

    void dispose_compute();

    void init_import_mode(std::string &file_src,
      holovibes::ThreadReader::FrameDescriptor frame_desc,
      bool loop,
      unsigned int fps,
      unsigned int spanStart,
      unsigned int spanEnd,
      unsigned int q_max_size_);

    /*! \{ \name Getters/Setters */
    std::shared_ptr<ICompute> get_pipe()
    {
      if (tcompute_)
        return tcompute_->get_pipe();
      throw std::runtime_error("cannot get pipe, no compute thread");
    }

    /*! \return Common ComputeDescriptor */
    ComputeDescriptor& get_compute_desc()
    {
      return compute_desc_;
    }

    /*! \brief Set ComputeDescriptor options
     *
     * \param compute_desc ComputeDescriptor to load
     *
     * Used when options are loaded from an INI file. */
    void set_compute_desc(const ComputeDescriptor& compute_desc)
    {
      compute_desc_ = compute_desc;
    }

    /*! \return Corresponding Camera INI file path */
    const char* get_camera_ini_path() const
    {
      return camera_->get_ini_path();
    }

    ConcurrentDeque<std::tuple<float, float, float>>& get_average_queue()
    {
      return average_queue_;
    }
    /*! \} */

    const camera::FrameDescriptor& get_cam_frame_desc();

    /* \brief Get zb = N d^2 / lambda
      Is updated everytime the camera changes or lamdba changes
      */
    const float get_boundary();

    /* \brief Getter onto launch_path
    */
    const std::string get_launch_path();

  private:
    /* Use shared pointers to ensure each ressources will freed. */
    /*! \brief ICamera use to acquire image */
    std::shared_ptr<camera::ICamera> camera_;
    bool camera_initialized_;
    /*! \brief IThread which acquiring continuously frames */
    std::unique_ptr<IThreadInput> tcapture_;
    /*! \brief Thread which compute continuously frames */
    std::unique_ptr<ThreadCompute> tcompute_;

    /*! \{ \name Frames queue (GPU) */
    std::unique_ptr<Queue> input_;
    std::unique_ptr<Queue> output_;
    /*! \} */

    /*! \brief Common compute descriptor shared between CLI/GUI and the
     * Pipe. */
    ComputeDescriptor compute_desc_;

    /*! \brief Store average of zone signal/noise
     *
     * Average are computes in ThreadCompute and use in CurvePlot
     * \note see void MainWindow::set_average_graphic() for example
     */
    ConcurrentDeque<std::tuple<float, float, float>> average_queue_;

    /*! \brief Max input queue size */
    unsigned int input_max_elts_;
    /*! \brief Max output queue size */
    unsigned int output_max_elts_;
    /* \brief Store the path of holovibes when it is launched.
       so that holovibes.ini is saved at the right place. The problem
       is that QT's functions actually change the current directory so
       saving holovibes.ini in "$PWD" isn't working*/
    std::string launch_path;
  };
}
