/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
 *
 * Encapsulation of a thread used to import raw data from a file,
 * and use it as the source for the input queue. */
#pragma once

#include <QProgressBar>
#include <QObject>

#include "frame_desc.hh"
#include "ithread_input.hh"
#include "power_of_two.hh"
#include "holovibes.hh"
#include "thread_timer.hh"

/* Forward declaration. */
namespace holovibes
{
  enum class FileType;
  class Queue;
  class Holovibes;
}

namespace holovibes
{
  class FpsHandler; // Forwrd declaration. Define in thread_reader.cc

	namespace gui
	{
		class MainWindow;
	}

  /*! \brief Thread encapsulation for reading data from a file.
  *
  * Reads raw data from a file, and interpret it as images of a specified format.
  * The data is transferred to the input queue, and so can be processed as regular
  * images recorded from a camera. */
  class ThreadReader : public QObject, public IThreadInput
  {
			Q_OBJECT

  public:

    /*! \brief Create a preconfigured ThreadReader. */
    ThreadReader(std::string file_src
      , camera::FrameDescriptor& fd
      , SquareInputMode mode
      , bool loop
      , unsigned int fps
      , size_t first_frame_id
      , size_t last_frame_id
      , Queue& input
	    , FileType file_type
      , bool load_file_in_gpu
	    , QProgressBar *reader_progress_bar
	    , gui::MainWindow *main_window);

    virtual ~ThreadReader();

    const camera::FrameDescriptor& get_input_fd() const override;

    const camera::FrameDescriptor& get_queue_fd() const override;

	signals:
    /*! \brief Signal used to synchronize recording
    ** Emit the signal when begin the reading of the file
    */
		void at_begin();

  private:
    /*! \brief Read frames while thread is running
    **
	  ** Proc that will launch the import mode. */
    void thread_proc();

    /*! \brief Load the entire file in gpu, then enqueue the frames one by one
    ** in the destination queue (DeviceToDevice)
    **
    **  Many parameters are needed to enable code factorisation. These
    **  parameters must not be attributes.
    **
    ** \param cpu_buffer preallocated buffer in the cpu (read file)
    ** \param gpu_buffer preallocated buffer in the gpu (copy from cpu_buffer)
    ** \param buffer_size Number of frames in the file (in bytes)
    ** \param file the file being read
    ** \param fps_handler handler for simulating fps
    ** \param thread_timer timer to count the fps
    ** \param nb_frames_one_second number of frames increased at each enqueue
    ** (use by the thread timer)
    */
	  void read_file_in_gpu(char* cpu_buffer,
	  								      char* gpu_buffer,
	  								      size_t buffer_size,
	  								      FILE* file,
	  								      FpsHandler& fps_handler,
	  								      ThreadTimer& thread_timer,
	  								      std::atomic<uint>& nb_frames_one_second);

    /*! \brief Load the file by batch, copy the batch to gpu.
    ** Then enqueue the frames one by one in the destination queue
    ** (DeviceToDevice)
    **
    **  Many parameters are needed to enable code factorisation. These
    **  parameters must not be attributes.
    **
    ** \param cpu_buffer preallocated buffer in the cpu (read file)
    ** \param gpu_buffer preallocated buffer in the gpu (copy from cpu_buffer)
    ** \param buffer_size Number of frames in the file (in bytes)
    ** \param file the file being read
    ** \param start_pos position of the first frame
    ** \param fps_handler handler for simulating fps
    ** \param thread_timer timer to count the fps
    ** \param nb_frames_one_second number of frames increased at each enqueue
    ** (use by the thread timer)
    */
    void read_file_batch(char* cpu_buffer,
                        char* gpu_buffer,
                        size_t buffer_size,
                        FILE* file,
                        fpos_t* start_pos,
                        FpsHandler& fps_handler,
                        ThreadTimer& thread_timer,
                        std::atomic<uint>& nb_frames_one_second);

    /*! \brief read the file (buffer_size bytes) and copy it to the gpu buffer
    ** \param cpu_buffer buffer used to read
    ** \param gpu_buffer store the bytes read in this buffer
    ** \param buffer_size number of bytes to read
    ** \param file file being read
    */
    size_t read_copy_file(char* cpu_buffer,
                          char* gpu_buffer,
                          size_t buffer_size,
                          FILE* file);

    /*! \brief enqueue frames_read in the destination queue with a speed
    ** according to the given fps
    ** \param gpu_buffer frames are stored in this buffer
    ** \param nb_frames_to_enqueue number of frames to enqueue
    ** \param fps_handler handler for simulating fps
    ** \param nb_frames_one_second number of frames increased at each enqueue
    ** (use by the thread timer)
    */
    void enqueue_loop(char* gpu_buffer,
                      size_t nb_frames_to_enqueue,
                      FpsHandler& fps_handler,
                      std::atomic<uint>& nb_frames_one_second);

    /*! \brief handle the case of the last frame
    ** Reset to the first frames if the file should be read several times
    ** Stop if no loop.
    ** If a file is given, set the position in the file to start_pos
    ** \param file file being read
    ** \param start_pos position of the first frame
    */
    void handle_last_frame(FILE* file = nullptr, fpos_t* start_pos = nullptr);

    /*! \brief Seek the offset to attain the .cine file first image */
    long int  offset_cine_first_image(FILE *file);

    /*!
    ** \brief open file, set the offset and update the frame size if needed
    ** \return return the file or nullptr if an error occurs
    */
    FILE* init_file(fpos_t* start_pos);

  private: /* Attributes */
    /*! \brief Source file */
    std::string file_src_;
    /*! \brief If true, the reading will start over when meeting the end of the file. */
    bool loop_;
    /*! \brief Frames Per Second to be displayed. */
    unsigned int fps_;
    /*! \brief Describes the image format asked by the user. */
    camera::FrameDescriptor fd_;
    /*! \brief Size of an input frame */
    unsigned int frame_size_;
    /*! \brief Current frame id in file. */
    size_t cur_frame_id_;
    /*! \brief Id of the first frame to read. */
    size_t first_frame_id_;
    /*! \brief Id of the last frame to read. */
    size_t last_frame_id_;
    /*! \brief The destination Queue in which the frames are enqueued */
    Queue& dst_queue_;
    /*! \brief The type of the file to read */
    FileType file_type_;
    /*! \brief The size in byte of the annotation (8 for cine file) */
    uint frame_annotation_size_;
    /*! \brief Bool to know whether the entire file should be loaded in gpu */
    bool load_file_in_gpu_;

    /*! \brief Frequency of the progress bar refresh */
    const double progress_bar_refresh_frequency_ = 10.0; // 10 Hz
    /*! \brief Number of frames for an update of the progress bar */
    size_t progress_bar_refresh_interval_;

    /*! \brief progress bar showing position in the file */
    QProgressBar *reader_progress_bar_;
    /*! \brief Pointer to main window, used to update the progress bar asyncronously */
    gui::MainWindow *main_window_;

    /*! \brief The thread which shall run thread_proc(). */
    std::thread thread_;
  };
}
