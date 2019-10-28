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

# include "frame_desc.hh"
# include "ithread_input.hh"
# include "power_of_two.hh"
# include "holovibes.hh"

/* Forward declaration. */
namespace holovibes
{
  class Queue;
  class Holovibes;
}

namespace holovibes
{

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
      , camera::FrameDescriptor& frame_desc
      , IThreadInput::SquareInputMode mode
      , bool loop
      , unsigned int fps
      , unsigned int spanStart
      , unsigned int spanEnd
      , Queue& input
	    , bool is_cine_file
	    , bool is_holo_file
	    , Holovibes& holovibes
	    , QProgressBar *reader_progress_bar
	    , gui::MainWindow *main_window);

    virtual ~ThreadReader();

    const camera::FrameDescriptor& get_input_frame_descriptor() const override;

    const camera::FrameDescriptor& get_queue_frame_descriptor() const override;

	signals:
		void at_begin();

  private:
    /*! \brief Read frames while thread is running */

	  /* Proc that will launch the import mode. */
    void  thread_proc(void);

	/* the loop that will read elements from a specified file. It is launched FPS times */
	bool reader_loop(FILE* file,
		char* buffer,
		const unsigned int frame_size,
		const unsigned int elts_max_nbr,
		fpos_t pos);
	/*! \brief Seek the offset to attain the .cine file first image */
	long int  offset_cine_first_image(FILE *file);

    /*! \brief Source file */
    std::string file_src_;
    /*! \brief If true, the reading will start over when meeting the end of the file. */
    bool loop_;
    /*! \brief Frames Per Second to be displayed. */
    unsigned int fps_;
    /*! \brief Describes the image format asked by the user. */
    camera::FrameDescriptor frame_desc_;
    /*! \brief Current frame id in file. */
    unsigned int frameId_;
    /*! \brief Id of the first frame to read. */
    unsigned int spanStart_;
    /*! \brief Id of the last frame to read. */
    unsigned int spanEnd_;
    /*! \brief The destination Queue. */
    Queue& queue_;
	/*! \brief Reading a cine file */
	bool is_cine_file_;
	/*! \brief Reading a holo file */
	bool is_holo_file_;
	/*! \brief Holovibes class */
	Holovibes& holovibes_;
	/*\ current buffer frame to be read */
	unsigned int act_frame_;
	/*\ progress bar showing position in the file */
	QProgressBar *reader_progress_bar_;
	/*! \brief Pointer to main window, used to update the progress bar asyncronously */
	gui::MainWindow *main_window_;
	
	/*\ current number of frames effectively stacked in the buffer (not always elts_max_nbr whenever eof is reached)*/
	unsigned int nbr_stored_;

    /*! The thread which shall run thread_proc(). */
    std::thread thread_;
  };
}