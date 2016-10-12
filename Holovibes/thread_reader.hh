/* \file
 *
 * Encapsulation of a thread used to import raw data from a file,
 * and use it as the source for the input queue. */
#pragma once

# include <iostream>
# include <thread>
# include <string>

# include "frame_desc.hh"
# include "ithread_input.hh"
# include "power_of_two.hh"
# include "holovibes.hh"

/* Forward declaration. */
namespace holovibes
{
  class Queue;
}

namespace holovibes
{
  /*! \brief Thread encapsulation for reading data from a file.
  *
  * Reads raw data from a file, and interpret it as images of a specified format.
  * The data is transferred to the input queue, and so can be processed as regular
  * images recorded from a camera. */
  class ThreadReader : public IThreadInput
  {
  public:

    /*! \brief Create a preconfigured ThreadReader. */
    ThreadReader(std::string file_src
      , camera::FrameDescriptor& frame_desc
      , bool loop
      , unsigned int fps
      , unsigned int spanStart
      , unsigned int spanEnd
      , Queue& input
	  , bool is_cine_file);

    virtual ~ThreadReader();

    const camera::FrameDescriptor& get_frame_descriptor() const;
  private:
    /*! \brief Read frames while thread is running */
    void  thread_proc(void);

    /*! \brief Source file */
    std::string file_src_;
    /*! \brief If true, the reading will start over when meeting the end of the file. */
    bool loop_;
    /*! \brief Frames Per Second to be displayed. */
    unsigned int fps_;
    /*! \brief Describes the image format read by thread_reader. */
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

    /*! The thread which shall run thread_proc(). */
    std::thread thread_;
  };
}