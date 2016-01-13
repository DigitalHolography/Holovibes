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
    /*! This structure contains everything related to the format of the images
    * stored in file.
    * Used by ThreadReader to read frames.
    * ThreadReader::FrameDescriptor fill camera::FrameDescriptor
    * in order that holovibes use ThreadReader as ThreadCapture.
    */
    struct FrameDescriptor
    {
    public:

      /*! \brief compute squared FrameDescriptor with power of 2 border size */
      void compute_sqared_image(void)
      {
        unsigned short biggestBorder = (desc.width > desc.height ? desc.width : desc.height);

        img_width = desc.width;
        img_height = desc.height;

        if (desc.width != desc.height)
          desc.width = desc.height = biggestBorder;

        if (!isPowerOfTwo(biggestBorder))
          desc.width = desc.height = static_cast<unsigned short>(nextPowerOf2(biggestBorder));
      }

      /*! \brief Adjust a camera::FrameDescriptor and store it. */
      FrameDescriptor(camera::FrameDescriptor d)
        : desc(d)
        , img_width(d.width)
        , img_height(d.height)
      {
        this->compute_sqared_image();
      }

      camera::FrameDescriptor desc;
      /*! Width of the image. != frame width */
      unsigned short         img_width;
      /*! Height of the image. != frame height */
      unsigned short         img_height;
    };

    /*! \brief Create a preconfigured ThreadReader. */
    ThreadReader(std::string file_src
      , holovibes::ThreadReader::FrameDescriptor frame_desc
      , bool loop
      , unsigned int fps
      , unsigned int spanStart
      , unsigned int spanEnd
      , Queue& input);

    virtual ~ThreadReader();

  private:
    /*! \brief Read frames while thread is running */
    void  thread_proc(void);

    /*! \brief Source file */
    std::string file_src_;
    /*! \brief If true, the reading will start over when meeting the end of the file. */
    bool loop_;
    /*! \brief Frames Per Second to be displayed. */
    unsigned int fps_;
    /*! \brief Describes the image format used by the camera. */
    camera::FrameDescriptor& frame_desc_;
    /*! \brief Describes the image format used for reading. */
    holovibes::ThreadReader::FrameDescriptor desc_;
    /*! \brief Current frame id in file. */
    unsigned int frameId_;
    /*! \brief Id of the first frame to read. */
    unsigned int spanStart_;
    /*! \brief Id of the last frame to read. */
    unsigned int spanEnd_;
    /*! \brief The destination Queue. */
    Queue& queue_;

    /*! The thread which shall run thread_proc(). */
    std::thread thread_;
  };
}