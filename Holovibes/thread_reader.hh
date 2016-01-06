#pragma once

# include <iostream>
# include <thread>
# include <string>

# include "queue.hh"
# include "ithread_input.hh"
# include "power_of_two.hh"

namespace holovibes
{
  /*! \brief Thread add frames to queue from file */
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

      /*! \brief compute sqared FrameDescriptor with power of 2 border size */
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

      /*! \brief Contructor
       * Construct ThreadReader::FrameDescriptor and compute camera::FrameDescriptor */
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

    /*! \brief Constructor */
    ThreadReader(std::string file_src
      , holovibes::ThreadReader::FrameDescriptor frame_desc
      , bool loop
      , unsigned int fps
      , unsigned int spanStart
      , unsigned int spanEnd
      , Queue& input);

    /*! \brief Destructor */
    virtual ~ThreadReader();

  private:
    /*! \brief Read frames while thread is running */
    void  thread_proc(void);

    /*! File source */
    std::string file_src_;
    /*! \brief Does it read file in loop */
    bool loop_;
    /*! Fps readed*/
    unsigned int fps_;
    /*! \brief desc of returned frames */
    camera::FrameDescriptor& frame_desc_;
    /*! \brief desc of readed frames*/
    holovibes::ThreadReader::FrameDescriptor desc_;
    /*! \brief current frame id in file */
    unsigned int frameId_;
    /*! \brief id of the first frame to read */
    unsigned int spanStart_;
    /*! \brief id of the last frame to read */
    unsigned int spanEnd_;
    Queue& queue_;

    std::thread thread_;

    /*! \var unsigned int frameId_
    * Begin at 1.*/
    /*! \var unsigned int spanStart_
    * Begin at 1.*/
    /*! \var unsigned int spanEnd_
    * Begin at 1.
    * Use last frames of file if spanEnd_ is larger. */
  };
}