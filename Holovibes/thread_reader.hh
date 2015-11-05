#pragma once

# include <iostream>
# include <thread>
#include <string>

# include "queue.hh"
# include "ithread_input.hh"

/*! Max number of frames read each time
 ** Use in order to limit disc usage
 */
#define NBR 5

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
      /*! FrameDescriptor use by holovibes */
      camera::FrameDescriptor desc;
      /*! Width of the image. != frame width */
      unsigned short         img_width;
      /*! Height of the image. != frame height */
      unsigned short         img_height;

      /* \brief check if x is power of two
       * http://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c
       * http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
       */
      inline bool isPowerOfTwo(unsigned int x) const
      {
        return ((x != 0) && ((x & (~x + 1)) == x));
      }

      /*! \brief Return the next power of two */
      inline unsigned int nextHightestPowerOf2(unsigned int x) const
      {
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        x++;
        return (x);
      }

      /*! \brief compute sqared FrameDescriptor with power of 2 border size */
      void compute_sqared_image(void)
      {
        unsigned short biggestBorder = (desc.width > desc.height ? desc.width : desc.height);

        img_width = desc.width;
        img_height = desc.height;

        if (desc.width != desc.height)
          desc.width = desc.height = biggestBorder;

        if (!isPowerOfTwo(biggestBorder))
          desc.width = desc.height = static_cast<unsigned short>(nextHightestPowerOf2(biggestBorder));
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