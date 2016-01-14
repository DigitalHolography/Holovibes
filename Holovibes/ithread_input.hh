/*! \file
 *
 * Interface for a thread encapsulation class that
 * grabs images from a source. */
#pragma once

/*! Forward declaration*/
namespace camera
{
  class FrameDescriptor;
}

namespace holovibes
{
  /*! \brief Interface for a thread encapsulation class that
   * grabs images from a source.
   */
  class IThreadInput
  {
  public:
    virtual ~IThreadInput();

    virtual const camera::FrameDescriptor& get_frame_descriptor() const = 0;
  public:
    /*! \brief Stop thread and join it */
    bool stop_requested_;

  protected:
    IThreadInput();
  };
}