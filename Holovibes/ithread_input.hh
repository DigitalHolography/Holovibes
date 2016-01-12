/*! \file
 *
 * Interface for a thread encapsulation class that
 * grabs images from a source. */
#pragma once

namespace holovibes
{
  /*! \brief Interface for a thread encapsulation class that
   * grabs images from a source.
   */
  class IThreadInput
  {
  public:
    virtual ~IThreadInput();

  public:
    /*! \brief Stop thread and join it */
    bool stop_requested_;

  protected:
    IThreadInput();
  };
}