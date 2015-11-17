#pragma once

namespace holovibes
{
  /*! \brief Interface between ThreadCapture and ThreadReader
   *
   * Both adds frame to queue
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