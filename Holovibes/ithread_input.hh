#pragma once

namespace holovibes
{
  /*! \brief Interface between ThreadCapture and ThreadReader
   *
   * Both adds frame to queue
   */
  class IThreadInput
  {
  protected:
    IThreadInput();
  public:
    virtual ~IThreadInput();
    /*! \brief Stop thread and join it */
    bool stop_requested_;
  };
}