#pragma once

namespace holovibes
{
  class IThreadInput
  {
  protected:
    IThreadInput();
  public:
    virtual ~IThreadInput();
    bool stop_requested_;
  };
}