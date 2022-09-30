#pragma once

#include <functional>
#include <string>
#include "core/types.hh"

namespace holovibes
{
class Pipe;

class IParameter
{
  public:
    IParameter() {}
    virtual ~IParameter() {}

  public:
    virtual const char* get_key() { return ""; };
    virtual void sync_with(IParameter* ref){};
};
} // namespace holovibes
