#pragma once

#include <functional>
#include <string>
#include <utility>
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
    virtual const char* get_key() const { return ""; };
    virtual void sync_with(IParameter* ref){};

    bool get_has_been_synchronized() const { return has_been_synchronized_; }
    void set_has_been_synchronized(bool value) { has_been_synchronized_ = value; }

  protected:
    bool has_been_synchronized_ = false;
};
} // namespace holovibes
