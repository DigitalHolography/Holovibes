#pragma once

#include <functional>
#include <string>
#include <utility>
#include "core/types.hh"
#include "logger.hh"

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

  public:
    virtual bool value_has_changed(IParameter* ref) const = 0;
    virtual void sync_with(IParameter* ref) = 0;

    bool get_has_been_synchronized() const { return has_been_synchronized_; }
    void set_has_been_synchronized(bool value) { has_been_synchronized_ = value; }

  protected:
    bool has_been_synchronized_ = false;
};

class IDuplicatedParameter
{
  public:
    virtual void save_current_value(const IParameter* param) = 0;
};

} // namespace holovibes
