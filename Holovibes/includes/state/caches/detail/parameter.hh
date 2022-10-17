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

// T is a class that herit from IParameter like CustomParameter, ...
template <typename T>
class DuplicatedParameter : public IDuplicatedParameter
{
  public:
    using BaseParameter = T;

  public:
    typename BaseParameter::ValueType& get_value() { return value_; }
    void set_value(typename BaseParameter::ValueType& value) { value_ = value; }
    virtual void save_current_value(const IParameter* param) override
    {
        const T* param_as_t = dynamic_cast<const T*>(param);
        if (param_as_t == nullptr)
        {
            LOG_ERROR(main, "Not supposed to end here : Not the good type casted when syncing");
            return;
        }

        value_ = param_as_t->get_value();
    }

  protected:
    typename BaseParameter::ValueType value_;
};

} // namespace holovibes
