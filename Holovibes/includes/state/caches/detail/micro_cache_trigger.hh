#pragma once

#include <functional>
#include <memory>
#include <iostream>

namespace holovibes
{

template <typename T>
class TriggerChangeValue
{
  public:
    T* value_;
    std::function<void(void)> callback_;

  private:
    bool call_callback_ = true;

  public:
    TriggerChangeValue(std::function<void(void)> callback, T* value)
        : value_(value)
        , callback_(callback)
    {
    }

    TriggerChangeValue(const TriggerChangeValue&) = delete;
    TriggerChangeValue& operator=(const TriggerChangeValue&) = delete;

    template <typename Ref>
    TriggerChangeValue(TriggerChangeValue<Ref>&& ref)
        : value_(static_cast<T*>(ref.value_))
        , callback_(ref.callback_)
    {
        ref.dont_call_callback_W();
    }

    template <typename Ref>
    TriggerChangeValue& operator=(TriggerChangeValue<Ref>&& ref)
    {
        callback_ = ref.callback_;
        value_ = static_cast<T*>(ref.value_);
        ref.dont_call_callback_W();
    }

    ~TriggerChangeValue()
    {
        if (call_callback_)
            callback_();
    }

  public:
    //! this function must be handle with care (hence the W, may_be we can change this...)
    void dont_call_callback_W() { call_callback_ = false; }

  public:
    T* operator->() { return value_; }
};

} // namespace holovibes
