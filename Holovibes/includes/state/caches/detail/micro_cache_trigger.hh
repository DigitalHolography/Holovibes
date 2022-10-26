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
    template <typename K>
    friend class TriggerChangeValue;

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

    template <typename Ref>
    explicit TriggerChangeValue(TriggerChangeValue<Ref>&& ref)
        : value_(static_cast<T*>(ref.value_))
        , callback_(ref.callback_)
    {
        take_ownership(ref);
    }

    template <typename Ref>
    TriggerChangeValue& operator=(TriggerChangeValue<Ref>&& ref)
    {
        callback_ = ref.callback_;
        value_ = static_cast<T*>(ref.value_);
        take_ownership(ref);
    }

    ~TriggerChangeValue()
    {
        if (call_callback_)
            callback_();
    }

  public:
    void trigger() {}

  private:
    template <typename Parent>
    void take_ownership(Parent&& parent)
    {
        if (parent.call_callback_)
            parent.dont_call_callback_W();
        else
            dont_call_callback_W();
    }

    //! this function must be handle with care (hence the W, may_be we can change this...)
    void dont_call_callback_W() { call_callback_ = false; }

  public:
    T* operator->() { return value_; }
};

} // namespace holovibes
